import threading
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from django.conf import settings
from django.db import close_old_connections

from .models import ReconstructionJob


RUNNING_JOBS = set()
RUNNING_LOCK = threading.Lock()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def launch_job(job_id: int) -> None:
    with RUNNING_LOCK:
        if job_id in RUNNING_JOBS:
            return
        RUNNING_JOBS.add(job_id)
    thread = threading.Thread(target=_process_job_thread, args=(job_id,), daemon=True)
    thread.start()


def _process_job_thread(job_id: int) -> None:
    close_old_connections()
    try:
        process_job(job_id)
    finally:
        close_old_connections()
        with RUNNING_LOCK:
            RUNNING_JOBS.discard(job_id)


def _save_job(job: ReconstructionJob, **kwargs) -> None:
    for key, value in kwargs.items():
        setattr(job, key, value)
    job.save(update_fields=[*kwargs.keys(), "updated_at"])


def _list_images(folder: Path) -> list[Path]:
    return sorted([path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTS])


def _extract_video_frames(video_path: Path, output_dir: Path, frame_count: int) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frame_count = max(2, min(frame_count, total))
    indices = np.linspace(0, total - 1, frame_count).astype(int)
    saved_paths = []
    cursor = 0
    target_idx = indices[cursor]
    frame_i = 0
    while cap.isOpened() and cursor < len(indices):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i == target_idx:
            out_path = output_dir / f"frame_{cursor:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)
            cursor += 1
            if cursor < len(indices):
                target_idx = indices[cursor]
        frame_i += 1
    cap.release()
    return saved_paths


def _make_detector(name: str):
    upper = name.upper()
    if upper == "ORB":
        return cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING
    if upper == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    return cv2.SIFT_create(), cv2.NORM_L2


def _match_homography(detector, norm_type, image_a, image_b):
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    key_a, desc_a = detector.detectAndCompute(gray_a, None)
    key_b, desc_b = detector.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(key_a) < 8 or len(key_b) < 8:
        return None, [], []

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, key_a, key_b

    src_pts = np.float32([key_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    return homography, key_a, key_b


def _camera_from_first(detector, norm_type, first_image, other_image):
    gray_a = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)

    key_a, desc_a = detector.detectAndCompute(gray_a, None)
    key_b, desc_b = detector.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        return None

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None

    points_a = np.float32([key_a[m.queryIdx].pt for m in good])
    points_b = np.float32([key_b[m.trainIdx].pt for m in good])
    h, w = gray_a.shape
    focal = max(h, w)
    k_mat = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    essential, _ = cv2.findEssentialMat(points_a, points_b, k_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None:
        return None

    _, rot, trans, _ = cv2.recoverPose(essential, points_a, points_b, k_mat)
    center = (-rot.T @ trans).reshape(3)
    rotation_world = rot.T
    return center, rotation_world


def _normalize_scene(
    cameras: list[dict],
    rectangles: list[dict],
    projection_rectangles: list[dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict] | None]:
    points = []
    for rect in rectangles:
        for pt in rect["points"]:
            points.append(np.array(pt, dtype=np.float64))
    if projection_rectangles:
        for rect in projection_rectangles:
            for pt in rect["points"]:
                points.append(np.array(pt, dtype=np.float64))
    for cam in cameras:
        points.append(np.array(cam["position"], dtype=np.float64))

    if not points:
        return cameras, rectangles, projection_rectangles

    stack = np.vstack(points)
    center = stack.mean(axis=0)
    max_span = np.max(np.ptp(stack, axis=0))
    scale = 1.0 if max_span < 1e-9 else 10.0 / max_span

    for cam in cameras:
        p = (np.array(cam["position"], dtype=np.float64) - center) * scale
        cam["position"] = p.tolist()

    for rect in rectangles:
        norm_points = []
        for pt in rect["points"]:
            p = (np.array(pt, dtype=np.float64) - center) * scale
            norm_points.append(p.tolist())
        rect["points"] = norm_points

    if projection_rectangles:
        for rect in projection_rectangles:
            norm_points = []
            for pt in rect["points"]:
                p = (np.array(pt, dtype=np.float64) - center) * scale
                norm_points.append(p.tolist())
            rect["points"] = norm_points

    return cameras, rectangles, projection_rectangles


def _build_rectangles(transforms: list[np.ndarray], shape_pairs: list[tuple[int, int]]) -> list[dict]:
    rectangles = []
    for idx, (transform, shape) in enumerate(zip(transforms, shape_pairs)):
        h, w = shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners.astype(np.float32), transform.astype(np.float32)).reshape(-1, 2)
        points_3d = [[float(pt[0]), float(pt[1]), 0.0] for pt in projected]
        rectangles.append({"index": idx, "points": points_3d})
    return rectangles


def _compute_incremental_transforms(
    images: list[np.ndarray],
    detector,
    norm_type,
    reference_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = len(images)
    if n == 0:
        return [], []

    reference_index = max(0, min(reference_index, n - 1))
    local_transforms: list[np.ndarray] = []

    for i in range(n - 1):
        homography, _, _ = _match_homography(detector, norm_type, images[i], images[i + 1])
        if homography is None:
            local_transforms.append(np.eye(3, dtype=np.float64))
        else:
            local_transforms.append(homography.astype(np.float64))

    transforms = [np.eye(3, dtype=np.float64) for _ in range(n)]

    for i in range(reference_index, n - 1):
        try:
            inv_h = np.linalg.inv(local_transforms[i])
        except np.linalg.LinAlgError:
            inv_h = np.eye(3, dtype=np.float64)
        transforms[i + 1] = transforms[i] @ inv_h

    for i in range(reference_index - 1, -1, -1):
        transforms[i] = transforms[i + 1] @ local_transforms[i]

    return transforms, local_transforms


def _normalize_rectangles(rectangles: list[dict]) -> list[dict]:
    points = []
    for rect in rectangles:
        for pt in rect.get("points", []):
            points.append(np.array(pt, dtype=np.float64))

    if not points:
        return rectangles

    stack = np.vstack(points)
    center = stack.mean(axis=0)
    max_span = np.max(np.ptp(stack, axis=0))
    scale = 1.0 if max_span < 1e-9 else 10.0 / max_span

    normalized = []
    for rect in rectangles:
        rect_points = []
        for pt in rect.get("points", []):
            p = (np.array(pt, dtype=np.float64) - center) * scale
            rect_points.append(p.tolist())
        normalized.append({"index": rect.get("index", 0), "points": rect_points})
    return normalized


def _build_cameras_from_transforms(
    transforms: list[np.ndarray],
    shape_pairs: list[tuple[int, int]],
    projection_rectangles: list[dict],
) -> list[dict]:
    if not transforms or not shape_pairs:
        return [{"index": 0, "position": [0.0, 0.0, 1.0], "rotation": np.eye(3).tolist()}]

    size_samples = []
    for rect in projection_rectangles:
        pts = rect.get("points", [])
        if len(pts) < 4:
            continue
        p0 = np.array(pts[0], dtype=np.float64)
        p1 = np.array(pts[1], dtype=np.float64)
        p2 = np.array(pts[2], dtype=np.float64)
        width = float(np.linalg.norm(p1 - p0))
        height = float(np.linalg.norm(p2 - p1))
        if width > 1e-6:
            size_samples.append(width)
        if height > 1e-6:
            size_samples.append(height)

    scale_ref = float(np.median(size_samples)) if size_samples else 100.0
    camera_height = max(20.0, 0.9 * scale_ref)

    view_samples = []
    for index, (transform, shape) in enumerate(zip(transforms, shape_pairs)):
        h, w = shape
        center_pt = np.array([[w * 0.5, h * 0.5]], dtype=np.float32).reshape(-1, 1, 2)
        right_pt = np.array([[w * 0.75, h * 0.5]], dtype=np.float32).reshape(-1, 1, 2)
        down_pt = np.array([[w * 0.5, h * 0.75]], dtype=np.float32).reshape(-1, 1, 2)

        center_proj = cv2.perspectiveTransform(center_pt, transform.astype(np.float32)).reshape(2)
        right_proj = cv2.perspectiveTransform(right_pt, transform.astype(np.float32)).reshape(2)
        down_proj = cv2.perspectiveTransform(down_pt, transform.astype(np.float32)).reshape(2)

        view_samples.append(
            {
                "index": index,
                "center": center_proj,
                "right": right_proj,
                "down": down_proj,
            }
        )

    if view_samples:
        global_center_2d = np.mean(np.array([sample["center"] for sample in view_samples], dtype=np.float64), axis=0)
    else:
        global_center_2d = np.array([0.0, 0.0], dtype=np.float64)

    cameras = []
    converge_weight = 0.85
    for sample in view_samples:
        index = sample["index"]
        center_proj = sample["center"]
        right_proj = sample["right"]
        down_proj = sample["down"]

        position = [float(center_proj[0]), float(center_proj[1]), float(camera_height)]
        cam_pos = np.array(position, dtype=np.float64)

        blended_center = converge_weight * global_center_2d + (1.0 - converge_weight) * center_proj
        target = np.array([blended_center[0], blended_center[1], 0.0], dtype=np.float64)
        z_dir = target - cam_pos
        if np.linalg.norm(z_dir) < 1e-9:
            z_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        z_dir = z_dir / np.linalg.norm(z_dir)

        x_seed = np.array([right_proj[0] - center_proj[0], right_proj[1] - center_proj[1], 0.0], dtype=np.float64)
        if np.linalg.norm(x_seed) < 1e-9:
            x_seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_dir = x_seed - np.dot(x_seed, z_dir) * z_dir
        if np.linalg.norm(x_dir) < 1e-9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_dir = fallback - np.dot(fallback, z_dir) * z_dir
        if np.linalg.norm(x_dir) < 1e-9:
            x_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_dir = x_dir / np.linalg.norm(x_dir)

        y_dir = np.cross(z_dir, x_dir)
        if np.linalg.norm(y_dir) < 1e-9:
            y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_dir = y_dir / np.linalg.norm(y_dir)

        down_seed = np.array([down_proj[0] - center_proj[0], down_proj[1] - center_proj[1], 0.0], dtype=np.float64)
        if np.linalg.norm(down_seed) > 1e-9 and np.dot(y_dir, down_seed) < 0:
            y_dir = -y_dir
            x_dir = -x_dir

        rotation = np.column_stack([x_dir, y_dir, z_dir])

        cameras.append(
            {
                "index": index,
                "position": position,
                "rotation": rotation.tolist(),
            }
        )

    if not cameras:
        cameras = [{"index": 0, "position": [0.0, 0.0, float(camera_height)], "rotation": np.eye(3).tolist()}]
    return cameras


def _build_rectangles_from_cameras(cameras: list[dict], shape_pairs: list[tuple[int, int]]) -> list[dict]:
    rectangles = []

    camera_centers = [np.array(cam["position"], dtype=np.float64) for cam in cameras if "position" in cam]
    distance_samples = []
    for idx, center in enumerate(camera_centers):
        nearest = None
        for jdx, other in enumerate(camera_centers):
            if idx == jdx:
                continue
            dist = float(np.linalg.norm(center - other))
            if dist <= 1e-9:
                continue
            if nearest is None or dist < nearest:
                nearest = dist
        if nearest is not None:
            distance_samples.append(nearest)

    if distance_samples:
        scale_ref = float(np.median(distance_samples))
    else:
        scale_ref = 1.0

    depth = max(0.2, min(2.4, 0.8 * scale_ref))

    for index, (camera, shape) in enumerate(zip(cameras, shape_pairs)):
        h, w = shape
        focal = float(max(h, w))

        half_w = depth * (float(w) / (2.0 * focal))
        half_h = depth * (float(h) / (2.0 * focal))

        min_half = max(0.08, 0.18 * scale_ref)
        max_half = max(min_half * 1.2, 0.95 * scale_ref)
        half_w = max(min_half, min(max_half, half_w))
        half_h = max(min_half, min(max_half, half_h))

        center = np.array(camera["position"], dtype=np.float64)
        rotation = np.array(camera["rotation"], dtype=np.float64)

        corners_local = [
            np.array([-half_w, -half_h, depth], dtype=np.float64),
            np.array([half_w, -half_h, depth], dtype=np.float64),
            np.array([half_w, half_h, depth], dtype=np.float64),
            np.array([-half_w, half_h, depth], dtype=np.float64),
        ]
        corners_world = [rotation @ corner + center for corner in corners_local]
        rectangles.append({"index": index, "points": [point.tolist() for point in corners_world]})
    return rectangles


def _panorama_fallback(images: list[np.ndarray]) -> np.ndarray:
    resized = []
    target_h = min(img.shape[0] for img in images)
    for image in images:
        ratio = target_h / image.shape[0]
        resized.append(cv2.resize(image, (int(image.shape[1] * ratio), target_h)))
    return cv2.hconcat(resized)


def _compute_panorama_canvas(transforms: list[np.ndarray], shape_pairs: list[tuple[int, int]]):
    corners_all = []
    for transform, shape in zip(transforms, shape_pairs):
        h, w = shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners.astype(np.float32), transform.astype(np.float32)).reshape(-1, 2)
        corners_all.append(projected)

    if not corners_all:
        return np.eye(3), 1, 1

    all_pts = np.vstack(corners_all)
    min_x = float(np.floor(np.min(all_pts[:, 0])))
    min_y = float(np.floor(np.min(all_pts[:, 1])))
    max_x = float(np.ceil(np.max(all_pts[:, 0])))
    max_y = float(np.ceil(np.max(all_pts[:, 1])))

    width = max(1, int(max_x - min_x))
    height = max(1, int(max_y - min_y))
    translation = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    return translation, width, height


def _build_planar_projected_previews(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    shape_pairs: list[tuple[int, int]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    translation, width, height = _compute_panorama_canvas(transforms, shape_pairs)
    for idx, (image, transform) in enumerate(zip(images, transforms)):
        matrix = translation @ transform
        warped = cv2.warpPerspective(image, matrix.astype(np.float32), (width, height))
        cv2.imwrite(str(output_dir / f"img_{idx:04d}.jpg"), warped)


def _render_panorama_from_transforms(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    shape_pairs: list[tuple[int, int]],
) -> np.ndarray:
    translation, width, height = _compute_panorama_canvas(transforms, shape_pairs)

    acc = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for image, transform in zip(images, transforms):
        matrix = translation @ transform
        warped = cv2.warpPerspective(image, matrix.astype(np.float32), (width, height))
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
        acc[mask] += warped[mask].astype(np.float32)
        weight[mask] += 1.0

    valid = weight > 0
    pano = np.zeros((height, width, 3), dtype=np.uint8)
    if np.any(valid):
        pano_float = acc[valid] / weight[valid, None]
        pano[valid] = np.clip(pano_float, 0, 255).astype(np.uint8)

    if not np.any(valid):
        return _panorama_fallback(images)
    return pano


def _build_pedagogy_assets(images: list[np.ndarray], detector, norm_type, output_dir: Path, reference_index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(images) < 2:
        return

    reference_index = max(0, min(reference_index, len(images) - 1))
    pair_index = reference_index + 1 if reference_index + 1 < len(images) else reference_index - 1
    if pair_index < 0 or pair_index >= len(images):
        return

    img1 = images[reference_index]
    img2 = images[pair_index]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return

    feat1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    feat2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(output_dir / "features_0.jpg"), feat1)
    cv2.imwrite(str(output_dir / "features_1.jpg"), feat2)

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    matches_viz = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good[:120],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_dir / "matches_0_1.jpg"), matches_viz)

    if len(good) < 8:
        return

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if homography is None:
        return

    h1, w1 = img1.shape[:2]
    warped2 = cv2.warpPerspective(img2, homography, (w1, h1))
    blend = cv2.addWeighted(img1, 0.5, warped2, 0.5, 0)
    cv2.imwrite(str(output_dir / "warp_0_1.jpg"), blend)


def _build_pedagogy_iteration_asset(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    shape_pairs: list[tuple[int, int]],
    output_dir: Path,
    max_frames: int = 10,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(max_frames, len(images))
    if count <= 0:
        return 0

    thumbs = []
    target_h = 120
    for idx in range(count):
        image = images[idx]
        h, w = image.shape[:2]
        ratio = target_h / max(1, h)
        thumb = cv2.resize(image, (max(1, int(w * ratio)), target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"I{idx+1}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)

    selected_images = images[:count]
    selected_transforms = transforms[:count]
    selected_shapes = shape_pairs[:count]

    translation, width, height = _compute_panorama_canvas(selected_transforms, selected_shapes)
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, (image, transform, shape) in enumerate(zip(selected_images, selected_transforms, selected_shapes)):
        matrix = translation @ transform
        warped = cv2.warpPerspective(image, matrix.astype(np.float32), (width, height))

        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
        mosaic[mask] = warped[mask]

        h, w = shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners.astype(np.float32), matrix.astype(np.float32)).reshape(-1, 2)
        polygon = np.int32(projected)
        cv2.polylines(mosaic, [polygon], True, (80, 200, 120), 2, cv2.LINE_AA)

        center = np.mean(projected, axis=0)
        cx, cy = int(center[0]), int(center[1])
        cv2.putText(
            mosaic,
            f"I{idx + 1}",
            (cx - 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_dir / "iterate_mosaic.jpg"), mosaic)
    return count


def process_job(job_id: int) -> None:
    job = ReconstructionJob.objects.get(pk=job_id)
    try:
        _save_job(job, status=ReconstructionJob.Status.RUNNING, progress=2, message="Préparation des données")

        work_dir = Path(job.working_dir)
        input_dir = work_dir / "input"
        extract_dir = work_dir / "extracted"
        processed_dir = work_dir / "processed"
        extract_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        images = []
        if job.input_mode == ReconstructionJob.InputMode.IMAGES:
            images = _list_images(input_dir)
        elif job.input_mode == ReconstructionJob.InputMode.ZIP:
            zip_files = sorted(input_dir.glob("*.zip"))
            if zip_files:
                with ZipFile(zip_files[0]) as archive:
                    archive.extractall(extract_dir)
                images = _list_images(extract_dir)
        elif job.input_mode == ReconstructionJob.InputMode.VIDEO:
            videos = sorted(
                [
                    p
                    for p in input_dir.glob("*")
                    if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}
                ]
            )
            if videos:
                images = _extract_video_frames(videos[0], extract_dir, job.frame_count)

        if len(images) < 2:
            _save_job(
                job,
                status=ReconstructionJob.Status.FAILED,
                progress=100,
                message="Erreur",
                error_text="Impossible de trouver au moins 2 images exploitables.",
            )
            return

        _save_job(job, progress=10, message=f"Prétraitement et recadrage ({len(images)} images)")

        loaded_images = []
        shape_pairs = []
        max_side = 1600
        for idx, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            if job.crop_w and job.crop_h and job.crop_w > 1 and job.crop_h > 1:
                h, w = image.shape[:2]
                x = max(0, min(job.crop_x or 0, w - 1))
                y = max(0, min(job.crop_y or 0, h - 1))
                crop_w = max(1, min(job.crop_w, w - x))
                crop_h = max(1, min(job.crop_h, h - y))
                image = image[y : y + crop_h, x : x + crop_w]

            h, w = image.shape[:2]
            longest = max(h, w)
            if longest > max_side:
                ratio = max_side / float(longest)
                new_w = max(1, int(w * ratio))
                new_h = max(1, int(h * ratio))
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = processed_dir / f"img_{idx:04d}.jpg"
            cv2.imwrite(str(out_path), image)
            loaded_images.append(image)
            shape_pairs.append((image.shape[0], image.shape[1]))

            preprocess_progress = 10 + int(((idx + 1) / max(1, len(images))) * 10)
            _save_job(
                job,
                progress=min(20, preprocess_progress),
                message=f"Prétraitement image {idx + 1}/{len(images)}",
            )

        if len(loaded_images) < 2:
            _save_job(
                job,
                status=ReconstructionJob.Status.FAILED,
                progress=100,
                message="Erreur",
                error_text="Le prétraitement n'a pas produit assez d'images valides.",
            )
            return

        detector, norm = _make_detector(job.algorithm)
        _save_job(job, progress=22, message=f"Détection et appariement ({job.algorithm})")

        pedagogy_dir = work_dir / "pedagogy"
        reference_index = len(loaded_images) // 2
        _build_pedagogy_assets(loaded_images, detector, norm, pedagogy_dir, reference_index)

        transforms, local_transforms = _compute_incremental_transforms(loaded_images, detector, norm, reference_index)

        for index in range(1, len(transforms)):
            partial_transforms = transforms[: index + 1]
            partial_shapes = shape_pairs[: index + 1]
            partial_projection = _build_rectangles(partial_transforms, partial_shapes)
            partial_rectangles = _normalize_rectangles(partial_projection)
            partial_cams = []

            progress = 22 + int((index / max(1, len(loaded_images) - 1)) * 33)
            _save_job(
                job,
                progress=progress,
                message=f"Reconstruction progressive ({index + 1}/{len(loaded_images)})",
                rectangles_json=partial_rectangles,
                cameras_json=partial_cams,
            )

        _build_pedagogy_iteration_asset(
            loaded_images,
            transforms,
            shape_pairs,
            pedagogy_dir,
            max_frames=10,
        )

        _save_job(job, progress=58, message="Estimation des caméras")

        projection_rectangles_planar = _build_rectangles(transforms, shape_pairs)
        rectangles_planar = _normalize_rectangles(projection_rectangles_planar)

        _save_job(job, progress=68, message="Finalisation des géométries")

        planar_dir = work_dir / "processed_planar"
        _build_planar_projected_previews(loaded_images, transforms, shape_pairs, planar_dir)

        projection_path_planar = work_dir / "projection_rectangles_planar.json"
        with projection_path_planar.open("w", encoding="utf-8") as fp:
            json.dump(rectangles_planar, fp)

        _save_job(
            job,
            progress=72,
            message="Visualisation 3D mise à jour",
            cameras_json=[],
            rectangles_json=rectangles_planar,
        )

        _save_job(job, progress=82, message="Assemblage panorama")

        panorama_path_planar_png = work_dir / "panorama_planar.png"
        panorama_path_planar_jpg = work_dir / "panorama_planar.jpg"

        panorama_path_png = work_dir / "panorama.png"
        panorama_path_jpg = work_dir / "panorama.jpg"

        _save_job(job, progress=88, message="Assemblage panorama (transforms chainés)")
        pano_planar = _render_panorama_from_transforms(loaded_images, transforms, shape_pairs)
        if pano_planar is None or pano_planar.size == 0:
            pano_planar = _panorama_fallback(loaded_images)

        _save_job(job, progress=96, message="Finalisation des exports panorama")
        cv2.imwrite(str(panorama_path_planar_png), pano_planar)
        cv2.imwrite(str(panorama_path_planar_jpg), pano_planar, [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(str(panorama_path_png), pano_planar)
        cv2.imwrite(str(panorama_path_jpg), pano_planar, [cv2.IMWRITE_JPEG_QUALITY, 93])

        rel_png = panorama_path_png.relative_to(settings.MEDIA_ROOT)
        rel_jpg = panorama_path_jpg.relative_to(settings.MEDIA_ROOT)
        _save_job(
            job,
            status=ReconstructionJob.Status.COMPLETED,
            progress=100,
            message="Terminé",
            cameras_json=[],
            rectangles_json=rectangles_planar,
            panorama_png=str(rel_png),
            panorama_jpg=str(rel_jpg),
            error_text="",
        )
    except Exception as exc:
        _save_job(
            job,
            status=ReconstructionJob.Status.FAILED,
            progress=100,
            message="Erreur",
            error_text=f"{type(exc).__name__}: {exc}",
        )


def job_public_payload(job: ReconstructionJob) -> dict:
    frame_urls = []
    frame_planar_urls = []
    projection_rectangles = []
    rectangles_planar = []
    panorama_planar_png = None
    panorama_planar_jpg = None
    reference_index = 0
    pedagogy = {}
    pedagogy_meta = {"iter_count": 0}
    try:
        if job.working_dir:
            processed_dir = Path(job.working_dir) / "processed"
            if processed_dir.exists():
                for image_path in sorted(processed_dir.glob("img_*.jpg")):
                    rel = image_path.relative_to(settings.MEDIA_ROOT)
                    frame_urls.append(settings.MEDIA_URL + str(rel))

            planar_dir = Path(job.working_dir) / "processed_planar"
            if planar_dir.exists():
                for image_path in sorted(planar_dir.glob("img_*.jpg")):
                    rel = image_path.relative_to(settings.MEDIA_ROOT)
                    frame_planar_urls.append(settings.MEDIA_URL + str(rel))

            if frame_urls:
                reference_index = len(frame_urls) // 2

            projection_path = Path(job.working_dir) / "projection_rectangles_planar.json"
            if projection_path.exists():
                try:
                    with projection_path.open("r", encoding="utf-8") as fp:
                        loaded = json.load(fp)
                    if isinstance(loaded, list):
                        projection_rectangles = loaded
                        rectangles_planar = loaded
                except Exception:
                    projection_rectangles = []
                    rectangles_planar = []

            pano_planar_png_path = Path(job.working_dir) / "panorama_planar.png"
            pano_planar_jpg_path = Path(job.working_dir) / "panorama_planar.jpg"

            if pano_planar_png_path.exists():
                rel = pano_planar_png_path.relative_to(settings.MEDIA_ROOT)
                panorama_planar_png = settings.MEDIA_URL + str(rel)
            if pano_planar_jpg_path.exists():
                rel = pano_planar_jpg_path.relative_to(settings.MEDIA_ROOT)
                panorama_planar_jpg = settings.MEDIA_URL + str(rel)

            pedagogy_dir = Path(job.working_dir) / "pedagogy"
            pedagogy_files = {
                "features_0": pedagogy_dir / "features_0.jpg",
                "features_1": pedagogy_dir / "features_1.jpg",
                "matches_0_1": pedagogy_dir / "matches_0_1.jpg",
                "warp_0_1": pedagogy_dir / "warp_0_1.jpg",
                "iterate_mosaic": pedagogy_dir / "iterate_mosaic.jpg",
            }
            for key, file_path in pedagogy_files.items():
                if file_path.exists():
                    rel = file_path.relative_to(settings.MEDIA_ROOT)
                    pedagogy[key] = settings.MEDIA_URL + str(rel)

            frame_count = len(frame_urls) if frame_urls else 0
            pedagogy_meta["iter_count"] = min(10, frame_count)
            pedagogy_meta["reference_index"] = reference_index
    except Exception:
        frame_urls = []
        frame_planar_urls = []
        projection_rectangles = []
        rectangles_planar = []
        panorama_planar_png = None
        panorama_planar_jpg = None
        reference_index = 0
        pedagogy = {}
        pedagogy_meta = {"iter_count": 0}

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error_text,
        "algorithm": job.algorithm,
        "cameras": job.cameras_json,
        "rectangles": job.rectangles_json,
        "projection_rectangles": projection_rectangles,
        "rectangles_planar": rectangles_planar or job.rectangles_json,
        "frame_urls": frame_urls,
        "frame_planar_urls": frame_planar_urls,
        "pedagogy": pedagogy,
        "pedagogy_meta": pedagogy_meta,
        "panorama_png": settings.MEDIA_URL + job.panorama_png if job.panorama_png else None,
        "panorama_jpg": settings.MEDIA_URL + job.panorama_jpg if job.panorama_jpg else None,
        "panorama_planar_png": panorama_planar_png,
        "panorama_planar_jpg": panorama_planar_jpg,
    }
