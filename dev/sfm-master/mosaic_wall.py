import os
import cv2
import numpy as np

# -------- Config -------- #
DATA_NPZ = "results/out.npz"           # chemin vers out.npz
IMG_DIR = "data/horizontal_4m80_cropped"  # dossier des images déjà croppées
STEP = 2                                # même step que lors du run SfM
CROP_DX, CROP_DY = 220, 77              # offset du crop appliqué avant SfM (à ajuster au besoin)
SCALE = 100                             # échelle utilisée au run (pour cohérence K)
CANVAS_MARGIN = 200                     # marge en pixels autour de la mosaïque
BLEND = True                            # moyenne pondérée simple

# ------------------------- #

def estimate_plane(points):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < 3:
        raise ValueError("Pas assez de points 3D pour estimer un plan")
    centroid = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    d = normal @ centroid
    return normal, d

def homography_from_pose(K, R, t, normal, d):
    """H = K (R - t n^T / d) K^-1"""
    K3 = K[:3, :3]
    H = (R - (t @ normal[None, :]) / d)
    return K3 @ H @ np.linalg.inv(K3)

def warp_on_canvas(img, H, canvas, weight):
    h, w = img.shape[:2]
    # warp corners to find bbox
    corners = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=np.float32)
    warped_c = (H @ corners.T).T
    warped_c = warped_c[:, :2] / warped_c[:, 2:3]
    min_xy = warped_c.min(axis=0)
    max_xy = warped_c.max(axis=0)
    return min_xy, max_xy

def main():
    if not os.path.exists(DATA_NPZ):
        raise FileNotFoundError(f"Fichier npz introuvable: {DATA_NPZ}")
    data = np.load(DATA_NPZ)
    K = data["K"].copy()
    Ts = data["T"]
    P = data["P"]

    # Ajuster le centre optique pour le crop
    K[0,2] -= CROP_DX
    K[1,2] -= CROP_DY
    K[:2] *= SCALE/100.0

    normal, d = estimate_plane(P)
    print(f"Plan estimé: n={normal}, d={d:.2f}")

    # Charger images (avec step identique à SfM)
    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png','.jpg','.jpeg')))
    files = files[::STEP]
    if len(files) < len(Ts):
        print(f"Attention: {len(files)} images pour {len(Ts)} poses. On tronque au plus petit.")
    N = min(len(files), len(Ts))
    files = files[:N]
    Ts = Ts[:N]

    # Pré-calcul bbox globale
    all_min = []
    all_max = []
    for pose, fname in zip(Ts, files):
        R = np.linalg.inv(pose)[:3,:3]
        t = np.linalg.inv(pose)[:3,3]
        H = homography_from_pose(K, R, t, normal, d)
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        if img is None:
            continue
        mn, mx = warp_on_canvas(img, H, None, None)
        all_min.append(mn)
        all_max.append(mx)

    if not all_min:
        raise RuntimeError("Aucune image valide lue.")

    global_min = np.min(np.vstack(all_min), axis=0) - CANVAS_MARGIN
    global_max = np.max(np.vstack(all_max), axis=0) + CANVAS_MARGIN
    width = int(np.ceil(global_max[0] - global_min[0]))
    height = int(np.ceil(global_max[1] - global_min[1]))
    shift = np.array([[1,0,-global_min[0]], [0,1,-global_min[1]], [0,0,1]])

    print(f"Canvas: {width} x {height}")
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for i, (pose, fname) in enumerate(zip(Ts, files)):
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skip {fname} (non lu)")
            continue
        R = np.linalg.inv(pose)[:3,:3]
        t = np.linalg.inv(pose)[:3,3]
        H = homography_from_pose(K, R, t, normal, d)
        Hs = shift @ H
        warped = cv2.warpPerspective(img, Hs, (width, height))
        mask = (warped.sum(axis=2) > 0).astype(np.float32)
        if BLEND:
            canvas += warped.astype(np.float32)
            weight += mask
        else:
            canvas[mask > 0] = warped[mask > 0]
        print(f"{i+1}/{N} : {fname} placé")

    if BLEND:
        weight[weight == 0] = 1
        canvas = (canvas / weight[...,None]).astype(np.uint8)
    else:
        canvas = canvas.astype(np.uint8)

    out_path = "mosaic_wall.png"
    cv2.imwrite(out_path, canvas)
    print(f"Mosaïque sauvegardée: {out_path}")

if __name__ == "__main__":
    main()
