from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from django.conf import settings
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_http_methods

from .models import ReconstructionJob
from .pipeline import job_public_payload, launch_job


def _save_upload(uploaded_file, destination: Path) -> None:
	with destination.open("wb+") as output:
		for chunk in uploaded_file.chunks():
			output.write(chunk)


def _int_or_none(value: str):
	if value in (None, ""):
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _create_preview_from_zip(zip_path: Path, preview_path: Path) -> bool:
	image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
	try:
		with ZipFile(zip_path) as archive:
			for member in archive.namelist():
				suffix = Path(member).suffix.lower()
				if suffix not in image_exts:
					continue
				with archive.open(member) as source:
					data = source.read()
				arr = np.frombuffer(data, dtype=np.uint8)
				img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
				if img is None:
					continue
				cv2.imwrite(str(preview_path), img)
				return True
	except Exception:
		return False
	return False


def _create_preview_from_video(video_path: Path, preview_path: Path) -> bool:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		return False
	ok, frame = cap.read()
	cap.release()
	if not ok or frame is None:
		return False
	cv2.imwrite(str(preview_path), frame)
	return True


def _create_preview_from_paths(image_paths: list[Path], preview_path: Path) -> bool:
	for image_path in image_paths:
		img = cv2.imread(str(image_path))
		if img is None:
			continue
		h, w = img.shape[:2]
		max_side = max(h, w)
		if max_side > 1200:
			ratio = 1200.0 / float(max_side)
			img = cv2.resize(img, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
		cv2.imwrite(str(preview_path), img)
		return True
	return False


@require_http_methods(["GET", "POST"])
def index(request):
	if request.method == "GET":
		return render(request, "reconstruction/index.html")

	upload_files = request.FILES.getlist("uploads")
	image_files = []
	zip_file = None
	video_file = None

	if not upload_files:
		return render(
			request,
			"reconstruction/index.html",
			{"error": "Ajoute un ZIP, une vidéo ou plusieurs images pour lancer la reconstruction."},
			status=400,
		)

	image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
	video_exts = {".mp4", ".mov"}

	video_count = 0
	zip_count = 0
	image_count = 0
	other_count = 0

	for uploaded in upload_files:
		suffix = Path(uploaded.name).suffix.lower()
		content_type = uploaded.content_type or ""
		if suffix == ".zip" or content_type in {"application/zip", "application/x-zip-compressed"}:
			zip_count += 1
		elif suffix in video_exts or content_type.startswith("video/"):
			video_count += 1
		elif suffix in image_exts or content_type.startswith("image/"):
			image_count += 1
		else:
			other_count += 1

	if other_count > 0:
		return render(
			request,
			"reconstruction/index.html",
			{"error": "Un ou plusieurs fichiers ne sont pas supportés."},
			status=400,
		)

	if zip_count == 1 and video_count == 0 and image_count == 0 and len(upload_files) == 1:
		zip_file = upload_files[0]
	elif video_count == 1 and zip_count == 0 and image_count == 0 and len(upload_files) == 1:
		video_file = upload_files[0]
	elif image_count >= 1 and zip_count == 0 and video_count == 0:
		image_files = upload_files
	else:
		return render(
			request,
			"reconstruction/index.html",
			{"error": "Sélection invalide: choisis soit 1 ZIP, soit 1 vidéo, soit des images uniquement."},
			status=400,
		)

	input_mode = None
	if image_files:
		input_mode = ReconstructionJob.InputMode.IMAGES
	elif zip_file:
		input_mode = ReconstructionJob.InputMode.ZIP
	elif video_file:
		input_mode = ReconstructionJob.InputMode.VIDEO

	if input_mode is None:
		return render(
			request,
			"reconstruction/index.html",
			{"error": "Ajoute un ZIP, une vidéo ou plusieurs images pour lancer la reconstruction."},
			status=400,
		)

	algorithm = request.POST.get("algorithm", ReconstructionJob.Algorithm.SIFT).upper()
	if algorithm not in {choice[0] for choice in ReconstructionJob.Algorithm.choices}:
		algorithm = ReconstructionJob.Algorithm.SIFT

	frame_count = _int_or_none(request.POST.get("frame_count")) or 20
	frame_count = max(2, min(frame_count, 200))

	job = ReconstructionJob.objects.create(
		input_mode=input_mode,
		algorithm=algorithm,
		frame_count=frame_count,
		crop_x=_int_or_none(request.POST.get("crop_x")),
		crop_y=_int_or_none(request.POST.get("crop_y")),
		crop_w=_int_or_none(request.POST.get("crop_w")),
		crop_h=_int_or_none(request.POST.get("crop_h")),
		status=ReconstructionJob.Status.PENDING,
		message="Initialisation",
	)

	media_root = Path(settings.MEDIA_ROOT)
	working_dir = media_root / "jobs" / str(job.id)
	input_dir = working_dir / "input"
	input_dir.mkdir(parents=True, exist_ok=True)

	preview_relative = ""
	if input_mode == ReconstructionJob.InputMode.IMAGES:
		saved_paths = []
		for idx, image_file in enumerate(image_files):
			extension = Path(image_file.name).suffix.lower() or ".jpg"
			filename = f"img_{idx:04d}{extension}"
			destination = input_dir / filename
			_save_upload(image_file, destination)
			saved_paths.append(destination)
		preview_path = working_dir / "preview.jpg"
		if _create_preview_from_paths(saved_paths, preview_path):
			preview_relative = str(preview_path.relative_to(media_root))
	elif input_mode == ReconstructionJob.InputMode.ZIP:
		destination = input_dir / (Path(zip_file.name).stem + ".zip")
		_save_upload(zip_file, destination)
		preview_path = working_dir / "preview.jpg"
		if _create_preview_from_zip(destination, preview_path):
			preview_relative = str(preview_path.relative_to(media_root))
	else:
		video_ext = Path(video_file.name).suffix.lower()
		if video_ext not in {".mp4", ".mov"}:
			video_ext = ".mp4"
		destination = input_dir / (Path(video_file.name).stem + video_ext)
		_save_upload(video_file, destination)
		preview_path = working_dir / "preview.jpg"
		if _create_preview_from_video(destination, preview_path):
			preview_relative = str(preview_path.relative_to(media_root))

	job.working_dir = str(working_dir)
	job.preview_image = preview_relative
	job.save(update_fields=["working_dir", "preview_image", "updated_at"])

	launch_job(job.id)
	return redirect("job_view", job_id=job.id)


@require_GET
def job_view(request, job_id: int):
	job = get_object_or_404(ReconstructionJob, pk=job_id)
	preview_url = f"{settings.MEDIA_URL}{job.preview_image}" if job.preview_image else ""
	return render(
		request,
		"reconstruction/job_view.html",
		{
			"job": job,
			"preview_url": preview_url,
		},
	)


@require_GET
def job_status_api(request, job_id: int):
	job = get_object_or_404(ReconstructionJob, pk=job_id)
	return JsonResponse(job_public_payload(job))
