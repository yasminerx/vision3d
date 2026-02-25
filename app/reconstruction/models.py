from django.db import models


class ReconstructionJob(models.Model):
	class Status(models.TextChoices):
		PENDING = "pending", "Pending"
		RUNNING = "running", "Running"
		COMPLETED = "completed", "Completed"
		FAILED = "failed", "Failed"

	class InputMode(models.TextChoices):
		IMAGES = "images", "Images"
		ZIP = "zip", "ZIP"
		VIDEO = "video", "Video"

	class Algorithm(models.TextChoices):
		SIFT = "SIFT", "SIFT"
		ORB = "ORB", "ORB"
		AKAZE = "AKAZE", "AKAZE"

	status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
	progress = models.PositiveSmallIntegerField(default=0)
	message = models.CharField(max_length=255, blank=True)
	error_text = models.TextField(blank=True)

	input_mode = models.CharField(max_length=12, choices=InputMode.choices)
	algorithm = models.CharField(max_length=12, choices=Algorithm.choices, default=Algorithm.SIFT)
	frame_count = models.PositiveIntegerField(default=20)

	crop_x = models.IntegerField(null=True, blank=True)
	crop_y = models.IntegerField(null=True, blank=True)
	crop_w = models.IntegerField(null=True, blank=True)
	crop_h = models.IntegerField(null=True, blank=True)

	preview_image = models.CharField(max_length=400, blank=True)
	working_dir = models.CharField(max_length=400, blank=True)
	panorama_png = models.CharField(max_length=400, blank=True)
	panorama_jpg = models.CharField(max_length=400, blank=True)

	cameras_json = models.JSONField(default=list, blank=True)
	rectangles_json = models.JSONField(default=list, blank=True)

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self):
		return f"Job #{self.pk} - {self.status}"
