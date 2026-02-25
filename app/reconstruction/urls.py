from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("jobs/<int:job_id>/", views.job_view, name="job_view"),
    path("api/jobs/<int:job_id>/status/", views.job_status_api, name="job_status_api"),
]
