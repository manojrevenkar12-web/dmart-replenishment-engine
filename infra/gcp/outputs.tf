output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "data_bucket" {
  value = google_storage_bucket.data.name
}

output "image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
