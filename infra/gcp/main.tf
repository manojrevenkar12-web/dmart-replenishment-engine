locals {
  name   = "dre-${var.env}"
  labels = { app = "dmart-replenishment-engine", env = var.env }
}

resource "google_project_service" "apis" {
  for_each           = toset(["run.googleapis.com", "cloudscheduler.googleapis.com", "artifactregistry.googleapis.com", "secretmanager.googleapis.com", "storage.googleapis.com"])
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "dre"
  location      = var.region
  format        = "DOCKER"
  labels        = local.labels
  depends_on    = [google_project_service.apis]
}

# Input data, model registry (artifacts + metadata + forecasts) and reports
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-${local.name}-data"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  labels                      = local.labels
  versioning { enabled = true } # every registry version stays recoverable
  lifecycle_rule {
    condition { age = 400 }
    action { type = "Delete" }
  }
}

# Optional: LLM key for order explanations; the service runs on templates without it
resource "google_secret_manager_secret" "anthropic" {
  secret_id = "${local.name}-anthropic-api-key"
  labels    = local.labels
  replication {
    user_managed {
      replicas { location = var.region }
    }
  }
  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------- identities (least privilege)
resource "google_service_account" "batch" {
  account_id   = "${local.name}-batch"
  display_name = "Nightly forecast and replenishment run"
}

resource "google_service_account" "api" {
  account_id   = "${local.name}-api"
  display_name = "Order advice API (read-only on the registry)"
}

resource "google_service_account" "scheduler" {
  account_id   = "${local.name}-scheduler"
  display_name = "Triggers the nightly job"
}

resource "google_storage_bucket_iam_member" "batch_rw" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_storage_bucket_iam_member" "api_ro" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret_iam_member" "api_secret" {
  secret_id = google_secret_manager_secret.anthropic.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# ---------------------------------------------------------------- nightly batch: Cloud Run job
resource "google_cloud_run_v2_job" "batch" {
  name     = "${local.name}-batch"
  location = var.region
  labels   = local.labels
  template {
    task_count = 1
    template {
      service_account = google_service_account.batch.email
      timeout         = "7200s"
      max_retries     = 1
      containers {
        image   = var.image
        command = ["dre"]
        args    = ["run", "--config", "config/dre.yaml", "--out", "/work/out"]
        resources {
          limits = { cpu = "8", memory = "32Gi" }
        }
        volume_mounts {
          name       = "work"
          mount_path = "/work"
        }
      }
      volumes {
        name = "work"
        gcs {
          bucket    = google_storage_bucket.data.name
          read_only = false
        }
      }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_invoker" {
  name     = google_cloud_run_v2_job.batch.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "nightly" {
  name      = "${local.name}-nightly"
  region    = var.region
  schedule  = var.batch_schedule
  time_zone = "Europe/Berlin"
  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.batch.name}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }
}

# ---------------------------------------------------------------- order API: Cloud Run service
resource "google_cloud_run_v2_service" "api" {
  name     = "${local.name}-api"
  location = var.region
  labels   = local.labels
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    service_account = google_service_account.api.email
    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = 10
    }
    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      env {
        name  = "DRE_REGISTRY"
        value = "/work/out/registry"
      }
      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic.secret_id
            version = "latest"
          }
        }
      }
      resources {
        limits = { cpu = "2", memory = "4Gi" }
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        failure_threshold     = 10
      }
      volume_mounts {
        name       = "work"
        mount_path = "/work"
      }
    }
    volumes {
      name = "work"
      gcs {
        bucket    = google_storage_bucket.data.name
        read_only = true
      }
    }
  }
  depends_on = [google_project_service.apis]
}
