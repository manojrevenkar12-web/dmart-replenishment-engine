variable "project_id" {
  description = "GCP project for the replenishment engine."
  type        = string
}

variable "region" {
  description = "EU region; Frankfurt by default."
  type        = string
  default     = "europe-west3"
  validation {
    condition     = startswith(var.region, "europe-")
    error_message = "Sales and vendor data must stay in an EU region."
  }
}

variable "env" {
  type    = string
  default = "dev"
}

variable "image" {
  description = "Container image (Artifact Registry) built by CI, e.g. europe-west3-docker.pkg.dev/<project>/dre/dre:<sha>."
  type        = string
}

variable "batch_schedule" {
  description = "Nightly forecast and order run (Cloud Scheduler, Europe/Berlin)."
  type        = string
  default     = "30 1 * * *"
}

variable "api_min_instances" {
  type    = number
  default = 1
}
