terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  # backend "gcs" {}  # state bucket configured per environment in the deployment pipeline
}

provider "google" {
  project = var.project_id
  region  = var.region
}
