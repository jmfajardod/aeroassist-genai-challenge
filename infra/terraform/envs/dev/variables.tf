variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  description = "First image digest to create the service with; empty skips the service until an image exists."
  type        = string
  default     = ""
}

variable "app_env" {
  type    = string
  default = "dev"
}

variable "github_repository" {
  description = "owner/repo trusted by Workload Identity Federation for deploys."
  type        = string
}
