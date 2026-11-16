variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "name" {
  type = string
}

variable "image" {
  description = "Initial image; later revisions are deployed by CD, so Terraform ignores changes."
  type        = string
}

variable "service_account_email" {
  type = string
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  type    = string
  default = "2Gi"
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 1
}

variable "concurrency" {
  type    = number
  default = 10
}

variable "timeout_seconds" {
  type    = number
  default = 80
}

variable "env" {
  description = "Plain environment variables."
  type        = map(string)
  default     = {}
}

variable "secret_env" {
  description = "Environment variable name to Secret Manager secret ID."
  type        = map(string)
  default     = {}
}

variable "public" {
  description = "Allow unauthenticated invocations (the evaluator calls without auth)."
  type        = bool
  default     = true
}

variable "deployer_members" {
  description = "IAM members allowed to deploy new revisions."
  type        = list(string)
  default     = []
}
