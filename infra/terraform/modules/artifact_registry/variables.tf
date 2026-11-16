variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "repository_id" {
  type = string
}

variable "keep_count" {
  description = "Most recent image versions to keep; older ones are deleted."
  type        = number
  default     = 5
}

variable "writer_members" {
  description = "IAM members allowed to push images."
  type        = list(string)
  default     = []
}
