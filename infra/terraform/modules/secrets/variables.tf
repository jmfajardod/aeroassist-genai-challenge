variable "project_id" {
  type = string
}

variable "secret_ids" {
  description = "Secret containers to create; values are added outside Terraform."
  type        = list(string)
}

variable "accessor_member" {
  description = "The only member allowed to read these secrets."
  type        = string
}
