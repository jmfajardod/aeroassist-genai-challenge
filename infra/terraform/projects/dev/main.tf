locals {
  name = "aeroassist"
  secrets = {
    OPENAI_API_KEY      = "aeroassist-openai-api-key"
    LANGFUSE_PUBLIC_KEY = "aeroassist-langfuse-public-key"
    LANGFUSE_SECRET_KEY = "aeroassist-langfuse-secret-key"
  }
}

resource "google_service_account" "runtime" {
  account_id   = "${local.name}-runtime"
  display_name = "AeroAssist Cloud Run runtime"
}

resource "google_service_account" "deployer" {
  account_id   = "${local.name}-deployer"
  display_name = "AeroAssist CD deployer"
}

data "google_project" "this" {}

resource "google_service_account_iam_member" "deployer_wif" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${data.google_project.this.number}/locations/global/workloadIdentityPools/github/attribute.release/true"
}

resource "google_service_account_iam_member" "terraform_acts_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:aeroassist-terraform@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_service_account_iam_member" "deployer_acts_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.deployer.member
}

module "registry" {
  source         = "../../modules/artifact_registry"
  project_id     = var.project_id
  region         = var.region
  repository_id  = local.name
  writer_members = [google_service_account.deployer.member]
}

module "secrets" {
  source          = "../../modules/secrets"
  project_id      = var.project_id
  secret_ids      = values(local.secrets)
  accessor_member = google_service_account.runtime.member
}

module "service" {
  count                 = var.image == "" ? 0 : 1
  source                = "../../modules/cloud_run"
  project_id            = var.project_id
  region                = var.region
  name                  = local.name
  image                 = var.image
  service_account_email = google_service_account.runtime.email
  deployer_members      = [google_service_account.deployer.member]
  min_instances         = 1
  env = {
    APP_ENV       = var.app_env
    LLM_BACKEND   = "openai"
    PROMPT_SOURCE = "repo"
    LOG_LEVEL     = "INFO"
  }
  secret_env = { for env, id in local.secrets : env => module.secrets.secret_ids[id] }
  depends_on = [module.secrets]
}
