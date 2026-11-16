#!/usr/bin/env bash
# One-time project bootstrap: APIs, Terraform state bucket, Workload Identity Federation for
# GitHub Actions and the Terraform service account. Idempotent; touches only named resources.
#
# Usage: PROJECT_ID=my-project GITHUB_REPO=owner/repo GITHUB_ACTOR=owner ./infra/bootstrap.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${GITHUB_REPO:?set GITHUB_REPO (owner/repo)}"
: "${GITHUB_ACTOR:?set GITHUB_ACTOR (the only GitHub user allowed to obtain GCP credentials)}"
REGION="${REGION:-us-central1}"
STATE_BUCKET="${STATE_BUCKET:-${PROJECT_ID}-aeroassist-tfstate}"
POOL="github"
PROVIDER="github-actions"
TF_SA="aeroassist-terraform"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

gcloud services enable \
  run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  cloudresourcemanager.googleapis.com serviceusage.googleapis.com

if ! gcloud storage buckets describe "gs://${STATE_BUCKET}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${STATE_BUCKET}" --location="$REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi
gcloud storage buckets update "gs://${STATE_BUCKET}" --versioning

if ! gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --location=global \
    --display-name="GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" \
  --workload-identity-pool="$POOL" --location=global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --workload-identity-pool="$POOL" --location=global \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref,attribute.actor=assertion.actor" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}' && assertion.actor == '${GITHUB_ACTOR}'"
fi

if ! gcloud iam service-accounts describe "${TF_SA}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$TF_SA" --display-name="AeroAssist Terraform (CI)"
fi
for role in roles/run.admin roles/artifactregistry.admin roles/secretmanager.admin \
  roles/iam.serviceAccountAdmin roles/iam.serviceAccountUser roles/resourcemanager.projectIamAdmin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet \
    --member="serviceAccount:${TF_SA}@${PROJECT_ID}.iam.gserviceaccount.com" --role="$role" >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://${STATE_BUCKET}" \
  --member="serviceAccount:${TF_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role=roles/storage.objectAdmin >/dev/null
gcloud iam service-accounts add-iam-policy-binding \
  "${TF_SA}@${PROJECT_ID}.iam.gserviceaccount.com" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" >/dev/null

cat <<OUT
Set these GitHub repository variables (Settings > Secrets and variables > Actions > Variables):
  GCP_PROJECT_ID=${PROJECT_ID}
  GCP_REGION=${REGION}
  GCP_WIF_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_TERRAFORM_SA=${TF_SA}@${PROJECT_ID}.iam.gserviceaccount.com
  GCP_DEPLOYER_SA=aeroassist-deployer@${PROJECT_ID}.iam.gserviceaccount.com
  TF_STATE_BUCKET=${STATE_BUCKET}
OUT
