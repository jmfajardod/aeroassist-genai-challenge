# Infrastructure

Terraform for the dev environment on GCP Cloud Run, plus a one-time bootstrap script. No state
files, credentials or project-wide destructive commands are committed; state lives in a
versioned GCS bucket.

| Path | What it creates |
|---|---|
| bootstrap.sh | APIs, the Terraform state bucket, the GitHub Workload Identity pool and provider (one repository, one user, release-branch attribute), the Terraform and read-only plan service accounts, and an optional budget alert |
| terraform/modules/artifact_registry | Docker repository with a cleanup policy (keep the 5 most recent images) and writer bindings |
| terraform/modules/secrets | Secret Manager containers, readable only by the runtime service account; values are added outside Terraform |
| terraform/modules/cloud_run | The Cloud Run service (1 vCPU, 2 GiB, min 0, max 1, concurrency 10, 80 s timeout, startup probe on /ready, liveness on /health, public invoker) and deployer binding; Terraform ignores image changes because CD deploys images |
| terraform/projects/dev | Runtime and deployer service accounts, least-privilege bindings, and the modules above |

Identities (no JSON keys anywhere): the runtime account can only read its three secrets; the
deployer can push images, deploy revisions of this service and act as the runtime account; the
Terraform account manages these resources and can act only as the runtime account; the plan
account is read-only. GitHub authenticates with Workload Identity Federation: the provider accepts
only this repository and the owner's user, and the Terraform and deployer accounts are available
only to runs on release/* branches. Third-party actions use major version tags.

Order for a new project: run bootstrap.sh, apply Terraform once without an image, add the
secret values, push a first image and apply again with INITIAL_IMAGE set. From then on both
terraform-apply and cd run only from release/* branches, only for the repository owner, and only
after the owner approves the production environment. docs/RUNBOOK.md has the exact commands, rollback and
cleanup.
