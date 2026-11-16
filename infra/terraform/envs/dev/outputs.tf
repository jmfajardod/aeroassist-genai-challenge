output "image_base" {
  value = module.registry.image_base
}

output "service_url" {
  value = try(module.service[0].url, null)
}

output "runtime_service_account" {
  value = google_service_account.runtime.email
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}
