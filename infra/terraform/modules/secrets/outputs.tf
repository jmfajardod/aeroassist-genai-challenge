output "secret_ids" {
  value = { for id, secret in google_secret_manager_secret.this : id => secret.secret_id }
}
