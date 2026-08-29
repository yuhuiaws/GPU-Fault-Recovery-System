output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  value = data.aws_region.current.region
}

output "runtime_image_repository" {
  value = aws_ecr_repository.runtime.repository_url
}

output "runtime_image_cache_repository" {
  value = aws_ecr_repository.runtime_cache.repository_url
}

output "public_subnet_ids" {
  value = var.public_subnet_ids
}

output "nlb_security_group_id" {
  value = aws_security_group.nlb.id
}

output "certificate_arn" {
  value = var.certificate_arn
}

output "hosted_zone_id" {
  value = var.hosted_zone_id
}

output "hostname" {
  value = var.hostname
}

output "aurora_cluster_id" {
  value = aws_rds_cluster.aurora.cluster_identifier
}

output "aurora_master_secret_arn" {
  value     = aws_rds_cluster.aurora.master_user_secret[0].secret_arn
  sensitive = true
}

output "amp_workspace_id" {
  value = aws_prometheus_workspace.this.id
}

output "sns_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "sqs_queue_arn" {
  value = aws_sqs_queue.alerts.arn
}

output "sqs_queue_url" {
  value = aws_sqs_queue.alerts.url
}
