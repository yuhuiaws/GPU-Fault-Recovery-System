data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  name = "gpu-fault-${var.site_id}"
  tags = merge(
    {
      "gpu-fault:site-id" = var.site_id
      "gpu-fault:owner"   = "terraform"
    },
    var.tags,
  )
}

resource "aws_ecr_repository" "runtime" {
  name                 = "${local.name}-runtime"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.tags
}

resource "aws_ecr_repository" "runtime_cache" {
  name                 = "${local.name}-runtime-cache"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(local.tags, {
    "gpu-fault:resource-role" = "build-cache"
  })
}

resource "aws_prometheus_workspace" "this" {
  alias = local.name
  tags  = local.tags
}

resource "aws_sns_topic" "alerts" {
  name              = "${local.name}-alerts"
  kms_master_key_id = "alias/aws/sns"
  tags              = local.tags
}

resource "aws_sqs_queue" "alerts" {
  name                      = "${local.name}-alerts"
  message_retention_seconds = 1209600
  kms_master_key_id         = "alias/aws/sqs"
  tags                      = local.tags
}

resource "aws_sqs_queue_policy" "alerts" {
  queue_url = aws_sqs_queue.alerts.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowSns"
      Effect    = "Allow"
      Principal = { Service = "sns.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.alerts.arn
      Condition = {
        ArnEquals = {
          "aws:SourceArn" = aws_sns_topic.alerts.arn
        }
      }
    }]
  })
}

resource "aws_sns_topic_subscription" "queue" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "sqs"
  endpoint  = aws_sqs_queue.alerts.arn
}

resource "aws_sns_topic_subscription" "email" {
  count = var.alert_email == null ? 0 : 1

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_security_group" "nlb" {
  name        = "${local.name}-nlb"
  description = "GPU fault regional NLB ingress"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "nlb" {
  for_each = var.gpu_egress_cidrs

  security_group_id = aws_security_group.nlb.id
  cidr_ipv4         = each.value
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_security_group" "aurora" {
  name        = "${local.name}-aurora"
  description = "GPU fault Aurora PostgreSQL"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "aurora" {
  for_each = var.cpu_node_security_group_ids

  security_group_id            = aws_security_group.aurora.id
  referenced_security_group_id = each.value
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

resource "aws_db_subnet_group" "aurora" {
  name       = "${local.name}-aurora"
  subnet_ids = var.private_subnet_ids
  tags       = local.tags
}

resource "aws_rds_cluster" "aurora" {
  cluster_identifier                  = "${local.name}-aurora"
  engine                              = "aurora-postgresql"
  engine_mode                         = "provisioned"
  database_name                       = "gpu_fault"
  master_username                     = "gpu_fault_admin"
  manage_master_user_password         = true
  db_subnet_group_name                = aws_db_subnet_group.aurora.name
  vpc_security_group_ids              = [aws_security_group.aurora.id]
  storage_encrypted                   = true
  backup_retention_period             = 7
  deletion_protection                 = true
  copy_tags_to_snapshot               = true
  iam_database_authentication_enabled = true

  serverlessv2_scaling_configuration {
    min_capacity = var.aurora_min_acu
    max_capacity = var.aurora_max_acu
  }

  tags = local.tags
}

resource "aws_rds_cluster_instance" "aurora" {
  for_each = toset(["writer", "reader"])

  identifier         = "${local.name}-aurora-${each.value}"
  cluster_identifier = aws_rds_cluster.aurora.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.aurora.engine
  promotion_tier     = each.value == "writer" ? 0 : 1
  tags               = local.tags
}
