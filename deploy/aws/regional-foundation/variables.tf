variable "site_id" {
  type = string

  validation {
    condition     = can(regex("^[A-Za-z0-9][A-Za-z0-9._-]{2,47}$", var.site_id))
    error_message = "site_id must be 3..48 safe characters."
  }
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)

  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "At least two private subnets are required."
  }
}

variable "public_subnet_ids" {
  type = list(string)

  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "At least two public subnets are required."
  }
}

variable "cpu_node_security_group_ids" {
  type = set(string)

  validation {
    condition     = length(var.cpu_node_security_group_ids) >= 1
    error_message = "At least one CPU node security group is required."
  }
}

variable "gpu_egress_cidrs" {
  type = set(string)

  validation {
    condition     = length(var.gpu_egress_cidrs) >= 1
    error_message = "At least one GPU egress CIDR is required."
  }
}

variable "certificate_arn" {
  type = string
}

variable "hosted_zone_id" {
  type = string
}

variable "hostname" {
  type = string
}

variable "alert_email" {
  type    = string
  default = null
}

variable "aurora_min_acu" {
  type    = number
  default = 0.5
}

variable "aurora_max_acu" {
  type    = number
  default = 8
}

variable "tags" {
  type    = map(string)
  default = {}
}
