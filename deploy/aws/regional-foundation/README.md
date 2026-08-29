# Regional foundation

This Terraform module owns the reusable AWS foundation for one regional site:
Aurora Serverless v2, AMP, SNS/SQS, the runtime ECR repository, and the
security groups consumed by the Kubernetes release.

Existing CPU/GPU HyperPod EKS clusters, Route53 zone, ACM certificate, and
public/private subnets are inputs. Apply this module in the infrastructure
pipeline, then populate `site.yaml` from `terraform output -json`. The normal
`gpu-fault-admin deploy -f site.yaml` path performs discovery, validation, and
application release only.

The former ARN-only Python foundation bootstrap is migration-only and requires
`--allow-legacy-python-foundation`.
