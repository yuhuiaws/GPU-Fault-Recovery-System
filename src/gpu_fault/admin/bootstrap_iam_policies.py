"""The IAM policy documents the bootstrap grants its two AWS identities.

Stated as documents so the same validator the release path applies to the
executor role (``regional_release_iam``) can check them without an account:
the control plane reads SageMaker and notifies through the site's own channel;
an executor holds exactly one cluster's node reads and reboot, scoped to the
provider that cluster runs on.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError
from gpu_fault.admin.site import archive_s3_prefix_arn
from gpu_fault.cluster_scope import ClusterProvider

__all__ = ["control_plane_policy_document", "executor_policy_document"]


def control_plane_policy_document(
    *,
    region: str,
    account_id: str,
    email_sender: str | None = None,
    archive_s3_uri: str | None = None,
    sns_topic_arn: str | None = None,
) -> dict[str, Any]:
    """Every AWS permission the regional control plane is allowed to hold.

    This is a value rather than a literal buried in ``ensure_control_plane_role``
    because it is the whole blast-radius argument for the CPU side: SageMaker
    read-only, no node mutation of any kind, notifications only to the site's
    own SNS topic (``channel: sns``) or only from the verified sender
    (``channel: ses``), and -- only when ``site.yaml`` ``spec.retention`` names
    an archive bucket -- ``PutObject`` under that one prefix, because the
    archiver writes bundles and never reads or deletes them. Stated as a
    document it can be checked against the same validator the release path
    applies to the executor role, without an AWS account.
    """

    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeCluster",
                "sagemaker:ListClusterNodes",
                "sagemaker:DescribeClusterNode",
            ],
            "Resource": (f"arn:aws:sagemaker:{region}:{account_id}:cluster/*"),
        }
    ]
    if email_sender:
        statements.append(
            {
                "Sid": "AdministratorEmail",
                "Effect": "Allow",
                "Action": "ses:SendEmail",
                "Resource": (
                    f"arn:aws:ses:{region}:{account_id}:identity/{email_sender}"
                ),
                "Condition": {
                    "StringEquals": {
                        "ses:FromAddress": email_sender,
                    }
                },
            }
        )
    if sns_topic_arn:
        # Publish only: the control plane never subscribes, lists or deletes;
        # the topic itself stays a bootstrap-owned resource.
        statements.append(
            {
                "Sid": "AdministratorNotificationTopic",
                "Effect": "Allow",
                "Action": "sns:Publish",
                "Resource": sns_topic_arn,
            }
        )
    if archive_s3_uri:
        statements.append(
            {
                "Sid": "ControlRecordArchive",
                "Effect": "Allow",
                "Action": "s3:PutObject",
                "Resource": archive_s3_prefix_arn(archive_s3_uri),
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}


def executor_policy_document(
    *,
    hyperpod_arn: str = "",
    provider: str = ClusterProvider.HYPERPOD.value,
    eks_cluster_arn: str = "",
) -> dict[str, Any]:
    """Every AWS permission one cluster's executor is allowed to hold.

    HyperPod: scoped to that cluster's HyperPod ARN, with reboot as the only
    mutation and no SES: notification stays a control-plane concern so
    deduplication cannot be bypassed from the data plane. Plain EKS
    (``eks-managed``): ``ec2:DescribeInstances`` (a read EC2 only grants on
    ``*``) and ``ec2:RebootInstances`` on the instances tagged by EKS with this
    cluster's name -- no ``sagemaker:*`` at all, and no stop, terminate or
    launch, so a replacement can never come from the data plane.
    """

    if ClusterProvider(provider) is ClusterProvider.EKS_MANAGED:
        eks = Arn.parse(eks_cluster_arn)
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DescribeClusterInstances",
                    "Effect": "Allow",
                    "Action": "ec2:DescribeInstances",
                    "Resource": "*",
                },
                {
                    "Sid": "RebootClusterInstances",
                    "Effect": "Allow",
                    "Action": "ec2:RebootInstances",
                    "Resource": (
                        f"arn:{eks.partition}:ec2:{eks.region}:{eks.account}:instance/*"
                    ),
                    "Condition": {
                        "StringEquals": {
                            "aws:ResourceTag/eks:cluster-name": eks.resource_name
                        }
                    },
                },
            ],
        }
    if not hyperpod_arn:
        raise BootstrapError("a HyperPod executor policy needs the HyperPod ARN")
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:DescribeCluster",
                    "sagemaker:ListClusterNodes",
                    "sagemaker:DescribeClusterNode",
                    "sagemaker:BatchRebootClusterNodes",
                ],
                "Resource": hyperpod_arn,
            }
        ],
    }
