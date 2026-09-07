"""How uninstall proves a single AWS resource is present or absent.

Every one of these probes is the sole evidence behind one line of the uninstall
report. Two failure modes matter: a probe that asks the wrong question (wrong
region, wrong identifier, wrong API) and a probe that reads an error as "gone".
The first deletes or preserves the wrong thing; the second reports a clean
uninstall over a resource that is still running and still billing.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from gpu_fault.admin import aws_commands as admin_aws_commands
from gpu_fault.admin.aws_cleanup import SUPPORTED_RESOURCE_TYPES, ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)
from tests.admin.test_admin_site import site_file

REGION = "us-east-1"
ACCOUNT = "123456789012"
ZONE = "Z123"


def _resource(
    resource_type: str,
    resource_id: str,
    *,
    arn: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    resource = InstallationResource(
        site_id="test-site",
        resource_key=f"aws/{resource_type}",
        resource_type=resource_type,
        resource_id=resource_id,
        resource_arn=arn,
        region=REGION,
        account_id=ACCOUNT,
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
        created_at=now,
        updated_at=now,
    )
    if attributes:
        return resource.model_copy(update={"attributes": attributes})
    return resource


class Aws:
    """A fake AWS CLI that records the probe and answers as instructed."""

    def __init__(
        self, *, stdout: str = "{}", returncode: int = 0, stderr: str = ""
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(
        self, arguments: Sequence[Any], **_keywords: Any
    ) -> subprocess.CompletedProcess:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


# (resource type, resource kwargs, expected probe, the API's own "absent" error)
PROBES: list[tuple[str, dict[str, Any], list[str], str]] = [
    (
        "nlb",
        {"resource_id": "gpu-fault-nlb", "arn": "arn:aws:elasticloadbalancing:nlb/a"},
        [
            "aws",
            "elbv2",
            "describe-load-balancers",
            "--region",
            REGION,
            "--load-balancer-arns",
            "arn:aws:elasticloadbalancing:nlb/a",
        ],
        "LoadBalancerNotFound",
    ),
    (
        "nlb",
        {"resource_id": "gpu-fault-nlb"},
        [
            "aws",
            "elbv2",
            "describe-load-balancers",
            "--region",
            REGION,
            "--names",
            "gpu-fault-nlb",
        ],
        "LoadBalancerNotFound",
    ),
    (
        "nlb_listener",
        {"resource_id": "arn:listener/a"},
        [
            "aws",
            "elbv2",
            "describe-listeners",
            "--region",
            REGION,
            "--listener-arns",
            "arn:listener/a",
        ],
        "ListenerNotFound",
    ),
    (
        "nlb_target_group",
        {"resource_id": "arn:tg/a"},
        [
            "aws",
            "elbv2",
            "describe-target-groups",
            "--region",
            REGION,
            "--target-group-arns",
            "arn:tg/a",
        ],
        "TargetGroupNotFound",
    ),
    (
        "security_group",
        {"resource_id": "sg-a"},
        [
            "aws",
            "ec2",
            "describe-security-groups",
            "--region",
            REGION,
            "--group-ids",
            "sg-a",
        ],
        "InvalidGroup.NotFound",
    ),
    (
        "ec2_subnet",
        {"resource_id": "subnet-a"},
        [
            "aws",
            "ec2",
            "describe-subnets",
            "--region",
            REGION,
            "--subnet-ids",
            "subnet-a",
        ],
        "InvalidSubnetID.NotFound",
    ),
    (
        "ec2_route_table",
        {"resource_id": "rtb-a"},
        [
            "aws",
            "ec2",
            "describe-route-tables",
            "--region",
            REGION,
            "--route-table-ids",
            "rtb-a",
        ],
        "InvalidRouteTableID.NotFound",
    ),
    (
        "internet_gateway",
        {"resource_id": "igw-a"},
        [
            "aws",
            "ec2",
            "describe-internet-gateways",
            "--region",
            REGION,
            "--internet-gateway-ids",
            "igw-a",
        ],
        "InvalidInternetGatewayID.NotFound",
    ),
    (
        "route53_zone",
        {"resource_id": ZONE},
        ["aws", "route53", "get-hosted-zone", "--region", REGION, "--id", ZONE],
        "NoSuchHostedZone",
    ),
    (
        "acm_certificate",
        {"resource_id": "arn:acm/a"},
        [
            "aws",
            "acm",
            "describe-certificate",
            "--region",
            REGION,
            "--certificate-arn",
            "arn:acm/a",
        ],
        "ResourceNotFoundException",
    ),
    (
        "secretsmanager_secret",
        {"resource_id": "gpu-fault-pki"},
        [
            "aws",
            "secretsmanager",
            "describe-secret",
            "--region",
            REGION,
            "--secret-id",
            "gpu-fault-pki",
        ],
        "ResourceNotFoundException",
    ),
    (
        "rds_managed_secret",
        {"resource_id": "arn:secret/aurora"},
        [
            "aws",
            "secretsmanager",
            "describe-secret",
            "--region",
            REGION,
            "--secret-id",
            "arn:secret/aurora",
        ],
        "ResourceNotFoundException",
    ),
    (
        "amp_workspace",
        {"resource_id": "ws-0001"},
        [
            "aws",
            "amp",
            "describe-workspace",
            "--region",
            REGION,
            "--workspace-id",
            "ws-0001",
        ],
        "ResourceNotFoundException",
    ),
    (
        "grafana_workspace",
        {"resource_id": "g-5b81a13d97"},
        [
            "aws",
            "grafana",
            "describe-workspace",
            "--region",
            REGION,
            "--workspace-id",
            "g-5b81a13d97",
        ],
        "ResourceNotFoundException",
    ),
    (
        "sns_topic",
        {"resource_id": "arn:sns:topic"},
        [
            "aws",
            "sns",
            "get-topic-attributes",
            "--region",
            REGION,
            "--topic-arn",
            "arn:sns:topic",
        ],
        "NotFound",
    ),
    (
        "sns_subscription",
        {"resource_id": "arn:sns:topic:sub"},
        [
            "aws",
            "sns",
            "get-subscription-attributes",
            "--region",
            REGION,
            "--subscription-arn",
            "arn:sns:topic:sub",
        ],
        "NotFoundException",
    ),
    (
        "ses_email_identity",
        {"resource_id": "ops@example.com"},
        [
            "aws",
            "sesv2",
            "get-email-identity",
            "--region",
            REGION,
            "--email-identity",
            "ops@example.com",
        ],
        "NotFoundException",
    ),
    (
        "iam_role",
        {"resource_id": "gpu-fault-control"},
        ["aws", "iam", "get-role", "--role-name", "gpu-fault-control"],
        "NoSuchEntity",
    ),
    (
        "iam_policy",
        {"resource_id": "arn:iam:policy/a"},
        ["aws", "iam", "get-policy", "--policy-arn", "arn:iam:policy/a"],
        "NoSuchEntity",
    ),
    (
        "iam_oidc_provider",
        {"resource_id": "arn:iam:oidc-provider/a"},
        [
            "aws",
            "iam",
            "get-open-id-connect-provider",
            "--open-id-connect-provider-arn",
            "arn:iam:oidc-provider/a",
        ],
        "NoSuchEntity",
    ),
    (
        "eks_pod_identity_association",
        {"resource_id": "assoc-a", "attributes": {"cluster_name": "control"}},
        [
            "aws",
            "eks",
            "describe-pod-identity-association",
            "--region",
            REGION,
            "--cluster-name",
            "control",
            "--association-id",
            "assoc-a",
        ],
        "ResourceNotFoundException",
    ),
    (
        "eks_addon",
        {
            "resource_id": "eks-pod-identity-agent",
            "attributes": {"cluster_name": "control"},
        },
        [
            "aws",
            "eks",
            "describe-addon",
            "--region",
            REGION,
            "--cluster-name",
            "control",
            "--addon-name",
            "eks-pod-identity-agent",
        ],
        "ResourceNotFoundException",
    ),
    (
        "ecr_repository",
        {"resource_id": "gpu-fault"},
        [
            "aws",
            "ecr",
            "describe-repositories",
            "--region",
            REGION,
            "--repository-names",
            "gpu-fault",
        ],
        "RepositoryNotFoundException",
    ),
    (
        "rds_db_subnet_group",
        {"resource_id": "gpu-fault-subnets"},
        [
            "aws",
            "rds",
            "describe-db-subnet-groups",
            "--region",
            REGION,
            "--db-subnet-group-name",
            "gpu-fault-subnets",
        ],
        "DBSubnetGroupNotFoundFault",
    ),
    (
        "aurora_cluster",
        {"resource_id": "gpu-fault-aurora"},
        [
            "aws",
            "rds",
            "describe-db-clusters",
            "--region",
            REGION,
            "--db-cluster-identifier",
            "gpu-fault-aurora",
        ],
        "DBClusterNotFoundFault",
    ),
    (
        "rds_cluster_parameter_group",
        {"resource_id": "gpu-fault-aurora-pg"},
        [
            "aws",
            "rds",
            "describe-db-cluster-parameter-groups",
            "--region",
            REGION,
            "--db-cluster-parameter-group-name",
            "gpu-fault-aurora-pg",
        ],
        "DBParameterGroupNotFound",
    ),
    (
        "aurora_instance",
        {"resource_id": "gpu-fault-aurora-writer"},
        [
            "aws",
            "rds",
            "describe-db-instances",
            "--region",
            REGION,
            "--db-instance-identifier",
            "gpu-fault-aurora-writer",
        ],
        "DBInstanceNotFound",
    ),
    (
        "rds_snapshot",
        {"resource_id": "gpu-fault-final"},
        [
            "aws",
            "rds",
            "describe-db-cluster-snapshots",
            "--region",
            REGION,
            "--db-cluster-snapshot-identifier",
            "gpu-fault-final",
        ],
        "DBClusterSnapshotNotFoundFault",
    ),
    (
        "cpu_eks",
        {"resource_id": "control"},
        ["aws", "eks", "describe-cluster", "--region", REGION, "--name", "control"],
        "ResourceNotFoundException",
    ),
    (
        "gpu_eks",
        {"resource_id": "gpu-a"},
        ["aws", "eks", "describe-cluster", "--region", REGION, "--name", "gpu-a"],
        "ResourceNotFoundException",
    ),
    (
        "cpu_hyperpod",
        {"resource_id": "hp-control"},
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            REGION,
            "--cluster-name",
            "hp-control",
        ],
        "ResourceNotFound",
    ),
    (
        "gpu_hyperpod",
        {"resource_id": "hp-gpu-a"},
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            REGION,
            "--cluster-name",
            "hp-gpu-a",
        ],
        "ResourceNotFound",
    ),
]


@pytest.fixture(name="site")
def site_fixture(tmp_path: Path) -> Any:
    return load_site(site_file(tmp_path))


@pytest.fixture(name="cleaner")
def cleaner_fixture(site: Any) -> ResourceCleaner:
    return ResourceCleaner(site)


def _identifier(case: tuple[str, dict[str, Any], list[str], str]) -> str:
    resource_type, keywords, _command, _absent = case
    return f"{resource_type}-{keywords['resource_id']}"


@pytest.mark.parametrize("case", PROBES, ids=_identifier)
def test_each_resource_type_is_probed_with_its_own_api(
    cleaner: ResourceCleaner,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, dict[str, Any], list[str], str],
) -> None:
    """The probe command is the contract, argument for argument.

    A probe in the wrong region, or against a name where the API wants an ARN,
    answers "not found" for a resource that is really there -- and uninstall then
    reports it deleted. IAM is deliberately region-free; everything else is pinned
    to the site's region.
    """

    resource_type, keywords, command, _absent = case
    aws = Aws()
    monkeypatch.setattr(admin_aws_commands.subprocess, "run", aws)

    assert cleaner.exists(_resource(resource_type, **keywords)) is True
    assert aws.calls == [command]


@pytest.mark.parametrize("case", PROBES, ids=_identifier)
def test_each_api_absent_error_means_absent(
    cleaner: ResourceCleaner,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, dict[str, Any], list[str], str],
) -> None:
    """Each service spells "gone" differently, and each spelling is accepted.

    Getting one of these tokens wrong turns a completed deletion into a permanent
    ``still exists`` failure that no retry can clear.
    """

    resource_type, keywords, _command, absent = case
    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(returncode=254, stdout="", stderr=f"An error occurred ({absent}) ..."),
    )

    assert cleaner.exists(_resource(resource_type, **keywords)) is False


@pytest.mark.parametrize("case", PROBES, ids=_identifier)
def test_no_probe_reads_an_unexpected_error_as_absent(
    cleaner: ResourceCleaner,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, dict[str, Any], list[str], str],
) -> None:
    """Throttling and AccessDenied must never be read as deletion.

    This is the failure that produces a false clean uninstall, so it is asserted
    for every resource type rather than for a representative one.
    """

    resource_type, keywords, _command, _absent = case
    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(returncode=254, stdout="", stderr="An error occurred (AccessDenied) ..."),
    )

    with pytest.raises(BootstrapError, match="AccessDenied"):
        cleaner.exists(_resource(resource_type, **keywords))


def test_a_route53_record_is_matched_by_name_and_type(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the record this site created counts as present.

    A zone holds records for everything in the VPC; matching the zone alone would
    make uninstall refuse to finish because somebody else's record exists.
    """

    record = _resource(
        "route53_record",
        "api.gpu-fault.internal",
        attributes={"hosted_zone_id": ZONE, "record_type": "A"},
    )
    document = {
        "ResourceRecordSets": [
            {"Name": "api.gpu-fault.internal.", "Type": "CNAME"},
            {"Name": "other.gpu-fault.internal.", "Type": "A"},
        ]
    }
    aws = Aws(stdout=json.dumps(document))
    monkeypatch.setattr(admin_aws_commands.subprocess, "run", aws)

    assert cleaner.exists(record) is False
    assert aws.calls[0][-2:] == ["--output", "json"]

    document["ResourceRecordSets"].append(
        {"Name": "api.gpu-fault.internal.", "Type": "A"}
    )
    monkeypatch.setattr(
        admin_aws_commands.subprocess, "run", Aws(stdout=json.dumps(document))
    )

    assert cleaner.exists(record) is True


def test_a_missing_zone_makes_its_records_and_associations_absent(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deleted zone takes its records and VPC associations with it.

    Probing them individually after the zone is gone would raise
    ``NoSuchHostedZone`` and fail an uninstall that has already succeeded.
    """

    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(returncode=254, stderr="An error occurred (NoSuchHostedZone) ..."),
    )

    assert (
        cleaner.exists(
            _resource(
                "route53_record",
                "api.gpu-fault.internal",
                attributes={"hosted_zone_id": ZONE},
            )
        )
        is False
    )
    assert (
        cleaner.exists(
            _resource(
                "route53_vpc_association",
                "vpc-a",
                attributes={
                    "hosted_zone_id": ZONE,
                    "vpc_id": "vpc-a",
                    "vpc_region": REGION,
                },
            )
        )
        is False
    )


def test_a_vpc_association_is_matched_by_vpc_and_region(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private zone is shared by every GPU cluster's VPC.

    Reading "the zone has associations" as "this association exists" would leave
    one cluster's VPC attached to the control plane's private zone after its
    removal, and uninstall would never notice.
    """

    attributes = {"hosted_zone_id": ZONE, "vpc_id": "vpc-gpu-b", "vpc_region": REGION}
    association = _resource(
        "route53_vpc_association", "vpc-gpu-b", attributes=attributes
    )
    others = {"VPCs": [{"VPCId": "vpc-gpu-a", "VPCRegion": REGION}]}
    monkeypatch.setattr(
        admin_aws_commands.subprocess, "run", Aws(stdout=json.dumps(others))
    )

    assert cleaner.exists(association) is False

    both = {
        "VPCs": [
            {"VPCId": "vpc-gpu-a", "VPCRegion": REGION},
            {"VPCId": "vpc-gpu-b", "VPCRegion": REGION},
        ]
    }
    monkeypatch.setattr(
        admin_aws_commands.subprocess, "run", Aws(stdout=json.dumps(both))
    )

    assert cleaner.exists(association) is True


def test_a_route_table_association_is_looked_up_by_its_own_id(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The association has no describe API of its own, only a filter.

    An empty result is the only proof it is detached; the route table itself
    survives and would answer "present" to any coarser query.
    """

    association = _resource("ec2_route_table_association", "rtbassoc-a")
    aws = Aws(stdout=json.dumps({"RouteTables": []}))
    monkeypatch.setattr(admin_aws_commands.subprocess, "run", aws)

    assert cleaner.exists(association) is False
    assert (
        "Name=association.route-table-association-id,Values=rtbassoc-a" in aws.calls[0]
    )


def test_an_sqs_policy_binding_is_the_topic_inside_the_queue_policy(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detaching the binding leaves the queue; only the policy changes.

    Probing the queue instead of the policy would report the binding as present
    forever, because the queue is meant to survive when it was reused.
    """

    binding = _resource(
        "sqs_policy_binding",
        "https://sqs.us-east-1.amazonaws.com/123456789012/gpu-fault",
        attributes={"topic_arn": "arn:aws:sns:us-east-1:123456789012:gpu-fault"},
    )
    empty = {"Attributes": {"Policy": ""}}
    monkeypatch.setattr(
        admin_aws_commands.subprocess, "run", Aws(stdout=json.dumps(empty))
    )

    assert cleaner.exists(binding) is False

    bound = {
        "Attributes": {
            "Policy": json.dumps(
                {
                    "Statement": [
                        {
                            "Condition": {
                                "ArnEquals": {
                                    "aws:SourceArn": binding.attributes["topic_arn"]
                                }
                            }
                        }
                    ]
                }
            )
        }
    }
    monkeypatch.setattr(
        admin_aws_commands.subprocess, "run", Aws(stdout=json.dumps(bound))
    )

    assert cleaner.exists(binding) is True


def test_a_deleted_queue_makes_its_binding_absent(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(
            returncode=254,
            stderr="An error occurred (AWS.SimpleQueueService.NonExistentQueue) ...",
        ),
    )
    url = "https://sqs.us-east-1.amazonaws.com/123456789012/gpu-fault"

    assert cleaner.exists(_resource("sqs_queue", url)) is False
    assert (
        cleaner.exists(
            _resource("sqs_policy_binding", url, attributes={"topic_arn": "arn:sns:a"})
        )
        is False
    )


def _service_account() -> InstallationResource:
    return _resource(
        "grafana_service_account", "9", attributes={"workspace_id": "g-5b81a13d97"}
    )


def test_a_grafana_service_account_is_looked_up_inside_its_workspace(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Service accounts have no describe call, so presence is the id being listed."""

    listed = json.dumps(
        {
            "serviceAccounts": [
                {"id": "3", "name": "SageMakerObservability", "grafanaRole": "ADMIN"},
                {"id": "9", "name": "gpu-fault-provisioner", "grafanaRole": "ADMIN"},
            ]
        }
    )
    aws = Aws(stdout=listed)
    monkeypatch.setattr(admin_aws_commands.subprocess, "run", aws)

    assert cleaner.exists(_service_account()) is True
    assert aws.calls == [
        [
            "aws",
            "grafana",
            "list-workspace-service-accounts",
            "--region",
            REGION,
            "--workspace-id",
            "g-5b81a13d97",
            "--output",
            "json",
        ]
    ]

    only_theirs = json.dumps(
        {"serviceAccounts": [{"id": "3", "name": "SageMakerObservability"}]}
    )
    monkeypatch.setattr(admin_aws_commands.subprocess, "run", Aws(stdout=only_theirs))
    assert cleaner.exists(_service_account()) is False


def test_a_deleted_grafana_workspace_makes_our_service_account_absent(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(
            returncode=254,
            stdout="",
            stderr="An error occurred (ResourceNotFoundException) ...",
        ),
    )

    assert cleaner.exists(_service_account()) is False

    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(returncode=254, stdout="", stderr="An error occurred (AccessDenied) ..."),
    )
    with pytest.raises(BootstrapError, match="AccessDenied"):
        cleaner.exists(_service_account())


def test_a_helm_release_is_probed_through_the_control_plane_kubeconfig(
    site: Any, cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Helm has no AWS API, so the probe is a cluster call.

    Without the site's kubeconfig it would read whatever ``KUBECONFIG`` happens to
    point at, which is how an uninstall ends up verifying the wrong cluster.
    """

    release = _resource(
        "helm_release",
        "aws-load-balancer-controller",
        attributes={"namespace": "kube-system"},
    )
    aws = Aws()
    monkeypatch.setattr(admin_aws_commands.subprocess, "run", aws)

    assert cleaner.exists(release) is True
    assert aws.calls == [
        [
            "helm",
            "--kubeconfig",
            str(site.release_config["cpu_kubeconfig"]),
            "-n",
            "kube-system",
            "status",
            "aws-load-balancer-controller",
        ]
    ]


def test_an_unreadable_helm_release_is_not_reported_absent(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(returncode=1, stderr="Error: Kubernetes cluster unreachable"),
    )
    release = _resource("helm_release", "aws-load-balancer-controller")

    with pytest.raises(BootstrapError, match="Helm release verification failed"):
        cleaner.exists(release)

    monkeypatch.setattr(
        admin_aws_commands.subprocess,
        "run",
        Aws(returncode=1, stderr="Error: release: not found"),
    )

    assert cleaner.exists(release) is False


def test_every_supported_resource_type_has_a_probe(
    cleaner: ResourceCleaner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A type the registry can hold but the prober cannot read is a trap.

    Uninstall would either crash halfway or, worse, skip the resource; the guard
    turns adding a resource type without a probe into an immediate failure.
    """

    monkeypatch.setattr(admin_aws_commands.subprocess, "run", Aws())
    covered = {resource_type for resource_type, _keywords, _c, _a in PROBES}
    attributes = {
        "cluster_name": "control",
        "hosted_zone_id": ZONE,
        "vpc_id": "vpc-a",
        "vpc_region": REGION,
        "topic_arn": "arn:sns:a",
        "namespace": "kube-system",
        "workspace_id": "g-5b81a13d97",
    }

    for resource_type in sorted(SUPPORTED_RESOURCE_TYPES - covered):
        cleaner.exists(_resource(resource_type, "identifier", attributes=attributes))

    with pytest.raises(BootstrapError, match="unsupported resource verification"):
        cleaner.exists(_resource("quantum_accelerator", "qa-1"))
