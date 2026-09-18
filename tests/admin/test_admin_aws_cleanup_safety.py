from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from tests.admin._aws_cleanup_support import (
    ACCOUNT,
    QUEUE,
    REGION,
    SITE,
    TAGS,
    TOPIC,
    Aws,
    absent,
    resource,
)
from tests.admin.test_admin_site import site_file


@pytest.fixture
def site(tmp_path: Path) -> RenderedSite:
    return load_site(site_file(tmp_path))


@pytest.mark.parametrize(
    "updates",
    [
        {"site_id": "another-site"},
        {"region": "us-west-2"},
        {"region": None},
        {"account_id": "111122223333"},
        {"account_id": None},
        {"provider": "other"},
        {"resource_arn": "arn:aws:ecr:us-west-2:123456789012:repository/test"},
        {"resource_arn": "arn:aws:ecr:us-east-1:111122223333:repository/test"},
        {"ownership": Ownership.EXTERNAL},
        {"delete_policy": Policy.PRESERVE},
        {"ownership": "unknown"},
        {"delete_policy": "unknown"},
    ],
)
def test_scope_or_ownership_mismatch_prevents_every_aws_call(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, updates: dict[str, Any]
) -> None:
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    repository = resource("ecr_repository", "test").model_copy(update=updates)
    with pytest.raises(BootstrapError):
        ResourceCleaner(site).delete(repository)
    assert aws.calls == []


def test_wrong_aws_account_cannot_prove_a_name_only_resource_absent(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws({("sts", "get-caller-identity"): {"Account": "111122223333"}})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="caller account"):
        ResourceCleaner(site).delete(resource("ecr_repository", "test"))
    assert [call[2] for call in aws.calls] == ["get-caller-identity"]


TAGGED = [
    (
        "ecr_repository",
        "test",
        "ecr",
        "describe-repositories",
        "list-tags-for-resource",
        "tags",
    ),
    ("iam_role", "test", "iam", "get-role", "list-role-tags", "Tags"),
    (
        "iam_policy",
        f"arn:aws:iam::{ACCOUNT}:policy/test",
        "iam",
        "get-policy",
        "list-policy-tags",
        "Tags",
    ),
    (
        "iam_oidc_provider",
        f"arn:aws:iam::{ACCOUNT}:oidc-provider/example",
        "iam",
        "get-open-id-connect-provider",
        "list-open-id-connect-provider-tags",
        "Tags",
    ),
    (
        "security_group",
        "sg-test",
        "ec2",
        "describe-security-groups",
        "describe-security-groups",
        "SecurityGroups",
    ),
    (
        "ec2_subnet",
        "subnet-test",
        "ec2",
        "describe-subnets",
        "describe-subnets",
        "Subnets",
    ),
    (
        "ec2_route_table",
        "rtb-test",
        "ec2",
        "describe-route-tables",
        "describe-route-tables",
        "RouteTables",
    ),
    (
        "internet_gateway",
        "igw-test",
        "ec2",
        "describe-internet-gateways",
        "describe-internet-gateways",
        "InternetGateways",
    ),
    (
        "rds_db_subnet_group",
        "test",
        "rds",
        "describe-db-subnet-groups",
        "list-tags-for-resource",
        "TagList",
    ),
    (
        "rds_cluster_parameter_group",
        "test",
        "rds",
        "describe-db-cluster-parameter-groups",
        "list-tags-for-resource",
        "TagList",
    ),
    (
        "acm_certificate",
        f"arn:aws:acm:{REGION}:{ACCOUNT}:certificate/test",
        "acm",
        "describe-certificate",
        "list-tags-for-certificate",
        "Tags",
    ),
    (
        "amp_workspace",
        "ws-test",
        "amp",
        "describe-workspace",
        "list-tags-for-resource",
        "tags",
    ),
    (
        "grafana_workspace",
        "g-test",
        "grafana",
        "describe-workspace",
        "list-tags-for-resource",
        "tags",
    ),
    (
        "sns_topic",
        TOPIC,
        "sns",
        "get-topic-attributes",
        "list-tags-for-resource",
        "Tags",
    ),
    ("sqs_queue", QUEUE, "sqs", "get-queue-attributes", "list-queue-tags", "Tags"),
    (
        "secretsmanager_secret",
        "gpu-fault/pki",
        "secretsmanager",
        "describe-secret",
        "describe-secret",
        "Tags",
    ),
    (
        "ses_email_identity",
        "ops@example.com",
        "sesv2",
        "get-email-identity",
        "get-email-identity",
        "Tags",
    ),
]


@pytest.mark.parametrize(
    ("kind", "identifier", "service", "probe", "tag_operation", "field"), TAGGED
)
@pytest.mark.parametrize(
    "tags", [[], [{"Key": "gpu-fault:site-id", "Value": "another-site"}]]
)
def test_missing_or_retagged_resources_are_never_deleted(
    site: RenderedSite,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    identifier: str,
    service: str,
    probe: str,
    tag_operation: str,
    field: str,
    tags: list[dict[str, str]],
) -> None:
    tag_response: dict[str, Any] = (
        {
            field: [
                {
                    "OwnerId": ACCOUNT,
                    "Tags": tags,
                    {
                        "security_group": "GroupId",
                        "ec2_subnet": "SubnetId",
                        "ec2_route_table": "RouteTableId",
                        "internet_gateway": "InternetGatewayId",
                    }[kind]: identifier,
                }
            ]
        }
        if service == "ec2"
        else {field: tags}
    )
    aws = Aws({(service, probe): {}, (service, tag_operation): tag_response})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="site"):
        ResourceCleaner(site).delete(resource(kind, identifier))
    assert aws.mutations == []


def test_tag_access_denied_is_not_absence(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied = absent("AccessDenied")
    denied.stderr += ": RepositoryNotFoundException is not authorized"
    aws = Aws(
        {
            ("ecr", "describe-repositories"): {},
            ("ecr", "list-tags-for-resource"): denied,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="AccessDenied"):
        ResourceCleaner(site).delete(resource("ecr_repository", "test"))
    assert aws.mutations == []


def test_guarded_ecr_delete_is_idempotent_and_caches_only_caller_identity(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted = False

    def describe(_arguments: list[str]) -> Any:
        return absent("RepositoryNotFoundException") if deleted else {}

    def delete(_arguments: list[str]) -> dict[str, Any]:
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("ecr", "describe-repositories"): describe,
            ("ecr", "list-tags-for-resource"): {"tags": TAGS},
            ("ecr", "delete-repository"): delete,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    repository = resource("ecr_repository", "test")
    cleaner.delete(repository)
    cleaner.delete(repository)
    assert len(aws.mutations) == 1
    assert "--force" in aws.mutations[0]
    assert [call[2] for call in aws.calls].count("get-caller-identity") == 1
    assert [call[2] for call in aws.calls].count("list-tags-for-resource") == 1


def test_a_failed_delete_rechecks_tags_on_retry(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws(
        {
            ("ecr", "describe-repositories"): {},
            ("ecr", "list-tags-for-resource"): {"tags": TAGS},
            ("ecr", "delete-repository"): absent("AccessDenied"),
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    repository = resource("ecr_repository", "test")
    with pytest.raises(BootstrapError, match="AccessDenied"):
        cleaner.delete(repository)
    aws.responses[("ecr", "list-tags-for-resource")] = {"tags": []}
    with pytest.raises(BootstrapError, match="site-id"):
        cleaner.delete(repository)
    assert len(aws.mutations) == 1
    assert [call[2] for call in aws.calls].count("list-tags-for-resource") == 2


def test_iam_instance_profile_users_block_all_role_mutation(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws(
        {
            ("iam", "get-role"): {},
            ("iam", "list-role-tags"): {"Tags": TAGS},
            ("iam", "list-instance-profiles-for-role"): {
                "InstanceProfiles": [{"InstanceProfileName": "customer-training"}]
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="instance profiles"):
        ResourceCleaner(site).delete(resource("iam_role", "test"))
    assert aws.mutations == []


def test_attached_iam_policy_is_not_force_detached(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    arn = f"arn:aws:iam::{ACCOUNT}:policy/lbc"
    aws = Aws(
        {
            ("iam", "get-policy"): {},
            ("iam", "list-policy-tags"): {"Tags": TAGS},
            ("iam", "list-entities-for-policy"): {
                "PolicyGroups": [],
                "PolicyUsers": [],
                "PolicyRoles": [{"RoleName": "lbc"}],
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="IAM policy remains attached"):
        ResourceCleaner(site).delete(resource("iam_policy", arn))
    assert aws.mutations == []


def test_nlb_deletion_is_pinned_to_the_verified_arn(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    arn = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/net/test/old"
    deleted = False
    lookups = 0

    def describe(arguments: list[str]) -> Any:
        nonlocal lookups
        lookups += 1
        if deleted:
            assert "--load-balancer-arns" in arguments
            assert arn in arguments
            return absent("LoadBalancerNotFound")
        if lookups > 2:
            raise AssertionError("unbound name lookup after ownership verification")
        return {"LoadBalancers": [{"LoadBalancerArn": arn, "LoadBalancerName": "test"}]}

    def delete(arguments: list[str]) -> dict[str, Any]:
        nonlocal deleted
        assert arn in arguments
        deleted = True
        return {}

    aws = Aws(
        {
            ("elbv2", "describe-load-balancers"): describe,
            ("elbv2", "describe-tags"): {
                "TagDescriptions": [
                    {
                        "ResourceArn": arn,
                        "Tags": [
                            {"Key": "elbv2.k8s.aws/cluster", "Value": "control"},
                            {
                                "Key": "service.k8s.aws/stack",
                                "Value": "gpu-fault-system/gpu-fault-api-nlb",
                            },
                        ],
                    }
                ]
            },
            ("elbv2", "delete-load-balancer"): delete,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(resource("nlb", "test"))
    assert len(aws.mutations) == 1
    assert lookups == 3


def test_sqs_detach_preserves_unrelated_and_prefix_topic_permissions(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    unrelated = {
        "Sid": TOPIC,
        "Effect": "Allow",
        "Condition": {"ArnEquals": {"aws:SourceArn": TOPIC + "-other"}},
    }
    policy = {
        "Statement": [
            {"Effect": "Allow", "Condition": {"ArnEquals": {"aws:SourceArn": TOPIC}}},
            unrelated,
        ]
    }

    def read(_arguments: list[str]) -> dict[str, Any]:
        return {"Attributes": {"Policy": json.dumps(policy)}}

    def write(arguments: list[str]) -> dict[str, Any]:
        nonlocal policy
        attributes = json.loads(arguments[arguments.index("--attributes") + 1])
        policy = json.loads(attributes["Policy"])
        return {}

    aws = Aws(
        {("sqs", "get-queue-attributes"): read, ("sqs", "set-queue-attributes"): write}
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(
        resource(
            "sqs_policy_binding",
            QUEUE,
            policy=Policy.DETACH,
            attributes={"topic_arn": TOPIC},
        )
    )
    assert policy["Statement"] == [unrelated]
    assert len(aws.mutations) == 1


def test_sqs_shared_statement_requires_explicit_resolution(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = {
        "Statement": {
            "Effect": "Allow",
            "Condition": {"ArnEquals": {"aws:SourceArn": [TOPIC, TOPIC + "-other"]}},
        }
    }
    aws = Aws(
        {
            ("sqs", "get-queue-attributes"): {
                "Attributes": {"Policy": json.dumps(policy)}
            }
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="shares"):
        ResourceCleaner(site).delete(
            resource(
                "sqs_policy_binding",
                QUEUE,
                policy=Policy.DETACH,
                attributes={"topic_arn": TOPIC},
            )
        )
    assert aws.mutations == []


@pytest.mark.parametrize("target", ["hyperpod", "eks"])
def test_cpu_deletion_rejects_gpu_or_rebound_cluster_records(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    hyperpod = resource("cpu_hyperpod", "control", policy=Policy.PRESERVE)
    eks = resource(
        "cpu_eks",
        "control",
        arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/control",
        policy=Policy.PRESERVE,
    )
    if target == "hyperpod":
        hyperpod = hyperpod.model_copy(update={"resource_id": "hp-gpu-a"})
    else:
        eks = eks.model_copy(
            update={"resource_arn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-a"}
        )
    with pytest.raises(BootstrapError, match="CPU deletion targets"):
        ResourceCleaner(site).delete_cpu_cluster(hyperpod, eks)
    assert aws.calls == []


def test_created_grafana_without_creation_tag_is_not_destroyed(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws(
        {
            ("grafana", "describe-workspace"): {},
            ("grafana", "list-tags-for-resource"): {
                "tags": {"gpu-fault:site-id": SITE}
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="creation tag"):
        ResourceCleaner(site).delete(resource("grafana_workspace", "g-test"))
    assert aws.mutations == []


@pytest.mark.parametrize(
    "identifier",
    [
        f"https://sqs.us-west-2.amazonaws.com/{ACCOUNT}/gpu-fault",
        "https://sqs.us-east-1.amazonaws.com/111122223333/gpu-fault",
        f"http://sqs.{REGION}.amazonaws.com/{ACCOUNT}/gpu-fault",
        f"https://example.com/{ACCOUNT}/gpu-fault",
        QUEUE + "?other-account=1",
    ],
)
def test_queue_url_identity_cannot_override_registry_identity(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="SQS queue URL"):
        ResourceCleaner(site).delete(resource("sqs_queue", identifier))
    assert aws.calls == []


def test_shared_ec2_resource_is_not_owned_by_the_caller_account(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws(
        {
            ("ec2", "describe-subnets"): {
                "Subnets": [
                    {"OwnerId": "111122223333", "SubnetId": "subnet-test", "Tags": TAGS}
                ]
            }
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="EC2 resource owner"):
        ResourceCleaner(site).delete(resource("ec2_subnet", "subnet-test"))
    assert aws.mutations == []


def test_shared_sqs_wildcard_cannot_be_reported_as_a_detached_topic(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = {
        "Statement": [
            {
                "Effect": "Allow",
                "Condition": {"ArnLike": {"aws:SourceArn": TOPIC + "*"}},
            }
        ]
    }
    aws = Aws(
        {
            ("sqs", "get-queue-attributes"): {
                "Attributes": {"Policy": json.dumps(policy)}
            }
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="shared wildcard"):
        ResourceCleaner(site).delete(
            resource(
                "sqs_policy_binding",
                QUEUE,
                policy=Policy.DETACH,
                attributes={"topic_arn": TOPIC},
            )
        )
    assert aws.mutations == []


@pytest.mark.parametrize("encoded", [False, True])
def test_trusted_oidc_provider_is_never_deleted(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, encoded: bool
) -> None:
    provider = f"arn:aws:iam::{ACCOUNT}:oidc-provider/issuer.example"
    policy = {"Statement": {"Principal": {"Federated": [provider]}}}
    aws = Aws(
        {
            ("iam", "get-open-id-connect-provider"): {},
            ("iam", "list-open-id-connect-provider-tags"): {"Tags": TAGS},
            ("iam", "list-roles"): {
                "Roles": [
                    {
                        "RoleName": "training",
                        "AssumeRolePolicyDocument": quote(json.dumps(policy))
                        if encoded
                        else policy,
                    }
                ]
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="remaining IAM roles"):
        ResourceCleaner(site).delete(resource("iam_oidc_provider", provider))
    assert aws.mutations == []


def test_unknown_role_trust_cannot_prove_an_oidc_provider_unused(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = f"arn:aws:iam::{ACCOUNT}:oidc-provider/issuer.example"
    aws = Aws(
        {
            ("iam", "get-open-id-connect-provider"): {},
            ("iam", "list-open-id-connect-provider-tags"): {"Tags": TAGS},
            ("iam", "list-roles"): {"Roles": [{"RoleName": "training"}]},
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="trust document"):
        ResourceCleaner(site).delete(resource("iam_oidc_provider", provider))
    assert aws.mutations == []


def test_owned_gateway_with_foreign_routes_is_not_detached(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws(
        {
            ("ec2", "describe-internet-gateways"): {
                "InternetGateways": [
                    {"InternetGatewayId": "igw-test", "OwnerId": ACCOUNT, "Tags": TAGS}
                ]
            },
            ("ec2", "describe-route-tables"): {
                "RouteTables": [{"RouteTableId": "rtb-customer", "Tags": []}]
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="non-solution route tables"):
        ResourceCleaner(site).delete(
            resource("internet_gateway", "igw-test", attributes={"vpc_id": "vpc-test"})
        )
    assert aws.mutations == []


@pytest.mark.parametrize("drift", ["none", "target", "nlb_owner"])
def test_dns_deletion_requires_the_current_owned_nlb_target(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    site.release_config["dns"] = {
        "hosted_zone_id": "Z123",
        "hostname": "control.example",
    }
    name = site.release_config["nlb"]["name"]
    arn = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/net/{name}/id"
    records = [
        {
            "Name": "CONTROL.EXAMPLE.",
            "Type": "CNAME",
            "TTL": 60,
            "ResourceRecords": [
                {"Value": "other.example" if drift == "target" else "nlb.example"}
            ],
        }
    ]

    def delete(_arguments: list[str]) -> dict[str, Any]:
        records.clear()
        return {}

    aws = Aws(
        {
            ("route53", "list-resource-record-sets"): lambda _args: {
                "ResourceRecordSets": records
            },
            ("elbv2", "describe-load-balancers"): {
                "LoadBalancers": [
                    {
                        "LoadBalancerArn": arn,
                        "LoadBalancerName": name,
                        "DNSName": "nlb.example",
                    }
                ]
            },
            ("elbv2", "describe-tags"): {
                "TagDescriptions": [
                    {"ResourceArn": arn, "Tags": [] if drift == "nlb_owner" else TAGS}
                ]
            },
            ("route53", "change-resource-record-sets"): delete,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    record = resource(
        "route53_record",
        "control.example",
        attributes={"hosted_zone_id": "Z123", "record_type": "CNAME"},
    )
    if drift == "none":
        ResourceCleaner(site).delete(record)
        assert not records, "the owned NLB DNS record was not deleted"
        assert [call[2] for call in aws.mutations] == ["change-resource-record-sets"]
    else:
        with pytest.raises(BootstrapError):
            ResourceCleaner(site).delete(record)
        assert records, "DNS records were deleted despite target or ownership drift"
        assert aws.mutations == []


@pytest.mark.parametrize(
    "drift",
    [
        "none",
        "record-target",
        "record-ttl",
        "nlb-incarnation",
        "nlb-read-denied",
        "missing-binding",
        "invalid-binding",
        "record-reappeared",
    ],
)
def test_saved_dns_binding_allows_only_the_original_record_after_nlb_removal(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    site.release_config["dns"] = {
        "hosted_zone_id": "Z123",
        "hostname": "control.example",
    }
    name = site.release_config["nlb"]["name"]
    arn = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/net/{name}/original"
    records = [
        {
            "Name": "control.example.",
            "Type": "CNAME",
            "TTL": 60,
            "ResourceRecords": [{"Value": "original.elb.example"}],
        }
    ]
    nlb_gone = False
    denied = False

    def nlb(_arguments: list[str]) -> Any:
        if denied:
            return absent("AccessDenied")
        if nlb_gone:
            return absent("LoadBalancerNotFound")
        return {
            "LoadBalancers": [
                {
                    "LoadBalancerName": name,
                    "LoadBalancerArn": arn,
                    "DNSName": "original.elb.example",
                }
            ]
        }

    def delete(_arguments: list[str]) -> dict[str, Any]:
        records.clear()
        return {}

    aws = Aws(
        {
            ("route53", "list-resource-record-sets"): lambda _args: {
                "ResourceRecordSets": records
            },
            ("route53", "change-resource-record-sets"): delete,
            ("elbv2", "describe-load-balancers"): nlb,
            ("elbv2", "describe-tags"): lambda _args: {
                "TagDescriptions": [{"ResourceArn": arn, "Tags": TAGS}]
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    record = resource(
        "route53_record", "control.example", attributes={"hosted_zone_id": "Z123"}
    )
    if drift == "record-reappeared":
        original_records = list(records)
        records.clear()
        binding = cleaner.prepare_dns_delete(record)
        records.extend(original_records)
    else:
        binding = cleaner.prepare_dns_delete(record)
    assert aws.mutations == [], "capturing DNS ownership must remain read-only"
    attributes = {**record.attributes, "uninstall_dns_binding": json.dumps(binding)}
    nlb_gone = True
    if drift == "record-target":
        records[0]["ResourceRecords"] = [{"Value": "customer.example"}]
    elif drift == "record-ttl":
        records[0]["TTL"] = 120
    elif drift == "nlb-incarnation":
        nlb_gone = False
        arn += "-replacement"
    elif drift == "nlb-read-denied":
        denied = True
    elif drift == "missing-binding":
        attributes.pop("uninstall_dns_binding")
    elif drift == "invalid-binding":
        attributes["uninstall_dns_binding"] = "null"
    target = record.model_copy(update={"attributes": attributes})
    if drift == "none":
        cleaner.delete(target)
        assert records == [], "the unchanged, previously bound CNAME must be removed"
        assert [call[2] for call in aws.mutations] == ["change-resource-record-sets"], (
            "DNS cleanup must delete only the bound record"
        )
    else:
        with pytest.raises(BootstrapError):
            cleaner.delete(target)
        assert records, "DNS drift or missing proof must preserve the live record"
        assert aws.mutations == [], "DNS drift must block all AWS mutations"


@pytest.mark.parametrize("drift", ["namespace", "role_owner"])
def test_pod_identity_detachment_requires_an_exact_owned_binding(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    role = f"arn:aws:iam::{ACCOUNT}:role/test"
    aws = Aws(
        {
            ("eks", "describe-pod-identity-association"): {
                "association": {
                    "associationId": "assoc-test",
                    "clusterName": "control",
                    "namespace": "training"
                    if drift == "namespace"
                    else "gpu-fault-system",
                    "serviceAccount": "gpu-fault-adot",
                    "roleArn": role,
                }
            },
            ("iam", "get-role"): {
                "Role": {"Arn": role, "Tags": [] if drift == "role_owner" else TAGS}
            },
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError):
        ResourceCleaner(site).delete(
            resource(
                "eks_pod_identity_association",
                "assoc-test",
                policy=Policy.DETACH,
                attributes={
                    "cluster_name": "control",
                    "namespace": "gpu-fault-system",
                    "service_account": "gpu-fault-adot",
                },
            )
        )
    assert aws.mutations == []


def test_grafana_account_id_cannot_authorize_a_renamed_foreign_account(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws(
        {
            ("grafana", "list-workspace-service-accounts"): {
                "serviceAccounts": [{"id": "9", "name": "customer-account"}]
            }
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    with pytest.raises(BootstrapError, match="service account identity"):
        ResourceCleaner(site).delete(
            resource(
                "grafana_service_account",
                "9",
                attributes={"workspace_id": "g-test", "name": "gpu-fault-dashboards"},
            )
        )
    assert aws.mutations == []


@pytest.mark.parametrize("controller", ["ingress.k8s.aws/alb", None])
def test_lbc_cannot_be_removed_with_alb_or_unclassified_ingresses(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, controller: str | None
) -> None:
    calls: list[list[str]] = []

    def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[:3] == ["aws", "sts", "get-caller-identity"]:
            document: dict[str, Any] = {"Account": ACCOUNT}
        elif arguments[0] == "helm":
            assert "uninstall" not in arguments
            document = {"clusterName": "control"}
        else:
            assert arguments[0] == "kubectl"
            kind = arguments[arguments.index("get") + 1]
            documents: dict[str, dict[str, Any]] = {
                "service": {"items": []},
                "ingress": {
                    "items": [
                        {
                            "metadata": {"namespace": "customer", "name": "api"},
                            "spec": {"ingressClassName": "customer-class"},
                        }
                    ]
                },
                "ingressclass": {
                    "items": [
                        {
                            "metadata": {"name": "customer-class"},
                            "spec": {"controller": controller},
                        }
                    ]
                },
            }
            document = documents[kind]
        return subprocess.CompletedProcess(
            arguments, 0, stdout=json.dumps(document), stderr=""
        )

    monkeypatch.setattr(aws_commands, "bounded_command", command)
    with pytest.raises(BootstrapError, match="remaining Ingresses"):
        ResourceCleaner(site).delete(
            resource(
                "helm_release",
                "aws-load-balancer-controller",
                attributes={"namespace": "kube-system"},
            )
        )
    assert not any("uninstall" in call for call in calls), (
        "the load balancer controller was uninstalled while ingresses still needed it"
    )
