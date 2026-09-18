from __future__ import annotations

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin._aws_cleanup_support import (
    ACCOUNT,
    QUEUE,
    REGION,
    TAGS,
    TOPIC,
    Aws,
    absent,
    resource,
)
from tests.admin.test_admin_site import site_file

EDGE_CASES = [
    (
        "sns_topic",
        TOPIC,
        "sns",
        "get-topic-attributes",
        "list-tags-for-resource",
        "delete-topic",
        "Tags",
        "NotFound",
    ),
    (
        "sqs_queue",
        QUEUE,
        "sqs",
        "get-queue-attributes",
        "list-queue-tags",
        "delete-queue",
        "Tags",
        "QueueDoesNotExist",
    ),
    (
        "acm_certificate",
        f"arn:aws:acm:{REGION}:{ACCOUNT}:certificate/example",
        "acm",
        "describe-certificate",
        "list-tags-for-certificate",
        "delete-certificate",
        "Tags",
        "ResourceNotFoundException",
    ),
    (
        "secretsmanager_secret",
        "gpu-fault/example",
        "secretsmanager",
        "describe-secret",
        "describe-secret",
        "delete-secret",
        "Tags",
        "ResourceNotFoundException",
    ),
    (
        "ses_email_identity",
        "ops@example.com",
        "sesv2",
        "get-email-identity",
        "get-email-identity",
        "delete-email-identity",
        "Tags",
        "NotFoundException",
    ),
    (
        "rds_db_subnet_group",
        "example",
        "rds",
        "describe-db-subnet-groups",
        "list-tags-for-resource",
        "delete-db-subnet-group",
        "TagList",
        "DBSubnetGroupNotFoundFault",
    ),
]


@pytest.fixture
def site(tmp_path):
    return load_site(site_file(tmp_path))


@pytest.mark.parametrize(
    "kind,identifier,service,probe,tags,delete,tag_field,code", EDGE_CASES
)
def test_edge_resource_delete_requires_fresh_tags_and_proves_absence(
    site, monkeypatch, kind, identifier, service, probe, tags, delete, tag_field, code
):
    deleted = False
    document = {tag_field: TAGS, "Attributes": {}}

    def describe(_arguments):
        return absent(code) if deleted else document

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            (service, tags): {tag_field: TAGS},
            (service, probe): describe,
            (service, delete): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    entry = resource(kind, identifier)
    cleaner.delete(entry)
    cleaner.delete(entry)
    assert [call[2] for call in aws.mutations] == [delete]
    assert not cleaner.exists(entry), "deleted edge resource remains observable"


@pytest.mark.parametrize(
    "kind,identifier,operation,collection,id_field,missing",
    [
        (
            "security_group",
            "sg-example",
            "security-group",
            "SecurityGroups",
            "GroupId",
            "InvalidGroup.NotFound",
        ),
        (
            "ec2_subnet",
            "subnet-example",
            "subnet",
            "Subnets",
            "SubnetId",
            "InvalidSubnetID.NotFound",
        ),
        (
            "ec2_route_table",
            "rtb-example",
            "route-table",
            "RouteTables",
            "RouteTableId",
            "InvalidRouteTableID.NotFound",
        ),
    ],
)
def test_owned_ec2_deletion_is_idempotent(
    site, monkeypatch, kind, identifier, operation, collection, id_field, missing
):
    deleted = False
    probe = "describe-" + operation + "s"

    def describe(_arguments):
        return (
            absent(missing)
            if deleted
            else {
                collection: [{"OwnerId": ACCOUNT, id_field: identifier, "Tags": TAGS}]
            }
        )

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws({("ec2", probe): describe, ("ec2", "delete-" + operation): remove})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    cleaner.delete(resource(kind, identifier))
    assert [call[2] for call in aws.mutations] == ["delete-" + operation]
    assert deleted, "owned EC2 resource never reached its fake delete transport"


@pytest.mark.parametrize("attached", [False, True])
def test_owned_internet_gateway_requires_exclusive_routes_then_detaches_if_needed(
    site, monkeypatch, attached
):
    deleted = False

    def describe(_arguments):
        return (
            absent("InvalidInternetGatewayID.NotFound")
            if deleted
            else {
                "InternetGateways": [
                    {
                        "InternetGatewayId": "igw-example",
                        "OwnerId": ACCOUNT,
                        "Tags": TAGS,
                    }
                ]
            }
        )

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("ec2", "describe-internet-gateways"): describe,
            ("ec2", "describe-route-tables"): {
                "RouteTables": [{"RouteTableId": "rtb-example", "Tags": TAGS}]
            },
            ("ec2", "detach-internet-gateway"): {},
            ("ec2", "delete-internet-gateway"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(
        resource(
            "internet_gateway",
            "igw-example",
            attributes={"vpc_id": "vpc-example"} if attached else {},
        )
    )
    assert [call[2] for call in aws.mutations] == (
        ["detach-internet-gateway"] if attached else []
    ) + ["delete-internet-gateway"]


@pytest.mark.parametrize("gone_at", ["profiles", "inline", "none"])
def test_iam_role_cleans_inline_and_attached_policies_or_stops_when_absent(
    site, monkeypatch, gone_at
):
    deleted = False

    def missing_at(stage, value):
        nonlocal deleted
        if stage == gone_at:
            deleted = True
            return absent("NoSuchEntity")
        return value

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("iam", "get-role"): lambda _args: absent("NoSuchEntity")
            if deleted
            else {},
            ("iam", "list-role-tags"): {"Tags": TAGS},
            ("iam", "list-instance-profiles-for-role"): lambda _args: missing_at(
                "profiles", {"InstanceProfiles": []}
            ),
            ("iam", "list-role-policies"): lambda _args: missing_at(
                "inline", {"PolicyNames": ["example-inline"]}
            ),
            ("iam", "delete-role-policy"): {},
            ("iam", "list-attached-role-policies"): {
                "AttachedPolicies": [
                    {"PolicyArn": f"arn:aws:iam::{ACCOUNT}:policy/example"}
                ]
            },
            ("iam", "detach-role-policy"): {},
            ("iam", "delete-role"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(resource("iam_role", "example"))
    assert [call[2] for call in aws.mutations] == (
        []
        if gone_at != "none"
        else ["delete-role-policy", "detach-role-policy", "delete-role"]
    )


@pytest.mark.parametrize("gone", [False, True])
def test_iam_policy_preserves_default_version_until_policy_deletion(
    site, monkeypatch, gone
):
    arn = f"arn:aws:iam::{ACCOUNT}:policy/example"
    deleted = False

    def entities(_arguments):
        nonlocal deleted
        if gone:
            deleted = True
            return absent("NoSuchEntity")
        return {"PolicyGroups": [], "PolicyUsers": [], "PolicyRoles": []}

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("iam", "get-policy"): lambda _args: absent("NoSuchEntity")
            if deleted
            else {},
            ("iam", "list-policy-tags"): {"Tags": TAGS},
            ("iam", "list-entities-for-policy"): entities,
            ("iam", "list-policy-versions"): {
                "Versions": [
                    {"VersionId": "v1", "IsDefaultVersion": True},
                    {"VersionId": "v2", "IsDefaultVersion": False},
                ]
            },
            ("iam", "delete-policy-version"): {},
            ("iam", "delete-policy"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(resource("iam_policy", arn))
    if gone:
        assert aws.mutations == []
    else:
        assert [call[2] for call in aws.mutations] == [
            "delete-policy-version",
            "delete-policy",
        ]
        assert "v2" in aws.mutations[0] and "v1" not in aws.mutations[0]


@pytest.mark.parametrize(
    "records", [[], [{"Type": "NS"}, {"Type": "SOA"}], [{"Type": "A"}]]
)
def test_hosted_zone_deletion_requires_no_customer_records(site, monkeypatch, records):
    deleted = False

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("route53", "get-hosted-zone"): lambda _args: absent("NoSuchHostedZone")
            if deleted
            else {},
            ("route53", "list-tags-for-resource"): {"ResourceTagSet": {"Tags": TAGS}},
            ("route53", "list-resource-record-sets"): {"ResourceRecordSets": records},
            ("route53", "delete-hosted-zone"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    if any(item["Type"] == "A" for item in records):
        with pytest.raises(BootstrapError, match="non-default records"):
            ResourceCleaner(site).delete(resource("route53_zone", "ZEXAMPLE"))
        assert aws.mutations == []
    else:
        ResourceCleaner(site).delete(resource("route53_zone", "ZEXAMPLE"))
        assert [call[2] for call in aws.mutations] == ["delete-hosted-zone"]


@pytest.mark.parametrize("busy", [False, True])
def test_pod_identity_addon_must_be_unused_before_delete(site, monkeypatch, busy):
    deleted = False

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("eks", "describe-addon"): lambda _args: absent("ResourceNotFoundException")
            if deleted
            else {"addon": {"tags": TAGS}},
            ("eks", "list-pod-identity-associations"): {
                "associations": [{"associationId": "in-use"}] if busy else []
            },
            ("eks", "delete-addon"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    item = resource(
        "eks_addon", "eks-pod-identity-agent", attributes={"cluster_name": "control"}
    )
    if busy:
        with pytest.raises(BootstrapError, match="remaining associations"):
            ResourceCleaner(site).delete(item)
        assert aws.mutations == []
    else:
        ResourceCleaner(site).delete(item)
        assert [call[2] for call in aws.mutations] == ["delete-addon"]


def test_oidc_provider_without_remaining_trust_is_deleted(site, monkeypatch):
    arn = f"arn:aws:iam::{ACCOUNT}:oidc-provider/example.invalid"
    deleted = False

    def remove(_arguments):
        nonlocal deleted
        deleted = True
        return {}

    aws = Aws(
        {
            ("iam", "get-open-id-connect-provider"): lambda _args: absent(
                "NoSuchEntity"
            )
            if deleted
            else {},
            ("iam", "list-open-id-connect-provider-tags"): {"Tags": TAGS},
            ("iam", "list-roles"): {
                "Roles": [
                    {
                        "RoleName": "unrelated",
                        "AssumeRolePolicyDocument": {
                            "Statement": [
                                {"Principal": {"Service": "example.amazonaws.com"}}
                            ]
                        },
                    }
                ]
            },
            ("iam", "delete-open-id-connect-provider"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(resource("iam_oidc_provider", arn))
    assert [call[2] for call in aws.mutations] == ["delete-open-id-connect-provider"]


@pytest.mark.parametrize("kind", ["nlb_listener", "nlb_target_group"])
def test_nlb_children_are_deleted_by_verified_arn(site, monkeypatch, kind):
    suffix = (
        "listener/net/example/a/b"
        if kind == "nlb_listener"
        else "targetgroup/example/a"
    )
    arn = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:{suffix}"
    tagged_arn = (
        f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/net/example/a"
        if kind == "nlb_listener"
        else arn
    )
    probe, delete, missing = (
        ("describe-listeners", "delete-listener", "ListenerNotFound")
        if kind == "nlb_listener"
        else ("describe-target-groups", "delete-target-group", "TargetGroupNotFound")
    )
    deleted = False

    def remove(arguments):
        nonlocal deleted
        assert arn in arguments
        deleted = True
        return {}

    aws = Aws(
        {
            ("elbv2", probe): lambda _args: absent(missing) if deleted else {},
            ("elbv2", "describe-tags"): {
                "TagDescriptions": [{"ResourceArn": tagged_arn, "Tags": TAGS}]
            },
            ("elbv2", delete): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(resource(kind, arn))
    assert [call[2] for call in aws.mutations] == [delete]


def test_unknown_resource_kind_is_refused_before_aws(site, monkeypatch):
    aws = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    with pytest.raises(BootstrapError, match="unsupported"):
        cleaner.validate_supported([resource("unknown", "example")])
    assert aws.calls == []
