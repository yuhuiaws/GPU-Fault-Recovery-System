from __future__ import annotations

import copy
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.grafana import CREATED_TAG_KEY, CREATED_TAG_VALUE
from gpu_fault.admin.monitoring_policy import SNS_TOPIC_GENERATION_TAG
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
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
from tests.admin.test_admin_aws_cleanup_safety import TAGGED
from tests.admin.test_admin_site import site_file


@pytest.fixture
def owner(tmp_path, monkeypatch):
    value = CleanupOwnership(load_site(site_file(tmp_path)))
    transport = Aws()
    monkeypatch.setattr(aws_commands, "bounded_command", transport)
    return value, transport


TAG_PROBES = [
    (kind, identifier, service, operation, field)
    for kind, identifier, service, _probe, operation, field in TAGGED
] + [
    ("aurora_cluster", "database", "rds", "list-tags-for-resource", "TagList"),
    ("aurora_instance", "writer", "rds", "list-tags-for-resource", "TagList"),
]


@pytest.mark.parametrize("kind,identifier,service,operation,field", TAG_PROBES)
@pytest.mark.parametrize("missing", [False, True])
def test_each_tagged_type_requires_live_ownership_proof(
    owner, kind, identifier, service, operation, field, missing
):
    ownership, aws = owner
    item = resource(kind, identifier)
    tags = TAGS + (
        [{"Key": CREATED_TAG_KEY, "Value": CREATED_TAG_VALUE}]
        if kind == "grafana_workspace"
        else []
    )
    if service == "ec2":
        id_field = {
            "security_group": "GroupId",
            "ec2_subnet": "SubnetId",
            "ec2_route_table": "RouteTableId",
            "internet_gateway": "InternetGatewayId",
        }[kind]
        result = {field: [{"OwnerId": ACCOUNT, id_field: identifier, "Tags": tags}]}
    else:
        result = {field: tags}
    not_found = {
        "ecr_repository": "RepositoryNotFoundException",
        "iam_role": "NoSuchEntity",
        "iam_policy": "NoSuchEntity",
        "iam_oidc_provider": "NoSuchEntity",
        "security_group": "InvalidGroup.NotFound",
        "ec2_subnet": "InvalidSubnetID.NotFound",
        "ec2_route_table": "InvalidRouteTableID.NotFound",
        "internet_gateway": "InvalidInternetGatewayID.NotFound",
        "aurora_cluster": "DBClusterNotFoundFault",
        "aurora_instance": "DBInstanceNotFound",
        "rds_db_subnet_group": "DBSubnetGroupNotFoundFault",
        "rds_cluster_parameter_group": "DBParameterGroupNotFound",
        "sns_topic": "NotFound",
        "sqs_queue": "QueueDoesNotExist",
        "ses_email_identity": "NotFoundException",
    }.get(kind, "ResourceNotFoundException")
    aws.responses[(service, operation)] = absent(not_found) if missing else result
    assert ownership.before_delete(item, lambda _item: True) == (
        None if missing else item
    )
    assert aws.mutations == []
    assert aws.calls[-1][1:3] == [service, operation]


@pytest.mark.parametrize("field", ["resource_arn", "resource_id"])
@pytest.mark.parametrize(
    "arn",
    [
        "not-an-arn",
        "arn:aws-cn:ecr:us-east-1:123456789012:repository/example",
        "arn:aws:ecr:us-west-2:123456789012:repository/example",
        "arn:aws:ecr:us-east-1:111122223333:repository/example",
    ],
)
def test_arn_scope_errors_precede_every_transport(owner, field, arn):
    ownership, aws = owner
    if field == "resource_id" and arn == "not-an-arn":
        with pytest.raises(BootstrapError, match="malformed"):
            ownership.validate_arn(arn)
    else:
        with pytest.raises(BootstrapError, match="ARN"):
            ownership.before_delete(
                resource("ecr_repository", "example").model_copy(update={field: arn}),
                lambda _item: True,
            )
    assert aws.calls == []


def test_distinct_resource_id_and_arn_are_not_interchangeable(owner):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="ID and ARN"):
        ownership.before_delete(
            resource("sns_topic", TOPIC, arn=TOPIC + "-other"), lambda _item: True
        )
    assert aws.calls == []


@pytest.mark.parametrize(
    "kind", ["gpu_eks", "gpu_hyperpod", "cpu_eks", "cpu_hyperpod", "rds_snapshot"]
)
def test_general_cleanup_never_authorizes_cluster_or_snapshot_deletion(owner, kind):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="preserved or external"):
        ownership.before_delete(resource(kind, "example"), lambda _item: True)
    assert aws.calls == []


def test_external_cluster_attribute_is_rejected_before_caller_probe(owner):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="outside the site"):
        ownership.before_delete(
            resource("iam_role", "example", attributes={"cluster_name": "unknown"}),
            lambda _item: True,
        )
    assert aws.calls == []


@pytest.mark.parametrize("suffix", ["#fragment", "/", "/other"])
def test_queue_path_or_fragment_cannot_change_scope(owner, suffix):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="SQS queue URL"):
        ownership.validate_scope(resource("sqs_queue", QUEUE + suffix))
    assert aws.calls == []


def test_china_partition_queue_and_global_route53_arns(owner):
    ownership, aws = owner
    site = copy.deepcopy(ownership.site)
    site.release_config.update(
        aws_region="cn-north-1",
        cpu_eks_arn=f"arn:aws-cn:eks:cn-north-1:{ACCOUNT}:cluster/control",
    )
    china = CleanupOwnership(site)
    item = resource(
        "sqs_queue", f"https://sqs.cn-north-1.amazonaws.com.cn/{ACCOUNT}/example"
    ).model_copy(update={"region": "cn-north-1"})
    china.validate_scope(item)
    china.validate_arn("arn:aws-cn:route53:::hostedzone/ZEXAMPLE")
    assert aws.calls == []


def test_caller_identity_is_single_flight_but_failure_is_not_cached(owner):
    ownership, aws = owner
    aws.responses[("sts", "get-caller-identity")] = {}
    with pytest.raises(BootstrapError, match="caller account"):
        ownership.assert_caller()
    aws.responses[("sts", "get-caller-identity")] = {"Account": ACCOUNT}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _index: ownership.assert_caller(), range(12)))
    assert len(aws.calls) == 2


@pytest.mark.parametrize("kind", ["route53_zone", "eks_addon"])
@pytest.mark.parametrize("missing", [False, True])
def test_zone_and_cpu_addon_proofs(owner, kind, missing):
    ownership, aws = owner
    if kind == "route53_zone":
        item = resource(kind, "ZEXAMPLE")
        key = ("route53", "list-tags-for-resource")
        response = {"ResourceTagSet": {"Tags": TAGS}}
        code = "NoSuchHostedZone"
    else:
        item = resource(
            kind, "eks-pod-identity-agent", attributes={"cluster_name": "control"}
        )
        key = ("eks", "describe-addon")
        response = {"addon": {"tags": {"gpu-fault:site-id": SITE}}}
        code = "ResourceNotFoundException"
    aws.responses[key] = absent(code) if missing else response
    assert ownership.before_delete(item, lambda _item: True) == (
        None if missing else item
    )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "identifier,cluster",
    [("other-addon", "control"), ("eks-pod-identity-agent", "gpu-a")],
)
def test_addon_binding_does_not_extend_to_gpu_or_other_addons(
    owner, identifier, cluster
):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="CPU platform binding"):
        ownership.before_delete(
            resource("eks_addon", identifier, attributes={"cluster_name": cluster}),
            lambda _item: True,
        )
    assert [call[1:3] for call in aws.calls] == [["sts", "get-caller-identity"]]


@pytest.mark.parametrize(
    "kind,arn",
    [
        ("ecr_repository", f"arn:aws:ecr:{REGION}:{ACCOUNT}:repository/other"),
        ("aurora_cluster", f"arn:aws:rds:{REGION}:{ACCOUNT}:cluster:other"),
        ("iam_role", f"arn:aws:iam::{ACCOUNT}:role/other"),
    ],
)
def test_same_account_different_resource_name_is_not_authorized(owner, kind, arn):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="name and ARN|ID and ARN"):
        ownership.before_delete(resource(kind, "example", arn=arn), lambda _item: True)
    assert aws.mutations == []
    assert len(aws.calls) == 1


@pytest.mark.parametrize("scenario", ["ok", "missing", "duplicate", "renamed"])
def test_grafana_service_account_binding(owner, scenario):
    ownership, aws = owner
    accounts = [{"id": "sa-example", "name": "site-account"}]
    if scenario == "missing":
        accounts = []
    elif scenario == "duplicate":
        accounts *= 2
    elif scenario == "renamed":
        accounts[0]["name"] = "foreign-account"
    aws.responses[("grafana", "list-workspace-service-accounts")] = {
        "serviceAccounts": accounts
    }
    item = resource(
        "grafana_service_account",
        "sa-example",
        attributes={"workspace_id": "g-example", "name": "site-account"},
    )
    if scenario in {"duplicate", "renamed"}:
        with pytest.raises(BootstrapError, match="identity has drifted"):
            ownership.before_delete(item, lambda _item: True)
    else:
        assert ownership.before_delete(item, lambda _item: True) == (
            None if scenario == "missing" else item
        )
    assert aws.mutations == []


@pytest.mark.parametrize(
    "scenario",
    ["ok", "absent", "multiple", "main", "missing-association", "foreign-owner"],
)
def test_route_table_association_requires_unique_non_main_owned_table(owner, scenario):
    ownership, aws = owner
    table = {
        "OwnerId": ACCOUNT,
        "Tags": TAGS,
        "Associations": [
            {"RouteTableAssociationId": "rtbassoc-example", "Main": False}
        ],
    }
    tables = [table]
    if scenario == "absent":
        tables = []
    elif scenario == "multiple":
        tables *= 2
    elif scenario == "main":
        table["Associations"][0]["Main"] = True
    elif scenario == "missing-association":
        table["Associations"] = []
    elif scenario == "foreign-owner":
        table["OwnerId"] = "111122223333"
    aws.responses[("ec2", "describe-route-tables")] = {"RouteTables": tables}
    item = resource("ec2_route_table_association", "rtbassoc-example")
    if scenario in {"ok", "absent"}:
        assert ownership.before_delete(item, lambda _item: True) == (
            item if scenario == "ok" else None
        )
    else:
        with pytest.raises(BootstrapError, match="association"):
            ownership.before_delete(item, lambda _item: True)
    assert aws.mutations == []


@pytest.mark.parametrize("scenario", ["ok", "absent", "namespace", "role", "tags"])
def test_pod_identity_proves_association_and_role_before_detaching(owner, scenario):
    ownership, aws = owner
    role_arn = f"arn:aws:iam::{ACCOUNT}:role/site"
    association = {
        "associationId": "a-example",
        "clusterName": "control",
        "namespace": "gpu-fault-system",
        "serviceAccount": "example",
        "roleArn": role_arn,
    }
    if scenario == "namespace":
        association["namespace"] = "foreign"
    aws.responses[("eks", "describe-pod-identity-association")] = (
        absent("ResourceNotFoundException")
        if scenario == "absent"
        else {"association": association}
    )
    aws.responses[("iam", "get-role")] = {
        "Role": {
            "Arn": role_arn + "-other" if scenario == "role" else role_arn,
            "Tags": [] if scenario == "tags" else TAGS,
        }
    }
    item = resource(
        "eks_pod_identity_association",
        "a-example",
        attributes={
            "cluster_name": "control",
            "namespace": "gpu-fault-system",
            "service_account": "example",
        },
        policy=Policy.DETACH,
    )
    if scenario in {"ok", "absent"}:
        assert ownership.before_delete(item, lambda _item: True) == (
            item if scenario == "ok" else None
        )
    else:
        with pytest.raises(BootstrapError, match="binding|identity|site-id"):
            ownership.before_delete(item, lambda _item: True)
    assert aws.mutations == []


@pytest.mark.parametrize(
    "scenario", ["ok", "absent", "arn", "topic", "owner", "endpoint", "generation"]
)
def test_subscription_requires_topic_incarnation_and_endpoint_binding(owner, scenario):
    ownership, aws = owner
    arn = TOPIC + ":example"
    endpoint = "Ops@Example.com"
    attributes = {
        "SubscriptionArn": arn,
        "TopicArn": TOPIC,
        "Owner": ACCOUNT,
        "Endpoint": endpoint,
    }
    key = {
        "arn": "SubscriptionArn",
        "topic": "TopicArn",
        "owner": "Owner",
        "endpoint": "Endpoint",
    }.get(scenario)
    if key:
        attributes[key] += "-other"
    aws.responses[("sns", "get-subscription-attributes")] = (
        absent("NotFound") if scenario == "absent" else {"Attributes": attributes}
    )
    aws.responses[("sns", "list-tags-for-resource")] = {
        "Tags": [
            {
                "Key": SNS_TOPIC_GENERATION_TAG,
                "Value": "other" if scenario == "generation" else "example-generation",
            }
        ]
    }
    item = resource(
        "sns_subscription",
        arn,
        attributes={
            "endpoint_sha256": hashlib.sha256(endpoint.casefold().encode()).hexdigest(),
            "topic_generation": "example-generation",
        },
    )
    if scenario in {"ok", "absent"}:
        assert ownership.before_delete(item, lambda _item: True) == (
            item if scenario == "ok" else None
        )
    else:
        with pytest.raises(BootstrapError, match="drifted"):
            ownership.before_delete(item, lambda _item: True)
    assert aws.mutations == []


@pytest.mark.parametrize("kind", ["route53_record", "route53_vpc_association"])
@pytest.mark.parametrize("drift", [False, True])
def test_dns_binding_is_exact(owner, kind, drift):
    ownership, aws = owner
    ownership.config["dns"] = {
        "hosted_zone_id": "ZEXAMPLE",
        "hostname": "api.example.invalid",
    }
    if kind == "route53_record":
        item = resource(
            kind,
            "api.example.invalid.",
            attributes={
                "hosted_zone_id": "ZOTHER" if drift else "ZEXAMPLE",
                "record_type": "CNAME",
            },
        )
    else:
        item = resource(
            kind,
            f"ZEXAMPLE:{REGION}:vpc-example",
            attributes={
                "hosted_zone_id": "ZEXAMPLE",
                "vpc_region": REGION,
                "vpc_id": "vpc-other" if drift else "vpc-example",
            },
        )
    if drift:
        with pytest.raises(BootstrapError, match="binding|identity"):
            ownership.before_delete(item, lambda _item: True)
    else:
        assert ownership.before_delete(item, lambda _item: True) == item
    assert aws.mutations == []


@pytest.mark.parametrize(
    "scenario",
    [
        "ok",
        "name-missing",
        "name-drift",
        "tag-missing",
        "tag-identity",
        "controller-drift",
    ],
)
def test_nlb_name_and_tag_proofs_bind_the_same_incarnation(owner, scenario):
    ownership, aws = owner
    arn = (
        f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/net/example/abc"
    )
    aws.responses[("elbv2", "describe-load-balancers")] = (
        absent("LoadBalancerNotFound")
        if scenario == "name-missing"
        else {
            "LoadBalancers": [
                {
                    "LoadBalancerArn": arn,
                    "LoadBalancerName": "other"
                    if scenario == "name-drift"
                    else "example",
                }
            ]
        }
    )
    tags = (
        TAGS
        if scenario != "controller-drift"
        else [{"Key": "elbv2.k8s.aws/cluster", "Value": "foreign"}]
    )
    aws.responses[("elbv2", "describe-tags")] = (
        absent("LoadBalancerNotFound")
        if scenario == "tag-missing"
        else {
            "TagDescriptions": [
                {
                    "ResourceArn": arn + "-other"
                    if scenario == "tag-identity"
                    else arn,
                    "Tags": tags,
                }
            ]
        }
    )
    item = resource("nlb", "example")
    if scenario in {"name-drift", "tag-identity", "controller-drift"}:
        with pytest.raises(BootstrapError, match="NLB"):
            ownership.before_delete(item, lambda _item: True)
    else:
        result = ownership.before_delete(item, lambda _item: True)
        if scenario == "ok":
            assert result.resource_arn == arn
        else:
            assert result is None
    assert aws.mutations == []


@pytest.mark.parametrize(
    "kind,identifier,arn",
    [
        (
            "nlb",
            "example",
            f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/app/example/a",
        ),
        (
            "nlb_listener",
            f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:listener/app/example/a/b",
            None,
        ),
        ("nlb_listener", f"arn:aws:sns:{REGION}:{ACCOUNT}:example", None),
    ],
)
def test_nlb_resource_shape_cannot_authorize_other_load_balancer_types(
    owner, kind, identifier, arn
):
    ownership, aws = owner
    with pytest.raises(BootstrapError, match="NLB"):
        ownership.before_delete(resource(kind, identifier, arn=arn), lambda _item: True)
    assert aws.mutations == []
    assert len(aws.calls) == 1


def test_helm_binding_validates_cpu_values_through_fake_transport(owner, monkeypatch):
    ownership, _aws = owner
    calls = []

    def command(arguments, **_kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments,
            0,
            '{"Account":"123456789012"}'
            if arguments[0] == "aws"
            else '{"clusterName":"control"}',
            "",
        )

    monkeypatch.setattr(aws_commands, "bounded_command", command)
    item = resource(
        "helm_release",
        "aws-load-balancer-controller",
        attributes={"namespace": "kube-system"},
    )
    assert ownership.before_delete(item, lambda _item: True) == item
    assert calls[-1][:3] == ["helm", "--kubeconfig", ownership.config["cpu_kubeconfig"]]
    assert "uninstall" not in calls[-1]
