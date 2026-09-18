from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.monitoring_policy import SNS_TOPIC_GENERATION_TAG
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
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


@pytest.fixture
def site(tmp_path):
    return load_site(site_file(tmp_path))


@pytest.mark.parametrize("kind", ["sns", "pod-identity", "route-table", "zone-vpc"])
def test_owned_detachment_deletes_only_its_exact_binding_and_proves_absence(
    site, monkeypatch, kind
):
    gone = False

    def remove(_arguments):
        nonlocal gone
        gone = True
        return {}

    if kind == "sns":
        identifier = TOPIC + ":example"
        email = "example@example.invalid"
        entry = resource(
            "sns_subscription",
            identifier,
            policy=Policy.DETACH,
            attributes={
                "endpoint_sha256": hashlib.sha256(
                    email.casefold().encode()
                ).hexdigest(),
                "topic_generation": "example-generation",
            },
        )
        responses = {
            ("sns", "get-subscription-attributes"): lambda _args: absent("NotFound")
            if gone
            else {
                "Attributes": {
                    "SubscriptionArn": identifier,
                    "TopicArn": TOPIC,
                    "Owner": ACCOUNT,
                    "Endpoint": email,
                }
            },
            ("sns", "list-tags-for-resource"): {
                "Tags": [
                    {"Key": SNS_TOPIC_GENERATION_TAG, "Value": "example-generation"}
                ]
            },
            ("sns", "unsubscribe"): remove,
        }
        mutation = "unsubscribe"
    elif kind == "pod-identity":
        role = f"arn:aws:iam::{ACCOUNT}:role/example"
        entry = resource(
            "eks_pod_identity_association",
            "a-example",
            policy=Policy.DETACH,
            attributes={
                "cluster_name": "control",
                "namespace": "gpu-fault-system",
                "service_account": "example",
            },
        )
        responses = {
            ("eks", "describe-pod-identity-association"): lambda _args: absent(
                "ResourceNotFoundException"
            )
            if gone
            else {
                "association": {
                    "associationId": "a-example",
                    "clusterName": "control",
                    "namespace": "gpu-fault-system",
                    "serviceAccount": "example",
                    "roleArn": role,
                }
            },
            ("iam", "get-role"): {"Role": {"Arn": role, "Tags": TAGS}},
            ("eks", "delete-pod-identity-association"): remove,
        }
        mutation = "delete-pod-identity-association"
    elif kind == "route-table":
        entry = resource(
            "ec2_route_table_association", "rtbassoc-example", policy=Policy.DETACH
        )
        responses = {
            ("ec2", "describe-route-tables"): lambda _args: {
                "RouteTables": []
                if gone
                else [
                    {
                        "OwnerId": ACCOUNT,
                        "Tags": TAGS,
                        "Associations": [
                            {
                                "RouteTableAssociationId": "rtbassoc-example",
                                "Main": False,
                            }
                        ],
                    }
                ]
            },
            ("ec2", "disassociate-route-table"): remove,
        }
        mutation = "disassociate-route-table"
    else:
        site.release_config["dns"] = {
            "hosted_zone_id": "ZEXAMPLE",
            "hostname": "api.example.invalid",
        }
        entry = resource(
            "route53_vpc_association",
            f"ZEXAMPLE:{REGION}:vpc-example",
            policy=Policy.DETACH,
            attributes={
                "hosted_zone_id": "ZEXAMPLE",
                "vpc_region": REGION,
                "vpc_id": "vpc-example",
            },
        )
        responses = {
            ("route53", "get-hosted-zone"): lambda _args: {
                "VPCs": [] if gone else [{"VPCId": "vpc-example", "VPCRegion": REGION}]
            },
            ("route53", "disassociate-vpc-from-hosted-zone"): remove,
        }
        mutation = "disassociate-vpc-from-hosted-zone"
    aws = Aws(responses)
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    cleaner = ResourceCleaner(site)
    cleaner.delete(entry)
    cleaner.delete(entry)
    assert gone, "the owned fake binding was never detached"
    assert [call[2] for call in aws.mutations] == [mutation]
    assert not cleaner.exists(entry), "detached binding remains observable"


@pytest.mark.parametrize("disappeared", [False, True])
def test_sqs_binding_disappearance_between_reads_does_not_erase_other_policy(
    site, monkeypatch, disappeared
):
    reads = 0

    def read(_arguments):
        nonlocal reads
        reads += 1
        if reads > 1 and disappeared:
            return absent("QueueDoesNotExist")
        return {
            "Attributes": {
                "Policy": json.dumps(
                    {
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Condition": {
                                    "ArnEquals": {
                                        "aws:SourceArn": TOPIC
                                        if reads == 1
                                        else TOPIC + "-other"
                                    }
                                },
                            }
                        ]
                    }
                )
            }
        }

    aws = Aws({("sqs", "get-queue-attributes"): read})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(
        resource(
            "sqs_policy_binding",
            QUEUE,
            policy=Policy.DETACH,
            attributes={"topic_arn": TOPIC},
        )
    )
    assert aws.mutations == []
    assert reads >= 2


@pytest.mark.parametrize(
    "inventory", ["load-balancer-service", "empty-ingresses", "other-controller"]
)
def test_lbc_cleanup_respects_remaining_service_and_ingress_ownership(
    site, monkeypatch, inventory
):
    calls = []
    alive = True

    def command(arguments, **_options):
        nonlocal alive
        calls.append(list(arguments))
        if arguments[:3] == ["aws", "sts", "get-caller-identity"]:
            document = {"Account": ACCOUNT}
        elif arguments[0] == "helm":
            if "uninstall" in arguments:
                alive = False
                document = {}
            elif not alive:
                return subprocess.CompletedProcess(
                    arguments, 1, "", "Error: release: not found"
                )
            else:
                document = {"clusterName": "control"}
        else:
            assert arguments[0] == "kubectl", "unexpected LBC cleanup transport"
            kind = arguments[arguments.index("get") + 1]
            documents = {
                "service": {
                    "items": [
                        {
                            "metadata": {"name": "customer"},
                            "spec": {"type": "LoadBalancer"},
                        }
                    ]
                    if inventory == "load-balancer-service"
                    else []
                },
                "ingress": {
                    "items": []
                    if inventory == "empty-ingresses"
                    else [
                        {
                            "metadata": {"name": "customer"},
                            "spec": {"ingressClassName": "other"},
                        }
                    ]
                },
                "ingressclass": {
                    "items": [
                        {
                            "metadata": {"name": "other"},
                            "spec": {"controller": "k8s.io/ingress-nginx"},
                        }
                    ]
                },
            }
            document = documents[kind]
        return subprocess.CompletedProcess(arguments, 0, json.dumps(document), "")

    monkeypatch.setattr(aws_commands, "bounded_command", command)
    entry = resource(
        "helm_release",
        "aws-load-balancer-controller",
        attributes={"namespace": "kube-system"},
    )
    if inventory == "load-balancer-service":
        with pytest.raises(BootstrapError, match="remaining Services"):
            ResourceCleaner(site).delete(entry)
        assert alive, "a controller used by a preserved Service was removed"
    else:
        ResourceCleaner(site).delete(entry)
        assert not alive, "an unused owned LBC release was not removed"
    assert sum("uninstall" in call for call in calls) == int(
        inventory != "load-balancer-service"
    )


def test_unused_aurora_parameter_group_deletes_through_the_registered_phase(
    site, monkeypatch
):
    gone = False

    def remove(_arguments):
        nonlocal gone
        gone = True
        return {}

    aws = Aws(
        {
            ("rds", "describe-db-cluster-parameter-groups"): lambda _args: absent(
                "DBParameterGroupNotFound"
            )
            if gone
            else {
                "DBClusterParameterGroups": [{"DBClusterParameterGroupName": "example"}]
            },
            ("rds", "list-tags-for-resource"): {"TagList": TAGS},
            ("rds", "delete-db-cluster-parameter-group"): remove,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    ResourceCleaner(site).delete(resource("rds_cluster_parameter_group", "example"))
    assert [call[2] for call in aws.mutations] == ["delete-db-cluster-parameter-group"]
