"""COLLECT-004 on a plain EKS cluster: scope capture, CloudTrail source, proof.

Live on ``eks-managed`` the COLLECT-004 plan refused "reboot scope preflight
failed: read-only AWS scope query failed": the scope capture described a
SageMaker HyperPod cluster that a managed node group does not have. The
plain-EKS branch describes the EKS cluster and the Node's ACTIVE node group,
binds the EC2 instance behind the Node's providerID, takes the GPU Nodes as
the provider inventory, and the reboot proof reads EC2 ``RebootInstances``
from CloudTrail's ``ec2.amazonaws.com`` source instead of SageMaker's.
HyperPod keeps its SageMaker scope and events byte for byte.
"""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from gpu_fault.hyperpod import HyperPodSubmissionRecord
from scripts.e2e.regional import collector_inventory_reboot as runner
from scripts.e2e.regional import collector_reboot_evidence as evidence
from scripts.e2e.regional.regional_live_fixture import (
    PROVIDER_MUTATIONS,
    RegionalFixtureError,
    RegionalLiveFixture,
)
from tests.regional._cov95_common_live import LiveModel
from tests.regional.test_collector_reboot_evidence import (
    ACCOUNT,
    CALLER,
    EKS_ARN,
    END,
    INSTANCE,
    KEY,
    NODE,
    OTHER_INSTANCE,
    REGION,
    ROLE,
    START,
    ReadOnlyRegional,
    inputs,
    set_path,
)

EKS_NAME = "actual-eks"
NODEGROUP = "gpu-ng"
OTHER_NODE = f"ip-10-0-41-164.{REGION}.compute.internal"


def gpu_node(name: str, instance_id: str, *, gpus: str = "8") -> dict[str, Any]:
    allocatable = {"nvidia.com/gpu": gpus} if gpus else {}
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {"eks.amazonaws.com/nodegroup": NODEGROUP},
        },
        "spec": {"providerID": f"aws:///{REGION}a/{instance_id}"},
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "nodeInfo": {"bootID": f"boot-{name}"},
            "allocatable": allocatable,
        },
    }


class EksReadOnlyRegional(ReadOnlyRegional):
    """The HyperPod fake with a plain-EKS cluster behind it: no SageMaker."""

    def __init__(self) -> None:
        super().__init__()
        del self.data["cluster"], self.data["inventory"], self.data["node_detail"]
        self.data["node"] = gpu_node(NODE, INSTANCE)
        self.data["nodes"] = {
            "items": [
                gpu_node(OTHER_NODE, OTHER_INSTANCE),
                gpu_node(NODE, INSTANCE),
                # A CPU-only node (no GPU allocatable) never enters the inventory.
                gpu_node("cpu-only", OTHER_INSTANCE, gpus=""),
            ]
        }
        self.data["nodegroup"] = {
            "nodegroup": {
                "nodegroupName": NODEGROUP,
                "clusterName": EKS_NAME,
                "status": "ACTIVE",
            }
        }
        self.data["ec2"] = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": INSTANCE,
                            "State": {"Name": "running"},
                            "Tags": [
                                {"Key": "eks:cluster-name", "Value": EKS_NAME},
                                {"Key": "eks:nodegroup-name", "Value": NODEGROUP},
                            ],
                        }
                    ]
                }
            ]
        }
        self.data["environment"]["hyperpod_cluster"] = EKS_NAME

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "aws" and command[1] != "sagemaker", (
            f"a plain EKS scope never asks SageMaker: {command[:3]}"
        )
        assert command[-4:] == ["--region", REGION, "--output", "json"]
        self.calls.append(tuple(command))
        keys = {
            ("eks", "describe-cluster"): "eks",
            ("eks", "describe-nodegroup"): "nodegroup",
            ("ec2", "describe-instances"): "ec2",
        }
        key = keys[(command[1], command[2])]
        if key == "eks":
            assert command[3:5] == ["--name", EKS_NAME]
        if key == "nodegroup":
            assert command[3:7] == [
                "--cluster-name",
                EKS_NAME,
                "--nodegroup-name",
                NODEGROUP,
            ]
        if key == "ec2":
            assert command[3:5] == ["--instance-ids", INSTANCE]
        return subprocess.CompletedProcess(command, 0, json.dumps(self.read(key)), "")

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        if arguments[:2] == ("get", "node") and arguments[2:] == ("-o", "json"):
            self.calls.append(("kubectl", *arguments))
            return json.dumps(self.read("nodes"))
        return super().kubectl(plane, *arguments, **kwargs)

    def capture(self, provider: str | None = None) -> evidence.RebootScope:
        return evidence.capture_reboot_scope(
            cast(RegionalLiveFixture, self),
            node=NODE,
            hyperpod_cluster=EKS_NAME,
            executor_role_arn=ROLE,
            provider=provider,
        )


def eks_scope() -> dict[str, Any]:
    return {
        "provider": "eks-managed",
        "cluster_id": "managed-gpu",
        "cluster_name": EKS_NAME,
        "cluster_arn": EKS_ARN,
        "eks_cluster_arn": EKS_ARN,
        "region": REGION,
        "account_id": ACCOUNT,
        "node_id": NODE,
        "node_uid": "node-uid-001",
        "boot_id": "boot-id-001",
        "instance_id": INSTANCE,
        "node_logical_id": NODE,
        "executor_role_arn": ROLE,
        "node_recovery": "not-applicable",
        "cluster_status": "ACTIVE",
        "provider_inventory": [
            {"node_logical_id": OTHER_NODE, "instance_id": OTHER_INSTANCE},
            {"node_logical_id": NODE, "instance_id": INSTANCE},
        ],
    }


def ec2_cloudtrail_item(*, returned: bool = True) -> dict[str, Any]:
    stamp = (START + timedelta(seconds=4)).isoformat()
    detail = {
        "eventID": "event-ec2-001",
        "requestID": "aws-request-ec2-001",
        "eventName": "RebootInstances",
        "eventTime": stamp,
        "eventSource": "ec2.amazonaws.com",
        "awsRegion": REGION,
        "recipientAccountId": ACCOUNT,
        "userIdentity": {
            "type": "AssumedRole",
            "accountId": ACCOUNT,
            "arn": CALLER,
            "accessKeyId": "SHOULD-NOT-BE-EXPORTED",
            "sessionContext": {
                "sessionIssuer": {"type": "Role", "accountId": ACCOUNT, "arn": ROLE}
            },
        },
        "requestParameters": {"instancesSet": {"items": [{"instanceId": INSTANCE}]}},
        "responseElements": {"requestId": "aws-request-ec2-001", "_return": returned},
    }
    return {
        "EventName": "RebootInstances",
        "EventTime": stamp,
        "EventId": "event-ec2-001",
        "Username": "executor-session",
        "CloudTrailEvent": json.dumps(detail),
    }


KARPENTER_ROLE = f"arn:aws:iam::{ACCOUNT}:role/KarpenterController-synthetic"
KARPENTER_SESSION = f"arn:aws:sts::{ACCOUNT}:assumed-role/KarpenterController-synthetic/eks-cl-karpenter-session"


def karpenter_run_instances_item(
    index: int, *, dry_run: bool = True, role: str = KARPENTER_ROLE
) -> dict[str, Any]:
    """CloudTrail's record of Karpenter's ``RunInstances`` on a plain EKS cluster.

    Karpenter dry-runs the launch about once a minute (``Client.DryRunOperation``,
    no response elements); ``dry_run=False`` is the accepted launch for contrast.
    """

    stamp = (START + timedelta(minutes=5, seconds=index)).isoformat()
    detail: dict[str, Any] = {
        "eventID": f"event-karpenter-{index:03d}",
        "requestID": f"aws-request-karpenter-{index:03d}",
        "eventName": "RunInstances",
        "eventTime": stamp,
        "eventSource": "ec2.amazonaws.com",
        "awsRegion": REGION,
        "recipientAccountId": ACCOUNT,
        "userIdentity": {
            "type": "AssumedRole",
            "accountId": ACCOUNT,
            "arn": KARPENTER_SESSION,
            "accessKeyId": "SHOULD-NOT-BE-EXPORTED",
            "sessionContext": {
                "sessionIssuer": {"type": "Role", "accountId": ACCOUNT, "arn": role}
            },
        },
        "requestParameters": {
            "instancesSet": {"items": [{"minCount": 1, "maxCount": 1}]},
            "instanceType": "p5en.48xlarge",
            "launchTemplate": {"launchTemplateName": "karpenter-synthetic"},
        },
        "responseElements": None,
    }
    if dry_run:
        detail["errorCode"] = "Client.DryRunOperation"
        detail["errorMessage"] = "Request would have succeeded, but DryRun flag is set."
    else:
        detail["responseElements"] = {
            "instancesSet": {"items": [{"instanceId": OTHER_INSTANCE}]}
        }
    return {
        "EventName": "RunInstances",
        "EventTime": stamp,
        "EventId": f"event-karpenter-{index:03d}",
        "Username": "eks-cl-karpenter-session",
        "CloudTrailEvent": json.dumps(detail),
    }


def refused_reboot_item(code: str, *, caller: str, role: str) -> dict[str, Any]:
    item = ec2_cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail.update(
        {
            "eventID": "event-ec2-refused",
            "requestID": "aws-request-ec2-refused",
            "errorCode": code,
            "errorMessage": "PRIVATE-FAILURE-CONTENT",
            "responseElements": None,
        }
    )
    detail["userIdentity"]["arn"] = caller
    detail["userIdentity"]["sessionContext"]["sessionIssuer"]["arn"] = role
    item.update({"EventId": "event-ec2-refused", "CloudTrailEvent": json.dumps(detail)})
    return item


def eks_inputs() -> dict[str, Any]:
    """The HyperPod proof inputs rebound to the EKS provider's records."""

    value = inputs()
    value["scope"] = eks_scope()
    details = {"action": "REBOOT", "submission_idempotency_key": KEY}
    details["submitted_nodes"] = [NODE]
    step = value["workflow"]["official_steps"][0]
    step["execution_owner"] = "gpu-fault-eks-adapter"
    value["workflow"]["step_executions"][0]["details"] = deepcopy(details)
    value["commands"][0]["step"] = deepcopy(step)
    value["commands"][0]["result_details"] = deepcopy(details)
    value["submission"] = HyperPodSubmissionRecord.model_validate(
        {
            "cluster_name": EKS_NAME,
            "idempotency_key": KEY,
            "action": "REBOOT",
            "requested_node_identifiers": [NODE],
            "state": "SUBMITTED",
            "created_at": START + timedelta(seconds=3, microseconds=100000),
            "updated_at": START + timedelta(seconds=4, microseconds=200000),
            "result": {
                "operation_id": "eks-op-local-not-aws-id",
                "idempotency_key": KEY,
                "action": "REBOOT",
                "cluster_name": EKS_NAME,
                "requested_node_logical_ids": [NODE],
                "successful_node_logical_ids": [NODE],
            },
        }
    ).model_dump(mode="json")
    value["events"] = [dict(evidence.normalize_provider_event(ec2_cloudtrail_item()))]
    return value


def test_eks_scope_is_captured_from_eks_ec2_and_the_gpu_nodes_without_sagemaker() -> (
    None
):
    regional = EksReadOnlyRegional()

    result = regional.capture()

    assert result["provider"] == "eks-managed"
    assert result["cluster_name"] == EKS_NAME
    assert result["cluster_arn"] == EKS_ARN == result["eks_cluster_arn"], (
        "a plain EKS cluster has no provider cluster of its own: its ARN is the EKS ARN"
    )
    assert result["node_logical_id"] == NODE, (
        "the EKS provider's submission names the Kubernetes Node, so the logical "
        "id the proof binds is the Node name"
    )
    assert result["instance_id"] == INSTANCE and result["node_id"] == NODE
    assert result["node_recovery"] == "not-applicable"
    assert result["cluster_status"] == "ACTIVE"
    assert result["executor_role_arn"] == ROLE and result["account_id"] == ACCOUNT
    assert result["provider_inventory"] == [
        {"node_logical_id": NODE, "instance_id": INSTANCE},
        {"node_logical_id": OTHER_NODE, "instance_id": OTHER_INSTANCE},
    ], "the GPU Nodes, sorted by name, are the provider inventory; CPU nodes are not"
    assert len(result["executor_pods"]) == 2
    aws = [call[1:3] for call in regional.calls if call[0] == "aws"]
    assert ("eks", "describe-nodegroup") in aws and ("ec2", "describe-instances") in aws
    assert all(call[0] != "sagemaker" for call in aws), aws
    assert regional.capture(provider="eks-managed")["provider"] == "eks-managed"
    assert "certificate-authority-data" not in json.dumps(result)


def test_the_provider_is_read_from_the_node_labels_when_not_given() -> None:
    regional = EksReadOnlyRegional()
    regional.data["node"]["metadata"]["labels"] = {
        "eks.amazonaws.com/nodegroup": NODEGROUP,
        "sagemaker.amazonaws.com/cluster-name": "hp",
    }

    with pytest.raises(RegionalFixtureError):
        # A HyperPod-labelled Node takes the SageMaker branch, which this fake
        # refuses: the label decides, not the node group alone.
        regional.capture()


@pytest.mark.parametrize(
    ("path", "replacement", "reason"),
    [
        (("eks", "cluster", "status"), "UPDATING", "EKS name/ARN/Region/account or"),
        (("eks", "cluster", "name"), "foreign", "EKS name/ARN/Region/account or"),
        (("nodegroup", "nodegroup", "status"), "UPDATING", "ACTIVE managed node group"),
        (("nodegroup", "nodegroup", "clusterName"), "foreign", "managed node group"),
        (
            ("ec2", "Reservations", 0, "Instances", 0, "State", "Name"),
            "stopped",
            "running",
        ),
        (
            ("ec2", "Reservations", 0, "Instances", 0, "Tags", 0, "Value"),
            "foreign",
            "running member of this cluster",
        ),
        (
            ("ec2", "Reservations", 0, "Instances", 0, "InstanceId"),
            OTHER_INSTANCE,
            "running member of this cluster",
        ),
        (("ec2", "Reservations", 0, "Instances"), [], "singleton"),
        (("nodes", "items", 1, "status", "allocatable"), {}, "no unique GPU node"),
        (
            ("nodes", "items", 0, "spec", "providerID"),
            f"aws:///{REGION}a/{INSTANCE}",
            "no unique GPU node",
        ),
        (("node", "metadata", "labels"), {}, "node group label"),
        (("environment", "hyperpod_cluster"), "foreign", "Executor cluster/role"),
        (("environment", "allow_reboot"), False, "Executor cluster/role"),
    ],
)
def test_eks_scope_refuses_drifted_or_incomplete_facts(
    path: tuple[Any, ...], replacement: Any, reason: str
) -> None:
    regional = EksReadOnlyRegional()
    set_path(regional.data, path, replacement)

    with pytest.raises(RegionalFixtureError, match=reason):
        regional.capture(provider="eks-managed")


def test_eks_scope_capture_detects_node_identity_drift_between_reads() -> None:
    regional = EksReadOnlyRegional()
    regional.second["node"] = gpu_node(NODE, OTHER_INSTANCE)

    with pytest.raises(RegionalFixtureError, match="changed during scope capture"):
        regional.capture()


def test_ec2_reboot_instances_normalizes_to_the_instance_singleton() -> None:
    event = evidence.normalize_provider_event(ec2_cloudtrail_item())

    assert event["normalization_errors"] == []
    assert event["event_name"] == "RebootInstances"
    assert event["event_source"] == "ec2.amazonaws.com"
    assert event["request_node_ids"] == [INSTANCE]
    assert event["request_node_logical_ids"] is None
    assert event["request_cluster_name"] is None
    assert event["response_failed_node_count"] == 0
    assert event["response_successful_node_logical_ids"] is None
    assert "SHOULD-NOT-BE-EXPORTED" not in json.dumps(event)

    refused = evidence.normalize_provider_event(ec2_cloudtrail_item(returned=False))
    assert refused["response_failed_node_count"] == 1, (
        "EC2 answers with a boolean; false is a failed reboot of the whole request"
    )


def test_eks_reboot_proof_binds_the_ec2_event_to_the_eks_submission() -> None:
    proof = evidence.prove_reboot_scope(**eks_inputs())

    assert proof["errors"] == [], proof
    assert proof["valid"] is True
    assert proof["event_id"] == "event-ec2-001"
    assert proof["operation_id"] == "eks-op-local-not-aws-id"
    assert proof["submission_idempotency_key"] == KEY
    assert proof["aws_request_id_joined"] is False


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value.update(
                events=[
                    dict(
                        evidence.normalize_provider_event(
                            ec2_cloudtrail_item(returned=False)
                        )
                    )
                ]
            ),
            "CloudTrail response is failed",
        ),
        (
            lambda value: value.update(events=inputs()["events"]),
            "CloudTrail API/Region/account differs",
        ),
        (
            lambda value: set_path(
                value, ("events", 0, "request_node_ids"), [OTHER_INSTANCE]
            ),
            "EC2 request does not match the exact instance singleton",
        ),
        (
            lambda value: set_path(value, ("scope", "node_recovery"), "None"),
            "recovery invariants differ",
        ),
        (
            lambda value: set_path(value, ("scope", "cluster_status"), "InService"),
            "recovery invariants differ",
        ),
        (
            lambda value: set_path(
                value, ("submission", "cluster_name"), "approved-hyperpod"
            ),
            "durable provider submission identity",
        ),
    ],
)
def test_eks_reboot_proof_fails_closed_on_wrong_provider_facts(
    mutate: Any, reason: str
) -> None:
    value = eks_inputs()
    mutate(value)

    proof = evidence.prove_reboot_scope(**value)

    assert proof["valid"] is False
    assert any(reason in error for error in proof["errors"]), proof["errors"]


def test_karpenter_dry_runs_are_refused_calls_beside_the_one_accepted_reboot() -> None:
    """The live attempt-6 inventory: the product's accepted RebootInstances and
    three Karpenter RunInstances dry-runs (``Client.DryRunOperation``), newest
    first. Nothing ran for the dry-runs, so they are refused calls on the record,
    not mutations, and the singleton rule holds for the one accepted call."""

    value = eks_inputs()
    dry_runs = [
        dict(evidence.normalize_provider_event(karpenter_run_instances_item(index)))
        for index in (3, 2, 1)
    ]
    value["events"] = [*dry_runs, *value["events"]]

    proof = evidence.prove_reboot_scope(**value)

    assert proof["errors"] == [], proof
    assert proof["valid"] is True and proof["event_id"] == "event-ec2-001"
    assert [item["event_name"] for item in proof["refused_provider_calls"]] == [
        "RunInstances"
    ] * 3
    assert {item["api_error_code"] for item in proof["refused_provider_calls"]} == {
        "Client.DryRunOperation"
    }
    assert {
        item["session_issuer_role_name"] for item in proof["refused_provider_calls"]
    } == {"KarpenterController-synthetic"}
    for record in proof["refused_provider_calls"]:
        assert record["normalization_errors"] == [], (
            "a launch request's instancesSet is a count, not a malformed target list"
        )
        assert record["request_node_ids"] is None
        assert set(record) == set(evidence.REFUSED_CALL_FIELDS)
    exported = json.dumps(proof)
    assert "SHOULD-NOT-BE-EXPORTED" not in exported
    assert "DryRun flag is set" not in exported, "error messages stay unexported"
    assert evidence.reboot_evidence_errors(**value) == []


def test_a_refused_call_is_reported_but_never_counted_as_the_mutation() -> None:
    """Refused RebootInstances, by a foreign role or by the executor itself, are
    on the record; only the accepted call is the mutation. Alone, a refused
    reboot proves nothing."""

    foreign = refused_reboot_item(
        "Client.UnauthorizedOperation", caller=KARPENTER_SESSION, role=KARPENTER_ROLE
    )
    throttled = refused_reboot_item(
        "Client.RequestLimitExceeded", caller=CALLER, role=ROLE
    )
    value = eks_inputs()
    value["events"] = [
        dict(evidence.normalize_provider_event(foreign)),
        *value["events"],
        dict(evidence.normalize_provider_event(throttled)),
    ]
    proof = evidence.prove_reboot_scope(**value)
    assert proof["valid"] is True and proof["errors"] == [], proof
    assert [
        (item["api_error_code"], item["session_issuer_role_name"])
        for item in proof["refused_provider_calls"]
    ] == [
        ("Client.UnauthorizedOperation", "KarpenterController-synthetic"),
        ("Client.RequestLimitExceeded", "executor"),
    ]
    assert "PRIVATE-FAILURE-CONTENT" not in json.dumps(proof)

    alone = eks_inputs()
    alone["events"] = [dict(evidence.normalize_provider_event(throttled))]
    proof = evidence.prove_reboot_scope(**alone)
    assert proof["valid"] is False
    assert proof["errors"] == [
        "exactly one provider mutation event is required "
        "(0 accepted and 1 refused provider calls observed)"
    ]
    assert len(proof["refused_provider_calls"]) == 1


def test_an_accepted_foreign_launch_still_breaks_the_singleton_rule() -> None:
    """A RunInstances AWS accepted is a mutation in the window whatever its
    target or actor: the inventory is not scoped down to the instance or the
    reboot verb, so the proof fails closed on it."""

    value = eks_inputs()
    launch = evidence.normalize_provider_event(
        karpenter_run_instances_item(7, dry_run=False)
    )
    assert launch["api_error"] is False and launch["normalization_errors"] == []
    value["events"] = [*value["events"], dict(launch)]
    proof = evidence.prove_reboot_scope(**value)
    assert proof["valid"] is False
    assert proof["errors"] == [
        "exactly one provider mutation event is required "
        "(2 accepted and 0 refused provider calls observed)"
    ]
    assert proof["refused_provider_calls"] == []


@pytest.mark.parametrize(
    "event",
    [
        {"event_name": "RunInstances", "api_error": True, "api_error_code": None},
        {"event_name": "RunInstances", "api_error": True, "api_error_code": ""},
        {"event_name": "RunInstances", "api_error_code": 403},
        "not-a-record",
    ],
)
def test_only_a_readable_error_code_makes_a_refused_call(event: Any) -> None:
    """An unreadable record is an invalid mutation, never an absence."""

    assert evidence.refused_provider_call(event) is False
    mutations, refused = evidence.partition_provider_events([event])
    assert mutations == [event] and refused == []


def test_hyperpod_proof_inputs_are_unchanged_by_the_eks_branch() -> None:
    proof = evidence.prove_reboot_scope(**inputs())
    assert proof["valid"] is True and proof["errors"] == [], proof
    assert evidence.scope_provider(inputs()["scope"]) == "hyperpod", (
        "a scope without the provider field is a HyperPod scope"
    )


def test_provider_event_names_and_sources_are_fixed_per_provider() -> None:
    assert evidence.reboot_event_names("eks-managed") == {"RebootInstances"}
    assert evidence.reboot_event_names("hyperpod") == {
        "BatchRebootClusterNodes",
        "RebootClusterNodes",
    }
    assert evidence.reboot_event_names(None) == evidence.reboot_event_names("hyperpod")
    assert evidence.provider_event_source("eks-managed") == "ec2.amazonaws.com"
    assert evidence.provider_event_source(None) == "sagemaker.amazonaws.com"
    assert evidence.scope_provider({"provider": "eks-managed"}) == "eks-managed"
    with pytest.raises(evidence.RebootEvidenceError, match="provider is unknown"):
        evidence.scope_provider({"provider": "bare-metal"})
    assert {"RebootInstances", "TerminateInstances", "StopInstances"} <= (
        PROVIDER_MUTATIONS
    ), "EC2 instance mutations are provider mutations the audits must see"


@pytest.mark.parametrize(
    ("scope", "names", "source"),
    [
        (eks_scope(), {"RebootInstances"}, "ec2.amazonaws.com"),
        (
            inputs()["scope"],
            {"BatchRebootClusterNodes", "RebootClusterNodes"},
            "sagemaker.amazonaws.com",
        ),
    ],
)
def test_collect004_waits_for_the_providers_own_reboot_event(
    scope: dict[str, Any], names: set[str], source: str
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    seen = [{"event_name": "RebootInstances"}]

    def wait(started_at: datetime, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(("wait", kwargs))
        return seen

    def events(
        started_at: datetime, ended_at: datetime, **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls.append(("events", {**kwargs, "ended_at": ended_at}))
        return seen

    regional = cast(
        RegionalLiveFixture,
        type(
            "Regional",
            (),
            {
                "wait_provider_events": staticmethod(wait),
                "provider_events": staticmethod(events),
            },
        )(),
    )
    started = datetime.now(timezone.utc) - timedelta(minutes=1)

    ended_at, result = runner.reboot_provider_events(regional, scope, started)

    assert result == seen and ended_at >= started
    assert calls[0] == (
        "wait",
        {"event_names": names, "expected_count": 1, "event_source": source},
    )
    assert calls[1][0] == "events" and calls[1][1]["event_source"] == source


def test_the_live_fixture_looks_up_the_requested_event_source(tmp_path: Path) -> None:
    fixture = LiveModel(tmp_path)
    fixture.events = [
        {
            "Events": [
                ec2_cloudtrail_item(),
                {**ec2_cloudtrail_item(), "EventName": "DescribeInstances"},
                {**ec2_cloudtrail_item(), "EventName": "TerminateInstances"},
            ]
        }
    ]
    now = datetime.now(timezone.utc)

    result = fixture.provider_events(now, now, event_source="ec2.amazonaws.com")

    assert [item["event_name"] for item in result] == [
        "RebootInstances",
        "TerminateInstances",
    ], "EC2 mutations are kept, EC2 reads are not provider mutations"
    (command, kwargs) = fixture.calls[-1]
    assert "AttributeKey=EventSource,AttributeValue=ec2.amazonaws.com" in command
    assert kwargs["timeout"] == 180

    fixture.provider_events(now, now)
    assert (
        "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com"
        in (fixture.calls[-1][0])
    ), "the default stays the HyperPod source"
    assert END > START, "the proof fixtures keep a non-empty window"
