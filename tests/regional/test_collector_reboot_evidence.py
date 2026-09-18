"""Read-only COLLECT004 provider proof, including adversarial identity inputs."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpu_fault.hyperpod import HyperPodSubmissionRecord
from scripts.e2e.regional import collector_reboot_evidence as evidence
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)

ACCOUNT = "123456789012"
REGION = "us-west-2"
HP = "approved-hyperpod"
HP_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-unique-resource"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/actual-eks"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/approved/path/executor"
CALLER = f"arn:aws:sts::{ACCOUNT}:assumed-role/executor/pod-session"
INSTANCE = "i-00000000000000001"
OTHER_INSTANCE = "i-00000000000000002"
NODE = f"hyperpod-{INSTANCE}"
LOGICAL = "logical-node-001"
START = datetime(2026, 9, 10, 10, tzinfo=timezone.utc)
END = START + timedelta(minutes=10)
CA = base64.b64encode(b"public-test-ca").decode()
IMAGE = "registry.example/executor@sha256:" + "a" * 64
KEY = "workflow-001/RESTART_NODE/0"


def scope() -> dict[str, Any]:
    return {
        "cluster_id": "managed-gpu",
        "cluster_name": HP,
        "cluster_arn": HP_ARN,
        "eks_cluster_arn": EKS_ARN,
        "region": REGION,
        "account_id": ACCOUNT,
        "node_id": NODE,
        "node_uid": "node-uid-001",
        "boot_id": "boot-id-001",
        "instance_id": INSTANCE,
        "node_logical_id": LOGICAL,
        "executor_role_arn": ROLE,
        "node_recovery": "None",
        "cluster_status": "InService",
        "provider_inventory": [
            {"node_logical_id": LOGICAL, "instance_id": INSTANCE},
            {"node_logical_id": "other-logical", "instance_id": OTHER_INSTANCE},
        ],
    }


def cloudtrail_item() -> dict[str, Any]:
    stamp = (START + timedelta(seconds=4)).isoformat()
    detail = {
        "eventID": "event-001",
        "requestID": "aws-request-independent-001",
        "eventName": "BatchRebootClusterNodes",
        "eventTime": stamp,
        "eventSource": "sagemaker.amazonaws.com",
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
        "requestParameters": {"clusterName": HP, "nodeLogicalIds": [LOGICAL]},
        "responseElements": {"successfulNodeLogicalIds": [LOGICAL], "failed": []},
    }
    return {
        "EventName": "BatchRebootClusterNodes",
        "EventTime": stamp,
        "EventId": "event-001",
        "Username": "legacy-session",
        "CloudTrailEvent": json.dumps(detail),
    }


def inputs() -> dict[str, Any]:
    details = {
        "action": "REBOOT",
        "submission_idempotency_key": KEY,
        "submitted_nodes": [LOGICAL],
    }
    step = {
        "operation": "RESTART_NODE",
        "execution_owner": "gpu-fault-hyperpod-adapter",
        "node_ids": [NODE],
    }
    workflow = {
        "request_id": "workflow-001",
        "incident_id": "incident-001",
        "status": "SUCCEEDED",
        "fencing_token": 4,
        "safety_only": False,
        "official_steps": [step],
        "safety_steps": [],
        "completed_step_indexes": [0],
        "superseded_step_indexes": [],
        "created_at": (START + timedelta(seconds=1)).isoformat(),
        "updated_at": END.isoformat(),
        "step_executions": [
            {
                "operation": "RESTART_NODE",
                "status": "SUCCEEDED",
                "phase": "official",
                "step_index": 0,
                "adapter_operation_id": "remote/command-001",
                "details": deepcopy(details),
                "updated_at": END.isoformat(),
            }
        ],
    }
    command = {
        "command_id": "command-001",
        "cluster_id": "managed-gpu",
        "workflow_request_id": "workflow-001",
        "incident_id": "incident-001",
        "step_index": 0,
        "fencing_token": 4,
        "step": deepcopy(step),
        "status": "SUCCEEDED",
        "result_details": deepcopy(details),
        "created_at": (START + timedelta(seconds=2)).isoformat(),
        "updated_at": END.isoformat(),
    }
    submission = HyperPodSubmissionRecord.model_validate(
        {
            "cluster_name": HP,
            "idempotency_key": KEY,
            "action": "REBOOT",
            "requested_node_identifiers": [NODE],
            "state": "SUBMITTED",
            "created_at": START + timedelta(seconds=3, microseconds=100000),
            "updated_at": START + timedelta(seconds=4, microseconds=200000),
            "result": {
                "operation_id": "hyperpod-op-local-not-aws-id",
                "idempotency_key": KEY,
                "action": "REBOOT",
                "cluster_name": HP,
                "requested_node_logical_ids": [LOGICAL],
                "successful_node_logical_ids": [LOGICAL],
            },
        }
    ).model_dump(mode="json")
    return {
        "scope": scope(),
        "workflow": workflow,
        "commands": [command],
        "submission": submission,
        "events": [dict(evidence.normalize_provider_event(cloudtrail_item()))],
        "started_at": START,
        "ended_at": END,
    }


def set_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    current = root
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def test_success_is_a_bounded_scope_correlation_not_an_invented_request_id_join() -> (
    None
):
    value = inputs()
    proof = evidence.prove_reboot_scope(**value)
    assert proof["valid"] and not proof["errors"], proof
    assert proof["aws_request_id_joined"] is False
    assert proof["request_id"] == "aws-request-independent-001"
    assert proof["operation_id"] == "hyperpod-op-local-not-aws-id"
    assert proof["request_id"] != proof["operation_id"]
    assert proof["binding"] == "unique-scoped-submission-window"
    assert len(proof["limitations"]) == 2
    assert evidence.reboot_evidence_errors(**value) == []
    assert (
        evidence.submitted_reboot_errors(
            value["scope"], value["workflow"], value["commands"], value["submission"]
        )
        == []
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("scope", "cluster_id"), ""),
        (("scope", "node_uid"), None),
        (("scope", "boot_id"), ""),
        (("scope", "instance_id"), "i-short"),
        (("scope", "region"), "us-east-1"),
        (("scope", "account_id"), "111122223333"),
        (("scope", "cluster_arn"), HP),
        (("scope", "eks_cluster_arn"), EKS_ARN.replace("aws:", "aws-cn:")),
        (("scope", "executor_role_arn"), ROLE.replace("iam::", "iam:us-west-2:")),
        (("scope", "node_recovery"), "Automatic"),
        (("scope", "cluster_status"), "Updating"),
        (("scope", "provider_inventory"), []),
        (("scope", "provider_inventory", 0, "instance_id"), OTHER_INSTANCE),
        (("workflow", "status"), "RUNNING"),
        (("workflow", "safety_only"), True),
        (("workflow", "fencing_token"), True),
        (("workflow", "request_id"), "workflow-other"),
        (("workflow", "incident_id"), "incident-other"),
        (("workflow", "official_steps"), []),
        (("workflow", "official_steps", 0, "node_ids"), [NODE, "other-node"]),
        (("workflow", "safety_steps"), [{"operation": "RESTART_NODE"}]),
        (("workflow", "safety_steps"), [{"operation": "REPLACE_NODE"}]),
        (("workflow", "completed_step_indexes"), []),
        (("workflow", "completed_step_indexes"), [True]),
        (("workflow", "completed_step_indexes"), [0, 0]),
        (("workflow", "superseded_step_indexes"), [0]),
        (("workflow", "step_executions"), []),
        (("workflow", "step_executions", 0, "status"), "WAITING"),
        (("workflow", "step_executions", 0, "phase"), None),
        (("workflow", "step_executions", 0, "step_index"), True),
        (("workflow", "step_executions", 0, "adapter_operation_id"), "remote/other"),
        (("workflow", "step_executions", 0, "error"), "failed"),
        (("workflow", "step_executions", 0, "details", "action"), "REPLACE"),
        (("workflow", "step_executions", 0, "details", "submitted_nodes"), []),
        (("workflow", "created_at"), (START - timedelta(seconds=1)).isoformat()),
        (("commands",), []),
        (("commands", 0, "status"), "WAITING"),
        (("commands", 0, "status"), "FAILED"),
        (("commands", 0, "error"), "failure"),
        (("commands", 0, "cluster_id"), "foreign-cluster"),
        (("commands", 0, "workflow_request_id"), "other-workflow"),
        (("commands", 0, "incident_id"), "other-incident"),
        (("commands", 0, "step_index"), True),
        (("commands", 0, "step_index"), 1),
        (("commands", 0, "fencing_token"), 5),
        (("commands", 0, "command_id"), ""),
        (("commands", 0, "step", "node_ids"), [NODE, "other-node"]),
        (("commands", 0, "step", "operation"), "REPLACE_NODE"),
        (("commands", 0, "batched_steps"), [{"step": {"operation": "RESTART_NODE"}}]),
        (("commands", 0, "result_details", "submission_idempotency_key"), "wrong"),
        (("commands", 0, "created_at"), END.isoformat()),
        (("commands", 0, "updated_at"), START.isoformat()),
        (("submission", "state"), "UNKNOWN"),
        (("submission", "state"), "INTENDED"),
        (("submission", "state"), "FAILED"),
        (("submission", "action"), "REPLACE"),
        (("submission", "error"), "failed"),
        (("submission", "cluster_name"), "wrong-cluster"),
        (("submission", "idempotency_key"), "other-key"),
        (("submission", "requested_node_identifiers"), [NODE, "other-node"]),
        (("submission", "requested_node_identifiers"), [NODE, NODE]),
        (("submission", "result"), None),
        (("submission", "result", "submitted"), False),
        (("submission", "result", "duplicate"), "false"),
        (("submission", "result", "operation_id"), None),
        (("submission", "result", "idempotency_key"), "other"),
        (("submission", "result", "cluster_name"), "other"),
        (("submission", "result", "action"), "REPLACE"),
        (("submission", "result", "requested_node_logical_ids"), [LOGICAL, "other"]),
        (("submission", "result", "successful_node_logical_ids"), []),
        (("submission", "result", "successful_node_logical_ids"), [LOGICAL, LOGICAL]),
        (
            ("submission", "result", "failures"),
            [{"message": "PRIVATE-FAILURE-CONTENT"}],
        ),
        (("submission", "created_at"), END.isoformat()),
        (("submission", "updated_at"), "2026-09-10T10:00:04"),
        (("submission", "updated_at"), "invalid-time"),
        (("events",), []),
        (("events", 0, "event_name"), "RebootClusterNodes"),
        (("events", 0, "event_name"), "BatchReplaceClusterNodes"),
        (("events", 0, "event_source"), "other.amazonaws.com"),
        (("events", 0, "aws_region"), "us-east-1"),
        (("events", 0, "recipient_account_id"), "111122223333"),
        (("events", 0, "identity_type"), "IAMUser"),
        (("events", 0, "identity_account_id"), "111122223333"),
        (("events", 0, "identity_arn"), CALLER + "/extra"),
        (("events", 0, "identity_arn"), CALLER.rsplit("/", 1)[0] + "/"),
        (
            ("events", 0, "session_issuer_arn"),
            ROLE.replace("approved/path", "other/path"),
        ),
        (("events", 0, "session_issuer_account_id"), "111122223333"),
        (("events", 0, "session_issuer_type"), "User"),
        (("events", 0, "request_cluster_name"), "wrong-cluster"),
        (("events", 0, "request_cluster_name"), None),
        (("events", 0, "request_cluster_arn"), HP_ARN + "-wrong"),
        (("events", 0, "request_node_logical_ids"), [LOGICAL, "other"]),
        (("events", 0, "request_node_ids"), [OTHER_INSTANCE]),
        (("events", 0, "request_node_ids"), [INSTANCE, OTHER_INSTANCE]),
        (("events", 0, "response_successful_node_logical_ids"), [LOGICAL, "other"]),
        (("events", 0, "response_successful_node_ids"), [OTHER_INSTANCE]),
        (("events", 0, "response_failed_node_count"), 1),
        (("events", 0, "response_failed_node_count"), False),
        (("events", 0, "api_error"), True),
        (("events", 0, "api_error_code"), "AccessDenied"),
        (("events", 0, "api_error_message_present"), True),
        (("events", 0, "event_id"), None),
        (("events", 0, "request_id"), None),
        (("events", 0, "event_time"), START.isoformat()),
        (("events", 0, "event_time"), END.isoformat()),
        (("events", 0, "normalization_errors"), ["malformed"]),
        (("started_at",), START.replace(tzinfo=None)),
        (("ended_at",), START),
    ],
)
def test_final_proof_rejects_wrong_failed_or_ambiguous_inputs(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    value = inputs()
    set_path(value, path, replacement)
    result = evidence.prove_reboot_scope(**value)
    assert result["valid"] is False and result["errors"], (path, result)
    assert "PRIVATE-FAILURE-CONTENT" not in json.dumps(result)
    assert result["aws_request_id_joined"] is False


@pytest.mark.parametrize("key", ["events", "commands"])
def test_duplicate_events_or_commands_do_not_prove_one_reboot(key: str) -> None:
    value = inputs()
    value[key].append(deepcopy(value[key][0]))
    assert evidence.reboot_evidence_errors(**value), (
        "test_duplicate_events_or_commands_do_not_prove_one_reboot: expected evidence.reboot_evidence_errors(**value)"
    )


def test_additional_provider_mutation_is_not_filtered_out_of_strict_proof() -> None:
    value = inputs()
    value["events"].append({"event_name": "BatchDeleteClusterNodes"})
    assert evidence.reboot_evidence_errors(**value), (
        "test_additional_provider_mutation_is_not_filtered_out_of_strict_proof: expected evidence.reboot_evidence_errors(**value)"
    )


def test_cloudtrail_whole_second_precision_does_not_require_fabricated_clock_skew() -> (
    None
):
    value = inputs()
    value["events"][0]["event_time"] = (START + timedelta(seconds=3)).isoformat()
    assert evidence.reboot_evidence_errors(**value) == []
    value["events"][0]["event_time"] = (START + timedelta(seconds=5)).isoformat()
    assert evidence.reboot_evidence_errors(**value), (
        "test_cloudtrail_whole_second_precision_does_not_require_fabricated_clock_skew: expected evidence.reboot_evidence_errors(**value)"
    )


def test_submitted_waiting_reboot_is_usable_without_claiming_it_has_completed() -> None:
    value = inputs()
    value["workflow"]["status"] = "RUNNING"
    value["workflow"]["completed_step_indexes"] = []
    value["workflow"]["step_executions"][0]["status"] = "WAITING"
    value["commands"][0]["status"] = "WAITING"
    assert (
        evidence.submitted_reboot_errors(
            value["scope"], value["workflow"], value["commands"], value["submission"]
        )
        == []
    )
    assert evidence.reboot_evidence_errors(**value), (
        "test_submitted_waiting_reboot_is_usable_without_claiming_it_has_completed: expected evidence.reboot_evidence_errors(**value)"
    )


def reclaimed_submission_inputs(execution_status: str = "WAITING") -> dict[str, Any]:
    value = inputs()
    value["workflow"].update(
        status="RUNNING",
        completed_step_indexes=[],
        updated_at=(START + timedelta(seconds=6)).isoformat(),
    )
    value["workflow"]["step_executions"][0].update(
        status=execution_status, updated_at=(START + timedelta(seconds=5)).isoformat()
    )
    value["commands"][0].update(
        status="LEASED", updated_at=(START + timedelta(seconds=8)).isoformat()
    )
    return value


@pytest.mark.parametrize("execution_status", ["WAITING", "SUCCEEDED"])
def test_reclaimed_submission_preserves_the_earlier_bound_receipt(
    execution_status: str,
) -> None:
    value = reclaimed_submission_inputs(execution_status)
    assert (
        evidence.submitted_reboot_errors(
            value["scope"], value["workflow"], value["commands"], value["submission"]
        )
        == []
    )
    assert (
        evidence.submitted_reboot_errors(
            value["scope"],
            value["workflow"],
            value["commands"],
            value["submission"],
            observed_at=END,
        )
        == []
    )
    assert evidence.reboot_evidence_errors(**value), (
        "test_reclaimed_submission_preserves_the_earlier_bound_receipt: expected evidence.reboot_evidence_errors(**value)"
    )


def test_reclaimed_submission_never_counts_as_final_command_success() -> None:
    value = reclaimed_submission_inputs("SUCCEEDED")
    value["workflow"].update(status="SUCCEEDED", completed_step_indexes=[0])
    errors = evidence.reboot_evidence_errors(**value)
    assert errors == [
        "remote reboot command outcome, identity, fencing or scope differs"
    ]
    proof = evidence.prove_reboot_scope(**value)
    assert proof["valid"] is False
    assert proof["aws_request_id_joined"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("submission",), {}),
        (("submission",), None),
        (("submission", "state"), "INTENDED"),
        (("submission", "state"), "UNKNOWN"),
        (("submission", "state"), "FAILED"),
        (("submission", "action"), "REPLACE"),
        (("submission", "cluster_name"), "foreign-cluster"),
        (("submission", "idempotency_key"), "other-command-key"),
        (("submission", "requested_node_identifiers"), [NODE, "other-node"]),
        (("submission", "requested_node_identifiers"), ["other-node"]),
        (("submission", "error"), "PRIVATE-ERROR"),
        (("submission", "result"), {}),
        (("submission", "result"), None),
        (("submission", "result", "idempotency_key"), "other-command-key"),
        (("submission", "result", "operation_id"), ""),
        (("submission", "result", "submitted"), False),
        (("submission", "result", "requested_node_logical_ids"), ["other-logical"]),
        (("submission", "result", "successful_node_logical_ids"), [LOGICAL, "other"]),
        (("submission", "result", "failures"), [{"message": "PRIVATE-ERROR"}]),
        (("workflow", "status"), "FAILED"),
        (("workflow", "safety_only"), True),
        (("workflow", "fencing_token"), True),
        (("workflow", "superseded_step_indexes"), [0]),
        (("workflow", "official_steps", 0, "node_ids"), ["other-node"]),
        (("workflow", "safety_steps"), [{"operation": "REPLACE_NODE"}]),
        (("workflow", "step_executions"), []),
        (("workflow", "step_executions", 0, "status"), "RUNNING"),
        (("workflow", "step_executions", 0, "status"), "LEASED"),
        (("workflow", "step_executions", 0, "phase"), "safety"),
        (("workflow", "step_executions", 0, "step_index"), 1),
        (("workflow", "step_executions", 0, "adapter_operation_id"), "remote/other"),
        (("workflow", "step_executions", 0, "error"), "PRIVATE-ERROR"),
        (("workflow", "step_executions", 0, "details"), {}),
        (("workflow", "step_executions", 0, "details"), None),
        (
            ("workflow", "step_executions", 0, "details", "submission_idempotency_key"),
            "other-command-key",
        ),
        (("workflow", "step_executions", 0, "details", "action"), "REPLACE"),
        (("workflow", "step_executions", 0, "details", "submitted_nodes"), []),
        (("commands",), []),
        (("commands", 0, "status"), "PENDING"),
        (("commands", 0, "status"), "FAILED"),
        (("commands", 0, "error"), "PRIVATE-ERROR"),
        (("commands", 0, "cluster_id"), "foreign-cluster"),
        (("commands", 0, "workflow_request_id"), "foreign-workflow"),
        (("commands", 0, "incident_id"), "foreign-incident"),
        (("commands", 0, "fencing_token"), 5),
        (("commands", 0, "fencing_token"), True),
        (("commands", 0, "command_id"), "other-command"),
        (("commands", 0, "step_index"), 1),
        (("commands", 0, "step", "node_ids"), [NODE, "other-node"]),
        (("commands", 0, "step", "execution_owner"), "other-owner"),
        (("commands", 0, "batched_steps"), [{"step": {"operation": "RESTART_NODE"}}]),
        (("commands", 0, "result_details"), {}),
        (("commands", 0, "result_details"), None),
        (("commands", 0, "result_details", "submission_idempotency_key"), "other-key"),
        (("commands", 0, "result_details", "action"), "REPLACE"),
        (("commands", 0, "result_details", "submitted_nodes"), [LOGICAL, "other"]),
    ],
)
def test_reclaimed_submission_requires_all_original_receipts_and_bindings(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    value = reclaimed_submission_inputs()
    set_path(value, path, replacement)
    errors = evidence.submitted_reboot_errors(
        value["scope"],
        value["workflow"],
        value["commands"],
        value["submission"],
        observed_at=END,
    )
    assert errors, path
    assert "PRIVATE-ERROR" not in json.dumps(errors)


@pytest.mark.parametrize(
    "path",
    [
        ("workflow", "created_at"),
        ("workflow", "updated_at"),
        ("workflow", "step_executions", 0, "updated_at"),
        ("commands", 0, "created_at"),
        ("commands", 0, "updated_at"),
        ("submission", "created_at"),
        ("submission", "updated_at"),
    ],
)
def test_submitted_reboot_timestamps_cannot_exceed_the_actual_read(
    path: tuple[str | int, ...],
) -> None:
    value = reclaimed_submission_inputs()
    set_path(value, path, (END + timedelta(microseconds=1)).isoformat())
    assert evidence.submitted_reboot_errors(
        value["scope"],
        value["workflow"],
        value["commands"],
        value["submission"],
        observed_at=END,
    ), (
        'test_submitted_reboot_timestamps_cannot_exceed_the_actual_read: expected evidence.submitted_reboot_errors( value["scope"], value["workflo...'
    )


@pytest.mark.parametrize(
    ("observed_at", "accepted"),
    [
        (START, False),
        (START + timedelta(seconds=7, microseconds=999999), False),
        (START + timedelta(seconds=8), True),
        (END.astimezone(timezone(timedelta(hours=8))), True),
    ],
    ids=["before-submission", "before-re-lease", "at-re-lease", "aware-offset"],
)
def test_submitted_reboot_uses_read_time_instead_of_workflow_update(
    observed_at: datetime, accepted: bool
) -> None:
    value = reclaimed_submission_inputs()
    errors = evidence.submitted_reboot_errors(
        value["scope"],
        value["workflow"],
        value["commands"],
        value["submission"],
        observed_at=observed_at,
    )
    assert (not errors) is accepted, errors


@pytest.mark.parametrize(
    "observed_at",
    [END.replace(tzinfo=None), "invalid", END.isoformat(), False, 0, [], {}],
    ids=[
        "naive",
        "invalid-string",
        "string-not-datetime",
        "bool",
        "int",
        "list",
        "dict",
    ],
)
def test_submitted_reboot_rejects_invalid_read_time(observed_at: Any) -> None:
    value = reclaimed_submission_inputs()
    errors = evidence.submitted_reboot_errors(
        value["scope"],
        value["workflow"],
        value["commands"],
        value["submission"],
        observed_at=observed_at,
    )
    assert errors == ["reboot observation must be a timezone-aware timestamp"]


def test_submitted_reboot_rejects_future_read_time() -> None:
    value = reclaimed_submission_inputs()
    errors = evidence.submitted_reboot_errors(
        value["scope"],
        value["workflow"],
        value["commands"],
        value["submission"],
        observed_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    assert errors == ["reboot observation is in the future"]


def test_submitted_reboot_default_read_time_does_not_accept_future_command() -> None:
    value = reclaimed_submission_inputs()
    value["commands"][0]["updated_at"] = (
        datetime.now(timezone.utc) + timedelta(days=1)
    ).isoformat()
    errors = evidence.submitted_reboot_errors(
        value["scope"], value["workflow"], value["commands"], value["submission"]
    )
    assert errors == [
        "workflow/command/execution timestamps do not contain the provider submission"
    ]


@pytest.mark.parametrize("state", ["INTENDED", "UNKNOWN", "FAILED", None])
def test_unconfirmed_submission_never_authorizes_reboot_transport_deferral(
    state: Any,
) -> None:
    value = inputs()
    value["submission"]["state"] = state
    assert evidence.submitted_reboot_errors(
        value["scope"], value["workflow"], value["commands"], value["submission"]
    ), (
        'test_unconfirmed_submission_never_authorizes_reboot_transport_deferral: expected evidence.submitted_reboot_errors( value["scope"], value[...'
    )


def test_normalization_keeps_legacy_fields_but_never_exports_credentials_or_raw_errors() -> (
    None
):
    item = cloudtrail_item()
    result = evidence.normalize_provider_event(item)
    assert result["normalization_errors"] == []
    assert result["event_name"] == item["EventName"]
    assert result["event_time"] == item["EventTime"]
    assert result["username"] == item["Username"]
    assert result["session_issuer_role_name"] == "executor"
    assert result["session_issuer_arn"] == ROLE
    assert result["request_node_logical_ids"] == [LOGICAL]
    assert "SHOULD-NOT-BE-EXPORTED" not in json.dumps(result)
    detail = json.loads(item["CloudTrailEvent"])
    detail.update(errorCode="AccessDenied", errorMessage="PRIVATE-ERROR-CONTENT")
    item["CloudTrailEvent"] = json.dumps(detail)
    result = evidence.normalize_provider_event(item)
    assert result["api_error"] is True
    assert result["api_error_code"] == "AccessDenied"
    assert result["api_error_message_present"] is True
    assert "PRIVATE-ERROR-CONTENT" not in json.dumps(result)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("eventID",), "different"),
        (("eventName",), "ListClusters"),
        (("eventTime",), END.isoformat()),
        (("eventTime",), "no-time"),
        (("requestParameters",), None),
        (("requestParameters", "clusterName"), 123),
        (("requestParameters", "clusterArn"), ""),
        (("requestParameters", "nodeLogicalIds"), [LOGICAL, LOGICAL]),
        (("requestParameters", "nodeLogicalIds"), LOGICAL),
        (("requestParameters", "nodeLogicalIds"), [False]),
        (("requestParameters", "nodeIds"), None),
        (("requestParameters", "allNodes"), True),
        (("responseElements",), None),
        (("responseElements", "successfulNodeLogicalIds"), [LOGICAL, LOGICAL]),
        (("responseElements", "failed"), {}),
        (("responseElements", "failed"), [False]),
        (("userIdentity",), []),
        (("userIdentity", "sessionContext"), "bad"),
        (("userIdentity", "sessionContext", "sessionIssuer"), None),
    ],
)
def test_malformed_cloudtrail_fields_remain_invalid_not_successful(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    set_path(detail, path, replacement)
    item["CloudTrailEvent"] = json.dumps(detail)
    normalized = evidence.normalize_provider_event(item)
    assert normalized["normalization_errors"], (path, normalized)
    value = inputs()
    value["events"] = [normalized]
    assert evidence.reboot_evidence_errors(**value), (
        "test_malformed_cloudtrail_fields_remain_invalid_not_successful: expected evidence.reboot_evidence_errors(**value)"
    )


@pytest.mark.parametrize(
    "raw", ["{", "[]", "null", '{"eventID":1,"eventID":2}', '{"x":NaN}']
)
def test_unparseable_cloudtrail_is_retained_as_an_invalid_mutation(raw: str) -> None:
    item = cloudtrail_item()
    item["CloudTrailEvent"] = raw
    result = evidence.normalize_provider_event(item)
    assert result["event_name"] == "BatchRebootClusterNodes"
    assert result["normalization_errors"] and result["api_error"]


@pytest.mark.parametrize("pascal_case", [False, True])
def test_both_target_forms_must_bind_the_same_single_physical_node(
    pascal_case: bool,
) -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail["requestParameters"].update(clusterArn=HP_ARN, nodeIds=[INSTANCE])
    detail["responseElements"].update(successful=[INSTANCE], failedNodeLogicalIds=[])
    if pascal_case:
        for key in ("requestParameters", "responseElements"):
            detail[key] = {
                name[0].upper() + name[1:]: val for name, val in detail[key].items()
            }
    item["CloudTrailEvent"] = json.dumps(detail)
    value = inputs()
    value["events"] = [evidence.normalize_provider_event(item)]
    assert evidence.reboot_evidence_errors(**value) == []


def test_ambiguous_case_aliases_are_not_silently_preferred() -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail["requestParameters"]["NodeLogicalIds"] = ["foreign"]
    item["CloudTrailEvent"] = json.dumps(detail)
    assert evidence.normalize_provider_event(item)["normalization_errors"]


def test_cluster_arn_as_the_only_request_binding_is_supported() -> None:
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    del detail["requestParameters"]["clusterName"]
    detail["requestParameters"]["clusterArn"] = HP_ARN
    item["CloudTrailEvent"] = json.dumps(detail)
    value = inputs()
    value["events"] = [evidence.normalize_provider_event(item)]
    assert evidence.reboot_evidence_errors(**value) == []


def test_legacy_instance_only_mutation_is_preserved_but_cannot_match_logical_submission() -> (
    None
):
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    detail["requestParameters"] = {"clusterName": HP, "nodeIds": [INSTANCE]}
    detail["responseElements"] = {"successful": [INSTANCE]}
    detail["userIdentity"].pop("sessionContext")
    item["CloudTrailEvent"] = json.dumps(detail)
    normalized = evidence.normalize_provider_event(item)
    assert normalized["normalization_errors"] == []
    assert normalized["request_node_ids"] == [INSTANCE]
    assert normalized["session_issuer_role_name"] == ""
    value = inputs()
    value["events"] = [normalized]
    assert evidence.reboot_evidence_errors(**value), (
        "test_legacy_instance_only_mutation_is_preserved_but_cannot_match_logical_submission: expected evidence.reboot_evidence_errors(**value)"
    )


def test_other_mutation_verbs_remain_visible_without_reboot_request_shape_assumptions() -> (
    None
):
    item = cloudtrail_item()
    detail = json.loads(item["CloudTrailEvent"])
    item["EventName"] = detail["eventName"] = "BatchDeleteClusterNodes"
    detail["requestParameters"]["someOtherVerbOption"] = True
    item["CloudTrailEvent"] = json.dumps(detail)
    normalized = evidence.normalize_provider_event(item)
    assert normalized["event_name"] == "BatchDeleteClusterNodes"
    assert normalized["normalization_errors"] == []
    value = inputs()
    value["events"] = [normalized]
    assert evidence.reboot_evidence_errors(**value), (
        "test_other_mutation_verbs_remain_visible_without_reboot_request_shape_assumptions: expected evidence.reboot_evidence_errors(**value)"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"Events":null}',
        '{"Events":{}}',
        '{"Events":[null]}',
        '{"Events":[],"Events":[]}',
        '{"Events":[],"NextToken":""}',
        "invalid",
    ],
)
def test_incomplete_or_invalid_event_pages_fail_closed(raw: str) -> None:
    with pytest.raises(evidence.RebootEvidenceError):
        evidence.provider_event_page(raw)


def test_page_parser_preserves_continuation_instead_of_claiming_complete_inventory() -> (
    None
):
    items, token = evidence.provider_event_page(
        json.dumps({"Events": [cloudtrail_item()], "NextToken": "opaque-next"})
    )
    assert token == "opaque-next" and len(items) == 1


def pod(name: str) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "namespace": "gpu-fault-system",
            "uid": name + "-uid",
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "ReplicaSet",
                    "controller": True,
                    "name": "executor-rs",
                    "uid": "rs-uid",
                }
            ],
        },
        "spec": {
            "serviceAccountName": "executor-sa",
            "nodeName": NODE,
            "containers": [{"name": "executor", "image": IMAGE}],
        },
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "executor",
                    "ready": True,
                    "containerID": "containerd://" + name,
                    "restartCount": 0,
                    "state": {"running": {"startedAt": START.isoformat()}},
                }
            ],
        },
    }


class ReadOnlyRegional:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            cluster_id="managed-gpu",
            gpu_context="selected-context",
            region=REGION,
            namespace="gpu-fault-system",
        )
        self.calls: list[tuple[str, ...]] = []
        self.scripts: list[str] = []
        self.reads: dict[str, int] = {}
        self.second: dict[str, Any] = {}
        self.data: dict[str, Any] = {
            "cluster": {
                "ClusterName": HP,
                "ClusterArn": HP_ARN,
                "ClusterStatus": "InService",
                "NodeRecovery": "None",
                "Orchestrator": {"Eks": {"ClusterArn": EKS_ARN}},
            },
            "eks": {
                "cluster": {
                    "name": "actual-eks",
                    "arn": EKS_ARN,
                    "status": "ACTIVE",
                    "endpoint": "https://actual.eks.example",
                    "certificateAuthority": {"data": CA},
                }
            },
            "kubeconfig_clusters": [
                {
                    "name": "arbitrary-alias-not-an-arn",
                    "cluster": {
                        "server": "https://actual.eks.example",
                        "certificate-authority-data": CA,
                    },
                }
            ],
            "anchor": {"metadata": {"name": "kube-system", "uid": "eks-namespace-uid"}},
            "node": {
                "metadata": {"name": NODE, "uid": "node-uid-001"},
                "spec": {"providerID": f"aws:///{REGION}a/{INSTANCE}"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "nodeInfo": {"bootID": "boot-id-001"},
                },
            },
            "inventory": {
                "ClusterNodeSummaries": [
                    {
                        "NodeLogicalId": LOGICAL,
                        "InstanceId": INSTANCE,
                        "InstanceStatus": {"Status": "Running"},
                    }
                ]
            },
            "node_detail": {
                "NodeDetails": {"NodeLogicalId": LOGICAL, "InstanceId": INSTANCE}
            },
            "deployment": {
                "metadata": {
                    "name": "gpu-fault-cluster-executor",
                    "uid": "deployment-uid",
                    "namespace": "gpu-fault-system",
                    "generation": 1,
                },
                "spec": {
                    "replicas": 2,
                    "template": {
                        "spec": {
                            "serviceAccountName": "executor-sa",
                            "containers": [{"name": "executor", "image": IMAGE}],
                        }
                    },
                },
                "status": {
                    "observedGeneration": 1,
                    "replicas": 2,
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                    "availableReplicas": 2,
                },
            },
            "pods": {"items": [pod("executor-a"), pod("executor-b")]},
            "replicaset": {
                "metadata": {
                    "name": "executor-rs",
                    "uid": "rs-uid",
                    "namespace": "gpu-fault-system",
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "controller": True,
                            "name": "gpu-fault-cluster-executor",
                            "uid": "deployment-uid",
                        }
                    ],
                }
            },
            "serviceaccount": {
                "metadata": {
                    "name": "executor-sa",
                    "namespace": "gpu-fault-system",
                    "uid": "sa-uid",
                    "annotations": {"eks.amazonaws.com/role-arn": ROLE},
                }
            },
            "environment": {
                "cluster_id": "managed-gpu",
                "hyperpod_cluster": HP,
                "region": REGION,
                "role_arn": ROLE,
                "allow_reboot": True,
                "allow_replace": False,
                "allow_automatic": False,
                "credential_method": "assume-role-with-web-identity",
                "caller_arn": CALLER,
                "caller_account": ACCOUNT,
            },
        }

    def read(self, key: str) -> Any:
        self.reads[key] = self.reads.get(key, 0) + 1
        value = (
            self.second.get(key, self.data[key])
            if self.reads[key] > 1
            else self.data[key]
        )
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "aws"
        assert command[-4:] == ["--region", REGION, "--output", "json"]
        assert kwargs["timeout"] == 180
        self.calls.append(tuple(command))
        keys = {
            ("sagemaker", "describe-cluster"): "cluster",
            ("sagemaker", "list-cluster-nodes"): "inventory",
            ("sagemaker", "describe-cluster-node"): "node_detail",
            ("eks", "describe-cluster"): "eks",
        }
        key = keys[(command[1], command[2])]
        if key == "inventory":
            assert "--include-node-logical-ids" in command
        if key == "node_detail":
            assert command[command.index("--node-logical-id") + 1] == LOGICAL
        return subprocess.CompletedProcess(command, 0, json.dumps(self.read(key)), "")

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        assert plane == "gpu" and kwargs["timeout"] == 60
        self.calls.append(("kubectl", *arguments))
        if arguments[0] == "config":
            assert arguments == (
                "config",
                "view",
                "--minify",
                "--flatten",
                "--raw",
                "-o",
                "jsonpath={.clusters}",
            )
            return json.dumps(self.read("kubeconfig_clusters"))
        if arguments[0] == "exec":
            assert arguments[1] == "-i"
            assert arguments[3:6] == ("-c", "executor", "--")
            assert kwargs["input_text"]
            self.scripts.append(kwargs["input_text"])
            return json.dumps({**self.read("environment"), "pod": arguments[2]})
        assert arguments[0] == "get"
        kinds = {
            "node": "node",
            "namespace": "anchor",
            "deployment": "deployment",
            "pod": "pods",
            "serviceaccount": "serviceaccount",
            "replicaset": "replicaset",
        }
        return json.dumps(self.read(kinds[arguments[1]]))

    def capture(self) -> evidence.RebootScope:
        return evidence.capture_reboot_scope(
            cast(RegionalLiveFixture, self),
            node=NODE,
            hyperpod_cluster=HP,
            executor_role_arn=ROLE,
        )


def test_scope_capture_uses_actual_eks_endpoint_and_full_role_without_mutation() -> (
    None
):
    regional = ReadOnlyRegional()
    result = regional.capture()
    assert result["cluster_name"] == HP and result["cluster_arn"] == HP_ARN
    assert result["eks_cluster_arn"] == EKS_ARN
    assert result["executor_role_arn"] == ROLE and result["account_id"] == ACCOUNT
    assert result["node_id"] == NODE and result["instance_id"] == INSTANCE
    assert result["node_uid"] == "node-uid-001" and result["boot_id"] == "boot-id-001"
    assert result["node_logical_id"] == LOGICAL
    assert result["provider_inventory"] == [
        {"node_logical_id": LOGICAL, "instance_id": INSTANCE}
    ]
    assert len(result["executor_pods"]) == 2
    assert all(
        call[2] in {"describe-cluster", "list-cluster-nodes", "describe-cluster-node"}
        for call in regional.calls
        if call[0] == "aws"
    ), (
        'test_scope_capture_uses_actual_eks_endpoint_and_full_role_without_mutation: expected all( call[2] in {"describe-cluster", "list-cluster-n...'
    )
    assert any(
        call[1:4] == ("eks", "describe-cluster", "--name") and call[4] == "actual-eks"
        for call in regional.calls
    ), (
        'test_scope_capture_uses_actual_eks_endpoint_and_full_role_without_mutation: expected any( call[1:4] == ("eks", "describe-cluster", "--nam...'
    )
    assert "certificate-authority-data" not in json.dumps(result)
    assert CA not in json.dumps(result)
    assert regional.reads["node"] == 2 and regional.reads["cluster"] == 2


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("cluster", "ClusterName"), "foreign"),
        (("cluster", "ClusterArn"), HP_ARN.replace(ACCOUNT, "111122223333")),
        (("cluster", "ClusterArn"), HP_ARN.replace(REGION, "us-east-1")),
        (("cluster", "NodeRecovery"), "Automatic"),
        (("cluster", "NodeRecovery"), None),
        (("cluster", "ClusterStatus"), "Updating"),
        (("cluster", "Orchestrator"), {"Slurm": {}}),
        (("cluster", "Orchestrator"), {"Eks": {"ClusterArn": EKS_ARN}, "Slurm": {}}),
        (
            ("cluster", "Orchestrator", "Eks", "ClusterArn"),
            EKS_ARN.replace(ACCOUNT, "111122223333"),
        ),
        (("eks", "cluster", "arn"), EKS_ARN + "-foreign"),
        (("eks", "cluster", "name"), "foreign-eks"),
        (("eks", "cluster", "status"), "CREATING"),
        (("eks", "cluster", "endpoint"), "https://other.eks.example"),
        (("eks", "cluster", "certificateAuthority", "data"), "not-base64"),
        (("kubeconfig_clusters",), []),
        (("kubeconfig_clusters", 0, "cluster", "insecure-skip-tls-verify"), True),
        (("kubeconfig_clusters", 0, "cluster", "insecure-skip-tls-verify"), 0),
        (("kubeconfig_clusters", 0, "cluster", "tls-server-name"), "other"),
        (("kubeconfig_clusters", 0, "cluster", "proxy-url"), "https://proxy"),
        (("kubeconfig_clusters", 0, "cluster", "certificate-authority-data"), ""),
        (("anchor", "metadata", "uid"), ""),
        (("node", "metadata", "uid"), ""),
        (("node", "metadata", "name"), "other-node"),
        (("node", "metadata", "deletionTimestamp"), START.isoformat()),
        (("node", "status", "nodeInfo", "bootID"), ""),
        (("node", "status", "conditions"), []),
        (("node", "spec", "providerID"), f"aws:///us-east-1a/{INSTANCE}"),
        (("node", "spec", "providerID"), f"aws:///{REGION}a/i-short"),
        (("inventory", "NextToken"), "unconsumed-page"),
        (("inventory", "ClusterNodeSummaries"), []),
        (("inventory", "ClusterNodeSummaries", 0, "NodeLogicalId"), None),
        (("inventory", "ClusterNodeSummaries", 0, "InstanceId"), OTHER_INSTANCE),
        (
            ("inventory", "ClusterNodeSummaries", 0, "InstanceStatus", "Status"),
            "Failure",
        ),
        (("node_detail", "NodeDetails", "NodeLogicalId"), "other"),
        (("node_detail", "NodeDetails", "InstanceId"), OTHER_INSTANCE),
        (("deployment", "metadata", "uid"), ""),
        (("deployment", "metadata", "name"), "foreign"),
        (("deployment", "status", "readyReplicas"), 1),
        (("pods", "items", 0, "metadata", "namespace"), "other-namespace"),
        (("pods", "items", 0, "spec", "serviceAccountName"), "other-sa"),
        (("pods", "items", 0, "spec", "containers", 0, "image"), "other-image"),
        (("pods", "items", 0, "status", "containerStatuses", 0, "ready"), False),
        (("pods", "items", 0, "status", "containerStatuses", 0, "restartCount"), True),
        (("pods", "items", 0, "status", "containerStatuses", 0, "containerID"), ""),
        (("pods", "items", 0, "metadata", "ownerReferences"), []),
        (("replicaset", "metadata", "uid"), "foreign-rs"),
        (("replicaset", "metadata", "ownerReferences", 0, "uid"), "foreign-deployment"),
        (
            ("serviceaccount", "metadata", "annotations", "eks.amazonaws.com/role-arn"),
            ROLE.replace("approved", "foreign"),
        ),
        (("environment", "hyperpod_cluster"), "foreign"),
        (("environment", "cluster_id"), "foreign"),
        (("environment", "region"), "us-east-1"),
        (("environment", "role_arn"), ROLE.replace("approved/path", "foreign/path")),
        (("environment", "allow_reboot"), False),
        (("environment", "allow_reboot"), "true"),
        (("environment", "allow_replace"), True),
        (("environment", "allow_automatic"), True),
        (("environment", "credential_method"), "iam-role"),
        (("environment", "caller_account"), "111122223333"),
        (("environment", "caller_arn"), CALLER.replace("executor/", "other-role/")),
    ],
)
def test_scope_capture_refuses_mismatched_or_incomplete_runtime_identities(
    path: tuple[str | int, ...], replacement: Any
) -> None:
    regional = ReadOnlyRegional()
    set_path(regional.data, path, replacement)
    with pytest.raises(evidence.RebootEvidenceError):
        regional.capture()


@pytest.mark.parametrize(
    ("key", "path", "replacement"),
    [
        ("node", ("metadata", "uid"), "recreated-node"),
        ("node", ("status", "nodeInfo", "bootID"), "changed-boot"),
        ("cluster", ("NodeRecovery",), "Automatic"),
        ("cluster", ("ClusterArn",), HP_ARN + "-other"),
        ("pods", ("items", 0, "metadata", "uid"), "recreated-pod"),
        ("pods", ("items", 0, "status", "containerStatuses", 0, "restartCount"), 1),
        ("serviceaccount", ("metadata", "uid"), "recreated-sa"),
        ("replicaset", ("metadata", "ownerReferences", 0, "uid"), "other-deployment"),
    ],
)
def test_scope_capture_does_not_accept_identity_drift_between_reads(
    key: str, path: tuple[str | int, ...], replacement: Any
) -> None:
    regional = ReadOnlyRegional()
    regional.second[key] = deepcopy(regional.data[key])
    set_path(regional.second[key], path, replacement)
    with pytest.raises(evidence.RebootEvidenceError):
        regional.capture()


def test_normal_pod_status_version_churn_is_not_an_identity_change() -> None:
    regional = ReadOnlyRegional()
    regional.second["pods"] = deepcopy(regional.data["pods"])
    regional.second["pods"]["items"][0]["metadata"]["resourceVersion"] = "new-version"
    regional.second["pods"]["items"][0]["status"]["conditions"][0]["lastProbeTime"] = (
        END.isoformat()
    )
    assert regional.capture()["executor_deployment_uid"] == "deployment-uid"


@pytest.mark.parametrize("logical", [LOGICAL, "second-logical-for-same-instance"])
def test_duplicate_provider_node_mapping_is_rejected(logical: str) -> None:
    regional = ReadOnlyRegional()
    row = deepcopy(regional.data["inventory"]["ClusterNodeSummaries"][0])
    row["NodeLogicalId"] = logical
    regional.data["inventory"]["ClusterNodeSummaries"].append(row)
    with pytest.raises(evidence.RebootEvidenceError):
        regional.capture()


def test_provider_inventory_order_does_not_change_scope_identity() -> None:
    first, second = ReadOnlyRegional(), ReadOnlyRegional()
    extra = {
        "NodeLogicalId": "other-logical",
        "InstanceId": OTHER_INSTANCE,
        "InstanceStatus": {"Status": "Running"},
    }
    first.data["inventory"]["ClusterNodeSummaries"].append(extra)
    second.data["inventory"]["ClusterNodeSummaries"].insert(0, extra)
    before, after = dict(first.capture()), dict(second.capture())
    before.pop("observed_at")
    after.pop("observed_at")
    assert before == after


def test_invalid_fixture_settings_fail_before_any_scope_read() -> None:
    regional = ReadOnlyRegional()
    del regional.settings.cluster_id
    with pytest.raises(
        evidence.RebootEvidenceError, match="identity evidence is malformed"
    ):
        regional.capture()
    assert regional.calls == []


def test_actual_scope_probe_only_queries_sts_and_returns_whitelisted_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    regional = ReadOnlyRegional()
    regional.capture()
    calls: list[tuple[str, dict[str, Any]]] = []

    class Session:
        def get_credentials(self) -> SimpleNamespace:
            return SimpleNamespace(method="assume-role-with-web-identity")

        def client(self, service: str, **kwargs: Any) -> SimpleNamespace:
            calls.append((service, kwargs))
            return SimpleNamespace(
                get_caller_identity=lambda: {"Arn": CALLER, "Account": ACCOUNT}
            )

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=Session))
    for key, value in {
        "GPU_FAULT_HYPERPOD_CLUSTER": HP,
        "GPU_FAULT_CLUSTER_ID": "managed-gpu",
        "AWS_REGION": REGION,
        "AWS_ROLE_ARN": ROLE,
        "HOSTNAME": "executor-a",
        "GPU_FAULT_ALLOW_HYPERPOD_REBOOT": "true",
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
        "GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY": "false",
        "AWS_SECRET_ACCESS_KEY": "PRIVATE-VALUE-NEVER-EXPORTED",
    }.items():
        monkeypatch.setenv(key, value)
    exec(compile(regional.scripts[0], "<scope-probe>", "exec"), {})
    output = capsys.readouterr().out
    observed = json.loads(output)
    assert observed == {**regional.data["environment"], "pod": "executor-a"}
    assert "PRIVATE-VALUE-NEVER-EXPORTED" not in output
    assert len(calls) == 1 and calls[0][0] == "sts"
    config = calls[0][1]["config"]
    assert config.connect_timeout == config.read_timeout == 10
    assert config.retries == {"max_attempts": 0}


@pytest.mark.parametrize("key", ["cluster", "eks", "node", "environment"])
def test_read_failures_are_sanitized_and_never_recast_as_missing_resources(
    key: str,
) -> None:
    regional = ReadOnlyRegional()
    regional.data[key] = RuntimeError("PRIVATE-RAW-CREDENTIAL-ERROR")
    with pytest.raises(evidence.RebootEvidenceError) as exc:
        regional.capture()
    assert "PRIVATE-RAW-CREDENTIAL-ERROR" not in str(exc.value)


def test_provider_fixture_retains_strict_metadata_through_the_waiter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "cpu").write_text("apiVersion: v1\n", encoding="utf-8")
    (tmp_path / "gpu").write_text("apiVersion: v1\n", encoding="utf-8")
    fixture = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu",
            gpu_kubeconfig=tmp_path / "gpu",
            gpu_context="context",
            namespace="gpu-fault-system",
            cluster_id="managed-gpu",
            region=REGION,
        )
    )
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"Events": [cloudtrail_item()]}), ""
        )

    monkeypatch.setattr(fixture, "run", run)
    events = fixture.wait_provider_events(
        START,
        ended_at=END,
        event_names={"BatchRebootClusterNodes"},
        expected_count=1,
        timeout_seconds=0,
    )
    value = inputs()
    value["events"] = events
    assert evidence.reboot_evidence_errors(**value) == []
    assert len(calls) == 1 and events[0]["session_issuer_arn"] == ROLE


@pytest.mark.parametrize("events", [None, {}, [None], [{"EventName": None}]])
def test_provider_fixture_rejects_incomplete_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: Any
) -> None:
    (tmp_path / "cpu").write_text("apiVersion: v1\n", encoding="utf-8")
    (tmp_path / "gpu").write_text("apiVersion: v1\n", encoding="utf-8")
    fixture = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu",
            gpu_kubeconfig=tmp_path / "gpu",
            gpu_context="context",
            namespace="gpu-fault-system",
            cluster_id="managed-gpu",
            region=REGION,
        )
    )
    monkeypatch.setattr(
        fixture,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps({"Events": events}), ""
        ),
    )
    with pytest.raises(RegionalFixtureError):
        fixture.provider_events(START, END)


def test_provider_fixture_rejects_truncated_cli_pagination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "cpu").write_text("apiVersion: v1\n", encoding="utf-8")
    (tmp_path / "gpu").write_text("apiVersion: v1\n", encoding="utf-8")
    fixture = RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu",
            gpu_kubeconfig=tmp_path / "gpu",
            gpu_context="context",
            namespace="gpu-fault-system",
            cluster_id="managed-gpu",
            region=REGION,
        )
    )
    monkeypatch.setattr(
        fixture,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"Events": [cloudtrail_item()], "NextToken": "incomplete"}),
            "",
        ),
    )
    with pytest.raises(RegionalFixtureError, match="incomplete inventory"):
        fixture.provider_events(START, END)
