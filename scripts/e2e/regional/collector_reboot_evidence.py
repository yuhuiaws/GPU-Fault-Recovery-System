"""Bounded, read-only provider scope evidence for COLLECT-004.

The runtime does not persist AWS RequestId. A unique matching CloudTrail event
inside the durable submission interval is a scoped correlation, not an ID join.
No helper in this module authorizes or submits a provider mutation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, NoReturn, TypedDict, cast

from gpu_fault.hyperpod import hyperpod_submission_idempotency_key
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional.guardrail_audit_evidence import complete_pod_population
from scripts.e2e.regional.regional_commands import RegionalFixtureError

if TYPE_CHECKING:
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture


class RebootEvidenceError(RegionalFixtureError):
    """A controlled error with no raw AWS, Kubernetes, or probe output."""


class ProviderEvent(TypedDict):
    event_name: str
    event_time: str
    username: str
    session_issuer_role_name: str
    event_id: str | None
    request_id: str | None
    event_source: str | None
    aws_region: str | None
    recipient_account_id: str | None
    identity_type: str | None
    identity_arn: str | None
    identity_account_id: str | None
    session_issuer_arn: str | None
    session_issuer_account_id: str | None
    session_issuer_type: str | None
    request_cluster_name: str | None
    request_cluster_arn: str | None
    request_node_logical_ids: list[str] | None
    request_node_ids: list[str] | None
    request_dry_run: bool | None
    response_successful_node_logical_ids: list[str] | None
    response_successful_node_ids: list[str] | None
    response_failed_node_count: int | None
    api_error: bool
    api_error_code: str | None
    api_error_message_present: bool
    normalization_errors: list[str]


class RebootScope(TypedDict):
    cluster_id: str
    node_id: str
    node_uid: str
    node_provider_id: str
    boot_id: str
    instance_id: str
    node_logical_id: str
    cluster_name: str
    cluster_arn: str
    eks_cluster_arn: str
    region: str
    account_id: str
    executor_role_arn: str
    provider_inventory: list[dict[str, object]]
    gpu_context: str
    eks_namespace_uid: str
    eks_endpoint_sha256: str
    eks_ca_sha256: str
    executor_deployment_uid: str
    executor_template_sha256: str
    executor_pods: list[dict[str, object]]
    node_recovery: str
    cluster_status: str
    observed_at: str


class RebootProof(TypedDict):
    valid: bool
    errors: list[str]
    binding: str
    aws_request_id_joined: bool
    event_id: str | None
    request_id: str | None
    operation_id: str | None
    submission_idempotency_key: str | None
    submission_started_at: str | None
    submission_completed_at: str | None
    limitations: list[str]


_INSTANCE = re.compile(r"i-[0-9a-f]{17}\Z")
_TEXT = re.compile(r"[A-Za-z0-9_./:@+=,-]{1,512}\Z")
_EXECUTOR = "gpu-fault-cluster-executor"
_REBOOT_EVENTS = frozenset({"BatchRebootClusterNodes", "RebootClusterNodes"})
_JSON_LIMIT = 16 * 1024 * 1024


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RebootEvidenceError(reason)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    result = cast(Mapping[str, Any], value)
    _require(all(isinstance(key, str) for key in result), f"{label} has invalid keys")
    return result


def _rows(value: object, label: str) -> list[Mapping[str, Any]]:
    _require(isinstance(value, list), f"{label} must be a complete list")
    return [_mapping(row, label) for row in cast(list[object], value)]


def _text(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _TEXT.fullmatch(value) is not None,
        f"{label} is missing or malformed",
    )
    return cast(str, value)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and _TEXT.fullmatch(value) else None


def _time(value: object, label: str) -> datetime:
    try:
        stamp = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(cast(str, value))
        )
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        raise RebootEvidenceError(
            f"{label} must be a timezone-aware timestamp"
        ) from None


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "JSON has duplicate fields")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> NoReturn:
    raise RebootEvidenceError("JSON contains a non-finite number")


def _json(text: object, label: str) -> Any:
    _require(
        isinstance(text, str) and len(text) <= _JSON_LIMIT,
        f"{label} is missing or exceeds the read bound",
    )
    try:
        return json.loads(
            cast(str, text),
            object_pairs_hook=_object_pairs,
            parse_constant=_invalid_constant,
        )
    except (ValueError, RecursionError, RebootEvidenceError):
        raise RebootEvidenceError(f"{label} is malformed JSON") from None


def _arn(value: object, service: str) -> tuple[str, str, str, str]:
    parts = _text(value, f"{service} ARN").split(":", 5)
    _require(
        len(parts) == 6
        and parts[0] == "arn"
        and parts[1] in {"aws", "aws-cn", "aws-us-gov"}
        and parts[2] == service
        and re.fullmatch(r"[0-9]{12}", parts[4]) is not None
        and parts[5].startswith("role/" if service == "iam" else "cluster/")
        and len(parts[5].split("/", 1)[1]) > 0,
        f"{service} ARN is malformed",
    )
    return parts[1], parts[3], parts[4], parts[5]


def _strings(value: object, label: str) -> list[str]:
    _require(isinstance(value, list), f"{label} must be a list")
    values = [_text(item, label) for item in cast(list[object], value)]
    _require(len(values) == len(set(values)), f"{label} repeats a target")
    return values


def _alias(mapping: Mapping[str, Any], lower: str, upper: str) -> object:
    keys = [key for key in (lower, upper) if key in mapping]
    _require(len(keys) <= 1, "provider fields contain ambiguous aliases")
    _require(
        not keys or mapping[keys[0]] is not None, "provider field is explicitly null"
    )
    return mapping[keys[0]] if keys else None


def normalize_provider_event(item: Mapping[str, Any]) -> ProviderEvent:
    """Keep legacy fields while exporting only whitelisted CloudTrail facts.

    Malformed details remain visible as an invalid mutation, never as absence.
    Error messages, credentials, request bodies and full events are not exported.
    """

    result: ProviderEvent = {
        "event_name": _optional_text(item.get("EventName")) or "",
        "event_time": _optional_text(item.get("EventTime")) or "",
        "username": _optional_text(item.get("Username")) or "",
        "session_issuer_role_name": "",
        "event_id": None,
        "request_id": None,
        "event_source": None,
        "aws_region": None,
        "recipient_account_id": None,
        "identity_type": None,
        "identity_arn": None,
        "identity_account_id": None,
        "session_issuer_arn": None,
        "session_issuer_account_id": None,
        "session_issuer_type": None,
        "request_cluster_name": None,
        "request_cluster_arn": None,
        "request_node_logical_ids": None,
        "request_node_ids": None,
        "request_dry_run": None,
        "response_successful_node_logical_ids": None,
        "response_successful_node_ids": None,
        "response_failed_node_count": None,
        "api_error": True,
        "api_error_code": None,
        "api_error_message_present": False,
        "normalization_errors": [],
    }
    try:
        detail = _mapping(
            _json(item.get("CloudTrailEvent"), "CloudTrail event"), "event"
        )
        identity = _mapping(detail.get("userIdentity", {}), "CloudTrail identity")
        session = _mapping(identity.get("sessionContext", {}), "CloudTrail session")
        issuer = _mapping(session.get("sessionIssuer", {}), "CloudTrail issuer")
        result.update(
            {
                "event_id": _optional_text(detail.get("eventID")),
                "request_id": _optional_text(detail.get("requestID")),
                "event_source": _optional_text(detail.get("eventSource")),
                "aws_region": _optional_text(detail.get("awsRegion")),
                "recipient_account_id": _optional_text(
                    detail.get("recipientAccountId")
                ),
                "identity_type": _optional_text(identity.get("type")),
                "identity_arn": _optional_text(identity.get("arn")),
                "identity_account_id": _optional_text(identity.get("accountId")),
                "session_issuer_arn": _optional_text(issuer.get("arn")),
                "session_issuer_account_id": _optional_text(issuer.get("accountId")),
                "session_issuer_type": _optional_text(issuer.get("type")),
                "api_error": "errorCode" in detail or "errorMessage" in detail,
                "api_error_code": _optional_text(detail.get("errorCode")),
                "api_error_message_present": "errorMessage" in detail,
            }
        )
        arn = result["session_issuer_arn"] or ""
        if ":role/" in arn:
            result["session_issuer_role_name"] = arn.rsplit("/", 1)[-1]
        _require(
            item.get("EventName") == detail.get("eventName"),
            "CloudTrail lookup/detail event names differ",
        )
        _require(
            result["event_id"] is not None
            and item.get("EventId") == result["event_id"],
            "CloudTrail lookup/detail event IDs differ or are missing",
        )
        _require(
            _time(item.get("EventTime"), "lookup time")
            == _time(detail.get("eventTime"), "event time"),
            "CloudTrail lookup/detail event times differ",
        )
        parameters = _mapping(detail.get("requestParameters"), "provider request")
        cluster_name = _alias(parameters, "clusterName", "ClusterName")
        cluster_arn = _alias(parameters, "clusterArn", "ClusterArn")
        if cluster_name is not None:
            result["request_cluster_name"] = _text(cluster_name, "request cluster name")
        if cluster_arn is not None:
            result["request_cluster_arn"] = _text(cluster_arn, "request cluster ARN")
        logical_ids = _alias(parameters, "nodeLogicalIds", "NodeLogicalIds")
        node_ids = _alias(parameters, "nodeIds", "NodeIds")
        if logical_ids is not None:
            result["request_node_logical_ids"] = _strings(logical_ids, "nodeLogicalIds")
        if node_ids is not None:
            result["request_node_ids"] = _strings(node_ids, "nodeIds")
        dry_run = _alias(parameters, "dryRun", "DryRun")
        if dry_run is not None:
            _require(type(dry_run) is bool, "provider dryRun must be boolean")
            result["request_dry_run"] = cast(bool, dry_run)
        if result["event_name"] in _REBOOT_EVENTS:
            _require(
                not parameters.keys()
                - {
                    "clusterName",
                    "ClusterName",
                    "clusterArn",
                    "ClusterArn",
                    "nodeLogicalIds",
                    "NodeLogicalIds",
                    "nodeIds",
                    "NodeIds",
                    "dryRun",
                    "DryRun",
                },
                "reboot request has unrecognized scope fields",
            )
        response = _mapping(detail.get("responseElements"), "provider response")
        logical_success = _alias(
            response, "successfulNodeLogicalIds", "SuccessfulNodeLogicalIds"
        )
        node_success = _alias(response, "successful", "Successful")
        if logical_success is not None:
            result["response_successful_node_logical_ids"] = _strings(
                logical_success, "successfulNodeLogicalIds"
            )
        if node_success is not None:
            result["response_successful_node_ids"] = _strings(
                node_success, "successful"
            )
        failures = []
        for lower, upper in (
            ("failedNodeLogicalIds", "FailedNodeLogicalIds"),
            ("failed", "Failed"),
        ):
            value = _alias(response, lower, upper)
            if value is not None:
                failures.extend(_rows(value, "provider failures"))
        result["response_failed_node_count"] = len(failures)
    except RebootEvidenceError as exc:
        result["normalization_errors"].append(str(exc))
    return result


def provider_event_page(text: str) -> tuple[list[Mapping[str, Any]], str | None]:
    """Parse a complete lookup page without treating an unreadable page as empty."""
    page = _mapping(_json(text, "CloudTrail page"), "CloudTrail page")
    rows = _rows(page.get("Events"), "CloudTrail Events")
    token = page.get("NextToken")
    if token is not None:
        token = _text(token, "CloudTrail continuation token")
    return rows, token


def _scope_identity(scope: Mapping[str, Any]) -> None:
    for field in (
        "cluster_id",
        "node_id",
        "node_uid",
        "boot_id",
        "node_logical_id",
        "cluster_name",
        "instance_id",
    ):
        _text(scope.get(field), f"scope {field}")
    hp = _arn(scope.get("cluster_arn"), "sagemaker")
    eks = _arn(scope.get("eks_cluster_arn"), "eks")
    role = _arn(scope.get("executor_role_arn"), "iam")
    _require(
        hp[:3] == eks[:3]
        and hp[0] == role[0]
        and hp[2] == role[2] == scope.get("account_id")
        and hp[1] == scope.get("region")
        and role[1] == ""
        and _INSTANCE.fullmatch(scope["instance_id"]) is not None
        and scope.get("node_recovery") == "None"
        and scope.get("cluster_status") == "InService",
        "scope HyperPod/EKS/role/account/Region or recovery invariants differ",
    )
    inventory = _rows(scope.get("provider_inventory"), "scope provider inventory")
    _require(
        [
            row.get("node_logical_id")
            for row in inventory
            if row.get("instance_id") == scope["instance_id"]
        ]
        == [scope["node_logical_id"]]
        and [
            row.get("instance_id")
            for row in inventory
            if row.get("node_logical_id") == scope["node_logical_id"]
        ]
        == [scope["instance_id"]],
        "scope provider inventory does not bind exactly one logical/EC2 node",
    )


def _submission_binding(
    scope: Mapping[str, Any],
    workflow: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any],
    started_at: datetime,
    ended_at: datetime,
    *,
    require_complete: bool = True,
) -> tuple[Mapping[str, Any], datetime, datetime]:
    _require(
        workflow.get("status")
        in ({"SUCCEEDED"} if require_complete else {"RUNNING", "SUCCEEDED"}),
        "reboot workflow is not in the required execution state",
    )
    _require(workflow.get("safety_only") is False, "reboot workflow phase is ambiguous")
    request_id = _text(workflow.get("request_id"), "workflow ID")
    incident_id = _text(workflow.get("incident_id"), "incident ID")
    official = _rows(workflow.get("official_steps"), "official steps")
    safety = _rows(workflow.get("safety_steps"), "safety steps")
    reboots = [
        (index, row)
        for index, row in enumerate(official)
        if row.get("operation") == "RESTART_NODE"
    ]
    _require(
        len(reboots) == 1
        and not any(row.get("operation") == "RESTART_NODE" for row in safety)
        and not any(
            row.get("operation") == "REPLACE_NODE" for row in [*official, *safety]
        ),
        "workflow does not contain exactly one unambiguous reboot without replacement",
    )
    index, step = reboots[0]
    _require(
        step.get("node_ids") == [scope["node_id"]], "workflow reboot scope differs"
    )
    _require(
        type(workflow.get("fencing_token")) is int and workflow["fencing_token"] >= 1,
        "workflow fencing token is invalid",
    )
    _require(
        (
            not require_complete
            or index in _step_indexes(workflow.get("completed_step_indexes"))
        )
        and index not in _step_indexes(workflow.get("superseded_step_indexes")),
        "workflow reboot is incomplete or superseded",
    )
    candidates = []
    for raw in commands:
        command = _mapping(raw, "remote command")
        command_step = _mapping(command.get("step"), "remote command step")
        if command_step.get("operation") == "RESTART_NODE":
            candidates.append(command)
            _require(
                not command.get("batched_steps"),
                "compound commands cannot prove reboot scope",
            )
        _require(
            command_step.get("operation") != "REPLACE_NODE",
            "remote commands include provider replacement",
        )
    _require(len(candidates) == 1, "exactly one reboot remote command is required")
    command = candidates[0]
    _require(
        command.get("status")
        in ({"SUCCEEDED"} if require_complete else {"LEASED", "WAITING", "SUCCEEDED"})
        and command.get("error") in (None, "")
        and command.get("cluster_id") == scope["cluster_id"]
        and command.get("workflow_request_id") == request_id
        and command.get("incident_id") == incident_id
        and type(command.get("step_index")) is int
        and command["step_index"] == index
        and type(command.get("fencing_token")) is int
        and command["fencing_token"] == workflow["fencing_token"]
        and command["step"] == step,
        "remote reboot command outcome, identity, fencing or scope differs",
    )
    command_id = _text(command.get("command_id"), "remote command ID")
    records = _rows(workflow.get("step_executions"), "workflow executions")
    matching = [
        row
        for row in records
        if row.get("operation") == "RESTART_NODE"
        and row.get("status")
        in ({"SUCCEEDED"} if require_complete else {"WAITING", "SUCCEEDED"})
    ]
    _require(
        bool(matching) and (not require_complete or len(matching) == 1),
        "workflow lacks a bound reboot execution",
    )
    execution = max(
        matching, key=lambda row: _time(row.get("updated_at"), "execution time")
    )
    _require(
        execution.get("phase") == "official"
        and type(execution.get("step_index")) is int
        and execution["step_index"] == index
        and execution.get("adapter_operation_id") == f"remote/{command_id}"
        and execution.get("error") in (None, ""),
        "workflow reboot execution is not bound to the remote command",
    )
    details = _mapping(command.get("result_details"), "remote reboot result")
    execution_details = _mapping(execution.get("details"), "workflow reboot result")
    key = hyperpod_submission_idempotency_key(
        request_id, index, WorkflowOperation.RESTART_NODE
    )
    for result_details in (details, execution_details):
        _require(
            result_details.get("submission_idempotency_key") == key
            and result_details.get("action") == "REBOOT"
            and result_details.get("submitted_nodes") == [scope["node_logical_id"]],
            "workflow/command result lacks the exact successful provider submission",
        )
    _require(
        submission.get("state") == "SUBMITTED"
        and submission.get("action") == "REBOOT"
        and submission.get("cluster_name") == scope["cluster_name"]
        and submission.get("idempotency_key") == key
        and submission.get("requested_node_identifiers") == [scope["node_id"]]
        and submission.get("error") in (None, ""),
        "durable provider submission identity, target or state differs",
    )
    result = _mapping(submission.get("result"), "provider submission result")
    _text(result.get("operation_id"), "local provider operation ID")
    _require(
        result.get("idempotency_key") == key
        and result.get("action") == "REBOOT"
        and result.get("cluster_name") == scope["cluster_name"]
        and result.get("submitted") is True
        and type(result.get("duplicate")) is bool
        and result.get("requested_node_logical_ids") == [scope["node_logical_id"]]
        and result.get("successful_node_logical_ids") == [scope["node_logical_id"]]
        and result.get("failures") == [],
        "provider submission was not a complete successful singleton reboot",
    )
    created = _time(submission.get("created_at"), "submission creation")
    completed = _time(submission.get("updated_at"), "submission completion")
    _require(
        started_at <= created <= completed <= ended_at,
        "provider submission is outside the approved observation interval",
    )
    _require(
        _time(workflow.get("created_at"), "workflow creation") >= started_at
        and _time(command.get("created_at"), "command creation") <= created
        and completed
        <= _time(command.get("updated_at"), "command completion")
        <= ended_at
        and completed
        <= _time(execution.get("updated_at"), "execution completion")
        <= ended_at,
        "workflow/command/execution timestamps do not contain the provider submission",
    )
    return result, created, completed


def _step_indexes(value: object) -> list[int]:
    _require(
        isinstance(value, list)
        and all(type(item) is int and item >= 0 for item in value),
        "workflow step indexes must be explicit nonnegative integers",
    )
    indexes = cast(list[int], value)
    _require(len(indexes) == len(set(indexes)), "workflow step indexes repeat")
    return indexes


def _event_binding(
    scope: Mapping[str, Any],
    event: Mapping[str, Any],
    created: datetime,
    completed: datetime,
) -> None:
    _require(event.get("normalization_errors") == [], "CloudTrail normalization failed")
    _require(
        event.get("request_dry_run") is None or event.get("request_dry_run") is False,
        "CloudTrail dry-run cannot prove an actual reboot",
    )
    _require(
        event.get("event_name") == "BatchRebootClusterNodes"
        and event.get("event_source") == "sagemaker.amazonaws.com"
        and event.get("aws_region") == scope["region"]
        and event.get("recipient_account_id") == scope["account_id"],
        "CloudTrail API/Region/account differs from the approved scope",
    )
    role = _arn(scope["executor_role_arn"], "iam")
    identity_arn = _text(event.get("identity_arn"), "CloudTrail session ARN")
    session_prefix = (
        f"arn:{role[0]}:sts::{role[2]}:assumed-role/"
        f"{scope['executor_role_arn'].rsplit('/', 1)[-1]}/"
    )
    _require(
        event.get("identity_type") == "AssumedRole"
        and event.get("identity_account_id") == scope["account_id"]
        and event.get("session_issuer_type") == "Role"
        and event.get("session_issuer_account_id") == scope["account_id"]
        and event.get("session_issuer_arn") == scope["executor_role_arn"]
        and identity_arn.startswith(session_prefix)
        and bool(identity_arn.removeprefix(session_prefix))
        and "/" not in identity_arn.removeprefix(session_prefix),
        "CloudTrail caller is not the exact approved executor role/account",
    )
    clusters = [
        event[key]
        for key in ("request_cluster_name", "request_cluster_arn")
        if event.get(key) is not None
    ]
    _require(
        bool(clusters)
        and all(
            value in {scope["cluster_name"], scope["cluster_arn"]} for value in clusters
        ),
        "CloudTrail request cluster differs or is ambiguous",
    )
    requested = event.get("request_node_logical_ids")
    instances = event.get("request_node_ids")
    _require(
        requested == [scope["node_logical_id"]]
        and instances in (None, [scope["instance_id"]]),
        "CloudTrail request does not match the exact logical/EC2 singleton",
    )
    _require(
        event.get("api_error") is False
        and event.get("api_error_code") is None
        and event.get("api_error_message_present") is False
        and type(event.get("response_failed_node_count")) is int
        and event["response_failed_node_count"] == 0
        and event.get("response_successful_node_logical_ids")
        == [scope["node_logical_id"]]
        and event.get("response_successful_node_ids")
        in (None, [], [scope["instance_id"]]),
        "CloudTrail response is failed, incomplete or names other nodes",
    )
    _text(event.get("event_id"), "CloudTrail event ID")
    _text(event.get("request_id"), "CloudTrail request ID")
    stamp = _time(event.get("event_time"), "CloudTrail event time")
    # CloudTrail timestamps have whole-second resolution. Widen only to the
    # enclosing seconds, not to an arbitrary clock-skew or whole-case window.
    _require(
        created.replace(microsecond=0)
        <= stamp
        < completed.replace(microsecond=0) + timedelta(seconds=1),
        "CloudTrail event is outside the durable submission interval",
    )


def prove_reboot_scope(
    scope: Mapping[str, Any],
    *,
    workflow: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    started_at: datetime,
    ended_at: datetime,
) -> RebootProof:
    """Validate the bounded singleton proof; raw inputs never enter diagnostics.

    Pass the complete provider mutation inventory, not a prefiltered matching
    event. Visibility delay and final negative audits remain caller obligations.
    """
    proof: RebootProof = {
        "valid": False,
        "errors": [],
        "binding": "unique-scoped-submission-window",
        "aws_request_id_joined": False,
        "event_id": None,
        "request_id": None,
        "operation_id": None,
        "submission_idempotency_key": None,
        "submission_started_at": None,
        "submission_completed_at": None,
        "limitations": [
            "Runtime does not persist AWS RequestId; this is not an AWS request-ID join.",
            "CloudTrail delivery is eventual; a bounded observation is not global exactly-once proof.",
        ],
    }
    try:
        _scope_identity(scope)
        start, end = _time(started_at, "window start"), _time(ended_at, "window end")
        _require(start < end, "reboot observation interval is empty or reversed")
        result, created, completed = _submission_binding(
            scope, workflow, commands, submission, start, end
        )
        _require(len(events) == 1, "exactly one provider mutation event is required")
        event = _mapping(events[0], "provider event")
        _event_binding(scope, event, created, completed)
        proof.update(
            {
                "valid": True,
                "event_id": event["event_id"],
                "request_id": event["request_id"],
                "operation_id": result["operation_id"],
                "submission_idempotency_key": result["idempotency_key"],
                "submission_started_at": created.isoformat(),
                "submission_completed_at": completed.isoformat(),
            }
        )
    except (RebootEvidenceError, KeyError, TypeError, AttributeError) as exc:
        proof["errors"].append(
            str(exc)
            if isinstance(exc, RebootEvidenceError)
            else "reboot proof input is malformed"
        )
    return proof


def reboot_evidence_errors(
    events: Sequence[Mapping[str, Any]],
    *,
    scope: Mapping[str, Any],
    workflow: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any],
    started_at: datetime,
    ended_at: datetime,
) -> list[str]:
    """Return strict final-proof errors using the complete mutation inventory."""
    return prove_reboot_scope(
        scope,
        workflow=workflow,
        commands=commands,
        submission=submission,
        events=events,
        started_at=started_at,
        ended_at=ended_at,
    )["errors"]


def submitted_reboot_errors(
    scope: Mapping[str, Any],
    workflow: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any],
    *,
    observed_at: datetime | None = None,
) -> list[str]:
    """Require a confirmed singleton submission, including an in-flight reboot.

    This receipt alone does not identify a transport error or authorize cleanup.
    It excludes INTENDED/UNKNOWN and unconfirmed provider requests.
    Re-leased commands still require their earlier WAITING/SUCCEEDED execution receipt.
    observed_at is the state read time, not a workflow update or a wait deadline.
    """
    try:
        _scope_identity(scope)
        now = datetime.now(timezone.utc)
        observed = now if observed_at is None else observed_at
        _require(
            isinstance(observed, datetime),
            "reboot observation must be a timezone-aware timestamp",
        )
        observed = _time(observed, "reboot observation")
        _require(observed <= now, "reboot observation is in the future")
        created = _time(workflow.get("created_at"), "workflow creation")
        _require(
            created <= _time(workflow.get("updated_at"), "workflow update") <= observed,
            "workflow timestamps are outside the reboot observation interval",
        )
        _submission_binding(
            scope,
            workflow,
            commands,
            submission,
            created,
            observed,
            require_complete=False,
        )
    except (RebootEvidenceError, KeyError, TypeError, AttributeError) as exc:
        return [
            str(exc)
            if isinstance(exc, RebootEvidenceError)
            else "reboot submission input is malformed"
        ]
    return []


_EXECUTOR_SCOPE_PROBE = r"""
import json, os
import boto3
from botocore.config import Config
from gpu_fault.hyperpod import HyperPodAction, HyperPodAdapterConfig
from gpu_fault.env import env_bool
config = HyperPodAdapterConfig.from_environment()
session = boto3.Session()
credentials = session.get_credentials()
caller = session.client("sts", config=Config(
    connect_timeout=10, read_timeout=10, retries={"max_attempts": 0}
)).get_caller_identity()
print(json.dumps({
    "cluster_id": os.getenv("GPU_FAULT_CLUSTER_ID"),
    "hyperpod_cluster": config.cluster_name,
    "region": config.region_name,
    "role_arn": os.getenv("AWS_ROLE_ARN"),
    "allow_reboot": config.action_enabled(HyperPodAction.REBOOT),
    "allow_replace": env_bool("GPU_FAULT_ALLOW_HYPERPOD_REPLACE", False),
    "allow_automatic": config.allow_when_node_recovery_automatic,
    "pod": os.getenv("HOSTNAME"),
    "credential_method": credentials.method if credentials else None,
    "caller_arn": caller.get("Arn"),
    "caller_account": caller.get("Account"),
}, sort_keys=True))
"""


def _aws(regional: RegionalLiveFixture, *arguments: str) -> Mapping[str, Any]:
    try:
        completed = regional.run(
            [
                "aws",
                *arguments,
                "--region",
                regional.settings.region,
                "--output",
                "json",
            ],
            timeout=180,
        )
        return _mapping(_json(completed.stdout, "AWS read"), "AWS read")
    except Exception:
        raise RebootEvidenceError("read-only AWS scope query failed") from None


def _kube(regional: RegionalLiveFixture, *arguments: str, **kwargs: Any) -> Any:
    try:
        return _json(
            regional.kubectl("gpu", *arguments, timeout=60, **kwargs), "Kubernetes read"
        )
    except Exception:
        raise RebootEvidenceError("read-only Kubernetes scope query failed") from None


def _node_identity(
    value: Mapping[str, Any],
    node: str,
    region: str,
    *,
    hyperpod_cluster_arn: str,
    hyperpod_cluster_name: str,
) -> dict[str, str]:
    metadata = _mapping(value.get("metadata"), "Node metadata")
    spec = _mapping(value.get("spec"), "Node spec")
    status = _mapping(value.get("status"), "Node status")
    _require(
        metadata.get("name") == node and not metadata.get("deletionTimestamp"),
        "Node name differs or Node is being deleted",
    )
    ready = [
        row.get("status")
        for row in _rows(status.get("conditions"), "Node conditions")
        if row.get("type") == "Ready"
    ]
    _require(ready == ["True"], "target Node is not uniquely Ready")
    provider = _text(spec.get("providerID"), "Node providerID")
    match = re.fullmatch(r"aws:///([^/]+)/(i-[0-9a-f]{17})", provider)
    if match is not None:
        _require(
            re.fullmatch(re.escape(region) + r"(?:[a-z]|-[a-z0-9-]+)", match.group(1))
            is not None,
            "Node providerID has no exact EC2 instance/Region binding",
        )
        instance_id = match.group(2)
    else:
        cluster_id = _arn(hyperpod_cluster_arn, "sagemaker")[3].removeprefix("cluster/")
        hyperpod = re.fullmatch(
            r"aws:///([a-z0-9]+-az[1-9][0-9]*)/sagemaker/cluster/hyperpod-"
            + re.escape(cluster_id)
            + r"-(i-[0-9a-f]{17})",
            provider,
        )
        _require(
            hyperpod is not None,
            "Node HyperPod providerID has no exact cluster/instance binding",
        )
        assert hyperpod is not None
        labels = _mapping(metadata.get("labels"), "HyperPod Node labels")
        _require(
            labels.get("topology.kubernetes.io/region") == region
            and labels.get("topology.k8s.aws/zone-id") == hyperpod.group(1)
            and labels.get("sagemaker.amazonaws.com/cluster-name")
            == hyperpod_cluster_name,
            "Node HyperPod providerID and topology labels disagree",
        )
        # Region authority is the checked HyperPod/EKS ARN and subsequent exact
        # instance lookup, not an inferred expansion of an AZ ID.
        instance_id = hyperpod.group(2)
    info = _mapping(status.get("nodeInfo"), "Node system identity")
    return {
        "node": node,
        "node_uid": _text(metadata.get("uid"), "Node UID"),
        "boot_id": _text(info.get("bootID"), "Node boot ID"),
        "instance_id": instance_id,
        "provider_id": provider,
    }


def _eks_identity(
    regional: RegionalLiveFixture, eks_arn: str, account: str
) -> dict[str, str]:
    _partition, region, eks_account, resource = _arn(eks_arn, "eks")
    _require(
        region == regional.settings.region and eks_account == account,
        "EKS account/Region differs from HyperPod",
    )
    description = _mapping(
        _aws(
            regional, "eks", "describe-cluster", "--name", resource.split("/", 1)[1]
        ).get("cluster"),
        "EKS cluster",
    )
    _require(
        description.get("arn") == eks_arn
        and description.get("name") == resource.split("/", 1)[1]
        and description.get("status") == "ACTIVE",
        "EKS description is not the exact active orchestrator",
    )
    # Select only the cluster section; never request or persist kubeconfig users.
    clusters = _rows(
        _kube(
            regional,
            "config",
            "view",
            "--minify",
            "--flatten",
            "--raw",
            "-o",
            "jsonpath={.clusters}",
        ),
        "selected kubeconfig clusters",
    )
    _require(len(clusters) == 1, "selected kubeconfig cluster is not singular")
    cluster = _mapping(clusters[0].get("cluster"), "selected kubeconfig cluster")
    endpoint = description.get("endpoint")
    _require(
        isinstance(endpoint, str)
        and endpoint.startswith("https://")
        and cluster.get("server") == endpoint
        and (
            cluster.get("insecure-skip-tls-verify") is None
            or cluster.get("insecure-skip-tls-verify") is False
        )
        and cluster.get("tls-server-name") in (None, "")
        and cluster.get("proxy-url") in (None, ""),
        "selected kubeconfig endpoint/TLS differs from the actual EKS cluster",
    )
    try:
        ca = base64.b64decode(
            _mapping(description.get("certificateAuthority"), "EKS CA")["data"],
            validate=True,
        )
        kube_ca = base64.b64decode(cluster["certificate-authority-data"], validate=True)
        _require(bool(ca) and ca == kube_ca, "selected kubeconfig CA differs from EKS")
    except (KeyError, TypeError, ValueError):
        raise RebootEvidenceError("selected kubeconfig/EKS CA is invalid") from None
    namespace = _mapping(
        _kube(regional, "get", "namespace", "kube-system", "-o", "json"),
        "EKS namespace anchor",
    )
    metadata = _mapping(namespace.get("metadata"), "EKS namespace anchor metadata")
    _require(
        metadata.get("name") == "kube-system" and not metadata.get("deletionTimestamp"),
        "EKS namespace anchor differs or is deleting",
    )
    return {
        "eks_namespace_uid": _text(metadata.get("uid"), "EKS namespace UID"),
        "eks_endpoint_sha256": hashlib.sha256(cast(str, endpoint).encode()).hexdigest(),
        "eks_ca_sha256": hashlib.sha256(ca).hexdigest(),
    }


def _executor_population(
    regional: RegionalLiveFixture,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    deployment = _mapping(
        _kube(regional, "get", "deployment", _EXECUTOR, "-o", "json"),
        "Executor deployment",
    )
    inventory = _mapping(
        _kube(regional, "get", "pod", "-l", f"app={_EXECUTOR}", "-o", "json"),
        "Executor Pods",
    )
    try:
        complete_pod_population(dict(deployment), dict(inventory))
    except (RuntimeError, KeyError, TypeError):
        raise RebootEvidenceError(
            "Executor population is not complete and stable Ready"
        ) from None
    return deployment, sorted(
        [dict(row) for row in _rows(inventory.get("items"), "Executor Pods")],
        key=lambda row: str(row["metadata"]["name"]),
    )


def _executor_fingerprint(
    deployment: Mapping[str, Any], pods: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Ignore status/resourceVersion churn, retain actual execution identities."""
    return {
        "deployment": {
            key: deployment["metadata"].get(key)
            for key in ("name", "namespace", "uid", "generation", "deletionTimestamp")
        },
        "spec": deployment["spec"],
        "pods": [
            {
                "metadata": {
                    key: pod["metadata"].get(key)
                    for key in (
                        "name",
                        "namespace",
                        "uid",
                        "ownerReferences",
                        "deletionTimestamp",
                    )
                },
                "spec": pod["spec"],
                "containers": [
                    {
                        key: row.get(key)
                        for key in (
                            "name",
                            "containerID",
                            "imageID",
                            "restartCount",
                            "ready",
                            "state",
                        )
                    }
                    for row in pod["status"]["containerStatuses"]
                ],
            }
            for pod in pods
        ],
    }


def _owner(value: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    owners = _rows(value.get("ownerReferences"), "Executor owner references")
    _require(
        len(owners) == 1
        and owners[0].get("kind") == kind
        and owners[0].get("controller") is True
        and owners[0].get("apiVersion") == "apps/v1",
        "Executor owner lineage is ambiguous or differs",
    )
    _text(owners[0].get("name"), "Executor owner name")
    _text(owners[0].get("uid"), "Executor owner UID")
    return owners[0]


def _executor_identity(
    regional: RegionalLiveFixture, hyperpod_cluster: str, role_arn: str
) -> tuple[str, str, list[dict[str, object]]]:
    from scripts.e2e.regional.regional_live_fixture import component_python

    deployment, pods = _executor_population(regional)
    deployment_metadata = _mapping(deployment.get("metadata"), "Executor metadata")
    _require(
        deployment_metadata.get("name") == _EXECUTOR
        and deployment_metadata.get("namespace") == regional.settings.namespace,
        "Executor deployment identity differs",
    )
    template = _mapping(
        _mapping(deployment.get("spec"), "Executor spec").get("template"),
        "Executor template",
    )
    pod_spec = _mapping(template.get("spec"), "Executor Pod template")
    service_account = _text(
        pod_spec.get("serviceAccountName"), "Executor ServiceAccount"
    )
    account = _mapping(
        _kube(regional, "get", "serviceaccount", service_account, "-o", "json"),
        "Executor ServiceAccount",
    )
    metadata = _mapping(account.get("metadata"), "Executor ServiceAccount metadata")
    _require(
        metadata.get("name") == service_account
        and metadata.get("namespace") == regional.settings.namespace
        and not metadata.get("deletionTimestamp")
        and _mapping(metadata.get("annotations"), "Executor role annotations").get(
            "eks.amazonaws.com/role-arn"
        )
        == role_arn,
        "Executor ServiceAccount does not name the exact approved role",
    )
    account_uid = _text(metadata.get("uid"), "Executor ServiceAccount UID")
    templates = [
        row
        for row in _rows(pod_spec.get("containers"), "Executor containers")
        if row.get("name") == "executor"
    ]
    _require(len(templates) == 1, "Executor template container is ambiguous")
    role = _arn(role_arn, "iam")
    session_prefix = (
        f"arn:{role[0]}:sts::{role[2]}:assumed-role/{role_arn.rsplit('/', 1)[-1]}/"
    )
    receipts: list[dict[str, object]] = []
    replicas: dict[str, Mapping[str, Any]] = {}
    for pod in pods:
        meta, spec, status = (
            _mapping(pod.get(key), f"Executor {key}")
            for key in ("metadata", "spec", "status")
        )
        containers = [
            row
            for row in _rows(spec.get("containers"), "Executor containers")
            if row.get("name") == "executor"
        ]
        statuses = [
            row
            for row in _rows(status.get("containerStatuses"), "Executor status")
            if row.get("name") == "executor"
        ]
        _require(
            meta.get("namespace") == regional.settings.namespace
            and spec.get("serviceAccountName") == service_account
            and len(containers) == len(statuses) == 1
            and containers[0].get("image") == templates[0].get("image")
            and statuses[0].get("ready") is True
            and type(statuses[0].get("restartCount")) is int
            and statuses[0]["restartCount"] >= 0,
            "Executor container or ServiceAccount differs from its deployment",
        )
        running = _mapping(
            _mapping(statuses[0].get("state"), "Executor container state").get(
                "running"
            ),
            "Executor running container",
        )
        _time(running.get("startedAt"), "Executor container start")
        owner = _owner(meta, "ReplicaSet")
        replica_name = str(owner["name"])
        if replica_name not in replicas:
            replicas[replica_name] = _mapping(
                _kube(regional, "get", "replicaset", replica_name, "-o", "json"),
                "Executor ReplicaSet",
            )
        replica_meta = _mapping(
            replicas[replica_name].get("metadata"), "Executor ReplicaSet metadata"
        )
        deployment_owner = _owner(replica_meta, "Deployment")
        _require(
            replica_meta.get("uid") == owner["uid"]
            and replica_meta.get("name") == replica_name
            and replica_meta.get("namespace") == regional.settings.namespace
            and not replica_meta.get("deletionTimestamp")
            and deployment_owner.get("name") == _EXECUTOR
            and deployment_owner.get("uid") == deployment_metadata["uid"],
            "Executor Pod is not owned by the actual deployment",
        )
        name = _text(meta.get("name"), "Executor Pod name")
        observed = _mapping(
            _kube(
                regional,
                "exec",
                "-i",
                name,
                "-c",
                "executor",
                "--",
                component_python("gpu"),
                "-B",
                "-",
                input_text=_EXECUTOR_SCOPE_PROBE,
            ),
            "Executor scope probe",
        )
        caller = _text(observed.get("caller_arn"), "Executor STS caller ARN")
        _require(
            observed.get("cluster_id") == regional.settings.cluster_id
            and observed.get("hyperpod_cluster") == hyperpod_cluster
            and observed.get("region") == regional.settings.region
            and observed.get("role_arn") == role_arn
            and observed.get("pod") == name
            and observed.get("allow_reboot") is True
            and observed.get("allow_replace") is False
            and observed.get("allow_automatic") is False
            and observed.get("credential_method") == "assume-role-with-web-identity"
            and observed.get("caller_account") == role[2]
            and caller.startswith(session_prefix)
            and bool(caller.removeprefix(session_prefix))
            and "/" not in caller.removeprefix(session_prefix),
            "deployed Executor cluster/role/Region/recovery configuration differs",
        )
        receipts.append(
            {
                "name": name,
                "uid": _text(meta.get("uid"), "Executor Pod UID"),
                "container_id": _text(
                    statuses[0].get("containerID"), "Executor container ID"
                ),
                "restart_count": statuses[0].get("restartCount"),
                "service_account": service_account,
                "service_account_uid": account_uid,
                "image": templates[0].get("image"),
                "caller_arn": caller,
            }
        )
    after, pods_after = _executor_population(regional)
    _require(
        _executor_fingerprint(after, pods_after)
        == _executor_fingerprint(deployment, pods),
        "Executor population changed while its configuration was read",
    )
    account_after = _kube(
        regional, "get", "serviceaccount", service_account, "-o", "json"
    )
    _require(
        account_after == account, "Executor ServiceAccount changed during scope capture"
    )
    for name, replica in replicas.items():
        current = _mapping(
            _kube(regional, "get", "replicaset", name, "-o", "json"),
            "Executor ReplicaSet",
        )
        _require(
            all(
                current["metadata"].get(key) == replica["metadata"].get(key)
                for key in (
                    "uid",
                    "name",
                    "namespace",
                    "ownerReferences",
                    "deletionTimestamp",
                )
            ),
            "Executor owner lineage changed during scope capture",
        )
    return (
        _text(deployment_metadata.get("uid"), "Executor deployment UID"),
        hashlib.sha256(
            json.dumps(dict(template), sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        receipts,
    )


def capture_reboot_scope(
    regional: RegionalLiveFixture,
    *,
    node: str,
    hyperpod_cluster: str,
    executor_role_arn: str,
) -> RebootScope:
    """Capture actual read-only identities; raise safely on missing/drifted facts.

    Call before injection and after recovery. Node UID/instance/logical identity
    must stay fixed; a new boot ID is an expected *separate* reboot observation.
    These sequential reads do not lock external actors or promise atomicity.
    """
    try:
        _text(node, "target Node")
        _text(hyperpod_cluster, "HyperPod cluster")
        _text(regional.settings.cluster_id, "managed cluster ID")
        role = _arn(executor_role_arn, "iam")
        cluster = _aws(
            regional,
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            hyperpod_cluster,
        )
        hp_arn = _text(cluster.get("ClusterArn"), "HyperPod ARN")
        hp = _arn(hp_arn, "sagemaker")
        _require(
            cluster.get("ClusterName") == hyperpod_cluster
            and hp[0] == role[0]
            and hp[2] == role[2]
            and not role[1]
            and hp[1] == regional.settings.region
            and cluster.get("NodeRecovery") == "None"
            and cluster.get("ClusterStatus") == "InService",
            "actual HyperPod name/ARN/Region/account or recovery state differs",
        )
        orchestrator = _mapping(cluster.get("Orchestrator"), "HyperPod orchestrator")
        _require(
            set(orchestrator) == {"Eks"}, "HyperPod orchestrator is not exclusively EKS"
        )
        eks = _mapping(orchestrator["Eks"], "HyperPod EKS orchestrator")
        eks_arn = _text(eks.get("ClusterArn"), "orchestrator EKS ARN")
        eks_facts = _eks_identity(regional, eks_arn, hp[2])
        before = _node_identity(
            _mapping(_kube(regional, "get", "node", node, "-o", "json"), "Node"),
            node,
            regional.settings.region,
            hyperpod_cluster_arn=hp_arn,
            hyperpod_cluster_name=hyperpod_cluster,
        )
        inventory = _aws(
            regional,
            "sagemaker",
            "list-cluster-nodes",
            "--cluster-name",
            hp_arn,
            "--include-node-logical-ids",
        )
        _require(not inventory.get("NextToken"), "HyperPod inventory is incomplete")
        rows = _rows(inventory.get("ClusterNodeSummaries"), "HyperPod node inventory")
        logical_ids = [
            _text(row.get("NodeLogicalId"), "provider logical node ID") for row in rows
        ]
        _require(
            len(set(logical_ids)) == len(logical_ids),
            "HyperPod inventory repeats a logical node",
        )
        matches = [
            row for row in rows if row.get("InstanceId") == before["instance_id"]
        ]
        _require(
            len(matches) == 1, "Node providerID has no unique HyperPod node mapping"
        )
        target = matches[0]
        _require(
            _mapping(target.get("InstanceStatus"), "HyperPod node status").get("Status")
            == "Running",
            "HyperPod target instance is not Running",
        )
        deployment_uid, template_digest, executors = _executor_identity(
            regional, hyperpod_cluster, executor_role_arn
        )
        after = _node_identity(
            _mapping(_kube(regional, "get", "node", node, "-o", "json"), "Node"),
            node,
            regional.settings.region,
            hyperpod_cluster_arn=hp_arn,
            hyperpod_cluster_name=hyperpod_cluster,
        )
        _require(after == before, "Node identity changed during scope capture")
        detail = _aws(
            regional,
            "sagemaker",
            "describe-cluster-node",
            "--cluster-name",
            hp_arn,
            "--node-logical-id",
            target["NodeLogicalId"],
        )
        details = _mapping(detail.get("NodeDetails"), "HyperPod node detail")
        _require(
            details.get("NodeLogicalId") == target["NodeLogicalId"]
            and details.get("InstanceId") == before["instance_id"],
            "HyperPod node mapping changed during scope capture",
        )
        cluster_after = _aws(
            regional, "sagemaker", "describe-cluster", "--cluster-name", hp_arn
        )
        _require(
            all(
                cluster_after.get(key) == cluster.get(key)
                for key in (
                    "ClusterArn",
                    "ClusterName",
                    "NodeRecovery",
                    "ClusterStatus",
                    "Orchestrator",
                )
            ),
            "HyperPod identity or recovery settings changed during scope capture",
        )
        result: RebootScope = {
            "cluster_id": regional.settings.cluster_id,
            "node_id": node,
            "node_uid": before["node_uid"],
            "node_provider_id": before["provider_id"],
            "boot_id": before["boot_id"],
            "instance_id": before["instance_id"],
            "node_logical_id": target["NodeLogicalId"],
            "cluster_name": hyperpod_cluster,
            "cluster_arn": hp_arn,
            "eks_cluster_arn": eks_arn,
            "region": regional.settings.region,
            "account_id": hp[2],
            "executor_role_arn": executor_role_arn,
            "gpu_context": regional.settings.gpu_context,
            "provider_inventory": [
                {
                    "node_logical_id": row["NodeLogicalId"],
                    "instance_id": _optional_text(row.get("InstanceId")),
                }
                for row in sorted(rows, key=lambda item: str(item["NodeLogicalId"]))
            ],
            "eks_namespace_uid": eks_facts["eks_namespace_uid"],
            "eks_endpoint_sha256": eks_facts["eks_endpoint_sha256"],
            "eks_ca_sha256": eks_facts["eks_ca_sha256"],
            "executor_deployment_uid": deployment_uid,
            "executor_template_sha256": template_digest,
            "executor_pods": executors,
            "node_recovery": "None",
            "cluster_status": "InService",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        _scope_identity(result)
        return result
    except RebootEvidenceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise RebootEvidenceError(
            "reboot scope identity evidence is malformed"
        ) from None
