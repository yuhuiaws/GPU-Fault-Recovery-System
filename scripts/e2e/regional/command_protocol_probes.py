"""Transport, probe assertions, seed models and cleanup for the in-Pod CMD audit."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import SqliteStore


class ProtocolAuditError(RuntimeError):
    """A case assertion that did not hold, or a preflight that refused to run."""


def build_audit_parser(
    case_ids: tuple[str, ...], *, description: str
) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--cluster-id", default="")
    result.add_argument("--other-cluster-id", default="")
    result.add_argument("--executor-sha256", default="")
    result.add_argument("--executor-digest", default="")
    result.add_argument(
        "--case",
        action="append",
        choices=case_ids,
        help="explicit CMD case to execute; repeat only in formal order",
    )
    result.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="write cases/<id>/<id>.json per CMD case under this directory",
    )
    result.add_argument(
        "--release-id",
        default="",
        help="release the evidence is bound to (from the release state ConfigMap)",
    )
    local = result.add_mutually_exclusive_group()
    local.add_argument(
        "--emit-probe",
        action="store_true",
        help="print the standalone source bundle for the CPU component interpreter",
    )
    local.add_argument(
        "--write-evidence",
        type=Path,
        default=None,
        metavar="SUMMARY_JSON",
        help="write per-case evidence from a printed summary without touching a cluster",
    )
    isolation = result.add_mutually_exclusive_group()
    isolation.add_argument(
        "--isolated-cluster",
        action="store_true",
        help="the operator asserts no production executor serves this cluster",
    )
    isolation.add_argument(
        "--executor-ready-replicas",
        type=int,
        default=None,
        help="Ready replicas observed outside the Pod; the audit requires zero",
    )
    return result


def expect(condition: object, message: str) -> None:
    """Keep audit assertions active even when Python runs with optimization."""
    if not condition:
        raise ProtocolAuditError(message)


def request_json(
    method: str,
    path: str,
    *,
    cluster_id: str | None,
    token: str | None,
    payload: Any = None,
) -> tuple[int, Any]:
    headers: dict[str, str] = {}
    if cluster_id is not None:
        headers["X-GPU-Fault-Cluster-ID"] = cluster_id
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
            return response.status, json.loads(raw or b"null")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw or b"null")


def make_audit_pair(
    run_id: str,
    suffix: str,
    *,
    owner: str,
    cluster_id: str,
    fencing_token: int,
    step: WorkflowStepSpec | None = None,
    not_before: datetime | None = None,
) -> tuple[FaultIncident, WorkflowRequest, WorkflowStepSpec]:
    incident_id = f"{run_id}-{suffix}-incident"
    workflow_id = f"{run_id}-{suffix}-workflow"
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=f"{run_id}-{suffix}-event",
        event_type="CMD_PROTOCOL_PROBE",
        cluster_id=cluster_id,
        node_ids=[f"{run_id}-nonexistent-node"],
        policy_version="cmd-probe",
        policy_source="cmd-probe",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow_id,
        fencing_token=fencing_token,
    )
    selected = step or WorkflowStepSpec(
        operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        execution_owner=owner,
        node_ids=[f"{run_id}-nonexistent-node"],
    )
    workflow = WorkflowRequest(
        request_id=workflow_id,
        incident_id=incident_id,
        runtime_profile_version="cmd-probe",
        status=WorkflowStatus.RUNNING,
        fencing_token=fencing_token,
        not_before=not_before,
        official_steps=[selected],
    )
    return incident, workflow, selected


def claim_records(
    status: int, body: Any, expected_ids: set[str], *, count: int | None = None
) -> list[dict[str, Any]]:
    expect(status == 200, f"claim answered {status}, expected 200")
    expect(isinstance(body, dict), "claim response is not an object")
    commands = body.get("commands")
    expect(
        isinstance(commands, list) and all(isinstance(item, dict) for item in commands),
        "claim response has no command list",
    )
    ids = [item.get("command_id") for item in commands]
    expect(
        all(isinstance(item, str) and item in expected_ids for item in ids),
        "claim returned a command this audit did not seed",
    )
    expect(len(ids) == len(set(ids)), "claim returned duplicate commands")
    expect(
        len(ids) == (len(expected_ids) if count is None else count),
        "claim returned the wrong number of seeded commands",
    )
    return cast(list[dict[str, Any]], commands)


def check_result_payloads(audit: Any, command_id: str, lease_token: str) -> None:
    before = audit.store.get_remote_command(command_id).model_dump(mode="json")
    payloads: dict[str, dict[str, Any]] = {
        "PENDING": {"lease_token": lease_token, "status": "PENDING"},
        "LEASED": {"lease_token": lease_token, "status": "LEASED"},
        "FAILED_NO_ERROR": {"lease_token": lease_token, "status": "FAILED"},
        "FAILED_EMPTY_ERROR": {
            "lease_token": lease_token,
            "status": "FAILED",
            "error": "",
        },
        "MISSING_TOKEN": {"status": "WAITING"},
        "EXTRA_FIELD": {
            "lease_token": lease_token,
            "status": "WAITING",
            "foo": 1,
        },
    }
    observed = {}
    for name, payload in payloads.items():
        status, _ = audit.complete(command_id, payload=payload)
        observed[name] = status
        expect(status == 422, f"{name} answered {status}")
        expect(
            audit.store.get_remote_command(command_id).model_dump(mode="json")
            == before,
            f"{name} changed the command record",
        )
    waiting, body = audit.complete(
        command_id,
        payload={"lease_token": lease_token, "status": "WAITING"},
    )
    expect(
        waiting == 200 and body["status"] == "WAITING",
        f"legal WAITING answered {waiting}",
    )
    audit.record(
        "GF-REGIONAL-CMD-008",
        invalid_statuses=observed,
        waiting_status=waiting,
    )


def make_submission_probe_record(audit: Any) -> dict[str, Any]:
    """Register every possible cleanup key before a submission request can commit."""
    key = f"{audit.run_id}-cmd016"
    cluster_name = audit.registry[audit.cluster_id]["hyperpod_cluster_name"]
    other_name = audit.registry[audit.other_cluster_id]["hyperpod_cluster_name"]
    for name in (cluster_name, other_name):
        for suffix in ("", "-missing", "-unreserved"):
            audit.created_auxiliary.add(
                (
                    "hyperpod_submission",
                    audit.store._hyperpod_submission_key(name, key + suffix),
                )
            )
    return {
        "cluster_name": cluster_name,
        "idempotency_key": key,
        "action": "REBOOT",
        "requested_node_identifiers": [f"{audit.run_id}-node"],
    }


def delete_audit_event_link(store: Any, event_id: str, incident_id: str) -> None:
    if isinstance(store, SqliteStore):
        with store._db:
            store._db.execute(
                "DELETE FROM links WHERE kind='incident_by_event' "
                "AND key=? AND value=?",
                (event_id, incident_id),
            )
    else:
        with store._db.cursor() as cursor:
            cursor.execute(
                "DELETE FROM gpu_fault_links WHERE kind='incident_by_event' "
                "AND key=%s AND value=%s",
                (event_id, incident_id),
            )
    expect(
        store._get_link("incident_by_event", event_id) is None,
        "audit event link remains or changed ownership",
    )


def cleanup_audit_records(
    store: Any,
    *,
    cluster_ids: set[str],
    created_commands: set[str],
    created_workflows: set[str],
    created_incidents: set[str],
    created_event_links: dict[str, str],
    created_auxiliary: set[tuple[str, str]],
    commands_before: set[str],
    workflows_before: set[str],
    incidents_before: set[str],
) -> None:
    errors = []
    workflows = created_workflows - workflows_before
    if workflows:
        try:
            for command in store.list_remote_commands(
                workflow_request_ids=sorted(workflows)
            ):
                expect(
                    command.incident_id in created_incidents
                    and command.cluster_id in cluster_ids,
                    "cleanup command identity does not match the audit",
                )
                created_commands.add(command.command_id)
        except Exception as exc:
            errors.append(f"command discovery: {type(exc).__name__}")
    for event_id, incident_id in list(created_event_links.items()):
        try:
            delete_audit_event_link(store, event_id, incident_id)
            del created_event_links[event_id]
        except Exception as exc:
            errors.append(f"event-link cleanup: {type(exc).__name__}")
    for kind, keys, before in (
        ("remote_command", created_commands, commands_before),
        ("workflow", created_workflows, workflows_before),
        ("incident", created_incidents, incidents_before),
    ):
        for key in keys - before:
            try:
                store._delete(kind, key)
                expect(
                    store._get_optional(kind, key) is None,
                    f"{kind} cleanup left an audit record",
                )
                keys.discard(key)
            except Exception as exc:
                errors.append(f"{kind} cleanup: {type(exc).__name__}")
    for kind, key in set(created_auxiliary):
        try:
            store._delete(kind, key)
            expect(
                store._get_optional(kind, key) is None,
                "auxiliary cleanup left an audit record",
            )
            created_auxiliary.discard((kind, key))
        except Exception as exc:
            errors.append(f"{kind} cleanup: {type(exc).__name__}")
    expect(not errors, "; ".join(errors))


def anonymous_submission_checks(
    request: Callable[..., tuple[int, Any]],
    *,
    cluster_id: str,
    reserve_payload: dict[str, Any],
    outcome_record: dict[str, Any],
    query: str,
) -> dict[str, int]:
    anonymous = {}
    for method, path, payload in (
        (
            "POST",
            "/v1/regional/executors/hyperpod-submissions/reserve",
            reserve_payload,
        ),
        (
            "GET",
            f"/v1/regional/executors/hyperpod-submissions?{query}",
            None,
        ),
        (
            "POST",
            "/v1/regional/executors/hyperpod-submissions/outcome",
            {"cluster_id": cluster_id, "record": outcome_record},
        ),
    ):
        status, _ = request(
            method,
            path,
            cluster_id=None,
            token=None,
            payload=payload,
        )
        anonymous[method + " " + path.split("?")[0]] = status
        expect(
            status == 401, f"anonymous {method} {path} answered {status}, expected 401"
        )
    return anonymous
