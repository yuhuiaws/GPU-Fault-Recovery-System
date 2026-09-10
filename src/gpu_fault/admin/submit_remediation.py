"""Submit the administrator's disposition after a ``CHECK_MECHANICALS`` step.

``CHECK_MECHANICALS`` is an investigation: the workflow notifies and then
waits, up to ``GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS``, for the
node to carry ``gpu-fault.io/mechanical-inspection-complete=<incident>:<fencing
token>``. Manual 8.3 had the administrator type that value, then hand-build a
trusted marker (posted with the execution token) and an operator terminal
attempt (posted with the cluster token) with the node, GPU UUIDs, job and
attempt identifiers copied by hand -- exactly the fabrication the same page
forbids.

This verb reads the incident, its workflow and the waiting step from the
control plane through the CPU ingress Pod's exec channel, derives every value
from the record, refuses on any mismatch (not waiting, node set changed, stale
generation, deadline passed), and submits:

``inspected``
    the acknowledgement annotation only -- the investigation workflow finishes
    on its own and restores the workload.
``reset-gpu`` / ``reboot-node`` / ``quarantine``
    the acknowledgement, then a trusted ``operator-change`` marker and the
    operator terminal trigger that compile a *new* fenced workflow for the
    hardware action; the original incident is never rewritten.
``restore``
    not a CHECK_MECHANICALS answer but the exit of a QUARANTINED incident whose
    node was repaired by hand: a validated restore workflow (VALIDATE_GPU ->
    VALIDATE_HOST -> VALIDATE_FABRIC -> RESTORE_SCHEDULING) under the *same*
    incident and fencing token, built by
    ``gpu_fault.orchestration.validated_restore`` inside the CPU Pod. Refused
    unless the incident is QUARANTINED with no open workflow and every node
    carries the ``gpu-fault.io/quarantined`` taint this incident owns; a node
    another incident quarantined is named. Until 2026-09-10 only the
    acceptance fixture could create this workflow.

Resubmitting the same disposition is a no-op: the annotation is compared
before it is written, and the terminal decision for the same attempt is
returned as ``duplicate`` by the control plane rather than re-planned; a
``restore`` rerun while its workflow is PENDING/RUNNING returns that id.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from gpu_fault.adapters.common import (
    ANNOTATION_INCIDENT,
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    QUARANTINE_TAINT,
)
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, safe_name
from gpu_fault.admin.operator_identity import resolve_operator_identity
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import (
    cluster_nodes,
    gpu_kubectl_command,
    run_control_plane_script,
)
from gpu_fault.models import FaultIncident, RecoveryAction
from gpu_fault.orchestration.incident_closure import owned_quarantine_taint_values
from gpu_fault.orchestration.validated_restore import (
    build_validated_restore_workflow,
    is_validated_restore_workflow,
    restore_reason,
)

STATE_ROOT = "submit-remediation"
DISPOSITION_INSPECTED = "inspected"
DISPOSITION_RESTORE = "restore"
DISPOSITION_ACTIONS: dict[str, RecoveryAction | None] = {
    DISPOSITION_INSPECTED: None,
    "reset-gpu": RecoveryAction.RESET_GPU,
    "reboot-node": RecoveryAction.REBOOT_NODE,
    "quarantine": RecoveryAction.QUARANTINE,
}
DISPOSITIONS = (*DISPOSITION_ACTIONS, DISPOSITION_RESTORE)
# A workflow in one of these still owns its node; ``restore`` waits for it.
OPEN_WORKFLOW_STATUSES = frozenset({"PENDING", "RUNNING", "SAFETY_PENDING"})
QUARANTINED_STATE = "QUARANTINED"
OPERATOR_MARKER_SOURCE = "operator-change"
OPERATOR_MAPPING_VERSION = "operator-change-v1"
OPERATOR_MARKER_TTL = timedelta(hours=1)
DEFAULT_ENVIRONMENT = "hyperpod-eks"
WAITING_OPERATION = "CHECK_MECHANICALS"
ACTIVE_WORKLOAD_PHASES = frozenset({"PENDING", "RUNNING"})
INCIDENT_ANNOTATION = "gpu-fault.io/incident-id"

REMEDIATION_SCRIPT = """
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    NodeMarker,
    TerminalEvent,
    WorkflowOperation,
    WorkflowStatus,
)

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
store = context.store
OPEN = {WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING}


def dump(model):
    return None if model is None else json.loads(model.model_dump_json())


def load(incident_id):
    incident = store.get_incident(incident_id)
    workflow = (
        store.get_workflow(incident.workflow_request_id)
        if incident.workflow_request_id
        else None
    )
    return incident, workflow


def open_workflows(incident):
    # Every open workflow of the incident on its nodes, the pointer included.
    rows = {}
    if incident.workflow_request_id:
        current = store.get_workflow(incident.workflow_request_id)
        if current.status in OPEN:
            rows[current.request_id] = current
    if incident.node_ids:
        for owner, workflow in store.list_active_workflow_incidents(
            incident.cluster_id, node_ids=set(incident.node_ids)
        ):
            if owner.incident_id == incident.incident_id and workflow.status in OPEN:
                rows[workflow.request_id] = workflow
    return [rows[key] for key in sorted(rows)]


if payload["mode"] == "inspect":
    incident, workflow = load(payload["incident_id"])
    nodes = set(incident.node_ids)
    observations = [
        dump(state)
        for state in store.list_attempt_observation_states(incident.cluster_id)
        if str(getattr(state.observation.workload_phase, "value",
                       state.observation.workload_phase)) in ("PENDING", "RUNNING")
        and any(item.node_id in nodes for item in state.observation.containers)
    ]
    result = {
        "incident": dump(incident),
        "workflow": dump(workflow),
        "open_workflows": [dump(item) for item in open_workflows(incident)],
        "active_observations": observations,
        "existing_markers": [
            dump(item)
            for item in store.list_markers_for_incident(payload["action_incident_id"])
        ],
        "acknowledgement_timeout_seconds": (
            context.production_executor_config.step_waiting_limit(
                WorkflowOperation.CHECK_MECHANICALS
            )
        ),
    }
elif payload["mode"] == "restore":
    from gpu_fault.orchestration.validated_restore import (
        build_validated_restore_workflow,
        is_validated_restore_workflow,
    )

    incident, workflow = load(payload["incident_id"])
    expected = payload["expected"]
    observed = {
        "fencing_token": incident.fencing_token,
        "workflow_request_id": incident.workflow_request_id,
        "state": incident.state.value,
        "node_ids": sorted(incident.node_ids),
    }
    drift = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if drift:
        raise SystemExit(
            "submit-remediation: incident moved since inspection: "
            + json.dumps(drift, sort_keys=True)
        )
    still_open = open_workflows(incident)
    existing = [
        item for item in still_open if is_validated_restore_workflow(item.request_id)
    ]
    if existing:
        result = {"no_op": True, "workflow": dump(existing[0]), "incident": dump(incident)}
    elif still_open:
        raise SystemExit(
            "submit-remediation: incident still has an open workflow "
            + ", ".join(f"{item.request_id} ({item.status.value})" for item in still_open)
        )
    else:
        updated, created = build_validated_restore_workflow(
            incident,
            operator=payload["operator"],
            reference=payload.get("reference"),
            now=datetime.now(timezone.utc),
            runtime_profile_version=payload.get("runtime_profile_version"),
        )
        store.save_incident_and_workflow(updated, created)
        context.dispatcher.wake()
        result = {"no_op": False, "workflow": dump(created), "incident": dump(updated)}
elif payload["mode"] == "submit":
    incident, workflow = load(payload["incident_id"])
    expected = payload["expected"]
    observed = {
        "fencing_token": incident.fencing_token,
        "workflow_request_id": incident.workflow_request_id,
        "node_ids": sorted(incident.node_ids),
    }
    drift = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if drift:
        raise SystemExit(
            "submit-remediation: incident moved since inspection: "
            + json.dumps(drift, sort_keys=True)
        )
    terminal = TerminalEvent.model_validate(payload["terminal"])
    existing = store.get_decision_by_event(terminal.event_key)
    if existing is not None:
        result = {"duplicate": True, "decision": dump(existing)}
        plan = (
            store.get_plan(existing.recovery_plan_id)
            if existing.recovery_plan_id
            else None
        )
    else:
        context.completion.add_marker(NodeMarker.model_validate(payload["marker"]))
        decision = context.completion.handle_terminal(terminal)
        result = {"duplicate": bool(decision.duplicate), "decision": dump(decision)}
        plan = (
            store.get_plan(decision.recovery_plan_id)
            if decision.recovery_plan_id
            else None
        )
    result["plan"] = dump(plan)
    result["workflow"] = (
        dump(store.get_workflow(plan.workflow_request_id))
        if plan is not None and plan.workflow_request_id
        else None
    )
else:
    raise ValueError("unsupported submit-remediation mode")
print(json.dumps(result, sort_keys=True))
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SubmitRemediationRequest:
    site: RenderedSite
    incident_id: str
    disposition: str
    reference: str | None = None
    plan_only: bool = False

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise BootstrapError(
                f"unknown disposition {self.disposition!r}; choose one of "
                + ", ".join(DISPOSITIONS)
            )
        if not self.incident_id.strip():
            raise BootstrapError("submit-remediation requires --incident-id")


@dataclass(frozen=True)
class RemediationPlan:
    """Everything the submission will write, derived from the record."""

    incident_id: str
    cluster_id: str
    disposition: str
    node_ids: tuple[str, ...]
    gpu_uuids: tuple[str, ...]
    fencing_token: int
    workflow_request_id: str
    acknowledgement_value: str
    pending_nodes: tuple[str, ...]
    next_operations: tuple[str, ...]
    acknowledgement_timeout_seconds: int | None
    action: str | None = None
    action_incident_id: str | None = None
    marker: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    workload_source: str | None = None
    resubmission: bool = False
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "cluster_id": self.cluster_id,
            "disposition": self.disposition,
            "node_ids": list(self.node_ids),
            "gpu_uuids": list(self.gpu_uuids),
            "fencing_token": self.fencing_token,
            "workflow_request_id": self.workflow_request_id,
            "acknowledgement": {
                "annotation": ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
                "value": self.acknowledgement_value,
                "pending_nodes": list(self.pending_nodes),
                "timeout_seconds": self.acknowledgement_timeout_seconds,
            },
            "action": self.action,
            "action_incident_id": self.action_incident_id,
            "marker": self.marker,
            "terminal": self.terminal,
            "workload_source": self.workload_source,
            "next_operations": list(self.next_operations),
            "resubmission": self.resubmission,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RestorePlan:
    """The validated restore ``--disposition restore`` will create, derived
    from the QUARANTINED incident record and the live nodes.

    ``steps`` are the exact steps the product builder yields (operation,
    owner, nodes, GPU scope); the request id is minted in the Pod at submit
    time. ``existing_restore_workflow_id`` is set when such a workflow is
    already PENDING/RUNNING under the incident: the rerun is a no-op.
    """

    incident_id: str
    cluster_id: str
    node_ids: tuple[str, ...]
    gpu_uuids: tuple[str, ...]
    fencing_token: int
    workflow_request_id: str
    state: str
    reason: str
    steps: tuple[dict[str, Any], ...]
    existing_restore_workflow_id: str | None = None
    warnings: tuple[str, ...] = ()
    disposition: str = DISPOSITION_RESTORE

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "cluster_id": self.cluster_id,
            "disposition": self.disposition,
            "node_ids": list(self.node_ids),
            "gpu_uuids": list(self.gpu_uuids),
            "fencing_token": self.fencing_token,
            "workflow_request_id": self.workflow_request_id,
            "state": self.state,
            "reason": self.reason,
            "steps": [dict(item) for item in self.steps],
            "existing_restore_workflow_id": self.existing_restore_workflow_id,
            "next_operations": [str(item["operation"]) for item in self.steps],
            "warnings": list(self.warnings),
        }


@dataclass
class SubmissionResult:
    plan: RemediationPlan | RestorePlan
    acknowledgement: dict[str, Any] = field(default_factory=dict)
    submission: dict[str, Any] | None = None
    no_op: bool = False
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "acknowledgement": self.acknowledgement,
            "submission": self.submission,
            "no_op": self.no_op,
            "message": self.message,
        }


# --------------------------------------------------------------------------
# Naming


def action_incident_id(incident_id: str, disposition: str) -> str:
    return f"inc-operator-{incident_id}-{disposition}"


def operator_marker_id(incident_id: str, disposition: str) -> str:
    return f"marker-operator-{incident_id}-{disposition}"


def operator_attempt_id(incident_id: str, disposition: str) -> str:
    return f"operator-{incident_id}-{disposition}"


def acknowledgement_value(incident_id: str, fencing_token: int) -> str:
    """The exact string the adapter compares (``node_operations._check_mechanicals``)."""

    return f"{incident_id}:{fencing_token}"


# --------------------------------------------------------------------------
# Control-plane reads


def inspect_incident(
    site: RenderedSite, incident_id: str, *, disposition: str
) -> dict[str, Any]:
    return run_control_plane_script(
        site,
        {
            "mode": "inspect",
            "incident_id": incident_id,
            "action_incident_id": action_incident_id(incident_id, disposition),
        },
        script=REMEDIATION_SCRIPT,
    )


def submit_operator_action(site: RenderedSite, plan: RemediationPlan) -> dict[str, Any]:
    return run_control_plane_script(
        site,
        {
            "mode": "submit",
            "incident_id": plan.incident_id,
            "expected": {
                "fencing_token": plan.fencing_token,
                "workflow_request_id": plan.workflow_request_id,
                "node_ids": sorted(plan.node_ids),
            },
            "marker": plan.marker,
            "terminal": plan.terminal,
        },
        script=REMEDIATION_SCRIPT,
    )


# --------------------------------------------------------------------------
# Validation


def _waiting_step(
    workflow: Mapping[str, Any], incident_id: str, fencing_token: int
) -> tuple[int, dict[str, Any]]:
    waiting = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == WAITING_OPERATION
        and item.get("status") == "WAITING"
        and item.get("phase") in (None, "official")
    ]
    if not waiting:
        raise BootstrapError(
            f"incident {incident_id} is not waiting on {WAITING_OPERATION}; "
            f"workflow {workflow.get('request_id')} is {workflow.get('status')}"
        )
    execution = max(waiting, key=lambda item: str(item.get("updated_at") or ""))
    index = int(execution["step_index"])
    steps = list(workflow.get("official_steps") or [])
    if index >= len(steps) or steps[index].get("operation") != WAITING_OPERATION:
        raise BootstrapError(
            f"workflow {workflow.get('request_id')} step {index} is not "
            f"{WAITING_OPERATION}"
        )
    details = execution.get("details") or {}
    required = details.get("required_annotation_value")
    expected = acknowledgement_value(incident_id, fencing_token)
    if required is not None and str(required) != expected:
        raise BootstrapError(
            "the waiting step expects acknowledgement "
            f"{required!r} but the incident record yields {expected!r}; "
            "the generation moved"
        )
    return index, dict(steps[index])


def _deadline_passed(workflow: Mapping[str, Any], now: datetime) -> str | None:
    for key in ("execution_deadline", "lifetime_deadline_at"):
        raw = workflow.get(key)
        if not raw:
            continue
        deadline = datetime.fromisoformat(str(raw))
        if deadline <= now:
            return f"{key} {deadline.isoformat()} has passed"
    return None


def _validate_record(
    inspection: Mapping[str, Any], *, incident_id: str, now: datetime
) -> tuple[dict[str, Any], dict[str, Any]]:
    incident = cast(dict[str, Any] | None, inspection.get("incident"))
    workflow = cast(dict[str, Any] | None, inspection.get("workflow"))
    if incident is None:
        raise BootstrapError(f"incident {incident_id} was not returned")
    if str(incident.get("incident_id")) != incident_id:
        raise BootstrapError("the control plane returned a different incident")
    if workflow is None:
        raise BootstrapError(f"incident {incident_id} has no workflow")
    if workflow.get("request_id") != incident.get("workflow_request_id"):
        raise BootstrapError(
            f"incident {incident_id} points at workflow "
            f"{incident.get('workflow_request_id')} but "
            f"{workflow.get('request_id')} was returned"
        )
    if int(workflow.get("fencing_token") or 0) != int(
        incident.get("fencing_token") or 0
    ):
        raise BootstrapError(
            f"stale generation: incident fencing token {incident.get('fencing_token')} "
            f"differs from workflow {workflow.get('fencing_token')}"
        )
    if not incident.get("node_ids"):
        raise BootstrapError(f"incident {incident_id} names no nodes")
    passed = _deadline_passed(workflow, now)
    if passed:
        raise BootstrapError(
            f"workflow {workflow.get('request_id')} {passed}; the workflow will "
            "fail on its own -- open a new remediation instead of acknowledging"
        )
    return incident, workflow


def _active_attempt(
    observations: list[dict[str, Any]], node_ids: set[str]
) -> dict[str, Any] | None:
    attempts: dict[str, dict[str, Any]] = {}
    for state in observations:
        observation = cast(dict[str, Any], state.get("observation") or {})
        containers = [
            item
            for item in observation.get("containers") or []
            if item.get("node_id") in node_ids
        ]
        if not containers:
            continue
        attempts[str(observation.get("attempt_id"))] = observation
    if not attempts:
        return None
    if len(attempts) > 1:
        raise BootstrapError(
            "more than one active attempt is observed on the incident's nodes ("
            + ", ".join(sorted(attempts))
            + "); refusing to guess which one the disposition terminates"
        )
    return next(iter(attempts.values()))


def _allocation(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for container in observation.get("containers") or []:
        node_id = container.get("node_id")
        if not node_id:
            continue
        entry = grouped.setdefault(
            str(node_id),
            {"node_id": str(node_id), "gpu_uuids": [], "gpu_count": 0},
        )
        for uuid in container.get("gpu_uuids") or []:
            if uuid not in entry["gpu_uuids"]:
                entry["gpu_uuids"].append(uuid)
        entry["gpu_count"] += int(container.get("gpu_count") or 0)
        if container.get("instance_id") and "instance_id" not in entry:
            entry["instance_id"] = container["instance_id"]
    return [grouped[key] for key in sorted(grouped)]


def _terminal_payload(
    incident: Mapping[str, Any],
    *,
    disposition: str,
    observation: Mapping[str, Any] | None,
    profile_version: str,
    now: datetime,
) -> tuple[dict[str, Any], str]:
    incident_id = str(incident["incident_id"])
    if observation is not None:
        return (
            {
                "cluster_id": str(incident["cluster_id"]),
                "environment": str(
                    observation.get("environment") or DEFAULT_ENVIRONMENT
                ),
                "job_id": str(observation["job_id"]),
                "attempt_id": str(observation["attempt_id"]),
                "terminal_status": "FAILED",
                "ended_at": now.isoformat(),
                "allocation": _allocation(observation),
                "workload_ids": list(observation.get("workload_ids") or []),
                "runtime_profile_version": str(
                    observation.get("runtime_profile_version") or profile_version
                ),
                "restart_budget": int(observation.get("restart_budget") or 1),
            },
            "attempt-observation",
        )
    return (
        {
            "cluster_id": str(incident["cluster_id"]),
            "environment": DEFAULT_ENVIRONMENT,
            "job_id": f"operator-{incident_id}",
            "attempt_id": operator_attempt_id(incident_id, disposition),
            "terminal_status": "FAILED",
            "ended_at": now.isoformat(),
            "allocation": [
                {"node_id": node_id, "gpu_uuids": list(incident.get("gpu_uuids") or [])}
                for node_id in incident["node_ids"]
            ],
            "workload_ids": [],
            "runtime_profile_version": profile_version,
            "restart_budget": 1,
        },
        "idle-node",
    )


def _marker_payload(
    incident: Mapping[str, Any],
    *,
    disposition: str,
    action: RecoveryAction,
    reference: str | None,
    now: datetime,
) -> dict[str, Any]:
    incident_id = str(incident["incident_id"])
    return {
        "marker_id": operator_marker_id(incident_id, disposition),
        "source": OPERATOR_MARKER_SOURCE,
        "cluster_id": str(incident["cluster_id"]),
        "trusted": True,
        "incident_id": action_incident_id(incident_id, disposition),
        "observed_at": now.isoformat(),
        "expires_at": (now + OPERATOR_MARKER_TTL).isoformat(),
        "scope": {
            "node_ids": list(incident["node_ids"]),
            "gpu_uuids": list(incident.get("gpu_uuids") or []),
        },
        "severity": "critical",
        "recommended_action": action.value,
        "mapping_version": OPERATOR_MAPPING_VERSION,
        "raw_reason": (
            f"administrator disposition {disposition} after {WAITING_OPERATION} "
            f"on {incident_id}" + (f"; reference {reference}" if reference else "")
        ),
    }


def build_remediation_plan(
    inspection: Mapping[str, Any],
    *,
    incident_id: str,
    disposition: str,
    profile_version: str,
    reference: str | None = None,
    live_nodes: Mapping[str, Mapping[str, Any]] | None = None,
    resubmission: bool = False,
    now: datetime | None = None,
) -> RemediationPlan:
    """Turn the control plane's record into the exact writes, or refuse.

    ``resubmission`` relaxes only the "still waiting" check: the first
    submission's acknowledgement legitimately lets the investigation workflow
    move on, and the terminal decision it compiled is what the control plane
    returns as a duplicate.
    """

    if disposition not in DISPOSITION_ACTIONS:
        raise BootstrapError(f"unknown disposition {disposition!r}")
    stamp = now or _utc_now()
    incident, workflow = _validate_record(
        inspection, incident_id=incident_id, now=stamp
    )
    fencing_token = int(incident["fencing_token"])
    node_ids = tuple(str(item) for item in incident["node_ids"])
    expected_value = acknowledgement_value(incident_id, fencing_token)
    warnings: list[str] = []
    if resubmission:
        pending: tuple[str, ...] = ()
        next_operations: tuple[str, ...] = ()
    else:
        index, step = _waiting_step(workflow, incident_id, fencing_token)
        if set(step.get("node_ids") or ()) != set(node_ids):
            raise BootstrapError(
                f"the waiting step targets {sorted(step.get('node_ids') or ())} but "
                f"the incident names {sorted(node_ids)}; the node set changed"
            )
        pending = tuple(sorted(node_ids))
        completed = set(workflow.get("completed_step_indexes") or [])
        next_operations = tuple(
            str(item.get("operation"))
            for position, item in enumerate(workflow.get("official_steps") or [])
            if position > index and position not in completed
        )
    if live_nodes is not None:
        missing = sorted(set(node_ids) - set(live_nodes))
        if missing:
            raise BootstrapError(
                f"incident nodes are missing from cluster {incident['cluster_id']}: "
                + ", ".join(missing)
            )
        for node_id in node_ids:
            metadata = cast(dict[str, Any], live_nodes[node_id].get("metadata") or {})
            owner = (metadata.get("annotations") or {}).get(INCIDENT_ANNOTATION)
            if owner and str(owner) != incident_id:
                raise BootstrapError(
                    f"node {node_id} is isolated by incident {owner}, not {incident_id}"
                )
    action = DISPOSITION_ACTIONS[disposition]
    marker: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    source: str | None = None
    if action is not None:
        gpu_uuids = list(incident.get("gpu_uuids") or [])
        if action is RecoveryAction.RESET_GPU and not gpu_uuids:
            raise BootstrapError(
                f"{disposition} needs the incident to name the GPU UUIDs; "
                f"{incident_id} names none"
            )
        observation = _active_attempt(
            list(inspection.get("active_observations") or []), set(node_ids)
        )
        terminal, source = _terminal_payload(
            incident,
            disposition=disposition,
            observation=observation,
            profile_version=profile_version,
            now=stamp,
        )
        marker = _marker_payload(
            incident,
            disposition=disposition,
            action=action,
            reference=reference,
            now=stamp,
        )
        if source == "attempt-observation":
            warnings.append(
                f"attempt {terminal['attempt_id']} is active on the node; the "
                "disposition stops it (STOP_WORKLOADS -> action -> RESTART_WORKLOAD)"
            )
    return RemediationPlan(
        incident_id=incident_id,
        cluster_id=str(incident["cluster_id"]),
        disposition=disposition,
        node_ids=node_ids,
        gpu_uuids=tuple(str(item) for item in incident.get("gpu_uuids") or []),
        fencing_token=fencing_token,
        workflow_request_id=str(workflow["request_id"]),
        acknowledgement_value=expected_value,
        pending_nodes=pending,
        next_operations=next_operations,
        acknowledgement_timeout_seconds=(
            int(inspection["acknowledgement_timeout_seconds"])
            if inspection.get("acknowledgement_timeout_seconds") is not None
            else None
        ),
        action=None if action is None else action.value,
        action_incident_id=(
            None if action is None else action_incident_id(incident_id, disposition)
        ),
        marker=marker,
        terminal=terminal,
        workload_source=source,
        resubmission=resubmission,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# restore: the validated restore of a QUARANTINED incident


def _quarantine_taint(node: Mapping[str, Any]) -> str | None:
    spec = node.get("spec")
    if not isinstance(spec, dict):
        return None
    for item in spec.get("taints") or []:
        if isinstance(item, dict) and item.get("key") == QUARANTINE_TAINT:
            return str(item.get("value") or "")
    return None


def _node_annotations(node: Mapping[str, Any]) -> dict[str, Any]:
    metadata = cast(dict[str, Any], node.get("metadata") or {})
    return dict(metadata.get("annotations") or {})


def _validate_restore_record(
    inspection: Mapping[str, Any],
    *,
    incident_id: str,
    live_nodes: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """The restore preconditions, all before any write.

    Returns the incident, its last workflow and the id of a validated restore
    workflow already open under it (the idempotent rerun), or raises the
    first refusal: not QUARANTINED; another workflow still open; a node
    missing from the cluster, without the quarantine taint, or quarantined by
    another incident (named); incident and workflow fencing tokens differing
    (a stale generation the node's annotation would reject).
    """

    incident = cast(dict[str, Any] | None, inspection.get("incident"))
    workflow = cast(dict[str, Any] | None, inspection.get("workflow"))
    if incident is None:
        raise BootstrapError(f"incident {incident_id} was not returned")
    if str(incident.get("incident_id")) != incident_id:
        raise BootstrapError("the control plane returned a different incident")
    state = str(incident.get("state"))
    if state != QUARANTINED_STATE:
        raise BootstrapError(
            f"incident {incident_id} is {state}, not {QUARANTINED_STATE}; "
            f"{DISPOSITION_RESTORE} applies to a quarantined node only"
        )
    if workflow is None:
        raise BootstrapError(
            f"incident {incident_id} has no workflow; the quarantine it records "
            "was not placed by this product"
        )
    if workflow.get("request_id") != incident.get("workflow_request_id"):
        raise BootstrapError(
            f"incident {incident_id} points at workflow "
            f"{incident.get('workflow_request_id')} but "
            f"{workflow.get('request_id')} was returned"
        )
    if int(workflow.get("fencing_token") or 0) != int(
        incident.get("fencing_token") or 0
    ):
        raise BootstrapError(
            f"stale generation: incident fencing token {incident.get('fencing_token')} "
            f"differs from workflow {workflow.get('fencing_token')}"
        )
    node_ids = [str(item) for item in incident.get("node_ids") or []]
    if not node_ids:
        raise BootstrapError(f"incident {incident_id} names no nodes")
    open_rows = (
        [dict(workflow)] if workflow.get("status") in OPEN_WORKFLOW_STATUSES else []
    )
    for item in inspection.get("open_workflows") or []:
        if isinstance(item, dict) and item.get("request_id") != workflow.get(
            "request_id"
        ):
            open_rows.append(item)
    existing = [
        str(item.get("request_id"))
        for item in open_rows
        if is_validated_restore_workflow(str(item.get("request_id") or ""))
    ]
    if existing:
        return incident, workflow, existing[0]
    if open_rows:
        raise BootstrapError(
            f"incident {incident_id} still has an open workflow "
            + ", ".join(
                f"{item.get('request_id')} ({item.get('status')})" for item in open_rows
            )
            + "; wait for it to end or reconcile it first"
        )
    missing = sorted(set(node_ids) - set(live_nodes))
    if missing:
        raise BootstrapError(
            f"incident nodes are missing from cluster {incident['cluster_id']}: "
            + ", ".join(missing)
        )
    owned = owned_quarantine_taint_values(incident_id)
    for node_id in node_ids:
        node = live_nodes[node_id]
        annotations = _node_annotations(node)
        owner = annotations.get(ANNOTATION_INCIDENT)
        taint = _quarantine_taint(node)
        if owner and str(owner) != incident_id:
            raise BootstrapError(
                f"node {node_id} is isolated by incident {owner}, not {incident_id}"
            )
        if taint is None:
            raise BootstrapError(
                f"node {node_id} carries no {QUARANTINE_TAINT} taint; there is no "
                f"isolation of {incident_id} to restore -- close the incident with "
                "gpu-fault-admin workflow-reconcile --close-incident instead"
            )
        if taint not in owned:
            # No incident-id annotation named another owner above, so the
            # taint value is all that identifies who placed it.
            raise BootstrapError(
                f"node {node_id} is quarantined by another incident (taint {taint}), "
                f"not {incident_id}"
            )
    return incident, workflow, None


def build_restore_plan(
    inspection: Mapping[str, Any],
    *,
    incident_id: str,
    operator: str,
    reference: str | None,
    live_nodes: Mapping[str, Mapping[str, Any]],
    now: datetime | None = None,
) -> RestorePlan:
    """Turn the QUARANTINED record into the restore it will get, or refuse."""

    stamp = now or _utc_now()
    incident, workflow, existing = _validate_restore_record(
        inspection, incident_id=incident_id, live_nodes=live_nodes
    )
    record = FaultIncident.model_validate(incident)
    _, preview = build_validated_restore_workflow(
        record, operator=operator, reference=reference, now=stamp
    )
    return RestorePlan(
        incident_id=incident_id,
        cluster_id=str(incident["cluster_id"]),
        node_ids=tuple(str(item) for item in incident["node_ids"]),
        gpu_uuids=tuple(str(item) for item in incident.get("gpu_uuids") or []),
        fencing_token=int(incident["fencing_token"]),
        workflow_request_id=str(workflow["request_id"]),
        state=str(incident["state"]),
        reason=restore_reason(operator, reference),
        steps=tuple(
            {
                "operation": step.operation.value,
                "execution_owner": step.execution_owner,
                "node_ids": list(step.node_ids),
                "gpu_uuids": list(step.gpu_uuids),
            }
            for step in preview.official_steps
        ),
        existing_restore_workflow_id=existing,
    )


def submit_restore(
    site: RenderedSite,
    plan: RestorePlan,
    *,
    operator: str,
    reference: str | None,
) -> dict[str, Any]:
    """Create the restore workflow in the Pod, re-checking the record first."""

    return run_control_plane_script(
        site,
        {
            "mode": "restore",
            "incident_id": plan.incident_id,
            "expected": {
                "fencing_token": plan.fencing_token,
                "workflow_request_id": plan.workflow_request_id,
                "state": plan.state,
                "node_ids": sorted(plan.node_ids),
            },
            "operator": operator,
            "reference": reference,
            "runtime_profile_version": str(
                site.release_config["runtime_profile"]["version"]
            ),
        },
        script=REMEDIATION_SCRIPT,
    )


def _submit_restore(
    request: SubmitRemediationRequest, *, now: Callable[[], datetime]
) -> SubmissionResult:
    site = request.site
    stamp = now()
    operator = resolve_operator_identity()
    inspection = inspect_incident(
        site, request.incident_id, disposition=request.disposition
    )
    incident = cast(dict[str, Any], inspection.get("incident") or {})
    live_nodes = cluster_nodes(site, str(incident.get("cluster_id") or ""))
    plan = build_restore_plan(
        inspection,
        incident_id=request.incident_id,
        operator=operator,
        reference=request.reference,
        live_nodes=live_nodes,
        now=stamp,
    )
    result = SubmissionResult(plan=plan)
    operations = " -> ".join(str(item["operation"]) for item in plan.steps)
    if request.plan_only:
        result.message = (
            f"plan only; nothing was written (would create {operations} under "
            f"incident {plan.incident_id}, fencing token {plan.fencing_token})"
        )
        return result
    if plan.existing_restore_workflow_id:
        result.no_op = True
        result.message = (
            f"restore workflow {plan.existing_restore_workflow_id} is already open "
            f"under incident {plan.incident_id}; nothing to do"
        )
        record_submission(site, result, now=stamp)
        return result
    submission = submit_restore(
        site, plan, operator=operator, reference=request.reference
    )
    result.submission = submission
    workflow = cast(dict[str, Any], submission.get("workflow") or {})
    if submission.get("no_op"):
        result.no_op = True
        result.message = (
            f"restore workflow {workflow.get('request_id')} is already open under "
            f"incident {plan.incident_id}; nothing to do"
        )
    else:
        result.message = (
            f"workflow {workflow.get('request_id')} (fencing token "
            f"{workflow.get('fencing_token')}, {workflow.get('status')}) runs "
            f"{operations}; incident {plan.incident_id} is now "
            f"{(submission.get('incident') or {}).get('state')}"
        )
    record_submission(site, result, now=stamp)
    return result


# --------------------------------------------------------------------------
# Writes


def _site_cluster(site: RenderedSite, cluster_id: str) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in site.release_config["clusters"]
        if str(item.get("cluster_id")) == cluster_id
    ]
    if len(matches) != 1:
        raise BootstrapError(
            f"incident cluster {cluster_id} is not in the managed site"
        )
    return matches[0]


def acknowledge_inspection(
    site: RenderedSite,
    plan: RemediationPlan,
    *,
    live_nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the acknowledgement onto the nodes that do not carry it yet."""

    target = _site_cluster(site, plan.cluster_id)
    already = []
    written = []
    for node_id in plan.node_ids:
        metadata = cast(dict[str, Any], live_nodes[node_id].get("metadata") or {})
        current = (metadata.get("annotations") or {}).get(
            ANNOTATION_MECHANICAL_INSPECTION_COMPLETE
        )
        if current == plan.acknowledgement_value:
            already.append(node_id)
            continue
        completed = subprocess.run(
            [
                *gpu_kubectl_command(site, target),
                "annotate",
                "node",
                node_id,
                f"{ANNOTATION_MECHANICAL_INSPECTION_COMPLETE}="
                f"{plan.acknowledgement_value}",
                "--overwrite",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise BootstrapError(
                f"cannot annotate node {node_id}: "
                + (completed.stderr.strip() or "kubectl annotate failed")
            )
        written.append(node_id)
    return {
        "annotation": ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
        "value": plan.acknowledgement_value,
        "annotated": written,
        "already_acknowledged": already,
    }


def evidence_directory(site: RenderedSite, incident_id: str) -> Path:
    return site.source.parent / STATE_ROOT / safe_name(incident_id)


def previous_submission(
    site: RenderedSite, incident_id: str, disposition: str
) -> dict[str, Any] | None:
    directory = evidence_directory(site, incident_id)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(f"{disposition}-*.json"))
    for path in reversed(candidates):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("status") == "SUBMITTED":
            return cast(dict[str, Any], value)
    return None


def record_submission(
    site: RenderedSite, result: SubmissionResult, *, now: datetime
) -> Path:
    directory = evidence_directory(site, result.plan.incident_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{result.plan.disposition}-{now:%Y%m%dT%H%M%SZ}.json"
    write_json_atomic(
        path,
        {
            "status": "NO_OP" if result.no_op else "SUBMITTED",
            "recorded_at": now.isoformat(),
            "operator": resolve_operator_identity(),
            "fencing_token": result.plan.fencing_token,
            **result.as_dict(),
        },
    )
    return path


def _describe_next(plan: RemediationPlan, submission: Mapping[str, Any] | None) -> str:
    if submission is None:
        if plan.next_operations:
            return (
                f"workflow {plan.workflow_request_id} completes {WAITING_OPERATION} "
                "and continues with " + " -> ".join(plan.next_operations)
            )
        return f"workflow {plan.workflow_request_id} completes {WAITING_OPERATION}"
    workflow = cast(dict[str, Any] | None, submission.get("workflow"))
    if workflow is None:
        decision = cast(dict[str, Any], submission.get("decision") or {})
        return (
            f"decision {decision.get('status')} ({decision.get('reason')}); "
            "no workflow was compiled"
        )
    operations = [
        str(item.get("operation")) for item in workflow.get("official_steps") or []
    ]
    return (
        f"workflow {workflow.get('request_id')} (fencing token "
        f"{workflow.get('fencing_token')}, {workflow.get('status')}) runs "
        + " -> ".join(operations)
    )


def submit_remediation(
    request: SubmitRemediationRequest,
    *,
    now: Callable[[], datetime] = _utc_now,
) -> SubmissionResult:
    """Inspect, validate, acknowledge and (for hardware dispositions) submit;
    ``restore`` inspects, validates and creates the validated restore."""

    if request.disposition == DISPOSITION_RESTORE:
        return _submit_restore(request, now=now)
    site = request.site
    stamp = now()
    inspection = inspect_incident(
        site, request.incident_id, disposition=request.disposition
    )
    previous = previous_submission(site, request.incident_id, request.disposition)
    incident = cast(dict[str, Any], inspection.get("incident") or {})
    resubmission = bool(
        previous is not None
        and previous.get("fencing_token") == incident.get("fencing_token")
        and request.disposition != DISPOSITION_INSPECTED
    )
    live_nodes = cluster_nodes(site, str(incident.get("cluster_id") or ""))
    plan = build_remediation_plan(
        inspection,
        incident_id=request.incident_id,
        disposition=request.disposition,
        profile_version=str(site.release_config["runtime_profile"]["version"]),
        reference=request.reference,
        live_nodes=live_nodes,
        resubmission=resubmission,
        now=stamp,
    )
    result = SubmissionResult(plan=plan)
    if request.plan_only:
        result.message = "plan only; nothing was written"
        return result
    result.acknowledgement = acknowledge_inspection(site, plan, live_nodes=live_nodes)
    if plan.action is None:
        result.no_op = not result.acknowledgement["annotated"]
        result.message = (
            "already acknowledged; nothing to do"
            if result.no_op
            else _describe_next(plan, None)
        )
        record_submission(site, result, now=stamp)
        return result
    submission = submit_operator_action(site, plan)
    result.submission = submission
    decision = cast(dict[str, Any], submission.get("decision") or {})
    marker_id = str((plan.marker or {}).get("marker_id"))
    matched = [str(item) for item in decision.get("matched_marker_ids") or []]
    if submission.get("duplicate"):
        result.no_op = True
        if marker_id in matched:
            result.message = (
                f"already submitted: decision for attempt {decision.get('attempt_id')} "
                "exists and matched this disposition's marker; "
                + _describe_next(plan, submission)
            )
        else:
            record_submission(site, result, now=stamp)
            raise BootstrapError(
                f"attempt {decision.get('attempt_id')} already has a completion "
                f"decision that did not match marker {marker_id}; inspect workflow "
                f"{(submission.get('workflow') or {}).get('request_id')} before "
                "deciding -- do not fabricate a new attempt id"
            )
    elif marker_id not in matched:
        record_submission(site, result, now=stamp)
        raise BootstrapError(
            f"the terminal decision did not match marker {marker_id} "
            f"(matched: {matched or 'none'}); the compiled plan does not carry the "
            "operator disposition"
        )
    else:
        result.message = _describe_next(plan, submission)
    record_submission(site, result, now=stamp)
    return result


# --------------------------------------------------------------------------
# CLI


def add_submit_remediation_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin submit-remediation``; the CLI passes site options."""

    command = commands.add_parser(
        "submit-remediation",
        usage=(
            "gpu-fault-admin submit-remediation --state-dir STATE_DIR "
            "--incident-id INCIDENT_ID --disposition "
            "{" + ",".join(DISPOSITIONS) + "} [--reference CHANGE_ID] [--plan]"
        ),
        help=(
            "acknowledge a CHECK_MECHANICALS inspection and, for a hardware "
            "disposition, submit the operator marker and terminal trigger built "
            "from the incident record; restore: create the validated restore of "
            "a QUARANTINED incident whose node was repaired by hand"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--incident-id",
        required=True,
        metavar="INCIDENT_ID",
        help=(
            "the incident whose workflow is waiting on CHECK_MECHANICALS, or "
            "(restore) the QUARANTINED incident that owns the node's isolation"
        ),
    )
    command.add_argument(
        "--disposition",
        required=True,
        choices=DISPOSITIONS,
        help=(
            "inspected: inspection complete, no hardware action; reset-gpu, "
            "reboot-node, quarantine: compile a new fenced workflow for that action; "
            "restore: VALIDATE_GPU -> VALIDATE_HOST -> VALIDATE_FABRIC -> "
            "RESTORE_SCHEDULING under the same QUARANTINED incident"
        ),
    )
    command.add_argument(
        "--reference",
        metavar="CHANGE_ID",
        help="change ticket recorded on the marker and the evidence file",
    )
    command.add_argument(
        "--plan",
        action="store_true",
        help="print what would be written and exit without writing",
    )


def run_submit_remediation_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    """Submit the disposition named on the command line."""

    request = SubmitRemediationRequest(
        site=site,
        incident_id=str(arguments.incident_id),
        disposition=str(arguments.disposition),
        reference=cast(str | None, arguments.reference),
        plan_only=bool(arguments.plan),
    )
    result = submit_remediation(request)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0
