"""CPU-only, guarded lifecycle of the live probe's auditable workflow holder."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
)

FINISH_REASON = "owned physical acceptance finished; no restart authorized"


def hold_workflow(
    store: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    *,
    run_id: str,
    window_end: datetime,
) -> dict[str, Any]:
    if (
        window_end.tzinfo is None
        or not 0 < (window_end - datetime.now(timezone.utc)).total_seconds() <= 3600
        or workflow.status is not WorkflowStatus.RUNNING
        or incident.state is not IncidentState.ACTION_PENDING
        or workflow.incident_id != incident.incident_id
        or workflow.request_id != incident.workflow_request_id
        or incident.drill_id != run_id
        or workflow.fencing_token != incident.fencing_token
    ):
        raise ValueError("acceptance workflow binding is incomplete")
    try:
        store.get_workflow(workflow.request_id)
    except KeyError:
        pass
    else:
        raise ValueError("acceptance workflow already exists")
    try:
        store.get_incident(incident.incident_id)
    except KeyError:
        pass
    else:
        raise ValueError("acceptance incident already exists")
    held = workflow.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "execution_owner_id": f"late-ownership/{run_id}",
            "execution_epoch": 1,
            # A lost probe cannot be taken over until its mutation lifetime ended.
            "execution_lease_expires_at": window_end + timedelta(minutes=10),
            "lifetime_deadline_at": window_end,
        }
    )
    store.save_incident_and_workflow(incident, held)
    observed = store.get_workflow(held.request_id)
    if observed != held:
        raise ValueError("acceptance workflow holder was not confirmed")
    return {
        "workflow": held.model_dump(mode="json"),
        "incident": incident.model_dump(mode="json"),
    }


def finish_workflow(
    store: Any,
    observed: WorkflowRequest,
    *,
    run_id: str,
) -> dict[str, Any]:
    current = store.get_workflow(observed.request_id)
    terminal_owned = (
        current.status is WorkflowStatus.SUPERSEDED
        and current.execution_owner_id is None
        and current.execution_lease_expires_at is None
        and current.workload_withdrawn_reason == FINISH_REASON
        and current.workload_withdrawn_at == observed.updated_at
        and current.step_executions == observed.step_executions
        and current.completed_step_indexes == observed.completed_step_indexes
    )
    if (
        current.execution_epoch != observed.execution_epoch
        or current.fencing_token != observed.fencing_token
        or current.incident_id != observed.incident_id
        or current.official_steps != observed.official_steps
        or (
            not terminal_owned
            and (
                current.execution_owner_id != f"late-ownership/{run_id}"
                or current.status is not WorkflowStatus.RUNNING
            )
        )
    ):
        raise ValueError("acceptance workflow holder changed before completion")
    incident = store.get_incident(current.incident_id)
    if (
        incident.drill_id != run_id
        or incident.fencing_token != current.fencing_token
        or incident.workflow_request_id != current.request_id
    ):
        raise ValueError("acceptance incident holder changed before completion")
    complete = (
        current
        if terminal_owned
        else current.model_copy(
            update={
                "status": WorkflowStatus.SUPERSEDED,
                "step_executions": observed.step_executions,
                "completed_step_indexes": observed.completed_step_indexes,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "workload_withdrawn_at": observed.updated_at,
                "workload_withdrawn_reason": FINISH_REASON,
                "preemption_reason": "owned physical acceptance stopped",
                "updated_at": observed.updated_at,
            }
        )
    )
    if not terminal_owned:
        store.save_workflow(complete, expected=current)
    recovered = incident.model_copy(update={"state": IncidentState.RECOVERED})
    if incident != recovered:
        store.save_incident(recovered, expected=incident)
    final = store.get_workflow(complete.request_id)
    if final != complete or store.get_incident(complete.incident_id) != recovered:
        raise ValueError("acceptance workflow terminal state was not confirmed")
    return {
        "workflow_id": final.request_id,
        "fencing_token": final.fencing_token,
        "execution_epoch": final.execution_epoch,
        "status": final.status.value,
    }


def check_workflow(
    store: Any, observed: WorkflowRequest, *, run_id: str
) -> dict[str, Any]:
    current = store.get_workflow(observed.request_id)
    incident = store.get_incident(current.incident_id)
    now = datetime.now(timezone.utc)
    if (
        current.request_id != observed.request_id
        or current.incident_id != observed.incident_id
        or current.execution_owner_id != f"late-ownership/{run_id}"
        or current.execution_epoch != observed.execution_epoch
        or current.fencing_token != observed.fencing_token
        or current.official_steps != observed.official_steps
        or current.status is not WorkflowStatus.RUNNING
        or current.execution_lease_expires_at is None
        or current.execution_lease_expires_at.tzinfo is None
        or current.execution_lease_expires_at <= now
        or current.lifetime_deadline_at is None
        or current.lifetime_deadline_at != observed.lifetime_deadline_at
        or current.lifetime_deadline_at.tzinfo is None
        or current.lifetime_deadline_at <= now
        or incident.drill_id != run_id
        or incident.workflow_request_id != current.request_id
        or incident.fencing_token != current.fencing_token
        or incident.state is not IncidentState.ACTION_PENDING
    ):
        raise ValueError("owned acceptance workflow lease is no longer current")
    return {
        "workflow_id": current.request_id,
        "fencing_token": current.fencing_token,
        "execution_epoch": current.execution_epoch,
        "holder_valid": True,
        "lifetime_deadline_at": current.lifetime_deadline_at.isoformat(),
    }


def cpu_program() -> str:
    """Send trusted source, not a database URL or execution credential."""
    from pathlib import Path

    source = Path(__file__).read_text(encoding="utf-8")
    return source + "\nraise SystemExit(main())\n"


def dispatch(store: Any, value: dict[str, Any]) -> dict[str, Any]:
    workflow = WorkflowRequest.model_validate(value["workflow"])
    if value["action"] == "hold":
        return hold_workflow(
            store,
            workflow,
            FaultIncident.model_validate(value["incident"]),
            run_id=value["run_id"],
            window_end=datetime.fromisoformat(value["window_end"]),
        )
    if value["action"] == "finish":
        return finish_workflow(store, workflow, run_id=value["run_id"])
    if value["action"] == "check":
        return check_workflow(store, workflow, run_id=value["run_id"])
    raise ValueError("unsupported acceptance state action")


def main() -> int:
    from gpu_fault.app import ApplicationContext

    try:
        value = json.loads(sys.argv[1])
        result = dispatch(ApplicationContext.from_environment().store, value)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error_kind": type(exc).__name__}))
        return 1
