from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from gpu_fault.models import (
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.workflow_resolution import (
    restore_reconciliation_reasons,
    verified_restore_successor,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _workflow_ids(
    store: ControlPlaneStore,
    requested: Iterable[str] | None,
) -> list[str]:
    if requested:
        values = sorted({str(item).strip() for item in requested if str(item).strip()})
        if len(values) > 1000:
            raise ValueError("workflow reconcile accepts at most 1000 workflow IDs")
        return values
    workflows = store.list_workflows(
        statuses={WorkflowStatus.BLOCKED},
        limit=1001,
    )
    if len(workflows) > 1000:
        raise ValueError("workflow reconcile found more than 1000 BLOCKED workflows")
    return sorted(item.request_id for item in workflows)


def build_workflow_reconcile_plan(
    store: ControlPlaneStore,
    workflow_ids: Iterable[str] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    evaluated_at = now or datetime.now(timezone.utc)
    request_ids = _workflow_ids(store, workflow_ids)
    # Scoped to the workflows under reconciliation rather than reading the whole
    # command history. Every use of this list -- here and inside
    # ``restore_reconciliation_reasons`` -- filters on
    # ``workflow_request_id == workflow.request_id``, and ``_workflow_ids``
    # refuses to return more than 1000 ids, so the narrowed read answers exactly
    # the same question against a bounded number of rows.
    commands = store.list_remote_commands(workflow_request_ids=request_ids)
    items = []
    for request_id in request_ids:
        try:
            workflow = store.get_workflow(request_id)
        except (KeyError, NotFoundError):
            items.append(
                {
                    "request_id": request_id,
                    "eligible": False,
                    "reasons": ["workflow does not exist"],
                }
            )
            continue
        open_commands = sorted(
            item.command_id
            for item in commands
            if item.workflow_request_id == request_id
            and item.status
            in {
                RemoteCommandStatus.PENDING,
                RemoteCommandStatus.WAITING,
                RemoteCommandStatus.LEASED,
            }
        )
        waiting_steps = sorted(
            item.step_index
            for item in workflow.step_executions
            if item.status is WorkflowStepStatus.WAITING
        )
        successor = verified_restore_successor(store, workflow)
        source_plan = None
        if workflow.source_plan_id:
            try:
                source_plan = store.get_plan(workflow.source_plan_id)
            except (KeyError, NotFoundError):
                source_plan = None
        try:
            incident = store.get_incident(workflow.incident_id)
        except (KeyError, NotFoundError):
            incident = None
        reasons = restore_reconciliation_reasons(
            workflow,
            incident,
            successor,
            source_plan,
            commands,
            evaluated_at=evaluated_at,
        )
        items.append(
            {
                "request_id": request_id,
                "incident_id": workflow.incident_id,
                "cluster_id": incident.cluster_id if incident is not None else None,
                "node_ids": sorted(incident.node_ids) if incident is not None else [],
                "fencing_token": workflow.fencing_token,
                "workflow_updated_at": workflow.updated_at.isoformat(),
                "successor_workflow_id": (
                    successor.request_id if successor is not None else None
                ),
                "source_plan_id": workflow.source_plan_id,
                "source_plan_status": (
                    source_plan.status.value if source_plan is not None else None
                ),
                "open_remote_commands": open_commands,
                "waiting_step_indexes": waiting_steps,
                "eligible": not reasons,
                "reasons": reasons,
            }
        )
    plan = {
        "schema_version": 1,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": evaluated_at.isoformat(),
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            "items": plan["items"],
        }
    )
    return plan


def apply_workflow_reconcile_plan(
    store: ControlPlaneStore,
    *,
    workflow_ids: Iterable[str],
    expected_plan_sha256: str,
    reference: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    applied_at = now or datetime.now(timezone.utc)
    plan = build_workflow_reconcile_plan(
        store,
        workflow_ids,
        now=applied_at,
    )
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("workflow reconcile plan changed before apply")
    rejected = [item for item in plan["items"] if not item["eligible"]]
    if rejected:
        raise ValueError("workflow reconcile plan contains ineligible records")
    applied = []
    resolved_plan_ids = []
    incident_ids = set()
    for item in plan["items"]:
        updated_workflow, incident, source_plan = store.reconcile_restored_workflow(
            item["request_id"],
            item["successor_workflow_id"],
            expected_fencing_token=int(item["fencing_token"]),
            expected_workflow_updated_at=datetime.fromisoformat(
                item["workflow_updated_at"]
            ),
            reference=reference,
            reconciled_at=applied_at,
        )
        incident_ids.add(incident.incident_id)
        applied.append(updated_workflow.request_id)
        resolved_plan_ids.append(source_plan.plan_id)
    return {
        "schema_version": 1,
        "mode": "workflow-reconcile-apply",
        "plan_sha256": expected_plan_sha256,
        "reference": reference,
        "applied_at": applied_at.isoformat(),
        "applied_workflow_ids": applied,
        "resolved_plan_ids": resolved_plan_ids,
        "archive_eligible_incident_ids": sorted(incident_ids),
        "records_deleted": 0,
    }
