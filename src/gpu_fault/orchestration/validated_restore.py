"""The validated restore an operator requests for a QUARANTINED node.

A QUARANTINED incident holds its node cordoned and tainted until a workflow
validates the hardware and puts scheduling back: VALIDATE_GPU -> VALIDATE_HOST
-> VALIDATE_FABRIC -> RESTORE_SCHEDULING, run under the *same* incident so the
kubernetes adapter's ownership check (``_node_restore_patch``: incident id and
fencing token on the node must match) lets the restore release the isolation.
Until 2026-09-10 the only thing that created such a workflow was the acceptance
fixture (``scripts/e2e/regional/warm_spare_fixture.py``), which hand-built the
``WorkflowRequest`` inside the CPU Pod; ``gpu-fault-admin submit-remediation
--disposition restore`` now does it for a node an operator repaired by hand,
and both build the pair here so the product and the fixture cannot drift.

``build_validated_restore_workflow`` is pure: it returns the updated incident
and the new workflow; the caller persists them with
``store.save_incident_and_workflow`` (one transaction) and wakes the
dispatcher. It does not check preconditions -- state, open workflows, node
ownership -- because those need the store and the cluster; the admin verb
(``gpu_fault.admin.submit_remediation``) and the fixture check them first.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence
from uuid import uuid4

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    bounded_reasons,
)

RESTORE_WORKFLOW_PREFIX = "workflow-validated-restore-"
RESTORE_OFFICIAL_ACTION = WorkflowOperation.RESTORE_SCHEDULING.value
VALIDATED_RESTORE_OPERATIONS: tuple[WorkflowOperation, ...] = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
)
# Execution owners as the runtime profiles bind them: validation steps run on
# the validation adapter, the release on the kubernetes adapter.
KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
VALIDATION_OWNER = "gpu-fault-validation-adapter"


def is_validated_restore_workflow(request_id: str | None) -> bool:
    return bool(request_id) and str(request_id).startswith(RESTORE_WORKFLOW_PREFIX)


def restore_reason(operator: str, reference: str | None) -> str:
    """The incident reason an operator-requested restore appends."""

    return f"operator restore requested by {operator} ({reference or 'no reference'})"


def validated_restore_steps(
    node_ids: Sequence[str], gpu_uuids: Sequence[str] = ()
) -> list[WorkflowStepSpec]:
    """The four steps over ``node_ids``; ``gpu_uuids`` scopes the validation."""

    nodes = list(node_ids)
    gpus = list(gpu_uuids)
    return [
        WorkflowStepSpec(
            operation=operation,
            execution_owner=(
                KUBERNETES_OWNER
                if operation is WorkflowOperation.RESTORE_SCHEDULING
                else VALIDATION_OWNER
            ),
            node_ids=nodes,
            gpu_uuids=gpus,
        )
        for operation in VALIDATED_RESTORE_OPERATIONS
    ]


def build_validated_restore_workflow(
    incident: FaultIncident,
    *,
    operator: str,
    reference: str | None,
    now: datetime,
    node_ids: Sequence[str] | None = None,
    runtime_profile_version: str | None = None,
    reason: str | None = None,
) -> tuple[FaultIncident, WorkflowRequest]:
    """The ``(incident, workflow)`` pair of a validated restore of ``incident``.

    The workflow is PENDING under the incident's current fencing token (the
    token the node's isolation annotation carries -- a bumped token would fail
    the adapter's ownership check), official action RESTORE_SCHEDULING, steps
    ``VALIDATED_RESTORE_OPERATIONS`` over ``node_ids`` (default: the
    incident's nodes). The incident's GPU scope goes on the steps only when
    the steps cover exactly the incident's nodes: a restore of some other node
    (the fixture restoring a spare) must not name GPUs that node does not have
    (DESTR-003, 2026-09-08). The incident moves to ACTION_PENDING, points at
    the workflow, and records ``reason`` (default ``restore_reason``).
    """

    step_nodes = sorted(set(node_ids if node_ids is not None else incident.node_ids))
    if not step_nodes:
        raise ValueError(f"incident {incident.incident_id} names no nodes to restore")
    gpu_uuids = (
        list(incident.gpu_uuids) if set(step_nodes) == set(incident.node_ids) else []
    )
    workflow = WorkflowRequest(
        request_id=f"{RESTORE_WORKFLOW_PREFIX}{uuid4()}",
        incident_id=incident.incident_id,
        runtime_profile_version=runtime_profile_version,
        status=WorkflowStatus.PENDING,
        official_action=RESTORE_OFFICIAL_ACTION,
        fencing_token=incident.fencing_token,
        official_steps=validated_restore_steps(step_nodes, gpu_uuids),
        created_at=now,
        updated_at=now,
    )
    updated = incident.model_copy(
        update={
            "state": IncidentState.ACTION_PENDING,
            "node_ids": sorted(set(incident.node_ids) | set(step_nodes)),
            "workflow_request_id": workflow.request_id,
            "reasons": bounded_reasons(
                [*incident.reasons, reason or restore_reason(operator, reference)]
            ),
            "updated_at": now,
        }
    )
    return updated, workflow
