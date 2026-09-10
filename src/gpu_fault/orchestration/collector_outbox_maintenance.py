"""The workflow behind ``gpu-fault-admin collector-outbox`` (ARCH-G2).

Everything that reaches a node agent is a NODE_ACTION step of a workflow, so
an operator's remote ``gpu-fault-collector outbox`` command is one too: an
operator-created incident and a single-step workflow, built here, persisted
from inside the CPU Pod (``store.save_incident_and_workflow``, one
transaction), the dispatcher woken, the CLI waiting for the terminal state and
printing the node result. The same shape as ``submit-remediation --disposition
restore`` (``validated_restore``), minus the precondition checks: the request
has no node isolation to respect, only a node to reach.

``build_collector_outbox_workflow`` is pure: it returns the pair and never
touches the store. The one runtime precondition -- the node's agent must
advertise ``COLLECTOR_OUTBOX_MAINTENANCE`` in its heartbeat, or it predates
the operation -- needs the store and is checked by the admin verb's in-Pod
script before it builds anything.

Terminal states, as the executor derives them (``_terminal_incident_state``):
a SUCCEEDED step leaves the incident RECOVERED; a FAILED step (lock held,
unreadable outbox, agent refusal) also leaves it RECOVERED, because a workflow
of nothing but a non-destructive, non-node-wide step is *diagnostic-only* to
``_failure_incident_state`` -- the failure is recorded on the incident's
reasons as an inconclusive diagnostic instead of parking an ESCALATED record
that the operator, who reads the failure on the CLI, would then have to close
by hand. ``tests/orchestration/test_collector_outbox_maintenance.py`` pins both.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from gpu_fault.collectors.outbox_maintenance import OutboxMaintenanceRequest
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    bounded_reasons,
)

OUTBOX_WORKFLOW_PREFIX = "workflow-collector-outbox-"
OUTBOX_INCIDENT_PREFIX = "inc-operator-outbox-"
OUTBOX_OFFICIAL_ACTION = WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE.value
#: An incident an operator opened for maintenance, not a fault the policy saw.
OPERATOR_EVENT_TYPE = "OPERATOR_MAINTENANCE"
OPERATOR_EVENT_SOURCE = "gpu-fault-admin"
OPERATOR_POLICY_SOURCE = "OPERATOR"
OPERATOR_POLICY_VERSION = "operator"
# The node-action adapter's owner as every runtime profile binds it
# (``GPU_FAULT_NODE_ACTION_OWNER`` default).
NODE_ACTION_OWNER = "gpu-fault-node-agent"


def is_collector_outbox_workflow(request_id: str | None) -> bool:
    return bool(request_id) and str(request_id).startswith(OUTBOX_WORKFLOW_PREFIX)


def outbox_incident_id(node_id: str, action: str, now: datetime) -> str:
    """``inc-operator-outbox-<node>-<action>-<utc compact time>``."""

    return f"{OUTBOX_INCIDENT_PREFIX}{node_id}-{action}-{now:%Y%m%dT%H%M%SZ}"


def outbox_reason(
    operator: str, reference: str | None, request: OutboxMaintenanceRequest
) -> str:
    """The incident reason: who asked for what, on which collector."""

    scope = f" path {request.path}" if request.path else ""
    return (
        f"operator collector outbox {request.action} on {request.collector}{scope} "
        f"requested by {operator} ({reference or 'no reference'})"
    )


def build_collector_outbox_workflow(
    cluster_id: str,
    node_id: str,
    *,
    collector: str,
    action: str,
    confirm: bool,
    path: str | None,
    operator: str,
    reference: str | None,
    now: datetime,
    runtime_profile_version: str | None = None,
) -> tuple[FaultIncident, WorkflowRequest]:
    """The ``(incident, workflow)`` pair of one remote outbox command.

    The request is validated first (``OutboxMaintenanceRequest``: known
    collector, known action, ``requeue-dead`` confirmed, path filter shaped
    like a control-plane path) and ``ValueError`` names the first bad field.
    The incident is ACTION_PENDING, policy source OPERATOR, fencing token 1,
    pointing at the workflow; the workflow is PENDING with the single
    NODE_ACTION step on ``node_id`` whose parameters are the request plus the
    operator and reference (the node agent writes them into a requeued
    record's error prefix).
    """

    request = OutboxMaintenanceRequest(
        collector=collector, action=action, confirm=confirm, path=path
    )
    if not node_id.strip():
        raise ValueError("collector outbox maintenance needs a node")
    if not operator.strip():
        raise ValueError("collector outbox maintenance needs the operator identity")
    incident_id = outbox_incident_id(node_id, request.action, now)
    workflow = WorkflowRequest(
        request_id=f"{OUTBOX_WORKFLOW_PREFIX}{uuid4()}",
        incident_id=incident_id,
        runtime_profile_version=runtime_profile_version,
        status=WorkflowStatus.PENDING,
        official_action=OUTBOX_OFFICIAL_ACTION,
        fencing_token=1,
        official_steps=[
            WorkflowStepSpec(
                operation=WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE,
                execution_owner=NODE_ACTION_OWNER,
                node_ids=[node_id],
                parameters={
                    **request.as_parameters(),
                    "operator": operator,
                    "reference": reference,
                },
            )
        ],
        created_at=now,
        updated_at=now,
    )
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=f"{incident_id}:event",
        event_type=OPERATOR_EVENT_TYPE,
        event_source=OPERATOR_EVENT_SOURCE,
        cluster_id=cluster_id,
        node_ids=[node_id],
        policy_version=OPERATOR_POLICY_VERSION,
        policy_source=OPERATOR_POLICY_SOURCE,
        policy_reference=reference,
        official_action=OUTBOX_OFFICIAL_ACTION,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
        fencing_token=1,
        reasons=bounded_reasons([outbox_reason(operator, reference, request)]),
        created_at=now,
        updated_at=now,
    )
    return incident, workflow
