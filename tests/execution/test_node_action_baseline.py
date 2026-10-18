"""The STOP receipt as the pre-action boot of ``confirm-node-action`` (batch 139, G4).

``execution/node_action_baseline.py`` reads the boot a node ran when a node
action of the workflow started from the workflow's own contained STOP receipt,
carried along the workflow's confirmed reboots; the incident's fault-time
``source_boot_id`` is the fallback only for a workflow without such a receipt.
These tests pin each branch: the receipt as is, the chain carried through a
confirmed reboot, a reboot the chain cannot follow (no baseline at all), a
reboot that never left the executor (skipped), and the receipts that do not
apply (another workflow's, uncontained, completed after the action).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.execution.node_action_baseline import (
    STOP_RECEIPT_CHAIN_SOURCE,
    STOP_RECEIPT_SOURCE,
    UNPROVABLE,
    stop_receipt_boot,
)
from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    confirm_node_action,
)
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY
from tests._builders import (
    contained_stop_receipt,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
STOP_AT = NOW - timedelta(hours=2)
NODE = "hyperpod-i-0b139"
SIBLING = "hyperpod-i-0b139sib"
WORKFLOW = "workflow-b139-baseline"
INCIDENT = "incident-b139-baseline"
RECEIPT_BOOT = "boot-at-the-stop"
REBOOTED_BOOT = "boot-after-the-confirmed-reboot"
FAULT_BOOT = "boot-at-the-xid"


def _reboot_record(
    index: int,
    *,
    status: WorkflowStepStatus = WorkflowStepStatus.SUCCEEDED,
    old_boot: str = RECEIPT_BOOT,
    new_boot: str = REBOOTED_BOOT,
    confirmed: bool = True,
    **details: Any,
):
    return workflow_step_execution(
        index,
        OP.RESTART_NODE,
        status,
        phase="official",
        adapter_operation_id="remote/remote-reboot",
        details={
            "action": "REBOOT",
            "submitted_nodes": ["i-0b139"],
            "externally_confirmed": confirmed,
            "agent_baselines": {NODE: {"boot_id": old_boot}},
            "agent_observations": [{"node_id": NODE, "boot_id": new_boot}],
            **details,
        },
        started_at=STOP_AT + timedelta(minutes=5),
        updated_at=STOP_AT + timedelta(minutes=15),
    )


def _parked(
    *,
    reboot=None,
    receipt: dict[str, Any] | None = None,
    quiesce_started: datetime | None = None,
    node_ids: list[str] | None = None,
):
    """MARK, STOP, (RESTART_NODE), QUIESCE parked unknown, the incident."""

    nodes = node_ids or [NODE, SIBLING]
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=nodes),
        workflow_step(
            OP.STOP_WORKLOADS, node_ids=nodes, workload_ids=["training/job/a"]
        ),
        workflow_step(OP.RESTART_NODE, node_ids=[NODE]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[NODE], gpu_uuids=["GPU-1"]),
    ]
    stop_receipt = receipt or contained_stop_receipt(
        cluster_id="hp-cluster",
        workflow_id=WORKFLOW,
        incident_id=INCIDENT,
        fencing_token=2,
        nodes=[
            {"name": NODE, "uid": f"{NODE}-uid", "boot_id": RECEIPT_BOOT},
            {"name": SIBLING, "uid": f"{SIBLING}-uid", "boot_id": "sibling-boot"},
        ],
        completed_at=STOP_AT.isoformat(),
    )
    started = quiesce_started or STOP_AT + timedelta(minutes=30)
    executions = [
        workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
        workflow_step_execution(
            1,
            OP.STOP_WORKLOADS,
            phase="official",
            details={STOP_OWNERSHIP_RECEIPT_KEY: stop_receipt},
            updated_at=STOP_AT,
        ),
    ]
    completed = [0, 1]
    if reboot is not None:
        executions.append(reboot)
        if reboot.status is WorkflowStepStatus.SUCCEEDED:
            completed.append(2)
    quiesce = workflow_step_execution(
        3,
        OP.QUIESCE_GPU_SERVICES,
        WorkflowStepStatus.FAILED,
        phase="official",
        adapter_operation_id="remote/remote-quiesce",
        details={
            "outcome_unknown": True,
            "manual_confirmation_required": True,
            "remote_command_id": "remote-quiesce",
        },
        started_at=started,
        updated_at=started + timedelta(minutes=10),
    )
    executions.append(quiesce)
    workflow = workflow_request(
        WORKFLOW,
        INCIDENT,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=2,
        execution_epoch=3,
        official_steps=steps,
        completed_step_indexes=completed,
        step_executions=executions,
        created_at=STOP_AT - timedelta(minutes=5),
        updated_at=started + timedelta(minutes=10),
    )
    incident = fault_incident(
        INCIDENT,
        "event-b139-baseline",
        cluster_id="hp-cluster",
        node_ids=nodes,
        gpu_uuids=["GPU-1"],
        state=IncidentState.QUARANTINED,
        workflow_request_id=WORKFLOW,
        fencing_token=2,
        source_boot_id=FAULT_BOOT,
    )
    return workflow, incident, quiesce


def _confirm(workflow, incident, boot: str):
    node = KubernetesNodeEvidence.from_mapping(
        {
            "node_id": NODE,
            "uid": f"{NODE}-uid",
            "boot_id": boot,
            "ready": True,
            "ready_since": (NOW - timedelta(minutes=10)).isoformat(),
        }
    )
    agent = AgentEvidence.from_mapping(
        {
            "node_id": NODE,
            "boot_id": boot,
            "agent_incarnation_id": f"incarnation-{boot}",
            "lifecycle_state": "ACTIVE",
            "generation": 9,
            "last_seen_at": (NOW - timedelta(seconds=10)).isoformat(),
            "lease_expires_at": (NOW + timedelta(seconds=60)).isoformat(),
        }
    )
    return confirm_node_action(
        workflow=workflow,
        incident=incident,
        node_id=NODE,
        node=node,
        agent=agent,
        remote_commands=[],
        actor="ops",
        reference="CHG-B139",
        now=NOW,
    )


def test_the_receipt_is_the_baseline_of_an_action_after_the_stop() -> None:
    workflow, incident, quiesce = _parked()

    assert stop_receipt_boot(workflow, incident, NODE, quiesce) == (
        RECEIPT_BOOT,
        STOP_RECEIPT_SOURCE,
    )


def test_a_confirmed_reboot_before_the_action_carries_the_receipt_forward() -> None:
    """The QUIESCE ran on the boot the reboot rung brought up, not on the one the
    STOP saw: a reboot after the QUIESCE is proven only against that boot."""

    workflow, incident, quiesce = _parked(reboot=_reboot_record(2))

    assert stop_receipt_boot(workflow, incident, NODE, quiesce) == (
        REBOOTED_BOOT,
        STOP_RECEIPT_CHAIN_SOURCE,
    )
    still = _confirm(workflow, incident, REBOOTED_BOOT)
    assert still.confirmations == ()
    assert any(f"still runs boot {REBOOTED_BOOT}" in item for item in still.refusals), (
        still.refusals
    )
    rebooted = _confirm(workflow, incident, "boot-after-the-reimage")
    assert rebooted.refusals == (), rebooted.refusals
    (confirmation,) = rebooted.confirmations
    assert confirmation["settled_by"] == "reboot_since_action"
    assert confirmation["previous_boot_id"] == REBOOTED_BOOT
    assert confirmation["previous_boot_id_source"] == STOP_RECEIPT_CHAIN_SOURCE


def test_a_reboot_the_chain_cannot_follow_leaves_no_baseline() -> None:
    """A reboot of the node before the action whose outcome is unknown: the boot
    the QUIESCE ran on is unknown, and the incident's older boot is not used."""

    unknown = _reboot_record(
        2,
        status=WorkflowStepStatus.FAILED,
        confirmed=False,
        outcome_unknown=True,
        requires_external_confirmation=True,
    )
    workflow, incident, quiesce = _parked(reboot=unknown)

    assert stop_receipt_boot(workflow, incident, NODE, quiesce) is UNPROVABLE
    verdict = _confirm(workflow, incident, "boot-after-the-reimage")
    assert verdict.confirmations == ()
    assert any("STOP receipt" in item for item in verdict.refusals), verdict.refusals


def test_a_link_that_does_not_chain_leaves_no_baseline() -> None:
    broken = _reboot_record(2, old_boot="a-boot-the-receipt-never-saw")
    workflow, incident, quiesce = _parked(reboot=broken)

    assert stop_receipt_boot(workflow, incident, NODE, quiesce) is UNPROVABLE


def test_a_reboot_that_never_left_the_executor_is_no_link() -> None:
    refused = workflow_step_execution(
        2,
        OP.RESTART_NODE,
        WorkflowStepStatus.FAILED,
        phase="official",
        details={"node_action_not_started": True, "safety_rejection": True},
        started_at=STOP_AT + timedelta(minutes=5),
        updated_at=STOP_AT + timedelta(minutes=6),
    )
    workflow, incident, quiesce = _parked(reboot=refused)

    assert stop_receipt_boot(workflow, incident, NODE, quiesce) == (
        RECEIPT_BOOT,
        STOP_RECEIPT_SOURCE,
    )


def test_receipts_that_do_not_describe_the_action_fall_back() -> None:
    """Another workflow's receipt, an uncontained one, a node it does not name
    and a STOP that completed after the action are no baseline: the caller's
    other sources apply as before (here the single-node incident's boot)."""

    foreign = contained_stop_receipt(
        workflow_id="workflow-of-another-generation",
        incident_id=INCIDENT,
        fencing_token=2,
        nodes=[{"name": NODE, "uid": f"{NODE}-uid", "boot_id": RECEIPT_BOOT}],
        completed_at=STOP_AT.isoformat(),
    )
    uncontained = {
        **contained_stop_receipt(
            workflow_id=WORKFLOW,
            incident_id=INCIDENT,
            fencing_token=2,
            nodes=[{"name": NODE, "uid": f"{NODE}-uid", "boot_id": RECEIPT_BOOT}],
            completed_at=STOP_AT.isoformat(),
        ),
        "contained": False,
    }
    other_node = contained_stop_receipt(
        workflow_id=WORKFLOW,
        incident_id=INCIDENT,
        fencing_token=2,
        nodes=[{"name": SIBLING, "uid": f"{SIBLING}-uid", "boot_id": "sibling-boot"}],
        completed_at=STOP_AT.isoformat(),
    )
    for receipt in (foreign, uncontained, other_node):
        workflow, incident, quiesce = _parked(receipt=receipt, node_ids=[NODE])
        assert stop_receipt_boot(workflow, incident, NODE, quiesce) is None, receipt
    workflow, incident, quiesce = _parked(
        quiesce_started=STOP_AT - timedelta(minutes=1), node_ids=[NODE]
    )
    assert stop_receipt_boot(workflow, incident, NODE, quiesce) is None
    verdict = _confirm(workflow, incident, "boot-after-the-reimage")
    (confirmation,) = verdict.confirmations
    assert confirmation["previous_boot_id"] == FAULT_BOOT
    assert confirmation["previous_boot_id_source"] == "incident.source_boot_id"
