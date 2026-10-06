"""A parked GPU reset with a lost answer is confirmed by the reboot that followed.

Live (gpu-eks-cluster, XID 46, release 5377615bd785): RESET_GPU ended
``outcome_unknown`` and the record parked BLOCKED. The operator rebooted the
node from EC2 (new boot id, Agent ACTIVE) and ``confirm-node-action`` still
refused -- "no terminal Node Agent result for RESET_GPU ... and no later
validated restore of this incident" -- while ``restore`` refused on the open
record and ``workflow-reconcile`` found it ineligible: an open workflow nothing
could close, the shape that blocked ``uninstall`` in an earlier round. No GPU
reset survives a boot, so a reboot proven the way RESTART_NODE proves its own
(boot id before and after, Agent incarnation) now confirms the reset too; the
confirmation names its basis in ``settled_by``, and the ESCALATED incident then
closes RECOVERED on the operator's authority, ending the record.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.compile_blocked import close_compile_blocked_workflows
from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    apply_node_action_confirmation,
    confirm_node_action,
)
from gpu_fault.execution.node_action_uncertainty import has_unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.incident_closure import (
    IncidentClosureService,
    IncidentNotClosable,
    NodeIsolationEvidence,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=2)
CLUSTER = "gpu-eks-cluster"
NODE = "ip-10-0-41-164"
INCIDENT = "inc-3731-xid-46"
RESET_WORKFLOW = "workflow-ec95fa59"
RESTORE_ATTEMPT = "workflow-validated-restore-attempt"
OLD_BOOT = "boot-before-reset"
NEW_BOOT = "boot-after-ec2-reboot"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
OP = WorkflowOperation


def _parked(store, *, incident_state: IncidentState = IncidentState.ESCALATED):
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[NODE]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[NODE]),
        workflow_step(OP.RESET_GPU, node_ids=[NODE], gpu_uuids=["GPU-3df3"]),
        workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=[NODE]),
    ]
    workflow = workflow_request(
        RESET_WORKFLOW,
        INCIDENT,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=1,
        execution_epoch=4,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[OP.MARK_UNSCHEDULABLE, OP.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
            workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES, phase="official"),
            workflow_step_execution(
                2,
                OP.RESET_GPU,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/remote-reset",
                error="OWNERSHIP_FINAL_CLIENTS_CHANGED",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "remote_command_id": "remote-reset",
                    "reason": "OWNERSHIP_FINAL_CLIENTS_CHANGED",
                },
                started_at=STARTED,
                updated_at=STARTED + timedelta(minutes=3),
            ),
        ],
        blocked_reasons=["node action outcome unknown"],
        execution_deadline=NOW - timedelta(hours=1),
        created_at=STARTED,
        updated_at=STARTED + timedelta(minutes=5),
    )
    pointer = RESET_WORKFLOW
    if incident_state is IncidentState.ESCALATED:
        # The operator's restore attempt that FAILED (its VALIDATE_GPU timed
        # out) and took the incident pointer with it.
        store.save_workflow(
            workflow_request(
                RESTORE_ATTEMPT,
                INCIDENT,
                status=WorkflowStatus.FAILED,
                fencing_token=1,
                official_steps=[workflow_step(OP.VALIDATE_GPU, node_ids=[NODE])],
                created_at=STARTED + timedelta(hours=1),
                updated_at=NOW - timedelta(minutes=20),
            )
        )
        pointer = RESTORE_ATTEMPT
    incident = fault_incident(
        INCIDENT,
        "kernel-log-kmsg-xid-46",
        cluster_id=CLUSTER,
        node_ids=[NODE],
        gpu_uuids=["GPU-3df3"],
        state=incident_state,
        workflow_request_id=pointer,
        fencing_token=1,
        source_boot_id=OLD_BOOT,
        created_at=STARTED,
        updated_at=NOW - timedelta(minutes=20),
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def _node(boot_id: str = NEW_BOOT) -> KubernetesNodeEvidence:
    return KubernetesNodeEvidence.from_mapping(
        {
            "node_id": NODE,
            "uid": "uid-41-164",
            "boot_id": boot_id,
            "ready": True,
            "ready_since": (NOW - timedelta(minutes=15)).isoformat(),
            "unschedulable": False,
        }
    )


def _agent(boot_id: str = NEW_BOOT) -> AgentEvidence:
    return AgentEvidence.from_mapping(
        {
            "node_id": NODE,
            "boot_id": boot_id,
            "agent_incarnation_id": "incarnation-after-reboot",
            "lifecycle_state": "ACTIVE",
            "generation": 21,
            "last_seen_at": (NOW - timedelta(seconds=10)).isoformat(),
            "lease_expires_at": (NOW + timedelta(seconds=50)).isoformat(),
        }
    )


def _confirm(store, workflow, *, boot_id: str = NEW_BOOT):
    return confirm_node_action(
        workflow=workflow,
        incident=store.get_incident(workflow.incident_id),
        node_id=NODE,
        node=_node(boot_id),
        agent=_agent(boot_id),
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-2026-1006-01",
        now=NOW,
    )


def test_a_reboot_since_the_reset_confirms_it_and_says_so() -> None:
    store = build_store()
    _incident, workflow = _parked(store)

    verdict = _confirm(store, workflow)

    assert verdict.refusals == (), verdict.refusals
    (confirmation,) = verdict.confirmations
    assert confirmation["operation"] == "RESET_GPU"
    assert confirmation["settled_by"] == "reboot_since_action"
    assert confirmation["previous_boot_id"] == OLD_BOOT
    assert confirmation["previous_boot_id_source"] == "incident.source_boot_id"
    assert confirmation["observed_boot_id"] == NEW_BOOT
    assert confirmation["terminal_node_result"] is None
    confirmed = apply_node_action_confirmation(workflow, verdict, now=NOW)
    assert has_unresolved_node_action(confirmed) is False


def test_the_same_boot_still_refuses_and_names_the_missing_proof() -> None:
    store = build_store()
    _incident, workflow = _parked(store)

    verdict = _confirm(store, workflow, boot_id=OLD_BOOT)

    assert verdict.confirmations == ()
    (refusal,) = verdict.refusals
    assert "no terminal Node Agent result for RESET_GPU" in refusal
    assert "no reboot since the action is proven" in refusal
    assert f"still runs boot {OLD_BOOT}" in refusal


def test_the_confirmed_record_ends_with_the_operators_close_of_the_incident() -> None:
    store = build_store()
    incident, workflow = _parked(store)
    verdict = _confirm(store, workflow)
    assert verdict.refusals == (), verdict.refusals
    confirmed = apply_node_action_confirmation(workflow, verdict, now=NOW)
    store.save_workflow(confirmed, expected=workflow)

    closed, changed = IncidentClosureService(store).close_incident(
        incident.incident_id,
        reason="node rebooted from EC2; GPU validated healthy by the later restore",
        operator=ACTOR,
        reference="CHG-2026-1006-01",
        # The node evidence the CLI reads through the GPU kubeconfig: clean.
        evidence=[NodeIsolationEvidence(node_id=NODE)],
    )
    close_compile_blocked_workflows(store, now=NOW)

    assert changed is True and closed.state is IncidentState.RECOVERED
    settled = store.get_workflow(RESET_WORKFLOW)
    assert settled.status is WorkflowStatus.SUPERSEDED, settled.status
    assert store.list_active_workflow_incidents(CLUSTER, node_ids={NODE}) == [], (
        "no open workflow holds the node any more: the uninstall gate counts none"
    )


def test_without_clearing_evidence_the_confirmed_record_still_refuses_the_close() -> (
    None
):
    store = build_store()
    incident, workflow = _parked(store)
    verdict = _confirm(store, workflow)
    confirmed = apply_node_action_confirmation(workflow, verdict, now=NOW)
    store.save_workflow(confirmed, expected=workflow)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match="still has an open workflow"):
        service.close_incident(
            incident.incident_id, reason="no evidence", operator=ACTOR, reference="CHG"
        )
    assert store.get_workflow(RESET_WORKFLOW).status is WorkflowStatus.BLOCKED


def test_the_preview_asks_for_node_evidence_once_the_record_is_confirmed() -> None:
    """``workflow-reconcile --close-incident`` runs a first pass without
    evidence; a refusal reported with ``evidence_required`` makes the CLI read
    the nodes through the GPU kubeconfig and close on the second pass."""

    store = build_store()
    incident, workflow = _parked(store)
    service = IncidentClosureService(store)

    before = service.preview(incident.incident_id)
    assert before["closable"] is False and before["evidence_required"] is False, (
        "an unconfirmed parked record is not something evidence can settle"
    )

    verdict = _confirm(store, workflow)
    store.save_workflow(
        apply_node_action_confirmation(workflow, verdict, now=NOW), expected=workflow
    )
    asked = service.preview(incident.incident_id)
    assert asked["closable"] is False
    assert asked["evidence_required"] is True
    assert asked["open_workflow_id"] == RESET_WORKFLOW

    with_evidence = service.preview(
        incident.incident_id, evidence=[NodeIsolationEvidence(node_id=NODE)]
    )
    assert with_evidence["closable"] is True, with_evidence
    assert with_evidence["isolation_reasons"]
