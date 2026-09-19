"""The release path of a parked record whose node action an operator confirmed.

GF-REGIONAL-DESTR-014 (unknown-reboot) leaves: incident QUARANTINED over two
nodes, workflow BLOCKED / NEEDS_OPERATOR, not plan-driven, RESTART_NODE on one
node FAILED with ``outcome_unknown`` (the node rebooted while its agent was
down), the other node restored by the workflow itself. Every lever refused
and the parked record kept both nodes out of fault handling. This module
walks both exits end to end on the store:

* the product exit -- confirm the node action, build the validated restore
  for the node still isolated, let it succeed, reconcile the parked record
  as superseded by that verified successor;
* the hand-released exit -- the operator already lifted taint and cordon by
  hand; after the confirmation the incident closes on node evidence and the
  close settles the parked record in the same breath.

And it pins that an *unconfirmed* record still refuses both, so confirmation
is the only thing that opens the door.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.compile_blocked import (
    close_compile_blocked_workflows,
    settled_incident_blocked_reasons,
)
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
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.incident_closure import (
    IncidentClosureService,
    IncidentNotClosable,
    NodeIsolationEvidence,
)
from gpu_fault.orchestration.validated_restore import (
    build_validated_restore_workflow,
    is_validated_restore_workflow,
)
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=3)
CLUSTER = "gpu-a"
NODE = "hyperpod-i-0246"
SIBLING = "hyperpod-i-096b"
INCIDENT = "inc-kernel-log-kmsg-xid-79"
WORKFLOW = "workflow-2a1858b1"
OLD_BOOT = "boot-old"
NEW_BOOT = "boot-new"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
RESTART = WorkflowOperation.RESTART_NODE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
RESTORE = WorkflowOperation.RESTORE_SCHEDULING


def _parked(store):
    """The DESTR-014 leftovers, as the store holds them."""

    workflow = workflow_request(
        WORKFLOW,
        INCIDENT,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=2,
        execution_epoch=5,
        dag_enabled=True,
        official_steps=[
            workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=[NODE, SIBLING]),
            workflow_step(MARK, node_ids=[NODE]),
            workflow_step(RESTART, node_ids=[NODE]),
            workflow_step(MARK, node_ids=[SIBLING]),
            workflow_step(RESTART, node_ids=[SIBLING]),
            workflow_step(RESTORE, node_ids=[SIBLING]),
        ],
        completed_step_indexes=[0, 1, 3, 4, 5],
        completed_operations=[WorkflowOperation.STOP_WORKLOADS, MARK, RESTART, RESTORE],
        step_executions=[
            workflow_step_execution(
                2,
                RESTART,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/cmd-0246",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "remote_command_id": "cmd-0246",
                    "agent_baselines": {NODE: {"boot_id": OLD_BOOT}},
                },
                started_at=STARTED,
                updated_at=STARTED + timedelta(minutes=10),
            ),
            workflow_step_execution(
                4, RESTART, WorkflowStepStatus.SUCCEEDED, phase="official"
            ),
            workflow_step_execution(
                5, RESTORE, WorkflowStepStatus.SUCCEEDED, phase="official"
            ),
        ],
        blocked_reasons=["node branch escalation exhausted"],
        execution_deadline=NOW - timedelta(hours=2),
        lifetime_deadline_at=NOW - timedelta(hours=1),
        created_at=STARTED,
        updated_at=NOW - timedelta(hours=1),
    )
    incident = fault_incident(
        INCIDENT,
        "event-79",
        cluster_id=CLUSTER,
        node_ids=[NODE, SIBLING],
        gpu_uuids=["GPU-0246-0", "GPU-096B-0"],
        state=IncidentState.QUARANTINED,
        workflow_request_id=WORKFLOW,
        fencing_token=2,
        created_at=STARTED,
        updated_at=NOW - timedelta(hours=1),
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def _node(node_id: str = NODE) -> KubernetesNodeEvidence:
    return KubernetesNodeEvidence.from_mapping(
        {
            "node_id": node_id,
            "uid": f"uid-{node_id}",
            "boot_id": NEW_BOOT,
            "ready": True,
            "ready_since": (NOW - timedelta(minutes=40)).isoformat(),
        }
    )


def _agent(node_id: str = NODE) -> AgentEvidence:
    return AgentEvidence.from_mapping(
        {
            "node_id": node_id,
            "boot_id": NEW_BOOT,
            "agent_incarnation_id": "inc-new",
            "lifecycle_state": "ACTIVE",
            "generation": 12,
            "last_seen_at": (NOW - timedelta(seconds=10)).isoformat(),
            "lease_expires_at": (NOW + timedelta(seconds=50)).isoformat(),
        }
    )


def _confirm(store, workflow):
    verdict = confirm_node_action(
        workflow=workflow,
        incident=store.get_incident(workflow.incident_id),
        node_id=NODE,
        node=_node(),
        agent=_agent(),
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-2026-0919-03",
        now=NOW,
    )
    assert verdict.refusals == (), verdict.refusals
    confirmed = apply_node_action_confirmation(workflow, verdict, now=NOW)
    store.save_workflow(confirmed, expected=workflow)
    return store.get_workflow(workflow.request_id)


def _evidence(node_id: str) -> NodeIsolationEvidence:
    return NodeIsolationEvidence(node_id=node_id)


def test_the_parked_record_holds_both_nodes_until_it_is_dealt_with() -> None:
    store = build_store()
    incident, workflow = _parked(store)

    assert has_unresolved_node_action(workflow), "the fixture is the live shape"
    held = store.list_active_workflow_incidents(CLUSTER, node_ids={NODE, SIBLING})
    assert [item.request_id for _, item in held] == [WORKFLOW], (
        "the BLOCKED / NEEDS_OPERATOR record occupies both nodes"
    )
    service = IncidentClosureService(store)
    with pytest.raises(IncidentNotClosable, match="open workflow"):
        service.close_incident(
            incident.incident_id,
            reason="hand-released",
            operator=ACTOR,
            evidence=[_evidence(NODE), _evidence(SIBLING)],
        )
    assert settled_incident_blocked_reasons(
        workflow, copy_model(incident, state=IncidentState.RECOVERED), []
    ), "the sweep must not end an unconfirmed record even for a RECOVERED incident"


def test_product_exit_confirm_then_validated_restore_then_reconcile() -> None:
    store = build_store()
    incident, workflow = _parked(store)

    confirmed = _confirm(store, workflow)
    assert not has_unresolved_node_action(confirmed), "confirmation resolved the step"
    assert confirmed.status is WorkflowStatus.BLOCKED, "the record is still parked"
    assert confirmed.events[-1].code == WorkflowEventCode.NODE_ACTION_CONFIRMED.value

    # The validated restore for the node still isolated by this incident.
    updated, restore = build_validated_restore_workflow(
        incident,
        operator=ACTOR,
        reference="CHG-2026-0919-03",
        now=NOW,
        node_ids=[NODE],
        node_gpu_uuids={NODE: ["GPU-0246-0"], SIBLING: ["GPU-096B-0"]},
    )
    assert is_validated_restore_workflow(restore.request_id), restore.request_id
    assert all(step.node_ids == [NODE] for step in restore.official_steps), (
        "only the node still isolated is restored"
    )
    assert all(step.gpu_uuids == [] for step in restore.official_steps), (
        "a restore of a subset of the incident's nodes validates node-wide: an "
        "incident GPU may sit on the node already restored, and a scope that can "
        "only time out looks like a sick node (the running builder's rule)"
    )
    assert updated.state is IncidentState.ACTION_PENDING
    store.save_incident_and_workflow(updated, restore)

    # The executor runs it to completion and writes the incident RECOVERED.
    succeeded = copy_model(
        restore,
        status=WorkflowStatus.SUCCEEDED,
        completed_step_indexes=list(range(len(restore.official_steps))),
        completed_operations=[step.operation for step in restore.official_steps],
        updated_at=NOW + timedelta(minutes=3),
    )
    store.save_workflow(succeeded, expected=restore)
    recovered = copy_model(
        updated, state=IncidentState.RECOVERED, updated_at=NOW + timedelta(minutes=3)
    )
    store.save_incident(recovered, expected=updated)

    plan = build_workflow_reconcile_plan(
        store, incident_ids=[INCIDENT], now=NOW + timedelta(minutes=5)
    )
    item = next(entry for entry in plan["items"] if entry["request_id"] == WORKFLOW)
    assert item["eligible"] is True, item["reasons"]
    assert item["terminalization"] == "verified-restore"
    assert item["successor_workflow_id"] == restore.request_id

    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=[WORKFLOW],
        expected_plan_sha256=build_workflow_reconcile_plan(
            store, [WORKFLOW], now=NOW + timedelta(minutes=5)
        )["plan_sha256"],
        reference="CHG-2026-0919-03",
        now=NOW + timedelta(minutes=5),
        actor=ACTOR,
    )
    assert result["applied_workflow_ids"] == [WORKFLOW], result
    closed = store.get_workflow(WORKFLOW)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id == restore.request_id
    assert store.list_active_workflow_incidents(CLUSTER, node_ids={NODE, SIBLING}) == []


def test_hand_released_exit_close_on_evidence_settles_the_confirmed_record() -> None:
    store = build_store()
    incident, workflow = _parked(store)
    _confirm(store, workflow)
    service = IncidentClosureService(store)

    preview = service.preview(
        incident.incident_id, evidence=[_evidence(NODE), _evidence(SIBLING)]
    )
    assert preview["closable"] is True, preview
    closed, changed = service.close_incident(
        incident.incident_id,
        reason="node rebooted on its own; taint and cordon lifted by hand",
        operator=ACTOR,
        reference="CHG-2026-0919-03",
        evidence=[_evidence(NODE), _evidence(SIBLING)],
    )

    assert changed is True and closed.state is IncidentState.RECOVERED
    settled = store.get_workflow(WORKFLOW)
    assert settled.status is WorkflowStatus.SUPERSEDED, settled.status
    assert "closed BLOCKED workflow of a RECOVERED incident" in (
        settled.preemption_reason or ""
    )
    assert store.list_active_workflow_incidents(CLUSTER, node_ids={NODE, SIBLING}) == []


def test_a_confirmed_record_still_needs_clear_isolation_evidence_to_close() -> None:
    store = build_store()
    incident, workflow = _parked(store)
    _confirm(store, workflow)
    service = IncidentClosureService(store)
    still_tainted = NodeIsolationEvidence(
        node_id=NODE,
        unschedulable=True,
        quarantine_taint_value=incident.incident_id,
        isolation_annotations={"gpu-fault.io/incident-id": incident.incident_id},
    )

    with pytest.raises(IncidentNotClosable, match="still cordoned"):
        service.close_incident(
            incident.incident_id,
            reason="too early",
            operator=ACTOR,
            evidence=[still_tainted, _evidence(SIBLING)],
        )
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED


def test_without_evidence_a_confirmed_record_is_not_a_reason_to_close() -> None:
    """Confirmation answers the physical question only; the QUARANTINED close
    still needs its node evidence, exactly as before."""

    store = build_store()
    incident, workflow = _parked(store)
    _confirm(store, workflow)

    preview = IncidentClosureService(store).preview(incident.incident_id)

    assert preview["closable"] is False
    assert preview["evidence_required"] is True


def test_the_sweep_ends_a_confirmed_record_once_its_incident_is_recovered() -> None:
    store = build_store()
    incident, workflow = _parked(store)
    confirmed = _confirm(store, workflow)
    assert close_compile_blocked_workflows(store, now=NOW) == [], (
        "a confirmed record of a QUARANTINED incident is still somebody's business"
    )
    store.save_incident(
        copy_model(incident, state=IncidentState.RECOVERED), expected=incident
    )

    closed = close_compile_blocked_workflows(store, now=NOW + timedelta(minutes=1))

    assert closed == [WORKFLOW]
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.SUPERSEDED
    assert confirmed.status is WorkflowStatus.BLOCKED, "the read copy is untouched"


def test_an_unconfirmed_record_of_a_recovered_incident_is_left_for_the_operator() -> (
    None
):
    store = build_store()
    incident, _workflow = _parked(store)
    store.save_incident(
        copy_model(incident, state=IncidentState.RECOVERED), expected=incident
    )

    assert close_compile_blocked_workflows(store, now=NOW) == []
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED
