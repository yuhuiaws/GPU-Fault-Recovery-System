"""The operator exit of a node a containment prefix holds.

A plan refused only for a missing executable owner now runs its containment
first and settles BLOCKED/SAFETY_SETTLED with the incident QUARANTINED
(``tests/orchestration/test_containment_before_compile_block.py``). The
coordinator's question for that shape: which command releases the node, and
does anything stand in its way? This module walks the two documented exits on
the store, the way ``test_confirmed_node_action_release_path.py`` does for the
parked node action:

* the node is cleared (the GPU passes the manual field diagnostic, or the
  replay stops): ``submit-remediation --disposition restore`` -- the validated
  restore under the same incident -- is accepted because a settled safety plan
  is not an open workflow; its success recovers the incident and the
  dispatcher's sweep ends the settled record as superseded by it;
* the node is replaced: the incident closes on node evidence (the departure
  proof or a clean node), and the close settles the record in the same call.

And it pins that the record does not occupy the node, that the compile-time
no-op sweep never touches it, and that an unsettled containment prefix (still
running) refuses both exits.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.adapters.common import ANNOTATION_INCIDENT, quarantine_taint_value
from gpu_fault.compile_blocked import (
    close_compile_blocked_workflows,
    never_dispatched_reasons,
    settled_incident_blocked_reasons,
)
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    workflow_is_open,
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
from gpu_fault.recovery_safety import workflow_recovery_error
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 10, 26, 7, 0, tzinfo=timezone.utc)
CLUSTER = "gpu-eks-cluster"
NODE = "ip-10-0-43-5"
INCIDENT = "inc-gpu-dcgm-ip-10-0-43-5-GPU-a-row_remap_failure"
WORKFLOW = "workflow-e75cb4b421836a80d2edac7c"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
QUARANTINE = WorkflowOperation.QUARANTINE
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
MEMORY_REASON = "no executable owner for memoryDiagnostics"


def _settled(store, *, incident_state=IncidentState.QUARANTINED):
    """The record the executor leaves after the containment prefix ran."""

    containment = [
        workflow_step(FREEZE, node_ids=[NODE]),
        workflow_step(MARK, node_ids=[NODE]),
        workflow_step(
            QUARANTINE,
            node_ids=[NODE],
            parameters={"terminal_quarantine_node_ids": [NODE]},
        ),
        workflow_step(BUNDLE, node_ids=[NODE]),
    ]
    workflow = workflow_request(
        WORKFLOW,
        INCIDENT,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.SAFETY_SETTLED,
        fencing_token=1,
        execution_epoch=1,
        runtime_profile_version="regional-eks-observe",
        official_action="RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        safety_steps=containment,
        safety_only=True,
        official_steps=[
            *containment,
            workflow_step(WorkflowOperation.VALIDATE_GPU, node_ids=[NODE]),
            workflow_step(WorkflowOperation.ESCALATE_SUPPORT, node_ids=[NODE]),
        ],
        blocked_reasons=[MEMORY_REASON],
        completed_step_indexes=[0, 1, 2, 3],
        completed_operations=[FREEZE, MARK, QUARANTINE, BUNDLE],
        step_executions=[
            workflow_step_execution(index, operation, phase="safety")
            for index, operation in enumerate((FREEZE, MARK, QUARANTINE, BUNDLE))
        ],
        created_at=NOW - timedelta(minutes=5),
        updated_at=NOW,
    )
    incident = fault_incident(
        INCIDENT,
        "gpu-dcgm-ip-10-0-43-5-GPU-a-row_remap_failure",
        event_type="NODE_HEALTH",
        cluster_id=CLUSTER,
        node_ids=[NODE],
        gpu_uuids=["GPU-a"],
        policy_source="NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
        official_action="RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        state=incident_state,
        workflow_request_id=WORKFLOW,
        fencing_token=1,
        created_at=NOW - timedelta(minutes=5),
        updated_at=NOW,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def _isolated(node_id: str) -> NodeIsolationEvidence:
    return NodeIsolationEvidence(
        node_id=node_id,
        unschedulable=True,
        quarantine_taint_value=quarantine_taint_value(INCIDENT),
        isolation_annotations={ANNOTATION_INCIDENT: INCIDENT},
    )


def _clean(node_id: str) -> NodeIsolationEvidence:
    return NodeIsolationEvidence(node_id=node_id)


def test_the_settled_record_holds_its_node_only_through_the_incident() -> None:
    store = build_store()
    incident, workflow = _settled(store)

    assert not workflow_is_open(workflow.status, workflow.blocked_kind), (
        "BLOCKED/SAFETY_SETTLED releases the node (F-A4)"
    )
    assert store.list_active_workflow_incidents(CLUSTER, node_ids={NODE}) == [], (
        "a settled safety plan is not a node incumbent; the QUARANTINED incident "
        "and the taint on the node are what hold it"
    )
    assert never_dispatched_reasons(workflow), "it ran steps: not a compile-time no-op"
    assert close_compile_blocked_workflows(store, now=NOW) == []
    assert workflow_recovery_error(workflow.model_dump(mode="json")) is None, (
        "the restore hold reads a settled prefix as not owning recovery"
    )
    assert settled_incident_blocked_reasons(workflow, incident, []), (
        "while the incident is QUARANTINED nothing ends the record"
    )


def test_exit_one_validated_restore_recovers_the_incident_and_the_sweep_ends_the_record() -> (
    None
):
    store = build_store()
    incident, _ = _settled(store)

    updated, restore = build_validated_restore_workflow(
        incident, operator=ACTOR, reference="CHG-2026-1026-19", now=NOW
    )
    assert is_validated_restore_workflow(restore.request_id), restore.request_id
    assert updated.state is IncidentState.ACTION_PENDING
    assert [step.operation for step in restore.official_steps][-1] is (
        WorkflowOperation.RESTORE_SCHEDULING
    )
    store.save_incident_and_workflow(updated, restore)

    # The executor runs the restore to completion and writes RECOVERED.
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

    assert close_compile_blocked_workflows(store, now=NOW + timedelta(minutes=4)) == [
        WORKFLOW
    ]
    ended = store.get_workflow(WORKFLOW)
    assert ended.status is WorkflowStatus.SUPERSEDED
    assert ended.preempted_by_workflow_id == restore.request_id, (
        "superseded by the verified restore, the third sweep shape"
    )
    (event,) = [
        e for e in ended.events if e.kind is WorkflowEventKind.OPERATOR_RECONCILED
    ]
    assert event.actor == "dispatcher"
    assert store.list_active_workflow_incidents(CLUSTER, node_ids={NODE}) == []


def test_exit_two_a_replaced_node_closes_on_evidence_and_settles_the_record() -> None:
    store = build_store()
    incident, _ = _settled(store)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match="only an ESCALATED"):
        service.close_incident(incident.incident_id, reason="rma", operator=ACTOR)
    refused = service.preview(incident.incident_id, evidence=[_isolated(NODE)])
    assert refused["closable"] is False, refused
    assert "does not clear it" in refused["refusal"]

    closed, _changed = service.close_incident(
        incident.incident_id,
        reason="GPU replaced after the manual field diagnostic",
        operator=ACTOR,
        reference="CHG-2026-1026-19",
        evidence=[_clean(NODE)],
    )
    assert closed.state is IncidentState.RECOVERED
    ended = store.get_workflow(WORKFLOW)
    assert ended.status is WorkflowStatus.SUPERSEDED, (
        "the close settles the record it leaves behind"
    )


def test_a_prefix_still_running_refuses_both_exits() -> None:
    """SAFETY_PENDING is live work: neither the restore nor the close passes
    it, exactly as for the fault family's safety plan."""

    store = build_store()
    incident, workflow = _settled(store, incident_state=IncidentState.SAFETY_PENDING)
    running = copy_model(
        workflow,
        status=WorkflowStatus.SAFETY_PENDING,
        blocked_kind=None,
        completed_step_indexes=[0, 1],
        completed_operations=[FREEZE, MARK],
        step_executions=[
            workflow_step_execution(0, FREEZE, phase="safety"),
            workflow_step_execution(1, MARK, phase="safety"),
            workflow_step_execution(
                2, QUARANTINE, WorkflowStepStatus.WAITING, phase="safety"
            ),
        ],
    )
    store.save_workflow(running, expected=workflow)

    assert workflow_is_open(running.status, running.blocked_kind), (
        "SAFETY_PENDING is live work"
    )
    assert workflow_recovery_error(running.model_dump(mode="json")) is not None
    with pytest.raises(IncidentNotClosable):
        IncidentClosureService(store).close_incident(
            incident.incident_id,
            reason="too early",
            operator=ACTOR,
            evidence=[_clean(NODE)],
        )
