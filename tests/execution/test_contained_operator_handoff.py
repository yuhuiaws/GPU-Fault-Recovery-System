"""A containment prefix that settles hands its node to an operator, in words.

The node-health family compiles a plan refused only for a missing executable
owner as SAFETY_PENDING on its containment prefix
(``tests/orchestration/test_containment_before_compile_block.py``). This is the
executor's half: the prefix runs, the record settles BLOCKED/SAFETY_SETTLED,
the incident ends QUARANTINED with a reason that names the contained node and
the capability an operator owns -- written in the terminal transaction -- and
one ``contained-operator-handoff-zh-v1`` notice goes out. The unowned step is
never dispatched, the compile-time sweep keeps its hands off a record that ran
steps, and the restore hold does not read the settled record as owning
recovery, so ``submit-remediation --disposition restore`` is the exit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.compile_blocked import (
    SETTLED_CLOSE_MARKER,
    close_compile_blocked_workflows,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    workflow_is_open,
)
from gpu_fault.notifications.contained_operator_handoff import (
    CONTAINED_OPERATOR_HANDOFF_CATEGORY,
    CONTAINED_OPERATOR_HANDOFF_TEMPLATE_VERSION,
)
from gpu_fault.recovery_safety import workflow_recovery_error
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
QUARANTINE = WorkflowOperation.QUARANTINE
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
FIELD = WorkflowOperation.RUN_FIELD_DIAGNOSTIC
VALIDATE = WorkflowOperation.VALIDATE_GPU
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
INCIDENT = "inc-gpu-dcgm-node-a-GPU-a-row_remap_failure"
WORKFLOW = "workflow-row-remap-observe"
MEMORY_REASON = "no executable owner for memoryDiagnostics"
NOW = datetime(2026, 10, 26, 6, 25, tzinfo=timezone.utc)


def _contained_plan(store, *, blocked_reasons=(MEMORY_REASON,)):
    """The live 019 ``remap_failure`` plan as the node-health family now
    compiles it under ``memoryDiagnostics: OBSERVE``."""

    incident = fault_incident(
        INCIDENT,
        "gpu-dcgm-node-a-GPU-a-row_remap_failure",
        event_type="NODE_HEALTH",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        policy_source="NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
        official_action="RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        state=IncidentState.SAFETY_PENDING,
        workflow_request_id=WORKFLOW,
        fencing_token=3,
        reasons=[
            "row remap failure; field diagnosis required before any RMA conclusion"
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    containment = [
        workflow_step(FREEZE),
        workflow_step(MARK),
        workflow_step(
            QUARANTINE, parameters={"terminal_quarantine_node_ids": ["node-a"]}
        ),
        workflow_step(BUNDLE),
    ]
    workflow = workflow_request(
        WORKFLOW,
        INCIDENT,
        status=WorkflowStatus.SAFETY_PENDING,
        runtime_profile_version="observe-v1",
        official_action="RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        safety_steps=containment,
        safety_only=True,
        official_steps=[*containment, workflow_step(VALIDATE), workflow_step(SUPPORT)],
        blocked_reasons=list(blocked_reasons),
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def _run(store, sent):
    adapter = FakeAdapter(
        {
            operation: WorkflowStepOutcome.succeeded(operation_id=operation.value)
            for operation in (
                FREEZE,
                MARK,
                QUARANTINE,
                BUNDLE,
                FIELD,
                VALIDATE,
                SUPPORT,
            )
        }
    )
    executor = active_workflow_executor(
        store,
        [adapter],
        [FREEZE, MARK, QUARANTINE, BUNDLE, FIELD, VALIDATE, SUPPORT],
        notification_sender=sent.append,
    )
    return adapter, execute_workflow(executor, WORKFLOW)


def test_the_prefix_runs_and_the_record_settles_as_a_contained_hand_off():
    store = build_store()
    sent: list[str] = []
    incident, _ = _contained_plan(store)

    adapter, result = _run(store, sent)

    assert result.status is WorkflowStatus.BLOCKED, result
    settled = store.get_workflow(WORKFLOW)
    assert settled.status is WorkflowStatus.BLOCKED
    assert settled.blocked_kind is BlockedKind.SAFETY_SETTLED
    assert settled.blocked_reasons == [MEMORY_REASON], "the refusal survives the run"
    assert [call.rsplit("/", 1)[1] for call in adapter.calls] == [
        FREEZE.value,
        MARK.value,
        QUARANTINE.value,
        BUNDLE.value,
    ], "only the containment prefix is dispatched; the unowned step never is"
    assert settled.completed_operations == [FREEZE, MARK, QUARANTINE, BUNDLE]
    assert settled.execution_owner_id is None
    assert not workflow_is_open(settled.status, settled.blocked_kind), (
        "a settled safety plan releases its node to successors (F-A4)"
    )

    closed = store.get_incident(incident.incident_id)
    assert closed.state is IncidentState.QUARANTINED, closed.state
    (handoff,) = [reason for reason in closed.reasons if "operator action" in reason]
    assert "node node-a contained (cordoned and quarantined)" in handoff
    assert "FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> QUARANTINE" in handoff
    assert "no executable owner for memoryDiagnostics" in handoff
    assert "RUN_FIELD_DIAGNOSTIC" in handoff and "VALIDATE_GPU" in handoff
    assert "--disposition restore" in handoff
    (terminal,) = [e for e in settled.events if e.kind is WorkflowEventKind.TERMINAL]
    assert terminal.details["incident_state"] == IncidentState.QUARANTINED.value


def test_one_hand_off_notice_names_the_containment_and_the_missing_owner():
    store = build_store()
    sent: list[str] = []
    incident, _ = _contained_plan(store)

    _run(store, sent)

    (notification,) = store.list_notifications()
    assert notification.incident_id == incident.incident_id
    assert notification.category == CONTAINED_OPERATOR_HANDOFF_CATEGORY
    assert notification.deduplication_key == (
        f"cluster-a/{INCIDENT}/contained-operator-handoff/{WORKFLOW}"
    )
    assert notification.subject.startswith(
        "[GPU FAULT][contained, operator action required] cluster-a: node-a"
    ), notification.subject
    body = notification.body_text
    assert (
        "FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> QUARANTINE -> COLLECT_DIAGNOSTIC_BUNDLE"
        in body
    )
    assert "- memoryDiagnostics" in body
    assert "VALIDATE_GPU -> ESCALATE_SUPPORT -> RUN_FIELD_DIAGNOSTIC" in body
    assert MEMORY_REASON in body
    assert "workflow BLOCKED/SAFETY_SETTLED, incident QUARANTINED" in body
    assert "--disposition restore" in body
    assert "--close-quarantined" in body
    assert "Runtime Profile: observe-v1" in body
    assert body.rstrip().endswith(
        f"Template: {CONTAINED_OPERATOR_HANDOFF_TEMPLATE_VERSION}"
    ), body[-120:]
    assert notification.support_case_draft == ""
    assert sent == [notification.notification_id]


def test_a_safety_plan_settled_for_another_reason_keeps_its_quiet_hand_off():
    """The fault family's SITE_SAFETY / missing-evidence settle is unchanged:
    no missing owner, so no notice and no extra reason."""

    store = build_store()
    sent: list[str] = []
    incident, _ = _contained_plan(
        store, blocked_reasons=("fatal SXID requires a trusted fabric_partition",)
    )

    _run(store, sent)

    settled = store.get_workflow(WORKFLOW)
    assert settled.blocked_kind is BlockedKind.SAFETY_SETTLED
    closed = store.get_incident(incident.incident_id)
    assert closed.state is IncidentState.QUARANTINED
    assert not any("operator action required" in reason for reason in closed.reasons), (
        closed.reasons
    )
    assert store.list_notifications() == []
    assert sent == []


def test_the_compile_time_sweep_leaves_the_settled_record_to_its_operator():
    """It ran steps and changed a node: not a "never changed a node" no-op.
    The settled-incident shape ends it once the operator's exit recovered the
    incident, and only then."""

    store = build_store()
    incident, _ = _contained_plan(store)
    _run(store, [])

    assert close_compile_blocked_workflows(store, now=NOW) == []
    settled = store.get_workflow(WORKFLOW)
    assert settled.status is WorkflowStatus.BLOCKED
    assert settled.blocked_kind is BlockedKind.SAFETY_SETTLED
    assert not any(
        event.kind is WorkflowEventKind.OPERATOR_RECONCILED for event in settled.events
    ), "a refusal writes no event"

    quarantined = store.get_incident(incident.incident_id)
    store.save_incident(
        copy_model(quarantined, state=IncidentState.RECOVERED), expected=quarantined
    )
    assert close_compile_blocked_workflows(store, now=NOW) == [WORKFLOW]
    ended = store.get_workflow(WORKFLOW)
    assert ended.status is WorkflowStatus.SUPERSEDED
    assert SETTLED_CLOSE_MARKER in (ended.preemption_reason or "")
    assert MEMORY_REASON in (ended.preemption_reason or "")


def test_the_restore_hold_does_not_read_a_settled_prefix_as_owning_recovery():
    """``recovery_safety`` refused the 019 fixture's validated restore with
    "workflow still owns recovery; operator hold retained" when the record was
    NEEDS_OPERATOR; a settled safety plan is the shape the restore exits."""

    store = build_store()
    _contained_plan(store)
    _run(store, [])

    settled = store.get_workflow(WORKFLOW)
    assert workflow_recovery_error(settled.model_dump(mode="json")) is None
