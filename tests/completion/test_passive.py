from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.execution import (
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    RECOVERY_ACTION_RANK,
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    EffectiveRuntimeProfile,
    Environment,
    MarkerScope,
    NodeMarker,
    PlanStatus,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    WorkflowOperation,
    WorkflowStatus,
    recovery_action_sort_key,
)
from gpu_fault.passive import PassiveWorkflowCompiler
from gpu_fault.planner import PlanBuilder, UnsupportedPlanError
from tests._builders import active_workflow_executor, build_store, copy_model


def test_passive_plan_compiles_to_fenced_workflow(failed_event: TerminalEvent) -> None:
    store = build_store()
    store.save_profile(default_simulated_profile())
    event = copy_model(failed_event, workload_ids=["training/job/distributed-train"])
    plan = RecoveryPlan(
        incident_id="incident-passive",
        attempt_id=event.attempt_id,
        trigger="quick-triage:PASS",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a"],
                "execution_owner": "simulated-runtime",
            }
        ],
    )

    compiled = PassiveWorkflowCompiler(store).compile(plan, event)
    workflow = store.get_workflow(compiled.workflow_request_id)
    incident = store.get_incident(compiled.incident_id)

    assert workflow.status is WorkflowStatus.PENDING
    assert incident.workflow_request_id == workflow.request_id
    assert workflow.official_steps[0].operation is (WorkflowOperation.RESTART_WORKLOAD)
    assert workflow.official_steps[0].workload_ids == ["training/job/distributed-train"]
    assert workflow.official_steps[0].parameters == {
        "cluster_id": "cluster-a",
        "job_id": "train-123",
        "source_attempt_id": "train-123-a1",
        "source_gpu_count": 2,
        "restart_budget": 1,
    }


def test_gpu_reset_plan_inserts_no_client_gate(failed_event: TerminalEvent) -> None:
    store = build_store()
    plan = RecoveryPlan(
        incident_id="incident-reset",
        attempt_id=failed_event.attempt_id,
        trigger="marker:xid",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESET_GPU,
                "node_ids": ["node-a"],
                "gpu_uuids": ["GPU-a"],
                "execution_owner": "gpu-fault-node-agent",
            }
        ],
    )

    compiled = PassiveWorkflowCompiler(store).compile(plan, failed_event)
    workflow = store.get_workflow(compiled.workflow_request_id)
    incident = store.get_incident(compiled.incident_id)

    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    assert incident.official_action == "RESET_GPU"
    assert incident.effective_action is RecoveryAction.RESET_GPU


@pytest.mark.parametrize(
    "action", [RecoveryAction.REBOOT_NODE, RecoveryAction.REPLACE_NODE]
)
def test_node_lifecycle_plan_validates_gpu_host_and_fabric(
    failed_event: TerminalEvent, action: RecoveryAction
) -> None:
    store = build_store()
    plan = RecoveryPlan(
        incident_id=f"incident-{action.value.lower()}",
        attempt_id=failed_event.attempt_id,
        trigger="marker:lifecycle",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": action,
                "node_ids": ["node-a"],
                "execution_owner": "gpu-fault-hyperpod-adapter",
            },
            {
                "action": RecoveryAction.VALIDATE_NODE,
                "node_ids": ["node-a"],
                "execution_owner": "gpu-fault-validation-adapter",
            },
        ],
    )

    compiled = PassiveWorkflowCompiler(store).compile(plan, failed_event)
    workflow = store.get_workflow(compiled.workflow_request_id)

    expected_lifecycle_operation = (
        WorkflowOperation.RESTART_NODE
        if action is RecoveryAction.REBOOT_NODE
        else WorkflowOperation.REPLACE_NODE
    )
    assert [step.operation for step in workflow.official_steps] == [
        expected_lifecycle_operation,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    ]


def test_quarantine_marker_uses_scheduler_drain_owner(
    failed_event: TerminalEvent,
) -> None:
    store = build_store()
    profile = EffectiveRuntimeProfile(
        cluster_id=failed_event.cluster_id,
        environment=Environment.HYPERPOD_EKS,
        profile_version="hyperpod-v1",
        capabilities=[
            EffectiveCapability(
                capability=CapabilityName.SCHEDULER_DRAIN,
                mode=CapabilityMode.OWN,
                owner="gpu-fault-kubernetes-adapter",
                adapter="regional-cluster-executor",
                observed_version="v1",
            )
        ],
    )
    marker = NodeMarker(
        marker_id="marker-quarantine-owner",
        source="test",
        trusted=True,
        incident_id="incident-quarantine-owner",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity="critical",
        recommended_action=RecoveryAction.QUARANTINE,
        mapping_version="test",
    )
    plan = PlanBuilder().from_marker(failed_event, marker, profile)

    assert [step.execution_owner for step in plan.steps] == [
        "gpu-fault-kubernetes-adapter",
        "gpu-fault-kubernetes-adapter",
    ]

    compiled = PassiveWorkflowCompiler(store).compile(plan, failed_event)
    workflow = store.get_workflow(compiled.workflow_request_id)
    assert [step.execution_owner for step in workflow.official_steps] == [
        "gpu-fault-kubernetes-adapter",
        "gpu-fault-kubernetes-adapter",
    ]


def test_operator_reset_stops_and_restarts_owned_workload(
    failed_event: TerminalEvent,
) -> None:
    profile = default_simulated_profile()
    event = copy_model(
        failed_event, workload_ids=["training/pytorchjob/distributed-training"]
    )
    marker = NodeMarker(
        marker_id="operator-reset-active",
        source="operator-change",
        trusted=True,
        incident_id="operator-reset-active",
        observed_at=event.ended_at,
        expires_at=event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        severity="critical",
        recommended_action=RecoveryAction.RESET_GPU,
        mapping_version="operator-change-v1",
    )

    plan = PlanBuilder().from_marker(event, marker, profile)

    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.STOP_WORKLOAD,
        RecoveryAction.RESET_GPU,
        RecoveryAction.VALIDATE_NODE,
        RecoveryAction.RESTORE_SCHEDULING,
        RecoveryAction.RESTART_WORKLOAD,
    ]


def test_operator_reset_on_idle_node_has_no_workload_steps(
    failed_event: TerminalEvent,
) -> None:
    profile = default_simulated_profile()
    event = copy_model(failed_event, workload_ids=[])
    marker = NodeMarker(
        marker_id="operator-reset-idle",
        source="operator-change",
        trusted=True,
        incident_id="operator-reset-idle",
        observed_at=event.ended_at,
        expires_at=event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        severity="critical",
        recommended_action=RecoveryAction.RESET_GPU,
        mapping_version="operator-change-v1",
    )

    plan = PlanBuilder().from_marker(event, marker, profile)

    assert RecoveryAction.STOP_WORKLOAD not in {step.action for step in plan.steps}
    assert RecoveryAction.RESTART_WORKLOAD not in {step.action for step in plan.steps}
    assert RecoveryAction.RESTORE_SCHEDULING in {step.action for step in plan.steps}


def test_destructive_marker_requires_explicit_scope(
    failed_event: TerminalEvent,
) -> None:
    marker = NodeMarker(
        marker_id="operator-reboot-empty-scope",
        source="operator-change",
        trusted=True,
        incident_id="operator-reboot-empty-scope",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(),
        severity="critical",
        recommended_action=RecoveryAction.REBOOT_NODE,
        mapping_version="operator-change-v1",
    )

    with pytest.raises(UnsupportedPlanError, match="requires explicit node scope"):
        PlanBuilder().from_marker(failed_event, marker, default_simulated_profile())


def test_reset_marker_requires_explicit_gpu_scope(failed_event: TerminalEvent) -> None:
    marker = NodeMarker(
        marker_id="operator-reset-empty-gpu",
        source="operator-change",
        trusted=True,
        incident_id="operator-reset-empty-gpu",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity="critical",
        recommended_action=RecoveryAction.RESET_GPU,
        mapping_version="operator-change-v1",
    )

    with pytest.raises(UnsupportedPlanError, match="requires explicit GPU scope"):
        PlanBuilder().from_marker(failed_event, marker, default_simulated_profile())


def test_recovery_action_rank_is_exhaustive_and_deterministic() -> None:
    assert set(RECOVERY_ACTION_RANK) == set(RecoveryAction)
    assert recovery_action_sort_key(
        RecoveryAction.ESCALATE_OPERATOR
    ) > recovery_action_sort_key(RecoveryAction.RESTART_WORKLOAD)
    left = max(
        [RecoveryAction.DRAIN, RecoveryAction.QUARANTINE], key=recovery_action_sort_key
    )
    right = max(
        [RecoveryAction.QUARANTINE, RecoveryAction.DRAIN], key=recovery_action_sort_key
    )
    assert left is right is RecoveryAction.QUARANTINE


def test_missing_terminal_capability_keeps_containment_and_support(
    failed_event: TerminalEvent,
) -> None:
    profile = default_simulated_profile()
    profile = copy_model(
        profile,
        capabilities=[
            item
            for item in profile.capabilities
            if item.capability is not CapabilityName.NODE_REBOOT
        ],
    )
    marker = NodeMarker(
        marker_id="reboot-without-owner",
        source="test",
        trusted=True,
        incident_id="reboot-without-owner",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity="critical",
        recommended_action=RecoveryAction.REBOOT_NODE,
        mapping_version="test",
    )
    event = copy_model(
        failed_event, workload_ids=["training/pytorchjob/distributed-training"]
    )

    plan = PlanBuilder().from_marker(event, marker, profile)

    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.STOP_WORKLOAD,
        RecoveryAction.QUARANTINE,
        RecoveryAction.ESCALATE_OPERATOR,
    ]
    assert all(step.action is not RecoveryAction.REBOOT_NODE for step in plan.steps)
    assert all(
        step.action is not RecoveryAction.RESTART_WORKLOAD for step in plan.steps
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (RecoveryAction.COLLECT_EVIDENCE, [RecoveryAction.COLLECT_EVIDENCE]),
        (RecoveryAction.RESTORE_SCHEDULING, [RecoveryAction.RESTORE_SCHEDULING]),
        (
            RecoveryAction.RUN_DIAGNOSTICS,
            [RecoveryAction.COLLECT_EVIDENCE, RecoveryAction.RUN_DIAGNOSTICS],
        ),
    ],
)
def test_marker_actions_keep_their_declared_semantics(
    failed_event: TerminalEvent, action: RecoveryAction, expected: list[RecoveryAction]
) -> None:
    marker = NodeMarker(
        marker_id=f"marker-{action.value.lower()}",
        source="test",
        trusted=True,
        incident_id=f"incident-{action.value.lower()}",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity="warning",
        recommended_action=action,
        mapping_version="test",
    )

    plan = PlanBuilder().from_marker(failed_event, marker, default_simulated_profile())

    assert [step.action for step in plan.steps] == expected


def test_operator_drill_id_reaches_compiled_incident(
    failed_event: TerminalEvent,
) -> None:
    store = build_store()
    marker = NodeMarker(
        marker_id="operator-reset-drill",
        source="operator-change",
        trusted=True,
        incident_id="operator-reset-drill",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        severity="critical",
        recommended_action=RecoveryAction.RESET_GPU,
        mapping_version="operator-change-v1",
        drill_id="ha003-20260811",
    )

    plan = PlanBuilder().from_marker(failed_event, marker, default_simulated_profile())
    compiled = PassiveWorkflowCompiler(store).compile(plan, failed_event)

    assert plan.drill_id == "ha003-20260811"
    assert store.get_incident(compiled.incident_id).drill_id == "ha003-20260811"


def test_dispatcher_updates_passive_plan_status(failed_event: TerminalEvent) -> None:
    class RestartAdapter:
        def supports(self, step):
            return step.operation is WorkflowOperation.RESTART_WORKLOAD

        def execute(self, _context):
            return WorkflowStepOutcome.succeeded()

    store = build_store()
    event = copy_model(failed_event, workload_ids=["training/job/train"])
    plan = RecoveryPlan(
        incident_id="incident-dispatch",
        attempt_id=event.attempt_id,
        trigger="quick-triage:PASS",
        runtime_profile_version="simulated-v1",
        steps=[
            {
                "action": RecoveryAction.RESTART_WORKLOAD,
                "node_ids": ["node-a"],
                "execution_owner": "simulated-runtime",
            }
        ],
    )
    compiled = PassiveWorkflowCompiler(store).compile(plan, event)
    store.save_plan(compiled)
    executor = active_workflow_executor(
        store, [RestartAdapter()], {WorkflowOperation.RESTART_WORKLOAD}
    )
    dispatcher = WorkflowDispatcher(
        store, executor, WorkflowDispatcherConfig(enabled=True)
    )

    report = dispatcher.run_once()

    assert report.completed == 1
    assert store.get_plan(compiled.plan_id).status is (PlanStatus.SUCCEEDED)
