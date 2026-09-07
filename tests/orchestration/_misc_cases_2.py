from __future__ import annotations

from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import (
    copy_model,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

from ._support import (
    NOW,
    ApplicationContext,
    CapabilityMode,
    CapabilityName,
    IncidentOrchestrator,
    IncidentState,
    NodeHealthCategory,
    PlanStatus,
    RecoveryAction,
    TerminalEvent,
    WorkflowFencingError,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
    event,
    ingest,
    pytest,
)


def test_reset_operation_failure_escalates_to_reboot(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-reset-not-completed",
        "reset-not-completed",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="retired pages pending",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )
    _, reset = context.orchestrator.ingest_node_health(finding)
    assert reset is not None
    failed = copy_model(
        reset,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                2,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="reset rejected",
            )
        ],
    )

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)
    assert escalated is not None
    assert escalated[0].effective_action is RecoveryAction.REBOOT_NODE
    assert any(
        step.operation is WorkflowOperation.RESTART_NODE
        for step in escalated[1].official_steps
    )


def test_reset_failure_is_not_misclassified_by_earlier_dcgm_step(
    context: ApplicationContext,
) -> None:
    incident = fault_incident(
        "incident-dcgm-then-reset",
        "dcgm-then-reset",
        "NODE_HEALTH",
        gpu_uuids=["GPU-a"],
        policy_version="site-v1",
        policy_source="SITE_POLICY",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-dcgm-then-reset",
    )
    workflow = workflow_request(
        "workflow-dcgm-then-reset",
        incident.incident_id,
        WorkflowStatus.FAILED,
        incident.fencing_token,
        runtime_profile_version="simulated-v1",
        official_steps=[
            workflow_step(
                WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                "gpu-fault-node-agent",
                gpu_uuids=["GPU-a"],
            ),
            workflow_step(
                WorkflowOperation.RESET_GPU, "gpu-fault-node-agent", gpu_uuids=["GPU-a"]
            ),
        ],
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.RUN_DCGM_DIAGNOSTIC],
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.RUN_DCGM_DIAGNOSTIC),
            workflow_step_execution(
                1,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="reset rejected",
            ),
        ],
    )
    context.store.save_incident(incident)
    context.store.save_workflow(workflow)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(workflow)

    assert escalated is not None
    assert escalated[0].effective_action is RecoveryAction.REBOOT_NODE
    assert any(
        step.operation is WorkflowOperation.RESTART_NODE
        for step in escalated[1].official_steps
    )


@pytest.mark.parametrize(
    "failed_operation",
    [
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
        WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
        WorkflowOperation.RUN_NVLINK74_WORKFLOW,
    ],
)
def test_failed_firmware_remediation_escalates_directly_to_support(
    context: ApplicationContext, failed_operation: WorkflowOperation
) -> None:
    incident = fault_incident(
        "incident-firmware-failed",
        "firmware-failed",
        "SXID",
        gpu_uuids=["GPU-a"],
        policy_version="fm-1",
        policy_source="NVIDIA_FABRIC_MANAGER",
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-firmware-failed",
    )
    workflow = workflow_request(
        "workflow-firmware-failed",
        incident.incident_id,
        WorkflowStatus.FAILED,
        incident.fencing_token,
        runtime_profile_version="simulated-v1",
        official_action=incident.official_action,
        official_steps=[
            workflow_step(failed_operation, "simulated-runtime", gpu_uuids=["GPU-a"]),
            workflow_step(
                WorkflowOperation.RESET_GPU, "simulated-runtime", gpu_uuids=["GPU-a"]
            ),
        ],
        step_executions=[
            workflow_step_execution(
                0,
                failed_operation,
                WorkflowStepStatus.FAILED,
                error="version verification failed",
            )
        ],
    )
    context.store.save_incident(incident)
    context.store.save_workflow(workflow)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(workflow)

    assert escalated is not None
    assert escalated[0].effective_action is RecoveryAction.ESCALATE_OPERATOR
    operations = {step.operation for step in escalated[1].official_steps}
    assert WorkflowOperation.ESCALATE_SUPPORT in operations
    assert WorkflowOperation.RESTART_NODE not in operations


def test_remediation_failure_chain_ends_in_hardware_offline_support(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-escalation-chain",
        "escalation-chain",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="persistent fabric fault",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    _, reset = context.orchestrator.ingest_node_health(finding)
    assert reset is not None
    reset_index = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    failed_reset = copy_model(
        reset,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                reset_index,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="GPU reset failed",
            )
        ],
    )
    context.store.save_workflow(failed_reset)
    _, reboot = context.orchestrator.escalate_failed_hardware_remediation(failed_reset)

    reboot_index = next(
        index
        for index, step in enumerate(reboot.official_steps)
        if step.operation is WorkflowOperation.RESTART_NODE
    )
    failed_reboot = copy_model(
        reboot,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                reboot_index,
                WorkflowOperation.RESTART_NODE,
                WorkflowStepStatus.FAILED,
                error="node did not return healthy",
            )
        ],
    )
    context.store.save_workflow(failed_reboot)
    replacement_incident, replacement = (
        context.orchestrator.escalate_failed_hardware_remediation(failed_reboot)
    )
    assert replacement_incident.effective_action is RecoveryAction.REPLACE_NODE
    assert any(
        step.operation is WorkflowOperation.REPLACE_NODE
        for step in replacement.official_steps
    )
    replace_step = next(
        step
        for step in replacement.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert replace_step.parameters == {
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"
    }

    replace_index = next(
        index
        for index, step in enumerate(replacement.official_steps)
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    failed_replacement = copy_model(
        replacement,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                replace_index,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.FAILED,
                error="healthy warm-spare capacity exhausted",
            )
        ],
    )
    context.store.save_workflow(failed_replacement)
    support_incident, support = (
        context.orchestrator.escalate_failed_hardware_remediation(failed_replacement)
    )

    assert support_incident.effective_action is RecoveryAction.ESCALATE_OPERATOR
    assert [step.operation for step in support.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.ESCALATE_SUPPORT,
    ]


def test_failed_validation_delegates_reboot_to_managed_hyperpod(
    context: ApplicationContext,
) -> None:
    profile = context.store.get_profile("simulated-v1")
    managed = copy_model(
        profile,
        profile_version="managed-node-recovery-v1",
        capabilities=[
            copy_model(
                capability,
                mode=CapabilityMode.DELEGATE,
                owner="hyperpod-managed-node-recovery",
                adapter="hyperpod-managed",
            )
            if capability.capability is CapabilityName.NODE_REBOOT
            else capability
            for capability in profile.capabilities
        ],
    )
    context.store.save_profile(managed)
    finding = node_health_finding(
        "finding-managed-reset",
        "managed-reset",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="uncorrectable ECC errors detected",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version=managed.profile_version,
        workload_state=WorkloadState.IDLE,
    )
    _, reset = context.orchestrator.ingest_node_health(finding)
    assert reset is not None
    failed = copy_model(
        reset,
        status=WorkflowStatus.FAILED,
        completed_operations=[WorkflowOperation.RESET_GPU],
        step_executions=[
            workflow_step_execution(
                3, WorkflowOperation.VALIDATE_GPU, WorkflowStepStatus.FAILED
            )
        ],
    )

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None
    reboot_step = next(
        step
        for step in escalated[1].official_steps
        if step.operation is WorkflowOperation.RESTART_NODE
    )
    assert reboot_step.execution_owner == "hyperpod-managed-node-recovery"


def test_unknown_workload_state_blocks_destructive_action(
    context: ApplicationContext,
) -> None:
    _, incident, workflow = ingest(
        context,
        event(95, event_id="unknown-workloads", workload_state=WorkloadState.UNKNOWN),
    )

    assert incident.state is IncidentState.SAFETY_PENDING
    assert workflow.status is WorkflowStatus.SAFETY_PENDING
    assert "node workload state is UNKNOWN" in workflow.blocked_reasons
    assert [step.operation for step in workflow.safety_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
    ]


def test_approved_adapter_resolves_official_workflow(
    context: ApplicationContext,
) -> None:
    decision, incident, workflow = ingest(
        context, event(54, event_id="mechanical-check")
    )

    assert decision.official_action == "CHECK_MECHANICALS"
    assert workflow.status is WorkflowStatus.PENDING
    assert WorkflowOperation.CHECK_MECHANICALS in {
        step.operation for step in workflow.official_steps
    }

    result = context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)
    updated = context.store.get_incident(incident.incident_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert result.completed_operations == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.CHECK_MECHANICALS,
    ]
    assert updated.state is IncidentState.RECOVERED


def test_update_swfw_compiles_quiesced_version_pinned_workflow(
    context: ApplicationContext,
) -> None:
    orchestrator = IncidentOrchestrator(
        context.store, target_firmware_version="92.10.14"
    )
    xid_event = event(78, event_id="xid-78-update-swfw")
    decision = context.policy.evaluate_xid(xid_event)

    _, workflow = orchestrator.ingest(xid_event, decision)

    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    update_step = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE
    )
    assert update_step.parameters == {"target_firmware_version": "92.10.14"}


def test_missing_workflow_adapter_runs_only_safety_steps(
    context: ApplicationContext,
) -> None:
    default = context.store.get_profile("simulated-v1")
    limited = copy_model(
        default,
        profile_version="limited-v1",
        capabilities=[
            item
            for item in default.capabilities
            if item.capability is not CapabilityName.MECHANICAL_INSPECTION
        ],
    )
    context.store.save_profile(limited)
    xid_event = event(54, event_id="mechanical-no-adapter")
    xid_event = copy_model(xid_event, runtime_profile_version="limited-v1")
    _, incident, workflow = ingest(context, xid_event)

    assert workflow.status is WorkflowStatus.SAFETY_PENDING
    assert "no executable owner for mechanicalInspection" in (workflow.blocked_reasons)

    result = context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)

    assert result.status is WorkflowStatus.BLOCKED
    assert result.completed_operations == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
    ]
    assert (
        context.store.get_incident(incident.incident_id).state
        is IncidentState.QUARANTINED
    )


def test_fencing_and_idempotent_execution(context: ApplicationContext) -> None:
    _, incident, workflow = ingest(context, event(79, event_id="fenced-reboot"))

    with pytest.raises(WorkflowFencingError, match="stale fencing"):
        context.orchestrator.simulate(workflow.request_id, workflow.fencing_token + 1)

    first = context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)
    second = context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)

    assert first.status is WorkflowStatus.SUCCEEDED
    assert second.status is WorkflowStatus.SUCCEEDED
    assert second.completed_operations == first.completed_operations
    assert (
        context.store.get_incident(incident.incident_id).state
        is IncidentState.RECOVERED
    )


def test_simulate_refuses_to_overwrite_an_incident_moved_under_it(
    context: ApplicationContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARCH-D1: the incident write in ``simulate`` is a CAS on the read copy.

    An incident another replica moved between the read and the write raises
    ``StaleWriteError`` like a stale fencing token would, instead of being
    overwritten with the simulated state.
    """
    _, incident, workflow = ingest(context, event(79, event_id="cas-reboot"))
    store = context.store
    original_get_incident = store.get_incident

    def racing_get_incident(incident_id: str):
        read = original_get_incident(incident_id)
        store.save_incident(
            copy_model(read, reasons=[*read.reasons, "moved by another replica"]),
            expected=read,
        )
        return read

    monkeypatch.setattr(store, "get_incident", racing_get_incident)

    with pytest.raises(StaleWriteError):
        context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)

    monkeypatch.undo()
    moved = store.get_incident(incident.incident_id)
    assert moved.state is not IncidentState.RECOVERED, (
        "a stale simulate write reached the incident row"
    )
    assert "moved by another replica" in moved.reasons, (
        "the other replica's write must survive the lost race"
    )


def test_event_ingestion_is_idempotent(context: ApplicationContext) -> None:
    xid_event = event(79, event_id="duplicate-event")
    decision = context.policy.evaluate_xid(xid_event)

    first_incident, first_workflow = context.orchestrator.ingest(xid_event, decision)
    second_incident, second_workflow = context.orchestrator.ingest(xid_event, decision)

    assert second_incident.incident_id == first_incident.incident_id
    assert second_workflow.request_id == first_workflow.request_id


def test_ignore_event_has_incident_without_workflow(
    context: ApplicationContext,
) -> None:
    _, incident, workflow = ingest(context, event(63, event_id="ignore-event"))

    assert workflow is None
    assert incident.state is IncidentState.RECOVERED


def test_policy_pre_actions_are_compiled_into_the_workflow(
    context: ApplicationContext,
) -> None:
    """Containment the policy demanded must reach the workflow.

    The action branches happen to cordon for today's catalog, so this
    pins the guarantee directly: a decision whose action alone implies
    no containment still gets the cordon and stop its pre_actions ask
    for, ordered after evidence capture.
    """
    xid_event = event(
        13,
        event_id="pre-action-contract",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    decision = context.policy.evaluate_xid(xid_event)
    contained = copy_model(
        decision,
        pre_actions=[RecoveryAction.MARK_UNSCHEDULABLE, RecoveryAction.STOP_WORKLOAD],
    )

    baseline = context.orchestrator._builder.official_operations(xid_event, decision)
    operations = context.orchestrator._builder.official_operations(xid_event, contained)

    assert WorkflowOperation.MARK_UNSCHEDULABLE not in baseline
    assert operations[0] is WorkflowOperation.FREEZE_EVIDENCE
    assert operations[1] is WorkflowOperation.MARK_UNSCHEDULABLE
    assert operations.index(WorkflowOperation.STOP_WORKLOADS) > operations.index(
        WorkflowOperation.MARK_UNSCHEDULABLE
    )
    assert operations.count(WorkflowOperation.STOP_WORKLOADS) == 1


def test_pre_actions_do_not_stop_workloads_on_an_idle_node(
    context: ApplicationContext,
) -> None:
    """Containment must not invent a stop for a node running nothing."""
    xid_event = event(63, event_id="pre-action-idle")
    decision = copy_model(
        context.policy.evaluate_xid(xid_event),
        pre_actions=[RecoveryAction.MARK_UNSCHEDULABLE, RecoveryAction.STOP_WORKLOAD],
    )

    operations = context.orchestrator._builder.official_operations(xid_event, decision)

    assert WorkflowOperation.MARK_UNSCHEDULABLE in operations
    assert WorkflowOperation.STOP_WORKLOADS not in operations


def test_restart_app_requires_workload_identity(context: ApplicationContext) -> None:
    _, _, workflow = ingest(
        context,
        event(13, event_id="restart-without-job", workload_state=WorkloadState.ACTIVE),
    )

    assert workflow.status is WorkflowStatus.SAFETY_PENDING
    assert (
        "workload restart requires an affected workload or job ID"
        in workflow.blocked_reasons
    )


def test_passive_recovery_waits_for_proactive_incident(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    xid_event = copy_model(
        event(79, event_id="shared-incident"), observed_at=failed_event.ended_at
    )
    decision = context.policy.evaluate_xid(xid_event)
    context.completion.add_marker(decision.marker)
    incident, workflow = context.orchestrator.ingest(xid_event, decision)

    completion = context.completion.handle_terminal(failed_event)
    plan = context.store.get_plan(completion.recovery_plan_id)
    blocked = context.executor.execute(plan)

    assert [step.operation for step in workflow.official_steps]
    assert len(plan.steps) == 1
    assert plan.steps[0].parameters["incident_id"] == (incident.incident_id)
    assert blocked.status is PlanStatus.FAILED
    assert "requires RECOVERED" in blocked.error

    context.orchestrator.simulate(workflow.request_id, workflow.fencing_token)
    resumed = context.executor.execute(plan)

    assert resumed.status is PlanStatus.SUCCEEDED
