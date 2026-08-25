from __future__ import annotations

from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
    node_health_finding,
)

from ._support import (
    NOW,
    ApplicationContext,
    IncidentOrchestrator,
    IncidentState,
    NodeHealthCategory,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadPhase,
    WorkloadState,
    _node_event,
    _save_attempt,
    event,
    ingest,
    pytest,
    timedelta,
)


def test_xid_48_solo_and_companion_compile_the_same_containment() -> None:
    """The XID 48 branches differ in label only, never in containment.

    ``_workflow_xid_48`` decides in place at ingest, so arrival order
    picks the branch: XID 63/64 are the row remapper *responding* to
    the double-bit error, so they normally land after it and the solo
    branch is taken. That asymmetry is safe only because both branches
    compile to the identical operation list -- ``_catalog_operations``
    already appends MARK_UNSCHEDULABLE for every RESET_GPU, and
    ``_with_pre_actions`` skips an operation that is already present,
    so ``pre_actions=[MARK_UNSCHEDULABLE]`` is absorbed.

    Pin that here. If this ever diverges, the ordering asymmetry stops
    being cosmetic and starts dropping a cordon on the *common* real
    ordering -- and the fix is to make the branches agree, not to hold
    XID 48 open for the 30s companion window, which would delay
    containment of a real uncorrectable error for no safety gain.
    """
    # Separate contexts on purpose: node-scoped merging would fold a
    # second same-node reset into the first workflow, and comparing an
    # object with itself would assert nothing.
    solo_decision, _, solo_workflow = ingest(
        build_context(), event(48, event_id="xid48-solo")
    )
    companion_context = build_context()
    companion = event(63, event_id="xid48-companion-63")
    primary = event(48, event_id="xid48-with-63")
    companion_decision = companion_context.policy.evaluate_xid(
        primary, companion_events=[companion]
    )
    _, companion_workflow = companion_context.orchestrator.ingest(
        primary, companion_decision
    )
    assert solo_workflow.request_id != companion_workflow.request_id

    assert solo_decision.official_action == "RESET_GPU"
    assert solo_decision.pre_actions == []
    assert companion_decision.official_action == "DRAIN_AND_RESET"
    assert companion_decision.pre_actions == [RecoveryAction.MARK_UNSCHEDULABLE]

    # The label differs; the executed containment must not.
    assert solo_decision.action is companion_decision.action
    assert solo_decision.containment is companion_decision.containment
    assert solo_decision.severity == companion_decision.severity
    assert [step.operation for step in companion_workflow.official_steps] == [
        step.operation for step in solo_workflow.official_steps
    ]
    assert WorkflowOperation.MARK_UNSCHEDULABLE in [
        step.operation for step in solo_workflow.official_steps
    ]
    assert companion_workflow.safety_steps == (solo_workflow.safety_steps)


def test_late_xid_inherits_attempt_from_active_recovery(
    context: ApplicationContext,
) -> None:
    _save_attempt(context, ("node-a",))
    low = copy_model(
        _node_event(48, event_id="late-xid-reset", gpu_uuid="GPU-a"),
        workload_state=WorkloadState.ACTIVE,
        job_id="job-a",
        attempt_id="attempt-a",
        affected_workload_ids=["training/job/job-a"],
    )
    _, _, reset = ingest(context, low)
    reset = copy_model(
        reset,
        status=WorkflowStatus.RUNNING,
        not_before=None,
        execution_owner_id="executor-a",
        completed_step_indexes=[0, 1, 2, 3],
    )
    context.store.save_workflow(reset)
    reset_incident = context.store.get_incident(reset.incident_id)
    assert reset_incident.job_id == "job-a"
    assert reset_incident.attempt_id == "attempt-a"
    assert (
        context.orchestrator._active_node_exclusive_workflow("cluster-a", {"node-a"})
        is not None
    )
    recovery_started_at = reset.created_at
    observation = context.store.list_attempt_observations("cluster-a")[0]
    context.store.save_attempt_observation(
        copy_model(
            observation,
            workload_phase=WorkloadPhase.STOPPED,
            observed_at=recovery_started_at + timedelta(seconds=5),
            containers=[
                copy_model(container, terminated=True, gpu_uuids=[])
                for container in observation.containers
            ],
        )
    )
    high = copy_model(
        _node_event(79, event_id="late-xid-reboot", gpu_uuid="GPU-a"),
        observed_at=recovery_started_at + timedelta(seconds=6),
        workload_state=WorkloadState.IDLE,
        job_id=None,
        attempt_id=None,
        affected_workload_ids=[],
    )
    assert (
        context.orchestrator._active_recovery_attempt_observation(high, {"GPU-a"})
        is not None
    )

    _, _, reboot = ingest(context, high)
    reboot_incident = context.store.get_incident(reboot.incident_id)

    assert reboot_incident.job_id == "job-a"
    assert reboot_incident.attempt_id == "attempt-a"
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in reboot.official_steps
    }
    assert reboot.request_id == reset.request_id
    reset_index = next(
        index
        for index, step in enumerate(reboot.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    assert reset_index in reboot.superseded_step_indexes
    cleanup = next(
        step
        for step in reboot.official_steps
        if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
        and step.parameters.get("preemption_quiesce_handoff_after_reboot")
    )
    assert cleanup.node_ids == ["node-a"]


@pytest.mark.parametrize("replacement_first", [True, False])
def test_xid_and_replacement_arrival_order_joins_same_attempt_dag(
    replacement_first: bool,
) -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "job-a-a001",
            NOW,
            expected_critical_ranks=2,
            containers=[
                container_observation(
                    "pod-a", "worker-a", 0, "node-a", gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-b", "worker-b", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
            ],
            workload_ids=["training/pytorchjob/job-a"],
            restart_budget=1,
        )
    )
    replacement = node_health_finding(
        f"finding-replacement-{replacement_first}",
        f"replacement-{replacement_first}",
        node_id="node-b",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="replace node-b from warm spare",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=["GPU-b"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
        job_id="job-a",
        attempt_id="job-a-a001",
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )
    xid = copy_model(
        event(
            48,
            event_id=f"xid-replacement-order-{replacement_first}",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=["training/pytorchjob/job-a"],
        ),
        node_id="node-a",
        gpu_uuid="GPU-a",
        job_id="job-a",
        attempt_id="job-a-a001",
    )

    if replacement_first:
        replacement_incident, replacement_workflow = (
            context.orchestrator.ingest_node_health(replacement)
        )
        _, xid_incident, xid_workflow = ingest(context, xid)
        workflow = xid_workflow
    else:
        _, xid_incident, xid_workflow = ingest(context, xid)
        replacement_incident, replacement_workflow = (
            context.orchestrator.ingest_node_health(replacement)
        )
        workflow = replacement_workflow

    assert xid_incident.incident_id == replacement_incident.incident_id
    assert xid_workflow.request_id == replacement_workflow.request_id
    assert workflow.dag_enabled
    assert workflow.predecessor_workflow_id is None
    assert (
        sum(
            step.operation is WorkflowOperation.STOP_WORKLOADS
            for step in workflow.official_steps
        )
        == 1
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )
        == 1
    )
    reset = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    )
    replace = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert reset.node_ids == ["node-a"]
    assert replace.node_ids == ["node-b"]


def test_xid_drill_id_is_persisted_on_incident(context: ApplicationContext) -> None:
    xid_event = copy_model(
        event(54, event_id="mechanical-drill"), drill_id="maintenance-20260811"
    )

    _, incident, _ = ingest(context, xid_event)

    assert incident.drill_id == "maintenance-20260811"


def test_h200_xid63_family_scope_has_no_recovery_workflow(
    context: ApplicationContext,
) -> None:
    xid_event = copy_model(event(63, event_id="h200-ignore"), product="NVIDIA H200 NVL")
    decision, incident, workflow = ingest(context, xid_event)

    assert decision.official_action == "IGNORE"
    assert decision.action is RecoveryAction.NO_ACTION
    assert workflow is None
    assert incident.state is IncidentState.RECOVERED


def test_xid_74_first_mechanical_event_resets_before_inspection(
    context: ApplicationContext,
) -> None:
    xid_event = event(74, event_id="xid-74-no-registers")
    _, _, blocked = ingest(context, xid_event)

    complete_event = copy_model(
        xid_event,
        event_id="xid-74-complete",
        registers=[1 << 8, 0, 0, 0, 0, 0, 0],
        nvlink_link_id=3,
        nvlink_occurrence_counts={"register1.bit8": 1},
    )
    _, _, ready = ingest(context, complete_event)

    assert blocked.status is WorkflowStatus.SAFETY_PENDING
    assert (
        "XID 74 decode requires exactly seven register fields "
        "from the kernel event" in blocked.blocked_reasons
    )
    assert ready.status is WorkflowStatus.PENDING
    assert [step.operation for step in ready.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]


@pytest.mark.parametrize(
    ("bit", "count"),
    [pytest.param(4, 3, id="ecc-third"), pytest.param(27, 2, id="report-second")],
)
def test_xid_74_repeated_threshold_runs_diagnostics_and_reset(
    context: ApplicationContext, bit: int, count: int
) -> None:
    threshold_event = copy_model(
        event(74, event_id=f"xid74-threshold-{bit}"),
        registers=[1 << bit, 0, 0, 0, 0, 0, 0],
        nvlink_link_id=3,
        nvlink_occurrence_counts={f"register1.bit{bit}": count},
    )

    decision, incident, workflow = ingest(context, threshold_event)

    assert decision.action is RecoveryAction.RESET_GPU
    assert incident.effective_action is RecoveryAction.RESET_GPU
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RUN_NVLINK74_WORKFLOW,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    diagnostic = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RUN_NVLINK74_WORKFLOW
    )
    assert diagnostic.parameters["nvlink_link_id"] == 3
    assert diagnostic.parameters["nvlink_occurrence_counts"] == {
        f"register1.bit{bit}": count
    }


def test_xid_74_marginal_channel_remediates_then_supports(
    context: ApplicationContext,
) -> None:
    marginal_event = copy_model(
        event(74, event_id="xid74-marginal-channel"),
        registers=[1 << 21, 0, 0, 0, 0, 0, 0],
        nvlink_link_id=3,
        nvlink_occurrence_counts={"register1.bit21": 1},
    )

    decision, _, workflow = ingest(context, marginal_event)

    assert decision.action is RecoveryAction.ESCALATE_OPERATOR
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RUN_NVLINK74_WORKFLOW,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.ESCALATE_SUPPORT,
    ]


def test_persistent_xid74_mechanical_issue_resets_after_diagnostics(
    context: ApplicationContext,
) -> None:
    repeated_event = copy_model(
        event(74, event_id="xid74-repeated"),
        registers=[1 << 8, 0, 0, 0, 0, 0, 0],
        nvlink_link_id=3,
        nvlink_occurrence_counts={"register1.bit8": 2},
    )
    _, incident, repeated = ingest(context, repeated_event)

    assert incident.effective_action is RecoveryAction.RESET_GPU
    assert any("same-link threshold reached" in reason for reason in (incident.reasons))
    assert [step.operation for step in repeated.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RUN_NVLINK74_WORKFLOW,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    field_diag = next(
        step
        for step in repeated.official_steps
        if step.operation is WorkflowOperation.RUN_NVLINK74_WORKFLOW
    )
    assert field_diag.parameters["nvlink_link_id"] == 3
    assert field_diag.parameters["nvlink_occurrence_counts"] == {"register1.bit8": 2}
