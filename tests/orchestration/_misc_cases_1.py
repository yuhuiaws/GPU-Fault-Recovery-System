from __future__ import annotations

from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
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
    AttemptObservation,
    ContainerObservation,
    Environment,
    IncidentOrchestrator,
    IncidentState,
    NodeHealthCategory,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadPhase,
    WorkloadState,
    _node_event,
    event,
    ingest,
    pytest,
    timedelta,
)


@pytest.mark.parametrize(
    ("nodes", "expected_seconds"), [(1, 5), (2, 5), (8, 15), (512, 30)]
)
def test_aggregation_window_scales_with_attempt_size(
    nodes: int, expected_seconds: int
) -> None:
    context = build_context()
    workflow = workflow_request(
        f"window-{nodes}",
        f"incident-window-{nodes}",
        fencing_token=1,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "simulated-runtime",
                node_ids=[f"node-{index}" for index in range(nodes)],
            )
        ],
        created_at=NOW,
        updated_at=NOW,
    )

    not_before, maximum = context.orchestrator._aggregation_deadlines(NOW, workflow)

    assert (not_before - NOW).total_seconds() == expected_seconds
    assert (maximum - NOW).total_seconds() == 60


def test_direct_official_action_creates_owned_proactive_workflow(
    context: ApplicationContext,
) -> None:
    decision, incident, workflow = ingest(
        context, event(79, event_id="proactive-reboot")
    )

    assert decision.official_action == "RESTART_BM"
    assert incident.state is IncidentState.ACTION_PENDING
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    assert {step.execution_owner for step in workflow.official_steps} == {
        "simulated-runtime"
    }


def test_idle_node_equal_rank_widens_unsubmitted_reset(
    context: ApplicationContext,
) -> None:
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    _, _, first = ingest(
        context, _node_event(48, event_id="equal-preempt-gpu-a", gpu_uuid="GPU-a")
    )
    context.store.save_workflow(
        copy_model(
            first,
            status=WorkflowStatus.RUNNING,
            execution_owner_id="executor-a",
            completed_step_indexes=[0],
        )
    )

    _, _, second = ingest(
        context, _node_event(48, event_id="equal-preempt-gpu-b", gpu_uuid="GPU-b")
    )

    assert second.request_id == first.request_id
    assert second.predecessor_workflow_id is None
    assert not second.preempt_predecessor, (
        "expected second.preempt_predecessor to be falsy"
    )
    reset = next(
        step
        for step in second.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    )
    assert set(reset.gpu_uuids) == {"GPU-a", "GPU-b"}


def test_active_job_recovery_is_scoped_to_the_exact_attempt(monkeypatch) -> None:
    context = build_context()
    old_incident = fault_incident(
        "incident-old-attempt",
        "event-old-attempt",
        "GPU_FAULT_GROUP",
        job_id="job-a",
        attempt_id="attempt-old",
        policy_version="test-v1",
        policy_source="test",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-old-attempt",
    )
    old_workflow = workflow_request(
        "workflow-old-attempt",
        old_incident.incident_id,
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action="RESET_GPU",
        official_steps=[
            workflow_step(
                WorkflowOperation.RESET_GPU, "simulated-runtime", gpu_uuids=["GPU-a"]
            )
        ],
    )
    context.store.save_incident(old_incident)
    context.store.save_workflow(old_workflow)
    monkeypatch.setattr(
        context.store,
        "list_workflows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("job recovery must not scan workflow history")
        ),
    )
    new_observation = attempt_observation(
        "job-a",
        "attempt-new",
        NOW,
        containers=[
            container_observation(
                "pod-new", "trainer-new", 0, "node-b", gpu_uuids=["GPU-b"]
            )
        ],
        workload_ids=["training/job/job-a"],
        restart_budget=1,
    )

    assert context.orchestrator._active_job_recovery_workflow(new_observation) is None
    assert context.orchestrator._job_recovery_workflow(new_observation) is None


@pytest.mark.parametrize(
    ("category", "metric_name", "reason"),
    [
        (
            NodeHealthCategory.MCE,
            None,
            "machine check reports an uncorrectable hardware error",
        ),
        (
            NodeHealthCategory.MCE,
            "edac_uncorrectable_errors",
            "EDAC reports an uncorrectable memory error",
        ),
        (
            NodeHealthCategory.STORAGE,
            "smart_health_failed",
            "SMART reports disk failure",
        ),
        (
            NodeHealthCategory.STORAGE,
            "filesystem_used_percent",
            "filesystem is critically full",
        ),
        (
            NodeHealthCategory.STORAGE,
            "buffer_io_error",
            "Buffer I/O error on the training volume",
        ),
        (NodeHealthCategory.RDMA, None, "EFA fatal transport error"),
        (NodeHealthCategory.RDMA, "rdma_fatal_error", "RDMA fatal transport error"),
        (
            NodeHealthCategory.BMC,
            "bmc_critical_sensor",
            "BMC reports a critical sensor",
        ),
    ],
)
def test_critical_host_quarantine_joins_running_attempt_recovery(
    category: NodeHealthCategory, metric_name: str | None, reason: str
) -> None:
    context = build_context()
    workload_id = "training/pytorchjob/train-a"
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            NOW,
            containers=[
                container_observation(
                    "pod-a", "worker-a", 0, "node-a", gpu_uuids=["GPU-a"]
                )
            ],
            workload_ids=[workload_id],
            restart_budget=1,
        )
    )
    xid = copy_model(
        event(
            48,
            event_id=f"reset-before-{category.value.lower()}",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=[workload_id],
        ),
        job_id="job-a",
        attempt_id="attempt-a",
    )
    _, reset_incident, reset = ingest(context, xid)
    context.store.save_workflow(copy_model(reset, status=WorkflowStatus.RUNNING))
    finding = node_health_finding(
        f"finding-{category.value.lower()}",
        f"host-{category.value.lower()}-critical",
        observed_at=NOW,
        category=category,
        severity="critical",
        reason=reason,
        recommended_action=RecoveryAction.QUARANTINE,
        metric_name=metric_name,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[workload_id],
        job_id="job-a",
        attempt_id="attempt-a",
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)

    assert incident.incident_id == reset_incident.incident_id
    assert workflow.request_id == reset.request_id
    assert workflow.dag_enabled, "expected workflow.dag_enabled to be truthy"
    assert len(context.store.list_workflows(limit=10)) == 1
    assert WorkflowOperation.QUARANTINE in {
        step.operation for step in workflow.official_steps
    }
    restart_index = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    restore_indexes = {
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING
        and "node-a" in step.node_ids
    }
    assert restart_index in workflow.superseded_step_indexes
    assert restore_indexes <= set(workflow.superseded_step_indexes)
    assert any(reason in item for item in incident.reasons), (
        "expected any(reason in item for item in incident.reasons) to be truthy"
    )
    assert any("WORKFLOW_XID_48" in item for item in incident.reasons), (
        'expected any("WORKFLOW_XID_48" in item for item in incident.reasons) to be truthy'
    )


def test_idle_host_quarantine_joins_active_node_reset() -> None:
    context = build_context()
    _, reset_incident, reset = ingest(
        context, _node_event(48, event_id="idle-reset-before-mce", gpu_uuid="GPU-a")
    )
    context.store.save_workflow(copy_model(reset, status=WorkflowStatus.RUNNING))
    finding = node_health_finding(
        "finding-idle-mce",
        "idle-mce-critical",
        observed_at=NOW,
        category=NodeHealthCategory.MCE,
        severity="critical",
        reason="machine check on idle recovery node",
        recommended_action=RecoveryAction.QUARANTINE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)

    assert incident.incident_id == reset_incident.incident_id
    assert workflow.request_id == reset.request_id
    assert workflow.dag_enabled, "expected workflow.dag_enabled to be truthy"
    assert WorkflowOperation.QUARANTINE in {
        step.operation for step in workflow.official_steps
    }
    restore_indexes = {
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING
    }
    assert restore_indexes <= set(workflow.superseded_step_indexes)
    assert len(context.store.list_workflows(limit=10)) == 1


def test_running_gpu_recovery_accepts_same_node_rdma_diagnostics() -> None:
    context = build_context()
    workload_id = "training/pytorchjob/rdma-diagnostic"
    context.store.save_attempt_observation(
        attempt_observation(
            "job-rdma",
            "attempt-rdma",
            NOW,
            containers=[
                container_observation(
                    "pod-rdma", "worker-rdma", 0, "node-a", gpu_uuids=["GPU-a"]
                )
            ],
            workload_ids=[workload_id],
            restart_budget=1,
        )
    )
    xid = copy_model(
        event(
            48,
            event_id="reset-before-rdma-diagnostic",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=[workload_id],
        ),
        job_id="job-rdma",
        attempt_id="attempt-rdma",
    )
    _, reset_incident, reset = ingest(context, xid)
    context.store.save_workflow(copy_model(reset, status=WorkflowStatus.RUNNING))
    finding = node_health_finding(
        "finding-rdma-diagnostic",
        "rdma-diagnostic-after-reset",
        observed_at=NOW,
        category=NodeHealthCategory.RDMA,
        severity="warning",
        reason="RDMA errors increased",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name="rdma_errors_delta",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[workload_id],
        job_id="job-rdma",
        attempt_id="attempt-rdma",
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)

    assert incident.incident_id == reset_incident.incident_id
    assert workflow.request_id == reset.request_id
    assert workflow.dag_enabled, "expected workflow.dag_enabled to be truthy"
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
    assert {step.operation for step in workflow.official_steps} >= {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
    }
    assert (
        next(
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is WorkflowOperation.RESTART_WORKLOAD
        )
        not in workflow.superseded_step_indexes
    )


def test_workload_only_actions_stay_separate_per_gpu(
    context: ApplicationContext,
) -> None:
    """Two RESTART_APP faults on one node must not collapse together.

    Merging exists to stop actions with overlapping host blast radii
    from interleaving. STOP_WORKLOADS plus RESTART_WORKLOAD touch no
    host state, so two of them cannot corrupt each other, and folding
    them in would erase which GPU each fault came from -- exactly what
    the PCI-address identification exists to preserve.
    """
    _, first, _ = ingest(
        context, _node_event(94, event_id="app-gpu-a", gpu_uuid="GPU-a")
    )
    _, second, _ = ingest(
        context, _node_event(94, event_id="app-gpu-b", gpu_uuid="GPU-b")
    )

    assert first.official_action == "RESTART_APP"
    assert first.incident_id != second.incident_id
    assert first.gpu_uuids == ["GPU-a"]
    assert second.gpu_uuids == ["GPU-b"]


def test_finished_node_workflow_does_not_absorb_a_later_fault(
    context: ApplicationContext,
) -> None:
    """A node's merge group must not outlive its workflow.

    The group key of a node is permanent, so a fault arriving after the
    group's workflow finished was merged into it and nothing ran: the
    dispatcher never picks a SUCCEEDED workflow up again, so the caller
    got a SUCCEEDED workflow back as if the new fault were handled.
    """
    _, _, first = ingest(
        context, _node_event(48, event_id="done-gpu-a", gpu_uuid="GPU-a")
    )
    context.store.save_workflow(
        copy_model(first, status=WorkflowStatus.SUCCEEDED, not_before=None)
    )

    _, incident, second = ingest(
        context, _node_event(48, event_id="done-gpu-b", gpu_uuid="GPU-b")
    )

    assert second.request_id != first.request_id
    assert second.status is WorkflowStatus.PENDING
    # Nothing is in flight, so nothing to queue behind.
    assert second.predecessor_workflow_id is None
    assert incident.gpu_uuids == ["GPU-b"]


def test_hyperpod_restart_vm_uses_node_reboot_not_stop_start(
    context: ApplicationContext,
) -> None:
    base = context.store.get_profile("simulated-v1")
    profile = copy_model(
        base,
        environment=Environment.HYPERPOD_EKS,
        profile_version="hyperpod-vm-reboot-v1",
    )
    context.store.save_profile(profile)
    xid_event = copy_model(
        event(
            151,
            event_id="hyperpod-restart-vm",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=["training/job/job-a"],
        ),
        runtime_profile_version=profile.profile_version,
    )

    decision, incident, workflow = ingest(context, xid_event)

    assert decision.official_action == "RESTART_VM"
    assert decision.action is None
    assert incident.official_action == "RESTART_VM"
    assert incident.effective_action is RecoveryAction.REBOOT_NODE
    assert incident.state is IncidentState.ACTION_PENDING
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    assert WorkflowOperation.RESTART_VM not in {
        step.operation for step in workflow.official_steps
    }
    assert any(
        "BatchRebootClusterNodes" in reason and "not EC2 stop/start" in reason
        for reason in incident.reasons
    ), (
        'expected any( "BatchRebootClusterNodes" in reason and "not EC2 stop/start" in reason for reason in incident.reasons ) to be truthy'
    )


def test_non_hyperpod_restart_vm_keeps_distinct_operation(
    context: ApplicationContext,
) -> None:
    xid_event = event(
        151, event_id="generic-restart-vm", affected_workload_ids=["training/job/job-a"]
    )

    decision, incident, workflow = ingest(context, xid_event)

    assert decision.official_action == "RESTART_VM"
    assert incident.effective_action is None
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.RESTART_VM,
    ]


def test_restart_parameters_prioritize_exact_logical_job_id(
    context: ApplicationContext,
) -> None:
    shared_workload = ["training/pytorchjob/shared-name"]
    for job_id, observed_at in [
        ("job-current", NOW),
        ("job-old", NOW + timedelta(minutes=1)),
    ]:
        context.store.save_attempt_observation(
            attempt_observation(
                job_id,
                f"{job_id}-a001",
                observed_at,
                containers=[
                    container_observation(
                        f"pod-{job_id}",
                        "worker-0",
                        0,
                        "node-a",
                        gpu_uuids=[f"GPU-{job_id}"],
                    )
                ],
                workload_ids=shared_workload,
                restart_budget=2,
            )
        )

    parameters = context.orchestrator._builder.restart_step_parameters(
        "cluster-a", shared_workload, job_id="job-current"
    )

    assert parameters == {
        "cluster_id": "cluster-a",
        "job_id": "job-current",
        "source_attempt_id": "job-current-a001",
        "source_gpu_count": 1,
        "restart_budget": 2,
    }


def test_gpu_memory_signal_uses_official_drain_and_reset_boundaries(
    context: ApplicationContext,
) -> None:
    first = context.orchestrator.gpu_metric_action(
        cluster_id="cluster-a",
        node_id="node-a",
        metric_name="ecc_dbe_volatile_total",
        has_explicit_gpu=True,
        default=RecoveryAction.QUARANTINE,
    )
    assert first is RecoveryAction.DRAIN
    pending = context.orchestrator.gpu_metric_action(
        cluster_id="cluster-a",
        node_id="node-a",
        metric_name="row_remap_pending",
        has_explicit_gpu=True,
        default=RecoveryAction.QUARANTINE,
    )
    assert pending is RecoveryAction.RESET_GPU
    assert (
        context.orchestrator.gpu_metric_action(
            cluster_id="cluster-a",
            node_id="node-b",
            metric_name="row_remap_failure",
            has_explicit_gpu=True,
            default=RecoveryAction.QUARANTINE,
        )
        is RecoveryAction.DRAIN
    )


def test_failed_gpu_validation_escalates_to_idempotent_reboot(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-reset-validation",
        "reset-validation",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="uncorrectable ECC errors detected",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    _, reset = context.orchestrator.ingest_node_health(finding)
    assert reset is not None
    failed = copy_model(
        reset,
        status=WorkflowStatus.FAILED,
        completed_operations=[
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ],
        step_executions=[
            workflow_step_execution(
                5,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowStepStatus.FAILED,
                error="DCGM validation failed",
            )
        ],
    )
    context.store.save_workflow(failed)

    first = context.orchestrator.escalate_failed_hardware_remediation(failed)
    second = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert first is not None
    assert second is not None
    incident, reboot = first
    assert second[0].incident_id == incident.incident_id
    assert second[1].request_id == reboot.request_id
    assert reboot.request_id == (f"workflow-reboot-after-{failed.request_id}")
    assert incident.effective_action is RecoveryAction.REBOOT_NODE
    assert incident.gpu_uuids == ["GPU-a"]
    assert [step.operation for step in reboot.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]


def test_gpu_warning_runs_dcgm_diagnostic_and_failure_drains(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-temperature-warning",
        "temperature-warning",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="warning",
        reason="GPU temperature exceeds warning threshold",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
    )

    _, diagnostic = context.orchestrator.ingest_node_health(finding)

    assert diagnostic is not None
    assert [step.operation for step in diagnostic.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        WorkflowOperation.VALIDATE_GPU,
    ]
    diagnostic_index = next(
        index
        for index, step in enumerate(diagnostic.official_steps)
        if step.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC
    )
    failed = copy_model(
        diagnostic,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                diagnostic_index,
                WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                WorkflowStepStatus.FAILED,
                error="DCGM diagnostic returned FAIL",
                details={
                    "failed_nodes": ["node-a"],
                    "node_failures": {"node-a": ["dcgm_diagnostic_fail"]},
                    "node_results": {
                        "node-a": {
                            "evidence_ref": "s3://evidence/node-a/dcgm.json",
                            "recommended_actions": [
                                {
                                    "action_code": "THERMAL_COOLING_INSPECTION",
                                    "priority": "IMMEDIATE",
                                    "instruction": "Inspect airflow and cooling before revalidation.",
                                    "trigger_tests": ["Thermal"],
                                }
                            ],
                        }
                    },
                },
            )
        ],
    )
    context.store.save_workflow(failed)

    first = context.orchestrator.escalate_failed_hardware_remediation(failed)
    second = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert first is not None
    assert second is not None
    incident, drain = first
    assert incident.effective_action is RecoveryAction.DRAIN
    assert any(
        "s3://evidence/node-a/dcgm.json" in reason for reason in incident.reasons
    ), (
        'expected any( "s3://evidence/node-a/dcgm.json" in reason for reason in incident.reasons ) to be truthy'
    )
    assert any(
        "THERMAL_COOLING_INSPECTION" in reason
        and "Inspect airflow and cooling" in reason
        for reason in incident.reasons
    ), (
        'expected any( "THERMAL_COOLING_INSPECTION" in reason and "Inspect airflow and cooling" in reason for reason in incident.reasons ) to be truthy'
    )
    assert second[1].request_id == drain.request_id
    operations = [step.operation for step in drain.official_steps]
    # The DRAIN successor mirrors the node-health DRAIN chain: the node stays
    # quarantined (RMA class) and the chain ends in an explicit operator
    # hand-off, so the incident reaches ESCALATED instead of parking in
    # QUARANTINED with nobody told.
    assert operations == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.ESCALATE_SUPPORT,
    ]
    assert WorkflowOperation.RESTORE_SCHEDULING not in operations
    assert WorkflowOperation.RESTART_WORKLOAD not in operations


def test_active_workload_dcgm_execution_review_does_not_drain(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-busy-dcgm",
        "busy-dcgm",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="warning",
        reason="GPU utilization remained near zero",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
    )
    _, diagnostic = context.orchestrator.ingest_node_health(finding)
    diagnostic_index = next(
        index
        for index, step in enumerate(diagnostic.official_steps)
        if step.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC
    )
    failed = copy_model(
        diagnostic,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                diagnostic_index,
                WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                WorkflowStepStatus.FAILED,
                error="DCGM diagnostic failed or was inconclusive",
                details={
                    "failed_nodes": ["node-a"],
                    "node_failures": {
                        "node-a": ["dcgm_diagnostic_fail", "DCGM_EXECUTION_REVIEW"]
                    },
                    "node_results": {
                        "node-a": {
                            "diagnostic_findings": [],
                            "failed_checks": [],
                            "warning_checks": [],
                            "returncode": 222,
                            "recommended_actions": [
                                {
                                    "action_code": "DCGM_EXECUTION_REVIEW",
                                    "instruction": "rerun during a maintenance window",
                                }
                            ],
                        }
                    },
                },
            )
        ],
    )
    context.store.save_workflow(failed)

    result = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert result is None
    assert not any(
        workflow.request_id.startswith("workflow-drain-after-")
        for workflow in context.store.list_workflows()
    ), (
        'expected any( workflow.request_id.startswith("workflow-drain-after-") for workflow in context.store.list_workflows() ) to be falsy'
    )


def test_partial_dcgm_execution_review_still_escalates(
    context: ApplicationContext,
) -> None:
    """A two-node DCGM step that only folded node-a is not a review-only verdict.

    The node-action adapter reports the per-node results it already collected
    when it exits mid-batch, so a step that failed hard on node-b can carry
    node-a's benign "rerun the diagnostic" verdict and nothing else. Reading
    that partial set as "review only" would suppress the escalation for the
    node that actually failed, so the criterion demands a result for every
    node the step targeted.
    """
    finding = node_health_finding(
        "finding-busy-dcgm-pair",
        "busy-dcgm-pair",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="warning",
        reason="GPU utilization remained near zero",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
    )
    _, diagnostic = context.orchestrator.ingest_node_health(finding)
    diagnostic_index = next(
        index
        for index, step in enumerate(diagnostic.official_steps)
        if step.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC
    )
    failed = copy_model(
        diagnostic,
        status=WorkflowStatus.FAILED,
        official_steps=[
            copy_model(step, node_ids=["node-a", "node-b"])
            if index == diagnostic_index
            else step
            for index, step in enumerate(diagnostic.official_steps)
        ],
        step_executions=[
            workflow_step_execution(
                diagnostic_index,
                WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                WorkflowStepStatus.FAILED,
                error="DCGM diagnostic failed on node-b",
                details={
                    "failed_nodes": ["node-b"],
                    "node_failures": {"node-b": ["dcgm_diagnostic_fail"]},
                    # Only node-a folded before the batch gave up, and its
                    # verdict is the benign one.
                    "node_results": {
                        "node-a": {
                            "diagnostic_findings": [],
                            "failed_checks": [],
                            "warning_checks": [],
                            "returncode": 222,
                            "recommended_actions": [
                                {
                                    "action_code": "DCGM_EXECUTION_REVIEW",
                                    "instruction": "rerun during a maintenance window",
                                }
                            ],
                        }
                    },
                    "completed_nodes": ["node-a"],
                },
            )
        ],
    )
    context.store.save_workflow(failed)

    result = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert result is not None, (
        "expected node-b's hard failure to escalate even though node-a only "
        "asked for a diagnostic rerun"
    )


def test_wide_attempt_hung_triage_samples_representative_nodes(
    context: ApplicationContext, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_HUNG_TRIAGE_MAX_NODES", "4")
    node_ids = [f"node-{index:02d}" for index in range(12)]
    context.store.save_attempt_observation(
        AttemptObservation(
            cluster_id="cluster-a",
            environment=Environment.HYPERPOD_EKS,
            job_id="job-a",
            attempt_id="attempt-a",
            workload_phase=WorkloadPhase.RUNNING,
            observed_at=NOW,
            expected_critical_ranks=len(node_ids),
            containers=[
                ContainerObservation(
                    pod_uid=f"pod-{index}",
                    pod_name=f"worker-{index}",
                    container_name="trainer",
                    role="worker",
                    # Rank 0 deliberately does not sort first, so the
                    # assertion cannot pass by accident.
                    rank=(index - 5) % len(node_ids),
                    node_id=node_id,
                    gpu_uuids=[f"GPU-{index}"],
                )
                for index, node_id in enumerate(node_ids)
            ],
            workload_ids=["training/job/job-a"],
            runtime_profile_version="simulated-v1",
        )
    )
    finding = node_health_finding(
        "finding-efa-hung-wide",
        "efa-hung-wide",
        node_id="node-09",
        observed_at=NOW,
        category=NodeHealthCategory.RDMA,
        severity="critical",
        reason="EFA traffic remains at zero",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
        job_id="job-a",
        attempt_id="attempt-a",
        diagnostic_parameters={
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "capture_process_state": True,
            "job_id": "job-a",
            "attempt_id": "attempt-a",
            "workload_ids": ["training/job/job-a"],
        },
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)

    assert workflow is not None
    # The incident still owns the whole attempt: only the py-spy sweep
    # is narrowed, not the blast radius of the recovery.
    assert incident.node_ids == node_ids
    triage = workflow.official_steps[1]
    assert len(triage.node_ids) == 4
    # The node whose fabric went quiet and the node holding rank 0.
    assert "node-09" in triage.node_ids
    assert "node-05" in triage.node_ids
    assert triage.parameters["attempt_node_ids"] == node_ids
    assert sorted(triage.node_ids + triage.parameters["not_sampled_nodes"]) == node_ids
    assert all(
        step.node_ids == node_ids
        for index, step in enumerate(workflow.official_steps)
        if index != 1
    ), (
        "expected all( step.node_ids == node_ids for index, step in enumerate(workflow.official_steps) if index != 1 ) to be truthy"
    )


def test_hung_triage_node_cap_can_be_disabled(
    context: ApplicationContext, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_HUNG_TRIAGE_MAX_NODES", "0")
    node_ids = [f"node-{index:02d}" for index in range(12)]

    sampled, skipped = context.orchestrator._sample_hung_triage_nodes(
        node_ids, reporting_node_id="node-09", lowest_rank_by_node={"node-05": 0}
    )

    assert sampled == node_ids
    assert skipped == []


def test_hung_triage_node_cap_rejects_negative_values(
    context: ApplicationContext, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_HUNG_TRIAGE_MAX_NODES", "-1")

    with pytest.raises(ValueError, match="GPU_FAULT_HUNG_TRIAGE_MAX_NODES"):
        context.orchestrator._sample_hung_triage_nodes(
            ["node-a", "node-b"], reporting_node_id="node-a", lowest_rank_by_node={}
        )


def test_multi_node_validation_failure_reboots_all_failed_nodes_once(
    context: ApplicationContext,
) -> None:
    incident = fault_incident(
        "incident-distributed-validation",
        "distributed-validation",
        "XID_BATCH",
        node_ids=["worker-0", "worker-1"],
        gpu_uuids=["GPU-0", "GPU-1"],
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    operations = [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.VALIDATE_GPU,
    ]
    steps = [
        workflow_step(
            operation,
            "simulated-runtime",
            node_ids=["worker-0", "worker-1", "worker-2"]
            if operation is WorkflowOperation.STOP_WORKLOADS
            else ["worker-0", "worker-1"],
            gpu_uuids=["GPU-0", "GPU-1"],
            workload_ids=["training/job/distributed"],
            parameters={
                "gpu_uuids_by_node": {"worker-0": ["GPU-0"], "worker-1": ["GPU-1"]}
            }
            if operation is WorkflowOperation.RESET_GPU
            else {},
        )
        for operation in operations
    ]
    failed = workflow_request(
        "workflow-distributed-validation",
        incident.incident_id,
        WorkflowStatus.FAILED,
        1,
        runtime_profile_version="simulated-v1",
        official_steps=steps,
        completed_operations=[WorkflowOperation.RESET_GPU],
        step_executions=[
            workflow_step_execution(
                4,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowStepStatus.FAILED,
                error="validation failed on nodes",
                details={
                    "failed_nodes": ["worker-0", "worker-1"],
                    "node_failures": {
                        "worker-0": ["active_gpu_health_findings"],
                        "worker-1": ["active_gpu_health_findings"],
                    },
                },
            )
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    context.store.save_incident(incident)
    context.store.save_workflow(failed)

    first = context.orchestrator.escalate_failed_hardware_remediation(failed)
    second = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert first is not None
    assert second is not None
    reboot_incident, reboot = first
    assert reboot_incident.node_ids == ["worker-0", "worker-1"]
    assert second[1].request_id == reboot.request_id
    reboot_step = next(
        step
        for step in reboot.official_steps
        if step.operation is WorkflowOperation.RESTART_NODE
    )
    stop_step = next(
        step
        for step in reboot.official_steps
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    restart_step = next(
        step
        for step in reboot.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert reboot_step.node_ids == ["worker-0", "worker-1"]
    assert stop_step.node_ids == ["worker-0", "worker-1", "worker-2"]
    assert restart_step.node_ids == ["worker-0", "worker-1", "worker-2"]
    assert reboot_incident.gpu_uuids == ["GPU-0", "GPU-1"]

    legacy_incident = copy_model(
        incident, incident_id="incident-legacy-validation", event_id="legacy-validation"
    )
    legacy_failed = copy_model(
        failed,
        request_id="workflow-legacy-validation",
        incident_id=legacy_incident.incident_id,
        step_executions=[copy_model(failed.step_executions[0], details={})],
    )
    context.store.save_incident(legacy_incident)
    context.store.save_workflow(legacy_failed)

    legacy = context.orchestrator.escalate_failed_hardware_remediation(legacy_failed)

    assert legacy is not None
    assert legacy[0].node_ids == ["worker-0", "worker-1"]
    legacy_reboot = next(
        step
        for step in legacy[1].official_steps
        if step.operation is WorkflowOperation.RESTART_NODE
    )
    assert legacy_reboot.node_ids == ["worker-0", "worker-1"]

    legacy_incident = copy_model(
        incident, incident_id="incident-legacy-validation", event_id="legacy-validation"
    )
    legacy = copy_model(
        failed,
        request_id="workflow-legacy-validation",
        incident_id=legacy_incident.incident_id,
        step_executions=[copy_model(failed.step_executions[0], details={})],
    )
    context.store.save_incident(legacy_incident)
    context.store.save_workflow(legacy)

    legacy_escalation = context.orchestrator.escalate_failed_hardware_remediation(
        legacy
    )

    assert legacy_escalation is not None
    assert legacy_escalation[0].node_ids == ["worker-0", "worker-1"]
