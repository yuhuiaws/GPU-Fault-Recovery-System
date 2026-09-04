from __future__ import annotations

from tests._builders import (
    attempt_observation,
    container_observation,
    copy_model,
    node_health_finding,
    workflow_step,
)

from ._support import (
    NOW,
    ApplicationContext,
    IncidentOrchestrator,
    NodeHealthCategory,
    NodeHealthFinding,
    RecoveryAction,
    RecoveryArbiter,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkloadState,
    _claims_the_node,
    _device_resource_finding,
    _inventory_mismatch_finding,
    _node_event,
    _save_attempt,
    ingest,
    pytest,
    timedelta,
)


def test_resource_claims_allow_non_conflicting_node_mutations() -> None:
    claims = IncidentOrchestrator._operation_resource_claims
    conflicts = RecoveryArbiter.resource_claims_conflict

    gpu_runtime = claims(WorkflowOperation.RESET_GPU)
    efa_runtime = claims(WorkflowOperation.REMEDIATE_EFA_DRIVER)
    gpu_plugin = claims(WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN)
    efa_plugin = claims(WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN)
    dcgm_diagnostic = claims(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)
    node_lifecycle = claims(WorkflowOperation.RESTART_NODE)

    assert not conflicts(gpu_runtime, efa_runtime)
    assert not conflicts(gpu_plugin, efa_plugin)
    assert conflicts(gpu_runtime, gpu_runtime)
    assert conflicts(gpu_runtime, dcgm_diagnostic)
    assert conflicts(dcgm_diagnostic, dcgm_diagnostic)
    assert conflicts(node_lifecycle, gpu_runtime)
    assert conflicts(node_lifecycle, efa_plugin)


def test_node_predecessor_requires_same_node_conflicting_claims(
    context: ApplicationContext,
) -> None:
    _, _, reset = ingest(
        context, _node_event(48, event_id="node-ownership-reset", gpu_uuid="GPU-a")
    )
    reset = copy_model(reset, status=WorkflowStatus.RUNNING)
    context.store.save_workflow(reset)
    conflicts = context.orchestrator._conflicts

    def steps(operation: WorkflowOperation, node_id: str) -> list[WorkflowStepSpec]:
        return [
            workflow_step(operation, "owner", node_ids=[node_id], gpu_uuids=["GPU-a"])
        ]

    assert (
        conflicts.active_node_exclusive_workflow(
            "cluster-a",
            {"node-a"},
            candidate_steps=steps(WorkflowOperation.REMEDIATE_EFA_DRIVER, "node-a"),
        )
        is None
    )
    assert (
        conflicts.active_node_exclusive_workflow(
            "cluster-a",
            {"node-a"},
            candidate_steps=steps(WorkflowOperation.RESET_GPU, "node-a"),
        ).request_id
        == reset.request_id
    )
    assert (
        conflicts.active_node_exclusive_workflow(
            "cluster-a",
            {"node-a"},
            candidate_steps=steps(WorkflowOperation.RESTART_NODE, "node-a"),
        ).request_id
        == reset.request_id
    )
    assert (
        conflicts.active_node_exclusive_workflow(
            "cluster-a",
            {"node-b"},
            candidate_steps=steps(WorkflowOperation.RESET_GPU, "node-b"),
        )
        is None
    )


@pytest.mark.parametrize(
    ("category", "validation"),
    [
        (NodeHealthCategory.CPU, WorkflowOperation.VALIDATE_HOST),
        (NodeHealthCategory.MEMORY, WorkflowOperation.VALIDATE_HOST),
        (NodeHealthCategory.STORAGE, WorkflowOperation.VALIDATE_HOST),
        (NodeHealthCategory.RDMA, WorkflowOperation.VALIDATE_FABRIC),
        (NodeHealthCategory.NCCL, WorkflowOperation.VALIDATE_FABRIC),
    ],
)
def test_collector_backed_diagnostics_do_not_recapture_evidence(
    context: ApplicationContext,
    category: NodeHealthCategory,
    validation: WorkflowOperation,
) -> None:
    _, workflow = context.orchestrator.ingest_node_health(
        node_health_finding(
            f"finding-{category.value.lower()}",
            f"event-{category.value.lower()}",
            observed_at=NOW,
            category=category,
            severity="warning",
            reason="collector already captured the evidence",
            recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
            evidence_ref=f"s3://evidence/{category.value}.json",
            runtime_profile_version="simulated-v1",
        )
    )

    operations = [step.operation for step in workflow.official_steps]
    assert operations[0] is WorkflowOperation.FREEZE_EVIDENCE
    assert WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE not in operations
    assert validation in operations


def test_idle_node_resource_group_upgrades_plugin_to_driver_remediation(
    context: ApplicationContext,
) -> None:
    first_incident, plugin = context.orchestrator.ingest_node_health(
        _device_resource_finding(
            event_id="efa-plugin-first",
            metric_name="efa_kubernetes_allocatable_mismatch",
            action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
        )
    )

    incident, upgraded = context.orchestrator.ingest_node_health(
        _device_resource_finding(
            event_id="efa-driver-second",
            metric_name="efa_inventory_mismatch",
            action=RecoveryAction.REMEDIATE_EFA_DRIVER,
        )
    )

    assert upgraded.request_id == plugin.request_id
    assert upgraded.fencing_token == plugin.fencing_token + 1
    operations = {step.operation for step in upgraded.official_steps}
    assert WorkflowOperation.REMEDIATE_EFA_DRIVER in operations
    assert WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN in operations
    assert incident.incident_id == first_incident.incident_id
    assert len(incident.reasons) == 2


def test_idle_node_resource_group_absorbs_plugin_after_driver(
    context: ApplicationContext,
) -> None:
    first_incident, driver = context.orchestrator.ingest_node_health(
        _device_resource_finding(
            event_id="efa-driver-first",
            metric_name="efa_inventory_mismatch",
            action=RecoveryAction.REMEDIATE_EFA_DRIVER,
        )
    )

    incident, absorbed = context.orchestrator.ingest_node_health(
        _device_resource_finding(
            event_id="efa-plugin-second",
            metric_name="efa_kubernetes_allocatable_mismatch",
            action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
        )
    )

    assert absorbed.request_id == driver.request_id
    assert absorbed.fencing_token == driver.fencing_token
    assert incident.incident_id == first_incident.incident_id
    assert len(incident.reasons) == 2


@pytest.mark.parametrize("workload_active", [False, True])
def test_inventory_finding_queues_reboot_while_reset_validation_is_pending(
    context: ApplicationContext, workload_active: bool
) -> None:
    """A confirmed missing card is stronger than an in-flight reset.

    RESET_GPU cannot target a GPU that disappeared from inventory, and
    a running executor has already snapshotted its validation step.
    Queue/preempt with RESTART_NODE instead of silently absorbing the
    node-wide card-loss evidence.
    """
    fault = _node_event(48, event_id="reset-in-flight", gpu_uuid="GPU-a")
    if workload_active:
        _save_attempt(context, ("node-a",))
        fault = copy_model(
            fault,
            workload_state=WorkloadState.ACTIVE,
            job_id="job-a",
            attempt_id="attempt-a",
            affected_workload_ids=["training/job/job-a"],
        )
    _, _, reset = ingest(context, fault)
    context.store.save_workflow(
        copy_model(
            reset,
            status=WorkflowStatus.RUNNING,
            not_before=None,
            execution_owner_id="executor-1",
            completed_step_indexes=[0],
        )
    )

    finding = _inventory_mismatch_finding(
        event_id="inventory-mid-reset",
        workload_state=(
            WorkloadState.ACTIVE if workload_active else WorkloadState.IDLE
        ),
        affected_workload_ids=(["training/job/job-a"] if workload_active else []),
    )
    incident, reboot = context.orchestrator.ingest_node_health(finding)

    assert reboot is not None
    assert reboot.request_id != reset.request_id
    assert reboot.predecessor_workflow_id == reset.request_id
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in reboot.official_steps
    }
    assert (
        context.store.get_incident_by_event(finding.event_id).incident_id
        == incident.incident_id
    )


def test_inventory_finding_queues_after_reset_validation_completed(
    context: ApplicationContext,
) -> None:
    """Post-validation mismatch is new evidence and must reboot."""
    _, _, reset = ingest(
        context, _node_event(48, event_id="reset-validation-finished", gpu_uuid="GPU-a")
    )
    validation_index = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is WorkflowOperation.VALIDATE_GPU
    )
    context.store.save_workflow(
        copy_model(
            reset,
            status=WorkflowStatus.RUNNING,
            not_before=None,
            execution_owner_id="executor-1",
            completed_step_indexes=list(range(validation_index + 1)),
        )
    )

    _, reboot = context.orchestrator.ingest_node_health(
        _inventory_mismatch_finding(event_id="inventory-after-validation")
    )

    assert reboot is not None
    assert reboot.request_id != reset.request_id
    assert reboot.predecessor_workflow_id == reset.request_id
    assert WorkflowOperation.RESTART_NODE in [
        step.operation for step in reboot.official_steps
    ]


def test_host_finding_does_not_queue_behind_evidence_only_work(
    context: ApplicationContext,
) -> None:
    """Only workflows that take the node block a later one.

    A pure evidence or validation workflow neither holds the host nor
    cares who else does, and making a reboot wait on it would only age
    the node further.
    """
    _, diagnostics = context.orchestrator.ingest_node_health(
        node_health_finding(
            "finding-diagnose-only",
            "diagnose-only",
            observed_at=NOW,
            category=NodeHealthCategory.GPU,
            severity="warning",
            metric_name="dcgm_xid_errors",
            reason="synthetic diagnostics-only finding",
            recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        )
    )
    assert diagnostics is not None
    assert not _claims_the_node(diagnostics)
    context.store.save_workflow(
        copy_model(
            diagnostics,
            status=WorkflowStatus.RUNNING,
            not_before=None,
            execution_owner_id="executor-1",
        )
    )

    _, reboot = context.orchestrator.ingest_node_health(
        _inventory_mismatch_finding(event_id="inventory-after-diag")
    )

    assert reboot is not None
    assert reboot.predecessor_workflow_id is None


def test_an_unqueued_reboot_does_not_take_the_incident_from_the_open_workflow(
    context: ApplicationContext,
) -> None:
    """The reason this family needs no retired-generation audit of its own.

    A workflow whose incident stops naming it is unclosable by anything on the
    ingest path -- that is the 2026-09-04 wedge -- so it matters that this family
    cannot produce one. It cannot, and not because of the predecessor link:
    ``NodeHealthPlanBuilder`` builds a *fresh* ``FaultIncident`` per finding with
    no workflow pointer at all, so the reboot above gets its own incident and the
    diagnostics workflow keeps the one that names it. There is no pointer to move
    and nothing is displaced.

    Pinned because the ingest-side half of the fix rests on it. If a later change
    lets this family re-plan an existing incident onto a new workflow -- which is
    the shape that stranded a ``RESTART_APP`` generation with an open
    ``STOP_WORKLOADS`` command -- it has to fail here rather than surface as a
    record only ``WorkflowDispatcher``'s revocation sweep can clean up.
    """

    diag_incident, diagnostics = context.orchestrator.ingest_node_health(
        node_health_finding(
            "finding-diagnose-then-reboot",
            "diagnose-then-reboot",
            observed_at=NOW,
            category=NodeHealthCategory.GPU,
            severity="warning",
            metric_name="dcgm_xid_errors",
            reason="synthetic diagnostics-only finding",
            recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        )
    )
    assert diagnostics is not None
    context.store.save_workflow(
        copy_model(
            diagnostics,
            status=WorkflowStatus.RUNNING,
            not_before=None,
            execution_owner_id="executor-1",
        )
    )

    reboot_incident, reboot = context.orchestrator.ingest_node_health(
        _inventory_mismatch_finding(event_id="inventory-retires-diag")
    )

    assert reboot is not None
    assert reboot_incident.incident_id != diag_incident.incident_id
    assert reboot_incident.workflow_request_id == reboot.request_id
    held = context.store.get_incident(diag_incident.incident_id)
    assert held.workflow_request_id == diagnostics.request_id, (
        "the open workflow must keep the incident that names it, or nothing on "
        "the ingest path can ever close it"
    )


def test_site_replacement_finding_creates_full_active_workflow(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-replace-node-a",
        "replace-node-a",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="repeated hardware fault after remediation",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)

    assert incident.effective_action is RecoveryAction.REPLACE_NODE
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]


def test_site_replacement_can_require_healthy_warm_spare(
    context: ApplicationContext,
) -> None:
    finding = node_health_finding(
        "finding-warm-spare-only",
        "warm-spare-only",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic warm-spare replacement test",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )

    _, workflow = context.orchestrator.ingest_node_health(finding)

    replace_step = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert replace_step.parameters == {
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"
    }


def test_replacement_findings_for_same_attempt_form_one_multi_node_workflow(
    context: ApplicationContext,
) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "job-a-a001",
            NOW,
            expected_critical_ranks=3,
            containers=[
                container_observation(
                    "pod-0", "worker-0", 0, "node-a", gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-1", "worker-1", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
                container_observation(
                    "pod-2", "worker-2", 2, "node-c", gpu_uuids=["GPU-c"]
                ),
            ],
            workload_ids=["training/job/job-a"],
            restart_budget=2,
        )
    )

    def replacement(node_id: str) -> NodeHealthFinding:
        return node_health_finding(
            f"finding-{node_id}",
            f"event-{node_id}",
            node_id=node_id,
            observed_at=NOW,
            category=NodeHealthCategory.GPU,
            severity="critical",
            reason=f"unrecoverable GPU fault on {node_id}",
            recommended_action=RecoveryAction.REPLACE_NODE,
            gpu_uuids=[f"GPU-{node_id[-1]}"],
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=["training/job/job-a"],
        )

    first_incident, first_workflow = context.orchestrator.ingest_node_health(
        replacement("node-a")
    )
    incident, workflow = context.orchestrator.ingest_node_health(replacement("node-b"))

    assert incident.incident_id == first_incident.incident_id
    assert workflow.request_id == first_workflow.request_id
    assert incident.node_ids == ["node-a", "node-b"]
    assert workflow.not_before is not None
    assert len(context.store.list_workflows()) == 1
    assert [step.operation for step in workflow.official_steps].count(
        WorkflowOperation.STOP_WORKLOADS
    ) == 1
    assert [step.operation for step in workflow.official_steps].count(
        WorkflowOperation.RESTART_WORKLOAD
    ) == 1
    for step in workflow.official_steps:
        if step.operation in {
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }:
            assert step.node_ids == ["node-a", "node-b", "node-c"]
        else:
            assert step.node_ids == ["node-a", "node-b"]
    restart = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    replace = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    )
    assert replace.parameters == {}
    assert "job-a" not in replace.workload_ids
    assert restart.parameters == {
        "cluster_id": "cluster-a",
        "job_id": "job-a",
        "source_attempt_id": "job-a-a001",
        "source_gpu_count": 3,
        "restart_budget": 2,
    }


def test_replacement_findings_for_different_attempts_do_not_merge(
    context: ApplicationContext,
) -> None:
    for suffix, node_id, observed_at in [
        ("a", "node-a", NOW),
        ("b", "node-b", NOW + timedelta(seconds=1)),
    ]:
        context.store.save_attempt_observation(
            attempt_observation(
                f"job-{suffix}",
                f"job-{suffix}-a001",
                observed_at,
                containers=[
                    container_observation(
                        f"pod-{suffix}",
                        f"worker-{suffix}",
                        0,
                        node_id,
                        gpu_uuids=[f"GPU-{suffix}"],
                    )
                ],
                workload_ids=[f"training/job/job-{suffix}"],
            )
        )
    incidents = []
    for suffix, node_id in [("a", "node-a"), ("b", "node-b")]:
        incident, _ = context.orchestrator.ingest_node_health(
            node_health_finding(
                f"finding-{suffix}",
                f"event-{suffix}",
                node_id=node_id,
                observed_at=NOW,
                category=NodeHealthCategory.GPU,
                severity="critical",
                reason="unrecoverable GPU fault",
                recommended_action=RecoveryAction.REPLACE_NODE,
                runtime_profile_version="simulated-v1",
                workload_state=WorkloadState.ACTIVE,
                affected_workload_ids=[f"training/job/job-{suffix}"],
            )
        )
        incidents.append(incident.incident_id)

    assert len(set(incidents)) == 2
    assert len(context.store.list_workflows()) == 2


def test_efa_hung_finding_requests_process_diagnostic_bundle(
    context: ApplicationContext,
) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            NOW,
            expected_critical_ranks=2,
            containers=[
                container_observation(
                    "pod-a", "worker-0", 0, "node-a", gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-b", "worker-1", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
            ],
            workload_ids=["training/job/job-a"],
        )
    )
    parameters = {
        "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
        "capture_process_state": True,
        "strace_duration_seconds": 10,
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "workload_ids": ["training/job/job-a"],
    }
    finding = node_health_finding(
        "finding-efa-hung",
        "efa-hung",
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
        diagnostic_parameters=parameters,
    )
    peer_finding = copy_model(
        finding,
        finding_id="finding-efa-hung-peer",
        event_id="efa-hung-peer",
        node_id="node-b",
        reason="EFA traffic remains at zero on peer",
        diagnostic_parameters={**parameters, "baseline_bytes_per_second": 125000000.0},
    )

    incident, workflow = context.orchestrator.ingest_node_health(finding)
    peer_incident, peer_workflow = context.orchestrator.ingest_node_health(peer_finding)

    assert workflow is not None
    assert peer_workflow is not None
    assert peer_incident.incident_id == incident.incident_id
    assert peer_workflow.request_id == workflow.request_id
    assert len(context.store.list_workflows()) == 1
    peer_event_incident = context.store.get_incident_by_event("efa-hung-peer")
    assert peer_event_incident is not None
    assert peer_event_incident.incident_id == incident.incident_id
    assert incident.node_ids == ["node-a", "node-b"]
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.COLLECT_HUNG_TRIAGE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.VALIDATE_FABRIC,
    ]
    assert workflow.dag_enabled
    assert workflow.dag_revision == 1
    assert [step.depends_on_step_indexes for step in workflow.official_steps] == [
        [],
        [0],
        [1],
        [1],
    ]
    assert all(
        step.node_ids == ["node-a", "node-b"] for step in workflow.official_steps
    )
    triage = workflow.official_steps[1]
    assert triage.parameters == {
        **parameters,
        "attempt_node_ids": ["node-a", "node-b"],
        "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]},
        "triage_timeout_seconds": 10,
        "expand_python_cgroup_processes": False,
    }
    bundle = workflow.official_steps[2]
    assert bundle.parameters["capture_process_state"] is False
    assert bundle.parameters["hung_triage_target_pending"] is True


def test_node_health_without_ingested_at_matches_running_attempt(
    context: ApplicationContext,
) -> None:
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
            workload_ids=["training/job/job-a"],
            restart_budget=1,
        )
    )
    finding = node_health_finding(
        "finding-a",
        "event-a",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic replacement",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=["GPU-a"],
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
        job_id="job-a",
        attempt_id="attempt-a",
        runtime_profile_version="simulated-v1",
    )

    matched = context.orchestrator._attempt_observation(finding)

    assert matched is not None
    assert matched.attempt_id == "attempt-a"
