from __future__ import annotations

from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

from ._support import (
    CONTAINMENT_ONLY_OPERATIONS,
    DESTRUCTIVE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
    NOW,
    ActionDisposition,
    ApplicationContext,
    DagBrancher,
    FaultIncident,
    IncidentOrchestrator,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
    XidEvent,
    _active_agent,
    _node_event,
    datetime,
    event,
    ingest,
    pytest,
    timedelta,
    timezone,
)


def _workflow_merge_service() -> WorkflowMergeService:
    arbiter = RecoveryArbiter()
    return WorkflowMergeService(
        arbiter,
        DagBrancher(arbiter),
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=(
            NodeConflictService.workflow_resource_claims_by_node
        ),
    )


def _preemption_incident(
    incident_id: str,
    *,
    node_id: str,
    job_id: str | None = None,
    attempt_id: str | None = None,
) -> FaultIncident:
    return fault_incident(
        incident_id,
        f"event-{incident_id}",
        node_ids=[node_id],
        job_id=job_id,
        attempt_id=attempt_id,
        policy_version="test",
        policy_source="PREEMPT-003",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.parametrize(
    ("existing", "candidate", "expected"),
    [
        (
            _preemption_incident("idle-a", node_id="node-a"),
            _preemption_incident("idle-b", node_id="node-a"),
            True,
        ),
        (
            _preemption_incident("idle-c", node_id="node-a"),
            _preemption_incident("idle-d", node_id="node-b"),
            False,
        ),
        (
            _preemption_incident(
                "job-a", node_id="node-a", job_id="job-a", attempt_id="attempt-a"
            ),
            _preemption_incident("idle-e", node_id="node-a"),
            False,
        ),
        (
            _preemption_incident(
                "job-b", node_id="node-a", job_id="job-a", attempt_id="attempt-a"
            ),
            _preemption_incident(
                "job-c", node_id="node-a", job_id="job-b", attempt_id="attempt-b"
            ),
            False,
        ),
    ],
)
def test_preemption_scope_boundaries(
    existing: FaultIncident, candidate: FaultIncident, expected: bool
) -> None:
    assert (
        WorkflowMergeService.preemption_scope_matches(existing, candidate) is expected
    )


def test_preemption_requires_strictly_higher_recovery_rank() -> None:
    def workflow(request_id: str, operation: WorkflowOperation) -> WorkflowRequest:
        return workflow_request(
            request_id,
            "rank-gate-incident",
            WorkflowStatus.RUNNING
            if request_id == "existing"
            else WorkflowStatus.PENDING,
            1,
            official_steps=[workflow_step(operation, gpu_uuids=["GPU-a"])],
        )

    service = WorkflowMergeService(
        RecoveryArbiter(),
        None,
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations={
            WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTART_NODE,
        },
        workflow_resource_claims_by_node=lambda _workflow: {},
    )
    existing = workflow("existing", WorkflowOperation.RESET_GPU)

    lower = service.prepare_preempting_successor(
        existing, workflow("lower", WorkflowOperation.RUN_DCGM_DIAGNOSTIC)
    )
    equal = service.prepare_preempting_successor(
        existing, workflow("equal", WorkflowOperation.RESET_GPU)
    )
    higher = service.prepare_preempting_successor(
        existing, workflow("higher", WorkflowOperation.RESTART_NODE)
    )

    assert not lower.preempt_predecessor, (
        "expected lower.preempt_predecessor to be falsy"
    )
    assert not equal.preempt_predecessor, (
        "expected equal.preempt_predecessor to be falsy"
    )
    assert higher.preempt_predecessor, (
        "expected higher.preempt_predecessor to be truthy"
    )
    assert higher.preemption_reason == (
        "strictly stronger recovery action: rank 30 -> 50"
    )


def test_fault_action_generation_fence_blocks_previous_boot() -> None:
    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-current"))
    xid_event = copy_model(
        event(48, event_id="xid-old-boot"),
        source_boot_id="boot-previous",
        source_event_time=NOW,
    )
    decision = context.policy.evaluate_xid(xid_event)

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW
    )

    assert fenced.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert fenced.action is None
    assert fenced.safety_action is None
    assert not fenced.marker.active, "expected fenced.marker.active to be falsy"
    assert any(
        "does not match current Agent boot ID" in reason for reason in fenced.reasons
    ), (
        'expected any( "does not match current Agent boot ID" in reason for reason in fenced.reasons ) to be truthy'
    )
    incident, workflow = context.orchestrator.ingest(xid_event, fenced)
    assert incident.workflow_request_id == workflow.request_id
    assert workflow.status is WorkflowStatus.SAFETY_PENDING
    assert [step.operation for step in workflow.safety_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE
    ]


def test_fault_action_generation_fence_allows_current_boot() -> None:
    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-current"))
    xid_event = copy_model(
        event(48, event_id="xid-current-boot"),
        source_boot_id="boot-current",
        source_event_time=NOW,
    )
    decision = context.policy.evaluate_xid(xid_event)

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW
    )

    assert fenced.disposition is decision.disposition
    assert fenced.action is decision.action
    assert fenced.marker.active, "expected fenced.marker.active to be truthy"


def test_fault_action_generation_fence_blocks_aged_event_without_boot() -> None:
    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-current"))
    xid_event = copy_model(event(48, event_id="xid-aged"), source_event_time=NOW)
    decision = context.policy.evaluate_xid(xid_event)

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW + timedelta(seconds=901)
    )

    assert fenced.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert any(
        "exceeds automatic action limit" in reason for reason in fenced.reasons
    ), (
        'expected any("exceeds automatic action limit" in reason for reason in fenced.reasons) to be truthy'
    )


def test_fault_action_generation_fence_keeps_stale_containment() -> None:
    """Stale evidence must not buy a suspect node its schedulability.

    The fence exists so that a fault from an earlier boot cannot drive a
    GPU reset at today's runtime -- see the previous-boot test above,
    where XID 48 loses RESET_GPU. Cordon, quarantine and operator
    escalation are the opposite case: they only change what the
    scheduler may place, so nulling them would withdraw protection.
    """

    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-current"))
    xid_event = copy_model(
        event(48, event_id="xid-escalate-old-boot"),
        source_boot_id="boot-previous",
        source_event_time=NOW,
    )
    decision = copy_model(
        context.policy.evaluate_xid(xid_event), action=RecoveryAction.ESCALATE_OPERATOR
    )
    operations = context.orchestrator._builder.official_operations(
        xid_event, decision, RecoveryAction.ESCALATE_OPERATOR
    )

    assert operations == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.ESCALATE_SUPPORT,
    ]
    # Containment still counts as destructive for the audit trail; it is
    # only the node-mutating subset that the gates key on.
    assert set(operations) & DESTRUCTIVE_OPERATIONS
    assert set(operations) & CONTAINMENT_ONLY_OPERATIONS
    assert not set(operations) & NODE_MUTATING_OPERATIONS
    assert not context.orchestrator._builder.mutates_node(operations), (
        "expected context.orchestrator._builder.mutates_node(operations) to be falsy"
    )

    fenced = context.orchestrator.apply_fault_action_generation_fence(
        xid_event, decision, now=NOW
    )

    assert fenced.disposition is ActionDisposition.EXECUTABLE
    assert fenced.action is RecoveryAction.ESCALATE_OPERATOR
    assert fenced.marker.active, "expected fenced.marker.active to be truthy"
    assert fenced.reasons == decision.reasons

    incident, workflow = context.orchestrator.ingest(xid_event, fenced)

    assert incident.workflow_request_id == workflow.request_id
    assert WorkflowOperation.ESCALATE_SUPPORT in [
        step.operation for step in workflow.official_steps
    ]


def test_unknown_workload_state_still_allows_escalation() -> None:
    """UNKNOWN workload state blocks node mutation, not containment.

    Stopping or resetting anything while the node's workload state is
    unreadable risks killing an unknown job, so RESET_GPU is held for the
    operator. Cordoning that same node and opening a support case needs
    no workload knowledge at all.
    """

    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-current"))
    xid_event = copy_model(
        event(48, event_id="xid-unknown-workload"),
        source_boot_id="boot-current",
        source_event_time=NOW,
        workload_state=WorkloadState.UNKNOWN,
    )
    decision = context.policy.evaluate_xid(xid_event)

    _, blocked = context.orchestrator.ingest(
        xid_event, copy_model(decision, action=RecoveryAction.RESET_GPU)
    )

    assert blocked.status is WorkflowStatus.SAFETY_PENDING
    assert "node workload state is UNKNOWN" in blocked.blocked_reasons

    escalating_context = build_context()
    escalating_context.store.save_agent(_active_agent(boot_id="boot-current"))
    _, escalated = escalating_context.orchestrator.ingest(
        xid_event, copy_model(decision, action=RecoveryAction.ESCALATE_OPERATOR)
    )

    assert escalated.blocked_reasons == []
    assert escalated.status is WorkflowStatus.PENDING
    assert [step.operation for step in escalated.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.ESCALATE_SUPPORT,
    ]


def test_same_node_actions_merge_instead_of_overlapping(
    context: ApplicationContext,
) -> None:
    """One node must not run a GPU reset and its own reboot at once.

    Two GPUs on an idle node can demand actions with overlapping blast
    radii: GPU-a wants an in-place RESET_GPU, GPU-b wants the box
    rebooted. Both used to compile PENDING workflows with no
    predecessor, because serialization keyed on job_id and an idle node
    has none. Since a WAITING step returns control to the dispatcher
    mid-workflow, the reboot could land between QUIESCE_GPU_SERVICES
    and RESTORE_GPU_SERVICES.

    Ordering them would not fix it -- the reset would still run, aimed
    at pre-reboot state. So they merge and escalate to the single
    action that covers both.
    """
    _, first_incident, first_workflow = ingest(
        context, _node_event(48, event_id="node-a-gpu0", gpu_uuid="GPU-a")
    )
    assert first_incident.official_action == "RESET_GPU"

    _, incident, workflow = ingest(
        context, _node_event(79, event_id="node-a-gpu5", gpu_uuid="GPU-b")
    )

    assert len(context.store.list_workflows(limit=10)) == 1
    assert workflow.request_id == first_workflow.request_id
    assert incident.incident_id == first_incident.incident_id
    assert incident.official_action == "RESTART_BM"
    assert incident.gpu_uuids == ["GPU-a", "GPU-b"]
    operations = [step.operation for step in workflow.official_steps]
    assert WorkflowOperation.RESTART_NODE in operations
    # The absorbed weaker action must not also run.
    assert WorkflowOperation.RESET_GPU not in operations
    # Each fault keeps its own justification for the operator: the
    # first verbatim, later ones prefixed with node and XID.
    assert any("WORKFLOW_XID_48" in reason for reason in incident.reasons), (
        'expected any("WORKFLOW_XID_48" in reason for reason in incident.reasons) to be truthy'
    )
    assert any(reason.startswith("node-a: XID 79:") for reason in incident.reasons), (
        'expected any(reason.startswith("node-a: XID 79:") for reason in incident.reasons) to be truthy'
    )


def test_idle_node_stronger_action_requests_safe_preemption(
    context: ApplicationContext,
) -> None:
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    _, _, reset = ingest(
        context, _node_event(48, event_id="idle-preempt-reset", gpu_uuid="GPU-a")
    )
    mark_index = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is WorkflowOperation.MARK_UNSCHEDULABLE
    )
    context.store.save_workflow(
        copy_model(
            reset,
            status=WorkflowStatus.RUNNING,
            execution_owner_id="executor-a",
            completed_step_indexes=[mark_index],
            completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
            step_executions=[
                workflow_step_execution(
                    mark_index, WorkflowOperation.MARK_UNSCHEDULABLE
                )
            ],
        )
    )

    _, _, reboot = ingest(
        context, _node_event(79, event_id="idle-preempt-reboot", gpu_uuid="GPU-b")
    )

    assert reboot.request_id != reset.request_id
    assert reboot.predecessor_workflow_id == reset.request_id
    assert reboot.preempt_predecessor, (
        "expected reboot.preempt_predecessor to be truthy"
    )
    assert WorkflowOperation.MARK_UNSCHEDULABLE in {
        reboot.official_steps[index].operation
        for index in reboot.inherited_step_indexes
    }


def test_parallel_job_dag_accepts_three_and_five_node_branches() -> None:
    def workflow_for(node_id: str, request_id: str) -> WorkflowRequest:
        return workflow_request(
            request_id,
            "incident-dag",
            fencing_token=1,
            runtime_profile_version="simulated-v1",
            official_action="RESET_GPU",
            official_steps=[
                workflow_step(
                    WorkflowOperation.STOP_WORKLOADS,
                    "owner",
                    node_ids=["node-a", "node-b", "node-c", "node-d", "node-e"],
                    workload_ids=["training/job/job-a"],
                ),
                workflow_step(
                    WorkflowOperation.RESET_GPU,
                    "owner",
                    node_ids=[node_id],
                    gpu_uuids=[f"GPU-{node_id}"],
                ),
                workflow_step(
                    WorkflowOperation.RESTORE_SCHEDULING, "owner", node_ids=[node_id]
                ),
                workflow_step(
                    WorkflowOperation.RESTART_WORKLOAD,
                    "owner",
                    node_ids=["node-a", "node-b", "node-c", "node-d", "node-e"],
                    workload_ids=["training/job/job-a"],
                ),
            ],
        )

    completed_stop = workflow_step_execution(0, WorkflowOperation.STOP_WORKLOADS)
    dag = copy_model(
        workflow_for("node-a", "workflow-node-a"),
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.STOP_WORKLOADS],
        step_executions=[completed_stop],
    )
    brancher = DagBrancher(RecoveryArbiter())
    revisions = []
    for node_id in ["node-b", "node-c", "node-d", "node-e"]:
        dag = brancher.append_parallel_job_branch(
            dag, workflow_for(node_id, f"workflow-{node_id}")
        )
        revisions.append(dag.dag_revision)
        assert dag.status is WorkflowStatus.RUNNING
        assert dag.completed_step_indexes == [0]
        assert dag.completed_operations == [WorkflowOperation.STOP_WORKLOADS]
        assert dag.step_executions == [completed_stop]
        if node_id == "node-c":
            restart = next(
                step
                for step in dag.official_steps
                if step.operation is WorkflowOperation.RESTART_WORKLOAD
            )
            assert len(restart.depends_on_step_indexes) == 3

    assert dag.dag_enabled, "expected dag.dag_enabled to be truthy"
    assert revisions == [1, 2, 3, 4]
    assert dag.dag_revision == 4
    assert (
        sum(
            step.operation is WorkflowOperation.STOP_WORKLOADS
            for step in dag.official_steps
        )
        == 1
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in dag.official_steps
        )
        == 1
    )
    branch_ids = {
        step.branch_id
        for step in dag.official_steps
        if step.branch_id and step.branch_id.startswith("branch:")
    }
    assert branch_ids == {
        "branch:initial",
        "branch:node-b",
        "branch:node-c",
        "branch:node-d",
        "branch:node-e",
    }
    restart = next(
        step
        for step in dag.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert restart.branch_id == "join"
    assert len(restart.depends_on_step_indexes) == 5
    before_duplicate = dag
    duplicate = brancher.append_parallel_job_branch(
        dag, workflow_for("node-e", "workflow-node-e-duplicate")
    )
    assert duplicate == before_duplicate


def _cross_domain_workflow(
    request_id: str,
    operations: list[WorkflowOperation],
    *,
    status: WorkflowStatus = WorkflowStatus.PENDING,
    node_id: str = "node-a",
) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "incident-domain",
        status,
        1,
        runtime_profile_version="simulated-v1",
        official_action=operations[-1].value,
        official_steps=[
            workflow_step(
                operation,
                "simulated-runtime",
                node_ids=[node_id],
                gpu_uuids=["GPU-a"] if operation is WorkflowOperation.RESET_GPU else [],
            )
            for operation in [
                WorkflowOperation.STOP_WORKLOADS,
                *operations,
                WorkflowOperation.RESTART_WORKLOAD,
            ]
        ],
    )


def _assert_cross_domain_parallel_dependencies(
    merger: WorkflowMergeService,
    reset: WorkflowRequest,
    efa_diagnostic: WorkflowRequest,
) -> None:
    combined = merger.brancher.append_parallel_job_branch(reset, efa_diagnostic)
    stop_index = next(
        index
        for index, step in enumerate(combined.official_steps)
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    diagnostic_indexes = [
        index
        for index, step in enumerate(combined.official_steps)
        if step.branch_id == "branch:node-a"
    ]
    freeze_index = next(
        index
        for index in diagnostic_indexes
        if combined.official_steps[index].operation is WorkflowOperation.FREEZE_EVIDENCE
    )
    collect_index = next(
        index
        for index in diagnostic_indexes
        if combined.official_steps[index].operation
        is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
    )
    validate_index = next(
        index
        for index in diagnostic_indexes
        if combined.official_steps[index].operation is WorkflowOperation.VALIDATE_FABRIC
    )
    assert freeze_index in combined.official_steps[stop_index].depends_on_step_indexes
    assert stop_index in combined.official_steps[collect_index].depends_on_step_indexes
    assert (
        collect_index in combined.official_steps[validate_index].depends_on_step_indexes
    )


def test_same_node_cross_domain_actions_require_explicit_dominance() -> None:
    merger = _workflow_merge_service()

    reset = _cross_domain_workflow("workflow-reset", [WorkflowOperation.RESET_GPU])
    efa_diagnostic = workflow_request(
        "workflow-efa-diagnostic",
        "incident-domain",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="RUN_DIAGNOSTICS",
        official_steps=[
            workflow_step(operation, "simulated-runtime")
            for operation in [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.VALIDATE_FABRIC,
            ]
        ],
    )
    dcgm_diagnostic = workflow_request(
        "workflow-dcgm-diagnostic",
        "incident-domain",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="RUN_DIAGNOSTICS",
        official_steps=[
            workflow_step(operation, "simulated-runtime", gpu_uuids=["GPU-a"])
            for operation in [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                WorkflowOperation.VALIDATE_GPU,
            ]
        ],
    )
    reboot = _cross_domain_workflow("workflow-reboot", [WorkflowOperation.RESTART_NODE])
    read_only = {
        "efa": _cross_domain_workflow(
            "workflow-efa-read", [WorkflowOperation.VALIDATE_FABRIC]
        ),
        "cpu": _cross_domain_workflow(
            "workflow-cpu-read", [WorkflowOperation.VALIDATE_HOST]
        ),
        "memory": _cross_domain_workflow(
            "workflow-memory-read", [WorkflowOperation.RUN_FIELD_DIAGNOSTIC]
        ),
        "storage": _cross_domain_workflow(
            "workflow-storage-read", [WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE]
        ),
    }

    disposition = merger.disposition(
        copy_model(reset, status=WorkflowStatus.RUNNING),
        efa_diagnostic,
        "node-a",
        set(),
    )
    assert disposition == "PARALLEL_BRANCH"
    assert (
        merger.disposition(
            copy_model(reset, status=WorkflowStatus.RUNNING),
            dcgm_diagnostic,
            "node-a",
            {"GPU-a"},
        )
        == "QUEUE_BRANCH_SUCCESSOR"
    )
    _assert_cross_domain_parallel_dependencies(merger, reset, efa_diagnostic)
    assert merger.disposition(reboot, reset, "node-a", {"GPU-a"}) == "ABSORB"
    assert merger.disposition(reset, reboot, "node-a", set()) == "REPLACE_IN_PLACE"
    for diagnostic in read_only.values():
        assert (
            merger.disposition(
                copy_model(reset, status=WorkflowStatus.RUNNING),
                diagnostic,
                "node-a",
                set(),
            )
            == "PARALLEL_BRANCH"
        )
    assert (
        merger.disposition(
            copy_model(reset, status=WorkflowStatus.RUNNING),
            _cross_domain_workflow(
                "workflow-node-b-reboot",
                [WorkflowOperation.RESTART_NODE],
                node_id="node-b",
            ),
            "node-b",
            set(),
        )
        == "PARALLEL_BRANCH"
    )
    for operation in {
        WorkflowOperation.REMEDIATE_DRIVER,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    }:
        mutation = _cross_domain_workflow(
            f"workflow-{operation.value.lower()}",
            [operation],
            status=WorkflowStatus.RUNNING,
        )
        assert merger.disposition(
            copy_model(reset, status=WorkflowStatus.RUNNING),
            copy_model(mutation, status=WorkflowStatus.PENDING),
            "node-a",
            {"GPU-a"},
        ) not in {"PARALLEL_BRANCH", "REPLACE_IN_PLACE"}
        assert (
            merger.disposition(
                copy_model(reboot, status=WorkflowStatus.RUNNING),
                copy_model(mutation, status=WorkflowStatus.PENDING),
                "node-a",
                set(),
            )
            != "PARALLEL_BRANCH"
        )
        for diagnostic in read_only.values():
            assert (
                merger.disposition(mutation, diagnostic, "node-a", set())
                == "PARALLEL_BRANCH"
            )
    assert (
        merger.disposition(
            copy_model(reset, status=WorkflowStatus.RUNNING),
            _cross_domain_workflow(
                "workflow-mechanicals", [WorkflowOperation.CHECK_MECHANICALS]
            ),
            "node-a",
            set(),
        )
        == "PARALLEL_BRANCH"
    )


def test_stop_waits_only_for_available_live_process_capture() -> None:
    brancher = DagBrancher(RecoveryArbiter())

    def reset_workflow(*, stop_completed: bool) -> WorkflowRequest:
        workflow = workflow_request(
            "workflow-reset",
            "incident-capture",
            WorkflowStatus.RUNNING,
            1,
            runtime_profile_version="simulated-v1",
            official_action="RESET_GPU",
            official_steps=[
                workflow_step(
                    WorkflowOperation.STOP_WORKLOADS,
                    "owner",
                    workload_ids=["training/job/job-a"],
                ),
                workflow_step(
                    WorkflowOperation.RESET_GPU, "owner", gpu_uuids=["GPU-a"]
                ),
                workflow_step(
                    WorkflowOperation.RESTART_WORKLOAD,
                    "owner",
                    workload_ids=["training/job/job-a"],
                ),
            ],
        )
        if not stop_completed:
            return workflow
        return copy_model(
            workflow,
            completed_step_indexes=[0],
            completed_operations=[WorkflowOperation.STOP_WORKLOADS],
            step_executions=[
                workflow_step_execution(0, WorkflowOperation.STOP_WORKLOADS)
            ],
        )

    candidate = workflow_request(
        "workflow-live-capture",
        "incident-capture",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="RUN_DIAGNOSTICS",
        official_steps=[
            workflow_step(
                WorkflowOperation.FREEZE_EVIDENCE, "owner", node_ids=["node-b"]
            ),
            workflow_step(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                "owner",
                node_ids=["node-b"],
                parameters={"capture_process_state": True},
            ),
            workflow_step(
                WorkflowOperation.VALIDATE_HOST, "owner", node_ids=["node-b"]
            ),
        ],
    )

    before_stop = brancher.append_parallel_job_branch(
        reset_workflow(stop_completed=False), candidate
    )
    stop_index = next(
        index
        for index, step in enumerate(before_stop.official_steps)
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    bundle_index = next(
        index
        for index, step in enumerate(before_stop.official_steps)
        if step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
    )
    assert (
        bundle_index in before_stop.official_steps[stop_index].depends_on_step_indexes
    )

    after_stop = brancher.append_parallel_job_branch(
        reset_workflow(stop_completed=True), candidate
    )
    bundle = next(
        step
        for step in after_stop.official_steps
        if step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
    )
    assert bundle.parameters["capture_process_state"] is False
    assert bundle.parameters["running_process_evidence_unavailable"]
    assert (
        bundle.parameters["running_process_evidence_reason"]
        == "shared STOP_WORKLOADS already completed"
    )
    assert any(
        step.operation is WorkflowOperation.FREEZE_EVIDENCE
        and step.branch_id == bundle.branch_id
        for step in after_stop.official_steps
    ), (
        "expected any( step.operation is WorkflowOperation.FREEZE_EVIDENCE and step.branch_id == bundle.branch_id for step in after_stop.official_steps ) to be truthy"
    )


@pytest.mark.parametrize("closed_by", ["stop", "deadline"])
def test_read_only_branch_after_join_window_queues_successor(closed_by) -> None:
    merger = _workflow_merge_service()
    now = datetime.now(timezone.utc)
    existing = workflow_request(
        "workflow-restart-generation",
        "incident-restart-generation",
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action="RESTART_APP",
        aggregation_max_deadline=now - timedelta(seconds=1)
        if closed_by == "deadline"
        else now + timedelta(minutes=1),
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, "owner"),
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "owner",
                workload_ids=["training/job/job-a"],
            ),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                "owner",
                workload_ids=["training/job/job-a"],
            ),
        ],
        completed_step_indexes=[0, 1] if closed_by == "stop" else [],
        completed_operations=[
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.STOP_WORKLOADS,
        ]
        if closed_by == "stop"
        else [],
    )
    candidate = workflow_request(
        "workflow-late-diagnostic",
        existing.incident_id,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="RUN_DIAGNOSTICS",
        official_steps=[
            workflow_step(
                WorkflowOperation.FREEZE_EVIDENCE, "owner", node_ids=["node-b"]
            ),
            workflow_step(
                WorkflowOperation.VALIDATE_FABRIC, "owner", node_ids=["node-b"]
            ),
        ],
    )

    disposition = merger.disposition(existing, candidate, "node-b", set())

    assert disposition == "QUEUE_SUCCESSOR"


def test_quarantine_branch_suppresses_shared_restart_and_readmission() -> None:
    merger = _workflow_merge_service()
    existing = workflow_request(
        "workflow-reset-before-quarantine",
        "incident-quarantine",
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action="RESET_GPU",
        official_steps=[
            workflow_step(operation, "simulated-runtime", gpu_uuids=["GPU-a"])
            for operation in [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.STOP_WORKLOADS,
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTORE_SCHEDULING,
                WorkflowOperation.RESTART_WORKLOAD,
            ]
        ],
        step_executions=[
            workflow_step_execution(
                3,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/reset-in-flight",
            )
        ],
    )
    candidate = workflow_request(
        "workflow-storage-quarantine",
        "incident-quarantine",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="QUARANTINE",
        official_steps=[
            workflow_step(operation, "simulated-runtime")
            for operation in [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.STOP_WORKLOADS,
                WorkflowOperation.QUARANTINE,
            ]
        ],
    )

    disposition = merger.disposition(existing, candidate, "node-a", set())
    assert disposition == "QUEUE_BRANCH_SUCCESSOR"
    combined = merger.brancher.append_parallel_job_branch_successor(
        existing, candidate, "node-a"
    )
    restart_index = next(
        index
        for index, step in enumerate(combined.official_steps)
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    restore_index = next(
        index
        for index, step in enumerate(combined.official_steps)
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING
    )
    assert restart_index in combined.superseded_step_indexes
    assert restore_index in combined.superseded_step_indexes
    assert 3 not in combined.superseded_step_indexes
    assert combined.step_executions == existing.step_executions
    assert (
        sum(
            step.operation is WorkflowOperation.QUARANTINE
            for step in combined.official_steps
        )
        == 1
    )


def test_quarantine_after_restart_submission_queues_new_generation() -> None:
    merger = _workflow_merge_service()
    operations = [
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    existing = workflow_request(
        "workflow-restart-submitted",
        "incident-quarantine-next-generation",
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action="RESET_GPU",
        dag_enabled=True,
        official_steps=[
            workflow_step(
                operation, "simulated-runtime", workload_ids=["training/job/job-a"]
            )
            for operation in operations
        ],
        step_executions=[
            workflow_step_execution(
                3,
                WorkflowOperation.RESTART_WORKLOAD,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/restart-submitted",
            )
        ],
    )
    candidate = workflow_request(
        "workflow-quarantine-next-generation",
        existing.incident_id,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action="QUARANTINE",
        official_steps=[
            workflow_step(WorkflowOperation.QUARANTINE, "simulated-runtime")
        ],
    )
    before = existing.model_dump(mode="json")

    disposition = merger.disposition(existing, candidate, "node-a", set())
    queued = merger.prepare_preempting_successor(
        existing, copy_model(candidate, predecessor_workflow_id=existing.request_id)
    )

    assert disposition == "QUEUE_SUCCESSOR"
    assert existing.model_dump(mode="json") == before
    assert 3 not in existing.superseded_step_indexes
    assert existing.step_executions[0].status is (WorkflowStepStatus.WAITING)
    assert queued.request_id != existing.request_id
    assert queued.predecessor_workflow_id == existing.request_id
    assert queued.preempt_predecessor, (
        "expected queued.preempt_predecessor to be truthy"
    )


def test_node_merge_is_scoped_to_one_node_and_skips_operator_actions(
    context: ApplicationContext,
) -> None:
    """Merging must not blur separate nodes or human decisions.

    A fault on another node shares no blast radius, and actions that
    only ever run attended (ESCALATE_OPERATOR here) carry no
    interrupted-midpoint risk while folding them in would collapse
    faults needing separate judgement into one ticket.
    """
    ingest(context, _node_event(48, event_id="merge-node-a", gpu_uuid="GPU-a"))
    _, other_node, _ = ingest(
        context,
        _node_event(79, event_id="merge-node-b", gpu_uuid="GPU-b", node_id="node-b"),
    )
    escalating_decision, escalating_incident, _ = ingest(
        context, _node_event(74, event_id="merge-escalate", gpu_uuid="GPU-c")
    )

    assert other_node.node_ids == ["node-b"]
    assert other_node.official_action == "RESTART_BM"
    assert escalating_decision.action is RecoveryAction.STOP_WORKLOAD
    assert escalating_incident.gpu_uuids == ["GPU-c"]
    assert len(context.store.list_workflows(limit=10)) == 3


def test_merged_reset_targets_every_faulted_gpu(context: ApplicationContext) -> None:
    """A merged reset must reset both cards, not just the first.

    Merging used to widen only ``incident.gpu_uuids`` while each step
    kept the compiling event's single GPU. The workflow then claimed two
    faulty cards, reset one, validated the node, and readmitted it --
    handing an unreset uncorrectable-ECC GPU back to a workload.

    Node agents accept the wider target only via
    ``parameters.gpu_uuids_by_node``, so the mapping must be present
    alongside the widened ``gpu_uuids``.
    """
    ingest(context, _node_event(48, event_id="reset-gpu-a", gpu_uuid="GPU-a"))
    ingest(context, _node_event(48, event_id="reset-gpu-b", gpu_uuid="GPU-b"))
    _, incident, workflow = ingest(
        context, _node_event(48, event_id="reset-gpu-c", gpu_uuid="GPU-c")
    )

    assert incident.official_action == "RESET_GPU"
    assert incident.gpu_uuids == ["GPU-a", "GPU-b", "GPU-c"]
    reset_steps = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    ]
    assert len(reset_steps) == 1
    assert reset_steps[0].gpu_uuids == ["GPU-a", "GPU-b", "GPU-c"]
    assert reset_steps[0].parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-a", "GPU-b", "GPU-c"]
    }
    # Quiesce and its restore bracket the reset, so a narrow quiesce
    # would leave one card's clients attached during the reset.
    for operation in (
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ):
        step = next(
            item for item in workflow.official_steps if item.operation is operation
        )
        assert step.gpu_uuids == ["GPU-a", "GPU-b", "GPU-c"]


def test_merge_stops_once_the_reset_step_already_ran(
    context: ApplicationContext,
) -> None:
    """A widened step that already executed remediates nothing.

    The executor skips indexes in ``completed_step_indexes``, so folding
    a second card into a workflow whose RESET_GPU already ran produced an
    incident and a step both claiming a GPU that was never touched --
    and the workflow went on to validate and uncordon it. The second
    fault has to get its own workflow, chained behind the incumbent so
    the two never own the host at once.
    """
    _, _, running = ingest(
        context, _node_event(48, event_id="ran-gpu-a", gpu_uuid="GPU-a")
    )
    reset_index = next(
        index
        for index, step in enumerate(running.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    context.store.save_workflow(
        copy_model(
            running,
            status=WorkflowStatus.RUNNING,
            not_before=None,
            execution_owner_id="executor-1",
            completed_step_indexes=list(range(reset_index + 1)),
        )
    )

    _, _, second = ingest(
        context, _node_event(48, event_id="ran-gpu-b", gpu_uuid="GPU-b")
    )

    assert second.request_id != running.request_id
    assert second.status is WorkflowStatus.PENDING
    assert second.predecessor_workflow_id == running.request_id
    assert not second.completed_step_indexes, (
        "expected second.completed_step_indexes to be falsy"
    )
    reset = next(
        step
        for step in second.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    )
    assert reset.gpu_uuids == ["GPU-b"]
    completed_reset = running.official_steps[reset_index]
    assert completed_reset.gpu_uuids == ["GPU-a"]


def test_node_merge_chains_behind_a_workflow_already_running(
    context: ApplicationContext,
) -> None:
    """A late stronger fault must queue, never race the running one.

    Rewriting a workflow in place is only safe while nothing has
    claimed or partially executed it. Once steps have run, escalation
    has to chain behind instead -- which is also what catches a fault
    arriving long after the aggregation window closed.
    """
    _, _, running = ingest(
        context, _node_event(48, event_id="running-first", gpu_uuid="GPU-a")
    )
    context.store.save_workflow(
        copy_model(
            running,
            status=WorkflowStatus.RUNNING,
            execution_owner_id="executor-1",
            completed_step_indexes=[0, 1],
        )
    )

    _, incident, queued = ingest(
        context, _node_event(79, event_id="running-second", gpu_uuid="GPU-b")
    )

    assert queued.request_id != running.request_id
    assert queued.predecessor_workflow_id == running.request_id
    assert queued.not_before is None
    assert incident.incident_id == running.incident_id
    assert incident.official_action == "RESTART_BM"


def test_attempt_scoped_merge_remediates_every_faulted_node(
    context: ApplicationContext,
) -> None:
    """Two ranks losing a GPU must both be reset, cordoned, validated.

    The attempt-scoped merge folded both faults into one workflow but
    left every step naming the first event's node and GPU. node-b's card
    was never reset, its node never cordoned and never validated, and
    RESTART_WORKLOAD then handed the workload straight back onto it.

    The node-replacement grouped path already widened its steps this
    way; the XID path did not.
    """
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "job-a-a001",
            NOW,
            expected_critical_ranks=2,
            containers=[
                container_observation(
                    "pod-0", "worker-0", 0, "node-a", gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-1", "worker-1", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
            ],
            workload_ids=["training/job/job-a"],
            restart_budget=2,
        )
    )

    def fault(event_id: str, gpu_uuid: str, node_id: str) -> XidEvent:
        return copy_model(
            event(
                48,
                event_id=event_id,
                workload_state=WorkloadState.ACTIVE,
                affected_workload_ids=["training/job/job-a"],
            ),
            gpu_uuid=gpu_uuid,
            node_id=node_id,
            job_id="job-a",
            attempt_id="job-a-a001",
        )

    ingest(context, fault("attempt-gpu-a", "GPU-a", "node-a"))
    _, incident, workflow = ingest(context, fault("attempt-gpu-b", "GPU-b", "node-b"))

    assert len(context.store.list_workflows(limit=10)) == 1
    assert incident.node_ids == ["node-a", "node-b"]
    assert incident.gpu_uuids == ["GPU-a", "GPU-b"]
    resets = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    ]
    assert len(resets) == 2
    assert {tuple(step.node_ids): tuple(step.gpu_uuids) for step in resets} == {
        ("node-a",): ("GPU-a",),
        ("node-b",): ("GPU-b",),
    }
    for operation in (
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.RESTORE_SCHEDULING,
    ):
        steps = [
            item for item in workflow.official_steps if item.operation is operation
        ]
        assert {tuple(step.node_ids) for step in steps} == {("node-a",), ("node-b",)}
    assert (
        sum(
            step.operation is WorkflowOperation.STOP_WORKLOADS
            for step in workflow.official_steps
        )
        == 1
    )
    restart = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert restart.parameters["source_attempt_id"] == "job-a-a001"
    assert restart.node_ids == ["node-a", "node-b"]
    assert len(restart.depends_on_step_indexes) == 2
