from __future__ import annotations

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    processor_request,
    workflow_step_execution,
)
from tests.execution._support import (
    _assert_chained_preemption_result,
    _FleetPreflightRegistry,
    _run_chained_preemptions,
)

from ._support import (
    FakeAdapter,
    IncidentState,
    ProcessorRequestStatus,
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    RemoteActionCommand,
    RemoteCommandStatus,
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    _preempting_successor,
    datetime,
    executor,
    pytest,
    timedelta,
    timezone,
    workflow_state,
)


def test_fleet_preflight_holds_before_destructive_step() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    incident, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.succeeded() for operation in operations}
    )
    registry = _FleetPreflightRegistry(ready=False)
    active = active_workflow_executor(store, [adapter], operations)
    active.fleet_registry = registry

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.PENDING
    assert "blocked destructive workflow" in result.error
    assert adapter.calls == []
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.PENDING
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.ACTION_PENDING
    )
    assert registry.calls == [("cluster-a", ["node-a"])]


def test_fleet_preflight_retries_after_pin_alignment() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    _, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.succeeded() for operation in operations}
    )
    registry = _FleetPreflightRegistry(ready=False)
    active = active_workflow_executor(store, [adapter], operations)
    active.fleet_registry = registry
    first = execute_workflow(active, workflow.request_id)

    registry.ready = True
    second = execute_workflow(active, workflow.request_id)

    assert first.status is WorkflowStatus.PENDING
    assert second.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [
        "workflow-active/0/MARK_UNSCHEDULABLE",
        "workflow-active/1/COLLECT_DIAGNOSTIC_BUNDLE",
    ]


def test_fleet_preflight_does_not_withdraw_containment_only_plan() -> None:
    store = build_store()
    operations = [WorkflowOperation.MARK_UNSCHEDULABLE]
    _, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {WorkflowOperation.MARK_UNSCHEDULABLE: (WorkflowStepOutcome.succeeded())}
    )
    registry = _FleetPreflightRegistry(ready=False)
    active = active_workflow_executor(store, [adapter], operations)
    active.fleet_registry = registry

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert registry.calls == []
    assert adapter.calls == ["workflow-active/0/MARK_UNSCHEDULABLE"]


def test_executor_supersedes_at_clean_step_boundary() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
    ]
    incident, workflow = workflow_state(store, operations)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.MARK_UNSCHEDULABLE)
        ],
    )
    store.save_workflow(workflow)
    successor = _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {WorkflowOperation.QUIESCE_GPU_SERVICES: (WorkflowStepOutcome.succeeded())}
    )
    active = active_workflow_executor(store, [adapter], operations)

    result = execute_workflow(active, workflow.request_id)

    updated = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.SUPERSEDED
    assert updated.preempted_by_workflow_id == successor.request_id
    assert updated.superseded_at is not None
    assert adapter.calls == []


def test_executor_restores_when_successor_cannot_take_quiesce_handoff() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
    ]
    incident, workflow = workflow_state(store, operations)
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.QUIESCE_GPU_SERVICES)
        ],
    )
    store.save_workflow(workflow)
    _preempting_successor(store, incident, workflow, WorkflowOperation.QUARANTINE)
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESET_GPU: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESTART_NODE: (
                WorkflowStepOutcome.waiting(operation_id="reboot-submitted")
            ),
            WorkflowOperation.RESTORE_GPU_SERVICES: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.VALIDATE_GPU: (WorkflowStepOutcome.succeeded()),
        }
    )
    active = active_workflow_executor(
        store, [adapter], [*operations, WorkflowOperation.RESTART_NODE]
    )

    result = execute_workflow(active, workflow.request_id)

    updated = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.SUPERSEDED
    assert updated.completed_step_indexes == [0, 2]
    assert adapter.calls == ["workflow-active/2/RESTORE_GPU_SERVICES"]


def test_preemption_config_disabled_keeps_predecessor_running() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.FREEZE_EVIDENCE: (
                WorkflowStepOutcome.waiting(operation_id="evidence/wait")
            )
        }
    )
    active = active_workflow_executor(
        store,
        [adapter],
        {WorkflowOperation.FREEZE_EVIDENCE},
        workflow_preemption_enabled=False,
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    assert adapter.calls == ["workflow-active/0/FREEZE_EVIDENCE"]


def test_dispatcher_releases_successor_after_predecessor_superseded() -> None:
    store = build_store()
    incident, predecessor = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    predecessor = copy_model(predecessor, status=WorkflowStatus.SUPERSEDED)
    store.save_workflow(predecessor)
    successor = _preempting_successor(
        store, incident, predecessor, WorkflowOperation.RESTART_NODE
    )
    adapter = FakeAdapter(
        {WorkflowOperation.RESTART_NODE: (WorkflowStepOutcome.succeeded())}
    )
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], {WorkflowOperation.RESTART_NODE}),
        WorkflowDispatcherConfig(enabled=True),
    )

    report = dispatcher.run_once()

    assert report.completed == 1
    assert store.get_workflow(successor.request_id).status is WorkflowStatus.SUCCEEDED


@pytest.mark.parametrize("chain_length", [3, 5])
def test_dispatcher_resolves_chained_preemptions_to_highest_action(
    chain_length: int,
) -> None:
    result = _run_chained_preemptions(chain_length, incremental_arrival=False)

    _assert_chained_preemption_result(result, chain_length)


def test_five_level_preemption_is_arrival_order_independent() -> None:
    preregistered = _run_chained_preemptions(5, incremental_arrival=False)
    incremental = _run_chained_preemptions(5, incremental_arrival=True)

    _assert_chained_preemption_result(preregistered, 5)
    _assert_chained_preemption_result(incremental, 5)
    assert [
        preregistered[0].get_workflow(item.request_id).status
        for item in preregistered[2]
    ] == [
        incremental[0].get_workflow(item.request_id).status for item in incremental[2]
    ]
    assert preregistered[4].calls == incremental[4].calls


def test_dag_fans_out_node_branches_and_joins_once() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.RESTART_WORKLOAD,
        ],
    )
    steps = [
        copy_model(workflow.official_steps[0], branch_id="shared"),
        copy_model(
            workflow.official_steps[1],
            node_ids=["node-a"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-a",
        ),
        copy_model(
            workflow.official_steps[2],
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[3], depends_on_step_indexes=[1, 2], branch_id="join"
        ),
    ]
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.STOP_WORKLOADS],
        step_executions=[workflow_step_execution(0, WorkflowOperation.STOP_WORKLOADS)],
    )
    store.save_workflow(workflow)
    waiting_adapter = FakeAdapter(
        {
            WorkflowOperation.RESET_GPU: (
                WorkflowStepOutcome.waiting(operation_id="remote/reset-a")
            ),
            WorkflowOperation.RESTART_NODE: (
                WorkflowStepOutcome.waiting(operation_id="remote/reboot-b")
            ),
            WorkflowOperation.RESTART_WORKLOAD: (WorkflowStepOutcome.succeeded()),
        }
    )
    config = ProductionExecutorConfig(
        enabled=True,
        executor_id="executor-a",
        allowed_operations=frozenset(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTART_NODE,
                WorkflowOperation.RESTART_WORKLOAD,
            }
        ),
    )

    first = execute_workflow(
        ProductionWorkflowExecutor(store, [waiting_adapter], config),
        workflow.request_id,
    )

    assert first.status is WorkflowStatus.RUNNING
    assert waiting_adapter.calls == [
        "workflow-active/1/RESET_GPU",
        "workflow-active/2/RESTART_NODE",
    ]
    assert all(
        execution.status is WorkflowStepStatus.WAITING
        for execution in store.get_workflow(workflow.request_id).step_executions
        if execution.step_index in {1, 2}
    )
    assert all(
        execution.step_index != 3
        for execution in store.get_workflow(workflow.request_id).step_executions
    )

    success_adapter = FakeAdapter(
        {
            operation: WorkflowStepOutcome.succeeded()
            for operation in {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTART_NODE,
                WorkflowOperation.RESTART_WORKLOAD,
            }
        }
    )
    second = execute_workflow(
        ProductionWorkflowExecutor(store, [success_adapter], config),
        workflow.request_id,
    )

    assert second.status is WorkflowStatus.SUCCEEDED
    assert success_adapter.calls[:2] == [
        "workflow-active/1/RESET_GPU",
        "workflow-active/2/RESTART_NODE",
    ]
    assert success_adapter.calls[-1] == ("workflow-active/3/RESTART_WORKLOAD")
    assert success_adapter.calls.count("workflow-active/3/RESTART_WORKLOAD") == 1


def test_dag_branch_failure_still_dispatches_other_ready_branch() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.RESTART_WORKLOAD,
        ],
    )
    steps = [
        workflow.official_steps[0],
        copy_model(workflow.official_steps[1], depends_on_step_indexes=[0]),
        copy_model(workflow.official_steps[2], depends_on_step_indexes=[0]),
        copy_model(workflow.official_steps[3], depends_on_step_indexes=[1, 2]),
    ]
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.STOP_WORKLOADS],
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESET_GPU: (WorkflowStepOutcome.failed("reset failed")),
            WorkflowOperation.RESTART_NODE: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESTART_WORKLOAD: (WorkflowStepOutcome.succeeded()),
        }
    )

    active = active_workflow_executor(
        store,
        [adapter],
        {
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.RESTART_WORKLOAD,
        },
    )
    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    # A sibling node's repair still runs when another branch fails: the
    # per-node escalation planned as F-N1 builds on exactly this.
    assert adapter.calls == [
        "workflow-active/1/RESET_GPU",
        "workflow-active/2/RESTART_NODE",
    ]
    assert all("RESTART_WORKLOAD" not in call for call in adapter.calls)
    saved = store.get_workflow(workflow.request_id)
    executions = {
        execution.step_index: execution for execution in saved.step_executions
    }
    assert executions[1].status is WorkflowStepStatus.FAILED
    assert executions[2].status is WorkflowStepStatus.SUCCEEDED
    assert 3 not in executions
    assert saved.completed_step_indexes == [0, 2]
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED

    repeated = execute_workflow(active, workflow.request_id)

    assert repeated.status is WorkflowStatus.FAILED
    assert adapter.calls == [
        "workflow-active/1/RESET_GPU",
        "workflow-active/2/RESTART_NODE",
    ]


def test_terminal_quarantine_dag_finishes_without_shared_restart() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.QUARANTINE,
            WorkflowOperation.RESTORE_SCHEDULING,
            WorkflowOperation.RESTART_WORKLOAD,
        ],
    )
    steps = [
        workflow.official_steps[0],
        copy_model(workflow.official_steps[1], depends_on_step_indexes=[0]),
        copy_model(workflow.official_steps[2], depends_on_step_indexes=[0]),
        copy_model(workflow.official_steps[3], depends_on_step_indexes=[0]),
    ]
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        dag_enabled=True,
        official_steps=steps,
        superseded_step_indexes=[2, 3],
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {
            operation: WorkflowStepOutcome.succeeded()
            for operation in {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.QUARANTINE,
                WorkflowOperation.RESTORE_SCHEDULING,
                WorkflowOperation.RESTART_WORKLOAD,
            }
        }
    )

    result = execute_workflow(
        active_workflow_executor(store, [adapter], adapter.outcomes),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED
    assert adapter.calls == [
        "workflow-active/0/RESET_GPU",
        "workflow-active/1/QUARANTINE",
    ]


def test_dag_cycle_is_rejected_before_execution() -> None:
    store = build_store()
    _, workflow = workflow_state(
        store, [WorkflowOperation.RESET_GPU, WorkflowOperation.RESTART_NODE]
    )
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        official_steps=[
            copy_model(workflow.official_steps[0], depends_on_step_indexes=[1]),
            copy_model(workflow.official_steps[1], depends_on_step_indexes=[0]),
        ],
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.RESET_GPU: WorkflowStepOutcome.succeeded(),
            WorkflowOperation.RESTART_NODE: WorkflowStepOutcome.succeeded(),
        }
    )

    with pytest.raises(WorkflowExecutionError, match="contains a cycle"):
        execute_workflow(
            active_workflow_executor(
                store,
                [adapter],
                {WorkflowOperation.RESET_GPU, WorkflowOperation.RESTART_NODE},
            ),
            workflow.request_id,
        )

    assert adapter.calls == []


def test_active_executor_completes_and_is_idempotent() -> None:
    store = build_store()
    operations = [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
    ]
    _, workflow = workflow_state(store, operations)
    adapter = FakeAdapter(
        {
            operation: WorkflowStepOutcome.succeeded(operation_id=operation.value)
            for operation in operations
        }
    )
    active = executor(store, adapter, operations)
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    first = active.execute(workflow.request_id, request)
    second = active.execute(workflow.request_id, request)

    assert first.status is WorkflowStatus.SUCCEEDED
    assert not first.simulation_only
    assert second.status is WorkflowStatus.SUCCEEDED
    assert len(adapter.calls) == 2
    assert store.get_incident("incident-active").state is IncidentState.RECOVERED


def test_active_executor_waits_and_resumes_external_operation() -> None:
    store = build_store()
    operation = WorkflowOperation.RESTART_NODE
    _, workflow = workflow_state(store, [operation])
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.waiting(operation_id="provider-op-1")}
    )
    active = executor(store, adapter, [operation])

    waiting = execute_workflow(active, workflow.request_id)
    completed = execute_workflow(
        active, workflow.request_id, confirmed_adapter_operation_ids=["provider-op-1"]
    )

    assert waiting.status is WorkflowStatus.RUNNING
    assert waiting.waiting_step_index == 0
    assert completed.status is WorkflowStatus.SUCCEEDED
    assert store.get_workflow(workflow.request_id).completed_step_indexes == [0]


def test_dispatcher_executes_durable_pending_workflow() -> None:
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, workflow = workflow_state(store, [operation])
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
    )

    report = dispatcher.run_once()

    assert report.scanned == 1
    assert report.completed == 1
    assert not report.failures
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.SUCCEEDED


def test_dispatcher_internal_error_blocks_workflow_without_hot_loop() -> None:
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    incident, workflow = workflow_state(store, [operation])

    class BrokenAdapter:
        owner = "simulated-runtime"

        def __init__(self) -> None:
            self.calls = 0

        def supports(self, _step) -> bool:
            self.calls += 1
            raise AttributeError("adapter wiring is broken")

        def execute(self, _context):
            raise AssertionError("execute must not be reached")

    adapter = BrokenAdapter()
    active = active_workflow_executor(store, [adapter], {operation})
    dispatcher = WorkflowDispatcher(
        store, active, WorkflowDispatcherConfig(enabled=True)
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    current = store.get_workflow(workflow.request_id)
    # F-B4 (3): an unrecognised internal error says nothing about the record,
    # so it stays executable and is retried instead of being written BLOCKED.
    assert first.internal_errors == 1
    assert first.failed == 0
    assert second.scanned == 0
    assert adapter.calls == 1
    assert current.status is WorkflowStatus.PENDING
    assert current.execution_owner_id is None
    assert current.not_before is not None
    assert current.blocked_reasons == []
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.ACTION_PENDING
    )


def test_dag_rebind_persists_incident_before_later_step_waits() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.REPLACE_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    steps = [
        workflow.official_steps[0],
        copy_model(workflow.official_steps[1], depends_on_step_indexes=[0]),
    ]
    workflow = copy_model(
        workflow, official_steps=steps, dag_enabled=True, dag_revision=1
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.REPLACE_NODE: (
                WorkflowStepOutcome.succeeded(
                    details={"node_rebindings": {"node-a": "node-spare"}}
                )
            ),
            WorkflowOperation.VALIDATE_GPU: (
                WorkflowStepOutcome.waiting(operation_id="validate-spare")
            ),
        }
    )
    active = active_workflow_executor(
        store,
        [adapter],
        {WorkflowOperation.REPLACE_NODE, WorkflowOperation.VALIDATE_GPU},
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    assert store.get_incident(incident.incident_id).node_ids == ["node-spare"]
    assert store.get_workflow(workflow.request_id).official_steps[1].node_ids == [
        "node-spare"
    ]


def test_dispatcher_skips_workflow_until_aggregation_window_expires():
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, workflow = workflow_state(store, [operation])
    store.save_workflow(
        copy_model(
            workflow, not_before=datetime.now(timezone.utc) + timedelta(minutes=1)
        )
    )
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
    )

    report = dispatcher.run_once()

    assert report.scanned == 0
    assert report.executed == 0
    assert adapter.calls == []
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.PENDING


def test_dispatcher_waits_for_same_node_processor_backlog():
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, workflow = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    store.save_workflow(
        copy_model(
            workflow,
            not_before=now - timedelta(seconds=1),
            aggregation_max_deadline=now + timedelta(seconds=30),
        )
    )
    store.enqueue_processor_request(
        processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a"}')
    )
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
    )

    report = dispatcher.run_once()

    assert report.scanned == 0
    assert adapter.calls == []


@pytest.mark.parametrize(
    ("cluster_id", "status", "maximum_offset"),
    [
        ("cluster-b", ProcessorRequestStatus.PENDING, 30),
        ("cluster-a", ProcessorRequestStatus.COMPLETED, 30),
        ("cluster-a", ProcessorRequestStatus.PENDING, -1),
    ],
)
def test_dispatcher_backlog_gate_is_scoped_and_bounded(
    cluster_id, status, maximum_offset
):
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, workflow = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    store.save_workflow(
        copy_model(
            workflow,
            not_before=now - timedelta(seconds=2),
            aggregation_max_deadline=now + timedelta(seconds=maximum_offset),
        )
    )
    request = copy_model(
        processor_request("/v1/gpu-events/xid", cluster_id=cluster_id), status=status
    )
    store.enqueue_processor_request(request)
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
    )

    report = dispatcher.run_once()

    assert report.scanned == 1
    assert report.completed == 1
    assert adapter.calls


def test_dispatcher_waits_for_predecessor_terminal_state() -> None:
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, predecessor = workflow_state(store, [operation])
    successor = copy_model(
        predecessor,
        request_id="workflow-successor",
        predecessor_workflow_id=predecessor.request_id,
        created_at=predecessor.created_at + timedelta(seconds=1),
        updated_at=predecessor.updated_at + timedelta(seconds=1),
    )
    store.save_workflow(successor)
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    assert first.scanned == 1
    assert first.completed == 1
    assert second.scanned == 1
    assert second.completed == 1
    assert store.get_workflow(predecessor.request_id).status is (
        WorkflowStatus.SUCCEEDED
    )
    assert store.get_workflow(successor.request_id).status is (WorkflowStatus.SUCCEEDED)


def test_dispatcher_expires_stuck_predecessor_and_runs_successor() -> None:
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, predecessor = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    predecessor = copy_model(
        predecessor,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="dead-executor",
        execution_epoch=1,
        execution_lease_expires_at=now - timedelta(seconds=1),
        execution_deadline=now - timedelta(seconds=1),
    )
    successor = copy_model(
        predecessor,
        request_id="workflow-successor",
        predecessor_workflow_id=predecessor.request_id,
        status=WorkflowStatus.PENDING,
        execution_owner_id=None,
        execution_epoch=0,
        execution_lease_expires_at=None,
        execution_deadline=None,
        created_at=now,
        updated_at=now,
    )
    store.save_workflow(predecessor)
    store.save_workflow(successor)
    incident = store.get_incident(predecessor.incident_id)
    for suffix in ("leased", "pending"):
        store.ensure_remote_command(
            RemoteActionCommand(
                command_id=f"remote-{suffix}",
                cluster_id=incident.cluster_id,
                workflow_request_id=predecessor.request_id,
                incident_id=incident.incident_id,
                step_index=0,
                fencing_token=predecessor.fencing_token,
                idempotency_key=f"remote/{suffix}",
                step=predecessor.official_steps[0],
                workflow=predecessor,
                incident=incident,
            )
        )
    leased = store.claim_remote_commands(
        incident.cluster_id, "remote-executor", limit=1, lease_seconds=60
    )[0]
    assert leased.command_id == "remote-leased"
    handled = []
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
        failure_handler=handled.append,
    )

    report = dispatcher.run_once()

    assert report.failed == 1
    assert report.completed == 1
    assert store.get_workflow(predecessor.request_id).status is (WorkflowStatus.FAILED)
    assert store.get_workflow(successor.request_id).status is (WorkflowStatus.SUCCEEDED)
    assert handled[0].request_id == predecessor.request_id
    assert (
        store.get_remote_command("remote-pending").status is RemoteCommandStatus.FAILED
    )
    leased_after = store.get_remote_command("remote-leased")
    assert leased_after.status is RemoteCommandStatus.LEASED
    assert leased_after.cancellation_requested_at is not None
    assert "deadline exceeded" in leased_after.cancellation_reason
