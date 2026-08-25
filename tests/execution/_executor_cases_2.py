from __future__ import annotations

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
)

from ._support import (
    FakeAdapter,
    IncidentState,
    InMemoryStore,
    ProductionExecutorConfig,
    SqliteStore,
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    WorkflowLeaseError,
    WorkflowOperation,
    WorkflowRequest,
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


def test_dispatcher_reconciles_previously_failed_workflow() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_GPU])
    failed = copy_model(workflow, status=WorkflowStatus.FAILED)
    store.save_workflow(failed)
    handled = []
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True),
        failure_handler=handled.append,
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    assert first.scanned == 0
    assert second.scanned == 0
    assert handled == [failed]
    assert store.get_workflow(failed.request_id).failure_handled_at is not None


def test_dispatcher_background_loop_executes_and_stops() -> None:
    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, workflow = workflow_state(store, [operation])
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True, poll_interval_seconds=0.01),
    )

    import time
    from threading import Thread

    worker = Thread(target=dispatcher.run_forever)
    worker.start()
    for _ in range(50):
        if store.get_workflow(workflow.request_id).status is WorkflowStatus.SUCCEEDED:
            break
        time.sleep(0.01)
    dispatcher.stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.SUCCEEDED


def test_dispatcher_background_loop_survives_transient_store_error() -> None:
    store = build_store()
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True, poll_interval_seconds=0.001),
    )
    calls = []

    def flaky_run_once():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise RuntimeError("Aurora writer changed")
        dispatcher.stop()

    dispatcher.run_once = flaky_run_once
    dispatcher.run_forever()

    assert calls == [1, 2]


def test_transient_store_error_does_not_block_active_workflow() -> None:
    class OperationalError(Exception):
        pass

    OperationalError.__module__ = "psycopg"
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    adapter = FakeAdapter(
        {WorkflowOperation.FREEZE_EVIDENCE: (WorkflowStepOutcome.succeeded())}
    )
    active = executor(store, adapter, [WorkflowOperation.FREEZE_EVIDENCE])
    real_execute = active.execute
    attempts = 0

    def flaky_execute(request_id, request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            claimed = store.claim_workflow(
                request_id,
                active.config.executor_id,
                request.expected_fencing_token,
                lease_duration=timedelta(seconds=30),
            )
            store.save_workflow_if_leased(
                copy_model(claimed, status=WorkflowStatus.RUNNING),
                active.config.executor_id,
                claimed.execution_epoch,
            )
            raise OperationalError("SSL error: unexpected eof while reading")
        return real_execute(request_id, request)

    active.execute = flaky_execute
    dispatcher = WorkflowDispatcher(
        store, active, WorkflowDispatcherConfig(enabled=True)
    )

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    current = store.get_workflow(workflow.request_id)
    assert first.waiting == 1
    assert first.failed == 0
    assert second.completed == 1
    assert current.status is WorkflowStatus.SUCCEEDED
    assert current.blocked_reasons == []
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED


def test_preemption_survives_store_failover_and_lease_takeover() -> None:
    class OperationalError(Exception):
        pass

    OperationalError.__module__ = "psycopg"
    store = build_store()
    incident, predecessor = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    successor = _preempting_successor(
        store, incident, predecessor, WorkflowOperation.RESTART_NODE
    )
    adapter = FakeAdapter(
        {
            WorkflowOperation.FREEZE_EVIDENCE: (WorkflowStepOutcome.succeeded()),
            WorkflowOperation.RESTART_NODE: (WorkflowStepOutcome.succeeded()),
        }
    )
    executor_a = active_workflow_executor(store, [adapter], adapter.outcomes)
    executor_b = active_workflow_executor(
        store, [adapter], adapter.outcomes, executor_id="executor-b"
    )
    real_execute_a = executor_a.execute
    failed_at = datetime.now(timezone.utc)
    claimed_a: WorkflowRequest | None = None

    def fail_during_predecessor_dispatch(request_id, request):
        nonlocal claimed_a
        assert request_id == predecessor.request_id
        claimed_a = store.claim_workflow(
            request_id,
            executor_a.config.executor_id,
            request.expected_fencing_token,
            now=failed_at,
            lease_duration=timedelta(seconds=30),
        )
        store.save_workflow_if_leased(
            copy_model(claimed_a, status=WorkflowStatus.RUNNING),
            executor_a.config.executor_id,
            claimed_a.execution_epoch,
            now=failed_at,
        )
        raise OperationalError("read-only transaction during Aurora failover")

    executor_a.execute = fail_during_predecessor_dispatch
    dispatcher_a = WorkflowDispatcher(
        store, executor_a, WorkflowDispatcherConfig(enabled=True)
    )

    interrupted = dispatcher_a.run_once()

    assert interrupted.waiting == 1
    assert interrupted.failed == 0
    assert claimed_a is not None
    assert store.get_workflow(predecessor.request_id).status is WorkflowStatus.RUNNING
    assert (
        store.get_incident(incident.incident_id).workflow_request_id
        == successor.request_id
    )

    takeover_at = failed_at + timedelta(seconds=31)
    claimed_b = store.claim_workflow(
        predecessor.request_id,
        executor_b.config.executor_id,
        predecessor.fencing_token,
        now=takeover_at,
        lease_duration=timedelta(seconds=30),
    )
    assert claimed_b.execution_epoch == claimed_a.execution_epoch + 1
    with pytest.raises(WorkflowLeaseError, match="stale"):
        store.save_workflow_if_leased(
            copy_model(claimed_a, status=WorkflowStatus.SUPERSEDED),
            executor_a.config.executor_id,
            claimed_a.execution_epoch,
            now=takeover_at,
        )

    superseded = execute_workflow(executor_b, predecessor.request_id)
    assert superseded.status is WorkflowStatus.SUPERSEDED
    assert adapter.calls == []

    dispatcher_b = WorkflowDispatcher(
        store, executor_b, WorkflowDispatcherConfig(enabled=True)
    )
    completed = dispatcher_b.run_once()
    executor_a.execute = real_execute_a
    duplicate_scan = dispatcher_a.run_once()

    assert completed.completed == 1
    assert duplicate_scan.executed == 0
    assert store.get_workflow(successor.request_id).status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == ["workflow-successor/0/RESTART_NODE"]


def test_dispatcher_wake_interrupts_poll_wait() -> None:
    store = build_store()
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True, poll_interval_seconds=60),
    )
    calls = []
    dispatcher.run_once = lambda: calls.append(time.monotonic())

    import time
    from threading import Thread

    worker = Thread(target=dispatcher.run_forever)
    worker.start()
    for _ in range(50):
        if calls:
            break
        time.sleep(0.01)
    dispatcher.wake()
    for _ in range(50):
        if len(calls) >= 2:
            break
        time.sleep(0.01)
    dispatcher.stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(calls) == 2
    assert calls[1] - calls[0] < 1


def test_hung_triage_result_and_dag_rewrite_are_saved_together() -> None:
    class RecordingStore(InMemoryStore):
        def __init__(self):
            super().__init__()
            self.leased_saves = []

        def save_workflow_if_leased(self, workflow, *args, **kwargs):
            self.leased_saves.append(workflow.model_copy(deep=True))
            return super().save_workflow_if_leased(workflow, *args, **kwargs)

    class HungAdapter:
        def supports(self, step):
            return step.execution_owner == "owner-a"

        def execute(self, context):
            if context.step.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE:
                return WorkflowStepOutcome.succeeded(
                    operation_id=context.idempotency_key,
                    details={
                        "node_results": {
                            "node-a": {
                                "ranks": [
                                    {
                                        "pid": 100,
                                        "rank": 0,
                                        "gpu_uuid": "GPU-a",
                                        "flight_recorder": {
                                            "last_entry": {
                                                "pg_name": "default",
                                                "collective_seq_id": 9,
                                                "time_discovered_started": "t",
                                            }
                                        },
                                    }
                                ]
                            },
                            "node-b": {
                                "ranks": [
                                    {
                                        "pid": 200,
                                        "rank": 1,
                                        "gpu_uuid": "GPU-b",
                                        "flight_recorder": {
                                            "last_entry": {
                                                "pg_name": "default",
                                                "collective_seq_id": 10,
                                                "time_discovered_started": "t",
                                            }
                                        },
                                    }
                                ]
                            },
                            "node-c": {
                                "ranks": [
                                    {
                                        "pid": 300,
                                        "rank": 2,
                                        "gpu_uuid": "GPU-c",
                                        "flight_recorder": {
                                            "last_entry": {
                                                "pg_name": "default",
                                                "collective_seq_id": 10,
                                                "time_discovered_started": "t",
                                            }
                                        },
                                    }
                                ]
                            },
                        },
                        "undetermined_nodes": [],
                    },
                )
            return WorkflowStepOutcome.waiting(operation_id=context.idempotency_key)

    store = RecordingStore()
    incident = fault_incident(
        "incident-hung-atomic",
        "event-hung-atomic",
        "NODE_HEALTH",
        node_ids=["node-a", "node-b", "node-c"],
        policy_version="v1",
        policy_source="SITE_EFA_TRAFFIC",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-hung-atomic",
        fencing_token=1,
    )
    steps = [
        workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=incident.node_ids),
        workflow_step(
            WorkflowOperation.COLLECT_HUNG_TRIAGE,
            node_ids=incident.node_ids,
            depends_on_step_indexes=[0],
        ),
        workflow_step(
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            node_ids=incident.node_ids,
            depends_on_step_indexes=[1],
        ),
        workflow_step(
            WorkflowOperation.VALIDATE_FABRIC,
            node_ids=incident.node_ids,
            depends_on_step_indexes=[1],
        ),
    ]
    workflow = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        WorkflowStatus.RUNNING,
        1,
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.FREEZE_EVIDENCE],
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    executor = active_workflow_executor(
        store,
        [HungAdapter()],
        {
            WorkflowOperation.COLLECT_HUNG_TRIAGE,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.VALIDATE_FABRIC,
        },
    )

    execute_workflow(executor, workflow.request_id, expected_fencing_token=1)

    atomic_save = next(
        item for item in store.leased_saves if 1 in item.completed_step_indexes
    )
    assert atomic_save.dag_revision == 2
    assert any(
        execution.step_index == 1 and execution.status is WorkflowStepStatus.SUCCEEDED
        for execution in atomic_save.step_executions
    )
    assert (
        atomic_save.official_steps[2].parameters["hung_triage_target_pending"] is False
    )


def test_sqlite_store_persists_workflow_and_atomic_claim(tmp_path) -> None:
    path = tmp_path / "control-plane.db"
    first = SqliteStore(str(path))
    _, workflow = workflow_state(first, [WorkflowOperation.RESTART_NODE])

    second = SqliteStore(str(path))
    loaded = second.get_workflow(workflow.request_id)
    claimed = second.claim_workflow(
        loaded.request_id, "executor-a", loaded.fencing_token
    )

    assert claimed.execution_owner_id == "executor-a"
    assert first.get_workflow(loaded.request_id).execution_owner_id == "executor-a"


def test_executor_identity_is_unique_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_EXECUTOR_ID", raising=False)
    monkeypatch.setenv("HOSTNAME", "control-plane-pod")

    first = ProductionExecutorConfig.from_environment()
    second = ProductionExecutorConfig.from_environment()

    assert first.executor_id != second.executor_id
    assert "control-plane-pod" in first.executor_id


def test_workflow_preemption_config_is_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION", raising=False)
    enabled_by_default = ProductionExecutorConfig.from_environment()
    monkeypatch.setenv("GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION", "false")
    explicitly_disabled = ProductionExecutorConfig.from_environment()

    assert enabled_by_default.workflow_preemption_enabled
    assert not explicitly_disabled.workflow_preemption_enabled


def test_dispatcher_treats_lease_contention_as_waiting() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    store.claim_workflow(
        workflow.request_id,
        "executor-a",
        workflow.fencing_token,
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(minutes=3),
    )
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(
            store,
            [
                FakeAdapter(
                    {WorkflowOperation.RESTART_NODE: WorkflowStepOutcome.succeeded()}
                )
            ],
            {WorkflowOperation.RESTART_NODE},
            executor_id="executor-b",
        ),
        WorkflowDispatcherConfig(enabled=True),
    )

    report = dispatcher.run_once()

    assert report.scanned == 1
    assert report.executed == 0
    assert report.waiting == 1
    assert report.failed == 0
    assert report.failures == []
