from __future__ import annotations

import logging

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

from ._support import (
    FakeAdapter,
    IncidentState,
    InMemoryStore,
    ProductionExecutorConfig,
    RemoteActionCommand,
    RemoteCommandStatus,
    SqliteStore,
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
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


def test_lease_holder_enforces_deadline_the_watchdog_cannot_claim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The overdue workflow the watchdog can never reap.

    ``WorkflowDispatcher._expire_stuck_workflows`` has to ``claim_workflow`` to
    act, and the store refuses a claim from a different owner while the lease is
    live. A workflow that is being redispatched renews that lease every few
    seconds, so the watchdog loses every race: the only records it can reap are
    the ones whose executor already stopped touching them. That left the looping
    workflow -- the one the deadline exists for -- immune to its own deadline.
    Enforcement therefore lives in the lease holder, and the watchdog's lost
    claim is logged instead of silently skipped.
    """

    store = build_store()
    operation = WorkflowOperation.FREEZE_EVIDENCE
    _, created = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    workflow = copy_model(
        created,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-a",
        execution_epoch=1,
        execution_lease_expires_at=now + timedelta(seconds=180),
        execution_deadline=now - timedelta(seconds=600),
    )
    store.save_workflow(workflow, expected=created)
    incident = store.get_incident(workflow.incident_id)
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id="remote-live",
            cluster_id=incident.cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=workflow.fencing_token,
            idempotency_key="remote/live",
            step=workflow.official_steps[0],
            workflow=workflow,
            incident=incident,
        )
    )
    handled = []
    adapter = FakeAdapter({operation: WorkflowStepOutcome.succeeded()})
    dispatcher = WorkflowDispatcher(
        store,
        executor(store, adapter, [operation]),
        WorkflowDispatcherConfig(enabled=True),
        failure_handler=handled.append,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.execution.dispatcher"):
        report = dispatcher.run_once()

    assert report.failed == 1
    # The step itself never ran: the deadline is checked before dispatch.
    assert adapter.calls == []
    final = store.get_workflow(workflow.request_id)
    assert final.status is WorkflowStatus.FAILED
    execution = final.step_executions[-1]
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.details["workflow_deadline_overdue_seconds"] >= 600
    assert execution.details["workflow_deadline_remote_command_cancellation"] == {
        "cancelled": 1,
        "cancellation_requested": 0,
    }
    assert handled[0].request_id == workflow.request_id
    assert store.get_remote_command("remote-live").status is RemoteCommandStatus.FAILED
    assert [
        record.getMessage()
        for record in caplog.records
        if "watchdog cannot reap it" in record.getMessage()
    ]


def _waiting_workflow_past(
    store: InMemoryStore,
    operation: WorkflowOperation,
    *,
    waited_seconds: int,
    window_seconds: int = 200,
) -> WorkflowRequest:
    """A workflow that has been WAITING on step 0 for ``waited_seconds``.

    The deadline is placed so the execution window opened ``window_seconds`` ago,
    which is what lets ``waited_seconds`` be measured at all: the per-step clock
    is clamped to the start of the window, recovered from the deadline minus the
    workflow budget. Taking the budget from the config rather than repeating it
    keeps that placement correct when the budget is retuned.
    """

    _, workflow = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    budget = ProductionExecutorConfig.workflow_execution_timeout_seconds
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        execution_deadline=now + timedelta(seconds=budget - window_seconds),
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="provider-op-1",
                started_at=now - timedelta(seconds=waited_seconds),
            )
        ],
    )
    store.save_workflow(workflow)
    return workflow


def test_step_waiting_cap_fails_a_step_no_retry_counter_can_bound() -> None:
    """Elapsed time, not another attempt count.

    Every per-operation bound in the engine is a counter an adapter reads back
    out of its own step execution record, so a record whose key never advances
    -- a synthetic check borrowing its parent's ``step_index``, an outcome that
    is never persisted -- takes the counter's ceiling away with it. Live on
    2026-09-05 that left a REPLACE_NODE step re-asking the same question for
    hours with ``verify_max_attempts=60`` stuck at 1.

    Driven through a quiesce rather than that REPLACE_NODE, because the default
    cap is what is under test and the delegated node operations answer to the
    provider window instead -- which is also what this shows: the override does
    not leak onto everything else.
    """

    store = build_store()
    operation = WorkflowOperation.QUIESCE_GPU_SERVICES
    workflow = _waiting_workflow_past(store, operation, waited_seconds=150)
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.waiting(operation_id="provider-op-1")}
    )
    active = active_workflow_executor(
        store,
        [adapter],
        [operation],
        step_waiting_timeout_seconds=120,
        step_waiting_warning_seconds=60,
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    execution = store.get_workflow(workflow.request_id).step_executions[-1]
    assert execution.status is WorkflowStepStatus.FAILED
    assert "per-step cap" in execution.error
    assert execution.details["step_waiting_seconds"] >= 150
    assert execution.details["step_waiting_timeout_seconds"] == 120


def test_step_waiting_warning_latches_so_it_reports_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn on the crossing, not on every redispatch after it.

    The step keeps waiting, and the poll interval is seconds, so an unlatched
    warning would be one line every few seconds until the cap fires -- the shape
    that trains an operator to filter the logger out.
    """

    store = build_store()
    operation = WorkflowOperation.QUIESCE_GPU_SERVICES
    workflow = _waiting_workflow_past(store, operation, waited_seconds=90)
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.waiting(operation_id="provider-op-1")}
    )
    active = active_workflow_executor(
        store,
        [adapter],
        [operation],
        step_waiting_timeout_seconds=120,
        step_waiting_warning_seconds=60,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.execution.executor"):
        first = execute_workflow(active, workflow.request_id)
        second = execute_workflow(active, workflow.request_id)

    assert first.status is WorkflowStatus.RUNNING
    assert first.waiting_step_index == 0
    assert second.status is WorkflowStatus.RUNNING
    execution = store.get_workflow(workflow.request_id).step_executions[-1]
    assert execution.status is WorkflowStepStatus.WAITING
    assert execution.details["step_waiting_slow"] is True
    assert execution.details["step_waiting_seconds"] >= 90
    assert (
        len(
            [
                record
                for record in caplog.records
                if "waiting far longer than expected" in record.getMessage()
            ]
        )
        == 1
    )


def test_inherited_step_start_time_cannot_fire_the_cap_early() -> None:
    """A start time older than this execution window only delays the cap.

    A merged or branched workflow inherits the step executions of the record it
    absorbed, so ``started_at`` can predate the current window by hours. Measured
    raw, the first dispatch after such a merge would fail the step immediately on
    time the current executor never spent.
    """

    store = build_store()
    operation = WorkflowOperation.QUIESCE_GPU_SERVICES
    _, workflow = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="provider-op-1",
                started_at=now - timedelta(hours=6),
            )
        ],
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.waiting(operation_id="provider-op-1")}
    )
    active = active_workflow_executor(
        store,
        [adapter],
        [operation],
        step_waiting_timeout_seconds=120,
        step_waiting_warning_seconds=60,
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    execution = store.get_workflow(workflow.request_id).step_executions[-1]
    assert execution.status is WorkflowStepStatus.WAITING
    # The deadline was written by this claim, so the window starts now.
    assert execution.details["step_waiting_seconds"] < 60
    assert "step_waiting_slow" not in execution.details


def test_acknowledgement_wait_is_measured_from_its_own_claim() -> None:
    """CHECK_MECHANICALS reports how long it has really waited, never a negative.

    ``claim_deadlines`` floors an acknowledgement workflow's execution deadline
    at ``now + operator_acknowledgement_timeout`` (24 h). Reconstructing the
    window start by subtracting only the 30-minute execution timeout put it a
    day in the future and ``step_waiting_seconds`` came out as -84599 live, so
    the oldest-waiting-step age never grew for the one step that waits longest.
    """

    store = build_store()
    operation = WorkflowOperation.CHECK_MECHANICALS
    _, workflow = workflow_state(store, [operation])
    now = datetime.now(timezone.utc)
    waited = 300
    workflow = copy_model(
        workflow,
        # Stamped by the claim that admitted the step: now - waited + 24 h.
        execution_deadline=now + timedelta(seconds=86400 - waited),
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="ack-op-1",
                started_at=now - timedelta(seconds=waited),
            )
        ],
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.waiting(operation_id="ack-op-1")}
    )
    active = active_workflow_executor(store, [adapter], [operation])

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.RUNNING
    execution = store.get_workflow(workflow.request_id).step_executions[-1]
    assert execution.status is WorkflowStepStatus.WAITING
    reported = execution.details["step_waiting_seconds"]
    assert waited - 5 <= reported <= waited + 60, execution.details
    assert execution.details.get("step_waiting_timeout_seconds") in (None, 86400), (
        execution.details
    )


def test_step_start_time_survives_retries_but_not_a_rebound_index() -> None:
    """``started_at`` is the step's clock, and only while it is the same step.

    It used to be re-defaulted on every attempt, which made it a second copy of
    ``updated_at`` that nothing read. Kept across attempts it becomes the value
    the per-step cap measures. It must still reset when a workflow falls back to
    ``safety_steps``, because that reuses the same indexes for other operations.
    """

    store = build_store()
    operation = WorkflowOperation.RESTART_NODE
    _, workflow = workflow_state(store, [operation])
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.waiting(operation_id="provider-op-1")}
    )
    active = executor(store, adapter, [operation])

    execute_workflow(active, workflow.request_id)
    first = store.get_workflow(workflow.request_id).step_executions[0]
    execute_workflow(active, workflow.request_id)
    second = store.get_workflow(workflow.request_id).step_executions[0]

    assert second.started_at == first.started_at

    rebound_at = datetime.now(timezone.utc)
    store.save_workflow(
        copy_model(
            store.get_workflow(workflow.request_id),
            step_executions=[
                workflow_step_execution(
                    0,
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowStepStatus.WAITING,
                    started_at=rebound_at - timedelta(hours=2),
                )
            ],
        )
    )
    execute_workflow(active, workflow.request_id)
    rebound = store.get_workflow(workflow.request_id).step_executions[0]

    assert rebound.operation is operation
    assert rebound.started_at >= rebound_at


def test_step_cap_settings_are_checked_where_the_operator_is_watching(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cap that cannot fire is a configuration defect, not a runtime surprise.

    The impossible orderings are refused at start-up. The remaining one -- the
    default cap at or above the workflow budget -- is only reported, because
    lowering the workflow budget for a drill is legitimate and refusing it would
    turn a drill into an outage; but it silently restores the unbounded state this
    setting exists to end, so it cannot pass unremarked.
    """

    with pytest.raises(WorkflowExecutionError, match="must not exceed"):
        ProductionExecutorConfig.from_mapping(
            {"GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS": "601"}
        )
    with pytest.raises(WorkflowExecutionError, match="step timeout must be positive"):
        ProductionExecutorConfig.from_mapping(
            {"GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS": "0"}
        )
    with pytest.raises(WorkflowExecutionError, match="warning threshold must be"):
        ProductionExecutorConfig.from_mapping(
            {"GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS": "0"}
        )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.execution.config"):
        config = ProductionExecutorConfig.from_mapping(
            {"GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS": "600"}
        )

    assert config.step_waiting_timeout_seconds == 600
    assert [
        record.getMessage()
        for record in caplog.records
        if "no step will ever be capped" in record.getMessage()
    ]


def test_an_override_below_the_default_cap_is_refused() -> None:
    """The ordering that silently replaces an adapter's own timeout handling.

    An override exists to raise a ceiling for an operation that legitimately
    waits longer, so one below the default cap can only be a mistake -- and a
    damaging one: the generic per-step failure would pre-empt the managed
    recovery observer's escalation notification, which is the only thing that
    tells an operator to open a provider support case.
    """

    with pytest.raises(WorkflowExecutionError, match="below the default step timeout"):
        ProductionExecutorConfig.from_mapping(
            {
                "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS": "900",
                "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS": "600",
            }
        )


def test_the_shipped_defaults_start_without_a_configuration_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The managed recovery window and the workflow budget are both 1800s.

    That is deliberate: the observer clamps its own deadline inside the workflow
    deadline, so a delegated step is bounded by the workflow and reports through
    the observer. A check that warned about it would fire on every start-up of
    every replica, which is how a real warning stops being read.
    """

    with caplog.at_level(logging.WARNING, logger="gpu_fault.execution.config"):
        config = ProductionExecutorConfig.from_mapping({})

    assert config.step_waiting_limit(WorkflowOperation.REPLACE_NODE) == 1800
    assert config.workflow_execution_timeout_seconds == 1800
    assert caplog.records == []


def test_a_delegated_operation_waits_on_its_own_clock() -> None:
    """One ceiling for every operation can only be the largest of them.

    A quiesce that has not answered in ten minutes is already wrong, while a
    delegated node replacement waits on the provider for as long as the provider
    takes. Collapsing the two would either fail the replacement early or leave
    everything else effectively unbounded.
    """

    config = ProductionExecutorConfig.from_mapping({})

    assert config.step_waiting_limit(WorkflowOperation.REPLACE_NODE) == 1800
    assert config.step_waiting_limit(WorkflowOperation.RESTART_NODE) == 1800
    assert config.step_waiting_limit(WorkflowOperation.QUIESCE_GPU_SERVICES) == 600
    # The lead time the configured pair defines, carried onto the raised ceiling
    # rather than warning five minutes into a healthy twenty-minute replacement.
    assert (
        config.step_waiting_warning_limit(WorkflowOperation.QUIESCE_GPU_SERVICES) == 300
    )
    assert config.step_waiting_warning_limit(WorkflowOperation.REPLACE_NODE) == 1500
