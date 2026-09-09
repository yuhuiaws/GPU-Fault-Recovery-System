"""Rule A, composed: a job whose node another remediation is repairing.

The job waits one bounded window (``node_busy_wait_seconds``). If the repair
finishes inside it the job proceeds; past it the job is not restarted -- the
workflow ends FAILED with a reason naming the node, its incident ESCALATED.

The workflow under test is shaped like ``PassiveWorkflowCompiler.compile``'s
``after_incident`` restart: a derived incident id, a single RESTART_WORKLOAD
step carrying ``requires_incident_state``, nodes
that include the one under repair. Its incident id differs from the node
remediation's, so the dispatcher's hold treats it as another incident and it
waits like any job workflow; the step's premise is the backstop once it runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
)
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
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
from tests.execution._support import (
    RESTART_PARAMETERS,
    FakeAdapter,
    WorkflowStepOutcome,
)
from tests.execution.test_node_busy_wait import REBOOT, _busy_node

RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
WINDOW = 240
# ``<incident>-completion-<attempt>-<hash>``: what the compiler derives when the
# node incident already exists.
DERIVED_INCIDENT = "inc-node-completion-train-1-a1-0123456789ab"
HOLDING = WorkflowEventCode.NODE_UNDER_REMEDIATION.value
TIMEOUT = WorkflowEventCode.NODE_REMEDIATION_TIMEOUT.value


def _after_incident_restart(store, *, created_at: datetime) -> None:
    incident = fault_incident(
        DERIVED_INCIDENT,
        "cluster-a/train-1-a1/TrainingAttemptTerminal/plan-1",
        event_type="TRAINING_ATTEMPT_TERMINAL",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-restart",
        node_ids=["node-a", "node-b"],
        job_id="train-1",
        attempt_id="train-1-a1",
        created_at=created_at,
        updated_at=created_at,
        fencing_token=3,
    )
    workflow = workflow_request(
        "wf-restart",
        DERIVED_INCIDENT,
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(
                RESTART_JOB,
                node_ids=["node-a", "node-b"],
                parameters={
                    **RESTART_PARAMETERS,
                    "requires_incident_state": IncidentState.RECOVERED.value,
                    "incident_id": "inc-node",
                },
            )
        ],
        created_at=created_at,
        updated_at=created_at,
    )
    store.save_incident_and_workflow(incident, workflow)


def _dispatcher(store, adapter: FakeAdapter) -> WorkflowDispatcher:
    executor = active_workflow_executor(store, [adapter], {RESTART_JOB, REBOOT})
    return WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, node_busy_wait_seconds=WINDOW
        ),
    )


def _rewind(store, request_id: str, by: timedelta) -> None:
    """Move the row's clock ``by`` into the past: its ``created_at`` and every
    HOLD the dispatcher stamped on it (the rule A window opens at the first)."""

    current = store.get_workflow(request_id)
    store.amend_workflow(
        request_id,
        {
            "created_at": current.created_at - by,
            "events": [
                event.model_copy(update={"at": event.at - by})
                for event in current.events
            ],
        },
    )


def _repair_finished(store) -> None:
    node_workflow = store.get_workflow("wf-node")
    store.save_workflow(
        copy_model(
            node_workflow,
            status=WorkflowStatus.SUCCEEDED,
            execution_owner_id=None,
            execution_lease_expires_at=None,
            completed_step_indexes=[0],
        )
    )
    store.save_incident(
        copy_model(store.get_incident("inc-node"), state=IncidentState.RECOVERED)
    )


# --- (a) the repair outlasts the window ----------------------------------------


def test_a_restart_waits_on_the_busy_node_then_fails_without_restarting():
    store = build_store()
    _busy_node(store)
    now = datetime.now(timezone.utc)
    _after_incident_restart(store, created_at=now)
    adapter = FakeAdapter({RESTART_JOB: WorkflowStepOutcome.succeeded()})
    dispatcher = _dispatcher(store, adapter)

    waiting = dispatcher.run_once()

    assert waiting.filtered.get("node_busy") == 1, waiting.filtered
    assert adapter.calls == [], "nothing restarts while the node is under repair"
    assert store.get_workflow("wf-restart").status is WorkflowStatus.PENDING

    # The window passes with the node still under the other remediation. The
    # wait is measured from the first HOLD the dispatcher recorded (D-11), so
    # that is what moves into the past, along with ``created_at``.
    _rewind(store, "wf-restart", timedelta(seconds=WINDOW + 60))
    gave_up = dispatcher.run_once()
    dispatcher.run_once()

    assert gave_up.filtered.get("node_busy_timeout") == 1, gave_up.filtered
    saved = store.get_workflow("wf-restart")
    assert saved.status is WorkflowStatus.FAILED, saved.status
    reason = saved.terminal_failure_reason or ""
    assert "node-a" in reason and "wf-node" in reason, reason
    assert reason.startswith(TIMEOUT), reason
    assert store.get_incident(DERIVED_INCIDENT).state is IncidentState.ESCALATED
    assert adapter.calls == [], "past the window the job is never restarted"
    assert dispatcher.node_busy_timeouts_total == 1
    node_after = store.get_workflow("wf-node")
    assert node_after.status is WorkflowStatus.RUNNING, "the repair keeps its budget"
    assert node_after.execution_owner_id == "executor-elsewhere"


# --- (b) the repair finishes inside the window --------------------------------


def test_a_restart_proceeds_once_when_the_repair_finishes_inside_the_window():
    store = build_store()
    _busy_node(store)
    _after_incident_restart(store, created_at=datetime.now(timezone.utc))
    adapter = FakeAdapter({RESTART_JOB: WorkflowStepOutcome.succeeded()})
    dispatcher = _dispatcher(store, adapter)

    waiting = dispatcher.run_once()
    assert waiting.filtered.get("node_busy") == 1, waiting.filtered
    assert adapter.calls == [], "still under repair: nothing restarts yet"

    _repair_finished(store)
    dispatcher.run_once()
    dispatcher.run_once()  # a second tick must not restart the job again

    assert adapter.calls == ["wf-restart/0/RESTART_WORKLOAD"], adapter.calls
    saved = store.get_workflow("wf-restart")
    assert saved.status is WorkflowStatus.SUCCEEDED, saved.status
    assert saved.terminal_failure_reason is None, saved.terminal_failure_reason
    assert dispatcher.node_busy_timeouts_total == 0
    holds = [event for event in saved.events if event.details.get("reason") == HOLDING]
    assert len(holds) == 1, [event.model_dump() for event in saved.events]
    assert holds[0].details["remediation_workflow_id"] == "wf-node"


# --- (d) the premise backstop once the step is running -------------------------


def test_the_running_premise_gives_up_at_the_window_not_the_generic_step_cap():
    store = build_store()
    now = datetime.now(timezone.utc)
    _after_incident_restart(store, created_at=now - timedelta(minutes=10))
    # Already dispatched: the step has been WAITING on the premise for longer
    # than the window but far less than the 600s generic step cap. The step's
    # clock is floored at the execution window's start (deadline minus the
    # workflow budget), so the deadline is placed to open that window 600s ago.
    # Its first claim's preflight reserved the restart; dispatch signs that.
    store.reserve_job_restart(
        str(RESTART_PARAMETERS["cluster_id"]),
        str(RESTART_PARAMETERS["job_id"]),
        int(RESTART_PARAMETERS["restart_budget"]),
        "wf-restart/0/RESTART_WORKLOAD",
    )
    budget = ProductionExecutorConfig.workflow_execution_timeout_seconds
    store.save_workflow(
        copy_model(
            store.get_workflow("wf-restart"),
            status=WorkflowStatus.RUNNING,
            execution_deadline=now + timedelta(seconds=budget - 600),
            step_executions=[
                workflow_step_execution(
                    0,
                    RESTART_JOB,
                    WorkflowStepStatus.WAITING,
                    adapter_operation_id="restart-1",
                    details={"reason": HOLDING, "remediation_workflow_id": "wf-node"},
                    started_at=now - timedelta(seconds=WINDOW + 30),
                )
            ],
        )
    )
    adapter = FakeAdapter(
        {
            RESTART_JOB: WorkflowStepOutcome.waiting(
                operation_id="restart-1",
                details={
                    "reason": HOLDING,
                    "remediation_workflow_id": "wf-node",
                    "incident_id": "inc-node",
                    "incident_state": IncidentState.ACTION_PENDING.value,
                },
            )
        }
    )
    executor = active_workflow_executor(store, [adapter], {RESTART_JOB})
    assert executor.config.step_waiting_limit(RESTART_JOB) == 600, (
        "the generic cap must be the one this test shows is not what fires"
    )

    result = execute_workflow(executor, "wf-restart")

    assert result.status is WorkflowStatus.FAILED, result
    execution = store.get_workflow("wf-restart").step_executions[-1]
    assert execution.status is WorkflowStepStatus.FAILED, execution
    assert execution.details["reason"] == TIMEOUT, execution.details
    assert execution.details["step_waiting_timeout_seconds"] == WINDOW, (
        execution.details
    )
    assert execution.details["step_waiting_seconds"] < 600, execution.details
    assert store.get_workflow("wf-restart").status is WorkflowStatus.FAILED
