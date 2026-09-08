"""GF-REGIONAL-PREEMPT-035: a placement hold under the dispatcher's rule A.

The hold is a job workflow the orchestrator opened because a running attempt
was observed on a node another incident is repairing. It has one STOP step
and waits like any job workflow (F-N1 §8): inside the window it dissolves the
moment the nodes are freed (SUPERSEDED, incident RECOVERED, nothing sent to
the data plane); past the window the STOP executes with the
controller-initiated marker and the workflow ends FAILED / ESCALATED. The
node's own workflow is never touched either way.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.app.context import default_simulated_profile
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.store import SqliteStore
from tests._builders import (
    active_workflow_executor,
    attempt_observation,
    container_observation,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

STOP = WorkflowOperation.STOP_WORKLOADS
REBOOT = WorkflowOperation.RESTART_NODE
WINDOW = 240


class _AnyOwnerAdapter(FakeAdapter):
    """The hold's STOP is compiled against the runtime profile, so its owner
    is the profile's, not the builders' ``owner-a``."""

    def supports(self, step):
        return step.operation in self.outcomes


def _busy_node(store, now: datetime) -> None:
    incident = fault_incident(
        "inc-node",
        "event-node",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-node",
        node_ids=["node-a"],
        fencing_token=1,
    )
    workflow = workflow_request(
        "wf-node",
        "inc-node",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(REBOOT, node_ids=["node-a"])],
        execution_owner_id="executor-elsewhere",
        execution_lease_expires_at=now + timedelta(minutes=10),
    )
    store.save_incident_and_workflow(incident, workflow)


def _observation(now: datetime):
    return attempt_observation(
        "train-1",
        "train-1-a1",
        now,
        expected_critical_ranks=2,
        containers=[
            container_observation(
                "pod-0", "train-1-0", 0, "node-a", gpu_uuids=["GPU-a"]
            ),
            container_observation(
                "pod-1", "train-1-1", 1, "node-b", gpu_uuids=["GPU-b"]
            ),
        ],
        workload_ids=["training/job/train-1"],
    )


def _open_hold(store, now: datetime) -> tuple[str, str]:
    store.save_profile(default_simulated_profile())
    _busy_node(store, now)
    held = IncidentOrchestrator(store).hold_attempt_on_repairing_nodes(
        _observation(now)
    )
    assert held is not None, "the observation lands on a node under repair"
    incident, workflow = held
    return incident.incident_id, workflow.request_id


def _dispatcher(store, adapter) -> WorkflowDispatcher:
    executor = active_workflow_executor(store, [adapter], {STOP, REBOOT})
    return WorkflowDispatcher(
        store,
        executor,
        WorkflowDispatcherConfig(
            enabled=True, batch_size=10, max_workers=1, node_busy_wait_seconds=WINDOW
        ),
    )


def _finish_node_repair(store) -> None:
    node = store.get_workflow("wf-node")
    store.save_workflow(
        node.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "completed_step_indexes": [0],
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        )
    )


def test_preempt035_a_hold_waits_then_dissolves_when_the_node_is_freed(tmp_path):
    store = SqliteStore(str(tmp_path / "hold-dissolve.db"))
    try:
        now = datetime.now(timezone.utc)
        incident_id, request_id = _open_hold(store, now)
        adapter = _AnyOwnerAdapter({STOP: WorkflowStepOutcome.succeeded()})
        dispatcher = _dispatcher(store, adapter)

        waiting = dispatcher.run_once()

        assert waiting.filtered.get("node_busy") == 1
        assert adapter.calls == []
        assert store.get_workflow(request_id).status is WorkflowStatus.PENDING

        # The node repair finishes inside the window.
        _finish_node_repair(store)
        released = dispatcher.run_once()

        assert released.filtered.get("placement_hold_dissolved") == 1
        assert adapter.calls == [], "a dissolved hold sends nothing to the data plane"
        hold = store.get_workflow(request_id)
        assert hold.status is WorkflowStatus.SUPERSEDED
        assert hold.execution_owner_id is None
        assert hold.step_executions == []
        assert store.get_incident(incident_id).state is IncidentState.RECOVERED
        codes = [e.code for e in hold.events if e.kind is WorkflowEventKind.HOLD]
        assert codes == [
            WorkflowEventCode.PLACEMENT_HOLD_OPENED.value,
            WorkflowEventCode.NODE_UNDER_REMEDIATION.value,
            WorkflowEventCode.PLACEMENT_HOLD_DISSOLVED.value,
        ]
        terminal = [e for e in hold.events if e.kind is WorkflowEventKind.TERMINAL]
        assert [e.actor for e in terminal] == ["dispatcher-placement-hold"]
        assert dispatcher.placement_holds_dissolved_total == 1
        assert dispatcher.node_busy_timeouts_total == 0

        # Nothing left to do on a later tick.
        again = dispatcher.run_once()
        assert again.filtered.get("placement_hold_dissolved") is None
        assert dispatcher.placement_holds_dissolved_total == 1
    finally:
        store.close()


def test_preempt035_past_the_window_a_hold_stops_the_job_and_fails(tmp_path):
    store = SqliteStore(str(tmp_path / "hold-timeout.db"))
    try:
        now = datetime.now(timezone.utc)
        incident_id, request_id = _open_hold(store, now)
        adapter = _AnyOwnerAdapter({STOP: WorkflowStepOutcome.succeeded()})
        dispatcher = _dispatcher(store, adapter)

        assert dispatcher.run_once().filtered.get("node_busy") == 1

        # The window passes with node-a still under the other remediation. The
        # wait is measured from the first HOLD the dispatcher recorded (D-11),
        # so that moves into the past along with ``created_at``.
        held = store.get_workflow(request_id)
        store.amend_workflow(
            request_id,
            {
                "created_at": now - timedelta(seconds=WINDOW + 60),
                "events": [
                    event.model_copy(
                        update={"at": event.at - timedelta(seconds=WINDOW + 60)}
                    )
                    for event in held.events
                ],
            },
        )
        gave_up = dispatcher.run_once()  # keeps the STOP, marks the failure
        assert gave_up.filtered.get("node_busy_timeout") == 1
        dispatcher.run_once()  # executes the STOP

        hold = store.get_workflow(request_id)
        assert hold.status is WorkflowStatus.FAILED
        assert [call.rsplit("/", 1)[1] for call in adapter.calls] == ["STOP_WORKLOADS"]
        assert [step.operation for step in hold.official_steps] == [STOP]
        assert (
            hold.official_steps[0].parameters["termination_initiator_incident_id"]
            == incident_id
        )
        assert "node-a" in (hold.terminal_failure_reason or "")
        assert store.get_incident(incident_id).state is IncidentState.ESCALATED
        assert dispatcher.node_busy_timeouts_total == 1
        assert dispatcher.placement_holds_dissolved_total == 0

        # The node's own workflow was never touched.
        node = store.get_workflow("wf-node")
        assert node.status is WorkflowStatus.RUNNING
        assert node.execution_owner_id == "executor-elsewhere"
    finally:
        store.close()


def test_a_hold_that_lost_the_race_to_its_own_timeout_is_not_dissolved(tmp_path):
    # Once the window passed and the STOP was kept, freeing the node no longer
    # dissolves the hold: the job owner is owed the stop that was decided.
    store = SqliteStore(str(tmp_path / "hold-late-free.db"))
    try:
        now = datetime.now(timezone.utc)
        incident_id, request_id = _open_hold(store, now)
        adapter = _AnyOwnerAdapter({STOP: WorkflowStepOutcome.succeeded()})
        dispatcher = _dispatcher(store, adapter)
        assert dispatcher.run_once().filtered.get("node_busy") == 1
        held = store.get_workflow(request_id)
        store.amend_workflow(
            request_id,
            {
                "created_at": now - timedelta(seconds=WINDOW + 60),
                "events": [
                    event.model_copy(
                        update={"at": event.at - timedelta(seconds=WINDOW + 60)}
                    )
                    for event in held.events
                ],
            },
        )
        assert dispatcher.run_once().filtered.get("node_busy_timeout") == 1

        _finish_node_repair(store)
        dispatcher.run_once()

        assert store.get_workflow(request_id).status is WorkflowStatus.FAILED
        assert [call.rsplit("/", 1)[1] for call in adapter.calls] == ["STOP_WORKLOADS"]
        assert store.get_incident(incident_id).state is IncidentState.ESCALATED
        assert dispatcher.placement_holds_dissolved_total == 0
    finally:
        store.close()
