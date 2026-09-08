"""Integration scenarios behind GF-REGIONAL-PREEMPT-032/033/034.

Each scenario drives the production executor and dispatcher over a real
store (SQLite where the scenario spans several claims, the application
context where the completion service is involved) with a scripted adapter,
and ends by pushing the next event through the public ingestion API where
the rule concerns later events. They are the automated part of the F-N1
acceptance cases; the live part -- the node agent honouring a cancellation,
the job owner seeing the controller-initiated stop -- is described in the
public spec alongside each case.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.models import (
    IncidentState,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import SqliteStore
from tests._builders import (
    active_workflow_executor,
    build_context,
    copy_model,
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
from tests.orchestration._cross_fault_support import post_faults, xid_payload


def _idle_node_xid(xid: int, event_id: str) -> dict:
    # A node with no workload: the plan is executable, not safety-only.
    return {**xid_payload(xid, event_id), "workload_state": "IDLE"}


CORDON = WorkflowOperation.MARK_UNSCHEDULABLE
STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
VALIDATE = WorkflowOperation.VALIDATE_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD


def _execute(executor, workflow):
    return executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )


# ---------------------------------------------------------------- PREEMPT-032


def _idle_node_xid(xid: int, event_id: str) -> dict:
    # A node with no training workload: the single-node, no-job case of §0.
    return {**xid_payload(xid, event_id), "workload_state": "IDLE"}


class _AnyOwnerAdapter(FakeAdapter):
    """Scripted adapter for plans the real builder compiled."""

    def supports(self, step):
        return True

    def execute(self, context):
        self.calls.append(context.idempotency_key)
        outcome = self.outcomes.get(context.step.operation)
        if outcome is None:
            return WorkflowStepOutcome.succeeded()
        if (
            outcome.status is WorkflowStepStatus.WAITING
            and outcome.adapter_operation_id
            in context.request.confirmed_adapter_operation_ids
        ):
            return WorkflowStepOutcome.succeeded(
                operation_id=outcome.adapter_operation_id
            )
        return outcome


def test_preempt032_a_node_workflow_past_its_lifetime_ends_with_the_operator(tmp_path):
    store = SqliteStore(str(tmp_path / "lifetime.db"))
    try:
        context = build_context(store=store)
        # A GPU fault opens the node's remediation through the public API.
        first = asyncio.run(
            post_faults(
                context, [("/v1/gpu-events/xid", _idle_node_xid(48, "first-fault"))]
            )
        )[0]
        incident_id = first["incident_id"]
        request_id = first["workflow_request_id"]
        workflow = store.get_workflow(request_id)
        assert RESET in {step.operation for step in workflow.official_steps}

        adapter = _AnyOwnerAdapter(
            {RESET: WorkflowStepOutcome.waiting(operation_id="remote/reset-a")}
        )
        executor = active_workflow_executor(store, [adapter], set(WorkflowOperation))
        executor.config = dataclasses.replace(
            executor.config, node_workflow_lifetime_seconds=1
        )
        claimed_at = datetime.now(timezone.utc)

        first_run = _execute(executor, workflow)

        running = store.get_workflow(request_id)
        assert first_run.status is WorkflowStatus.RUNNING
        assert running.lifetime_deadline_at is not None
        assert running.lifetime_deadline_at <= claimed_at + timedelta(seconds=3)
        assert running.execution_deadline == running.lifetime_deadline_at
        reset_index = next(
            index
            for index, step in enumerate(running.official_steps)
            if step.operation is RESET
        )
        # The reset is on the node: a remote command the agent is working on.
        store.ensure_remote_command(
            RemoteActionCommand(
                command_id="cmd-reset-a",
                cluster_id="cluster-a",
                workflow_request_id=request_id,
                incident_id=incident_id,
                step_index=reset_index,
                fencing_token=running.fencing_token,
                idempotency_key=f"{request_id}/{reset_index}/RESET_GPU",
                step=running.official_steps[reset_index],
                workflow=running,
                incident=store.get_incident(incident_id),
                status=RemoteCommandStatus.WAITING,
            )
        )
        calls_before = list(adapter.calls)
        time.sleep(1.3)

        second_run = _execute(executor, running)

        saved = store.get_workflow(request_id)
        assert second_run.status is WorkflowStatus.FAILED
        assert saved.status is WorkflowStatus.FAILED
        # Nothing new started after the deadline except the undo of the
        # quiesce the workflow itself performed: no reboot, no validation.
        started_after = [
            call.rsplit("/", 1)[1] for call in adapter.calls[len(calls_before) :]
        ]
        assert set(started_after) <= {"RESTORE_GPU_SERVICES"}
        assert not any("RESTART_NODE" in call for call in adapter.calls), (
            'expected any("RESTART_NODE" in call for call in adapter.calls) to be false'
        )
        command = store.get_remote_command("cmd-reset-a")
        assert command.status is RemoteCommandStatus.FAILED
        assert command.status_source == "workflow-timeout"
        failed = [
            e for e in saved.step_executions if e.status is WorkflowStepStatus.FAILED
        ]
        assert failed and failed[-1].details.get("workflow_lifetime_exceeded") is True
        # The node stays isolated and the incident goes to the operator.
        assert store.get_incident(incident_id).state in {
            IncidentState.ESCALATED,
            IncidentState.QUARANTINED,
        }
        classification = HardwareEscalationService.classify(saved)
        assert classification is not None
        assert classification[0] == "lifetime_exceeded"
        assert classification[2] is WorkflowOperation.ESCALATE_SUPPORT
        assert executor.lifetime_exceeded_total == 1

        # A later, stronger fault on the same node is recorded on the incident
        # the operator already holds; no new workflow is planned.
        later = asyncio.run(
            post_faults(
                context, [("/v1/gpu-events/xid", _idle_node_xid(79, "after-deadline"))]
            )
        )[0]

        assert later["incident_id"] == incident_id
        assert later["workflow_request_id"] == request_id
        assert store.get_workflow(request_id).status is WorkflowStatus.FAILED
        assert [
            item.request_id
            for item in store.list_workflows()
            if item.status in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}
        ] == []
    finally:
        store.close()


# ---------------------------------------------------------------- PREEMPT-033


def test_preempt033_a_stopped_job_winds_its_workflow_down_without_restarting_it(
    context, failed_event
):
    store = context.store
    incident = fault_incident(
        "inc-job",
        "event-job",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-job",
        node_ids=["node-b", "node-c"],
        job_id=failed_event.job_id,
        attempt_id=failed_event.attempt_id,
        fencing_token=1,
    )
    steps = [
        workflow_step(
            STOP,
            node_ids=["node-b", "node-c"],
            workload_ids=["training/job/train-123"],
            branch_id="shared",
        ),
        workflow_step(
            CORDON,
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:1:node-b",
        ),
        workflow_step(
            RESET,
            node_ids=["node-b"],
            gpu_uuids=["GPU-b"],
            depends_on_step_indexes=[1],
            branch_id="branch:1:node-b",
        ),
        workflow_step(
            RESTORE,
            node_ids=["node-b"],
            depends_on_step_indexes=[2],
            branch_id="branch:1:node-b",
        ),
        workflow_step(
            CORDON,
            node_ids=["node-c"],
            depends_on_step_indexes=[0],
            branch_id="branch:2:node-c",
        ),
        workflow_step(
            RESET,
            node_ids=["node-c"],
            gpu_uuids=["GPU-c"],
            depends_on_step_indexes=[4],
            branch_id="branch:2:node-c",
        ),
        workflow_step(
            RESTORE,
            node_ids=["node-c"],
            depends_on_step_indexes=[5],
            branch_id="branch:2:node-c",
        ),
        workflow_step(
            RESTART_JOB,
            node_ids=["node-b", "node-c"],
            workload_ids=["training/job/train-123"],
            parameters=dict(RESTART_PARAMETERS),
            depends_on_step_indexes=[3, 6],
            branch_id="join",
        ),
    ]
    workflow = workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        dag_enabled=True,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[STOP, CORDON],
        step_executions=[
            workflow_step_execution(0, STOP),
            workflow_step_execution(1, CORDON),
            # node-b's reset is on the node right now; node-c has not started.
            workflow_step_execution(
                2,
                RESET,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/reset-b",
            ),
        ],
        lifetime_deadline_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    store.save_incident_and_workflow(incident, workflow)
    adapter = FakeAdapter(
        {
            CORDON: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.succeeded(),
            RESTORE: WorkflowStepOutcome.succeeded(),
            RESTART_JOB: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = active_workflow_executor(
        store, [adapter], {STOP, CORDON, RESET, RESTORE, RESTART_JOB}
    )

    # The job owner stops the training job while the workflow is repairing it.
    context.completion.handle_terminal(
        copy_model(failed_event, terminal_status=TerminalStatus.STOPPED)
    )
    withdrawn = store.get_workflow("wf-job")
    assert withdrawn.workload_withdrawn_at is not None

    result = _execute(executor, withdrawn)

    saved = store.get_workflow("wf-job")
    assert result.status is WorkflowStatus.SUPERSEDED
    assert saved.status is WorkflowStatus.SUPERSEDED
    # The in-flight reset finished and node-b was released; node-c was never
    # cordoned so it owes no release, and the job the owner stopped was not
    # restarted.
    assert [call.rsplit("/", 1)[1] for call in adapter.calls] == [
        "RESET_GPU",
        "RESTORE_SCHEDULING",
    ]
    assert {4, 5, 6, 7} <= set(saved.superseded_step_indexes)
    assert "stop" in (saved.preemption_reason or "")
    assert store.get_incident("inc-job").state is IncidentState.RECOVERED


# ---------------------------------------------------------------- PREEMPT-034


def test_preempt034_a_job_on_a_node_under_repair_waits_then_stops_the_job(tmp_path):
    store = SqliteStore(str(tmp_path / "node-busy.db"))
    try:
        node_incident = fault_incident(
            "inc-node",
            "event-node",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="wf-node",
            node_ids=["node-a"],
            fencing_token=1,
        )
        node_workflow = workflow_request(
            "wf-node",
            "inc-node",
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            official_steps=[workflow_step(REBOOT, node_ids=["node-a"])],
            execution_owner_id="executor-elsewhere",
            execution_lease_expires_at=datetime.now(timezone.utc)
            + timedelta(minutes=10),
        )
        store.save_incident_and_workflow(node_incident, node_workflow)
        now = datetime.now(timezone.utc)
        job_incident = fault_incident(
            "inc-job",
            "event-job",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="wf-job",
            node_ids=["node-a", "node-b"],
            job_id="train-1",
            attempt_id="train-1-a1",
            fencing_token=1,
            created_at=now,
            updated_at=now,
        )
        job_workflow = workflow_request(
            "wf-job",
            "inc-job",
            fencing_token=1,
            official_steps=[
                workflow_step(
                    STOP,
                    node_ids=["node-a", "node-b"],
                    workload_ids=["training/job/train-1"],
                ),
                workflow_step(
                    RESET,
                    node_ids=["node-a"],
                    gpu_uuids=["GPU-a"],
                    depends_on_step_indexes=[0],
                ),
                workflow_step(
                    RESTART_JOB,
                    node_ids=["node-a", "node-b"],
                    parameters=dict(RESTART_PARAMETERS),
                    depends_on_step_indexes=[1],
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        store.save_incident_and_workflow(job_incident, job_workflow)
        adapter = FakeAdapter(
            {
                STOP: WorkflowStepOutcome.succeeded(),
                RESET: WorkflowStepOutcome.succeeded(),
                RESTART_JOB: WorkflowStepOutcome.succeeded(),
            }
        )
        executor = active_workflow_executor(
            store, [adapter], {STOP, RESET, RESTART_JOB}
        )
        dispatcher = WorkflowDispatcher(
            store,
            executor,
            WorkflowDispatcherConfig(
                enabled=True, batch_size=10, max_workers=1, node_busy_wait_seconds=300
            ),
        )

        waiting = dispatcher.run_once()

        assert waiting.filtered.get("node_busy") == 1
        assert adapter.calls == []
        assert store.get_workflow("wf-job").status is WorkflowStatus.PENDING

        # Five minutes later the node is still under the other remediation
        # (the wait is measured from the first HOLD, D-11).
        held = store.get_workflow("wf-job")
        store.amend_workflow(
            "wf-job",
            {
                "created_at": now - timedelta(minutes=6),
                "events": [
                    event.model_copy(update={"at": event.at - timedelta(minutes=6)})
                    for event in held.events
                ],
            },
        )
        dispatcher.run_once()  # gives up: the plan becomes stop-only
        dispatcher.run_once()  # executes the stop

        saved = store.get_workflow("wf-job")
        assert saved.status is WorkflowStatus.FAILED
        assert [call.rsplit("/", 1)[1] for call in adapter.calls] == ["STOP_WORKLOADS"]
        assert [step.operation for step in saved.official_steps] == [STOP]
        # The stop the data plane receives names who initiated it.
        assert (
            saved.official_steps[0].parameters["termination_initiator_incident_id"]
            == "inc-job"
        )
        assert "node-a" in (saved.terminal_failure_reason or "")
        assert store.get_incident("inc-job").state is IncidentState.ESCALATED
        # The node's own workflow was never touched.
        node_after = store.get_workflow("wf-node")
        assert node_after.status is WorkflowStatus.RUNNING
        assert node_after.execution_owner_id == "executor-elsewhere"
        assert dispatcher.node_busy_timeouts_total == 1
    finally:
        store.close()
