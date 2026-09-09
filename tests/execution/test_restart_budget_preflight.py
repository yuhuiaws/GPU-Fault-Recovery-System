from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.execution import (
    ProductionWorkflowExecutor,
    WorkflowExecutionRequest,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.execution.restart_budget_preflight import (
    issue_restart_authorization,
    release_unattempted_restart_reservations,
)
from gpu_fault.models import (
    IncidentState,
    RestartAuthorization,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
    cluster_token_sha256,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 8, 20, 6, 30, tzinfo=timezone.utc)
OPERATIONS = (
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.STOP_WORKLOADS,
    WorkflowOperation.RESTART_WORKLOAD,
)


class RecordingAdapter:
    def __init__(
        self,
        store: InMemoryStore,
        outcomes: dict[WorkflowOperation, WorkflowStepOutcome] | None = None,
    ) -> None:
        self.store = store
        self.outcomes = outcomes or {}
        self.calls: list[WorkflowOperation] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        state = self.store.get_restart_budget("cluster-a", "training-a")
        assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]
        self.calls.append(context.step.operation)
        return self.outcomes.get(
            context.step.operation, WorkflowStepOutcome.succeeded()
        )


def _state(
    store: InMemoryStore,
    *,
    restart_parameters: dict[str, object] | None = None,
    operations: tuple[WorkflowOperation, ...] = OPERATIONS,
) -> WorkflowRequest:
    incident = fault_incident(
        "incident-a",
        "event-a",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    parameters = {
        "cluster_id": "cluster-a",
        "job_id": "training-a",
        "source_attempt_id": "attempt-a",
        "source_gpu_count": 1,
        "restart_budget": 1,
    }
    if restart_parameters is not None:
        parameters = restart_parameters
    workflow = workflow_request(
        "workflow-a",
        incident.incident_id,
        fencing_token=1,
        official_steps=[
            workflow_step(
                operation,
                workload_ids=["training/job/training-a"],
                parameters=parameters
                if operation is WorkflowOperation.RESTART_WORKLOAD
                else {},
            )
            for operation in operations
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident(copy_model(incident, workflow_request_id=workflow.request_id))
    store.save_workflow(workflow)
    return workflow


def _executor(
    store: InMemoryStore, adapter: RecordingAdapter
) -> ProductionWorkflowExecutor:
    return active_workflow_executor(store, [adapter], OPERATIONS)


def _request() -> WorkflowExecutionRequest:
    return WorkflowExecutionRequest(expected_fencing_token=1)


def test_exhausted_budget_fails_before_any_adapter_call() -> None:
    store = build_store()
    store.reserve_job_restart("cluster-a", "training-a", 1, "previous-restart")
    workflow = _state(store)
    adapter = RecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    persisted = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    assert persisted.completed_step_indexes == []
    assert len(persisted.step_executions) == 1
    execution = persisted.step_executions[0]
    assert execution.step_index == 2
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.details["reason"] == ("RESTART_BUDGET_EXHAUSTED")
    assert "1/1" in execution.error
    assert store.list_remote_commands() == []


def test_missing_restart_context_fails_before_stop() -> None:
    store = build_store()
    workflow = _state(
        store,
        restart_parameters={
            "cluster_id": "cluster-a",
            "job_id": "training-a",
            "source_attempt_id": "attempt-a",
            "restart_budget": 1,
        },
    )
    adapter = RecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    assert execution.step_index == 2
    assert execution.details["reason"] == ("RESTART_SAFETY_CONTEXT_MISSING")
    assert "source_gpu_count" in execution.error


def test_budget_is_reserved_before_first_adapter_call() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == list(OPERATIONS)
    assert state.restart_count == 1
    assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]


def test_failure_before_restart_releases_reservation() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(
        store,
        {
            WorkflowOperation.FREEZE_EVIDENCE: (
                WorkflowStepOutcome.failed("evidence collection failed")
            )
        },
    )

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == [WorkflowOperation.FREEZE_EVIDENCE]
    assert state.restart_count == 0
    assert state.reservation_ids == []


def test_waiting_retry_does_not_double_reserve() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(
        store,
        {
            WorkflowOperation.FREEZE_EVIDENCE: (
                WorkflowStepOutcome.waiting(operation_id="evidence/waiting")
            )
        },
    )
    active = _executor(store, adapter)

    first = active.execute(workflow.request_id, _request())
    second = active.execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert first.status is WorkflowStatus.RUNNING
    assert second.status is WorkflowStatus.RUNNING
    assert state.restart_count == 1
    assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]


def test_cancelled_waiting_restart_releases_reservation() -> None:
    store = build_store()
    workflow = _state(store)
    store.reserve_job_restart(
        "cluster-a", "training-a", 1, "workflow-a/2/RESTART_WORKLOAD"
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                2, WorkflowOperation.RESTART_WORKLOAD, WorkflowStepStatus.WAITING
            )
        ],
    )

    release_unattempted_restart_reservations(store, workflow, release_step_indexes={2})

    state = store.get_restart_budget("cluster-a", "training-a")
    assert state.restart_count == 0
    assert state.reservation_ids == []


def test_issue_authorization_requires_an_existing_reservation() -> None:
    store = build_store()
    incident = fault_incident("inc-1", "event-1", cluster_id="cluster-a")
    step = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "train-1",
            "source_attempt_id": "train-1-a1",
            "source_gpu_count": 8,
            "restart_budget": 1,
        },
    )

    missing = issue_restart_authorization(
        store, incident, step, "wf/0/RESTART_WORKLOAD"
    )
    assert isinstance(missing, WorkflowStepOutcome)
    assert missing.status is WorkflowStepStatus.FAILED
    assert missing.details["reason"] == "RESTART_RESERVATION_MISSING"
    assert missing.details["reservation_id"] == "wf/0/RESTART_WORKLOAD"

    store.reserve_job_restart("cluster-a", "train-1", 1, "wf/0/RESTART_WORKLOAD")
    granted = issue_restart_authorization(
        store, incident, step, "wf/0/RESTART_WORKLOAD"
    )
    assert isinstance(granted, RestartAuthorization)
    assert granted.reservation_id == "wf/0/RESTART_WORKLOAD"
    assert granted.restart_count == 1
    assert granted.restart_budget == 1
    assert granted.source_gpu_count == 8
    assert granted.source_attempt_id == "train-1-a1"


def test_issue_authorization_rejects_a_step_from_another_cluster() -> None:
    store = build_store()
    incident = fault_incident("inc-1", "event-1", cluster_id="cluster-a")
    step = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        parameters={
            "cluster_id": "cluster-b",
            "job_id": "train-1",
            "source_attempt_id": "train-1-a1",
            "source_gpu_count": 8,
            "restart_budget": 1,
        },
    )
    # Even a reservation on the step's own cluster does not make it the
    # incident's restart: the gate trusts the incident over the plan.
    store.reserve_job_restart("cluster-b", "train-1", 1, "wf/0/RESTART_WORKLOAD")

    outcome = issue_restart_authorization(
        store, incident, step, "wf/0/RESTART_WORKLOAD"
    )

    assert isinstance(outcome, WorkflowStepOutcome)
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.error == (
        "restart safety context cluster does not match incident: cluster-b != cluster-a"
    )
    assert outcome.details == {
        "reason": "RESTART_CLUSTER_MISMATCH",
        "restart_cluster_id": "cluster-b",
        "incident_cluster_id": "cluster-a",
    }


@pytest.mark.parametrize(
    ("overrides", "reserved"),
    [
        ({"source_gpu_count": "eight"}, True),
        ({"restart_budget": "one"}, False),
        ({"restart_budget": None}, True),
    ],
)
def test_issue_authorization_fails_closed_on_a_malformed_numeric_parameter(
    overrides: dict[str, object], reserved: bool
) -> None:
    store = build_store()
    incident = fault_incident("inc-1", "event-1", cluster_id="cluster-a")
    step = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "train-1",
            "source_attempt_id": "train-1-a1",
            "source_gpu_count": 8,
            "restart_budget": 1,
            **overrides,
        },
    )
    if reserved:
        store.reserve_job_restart("cluster-a", "train-1", 1, "wf/0/RESTART_WORKLOAD")

    outcome = issue_restart_authorization(
        store, incident, step, "wf/0/RESTART_WORKLOAD"
    )

    # A malformed plan is a FAILED step with a reason, not an adapter traceback.
    assert isinstance(outcome, WorkflowStepOutcome)
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.startswith("restart safety context is invalid: ")
    assert outcome.details["reason"] == "RESTART_SAFETY_CONTEXT_INVALID"


def test_issue_authorization_rejects_a_reservation_held_by_another_step() -> None:
    store = build_store()
    incident = fault_incident("inc-1", "event-1", cluster_id="cluster-a")
    step = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "train-1",
            "source_attempt_id": "train-1-a1",
            "source_gpu_count": 8,
            "restart_budget": 2,
        },
    )
    store.reserve_job_restart("cluster-a", "train-1", 2, "other-wf/0/RESTART_WORKLOAD")

    outcome = issue_restart_authorization(
        store, incident, step, "wf/0/RESTART_WORKLOAD"
    )

    # The budget row exists but this step's reservation is not on it: still
    # fail closed, and say how much of the budget is spoken for.
    assert isinstance(outcome, WorkflowStepOutcome)
    assert outcome.details["reason"] == "RESTART_RESERVATION_MISSING"
    assert outcome.details["restart_count"] == 1
    assert outcome.details["restart_budget"] == 2


def test_issue_authorization_reports_missing_safety_context() -> None:
    store = build_store()
    incident = fault_incident("inc-1", "event-1", cluster_id="cluster-a")
    step = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "train-1",
            "restart_budget": 1,
        },
    )

    outcome = issue_restart_authorization(
        store, incident, step, "wf/0/RESTART_WORKLOAD"
    )

    assert isinstance(outcome, WorkflowStepOutcome)
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.error == (
        "restart safety context is missing: source_attempt_id, source_gpu_count"
    )
    assert outcome.details["reason"] == "RESTART_SAFETY_CONTEXT_MISSING"
    assert outcome.details["missing_parameters"] == [
        "source_attempt_id",
        "source_gpu_count",
    ]


class AuthorizationRecordingAdapter(RecordingAdapter):
    """Records the authorization each step's request carried."""

    def __init__(self, store: InMemoryStore) -> None:
        super().__init__(store)
        self.authorizations: list[RestartAuthorization | None] = []

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.authorizations.append(context.request.restart_authorization)
        return super().execute(context)


def test_dispatch_hands_the_preflight_reservation_to_the_restart_adapter() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = AuthorizationRecordingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.authorizations == [
        None,
        None,
        RestartAuthorization(
            cluster_id="cluster-a",
            job_id="training-a",
            source_attempt_id="attempt-a",
            source_gpu_count=1,
            restart_budget=1,
            restart_count=1,
            reservation_id="workflow-a/2/RESTART_WORKLOAD",
        ),
    ]
    # Signing the reservation is a read: the preflight's row is untouched.
    assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]


class ReleasingAdapter:
    """Drops the restart reservation mid-workflow.

    Stands in for a release that raced dispatch (a reaper, an operator
    restore) so the test can see what the restart step does when the
    preflight's reservation is no longer there to sign.
    """

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store
        self.calls: list[WorkflowOperation] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append(context.step.operation)
        if context.step.operation is WorkflowOperation.STOP_WORKLOADS:
            self.store.release_job_restart(
                "cluster-a", "training-a", "workflow-a/2/RESTART_WORKLOAD"
            )
        return WorkflowStepOutcome.succeeded()


def test_dispatch_fails_closed_when_the_reservation_is_gone() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = ReleasingAdapter(store)

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    persisted = store.get_workflow(workflow.request_id)
    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.FAILED
    # The restart adapter never ran: dispatch does not reserve on its behalf.
    assert adapter.calls == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.STOP_WORKLOADS,
    ]
    execution = [item for item in persisted.step_executions if item.step_index == 2][-1]
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.details["reason"] == "RESTART_RESERVATION_MISSING"
    assert execution.details["reservation_id"] == "workflow-a/2/RESTART_WORKLOAD"
    assert state.restart_count == 0
    assert state.reservation_ids == []


def test_exhausted_budget_mails_the_operator_from_the_preflight() -> None:
    store = build_store()
    store.reserve_job_restart("cluster-a", "training-a", 1, "previous-restart")
    workflow = _state(store)
    adapter = RecordingAdapter(store)
    sent: list[str] = []
    active = active_workflow_executor(
        store, [adapter], OPERATIONS, notification_sender=sent.append
    )

    result = active.execute(workflow.request_id, _request())

    notifications = store.list_notifications()
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == []
    assert execution.details["reason"] == "RESTART_BUDGET_EXHAUSTED"
    # The budget-exhausted mail is the preflight's: one notification, sent
    # once, and the step record points at it.
    assert len(notifications) == 1
    (notification,) = notifications
    assert notification.deduplication_key == (
        "cluster-a/training-a/restart-budget-exhausted/1"
    )
    assert notification.incident_id == "incident-a"
    assert "training-a" in notification.subject
    assert "1 次上限" in notification.subject
    assert "attempt-a" in notification.body_text
    assert sent == [notification.notification_id]
    assert execution.details["notification_id"] == notification.notification_id


def test_a_restart_the_adapter_refused_hands_its_reservation_back() -> None:
    """In-process: a guard refusal (``restart_submitted: False``) is released.

    The adapter no longer reserves or releases budget itself, so the only
    thing that tells the terminal write "no restart was ever submitted" is the
    marker every guard rejection carries.
    """

    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(
        store,
        {
            WorkflowOperation.RESTART_WORKLOAD: WorkflowStepOutcome.failed(
                "restart authorization does not match the workload safety context",
                details={"restart_submitted": False},
            )
        },
    )

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    execution = store.get_workflow(workflow.request_id).step_executions[-1]
    assert result.status is WorkflowStatus.FAILED
    assert adapter.calls == list(OPERATIONS)
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.details["restart_submitted"] is False
    assert state.restart_count == 0
    assert state.reservation_ids == []


def test_a_restart_that_failed_after_submission_keeps_its_reservation() -> None:
    store = build_store()
    workflow = _state(store)
    adapter = RecordingAdapter(
        store,
        {
            WorkflowOperation.RESTART_WORKLOAD: WorkflowStepOutcome.failed(
                "job create timed out", details={"reason": "RESTART_CREATE_TIMEOUT"}
            )
        },
    )

    result = _executor(store, adapter).execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    assert result.status is WorkflowStatus.FAILED
    # Without the marker the restart may have happened: the budget stays spent.
    assert state.restart_count == 1
    assert state.reservation_ids == ["workflow-a/2/RESTART_WORKLOAD"]


def test_a_remote_restart_the_data_plane_refused_hands_its_reservation_back() -> None:
    """Regional: the guard's marker rides ``RemoteCommandResult.details``.

    The cluster executor forwards the adapter's ``outcome.details`` verbatim;
    the regional adapter spreads ``result_details`` into the step's FAILED
    outcome, so the same marker reaches the terminal write.
    """

    store = build_store()
    store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id="cluster-a",
            region="us-west-2",
            hyperpod_cluster_name="hp-cluster-a",
            eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/cluster-a",
            token_sha256=cluster_token_sha256("cluster-a-token-" + "0" * 32),
            allowed_namespaces=["training"],
            agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        )
    )
    workflow = _state(store, operations=(WorkflowOperation.RESTART_WORKLOAD,))
    remote = RegionalRemoteWorkflowAdapter(store, owners={"owner-a"})
    active = active_workflow_executor(store, [remote], OPERATIONS)

    dispatched = active.execute(workflow.request_id, _request())
    command = store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=1,
        lease_seconds=60,
        execution_owners={"owner-a"},
    )[0]
    store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=command.lease_token,
            status=RemoteCommandStatus.FAILED,
            error="restart authorization does not match the workload safety context",
            details={"restart_submitted": False},
        ),
    )
    ended = active.execute(workflow.request_id, _request())

    state = store.get_restart_budget("cluster-a", "training-a")
    execution = store.get_workflow(workflow.request_id).step_executions[-1]
    assert dispatched.status is WorkflowStatus.RUNNING
    assert command.restart_authorization is not None
    assert command.restart_authorization.reservation_id == (
        "workflow-a/0/RESTART_WORKLOAD"
    )
    assert ended.status is WorkflowStatus.FAILED
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.details["restart_submitted"] is False
    assert state.restart_count == 0
    assert state.reservation_ids == []


@pytest.mark.parametrize(
    ("details", "released"),
    [
        # The data-plane guard refused before submitting anything.
        ({"restart_submitted": False, "reason": "RESTART_TARGET_AVOIDED"}, True),
        # The cluster executor refused the command before any adapter ran.
        ({"remote_status_source": "executor-rejected"}, True),
        # The workflow cancelled a command still PENDING or WAITING.
        ({"remote_status_source": "workflow-timeout"}, True),
        ({"remote_status_source": "workflow-preempted"}, True),
        # A LEASED command cancelled mid-flight: the node's last word decides.
        (
            {
                "remote_status_source": "completed-after-cancellation",
                "post_cancellation_status": "WAITING",
            },
            True,
        ),
        (
            {
                "remote_status_source": "completed-after-cancellation",
                "post_cancellation_status": "SUCCEEDED",
            },
            False,
        ),
        # Nothing here says whether the restart was submitted: keep it spent.
        ({"remote_status_source": "executor-internal-error"}, False),
        ({"remote_status_source": "executor-configuration-error"}, False),
        ({"remote_status_source": "stale-fence"}, False),
        ({"reason": "RESTART_CREATE_TIMEOUT"}, False),
        ({}, False),
    ],
)
def test_terminal_release_reads_what_a_failed_restart_record_proves(
    details: dict[str, object], released: bool
) -> None:
    store = build_store()
    workflow = _state(store)
    store.reserve_job_restart(
        "cluster-a", "training-a", 1, "workflow-a/2/RESTART_WORKLOAD"
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                2,
                WorkflowOperation.RESTART_WORKLOAD,
                WorkflowStepStatus.FAILED,
                details=details,
            )
        ],
    )

    release_unattempted_restart_reservations(store, workflow)

    state = store.get_restart_budget("cluster-a", "training-a")
    assert state.restart_count == (0 if released else 1)
    assert state.reservation_ids == (
        [] if released else ["workflow-a/2/RESTART_WORKLOAD"]
    )
