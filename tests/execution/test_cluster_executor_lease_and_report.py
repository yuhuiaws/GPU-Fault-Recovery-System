"""Lease heartbeat, result reporting and adapter dispatch in the regional executor.

``tests/hyperpod/test_cluster_executor.py`` covers the HTTP client, the
``executor_from_environment`` wiring and the ``_execute`` error taxonomy of a
single command. What it never exercises is the part that only shows up across a
batch and across the lease heartbeat:

* the heartbeat interval ``_renew_lease`` derives from ``lease_seconds`` (that
  file drives the loop with a stop stand-in that ignores the interval), and what
  a failed renewal does to the command it was renewing;
* a batch in which one command's result is rejected by the control plane -- the
  remaining commands must still run and still report;
* ``last_cycle_advanced`` for a batch of more than one command, which is where
  the ``any(...)`` over statuses actually decides something;
* the exactly-one-adapter rule, the step-status mapping, and the synthetic prior
  execution a re-claimed command's ``result_details`` becomes.

``RegionalExecutorClient`` is faked here on purpose: claim/complete/renew are
HTTP round trips to the regional control plane, so the executor's lease and
reporting behaviour is only observable in what it sends across that boundary.
The executor itself is always the real one.
"""

from __future__ import annotations

from threading import Event
from typing import Any

import pytest
from pydantic import ValidationError

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from tests._builders import build_store, fault_incident, workflow_request, workflow_step

CLUSTER = "cluster-a"
EXECUTOR = "executor-a"
FENCING_TOKEN = 3


@pytest.fixture(autouse=True)
def claim_state_outside_shared_tmp(tmp_path, monkeypatch) -> None:
    """Keep the readiness breadcrumb out of the shared ``/tmp`` default.

    ``run_once`` writes ``last_successful_claim_at`` to a file for the
    out-of-process readiness probe. The default path is one fixed name in
    ``/tmp``, which every xdist worker would rewrite under every other.
    """

    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        str(tmp_path / "claim-state.json"),
    )


def remote_command(
    command_id: str,
    *,
    operation: WorkflowOperation = WorkflowOperation.VALIDATE_HOST,
    node_ids: list[str] | None = None,
    result_details: dict[str, Any] | None = None,
    workflow_steps: bool = False,
    execution_owner: str = "owner-a",
) -> RemoteActionCommand:
    """A claimed command whose fencing tokens agree, so ``_validate`` passes.

    ``workflow_steps`` puts this command's own step into the workflow's
    ``official_steps``, which is what the destructive fleet preflight reads.
    ``execution_owner`` routes the step to a real adapter instead of a stub.
    """

    step = workflow_step(operation, execution_owner, node_ids=node_ids)
    return RemoteActionCommand(
        command_id=command_id,
        cluster_id=CLUSTER,
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        step_index=0,
        fencing_token=FENCING_TOKEN,
        idempotency_key=f"idem-{command_id}",
        lease_token=f"lease-{command_id}",
        step=step,
        workflow=workflow_request(
            "workflow-a",
            "incident-a",
            fencing_token=FENCING_TOKEN,
            official_steps=[step] if workflow_steps else [],
        ),
        incident=fault_incident("incident-a", "event-a", fencing_token=FENCING_TOKEN),
        result_details=result_details or {},
    )


class FakeExecutorClient:
    """In-memory stand-in for the regional control-plane client.

    ``_validate`` compares the claimed command against the client's own
    cluster, so the fake carries one.
    """

    cluster_id = CLUSTER

    def __init__(
        self,
        *batches: list[RemoteActionCommand],
        complete_errors: dict[str, BaseException] | None = None,
        renew_error: BaseException | None = None,
    ) -> None:
        self.batches = list(batches)
        self.claims: list[dict[str, Any]] = []
        self.completed: list[tuple[str, RemoteCommandResult]] = []
        self.renewals: list[tuple[str, str, int]] = []
        self.complete_errors = dict(complete_errors or {})
        self.renew_error = renew_error

    def claim(
        self,
        executor_id: str,
        *,
        execution_owners: list[str],
        max_commands: int,
        lease_seconds: int,
    ) -> list[RemoteActionCommand]:
        self.claims.append(
            {
                "executor_id": executor_id,
                "execution_owners": execution_owners,
                "max_commands": max_commands,
                "lease_seconds": lease_seconds,
            }
        )
        return self.batches.pop(0) if self.batches else []

    def complete(
        self, command: RemoteActionCommand, result: RemoteCommandResult
    ) -> RemoteActionCommand:
        self.completed.append((command.command_id, result))
        error = self.complete_errors.get(command.command_id)
        if error is not None:
            raise error
        return command

    def renew(
        self, command: RemoteActionCommand, executor_id: str, lease_seconds: int
    ) -> RemoteActionCommand:
        self.renewals.append((command.command_id, executor_id, lease_seconds))
        if self.renew_error is not None:
            raise self.renew_error
        return command

    def reported(self, command_id: str) -> RemoteCommandResult:
        return next(
            result
            for reported_id, result in self.completed
            if reported_id == command_id
        )


class RecordingAdapter:
    """Local adapter stand-in that records the contexts it executed.

    The executor reads ``owner``, ``supports`` and ``execute`` off its adapters
    and nothing else. Leaving ``registry`` unset keeps ``fleet_registry`` None,
    so these tests exercise dispatch rather than the destructive preflight.
    """

    def __init__(
        self,
        owner: str = "owner-a",
        *,
        operation: WorkflowOperation | None = None,
        supported: bool = True,
        status: WorkflowStepStatus = WorkflowStepStatus.SUCCEEDED,
        step_error: str | None = None,
        details: dict[str, Any] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.owner = owner
        self.operation = operation
        self.supported = supported
        self.status = status
        self.step_error = step_error
        self.details = details
        self.raises = raises
        self.contexts: list[Any] = []

    def supports(self, step: Any) -> bool:
        return self.supported and (
            self.operation is None or step.operation is self.operation
        )

    def execute(self, context: Any) -> WorkflowStepOutcome:
        self.contexts.append(context)
        if self.raises is not None:
            raise self.raises
        return WorkflowStepOutcome(
            status=self.status, error=self.step_error, details=self.details
        )

    def executed_command_ids(self) -> list[str]:
        return sorted(
            context.idempotency_key.removeprefix("idem-") for context in self.contexts
        )


def build_executor(
    client: FakeExecutorClient, adapters: list[RecordingAdapter], **overrides: Any
) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        client,
        adapters,
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        **overrides,
    )


class OneRenewalStopEvent(Event):
    """Stop event that lets the renew loop run exactly one renewal.

    ``_renew_lease`` waits on this event between renewals, so the timeout it
    passes is the derived heartbeat interval. Returning False once and True
    afterwards runs one renewal and ends the loop without any real sleeping,
    whatever interval the executor derived.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_timeouts.append(timeout)
        return len(self.wait_timeouts) > 1


def stop_events(monkeypatch) -> list[OneRenewalStopEvent]:
    """Hand ``_execute_and_report`` stop events that record their waits."""

    created: list[OneRenewalStopEvent] = []

    def factory() -> OneRenewalStopEvent:
        event = OneRenewalStopEvent()
        created.append(event)
        return event

    monkeypatch.setattr("gpu_fault.cluster_executor.Event", factory)
    return created


@pytest.mark.parametrize(
    ("lease_seconds", "interval"), [(10, 10 / 3), (60, 20.0), (600, 30.0)]
)
def test_lease_heartbeat_waits_a_third_of_the_lease_capped_at_thirty_seconds(
    monkeypatch, lease_seconds: int, interval: float
) -> None:
    """Renew well before expiry, and no slower than twice a minute.

    A third of the lease leaves two whole heartbeats of slack before the
    control plane may steal the command back; the 30 s cap keeps a two-hour
    lease from going 40 minutes without proof the executor is still alive.
    """

    events = stop_events(monkeypatch)
    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(client, [RecordingAdapter()], lease_seconds=lease_seconds)

    assert executor.run_once() == 1
    assert [event.wait_timeouts[0] for event in events] == [pytest.approx(interval)]


def test_a_failed_lease_renewal_is_counted_without_abandoning_the_command(
    monkeypatch,
) -> None:
    """The heartbeat is best effort: losing one must not lose the result.

    The renewal runs on its own thread, so an exception there is invisible to
    the command being executed. It has to stay invisible -- the command may
    still succeed and its result is still worth reporting -- while remaining
    countable, because a renewal that never succeeds ends with the control
    plane handing the command to another executor.
    """

    stop_events(monkeypatch)
    client = FakeExecutorClient(
        [remote_command("command-a")],
        renew_error=ClusterExecutorError("stale lease token", status_code=409),
    )
    executor = build_executor(client, [RecordingAdapter()])

    assert executor.run_once() == 1
    assert executor.lease_renewal_failures == 1
    assert client.renewals == [("command-a", EXECUTOR, 120)]
    assert client.reported("command-a").status is RemoteCommandStatus.SUCCEEDED
    assert executor.reported_failures == 0


def test_a_command_result_is_posted_exactly_once() -> None:
    """One execution, one report: the control plane counts attempts."""

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(client, [RecordingAdapter()])

    assert executor.run_once() == 1
    assert [command_id for command_id, _ in client.completed] == ["command-a"]


def test_a_rejected_result_does_not_discard_the_rest_of_the_batch() -> None:
    """A stale lease on one command must not silence its batch siblings.

    ``complete`` raising means the control plane refused this result -- a stale
    lease, a stale fencing token, an already-terminal command. The other
    commands in the batch were executed for real, so swallowing their reports
    would strand work that already happened on the cluster.
    """

    client = FakeExecutorClient(
        [remote_command("command-a"), remote_command("command-b")],
        complete_errors={
            "command-a": ClusterExecutorError(
                "regional control plane rejected request (409): lease is stale",
                status_code=409,
            )
        },
    )
    adapter = RecordingAdapter()
    executor = build_executor(client, [adapter], max_concurrent_commands=2)

    assert executor.run_once() == 2
    assert adapter.executed_command_ids() == ["command-a", "command-b"]
    assert sorted(command_id for command_id, _ in client.completed) == [
        "command-a",
        "command-b",
    ]
    assert executor.reported_failures == 1
    assert executor.unexpected_failures == 0


def test_a_batch_of_only_waiting_results_does_not_count_as_progress() -> None:
    """Nothing moved, so the claim loop must fall back to the idle poll."""

    client = FakeExecutorClient(
        [remote_command("command-a"), remote_command("command-b")]
    )
    executor = build_executor(
        client,
        [RecordingAdapter(status=WorkflowStepStatus.WAITING)],
        max_concurrent_commands=2,
    )

    assert executor.run_once() == 2
    assert executor.last_cycle_advanced is False


def test_a_batch_advances_when_any_one_of_its_commands_is_terminal() -> None:
    """One terminal command is progress even next to a held sibling.

    Sleeping ``poll_seconds`` because a co-claimed command is still waiting
    would slow the workflow that just moved, so the batch verdict is ``any``
    terminal rather than ``all``.
    """

    held = remote_command("command-a", operation=WorkflowOperation.VALIDATE_HOST)
    advanced = remote_command(
        "command-b", operation=WorkflowOperation.MARK_UNSCHEDULABLE
    )
    client = FakeExecutorClient([held, advanced])
    executor = build_executor(
        client,
        [
            RecordingAdapter(
                "owner-waiting",
                operation=WorkflowOperation.VALIDATE_HOST,
                status=WorkflowStepStatus.WAITING,
            ),
            RecordingAdapter(
                "owner-terminal",
                operation=WorkflowOperation.MARK_UNSCHEDULABLE,
                status=WorkflowStepStatus.SUCCEEDED,
            ),
        ],
        max_concurrent_commands=2,
    )

    assert executor.run_once() == 2
    assert client.reported("command-a").status is RemoteCommandStatus.WAITING
    assert client.reported("command-b").status is RemoteCommandStatus.SUCCEEDED
    assert executor.last_cycle_advanced is True


@pytest.mark.parametrize(
    "status", [RemoteCommandStatus.PENDING, RemoteCommandStatus.LEASED]
)
def test_a_remote_result_may_not_report_a_queue_state_as_an_outcome(
    status: RemoteCommandStatus,
) -> None:
    """PENDING and LEASED are the queue's words, not an executor's.

    ``RemoteCommandStatus`` carries both queue states and executor outcomes.
    Letting a result post PENDING or LEASED would let an executor push a
    command back into the claimable queue through the result route.
    """

    with pytest.raises(ValidationError, match="must be WAITING, SUCCEEDED, or FAILED"):
        RemoteCommandResult(lease_token="lease-a", status=status)


def test_a_failed_remote_result_must_name_its_error() -> None:
    """A FAILED command with no reason is unreadable to the operator."""

    with pytest.raises(ValidationError, match="failed remote result requires error"):
        RemoteCommandResult(lease_token="lease-a", status=RemoteCommandStatus.FAILED)


def test_a_retryable_control_plane_status_records_what_to_retry_and_why() -> None:
    """A 503 from the control plane is not the recovery action failing.

    The adapters reach the control plane through the regional proxies, so a
    transient control-plane status surfaces inside ``_execute`` as a
    ``ClusterExecutorError``. Reporting it as FAILED would mark a GPU
    unrecoverable because a store was briefly unavailable, so it comes back
    WAITING with the status code and reason an operator needs to tell this
    apart from a real refusal.
    """

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(
        client,
        [
            RecordingAdapter(
                raises=ClusterExecutorError(
                    "regional control plane rejected request (503): store unavailable",
                    status_code=503,
                )
            )
        ],
    )

    assert executor.run_once() == 1
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING
    assert result.status_source == "executor-retryable-control-plane"
    assert result.details == {
        "retryable_control_plane_error": True,
        "status_code": 503,
        "reason": ("regional control plane rejected request (503): store unavailable"),
        "executor_id": EXECUTOR,
        "exception_type": "ClusterExecutorError",
    }
    assert result.error is None
    assert executor.unexpected_failures == 0


def test_a_control_plane_status_just_below_five_hundred_is_not_retryable() -> None:
    """499 is the boundary: below 500 and not 408/425/429 means refused.

    Retrying a 4xx that is not one of the three transient codes re-sends a
    request the control plane has already judged, so the command fails instead.
    """

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(
        client,
        [
            RecordingAdapter(
                raises=ClusterExecutorError(
                    "regional control plane rejected request (499)", status_code=499
                )
            )
        ],
    )

    assert executor.run_once() == 1
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-rejected"
    assert result.error == (
        "ClusterExecutorError: regional control plane rejected request (499)"
    )
    assert result.details == {}
    assert executor.unexpected_failures == 0


def test_a_command_no_local_adapter_supports_is_rejected() -> None:
    """The executor claimed a step it cannot run: fail it, do not hold it.

    The control plane hands out commands by ``execution_owner``, so no local
    adapter matching means the routing is wrong. Holding it WAITING would keep
    the workflow alive against an executor that can never run the step.
    """

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(client, [RecordingAdapter(supported=False)])

    assert executor.run_once() == 1
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-rejected"
    assert "found 0" in result.error


def test_a_command_two_local_adapters_claim_is_rejected() -> None:
    """Ambiguous dispatch must not be resolved by adapter order.

    Two adapters answering ``supports`` for one step means the recovery action
    would be decided by list position -- reboot via HyperPod or via the node
    agent, whichever was registered first. Refuse instead of guessing.
    """

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(
        client, [RecordingAdapter("owner-a"), RecordingAdapter("owner-b")]
    )

    assert executor.run_once() == 1
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-rejected"
    assert "found 2" in result.error


@pytest.mark.parametrize(
    ("step_status", "command_status"),
    [
        (WorkflowStepStatus.WAITING, RemoteCommandStatus.WAITING),
        (WorkflowStepStatus.SUCCEEDED, RemoteCommandStatus.SUCCEEDED),
        (WorkflowStepStatus.FAILED, RemoteCommandStatus.FAILED),
    ],
)
def test_the_adapter_step_status_becomes_the_reported_command_status(
    step_status: WorkflowStepStatus, command_status: RemoteCommandStatus
) -> None:
    """The executor reports the adapter's verdict, unchanged and untagged.

    ``status_source`` stays None for these: the source tags exist to mark
    results the executor itself manufactured, so tagging a genuine adapter
    outcome would hide the difference.
    """

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(
        client,
        [
            RecordingAdapter(
                status=step_status,
                step_error="adapter refused the step",
                details={"probe": "ran"},
            )
        ],
    )

    assert executor.run_once() == 1
    result = client.reported("command-a")
    assert result.status is command_status
    assert result.status_source is None
    assert result.details == {"probe": "ran"}


def test_a_reclaimed_command_replays_its_recorded_details_to_the_adapter() -> None:
    """The adapter has to see what its own earlier attempt already did.

    A WAITING command keeps its ``result_details`` on the control plane, and
    the executor holds no state between claims. Re-claiming it therefore
    replays those details as a synthetic WAITING execution of the same step, so
    an adapter polling its own asynchronous operation finds the operation id it
    started last time instead of starting a second one.
    """

    command = remote_command("command-a", result_details={"node_action_id": "action-7"})
    client = FakeExecutorClient([command])
    adapter = RecordingAdapter(status=WorkflowStepStatus.WAITING)
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1
    seen = adapter.contexts[0].workflow.step_executions
    assert [
        (
            execution.step_index,
            execution.operation,
            execution.status,
            execution.phase,
            execution.adapter_operation_id,
            execution.details,
        )
        for execution in seen
    ] == [
        (
            0,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowStepStatus.WAITING,
            "official",
            "remote/command-a",
            {"node_action_id": "action-7"},
        )
    ]
    # The replay is a copy: the claimed command still carries the record the
    # control plane sent, so reporting cannot echo a fabricated execution back.
    assert command.workflow.step_executions == []


def api_exception(status: int) -> Exception:
    """A ``kubernetes.client`` ApiException as the classifier recognises it.

    Built by name and module so the retryable-adapter-error classification is
    exercised without importing the Kubernetes client extra.
    """

    exc = type(
        "ApiException", (Exception,), {"__module__": "kubernetes.client.exceptions"}
    )(f"({status}) Reason: HTTP {status}")
    exc.status = status  # type: ignore[attr-defined]
    return exc


# The continuation state a multi-cycle adapter step keeps in its own details:
# the HyperPod lifecycle adapter resumes RESTART_NODE/REPLACE_NODE from
# ``agent_baselines``/``spare_failover_pending``, and VERIFY_NO_GPU_CLIENTS
# bounds its wait with ``gpu_client_quiesce_attempt``.
CONTINUATION_STATE: dict[str, Any] = {
    "agent_baselines": {"node-a": {"boot_id": "boot-1", "generation": 7}},
    "spare_failover_pending": True,
    "gpu_client_quiesce_attempt": 12,
    "reason": "waiting for the reboot to land",
}


def test_a_retryable_adapter_error_keeps_the_previous_cycles_details() -> None:
    """A transient adapter failure may not erase what the last cycle recorded.

    ``complete_remote_command`` *replaces* ``result_details``, so a WAITING
    result the executor manufactured itself is the whole record of the step
    from then on. Posting only ``retryable_adapter_error`` wiped the HyperPod
    adapter's ``agent_baselines``: the next cycle saw a WAITING execution with
    no baselines, could not auto-confirm the reboot it had already submitted,
    and idled to the step bound. The executor's own keys still win, so
    ``reason`` is this cycle's reason and not the stale one.
    """

    command = remote_command("command-a", result_details=dict(CONTINUATION_STATE))
    client = FakeExecutorClient([command])
    executor = build_executor(client, [RecordingAdapter(raises=api_exception(503))])

    assert executor.run_once() == 1, "run_once must report the hold, not raise"
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING, result
    assert result.status_source == "executor-retryable-adapter-error", result
    assert result.details["agent_baselines"] == CONTINUATION_STATE["agent_baselines"], (
        "the retryable hold dropped the adapter's continuation state"
    )
    assert result.details["spare_failover_pending"] is True, result.details
    assert result.details["gpu_client_quiesce_attempt"] == 12, result.details
    assert result.details["retryable_adapter_error"] is True, result.details
    assert result.details["reason"] == "RETRYABLE_ADAPTER_ERROR", (
        "the executor's own keys must win over the replayed ones"
    )


def test_a_retryable_control_plane_hold_keeps_the_previous_cycles_details() -> None:
    """Same rule for a 503 from the control plane behind a regional proxy."""

    command = remote_command("command-a", result_details=dict(CONTINUATION_STATE))
    client = FakeExecutorClient([command])
    executor = build_executor(
        client,
        [
            RecordingAdapter(
                raises=ClusterExecutorError(
                    "regional control plane rejected request (503): store unavailable",
                    status_code=503,
                )
            )
        ],
    )

    assert executor.run_once() == 1, "run_once must report the hold, not raise"
    result = client.reported("command-a")
    assert result.status_source == "executor-retryable-control-plane", result
    assert result.details["agent_baselines"] == CONTINUATION_STATE["agent_baselines"], (
        "a transient control-plane read dropped the adapter's continuation state"
    )
    assert result.details["retryable_control_plane_error"] is True, result.details
    assert result.details["reason"] == (
        "regional control plane rejected request (503): store unavailable"
    ), "the executor's own keys must win over the replayed ones"


class FenceRegistry:
    """Fleet registry stand-in that answers the rollout fence remotely.

    The regional proxy owns no store and answers the fence with a control-plane
    round trip, which ``fleet_rollout_fence`` resolves by this method's
    presence.
    """

    def fleet_rollout_fence_deployments(self, cluster_id: str) -> list[str]:
        return ["deployment-1"]


def test_a_fleet_preflight_hold_keeps_the_previous_cycles_details() -> None:
    """A rollout that starts mid-step must not cost the step its memory.

    The destructive preflight holds the command WAITING for as long as the
    fleet rollout runs. That hold is the executor's, not the adapter's, so
    without the merge a REPLACE_NODE already waiting on a warm spare loses
    ``spare_failover_pending`` and restarts its failover from scratch.
    """

    command = remote_command(
        "command-a",
        operation=WorkflowOperation.MARK_UNSCHEDULABLE,
        result_details=dict(CONTINUATION_STATE),
        workflow_steps=True,
    )
    client = FakeExecutorClient([command])
    adapter = RecordingAdapter()
    adapter.registry = FenceRegistry()  # type: ignore[attr-defined]
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1, "run_once must report the hold, not raise"
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING, result
    assert result.details["fleet_preflight_blocked"] is True, result.details
    assert result.details["agent_baselines"] == CONTINUATION_STATE["agent_baselines"], (
        "the preflight hold dropped the adapter's continuation state"
    )
    assert "fleet rollout fence blocked" in result.details["reason"], (
        "the executor's own keys must win over the replayed ones"
    )
    assert adapter.contexts == [], "the adapter must not run behind a closed fence"


def test_a_barrier_hold_keeps_the_previous_cycles_details() -> None:
    """The barrier refusal is a hold too, and holds do not truncate history."""

    command = remote_command(
        "command-a",
        operation=WorkflowOperation.RESET_GPU,
        node_ids=["node-a", "node-b"],
        result_details=dict(CONTINUATION_STATE),
    )
    client = FakeExecutorClient([command])
    executor = build_executor(client, [RecordingAdapter()])

    assert executor.run_once() == 1, "run_once must report the hold, not raise"
    result = client.reported("command-a")
    assert result.status_source == "executor-barrier-unavailable", result
    assert result.details["agent_baselines"] == CONTINUATION_STATE["agent_baselines"], (
        "the barrier hold dropped the adapter's continuation state"
    )
    assert result.details["multi_node_barrier_unavailable"] is True, result.details


@pytest.mark.parametrize(
    ("adapter", "status_source"),
    [
        (
            RecordingAdapter(raises=ValueError("adapter wiring is wrong")),
            "executor-internal-error",
        ),
        (
            RecordingAdapter(
                raises=ClusterExecutorError("stale fencing token", status_code=409)
            ),
            "executor-rejected",
        ),
        (
            RecordingAdapter(
                status=WorkflowStepStatus.FAILED,
                step_error="the GPU is still faulted",
                details={"probe": "ran"},
            ),
            None,
        ),
    ],
    ids=["defect", "rejected", "adapter-failed"],
)
def test_a_terminal_result_does_not_inherit_the_previous_cycles_details(
    adapter: RecordingAdapter, status_source: str | None
) -> None:
    """Merging is for holds only: a verdict must not carry waiting state.

    ``gpu_client_quiesce_attempt`` or ``spare_failover_pending`` surviving into
    a FAILED result would tell the control plane and the operator that a step
    that is over is still mid-flight, and the next reader of the details cannot
    tell an inherited key from a fresh one.
    """

    command = remote_command("command-a", result_details=dict(CONTINUATION_STATE))
    client = FakeExecutorClient([command])
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1, "run_once must report the verdict, not raise"
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.FAILED, result
    assert result.status_source == status_source, result
    assert "agent_baselines" not in result.details, (
        "a terminal result inherited the previous cycle's waiting state"
    )
    assert "spare_failover_pending" not in result.details, result.details
    assert "gpu_client_quiesce_attempt" not in result.details, result.details


def test_a_cancelled_command_keeps_the_merged_details_in_its_terminal_record() -> None:
    """Accepted: a cancellation turns a merged hold into the FAILED record.

    ``complete_remote_command`` writes FAILED with ``{**result.details,
    post_cancellation_*}`` when a cancellation was requested while the command
    was leased, so the continuation state a hold carried forward does land on a
    terminal row. That is deliberate and stays that way: the row is the only
    surviving account of what the executor was doing when the control plane
    pulled the command, ``post_cancellation_status`` says in the same record
    that the executor itself only ever asked to wait, and nothing replays a
    FAILED command's ``result_details`` (a re-claim needs a LEASED row).
    Stripping the keys in the store would delete exactly the evidence the
    operator who has to finish the action by hand needs.
    """

    store = build_store()
    command = remote_command("command-a")
    store.ensure_remote_command(command)
    claimed = store.claim_remote_commands(CLUSTER, EXECUTOR, limit=1, lease_seconds=60)[
        0
    ]
    store.cancel_remote_commands_for_workflow(
        command.workflow_request_id, reason="workflow deadline expired"
    )

    completed = store.complete_remote_command(
        CLUSTER,
        command.command_id,
        RemoteCommandResult(
            lease_token=claimed.lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-retryable-transport",
            details={**CONTINUATION_STATE, "retryable_transport_error": True},
        ),
    )

    assert completed.status is RemoteCommandStatus.FAILED, completed
    assert completed.error == "workflow deadline expired", completed.error
    assert completed.result_details["post_cancellation_status"] == "WAITING", (
        "the record must say the executor only asked to wait: "
        f"{completed.result_details}"
    )
    assert (
        completed.result_details["agent_baselines"]
        == (CONTINUATION_STATE["agent_baselines"])
    ), (
        "the cancelled record keeps the carried-forward state on purpose: "
        f"{completed.result_details}"
    )
