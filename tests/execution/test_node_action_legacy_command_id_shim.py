"""The upgrade-window read-back for the generation-stable command_id (R4 fix 1).

R4 renamed the driver, firmware and EFA remediation command from
``<key>/<node>/agent-N`` to ``<key>/<node>``. The steady state is closed; the
transition is not: a step that was WAITING when the new release landed holds an
agent-ledger row under the OLD id, and a control plane that polls only the new
id reads 404, submits, and the agent -- with no history for the new id -- runs
a SECOND install next to the first.

The shim under test: for the three stable operations only, a 404 on the new id
is followed by one read of the legacy id before anything is submitted. The
legacy id is taken from the step's own ``node_action_command_id`` pointer when
that pointer carries the old shape (exact: it is the id the old control plane
submitted), else derived from the live generation. IN_PROGRESS means wait, a
result is folded, and only two 404s submit.

The wire is faked at ``urlopen`` so the id the control plane asks for is what
is observed; the adapter, signing and folding are the real ones.

REMOVE WITH THE SHIM (one release after R4 ships): the ``agent-N`` rows age out
of the agent ledgers within the retention window, after which the second read
is a wasted round trip on every first dispatch of these operations.
"""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from gpu_fault.execution import step_bounds
from gpu_fault.node_agent import NodeActionExecutionState, NodeActionSubmission
from tests._builders import copy_model, workflow_step_execution
from tests.execution._support import (
    NodeActionResult,
    NodeActionStatus,
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    build_store,
    workflow_state,
)

SECRET = "s" * 32
ENDPOINT = "http://node-a:9099"
KEY = "workflow/r4-fix1"
NEW_ID = f"{KEY}/node-a"
LIVE_GENERATION = 7
LEGACY_ID = f"{NEW_ID}/agent-{LIVE_GENERATION}"
GENERATION_STABLE = [
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
]


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeAgentLedgerWire:
    """An agent whose ledger is a dict; unknown ids answer 404 like the app.

    Every GET is recorded by the ``command_id`` it asked for, every POST by
    the id in its body, so a test can pin the exact sequence of ids the
    control plane tried and whether anything was submitted at all.
    """

    def __init__(self, ledger: dict[str, NodeActionSubmission]) -> None:
        self.ledger = dict(ledger)
        self.polled: list[str] = []
        self.submitted: list[str] = []

    def __call__(
        self, request: Any, timeout: float | None = None, *, ssl_context: Any = None
    ) -> FakeResponse:
        if request.get_method() == "GET":
            command_id = parse_qs(urlsplit(request.full_url).query)["command_id"][0]
            self.polled.append(command_id)
            state = self.ledger.get(command_id)
            if state is None:
                raise HTTPError(
                    request.full_url,
                    404,
                    "not found",
                    {},
                    io.BytesIO(b'{"detail": "node action command is unknown"}'),
                )
            return FakeResponse(state.model_dump_json().encode())
        command_id = json.loads(request.data)["command"]["command_id"]
        self.submitted.append(command_id)
        accepted = NodeActionSubmission(
            command_id=command_id, state=NodeActionExecutionState.PENDING
        )
        self.ledger[command_id] = accepted
        return FakeResponse(accepted.model_dump_json().encode())


def in_progress(command_id: str) -> NodeActionSubmission:
    return NodeActionSubmission(
        command_id=command_id, state=NodeActionExecutionState.PENDING
    )


def succeeded(command_id: str, operation: WorkflowOperation) -> NodeActionSubmission:
    return NodeActionSubmission(
        command_id=command_id,
        state=NodeActionExecutionState.SUCCEEDED,
        result=NodeActionResult(
            command_id=command_id,
            operation=operation,
            status=NodeActionStatus.SUCCEEDED,
            attempt=1,
            details={"verified_driver_branches": [575]},
        ),
    )


def interrupted(command_id: str, operation: WorkflowOperation) -> NodeActionSubmission:
    """The row the agent closes when it restarts under a running install."""

    return NodeActionSubmission(
        command_id=command_id,
        state=NodeActionExecutionState.FAILED,
        result=NodeActionResult(
            command_id=command_id,
            operation=operation,
            status=NodeActionStatus.INTERRUPTED,
            error="agent restarted during attempt 1",
            retryable=False,
            attempt=1,
        ),
    )


def retryable_failure(
    command_id: str, operation: WorkflowOperation
) -> NodeActionSubmission:
    return NodeActionSubmission(
        command_id=command_id,
        state=NodeActionExecutionState.FAILED,
        result=NodeActionResult(
            command_id=command_id,
            operation=operation,
            status=NodeActionStatus.FAILED,
            error="apt lock held",
            retryable=True,
            attempt=1,
        ),
    )


def wire(
    monkeypatch: pytest.MonkeyPatch, ledger: dict[str, NodeActionSubmission]
) -> FakeAgentLedgerWire:
    fake = FakeAgentLedgerWire(ledger)
    monkeypatch.setattr("gpu_fault.adapters.node_action.transport.urlopen", fake)
    return fake


def unreachable_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent that is being restarted: every request fails to connect."""

    def refuse(request: Any, timeout: float | None = None, *, ssl_context: Any = None):
        raise URLError("[Errno 111] Connection refused")

    monkeypatch.setattr("gpu_fault.adapters.node_action.transport.urlopen", refuse)


def _step_context(operation: WorkflowOperation, executions: list, *, generation: int):
    """The adapter and one step context over ``executions`` (the step's records)."""

    store = build_store()
    incident, workflow = workflow_state(store, [operation])
    registry = StubFleetRegistry({"node-a": ENDPOINT}, generation=generation)
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    context = WorkflowStepContext(
        workflow=copy_model(
            workflow, official_steps=[step], step_executions=list(executions)
        ),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key=KEY,
    )
    return adapter, context


def _poll(
    operation: WorkflowOperation,
    *,
    previous_pointer: str | None = None,
    generation: int = LIVE_GENERATION,
):
    """One control-plane poll of a single-node step of ``operation``.

    ``previous_pointer`` is the ``node_action_command_id`` the step's last
    WAITING record carries -- what the old control plane wrote before the
    upgrade, in the shape it used.
    """

    store = build_store()
    incident, workflow = workflow_state(store, [operation])
    registry = StubFleetRegistry({"node-a": ENDPOINT}, generation=generation)
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=registry)
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    executions = []
    if previous_pointer is not None:
        executions.append(
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                adapter_operation_id=KEY,
                details={
                    "node_action_command_id": previous_pointer,
                    "node_action_state": "PENDING",
                },
            )
        )
    return adapter.execute(
        WorkflowStepContext(
            workflow=copy_model(
                workflow, official_steps=[step], step_executions=executions
            ),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key=KEY,
        )
    )


@pytest.mark.parametrize("operation", GENERATION_STABLE, ids=lambda op: op.value)
def test_an_in_progress_legacy_row_is_waited_on_not_resubmitted(
    monkeypatch: pytest.MonkeyPatch, operation: WorkflowOperation
) -> None:
    """(i) The install the old control plane started is still running."""

    fake = wire(monkeypatch, {LEGACY_ID: in_progress(LEGACY_ID)})

    outcome = _poll(operation, previous_pointer=LEGACY_ID)

    assert fake.submitted == [], (
        "the new control plane submitted on top of the install the old one "
        "started: a second install"
    )
    assert fake.polled == [NEW_ID, LEGACY_ID]
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] == "PENDING"
    # The pointer now names the row that is actually running, so the next
    # poll's read-back is exact and an operator can find the ledger row.
    assert outcome.details["node_action_command_id"] == LEGACY_ID
    assert outcome.details["node_action_legacy_command_id"] == LEGACY_ID


@pytest.mark.parametrize("operation", GENERATION_STABLE, ids=lambda op: op.value)
def test_a_finished_legacy_row_is_folded_not_rerun(
    monkeypatch: pytest.MonkeyPatch, operation: WorkflowOperation
) -> None:
    """(ii) The install finished under the old id; the step succeeds on it."""

    fake = wire(monkeypatch, {LEGACY_ID: succeeded(LEGACY_ID, operation)})

    outcome = _poll(operation, previous_pointer=LEGACY_ID)

    assert fake.submitted == []
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert outcome.details["node_results"]["node-a"] == {
        "verified_driver_branches": [575]
    }


@pytest.mark.parametrize("operation", GENERATION_STABLE, ids=lambda op: op.value)
def test_two_unknown_ids_submit_exactly_once_under_the_new_id(
    monkeypatch: pytest.MonkeyPatch, operation: WorkflowOperation
) -> None:
    """(iii) A step the old control plane never dispatched: one fresh submit."""

    fake = wire(monkeypatch, {})

    outcome = _poll(operation)

    assert fake.polled == [NEW_ID, LEGACY_ID], (
        "the legacy id must be read once, after the new id, before any submit"
    )
    assert fake.submitted == [NEW_ID]
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_command_id"] == NEW_ID
    assert "node_action_legacy_command_id" not in outcome.details


def test_the_pointer_names_the_legacy_id_when_the_generation_has_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded pointer is exact; the live generation is only a guess.

    The old control plane submitted under agent-5, the agent restarted twice
    since and now reports 7. Deriving ``agent-7`` alone misses the row; the
    pointer the old control plane wrote finds it, and is read first.
    """

    recorded = f"{NEW_ID}/agent-5"
    fake = wire(monkeypatch, {recorded: in_progress(recorded)})

    outcome = _poll(WorkflowOperation.REMEDIATE_DRIVER, previous_pointer=recorded)

    assert fake.submitted == []
    assert fake.polled == [NEW_ID, recorded]
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_command_id"] == recorded


def test_a_new_shape_pointer_falls_back_to_the_live_generation_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer already in the new shape carries no legacy id to read back.

    Both the pointer and the derived guess resolve to one legacy candidate,
    so the read-back is a single extra GET, never two for the same id.
    """

    fake = wire(monkeypatch, {})

    outcome = _poll(WorkflowOperation.REMEDIATE_DRIVER, previous_pointer=NEW_ID)

    assert fake.polled == [NEW_ID, LEGACY_ID]
    assert fake.submitted == [NEW_ID]
    assert outcome.status is WorkflowStepStatus.WAITING


def test_diagnostic_actions_do_not_read_a_legacy_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suffix never left the diagnostic ids, so there is nothing to read."""

    fake = wire(monkeypatch, {})

    outcome = _poll(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE)

    assert fake.polled == [f"{NEW_ID}/agent-{LIVE_GENERATION}"]
    assert fake.submitted == [f"{NEW_ID}/agent-{LIVE_GENERATION}"]
    assert outcome.status is WorkflowStepStatus.WAITING


def test_a_transport_retry_between_the_upgrade_and_an_agent_restart_keeps_the_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy pointer must survive a poll that never reached the agent.

    Old control plane: submitted under ``agent-5`` and recorded that pointer.
    Upgrade. New control plane, poll 1: the agent is mid-restart, the request
    fails to connect, and the TRANSPORT_RETRY record replaces the step's only
    record with a NEW-shape pointer (``record_attempt`` keeps one record per
    step). The agent comes back as generation 6 and has closed the ``agent-5``
    row INTERRUPTED. Poll 2 must still read ``agent-5`` back: guessing only
    ``agent-6`` reads 404 and submits a second install on top of an interrupted
    one that needs an operator.
    """

    operation = WorkflowOperation.REMEDIATE_DRIVER
    recorded = f"{NEW_ID}/agent-5"
    old_record = workflow_step_execution(
        0,
        operation,
        WorkflowStepStatus.WAITING,
        adapter_operation_id=KEY,
        details={"node_action_command_id": recorded, "node_action_state": "PENDING"},
    )

    unreachable_wire(monkeypatch)
    adapter, context = _step_context(operation, [old_record], generation=5)
    first = adapter.execute(context)
    assert first.status is WorkflowStepStatus.WAITING, first
    assert first.details["node_action_state"] == "TRANSPORT_RETRY", first.details
    after_retry = step_bounds.record_attempt(context.workflow, context.step, 0, first)
    assert [
        item.details["node_action_command_id"] for item in after_retry.step_executions
    ] == [NEW_ID], "the retry record replaced the step's record with the new pointer"

    fake = wire(monkeypatch, {recorded: interrupted(recorded, operation)})
    adapter, context = _step_context(
        operation, after_retry.step_executions, generation=6
    )
    second = adapter.execute(context)

    assert fake.submitted == [], (
        "a second install was submitted on top of the interrupted agent-5 row"
    )
    assert recorded in fake.polled, fake.polled
    assert fake.polled[0] == NEW_ID, fake.polled
    assert second.status is WorkflowStepStatus.FAILED, second
    assert second.details["manual_confirmation_required"] is True, second.details
    assert second.details["node_action_interrupted"] is True, second.details


def test_every_distinct_legacy_pointer_across_the_steps_records_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regional executor appends records; every old-shape pointer counts.

    Oldest first, deduplicated, then the live-generation guess -- and the
    read-back stops at the first row that answers.
    """

    operation = WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE
    first_pointer = f"{NEW_ID}/agent-4"
    second_pointer = f"{NEW_ID}/agent-5"
    records = [
        workflow_step_execution(
            0,
            operation,
            WorkflowStepStatus.WAITING,
            details={"node_action_command_id": pointer},
        )
        for pointer in (first_pointer, second_pointer, second_pointer)
    ]
    fake = wire(monkeypatch, {second_pointer: in_progress(second_pointer)})
    adapter, context = _step_context(operation, records, generation=LIVE_GENERATION)

    outcome = adapter.execute(context)

    assert fake.polled == [NEW_ID, first_pointer, second_pointer], fake.polled
    assert fake.submitted == []
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_command_id"] == second_pointer


def test_a_retryable_legacy_failure_is_resubmitted_as_attempt_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent has no history for the NEW id, so it runs attempt 1, not 2."""

    operation = WorkflowOperation.REMEDIATE_EFA_DRIVER
    fake = wire(monkeypatch, {LEGACY_ID: retryable_failure(LEGACY_ID, operation)})

    outcome = _poll(operation, previous_pointer=LEGACY_ID)

    assert fake.submitted == [NEW_ID], fake.submitted
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_action_resubmitted"] is True, outcome.details
    assert outcome.details["node_action_legacy_command_id"] == LEGACY_ID
    assert outcome.details["node_action_attempt"] == 1, (
        "the detail must say what the agent will run under the new id"
    )
    assert outcome.details["node_action_last_error"] == "apt lock held"
