"""Attempt bookkeeping on the node agent: one failure, one row, one claimant.

Three ways the ledger's attempt view drifted from what actually happened:

* the app's pool wrapper closed one pre-dispatch failure twice -- once from the
  poll, once from the pool's done-callback -- and saved attempt 1 then attempt 2
  for it, spending two of the control plane's ``node_action_retry_limit`` slots;
* those wrapper rows carry no audit columns, and ``_reject_command_id_reuse``
  compared only the newest row, so body validation for the command_id was lost
  the moment one wrapper failure landed on top of the attempt that ran;
* a resubmitted reset refused by the quiesce window was told the window was
  "already claimed by <its own command_id>", which reads as a contradiction.
"""

from __future__ import annotations

import sqlite3
from threading import Event, current_thread

import pytest

from tests._builders import copy_model, node_action_result

from ._support import (
    FakeRunner,
    NodeActionStatus,
    ServiceRunner,
    SignedNodeAction,
    TestClient,
    WorkflowOperation,
    command,
    create_node_agent_app,
    envelope,
    executor,
    quiesce_executor,
    quiesce_manager,
    result_params,
    submit_action,
)

POOL_THREAD_PREFIX = "gpu-fault-node-action"


def test_a_pre_dispatch_failure_closed_by_both_callers_spends_one_attempt(
    tmp_path, monkeypatch
) -> None:
    """The poll and the done-callback close the same future; one row results.

    ``execute`` fails before anything is dispatched (no IN_PROGRESS row), so the
    first closer saves a retryable FAILED at attempt 1. The second closer used
    to find that row, see it was retryable, and save the *same* failure again at
    attempt 2 -- so a single ledger hiccup cost two retry slots and the third
    real failure was final instead of the fourth.
    """

    monkeypatch.delenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_CLUSTER_ID", raising=False)
    agent = executor(tmp_path, FakeRunner())
    real_save = agent.ledger.save
    submit_returned = Event()
    first_row_saved = Event()
    poll_answered = Event()

    def fail_before_dispatch(value: SignedNodeAction):
        # Hold until the submit has registered the done-callback, so the pool
        # thread -- not the submit handler -- is the one that runs it.
        submit_returned.wait(timeout=5)
        raise sqlite3.OperationalError("database is locked")

    def save_then_hold_the_callback(result, **kwargs):
        real_save(result, **kwargs)
        first_row_saved.set()
        if current_thread().name.startswith(POOL_THREAD_PREFIX):
            # The done-callback has saved its row but not yet dropped the
            # future; the poll now has to close the same failure a second time.
            poll_answered.wait(timeout=5)

    monkeypatch.setattr(agent, "execute", fail_before_dispatch)
    monkeypatch.setattr(agent.ledger, "save", save_then_hold_the_callback)
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))
    command_id = signed.command.command_id

    with TestClient(create_node_agent_app(agent, heartbeat_reporter=None)) as client:
        submit_action(client, signed)
        submit_returned.set()
        assert first_row_saved.wait(timeout=5), "the failure was never saved"
        polled = client.get("/v1/node-actions/result", params=result_params(command_id))
        poll_answered.set()

    assert polled.status_code == 200, f"the poll must answer: {polled.text}"
    payload = polled.json()
    assert payload["result"]["retryable"] is True, payload
    assert payload["result"]["attempt"] == 1, (
        f"one failure was closed twice and counted as two attempts: {payload}"
    )
    history = agent.ledger.attempt_history(command_id)
    assert [row["attempt"] for row in history] == [1], (
        f"one pre-dispatch failure must leave exactly one row: {history}"
    )


def test_body_validation_survives_a_wrapper_row_without_audit_columns(tmp_path) -> None:
    """The newest row that carries targets decides whether the id was reused.

    The pool wrapper saves its failures with ``save`` alone -- no operation
    targets, no parameter digest. Comparing a different body against that row
    found nothing to disagree with, and the old attempt's SUCCEEDED result for
    GPU-a could again be replayed for a command naming GPU-b.
    """

    agent = executor(tmp_path, FakeRunner())
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))
    first = agent.execute(signed)
    assert first.status is NodeActionStatus.SUCCEEDED, first.error
    # What the app's pool wrapper writes for a retryable failure after the
    # attempt that ran: a bare result row, audit columns NULL.
    agent.ledger.save(
        node_action_result(
            signed.command.command_id,
            signed.command.operation,
            NodeActionStatus.FAILED,
            error="OperationalError: database is locked",
            retryable=True,
            attempt=2,
        )
    )
    other_targets = envelope(copy_model(signed.command, gpu_uuids=["GPU-b"]))

    with pytest.raises(ValueError, match="command_id reused"):
        agent.validate_submission(other_targets)


def test_a_refused_resubmit_names_the_earlier_attempt_not_its_own_id(tmp_path) -> None:
    """A second claim by the same command_id is its own earlier attempt.

    The one reset a quiesce window allows was claimed by attempt 1 of this
    command, whose outcome is unknown. Attempt 2 is refused -- correctly -- but
    the refusal quoted the command its own id as the claimant.
    """

    manager = quiesce_manager(tmp_path, ServiceRunner(active={"kubelet"}))
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")
    manager.assert_quiesced(
        incident_id="incident-a", command_id="workflow/r1/node-a", attempt=1
    )

    with pytest.raises(RuntimeError) as refused:
        manager.assert_quiesced(
            incident_id="incident-a", command_id="workflow/r1/node-a", attempt=2
        )
    with pytest.raises(RuntimeError) as refused_other:
        manager.assert_quiesced(
            incident_id="incident-a", command_id="workflow/r2/node-a", attempt=1
        )

    assert "an earlier attempt of this command (see attempt 1)" in str(refused.value), (
        str(refused.value)
    )
    assert "claimed by workflow/r1/node-a" not in str(refused.value), (
        f"the refusal quotes the caller its own command_id: {refused.value}"
    )
    assert "claimed by workflow/r1/node-a" in str(refused_other.value), (
        f"another command must still be told who holds the window: "
        f"{refused_other.value}"
    )


def test_a_ledger_read_failure_before_the_claim_does_not_fail_the_reset(
    tmp_path, monkeypatch
) -> None:
    """The attempt number only feeds the refusal text; it must not gate the reset.

    ``_claim_reset_window`` reads the attempt from the ledger inside the handler.
    With ``sqlite3.OperationalError`` no longer a retryable handler error, a read
    that failed there closed the attempt as a terminal FAILED -- and the ladder
    rebooted a node whose GPU nobody had touched.
    """

    runner = ServiceRunner(active={"kubelet"})
    agent = quiesce_executor(tmp_path, runner)
    agent.execute(envelope(command(WorkflowOperation.QUIESCE_GPU_SERVICES)))
    real_latest_row = agent.ledger.latest_row
    real_mark_in_progress = agent.ledger.mark_in_progress
    armed = Event()

    def arm_after_dispatch(*args, **kwargs):
        # The attempt is on disk: from here on every ledger read fails, which
        # is the handler's read and nothing before it.
        real_mark_in_progress(*args, **kwargs)
        armed.set()

    def failing_read(command_id: str):
        if armed.is_set():
            raise sqlite3.OperationalError("database is locked")
        return real_latest_row(command_id)

    monkeypatch.setattr(agent.ledger, "mark_in_progress", arm_after_dispatch)
    monkeypatch.setattr(agent.ledger, "latest_row", failing_read)

    result = agent.execute(
        envelope(command(WorkflowOperation.RESET_GPU, command_id="workflow/r1/node-a"))
    )

    assert result.status is NodeActionStatus.SUCCEEDED, (
        "a ledger read that only feeds a refusal text failed the reset: "
        f"{result.status.value} retryable={result.retryable} error={result.error}"
    )
    assert len([item for item in runner.commands if "--gpu-reset" in item]) == 1, (
        runner.commands
    )
