"""Every writer of one remote command serialises on the same advisory lock.

Store review 2026-09-07, item A. ``complete_remote_command``,
``renew_remote_command_lease`` and ``claim_remote_commands`` take
``remote_command/<id>``; the two cancel paths took ``remote_command/<id>/timeout``
and ``remote_command/<id>/cancel``. All of them read the row without a row lock
and write it back whole, so a cancel that overlapped a completion could put a
SUCCEEDED command back to LEASED with ``cancellation_requested_at`` set. The
cancel paths now take the shared key, which this test checks from the outside:
while another session holds ``remote_command/<id>``, a cancel must wait.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from gpu_fault.remote_command_models import RemoteCommandStatus
from tests.store._postgres_processor_claim_support import (  # noqa: F401
    POSTGRES_URL,
    _command,
    _reload,
    store,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)


def _hold_command_lock(command_id: str):
    import psycopg

    connection = psycopg.connect(POSTGRES_URL, autocommit=False)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"remote_command/{command_id}",),
        )
    return connection


def _blocks_until_released(action, connection, *, settle_seconds: float = 0.5):
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = action()
        except Exception as exc:  # noqa: BLE001 - reported to the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(settle_seconds)
    blocked = worker.is_alive()
    connection.commit()
    connection.close()
    worker.join(30)
    assert not worker.is_alive(), "cancel never finished after the lock was released"
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return blocked, outcome["result"]


def test_workflow_cancel_waits_for_the_per_command_lock(store):  # noqa: F811
    command = _command(store, "remote-lock-key-a", request_id="workflow-lock-a")
    connection = _hold_command_lock(command.command_id)

    blocked, result = _blocks_until_released(
        lambda: store.cancel_remote_commands_for_workflow(
            command.workflow_request_id, reason="workflow timed out"
        ),
        connection,
    )

    assert blocked, "cancel_remote_commands_for_workflow did not wait for the lock"
    assert result == {"cancelled": 1, "cancellation_requested": 0}
    assert _reload(store, command.command_id).status is RemoteCommandStatus.FAILED


def test_single_cancel_waits_for_the_per_command_lock(store):  # noqa: F811
    command = _command(store, "remote-lock-key-b", request_id="workflow-lock-b")
    connection = _hold_command_lock(command.command_id)

    blocked, result = _blocks_until_released(
        lambda: store.cancel_remote_command(
            command.command_id, reason="workflow preempted"
        ),
        connection,
    )

    assert blocked, "cancel_remote_command did not wait for the lock"
    assert result is True
    assert _reload(store, command.command_id).status is RemoteCommandStatus.FAILED


def test_completion_and_cancel_never_interleave_on_one_command(store):  # noqa: F811
    """A completion that lands while a cancel is pending is not overwritten.

    The cancel reads under the same lock the completion held, so it sees the
    terminal state and leaves the row alone.
    """

    command = _command(store, "remote-lock-key-c", request_id="workflow-lock-c")
    leased = store.claim_remote_commands(
        command.cluster_id, "executor-a", limit=1, lease_seconds=60
    )[0]
    connection = _hold_command_lock(command.command_id)
    from gpu_fault.regional import RemoteCommandResult

    def complete_then_release() -> None:
        time.sleep(0.2)
        connection.commit()
        connection.close()
        store.complete_remote_command(
            command.cluster_id,
            command.command_id,
            RemoteCommandResult(
                lease_token=leased.lease_token,
                status=RemoteCommandStatus.SUCCEEDED,
                status_source="executor",
            ),
        )

    completer = threading.Thread(target=complete_then_release)
    completer.start()
    result = store.cancel_remote_commands_for_workflow(
        command.workflow_request_id, reason="workflow timed out"
    )
    completer.join(30)

    current = _reload(store, command.command_id)
    if result == {"cancelled": 0, "cancellation_requested": 0}:
        assert current.status is RemoteCommandStatus.SUCCEEDED
    else:
        assert result == {"cancelled": 0, "cancellation_requested": 1}
        assert current.status is RemoteCommandStatus.FAILED
        assert current.status_source == "completed-after-cancellation"
