from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from gpu_fault.store import SqliteStore
from tests.store.test_state_table_payload import command


def test_sqlite_claim_rechecks_cancellation_after_its_candidate_scan(
    tmp_path, monkeypatch
) -> None:
    connections: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def capture_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", capture_connection)
    path = str(tmp_path / "claim-cancellation.db")
    claimer = SqliteStore(path)
    canceller = SqliteStore(path)
    cancellations = []
    value = command(datetime.now(UTC))
    try:
        claimer.ensure_remote_command(value)
        first = claimer.claim_remote_commands(
            value.cluster_id, "first-executor", limit=1, lease_seconds=-1
        )[0]

        def cancel_before_claim_transaction(statement: str) -> None:
            if statement == "BEGIN IMMEDIATE" and not cancellations:
                cancellations.append(
                    canceller.cancel_remote_commands_for_workflow(
                        value.workflow_request_id, reason="workflow stopped"
                    )
                )

        connections[0].set_trace_callback(cancel_before_claim_transaction)
        claimed = claimer.claim_remote_commands(
            value.cluster_id, "second-executor", limit=1, lease_seconds=60
        )
        connections[0].set_trace_callback(None)
        assert cancellations == [{"cancelled": 0, "cancellation_requested": 1}]
        assert claimed == [], "claim reissued a command cancelled after selection"
        current = claimer.get_remote_command(value.command_id)
        assert current.lease_owner == first.lease_owner
        assert current.lease_token == first.lease_token
        assert current.cancellation_reason == "workflow stopped"
    finally:
        connections[0].set_trace_callback(None)
        claimer.close()
        canceller.close()
