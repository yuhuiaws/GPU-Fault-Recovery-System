"""The node action ledger as an audit record, not just an idempotency cache.

One row per ``(command_id, attempt)``: a retry must not overwrite what the
first attempt did. Alongside the result the row keeps who asked (incident,
workflow, fencing token), what was targeted (GPU UUIDs), and digests of the
parameters and the envelope signature -- digests, because the parameters may
name workload paths and the signature is derived from the shared secret.

Existing nodes carry ledgers in the old single-row schema, so the upgrade has
to happen in place on open, with the old rows still readable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionLedger, NodeActionStatus
from gpu_fault.node_agent.ledger import LEDGER_SCHEMA_VERSION
from tests._builders import node_action_result

from ._support import NOW, SECRET, command, envelope

VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_every_attempt_is_kept_and_the_latest_answers_idempotent_reads(
    tmp_path: Path,
) -> None:
    ledger = NodeActionLedger(str(tmp_path / "audit.db"))
    submitted = command(
        VERIFY, fencing_token=4, parameters={"compute_clients_only": True}
    )
    signed = envelope(submitted)

    ledger.mark_in_progress(submitted, 1, signature=signed.signature)
    ledger.save(
        node_action_result(
            submitted.command_id,
            VERIFY,
            NodeActionStatus.FAILED,
            error="OSError: device busy",
            retryable=True,
            attempt=1,
            completed_at=NOW,
        ),
        exit_code=255,
    )
    ledger.mark_in_progress(submitted, 2, signature=signed.signature)
    ledger.save(
        node_action_result(
            submitted.command_id,
            VERIFY,
            attempt=2,
            completed_at=NOW + timedelta(seconds=5),
        )
    )

    latest = ledger.get(submitted.command_id)
    assert latest is not None
    assert latest.status is NodeActionStatus.SUCCEEDED
    assert latest.attempt == 2
    history = ledger.attempt_history(submitted.command_id)
    assert [row["attempt"] for row in history] == [1, 2]
    assert [row["state"] for row in history] == ["FAILED", "SUCCEEDED"]
    first = history[0]
    assert first["exit_code"] == 255
    assert first["incident_id"] == "incident-a"
    assert first["workflow_request_id"] == "workflow-a"
    assert first["fencing_token"] == 4
    assert first["gpu_uuids"] == ["GPU-a"]
    assert first["operation"] == VERIFY.value
    assert first["parameters_digest"] == canonical_digest(
        {"compute_clients_only": True}
    )
    assert (
        first["signature_digest"]
        == hashlib.sha256(signed.signature.encode()).hexdigest()
    )
    assert first["started_at"] is not None
    assert first["completed_at"] is not None
    stored_text = " ".join(
        str(value) for row in history for value in row.values() if value is not None
    )
    assert "compute_clients_only" not in stored_text
    assert signed.signature not in stored_text
    assert SECRET not in stored_text


def test_marking_the_next_attempt_in_progress_hides_the_previous_result(
    tmp_path: Path,
) -> None:
    ledger = NodeActionLedger(str(tmp_path / "progress.db"))
    submitted = command(VERIFY)
    ledger.save(
        node_action_result(
            submitted.command_id,
            VERIFY,
            NodeActionStatus.FAILED,
            retryable=True,
            attempt=1,
        )
    )

    ledger.mark_in_progress(submitted, 2)

    assert ledger.get(submitted.command_id) is None
    assert [row["state"] for row in ledger.attempt_history(submitted.command_id)] == [
        "FAILED",
        "IN_PROGRESS",
    ]


def legacy_ledger(path: Path) -> None:
    """Write a ledger exactly as the pre-audit code left it on disk."""

    db = sqlite3.connect(str(path), isolation_level=None)
    db.execute(
        """
        CREATE TABLE results (
            command_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    db.execute(
        """
        CREATE TABLE fencing (
            incident_id TEXT PRIMARY KEY,
            token INTEGER NOT NULL,
            updated_at TEXT
        )
        """
    )
    db.execute("ALTER TABLE results ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1")
    db.execute("ALTER TABLE results ADD COLUMN state TEXT NOT NULL DEFAULT 'COMPLETED'")
    db.execute("ALTER TABLE results ADD COLUMN operation TEXT")
    db.execute("ALTER TABLE results ADD COLUMN started_at TEXT")
    # Real timestamps: the ledger runs its retention purge on open, against
    # the wall clock, and the fixture must survive it.
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    old = node_action_result(
        "wf-old/1/VERIFY_NO_GPU_CLIENTS/node-a", VERIFY, attempt=2, completed_at=recent
    )
    db.execute(
        """
        INSERT INTO results(command_id, payload, completed_at, attempt, state,
                            operation, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            old.command_id,
            old.model_dump_json(),
            old.completed_at.isoformat(),
            2,
            "SUCCEEDED",
            VERIFY.value,
            (recent - timedelta(seconds=3)).isoformat(),
        ),
    )
    in_progress = node_action_result(
        "wf-old/2/RESET_GPU/node-a",
        WorkflowOperation.RESET_GPU,
        NodeActionStatus.INTERRUPTED,
        error="node action is in progress",
    )
    db.execute(
        """
        INSERT INTO results(command_id, payload, completed_at, attempt, state,
                            operation, started_at)
        VALUES (?, ?, NULL, 1, 'IN_PROGRESS', 'RESET_GPU', ?)
        """,
        (in_progress.command_id, in_progress.model_dump_json(), recent.isoformat()),
    )
    db.execute(
        "INSERT INTO fencing(incident_id, token, updated_at) VALUES (?, ?, ?)",
        ("incident-a", 5, recent.isoformat()),
    )
    db.close()


def test_a_ledger_written_by_the_old_schema_is_migrated_in_place(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    legacy_ledger(path)

    ledger = NodeActionLedger(str(path))

    old = ledger.get("wf-old/1/VERIFY_NO_GPU_CLIENTS/node-a")
    assert old is not None
    assert old.attempt == 2
    assert old.status is NodeActionStatus.SUCCEEDED
    interrupted = ledger.get("wf-old/2/RESET_GPU/node-a")
    assert interrupted is not None
    assert interrupted.status is NodeActionStatus.INTERRUPTED
    assert "agent restarted" in (interrupted.error or "")
    assert ledger.accept_fencing("incident-a", 4) is False, (
        "the fencing table must survive the migration"
    )
    # Writes go through the new per-attempt path on the migrated file.
    fresh = command(VERIFY, command_id="wf-new/1/VERIFY_NO_GPU_CLIENTS/node-a")
    ledger.mark_in_progress(fresh, 1)
    ledger.save(node_action_result(fresh.command_id, VERIFY, attempt=1))
    ledger.mark_in_progress(fresh, 2)
    ledger.save(node_action_result(fresh.command_id, VERIFY, attempt=2))
    assert len(ledger.attempt_history(fresh.command_id)) == 2
    db = sqlite3.connect(str(path))
    columns = list(db.execute("PRAGMA table_info(results)"))
    primary_key = sorted(row[1] for row in columns if row[5] > 0)
    assert primary_key == ["attempt", "command_id"]
    assert "agent_generation" in {row[1] for row in columns}, (
        "a v0 ledger must land on the current schema in one open"
    )
    assert db.execute("PRAGMA user_version").fetchone()[0] == LEDGER_SCHEMA_VERSION


def test_reopening_a_migrated_ledger_does_not_migrate_again(tmp_path: Path) -> None:
    path = tmp_path / "twice.db"
    legacy_ledger(path)
    NodeActionLedger(str(path))
    fresh = command(VERIFY, command_id="wf-new/1/VERIFY_NO_GPU_CLIENTS/node-a")

    reopened = NodeActionLedger(str(path))
    reopened.save(node_action_result(fresh.command_id, VERIFY, attempt=1))

    assert reopened.get(fresh.command_id) is not None
    assert reopened.get("wf-old/1/VERIFY_NO_GPU_CLIENTS/node-a") is not None


def test_retention_defaults_to_thirty_days_and_the_purge_logs_a_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ledger = NodeActionLedger(str(tmp_path / "retention.db"))
    assert ledger.retention_seconds == 30 * 24 * 3600
    ledger.save(
        node_action_result(
            "old", VERIFY, attempt=1, completed_at=NOW - timedelta(days=31)
        )
    )
    ledger.save(node_action_result("new", VERIFY, attempt=1, completed_at=NOW))

    with caplog.at_level(logging.INFO, logger="gpu_fault.node_agent.ledger"):
        removed = ledger.cleanup(now=NOW)

    assert removed["results"] == 1
    assert ledger.get("old") is None
    assert ledger.get("new") is not None
    messages = [record.getMessage() for record in caplog.records]
    assert any("results=1" in message for message in messages), messages


def test_a_writable_ledger_passes_the_health_probe_and_a_closed_one_fails(
    tmp_path: Path,
) -> None:
    ledger = NodeActionLedger(str(tmp_path / "probe.db"))

    ledger.probe_writable()
    ledger.close()

    with pytest.raises(sqlite3.Error):
        ledger.probe_writable()


def test_each_attempt_row_records_the_generation_of_the_agent_that_took_it(
    tmp_path: Path,
) -> None:
    """With the ``agent-N`` id suffix gone, the row is the only audit trail.

    The generation is written when the attempt is opened and kept by ``save``;
    a row written by ``save`` alone (the pool wrapper's failures) has none.
    """

    ledger = NodeActionLedger(str(tmp_path / "generation.db"))
    submitted = command(VERIFY)

    ledger.mark_in_progress(submitted, 1, agent_generation=7)
    ledger.save(
        node_action_result(
            submitted.command_id,
            VERIFY,
            NodeActionStatus.FAILED,
            error="OSError: device busy",
            retryable=True,
            attempt=1,
        )
    )
    ledger.mark_in_progress(submitted, 2, agent_generation=8)
    ledger.save(node_action_result(submitted.command_id, VERIFY, attempt=2))
    ledger.save(node_action_result(submitted.command_id, VERIFY, attempt=3))

    history = ledger.attempt_history(submitted.command_id)
    assert [row["agent_generation"] for row in history] == [7, 8, None], history


def schema_v2_ledger(path: Path) -> str:
    """Write a ledger exactly as the per-attempt (v2) agent left it on disk."""

    db = sqlite3.connect(str(path), isolation_level=None)
    db.execute(
        """
        CREATE TABLE results (
            command_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            payload TEXT NOT NULL,
            completed_at TEXT,
            state TEXT NOT NULL DEFAULT 'COMPLETED',
            operation TEXT,
            started_at TEXT,
            incident_id TEXT,
            workflow_request_id TEXT,
            fencing_token INTEGER,
            gpu_uuids TEXT,
            parameters_digest TEXT,
            signature_digest TEXT,
            exit_code INTEGER,
            PRIMARY KEY (command_id, attempt)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE fencing (
            incident_id TEXT PRIMARY KEY,
            token INTEGER NOT NULL,
            updated_at TEXT
        )
        """
    )
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    row = node_action_result(
        "wf-v2/1/REMEDIATE_DRIVER/node-a",
        WorkflowOperation.REMEDIATE_DRIVER,
        attempt=1,
        completed_at=recent,
    )
    db.execute(
        """
        INSERT INTO results(command_id, attempt, payload, completed_at, state,
                            operation, started_at, incident_id,
                            workflow_request_id, fencing_token, gpu_uuids,
                            parameters_digest, exit_code)
        VALUES (?, 1, ?, ?, 'SUCCEEDED', ?, ?, 'incident-a', 'workflow-a', 3,
                '["GPU-a"]', ?, 0)
        """,
        (
            row.command_id,
            row.model_dump_json(),
            row.completed_at.isoformat(),
            row.operation.value,
            (recent - timedelta(seconds=900)).isoformat(),
            canonical_digest({"target_driver_branch": 575}),
        ),
    )
    db.execute("PRAGMA user_version=2")
    db.close()
    return row.command_id


def test_a_v2_ledger_gains_the_generation_column_and_keeps_its_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2.db"
    legacy_id = schema_v2_ledger(path)

    ledger = NodeActionLedger(str(path))

    old = ledger.get(legacy_id)
    assert old is not None, "the v2 row must survive the migration"
    assert old.status is NodeActionStatus.SUCCEEDED
    (legacy_row,) = ledger.attempt_history(legacy_id)
    assert legacy_row["agent_generation"] is None, (
        f"a row written before the column has no generation to claim: {legacy_row}"
    )
    assert legacy_row["gpu_uuids"] == ["GPU-a"], (
        f"the audit columns must survive the migration: {legacy_row}"
    )
    fresh = command(VERIFY, command_id="wf-new/1/VERIFY_NO_GPU_CLIENTS/node-a")
    ledger.mark_in_progress(fresh, 1, agent_generation=9)
    ledger.save(node_action_result(fresh.command_id, VERIFY, attempt=1))
    (new_row,) = ledger.attempt_history(fresh.command_id)
    assert new_row["agent_generation"] == 9, new_row
    db = sqlite3.connect(str(path))
    assert db.execute("PRAGMA user_version").fetchone()[0] == LEDGER_SCHEMA_VERSION
    assert LEDGER_SCHEMA_VERSION > 2, "the column is a schema change"


def test_reopening_a_v2_migrated_ledger_does_not_migrate_again(tmp_path: Path) -> None:
    path = tmp_path / "v2-twice.db"
    legacy_id = schema_v2_ledger(path)
    NodeActionLedger(str(path))

    reopened = NodeActionLedger(str(path))

    assert reopened.get(legacy_id) is not None
    (legacy_row,) = reopened.attempt_history(legacy_id)
    assert legacy_row["agent_generation"] is None, legacy_row
