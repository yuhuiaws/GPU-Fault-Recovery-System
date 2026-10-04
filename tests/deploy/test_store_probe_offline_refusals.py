"""Offline refusals of the read-only bootstrap store probe.

The probe runs inside a CPU Job against the live database, so every verdict it
can reach must also be reachable here against a scripted cursor: declared table
discovery from the shipped DDL, the partial-schema emptiness proof and each
``ValueError`` the inspection raises before it would ever call a schema ensure.
No database is contacted; ``psycopg`` is imported lazily by the tests that patch its ``connect``.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any

import pytest

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS
from gpu_fault_release import regional_release_store_probe as probe

HEAP = ("r", "heap", False, False, False, False, False, False, False)
TRIGGERED = ("r", "heap", False, False, False, False, False, False, True)
METADATA_TABLES = ("gpu_fault_schema_migrations", "gpu_fault_schema_version")
RECORD_TABLES = (
    "gpu_fault_attempt_observations",
    "gpu_fault_control_records",
    "gpu_fault_control_state_modes",
    "gpu_fault_objects",
    *sorted(layout.table for layout in STATE_LAYOUTS.values()),
)


class ScriptedCursor:
    """Answers SELECTs from a queue; records every statement it was given."""

    def __init__(self, responses: list[list[Any]]) -> None:
        self.responses = list(responses)
        self.statements: list[str] = []
        self.rows: list[Any] = []

    def transaction(self) -> Any:
        return nullcontext()

    def cursor(self) -> Any:
        return nullcontext(self)

    def execute(self, statement: Any, _parameters: Any = None) -> None:
        text = statement if isinstance(statement, str) else statement.as_string()
        self.statements.append(text.strip())
        if not text.strip().startswith("SET "):
            assert text.strip().startswith("SELECT"), (
                "the probe attempted to mutate its database"
            )
            self.rows = self.responses.pop(0)

    def fetchall(self) -> list[Any]:
        return self.rows

    def fetchone(self) -> Any:
        return self.rows[0] if self.rows else None


def relations(*names: str) -> list[tuple[str, str, str]]:
    return [("public", name, "r") for name in names]


def test_declared_table_names_come_from_the_shipped_ddl() -> None:
    declared = probe.declared_table_names()
    assert {"gpu_fault_schema_version", "gpu_fault_objects"}.issubset(declared), (
        "the shipped DDL declares the metadata and object tables"
    )
    assert all(name.startswith("gpu_fault_") for name in declared), (
        "only product tables are declared"
    )


def test_probe_refuses_an_image_whose_schema_identity_differs() -> None:
    cursor = ScriptedCursor([])
    with pytest.raises(ValueError, match="schema identity differs"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION + 1)
    assert cursor.statements == [], "no query may run before the identity check"


def test_probe_refuses_foreign_schemas_and_foreign_tables() -> None:
    cursor = ScriptedCursor([[("audit", "gpu_fault_objects", "r")]])
    with pytest.raises(ValueError, match="unrecognized database schema"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)


def test_partial_schema_without_metadata_is_proven_empty_table_by_table() -> None:
    cursor = ScriptedCursor(
        [
            relations("gpu_fault_objects", "gpu_fault_attempt_observations"),
            [HEAP],
            [HEAP],
            [(False,)],
            [(False,)],
        ]
    )
    verdict = probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)
    assert verdict == {
        "safe": True,
        "database_state": "uninitialized_empty",
        "schema_version": 0,
        "blockers": {"workflow": 0, "remote_command": 0, "observation": 0},
    }
    emptiness = [s for s in cursor.statements if "FROM ONLY" in s and "LIMIT 1" in s]
    assert len(emptiness) == 2, "each heap must be checked for rows"


def test_partial_schema_with_an_undeclared_table_is_refused() -> None:
    cursor = ScriptedCursor([relations("gpu_fault_objects", "gpu_fault_scratch")])
    with pytest.raises(ValueError, match="unrecognized views or tables"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)


def test_partial_schema_with_unverified_triggers_is_refused() -> None:
    cursor = ScriptedCursor([relations("gpu_fault_objects"), [TRIGGERED]])
    with pytest.raises(ValueError, match="unverified triggers"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)


def test_partial_schema_holding_rows_is_not_empty() -> None:
    cursor = ScriptedCursor([relations("gpu_fault_objects"), [HEAP], [(True,)]])
    with pytest.raises(ValueError, match="incomplete schema contains rows"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)


def test_metadata_tables_without_a_version_row_fall_back_to_the_emptiness_proof() -> (
    None
):
    cursor = ScriptedCursor(
        [
            relations(*METADATA_TABLES),
            [HEAP],
            [HEAP],
            [],
            [HEAP],
            [HEAP],
            [(False,)],
            [(False,)],
        ]
    )
    verdict = probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)
    assert verdict["database_state"] == "uninitialized_empty"
    assert verdict["safe"] is True


def history(schema: int) -> list[tuple[int, str, str]]:
    return [
        (item.version, item.name, item.checksum)
        for item in POSTGRES_SCHEMA_MIGRATIONS
        if item.version <= schema
    ]


def test_initialized_schema_missing_record_tables_is_incomplete() -> None:
    schema = LATEST_POSTGRES_SCHEMA_VERSION
    cursor = ScriptedCursor(
        [relations(*METADATA_TABLES), [HEAP], [HEAP], [(schema,)], history(schema)]
    )
    with pytest.raises(ValueError, match="record schema is incomplete"):
        probe.inspect_database(cursor, schema)


def initialized(
    schema: int, *, modes: list[Any], statuses: list[Any]
) -> ScriptedCursor:
    return ScriptedCursor(
        [
            relations(*sorted((*METADATA_TABLES, *RECORD_TABLES))),
            [HEAP],
            [HEAP],
            [(schema,)],
            history(schema),
            [HEAP],
            [HEAP],
            [HEAP],
            [HEAP],
            [HEAP],
            modes,
            statuses,
            [(0,)],
        ]
    )


def test_unknown_control_state_mode_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpu_fault.store.postgres import state_table_definitions

    monkeypatch.setattr(
        state_table_definitions, "validate_state_definitions", lambda _cursor: None
    )
    cursor = initialized(
        LATEST_POSTGRES_SCHEMA_VERSION,
        modes=[("workflow", "legacy"), ("remote_command", "mirrored")],
        statuses=[],
    )
    with pytest.raises(ValueError, match="unknown control-state storage mode"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)


@pytest.mark.parametrize(
    "row", [("incident", "OPEN", 1), ("workflow", "RUNNING", -1)], ids=["kind", "count"]
)
def test_invalid_workflow_evidence_is_refused(
    monkeypatch: pytest.MonkeyPatch, row: tuple[str, str, int]
) -> None:
    from gpu_fault.store.postgres import state_table_definitions

    monkeypatch.setattr(
        state_table_definitions, "validate_state_definitions", lambda _cursor: None
    )
    cursor = initialized(
        LATEST_POSTGRES_SCHEMA_VERSION,
        modes=[("workflow", "legacy"), ("remote_command", "legacy")],
        statuses=[row],
    )
    with pytest.raises(ValueError, match="invalid workflow evidence"):
        probe.inspect_database(cursor, LATEST_POSTGRES_SCHEMA_VERSION)


# --- the Job entrypoint ------------------------------------------------------


def probe_environment(monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
    identity = {
        "endpoint": "db.example",
        "port": 5432,
        "database": "gpu_fault",
        "username": "probe",
        "schema_version": LATEST_POSTGRES_SCHEMA_VERSION,
    }
    values = {
        "GPU_FAULT_PROOF_DATABASE": json.dumps(identity),
        "GPU_FAULT_STORE_URL": (
            "postgresql://probe:fixture-not-a-real-password@db.example:5432/gpu_fault"
            f"?sslmode=verify-full&sslrootcert={probe.CA_PATH}"
        ),
        "GPU_FAULT_PROOF_RUN_ID": "run-1",
        "GPU_FAULT_PROOF_IDENTITY_SHA256": "c" * 64,
        **extra,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_main_emits_an_envelope_from_the_inspected_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg

    probe_environment(monkeypatch, GPU_FAULT_PROOF_RETAINED_SCHEMA_UPGRADE="true")
    connections: list[dict[str, Any]] = []
    cursor = ScriptedCursor([relations("gpu_fault_objects"), [HEAP], [(False,)]])

    def connect(**arguments: Any) -> Any:
        connections.append(arguments)
        return nullcontext(cursor)

    monkeypatch.setattr(psycopg, "connect", connect)
    probe.main()
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["database_state"] == "uninitialized_empty"
    assert envelope["run_id"] == "run-1" and envelope["identity_sha256"] == "c" * 64
    assert "finished_at" in envelope
    assert connections[0]["sslmode"] == "verify-full"
    assert connections[0]["connect_timeout"] == 10
    assert "password" not in capsys.readouterr().out


def test_main_reports_a_binding_mismatch_by_type_and_stage_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg

    probe_environment(monkeypatch, GPU_FAULT_STORE_URL="postgresql://x:y@other/db")
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda **_arguments: pytest.fail("a mismatched binding must not connect"),
    )
    probe.main()
    assert json.loads(capsys.readouterr().out) == {
        "safe": False,
        "error_type": "ValueError",
        "probe_stage": "bind_connection",
    }


def test_main_without_identity_fails_closed_at_the_identity_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_environment(monkeypatch)
    monkeypatch.delenv("GPU_FAULT_PROOF_DATABASE")
    assert "GPU_FAULT_PROOF_DATABASE" not in os.environ
    probe.main()
    assert json.loads(capsys.readouterr().out) == {
        "safe": False,
        "error_type": "KeyError",
        "probe_stage": "read_identity",
    }
