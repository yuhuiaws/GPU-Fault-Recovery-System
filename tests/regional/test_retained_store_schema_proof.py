"""Offline cursor protocol coverage; no database connection or schema mutation."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.store.postgres import state_table_definitions
from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS
from gpu_fault_release.regional_release_store_probe import inspect_database


class ReadOnlyDatabase:
    def __init__(self, *, schema=17, statuses=(), observation=0, drift=None):
        tables = {
            "gpu_fault_schema_version",
            "gpu_fault_schema_migrations",
            "gpu_fault_control_records",
            "gpu_fault_objects",
            "gpu_fault_attempt_observations",
            "gpu_fault_control_state_modes",
            *(layout.table for layout in STATE_LAYOUTS.values()),
        }
        heap = [("r", "heap", False, False, False, False, False, False, True)]
        history = [
            (item.version, item.name, item.checksum)
            for item in POSTGRES_SCHEMA_MIGRATIONS
            if item.version <= schema
        ]
        if drift == "history":
            history = history[:-1]
        self.responses = iter(
            [
                [
                    (
                        "public",
                        table,
                        "v" if table == "gpu_fault_control_records" else "r",
                    )
                    for table in sorted(tables)
                ],
                heap,
                heap,
                [(schema,)],
                history,
                heap,
                [("r", "heap", True, False, False, False, False, False, True)]
                if drift == "rls"
                else heap,
                heap,
                heap,
                heap,
                [("workflow", "legacy"), ("remote_command", "legacy")],
                list(statuses),
                [(observation,)],
            ]
        )
        self.rows = []
        self.commands = []
        self.history = {"incident": {"state": "QUARANTINED"}, "audit": ["preserved"]}

    def transaction(self):
        return nullcontext()

    def cursor(self):
        return nullcontext(self)

    def execute(self, statement, _parameters=None):
        command = statement if isinstance(statement, str) else statement.as_string()
        self.commands.append(command.strip())
        if not command.strip().startswith("SET "):
            assert command.strip().startswith("SELECT"), (
                "Store proof attempted to mutate its database"
            )
            self.rows = next(self.responses)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


def test_retained_v17_history_uses_physical_reads_before_candidate_v18_ensure(
    monkeypatch,
):
    assert LATEST_POSTGRES_SCHEMA_VERSION == 18, "this compatibility pair needs review"
    monkeypatch.setattr(
        state_table_definitions,
        "validate_state_definitions",
        lambda _cursor: pytest.fail(
            "candidate definitions were required before ensure"
        ),
    )
    database = ReadOnlyDatabase(
        statuses=[
            ("workflow", "SUCCEEDED", 2),
            ("workflow", "FAILED", 1),
            ("workflow", "SUPERSEDED", 1),
            ("remote_command", "FAILED", 2),
        ]
    )
    proof = inspect_database(database, 18, allow_schema_upgrade=True)
    assert proof == {
        "safe": True,
        "database_state": "initialized",
        "schema_version": 17,
        "schema_ensure_required": True,
        "blockers": {"workflow": 0, "remote_command": 0, "observation": 0},
    }
    assert database.history == {
        "incident": {"state": "QUARANTINED"},
        "audit": ["preserved"],
    }
    assert (
        database.commands[0]
        == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )


@pytest.mark.parametrize("schema", [16, 17, 19])
def test_only_reviewed_retained_upgrade_pair_is_allowed(schema):
    database = ReadOnlyDatabase(schema=schema)
    with pytest.raises(ValueError, match="schema version differs"):
        inspect_database(database, 18, allow_schema_upgrade=schema != 17)


@pytest.mark.parametrize("drift", ["history", "rls"])
def test_retained_schema_proof_rejects_migration_and_execution_policy_drift(drift):
    database = ReadOnlyDatabase(drift=drift)
    with pytest.raises(ValueError, match="migration history|execution policy"):
        inspect_database(database, 18, allow_schema_upgrade=True)


@pytest.mark.parametrize(
    "kind,status",
    [
        ("workflow", "BLOCKED"),
        ("workflow", "UNKNOWN"),
        ("remote_command", "LEASED"),
        ("remote_command", None),
    ],
)
def test_old_schema_still_blocks_live_or_unknown_records(kind, status):
    proof = inspect_database(
        ReadOnlyDatabase(statuses=[(kind, status, 1)]), 18, allow_schema_upgrade=True
    )
    assert proof["safe"] is False
    assert proof["blockers"][kind] == 1


def test_old_schema_still_blocks_active_observation():
    proof = inspect_database(
        ReadOnlyDatabase(observation=1), 18, allow_schema_upgrade=True
    )
    assert proof["safe"] is False and proof["blockers"]["observation"] == 1


def test_current_schema_still_requires_full_candidate_definition_validation(
    monkeypatch,
):
    checked = []
    monkeypatch.setattr(
        state_table_definitions,
        "validate_state_definitions",
        lambda cursor: checked.append(cursor),
    )
    database = ReadOnlyDatabase(schema=18)
    proof = inspect_database(database, 18, allow_schema_upgrade=True)
    assert proof["safe"] is True and proof["schema_ensure_required"] is False
    assert checked == [database]
