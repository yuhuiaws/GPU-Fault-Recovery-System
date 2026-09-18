"""Hermetic PostgreSQL query/locked-read wiring; not a native SQL execution test."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from gpu_fault.regional_compatibility import command_protocol_eligible
from gpu_fault.store.postgres.remote_commands import PostgresRemoteCommandMixin
from tests.store.test_activation_inhibition_claims import command


class Cursor:
    def __init__(self, database):
        self.database = database
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params):
        database = self.database
        database.calls.append((query, params))
        if "FROM gpu_fault_remote_command_records AS cmd" in query:
            restricted = database.protocol < 4
            assert ("$.step.parameters.activation_forbidden" in query) is restricted
            assert (
                "$.batched_steps[*].step.parameters.activation_forbidden" in query
            ) is restricted
            if restricted:
                assert query.index("$.batched_steps") < query.index("LIMIT")
            selected = [
                key
                for key, value in sorted(database.records.items())
                if command_protocol_eligible(value, database.protocol)
            ]
            self.rows = [(key,) for key in selected[: params[-1]]]
        elif "pg_advisory_xact_lock" in query:
            if database.after_lock is not None:
                database.after_lock(database.records)
            self.rows = []
        else:
            assert "WHERE kind='remote_command' AND key=ANY" in query
            self.rows = [(database.records[key],) for key in params[0]]

    def fetchall(self):
        return self.rows


class Database:
    def __init__(self, records, protocol):
        self.records = {record.command_id: record for record in records}
        self.protocol = protocol
        self.calls = []
        self.after_lock = None

    def transaction(self):
        return nullcontext()

    def cursor(self):
        return Cursor(self)


class Store(PostgresRemoteCommandMixin):
    def __init__(self, records, protocol):
        self._db = Database(records, protocol)
        self.database = self._db
        self.writes = []

    def _decode(self, kind, payload):
        assert kind == "remote_command"
        return payload

    def _get_optional(self, kind, key):
        assert kind == "workflow"
        return None

    def _put(self, kind, key, value):
        assert kind == "remote_command"
        self.writes.append(key)
        self._db.records[key] = value


@pytest.mark.parametrize("protocol", [1, 2, 3, 4])
@pytest.mark.parametrize("marker", [True, False, None])
@pytest.mark.parametrize("batched", [False, True])
def test_postgres_gate_precedes_limit_without_changing_ordinary_claims(
    protocol, marker, batched
):
    store = Store(
        [command("a-guarded", marker, batched=batched), command("b-ordinary")], protocol
    )
    result = store.claim_remote_commands(
        "cluster-a",
        "executor",
        limit=1,
        lease_seconds=60,
        executor_protocol_version=protocol,
    )
    assert [item.command_id for item in result] == [
        "a-guarded" if protocol == 4 else "b-ordinary"
    ]
    assert store.writes == [item.command_id for item in result]


def test_postgres_rechecks_inhibition_after_advisory_lock():
    store = Store([command("changed")], 3)
    store.database.after_lock = lambda rows: rows.update(
        changed=command("changed", True, batched=True)
    )
    result = store.claim_remote_commands(
        "cluster-a", "old", limit=1, lease_seconds=60, executor_protocol_version=3
    )
    assert result == []
    assert store.writes == []
