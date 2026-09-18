"""Raw-retention audit with an in-memory SQL transport and virtual sweeps."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_raw_evidence_periodic_cleanup as audit
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401
from tests.regional.test_preempt038_probe_transactions import Connection, Cursor

RUN_ID = "coverage-00000001"


@pytest.fixture
def database(monkeypatch: Any) -> Any:
    clock = Clock()
    connection = Connection()
    identity = audit.audit_identity(RUN_ID)

    def sweep(seconds: float) -> None:
        if not getattr(connection, "keep_unrelated", False):
            connection.periodic_delete(identity["unrelated_key"])
        if getattr(connection, "lose_pin", False) and seconds == audit.POLL_SECONDS * 3:
            connection.periodic_delete(identity["pinned_key"])
        incident = connection.rows.get(("incident", identity["incident_id"])) or {}
        if incident.get("state") == "RECOVERED" and not getattr(
            connection, "keep_pin", False
        ):
            connection.periodic_delete(identity["pinned_key"])

    clock.on_sleep = sweep
    monkeypatch.setattr(audit, "time", clock)
    return SimpleNamespace(connection=connection, identity=identity, clock=clock)


@pytest.mark.parametrize(
    "failure",
    [None, "unrelated-timeout", "pin-lost", "pin-timeout", "update", "commit-ack"],
)
def test_retention_audit_tracks_pin_then_cleans_every_owned_row(
    database: Any, failure: str | None
) -> None:
    connection = database.connection
    connection.keep_unrelated = failure == "unrelated-timeout"
    connection.keep_pin = failure == "pin-timeout"
    connection.lose_pin = failure == "pin-lost"
    connection.fail_update = failure == "update"
    connection.fail_commit_ack = False
    sentinel = ("incident", "foreign")
    connection.rows[sentinel] = {"cluster_id": "not-owned"}
    connection.commit()
    if failure == "commit-ack":
        connection.fail_commit_ack = True
    result = audit.audit(connection, RUN_ID)
    assert result["verdict"] == ("PASS" if failure is None else "FAIL")
    assert result["residual_rows"] == 0
    assert connection.rows == {sentinel: {"cluster_id": "not-owned"}}
    assert connection.rollback_count >= 2
    if failure is None:
        assert result["unrelated_deleted_after_seconds"] == 5
        assert result["pinned_present_after_extra_wait"] is True
        assert result["pinned_deleted_after_recovered_seconds"] == 5


@pytest.mark.parametrize("count", [None, (True,), (-1,), ("1",)])
def test_unknown_count_fails_audit_without_leaving_seed_rows(
    database: Any, monkeypatch: Any, count: Any
) -> None:
    class BadCount(Cursor):
        def fetchone(self) -> Any:
            return count

    monkeypatch.setattr(
        database.connection, "cursor", lambda: BadCount(database.connection)
    )
    result = audit.audit(database.connection, RUN_ID)
    assert result["verdict"] == "FAIL"
    assert result["residual_rows"] == 0


def test_missing_pinned_row_and_unpersisted_incident_recovery_are_refused(
    database: Any, monkeypatch: Any
) -> None:
    class MissingPin(Cursor):
        def execute(self, query: str, parameters: Any) -> None:
            super().execute(query, parameters)
            if (
                query.startswith("SELECT count")
                and parameters[0] == database.identity["pinned_key"]
            ):
                self.result = [(0,)]

    monkeypatch.setattr(
        database.connection, "cursor", lambda: MissingPin(database.connection)
    )
    result = audit.audit(database.connection, RUN_ID)
    assert result["verdict"] == "FAIL"
    assert result["pinned_present_after_unrelated_deleted"] is False
    assert result["residual_rows"] == 0


def test_recovery_update_must_affect_exactly_one_owned_incident(
    database: Any, monkeypatch: Any
) -> None:
    class LostUpdate(Cursor):
        def execute(self, query: str, parameters: Any) -> None:
            super().execute(query, parameters)
            if query.startswith("UPDATE"):
                self.rowcount = 0

    monkeypatch.setattr(
        database.connection, "cursor", lambda: LostUpdate(database.connection)
    )
    result = audit.audit(database.connection, RUN_ID)
    assert result["verdict"] == "FAIL"
    assert "error" in result, "unpersisted RECOVERED state must fail the audit"
    assert result["residual_rows"] == 0


@pytest.mark.parametrize(
    "change", ["cluster", "node", "payload", "incident", "event", "nonobject"]
)
def test_cleanup_refuses_changed_row_ownership_before_any_delete(
    database: Any, change: str
) -> None:
    identity, connection = database.identity, database.connection
    if change in {"incident", "event"}:
        kind, key = "incident", identity["incident_id"]
        payload = {
            "cluster_id": audit.CLUSTER_ID,
            "incident_id": key,
            "event_type": "AUDIT_EVIDENCE_PIN",
            "node_ids": [identity["pinned_node"]],
        }
        payload["incident_id" if change == "incident" else "event_type"] = "other"
    else:
        kind, key = "raw_evidence", identity["unrelated_key"]
        payload = {
            "cluster_id": audit.CLUSTER_ID,
            "record_id": key,
            "node_id": identity["unrelated_node"],
            "payload": {"audit": True},
        }
        if change == "cluster":
            payload["cluster_id"] = "other"
        elif change == "node":
            payload["node_id"] = "other"
        elif change == "payload":
            payload["payload"] = {}
        else:
            payload = []
    connection.rows[(kind, key)] = payload
    connection.commit()
    with pytest.raises(RuntimeError, match="ownership"):
        audit.cleanup_rows(connection, identity)
    assert connection.rows == {(kind, key): payload}


def test_json_payloads_cleanup_and_missing_ack_are_observable(
    database: Any, monkeypatch: Any
) -> None:
    connection, identity = database.connection, database.identity
    key = identity["pinned_key"]
    payload = {
        "cluster_id": audit.CLUSTER_ID,
        "record_id": key,
        "node_id": identity["pinned_node"],
        "payload": {"audit": True},
    }
    connection.rows[("raw_evidence", key)] = payload
    connection.commit()

    class JsonRows(Cursor):
        def fetchall(self) -> list[Any]:
            return [(kind, key, json.dumps(value)) for kind, key, value in self.result]

    monkeypatch.setattr(connection, "cursor", lambda: JsonRows(connection))
    assert audit.cleanup_rows(connection, identity) == {"residual_rows": 0}
    connection.rows[("raw_evidence", key)] = deepcopy(payload)
    connection.commit()
    result = audit.audit(connection, RUN_ID)
    assert result["cleanup_permitted"] is False
    assert connection.rows == {("raw_evidence", key): payload}


def test_cleanup_transport_failure_is_not_a_pass(
    database: Any, monkeypatch: Any
) -> None:
    def fail(*args: Any) -> Any:
        raise RuntimeError("cleanup unavailable")

    monkeypatch.setattr(audit, "cleanup_rows", fail)
    result = audit.audit(database.connection, RUN_ID)
    assert result["verdict"] == "FAIL"
    assert "cleanup_error" in result
    assert database.connection.rows, "unconfirmed cleanup must remain visible"


@pytest.mark.parametrize(
    "cleanup_only,remaining", [(False, False), (True, False), (True, True)]
)
def test_main_uses_fake_connection_and_returns_json_verdict_even_for_residuals(
    database: Any, monkeypatch: Any, capsys: Any, cleanup_only: bool, remaining: bool
) -> None:
    connection = database.connection

    class Session:
        def __enter__(self) -> Connection:
            return connection

        def __exit__(self, *args: Any) -> None:
            return None

    calls = []
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "example-in-memory-transport")
    monkeypatch.setattr(
        "psycopg.connect", lambda value: calls.append(value) or Session()
    )
    if remaining:
        monkeypatch.setattr(audit, "cleanup_rows", lambda *a: {"residual_rows": 1})
    args = [RUN_ID, *(["--cleanup-only"] if cleanup_only else [])]
    assert audit.main(args) is None
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == ("FAIL" if remaining else "PASS")
    assert len(calls) == 1
    with pytest.raises(ValueError, match="invalid"):
        audit.main(["invalid"])
    assert len(calls) == 1, "invalid run identity must fail before connection"
