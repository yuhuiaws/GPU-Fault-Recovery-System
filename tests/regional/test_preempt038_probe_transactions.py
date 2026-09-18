"""Transaction and ACK-loss tests for the raw-evidence audit, without PostgreSQL."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_raw_evidence_periodic_cleanup as probe
from scripts.e2e.regional import run_preempt038_evidence_pins as runner


class Connection:
    def __init__(self, *, fail_update: bool = False) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.committed: dict[tuple[str, str], dict[str, Any]] = {}
        self.aborted = False
        self.fail_update = fail_update
        self.fail_commit_ack = False
        self.rollback_count = 0

    def cursor(self) -> Cursor:
        return Cursor(self)

    def commit(self) -> None:
        if self.aborted:
            raise RuntimeError("aborted transaction")
        self.committed = copy.deepcopy(self.rows)
        if self.fail_commit_ack:
            self.fail_commit_ack = False
            raise RuntimeError("commit acknowledgement lost")

    def rollback(self) -> None:
        self.rollback_count += 1
        self.aborted = False
        self.rows = copy.deepcopy(self.committed)

    def periodic_delete(self, key: str) -> None:
        self.rows.pop(("raw_evidence", key), None)
        self.committed.pop(("raw_evidence", key), None)


class Cursor:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.result: list[Any] = []
        self.rowcount = 0

    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, query: str, parameters: tuple[Any, ...]) -> None:
        connection = self.connection
        if connection.aborted:
            raise RuntimeError("transaction must be rolled back first")
        if query.startswith("SELECT kind,key,payload"):
            self.result = [
                (kind, key, copy.deepcopy(value))
                for (kind, key), value in connection.rows.items()
                if (kind == "raw_evidence" and key in parameters[:2])
                or (kind == "incident" and key == parameters[2])
            ]
        elif query.startswith("SELECT count(*)"):
            self.result = [(int(("raw_evidence", parameters[0]) in connection.rows),)]
        elif query.startswith("INSERT"):
            kind, key, raw = parameters
            connection.rows[(kind, key)] = json.loads(raw)
            self.rowcount = 1
        elif query.startswith("UPDATE"):
            if connection.fail_update:
                connection.aborted = True
                raise RuntimeError("injected database statement failure")
            raw, key = parameters
            self.rowcount = int(("incident", key) in connection.rows)
            if self.rowcount:
                connection.rows[("incident", key)] = json.loads(raw)
        elif query.startswith("DELETE"):
            kind, key, raw = parameters
            if connection.rows.get((kind, key)) == json.loads(raw):
                del connection.rows[(kind, key)]
                self.rowcount = 1
        else:
            raise AssertionError(f"unexpected SQL operation: {query.split()[0]}")

    def fetchall(self) -> list[Any]:
        return self.result

    def fetchone(self) -> Any:
        return self.result[0] if self.result else None


@pytest.mark.parametrize("fail_update", [False, True])
def test_audit_cleans_committed_rows_after_success_or_aborted_transaction(
    monkeypatch: pytest.MonkeyPatch, fail_update: bool
) -> None:
    connection = Connection(fail_update=fail_update)
    sentinel = ("incident", "unrelated-owner")
    connection.rows[sentinel] = {"cluster_id": "other-cluster"}
    connection.commit()

    def sweep(db: Connection, key: str) -> float:
        db.periodic_delete(key)
        return 5.0

    monkeypatch.setattr(probe, "_wait_deleted", sweep)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)

    result = probe.audit(connection, "review-00000001")

    assert result["verdict"] == ("FAIL" if fail_update else "PASS")
    assert result["residual_rows"] == 0
    assert connection.rows == {sentinel: {"cluster_id": "other-cluster"}}
    assert connection.rollback_count >= 2


def test_cleanup_refuses_a_rebound_row_without_deleting_anything() -> None:
    connection = Connection()
    identity = probe.audit_identity("review-00000002")
    key = ("raw_evidence", identity["pinned_key"])
    connection.rows[key] = {"cluster_id": "other-cluster"}
    connection.commit()
    with pytest.raises(RuntimeError, match="ownership changed"):
        probe.cleanup_rows(connection, identity)
    assert connection.rows == {key: {"cluster_id": "other-cluster"}}


def test_seed_commit_ack_loss_is_rolled_back_before_exact_cleanup() -> None:
    connection = Connection()
    connection.fail_commit_ack = True
    result = probe.audit(connection, "review-00000004")
    assert result["verdict"] == "FAIL"
    assert result["residual_rows"] == 0
    assert result["error"] == "audit failed: RuntimeError"
    assert connection.rows == {}
    assert connection.rollback_count >= 2


def test_preexisting_identity_is_not_overwritten_or_cleaned() -> None:
    connection = Connection()
    identity = probe.audit_identity("review-00000003")
    sentinel = ("incident", identity["incident_id"])
    connection.rows[sentinel] = {"cluster_id": "foreign"}
    connection.commit()
    result = probe.audit(connection, identity["run_id"])
    assert result["verdict"] == "FAIL"
    assert result["cleanup_permitted"] is False
    assert connection.rows == {sentinel: {"cluster_id": "foreign"}}


def test_runner_recovers_cleanup_identity_after_lost_audit_ack(tmp_path: Path) -> None:
    calls: list[tuple[Any, ...]] = []

    class Regional:
        def pod_python(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["attempts"] == 1, "a mutating audit must not be replayed"
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError("audit response lost")
            return {
                **probe.audit_identity(str(args[3])),
                "verdict": "PASS",
                "residual_rows": 0,
            }

    with pytest.raises(RuntimeError, match="audit response lost"):
        runner.execute(Regional(), tmp_path)
    assert len(calls) == 2
    assert calls[0][3] == calls[1][3]
    assert calls[1][-1] == "--cleanup-only"
    persisted = json.loads((tmp_path / "seed-identity.json").read_text())
    assert persisted["run_id"] == calls[0][3]
    assert (
        json.loads((tmp_path / "seed-cleanup.json").read_text())["residual_rows"] == 0
    )


@pytest.mark.parametrize(
    "failure",
    [None, "foreign-report", "foreign-cleanup", "cleanup-error", "boolean-zero"],
)
def test_runner_binds_success_and_cleanup_to_the_caller_seed_identity(
    tmp_path: Path, failure: str | None
) -> None:
    class Regional:
        def __init__(self) -> None:
            self.identity: dict[str, str] = {}

        def pod_python(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["attempts"] == 1
            self.identity = probe.audit_identity(str(args[3]))
            result: dict[str, Any] = {
                **self.identity,
                "verdict": "PASS",
                "residual_rows": 0,
            }
            if args[-1] == "--cleanup-only":
                if failure == "cleanup-error":
                    raise RuntimeError("independent cleanup unavailable")
                if failure == "foreign-cleanup":
                    result["run_id"] = "foreign-run"
                if failure == "boolean-zero":
                    result["residual_rows"] = False
                return result
            result.update(
                unrelated_deleted_after_seconds=5.0,
                pinned_deleted_after_recovered_seconds=5.0,
                pinned_present_after_unrelated_deleted=True,
                pinned_present_after_extra_wait=True,
            )
            if failure == "foreign-report":
                result["pinned_key"] = "foreign-key"
            return result

        def ready_pods(self, *args: Any) -> list[dict[str, str]]:
            return [{"name": "worker"}]

        def kubectl(self, *args: Any, **kwargs: Any) -> str:
            return (
                "cleanup raw_evidence deleted 2 rows; keys[:20]="
                + self.identity["unrelated_key"]
                + ","
                + self.identity["pinned_key"]
            )

    result = runner.execute(Regional(), tmp_path)
    assert result["verdict"] == ("PASS" if failure is None else "FAIL")
    if failure == "foreign-report":
        assert result["stages"]["identity"]
    elif failure is not None:
        assert result["stages"]["cleanup"]
    if failure == "cleanup-error":
        cleanup = json.loads((tmp_path / "seed-cleanup.json").read_text())
        assert cleanup["verdict"] == "FAIL"


@pytest.mark.parametrize("elapsed", [True, -1, 181, "5", float("nan"), float("inf")])
def test_report_elapsed_times_cannot_be_unknown_or_outside_the_bound(
    elapsed: Any,
) -> None:
    report = {
        "verdict": "PASS",
        "residual_rows": 0,
        "unrelated_deleted_after_seconds": elapsed,
        "pinned_deleted_after_recovered_seconds": elapsed,
        "pinned_present_after_unrelated_deleted": True,
        "pinned_present_after_extra_wait": True,
    }
    assert runner.verdicts.audit_errors(report), (
        f"invalid sweep duration was accepted: {elapsed!r}"
    )


@pytest.mark.parametrize("run_id", ["", "../foreign", "x", "a" * 49])
def test_invalid_audit_identity_is_rejected_before_connecting(run_id: str) -> None:
    with pytest.raises(ValueError, match="identity"):
        probe.audit_identity(run_id)
