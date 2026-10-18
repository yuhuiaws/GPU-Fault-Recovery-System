"""The three Postgres audit scripts, driven offline against scripted fakes.

``audit_q113_q129_postgres``, ``audit_q05_q06_postgres`` and
``audit_processor_order_production`` are one-shot probes that normally need the
control-plane database. What CI can own is the harness around the queries: how
the DSN is resolved (file first, env fallback, refusals on empty input), that
every record the probe writes is deleted again even when an assertion fails,
and that the ordering audit gives up at its deadline instead of spinning.
"""

from __future__ import annotations

import base64
import json
import runpy
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import pytest

import gpu_fault.store
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    NodeMarker,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepStatus,
)
from scripts.e2e.regional import audit_q05_q06_postgres as q05_q06
from scripts.e2e.regional import audit_q113_q129_postgres as q113
from tests._script_loader import LazyScriptModule, lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "e2e" / "regional"
# Loaded lazily: the ordering audit imports ``psycopg`` at module scope, and
# the suite must still collect when the optional driver is absent.
processor_order = lazy_script_module(SCRIPTS / "audit_processor_order_production.py")
FAKE_DSN = "postgresql://audit:placeholder@db.invalid:5432/gpu_fault"
DSN_MODULES = pytest.mark.parametrize(
    "module",
    [q113, q05_q06, processor_order],
    ids=[
        "audit_q113_q129_postgres",
        "audit_q05_q06_postgres",
        "audit_processor_order_production",
    ],
)


# --------------------------------------------------------------------------- #
# store_dsn: the same resolution rules in every script
# --------------------------------------------------------------------------- #
@DSN_MODULES
def test_store_dsn_prefers_the_configured_file(
    module: ModuleType | LazyScriptModule,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text(f"  {FAKE_DSN}\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://env-must-lose")

    assert module.store_dsn() == FAKE_DSN


@DSN_MODULES
def test_store_dsn_refuses_an_empty_file_path(
    module: ModuleType | LazyScriptModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", FAKE_DSN)

    with pytest.raises(RuntimeError, match="file path is empty"):
        module.store_dsn()


@DSN_MODULES
def test_store_dsn_refuses_an_empty_file(
    module: ModuleType | LazyScriptModule,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text("\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))

    with pytest.raises(RuntimeError, match="file is empty"):
        module.store_dsn()


@DSN_MODULES
def test_store_dsn_configured_but_missing_file_is_an_error_not_a_fallback(
    module: ModuleType | LazyScriptModule,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "absent"))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", FAKE_DSN)

    with pytest.raises(FileNotFoundError):
        module.store_dsn()


@DSN_MODULES
def test_store_dsn_falls_back_to_the_env_only_for_the_default_path(
    module: ModuleType | LazyScriptModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default mount /etc/gpu-fault/aurora/postgres-url exists only inside a
    # control-plane Pod; on a developer or CI host the env is the only source.
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", FAKE_DSN)

    assert module.store_dsn() == FAKE_DSN


# --------------------------------------------------------------------------- #
# q1-13 / q1-29: failed-workflow scan and active-marker selection
# --------------------------------------------------------------------------- #
class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, parameters: Any = None) -> None:
        self.database.statements.append((" ".join(sql.split()), parameters))

    def executemany(self, sql: str, rows: list[tuple[str, str]]) -> None:
        self.database.deleted.extend(rows)

    def fetchall(self) -> list[tuple[str]]:
        return [(name,) for name in sorted(self.database.indexes)]


class FakeDatabase:
    def __init__(self, indexes: set[str]) -> None:
        self.indexes = indexes
        self.statements: list[tuple[str, Any]] = []
        self.deleted: list[tuple[str, str]] = []
        self.transactions = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        yield


class FakeControlStore:
    """Enough of PostgresStore for the q1-13 / q1-29 probe, in memory."""

    constructed: list[tuple[str, dict[str, Any]]] = []
    instances: list[FakeControlStore] = []
    indexes: set[str] = set(q113.EXPECTED_INDEXES)

    def __init__(self, dsn: str, **options: Any) -> None:
        type(self).constructed.append((dsn, options))
        type(self).instances.append(self)
        self.workflows: dict[str, WorkflowRequest] = {}
        self.markers: dict[str, NodeMarker] = {}
        self.closed = False
        self.database = FakeDatabase(set(type(self).indexes))
        self._db = self.database

    def save_workflow(self, workflow: WorkflowRequest) -> None:
        self.workflows[workflow.request_id] = workflow

    def list_unhandled_failed_workflows(
        self, *, limit: int = 1000
    ) -> list[WorkflowRequest]:
        return [
            workflow
            for workflow in self.workflows.values()
            if workflow.status.value == "FAILED" and workflow.failure_handled_at is None
        ][:limit]

    def add_marker(self, marker: NodeMarker) -> None:
        self.markers[marker.marker_id] = marker

    def list_active_markers_for_nodes(
        self, node_ids: set[str], actions: set[RecoveryAction]
    ) -> list[NodeMarker]:
        return [
            marker
            for marker in self.markers.values()
            if marker.active
            and set(marker.scope.node_ids) & node_ids
            and marker.recommended_action in actions
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def control_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text(FAKE_DSN, encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))
    FakeControlStore.constructed = []
    FakeControlStore.instances = []
    FakeControlStore.indexes = set(q113.EXPECTED_INDEXES)
    monkeypatch.setattr(q113, "PostgresStore", FakeControlStore)
    return FakeControlStore


def test_q113_q129_probe_passes_and_deletes_every_record_it_wrote(
    control_store: type, capsys: pytest.CaptureFixture[str]
) -> None:
    q113.main()

    (dsn, options), *rest = control_store.constructed
    assert rest == []
    assert dsn == FAKE_DSN
    assert options == {
        "pool_min_size": 1,
        "pool_max_size": 2,
        "pool_timeout_seconds": 5,
    }
    out = capsys.readouterr().out
    assert "q1-13" in out and "second_scan=[]" in out
    assert "q1-29" in out and "-target'" in out
    assert FAKE_DSN not in out, "the DSN must never be printed"


def test_q113_q129_probe_cleans_up_through_the_delete_function_when_an_index_is_missing(
    control_store: type,
) -> None:
    control_store.indexes = {"gpu_fault_failed_workflow_updated"}

    with pytest.raises(AssertionError):
        q113.main()

    (store,) = control_store.instances
    assert store.closed is True
    kinds = sorted({kind for kind, _key in store.database.deleted})
    assert kinds == ["marker", "workflow"]
    assert len(store.database.deleted) == 6, store.database.deleted
    assert store.database.transactions == 1
    assert all(
        "gpu_fault_delete_control_state" not in sql
        for sql, _parameters in store.database.statements
    ), "deletes go through executemany inside one transaction"


def test_q113_q129_module_entry_point_runs_the_probe(
    control_store: type,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ``from gpu_fault.store import PostgresStore`` is re-evaluated by the fresh
    # ``__main__`` namespace, so the fake is installed where that import looks.
    monkeypatch.setattr(gpu_fault.store, "PostgresStore", control_store)
    monkeypatch.setattr(sys, "argv", ["audit_q113_q129_postgres.py"])

    runpy.run_path(str(SCRIPTS / "audit_q113_q129_postgres.py"), run_name="__main__")

    assert len(control_store.constructed) == 1
    assert "indexes" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# q05/q06: the audit adapter's step answers
# --------------------------------------------------------------------------- #
def _context(operation: WorkflowOperation) -> SimpleNamespace:
    return SimpleNamespace(step=SimpleNamespace(operation=operation))


def test_q05_q06_adapter_rebinds_on_replace_and_waits_on_validate_and_restart() -> None:
    adapter = q05_q06.AuditAdapter()

    replace = adapter.execute(_context(WorkflowOperation.REPLACE_NODE))
    assert replace.status is WorkflowStepStatus.SUCCEEDED
    assert replace.details == {
        "node_rebindings": {"audit-node-old": "audit-node-spare"}
    }

    for operation in (WorkflowOperation.VALIDATE_GPU, WorkflowOperation.RESTART_NODE):
        waiting = adapter.execute(_context(operation))
        assert waiting.status is WorkflowStepStatus.WAITING
        assert waiting.adapter_operation_id == f"audit/{operation.value}"

    plain = adapter.execute(_context(WorkflowOperation.QUIESCE_GPU_SERVICES))
    assert plain == WorkflowStepOutcome.succeeded()


def test_q05_q06_adapter_supports_only_its_own_owner() -> None:
    adapter = q05_q06.AuditAdapter()
    assert adapter.supports(SimpleNamespace(execution_owner="audit-owner")) is True
    assert adapter.supports(SimpleNamespace(execution_owner="other")) is False


# --------------------------------------------------------------------------- #
# processor ordering audit against a scripted psycopg connection
# --------------------------------------------------------------------------- #
class QueueCursor:
    def __init__(self, connection: QueueConnection) -> None:
        self.connection = connection

    def __enter__(self) -> QueueCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, parameters: Any = None) -> None:
        flat = " ".join(sql.split())
        self.connection.statements.append((flat, parameters))
        if flat.startswith("INSERT INTO gpu_fault_processor_queue"):
            self.connection.queue[str(parameters[0])] = {
                "status": "PENDING",
                "payload": json.loads(parameters[-1]),
            }
        elif flat.startswith("DELETE"):
            self.connection.deletes.append(flat.split(" FROM ")[1].split(" ")[0])

    def fetchall(self) -> list[tuple[Any, ...]]:
        self.connection.polls += 1
        rows = []
        for request_id, row in self.connection.queue.items():
            status, updated_at = self.connection.status_for(request_id, row)
            rows.append((request_id, status, updated_at, row["payload"]))
        return rows

    def fetchone(self) -> tuple[dict[str, Any]]:
        return (self.connection.event,)


class QueueConnection:
    def __init__(self, *, complete: bool, event: dict[str, Any]) -> None:
        self.complete = complete
        self.event = event
        self.queue: dict[str, dict[str, Any]] = {}
        self.statements: list[tuple[str, Any]] = []
        self.deletes: list[str] = []
        self.polls = 0
        self.autocommit: bool | None = None
        self.rolled_back = False
        self.closed = False
        self.base = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

    def status_for(self, request_id: str, row: dict[str, Any]) -> tuple[str, datetime]:
        if not self.complete:
            return "LEASED", self.base
        # The observation completes first; the fault completes one second later.
        if row["payload"]["path"] == "/v1/workload-observations":
            return "COMPLETED", self.base
        return "COMPLETED", self.base + timedelta(seconds=1)

    def cursor(self) -> QueueCursor:
        return QueueCursor(self)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class FakeClock:
    """``time`` as the script sees it: monotonic advances only when asked."""

    def __init__(self, step: float) -> None:
        self.now = 1000.0
        self.step = step
        self.slept: list[float] = []

    def monotonic(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


@pytest.fixture
def ordering_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text(FAKE_DSN, encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))
    monkeypatch.setenv("GPU_FAULT_RUNTIME_PROFILE_VERSION", "profile-audit-1")


def _connect(monkeypatch: pytest.MonkeyPatch, connection: QueueConnection) -> list[str]:
    import psycopg

    urls: list[str] = []

    def connect(url: str) -> QueueConnection:
        urls.append(url)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    return urls


def test_ordering_audit_reports_the_observation_completing_first(
    ordering_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    connection = QueueConnection(complete=True, event={})
    urls = _connect(monkeypatch, connection)

    # The event row is looked up by the suffix main() draws; answer with whatever
    # identity the inserted observation carried so the audit's own join holds.
    def fetchone(self: QueueCursor) -> tuple[dict[str, Any]]:
        observation = next(
            row["payload"]
            for row in connection.queue.values()
            if row["payload"]["path"] == "/v1/workload-observations"
        )
        body = json.loads(base64.b64decode(observation["body_base64"]))
        return (
            {
                "job_id": body["job_id"],
                "attempt_id": body["attempt_id"],
                "workload_identity_source": "ACTIVE_MANAGED_ATTEMPT",
            },
        )

    monkeypatch.setattr(QueueCursor, "fetchone", fetchone)
    monkeypatch.setattr(processor_order, "time", FakeClock(step=0.0))

    processor_order.main()

    assert urls == [FAKE_DSN]
    report = json.loads(capsys.readouterr().out)
    assert report["fault_inserted_first"] is True
    assert report["observation_completed_at"] < report["fault_completed_at"]
    assert report["workload_identity_source"] == "ACTIVE_MANAGED_ATTEMPT"
    assert report["job_id"].startswith("audit-job-"), report["job_id"]
    assert connection.polls == 1
    assert connection.deletes == [
        "gpu_fault_processor_queue",
        "gpu_fault_objects",
        "gpu_fault_links",
    ]
    assert connection.rolled_back is True and connection.closed is True
    assert connection.autocommit is True, "cleanup runs with autocommit restored"


def test_ordering_audit_gives_up_at_its_deadline_and_still_cleans_up(
    ordering_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = QueueConnection(complete=False, event={})
    _connect(monkeypatch, connection)
    # 60 s budget, the clock advances 40 s per reading: one poll, one sleep, out.
    clock = FakeClock(step=40.0)
    monkeypatch.setattr(processor_order, "time", clock)

    with pytest.raises(AssertionError):
        processor_order.main()

    assert connection.polls == 1
    assert clock.slept == [0.05]
    assert connection.deletes == [
        "gpu_fault_processor_queue",
        "gpu_fault_objects",
        "gpu_fault_links",
    ]
    assert connection.closed is True


def test_ordering_audit_module_entry_point_runs_main(
    ordering_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = QueueConnection(complete=False, event={})
    _connect(monkeypatch, connection)
    monkeypatch.setattr(sys, "argv", ["audit_processor_order_production.py"])
    # The fresh ``__main__`` namespace binds its own ``time``; make the real
    # clock irrelevant by never completing and letting the deadline pass.
    monkeypatch.setattr(
        processor_order.time, "monotonic", FakeClock(step=40.0).monotonic
    )
    monkeypatch.setattr(processor_order.time, "sleep", lambda _seconds: None)

    with pytest.raises(AssertionError):
        runpy.run_path(
            str(SCRIPTS / "audit_processor_order_production.py"), run_name="__main__"
        )

    assert connection.closed is True
