"""The store audit scripts against an in-memory store, a fake bucket and database.

``q118_fallback_marker.py``, ``audit_control_record_archive.py`` and
``audit_node_health_atomicity_postgres.py`` run on the CPU control plane
against Aurora. Here every ``PostgresStore(dsn)`` they open is a handle on one
shared :class:`InMemoryStore`, ``psycopg.connect`` records the clean-up SQL and
``boto3.client("s3")`` is an in-memory bucket, so each audit's decisions and
its clean-up are observable without a database.
"""

from __future__ import annotations

import io
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import psycopg
import pytest

import gpu_fault.store as store_package
from gpu_fault.models import (
    Environment,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from scripts.e2e.regional import audit_control_record_archive as archive_audit
from scripts.e2e.regional import audit_node_health_atomicity_postgres as atomicity
from scripts.e2e.regional import q118_fallback_marker as q118

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "e2e" / "regional"
DEFAULT_DSN_FILE = Path("/etc/gpu-fault/aurora/postgres-url")
DSN_READERS = (q118, archive_audit, atomicity)


class StoreHandle:
    """One ``PostgresStore(dsn, **options)`` the audit opened."""

    def __init__(self, world: StoreWorld, dsn: str, options: dict[str, Any]) -> None:
        self.world = world
        self.dsn = dsn
        self.options = options
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.world.store, name)

    def close(self) -> None:
        self.closed = True


class StoreWorld:
    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.handles: list[StoreHandle] = []

    def open(self, dsn: str, **options: Any) -> StoreHandle:
        handle = StoreHandle(self, dsn, options)
        self.handles.append(handle)
        return handle


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.database.statements.append((" ".join(sql.split()), params))

    def fetchone(self) -> tuple[int]:
        return (self.database.remaining_rows,)


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.database)


class FakeDatabase:
    """``psycopg.connect``: records every statement, answers counts with 0."""

    def __init__(self) -> None:
        self.dsns: list[str] = []
        self.statements: list[tuple[str, Any]] = []
        self.remaining_rows = 0

    def connect(self, dsn: str) -> FakeConnection:
        self.dsns.append(dsn)
        return FakeConnection(self)


class FakeBucket:
    """``boto3.client("s3")`` plus what the fake archiver module reads."""

    def __init__(self, store: InMemoryStore, *, archive_active: bool = False) -> None:
        self.store = store
        self.archive_active = archive_active
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []
        self.archive_calls: list[str] = []
        self.clients: list[str] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.deleted.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)


ARCHIVER_MODULE = """
import gzip
import json


class ArchiveSafetyError(RuntimeError):
    pass


class ControlRecordArchiver:
    def __init__(self, dsn, archive_uri, *, s3_client):
        self.dsn = dsn
        self.archive_uri = archive_uri
        self.s3 = s3_client

    def archive_one(self, incident_id):
        self.s3.archive_calls.append(incident_id)
        store = self.s3.store
        incident = store.get_incident(incident_id)
        if incident.state.value != "RECOVERED" and not self.s3.archive_active:
            raise ArchiveSafetyError(incident_id + " is still active")
        records = [{"kind": "incident", "key": incident_id}]
        records.append({"kind": "workflow", "key": incident.workflow_request_id})
        records.extend(
            {"kind": "marker", "key": marker.marker_id}
            for marker in store.list_markers_for_incident(incident_id)
        )
        try:
            notification = store.get_notification("notification-" + incident_id)
        except KeyError:
            notification = None
        if notification is not None:
            records.append({"kind": "notification", "key": notification.notification_id})
        bucket, _, prefix = self.archive_uri[len("s3://"):].partition("/")
        key = prefix + "/" + incident_id + ".jsonl.gz"
        body = "\\n".join(json.dumps(record) for record in records)
        self.s3.put_object(Bucket=bucket, Key=key, Body=gzip.compress(body.encode()))
        return "s3://" + bucket + "/" + key
"""


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> StoreWorld:
    world = StoreWorld()
    for module in DSN_READERS:
        monkeypatch.setattr(module, "PostgresStore", world.open)
    # A ``__main__`` run re-imports the name from the package.
    monkeypatch.setattr(store_package, "PostgresStore", world.open)
    return world


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> FakeDatabase:
    database = FakeDatabase()
    monkeypatch.setattr(psycopg, "connect", database.connect)
    return database


@pytest.fixture
def dsn_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "postgres-url"
    path.write_text("postgresql://audit.invalid/gpu_fault\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    return path


@pytest.mark.parametrize("module", DSN_READERS, ids=lambda module: module.__name__)
def test_store_dsn_reads_the_configured_file_and_refuses_empty_inputs(
    module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "postgres-url"
    path.write_text("  postgresql://audit.invalid/db \n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://environment.invalid/db")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    assert module.store_dsn() == "postgresql://audit.invalid/db"

    path.write_text(" \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="store DSN file is empty"):
        module.store_dsn()

    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "")
    with pytest.raises(RuntimeError, match="configured store DSN file path is empty"):
        module.store_dsn()

    # A configured file that is missing is an error, never the env fallback.
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        module.store_dsn()


@pytest.mark.parametrize("module", DSN_READERS, ids=lambda module: module.__name__)
def test_store_dsn_falls_back_to_the_environment_without_the_mounted_file(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    if DEFAULT_DSN_FILE.exists():
        pytest.skip("this host mounts the real Aurora DSN file")
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://environment.invalid/db")
    assert module.store_dsn() == "postgresql://environment.invalid/db"


def marker(marker_id: str, *, active: bool = True) -> NodeMarker:
    now = datetime.now(timezone.utc)
    return NodeMarker(
        marker_id=marker_id,
        source="audit-q118",
        trusted=True,
        incident_id=f"{marker_id}-incident",
        observed_at=now,
        expires_at=now + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RESTART_WORKLOAD,
        mapping_version="audit-q118",
        active=active,
    )


def attempt(attempt_id: str, nodes: list[str | None]) -> AttemptObservation:
    return AttemptObservation(
        cluster_id="cluster-a",
        environment=Environment.HYPERPOD_EKS,
        job_id="job-a",
        attempt_id=attempt_id,
        workload_phase=WorkloadPhase.RUNNING,
        observed_at=datetime.now(timezone.utc),
        runtime_profile_version="profile-a",
        expected_critical_ranks=len(nodes),
        containers=[
            ContainerObservation(
                pod_uid=f"uid-{rank}",
                pod_name=f"worker-{rank}",
                container_name="trainer",
                role="worker",
                rank=rank,
                node_id=node,
                gpu_count=8,
            )
            for rank, node in enumerate(nodes)
        ],
    )


def test_q118_disable_turns_the_named_marker_inactive_and_closes_the_store(
    world: StoreWorld,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world.store.add_marker(marker("marker-other"))
    world.store.add_marker(marker("marker-1"))
    monkeypatch.setattr(
        sys, "argv", ["q118", "disable", "cluster-a", "attempt-1", "marker-1"]
    )

    q118.main()

    states = {item.marker_id: item.active for item in world.store.list_markers()}
    assert states == {"marker-other": True, "marker-1": False}
    assert capsys.readouterr().out == "disabled marker-1\n"
    [handle] = world.handles
    assert handle.dsn == "postgresql://audit.invalid/gpu_fault"
    assert handle.options == {
        "pool_min_size": 1,
        "pool_max_size": 2,
        "pool_timeout_seconds": 5,
    }
    assert handle.closed is True


def test_q118_create_marks_every_distinct_attempt_node_for_a_workload_restart(
    world: StoreWorld,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world.store.save_attempt_observation(attempt("attempt-0", ["node-z"]))
    world.store.save_attempt_observation(
        attempt("attempt-1", ["node-b", "node-a", None, "node-b"])
    )
    monkeypatch.setattr(
        sys, "argv", ["q118", "create", "cluster-a", "attempt-1", "marker-2"]
    )

    before = datetime.now(timezone.utc)
    q118.main()

    [created] = world.store.list_markers()
    assert created.marker_id == "marker-2"
    assert created.scope.node_ids == ["node-a", "node-b"]
    assert (created.source, created.trusted, created.mapping_version) == (
        "audit-q118",
        True,
        "audit-q118",
    )
    assert created.severity is Severity.CRITICAL
    assert created.recommended_action is RecoveryAction.RESTART_WORKLOAD
    assert created.incident_id == "marker-2-incident"
    assert created.drill_id == "q118-emergency-fallback"
    assert created.observed_at >= before
    assert created.expires_at - created.observed_at == timedelta(hours=1)
    assert capsys.readouterr().out == (
        "created marker-2 RUNNING ['node-a', 'node-b']\n"
    )
    assert world.handles[0].closed is True


def test_q118_runs_as_a_program_with_its_four_arguments(
    world: StoreWorld,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world.store.add_marker(marker("marker-3"))
    monkeypatch.setattr(
        sys, "argv", ["q118", "disable", "cluster-a", "attempt-1", "marker-3"]
    )

    runpy.run_path(str(SCRIPTS / "q118_fallback_marker.py"), run_name="__main__")

    assert [item.active for item in world.store.list_markers()] == [False]
    assert capsys.readouterr().out == "disabled marker-3\n"
    assert [handle.closed for handle in world.handles] == [True]


def test_q118_closes_the_store_when_the_named_marker_is_absent(
    world: StoreWorld, dsn_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["q118", "disable", "cluster-a", "attempt-1", "marker-missing"]
    )

    with pytest.raises(StopIteration):
        q118.main()

    assert [handle.closed for handle in world.handles] == [True]


def archive_world(
    monkeypatch: pytest.MonkeyPatch,
    world: StoreWorld,
    tmp_path: Path,
    *,
    archive_active: bool = False,
) -> tuple[FakeBucket, Path]:
    bucket = FakeBucket(world.store, archive_active=archive_active)

    def client(service: str) -> FakeBucket:
        bucket.clients.append(service)
        return bucket

    monkeypatch.setattr(boto3, "client", client)
    module = tmp_path / "control_record_archive.py"
    module.write_text(ARCHIVER_MODULE, encoding="utf-8")
    return bucket, module


def cleanup_patterns(database: FakeDatabase) -> list[Any]:
    return [params for sql, params in database.statements if "LIKE" in sql]


def test_archive_audit_archives_only_the_recovered_incident_and_cleans_up(
    world: StoreWorld,
    database: FakeDatabase,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket, module = archive_world(monkeypatch, world, tmp_path)
    monkeypatch.setattr(archive_audit, "uuid4", lambda: SimpleNamespace(hex="c0ffee"))

    archive_audit.main(str(module), "s3://audit-bucket/archive")

    incident_id = "audit-archive-c0ffee"
    active_id = "audit-archive-active-c0ffee"
    assert bucket.archive_calls == [active_id, incident_id]
    assert world.store.get_incident(incident_id).state is IncidentState.RECOVERED
    assert world.store.get_incident(active_id).state is IncidentState.ACTION_PENDING
    workflow = world.store.get_workflow(f"workflow-{incident_id}")
    assert workflow.status is WorkflowStatus.SUCCEEDED
    [audit_marker] = world.store.list_markers_for_incident(incident_id)
    assert audit_marker.active is False
    key = f"archive/{incident_id}.jsonl.gz"
    assert bucket.deleted == [("audit-bucket", key)]
    assert bucket.objects == {}
    output = capsys.readouterr().out
    assert output.startswith("PASS "), output
    assert f"'archive_uri': 's3://audit-bucket/{key}'" in output
    assert "'records': 4" in output
    count_sql = [sql for sql, _ in database.statements if "count(*)" in sql]
    assert len(count_sql) == 1
    assert cleanup_patterns(database) == [("%c0ffee%", "%c0ffee%")] * 2
    assert database.dsns == ["postgresql://audit.invalid/gpu_fault"] * 2
    assert world.handles[0].closed is True


def test_archive_audit_fails_when_the_active_incident_is_archived(
    world: StoreWorld,
    database: FakeDatabase,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bucket, module = archive_world(monkeypatch, world, tmp_path, archive_active=True)

    with pytest.raises(AssertionError, match="active incident was archived"):
        archive_audit.main(str(module), "s3://audit-bucket/archive")

    # Clean-up still runs; nothing was taken from the bucket by the audit.
    assert len(bucket.archive_calls) == 1
    assert bucket.deleted == []
    assert len(cleanup_patterns(database)) == 2
    assert world.handles[0].closed is True


def test_archive_audit_runs_as_a_program_with_its_two_arguments(
    world: StoreWorld,
    database: FakeDatabase,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket, module = archive_world(monkeypatch, world, tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["audit", str(module), "s3://program-bucket/runs/r7"]
    )

    runpy.run_path(
        str(SCRIPTS / "audit_control_record_archive.py"), run_name="__main__"
    )

    assert bucket.clients == ["s3"]
    [(bucket_name, key)] = bucket.deleted
    assert bucket_name == "program-bucket"
    assert key.startswith("runs/r7/audit-archive-"), key
    output = capsys.readouterr().out
    assert output.startswith("PASS "), output


def test_node_health_audit_serializes_the_two_creators_into_root_and_successor(
    world: StoreWorld,
    database: FakeDatabase,
    dsn_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pauses: list[float] = []
    monkeypatch.setattr(atomicity, "time", SimpleNamespace(sleep=pauses.append))
    monkeypatch.setattr(atomicity, "uuid4", lambda: SimpleNamespace(hex="beef"))

    atomicity.main()

    event_ids = [f"audit-p12-node-health-beef-{side}" for side in "ab"]
    workflows = [world.store.get_workflow(f"workflow-{event}") for event in event_ids]
    roots = [w for w in workflows if w.predecessor_workflow_id is None]
    successors = [w for w in workflows if w.predecessor_workflow_id is not None]
    assert len(roots) == len(successors) == 1
    root = roots[0]
    assert successors[0].predecessor_workflow_id == root.request_id
    assert {w.status for w in workflows} == {WorkflowStatus.BLOCKED}
    assert pauses == [0.5, 0.5]
    output = capsys.readouterr().out
    assert output.startswith("PASS "), output
    assert f"'root': '{root.request_id}'" in output
    assert "'duplicate_created': False" in output
    assert [handle.closed for handle in world.handles] == [True, True]
    links, records = database.statements
    assert links[0].startswith("DELETE FROM gpu_fault_links"), links[0]
    incident_ids = [f"inc-{event_id}" for event_id in event_ids]
    workflow_ids = [f"workflow-{event_id}" for event_id in event_ids]
    assert links[1] == (
        event_ids + incident_ids + workflow_ids,
        incident_ids + workflow_ids,
    )
    assert records[1] == (incident_ids, workflow_ids)


def test_node_health_audit_program_stops_before_the_database_without_a_dsn(
    world: StoreWorld, database: FakeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "")

    with pytest.raises(RuntimeError, match="configured store DSN file path is empty"):
        runpy.run_path(
            str(SCRIPTS / "audit_node_health_atomicity_postgres.py"),
            run_name="__main__",
        )

    assert world.handles == []
    assert database.statements == []


def test_load_module_executes_the_archiver_file_without_registering_it(
    tmp_path: Path,
) -> None:
    module = tmp_path / "control_record_archive.py"
    module.write_text(ARCHIVER_MODULE, encoding="utf-8")

    loaded = archive_audit.load_module(str(module))

    assert loaded.__name__ == "gpu_fault.control_record_archive_probe"
    assert issubclass(loaded.ArchiveSafetyError, RuntimeError), (
        "the audit catches the loaded module's own safety error class"
    )
    assert "gpu_fault.control_record_archive_probe" not in sys.modules
