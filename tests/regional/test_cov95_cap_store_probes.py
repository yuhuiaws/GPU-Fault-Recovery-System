from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional.probes import cap_seed_commands as seed
from scripts.e2e.regional.probes import cap_store_lock as locks
from tests.regional._cov95_cap_probes import RecordingDatabase, RecordingStore

STORE_URL = "postgresql://example.invalid/cap-local"


def store_factory(
    monkeypatch: pytest.MonkeyPatch, module: Any, *, fail_after: int | None = None
) -> tuple[RecordingStore, list[tuple[str, dict[str, Any]]]]:
    store = RecordingStore([], fail_after=fail_after)
    calls: list[tuple[str, dict[str, Any]]] = []

    def connect(url: str, **kwargs: Any) -> RecordingStore:
        calls.append((url, kwargs))
        return store

    monkeypatch.setattr(module, "PostgresStore", connect)
    return store, calls


@pytest.mark.parametrize("count,tag", [(None, "default"), (1, "one"), (4, "four")])
def test_store_lock_probe_seeds_bound_commands_and_unlocks_every_owned_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    count: int | None,
    tag: str,
) -> None:
    (tmp_path / "store-url").write_text(STORE_URL)
    ready = tmp_path / f"store-lock-ready-{tag}"
    released = tmp_path / f"store-lock-release-{tag}"
    ready.write_text("stale ready")
    released.write_text("stale release")
    store, stores = store_factory(monkeypatch, locks)
    database = RecordingDatabase([])
    monkeypatch.setattr(locks, "WORK", tmp_path)
    monkeypatch.setattr(
        locks,
        "sys",
        SimpleNamespace(
            argv=["cap-store-lock"] + ([] if count is None else [str(count), tag])
        ),
    )
    monkeypatch.setattr("psycopg.connect", database.connect)
    pauses = []

    def pause(seconds: float) -> None:
        pauses.append(seconds)
        assert ready.read_text() == "ready\n"
        assert not released.exists(), "a stale release sentinel skipped the lock window"
        released.write_text("release")

    monkeypatch.setattr(
        locks, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=pause)
    )
    assert locks.main() == 0
    expected = 4 if count is None else count
    ids = [f"command-cap002-{tag}-{index:03d}" for index in range(expected)]
    assert [command.command_id for command in store.commands] == ids
    for index, command in enumerate(store.commands):
        assert command.cluster_id == f"cap-cluster-{index:03d}"
        assert command.step.operation == WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
        assert command.step.execution_owner == "gpu-fault-kubernetes-adapter"
        assert command.step.parameters == {"capacity_test": True}
        assert command.incident.drill_id == "cap002"
        assert command.workflow.fencing_token == 1
    assert stores == [
        (
            STORE_URL,
            {"initialize_schema": False, "pool_min_size": 0, "pool_max_size": 2},
        )
    ]
    assert store.closed is True
    assert database.connections == [(STORE_URL, {})]
    assert database.closed == 1
    assert database.statements == [
        (
            "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
            (f"remote_command/{name}",),
        )
        for name in ids
    ] + [
        (
            "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
            (f"remote_command/{name}",),
        )
        for name in ids
    ]
    assert pauses == [0.25]
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"advisory_locks": "held", "command_count": expected},
        {"advisory_locks": "released"},
    ]


@pytest.mark.parametrize("count", ["0", "5", "not-an-integer"])
def test_invalid_lock_count_stops_before_file_or_database_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, count: str
) -> None:
    store, calls = store_factory(monkeypatch, locks)
    monkeypatch.setattr(locks, "WORK", tmp_path / "missing")
    monkeypatch.setattr(locks, "sys", SimpleNamespace(argv=["cap-store-lock", count]))
    with pytest.raises(ValueError):
        locks.main()
    assert calls == []
    assert store.commands == []


def test_lock_deadline_closes_connection_without_claiming_explicit_unlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "store-url").write_text(STORE_URL)
    store, _calls = store_factory(monkeypatch, locks)
    database = RecordingDatabase([])
    elapsed = [0.0]
    monkeypatch.setattr(locks, "WORK", tmp_path)
    monkeypatch.setattr(
        locks, "sys", SimpleNamespace(argv=["cap-store-lock", "1", "timeout"])
    )
    monkeypatch.setattr("psycopg.connect", database.connect)

    def pause(_seconds: float) -> None:
        elapsed[0] += 601

    monkeypatch.setattr(
        locks, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=pause)
    )
    with pytest.raises(RuntimeError, match="release sentinel was not created"):
        locks.main()
    assert store.closed is True
    assert database.closed == 1
    assert len(database.statements) == 1
    assert json.loads(capsys.readouterr().out) == {
        "advisory_locks": "held",
        "command_count": 1,
    }


def test_lock_seed_failure_closes_store_before_opening_a_lock_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "store-url").write_text(STORE_URL)
    store, _calls = store_factory(monkeypatch, locks, fail_after=0)
    database = RecordingDatabase([])
    monkeypatch.setattr(locks, "WORK", tmp_path)
    monkeypatch.setattr(locks, "sys", SimpleNamespace(argv=["cap-store-lock"]))
    monkeypatch.setattr("psycopg.connect", database.connect)
    with pytest.raises(RuntimeError, match="insertion failed"):
        locks.main()
    assert store.closed is True
    assert database.connections == []


@pytest.mark.parametrize("fail_after", [None, 1])
def test_cap004_seed_uses_isolated_non_ddl_store_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fail_after: int | None,
) -> None:
    path = tmp_path / "store-url"
    path.write_text(STORE_URL)
    store, calls = store_factory(monkeypatch, seed, fail_after=fail_after)

    def isolated_path(value: str) -> Path:
        assert value == "/work/store-url"
        return path

    monkeypatch.setattr(seed, "Path", isolated_path)
    if fail_after is not None:
        with pytest.raises(RuntimeError, match="insertion failed"):
            seed.main()
        assert capsys.readouterr().out == ""
    else:
        assert seed.main() == 0
        assert json.loads(capsys.readouterr().out) == {
            "cluster_id": "cap-cluster-000",
            "commands_created": 25,
        }
    assert len(store.commands) == (25 if fail_after is None else fail_after)
    assert store.closed is True
    assert calls == [
        (
            STORE_URL,
            {"initialize_schema": False, "pool_min_size": 0, "pool_max_size": 2},
        )
    ]
    for index, command in enumerate(store.commands):
        assert command.command_id == f"command-cap004-{index:03d}"
        assert command.cluster_id == "cap-cluster-000"
        assert command.step.operation == WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
        assert command.step.execution_owner == "gpu-fault-node-agent"
        assert command.step.parameters == {"capacity_test": True}
        assert command.incident.drill_id == "cap004"
