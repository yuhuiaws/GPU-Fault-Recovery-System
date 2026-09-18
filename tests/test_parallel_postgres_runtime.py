from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.postgres_grant import (
    ALLOCATION_ENV,
    POSTGRES_URL_ENV,
    PostgresGrantError,
    postgres_test_environment,
)
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.release_postgres import (
    ALLOCATION_DIRECTORY,
    OwnedPostgres,
    isolated_postgres_allocation,
)
from scripts import postgres_shard_worker as worker
from scripts import run_postgres_shards as runner
from scripts.postgres_shard_receipts import PostgresShardError, read_private_json
from tests.admin._release_postgres_support import CID, IMAGE, FakePostgresDocker


@pytest.fixture(autouse=True)
def no_inherited_database(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith(("PG", "POSTGRES_")) or name in (
            POSTGRES_URL_ENV,
            ALLOCATION_ENV,
        ):
            monkeypatch.delenv(name)


def allocation(tmp_path: Path) -> tuple[Path, FakePostgresDocker, OwnedPostgres]:
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    directory = state / ALLOCATION_DIRECTORY
    directory.mkdir(mode=0o700)
    info = directory.stat()
    docker = FakePostgresDocker()
    owned = OwnedPostgres(
        docker, directory, IMAGE, uuid4().hex, (info.st_dev, info.st_ino)
    )
    owned.record("PREPARED")
    owned.start()
    return state, docker, owned


@pytest.mark.parametrize(
    "url", ["postgresql://external.example/tests", "postgresql://localhost/tests"]
)
def test_external_urls_are_refused_before_any_allocation(url: str) -> None:
    parent = {POSTGRES_URL_ENV: url, "PGPASSWORD": "example-private-value"}
    with pytest.raises(PostgresShardError, match="external PostgreSQL"):
        worker.worker_environment(parent, root=worker.ROOT)
    assert parent["PGPASSWORD"] == "example-private-value"


def test_valid_parent_grant_is_only_read_then_removed_from_worker_env(
    tmp_path: Path,
) -> None:
    docker = FakePostgresDocker()
    state = tmp_path / "parent"
    state.mkdir(mode=0o700)
    with isolated_postgres_allocation(docker, state_dir=state) as parent_allocation:
        parent = postgres_test_environment(
            parent_allocation.build_environment(
                {
                    "HOME": "/original-home",
                    "COSIGN_PASSWORD": "example-signing-value",
                    "AWS_SECRET_ACCESS_KEY": "example-cloud-value",
                }
            )
        )
        parent.update(
            {
                "PGPASSWORD": "example-db-value",
                "POSTGRES_PASSWORD": "example-container-value",
                "PYTEST_ADDOPTS": "-k not_stress -n8",
                "PYTEST_XDIST_WORKER": "gw2",
                "PYTEST_GPU_FAULT_PARTITION_COUNT": "99",
                "PYTEST_GPU_FAULT_CASE_REPORT": "/foreign/report.json",
                "KUBECONFIG": "/foreign/config",
            }
        )
        before = dict(parent)
        calls = len(docker.calls)
        assert parent_allocation.directory is not None
        metadata = (parent_allocation.directory / "ownership.json").read_bytes()
        child = worker.worker_environment(parent, root=worker.ROOT)
        assert parent == before, "preparing a shard mutated the parent's environment"
        assert len(docker.calls) == calls, "admission touched the parent's database"
        assert (parent_allocation.directory / "ownership.json").read_bytes() == metadata
        assert not any(name.startswith(("PG", "POSTGRES_")) for name in child), (
            "worker inherited database credentials"
        )
        assert not (
            {POSTGRES_URL_ENV, ALLOCATION_ENV, "COSIGN_PASSWORD", "KUBECONFIG"}
            & child.keys()
        )
        assert child["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert child["PYTHONPATH"].split(os.pathsep) == [
            str(worker.ROOT / "src"),
            str(worker.ROOT),
        ]
        assert docker.removed == [], "admission stopped the parent allocation"
    assert docker.removed == [CID]


def test_invalid_inherited_grant_never_falls_back_to_local_postgres(
    tmp_path: Path,
) -> None:
    with pytest.raises(PostgresGrantError):
        worker.worker_environment(
            {
                POSTGRES_URL_ENV: "postgresql://external.example/test",
                ALLOCATION_ENV: str(tmp_path),
            },
            root=worker.ROOT,
        )


def test_worker_environment_only_preserves_explicit_test_controls() -> None:
    controls = {
        "GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS": "8",
        "GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS": "40",
        "GPU_FAULT_TEST_SHUFFLE_SEED": "12345",
    }
    parent = {
        **controls,
        "GPU_FAULT_STORE_URL": "postgresql://external.example/runtime",
        "GPU_FAULT_EXECUTION_TOKEN": "example-execution-value",
        "GPU_FAULT_CLUSTER_TOKEN": "example-cluster-value",
        "GPU_FAULT_NODE_ACTION_SECRET_KEY": "example-node-value",
        "GPU_FAULT_CONTROL_PLANE_URL": "https://external.example/control",
        "GPU_FAULT_REPOSITORY_ROOT": "/foreign/repository",
        "GPU_FAULT_TEST_UNAPPROVED_CONTROL": "example-unapproved-value",
        "GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC": "12345",
        "GPU_FAULT_DEPLOY_HARD_DEADLINE_MONOTONIC": "12445",
        "GPU_FAULT_DEPLOY_RECOVERY_ACTIVE": "true",
        "GPU_FAULT_DEPLOY_API_BUDGET_DIR": "/private/deploy-budget",
        "PATH": "/private/deploy-budget/bin:/usr/bin",
        "HOME": "/original-home",
    }
    original = dict(parent)
    child = worker.worker_environment(parent, root=worker.ROOT)
    assert parent == original, "worker isolation changed the caller's environment"
    assert {
        name: value for name, value in child.items() if name.startswith("GPU_FAULT_")
    } == controls, "ambient runtime credentials or endpoints entered the worker"
    assert child["PATH"] == "/usr/bin", "the stripped deployment shim remained active"
    assert child["HOME"] == parent["HOME"]


def test_first_failure_survives_later_phase_failures_and_worker_exit(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / worker.FAILURE_FILE
    worker.record_failure(
        path,
        nodeid="tests/test_example.py::test_original[value]",
        phase="setup",
        outcome="failed",
    )
    original = path.read_bytes()
    worker.record_failure(
        path,
        nodeid="tests/test_example.py::test_later",
        phase="teardown",
        outcome="failed",
    )
    worker.record_failure(path)
    assert path.read_bytes() == original, (
        "later failures replaced the first test failure"
    )


def test_cancel_before_allocation_never_creates_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    write_json_atomic(tmp_path / worker.CANCEL_FILE, {"cancelled": True})
    called: list[bool] = []

    def identify(_root: Path) -> str:
        called.append(True)
        return "a" * 64

    monkeypatch.setattr(worker, "source_identity", identify)
    with pytest.raises(KeyboardInterrupt):
        worker.run_worker(
            state=tmp_path, tests=(), workers=1, index=0, durations=0, identity="a" * 64
        )
    assert called == [], "a cancelled worker started preparation"
    assert not (tmp_path / ALLOCATION_DIRECTORY).exists(), (
        "a cancelled worker allocated a database"
    )


@pytest.mark.parametrize(
    "phase", ["identity", "run", "cleanup", "supervision", "interrupt"]
)
def test_failure_audit_retains_only_established_run_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    tests = ("tests/test_example.py",)
    (root / tests[0]).touch()
    state = tmp_path / "run"
    state.mkdir(mode=0o700)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **_options: str(state))
    identity = "b" * 64
    cleaned: list[int] = []

    def bind(_python: str, _environment: object) -> str:
        if phase == "identity":
            raise PostgresShardError("source identity unavailable")
        return identity

    def execute(jobs: Sequence[runner.ShardJob], **_options: Any) -> list[Any]:
        assert len(jobs) == 2
        assert read_private_json(state / "run.json") == {
            "kind": "local-postgres-shards",
            "status": "RUNNING",
            "source_identity": identity,
            "workers": 2,
            "tests": list(tests),
        }
        if phase == "supervision":
            raise ProcessSupervisionLost("example supervision loss")
        if phase == "interrupt":
            raise KeyboardInterrupt
        if phase == "run":
            raise PostgresShardError("example shard failure")
        return []

    def cleanup(jobs: Sequence[runner.ShardJob]) -> None:
        cleaned.extend(job.index for job in jobs)
        if phase == "cleanup":
            raise PostgresShardError("example cleanup failure")

    monkeypatch.setattr(runner, "checked_identity", bind)
    monkeypatch.setattr(runner, "run_jobs", execute)
    monkeypatch.setattr(runner, "reconcile_jobs", cleanup)
    expected_error = (
        ProcessSupervisionLost
        if phase == "supervision"
        else KeyboardInterrupt
        if phase == "interrupt"
        else PostgresShardError
    )
    with pytest.raises(expected_error):
        runner.run_postgres_shards(sys.executable, workers=2, durations=0, tests=tests)
    expected: dict[str, object] = {
        "kind": "local-postgres-shards",
        "status": "SUPERVISION_LOST" if phase == "supervision" else "FAILED",
    }
    if phase != "identity":
        expected.update(source_identity=identity, workers=2, tests=list(tests))
    assert read_private_json(state / "run.json") == expected, (
        "failure erased a known source binding or invented an unverified one"
    )
    assert cleaned == ([] if phase == "identity" else [0, 1])


@pytest.mark.parametrize("status", [0, 1])
def test_worker_uses_own_grant_serial_pytest_and_cleans_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    tmp_path.chmod(0o700)
    docker = FakePostgresDocker()
    monkeypatch.setattr(worker, "CommandRunner", lambda: docker)
    monkeypatch.setattr(worker, "source_identity", lambda _root: "a" * 64)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS", "8")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS", "40")
    seen: list[dict[str, str]] = []

    def pytest_child(
        command: Sequence[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        environment = options["environment"]
        seen.append(dict(environment))
        assert command[command.index("-n") + 1] == "0", "shards must not use xdist"
        assert "tools.pytest_case_reporter" in command
        assert "scripts.postgres_shard_worker" in command
        assert environment[ALLOCATION_ENV] == str(tmp_path / ALLOCATION_DIRECTORY)
        assert Path(environment["PGPASSFILE"]).is_file(), (
            "pytest lacks its private grant"
        )
        assert environment["GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS"] == "8"
        assert environment["GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS"] == "40"
        return subprocess.CompletedProcess(
            command, status, "example-private-output", "example-private-error"
        )

    monkeypatch.setattr(worker, "run_owned_command", pytest_child)
    result = worker.run_worker(
        state=tmp_path,
        tests=("tests/test_example.py",),
        workers=4,
        index=1,
        durations=7,
        identity="a" * 64,
    )
    assert result == status
    assert len(seen) == 1
    assert docker.removed == [CID]
    assert not (tmp_path / ALLOCATION_DIRECTORY).exists(), "worker left credentials"
    assert (tmp_path / worker.FAILURE_FILE).exists() is bool(status)


def test_worker_refuses_inherited_database_before_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setenv("PGPASSWORD", "example-do-not-use")
    with pytest.raises(PostgresShardError, match="inherited"):
        worker.run_worker(
            state=tmp_path, tests=(), workers=1, index=0, durations=0, identity="a" * 64
        )
    assert not (tmp_path / ALLOCATION_DIRECTORY).exists(), (
        "a refused worker created an allocation"
    )


def test_parent_reconciles_using_actual_allocator_metadata(tmp_path: Path) -> None:
    state, docker, owned = allocation(tmp_path)
    assert owned.phase == "READY"
    worker.reconcile_allocation(state, docker)
    assert docker.removed == [CID], (
        "parent did not remove exactly its interrupted shard"
    )
    assert not owned.directory.exists(), "confirmed cleanup retained shard credentials"


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "foreign"),
        ("container_id", "short"),
        ("image_id", "postgres:16"),
        ("docker_host", "tcp://external:2375"),
        ("directory_inode", 0),
        ("create_attempted", False),
        ("port", True),
        ("name", "foreign"),
        ("image_reference", "postgres:15"),
        ("phase", "UNKNOWN"),
    ],
)
def test_reconciliation_refuses_malformed_or_replaced_ownership(
    tmp_path: Path, field: str, value: object
) -> None:
    state, docker, owned = allocation(tmp_path)
    metadata = json.loads((owned.directory / "ownership.json").read_text())
    metadata[field] = value
    write_json_atomic(owned.directory / "ownership.json", metadata)
    calls = len(docker.calls)
    with pytest.raises(PostgresShardError):
        worker.reconcile_allocation(state, docker)
    assert len(docker.calls) == calls, "invalid ownership authorized a Docker operation"
    assert owned.directory.exists() and CID in docker.containers


def test_replaced_allocation_directory_is_not_removed(tmp_path: Path) -> None:
    state, docker, owned = allocation(tmp_path)
    moved = tmp_path / "original"
    owned.directory.rename(moved)
    owned.directory.mkdir(mode=0o700)
    write_json_atomic(
        owned.directory / "ownership.json",
        json.loads((moved / "ownership.json").read_text()),
    )
    with pytest.raises(PostgresShardError):
        worker.reconcile_allocation(state, docker)
    assert docker.removal_attempts == [], "replacement directory was adopted"
    assert moved.exists() and owned.directory.exists()


def test_uncertain_cleanup_retains_ownership_and_credentials(tmp_path: Path) -> None:
    state, docker, owned = allocation(tmp_path)
    docker.keep_after_remove = True
    with pytest.raises(PostgresShardError, match="unconfirmed"):
        worker.reconcile_allocation(state, docker)
    metadata = json.loads((owned.directory / "ownership.json").read_text())
    assert metadata["phase"] == "UNCONFIRMED"
    assert (owned.directory / "postgres.env").is_file(), (
        "uncertain cleanup erased private reconciliation inputs"
    )


def test_lost_worker_supervision_never_authorizes_parent_cleanup(
    tmp_path: Path,
) -> None:
    state, docker, owned = allocation(tmp_path)
    owned.retain(supervision_lost=True)
    calls = len(docker.calls)
    with pytest.raises(ProcessSupervisionLost):
        worker.reconcile_allocation(state, docker)
    assert len(docker.calls) == calls and CID in docker.containers
    assert owned.directory.exists(), "lost-supervision evidence was erased"


@pytest.mark.parametrize(
    "defect",
    [
        "container",
        "owner",
        "port",
        "inode",
        "image",
        "missing-image",
        "lost",
        "not-ready",
    ],
)
def test_independence_receipts_require_complete_local_pg16_identity(
    tmp_path: Path, defect: str
) -> None:
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    state, _docker, first = allocation(first_root)
    other_state, _other_docker, second = allocation(second_root)
    first_receipt = json.loads((first.directory / "ownership.json").read_text())
    second_receipt = json.loads((second.directory / "ownership.json").read_text())
    second_receipt.update(container_id="c" * 64, port=54322)
    if defect in {"container", "owner", "port", "inode"}:
        field = {
            "container": "container_id",
            "owner": "owner",
            "port": "port",
            "inode": "directory_inode",
        }[defect]
        second_receipt[field] = first_receipt[field]
    elif defect == "image":
        second_receipt["image_id"] = "sha256:" + "d" * 64
    elif defect == "missing-image":
        first_receipt.pop("image_id")
        second_receipt.pop("image_id")
    elif defect == "lost":
        second_receipt["supervision_lost"] = True
    elif defect == "not-ready":
        second_receipt["phase"] = "STARTED"
    write_json_atomic(state / worker.ALLOCATION_FILE, first_receipt)
    write_json_atomic(other_state / worker.ALLOCATION_FILE, second_receipt)
    jobs = [runner.ShardJob(0, state, (), {}), runner.ShardJob(1, other_state, (), {})]
    with pytest.raises(PostgresShardError):
        runner.validate_independent_allocations(jobs)


@pytest.mark.parametrize("lost", [False, True])
def test_group_cleanup_continues_only_while_supervision_is_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: bool
) -> None:
    visited: list[Path] = []
    jobs = [runner.ShardJob(index, tmp_path / str(index), (), {}) for index in range(3)]

    def reconcile(state: Path, _docker: object) -> None:
        visited.append(state)
        if state == jobs[0].state:
            if lost:
                raise ProcessSupervisionLost("example")
            raise PostgresShardError("example")

    monkeypatch.setattr(runner, "reconcile_allocation", reconcile)
    with pytest.raises(ProcessSupervisionLost if lost else PostgresShardError):
        runner.reconcile_jobs(jobs)
    assert visited == [job.state for job in (jobs[:1] if lost else jobs)]


@pytest.mark.parametrize("workers", [0, 17, -1])
def test_cli_worker_budget_fails_before_spawning(
    workers: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        runner.main(["--workers", str(workers), "--tests", "tests/test_example.py"])
        == 2
    )
    assert "within 1..16" in capsys.readouterr().err, (
        "the worker budget names its 1..16 bound before any shard is spawned"
    )


def test_cli_uses_agreed_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def run(python: str, **options: Any) -> Path:
        calls.append((python, options))
        return tmp_path

    monkeypatch.setattr(runner, "run_postgres_shards", run)
    assert runner.main(["--tests", "tests/test_example.py"]) == 0
    assert calls == [
        (
            sys.executable,
            {"workers": 4, "durations": 50, "tests": ["tests/test_example.py"]},
        )
    ]
