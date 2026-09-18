from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.postgres_grant import ALLOCATION_ENV, LOCAL_DOCKER_HOST
from gpu_fault.admin.release_postgres import ALLOCATION_DIRECTORY
from scripts import ci_coverage_config, run_postgres_shards
from scripts.e2e.regional import run_cap005_postgres_suite as cap005
from tests._parallel_postgres_support import (
    DISCOVERED,
    IDENTITY,
    TARGETS,
    receipt_payload,
)
from tools import pytest_result_identity

LOCAL_URL = "postgresql://localhost:55432/postgres"


class Harness:
    def __init__(self, root: Path, receipts: Path) -> None:
        self.root = root
        self.receipts = receipts
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self.database_events: list[str] = []
        self.workers: list[int] = []
        self.before_reference: Callable[[], None] = lambda: None
        self.after_reference: Callable[[], None] = lambda: None
        self.child_exit = 0
        self.contract_exit = 0
        self.database_residual = False
        self.change_after_contract = False
        self.change_during_cleanup = False
        self.malformed_serial_report = False
        self.identity = IDENTITY

    def allocation(self, index: int) -> dict[str, Any]:
        owner = f"{index + 1:032x}"
        return {
            "schema_version": 1,
            "owner": owner,
            "name": "gpu-fault-release-postgres-" + owner,
            "image_id": "sha256:" + "e" * 64,
            "image_reference": "postgres:16",
            "docker_host": LOCAL_DOCKER_HOST,
            "container_id": f"{index + 1:064x}",
            "port": 54000 + index,
            "directory_device": 1,
            "directory_inode": index + 1,
            "create_attempted": True,
            "supervision_lost": False,
            "phase": "READY",
        }

    def launch(self, python: str, *, workers: int, durations: int, tests: Any) -> Path:
        assert (python, durations, tuple(tests)) == (sys.executable, 20, TARGETS), (
            "parallel CAP-005 must use the active interpreter and canonical inventory"
        )
        assert os.environ["GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS"] == "8", (
            "parallelism must not reduce the stress workload"
        )
        assert os.environ["GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS"] == "40", (
            "parallelism must preserve all stress rounds"
        )
        self.workers.append(workers)
        now = datetime.now(timezone.utc).isoformat()
        for index in range(workers):
            state = self.receipts / f"shard-{index}"
            value = receipt_payload(index, workers)
            value["session"].update(started_at=now, finished_at=now)
            write_json_atomic(state / "pytest.json", value)
            write_json_atomic(state / "allocation.json", self.allocation(index))
        write_json_atomic(
            self.receipts / "run.json",
            {
                "kind": "local-postgres-shards",
                "status": "COMPLETE",
                "source_identity": IDENTITY,
                "workers": workers,
                "tests": list(TARGETS),
                "executed_tests": len(DISCOVERED),
                "elapsed_seconds": 1.0,
            },
        )
        self.before_reference()
        return self.receipts

    def command(
        self, command: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        environment = options["env"]
        self.environments.append(dict(environment))
        if command[:3] == [sys.executable, "-B", "-c"]:
            output = io.StringIO()
            with pytest.MonkeyPatch.context() as child:
                child.setattr(os, "environ", dict(environment))
                with redirect_stdout(output), redirect_stderr(io.StringIO()):
                    cap005.parallel_postgres_child(int(command[-1]))
            self.after_reference()
            return subprocess.CompletedProcess(
                command, self.child_exit, output.getvalue(), ""
            )
        arguments = (
            shlex.split(environment["PYTEST_ADDOPTS"])
            if command[0] == "make"
            else command
        )
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--junitxml", type=Path)
        parsed, _rest = parser.parse_known_args(arguments)
        assert parsed.junitxml is not None, (
            "serial children must retain real JUnit routing"
        )
        parsed.junitxml.write_text(
            "<testsuite"
            if command[0] == "make" and self.malformed_serial_report
            else f'<testsuite tests="3" failures="{self.contract_exit}" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        if command[0] != "make":
            assert (
                options["isolated_postgres_url"]
                == environment["GPU_FAULT_TEST_POSTGRES_URL"]
            ), "the separate full contract must explicitly use its CAP-005 database"
            if self.change_after_contract:
                self.identity = "b" * 64
        return subprocess.CompletedProcess(command, self.contract_exit, "", "")

    def connect(self, url: str, **_options: Any) -> Any:
        assert url == LOCAL_URL, (
            "database lifecycle must use only the local admin connection"
        )
        harness = self

        class Connection:
            info = SimpleNamespace(server_version=160004)
            row: tuple[Any, ...] = (False,)

            def __enter__(self) -> Connection:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def cursor(self) -> Connection:
                return self

            def execute(self, statement: Any, *_args: Any) -> Connection:
                text = (
                    statement if isinstance(statement, str) else statement.as_string()
                )
                if text.startswith("CREATE DATABASE"):
                    harness.database_events.append("create")
                elif text.startswith("DROP DATABASE"):
                    harness.database_events.append("drop")
                elif text.startswith("SELECT count(*) FROM pg_database"):
                    harness.database_events.append("verify_absence")
                    self.row = (int(harness.database_residual),)
                    if harness.change_during_cleanup:
                        harness.identity = "b" * 64
                return self

            def fetchone(self) -> tuple[Any, ...]:
                return self.row

        return Connection()


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    for target in TARGETS:
        (root / target).touch()
    value = Harness(root, tmp_path / "gpu-fault-postgres-shards-fixture")
    monkeypatch.setattr(cap005, "ROOT", root)
    monkeypatch.setattr(cap005, "run_fixture_command", value.command)
    monkeypatch.setattr(ci_coverage_config, "pytest_targets", lambda *_args: TARGETS)
    monkeypatch.setattr(run_postgres_shards, "run_postgres_shards", value.launch)
    monkeypatch.setattr(
        pytest_result_identity, "source_identity", lambda _root: value.identity
    )
    monkeypatch.setattr("psycopg.connect", value.connect)
    yield value


@pytest.mark.parametrize("workers", [2, 8])
def test_parallel_routes_to_owned_shards_and_separate_contract(
    harness: Harness, workers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "GPU_FAULT_TEST_POSTGRES_URL": "untrusted-test-reference",
        "GPU_FAULT_STORE_URL": "untrusted-store-reference",
        "GPU_FAULT_STORE_URL_FILE": "/private/do-not-read",
        ALLOCATION_ENV: "/private/do-not-adopt",
        "PGPASSFILE": "/private/do-not-read",
        "PGSERVICE": "untrusted",
        "DOCKER_HOST": "untrusted",
        "DOCKER_CONFIG": "/private/do-not-read",
        "AWS_PROFILE": "untrusted",
        "COSIGN_PASSWORD": "test-only-placeholder",
        "PYTEST_ADDOPTS": "-k omit",
        "POSTGRES_TEST_PARALLEL": "0",
        "POSTGRES_TEST_WORKERS": "1",
        "MAKEFLAGS": "POSTGRES_TESTS=tests/subset.py",
        "PYTHONPATH": "/private/do-not-import",
        "HOME": "/private/do-not-use",
        "TMPDIR": "/private/do-not-use",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    result = cap005.run_suite(LOCAL_URL, harness.root, postgres_workers=workers)
    assert result["status"] == "PASS", result
    assert harness.workers == [workers], (
        "the requested process count must reach the existing launcher"
    )
    assert harness.commands[0][:3] == [sys.executable, "-B", "-c"], (
        "shards need a clean child process"
    )
    assert harness.commands[1][:5] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/store/test_store_contracts.py",
    ], "the independent full-contract invocation must remain mandatory"
    environment = harness.environments[0]
    for name in inherited:
        assert environment.get(name) != inherited[name], (
            f"parallel child inherited {name}"
        )
    assert not any(
        name.startswith(("PG", "POSTGRES_", "PYTEST_", "MAKE", "DOCKER_"))
        for name in environment
    ), (
        "the shard bootstrap must not inherit a database, grant, filter or Docker authority"
    )
    assert (
        environment["AWS_CONFIG_FILE"]
        == environment["AWS_SHARED_CREDENTIALS_FILE"]
        == os.devnull
    ), "cloud configuration must remain disabled"
    assert environment["KUBECONFIG"] == os.devnull, (
        "parallel tests must not inherit cluster access"
    )
    assert result["suites"] == {
        "contract": {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}
    }, "parallel receipts must not be converted into invented PostgreSQL JUnit"
    proof = result["postgres_shards"]
    assert proof["executed_tests"] == len(DISCOVERED) and proof["cleanup_verified"], (
        "native success requires full discovery and shard cleanup"
    )
    assert len(proof["receipt_sha256"]) == 1 + 2 * workers, (
        "retain every original receipt binding"
    )
    assert harness.database_events == ["create", "drop", "verify_absence"], (
        "the separate contract database must still have verified owned cleanup"
    )
    assert not list(harness.root.rglob("postgres.xml")), (
        "parallel mode must not fabricate a JUnit report"
    )


def rewrite(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    write_json_atomic(path, value)


@pytest.mark.parametrize(
    "defect",
    [
        "run-failed",
        "run-source",
        "run-workers",
        "run-count",
        "run-targets",
        "run-duration",
        "run-boolean-duration",
        "missing-run",
        "missing-shard",
        "missing-allocation",
        "source",
        "failed",
        "incomplete",
        "skip",
        "collection-skip",
        "filtered",
        "partition",
        "discovery",
        "old",
        "shared-allocation",
        "cleanup",
        "cleanup-symlink",
        "failure-marker",
        "cancel-marker",
        "altered-after-return",
        "failed-child",
    ],
)
def test_parallel_proof_failures_never_become_native_success(
    harness: Harness, defect: str
) -> None:
    def corrupt() -> None:
        run = harness.receipts / "run.json"
        state = harness.receipts / "shard-0"
        receipt = state / "pytest.json"
        allocation = state / "allocation.json"
        if defect.startswith("run-"):
            field, value = {
                "run-failed": ("status", "FAILED"),
                "run-source": ("source_identity", "b" * 64),
                "run-workers": ("workers", 1),
                "run-count": ("executed_tests", 1),
                "run-targets": ("tests", list(TARGETS[:1])),
                "run-duration": ("elapsed_seconds", -1),
                "run-boolean-duration": ("elapsed_seconds", True),
            }[defect]
            rewrite(run, lambda payload: payload.update({field: value}))
        elif defect.startswith("missing-"):
            {
                "missing-run": run,
                "missing-shard": receipt,
                "missing-allocation": allocation,
            }[defect].unlink()
        elif defect in {"cleanup", "cleanup-symlink"}:
            path = state / ALLOCATION_DIRECTORY
            if defect == "cleanup":
                path.mkdir(mode=0o700)
            else:
                path.symlink_to(state / "missing-owned-directory")
        elif defect in {"failure-marker", "cancel-marker"}:
            (
                state
                / ("failure.json" if defect == "failure-marker" else "cancel.json")
            ).touch()
        elif defect == "shared-allocation":
            write_json_atomic(
                harness.receipts / "shard-1/allocation.json", harness.allocation(0)
            )
        else:

            def mutate(payload: dict[str, Any]) -> None:
                session = payload["session"]
                nodeid = next(iter(payload["records"]))
                if defect == "source":
                    payload["source_identity"] = "b" * 64
                elif defect == "failed":
                    payload["records"][nodeid]["phases"]["teardown"] = "failed"
                elif defect == "incomplete":
                    del payload["records"][nodeid]
                elif defect == "skip":
                    payload["records"][nodeid]["phases"]["call"] = "skipped"
                elif defect == "collection-skip":
                    session["collection_skips"] = [TARGETS[0]]
                elif defect == "filtered":
                    session["selection"]["keyword"] = "not stress"
                elif defect == "partition":
                    session["selection"]["partition"] = [8, 1]
                elif defect == "discovery":
                    session["discovered_nodeids"].remove(nodeid)
                elif defect == "old":
                    session["started_at"] = "2026-09-01T00:00:00+00:00"
                elif defect == "altered-after-return":
                    payload["records"][nodeid]["duration_seconds"] = 0.5
                else:
                    raise AssertionError("unhandled proof defect")

            rewrite(receipt, mutate)

    if defect == "altered-after-return":
        harness.after_reference = corrupt
    elif defect == "failed-child":
        harness.child_exit = 1
    else:
        harness.before_reference = corrupt
    result = cap005.run_suite(LOCAL_URL, harness.root, postgres_workers=8)
    assert result["status"] == "FAIL" and result["errors"], (
        "defective shard proof cannot pass CAP-005"
    )
    assert result["database_dropped"] is True, (
        "proof rejection must still clean the contract database"
    )
    assert harness.database_events[-2:] == ["drop", "verify_absence"], (
        "cleanup needs an absence readback"
    )
    assert len(harness.commands) == 1, (
        "a rejected parallel stage must stop before the contract"
    )


@pytest.mark.parametrize(
    "defect",
    ["contract", "database-cleanup", "source-after-contract", "source-during-cleanup"],
)
def test_parallel_pass_does_not_override_contract_cleanup_or_source_failure(
    harness: Harness, defect: str
) -> None:
    harness.contract_exit = int(defect == "contract")
    harness.database_residual = defect == "database-cleanup"
    harness.change_after_contract = defect == "source-after-contract"
    harness.change_during_cleanup = defect == "source-during-cleanup"
    result = cap005.run_suite(LOCAL_URL, harness.root, postgres_workers=8)
    assert result["status"] == "FAIL", (
        "the whole case must fail even when shards passed"
    )
    assert len(harness.commands) == 2 and harness.workers == [8], (
        "both stages must actually run"
    )
    assert result["database_dropped"] is not harness.database_residual, (
        "keep the observed cleanup outcome"
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_default_and_explicit_one_preserve_serial_native_case(
    harness: Harness, explicit: bool
) -> None:
    options = {"postgres_workers": 1} if explicit else {}
    result = cap005.run_suite(LOCAL_URL, harness.root, **options)
    assert result["status"] == "PASS", result
    assert harness.workers == [] and "postgres_shards" not in result, (
        "one worker must not allocate shards"
    )
    assert harness.commands[0] == [
        "make",
        "test-postgres-stress",
        f"PYTHON={sys.executable}",
        "PYTEST_XDIST_WORKERS=0",
    ], "the established serial Make invocation must remain unchanged"
    assert set(result["suites"]) == {"postgres", "contract"}, (
        "serial mode still requires both JUnit reports"
    )
    assert harness.database_events == ["create", "drop", "verify_absence"], (
        "serial ownership cleanup regressed"
    )


def test_serial_junit_validation_still_follows_both_suites(harness: Harness) -> None:
    harness.malformed_serial_report = True
    result = cap005.run_suite(LOCAL_URL, harness.root)
    assert result["status"] == "FAIL", "malformed JUnit cannot pass the serial case"
    assert len(harness.commands) == 2, (
        "serial mode must preserve both invocations before reading their reports"
    )
    assert result["database_dropped"], "report parsing failures must retain cleanup"


@pytest.mark.parametrize("workers", [0, 9, -1, True, 1.0, "8", None])
def test_invalid_api_budget_is_rejected_before_any_resource_use(
    harness: Harness, workers: Any
) -> None:
    with pytest.raises(cap005.SuiteError, match="1..8"):
        cap005.run_suite(LOCAL_URL, harness.root, postgres_workers=workers)
    assert not harness.commands and not harness.database_events, (
        "invalid budgets must fail before allocation"
    )
    assert not (harness.root / "artifacts").exists(), (
        "invalid budgets must not create case state"
    )


@pytest.mark.parametrize("workers", ["0", "9", "-1", "1.5", "eight"])
def test_invalid_cli_budget_is_rejected_before_reading_a_dsn(
    monkeypatch: pytest.MonkeyPatch, workers: str
) -> None:
    monkeypatch.setattr(
        cap005, "store_dsn", lambda: pytest.fail("invalid CLI read a DSN")
    )
    with pytest.raises(SystemExit) as error:
        cap005.main(["--postgres-workers", workers])
    assert error.value.code == 2, "invalid worker counts must be argparse usage errors"


def test_cli_eight_selects_the_parallel_case(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cap005, "store_dsn", lambda: LOCAL_URL)
    monkeypatch.setenv("GPU_FAULT_CAP005_WORKDIR", str(harness.root))
    cap005.main(["--postgres-workers", "8"])
    report = json.loads(capsys.readouterr().out)
    assert report["postgres_workers"] == 8 and harness.workers == [8], (
        "CLI must honor the explicit budget"
    )
    assert report["status"] == "PASS" and report["database_dropped"], (
        "CLI output must follow both stages and cleanup"
    )


def test_parallel_cannot_silently_switch_checkout(
    harness: Harness, tmp_path: Path
) -> None:
    with pytest.raises(cap005.SuiteError, match="own checkout"):
        cap005.run_suite(LOCAL_URL, tmp_path, postgres_workers=8)
    assert not harness.commands and not harness.database_events, (
        "source mismatch must fail before database access"
    )
