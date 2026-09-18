from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_preempt036_stuck_workflow_reconcile as runner
from tests.regional._cov95_cap_probes import RecordingDatabase, RecordingStore


class FakeContainers:
    def __init__(self, failure: str = "") -> None:
        self.failure = failure
        self.calls: list[list[str]] = []
        self.readiness_calls = 0

    def run(self, command, **kwargs):
        self.calls.append(command)
        assert command[0] == "/fixture/docker", "no real process may be launched"
        assert kwargs["timeout"] > 0, "every container operation needs a finite timeout"
        verb = command[1]
        output, code = "", 0
        if verb == "run":
            code = int(self.failure == "start")
            output = "fixture-container"
        elif verb == "exec":
            self.readiness_calls += 1
            code = int(self.failure == "ready" or self.readiness_calls == 1)
        elif verb == "inspect":
            if self.failure == "inspect":
                raise subprocess.CalledProcessError(
                    1, command, stderr="fixture inspect"
                )
            output = "54321\n"
        elif verb == "rm":
            code = int(self.failure == "remove")
        else:
            raise AssertionError(f"unexpected fake container operation: {verb}")
        return subprocess.CompletedProcess(command, code, output, "fixture refusal")


@pytest.mark.parametrize(
    ("backend", "docker", "failure", "code", "recorded_backend"),
    [
        ("auto", False, "", 0, "sqlite"),
        ("postgres", False, "", 1, None),
        ("auto", True, "start", 0, "sqlite"),
        ("postgres", True, "start", 1, None),
        ("postgres", True, "ready", 1, None),
        ("postgres", True, "inspect", 1, None),
        ("postgres", True, "remove", 1, "postgres"),
        ("postgres", True, "", 0, "postgres"),
    ],
)
def test_public_runner_provisions_only_through_recording_boundaries(
    backend, docker, failure, code, recorded_backend, tmp_path, monkeypatch
) -> None:
    import psycopg

    containers = FakeContainers(failure)
    database = RecordingDatabase([])
    stores = []
    shapes = []
    monkeypatch.setattr(
        runner,
        "shutil",
        SimpleNamespace(which=lambda _name: "/fixture/docker" if docker else None),
    )
    monkeypatch.setattr(
        runner,
        "subprocess",
        SimpleNamespace(run=containers.run, DEVNULL=subprocess.DEVNULL),
    )
    monkeypatch.setattr(runner, "time", SimpleNamespace(sleep=lambda _seconds: None))
    monkeypatch.setattr(runner, "POSTGRES_READY_ATTEMPTS", 2)
    monkeypatch.setattr(psycopg, "connect", database.connect)

    def open_store(url, **kwargs):
        stores.append((url, kwargs))
        return RecordingStore([])

    monkeypatch.setattr(runner, "PostgresStore", open_store)

    def shape(name, provisioner, **_kwargs):
        shapes.append(name)
        url = provisioner.create(name)
        assert provisioner.owns(url), "only newly registered targets may be opened"
        store = provisioner.open(url)
        store.close()
        with pytest.raises(runner.RunnerError, match="did not create"):
            provisioner.open("postgresql://unowned.invalid/never-connect")
        return {"verdict": "PASS", "errors": []}

    monkeypatch.setattr(runner, "run_shape", shape)
    assert runner.main(["--run-dir", str(tmp_path), "--backend", backend]) == code, (
        "provisioning/teardown errors must affect the public runner exit"
    )
    path = tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json"
    result = json.loads(path.read_text())
    assert result["store_backend"] == recorded_backend, "report the selected fake path"
    assert result["verdict"] == ("PASS" if code == 0 else "FAIL"), (
        "evidence must agree with the runner exit code"
    )
    verbs = [item[1] for item in containers.calls]
    if docker and failure != "start":
        assert verbs[-1] == "rm", "every started fake container must enter teardown"
    if recorded_backend == "postgres":
        assert len(stores) == len(runner.SHAPES), "each shape gets its own fake store"
        assert all(options == {"initialize_schema": True} for _, options in stores), (
            "the isolated harness must request the public schema initialization path"
        )
        assert database.closed == 3, "each fake CREATE DATABASE connection must close"
        assert all(
            statement.startswith('CREATE DATABASE "gpu_fault_p036_')
            for statement, _ in database.statements
        ), "database names must be escaped through SQL identifiers"
    if code == 1:
        assert result["errors"], "a provisioning failure needs an explicit diagnostic"


@pytest.mark.parametrize("name", ["../foreign", "a; DROP DATABASE other", "a" * 80])
def test_invalid_isolated_database_names_never_connect(name, monkeypatch) -> None:
    import psycopg

    database = RecordingDatabase([])
    monkeypatch.setattr(psycopg, "connect", database.connect)
    provisioner = runner.PostgresProvisioner(
        "postgresql://fixture.invalid/postgres", "fake"
    )
    with pytest.raises(runner.RunnerError, match="unexpected database name"):
        provisioner.create(name)
    assert database.connections == [], (
        "invalid names must fail before the fake connection"
    )


@pytest.mark.parametrize(
    ("output", "returncode", "message"),
    [("", 2, "exited 2"), ("not-json", 0, "non-JSON"), ("[]", 0, "non-object")],
)
def test_shipped_probe_failure_closes_owned_sqlite_store(
    output, returncode, message, tmp_path, monkeypatch
) -> None:
    provisioner = runner.SqliteProvisioner(tmp_path, "unit fixture")
    opened = []
    closed = []
    calls = []
    original_open = provisioner.open

    def open_store(url):
        store = original_open(url)
        opened.append(store)
        close = store.close

        def record_close():
            closed.append(store)
            close()

        monkeypatch.setattr(store, "close", record_close)
        return store

    def run(command, **kwargs):
        calls.append(kwargs)
        assert command[1] == "-c", "the harness passes its shipped public probe source"
        return subprocess.CompletedProcess(
            command, returncode, output, "fixture child error"
        )

    monkeypatch.setattr(provisioner, "open", open_store)
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=run))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "ambient-must-not-be-used")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fixture-not-inherited")
    with pytest.raises(runner.RunnerError, match=message):
        runner.run_shape(
            "compile-blocked",
            provisioner,
            safety_probe="safety-fixture",
            stats_probe="stats-fixture",
        )
    assert len(calls) == 1, "a failed read-only probe must not advance to later stages"
    environment = calls[0]["env"]
    assert provisioner.owns(environment["GPU_FAULT_STORE_URL"]), (
        "child probes must receive only a store created by this provisioner"
    )
    assert "AWS_SECRET_ACCESS_KEY" not in environment, (
        "ambient credentials are excluded"
    )
    assert environment["KUBECONFIG"] == "/dev/null", (
        "child probes must not reach clusters"
    )
    assert len(opened) == 1, "the failing shape must allocate only one store"
    assert closed == opened, "the store must close when its first probe fails"


def test_first_shape_failure_stops_later_shapes_and_remains_failed(
    tmp_path, monkeypatch
) -> None:
    shapes = []

    def fail(name, *_args, **_kwargs):
        shapes.append(name)
        return {"verdict": "FAIL", "errors": ["fixture sweep failed"]}

    monkeypatch.setattr(runner, "run_shape", fail)
    assert runner.main(["--run-dir", str(tmp_path), "--backend", "sqlite"]) == 1, (
        "a failed shape must produce a failed case exit"
    )
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert shapes == [runner.SHAPES[0]], "the first failed shape must stop execution"
    assert result["errors"][0].endswith("fixture sweep failed"), (
        "retain the causal failure"
    )
    assert "all three distinct" in result["errors"][-1], "partial evidence cannot pass"
