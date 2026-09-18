"""Window runner exit evidence survives interruption and uncertain cleanup."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_window_fixture as module
from tests.regional._cov95_collect_net import (  # noqa: F401
    StopLoop,
    no_external_effects,
)

CASE = "GF-REGIONAL-COLLECT-019"


@pytest.fixture
def window_runner(tmp_path: Path, monkeypatch: Any) -> Any:
    calls = []
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    settings = SimpleNamespace(
        regional=SimpleNamespace(cluster_id="cluster-a"),
        node="node-a",
        host_probe_image="example@sha256:" + "a" * 64,
        environment=lambda: {"fixture": "value"},
    )
    regional = SimpleNamespace(
        evidence_identity=lambda: {
            "release_id": "release-a",
            "cluster_id": "cluster-a",
        },
        cpu_blast_snapshot=lambda: {},
    )
    state = SimpleNamespace(
        residuals={
            "pod": False,
            "configmap": False,
            "host_script": False,
            "creation_unresolved": False,
        },
        cleanup_error=None,
        create_error=None,
        execute_error=None,
        preflight={"errors": [], "predecessor": {"valid": True}, "cpu_blast": {}},
        regional=regional,
        calls=calls,
    )

    class Fixture:
        def __init__(self, instance: Any, **kwargs: Any) -> None:
            assert instance is regional
            calls.append(("fixture", kwargs))

        def create(self) -> None:
            calls.append(("create",))
            if state.create_error:
                raise state.create_error

        def cleanup(self) -> Any:
            calls.append(("cleanup",))
            if state.cleanup_error:
                raise state.cleanup_error
            return state.residuals

    monkeypatch.setattr(module, "install_site_profile", lambda: calls.append(("site",)))
    monkeypatch.setattr(
        module, "install_abort_signals", lambda: calls.append(("signals",))
    )
    monkeypatch.setattr(
        module,
        "os",
        SimpleNamespace(
            **{**vars(os), "umask": lambda value: calls.append(("umask", value))}
        ),
    )
    monkeypatch.setattr(module, "configure", lambda *a: settings)
    monkeypatch.setattr(module, "read_only_preflight", lambda *a: state.preflight)
    monkeypatch.setattr(module, "authorize_execution", lambda *a, **k: deadline)
    monkeypatch.setattr(module, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(module, "CollectorWindowFixture", Fixture)
    monkeypatch.setattr(
        module,
        "build_plan",
        lambda **kwargs: {
            "case_id": CASE,
            "preflight_passed": kwargs["preflight_passed"],
        },
    )

    def execute(*args: Any) -> dict[str, Any]:
        calls.append(("execute",))
        if state.execute_error:
            raise state.execute_error
        return {"verdict": "PASS", "errors": []}

    def run(*extra: str) -> int:
        monkeypatch.setattr(
            sys, "argv", ["window-runner", "--run-dir", str(tmp_path), *extra]
        )
        return module.run_window_case(
            case_id=CASE,
            confirmation="CONFIRM",
            parser_description="fixture",
            plan_details=lambda *a: {},
            execute=execute,
        )

    state.run = run
    state.path = tmp_path / "cases" / CASE / f"{CASE}.json"
    return state


@pytest.mark.parametrize("residuals", [{}, {"pod": None}, {"pod": 0}])
def test_unknown_cleanup_cannot_pass(window_runner: Any, residuals: Any) -> None:
    window_runner.residuals = residuals
    assert window_runner.run("--execute", "--confirm", "CONFIRM") == 1
    assert json.loads(window_runner.path.read_text())["verdict"] == "FAIL"


def test_abort_persists_failed_evidence_after_owned_cleanup(window_runner: Any) -> None:
    window_runner.execute_error = StopLoop()
    with pytest.raises(StopLoop):
        window_runner.run("--execute", "--confirm", "CONFIRM")
    assert ("cleanup",) in window_runner.calls, "abort must restore owned fixture"
    assert window_runner.path.is_file(), "abort must persist final FAIL evidence"
    result = json.loads(window_runner.path.read_text())
    assert result["verdict"] == "FAIL"
    assert result["ended_at"], "interrupted evidence must have terminal timestamp"


@pytest.mark.parametrize(
    "failure", [None, "create", "execute", "cleanup", "blast", "residual"]
)
def test_window_runner_records_stage_failure_and_always_cleans(
    window_runner: Any, failure: str | None
) -> None:
    if failure in {"create", "execute", "cleanup"}:
        setattr(window_runner, failure + "_error", RuntimeError(failure + " failed"))
    elif failure == "blast":
        window_runner.regional.cpu_blast_snapshot = lambda: {"changed": True}
    elif failure == "residual":
        window_runner.residuals["pod"] = True
    assert window_runner.run("--execute", "--confirm", "CONFIRM") == (
        0 if failure is None else 1
    )
    result = json.loads(window_runner.path.read_text())
    assert result["verdict"] == ("PASS" if failure is None else "FAIL")
    assert window_runner.calls[-1] == ("cleanup",)


@pytest.mark.parametrize("valid", [False, True])
def test_default_plan_passes_actual_preflight_result_without_creating_fixture(
    window_runner: Any, valid: bool
) -> None:
    if not valid:
        window_runner.preflight["errors"] = ["not ready"]
    assert window_runner.run() == (0 if valid else 1)
    assert not any(call[0] == "fixture" for call in window_runner.calls), (
        "plan cannot create resources"
    )


@pytest.mark.parametrize("error", ["confirm", "preflight"])
def test_execute_admission_failure_cannot_create_fixture(
    window_runner: Any, error: str
) -> None:
    if error == "preflight":
        window_runner.preflight["errors"] = ["not ready"]
    with pytest.raises(module.RegionalFixtureError):
        window_runner.run(
            "--execute", "--confirm", "wrong" if error == "confirm" else "CONFIRM"
        )
    assert not any(call[0] == "fixture" for call in window_runner.calls), (
        "failed admission cannot create resources"
    )
