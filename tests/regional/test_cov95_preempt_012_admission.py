from __future__ import annotations

import copy
import json
import os
import subprocess
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import run_preempt012_acceptance as runner
from tests.regional._cov95_preempt_012 import PreemptionModel


@pytest.mark.parametrize("reuse", [False, True, None])
def test_focused_tests_reuse_only_a_valid_recorded_pass(
    reuse, tmp_path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        runner,
        "reusable_focused_tests",
        lambda _path: {"passed": True, "returncode": 0} if reuse else None,
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 1, "fixture stdout", "fixture stderr"
        )

    monkeypatch.setattr(runner, "RegionalLiveFixture", SimpleNamespace(run=run))
    result = runner.focused_tests(
        tmp_path, reuse_from=tmp_path / "plan.json" if reuse is not None else None
    )
    assert result["passed"] is bool(reuse), "only the recorded PASS may be reused"
    if reuse:
        assert result["reused_from_plan"] is True, "reuse must be explicit evidence"
        assert calls == [], "valid cached tests must not launch another process"
    else:
        assert len(calls) == 1, "missing or invalid cache must run the fake local tests"
        assert calls[0][1] == {"cwd": runner.ROOT, "check": False, "timeout": 600}, (
            "focused test execution must retain its bounded process contract"
        )
        assert (
            tmp_path / "focused-tests.log"
        ).read_text() == "fixture stdoutfixture stderr"
        assert (tmp_path / "focused-tests.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("predecessor", "predecessor evidence"),
        ("ready", "Ready and schedulable"),
        ("taint", "mutation ownership"),
        ("workload", "business workload"),
        ("agent", "not ACTIVE"),
        ("queue", "queue is not empty"),
        ("tests", "regression tests failed"),
    ],
)
def test_preflight_records_each_admission_failure_without_guessing_success(
    failure, message, tmp_path, monkeypatch
) -> None:
    node = {
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    store = {
        "agent": {"lifecycle_state": "ACTIVE"},
        "remote_commands": {"open_by_cluster": {}},
    }
    if failure == "ready":
        node["ready"] = "Unknown"
    elif failure == "taint":
        node["taints"] = [{"key": "gpu-fault.io/owner"}]
    elif failure == "agent":
        store["agent"] = {}
    elif failure == "queue":
        store["remote_commands"]["open_by_cluster"] = {"cluster-a": 1}
    regional = SimpleNamespace(
        node_snapshot=lambda _node: copy.deepcopy(node),
        store_snapshot=lambda **_kwargs: copy.deepcopy(store),
        business_workloads=lambda _node: ["busy"] if failure == "workload" else [],
        cpu_blast_snapshot=lambda: {"pods": ["cpu-one"]},
    )
    monkeypatch.setattr(
        runner, "focused_tests", lambda *_a, **_k: {"passed": failure != "tests"}
    )
    monkeypatch.setattr(runner, "predecessor_path", lambda *_a: ("previous", tmp_path))
    monkeypatch.setattr(
        runner, "predecessor_evidence", lambda *_a: {"valid": failure != "predecessor"}
    )
    result = runner.read_only_preflight(
        regional,
        node="node-a",
        predecessor_path_value=tmp_path / "previous.json",
        case_dir=tmp_path,
    )
    assert len(result["errors"]) == 1 and message in result["errors"][0], (
        "preflight must name the single unproved admission condition"
    )
    assert (
        json.loads((tmp_path / "preflight.json").read_text())["errors"]
        == result["errors"]
    )


def test_missing_formal_predecessor_identity_stops_preflight(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "focused_tests", lambda *_a, **_k: {"passed": True})
    monkeypatch.setattr(runner, "predecessor_path", lambda *_a: (None, None))
    regional = SimpleNamespace(
        node_snapshot=lambda _node: {}, store_snapshot=lambda **_k: {}
    )
    with pytest.raises(
        runner.PreemptAcceptanceError, match="formal predecessor identity"
    ):
        runner.read_only_preflight(
            regional,
            node="node-a",
            predecessor_path_value=tmp_path / "previous.json",
            case_dir=tmp_path,
        )
    assert not (tmp_path / "preflight.json").exists(), (
        "unknown predecessor must not produce valid evidence"
    )


@pytest.mark.parametrize("failure", ["missing-formal", "changed-identity"])
def test_execute_refuses_invalid_predecessor_before_host_creation(
    failure, tmp_path, monkeypatch
) -> None:
    predecessor_id, predecessor_file = runner.predecessor_path(
        tmp_path, runner.CASE_ID, None
    )
    assert predecessor_id is not None and predecessor_file is not None, (
        "the test must start from the real formal predecessor"
    )
    if failure == "missing-formal":
        monkeypatch.setattr(runner, "predecessor_path", lambda *_args: (None, None))
    else:
        predecessor_id = "GF-REGIONAL-PREEMPT-011"
    monkeypatch.setattr(
        runner,
        "authorize_execution",
        lambda *_a, **_k: datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setattr(runner, "focused_tests", lambda *_a, **_k: {"passed": True})
    monkeypatch.setattr(
        runner,
        "predecessor_evidence",
        lambda *_a: pytest.fail(
            "invalid predecessor identity reached evidence loading"
        ),
    )
    monkeypatch.setattr(
        runner,
        "HostProbeFixture",
        lambda *_a: pytest.fail("invalid predecessor identity reached host creation"),
    )
    regional = SimpleNamespace(
        node_snapshot=lambda _node: {}, store_snapshot=lambda **_k: {}
    )
    with pytest.raises(
        runner.PreemptAcceptanceError, match="formal predecessor identity"
    ):
        runner.execute_case(
            Namespace(run_dir=tmp_path, attempt=1),
            regional,
            node="node-a",
            image="fixture/probe@sha256:" + "a" * 64,
            predecessor_id=predecessor_id,
            predecessor_path_value=predecessor_file,
            environment={},
        )
    case_dir = tmp_path / "cases" / runner.CASE_ID
    assert not (case_dir / "preflight.json").exists(), (
        "invalid predecessor identity must not produce a preflight receipt"
    )
    assert not (case_dir / f"{runner.CASE_ID}.json").exists(), (
        "invalid predecessor identity must not produce execution evidence"
    )


def install_entry(tmp_path, monkeypatch):
    model = PreemptionModel(tmp_path, monkeypatch)
    args = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--node",
            "node-a",
            "--host-probe-image",
            "fixture/probe@sha256:" + "a" * 64,
        ]
    )
    monkeypatch.setattr(
        runner, "parser", lambda: SimpleNamespace(parse_args=lambda: args)
    )
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(
        runner, "os", SimpleNamespace(umask=lambda _mask: None, getenv=os.getenv)
    )
    monkeypatch.setattr(runner, "settings_from_arguments", lambda _args: model.settings)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _settings: model)
    monkeypatch.setattr(
        runner,
        "predecessor_path",
        lambda *_args: ("previous", tmp_path / "previous.json"),
    )
    monkeypatch.setattr(guard, "source_digest", lambda: "fixture-source")
    monkeypatch.setattr(
        runner,
        "build_plan",
        lambda **kwargs: {
            "details": kwargs["details"],
            "preflight_passed": kwargs["preflight_passed"],
        },
    )
    return model, args


@pytest.mark.parametrize("passed", [False, True])
def test_entry_plan_binds_focused_results_without_creating_host(
    passed, tmp_path, monkeypatch, capsys
) -> None:
    model, _args = install_entry(tmp_path, monkeypatch)
    if not passed:
        model.preflight_errors = ["fixture refusal"]
    assert runner.main() == int(not passed), (
        "the preflight determines the planning exit"
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["preflight_passed"] is passed, (
        "planning must retain failed preflight evidence"
    )
    assert plan["details"]["focused_tests_source_digest"] == "fixture-source"
    assert model.calls == [], "plan-only mode must not create or execute a host probe"


@pytest.mark.parametrize("failure", ["image", "predecessor"])
def test_entry_rejects_missing_identity_before_host_use(
    failure, tmp_path, monkeypatch
) -> None:
    model, args = install_entry(tmp_path, monkeypatch)
    if failure == "image":
        args.host_probe_image = "fixture/probe:mutable"
    else:
        monkeypatch.setattr(runner, "predecessor_path", lambda *_args: (None, None))
    with pytest.raises(
        runner.PreemptAcceptanceError, match="immutable digest|predecessor resolution"
    ):
        runner.main()
    assert model.calls == [], (
        "invalid image or predecessor must stop before a host action"
    )


def test_execute_entry_drives_the_fake_public_lifecycle(tmp_path, monkeypatch) -> None:
    model, args = install_entry(tmp_path, monkeypatch)
    args.execute = True
    assert runner.main() == 0, "approved fake execution should exercise the lifecycle"
    assert model.calls[-3:] == ["cleanup", "remove", "node"], (
        "entry execution must finish cleanup"
    )
