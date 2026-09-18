from __future__ import annotations

import argparse
import importlib
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr011_provider_replace as retired_case
from scripts.e2e.regional import run_destr016_preempting_reboot as preemption_case
from scripts.e2e.regional import run_destr020_identity_mismatch_isolation as alias_case
from scripts.e2e.regional import run_destr024_watcher_down_fail_closed as watcher_case
from scripts.e2e.regional.probes import destr014_node_probe as branch_probe
from scripts.e2e.regional.probes import destr016_node_probe as holder_probe
from scripts.e2e.regional.probes import destr019_node_probe as ledger_probe
from scripts.e2e.regional.probes.destr021_annotation_writer import AnnotationWriter
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError


@pytest.mark.parametrize(
    ("name", "first_mutation"),
    [
        ("run_destr014_branch_exhaustion", "_arm_and_inject"),
        ("run_destr015_parallel_branch_join", "_start_job_and_probes"),
        ("run_destr016_preempting_reboot", "_arm_and_park"),
        ("run_destr017_out_of_band_reboot_fence", "_start_probes"),
        ("run_destr018_lifetime_deadline", "_baseline_host"),
        ("run_destr019_agent_restart_ledger", "_baseline"),
        ("run_destr020_identity_mismatch_isolation", "_inject_and_observe"),
        ("run_destr021_adversarial_node_metadata", "_preseed"),
        ("run_destr022_spare_reservation_reclaim", "_inject"),
    ],
)
def test_expired_case_does_not_start_mutation_but_still_cleans_up(
    name: str, first_mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    calls: list[str] = []
    settings = SimpleNamespace(
        node="node-a",
        nodes=("node-a", "node-b"),
        reference_node="node-a",
        spare_node="node-b",
        job_id="job",
        attempt_id="job-a001",
        variant="sibling-exhausted",
    )
    run = SimpleNamespace(
        settings=settings,
        case_dir=tmp_path,
        regional=object(),
        warm=object(),
        workload=object(),
        prewarm=object(),
        fault_probe=object(),
        sibling_probe=object(),
        inject_fault=object(),
        inject_sibling=object(),
        run_id="run",
        incident_id="incident",
        alias="acceptance-alias-review",
        foreign_incident="acceptance-foreign-review",
        env_baseline=tmp_path / "executor.json",
        control_env_baseline=tmp_path / "control.json",
        env_opened=False,
        control_env_opened=False,
        holder_armed=False,
        agent_disabled=False,
        preflight={
            "store": {
                "profile": {"profile_version": "profile"},
                "queue": {"depth": 0, "fault_backlog_depth": 0},
            }
        },
    )

    def mutate(*_args: Any, **_kwargs: Any) -> None:
        calls.append("mutate")
        raise AssertionError("mutation after deadline")

    def cleanup(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("cleanup")
        return {"errors": []}

    monkeypatch.setattr(module, "_prepare_live_run", lambda *_args: run)
    monkeypatch.setattr(module, first_mutation, mutate)
    monkeypatch.setattr(module, "_cleanup", cleanup)
    end = datetime.now(timezone.utc) - timedelta(seconds=1)

    assert module.execute_case(settings, tmp_path, 1, end) == 1
    report = json.loads((tmp_path / f"{module.CASE_ID}.json").read_text())
    assert report["verdict"] == "FAIL", report
    assert "window" in report["error"], report
    assert calls == ["cleanup"], calls


def test_alias_absence_requires_a_successful_kubernetes_read() -> None:
    calls: list[dict[str, Any]] = []

    class Transport:
        def kubectl(self, *_args: Any, **kwargs: Any) -> str:
            calls.append(kwargs)
            if kwargs.get("check", True):
                raise RegionalFixtureError("read failed")
            return ""

    with pytest.raises(RegionalFixtureError, match="read failed"):
        alias_case.kubernetes_node_present(Transport(), "acceptance-alias-review")  # type: ignore[arg-type]
    assert calls[0].get("check", True) is True


def test_dead_watchdog_cannot_authorize_watcher_scale_down(tmp_path: Path) -> None:
    calls: list[str] = []

    class Transport:
        def kubectl(self, *_args: Any, **_kwargs: Any) -> str:
            calls.append("scale")
            raise AssertionError("dead watchdog must refuse before scaling")

    fixture = watcher_case.WatcherScaleFixture(
        Transport(),
        case_dir=tmp_path,
        baseline_replicas=1,  # type: ignore[arg-type]
    )
    fixture.watchdog = SimpleNamespace(poll=lambda: 1)  # type: ignore[assignment]

    with pytest.raises(RegionalFixtureError, match="watchdog"):
        fixture.scale_down()
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "run_destr020_identity_mismatch_isolation",
        "run_destr022_spare_reservation_reclaim",
    ],
)
@pytest.mark.parametrize("passed", [False, True])
def test_custom_plan_call_binds_parsed_arguments_and_real_preflight_verdict(
    name: str, passed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    arguments = argparse.Namespace(run_dir=tmp_path, attempt=1, execute=False)
    captured: dict[str, Any] = {}
    settings = SimpleNamespace(environment=lambda: {"cluster_id": "cluster-a"})
    monkeypatch.setattr(module, "install_site_profile", lambda: None)
    monkeypatch.setattr(module, "install_abort_signals", lambda: None)
    monkeypatch.setattr(
        module, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(module, "configure", lambda _args: settings)
    monkeypatch.setattr(
        module,
        "read_only_preflight",
        lambda *_args, **_kwargs: {"errors": [] if passed else ["unsafe"]},
    )
    monkeypatch.setattr(module, "plan_details", lambda *_args: {"target": "node-a"})

    def build_plan(
        *, arguments: argparse.Namespace, preflight_passed: bool, **kwargs: Any
    ) -> dict[str, Any]:
        captured.update(
            arguments=arguments, preflight_passed=preflight_passed, **kwargs
        )
        return {"schema_version": 3, "preflight_passed": preflight_passed}

    monkeypatch.setattr(module, "build_plan", build_plan)

    assert module.main() == (0 if passed else 1)
    assert captured["arguments"] is arguments
    assert captured["preflight_passed"] is passed


def test_writer_cannot_clear_while_a_patch_is_still_running() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[list[str]] = []

    def patch(command: list[str]) -> int:
        calls.append(command)
        entered.set()
        assert release.wait(5), "fake patch did not receive release"
        return 0

    writer = AnnotationWriter(["kubectl"], "node-a", runner=patch)
    writer.start()
    try:
        assert entered.wait(5), "fake patch did not start"
        writer.stop(join_timeout=0)
        assert writer.report()["stopped"] is False
        with pytest.raises(RuntimeError, match="running"):
            writer.clear()
        assert len(calls) == 1
    finally:
        release.set()
        writer.stop()
    assert writer.report()["stopped"] is True


def test_writer_records_transport_failure_instead_of_silently_dying() -> None:
    def broken(_command: list[str]) -> int:
        raise TimeoutError("mock timeout")

    writer = AnnotationWriter(["kubectl"], "node-a", runner=broken)
    writer.run_loop()

    report = writer.report()
    assert report["stopped"] is True
    assert report["running"] is False
    assert report["error"] == "TimeoutError: mock timeout"


def test_lost_holder_race_never_schedules_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "holder.json"
    rows = [
        {
            "command_id": "quiesce",
            "operation": "QUIESCE_GPU_SERVICES",
            "state": "SUCCEEDED",
            "completed_at": "2026-09-01T00:00:01+00:00",
        },
        {
            "command_id": "verify",
            "operation": "VERIFY_NO_GPU_CLIENTS",
            "state": "SUCCEEDED",
            "completed_at": "2026-09-01T00:00:02+00:00",
        },
    ]
    holder_probe.write_state(
        path,
        {
            "after_ledger_op": "QUIESCE_GPU_SERVICES",
            "device": "/dev/nvidia0",
            "max_hold_seconds": 60,
            "baseline_command_ids": [],
            "verify_baseline_ids": [],
            "armed_at": "2026-09-01T00:00:00+00:00",
            "injections": [
                {
                    "phase": "escalate",
                    "maintenance_window_end": "2099-01-01T00:00:00+00:00",
                }
            ],
        },
    )
    scheduled: list[str] = []
    monkeypatch.setattr(holder_probe, "state_path", lambda _run: path)
    monkeypatch.setattr(holder_probe, "ledger_rows", lambda: rows)
    monkeypatch.setattr(holder_probe, "run", lambda *_args, **_kwargs: None)

    def injection(_run: str, _item: dict[str, Any]) -> list[str]:
        scheduled.append("escalate")
        return ["mock-injection"]

    monkeypatch.setattr(holder_probe, "injection_command", injection)

    holder_probe.watch_ledger(argparse.Namespace(run_id="review"))

    assert scheduled == []
    assert holder_probe.read_state(path)["arm_race_lost"] is True
    # A pre-authorized timer that fires afterwards refuses for the same reason
    # and records it, instead of writing the escalation onto a committed reset.
    holder_probe.update_state(
        path,
        {
            "run_id": "review",
            "pre_authorization": {"kind": "conditional-barrier-pre-authorization"},
        },
    )
    monkeypatch.setattr(holder_probe, "holder_active", lambda _run: True)
    with pytest.raises(holder_probe.ProbeError, match="before the holder started"):
        holder_probe.fire_injection(
            argparse.Namespace(run_id="review", phase="escalate")
        )
    assert scheduled == []
    refusal = holder_probe.read_state(path)["injection_refusals"]["escalate"]
    assert "before the holder started" in refusal["reason"], refusal


def test_holder_is_bound_to_the_explicit_faulted_gpu() -> None:
    inventory = [
        {"index": 0, "pci_bdf": "0000:01:00.0"},
        {"index": 1, "pci_bdf": "0000:02:00.0"},
    ]
    assert (
        preemption_case.holder_device("", inventory, pci_bdf="0000:02:00.0")
        == "/dev/nvidia1"
    )
    with pytest.raises(RegionalFixtureError, match="does not match"):
        preemption_case.holder_device("/dev/nvidia0", inventory, pci_bdf="0000:02:00.0")
    with pytest.raises(RegionalFixtureError, match="inventory"):
        preemption_case.target_bdf("0000:03:00.0", inventory)


@pytest.mark.parametrize("healthy", [True, False])
def test_retired_diagnostic_never_writes_a_formal_pass(
    healthy: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retired_case, "node_inventory", lambda: {"count": 1})

    def command(argv: list[str], **_kwargs: Any) -> str:
        if "pytest" in argv:
            return "passed"
        if "get" in argv:
            return "executor-a executor-b"
        if "exec" in argv:
            return json.dumps({"allow_replace": "false", "allow_reboot": "true"})
        if "describe-cluster" in argv:
            return json.dumps({"ClusterArn": "arn:example:cluster"})
        if "simulate-principal-policy" in argv:
            if not healthy:
                raise RuntimeError("mock unavailable")
            return json.dumps(
                {
                    "EvaluationResults": [
                        {
                            "EvalActionName": "sagemaker:BatchReplaceClusterNodes",
                            "EvalDecision": "implicitDeny",
                        },
                        {
                            "EvalActionName": "sagemaker:BatchRebootClusterNodes",
                            "EvalDecision": "allowed",
                        },
                    ]
                }
            )
        if "lookup-events" in argv:
            return json.dumps({"Events": []})
        raise AssertionError(f"unexpected diagnostic request: {argv[0]}")

    monkeypatch.setattr(retired_case, "command", command)

    exit_code = retired_case.run_case(tmp_path, 1)

    report = json.loads(
        (
            tmp_path / "cases" / retired_case.CASE_ID / f"{retired_case.CASE_ID}.json"
        ).read_text()
    )
    assert report["verdict"] == "SUPERSEDED", report
    assert report["diagnostic_verdict"] == ("PASS" if healthy else "FAIL")
    assert report["formal_sequence_satisfied"] is False
    assert exit_code == (0 if healthy else 1)


def test_agent_disable_refuses_without_a_reboot_surviving_safeguard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[Any] = []
    monkeypatch.setattr(
        branch_probe, "run", lambda *args, **kwargs: commands.append((args, kwargs))
    )

    with pytest.raises(branch_probe.ProbeError, match="recovery safeguard"):
        branch_probe.disable_agent_restart(argparse.Namespace(run_id="review"))
    assert commands == []


def test_scratch_migration_failure_removes_only_the_owned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gpu_fault.node_agent.ledger as ledger

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("mock migration failure")

    monkeypatch.setattr(ledger, "NodeActionLedger", broken)

    with pytest.raises(RuntimeError, match="mock migration failure"):
        ledger_probe.migration_drill_report(tmp_path / "scratch.db")
    assert list(tmp_path.iterdir()) == []


def test_scratch_migration_does_not_overwrite_existing_data(tmp_path: Path) -> None:
    path = tmp_path / "existing.db"
    path.write_text("previous evidence", encoding="utf-8")

    with pytest.raises(ledger_probe.ProbeError, match="overwrite"):
        ledger_probe.migration_drill_report(path)
    assert path.read_text(encoding="utf-8") == "previous evidence"


@pytest.mark.parametrize("stage", ["health", "migration", "restart"])
def test_agent_restart_case_stops_after_any_failed_prerequisite(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional import run_destr019_agent_restart_ledger as case

    calls: list[str] = []
    settings = SimpleNamespace(node="node-a")
    run = SimpleNamespace(case_dir=tmp_path)
    monkeypatch.setattr(case, "_prepare_live_run", lambda *_args: run)
    monkeypatch.setattr(case, "_baseline", lambda _run: {})
    monkeypatch.setattr(
        case.verdicts,
        "health_errors",
        lambda *_args, **_kwargs: ["unhealthy"] if stage == "health" else [],
    )

    def step(label: str) -> list[str]:
        calls.append(label)
        return [f"{label} failed"] if label == stage else []

    monkeypatch.setattr(case, "_migration_drill", lambda _run: step("migration"))
    monkeypatch.setattr(case, "_restart", lambda _run: step("restart"))
    monkeypatch.setattr(
        case, "_inject_and_observe", lambda _run: calls.append("injection")
    )
    monkeypatch.setattr(case, "_cleanup", lambda _run: {"errors": []})
    end = datetime.now(timezone.utc) + timedelta(hours=1)

    assert case.execute_case(settings, tmp_path, 1, end) == 1
    assert "injection" not in calls
    report = json.loads((tmp_path / f"{case.CASE_ID}.json").read_text())
    assert report["verdict"] == "FAIL"


def test_writer_calls_the_public_supervised_command_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from scripts.e2e.regional import regional_commands
    from scripts.e2e.regional.probes import destr021_annotation_writer as module

    calls: list[dict[str, Any]] = []

    def command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(regional_commands, "run_command", command)

    assert module.subprocess_runner(["kubectl", "mock-only"]) == 0
    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == 60


@pytest.mark.parametrize(
    "scenario", ["active-gpu-pod", "kubernetes-not-ready", "agent-unavailable"]
)
def test_expiring_shortage_refuses_without_independent_cancellation(
    scenario: str,
) -> None:
    from scripts.e2e.regional import run_destr008_warm_spare_shortage as case

    assert case.scenario_admission_errors([scenario]), scenario
    assert (
        case.scenario_admission_errors(
            ["no-spare", "topology-mismatch", "reserved-by-other"]
        )
        == []
    )
