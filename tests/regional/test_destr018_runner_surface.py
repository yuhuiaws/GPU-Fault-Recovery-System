"""GF-REGIONAL-DESTR-018 runner surface: command line, probe invocations and
the cleanup that closes the lifetime-escalated incident.

Split from ``test_destr018_lifetime_deadline.py`` (the timing, verdict and
preflight contracts) to keep each file within the review size limit; these
need only the runner module, not the synthetic snapshots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr018_verdicts as verdicts
from scripts.e2e.regional import run_destr018_lifetime_deadline as destr018

ROOT = Path(__file__).resolve().parents[2]
NODE = "node-a"


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr018.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False
    assert plan.lifetime_seconds == verdicts.LIFETIME_SECONDS
    assert plan.execution_timeout_seconds == verdicts.EXECUTION_TIMEOUT_SECONDS
    assert plan.step_timeout_seconds == verdicts.STEP_TIMEOUT_SECONDS
    assert plan.managed_recovery_seconds == verdicts.MANAGED_RECOVERY_SECONDS
    assert plan.step_warning_seconds == verdicts.STEP_WARNING_SECONDS
    assert plan.lease_duration_seconds == verdicts.LEASE_DURATION_SECONDS
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr018.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
        ]
    )
    assert execute.execute is True
    assert execute.confirm == destr018.CONFIRMATION
    assert execute.node == NODE
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = destr018.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text


def test_the_runner_and_the_env_window_helper_are_executable_with_a_shebang() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_destr018_lifetime_deadline.py",
        ROOT / "scripts/e2e/regional/control_plane_env_window.py",
        ROOT / "scripts/e2e/regional/probes/destr018_node_probe.py",
    ):
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{path.name} is {oct(mode)}, not 0o775"
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "#!/usr/bin/env python3"


def test_the_expected_step_sequence_is_the_reset_contract() -> None:
    assert destr018.EXPECTED_STEPS.index(verdicts.WAITING_STEP) == 3
    assert destr018.EXPECTED_STEPS.index(verdicts.RESET_STEP) == 4
    assert destr018.EXPECTED_STEPS.index(verdicts.COMPENSATION_STEP) == 5
    assert isinstance(destr018.parser(), argparse.ArgumentParser) is True


def test_arm_holder_only_sends_flags_the_node_probe_actually_accepts(
    tmp_path: Path,
) -> None:
    """Every holder subcommand the runner invokes must parse against the real
    on-node probe contract. `holder-status` derives its device from persisted
    state and defines only `--run-id`; passing `--device` made argparse reject
    the call ("unrecognized arguments: --device") after arm-holder had already
    succeeded (observed live 2026-09-08, attempt 6)."""
    from types import SimpleNamespace

    from scripts.e2e.regional.probes import destr018_node_probe as probe

    probe_parser = probe.parser()
    calls: list[tuple[str, ...]] = []

    class _FakeHolder:
        settings = SimpleNamespace(run_id="run-abc")
        host_script = "/run/gpu-fault-host-probe-deadbeef01.py"

        def execute(self, *arguments: str, timeout: int = 0) -> dict[str, Any]:
            calls.append(arguments)
            return {"device_clients": [{"pid": "1234", "comm": "holder"}]}

    run = SimpleNamespace(
        holder=_FakeHolder(),
        settings=SimpleNamespace(hold_seconds=900),
        run_id="drill-xyz",
        case_dir=tmp_path,
        holder_armed=False,
    )

    destr018.arm_holder(run, "/dev/nvidia0")

    assert calls, "arm-holder was never invoked"
    subcommands = {call[0] for call in calls}
    assert {"arm-holder", "holder-status"} <= subcommands
    for call in calls:
        # Would raise SystemExit on an unrecognized flag, mirroring the pod.
        probe_parser.parse_args(list(call))


def test_cleanup_closes_the_reset_incident_through_a_validated_restore(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    calls: list[tuple[str, Any]] = []
    states = iter(["ESCALATED", "RECOVERED"])

    class _FakeWarm:
        def __init__(self, regional: Any, hyperpod: str) -> None:
            calls.append(("init", hyperpod))

        def incident_by_id(self, incident_id: str) -> dict[str, Any]:
            return {"incident_id": incident_id, "state": next(states)}

        def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
            calls.append(("idle", incident_id))
            return {}

        def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("restore", kwargs))
            return {"workflow_request_id": "workflow-validated-restore-1"}

        def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
            calls.append(("wait", workflow_id))
            return {"status": "SUCCEEDED"}

    monkeypatch.setattr(destr018, "WarmSpareLiveFixture", _FakeWarm)
    run = SimpleNamespace(
        incident_id="inc-reset",
        regional=object(),
        settings=SimpleNamespace(node="node-a"),
        profile_version="p1",
    )

    result = destr018.close_reset_incident(run)

    assert result["closed"] is True and result["state"] == "RECOVERED", result
    restore = next(kwargs for kind, kwargs in calls if kind == "restore")
    assert restore["incident_id"] == "inc-reset" and restore["node"] == "node-a"
    assert [kind for kind, _ in calls] == ["init", "idle", "restore", "wait"]


def test_cleanup_leaves_an_already_recovered_reset_incident_alone(monkeypatch) -> None:
    from types import SimpleNamespace

    class _Recovered:
        def __init__(self, regional: Any, hyperpod: str) -> None:
            pass

        def incident_by_id(self, incident_id: str) -> dict[str, Any]:
            return {"state": "RECOVERED"}

        def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("a RECOVERED incident must not be restored again")

    monkeypatch.setattr(destr018, "WarmSpareLiveFixture", _Recovered)
    run = SimpleNamespace(
        incident_id="inc-reset",
        regional=object(),
        settings=SimpleNamespace(node="node-a"),
        profile_version="p1",
    )

    assert destr018.close_reset_incident(run) == {"closed": False, "state": "RECOVERED"}
    assert (
        destr018.close_reset_incident(
            SimpleNamespace(
                incident_id="", regional=None, settings=None, profile_version=""
            )
        )["closed"]
        is False
    )
