"""run_env_live's entry point, plan fallbacks and execute-time failures.

``test_env_live_runner.py``, ``test_env_live_interruption.py`` and
``test_env_live_scope.py`` drive the cases through the scripted world; here the
runner is held to the paths those leave out: the script started by path from a
bare interpreter, the ``--case`` family check and its per-case token, the plan
fallbacks of a GPU without the T.Limit scale, plan and observation rules that
cannot be built, and an execute that loses a step, a pause waiter, its cleanup
or an operator abort -- each recorded in the case evidence or raised as the
runner promises, the cleanup always taken.
"""

from __future__ import annotations

import json
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import env_live_fixture as fixture
from scripts.e2e.regional import run_env_live as runner
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureAbort
from tests.regional._env_live_world import (
    GPU,
    H200_LIMITS,
    NODE,
    World,
    WorldController,
    WorldControlTask,
    WorldRegionalSettings,
    h200_preflight,
    install_world,
    review_chain,
)

#: A GPU without the T.Limit sensor: absolute slowdown/shutdown limits only.
ABSOLUTE_LIMITS = {
    "gpu_slowdown_temperature_c": 87.0,
    "gpu_shutdown_temperature_c": 92.0,
    "gpu_max_operating_temperature_c": 85.0,
    "memory_max_operating_temperature_c": 95.0,
}
CONTROL = "ip-10-0-32-162"
FIELD = "gpu_tlimit_c"


def _settings(case_id: str, **overrides: Any) -> fixture.EnvLiveSettings:
    values: dict[str, Any] = {
        "regional": WorldRegionalSettings(),
        "case_id": case_id,
        "confirmation": runner.confirmation_for(case_id),
        "node": NODE,
        "host_probe_image": "public.ecr.aws/x@sha256:" + "0" * 64,
        "gpu_uuid": GPU,
        "predecessor_path": None,
        "predecessor_id": None,
        "site_file": None,
        "isolate_confirm": runner.isolate_token(case_id),
        "stop_confirm": runner.stop_token(case_id),
    }
    values.update(overrides)
    return fixture.EnvLiveSettings(**values)


def _absolute_preflight() -> dict[str, Any]:
    target = {
        "gpu_uuid": GPU,
        "gpu_index": "0",
        "product": "NVIDIA A100-SXM4-80GB",
        "pci_bdf": "10:00",
        "limits": dict(ABSOLUTE_LIMITS),
    }
    return h200_preflight(devices=[target], target_gpu=target)


def _window() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _case(run_dir: Path, case_id: str) -> dict[str, Any]:
    path = run_dir / "cases" / case_id / f"{case_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    value = World(start=datetime.now(timezone.utc))
    install_world(monkeypatch, value, preflight=h200_preflight())
    return value


# --------------------------------------------------------------------------- #
# The entry point, the case family and the read-only preflight
# --------------------------------------------------------------------------- #
def test_the_runner_answers_help_when_started_by_path_from_a_bare_interpreter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(runner.ROOT)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SITE_PROFILE", raising=False)
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != root])
    monkeypatch.setattr(sys, "argv", [runner.__file__, "--help"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(runner.__file__, run_name="__main__")

    assert stopped.value.code == 0
    assert sys.path[0] == root, "the script puts its own checkout first on sys.path"
    out = capsys.readouterr().out
    assert "Run one guarded GF-ENV-LIVE acceptance case." in out, out
    for flag in ("--case", "--confirm-isolate", "--node-b"):
        assert flag in out, f"{flag} missing from --help"


def test_configure_accepts_only_the_env_live_family_with_its_own_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SITE_PROFILE", raising=False)
    parse = runner.parser().parse_args
    with pytest.raises(RegionalFixtureError, match=r"001\.\.012, not GF-ENV-LIVE-013"):
        runner.configure(
            parse(["--run-dir", str(tmp_path), "--case", "GF-ENV-LIVE-013"])
        )
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        fixture,
        "configure",
        lambda arguments, confirmation: seen.append((arguments.case, confirmation))
        or "settings",
    )
    arguments = parse(["--run-dir", str(tmp_path), "--case", "GF-ENV-LIVE-007"])
    assert runner.configure(arguments) == "settings"
    assert seen == [("GF-ENV-LIVE-007", "ENVLIVE007_EXECUTE")], (
        "the execute token is the case's own, not the generic placeholder"
    )


def test_the_not_run_case_writes_its_plan_from_a_copy_of_the_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors = ["GF-ENV-LIVE-012 needs a vendor lab"]
    shared = {"errors": errors, "node": {"name": NODE}}
    monkeypatch.setattr(fixture, "read_only_preflight", lambda *_args: shared)

    not_run = runner.read_only_preflight(_settings("GF-ENV-LIVE-012"), tmp_path)
    assert not_run == shared
    not_run["errors"].append("added by the plan writer")
    assert errors == ["GF-ENV-LIVE-012 needs a vendor lab"], (
        "the NOT_RUN plan must not write into the fixture's own error list"
    )
    assert runner.read_only_preflight(_settings("GF-ENV-LIVE-003"), tmp_path) is shared


# --------------------------------------------------------------------------- #
# Replay plans and observation rules
# --------------------------------------------------------------------------- #
def test_a_gpu_without_t_limit_runs_005_on_the_next_scale_and_009_not_at_all() -> None:
    preflight = _absolute_preflight()
    transient = runner.replay_plans_for_case("GF-ENV-LIVE-005", preflight)
    assert sorted(transient["plans"]) == [
        "transient/once_then_clear",
        "transient/twice_then_clear",
        "transient/warning_then_severe",
    ]
    assert transient["not_run"] == {
        "core_tlimit": "device publishes no plausible T.Limit reading (field 153)"
    }
    severe = transient["plans"]["transient/once_then_clear"]["steps"][0]["values"]
    assert list(severe) == ["gpu_temperature_c"], (
        "the absolute core scale is the fallback, not the T.Limit field"
    )
    interruption = runner.replay_plans_for_case("GF-ENV-LIVE-009", preflight)
    assert interruption == {
        "plans": {},
        "not_run": {
            "core_tlimit": "device publishes no plausible T.Limit reading (field 153)"
        },
    }


def test_a_plan_the_case_rule_cannot_build_is_reported_not_raised() -> None:
    one_sample = h200_preflight(confirmation_samples=1)
    result = runner.replay_plans_for_case("GF-ENV-LIVE-006", one_sample)
    assert result["plans"] == {} and result["not_run"] == {}, result
    assert result["error"].startswith(
        "a thermal duration plan needs at least two confirmation samples"
    ), result


def test_an_observation_plan_without_its_rule_is_named_in_the_details() -> None:
    observation = runner.observation_plan(
        "GF-ENV-LIVE-007",
        h200_preflight(),
        {"unreviewed_brake": {"steps": [{"reads": 3, "values": {}}]}},
    )
    assert observation == {
        "plan": "unreviewed_brake",
        "error": "no observation rule for GF-ENV-LIVE-007 plan unreviewed_brake",
    }


# --------------------------------------------------------------------------- #
# Execute: what a lost step, waiter or cleanup leaves in the evidence
# --------------------------------------------------------------------------- #
class NoReadCountController(WorldController):
    """An arm receipt whose /status carries no read count."""

    def arm(self, **options: Any) -> dict[str, Any]:
        receipt = super().arm(**options)
        receipt["status"] = {
            key: value for key, value in receipt["status"].items() if key != "reads"
        }
        return receipt


def test_009_without_the_arms_read_count_takes_no_pause_and_cleans_up(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixture, "EnvReplayController", NoReadCountController)
    code = runner.execute_case(_settings("GF-ENV-LIVE-009"), tmp_path, 1, _window())
    result = _case(tmp_path, "GF-ENV-LIVE-009")

    assert code == 1 and result["verdict"] == "INCONCLUSIVE", result["stages"]
    assert result["inconclusive"] == [
        "GF-ENV-LIVE-009: the arm receipt carried no proxy read count; the pause "
        "point is unknown"
    ]
    calls = WorldController.instances[0].calls
    assert "pause_collector" not in calls and calls[-1] == "cleanup", calls


def _severe(world: World, reads: int, finding_id: str, count: int) -> None:
    def action(w: World) -> None:
        observed = w.tick_time(reads)
        w.record_batch(observed, {FIELD: -3.0, "gpu_temperature_c": 30.0})
        w.record_finding(
            observed,
            finding_id,
            name=FIELD,
            value=-3.0,
            severity="WARNING",
            count=count,
            source="NVIDIA_DEVICE_TLIMIT",
        )

    world.at_read(reads, action)


def test_a_hold_waiter_that_never_starts_is_recorded_and_the_resume_goes_on(
    tmp_path: Path, world: World
) -> None:
    world.hold_waiter_dead = True
    _severe(world, 3, "f-warn", 1)
    _severe(world, 4, "f-warn", 2)
    code = runner.execute_case(_settings("GF-ENV-LIVE-009"), tmp_path, 1, _window())
    result = _case(tmp_path, "GF-ENV-LIVE-009")

    hold = result["receipts"]["observation"]["pause"]["hold"]
    assert hold["schedule_error"] == (
        "pause-collector[hold]: PauseNotTaken: the recount hold waiter is not "
        "running (inactive)"
    )
    assert hold["paused_at"] is None, hold
    calls = WorldController.instances[0].calls
    assert calls.index("pause_collector:hold") < calls.index("resume_collector")
    assert "resume_collector:hold" not in calls, calls
    assert code == 1 and result["verdict"] == "INCONCLUSIVE", result["stages"]
    assert not result["stages"]["steps"], "a refused hold is no failed step"


class DeployRefusedController(WorldController):
    def deploy_proxy(self, plan_path: Any, gpu_uuid: str, **options: Any) -> Any:
        self.calls.append("deploy_proxy")
        raise RuntimeError("the probe Pod could not be scheduled")

    def cleanup(self) -> dict[str, Any]:
        self.calls.append("cleanup")
        raise TimeoutError("the probe did not answer the cleanup")


def test_a_failed_step_and_a_failed_cleanup_are_both_in_the_evidence(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixture, "EnvReplayController", DeployRefusedController)
    code = runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())
    result = _case(tmp_path, "GF-ENV-LIVE-004")

    assert code == 1 and result["verdict"] == "FAIL"
    assert result["error"] == "RuntimeError: the probe Pod could not be scheduled"
    assert result["stages"]["execution"] == [result["error"]]
    assert result["receipts"]["cleanup_error"] == (
        "TimeoutError: the probe did not answer the cleanup"
    )
    assert WorldController.instances[0].calls[-1] == "cleanup"


class AbortedArmController(WorldController):
    def arm(self, **options: Any) -> dict[str, Any]:
        self.calls.append("arm")
        raise RegionalFixtureAbort(15)


def test_an_operator_abort_cleans_up_and_propagates_without_a_verdict(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixture, "EnvReplayController", AbortedArmController)
    with pytest.raises(RegionalFixtureAbort, match="received signal 15"):
        runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())

    calls = WorldController.instances[0].calls
    assert calls[-2:] == ["arm", "cleanup"], calls
    evidence = tmp_path / "cases" / "GF-ENV-LIVE-004" / "GF-ENV-LIVE-004.json"
    assert not evidence.exists(), "an aborted run publishes no verdict"


class StillStoppedController(WorldController):
    def cleanup(self) -> dict[str, Any]:
        receipt = super().cleanup()
        receipt["resume_error"] = "RuntimeError: systemctl kill -s CONT refused"
        receipt["resume_hold"] = {"stopped": True}
        return receipt


def test_a_collector_left_stopped_by_the_cleanup_fails_the_case(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_chain(world)
    monkeypatch.setattr(fixture, "EnvReplayController", StillStoppedController)
    code = runner.execute_case(_settings("GF-ENV-LIVE-004"), tmp_path, 1, _window())
    result = _case(tmp_path, "GF-ENV-LIVE-004")

    assert code == 1 and result["verdict"] == "FAIL"
    assert result["stages"]["cleanup"] == [
        "GF-ENV-LIVE-004: the paused collector was not resumed (resume): "
        "RuntimeError: systemctl kill -s CONT refused",
        "GF-ENV-LIVE-004: the metrics collector is still stopped after cleanup "
        "(resume_hold)",
    ]


def test_a_control_task_that_cannot_be_removed_fails_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices = [
        {
            "gpu_uuid": uuid,
            "gpu_index": index,
            "product": "NVIDIA H200",
            "pci_bdf": bdf,
            "limits": dict(H200_LIMITS),
        }
        for uuid, index, bdf in ((GPU, "0", "59:00"), ("GPU-bbbb", "1", "5a:00"))
    ]
    preflight = h200_preflight(
        devices=devices,
        target_gpu=devices[0],
        control_node={"name": CONTROL, "errors": []},
    )
    world = World(start=datetime.now(timezone.utc))
    install_world(monkeypatch, world, preflight=preflight)
    review_chain(world)

    def refuse(self: WorldControlTask) -> dict[str, Any]:
        self.calls.append("cleanup")
        raise RuntimeError("the control Job is stuck terminating")

    monkeypatch.setattr(WorldControlTask, "cleanup", refuse)
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n", encoding="utf-8")
    settings = _settings("GF-ENV-LIVE-010", site_file=site, control_node=CONTROL)
    code = runner.execute_case(settings, tmp_path, 1, _window())
    result = _case(tmp_path, "GF-ENV-LIVE-010")

    assert result["receipts"]["control_task_cleanup_error"] == (
        "RuntimeError: the control Job is stuck terminating"
    )
    assert result["stages"]["cleanup"] == [
        "GF-ENV-LIVE-010: control_task cleanup: RuntimeError: the control Job is "
        "stuck terminating"
    ]
    assert code == 1 and result["verdict"] == "FAIL"
    assert WorldControlTask.instances[0].calls[-1] == "cleanup"


def test_a_baseline_without_the_before_read_skips_that_check_only(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def records(regional: Any, node: str, since: datetime, names: Any) -> Any:
        if world.repoint_at is None:
            raise ConnectionError("the API Pod refused the store read")
        return world.gpu_state()

    monkeypatch.setattr(fixture, "gpu_records", records)
    runner.execute_case(_settings("GF-ENV-LIVE-001"), tmp_path, 1, _window())
    result = _case(tmp_path, "GF-ENV-LIVE-001")

    assert result["receipts"]["before_error"] == (
        "ConnectionError: the API Pod refused the store read"
    )
    assert "observation" in result["receipts"], "the baseline was still observed"
    assert "arm" not in WorldController.instances[0].calls, "001 never arms"
