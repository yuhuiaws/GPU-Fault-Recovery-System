"""GF-ENV-LIVE-009 through run_env_live over the scripted world (no cluster).

The world models the read-synchronous pause the on-node waiter takes (no
collector tick while paused), the judge's resume and restart steps and the
product's records at the collector's own read times; the runner's real
observation loop decides. A passing scenario is supporting logic coverage,
never a LIVE PASS.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.gpu_environment_policy import CONFIRMATION_MAX_GAP_SECONDS
from scripts.e2e.regional import env_live_expectations as expectations
from scripts.e2e.regional import env_live_fixture as fixture
from scripts.e2e.regional import env_live_interruption as interruption
from scripts.e2e.regional import env_live_plans as plans
from scripts.e2e.regional import env_live_watch as watch
from scripts.e2e.regional import run_env_live as runner
from tests.regional._env_live_world import (
    GPU,
    NODE,
    World,
    WorldController,
    WorldRegionalSettings,
    baseline_values,
    h200_preflight,
    install_world,
)

CASE = "GF-ENV-LIVE-009"
FIELD = "gpu_tlimit_c"
SEVERE = -3.0
HEALTHY = 58.0


def _settings(**overrides: Any) -> fixture.EnvLiveSettings:
    values: dict[str, Any] = {
        "regional": WorldRegionalSettings(),
        "case_id": CASE,
        "confirmation": runner.confirmation_for(CASE),
        "node": NODE,
        "host_probe_image": "public.ecr.aws/x@sha256:" + "0" * 64,
        "gpu_uuid": GPU,
        "predecessor_path": None,
        "predecessor_id": None,
        "site_file": None,
    }
    values.update(overrides)
    return fixture.EnvLiveSettings(**values)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    value = World(start=datetime.now(timezone.utc))
    install_world(monkeypatch, value, preflight=h200_preflight())
    return value


def _window() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _read_case(run_dir: Path) -> dict[str, Any]:
    import json

    return json.loads((run_dir / "cases" / CASE / f"{CASE}.json").read_text())


def severe_read(world: World, reads: int, finding_id: str, count: int) -> None:
    """The collector's ``reads``-th read served severe; the product graded it."""

    def action(w: World) -> None:
        observed = w.tick_time(reads)
        w.record_batch(observed, {FIELD: SEVERE, "gpu_temperature_c": 30.0})
        severity = "CRITICAL" if count >= 3 else "WARNING"
        w.record_finding(
            observed,
            finding_id,
            name=FIELD,
            value=SEVERE,
            severity=severity,
            count=count,
            source="NVIDIA_DEVICE_TLIMIT",
        )

    world.at_read(reads, action)


def healthy_read(world: World, reads: int, *clear: str) -> None:
    def action(w: World) -> None:
        w.record_batch(
            w.tick_time(reads),
            {FIELD: HEALTHY, "gpu_temperature_c": 30.0},
            reasons=("candidate-recovered",),
        )
        w.active.difference_update(clear)

    world.at_read(reads, action)


def fresh_after_restart(world: World, offset: float = 30.0) -> None:
    world.at_restart(
        offset,
        lambda w: w.record_batch(
            w.now, baseline_values(), reasons=("initial-baseline",)
        ),
    )


def product_recounts(world: World) -> World:
    """The product doing 009 right: 2/3 before the gap, 1/3 after it, a clear."""

    severe_read(world, 3, "f-warn", 1)
    severe_read(world, 4, "f-warn", 2)
    severe_read(world, 5, "f-after", 1)  # past the gap: counted from 1 again
    healthy_read(world, 6, "f-warn", "f-after")
    healthy_read(world, 7)
    healthy_read(world, 8)
    fresh_after_restart(world)
    return world


# --------------------------------------------------------------------------- #
def test_interruption_plan_pauses_after_the_second_severe_read() -> None:
    preflight = h200_preflight()
    plan = runner.replay_plans_for_case(CASE, preflight)["plans"][
        "core_tlimit/interruption"
    ]
    reads = [step["reads"] for step in plan["steps"]]
    assert reads == [2, 2, 1, 3]
    assert plan["steps"][1]["values"] == {FIELD: SEVERE}
    assert plan["steps"][2]["values"] == {FIELD: SEVERE}
    assert plan["steps"][3]["values"] == {}
    assert plans.pause_read_index(plan) == 4
    from tools.gpu_environment_replay import validate_plan

    validate_plan(plan)
    with pytest.raises(plans.EnvLivePlanError):
        plans.pause_read_index({"steps": [{"reads": 1, "values": {}}]})


def test_interruption_expectation_names_the_pause_and_the_gaps() -> None:
    preflight = h200_preflight()
    plan = runner.replay_plans_for_case(CASE, preflight)["plans"][
        "core_tlimit/interruption"
    ]
    expectation = expectations.expectation_for(
        case_id=CASE,
        plan_name="core_tlimit/interruption",
        plan=plan,
        preflight=preflight,
        cadence=watch.CollectorCadence(),
        node=NODE,
    )
    assert expectation.kind == "interruption"
    assert expectation.pause_after_reads == 4
    assert expectation.pause_seconds == CONFIRMATION_MAX_GAP_SECONDS + 15
    assert expectation.terms["collection_pause"] == expectation.pause_seconds
    assert expectation.require_clear and expectation.boot_id == "boot-1"
    assert any("source switch" in note for note in expectation.notes), expectation.notes
    assert any("cross-boot" in note for note in expectation.notes), expectation.notes
    details = runner.plan_details(_settings(), preflight)
    assert "verdict_if_blocked" not in details
    assert details["interruption"]["pause_after_reads"] == 4
    assert details["interruption"]["pause_deadman_seconds"] == 75 + 120
    assert details["risk"] == "live-collector-maintenance"


def test_execute_009_schedules_the_pause_right_after_the_arm(
    tmp_path: Path, world: World
) -> None:
    product_recounts(world)
    runner.execute_case(_settings(), tmp_path, 1, _window())
    calls = WorldController.instances[0].calls
    assert calls.index("arm") + 1 == calls.index("pause_collector")
    assert world.pause_target == 4, "absolute proxy reads: arm at 0 + pause read 4"
    receipt = _read_case(tmp_path)["receipts"]["pause_schedule"]
    assert receipt["waiter"]["ActiveState"] == "active"


def test_execute_009_passes_when_the_gap_recounts_from_one(
    tmp_path: Path, world: World
) -> None:
    product_recounts(world)
    code = runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 0 and result["verdict"] == "PASS", (
        result["stages"],
        result["inconclusive"],
    )
    observation = result["receipts"]["observation"]
    assert observation["gap_seconds"] > CONFIRMATION_MAX_GAP_SECONDS
    assert observation["pause"]["paused_at"] and observation["pause"]["resumed_at"]
    assert observation["pause"]["restart_at"]
    assert observation["post_gap_findings"][0]["confirmation_count"] == 1
    assert observation["fresh_batch_after_restart"]["edge_filter_reasons"] == [
        "initial-baseline"
    ]
    assert observation["sources"] == ["DCGM_EXPORTER"]
    calls = WorldController.instances[0].calls
    assert calls.index("resume_collector") < calls.index("restart_collector")
    assert calls.index("restart_collector") < calls.index("cleanup")
    # The judge resumed only after the planned pause had elapsed on the node.
    assert world.resumed_at is not None and world.paused_at is not None
    assert (world.resumed_at - world.paused_at).total_seconds() >= 75
    # The source switch and the cross-boot path are recorded as gaps, not PASSed.
    assert any("source switch" in item for item in result["limitations"]), result[
        "limitations"
    ]


def test_execute_009_fails_when_the_product_confirms_across_the_gap(
    tmp_path: Path, world: World
) -> None:
    severe_read(world, 3, "f-warn", 1)
    severe_read(world, 4, "f-warn", 2)
    severe_read(world, 5, "f-crit", 3)  # the gap did not restart the count
    healthy_read(world, 6, "f-warn", "f-crit")
    healthy_read(world, 7)
    healthy_read(world, 8)
    fresh_after_restart(world)
    code = runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 1 and result["verdict"] == "FAIL"
    assert any("CRITICAL" in item for item in result["stages"]["recount"]), result[
        "stages"
    ]


def test_execute_009_fails_when_the_gap_is_treated_as_healthy(
    tmp_path: Path, world: World
) -> None:
    product_recounts(world)

    def inside_gap(w: World) -> None:
        # A reading stamped inside the pause: nothing real could have produced it.
        assert w.paused_at is not None, "fires only once paused"
        w.record_batch(
            w.paused_at + timedelta(seconds=30),
            {FIELD: HEALTHY, "gpu_temperature_c": 30.0},
            reasons=("candidate-recovered",),
        )

    world.at_read(5, inside_gap)
    runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any(
        "inside the collection gap" in item for item in result["stages"]["gap"]
    ), result["stages"]


def test_execute_009_fails_when_the_prepause_finding_vanishes_in_the_gap(
    tmp_path: Path, world: World
) -> None:
    product_recounts(world)

    def clear_while_paused(w: World) -> None:
        w.active.discard("f-warn")

    # Fires on the first poll after the pause (read 4 is the pause read), with
    # no reading accepted: the product treated the silence as a recovery.
    world.at_read(4, lambda w: w.at(0.0, clear_while_paused))
    runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL", result["stages"]
    assert any("treated as healthy" in item for item in result["stages"]["gap"]), (
        result["stages"]
    )


def test_execute_009_isolation_or_a_forbidden_step_fails(
    tmp_path: Path, world: World
) -> None:
    product_recounts(world)
    world.incident(70, "f-warn", action="DRAIN", workflow="wf-x")
    world.workflow(70, "wf-x", "f-warn", operations=("FREEZE_EVIDENCE", "QUARANTINE"))
    runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "FAIL"
    assert any("QUARANTINE" in item for item in result["stages"]["reaction"]), result[
        "stages"
    ]


def test_execute_009_waiter_timeout_is_inconclusive_even_if_the_product_confirms(
    tmp_path: Path, world: World
) -> None:
    world.pause_timed_out = True  # the pause never lands; reads keep flowing
    severe_read(world, 3, "f-warn", 1)
    severe_read(world, 4, "f-warn", 2)
    severe_read(world, 5, "f-crit", 3)  # correct product behaviour, not graded
    healthy_read(world, 6, "f-warn", "f-crit")
    runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "INCONCLUSIVE", result["stages"]
    assert any("did not reach its read" in item for item in result["inconclusive"]), (
        result["inconclusive"]
    )
    assert not any(
        result["stages"].get(key) for key in ("recount", "gap", "reaction")
    ), result["stages"]


def test_execute_009_refused_pause_schedule_is_inconclusive_and_cleans_up(
    tmp_path: Path, world: World
) -> None:
    world.waiter_dead = True  # pause-collector comes back without a running waiter
    code = runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 1 and result["verdict"] == "INCONCLUSIVE", result
    assert any("waiter is not running" in item for item in result["inconclusive"]), (
        result["inconclusive"]
    )
    calls = WorldController.instances[0].calls
    assert calls[-1] == "cleanup" and "pause_status" not in calls


def test_execute_009_without_a_resume_before_the_deadline_is_inconclusive(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    severe_read(world, 3, "f-warn", 1)
    severe_read(world, 4, "f-warn", 2)
    # No post-gap records at all: the plan is never consumed past the pause
    # because the world's resume keeps the unit stopped (a stuck SIGCONT).
    monkeypatch.setattr(World, "resume_collector", lambda self: {"stopped": True})
    runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "INCONCLUSIVE", result["stages"]


def test_interruption_judge_is_never_built_by_judge_for() -> None:
    from scripts.e2e.regional import env_live_judges as judges

    preflight = h200_preflight()
    plan = runner.replay_plans_for_case(CASE, preflight)["plans"][
        "core_tlimit/interruption"
    ]
    expectation = expectations.expectation_for(
        case_id=CASE,
        plan_name="core_tlimit/interruption",
        plan=plan,
        preflight=preflight,
        cadence=watch.CollectorCadence(),
        node=NODE,
    )
    with pytest.raises(ValueError, match="env_live_interruption"):
        judges.judge_for(expectation, repoint_at=datetime.now(timezone.utc))
    judge = interruption.InterruptionJudge(
        expectation,
        steps=interruption.InterruptionSteps(resume=dict, restart=dict),
        interval_seconds=15,
    )
    assert judge.reads_extra and judge.reads_node
