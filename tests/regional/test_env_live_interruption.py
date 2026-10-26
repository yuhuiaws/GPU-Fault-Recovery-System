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
    # The recount hold: two polling rounds (the 15 s sleep plus the five remote
    # reads a 009 poll makes, 45 s budgeted) at the default cadence.
    assert expectation.recount_hold_seconds == 2 * (15 + watch.POLL_READ_BUDGET_SECONDS)
    assert expectation.recount_hold_seconds == 120
    assert expectation.terms["recount_hold"] == expectation.recount_hold_seconds
    assert expectation.evidence()["recount_hold_seconds"] == 120
    assert expectation.require_clear and expectation.boot_id == "boot-1"
    assert any("source switch" in note for note in expectation.notes), expectation.notes
    assert any("cross-boot" in note for note in expectation.notes), expectation.notes
    details = runner.plan_details(_settings(), preflight)
    assert "verdict_if_blocked" not in details
    assert details["interruption"]["pause_after_reads"] == 4
    assert details["interruption"]["pause_deadman_seconds"] == 75 + 120
    assert (
        details["interruption"]["recount_hold_label"] == interruption.RECOUNT_HOLD_LABEL
    )
    assert details["interruption"]["recount_hold_after_reads"] == 5
    assert details["interruption"]["recount_hold_seconds"] == 120
    assert details["interruption"]["recount_hold_deadman_seconds"] == 120 + 120
    assert "recount hold" in details["interruption"]["mechanism"]
    # The proxy window and the collector deadman carry the hold.
    assert details["proxy_mechanism"]["max_window_seconds"] == 600 + 120
    assert details["rollback"]["collector_repoint_deadman_seconds"] == 600 + 120
    assert details["observation"]["terms"]["recount_hold"] == 120
    assert details["risk"] == "live-collector-maintenance"
    # Other cases hold nothing and keep the plain window.
    other = runner.plan_details(_settings(case_id="GF-ENV-LIVE-003"), preflight)
    assert other["proxy_mechanism"]["max_window_seconds"] == 600
    assert "recount_hold" not in other["observation"]["terms"]


def test_execute_009_schedules_the_pause_right_after_the_arm(
    tmp_path: Path, world: World
) -> None:
    product_recounts(world)
    runner.execute_case(_settings(), tmp_path, 1, _window())
    calls = WorldController.instances[0].calls
    assert calls.index("arm") + 1 == calls.index("pause_collector")
    assert world.pause_target == 1 + 4, (
        "absolute proxy reads: the proven repoint's pass-through read (1) + pause read 4"
    )
    receipt = _read_case(tmp_path)["receipts"]["pause_schedule"]
    assert receipt["waiter"]["ActiveState"] == "active"
    # The recount hold is scheduled one read past the gap pause, right before
    # the resume, while the unit is still stopped (no read can slip in).
    assert calls.index("pause_collector:hold") + 1 == calls.index("resume_collector")
    assert world.hold is not None and world.hold.target == 1 + 5
    assert world.hold.scheduled_at is not None and world.paused_at is not None
    assert world.paused_at < world.hold.scheduled_at <= world.resumed_at


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
    # The recount hold landed after the post-gap read, was read once and was
    # resumed the moment the regraded finding had been read -- long before its
    # bound -- then the clear, the restart.
    hold = observation["pause"]["hold"]
    assert hold["scheduled"] and hold["schedule_error"] is None
    assert hold["target_reads"] == 1 + 5 and hold["missed"] is False
    assert hold["paused_at"] and hold["resumed_at"] and hold["regrade_seen_at"]
    assert hold["reads_while_held"] >= 1 and hold["reads_with_post_sample"] >= 1
    assert observation["recount_hold_seconds"] == 120
    assert world.hold is not None and world.hold.paused_at and world.hold.resumed_at
    held = (world.hold.resumed_at - world.hold.paused_at).total_seconds()
    assert 0 < held < observation["recount_hold_seconds"]
    assert calls.index("resume_collector") < calls.index("resume_collector:hold")
    assert calls.index("resume_collector:hold") < calls.index("restart_collector")
    assert (
        observation["first_seen"]["hold_resumed"]
        <= observation["first_seen"]["restarted"]
    )
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
    monkeypatch.setattr(World, "resume_collector", lambda self, **kw: {"stopped": True})
    runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert result["verdict"] == "INCONCLUSIVE", result["stages"]


# --------------------------------------------------------------------------- #
# The recount hold (live 2026-10-26: the regraded 1/3 state lived one collector
# interval, the polls were 46-56 s apart, INCONCLUSIVE "no finding graded it")
# --------------------------------------------------------------------------- #
def test_execute_009_holds_the_recount_for_polls_slower_than_one_interval(
    tmp_path: Path, world: World
) -> None:
    """Live cadence: five 8 s reads a poll (a 55 s round) against a 15 s state."""

    world.read_latency_seconds = 8.0
    product_recounts(world)
    code = runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 0 and result["verdict"] == "PASS", (
        result["stages"],
        result["inconclusive"],
    )
    observation = result["receipts"]["observation"]
    assert observation["post_gap_findings"][0]["confirmation_count"] == 1
    assert observation["gap_seconds"] > CONFIRMATION_MAX_GAP_SECONDS
    hold = observation["pause"]["hold"]
    assert hold["paused_at"] and hold["resumed_at"] and hold["missed"] is False
    assert hold["reads_while_held"] >= 1 and hold["regrade_seen_at"]
    # Nothing was accepted while the hold stood: the clear is the next read
    # after the resume, and the plan was still consumed inside the window.
    assert observation["replay"]["consumed_at"] and observation["cleared_at"]
    assert observation["cleared_at"] > hold["resumed_at"]
    assert not observation["window"]["capped"]


def test_execute_009_fails_when_the_held_state_never_grades_the_post_gap_sample(
    tmp_path: Path, world: World
) -> None:
    """The hold makes a missing grading a product verdict, not a missed read."""

    severe_read(world, 3, "f-warn", 1)
    severe_read(world, 4, "f-warn", 2)
    # The post-gap severe sample is accepted (its batch is on record) but the
    # field's finding state keeps the pre-gap finding: nothing graded it.
    world.at_read(
        5,
        lambda w: w.record_batch(
            w.tick_time(5), {FIELD: SEVERE, "gpu_temperature_c": 30.0}
        ),
    )
    healthy_read(world, 6, "f-warn")
    healthy_read(world, 7)
    healthy_read(world, 8)
    fresh_after_restart(world)
    code = runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    assert code == 1 and result["verdict"] == "FAIL", (
        result["stages"],
        result["inconclusive"],
    )
    assert any(
        "was not graded into a finding while the collector was held" in item
        for item in result["stages"]["recount"]
    ), result["stages"]
    hold = result["receipts"]["observation"]["pause"]["hold"]
    assert hold["paused_at"] and hold["reads_with_post_sample"] >= 2
    assert result["receipts"]["observation"]["post_gap_findings"] == []


def test_execute_009_records_a_refused_hold_and_judges_the_records(
    tmp_path: Path, world: World
) -> None:
    """A hold the node refuses is evidence, not a verdict: the resume goes ahead
    and the records decide.

    Without the hold the world reproduces the live miss even with instant
    reads: the resume is taken at a poll, so the post-gap read and the clearing
    read both fall between that poll and the next one, and the regraded 1/3
    state is never read -- INCONCLUSIVE, naming the refused hold.
    """

    world.hold_schedule_fails = True
    product_recounts(world)
    code = runner.execute_case(_settings(), tmp_path, 1, _window())
    result = _read_case(tmp_path)
    hold = result["receipts"]["observation"]["pause"]["hold"]
    assert hold["scheduled"] is False and hold["paused_at"] is None
    assert "refused by the node" in hold["schedule_error"]
    calls = WorldController.instances[0].calls
    assert "pause_collector:hold" in calls and "resume_collector" in calls
    assert "resume_collector:hold" not in calls
    assert calls.index("pause_collector:hold") < calls.index("resume_collector")
    assert code == 1 and result["verdict"] == "INCONCLUSIVE", (
        result["stages"],
        result["inconclusive"],
    )
    text = "\n".join(result["inconclusive"])
    assert "no finding graded it in any store read" in text
    assert "the recount hold could not be scheduled" in text
    assert "refused by the node" in text
    assert not any(result["stages"][key] for key in ("steps", "recount")), result[
        "stages"
    ]
    # The plan was still consumed and the collector restarted: the refused
    # hold cost the proof, not the maintenance steps.
    observation = result["receipts"]["observation"]
    assert observation["replay"]["consumed_at"] and observation["pause"]["restart_at"]


def _live17_finding(finding_id: str, observed: datetime, count: int) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "canonical_name": FIELD,
        "severity": "WARNING",
        "value": SEVERE,
        "gpu_uuid": GPU,
        "observed_at": observed.isoformat(),
        "automatic_action": "RUN_DIAGNOSTICS",
        "confirmation_count": count,
        "threshold_source": "NVIDIA_DEVICE_TLIMIT",
        "policy_source": "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
        "affected_gpu_uuids": [],
    }


def _live17_batch(
    batch_id: str, observed: datetime, value: float, *reasons: str
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "observed_at": observed.isoformat(),
        "edge_filter_reasons": list(reasons),
        "collection_errors": [],
        "gpus": [GPU],
        "samples": [
            {
                "gpu": GPU,
                "name": FIELD,
                "value": value,
                "labels": {"source_boot_id": "boot-1"},
            }
        ],
    }


def test_judge_names_the_unheld_regrade_the_live17_way() -> None:
    """The live 2026-10-26 poll sequence, replayed on the judge without a hold.

    The post-gap sample (-3, 13:32:57Z) was graded 1/3 into the current finding
    state only; the clearing read (13:33:12Z) replaced it; the history kept the
    first 1/3 finding alone. The poll at 13:33:16Z saw neither the 2/3 nor the
    regraded 1/3 as active, and no durable record of the regrade exists: the
    verdict is INCONCLUSIVE and says why the state was not readable.
    """

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
    utc = timezone.utc
    day = datetime(2026, 10, 26, tzinfo=utc)

    def at(hour: int, minute: int, second: float) -> datetime:
        return day + timedelta(hours=hour, minutes=minute, seconds=second)

    arm = at(13, 29, 43.5)
    paused = at(13, 30, 46.339)
    resumed = at(13, 32, 57.293)
    receipts: list[str] = []

    def resume() -> dict[str, Any]:
        receipts.append("resume")
        return {"pause": {"resumed_at": resumed.isoformat()}, "stopped": False}

    judge = interruption.InterruptionJudge(
        expectation,
        steps=interruption.InterruptionSteps(resume=resume, restart=dict),
        interval_seconds=15,
    )
    progress = watch.ReplayProgress(
        planned_reads=8, armed=True, proxy_endpoint="http://127.0.0.1:9411/metrics"
    )
    progress.begin({"status": {"reads": 2, "armed": True, "next_values": {}}}, arm)
    first = _live17_finding("f-1", at(13, 30, 30.918), 1)
    second = _live17_finding("f-2", at(13, 30, 45.986), 2)
    batches = [
        _live17_batch("b-3", at(13, 30, 30.918), SEVERE, "environmental-observation"),
        _live17_batch("b-4", at(13, 30, 45.986), SEVERE, "environmental-observation"),
    ]
    pause = {
        "target_reads": 6,
        "paused_at": paused.isoformat(),
        "reads_at_pause": 6,
        "missed": False,
    }

    def poll(
        moment: datetime,
        *,
        reads: int,
        active: list[dict[str, Any]],
        history: list[dict[str, Any]],
        batches: list[dict[str, Any]],
        pause: dict[str, Any],
        stopped: bool,
    ) -> None:
        progress.update(
            {"status": {"reads": reads, "armed": True, "next_values": {}}}, moment
        )
        judge.observe(
            watch.Snapshot(
                at=moment,
                gpu_state={
                    "active_findings": active,
                    "findings": history,
                    "batches": batches,
                    "latest": [],
                    "collector_statuses": [],
                },
                activity={"incidents": [], "workflows": [], "notifications": []},
                node={"unschedulable": False, "taints": []},
                extra={
                    "pause": pause,
                    "stopped": stopped,
                    "waiter": {"ActiveState": "inactive"},
                    "pauses": {"": pause},
                },
            ),
            progress,
        )

    poll(
        at(13, 30, 2.3),
        reads=3,
        active=[],
        history=[],
        batches=[],
        pause={"target_reads": 6},
        stopped=False,
    )
    for moment in (at(13, 30, 48.6), at(13, 31, 34.7), at(13, 32, 20.7)):
        poll(
            moment,
            reads=6,
            active=[second],
            history=[first],
            batches=batches,
            pause=pause,
            stopped=True,
        )
        assert judge.decide(progress, moment, final=False) is None
    assert receipts == ["resume"], "resumed at the first poll past the 75 s pause"
    assert judge.pause.resumed_at == resumed, "the node's SIGCONT stamp, not the poll's"
    after = [
        *batches,
        _live17_batch(
            "b-5",
            at(13, 32, 57.267),
            SEVERE,
            "health-summary",
            "environmental-observation",
        ),
        _live17_batch(
            "b-6",
            at(13, 33, 12.543),
            60.0,
            "environmental-observation",
            "candidate-recovered",
        ),
    ]
    late = at(13, 33, 16.2)
    poll(
        late,
        reads=8,
        active=[],
        history=[first],
        batches=after,
        pause={**pause, "resumed_at": resumed.isoformat()},
        stopped=False,
    )
    assert round(judge.gap_seconds() or 0) == 131
    assert judge.post_gap_findings() == []
    assert judge.cleared_at() is not None
    assert judge.decide(progress, late, final=False) is None
    # 13:34:02Z: the plan consumed (served 9 of 8), the restart taken; the
    # restarted collector's first batch is not in yet.
    consumed = at(13, 34, 2.4)
    poll(
        consumed,
        reads=11,
        active=[],
        history=[first],
        batches=after,
        pause={**pause, "resumed_at": resumed.isoformat()},
        stopped=False,
    )
    assert progress.consumed and judge.pause.restart_at == consumed
    assert judge.decide(progress, consumed, final=False) is None
    verdict = judge.decide(progress, consumed, final=True)
    assert verdict is not None and verdict.outcome == "INCONCLUSIVE", verdict
    text = "\n".join(verdict.inconclusive)
    assert "no finding graded it in any store read" in text
    assert "no recount hold was scheduled" in text
    assert "current finding state" in text
    assert verdict.evidence["pause"]["hold"]["scheduled"] is False
    assert (
        verdict.evidence["first_post_gap_sample_at"] == at(13, 32, 57.267).isoformat()
    )


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
