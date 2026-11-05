"""GF-ENV-LIVE-009's judge, polled by hand through every way a run can end.

``test_env_live_interruption.py`` runs the case through ``run_env_live`` over
the scripted world and replays the live 2026-10-26 poll sequence; here the
``InterruptionJudge`` is fed snapshots directly so that each verdict branch is
reached on purpose: the recount hold's notes, a pause or hold record the node
stamped itself, maintenance steps that raise, the reasons a deadline leaves the
case INCONCLUSIVE (in the order the judge asks them), and the FAIL stages a
second producer, an unauthorized recovery or a foreign boot id open.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import env_live_expectations as expectations
from scripts.e2e.regional import env_live_interruption as interruption
from scripts.e2e.regional import env_live_watch as watch
from scripts.e2e.regional import run_env_live as runner
from tests.regional._env_live_world import GPU, NODE, h200_preflight

CASE = "GF-ENV-LIVE-009"
FIELD = "gpu_tlimit_c"
SEVERE = -3.0
HEALTHY = 58.0
ARM = datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc)
ENDPOINT = "http://127.0.0.1:9411/metrics"


def at(seconds: float) -> datetime:
    return ARM + timedelta(seconds=seconds)


def stamp(seconds: float) -> str:
    return at(seconds).isoformat()


def expectation() -> expectations.Expectation:
    preflight = h200_preflight()
    plan = runner.replay_plans_for_case(CASE, preflight)["plans"][
        "core_tlimit/interruption"
    ]
    return expectations.expectation_for(
        case_id=CASE,
        plan_name="core_tlimit/interruption",
        plan=plan,
        preflight=preflight,
        cadence=watch.CollectorCadence(),
        node=NODE,
    )


def batch(
    batch_id: str, seconds: float, value: float, *reasons: str, boot: str = "boot-1"
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "observed_at": stamp(seconds),
        "edge_filter_reasons": list(reasons),
        "collection_errors": [],
        "gpus": [GPU],
        "samples": [
            {
                "gpu": GPU,
                "name": FIELD,
                "value": value,
                "labels": {"source_boot_id": boot},
            }
        ],
    }


def finding(finding_id: str, seconds: float, count: int) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "canonical_name": FIELD,
        "severity": "WARNING",
        "value": SEVERE,
        "gpu_uuid": GPU,
        "observed_at": stamp(seconds),
        "confirmation_count": count,
    }


class Scene:
    """One judge, its replay progress and the store the polls read."""

    def __init__(
        self,
        *,
        resume: Callable[[], dict[str, Any]] | None = None,
        restart: Callable[[], dict[str, Any]] | None = None,
        hold: Callable[[], dict[str, Any]] | None = None,
        resume_hold: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.x = expectation()
        self.judge = interruption.InterruptionJudge(
            self.x,
            steps=interruption.InterruptionSteps(
                resume=resume or self.step("resume"),
                restart=restart or self.step("restart"),
                hold=hold,
                resume_hold=resume_hold,
            ),
            interval_seconds=15,
        )
        self.progress = watch.ReplayProgress(
            planned_reads=8, armed=True, proxy_endpoint=ENDPOINT
        )
        self.progress.begin(
            {"status": {"reads": 2, "armed": True, "next_values": {}}}, ARM
        )
        self.batches: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []
        self.active: list[dict[str, Any]] = []
        self.latest: list[dict[str, Any]] = []
        self.workflows: list[dict[str, Any]] = []
        self.node: dict[str, Any] = {"unschedulable": False, "taints": []}

    def step(self, name: str) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            self.calls.append(name)
            return {}

        return run

    def reads(self, count: int, seconds: float) -> None:
        self.progress.update(
            {"status": {"reads": count, "armed": True, "next_values": {}}}, at(seconds)
        )

    def poll(
        self,
        seconds: float,
        *,
        pause: dict[str, Any] | None = None,
        hold: dict[str, Any] | None = None,
        waiter: str = "inactive",
        extra: bool = True,
    ) -> None:
        document = {
            "pause": pause or {},
            "stopped": bool(pause and pause.get("paused_at")),
            "waiter": {"ActiveState": waiter},
            "pauses": {interruption.RECOUNT_HOLD_LABEL: hold} if hold else {},
        }
        self.judge.observe(
            watch.Snapshot(
                at=at(seconds),
                gpu_state={
                    "active_findings": list(self.active),
                    "findings": list(self.findings),
                    "batches": list(self.batches),
                    "latest": list(self.latest),
                    "collector_statuses": [],
                },
                activity={
                    "incidents": [],
                    "workflows": list(self.workflows),
                    "notifications": [],
                },
                node=dict(self.node),
                extra=document if extra else None,
            ),
            self.progress,
        )

    def decide(self, seconds: float, *, final: bool = True) -> watch.Judgement:
        verdict = self.judge.decide(self.progress, at(seconds), final=final)
        assert verdict is not None, "the judge decides at the deadline"
        return verdict


PAUSED = {"target_reads": 6, "paused_at": stamp(46), "reads_at_pause": 6}


def pre_gap(scene: Scene) -> None:
    """Reads 3 and 4 severe, graded 1/3 and 2/3; the gap pause lands at +46 s."""

    scene.batches += [batch("b3", 30, SEVERE), batch("b4", 45, SEVERE)]
    scene.findings += [finding("f-warn", 30, 1)]
    scene.active[:] = [finding("f-warn", 45, 2)]


def post_gap(scene: Scene, *, graded: bool = True) -> None:
    """The first read after the resume, severe, graded 1/3 (counted anew)."""

    scene.batches.append(batch("b5", 130, SEVERE))
    if graded:
        scene.active[:] = [finding("f-after", 130, 1)]


def cleared(scene: Scene) -> None:
    scene.batches.append(batch("b6", 145, HEALTHY, "candidate-recovered"))
    scene.active[:] = []


RESUMED = {**PAUSED, "resumed_at": stamp(121)}


# --------------------------------------------------------------------------- #
# The recount hold's account of itself
# --------------------------------------------------------------------------- #
def test_the_hold_note_says_why_the_regrade_was_or_was_not_readable() -> None:
    held_at = at(131)
    notes = {
        "timed out": interruption.HoldFacts(scheduled=True, timed_out=True),
        "missed": interruption.HoldFacts(
            scheduled=True, missed=True, reads_at_pause=7, target_reads=5
        ),
        "never landed": interruption.HoldFacts(scheduled=True, waiter_state="failed"),
        "unknown waiter": interruption.HoldFacts(scheduled=True),
        "held": interruption.HoldFacts(
            scheduled=True, paused_at=held_at, reads_while_held=3
        ),
    }
    assert {name: facts.note() for name, facts in notes.items()} == {
        "timed out": "the recount hold waiter did not reach the post-gap read in time",
        "missed": "the recount hold landed after read 7, past its target 5",
        "never landed": "the recount hold never landed (waiter failed)",
        "unknown waiter": "the recount hold never landed (waiter unknown)",
        "held": "the collector was held after the post-gap read and the state was "
        "read 3 time(s) meanwhile",
    }
    assert notes["held"].taken and not notes["never landed"].taken, (
        "a hold is taken once the node stamped its pause"
    )
    assert notes["held"].evidence()["paused_at"] == held_at.isoformat()


# --------------------------------------------------------------------------- #
# Intake: the node's own records
# --------------------------------------------------------------------------- #
def test_the_node_stamps_of_the_pause_and_the_hold_are_taken_as_recorded() -> None:
    scene = Scene()
    pre_gap(scene)
    scene.poll(20, extra=False)
    assert scene.judge.pause.scheduled is False, "a poll without the fifth read"
    scene.poll(40, pause={"paused_at": "not-a-stamp", "reads_at_pause": 6})
    assert scene.judge.pause.scheduled is True
    assert scene.judge.pause.target_reads is None, "the record named no target"
    assert scene.judge.pause.paused_at is None, "an unparsable stamp is no pause"
    scene.poll(122, pause=RESUMED, hold={"paused_at": stamp(131)})
    pause = scene.judge.pause
    assert (pause.paused_at, pause.resumed_at) == (at(46), at(121)), (
        "the node's SIGSTOP and SIGCONT stamps, not the poll's"
    )
    assert scene.calls == [], "the resume is never repeated once the node took it"
    assert pause.hold.scheduled and pause.hold.target_reads is None, pause.hold
    assert pause.hold.paused_at == at(131)
    scene.poll(
        140,
        pause=RESUMED,
        hold={"paused_at": stamp(131), "resumed_at": stamp(139), "target_reads": 7},
    )
    assert (pause.hold.resumed_at, pause.hold.target_reads) == (at(139), 7)
    evidence = pause.evidence()
    assert evidence["resumed_at"] == stamp(121)
    assert evidence["hold"]["resumed_at"] == stamp(139)


def test_a_finding_cleared_in_the_gap_is_named_once_and_a_held_read_counted() -> None:
    scene = Scene()
    pre_gap(scene)
    scene.poll(50, pause=PAUSED)
    assert scene.judge.pause.active_while_paused == {"f-warn"}
    scene.active[:] = []
    scene.poll(65, pause=PAUSED)
    scene.poll(80, pause=PAUSED)
    assert scene.judge.pause.cleared_in_gap == ["f-warn"], (
        "a finding that vanished in the gap is listed once, however many polls"
    )
    scene.poll(125, pause=RESUMED, hold={"paused_at": stamp(124)})
    hold = scene.judge.pause.hold
    assert (hold.reads_while_held, hold.reads_with_post_sample) == (1, 0), (
        "the held state was read before the post-gap batch reached the store"
    )
    stages = scene.decide(126, final=False).stages
    assert stages["gap"] == [
        f"{CASE}: finding(s) ['f-warn'] were cleared while no reading was accepted: "
        "the gap was treated as healthy"
    ]


# --------------------------------------------------------------------------- #
# The maintenance steps
# --------------------------------------------------------------------------- #
def test_a_refused_resume_and_hold_resume_are_step_errors_and_no_restart() -> None:
    def refuse(name: str) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            raise RuntimeError(f"{name} refused by SSM")

        return run

    scene = Scene(resume=refuse("resume"), hold=dict, resume_hold=refuse("hold"))
    pre_gap(scene)
    scene.poll(50, pause=PAUSED)
    scene.poll(122, pause=PAUSED)
    pause = scene.judge.pause
    assert pause.hold.scheduled and pause.hold.target_reads is None, (
        "a hold receipt without target_reads leaves the target unknown"
    )
    assert pause.resumed_at == at(122), "no receipt: the poll's own moment"
    post_gap(scene)
    scene.poll(135, pause=PAUSED, hold={"paused_at": stamp(131)})
    assert pause.hold.resumed_at == at(135)
    cleared(scene)
    scene.reads(10, 150)
    scene.poll(150, pause=PAUSED)
    assert pause.restart_at is None, "no restart after a failed maintenance step"
    assert scene.judge.step_errors == [
        "resume-collector: RuntimeError: resume refused by SSM",
        "resume-collector[hold]: RuntimeError: hold refused by SSM",
    ]
    verdict = scene.decide(151)
    assert verdict.outcome == "FAIL"
    assert verdict.stages["steps"] == scene.judge.step_errors


def test_a_refused_restart_is_recorded_and_fails_the_case() -> None:
    def restart() -> dict[str, Any]:
        raise OSError("systemctl restart timed out")

    scene = Scene(restart=restart)
    pre_gap(scene)
    scene.poll(50, pause=PAUSED)
    scene.poll(122, pause=PAUSED)
    assert scene.calls == ["resume"]
    post_gap(scene)
    cleared(scene)
    scene.reads(10, 150)
    scene.poll(150, pause=RESUMED)
    assert scene.judge.pause.restart_at == at(150)
    verdict = scene.decide(151)
    assert verdict.outcome == "FAIL"
    assert verdict.stages["steps"] == [
        "restart-collector: OSError: systemctl restart timed out"
    ]


def test_the_wait_covers_the_clearing_read_after_the_plan_is_consumed() -> None:
    scene = Scene()
    pre_gap(scene)
    scene.poll(50, pause=PAUSED)
    scene.poll(122, pause=RESUMED)
    post_gap(scene)
    scene.reads(10, 140)
    scene.poll(140, pause=RESUMED)
    assert scene.judge.cleared_at() is None, "the post-gap finding is still active"
    assert scene.judge.wait_until(scene.progress, at(141)) == at(140) + timedelta(
        seconds=2 * scene.x.hold_seconds
    )


# --------------------------------------------------------------------------- #
# The deadline: why there is no verdict, in the judge's order
# --------------------------------------------------------------------------- #
def missed(scene: Scene) -> float:
    scene.poll(50, pause={**PAUSED, "reads_at_pause": 7, "missed": True})
    return 51


def never_paused(scene: Scene) -> float:
    scene.reads(10, 60)
    scene.poll(60, pause={"target_reads": 6}, waiter="active")
    return 61


def never_paused_unknown(scene: Scene) -> float:
    scene.reads(10, 60)
    scene.poll(60, pause={"target_reads": 6}, waiter="")
    return 61


def not_resumed(scene: Scene) -> float:
    pre_gap(scene)
    scene.reads(10, 60)
    scene.poll(60, pause=PAUSED)
    return 61


def no_post_sample(scene: Scene) -> float:
    pre_gap(scene)
    scene.reads(10, 125)
    scene.poll(125, pause=RESUMED)
    return 126


def short_gap(scene: Scene) -> float:
    pre_gap(scene)
    scene.batches.append(batch("b5", 90, SEVERE))
    scene.reads(10, 95)
    scene.poll(95, pause={**PAUSED, "resumed_at": stamp(80)})
    return 96


def hold_not_resumed(scene: Scene) -> float:
    pre_gap(scene)
    post_gap(scene)
    scene.reads(10, 140)
    scene.poll(140, pause=RESUMED, hold={"paused_at": stamp(131)})
    return 141


def not_cleared(scene: Scene) -> float:
    pre_gap(scene)
    post_gap(scene)
    scene.reads(10, 140)
    scene.poll(140, pause=RESUMED)
    return 141


def restart_not_taken(scene: Scene) -> float:
    pre_gap(scene)
    post_gap(scene)
    scene.findings.append(finding("f-after", 130, 1))
    cleared(scene)
    scene.poll(146, pause=RESUMED)
    scene.reads(10, 150)
    return 151


def no_fresh_batch(scene: Scene) -> float:
    restart_not_taken(scene)
    scene.poll(152, pause=RESUMED)
    assert scene.calls == ["restart"], "the restart is taken once the plan is served"
    return 153


def no_single_source(scene: Scene) -> float:
    no_fresh_batch(scene)
    scene.batches.append(batch("b7", 170, HEALTHY, "initial-baseline"))
    scene.poll(175, pause=RESUMED)
    return 176


UNFINISHED = {
    missed: "the pause landed after read 7, past its target 6; the product's "
    "reaction to three consecutive severe samples is not this case's evidence",
    never_paused: "the collector was never paused (waiter active)",
    never_paused_unknown: "the collector was never paused (waiter unknown)",
    not_resumed: "the collector was not resumed before the deadline",
    no_post_sample: f"no accepted severe {FIELD} sample after the resume shows the gap",
    short_gap: "the accepted samples are only 45s apart, not past the 60s "
    "confirmation gap",
    hold_not_resumed: "the recount hold was not resumed before the deadline",
    not_cleared: "the clearing reading was not accepted before the deadline",
    restart_not_taken: "the collector restart was not taken before the deadline",
    no_fresh_batch: "the restarted collector's initial-baseline batch did not reach "
    "the control plane before the deadline",
    no_single_source: "the case did not complete before the deadline",
}


@pytest.mark.parametrize("scenario", list(UNFINISHED), ids=lambda item: item.__name__)
def test_the_deadline_names_the_first_missing_piece(
    scenario: Callable[[Scene], float],
) -> None:
    scene = Scene()
    deadline = scenario(scene)
    verdict = scene.decide(deadline)
    assert verdict.outcome == "INCONCLUSIVE", verdict.stages
    assert verdict.inconclusive == [f"{CASE}: {UNFINISHED[scenario]}"]


# --------------------------------------------------------------------------- #
# Definite failures besides the recount
# --------------------------------------------------------------------------- #
def test_recovery_isolation_second_producer_and_foreign_boot_fail_their_stages() -> (
    None
):
    scene = Scene()
    no_fresh_batch(scene)
    restart = scene.judge.pause.restart_at
    assert restart == at(152)
    scene.batches += [
        batch(f"r{index}", 155 + 2 * index, HEALTHY, "initial-baseline", boot="boot-2")
        for index in range(6)
    ]
    scene.latest += [
        {"gpu": GPU, "name": FIELD, "observed_at": stamp(156), "source": "DCGM"},
        {"gpu": GPU, "name": FIELD, "observed_at": stamp(158), "source": "NVML"},
    ]
    scene.workflows += [
        {
            "request_id": "wf-restore",
            "official_steps": [{"operation": "RESTORE_SCHEDULING"}],
        },
        {"request_id": "wf-diag", "official_steps": [{"operation": "RUN_DIAGNOSTICS"}]},
    ]
    scene.node = {"unschedulable": True, "taints": []}
    scene.poll(170, pause=RESUMED)
    stages = scene.decide(171, final=False).stages

    assert stages["reaction"] == [
        f"{CASE}: workflow wf-restore restores scheduling: an unauthorized recovery",
        f"{CASE}: the node was isolated although the gap must restart the confirmation",
    ], "a diagnostic-only workflow is no reaction"
    assert stages["producer"] == [
        f"{CASE}: accepted readings carry ['DCGM', 'NVML'] sources; one producer "
        "is required",
        f"{CASE}: 6 batches in 10s after the restart exceed one producer's 3 at 15s",
    ]
    assert stages["restart"] == [
        f"{CASE}: the restarted collector's readings carry source_boot_id "
        "['boot-2'], the node booted boot-1"
    ]
    assert stages["recount"] == [] and stages["gap"] == [], stages
