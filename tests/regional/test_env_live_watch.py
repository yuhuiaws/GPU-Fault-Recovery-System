"""The bounded GF-ENV-LIVE observation after arming, driven by a scripted world.

Supporting logic coverage only: no cluster, node or store is touched, and a
passing test is never a LIVE PASS. The world serves proxy ``/status`` reads one
per collector interval and the product's records at scripted offsets; the
assertions pin the verdict rules -- INCONCLUSIVE when the replay or its
ingestion cannot be shown, FAIL on a missing or wrong reaction, PASS only on
positive evidence -- and the product settings the waits derive from.
"""

from __future__ import annotations

import inspect
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.adapters.common import QUARANTINE_TAINT
from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.collectors.cadence import CadenceJitter
from gpu_fault.collectors.context import CollectorContext
from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS
from scripts.e2e.regional import env_live_expectations as expectations
from scripts.e2e.regional import env_live_judges as judges
from scripts.e2e.regional import env_live_watch as watch
from scripts.e2e.regional import run_env_live as runner
from tests.regional._env_live_world import (
    DIAGNOSTIC_OPERATIONS,
    DROPIN,
    GPU,
    NODE,
    PROXY,
    REAL,
    REFUSAL,
    REVIEW_OPERATIONS,
    UNIT_INVOCATION,
    World,
    baseline_values,
    h200_preflight,
    review_chain,
)

COLLECTOR_ENV = {
    "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "3",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
}


def _plan(case_id: str, name: str) -> dict[str, Any]:
    if name == "passthrough":
        return {"steps": [{"reads": 1, "values": {}}]}
    return dict(runner.replay_plans_for_case(case_id, h200_preflight())["plans"][name])


def _run(
    world: World,
    case_id: str,
    plan_name: str,
    *,
    before: dict[str, Any] | None = None,
    reverify: bool = False,
) -> tuple[watch.WatchResult, watch.ObservationWindow]:
    """What ``run_env_live._run_replay`` does after the deploy, on the world.

    ``reverify`` binds the zero-read re-verification the runner binds: the
    first-read due time, the repoint receipt's baseline and the unit snapshot.
    """

    preflight = h200_preflight()
    cadence = watch.CollectorCadence.from_probe({"collector_env": COLLECTOR_ENV})
    expectation = expectations.expectation_for(
        case_id=case_id,
        plan_name=plan_name,
        plan=_plan(case_id, plan_name),
        preflight=preflight,
        cadence=cadence,
        node=NODE,
    )
    node_before = world.node_snapshot()
    receipt = world.repoint()
    armed = expectation.kind != "baseline"
    progress = watch.ReplayProgress(
        planned_reads=expectation.planned_reads,
        armed=armed,
        proxy_endpoint=PROXY,
        first_read_due_seconds=cadence.first_read_due_seconds if reverify else None,
        repoint_baseline=watch.repoint_baseline(receipt) if reverify else {},
    )
    if armed:
        start = world.now
        progress.begin(world.arm(), start)
    else:
        start = world.now
        progress.begin(world.status(), start)
    assert world.repoint_at is not None, "the world was repointed"
    window = watch.observation_window(
        started_at=start,
        terms=expectation.terms,
        replay_end=start + timedelta(seconds=600) if armed else None,
        deadman_at=world.repoint_at + timedelta(seconds=600),
        maintenance_end=start + timedelta(hours=1),
    )
    judge = judges.judge_for(
        expectation,
        repoint_at=world.repoint_at,
        before=before,
        collector_mode=cadence.metrics_mode,
        node_before=node_before,
    )
    reads = watch.WatchReads(
        status=world.status,
        gpu_state=world.gpu_state,
        activity=world.activity,
        node=world.node_snapshot,
        repoint=world.snapshot if reverify else None,
    )
    outcome = watch.watch(
        judge,
        progress,
        reads,
        window,
        poll_seconds=cadence.poll_seconds,
        clock=world.clock,
        sleep=world.sleep,
    )
    return outcome, window


def _errors(outcome: watch.WatchResult) -> str:
    return "\n".join(
        item for items in outcome.judgement.stages.values() for item in items
    )


# --------------------------------------------------------------------------- #
# The product settings the waits derive from
# --------------------------------------------------------------------------- #
def test_collector_defaults_are_the_metrics_collectors_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
        "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    class Sink:
        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            return {"accepted": True}

    collector = DcgmMetricsCollector(
        Sink(), CollectorContext(cluster_id="cluster-a"), node_id="node-a"
    )
    assert collector.interval_seconds == watch.DEFAULT_INTERVAL_SECONDS
    assert (
        collector.edge_confirmation_samples == watch.DEFAULT_EDGE_CONFIRMATION_SAMPLES
    )
    assert collector.health_summary_seconds == watch.DEFAULT_HEALTH_SUMMARY_SECONDS
    assert collector.startup_spread_seconds == watch.STARTUP_SPREAD_SECONDS


def test_control_plane_windows_are_the_products_own() -> None:
    grace = inspect.signature(GpuValidationAdapter.__init__).parameters[
        "temperature_warning_grace"
    ]
    assert grace.default == timedelta(
        seconds=expectations.TEMPERATURE_WARNING_GRACE_SECONDS
    )
    assert judges.QUARANTINE_TAINT == QUARANTINE_TAINT == "gpu-fault.io/quarantined"
    assert COOLDOWN_SECONDS == 300


def test_cadence_reads_the_nodes_collector_env_and_keeps_defaults() -> None:
    cadence = watch.CollectorCadence.from_probe(
        {"collector_env": {"GPU_FAULT_METRICS_INTERVAL_SECONDS": "30"}}
    )
    assert cadence.interval_seconds == 30
    assert cadence.edge_confirmation_samples == 3
    assert cadence.health_summary_seconds == 300
    assert cadence.configured == ("GPU_FAULT_METRICS_INTERVAL_SECONDS",)
    assert cadence.delivery_margin_seconds == 30 + watch.PROCESSING_MARGIN_SECONDS
    assert cadence.poll_seconds == watch.MAX_POLL_SECONDS
    assert watch.CollectorCadence.from_probe(None).interval_seconds == 15
    assert not watch.CollectorCadence(interval_seconds=61).confirmation_gap_ok(), (
        "a 61 s interval cannot confirm within the 60 s gap"
    )
    # The start-up spread, start-up jitter and cadence jitter ratio default to
    # the product's, come from collector.env when set (zero included), and
    # size the jittered tick, the start-up allowance and the first-read due.
    assert (cadence.startup_spread_seconds, cadence.startup_jitter_seconds) == (15, 20)
    assert cadence.jitter_ratio == 0.1
    assert cadence.max_interval_seconds == 30 * 1.1
    assert cadence.startup_allowance_seconds == 35
    assert cadence.first_read_due_seconds == 30 * 1.1 + 35
    assert cadence.jitter_seconds(7) == 7 * 30 * 0.1
    tuned = watch.CollectorCadence.from_probe(
        {
            "collector_env": {
                "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS": "5",
                "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS": "0",
                "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO": "0.2",
            }
        }
    )
    assert (tuned.startup_spread_seconds, tuned.startup_jitter_seconds) == (5, 0)
    assert tuned.jitter_ratio == 0.2 and tuned.first_read_due_seconds == 15 * 1.2 + 5
    assert set(tuned.configured) == {
        "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS",
        "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS",
        "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO",
    }
    assert tuned.evidence()["first_read_due_seconds"] == 15 * 1.2 + 5


def test_cadence_jitter_defaults_are_the_products_own() -> None:
    jitter = CadenceJitter()
    assert watch.STARTUP_JITTER_SECONDS == jitter.startup_seconds == 20
    assert watch.CADENCE_JITTER_RATIO == jitter.ratio == 0.1
    assert watch.CollectorCadence().max_interval_seconds == jitter.max_interval(15)


def test_review_terms_derive_from_the_plan_cadence_and_cooldown() -> None:
    cadence = watch.CollectorCadence.from_probe({"collector_env": COLLECTOR_ENV})
    four = expectations.expectation_for(
        case_id="GF-ENV-LIVE-004",
        plan_name="core_tlimit/severe",
        plan=_plan("GF-ENV-LIVE-004", "core_tlimit/severe"),
        preflight=h200_preflight(),
        cadence=cadence,
        node=NODE,
    )
    assert four.kind == "review" and four.field == "gpu_tlimit_c"
    assert four.trigger_values == (-3.0,) and four.required_samples == 3
    assert four.threshold_source == "NVIDIA_DEVICE_TLIMIT"
    # Sized from the collector cadence like the repoint's first-read budget:
    # the reads at the interval, the cadence jitter over them, the start-up
    # spread and jitter a restarted collector needs, the delivery margin.
    assert four.terms == {
        "replay_reads_x_interval": 5 * 15,
        "cadence_jitter": 5 * 15 * 0.1,
        "collector_startup_spread_and_jitter": 15 + 20,
        "delivery_margin": 15 + watch.PROCESSING_MARGIN_SECONDS,
        "cooldown_hold": COOLDOWN_SECONDS + 15,
        "workflow_dispatch": watch.PROCESSING_MARGIN_SECONDS,
    }
    three = expectations.expectation_for(
        case_id="GF-ENV-LIVE-003",
        plan_name="core_tlimit/warning",
        plan=_plan("GF-ENV-LIVE-003", "core_tlimit/warning"),
        preflight=h200_preflight(),
        cadence=cadence,
        node=NODE,
    )
    assert three.kind == "warning" and three.require_diagnostic
    assert three.fallback_field == "gpu_temperature_c"
    assert three.terms["validate_gpu_warning_grace_or_hold"] == (
        expectations.TEMPERATURE_WARNING_GRACE_SECONDS + watch.PROCESSING_MARGIN_SECONDS
    )
    transient = expectations.expectation_for(
        case_id="GF-ENV-LIVE-005",
        plan_name="transient/warning_then_severe",
        plan=_plan("GF-ENV-LIVE-005", "transient/warning_then_severe"),
        preflight=h200_preflight(),
        cadence=cadence,
        node=NODE,
    )
    # The warning reads never count toward the severe confirmation.
    assert transient.trigger_values == (-3.0,) and transient.max_confirmation == 1
    with pytest.raises(ValueError, match="no observation rule"):
        expectations.expectation_for(
            case_id="GF-ENV-LIVE-002",
            plan_name="x",
            plan={"steps": [{"reads": 1, "values": {}}]},
            preflight=h200_preflight(),
            cadence=cadence,
            node=NODE,
        )


def test_window_is_capped_by_the_earliest_hard_end_minus_cleanup() -> None:
    world = World()
    window = watch.observation_window(
        started_at=world.now,
        terms={"a": 500.0, "b": 200.0},
        replay_end=world.now + timedelta(seconds=600),
        deadman_at=world.now + timedelta(seconds=590),
        maintenance_end=world.now + timedelta(hours=2),
    )
    assert window.derived_seconds == 700
    assert window.cap_at == world.now + timedelta(
        seconds=590 - watch.CLEANUP_MARGIN_SECONDS
    )
    assert window.deadline == window.cap_at and window.capped
    assert window.evidence()["caps"]["binding"] == "collector_deadman"


def test_progress_counts_reads_from_the_arm_receipt() -> None:
    world = World()
    receipt = world.repoint()  # returns after the first pass-through read (one tick)
    assert receipt["first_read"] == {
        "observed": True,
        "reads_before": 0,
        "reads_after": 1,
        "waited_seconds": 15.0,
        "budget_seconds": 65,
    }
    world.sleep(45)  # four pass-through scrapes before the arm (60 s in all)
    progress = watch.ReplayProgress(planned_reads=3, armed=True, proxy_endpoint=PROXY)
    progress.begin(world.arm(), world.now)
    assert progress.start_reads == 4 and progress.served == 0
    world.sleep(30)
    progress.update(world.status(), world.now)
    assert progress.served == 2 and not progress.consumed
    world.sleep(15)
    progress.update(
        {"status": {**world.status()["status"], "next_values": {"x": 1}}}, world.now
    )
    assert not progress.consumed, "a pending step is not a consumed plan"
    progress.update(world.status(), world.now)
    assert progress.consumed and progress.inconclusive() == []
    restarted = watch.ReplayProgress(planned_reads=3, armed=True, proxy_endpoint=PROXY)
    restarted.begin(world.arm(), world.now)
    world.armed = False  # the tool restarted: pass-through, a fresh read count
    world.sleep(60)
    restarted.update(world.status(), world.now)
    assert not restarted.consumed, "a restarted proxy never served the plan"
    assert "reported itself unarmed" in "\n".join(restarted.inconclusive())
    world.armed = True
    world.aborted = "live safety stop: DCGM_FI_DEV_GPU_TEMP"
    stuck = watch.ReplayProgress(planned_reads=9, armed=True, proxy_endpoint=PROXY)
    stuck.begin(world.arm(), world.now)
    stuck.update({**world.status(), "effective_endpoint": None}, world.now)
    reasons = "\n".join(stuck.inconclusive())
    assert "aborted on a real upstream safety reading" in reasons
    assert "stopped reading the proxy" in reasons and "consumed 0 of 9" in reasons


def _baseline() -> dict[str, Any]:
    world = World()
    return watch.repoint_baseline(world.repoint())


def _snapshot(world: World | None = None, **unit: Any) -> dict[str, Any]:
    world = world or World()
    world.repoint()
    value = world.snapshot()
    value["metrics_unit"].update(unit)
    return value


def test_repoint_baseline_pins_the_unit_the_receipt_left() -> None:
    baseline = _baseline()
    assert baseline["proxy_endpoint"] == PROXY and baseline["original_endpoint"] == REAL
    assert baseline["dropin"] == DROPIN
    assert baseline["invocation_id"] == UNIT_INVOCATION
    assert baseline["exec_main_start"] and baseline["main_pid"] == "4242"
    assert baseline["first_read"]["observed"] is True
    assert watch.repoint_baseline(None)["invocation_id"] is None


def test_diagnose_zero_reads_names_the_reason_in_order_of_decisiveness() -> None:
    baseline = _baseline()
    diagnose = watch.diagnose_zero_reads
    # The live 2026-10-26 GF-ENV-LIVE-006 shape: unit and drop-in unchanged,
    # the process on the proxy, the proxy refusing every armed read.
    refused = diagnose(
        baseline,
        _snapshot(),
        {"failed_reads": 4, "last_error": REFUSAL},
        proxy_endpoint=PROXY,
    )
    assert refused["reason"] == watch.ZERO_READS_PROXY_REFUSED
    assert "4 collector read(s) with 503" in refused["detail"]
    assert REFUSAL in refused["detail"]
    assert refused["dropin_loaded"] is True and refused["unit_restarted"] is False
    # The endpoint wins over everything else; the drop-in over a restart.
    world = World()
    world.endpoint = REAL
    world.dropin_loaded = False
    reverted = diagnose(
        baseline, _snapshot(world, InvocationID="other"), {}, proxy_endpoint=PROXY
    )
    assert reverted["reason"] == watch.ZERO_READS_ENDPOINT_REVERTED
    assert "reads 'http://127.0.0.1:9400/metrics', not the proxy" in reverted["detail"]
    assert "drop-in is no longer loaded" in reverted["detail"]
    assert "the unit restarted" in reverted["detail"]
    gone = diagnose(baseline, _snapshot(DropInPaths=""), {}, proxy_endpoint=PROXY)
    assert gone["reason"] == watch.ZERO_READS_DROPIN_GONE
    inactive = diagnose(
        baseline, _snapshot(ActiveState="failed"), {}, proxy_endpoint=PROXY
    )
    assert inactive["reason"] == watch.ZERO_READS_UNIT_INACTIVE
    assert "is failed, not active" in inactive["detail"]
    restarted = diagnose(
        baseline,
        _snapshot(InvocationID="0f1e2d3c", ExecMainStartTimestamp="later"),
        {"failed_reads": 0},
        proxy_endpoint=PROXY,
    )
    assert restarted["reason"] == watch.ZERO_READS_UNIT_RESTARTED
    assert f"InvocationID {UNIT_INVOCATION} -> 0f1e2d3c" in restarted["detail"]
    assert "started later" in restarted["detail"]
    silent = diagnose(baseline, _snapshot(), {"failed_reads": 0}, proxy_endpoint=PROXY)
    assert silent["reason"] == watch.ZERO_READS_COLLECTOR_SILENT
    assert "refused none" in silent["detail"]
    older_tool = diagnose(baseline, _snapshot(), {}, proxy_endpoint=PROXY)
    assert older_tool["reason"] == watch.ZERO_READS_COLLECTOR_SILENT
    assert "reports no refusal count" in older_tool["detail"]
    unverified = diagnose(baseline, None, {}, proxy_endpoint=PROXY)
    assert unverified["reason"] == watch.ZERO_READS_UNVERIFIED


def test_watch_reverifies_the_repoint_once_when_the_first_read_is_overdue() -> None:
    world = World()
    world.refuse_armed_reads = True
    outcome, _window = _run(world, "GF-ENV-LIVE-006", "hw_thermal", reverify=True)
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    reasons = "\n".join(outcome.judgement.inconclusive)
    assert "consumed 0 of 7 planned replay reads" in reasons
    assert f"({watch.ZERO_READS_PROXY_REFUSED})" in reasons and REFUSAL in reasons
    assert world.snapshots == 1, "one snapshot, not one per poll"
    replay = outcome.judgement.evidence["replay"]
    assert replay["zero_reads_diagnosis"]["reason"] == watch.ZERO_READS_PROXY_REFUSED
    assert replay["failed_reads"] >= 3 and replay["served"] == 0
    assert replay["first_read_due_seconds"] == 15 * 1.1 + 35
    assert world.armed_at is not None
    diagnosed_at = replay["zero_reads_diagnosis"]["at"]
    assert (
        diagnosed_at >= (world.armed_at + timedelta(seconds=15 * 1.1 + 35)).isoformat()
    )


def test_watch_without_the_snapshot_reader_or_a_served_read_never_diagnoses() -> None:
    # No reader bound (an older runner wiring): the old inconclusive text only.
    world = World()
    world.freeze_reads_at_arm = True
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.evidence["replay"]["zero_reads_diagnosis"] is None
    assert world.snapshots == 0
    # A collector that reads (late or not) is never re-verified.
    served = World()
    outcome, _window = _run(
        served, "GF-ENV-LIVE-004", "core_tlimit/severe", reverify=True
    )
    assert outcome.judgement.evidence["replay"]["consumed_at"] is not None
    assert served.snapshots == 0


def test_a_failing_read_is_recorded_and_the_loop_still_ends() -> None:
    world = World()

    def broken() -> dict[str, Any]:
        raise RuntimeError("kubectl exec failed")

    world.gpu_state = broken  # type: ignore[method-assign]
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    assert any("kubectl exec failed" in item for item in outcome.read_errors), (
        outcome.read_errors
    )


# --------------------------------------------------------------------------- #
# Review cases (004 / 011 / 007 sustained and mixed)
# --------------------------------------------------------------------------- #
def test_review_without_any_product_reaction_is_inconclusive_never_pass() -> None:
    # The live defect: the proxy served the plan, the store shows nothing --
    # the old single snapshot either failed "no workflow" or (for the
    # read-only cases) passed. Neither is a verdict on the product.
    outcome, window = _run(World(), "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "INCONCLUSIVE", outcome.judgement
    assert outcome.judgement.evidence["replay"]["served"] >= 5
    assert any(
        "accepted 0 valid severe gpu_tlimit_c samples" in item
        for item in outcome.judgement.inconclusive
    ), outcome.judgement.inconclusive
    assert outcome.ended_at >= window.deadline and outcome.polls > 10


def test_review_with_accepted_samples_but_no_confirmation_fails() -> None:
    world = World()
    for offset in (45, 60, 75):
        world.accept(offset, {"gpu_tlimit_c": -3.0, "gpu_temperature_c": 30.0})
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "FAIL"
    assert "no CRITICAL finding confirmed them" in _errors(outcome)


def test_review_confirmed_without_a_workflow_fails_at_the_deadline() -> None:
    world = World()
    for offset in (45, 60, 75):
        world.accept(offset, {"gpu_tlimit_c": -3.0, "gpu_temperature_c": 30.0})
    world.finding(
        75,
        "f-crit",
        name="gpu_tlimit_c",
        value=-3.0,
        severity="CRITICAL",
        count=3,
        source="NVIDIA_DEVICE_TLIMIT",
    )
    outcome, window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "FAIL"
    assert "no ENVIRONMENT_REVIEW workflow was observed" in _errors(outcome)
    assert outcome.ended_at >= window.deadline


def test_review_full_chain_passes_only_after_the_cooldown_hold() -> None:
    world = review_chain(World())
    outcome, window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    judgement = outcome.judgement
    assert judgement.outcome == "PASS", (judgement.stages, judgement.inconclusive)
    hold = judgement.evidence["hold"]
    assert hold["held"] and hold["recovered_reading_at"]
    # The workflow finished at +200 s; the PASS waited for the 315 s hold past
    # the cooled reading, inside the derived window, and never past its cap.
    assert outcome.ended_at >= world.moment(90 + COOLDOWN_SECONDS + 15)
    assert outcome.ended_at <= window.cap_at
    assert judgement.evidence["valid_severe_run"] == 3
    assert judgement.evidence["task"]["measured"] is False
    assert "no owned test task" in judgement.evidence["task"]["reason"]


def test_review_that_is_not_consumed_is_inconclusive() -> None:
    world = review_chain(World())
    # The collector stopped after one plan read (absolute reads: the arm's
    # own pass-through read plus one).
    world.reads_frozen_at = 1 + 1
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    assert any("consumed" in item for item in outcome.judgement.inconclusive), (
        outcome.judgement.inconclusive
    )


def test_review_with_a_forbidden_step_fails_at_once() -> None:
    world = review_chain(World())
    world.workflow(
        80, "wf-reset", "f-crit", operations=("FREEZE_EVIDENCE", "RESET_GPU")
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "FAIL"
    assert "forbidden ['RESET_GPU']" in _errors(outcome)
    assert outcome.ended_at < world.moment(120), "a definite FAIL does not wait"


def test_review_on_the_wrong_gpu_scope_fails() -> None:
    world = review_chain(World())
    world.incident(
        75, "f-crit", action="DRAIN", workflow="wf-review", gpus=[GPU, "GPU-bbbb"]
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "FAIL"
    assert "scopes gpu_uuids" in _errors(outcome)


def test_review_still_running_inside_its_deadline_is_inconclusive() -> None:
    world = review_chain(World(), finish_at=5000)
    outcome, window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    assert any("still RUNNING" in item for item in outcome.judgement.inconclusive), (
        outcome.judgement.inconclusive
    )
    # A running required workflow keeps the loop going to the cap, not past it.
    assert window.deadline <= outcome.ended_at <= window.cap_at + timedelta(seconds=30)


def test_review_past_its_own_execution_deadline_fails() -> None:
    world = review_chain(World(), finish_at=5000)
    world.workflow(
        75,
        "wf-review",
        "f-crit",
        operations=REVIEW_OPERATIONS,
        status="RUNNING",
        deadline_seconds=120,
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "FAIL"
    assert "past its own execution deadline" in _errors(outcome)


def test_review_released_node_fails() -> None:
    world = review_chain(World())
    world.release(250)
    outcome, _window = _run(world, "GF-ENV-LIVE-004", "core_tlimit/severe")
    assert outcome.judgement.outcome == "FAIL"
    assert "released after the review isolated it" in _errors(outcome)


def test_power_brake_confirmation_must_be_a_power_delivery_check() -> None:
    world = review_chain(
        World(),
        name="clock_throttle_reasons",
        severe=float(0x180),
        healthy=0.0,
        first_severe=15,
        source="NVIDIA_HW_POWER_BRAKE",
        policy="SITE_NVIDIA_POWER_DELIVERY",
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-007", "mixed_0x180")
    assert outcome.judgement.outcome == "PASS", outcome.judgement.stages
    wrong = review_chain(
        World(),
        name="clock_throttle_reasons",
        severe=float(0x180),
        healthy=0.0,
        first_severe=15,
        source="NVIDIA_HW_POWER_BRAKE",
        policy="SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
    )
    outcome, _window = _run(wrong, "GF-ENV-LIVE-007", "mixed_0x180")
    assert outcome.judgement.outcome == "FAIL"
    assert "not a power-delivery check" in _errors(outcome)


# --------------------------------------------------------------------------- #
# Read-only signals
# --------------------------------------------------------------------------- #
def _warning_world(*, workflow_at: float | None = 75, done_at: float = 180) -> World:
    world = World()
    world.accept(
        75,
        {"gpu_tlimit_c": 0.5, "gpu_temperature_c": 30.0},
        reasons=("candidate-confirmed",),
    )
    world.finding(
        75,
        "f-warn",
        name="gpu_tlimit_c",
        value=0.5,
        severity="WARNING",
        source="NVIDIA_DEVICE_TLIMIT",
    )
    if workflow_at is not None:
        world.incident(workflow_at, "f-warn", workflow="wf-diag")
        world.workflow(
            workflow_at, "wf-diag", "f-warn", operations=DIAGNOSTIC_OPERATIONS
        )
        world.finish(done_at, "wf-diag")
        world.set_incident(done_at, "f-warn", state="RECOVERED")
    world.accept(
        120,
        {"gpu_tlimit_c": 58.0, "gpu_temperature_c": 30.0},
        reasons=("candidate-recovered",),
    )
    world.clear(120, "f-warn")
    return world


def test_warning_without_any_reaction_is_inconclusive_never_pass() -> None:
    outcome, _window = _run(World(), "GF-ENV-LIVE-003", "core_tlimit/warning")
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    assert any(
        "no accepted record proves the replayed gpu_tlimit_c" in item
        for item in outcome.judgement.inconclusive
    ), outcome.judgement.inconclusive


def test_warning_waits_for_the_diagnostic_workflow_and_passes() -> None:
    # The workflow appears with the delivered warning (+75 s) and finishes at
    # +180 s, several polls after arming: the PASS waits for it.
    world = _warning_world()
    outcome, _window = _run(world, "GF-ENV-LIVE-003", "core_tlimit/warning")
    judgement = outcome.judgement
    assert judgement.outcome == "PASS", (judgement.stages, judgement.inconclusive)
    assert outcome.ended_at >= world.moment(180) and outcome.polls >= 12
    assert judgement.evidence["diagnostic_workflows"] == ["wf-diag"]
    assert judgement.evidence["graded_evidence"] == ["f-warn"]


def test_warning_graded_but_without_the_required_workflow_fails() -> None:
    outcome, window = _run(
        _warning_world(workflow_at=None), "GF-ENV-LIVE-003", "core_tlimit/warning"
    )
    assert outcome.judgement.outcome == "FAIL"
    assert "opened no read-only diagnostic workflow" in _errors(outcome)
    assert outcome.ended_at >= window.deadline


def test_warning_accepted_but_never_graded_fails() -> None:
    world = World()
    world.accept(75, {"gpu_tlimit_c": 0.5, "gpu_temperature_c": 30.0}, reasons=("x",))
    outcome, _window = _run(world, "GF-ENV-LIVE-003", "core_tlimit/warning")
    assert outcome.judgement.outcome == "FAIL"
    assert "were never graded into a finding" in _errors(outcome)


@pytest.mark.parametrize(
    ("operations", "expected"),
    [
        (("FREEZE_EVIDENCE", "QUARANTINE"), "isolating steps ['QUARANTINE']"),
        (("FREEZE_EVIDENCE", "RESTORE_SCHEDULING"), "non-diagnostic steps"),
    ],
)
def test_warning_that_isolates_or_strays_fails(
    operations: tuple[str, ...], expected: str
) -> None:
    world = _warning_world(workflow_at=None)
    world.incident(75, "f-warn", workflow="wf-bad")
    world.workflow(75, "wf-bad", "f-warn", operations=operations)
    outcome, _window = _run(world, "GF-ENV-LIVE-003", "core_tlimit/warning")
    assert outcome.judgement.outcome == "FAIL"
    assert expected in _errors(outcome)


def test_warning_escalated_or_on_a_fallback_scale_fails() -> None:
    world = _warning_world()
    world.finding(
        75,
        "f-fallback",
        name="gpu_temperature_c",
        value=30.0,
        severity="WARNING",
        source="CONFIGURED_FALLBACK",
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-003", "core_tlimit/warning")
    assert outcome.judgement.outcome == "FAIL"
    assert "also raised WARNING (CONFIGURED_FALLBACK)" in _errors(outcome)
    world = _warning_world()
    world.finding(
        90, "f-crit", name="gpu_tlimit_c", value=0.5, severity="CRITICAL", count=3
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-003", "core_tlimit/warning")
    assert "graded CRITICAL/DRAIN on a read-only signal" in _errors(outcome)


def test_node_isolated_for_a_read_only_signal_fails() -> None:
    world = _warning_world()
    world.isolate(100)
    outcome, _window = _run(world, "GF-ENV-LIVE-003", "core_tlimit/warning")
    assert outcome.judgement.outcome == "FAIL"
    assert "node was isolated for a read-only signal" in _errors(outcome)


def _transient_world(*, count: int = 1, clear: bool = True) -> World:
    world = World()
    world.accept(15, {"gpu_tlimit_c": -3.0, "gpu_temperature_c": 30.0})
    world.finding(
        15,
        "f-t",
        name="gpu_tlimit_c",
        value=-3.0,
        severity="WARNING" if count < 3 else "CRITICAL",
        count=count,
        source="NVIDIA_DEVICE_TLIMIT",
    )
    world.incident(15, "f-t", workflow="wf-diag")
    world.workflow(15, "wf-diag", "f-t", operations=DIAGNOSTIC_OPERATIONS)
    world.finish(60, "wf-diag")
    if clear:
        world.accept(
            30,
            {"gpu_tlimit_c": 58.0, "gpu_temperature_c": 30.0},
            reasons=("candidate-recovered",),
        )
        world.clear(30, "f-t")
    return world


def test_transient_that_clears_through_the_edge_filter_passes() -> None:
    outcome, _window = _run(
        _transient_world(), "GF-ENV-LIVE-005", "transient/once_then_clear"
    )
    assert outcome.judgement.outcome == "PASS", outcome.judgement.stages
    assert outcome.judgement.evidence["cleared_at"]
    notes = outcome.judgement.evidence["expectation"]["not_covered_by_this_attempt"]
    assert notes and "separate attempts" in notes[0]


def test_transient_that_never_clears_or_confirms_fails() -> None:
    outcome, _window = _run(
        _transient_world(clear=False), "GF-ENV-LIVE-005", "transient/once_then_clear"
    )
    assert outcome.judgement.outcome == "FAIL"
    assert "did not pass the edge filter" in _errors(outcome)
    outcome, _window = _run(
        _transient_world(count=2), "GF-ENV-LIVE-005", "transient/once_then_clear"
    )
    assert "the count reached 2 after 1 transient" in _errors(outcome)


def test_unusable_field_without_ingestion_is_inconclusive() -> None:
    outcome, _window = _run(World(), "GF-ENV-LIVE-008", "dcgm_blank")
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    reasons = "\n".join(outcome.judgement.inconclusive)
    assert "no delivery edge" in reasons and "health summary" in reasons


def test_unusable_field_reported_unknown_passes_after_the_hold() -> None:
    world = World()
    world.at(
        20,
        lambda w: w.collector_errors.append(
            f"invalid or ambiguous GPU observation: {GPU}/clock_throttle_reasons"
        ),
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-008", "non_integer_mask")
    assert outcome.judgement.outcome == "PASS", outcome.judgement.stages


def test_unusable_field_accepted_without_unknown_fails() -> None:
    # A health summary carried the omitted field's batch: the GPU's temperature
    # refreshed after the arm, its reason mask did not, and nothing reported
    # the field missing (the whole-batch completeness check of a multi-GPU node).
    world = World()
    world.accept(-120, {"gpu_temperature_c": 30.0, "clock_throttle_reasons": 0.0})
    world.at(
        25,
        lambda w: w.latest.__setitem__(
            "gpu_temperature_c",
            {
                **w.latest[f"{GPU}:gpu_temperature_c"],
                "observed_at": w.moment(25).isoformat(),
            },
        ),
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-008", "omit_reason")
    assert outcome.judgement.outcome == "FAIL"
    assert "nothing reported it UNKNOWN" in _errors(outcome)


def test_non_brake_bit_needs_its_accepted_reading_and_no_finding() -> None:
    world = World()
    world.at(
        20,
        lambda w: w.latest.__setitem__(
            "clock_throttle_reasons",
            {
                "gpu": GPU,
                "name": "clock_throttle_reasons",
                "value": 4.0,
                "observed_at": w.moment(20).isoformat(),
            },
        ),
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-007", "non_brake_0x4")
    assert outcome.judgement.outcome == "PASS", outcome.judgement.stages
    graded = World()
    graded.finding(
        20, "f-x", name="clock_throttle_reasons", value=4.0, severity="WARNING"
    )
    outcome, _window = _run(graded, "GF-ENV-LIVE-007", "non_brake_0x4")
    assert outcome.judgement.outcome == "FAIL"
    assert "ordinary bit 4.0 was graded WARNING" in _errors(outcome)


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-001: post-repoint evidence or nothing
# --------------------------------------------------------------------------- #
def _before() -> dict[str, Any]:
    return {
        "latest": [
            {"gpu": GPU, "name": name, "value": value}
            for name, value in baseline_values().items()
        ]
    }


def test_baseline_from_preflight_data_alone_is_inconclusive() -> None:
    outcome, _window = _run(World(), "GF-ENV-LIVE-001", "passthrough", before=_before())
    assert outcome.judgement.outcome == "INCONCLUSIVE"
    assert any(
        "no GPU_METRICS batch of the repointed collector" in item
        for item in outcome.judgement.inconclusive
    ), outcome.judgement.inconclusive


def test_baseline_passes_on_the_repointed_collectors_first_batch() -> None:
    world = World()
    world.accept(15, baseline_values(), reasons=("initial-baseline",))
    outcome, _window = _run(world, "GF-ENV-LIVE-001", "passthrough", before=_before())
    judgement = outcome.judgement
    assert judgement.outcome == "PASS", judgement.stages
    assert judgement.evidence["passthrough_reads"] >= 3
    assert "power_limit_w" in judgement.evidence["unchanged_static_fields"]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            baseline_values(power_limit_w=650.0),
            "the pass-through changed power_limit_w",
        ),
        (
            {k: v for k, v in baseline_values().items() if k != "power_usage_w"},
            "power_usage_w has neither a reading",
        ),
    ],
)
def test_baseline_with_a_changed_or_missing_reading_fails(
    values: dict[str, float], expected: str
) -> None:
    world = World()
    world.accept(15, values, reasons=("initial-baseline",))
    outcome, _window = _run(world, "GF-ENV-LIVE-001", "passthrough", before=_before())
    assert outcome.judgement.outcome == "FAIL"
    assert expected in _errors(outcome)


def test_baseline_completes_a_first_batch_without_limits_from_a_later_one() -> None:
    # The restarted collector's first batch can miss the cached temperature
    # limits; the next delivered batch carries them.
    world = World()
    first = {
        key: value
        for key, value in baseline_values().items()
        if not key.endswith("_tlimit_c") or key == "gpu_tlimit_c"
    }
    world.accept(15, first, reasons=("initial-baseline",))
    world.accept(
        40,
        {"gpu_slowdown_tlimit_c": -2.0, "gpu_shutdown_tlimit_c": -8.0},
        reasons=("counter-increased",),
    )
    outcome, _window = _run(world, "GF-ENV-LIVE-001", "passthrough", before=_before())
    assert outcome.judgement.outcome == "PASS", outcome.judgement.stages


def test_baseline_with_a_new_environment_incident_fails() -> None:
    world = World()
    world.accept(15, baseline_values(), reasons=("initial-baseline",))
    world.incident(20, "f-x")
    outcome, _window = _run(world, "GF-ENV-LIVE-001", "passthrough", before=_before())
    assert outcome.judgement.outcome == "FAIL"
    assert "created actions" in _errors(outcome)
