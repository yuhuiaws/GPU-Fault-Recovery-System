from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
TIMING = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_timing.py"
)


def test_rollback_timing_records_safe_and_full_rto() -> None:
    timing = TIMING.initialize_rollback_timing(
        {"updated_at_epoch": 100.0}, observed_at_epoch=110.0
    )
    TIMING.start_timed_entry(timing, "phases", "restore", observed_at_epoch=111.0)
    TIMING.complete_timed_entry(timing, "phases", "restore", observed_at_epoch=130.0)
    TIMING.mark_rollback_safe(timing, observed_at_epoch=130.0)
    TIMING.start_timed_entry(timing, "phases", "verify", observed_at_epoch=131.0)
    TIMING.complete_timed_entry(timing, "phases", "verify", observed_at_epoch=140.0)
    TIMING.mark_rollback_complete(timing, observed_at_epoch=140.0)

    assert timing["rollback_start_latency_seconds"] == 10.0
    assert timing["t_safe_seconds"] == 30.0
    assert timing["t_full_seconds"] == 40.0
    assert timing["phases"]["restore"]["duration_seconds"] == 19.0
    assert timing["phases"]["verify"]["duration_seconds"] == 9.0


def test_rollback_timing_preserves_resume_baseline() -> None:
    original = TIMING.initialize_rollback_timing(
        {"updated_at_epoch": 100.0}, observed_at_epoch=110.0
    )
    TIMING.mark_rollback_safe(original, observed_at_epoch=130.0)

    resumed = TIMING.initialize_rollback_timing(
        {"rollback_timing": original, "updated_at_epoch": 150.0},
        observed_at_epoch=160.0,
    )

    assert resumed["failure_detected_at_epoch"] == 100.0
    assert resumed["rollback_started_at_epoch"] == 110.0
    assert resumed["safe_at_epoch"] == 130.0


def test_rollback_wave_timing_records_convergence_milestones() -> None:
    release = SimpleNamespace()
    wave = ("node-a", "node-b")
    for event, observed in (
        ("safety_started", 100.0),
        ("safety_completed", 101.0),
        ("reconciler_applied", 103.0),
        ("agents_converged", 110.0),
        ("completed", 111.0),
    ):
        TIMING.record_rollback_wave_event(
            release,
            cluster_id="gpu-a",
            wave=wave,
            event=event,
            observed_at_epoch=observed,
        )
    timing = TIMING.initialize_rollback_timing(
        {"updated_at_epoch": 90.0}, observed_at_epoch=95.0
    )
    TIMING.merge_rollback_wave_timings(release, timing)

    recorded = timing["clusters"]["gpu-a"]["waves"]["node-a,node-b"]
    assert recorded["status"] == "COMPLETED"
    assert recorded["duration_seconds"] == 11.0
    assert recorded["reconciler_applied_at_epoch"] == 103.0
    assert recorded["agents_converged_at_epoch"] == 110.0
