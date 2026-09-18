from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_safety as module
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_causal import CausalHarness, build_causal


def arm(harness: CausalHarness, *, window: int = 360) -> None:
    harness.safety.arm(
        harness.observation(),
        window_seconds=window,
        maintenance_window_end=datetime.fromtimestamp(
            harness.cpu.clock.now() + 3600, timezone.utc
        ),
    )


def test_fence_lifecycle_consumes_real_watchdog_and_scoped_store_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    h.safety.admit_fixture()
    claim = h.safety.before_post()
    h.acknowledge(claim)
    report = h.safety.finish()
    assert report["quiescent"] and report["retired"], report
    assert not report["errors"], report
    assert report["receipt"]["workflow_ids"] == ["workflow-a"], report
    assert report["receipt"]["source_complete"] is True, report
    assert not h.gpu.objects, h.gpu.objects
    assert h.cpu.journal()["closed"], h.cpu.journal()


def test_never_submitted_fixture_also_requires_real_quiet_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    h.safety.admit_fixture()
    before = h.cpu.clock.now()
    report = h.safety.finish()
    assert report["quiescent"] and not report["errors"], report
    assert h.cpu.clock.now() - before >= 5, report
    assert report["receipt"]["producer"]["state"] == "NOT_STARTED", report


def test_unknown_post_outcome_keeps_fence_after_controller_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    h.safety.admit_fixture()
    h.safety.before_post()
    fresh = module.ShortageSafety(
        h.cpu.regional,
        run_id=h.safety.run_id,
        attempt_id=h.safety.attempt_id,
        event_id=h.safety.event_id,
        fault_node=h.safety.fault_node,
        spare_node=h.safety.spare_node,
        spare_uid=h.gpu.binding.node_uid,
        release_id=h.safety.release_id,
        directory=h.safety.directory,
    )
    report = fresh.resume_cleanup()
    assert report["quiescent"] is False and report["errors"], report
    assert h.gpu.objects and not h.cpu.journal()["closed"], h.cpu.journal()
    control = fresh.watchdog.control_client() if fresh.watchdog is not None else None
    assert control is not None and control.read().control.revocation is not None
    with pytest.raises(RegionalFixtureError):
        control.claim()


def test_expiry_does_not_convert_original_failure_to_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    h.safety.admit_fixture()
    h.cpu.clock.sleep(361)
    report = h.safety.finish()
    assert report["quiescent"] and report["retired"], report
    assert report["receipt"]["revocation"]["reason"] == "DEADLINE", report
    assert report["errors"], "an expired drill cannot receive a passing verdict"


def test_cleanup_only_marks_original_case_failed_even_when_everything_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    h.safety.admit_fixture()
    claim = h.safety.before_post()
    h.acknowledge(claim)
    report = h.safety.resume_cleanup()
    assert report["quiescent"] and report["retired"], report
    assert report["receipt"]["case_failed"] and report["errors"], report


@pytest.mark.parametrize("resume", [False, True])
def test_completed_cleanup_revalidates_closure_without_recreating_observers(
    resume: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    h.safety.admit_fixture()
    assert h.safety.finish()["retired"], "the first cleanup must retire all resources"
    creations = sum(call[0] == "create" for call in h.cpu.calls)
    report = h.safety.resume_cleanup() if resume else h.safety.finish()
    assert report["quiescent"] and report["retired"] and not report["errors"], report
    assert sum(call[0] == "create" for call in h.cpu.calls) == creations, (
        "a closed attempt must never start a new observer or producer"
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda safety: safety.admit_fixture(),
        lambda safety: safety.require_bound(None, margin=60),
        lambda safety: safety.before_post(),
        lambda safety: safety.acknowledge("claim", {}),
        lambda safety: safety.remaining_seconds(),
    ],
)
def test_unarmed_controller_cannot_issue_or_certify_an_action(
    operation: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    with pytest.raises(RegionalFixtureError):
        operation(h.safety)
    assert not h.cpu.calls and not h.gpu.calls


def test_immutable_fixture_expiry_must_outlast_cancellation_and_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = build_causal(tmp_path, monkeypatch)
    arm(h)
    assert h.safety.plan is not None
    expiry = h.safety.plan.deadline_at
    with pytest.raises(RegionalFixtureError, match="precedes"):
        h.safety.require_bound(datetime.fromtimestamp(expiry, timezone.utc), margin=60)
    h.safety.require_bound(datetime.fromtimestamp(expiry + 60, timezone.utc), margin=60)
    assert 0 < h.safety.remaining_seconds() <= 360
    assert h.safety.finish()["retired"]
