"""A refusal at the DESTR-008 fence close recovers like one at the watchdog.

``ShortageSafety.finish`` certifies the original observer's terminal receipt
twice: the watchdog once, then the fence close on its own three round trips
later. A watchdog refusal already fell back to a fresh cleanup observer, but a
fence refusal -- b150's model: ~5.5 s per kubectl call, where the watchdog
certifies and the fence refuses -- kept the fence and deferred every cleanup
until an operator acted. Both refusals now start a cleanup observer and the fence
closes on its certified completion
(``CancellationControl.cleanup_completion``): the case still fails, and a
cleanup observer that cannot complete keeps the fence.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_watchdog_control as control_api
from scripts.e2e.regional.destr008_admission import RESOURCES
from scripts.e2e.regional.destr008_watchdog_control import MAX_RECEIPT_AGE
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_causal import CausalHarness, build_causal
from tests.regional._destr008_main import MainHarness
from tests.regional.test_destr008_receipt_freshness import (
    STALE,
    certifications,
    post,
    slow_kubectl,
)

STALLED = "watchdog terminal quiescence deadline expired"


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CausalHarness:
    return build_causal(tmp_path, monkeypatch)


def cleanup_reads(
    monkeypatch: pytest.MonkeyPatch, now: Callable[[], float]
) -> list[int]:
    """The age of a cleanup observer's terminal receipt at every control read.

    The watchdog's certifying read comes first, the fence's own read last.
    """

    ages: list[int] = []
    read = control_api.CancellationControl.read

    def recorded(
        control: control_api.CancellationControl, *, timeout: float = 30
    ) -> control_api.ControlSnapshot:
        snapshot = read(control, timeout=timeout)
        receipt = snapshot.receipt
        if (
            receipt is not None
            and receipt.cleanup is not None
            and receipt.state == "QUIESCENT"
        ):
            ages.append(int(now()) - receipt.observed_at)
        return snapshot

    monkeypatch.setattr(control_api.CancellationControl, "read", recorded)
    return ages


def refusing(
    monkeypatch: pytest.MonkeyPatch, call: int, then: Callable[[], None] = lambda: None
) -> None:
    """Refuse the ``call``-th freshness certification as stale.

    The first is the watchdog's own of the original receipt, the second the
    fence close's; ``then`` runs just before the refusal.
    """

    certify = control_api.CancellationControl.quiescence
    calls: list[int] = []

    def counted(
        control: control_api.CancellationControl, *, timeout: float = 30
    ) -> wire.Receipt:
        calls.append(len(calls) + 1)
        if len(calls) == call:
            then()
            raise RegionalFixtureError(STALE)
        return certify(control, timeout=timeout)

    monkeypatch.setattr(control_api.CancellationControl, "quiescence", counted)


def fence_journal(h: CausalHarness) -> dict[str, object]:
    value = locking.read_private_document(h.safety.fence.path)
    assert isinstance(value, dict), "the fence journal must be an object"
    return value


@pytest.mark.parametrize(
    "seconds,refused,certified",
    [(5.5, "original fence close", 2), (7.0, "original watchdog completion", 1)],
    ids=["fence-refuses", "watchdog-refuses"],
)
def test_a_refusal_closes_the_fence_on_the_cleanup_observer_s_completion(
    h: CausalHarness,
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
    refused: str,
    certified: int,
) -> None:
    post(h)
    ages = certifications(monkeypatch, h.cpu.clock.now)
    reads = cleanup_reads(monkeypatch, h.cpu.clock.now)
    slow_kubectl(monkeypatch, h.cpu.regional, h.cpu.clock, seconds)
    report = h.safety.finish()
    # At 5.5 s per call the watchdog certifies the original receipt and the
    # fence's own read three round trips later refuses it; at 7 s the watchdog
    # refuses it already. Either way a fresh cleanup observer replaces the proof.
    assert report["errors"] == [
        f"{refused}: RegionalFixtureError: {STALE}",
        "independent watchdog rejected this case",
    ], report
    assert report["quiescent"] is True and report["retired"] is True, report
    receipt = report["receipt"]
    assert receipt["cleanup"] is not None and receipt["case_failed"] is True, receipt
    assert len(ages) == certified and max(ages) <= MAX_RECEIPT_AGE, ages
    # The fence re-read the receipt the watchdog had certified fresh, past the
    # freshness bound: a plain retry of the fresh check would refuse it again.
    assert len(reads) == 2 and reads[0] <= MAX_RECEIPT_AGE < reads[1], reads
    saved = fence_journal(h)
    assert saved["phase"] == "CLOSED" and saved["retirement_receipt"] == receipt, (
        "the fence must retire on the cleanup observer's certified receipt"
    )
    assert h.gpu.objects == {} and h.cpu.journal()["closed"] is True, (
        "the fence and the watchdog's resources must be cleaned"
    )


@pytest.mark.parametrize(
    "seconds,refused,cause",
    [
        (5.5, "original fence close", STALLED),
        (8.0, "original watchdog completion", STALE),
    ],
    ids=["fence-refuses-observer-stalls", "watchdog-refuses-cleanup-receipt"],
)
def test_a_cleanup_observer_that_cannot_complete_keeps_the_fence(
    h: CausalHarness,
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
    refused: str,
    cause: str,
) -> None:
    post(h)
    fence = set(h.gpu.objects)
    certify = control_api.CancellationControl.quiescence

    def certify_or_stall(
        control: control_api.CancellationControl, *, timeout: float = 30
    ) -> wire.Receipt:
        try:
            return certify(control, timeout=timeout)
        except RegionalFixtureError:
            if cause == STALLED:
                h.observer_enabled = False  # the cleanup observer never reports
            raise

    monkeypatch.setattr(control_api.CancellationControl, "quiescence", certify_or_stall)
    slow_kubectl(monkeypatch, h.cpu.regional, h.cpu.clock, seconds)
    report = h.safety.finish()
    assert report["errors"] == [
        f"{refused}: RegionalFixtureError: {STALE}",
        f"cleanup observer: RegionalFixtureError: {cause}",
    ], "the evidence must name the refusal and the failed cleanup observer"
    assert report["quiescent"] is False and report["retired"] is False, report
    assert "receipt" not in report, report
    assert set(h.gpu.objects) == fence == set(RESOURCES), (
        "an unproven cleanup must keep the fence"
    )
    assert fence_journal(h)["phase"] == "ARMED", "the fence must not start closing"
    journal = h.cpu.journal()
    assert journal["jobs"][-1]["cleanup_id"] is not None, (
        "a cleanup observer must replace the refused proof"
    )
    assert journal["quiescence"] is None and journal["closed"] is False, journal


@pytest.mark.parametrize(
    "call,refused",
    [(1, "original watchdog completion"), (2, "original fence close")],
    ids=["watchdog", "fence"],
)
def test_either_refusal_fails_the_scenario_and_still_cleans_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: int, refused: str
) -> None:
    h = MainHarness(tmp_path, monkeypatch, "active-gpu-pod")
    refusing(monkeypatch, call)
    report = h.run()
    assert report["verdict"] == "FAIL", report
    assert report["errors"][-2:] == [
        f"independent safety: {refused}: RegionalFixtureError: {STALE}",
        "independent safety: independent watchdog rejected this case",
    ], report
    proof = report["independent_safety_cleanup"]
    assert proof["quiescent"] is True and proof["retired"] is True, proof
    assert proof["receipt"]["cleanup"] is not None, proof
    # cleanup_complete is what lets the matrix record the scenario as cleaned
    # and stop on its FAIL, exactly as after a watchdog refusal.
    assert report["cleanup_complete"] is True, report
    assert "cleanup_deferred" not in report, report
    names = [name for name, _ in h.warm.calls]
    assert "holder.cleanup" in names, names
    assert not h.causal.gpu.objects and h.causal.cpu.journal()["closed"], (
        "the fence must be gone and the watchdog retired"
    )


def test_a_fence_refusal_whose_cleanup_observer_fails_defers_the_scenario_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, "active-gpu-pod")

    def stall() -> None:
        # The loaded host that aged the receipt keeps the cleanup observer from
        # ever reporting; slow calls also keep its bounded wait to a few passes.
        h.causal.observer_enabled = False
        slow_kubectl(monkeypatch, h.causal.cpu.regional, h.causal.cpu.clock, 5.5)

    refusing(monkeypatch, 2, then=stall)
    report = h.run()
    assert report["verdict"] == "FAIL" and report["cleanup_deferred"] is True, report
    assert report["errors"][-2:] == [
        f"independent safety: original fence close: RegionalFixtureError: {STALE}",
        f"independent safety: cleanup observer: RegionalFixtureError: {STALLED}",
    ], report
    proof = report["independent_safety_cleanup"]
    assert proof["quiescent"] is False and proof["retired"] is False, proof
    assert report["cleanup_complete"] is False, report
    assert set(h.causal.gpu.objects) == set(RESOURCES), (
        "an unproven cleanup must keep the fence"
    )
    names = {name for name, _ in h.warm.calls}
    assert not names & {"workload.delete", "holder.cleanup", "restore.create"}, names
