"""DESTR-008 watchdog receipts stay certifiable under a slow CPU kubectl.

Live round 10, attempt 3: the observer's terminal QUIESCENT receipt was 36 s old
when the parent re-read it. About 22 sequential kubectl calls separated the close
request from the 30-s freshness check while parallel test suites saturated the
acceptance host's disk (1.6-5.8 s per call; ~1.2 s normally). The bound itself
(``MAX_RECEIPT_AGE``) is unchanged: these tests charge every kubectl call a fixed
latency and require that a fresh receipt is still certified, by the watchdog and
again by the fence close, while a receipt that really aged past the bound is
still refused. What either refusal leads to -- a cleanup observer, on whose
certified completion the fence closes -- is in ``test_destr008_fence_refusal.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pytest

from scripts.e2e.regional import destr008_parallel as parallel
from scripts.e2e.regional import destr008_watchdog_control as control_api
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.destr008_watchdog_control import MAX_RECEIPT_AGE
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_cancellation_controller import CpuApi, build_api
from tests.regional._destr008_causal import CausalHarness, build_causal
from tests.regional._destr008_main import MainHarness

T = TypeVar("T")
STALE = "fresh complete watchdog quiescence is unavailable"


def within(round_trips: int, seconds: float) -> float:
    # ``observed_at`` is truncated to a whole second, so an age can read up to
    # one second more than the time that really elapsed.
    return round_trips * seconds + 1


@dataclass
class SlowKubectl:
    """Charge each kubectl call ``seconds`` of the fake clock before it answers.

    The release-state read behind ``evidence_identity`` is a kubectl call live,
    so it is charged too. A ``destr008_parallel.gather`` batch runs its calls in
    order (the fakes stay single-threaded) but costs one round trip per
    ``workers`` calls, as the production pool of that size overlaps them.
    """

    clock: Any
    seconds: float
    workers: int = 8
    batch: int | None = None

    def charge(self) -> None:
        if self.batch is None or self.batch % self.workers == 0:
            self.clock.sleep(self.seconds)
        if self.batch is not None:
            self.batch += 1

    def wrap(self, call: Callable[..., T]) -> Callable[..., T]:
        def slow(*args: Any, **kwargs: Any) -> T:
            self.charge()
            return call(*args, **kwargs)

        return slow

    def gather(self, calls: Sequence[Callable[[], T]]) -> list[T]:
        outer, self.batch = self.batch, 0
        try:
            return [call() for call in calls]
        finally:
            self.batch = outer


def slow_kubectl(
    monkeypatch: pytest.MonkeyPatch, regional: Any, clock: Any, seconds: float
) -> SlowKubectl:
    slow = SlowKubectl(clock, seconds)
    monkeypatch.setattr(regional, "kubectl", slow.wrap(regional.kubectl))
    monkeypatch.setattr(
        regional, "evidence_identity", slow.wrap(regional.evidence_identity)
    )
    monkeypatch.setattr(parallel, "gather", slow.gather)
    return slow


def certifications(
    monkeypatch: pytest.MonkeyPatch, now: Callable[[], float]
) -> list[int]:
    """The receipt age, in seconds, at every successful freshness certification."""

    ages: list[int] = []
    certify = control_api.CancellationControl.quiescence

    def recorded(
        control: control_api.CancellationControl, *, timeout: float = 30
    ) -> wire.Receipt:
        receipt = certify(control, timeout=timeout)
        ages.append(int(now()) - receipt.observed_at)
        return receipt

    monkeypatch.setattr(control_api.CancellationControl, "quiescence", recorded)
    return ages


@pytest.fixture
def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CpuApi, wire.Plan, resources.CpuRuntime]:
    return build_api(tmp_path, monkeypatch)


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CausalHarness:
    return build_causal(tmp_path, monkeypatch)


def post(h: CausalHarness) -> None:
    h.safety.arm(
        h.observation(),
        window_seconds=360,
        maintenance_window_end=datetime.fromtimestamp(
            h.cpu.clock.now() + 3600, timezone.utc
        ),
    )
    h.safety.admit_fixture()
    h.acknowledge(h.safety.before_post())


@pytest.mark.parametrize("seconds", [2.5, 4.0])
def test_slow_kubectl_still_certifies_the_fresh_terminal_receipt(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    ages = certifications(monkeypatch, api.clock.now)
    slow_kubectl(monkeypatch, api.regional, api.clock, seconds)
    control.request_close()
    receipt = watchdog.wait_quiescence()
    assert receipt.state == "QUIESCENT" and not receipt.monitoring, receipt
    assert api.journal()["quiescence"] == receipt.model_dump(mode="json"), (
        "the certified receipt must be the recorded verdict"
    )
    # The fake observer writes its receipt on the first read after the close;
    # six round trips (runtime batch, control, Job, Pod, support batch,
    # certification) may follow it, where ~22 sequential reads did.
    assert len(ages) == 1 and ages[0] <= within(6, seconds), ages
    reads = [args[1:3] for verb, args, _ in api.calls[-6:] if verb == "get"]
    assert reads == [
        ("configmap", watchdog.name),
        ("serviceaccount", watchdog.name),
        ("role", watchdog.name),
        ("rolebinding", watchdog.name),
        ("configmap", watchdog.name + "-code"),
        ("configmap", watchdog.name),
    ], "the supporting resources must be verified before the certifying read"


def test_receipt_older_than_the_bound_is_still_refused(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    slow_kubectl(monkeypatch, api.regional, api.clock, 6)
    control.request_close()
    with pytest.raises(RegionalFixtureError, match=STALE):
        watchdog.wait_quiescence()
    saved = api.journal()
    last = saved["last_receipt"]
    assert last["state"] == "QUIESCENT" and not last["monitoring"], last
    assert int(api.clock.now()) - last["observed_at"] > MAX_RECEIPT_AGE, (
        "the refused receipt must really be older than the bound"
    )
    assert saved["quiescence"] is None, "a stale receipt must not become the verdict"


def test_runtime_batch_refuses_a_namespace_replaced_after_the_operation_check(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm().request_close()
    reads: list[str] = []
    kubectl = api.regional.kubectl

    def replaced(plane: str, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "namespace"):
            reads.append(args[2])
            if len(reads) == 2:
                namespace = api.objects["namespace", runtime.namespace]
                namespace["metadata"]["uid"] = "replacement-namespace"
        return kubectl(plane, *args, **kwargs)

    monkeypatch.setattr(api.regional, "kubectl", replaced)
    with pytest.raises(RegionalFixtureError, match="namespace was replaced"):
        watchdog.wait_quiescence()
    assert reads == [runtime.namespace, runtime.namespace], reads
    assert api.journal()["quiescence"] is None, "a replaced runtime has no verdict"


@pytest.mark.parametrize("seconds", [2.5, 4.0])
def test_slow_kubectl_close_certifies_twice_and_retires_the_fence(
    h: CausalHarness, monkeypatch: pytest.MonkeyPatch, seconds: float
) -> None:
    post(h)
    ages = certifications(monkeypatch, h.cpu.clock.now)
    slow_kubectl(monkeypatch, h.cpu.regional, h.cpu.clock, seconds)
    report = h.safety.finish()
    assert report["quiescent"] and report["retired"], report
    assert report["errors"] == [], report
    assert report["receipt"]["state"] == "QUIESCENT", report
    # The real probe needs QUIET_SECONDS of unchanged inventory; the watchdog
    # certifies four round trips after its receipt, the fence close (two
    # release-state reads, then its own control read) three after that.
    assert len(ages) == 2, ages
    assert ages[0] <= within(4, seconds) and ages[1] <= within(7, seconds), ages
    assert within(7, seconds) <= MAX_RECEIPT_AGE, "the model must stay in bound"
    assert h.gpu.objects == {} and h.cpu.journal()["closed"] is True, (
        "a fresh receipt must still retire the fence and the watchdog"
    )


def test_original_completion_failure_keeps_only_the_class_of_foreign_errors(
    h: CausalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post(h)
    certify = control_api.CancellationControl.quiescence
    failures: list[str] = []

    def foreign_once(
        control: control_api.CancellationControl, *, timeout: float = 30
    ) -> wire.Receipt:
        if not failures:
            failures.append("raised")
            raise ValueError("postgresql://private-diagnostic")
        return certify(control, timeout=timeout)

    monkeypatch.setattr(control_api.CancellationControl, "quiescence", foreign_once)
    report = h.safety.finish()
    assert report["quiescent"] and report["retired"], report
    assert report["errors"][0] == "original watchdog completion: ValueError", report
    assert "private-diagnostic" not in str(report), report


def test_scenario_errors_name_the_watchdog_check_that_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = MainHarness(tmp_path, monkeypatch, "active-gpu-pod")
    certify = control_api.CancellationControl.quiescence
    refused: list[str] = []

    def stale_once(
        control: control_api.CancellationControl, *, timeout: float = 30
    ) -> wire.Receipt:
        if not refused:
            refused.append("refused")
            raise RegionalFixtureError(STALE)
        return certify(control, timeout=timeout)

    monkeypatch.setattr(control_api.CancellationControl, "quiescence", stale_once)
    report = h.run()
    assert report["verdict"] == "FAIL", report
    proof = report["independent_safety_cleanup"]
    assert proof["quiescent"] and proof["retired"], proof
    assert report["errors"][-2:] == [
        "independent safety: original watchdog completion: RegionalFixtureError: "
        + STALE,
        "independent safety: independent watchdog rejected this case",
    ], report
