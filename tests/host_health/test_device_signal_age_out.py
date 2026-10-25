"""A device-scoped health signal whose device stopped reporting does not stay
notified forever, and a new episode on the same key mints a new finding
(batch 160, live GF-GENERAL-LIVE-006 on plain EKS, 2026-10-25).

The timeline the product's own records showed: ``disk_io_await_ms`` on a test
dm-delay device went active and notified at 15:03Z, the device was torn down
with no below-threshold sample, the signal stayed active+notified for two
hours, and a new same-named dm-0 that crossed the threshold at 17:04Z minted
nothing because the key still carried the latch.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.host_device_signals import (
    DEVICE_AGE_OUT_SUMMARIES_ENV,
    DEVICE_STATE_GONE,
    DEVICE_STATE_LABEL,
    device_signal_age_out_seconds,
)
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.store.shared.health_signals import (
    attempt_signal_key,
    finding_health_signal_fingerprint,
    finding_health_signal_key,
)
from tests._builders import build_store, host_telemetry_batch
from tests.host_health._support import NOW

SUMMARY = 300.0
WINDOW = NodeHealthPolicy.DISK_IO_SUSTAIN_SECONDS
DM0 = "cluster-a/node-a/disk_io_await_ms/dm-0"
CPU = "cluster-a/node-a/cpu_usage_percent/node"


def dm0(value: float, **labels: str) -> HostMetricSample:
    return HostMetricSample(
        name="disk_io_await_ms", value=value, device="dm-0", labels=labels
    )


def cpu(value: float) -> HostMetricSample:
    return HostMetricSample(name="cpu_usage_percent", value=value)


class Site:
    """One node's host telemetry fed to the policy, second by second from NOW.

    Every batch carries the node-scoped CPU reading -- the node is alive and
    reporting -- and the control plane receives each batch at the node's time
    unless a test says otherwise.
    """

    def __init__(self, policy: NodeHealthPolicy) -> None:
        self.policy = policy
        self.store = policy.store
        self.counter = 0

    def batch(self, seconds: float, samples, *, received_at=None, **values):
        self.counter += 1
        observed_at = NOW + timedelta(seconds=seconds)
        if not any(sample.name == "cpu_usage_percent" for sample in samples):
            samples = [cpu(20.0), *samples]
        return host_telemetry_batch(
            f"host-{self.counter}",
            observed_at,
            samples,
            received_at=observed_at if received_at is None else received_at,
            **values,
        )

    def evaluate(self, seconds: float, samples=(), **values):
        return self.policy.evaluate_metrics(
            self.batch(seconds, list(samples), **values)
        )

    def deliver(self, findings, seconds: float) -> None:
        """What the ingest's finish half does once the incident committed."""

        for finding in findings:
            self.store.mark_health_signal_notified(
                finding_health_signal_key(finding),
                notified_at=NOW + timedelta(seconds=seconds),
                semantic_fingerprint=finding_health_signal_fingerprint(finding),
            )

    def first_episode(self) -> float:
        """dm-0 at 250 ms held for the sustain window: one finding, delivered."""

        assert self.evaluate(0, [dm0(250.0)]) == []
        findings = self.evaluate(WINDOW, [dm0(250.0)])
        assert [item.metric_name for item in findings] == ["disk_io_await_ms"]
        self.deliver(findings, WINDOW)
        state = self.store.get_health_signal_state(DM0)
        assert state is not None and state.active and state.notified is True
        return WINDOW


@pytest.fixture
def site(monkeypatch) -> Site:
    monkeypatch.delenv(DEVICE_AGE_OUT_SUMMARIES_ENV, raising=False)
    monkeypatch.delenv("GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS", raising=False)
    return Site(NodeHealthPolicy(build_store()))


def test_default_window_is_three_host_health_summaries(monkeypatch) -> None:
    monkeypatch.delenv(DEVICE_AGE_OUT_SUMMARIES_ENV, raising=False)
    monkeypatch.delenv("GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS", raising=False)
    assert device_signal_age_out_seconds() == 3 * SUMMARY
    assert device_signal_age_out_seconds({DEVICE_AGE_OUT_SUMMARIES_ENV: "0"}) == 0
    assert (
        device_signal_age_out_seconds(
            {
                DEVICE_AGE_OUT_SUMMARIES_ENV: "4",
                "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "120",
            }
        )
        == 480
    )
    with pytest.raises(ValueError, match="at least 2"):
        device_signal_age_out_seconds({DEVICE_AGE_OUT_SUMMARIES_ENV: "1"})
    with pytest.raises(ValueError, match="positive"):
        device_signal_age_out_seconds({"GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "0"})


def test_live_timeline_vanished_device_then_same_name_new_episode_mints_a_finding(
    site: Site,
) -> None:
    """15:03Z notified; device torn down, summaries without dm-0 for two
    hours; 17:04Z a new dm-0 crosses again -> a new finding (was: silent)."""

    at = site.first_episode()
    # Health summaries without any dm-0 sample: the node reports, the device
    # does not. Nothing is cleared before the window has run on both clocks.
    for summaries in (1, 2):
        at += SUMMARY
        assert site.evaluate(at) == []
        state = site.store.get_health_signal_state(DM0)
        assert state is not None and state.active, summaries
    at += SUMMARY  # the third summary without the device: aged out
    assert site.evaluate(at) == []
    aged = site.store.get_health_signal_state(DM0)
    assert aged is not None
    assert aged.active is False and aged.notified is False, aged
    assert aged.observed_at == NOW + timedelta(seconds=at), (
        "the cleared state records when the signal was aged out"
    )
    assert site.policy.device_signal_age_outs_total == 1

    # Two hours later a new dm-0 crosses the threshold: a new episode that
    # needs the full sustain window again, then one finding.
    at = 2 * 3600.0
    assert site.evaluate(at, [dm0(300.0)]) == []
    findings = site.evaluate(at + WINDOW, [dm0(300.0)])
    assert [item.metric_name for item in findings] == ["disk_io_await_ms"]
    assert findings[0].device == "dm-0"
    assert findings[0].value == 300.0


def test_a_device_that_keeps_reporting_at_threshold_is_notified_once(
    site: Site,
) -> None:
    at = site.first_episode()
    for _ in range(24):  # two hours of health summaries, dm-0 still slow
        at += SUMMARY
        assert site.evaluate(at, [dm0(250.0)]) == []
    state = site.store.get_health_signal_state(DM0)
    assert state is not None and state.active and state.notified is True
    assert site.policy.device_signal_age_outs_total == 0


def test_node_scoped_and_attempt_scoped_signals_are_never_aged_out(site: Site) -> None:
    # A node-scoped rule held active and notified ...
    assert site.evaluate(0, [cpu(99.0)]) == []
    findings = site.evaluate(120, [cpu(99.0)])
    assert [item.metric_name for item in findings] == ["cpu_usage_percent"]
    site.deliver(findings, 120)
    # ... and an attempt-scoped low-utilisation window, both silent afterwards.
    gpu_window = attempt_signal_key(
        "cluster-a/node-a/host_gpu_utilization_percent/GPU-1/LOW_GPU_UTILIZATION",
        "job-a",
        "attempt-1",
    )
    site.store.claim_health_signal_transitions(
        [(gpu_window, True, NOW + timedelta(seconds=120), 0.0)],
        received_at=NOW + timedelta(seconds=120),
    )
    at = 120.0
    for _ in range(6):
        at += SUMMARY
        # The node keeps reporting (a healthy disk reading) without the CPU
        # metric and without the GPU window's sample.
        batch = host_telemetry_batch(
            f"quiet-{at}",
            NOW + timedelta(seconds=at),
            [dm0(1.0)],
            received_at=NOW + timedelta(seconds=at),
        )
        assert site.policy.evaluate_metrics(batch) == []
    cpu_state = site.store.get_health_signal_state(CPU)
    assert cpu_state is not None and cpu_state.active and cpu_state.notified is True
    window_state = site.store.get_health_signal_state(gpu_window)
    assert window_state is not None and window_state.active
    assert site.policy.device_signal_age_outs_total == 0


def test_a_backlog_delivered_after_an_outage_ages_nothing_out(site: Site) -> None:
    """Only the receive clock jumped: the node's clock says the device was
    missing for seconds, so the signal stays (and so does its latch)."""

    at = site.first_episode()
    late = NOW + timedelta(hours=1)
    for tick in (15, 30, 45):
        assert (
            site.evaluate(at + tick, received_at=late + timedelta(seconds=tick)) == []
        )
    state = site.store.get_health_signal_state(DM0)
    assert state is not None and state.active and state.notified is True
    assert site.policy.device_signal_age_outs_total == 0


def test_only_the_nodes_own_collector_ages_its_devices_out(site: Site) -> None:
    at = site.first_episode()
    assert site.evaluate(at + 4 * SUMMARY, producer="control-plane") == []
    state = site.store.get_health_signal_state(DM0)
    assert state is not None and state.active and state.notified is True


def test_age_out_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setenv(DEVICE_AGE_OUT_SUMMARIES_ENV, "0")
    site = Site(NodeHealthPolicy(build_store()))
    at = site.first_episode()
    for _ in range(6):
        at += SUMMARY
        assert site.evaluate(at) == []
    state = site.store.get_health_signal_state(DM0)
    assert state is not None and state.active and state.notified is True


def test_a_gone_sample_clears_the_device_signal_and_is_not_a_reading(
    site: Site,
) -> None:
    at = site.first_episode()
    gone = dm0(0.0, **{DEVICE_STATE_LABEL: DEVICE_STATE_GONE})
    assert site.evaluate(at + 15, [gone]) == []
    cleared = site.store.get_health_signal_state(DM0)
    assert cleared is not None
    assert cleared.active is False and cleared.notified is False
    latest = {
        (item.name, item.device): item.value
        for item in site.store.list_telemetry_metrics_latest("cluster-a", "node-a")
    }
    assert latest[("disk_io_await_ms", "dm-0")] == 250.0, (
        "the GONE sample's 0 must not become the device's latest reading"
    )
    # The same name back on the node and slow again: a new episode.
    assert site.evaluate(at + 600, [dm0(300.0)]) == []
    findings = site.evaluate(at + 600 + WINDOW, [dm0(300.0)])
    assert [item.device for item in findings] == ["dm-0"]


def test_a_gone_sample_for_an_unknown_device_writes_nothing(site: Site) -> None:
    gone = dm0(0.0, **{DEVICE_STATE_LABEL: DEVICE_STATE_GONE})
    assert site.evaluate(0, [gone]) == []
    assert site.store.get_health_signal_state(DM0) is None
