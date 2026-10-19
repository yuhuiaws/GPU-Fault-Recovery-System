"""A low-utilisation sustain window belongs to one managed attempt.

Live round 10: COLLECT-017's managed EFA-plugin job idled a HyperPod node's
GPUs for about four minutes, E2E-002's managed job replaced it on the same node,
and ``LOW_GPU_UTILIZATION`` opened an incident for E2E-002 (STOP_WORKLOADS +
RESTART_WORKLOAD) about one minute after it started. The window was keyed by
cluster/node/metric/device/rule only, so the new attempt inherited the old
attempt's low time -- and a delivered finding of the old attempt would have
silenced the new one for as long as the reading stayed low.

``LOW_CPU_UTILIZATION`` and ``LOW_GPU_UTILIZATION`` (the two
``active_workload_only`` rules) now time their window per resolved managed
attempt; a sample with no single managed attempt still resets it. The EFA
traffic rule needs no change: its state is keyed by job and attempt already.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.store.shared.health_signals import (
    finding_health_signal_fingerprint,
    finding_health_signal_key,
)
from gpu_fault.watcher import WorkloadPhase
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    host_telemetry_batch,
)
from tests.host_health._support import NOW

WINDOW = 300
NODE = "node-a"
WORKLOAD = "training/job/shared"
GPUS = ("GPU-0", "GPU-1")
A = ("job-a", "attempt-a")
B = ("job-b", "attempt-b")
CPU = ("cpu_usage_percent", None)
GPU0 = ("host_gpu_utilization_percent", "GPU-0")
GPU1 = ("host_gpu_utilization_percent", "GPU-1")


def _observe(
    store,
    attempt,
    offset,
    *,
    phase=WorkloadPhase.RUNNING,
    exit_code=None,
    restart_count=0,
):
    job_id, attempt_id = attempt
    store.save_attempt_observation(
        attempt_observation(
            job_id,
            attempt_id,
            NOW + timedelta(seconds=offset),
            workload_phase=phase,
            workload_ids=[WORKLOAD],
            containers=[
                container_observation(
                    f"pod-{attempt_id}",
                    f"worker-{attempt_id}",
                    0,
                    NODE,
                    gpu_uuids=list(GPUS),
                    terminated=exit_code is not None,
                    exit_code=exit_code,
                    restart_count=restart_count,
                )
            ],
        )
    )


def _end(store, attempt, offset):
    _observe(store, attempt, offset, phase=WorkloadPhase.SUCCEEDED, exit_code=0)


def _sample(
    policy,
    store,
    offset,
    attempt,
    *,
    gpu=(0.0, 0.0),
    cpu=1.0,
    workload_state="ACTIVE",
    deliver=False,
):
    """One host batch at ``offset``; ``attempt`` refreshes its observation.

    ``deliver`` plays the finish half of the ingest: it latches every emitted
    finding under the key the finding names, as ``_finish_host_telemetry``
    does after the incident committed.
    """

    if attempt is not None:
        _observe(store, attempt, offset)
    observed_at = NOW + timedelta(seconds=offset)
    findings = policy.evaluate_metrics(
        host_telemetry_batch(
            f"low-{offset}",
            observed_at,
            [
                HostMetricSample(name="cpu_usage_percent", value=cpu, unit="percent"),
                *(
                    HostMetricSample(
                        name="host_gpu_utilization_percent",
                        value=value,
                        unit="percent",
                        device=device,
                    )
                    for device, value in zip(GPUS, gpu, strict=True)
                ),
            ],
            node_id=NODE,
            workload_state=workload_state,
            affected_workload_ids=[WORKLOAD],
            received_at=observed_at,
        )
    )
    if deliver:
        for finding in findings:
            store.mark_health_signal_notified(
                finding_health_signal_key(finding),
                notified_at=observed_at,
                semantic_fingerprint=finding_health_signal_fingerprint(finding),
            )
    return findings


def _owners(findings):
    return sorted(
        (item.metric_name, item.device, (item.job_id, item.attempt_id))
        for item in findings
    )


def _first_fire(fired):
    return min((offset for offset, findings in fired.items() if findings), default=None)


def test_a_replacement_attempt_does_not_inherit_the_previous_attempts_low_time():
    """The live sequence: A idles the node for 240 s, B replaces it."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    fired = {}
    for offset in range(0, 241, 60):
        fired[offset] = _sample(policy, store, offset, A)
    _end(store, A, 260)
    for offset in range(270, 631, 60):
        fired[offset] = _sample(policy, store, offset, B)

    first = _first_fire(fired)
    assert first == 270 + WINDOW, (
        f"first low-utilisation finding at {first} s; attempt B started at 270 s "
        f"and must be low for {WINDOW} s itself: {_owners(fired.get(first, []))}"
    )
    assert _owners(fired[first]) == [(*CPU, B), (*GPU0, B), (*GPU1, B)]


def test_the_first_sample_of_a_replacement_attempt_does_not_fire_on_the_old_window():
    store = build_store()
    policy = NodeHealthPolicy(store)
    for offset in (0, 100, 200):
        assert _sample(policy, store, offset, A) == []
    assert _owners(_sample(policy, store, 300, A)) == [
        (*CPU, A),
        (*GPU0, A),
        (*GPU1, A),
    ]
    _end(store, A, 310)

    findings = _sample(policy, store, 320, B)

    assert findings == [], (
        "attempt B's first sample inherited attempt A's elapsed window: "
        f"{_owners(findings)}"
    )


def test_a_delivered_finding_for_one_attempt_does_not_silence_the_next():
    store = build_store()
    policy = NodeHealthPolicy(store)
    fired = {}
    for offset in range(0, 401, 100):
        fired[offset] = _sample(policy, store, offset, A, deliver=True)
    assert _first_fire(fired) == WINDOW
    assert fired[400] == [], "a delivered finding repeated for the same attempt"
    _end(store, A, 410)
    fired = {}
    for offset in range(420, 781, 60):
        fired[offset] = _sample(policy, store, offset, B, deliver=True)

    assert _first_fire(fired) == 420 + WINDOW, (
        "attempt A's latch silenced attempt B or B fired early: "
        f"{ {offset: _owners(items) for offset, items in fired.items() if items} }"
    )
    assert _owners(fired[420 + WINDOW]) == [(*CPU, B), (*GPU0, B), (*GPU1, B)]
    assert fired[780] == [], "a delivered finding repeated for attempt B"


def test_the_same_attempt_keeps_accumulating_across_batches():
    store = build_store()
    policy = NodeHealthPolicy(store)
    fired = {offset: _sample(policy, store, offset, A) for offset in range(0, 301, 30)}

    assert _first_fire(fired) == WINDOW
    assert _owners(fired[WINDOW]) == [(*CPU, A), (*GPU0, A), (*GPU1, A)]


def _collector_idle(policy, store, offset):
    return _sample(policy, store, offset, None, workload_state="IDLE")


def _container_restarting(policy, store, offset):
    _observe(store, A, offset, exit_code=137)
    findings = _sample(policy, store, offset, None)
    _observe(store, A, offset + 1, restart_count=1)
    return findings


def _two_attempts_on_the_node(policy, store, offset):
    _observe(store, B, offset)
    findings = _sample(policy, store, offset, A)
    _end(store, B, offset + 1)
    return findings


@pytest.mark.parametrize(
    "gap",
    [_collector_idle, _container_restarting, _two_attempts_on_the_node],
    ids=["collector-idle", "container-restarting", "ambiguous-ownership"],
)
def test_a_sample_without_one_managed_attempt_still_resets_the_window(gap):
    store = build_store()
    policy = NodeHealthPolicy(store)
    fired = {offset: _sample(policy, store, offset, A) for offset in (0, 100, 200)}
    fired[250] = gap(policy, store, 250)
    for offset in range(300, 601, 100):
        fired[offset] = _sample(policy, store, offset, A)

    assert _first_fire(fired) == 300 + WINDOW, (
        "the window did not restart after the sample without a managed attempt: "
        f"{ {offset: _owners(items) for offset, items in fired.items() if items} }"
    )
    assert _owners(fired[600]) == [(*CPU, A), (*GPU0, A), (*GPU1, A)]


def test_each_gpu_keeps_its_own_window_inside_each_attempt():
    store = build_store()
    policy = NodeHealthPolicy(store)
    fired = {}
    for offset in range(0, 421, 60):
        gpu1 = 90.0 if offset < 120 else 0.0
        fired[offset] = _sample(policy, store, offset, A, gpu=(0.0, gpu1), cpu=50.0)
    _end(store, A, 440)
    for offset in range(480, 781, 60):
        fired[offset] = _sample(policy, store, offset, B, cpu=50.0)

    first = {}
    for offset, findings in sorted(fired.items()):
        for metric_name, device, owner in _owners(findings):
            first.setdefault((device, owner), offset)
            assert metric_name == "host_gpu_utilization_percent", findings
    assert first == {
        ("GPU-0", A): WINDOW,
        ("GPU-1", A): 120 + WINDOW,
        ("GPU-0", B): 480 + WINDOW,
        ("GPU-1", B): 480 + WINDOW,
    }


def test_the_latch_key_of_a_low_utilisation_finding_names_its_attempt():
    store = build_store()
    policy = NodeHealthPolicy(store)
    for offset in (0, 100, 200):
        _sample(policy, store, offset, A)
    findings = _sample(policy, store, 300, A)

    assert len(findings) == 3, findings
    for finding in findings:
        key = finding_health_signal_key(finding)
        assert key.endswith("/job-a/attempt-a"), key
        state = store.get_health_signal_state(key)
        assert state is not None, f"nothing was claimed under the latch key {key}"
        assert state.active is True
        assert state.active_since == NOW, state


def test_efa_traffic_state_is_already_scoped_to_the_attempt(monkeypatch):
    """The third workload-gated rule: B starts from WARMUP, not A's zero clock."""

    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS", "0")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "30")
    store = build_store()
    policy = NodeHealthPolicy(store)

    def efa(offset, attempt, value):
        _observe(store, attempt, offset)
        findings = policy.evaluate_metrics(
            host_telemetry_batch(
                f"efa-{offset}",
                NOW + timedelta(seconds=offset),
                [
                    HostMetricSample(
                        name="efa_traffic_bytes_per_second",
                        value=value,
                        unit="bytes/second",
                    )
                ],
                node_id=NODE,
                workload_state="ACTIVE",
                affected_workload_ids=[WORKLOAD],
            )
        )
        return [
            (item.diagnostic_parameters["diagnostic_reason"], item.attempt_id)
            for item in findings
        ]

    assert efa(0, A, 1000) == []
    assert efa(15, A, 0) == []
    assert efa(30, A, 0) == [("EFA_TRAFFIC_ZERO_WARNING", "attempt-a")]
    _end(store, A, 35)

    assert efa(40, B, 0) == [], "attempt B inherited attempt A's zero-traffic clock"
    assert efa(50, B, 1000) == []
    assert efa(60, B, 0) == []
    assert efa(75, B, 0) == [("EFA_TRAFFIC_ZERO_WARNING", "attempt-b")]
