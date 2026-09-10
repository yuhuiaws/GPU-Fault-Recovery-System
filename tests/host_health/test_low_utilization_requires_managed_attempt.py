"""Low CPU/GPU utilization only counts while a real managed attempt owns the node.

``LOW_CPU_UTILIZATION`` and ``LOW_GPU_UTILIZATION`` were gated on the batch's
``workload_state``/``affected_workload_ids`` alone. Those two fields are the
node collector's static environment, not an observation, so every node whose
collector was configured ``ACTIVE`` kept the timer running while placeholder
jobs held it idle -- 58 ESCALATED incidents with ``job_id=None``. The gate now
requires ``NodeHealthPolicy._active_attempt`` to resolve exactly one
PENDING/RUNNING attempt with a live container on this node, the same resolver
the EFA-traffic rule already trusts, and the finding carries that attempt.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    host_telemetry_batch,
)
from tests.host_health._support import NOW, observe_running_attempt

GPUS = ("GPU-0", "GPU-1")
WINDOW = 300.0


def _low_utilization_batch(batch_id: str, offset: float, *, node_id: str = "node-a"):
    return host_telemetry_batch(
        batch_id,
        NOW + timedelta(seconds=offset),
        [
            HostMetricSample(name="cpu_usage_percent", value=1.0, unit="percent"),
            *(
                HostMetricSample(
                    name="host_gpu_utilization_percent",
                    value=0.0,
                    unit="percent",
                    device=device,
                )
                for device in GPUS
            ),
        ],
        node_id=node_id,
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
    )


def _run_low_utilization(policy: NodeHealthPolicy, *, seed=None):
    """Feed 400 s of near-zero utilization; ``seed`` refreshes observations."""

    emitted = []
    for offset in (0.0, 100.0, 200.0, 300.0, 400.0):
        if seed is not None:
            seed(NOW + timedelta(seconds=offset))
        emitted.extend(
            policy.evaluate_metrics(
                _low_utilization_batch(f"low-{int(offset)}", offset)
            )
        )
    return emitted


def _observe_other_attempt(store, observed_at) -> None:
    store.save_attempt_observation(
        attempt_observation(
            "job-b",
            "attempt-b",
            observed_at,
            workload_ids=["training/job/job-a"],
            containers=[container_observation("pod-b", "worker-b", 0, "node-a")],
        )
    )


def test_declared_active_state_without_an_attempt_observation_never_fires(
    caplog,
) -> None:
    assert NodeHealthPolicy(build_store()).low_utilization_duration_seconds == WINDOW
    policy = NodeHealthPolicy(build_store())

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.host_health"):
        assert _run_low_utilization(policy) == [], (
            "the collector's env-declared ACTIVE state alone kept the timer running"
        )

    gated = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and "managed attempt" in record.getMessage()
    ]
    assert len(gated) == 5, (
        "expected exactly one debug line per batch (five batches), got "
        f"{[record.getMessage() for record in gated]}"
    )


def test_running_attempt_on_the_node_fires_once_per_device_with_its_identity() -> None:
    store = build_store()
    policy = NodeHealthPolicy(store)

    emitted = _run_low_utilization(
        policy, seed=lambda observed_at: observe_running_attempt(store, observed_at)
    )

    assert all(
        item.observed_at >= NOW + timedelta(seconds=WINDOW) for item in emitted
    ), "a finding was minted before the sustain window elapsed"
    assert all(
        item.job_id == "job-a" and item.attempt_id == "attempt-a" for item in emitted
    ), emitted
    findings = [
        item for item in emitted if item.observed_at == NOW + timedelta(seconds=WINDOW)
    ]
    assert sorted((item.metric_name, item.device) for item in findings) == sorted(
        [
            ("cpu_usage_percent", None),
            ("host_gpu_utilization_percent", "GPU-0"),
            ("host_gpu_utilization_percent", "GPU-1"),
        ]
    ), findings
    gpu_findings = [item for item in findings if item.device is not None]
    assert len({item.event_id for item in gpu_findings}) == 2, (
        "per-GPU event ids collapsed"
    )
    assert all(
        item.event_id.endswith(f"-{item.device}-low_gpu_utilization")
        for item in gpu_findings
    ), gpu_findings


def test_attempt_running_on_another_node_does_not_count() -> None:
    store = build_store()
    policy = NodeHealthPolicy(store)

    findings = _run_low_utilization(
        policy,
        seed=lambda observed_at: observe_running_attempt(
            store, observed_at, ("node-b",)
        ),
    )

    assert findings == []


def test_two_attempts_on_the_node_are_ambiguous_and_never_fire(caplog) -> None:
    store = build_store()
    policy = NodeHealthPolicy(store)

    def seed(observed_at) -> None:
        observe_running_attempt(store, observed_at)
        _observe_other_attempt(store, observed_at)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.host_health"):
        findings = _run_low_utilization(policy, seed=seed)

    assert findings == []
    ambiguous = [
        record.getMessage()
        for record in caplog.records
        if "ambiguous" in record.getMessage()
    ]
    assert ambiguous, "ambiguous ownership must still be logged"
    assert all("EFA" not in message for message in ambiguous), ambiguous


def test_a_stale_attempt_observation_resets_the_window() -> None:
    """Observations older than the resolver's window stop counting."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    observe_running_attempt(store, NOW)

    assert _run_low_utilization(policy) == [], (
        "an observation from 400 s ago still counted as a running attempt"
    )


@pytest.mark.parametrize(
    ("metric_name", "value", "device"),
    [
        ("memory_available_percent", 2.0, None),
        ("page_cache_percent", 95.0, None),
        ("local_filesystem_used_percent", 92.0, "/"),
    ],
)
def test_rules_that_do_not_need_a_workload_are_unaffected(
    metric_name, value, device
) -> None:
    policy = NodeHealthPolicy(build_store())
    duration = {
        "memory_available_percent": policy.memory_pressure_duration_seconds,
        "page_cache_percent": policy.page_cache_duration_seconds,
        "local_filesystem_used_percent": policy.local_filesystem_duration_seconds,
    }[metric_name]

    def batch(offset: float):
        return host_telemetry_batch(
            f"{metric_name}-{int(offset)}",
            NOW + timedelta(seconds=offset),
            [
                HostMetricSample(
                    name=metric_name, value=value, unit="percent", device=device
                )
            ],
            workload_state="ACTIVE",
            affected_workload_ids=["training/job/job-a"],
        )

    assert policy.evaluate_metrics(batch(0)) == []
    findings = policy.evaluate_metrics(batch(duration))

    assert [item.metric_name for item in findings] == [metric_name], findings
    assert findings[0].job_id is None
    assert findings[0].attempt_id is None
