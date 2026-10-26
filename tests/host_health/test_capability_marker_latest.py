"""The control plane keeps every capability (NOT_SUPPORTED) marker as a latest record.

``NodeHealthPolicy.evaluate_metrics`` upserts the ``LATEST_METRICS_REQUIRED``
samples of every delivered host batch into ``telemetry_metric_latest``; the
markers (``host_capability_markers.CAPABILITY_MARKERS``) are among them since
GF-GENERAL-LIVE-015 ``unsupported_bmc`` (2026-10-26) read a node without
``ipmitool`` for a whole summary period and found no record at all: the steady
marker was no edge, and the summary that carried it is not evidence. The record
keeps the ``reason`` label and the device, is one per device, and is never a
rule input: no finding, and VALIDATE_HOST reads past it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.host_diagnostics import validate_host_metrics
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from tests._builders import build_store, host_telemetry_batch

T0 = datetime(2026, 10, 26, 4, 24, tzinfo=timezone.utc)


def reading(name: str, value: float, device: str | None = None) -> HostMetricSample:
    return HostMetricSample(name=name, value=value, device=device)


def marker(name: str, reason: str, device: str | None = None, **labels: str):
    return HostMetricSample(
        name=name, value=1.0, device=device, labels={"reason": reason, **labels}
    )


HOST_READINGS = [
    reading("cpu_usage_percent", 12.0),
    reading("load1_per_cpu", 0.1),
    reading("memory_used_percent", 20.0),
    reading("filesystem_used_percent", 30.0, "/"),
]


def latest(policy: NodeHealthPolicy) -> dict[tuple[str, str | None], tuple]:
    return {
        (item.name, item.device): (item.labels, item.observed_at)
        for item in policy.store.list_telemetry_metrics_latest("cluster-a", "node-a")
        if item.name.endswith("_unsupported")
    }


def test_markers_are_kept_per_device_with_their_reason_and_mint_nothing() -> None:
    policy = NodeHealthPolicy(build_store())
    batch = host_telemetry_batch(
        "host-1",
        T0,
        [
            *HOST_READINGS,
            marker("bmc_status_unsupported", "NO_DEVICE"),
            marker("smart_status_unsupported", "NO_SMART_STATUS", "/dev/nvme1"),
            marker(
                "efa_ethtool_counters_unsupported",
                "MISSING_COUNTER",
                "eth0",
                metrics="efa_rnr_errors_delta",
            ),
        ],
        edge_filter_reasons=["health-summary"],
    )

    assert policy.evaluate_metrics(batch) == [], "NOT_SUPPORTED never mints a finding"
    assert latest(policy) == {
        ("bmc_status_unsupported", None): ({"reason": "NO_DEVICE"}, T0),
        ("smart_status_unsupported", "/dev/nvme1"): ({"reason": "NO_SMART_STATUS"}, T0),
        ("efa_ethtool_counters_unsupported", "eth0"): (
            {"reason": "MISSING_COUNTER", "metrics": "efa_rnr_errors_delta"},
            T0,
        ),
    }
    rows = policy.store.list_telemetry_metrics_latest("cluster-a", "node-a")
    failures: dict[str, list[str]] = {}
    assert validate_host_metrics(rows, "node-a", failures) is None
    assert failures == {}, "a capability gap is not an unhealthy host metric"


def test_a_cleared_marker_keeps_its_last_record_while_a_steady_one_moves() -> None:
    """A reader tells a current gap from a closed one by the record's time."""

    policy = NodeHealthPolicy(build_store())
    first = host_telemetry_batch(
        "host-1",
        T0,
        [
            *HOST_READINGS,
            marker("bmc_status_unsupported", "TOOL_MISSING"),
            marker("smart_status_unsupported", "NO_DEVICES"),
        ],
        edge_filter_reasons=["baseline"],
    )
    later = T0 + timedelta(seconds=300)
    second = host_telemetry_batch(
        "host-2",
        later,
        [
            *HOST_READINGS,
            marker("bmc_status_unsupported", "TOOL_MISSING"),
            reading("smart_health_failed", 0.0, "/dev/nvme0"),
        ],
        edge_filter_reasons=["health-summary"],
    )

    assert policy.evaluate_metrics(first) == []
    assert policy.evaluate_metrics(second) == []
    assert latest(policy) == {
        ("bmc_status_unsupported", None): ({"reason": "TOOL_MISSING"}, later),
        ("smart_status_unsupported", None): ({"reason": "NO_DEVICES"}, T0),
    }, "the steady BMC gap is current; the SMART gap closed when a drive appeared"
