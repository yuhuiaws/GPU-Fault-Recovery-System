"""A port rate requires a complete, comparable pair of traffic counters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from gpu_fault.collectors.host.observation import read_counter
from gpu_fault.host_health import HostMetricSample

TRAFFIC_COUNTERS = {
    "rdma_read_wrs": ("efa_rdma_read_ops_delta", "operations"),
    "rdma_read_bytes": ("efa_rdma_read_bytes_delta", "bytes"),
    "rdma_write_wrs": ("efa_rdma_write_ops_delta", "operations"),
    "rdma_write_bytes": ("efa_rdma_write_bytes_delta", "bytes"),
    "rdma_write_recv_bytes": ("efa_rdma_write_recv_bytes_delta", "bytes"),
    "send_wrs": ("efa_send_ops_delta", "operations"),
    "send_bytes": ("efa_send_bytes_delta", "bytes"),
    "recv_wrs": ("efa_recv_ops_delta", "operations"),
    "recv_bytes": ("efa_recv_bytes_delta", "bytes"),
    "rx_bytes": ("efa_rx_bytes_delta", "bytes"),
    "tx_bytes": ("efa_tx_bytes_delta", "bytes"),
    "rx_pkts": ("efa_rx_packets_delta", "packets"),
    "tx_pkts": ("efa_tx_packets_delta", "packets"),
}


def port_traffic(
    port: Path,
    observed_at: datetime,
    previous: dict[str, tuple[float, datetime]],
    delta: Callable[[str, float, datetime], tuple[float, float] | None],
) -> tuple[list[HostMetricSample], tuple[float, float] | None, str | None]:
    device = f"{port.parent.parent.name}/{port.name}"
    try:
        paths = {path.name: path for path in (port / "hw_counters").iterdir()}
    except FileNotFoundError:
        return [], None, None
    selected = (
        ("rx_bytes", "tx_bytes")
        if {"rx_bytes", "tx_bytes"} & paths.keys()
        else ("send_bytes", "recv_bytes")
    )
    try:
        totals = {name: read_counter(paths[name]) for name in selected}
    except (OSError, ValueError, KeyError) as exc:
        return [], None, f"{device}: incomplete traffic counter pair ({exc})"
    samples = []
    pair_changes = []
    reset = False
    for name, (metric, unit) in TRAFFIC_COUNTERS.items():
        if name not in paths:
            continue
        try:
            current = totals[name] if name in totals else read_counter(paths[name])
        except (OSError, ValueError):
            continue
        key = f"rdma-traffic/{device}/{name}"
        before = previous.get(key)
        reset = reset or (
            name in selected and before is not None and current < before[0]
        )
        change = delta(key, current, observed_at)
        if change is None:
            continue
        if change[0]:
            samples.append(
                HostMetricSample(name=metric, value=change[0], unit=unit, device=device)
            )
        if name in selected:
            pair_changes.append(change)
    if reset:
        return samples, None, f"{device}: traffic counter reset; rate is unknown"
    if len(pair_changes) != 2:
        return samples, None, None
    if pair_changes[0][1] != pair_changes[1][1] or pair_changes[0][1] <= 0:
        return samples, None, f"{device}: traffic intervals do not match"
    return samples, (sum(item[0] for item in pair_changes), pair_changes[0][1]), None
