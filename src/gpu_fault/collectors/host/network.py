from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.host_health import (
    HostMetricSample,
)

LOGGER = logging.getLogger(__name__)


class HostNetworkMixin:
    # Attributes supplied by the composed concrete implementation.
    _delta: Callable[..., Any]
    _is_efa_device: Callable[..., Any]
    _sample: Callable[..., Any]
    infiniband_root: Any
    required_interfaces: Any
    runner: Callable[..., Any]

    def _network(self, observed_at: datetime) -> list[HostMetricSample]:
        result = []
        for interface in Path("/sys/class/net").iterdir():
            name = interface.name
            if name == "lo":
                continue
            state = (interface / "operstate").read_text().strip()
            result.append(
                self._sample(
                    "network_link_up",
                    1 if state == "up" else 0,
                    None,
                    name,
                )
            )
            if name in self.required_interfaces:
                result.append(
                    self._sample(
                        "network_link_down",
                        0 if state == "up" else 1,
                        None,
                        name,
                    )
                )
            stats = interface / "statistics"
            for metric, files in {
                "network_errors_delta": (
                    "rx_errors",
                    "tx_errors",
                ),
                "network_drops_delta": (
                    "rx_dropped",
                    "tx_dropped",
                ),
            }.items():
                total = sum(float((stats / filename).read_text()) for filename in files)
                change = self._delta(
                    f"net/{name}/{metric}",
                    total,
                    observed_at,
                )
                if change:
                    result.append(
                        self._sample(
                            metric,
                            change[0],
                            "packets",
                            name,
                        )
                    )
        return result

    def _tcp(self, observed_at: datetime) -> list[HostMetricSample]:
        lines = Path("/proc/net/snmp").read_text().splitlines()
        retransmits = None
        for index in range(0, len(lines) - 1, 2):
            header = lines[index].split()
            values = lines[index + 1].split()
            if header and values and header[0] == "Tcp:" and values[0] == "Tcp:":
                fields = dict(zip(header[1:], values[1:], strict=False))
                retransmits = float(fields.get("RetransSegs", 0))
                break
        if retransmits is None:
            return []
        change = self._delta(
            "tcp/retransmitted_segments",
            retransmits,
            observed_at,
        )
        if change is None:
            return []
        return [
            self._sample(
                "tcp_retransmits_delta",
                change[0],
                "segments",
            )
        ]

    def _rdma(self, observed_at: datetime) -> list[HostMetricSample]:
        result = []
        aggregate_traffic_delta = 0.0
        aggregate_interval = 0.0
        root = self.infiniband_root
        if not root.exists():
            return result
        counter_names = {
            "symbol_error",
            "link_error_recovery",
            "link_downed",
            "port_rcv_errors",
            "port_xmit_discards",
            "port_rcv_remote_physical_errors",
            "port_xmit_constraint_errors",
            "port_rcv_constraint_errors",
        }
        hw_counter_names = {
            "local_ack_timeout_err",
            "out_of_sequence",
            "packet_seq_err",
            "poll_cq_err",
            "rnr_nak_retry_err",
            "rx_drops",
            "rx_overrun_errors",
            "send_queue_overflow",
            "tx_drops",
        }
        traffic_counters = {
            "rdma_read_wrs": (
                "efa_rdma_read_ops_delta",
                "operations",
            ),
            "rdma_read_bytes": (
                "efa_rdma_read_bytes_delta",
                "bytes",
            ),
            "rdma_write_wrs": (
                "efa_rdma_write_ops_delta",
                "operations",
            ),
            "rdma_write_bytes": (
                "efa_rdma_write_bytes_delta",
                "bytes",
            ),
            "rdma_write_recv_bytes": (
                "efa_rdma_write_recv_bytes_delta",
                "bytes",
            ),
            "send_wrs": (
                "efa_send_ops_delta",
                "operations",
            ),
            "send_bytes": ("efa_send_bytes_delta", "bytes"),
            "recv_wrs": (
                "efa_recv_ops_delta",
                "operations",
            ),
            "recv_bytes": ("efa_recv_bytes_delta", "bytes"),
            "rx_bytes": ("efa_rx_bytes_delta", "bytes"),
            "tx_bytes": ("efa_tx_bytes_delta", "bytes"),
            "rx_pkts": ("efa_rx_packets_delta", "packets"),
            "tx_pkts": ("efa_tx_packets_delta", "packets"),
        }
        for device in root.iterdir():
            if not self._is_efa_device(device):
                continue
            for port in (device / "ports").glob("*"):
                state_path = port / "state"
                physical_path = port / "phys_state"
                if state_path.exists():
                    state = state_path.read_text().strip().split(":", 1)[0]
                    physical = (
                        physical_path.read_text().strip().split(":", 1)[0]
                        if physical_path.exists()
                        else "5"
                    )
                    result.append(
                        self._sample(
                            "rdma_link_down",
                            0 if state == "4" and physical == "5" else 1,
                            None,
                            f"{device.name}/{port.name}",
                        )
                    )
                total = 0.0
                for counter in counter_names:
                    path = port / "counters" / counter
                    if path.exists():
                        value = float(path.read_text().strip())
                        total += value
                        change = self._delta(
                            (f"rdma/{device.name}/{port.name}/{counter}"),
                            value,
                            observed_at,
                        )
                        if change:
                            result.append(
                                self._sample(
                                    f"rdma_{counter}_delta",
                                    change[0],
                                    "events",
                                    f"{device.name}/{port.name}",
                                )
                            )
                for counter in hw_counter_names:
                    path = port / "hw_counters" / counter
                    if path.exists():
                        raw = path.read_text().strip()
                        if raw.isdigit():
                            value = float(raw)
                            total += value
                            change = self._delta(
                                (f"rdma/{device.name}/{port.name}/{counter}"),
                                value,
                                observed_at,
                            )
                            if change:
                                result.append(
                                    self._sample(
                                        f"rdma_{counter}_delta",
                                        change[0],
                                        "events",
                                        (f"{device.name}/{port.name}"),
                                    )
                                )
                port_traffic_delta = 0.0
                port_interval = 0.0
                has_rx_tx = all(
                    (port / "hw_counters" / name).exists()
                    for name in ("rx_bytes", "tx_bytes")
                )
                for counter, (
                    metric,
                    unit,
                ) in traffic_counters.items():
                    path = port / "hw_counters" / counter
                    if not path.exists():
                        continue
                    raw = path.read_text().strip()
                    if not raw.isdigit():
                        continue
                    change = self._delta(
                        (f"rdma-traffic/{device.name}/{port.name}/{counter}"),
                        float(raw),
                        observed_at,
                    )
                    if change is None:
                        continue
                    result.append(
                        self._sample(
                            metric,
                            change[0],
                            unit,
                            f"{device.name}/{port.name}",
                        )
                    )
                    if (has_rx_tx and counter in {"rx_bytes", "tx_bytes"}) or (
                        not has_rx_tx and counter in {"send_bytes", "recv_bytes"}
                    ):
                        port_traffic_delta += change[0]
                        port_interval = max(port_interval, change[1])
                aggregate_traffic_delta += port_traffic_delta
                aggregate_interval = max(aggregate_interval, port_interval)
                key = f"rdma/{device.name}/{port.name}"
                change = self._delta(key, total, observed_at)
                if change:
                    result.append(
                        self._sample(
                            "rdma_errors_delta",
                            change[0],
                            "errors",
                            f"{device.name}/{port.name}",
                        )
                    )
        if aggregate_interval > 0:
            result.extend(
                [
                    self._sample(
                        "efa_traffic_bytes_delta",
                        aggregate_traffic_delta,
                        "bytes",
                    ),
                    self._sample(
                        "efa_traffic_bytes_per_second",
                        aggregate_traffic_delta / aggregate_interval,
                        "bytes/second",
                    ),
                ]
            )
        return result

    def _efa_network(self, observed_at: datetime) -> list[HostMetricSample]:
        if not shutil.which("ethtool"):
            return []
        interfaces = set()
        for device in self.infiniband_root.glob("*"):
            if not self._is_efa_device(device):
                continue
            interfaces.update(
                item.name for item in (device / "device" / "net").glob("*")
            )
        result = []
        groups = {
            "rnr": "efa_rnr_errors_delta",
            "retry": "efa_retry_errors_delta",
            "cq_err": "efa_cq_errors_delta",
            "pfc": "network_pfc_pause_delta",
            "ecn": "network_ecn_marks_delta",
        }
        for interface in interfaces:
            completed = self.runner(
                ["ethtool", "-S", interface],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if completed.returncode != 0:
                continue
            counters = {name: 0.0 for name in groups.values()}
            for line in completed.stdout.splitlines():
                if ":" not in line:
                    continue
                name, raw = (item.strip() for item in line.split(":", 1))
                try:
                    value = float(raw)
                except ValueError:
                    continue
                lowered = name.lower()
                matched = next(
                    (metric for token, metric in groups.items() if token in lowered),
                    None,
                )
                if matched is not None:
                    change = self._delta(
                        f"ethtool/{interface}/{name}",
                        value,
                        observed_at,
                    )
                    if change:
                        counters[matched] += change[0]
            result.extend(
                self._sample(metric, count, "events", interface)
                for metric, count in counters.items()
            )
        return result
