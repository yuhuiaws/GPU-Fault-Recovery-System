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

#: The sysfs counters behind each per-interface delta sample. Read as one unit:
#: ``_delta`` advances a baseline as it reads it, so a counter that fails to
#: parse must not have consumed another counter's interval already.
_COUNTER_FILES: dict[str, tuple[str, ...]] = {
    "network_errors_delta": ("rx_errors", "tx_errors"),
    "network_drops_delta": ("rx_dropped", "tx_dropped"),
}

#: Delta samples the consumer reads as *latest values* rather than as events.
#: A zero for these must still ship: ``NodeHealthPolicy.LATEST_METRICS_REQUIRED``
#: persists them for VALIDATE_FABRIC, and the sustained rules
#: (``METRIC_RULE_SUSTAIN_SECONDS["network_drops_delta"]``) can only *clear* a
#: signal on a below-threshold reading -- a fabric validation with no fresh zero
#: waits until its deadline instead. The PFC/ECN pair is here for the same
#: reason: they are congestion state, not events, and there are two per netdev.
_ZERO_IS_A_READING: frozenset[str] = frozenset(
    {
        "network_errors_delta",
        "network_drops_delta",
        "network_pfc_pause_delta",
        "network_ecn_marks_delta",
        "rdma_errors_delta",
        "efa_rnr_errors_delta",
        "efa_retry_errors_delta",
        "efa_cq_errors_delta",
    }
)


def _is_reportable(name: str, value: float) -> bool:
    """Whether a delta sample earns its place in the batch.

    A quiet p5en has 16 EFA ports and about 30 per-port deltas each, so roughly
    500 zero-valued samples -- about 100 KB of JSON -- were posted every batch to
    say that nothing happened (F-H8). Per-port traffic and per-counter deltas are
    events: their absence *is* the zero. The names in
    :data:`_ZERO_IS_A_READING` are not, and are kept whatever they read.
    """

    return value != 0.0 or name in _ZERO_IS_A_READING


class HostNetworkMixin:
    # Attributes supplied by the composed concrete implementation.
    _delta: Callable[..., Any]
    _is_efa_device: Callable[..., Any]
    _previous: dict[str, tuple[float, datetime]]
    _sample: Callable[..., Any]
    infiniband_root: Any
    net_class_root: Path
    required_interfaces: Any
    runner: Callable[..., Any]

    def _network(self, observed_at: datetime) -> list[HostMetricSample]:
        """Link state and error/drop counters for the node's real interfaces.

        Every read is guarded per interface and every virtual interface is
        skipped. Pod churn on an EKS node creates and deletes ``veth``/``eni``
        pairs constantly: one deleted between ``iterdir`` and ``read_text``
        raised ``FileNotFoundError`` and discarded the whole tick's network
        samples -- including ``network_link_down`` for a required interface,
        the one CRITICAL signal here -- and every transient veth was reported
        as a device of its own, growing the control plane's latest-metrics rows
        and tripping the ``network_drops_delta`` rule on drops that are routine
        on a veth (F-H4).
        """

        result: list[HostMetricSample] = []
        seen: set[str] = set()
        try:
            interfaces = sorted(self.net_class_root.iterdir())
        except OSError as exc:
            LOGGER.warning("cannot list %s: %s", self.net_class_root, exc)
            return result
        for interface in interfaces:
            name = interface.name
            if name == "lo":
                continue
            try:
                if not self._is_reported_interface(interface):
                    continue
                samples = self._interface_samples(interface, observed_at)
            except (OSError, ValueError):
                # Two different failures with two different answers. A veth
                # deleted between ``iterdir`` and the read is gone, so its
                # baseline may be pruned. A counter that reads back empty or
                # ``[N/A]`` -- ``float("")`` raises ``ValueError``, not
                # ``OSError``, so it used to discard the whole tick -- belongs
                # to an interface that is still there: keeping it in ``seen``
                # preserves the baseline, without which an intermittently
                # failing NIC, the one the error-rate rule exists for, would
                # never report ``network_errors_delta`` again.
                if interface.exists():
                    seen.add(name)
                continue
            seen.add(name)
            result.extend(samples)
        self._prune_interface_counters(seen)
        return result

    def _is_reported_interface(self, interface: Path) -> bool:
        """Whether one ``/sys/class/net`` entry is worth a sample.

        A required interface always is. Otherwise only interfaces with a
        parent device are: a bridge, a tunnel and a pod's veth have none.
        """

        if interface.name in self.required_interfaces:
            return True
        return (interface / "device").exists()

    def _interface_samples(
        self, interface: Path, observed_at: datetime
    ) -> list[HostMetricSample]:
        """Every sample for one interface, or nothing if any read fails.

        Built as a unit so a half-read interface contributes nothing rather
        than a link state without its counters. Raises ``OSError`` if the
        interface disappears and ``ValueError`` if a counter reads back
        unparseable; the caller tells those two apart.
        """

        name = interface.name
        state = (interface / "operstate").read_text().strip()
        # Read and parse every counter *before* any baseline moves. ``_delta``
        # advances the baseline as it reads it, so computing the error delta
        # before the drop counters had even been parsed let an unparseable drop
        # counter silently consume the error interval: the next tick then
        # reported two intervals of errors as one, halving the rate the
        # consumer's error-rate rule computes.
        totals = {
            metric: sum(
                float((interface / "statistics" / filename).read_text())
                for filename in files
            )
            for metric, files in _COUNTER_FILES.items()
        }
        result = [
            self._sample(
                "network_link_up",
                1 if state == "up" else 0,
                None,
                name,
            )
        ]
        if name in self.required_interfaces:
            result.append(
                self._sample(
                    "network_link_down",
                    0 if state == "up" else 1,
                    None,
                    name,
                )
            )
        for metric, total in totals.items():
            change = self._delta(
                f"net/{name}/{metric}",
                total,
                observed_at,
            )
            if change and _is_reportable(metric, change[0]):
                result.append(
                    self._sample(
                        metric,
                        change[0],
                        "packets",
                        name,
                    )
                )
        return result

    def _prune_interface_counters(self, seen: set[str]) -> None:
        """Drop the counter baselines of interfaces this tick did not see.

        A node runs this collector for weeks across thousands of pods; without
        this the ``net/<interface>/`` keys grow for every veth that ever
        existed (the ``rank/`` keys are pruned the same way).
        """

        stale = [
            key
            for key in self._previous
            if key.startswith("net/") and key.split("/")[1] not in seen
        ]
        for key in stale:
            del self._previous[key]

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
                        if change and _is_reportable(
                            f"rdma_{counter}_delta", change[0]
                        ):
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
                            if change and _is_reportable(
                                f"rdma_{counter}_delta", change[0]
                            ):
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
                    if _is_reportable(metric, change[0]):
                        result.append(
                            self._sample(
                                metric,
                                change[0],
                                unit,
                                f"{device.name}/{port.name}",
                            )
                        )
                    # The aggregate is accumulated whether or not the per-port
                    # sample ships: zero traffic is the EFA signal, and the
                    # traffic state machine reads only the aggregate.
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
                if _is_reportable(metric, count)
            )
        return result
