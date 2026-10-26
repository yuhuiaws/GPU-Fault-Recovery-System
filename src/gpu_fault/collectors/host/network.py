from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.host_capability_markers import EFA_ETHTOOL_UNSUPPORTED
from gpu_fault.host_health import (
    HostMetricSample,
)
from gpu_fault.collectors.host.observation import HostObservationError, read_counter
from gpu_fault.collectors.host.rdma_traffic import port_traffic

LOGGER = logging.getLogger(__name__)

#: The sysfs counters behind each per-interface delta sample. Read as one unit:
#: ``_delta`` advances a baseline as it reads it, so a counter that fails to
#: parse must not have consumed another counter's interval already. Each file
#: keeps its own baseline (``net/<interface>/<metric>/<file>``): a reset is
#: detected per counter, where ``rx_errors`` falling back to 1 while
#: ``tx_errors`` climbs would otherwise sum to an unchanged, "clean" total.
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


#: ``ethtool -S`` counter-name tokens and the delta each one feeds. A counter is
#: assigned to the first group whose token its lower-cased name contains, so
#: ``rnr_retry_err`` counts once, as RNR.
_ETHTOOL_GROUPS: dict[str, str] = {
    "rnr": "efa_rnr_errors_delta",
    "retry": "efa_retry_errors_delta",
    "cq_err": "efa_cq_errors_delta",
    "pfc": "network_pfc_pause_delta",
    "ecn": "network_ecn_marks_delta",
}

#: What ``ethtool -S`` prints for a driver that exposes no statistics at all:
#: ``no stats available`` (exit 94), or EOPNOTSUPP from the string-set query.
_ETHTOOL_NO_STATISTICS = ("no stats available", "Operation not supported")
_ETHTOOL_NO_STATISTICS_EXIT = 94

#: The NOT_SUPPORTED marker ``_ethtool_counters`` ships for the groups a netdev
#: cannot provide. Like every capability marker it is a state, not an edge of
#: its own; one that appears, changes or clears after the first tick is an
#: edge (``host_capability_markers.not_supported_edge_reason``, applied by
#: ``HostTelemetryCollector._edge_reasons``) and every delivered batch keeps it
#: in the node's latest records. Live GF-GENERAL-LIVE-010 ``missing_counter``
#: (2026-10-25): the marker named the withdrawn counter on every tick, but no
#: reason delivered the batch and a health summary without findings or
#: collection errors is not kept as evidence, so the product's records never
#: said the counter was gone.


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
        errors: list[str] = []
        try:
            interfaces = sorted(self.net_class_root.iterdir())
        except OSError as exc:
            raise HostObservationError(
                f"cannot list {self.net_class_root}: {exc}", []
            ) from exc
        missing = self.required_interfaces - {
            interface.name for interface in interfaces
        }
        for name in sorted(missing):
            result.extend(
                [
                    self._sample("network_link_up", 0, None, name),
                    self._sample("network_link_down", 1, None, name),
                ]
            )
        for interface in interfaces:
            name = interface.name
            if name == "lo":
                continue
            try:
                if not self._is_reported_interface(interface):
                    continue
                samples = self._interface_samples(interface, observed_at)
            except (OSError, ValueError, HostObservationError) as exc:
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
                if isinstance(exc, HostObservationError):
                    result.extend(exc.samples)
                errors.append(f"{name}: {exc}")
                continue
            seen.add(name)
            result.extend(samples)
        self._prune_interface_counters(seen)
        if errors:
            raise HostObservationError("; ".join(errors), result)
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
        """Keep link evidence even when the atomic counter snapshot fails."""

        name = interface.name
        state = (interface / "operstate").read_text().strip()
        if state not in {
            "up",
            "down",
            "lowerlayerdown",
            "notpresent",
            "dormant",
            "testing",
        }:
            raise HostObservationError(f"link state is unknown: {state!r}", [])
        # Read and parse every counter *before* any baseline moves. ``_delta``
        # advances the baseline as it reads it, so computing the error delta
        # before the drop counters had even been parsed let an unparseable drop
        # counter silently consume the error interval: the next tick then
        # reported two intervals of errors as one, halving the rate the
        # consumer's error-rate rule computes.
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
        try:
            counters = {
                (metric, filename): read_counter(interface / "statistics" / filename)
                for metric, files in _COUNTER_FILES.items()
                for filename in files
            }
        except (OSError, ValueError) as exc:
            raise HostObservationError(
                f"statistics unavailable: {exc}", result
            ) from exc
        # A counter below its baseline is a reset, not "no errors": the delta
        # ``_delta`` clamps to 0 is a reading (``_ZERO_IS_A_READING``) that
        # clears an active ``network_errors_delta`` signal, restarts the drops
        # sustain window and passes VALIDATE_FABRIC, and the next increase
        # then mints the same fault as a new incident. The metric is unknown
        # for this tick (a collection error naming the counter), its baseline
        # re-armed from the new value; the other metric is still a reading.
        problems: list[str] = []
        for metric, files in _COUNTER_FILES.items():
            total = 0.0
            complete = True
            for filename in files:
                value = float(counters[metric, filename])
                change, reset_from = self._monotonic_delta(
                    f"net/{name}/{metric}/{filename}", value, observed_at
                )
                if reset_from is not None:
                    problems.append(self._reset_problem(filename, reset_from, value))
                    complete = False
                elif change is None:
                    complete = False
                else:
                    total += change[0]
            if complete and _is_reportable(metric, total):
                result.append(self._sample(metric, total, "packets", name))
        if problems:
            raise HostObservationError("; ".join(problems), result)
        return result

    def _monotonic_delta(
        self, key: str, value: float, observed_at: datetime
    ) -> tuple[tuple[float, float] | None, float | None]:
        """``_delta`` plus the baseline a reset counter fell below.

        ``_delta`` clamps a counter that went backwards to a 0 delta and moves
        the baseline on. The clamp is right for the rates the other
        contributors derive; for the error counters here a 0 is a reading
        (:data:`_ZERO_IS_A_READING`) that would clear a signal nothing
        measured. Returns the ``_delta`` result and, when the counter read
        below its baseline, that baseline, so the caller can withhold the
        delta and name the reset. The baseline is re-armed either way.
        """

        before = self._previous.get(key)
        change = self._delta(key, value, observed_at)
        if before is not None and value < before[0]:
            return change, before[0]
        return change, None

    @staticmethod
    def _reset_problem(counter: str, before: float, value: float) -> str:
        """The one wording every counter reader uses for a reset tick."""

        return f"{counter} reset ({before:g} -> {value:g}); delta unknown"

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
        aggregate_rate = 0.0
        traffic_complete = True
        device_count = 0
        port_count = 0
        errors = []
        root = self.infiniband_root
        if not root.exists():
            return [self._sample("efa_traffic_observation_valid", 0)]
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
        for device in root.iterdir():
            if not self._is_efa_device(device):
                continue
            device_count += 1
            ports = list((device / "ports").iterdir())
            if not ports:
                traffic_complete = False
                errors.append(f"{device.name}: no observable RDMA ports")
            for port in ports:
                port_count += 1
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
                # A counter below its baseline is a reset (GF-GENERAL-LIVE-010
                # counter_reset), handled as ``_ethtool_counters`` handles its
                # groups: the counter's own delta and the port's
                # ``rdma_errors_delta`` -- the delta of the counter sum, which
                # a reset lowers or, with another counter climbing, leaves
                # misleadingly unchanged -- are unknown for this tick, named
                # in a collection error, and measured from the new baseline
                # on the next one. ``_delta`` clamped the reset to a 0 that
                # cleared the active signal, passed VALIDATE_FABRIC and let the
                # continuing increase mint the fault again.
                resets: list[str] = []
                readings: list[tuple[str, float]] = []
                for counter in sorted(counter_names):
                    path = port / "counters" / counter
                    if path.exists():
                        readings.append((counter, float(path.read_text().strip())))
                for counter in sorted(hw_counter_names):
                    path = port / "hw_counters" / counter
                    if path.exists():
                        raw = path.read_text().strip()
                        if raw.isdigit():
                            readings.append((counter, float(raw)))
                for counter, value in readings:
                    total += value
                    change, reset_from = self._monotonic_delta(
                        f"rdma/{device.name}/{port.name}/{counter}",
                        value,
                        observed_at,
                    )
                    if reset_from is not None:
                        resets.append(self._reset_problem(counter, reset_from, value))
                    elif change and _is_reportable(f"rdma_{counter}_delta", change[0]):
                        result.append(
                            self._sample(
                                f"rdma_{counter}_delta",
                                change[0],
                                "events",
                                f"{device.name}/{port.name}",
                            )
                        )
                try:
                    traffic_samples, traffic, error = port_traffic(
                        port, observed_at, self._previous, self._delta
                    )
                except OSError as exc:
                    traffic_samples, traffic, error = [], None, str(exc)
                result.extend(traffic_samples)
                if error:
                    errors.append(error)
                if traffic is None:
                    traffic_complete = False
                else:
                    aggregate_traffic_delta += traffic[0]
                    aggregate_rate += traffic[0] / traffic[1]
                key = f"rdma/{device.name}/{port.name}"
                change, reset_from = self._monotonic_delta(key, total, observed_at)
                if reset_from is not None and not resets:
                    # No counter fell below its own baseline, yet the sum did:
                    # a counter the port no longer exposes. Not zero errors.
                    resets.append(
                        self._reset_problem("error counter total", reset_from, total)
                    )
                if resets:
                    errors.append(f"{device.name}/{port.name}: " + "; ".join(resets))
                elif change:
                    result.append(
                        self._sample(
                            "rdma_errors_delta",
                            change[0],
                            "errors",
                            f"{device.name}/{port.name}",
                        )
                    )
        expected = getattr(self, "expected_efa_device_count", None)
        if expected is not None and device_count != expected:
            traffic_complete = False
            errors.append(
                f"incomplete EFA traffic coverage: {device_count}/{expected} devices"
            )
        valid = traffic_complete and port_count > 0
        result.append(self._sample("efa_traffic_observation_valid", int(valid)))
        if valid:
            result.extend(
                [
                    self._sample(
                        "efa_traffic_bytes_delta",
                        aggregate_traffic_delta,
                        "bytes",
                    ),
                    self._sample(
                        "efa_traffic_bytes_per_second",
                        aggregate_rate,
                        "bytes/second",
                    ),
                ]
            )
        if errors:
            raise HostObservationError("; ".join(errors), result)
        return result

    def _efa_network(self, observed_at: datetime) -> list[HostMetricSample]:
        """EFA congestion and error deltas from ``ethtool -S`` per EFA netdev.

        A delta is reported only from a real reading (GF-GENERAL-LIVE-010).
        Every group used to start at 0.0, so a counter the driver does not
        expose, the first reading (no baseline yet) and a counter reset all
        shipped as "0 errors" -- which ``VALIDATE_FABRIC`` accepts as a clean
        link -- and a failed ``ethtool`` was skipped without a trace. Now a
        failed or hung ``ethtool``, a reset and an unparseable counter are
        UNKNOWN (a collection error, no delta), and a counter the netdev does
        not have is NOT_SUPPORTED (``efa_ethtool_counters_unsupported``, no
        collection error: the gap is permanent; one that opens, changes or
        closes after the first tick is an edge,
        ``host_capability_markers.not_supported_edge_reason``).
        One netdev's failure no longer costs the others their readings.
        """

        if not shutil.which("ethtool"):
            return []
        interfaces = set()
        for device in self.infiniband_root.glob("*"):
            if not self._is_efa_device(device):
                continue
            interfaces.update(
                item.name for item in (device / "device" / "net").glob("*")
            )
        result: list[HostMetricSample] = []
        errors: list[str] = []
        for interface in sorted(interfaces):
            samples, error = self._ethtool_counters(interface, observed_at)
            result.extend(samples)
            if error is not None:
                errors.append(f"{interface}: {error}")
        if errors:
            raise HostObservationError("; ".join(errors), result)
        return result

    def _ethtool_counters(
        self, interface: str, observed_at: datetime
    ) -> tuple[list[HostMetricSample], str | None]:
        """One netdev's group deltas, and why some or all are unknown."""

        try:
            completed = self.runner(
                ["ethtool", "-S", interface],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return [], f"ethtool -S failed: {type(exc).__name__}: {exc}"
        if completed.returncode != 0:
            output = f"{completed.stderr or ''} {completed.stdout or ''}".strip()
            if completed.returncode == _ETHTOOL_NO_STATISTICS_EXIT or any(
                text in output for text in _ETHTOOL_NO_STATISTICS
            ):
                return [
                    self._ethtool_unsupported(
                        interface, _ETHTOOL_GROUPS.values(), "NO_STATISTICS"
                    )
                ], None
            return [], f"ethtool -S exited {completed.returncode}: {output[:200]}"
        deltas: dict[str, float] = {}
        matched: set[str] = set()
        incomplete: set[str] = set()
        problems: list[str] = []
        for line in completed.stdout.splitlines():
            if ":" not in line:
                continue
            name, raw = (item.strip() for item in line.split(":", 1))
            lowered = name.lower()
            metric = next(
                (
                    metric
                    for token, metric in _ETHTOOL_GROUPS.items()
                    if token in lowered
                ),
                None,
            )
            if metric is None:
                continue
            matched.add(metric)
            if not (raw.isascii() and raw.isdigit()):
                incomplete.add(metric)
                problems.append(f"{name} is not a counter: {raw[:40]!r}")
                continue
            value = float(raw)
            change, reset_from = self._monotonic_delta(
                f"ethtool/{interface}/{name}", value, observed_at
            )
            if reset_from is not None:
                # ``_delta`` clamps a reset to 0; that zero would clear an
                # active error signal although nothing was measured.
                incomplete.add(metric)
                problems.append(self._reset_problem(name, reset_from, value))
            elif change is None:
                # First reading, or no time has passed: a baseline, no delta.
                incomplete.add(metric)
            else:
                deltas[metric] = deltas.get(metric, 0.0) + change[0]
        result = [
            self._sample(metric, deltas[metric], "events", interface)
            for metric in _ETHTOOL_GROUPS.values()
            if metric in deltas
            and metric not in incomplete
            and _is_reportable(metric, deltas[metric])
        ]
        missing = [
            metric for metric in _ETHTOOL_GROUPS.values() if metric not in matched
        ]
        if missing:
            result.append(
                self._ethtool_unsupported(interface, missing, "MISSING_COUNTER")
            )
        return result, "; ".join(problems) or None

    @staticmethod
    def _ethtool_unsupported(
        interface: str, metrics: Iterable[str], reason: str
    ) -> HostMetricSample:
        """The deltas this netdev cannot provide, named instead of zeroed.

        ``MISSING_COUNTER``: ``ethtool -S`` answered but has no counter for
        them; ``NO_STATISTICS``: the driver exposes no statistics at all.
        """

        names = sorted(metrics)
        return HostMetricSample(
            name=EFA_ETHTOOL_UNSUPPORTED,
            value=len(names),
            unit="counters",
            device=interface,
            labels={"reason": reason, "metrics": ",".join(names)},
        )
