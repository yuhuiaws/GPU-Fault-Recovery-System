"""Kubernetes device-plugin liveness from the node (GF-GENERAL-LIVE-014).

The host collector probes every device-plugin socket under kubelet's
``device-plugins`` directory with one bounded ``GetDevicePluginOptions`` call
per tick (``device_plugin_probe.py``) and grades each resource family (GPU,
EFA) on the result: ``<family>_device_plugin_unresponsive`` is 1 once the
plugin accepted the connection and answered nothing for the required number of
consecutive ticks, 0 otherwise. Its labels carry what the probe saw, so the
finding, the incident and the restart step can say which socket, how long and
whether the gRPC transport was alive while the handler was not.

The host collector unit runs as root under ``ProtectSystem=strict``: the
kubelet tree is bind-mounted read-only, which forbids creating or writing
files but not connecting to a unix socket (``net/unix/af_unix.c``
``unix_find_bsd`` asks ``path_permission(MAY_WRITE)`` of the socket inode, and
``fs/namei.c`` ``sb_permission`` refuses write access on a read-only superblock
for regular files, directories and symlinks only). Verified on the deploy host
with a transient unit under the same directives: a write under ``/var/lib``
fails EROFS while a unix connect there exchanges bytes. No unit or node
template change is needed.

A node without the directory (no kubelet, or a kubelet with another root) is
simply not probed: no samples, one log line. A socket that is missing or
refuses the connection is a plugin kubelet already sees as gone (the
``ListAndWatch`` stream ended), covered by the allocatable collector, so it is
reported in the labels but never as a hang.
"""

from __future__ import annotations

import logging
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Mapping

from gpu_fault.collectors.host.device_plugin_probe import (
    ProbeResult,
    ProbeState,
)
from gpu_fault.collectors.host.observation import HostObservationError
from gpu_fault.device_plugin_liveness import (
    DEVICE_PLUGIN_FAMILIES,
    KUBELET_SOCKET,
    MAX_SOCKETS_PER_FAMILY,
    UNRESPONSIVE_FAILURE_MODE,
)
from gpu_fault.host_health import HostMetricSample

LOGGER = logging.getLogger(__name__)


class HostDevicePluginMixin:
    # Attributes supplied by the composed concrete implementation.
    device_plugin_socket_dir: Path | None
    device_plugin_socket_patterns: Mapping[str, tuple[str, ...]]
    device_plugin_probe_timeout_seconds: float
    device_plugin_unresponsive_samples: int
    expected_gpu_count: int | None
    expected_efa_device_count: int | None
    node_instance_type: str | None
    _device_plugin_probe: Callable[[Path, float], ProbeResult]
    _device_plugin_timeouts: dict[str, int]
    _device_plugin_directory_logged: bool

    def _device_plugins(self, _: datetime) -> list[HostMetricSample]:
        """One ``<family>_device_plugin_unresponsive`` sample per family.

        Each socket of a family is probed once, in name order, under the
        configured timeout; the family is unresponsive when any of its sockets
        timed out. The consecutive-timeout count is per family and resets on
        any tick without a timeout -- a socket that disappears mid-hang (the
        Pod was deleted) clears it too.
        """

        directory = self.device_plugin_socket_dir
        if directory is None:
            return []
        if not directory.is_dir():
            if not self._device_plugin_directory_logged:
                self._device_plugin_directory_logged = True
                LOGGER.info(
                    "device plugin socket directory %s is absent; the device "
                    "plugin liveness probe is not running on this node",
                    directory,
                )
            return []
        self._device_plugin_directory_logged = False
        try:
            entries = sorted(
                item
                for item in directory.iterdir()
                if item.name != KUBELET_SOCKET and item.is_socket()
            )
        except OSError as exc:
            raise HostObservationError(
                f"device plugin socket directory {directory} is unreadable: "
                f"{type(exc).__name__}: {exc}",
                [],
            ) from exc
        samples: list[HostMetricSample] = []
        for key, family in DEVICE_PLUGIN_FAMILIES.items():
            patterns = self.device_plugin_socket_patterns.get(key, family.patterns)
            sockets = [
                item
                for item in entries
                if any(fnmatch(item.name, pattern) for pattern in patterns)
            ][:MAX_SOCKETS_PER_FAMILY]
            results = {
                item.name: self._device_plugin_probe(
                    item, self.device_plugin_probe_timeout_seconds
                )
                for item in sockets
            }
            hung = [name for name, item in results.items() if item.hung]
            count = self._device_plugin_timeouts.get(key, 0) + 1 if hung else 0
            self._device_plugin_timeouts[key] = count
            persistent = count >= self.device_plugin_unresponsive_samples
            if not sockets:
                state = ProbeState.SOCKET_MISSING.value
            elif hung:
                state = ProbeState.TIMEOUT.value
            else:
                states = [item.state for item in results.values()]
                state = (
                    ProbeState.RESPONDED.value
                    if all(item is ProbeState.RESPONDED for item in states)
                    else next(
                        item.value
                        for item in states
                        if item is not ProbeState.RESPONDED
                    )
                )
            expected = (
                self.expected_gpu_count
                if key == "gpu"
                else self.expected_efa_device_count
            )
            labels = {
                "resource": family.resource,
                "resource_name": family.resource_name,
                "node_instance_type": self.node_instance_type or "UNKNOWN",
                "sockets": ",".join(results) or "none",
                "probe_state": state,
                "probe_states": ",".join(
                    f"{name}={item.state.value}" for name, item in results.items()
                )
                or "none",
                "probe_detail": "; ".join(
                    f"{name}: {item.detail}" for name, item in results.items()
                )
                or "no socket matched " + "|".join(patterns),
                "transport_answered": str(
                    any(item.transport_answered for item in results.values())
                ).lower(),
                "latency_ms": str(
                    max((item.latency_ms for item in results.values()), default=0.0)
                ),
                "timeout_seconds": f"{self.device_plugin_probe_timeout_seconds:g}",
                "consecutive_timeout_samples": str(count),
                "required_consecutive_samples": str(
                    self.device_plugin_unresponsive_samples
                ),
                "failure_mode": (
                    UNRESPONSIVE_FAILURE_MODE if persistent else "HEALTHY"
                ),
            }
            if expected is not None:
                labels["expected_count"] = str(expected)
            if hung:
                LOGGER.warning(
                    "device plugin %s did not answer GetDevicePluginOptions within "
                    "%gs (%d consecutive; finding after %d): %s",
                    ",".join(hung),
                    self.device_plugin_probe_timeout_seconds,
                    count,
                    self.device_plugin_unresponsive_samples,
                    labels["probe_detail"],
                )
            samples.append(
                HostMetricSample(
                    name=family.metric, value=1 if persistent else 0, labels=labels
                )
            )
        return samples
