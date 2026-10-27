"""What a Kubernetes device plugin owes the node, and how the product grades it.

kubelet learns about ``nvidia.com/gpu`` and ``vpc.amazonaws.com/efa`` from the
device plugins' ``ListAndWatch`` streams and publishes the result as the Node's
capacity and allocatable. The control-plane collector
(``collectors/cloud/kubernetes.py``) grades that advertisement:
``*_kubernetes_allocatable_mismatch`` when a node advertises fewer devices than
its instance type carries. A plugin that is *hung but still connected* -- its
process stopped or deadlocked while its unix socket under
``/var/lib/kubelet/device-plugins/`` stays open -- never shows there: kubelet's
device manager holds the ``ListAndWatch`` stream open without a deadline or a
keepalive (``pkg/kubelet/cm/devicemanager/plugin/v1beta1/client.go``,
``client.Run`` blocking in ``stream.Recv()``), so the devices stay healthy and
allocatable stays full (live 2026-10-27, HyperPod, GF-GENERAL-LIVE-014
``gpu_plugin``: the NVIDIA plugin frozen for four minutes, ``nvidia.com/gpu``
allocatable 8 throughout). Meanwhile kubelet's ``Allocate`` RPC to that plugin
(``pkg/kubelet/cm/devicemanager/endpoint.go``, ``context.Background()``) blocks
the Pod admission path -- and with it the kubelet sync loop -- for every new
Pod that asks for the resource.

The host collector therefore probes each plugin's own socket on the node with a
bounded ``GetDevicePluginOptions`` call (``collectors/host/device_plugin.py``):
``*_device_plugin_unresponsive`` is 1 once the plugin accepted the connection
and answered nothing for the required number of consecutive ticks. This module
holds what the collector, the rule table (``host_health.py``), the planner and
the Kubernetes adapter share about these findings, so none of them spells a
metric name the others do not know. Standard library only: part of the node
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeVar

from gpu_fault.models import RecoveryAction, Severity

CategoryT = TypeVar("CategoryT", bound=Enum)

#: Where kubelet expects device plugins to listen
#: (``k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1`` ``DevicePluginPath``).
DEFAULT_SOCKET_DIR = "/var/lib/kubelet/device-plugins"
#: kubelet's own registration server in that directory; never a plugin.
KUBELET_SOCKET = "kubelet.sock"
#: ``GPU_FAULT_DEVICE_PLUGIN_PROBE_TIMEOUT_SECONDS``. ``GetDevicePluginOptions``
#: returns a constant from the plugin's gRPC handler goroutine; a healthy plugin
#: answers in milliseconds however loaded the node is. Five seconds is two
#: orders of magnitude above that, so a merely slow plugin is never a timeout,
#: and two consecutive 15-second ticks must time out before the finding opens.
DEFAULT_PROBE_TIMEOUT_SECONDS = 5.0
#: ``GPU_FAULT_DEVICE_PLUGIN_UNRESPONSIVE_CONSECUTIVE_SAMPLES``: like the
#: allocatable rule's two consecutive mismatch samples.
DEFAULT_CONSECUTIVE_SAMPLES = 2
#: How many sockets of one family a tick probes (each probe is bounded by the
#: timeout, so this bounds the tick): the NVIDIA plugin opens one socket per
#: resource name (``nvidia-gpu.sock``; one per MIG profile under the ``mixed``
#: strategy), the EFA plugin one.
MAX_SOCKETS_PER_FAMILY = 8
#: Label value on the sample while the finding is active.
UNRESPONSIVE_FAILURE_MODE = "DEVICE_PLUGIN_UNRESPONSIVE"


@dataclass(frozen=True)
class DevicePluginFamily:
    """One device-plugin resource the product restarts a plugin for."""

    key: str
    #: The ``resource`` label the allocatable collector also writes.
    resource: str
    resource_name: str
    metric: str
    #: Socket basenames under the kubelet directory (``fnmatch`` globs). The
    #: NVIDIA plugin names its socket ``nvidia-<resource suffix>.sock``
    #: (``internal/plugin/server.go`` ``getPluginSocketPath``); the EFA plugin's
    #: basename is matched by the resource word, so a renamed build still
    #: counts. ``GPU_FAULT_DEVICE_PLUGIN_SOCKET_PATTERNS`` overrides.
    patterns: tuple[str, ...]
    #: ``host_health.NodeHealthCategory`` member name; the enum itself lives in
    #: ``host_health``, which imports this module, so it is named, not imported.
    category_name: str
    action: RecoveryAction
    reason: str


DEVICE_PLUGIN_FAMILIES: Mapping[str, DevicePluginFamily] = {
    "gpu": DevicePluginFamily(
        key="gpu",
        resource="GPU",
        resource_name="nvidia.com/gpu",
        metric="gpu_device_plugin_unresponsive",
        patterns=("nvidia-*.sock",),
        category_name="GPU",
        action=RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        reason=(
            "the Kubernetes GPU device plugin is registered with kubelet but "
            "does not answer; new Pods cannot be allocated nvidia.com/gpu"
        ),
    ),
    "efa": DevicePluginFamily(
        key="efa",
        resource="EFA",
        resource_name="vpc.amazonaws.com/efa",
        metric="efa_device_plugin_unresponsive",
        patterns=("*efa*.sock",),
        category_name="RDMA",
        action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
        reason=(
            "the Kubernetes EFA device plugin is registered with kubelet but "
            "does not answer; new Pods cannot be allocated vpc.amazonaws.com/efa"
        ),
    ),
}

#: The liveness metrics, by family key.
DEVICE_PLUGIN_UNRESPONSIVE_METRICS = frozenset(
    family.metric for family in DEVICE_PLUGIN_FAMILIES.values()
)
#: Every finding about a node's Kubernetes device resources that the
#: orchestration arbitrates as one family (expected-count stamping, reboot
#: escalation of a failed plugin restart, grouping behind a node-exclusive
#: incumbent): the host inventory, the allocatable advertisement and the
#: plugin liveness probe.
NODE_DEVICE_RESOURCE_METRICS = frozenset(
    {
        "efa_inventory_mismatch",
        "efa_kubernetes_allocatable_mismatch",
        "gpu_kubernetes_allocatable_mismatch",
        *DEVICE_PLUGIN_UNRESPONSIVE_METRICS,
    }
)
#: Step parameter the planner stamps on a ``RESTART_*_DEVICE_PLUGIN`` built from
#: a liveness finding: the plugin advertises every device, so the restart must
#: not read the full allocatable as "already healthy" and delete nothing.
PLUGIN_UNRESPONSIVE_PARAMETER = "plugin_unresponsive"


def kubernetes_device_plugin_rules(
    category: type[CategoryT],
) -> dict[str, tuple[float, CategoryT, Severity, RecoveryAction, str]]:
    """``host_health.NodeHealthPolicy.METRIC_RULES`` entries for the
    device-plugin availability signals: the control-plane allocatable
    comparison and the node liveness probe, one rule per resource, all
    CRITICAL plugin restarts. ``category`` is ``host_health.NodeHealthCategory``
    (passed in: that module imports this one)."""

    return {
        "efa_kubernetes_allocatable_mismatch": (
            1.0,
            category["RDMA"],
            Severity.CRITICAL,
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            "Kubernetes EFA allocatable does not match the host EFA inventory",
        ),
        "gpu_kubernetes_allocatable_mismatch": (
            1.0,
            category["GPU"],
            Severity.CRITICAL,
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
            "Kubernetes GPU allocatable does not match the host GPU inventory",
        ),
        **{
            family.metric: (
                1.0,
                category[family.category_name],
                Severity.CRITICAL,
                family.action,
                family.reason,
            )
            for family in DEVICE_PLUGIN_FAMILIES.values()
        },
    }


def parse_socket_patterns(text: str | None) -> dict[str, tuple[str, ...]]:
    """``GPU_FAULT_DEVICE_PLUGIN_SOCKET_PATTERNS`` -> family -> globs.

    ``gpu=nvidia-*.sock,efa=*efa*.sock``: comma-separated ``family=glob``
    entries; a family named more than once collects every glob; a family left
    out keeps its default; an unknown family or a malformed entry refuses to
    start rather than silently probing nothing.
    """

    patterns = {
        key: list(family.patterns) for key, family in DEVICE_PLUGIN_FAMILIES.items()
    }
    if text is None or not text.strip():
        return {key: tuple(value) for key, value in patterns.items()}
    seen: dict[str, list[str]] = {}
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        family, separator, glob = entry.partition("=")
        family = family.strip()
        glob = glob.strip()
        if not separator or family not in patterns or not glob:
            raise ValueError(
                "GPU_FAULT_DEVICE_PLUGIN_SOCKET_PATTERNS entries must be "
                f"<gpu|efa>=<glob>, got {entry!r}"
            )
        seen.setdefault(family, []).append(glob)
    patterns.update(seen)
    return {key: tuple(value) for key, value in patterns.items()}
