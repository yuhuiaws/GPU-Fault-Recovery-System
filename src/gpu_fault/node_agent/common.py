from __future__ import annotations

import re
from pathlib import Path

from gpu_fault.cluster_scope import (
    FABRIC_MANAGER_START_TIMEOUT_SECONDS,
    FABRIC_MANAGER_UNIT,
)

SYSTEMD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@:-]{1,128}$")

PROCESS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,15}$")

NVIDIA_GPU_DEVICE_PATTERN = re.compile(r"^/dev/nvidia\d+$")

# The persistence daemon's unit, and its ``comm`` (cut at 15 characters).
PERSISTENCE_DAEMON_UNIT = "nvidia-persistenced"
PERSISTENCE_DAEMON_PROCESS = "nvidia-persiste"
# Our oneshot ``nvidia-smi -pm 1``: the restore re-enables what the quiesce
# released through it, in whichever regime (daemon or legacy) the node runs.
PERSISTENCE_RESTORE_UNIT = "gpu-fault-gpu-persistence"
# Fabric Manager's ``comm`` (``nv-fabricmanager`` cut at 15 characters); its
# unit is ``cluster_scope.FABRIC_MANAGER_UNIT``.
FABRIC_MANAGER_PROCESS = "nv-fabricmanage"
# The NVLink Subnet Manager's ``comm`` (``/opt/nvidia/nvlsm/sbin/nvlsm``). On NVL5
# (HGX B200/B300) the Fabric Manager unit's start script starts it before
# Fabric Manager, in the same unit (NVIDIA ships no unit of its own for it), and
# the unit's stop stops both: signalled, the fabric loses its subnet manager.
NVLSM_PROCESS = "nvlsm"
# What the quiesce never signals by name, and no configured list may name.
FABRIC_PROCESSES = frozenset({FABRIC_MANAGER_PROCESS, NVLSM_PROCESS})
# How long one ``systemctl start``/``restart`` of a unit may take.
UNIT_START_TIMEOUT_SECONDS = 120


def unit_start_timeout_seconds(service: str) -> int:
    """120 s for a unit, 780 s for Fabric Manager's (``cluster_scope``)."""

    return (
        FABRIC_MANAGER_START_TIMEOUT_SECONDS
        if service.removesuffix(".service") == FABRIC_MANAGER_UNIT
        else UNIT_START_TIMEOUT_SECONDS
    )


#: What one GPU's driver initialisation may take when nothing holds the GPU.
#: With persistence mode off (the quiesce window) a GPU stays initialised only
#: while a client holds it -- on the AL2023 NVIDIA AMI the display stack's DRM
#: device does -- and every ``nvidia-smi`` or ``modprobe nvidia_drm`` after
#: that initialises every GPU of the node, one after another (live,
#: gpu-eks-cluster p5en: ``[nvidia-drm] Loading driver`` 2.2 s apart per H200
#: once the busy reset had unloaded the display stack). Bounds that cover a
#: whole node are sized from this, about twice the measured time.
COLD_GPU_INIT_BOUND_SECONDS = 5
#: The largest node shape (p5, p5en, p6: eight GPUs), for a node whose driver
#: GPU list cannot be read: the bound errs long, never short.
DEFAULT_NODE_GPU_COUNT = 8


def nvidia_gpu_count(proc_root: Path | str | None) -> int | None:
    """The GPUs the NVIDIA driver manages, from ``<proc>/driver/nvidia/gpus``.

    One directory per PCI device, listed whether or not the GPU is initialised,
    and read without spawning anything; ``None`` when there is no such list.
    """

    if proc_root is None:
        return None
    try:
        count = sum(1 for _ in (Path(proc_root) / "driver/nvidia/gpus").iterdir())
    except OSError:
        return None
    return count or None


DEFAULT_QUIESCE_SERVICES = (
    "nvidia-fabricmanager",
    "nvidia-dcgm",
    PERSISTENCE_DAEMON_UNIT,
    PERSISTENCE_RESTORE_UNIT,
    "gpu-fault-metrics-collector",
    "gpu-fault-host-collector",
    "kubelet",
)

DEFAULT_QUIESCE_PROCESSES: tuple[str, ...] = ()

DEFAULT_DEVICE_SWEEP_PROCESSES = (
    PERSISTENCE_DAEMON_PROCESS,
    "nv-hostengine",
)

# Selectors are ``namespace/container``; one that matches no running container
# (the HyperPod health-monitoring-agent on a plain EKS node) is simply skipped
# by the quiesce manager, so the same default list serves every provider.
DEFAULT_QUIESCE_CONTAINERS = (
    "aws-hyperpod/health-monitoring-agent",
    "gpu-fault-system/exporter",
    "kube-system/nvidia-device-plugin-ctr",
)
