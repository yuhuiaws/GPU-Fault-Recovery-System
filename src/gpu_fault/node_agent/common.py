from __future__ import annotations

import re

SYSTEMD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@:-]{1,128}$")

PROCESS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,15}$")

NVIDIA_GPU_DEVICE_PATTERN = re.compile(r"^/dev/nvidia\d+$")

# The persistence daemon's unit, and its ``comm`` (cut at 15 characters).
PERSISTENCE_DAEMON_UNIT = "nvidia-persistenced"
PERSISTENCE_DAEMON_PROCESS = "nvidia-persiste"
# Our oneshot ``nvidia-smi -pm 1``: the restore re-enables what the quiesce
# released through it, in whichever regime (daemon or legacy) the node runs.
PERSISTENCE_RESTORE_UNIT = "gpu-fault-gpu-persistence"

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
