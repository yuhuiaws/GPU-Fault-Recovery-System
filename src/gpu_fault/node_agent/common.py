from __future__ import annotations

import re

SYSTEMD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@:-]{1,128}$")

PROCESS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,15}$")

NVIDIA_GPU_DEVICE_PATTERN = re.compile(r"^/dev/nvidia\d+$")

DEFAULT_QUIESCE_SERVICES = (
    "nvidia-fabricmanager",
    "nvidia-dcgm",
    "nvidia-persistenced",
    "gpu-fault-gpu-persistence",
    "gpu-fault-metrics-collector",
    "gpu-fault-host-collector",
    "kubelet",
)

DEFAULT_QUIESCE_PROCESSES: tuple[str, ...] = ()

DEFAULT_DEVICE_SWEEP_PROCESSES = (
    "nvidia-persiste",
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
