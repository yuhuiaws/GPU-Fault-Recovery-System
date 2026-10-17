"""Read-only Kubernetes findings for a plain EKS GPU cluster before it joins.

``prepare-eks-nodegroup`` changes node groups only; what a join needs from the
cluster itself is reported and never changed (administrator operations
§5.0.1, plain-eks-gpu-cluster §3):

* every allowed workload namespace exists (the join creates only the
  control-plane namespace; a missing ``training`` failed a live join deep in
  its release);
* the NVIDIA device plugin (kube-system, ``app.kubernetes.io/name=
  nvidia-device-plugin``, the DaemonSet that mounts the kubelet device-plugins
  directory -- the chart's GFD and MPS DaemonSets carry the same labels) and
  the EFA device plugin (``name=aws-efa-k8s-device-plugin``) run: the recovery
  steps restart exactly these;
* no GPU client DaemonSet tolerates the quiesce taint ``gpu-fault.io/quiesce``
  (``NoExecute``): a key-less ``operator: Exists`` toleration keeps its Pods
  on the node through the quiesce window and the GPU reset then finds the
  device in use.

The reads go through a private kubeconfig ``aws eks update-kubeconfig``
writes into a temporary directory with the operator's credentials, so they
work before any site exists. A cluster the operator cannot read (no access
entry) yields ``available: false`` and the reason: findings never block.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.adapters.common import QUIESCE_TAINT
from gpu_fault.adapters.kubernetes.device_plugin_health import (
    PLUGIN_NAMESPACE,
    PLUGIN_SELECTOR,
)
from gpu_fault.adapters.kubernetes.node_operations import mounts_kubelet_device_plugins
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.cluster_scope import ClusterProvider
from gpu_fault.node_provider import EFA_DEVICE_PLUGIN_NAMES

__all__ = ["finding_lines", "kubernetes_findings", "quiesce_tolerations"]

KUBECONFIG_ALIAS = "gpu-fault-prepare-eks-nodegroup"
EFA_PLUGIN_NAME = EFA_DEVICE_PLUGIN_NAMES[ClusterProvider.EKS_MANAGED]
#: What marks a DaemonSet as a GPU client: it asks for GPUs, is handed every
#: GPU by the NVIDIA runtime, or reaches the device files or the device-plugin
#: socket directory. Images are a hint for the agents that do it by default.
_GPU_IMAGE_HINTS = ("nvidia", "dcgm", "gpu")
_GPU_HOST_PATHS = ("/dev/nvidia", "/var/lib/kubelet/device-plugins", "/run/nvidia")


def _labels(selector: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in selector.split(",") if "=" in item)


def quiesce_tolerations(tolerations: object) -> list[str]:
    """The tolerations in ``tolerations`` that tolerate the quiesce taint.

    The taint is ``gpu-fault.io/quiesce=<value>:NoExecute``: a toleration
    matches it when its key is empty with ``operator: Exists`` or equals the
    key, and its effect is empty or ``NoExecute``. ``tolerationSeconds`` only
    bounds how long the Pod stays, so it is named rather than excused.
    """

    matched: list[str] = []
    for item in tolerations if isinstance(tolerations, list) else ():
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "")
        operator = str(item.get("operator") or "Equal")
        effect = str(item.get("effect") or "")
        if effect not in ("", "NoExecute"):
            continue
        if (not key and operator == "Exists") or key == QUIESCE_TAINT:
            shown = f"key={key or '<any>'} operator={operator}"
            if effect:
                shown += f" effect={effect}"
            if item.get("tolerationSeconds") is not None:
                shown += f" tolerationSeconds={item['tolerationSeconds']}"
            matched.append(shown)
    return matched


def _gpu_signals(pod: Mapping[str, Any]) -> list[str]:
    signals: list[str] = []
    containers = [
        item
        for key in ("initContainers", "containers")
        for item in pod.get(key) or ()
        if isinstance(item, Mapping)
    ]
    for container in containers:
        resources = container.get("resources") or {}
        for kind in ("limits", "requests"):
            if "nvidia.com/gpu" in (resources.get(kind) or {}):
                signals.append(f"{container.get('name')} {kind} nvidia.com/gpu")
        for variable in container.get("env") or ():
            if (
                isinstance(variable, Mapping)
                and variable.get("name") == "NVIDIA_VISIBLE_DEVICES"
                and str(variable.get("value") or "").lower() not in ("", "void", "none")
            ):
                signals.append(f"{container.get('name')} NVIDIA_VISIBLE_DEVICES")
        image = str(container.get("image") or "").lower()
        if any(hint in image for hint in _GPU_IMAGE_HINTS):
            signals.append(f"{container.get('name')} image {container.get('image')}")
    for volume in pod.get("volumes") or ():
        path = str(((volume or {}).get("hostPath") or {}).get("path") or "")
        if path.startswith(_GPU_HOST_PATHS):
            signals.append(f"hostPath {path}")
    if str(pod.get("runtimeClassName") or "") == "nvidia":
        signals.append("runtimeClassName nvidia")
    return signals


def _plugin_status(item: Mapping[str, Any]) -> dict[str, Any]:
    status = item.get("status") or {}
    desired = int(status.get("desiredNumberScheduled") or 0)
    ready = int(status.get("numberReady") or 0)
    metadata = item.get("metadata") or {}
    return {
        "daemonset": f"{metadata.get('namespace')}/{metadata.get('name')}",
        "desired": desired,
        "ready": ready,
        "running": desired > 0 and ready == desired,
    }


def _device_plugins(daemonsets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nvidia_labels = _labels(PLUGIN_SELECTOR)
    nvidia: list[dict[str, Any]] = []
    efa: list[dict[str, Any]] = []
    for item in daemonsets:
        metadata = item.get("metadata") or {}
        if metadata.get("namespace") != PLUGIN_NAMESPACE:
            continue
        template = (item.get("spec") or {}).get("template") or {}
        labels = (template.get("metadata") or {}).get("labels") or {}
        pod = template.get("spec") or {}
        if all(labels.get(key) == value for key, value in nvidia_labels.items()):
            if mounts_kubelet_device_plugins(pod.get("volumes")) is not False:
                nvidia.append(_plugin_status(item))
        elif labels.get("name") == EFA_PLUGIN_NAME:
            efa.append(_plugin_status(item))
    return {
        "nvidia": {
            "selector": f"{PLUGIN_NAMESPACE} {PLUGIN_SELECTOR}",
            "daemonsets": nvidia,
            "running": any(entry["running"] for entry in nvidia),
        },
        "efa": {
            "selector": f"{PLUGIN_NAMESPACE} name={EFA_PLUGIN_NAME}",
            "daemonsets": efa,
            "running": any(entry["running"] for entry in efa),
        },
    }


def _quiesce_tolerating_gpu_clients(
    daemonsets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in daemonsets:
        metadata = item.get("metadata") or {}
        pod = ((item.get("spec") or {}).get("template") or {}).get("spec") or {}
        tolerating = quiesce_tolerations(pod.get("tolerations"))
        signals = _gpu_signals(pod)
        if tolerating and signals:
            findings.append(
                {
                    "daemonset": f"{metadata.get('namespace')}/{metadata.get('name')}",
                    "tolerations": tolerating,
                    "gpu_signals": signals,
                }
            )
    return findings


def _kubectl(kubeconfig: Path, *arguments: str) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        KUBECONFIG_ALIAS,
        *arguments,
    ]


def _items(runner: CommandRunner, command: list[str]) -> list[Mapping[str, Any]]:
    try:
        document = json.loads(runner.run(command, capture=True, timeout_seconds=90))
    except json.JSONDecodeError as exc:
        raise BootstrapError(f"{' '.join(command[5:8])} returned no JSON") from exc
    items = document.get("items") if isinstance(document, Mapping) else None
    return [item for item in items or () if isinstance(item, Mapping)]


def kubernetes_findings(
    runner: CommandRunner,
    *,
    region: str,
    eks_name: str,
    namespaces: Sequence[str],
    control_plane_namespace: str,
) -> dict[str, Any]:
    """The report; ``available: false`` with the reason when unreadable."""

    required = sorted(set(namespaces) - {control_plane_namespace})
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-fault-prepare-") as directory:
            kubeconfig = Path(directory) / "kubeconfig"
            runner.run(
                [
                    "aws",
                    "eks",
                    "update-kubeconfig",
                    "--region",
                    region,
                    "--name",
                    eks_name,
                    "--kubeconfig",
                    str(kubeconfig),
                    "--alias",
                    KUBECONFIG_ALIAS,
                ],
                capture=True,
            )
            names = {
                str((item.get("metadata") or {}).get("name") or "")
                for item in _items(
                    runner, _kubectl(kubeconfig, "get", "namespaces", "-o", "json")
                )
            }
            daemonsets = _items(
                runner,
                _kubectl(
                    kubeconfig, "get", "daemonsets", "--all-namespaces", "-o", "json"
                ),
            )
    except BootstrapError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "remedy": (
                "the operator identity needs Kubernetes read access to the cluster "
                "(an EKS access entry) for these checks; they never block"
            ),
        }
    missing = [name for name in required if name not in names]
    return {
        "available": True,
        "namespaces": {
            "required": required,
            "missing": missing,
            "control_plane_namespace": control_plane_namespace,
            "remedy": (
                [f"kubectl create namespace {name}" for name in missing]
                if missing
                else []
            ),
        },
        "device_plugins": _device_plugins(daemonsets),
        "quiesce_tolerating_gpu_daemonsets": _quiesce_tolerating_gpu_clients(
            daemonsets
        ),
    }


def finding_lines(findings: Mapping[str, Any]) -> list[str]:
    """One human line per thing that needs the operator's attention."""

    if not findings.get("available"):
        return [f"Kubernetes checks unavailable: {findings.get('reason')}"]
    lines: list[str] = []
    missing = findings["namespaces"]["missing"]
    if missing:
        lines.append(
            "allowed workload namespace(s) missing: "
            + ", ".join(missing)
            + " (kubectl create namespace <name>, or join with --allowed-namespace)"
        )
    for kind in ("nvidia", "efa"):
        plugin = findings["device_plugins"][kind]
        if not plugin["running"]:
            lines.append(
                f"{kind.upper()} device plugin not running ({plugin['selector']}: "
                + (
                    ", ".join(
                        f"{entry['daemonset']} ready {entry['ready']}/{entry['desired']}"
                        for entry in plugin["daemonsets"]
                    )
                    or "no DaemonSet"
                )
                + ")"
            )
    for entry in findings["quiesce_tolerating_gpu_daemonsets"]:
        lines.append(
            f"GPU client DaemonSet {entry['daemonset']} tolerates {QUIESCE_TAINT} ("
            + "; ".join(entry["tolerations"])
            + "): list toleration keys such as nvidia.com/gpu instead of a "
            "key-less operator: Exists"
        )
    return lines
