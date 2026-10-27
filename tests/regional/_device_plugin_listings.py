"""The device-plugin DaemonSets of both GPU clusters, as ``kubectl get daemonset
-A`` listed them read-only on 2026-10-27, for the GF-GENERAL-LIVE-014 tests.

Plain EKS: ``kube-system/aws-efa-k8s-device-plugin`` (selector ``name=aws-efa-
k8s-device-plugin``, Helm labels ``app.kubernetes.io/name=aws-efa-k8s-device-
plugin``) and the upstream k8s-device-plugin chart as ``nvdp`` -- the plugin
with its ``gpu-feature-discovery`` and ``mps-control-daemon`` siblings, all
three labelled alike and running the same image. HyperPod: the
``hyperpod-dependencies`` release of both charts, the EFA DaemonSet named
``hyperpod-dependencies-aws-efa-k8s-device-plugin`` (selector ``name=<that>``,
the same ``app.kubernetes.io/name`` identity). Image registries are
placeholders; names, labels, versions and mounts are the live ones.
"""

from __future__ import annotations

from typing import Any

DEVICE_PLUGINS_MOUNT = {
    "name": "device-plugin",
    "hostPath": {"path": "/var/lib/kubelet/device-plugins"},
}
EKS_LABELS = {"eks.amazonaws.com/nodegroup": "gpu-ng"}
HP_LABELS = {
    "sagemaker.amazonaws.com/compute-type": "hyperpod",
    "sagemaker.amazonaws.com/cluster-name": "hp-cluster",
}
EFA_IMAGE = "registry.example/eks/aws-efa-k8s-device-plugin"
NVDP_IMAGE = "nvcr.io/nvidia/k8s-device-plugin"
#: The EFA DaemonSet's UID on either cluster: the 014 tests' restart scenario
#: re-owns the replacement Pod by the fake cluster's DaemonSet UID.
EFA_UID = "ds-uid"


def chart_daemonset(
    name: str,
    *,
    labels: dict[str, str],
    image: str,
    volumes: list[dict[str, Any]],
    uid: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "namespace": "kube-system",
            "name": name,
            "uid": uid,
            "labels": labels,
        },
        "spec": {
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [{"name": name, "image": image}],
                    "volumes": volumes,
                },
            },
        },
        "status": {"desiredNumberScheduled": 1},
    }


def nvdp_daemonsets(release: str, version: str, *, gfd: bool) -> list[dict[str, Any]]:
    """The upstream k8s-device-plugin chart as deployed: the plugin and its
    mps-control-daemon / gpu-feature-discovery siblings, labelled alike and
    (since v0.15) running the same image; only the plugin mounts the kubelet
    device-plugins directory."""

    labels = {
        "app.kubernetes.io/instance": release,
        "app.kubernetes.io/name": "nvidia-device-plugin",
    }
    image = f"{NVDP_IMAGE}:{version}"
    items = [
        chart_daemonset(
            f"{release}-nvidia-device-plugin",
            labels=labels,
            image=image,
            volumes=[DEVICE_PLUGINS_MOUNT],
            uid=f"{release}-plugin-uid",
        ),
        chart_daemonset(
            f"{release}-nvidia-device-plugin-mps-control-daemon",
            labels=labels,
            image=image,
            volumes=[
                {"name": "mps-root", "hostPath": {"path": "/run/nvidia/mps"}},
                {"name": "mps-shm", "hostPath": {"path": "/run/nvidia/mps/shm"}},
            ],
            uid=f"{release}-mps-uid",
        ),
    ]
    if gfd:
        items.append(
            chart_daemonset(
                f"{release}-nvidia-device-plugin-gpu-feature-discovery",
                labels=labels,
                image=image,
                volumes=[
                    {
                        "name": "output-dir",
                        "hostPath": {
                            "path": "/etc/kubernetes/node-feature-discovery/features.d"
                        },
                    },
                    {"name": "host-sys", "hostPath": {"path": "/sys"}},
                ],
                uid=f"{release}-gfd-uid",
            )
        )
    return items


def eks_daemonsets() -> list[dict[str, Any]]:
    """kube-system of the plain-EKS cluster."""

    return [
        chart_daemonset(
            "aws-efa-k8s-device-plugin",
            labels={
                "name": "aws-efa-k8s-device-plugin",
                "app.kubernetes.io/name": "aws-efa-k8s-device-plugin",
                "app.kubernetes.io/instance": "aws-efa-k8s-device-plugin",
                "app.kubernetes.io/managed-by": "Helm",
            },
            image=f"{EFA_IMAGE}:v0.5.21-eksbuild.13",
            volumes=[DEVICE_PLUGINS_MOUNT],
            uid=EFA_UID,
        ),
        *nvdp_daemonsets("nvdp", "v0.17.1", gfd=True),
    ]


def hp_daemonsets() -> list[dict[str, Any]]:
    """kube-system of the HyperPod cluster."""

    return [
        chart_daemonset(
            "hyperpod-dependencies-aws-efa-k8s-device-plugin",
            labels={
                "name": "hyperpod-dependencies-aws-efa-k8s-device-plugin",
                "app.kubernetes.io/name": "aws-efa-k8s-device-plugin",
                "app.kubernetes.io/instance": "hyperpod-dependencies",
            },
            image=f"{EFA_IMAGE}:v0.5.12",
            volumes=[DEVICE_PLUGINS_MOUNT],
            uid=EFA_UID,
        ),
        *nvdp_daemonsets("hyperpod-dependencies", "v0.16.1", gfd=False),
    ]
