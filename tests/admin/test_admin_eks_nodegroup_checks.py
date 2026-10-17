"""Kubernetes findings of ``prepare-eks-nodegroup``: reported, never changed.

Allowed workload namespaces, the NVIDIA and EFA device plugin DaemonSets, and
GPU client DaemonSets tolerating the ``gpu-fault.io/quiesce`` taint.
"""

from __future__ import annotations

import pytest

from gpu_fault.admin import eks_nodegroup_checks as checks
from tests.admin._eks_nodegroup_world import (
    GPU_NAME,
    REGION,
    FakeAws,
    World,
    daemonset,
    efa_plugin,
    nvidia_plugin,
)


def _findings(world: World, namespaces=("gpu-fault-system", "training")) -> dict:
    return checks.kubernetes_findings(
        FakeAws(world),
        region=REGION,
        eks_name=GPU_NAME,
        namespaces=namespaces,
        control_plane_namespace="gpu-fault-system",
    )


def test_a_ready_cluster_has_no_finding_and_is_read_through_a_private_kubeconfig() -> (
    None
):
    world = World()

    report = _findings(world)

    assert report["available"] is True
    assert report["namespaces"] == {
        "required": ["training"],
        "missing": [],
        "control_plane_namespace": "gpu-fault-system",
        "remedy": [],
    }, "the control-plane namespace is the join's own to create"
    assert report["device_plugins"]["nvidia"]["running"] is True
    assert report["device_plugins"]["efa"]["running"] is True
    assert report["quiesce_tolerating_gpu_daemonsets"] == []
    assert checks.finding_lines(report) == []
    (kubeconfig,) = [argv for argv, _flag in world.calls if "update-kubeconfig" in argv]
    assert kubeconfig[kubeconfig.index("--kubeconfig") + 1].endswith("/kubeconfig"), (
        "the kubeconfig lives in a private temporary directory"
    )
    assert not any(flag for _argv, flag in world.calls), "every read is a read"
    assert all(
        argv[argv.index("--context") + 1] == checks.KUBECONFIG_ALIAS
        for argv, _flag in world.calls
        if argv[0] == "kubectl"
    ), "every kubectl call names the private context"


def test_missing_namespaces_and_stopped_plugins_are_named() -> None:
    world = World()
    world.namespaces = ["default", "kube-system"]
    world.daemonsets = [nvidia_plugin(ready=1, desired=2)]

    report = _findings(world)

    assert report["namespaces"]["missing"] == ["training"]
    assert report["namespaces"]["remedy"] == ["kubectl create namespace training"]
    assert checks.finding_lines(report) == [
        "allowed workload namespace(s) missing: training (kubectl create namespace "
        "<name>, or join with --allowed-namespace)",
        "NVIDIA device plugin not running (kube-system app.kubernetes.io/name="
        "nvidia-device-plugin: kube-system/nvdp-nvidia-device-plugin ready 1/2)",
        "EFA device plugin not running (kube-system name=aws-efa-k8s-device-plugin: "
        "no DaemonSet)",
    ]


def test_the_chart_companions_with_the_plugin_labels_are_not_the_plugin() -> None:
    world = World()
    gfd = daemonset(
        "kube-system",
        "nvdp-nvidia-device-plugin-gpu-feature-discovery",
        labels={"app.kubernetes.io/name": "nvidia-device-plugin"},
        volumes=[
            {
                "name": "features",
                "hostPath": {
                    "path": "/etc/kubernetes/node-feature-discovery/features.d"
                },
            }
        ],
    )
    world.daemonsets = [gfd, efa_plugin()]

    report = _findings(world)

    assert report["device_plugins"]["nvidia"] == {
        "selector": "kube-system app.kubernetes.io/name=nvidia-device-plugin",
        "daemonsets": [],
        "running": False,
    }, "only the DaemonSet mounting the kubelet device-plugins directory counts"


@pytest.mark.parametrize(
    ("toleration", "flagged"),
    [
        ({"operator": "Exists"}, True),
        ({"operator": "Exists", "effect": "NoExecute"}, True),
        ({"key": "gpu-fault.io/quiesce", "operator": "Exists"}, True),
        ({"operator": "Exists", "effect": "NoSchedule"}, False),
        ({"key": "nvidia.com/gpu", "operator": "Exists"}, False),
    ],
)
def test_gpu_clients_tolerating_the_quiesce_taint_are_findings(
    toleration: dict, flagged: bool
) -> None:
    world = World()
    exporter = daemonset(
        "monitoring",
        "dcgm-exporter",
        labels={"app": "dcgm-exporter"},
        tolerations=[toleration],
        containers=[
            {
                "name": "exporter",
                "image": "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1",
                "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}],
            }
        ],
    )
    world.daemonsets = [nvidia_plugin(), efa_plugin(), exporter]

    found = _findings(world)["quiesce_tolerating_gpu_daemonsets"]

    if flagged:
        assert [item["daemonset"] for item in found] == ["monitoring/dcgm-exporter"]
        assert "exporter NVIDIA_VISIBLE_DEVICES" in found[0]["gpu_signals"]
    else:
        assert found == []


def test_wildcard_tolerating_non_gpu_daemonsets_are_not_findings() -> None:
    world = World()
    world.daemonsets.append(
        daemonset(
            "kube-system",
            "kube-proxy",
            labels={"k8s-app": "kube-proxy"},
            tolerations=[{"operator": "Exists"}],
            containers=[{"name": "kube-proxy", "image": "eks/kube-proxy:v1.32"}],
        )
    )

    assert _findings(world)["quiesce_tolerating_gpu_daemonsets"] == []


def test_toleration_seconds_are_named_rather_than_excused() -> None:
    assert checks.quiesce_tolerations(
        [{"operator": "Exists", "effect": "NoExecute", "tolerationSeconds": 300}]
    ) == ["key=<any> operator=Exists effect=NoExecute tolerationSeconds=300"]
    assert checks.quiesce_tolerations("not a list") == []


def test_a_cluster_the_operator_cannot_read_reports_why_and_never_blocks() -> None:
    world = World()
    world.kubernetes_error = "error: You must be logged in to the server (Unauthorized)"

    report = _findings(world)

    assert report["available"] is False
    assert "Unauthorized" in report["reason"]
    assert "access entry" in report["remedy"]
    assert checks.finding_lines(report) == [
        "Kubernetes checks unavailable: error: You must be logged in to the server "
        "(Unauthorized)"
    ]
