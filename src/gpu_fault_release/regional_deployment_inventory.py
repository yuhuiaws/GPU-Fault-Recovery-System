"""Load deployment roles generated from source Manifest annotations."""

from __future__ import annotations

import json

from gpu_fault_release import repository_root

ROOT = repository_root()
RESOURCE_INVENTORY = ROOT / "deploy/control-plane/regional/cleanup-inventory.json"
DOCUMENT = json.loads(RESOURCE_INVENTORY.read_text(encoding="utf-8"))
GPU_RESOURCES = DOCUMENT["gpu"]["resources"]
CPU_RESOURCES = DOCUMENT["cpu"]["resources"]

GPU_ROLLOUT_DEPLOYMENTS = tuple(
    (resource["manifest"], resource["name"])
    for resource in GPU_RESOURCES
    if resource.get("deploy") == "regional_rollout"
)
GPU_RECONCILERS = [
    resource["name"]
    for resource in GPU_RESOURCES
    if resource.get("deploy") == "regional_reconciler"
]
GPU_EXECUTORS = [
    resource["name"]
    for resource in GPU_RESOURCES
    if resource["kind"] == "deployment" and resource["phase"] == "executor"
]
CPU_DEPLOYMENTS = tuple(
    resource["name"] for resource in CPU_RESOURCES if resource["kind"] == "deployment"
)
CPU_INGRESS = [
    resource["name"]
    for resource in CPU_RESOURCES
    if resource["kind"] == "deployment" and resource["phase"] == "ingress"
]
CPU_RUNTIME_DEPLOYMENTS = tuple(
    resource["name"]
    for resource in CPU_RESOURCES
    if resource["kind"] == "deployment" and resource["phase"] in {"ingress", "consumer"}
)

if (
    not GPU_ROLLOUT_DEPLOYMENTS
    or len(GPU_RECONCILERS) != 1
    or len(GPU_EXECUTORS) != 1
    or len(CPU_INGRESS) != 1
):
    raise RuntimeError("cleanup inventory deployment roles are incomplete")

GPU_RECONCILER_DEPLOYMENT = GPU_RECONCILERS[0]
GPU_EXECUTOR_DEPLOYMENT = GPU_EXECUTORS[0]
GPU_WATCHER_DEPLOYMENT = dict(GPU_ROLLOUT_DEPLOYMENTS)["completion-watcher.yaml"]
GPU_COLLECTOR_DEPLOYMENT = dict(GPU_ROLLOUT_DEPLOYMENTS)[
    "kubernetes-node-resource-collector.yaml"
]
CPU_INGRESS_DEPLOYMENT = CPU_INGRESS[0]
DEPLOYMENTS = tuple(name for _, name in GPU_ROLLOUT_DEPLOYMENTS)
