"""Read-only binding to one deployed control-plane component and CPU node."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.e2e.regional.notify008_bundle import PROBES
from scripts.e2e.regional.notify008_fixture import CpuAPI
from scripts.e2e.regional.notify008_resources import cpu_node_errors
from scripts.e2e.regional.probes.notify008_protocol import CASE_ID, ProbeError, Target
from scripts.e2e.regional.regional_case_contract import case_metadata, predecessor_path
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records

APP = "gpu-fault-control-worker"


@dataclass(frozen=True)
class Settings:
    cpu_kubeconfig: Path
    cpu_context: str
    namespace: str
    cluster_id: str
    region: str
    postgres_image: str

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "CPU_EKS_CONTEXT": self.cpu_context,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "GPU_FAULT_CLUSTER_ID": self.cluster_id,
            "AWS_REGION": self.region,
            "NOTIFY008_POSTGRES_IMAGE": self.postgres_image,
        }


def source_target(
    api: CpuAPI, settings: Settings, run_id: str, *, expected: Target | None = None
) -> Target:
    release = api.read(
        "configmap", "gpu-fault-regional-release-state", namespace=settings.namespace
    )
    if release is None:
        raise ProbeError("regional release state is missing")
    state = json.loads(release.get("data", {}).get("state.json", "null"))
    if (
        not isinstance(state, dict)
        or state.get("phase") != "complete"
        or state.get("transaction_committed") is not True
    ):
        raise ProbeError("regional release is not committed")
    deployment = api.read("deployment", APP, namespace=settings.namespace)
    if deployment is None:
        raise ProbeError("deployed control-worker is missing")
    meta = deployment.get("metadata") or {}
    spec, status = deployment.get("spec") or {}, deployment.get("status") or {}
    replicas = spec.get("replicas")
    generation = meta.get("generation")
    if (
        type(replicas) is not int
        or replicas < 1
        or type(generation) is not int
        or generation < 1
        or status.get("observedGeneration") != meta.get("generation")
        or any(
            status.get(key) != replicas
            for key in ("updatedReplicas", "readyReplicas", "availableReplicas")
        )
    ):
        raise ProbeError("deployed control-worker is not fully converged")
    containers = spec.get("template", {}).get("spec", {}).get("containers", [])
    if len(containers) != 1:
        raise ProbeError("deployed control-worker container identity is ambiguous")
    container = containers[0]
    inventory = json.loads(
        api.call(
            "get",
            "pods",
            "-l",
            f"app={APP}",
            "-o",
            "json",
            namespace=settings.namespace,
        )
    )
    ready = ready_pod_records(inventory)
    if len(ready) != replicas:
        raise ProbeError("deployed control-worker readiness inventory differs")
    selected = (
        next((item for item in ready if item["name"] == expected.source_pod), None)
        if expected
        else sorted(ready, key=lambda item: item["name"])[0]
    )
    if selected is None:
        raise ProbeError("approved source Pod disappeared; no automatic replacement")
    pod = api.read("pod", selected["name"], namespace=settings.namespace)
    if pod is None:
        raise ProbeError("source Pod disappeared")
    owners = pod.get("metadata", {}).get("ownerReferences") or []
    if len(owners) != 1 or owners[0].get("kind") != "ReplicaSet":
        raise ProbeError("source Pod is not owned by the selected deployment")
    replica = api.read("replicaset", owners[0]["name"], namespace=settings.namespace)
    replica_owners = (replica or {}).get("metadata", {}).get("ownerReferences") or []
    if (
        replica is None
        or replica["metadata"].get("uid") != owners[0].get("uid")
        or len(replica_owners) != 1
        or replica_owners[0].get("uid") != meta.get("uid")
        or replica_owners[0].get("kind") != "Deployment"
    ):
        raise ProbeError("source Pod deployment ownership differs")
    images = pod.get("status", {}).get("containerStatuses") or []
    if len(images) != 1 or str(images[0].get("imageID", "")).removeprefix(
        "docker-pullable://"
    ) != container.get("image"):
        raise ProbeError("source runtime resolved image does not match its digest pin")
    identity = json.loads(
        api.call(
            "exec",
            selected["name"],
            "-c",
            container["name"],
            "--",
            "/opt/gpu-fault/control-plane/bin/python",
            "-c",
            (PROBES / "notify008_identity.py").read_text(encoding="utf-8"),
            namespace=settings.namespace,
        )
    )
    if (
        not isinstance(identity, dict)
        or identity.get("distribution") != "gpu-fault-control-plane"
    ):
        raise ProbeError("source runtime did not attest its control-plane component")
    node_name = pod.get("spec", {}).get("nodeName", "")
    node = api.read("node", node_name)
    if node is None:
        raise ProbeError("source CPU node is missing")
    target = Target(
        run_id=run_id,
        cluster_id=settings.cluster_id,
        region=settings.region,
        release_id=state.get("release_id", ""),
        node=node_name,
        node_uid=node.get("metadata", {}).get("uid", ""),
        source_pod=selected["name"],
        source_pod_uid=pod["metadata"].get("uid", ""),
        deployment_uid=meta.get("uid", ""),
        deployment_generation=generation,
        runtime_image=container.get("image", ""),
        postgres_image=settings.postgres_image,
        runtime_version=identity.get("version", ""),
        runtime_module_digest=identity.get("module_digest", ""),
    )
    if cpu_node_errors(node, target):
        raise ProbeError("source node lacks positive CPU-only placement proof")
    if expected is not None and target != expected:
        raise ProbeError("approved source runtime or CPU target drifted")
    return target


def sandbox_preflight(api: CpuAPI, target: Target) -> None:
    for kind in ("namespace", "priorityclass"):
        if api.read(kind, target.run_id) is not None:
            raise ProbeError(f"proposed isolated {kind} already exists")


def predecessor(run_dir: Path, target: Target) -> dict[str, Any]:
    metadata = case_metadata(CASE_ID)
    if metadata.risk != "live-non-destructive" or metadata.automation != "manual":
        raise ProbeError("NOTIFY008 catalog risk or automation differs")
    case_id, path = predecessor_path(run_dir, CASE_ID, "")
    if case_id is None or path is None:
        raise ProbeError("NOTIFY008 requires its canonical formal predecessor")
    evidence = predecessor_evidence(
        path, case_id, release_id=target.release_id, cluster_id=target.cluster_id
    )
    if evidence.get("valid") is not True:
        raise ProbeError("canonical predecessor evidence is not valid")
    return evidence
