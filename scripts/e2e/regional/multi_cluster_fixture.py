from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_pod_inventory import ready_pod_records  # noqa: E402


@dataclass(frozen=True)
class ClusterTarget:
    cluster_id: str
    gpu_kubeconfig: Path
    gpu_context: str


@dataclass(frozen=True)
class MultiClusterSettings:
    cpu_kubeconfig: Path
    namespace: str
    region: str
    cluster_a: ClusterTarget
    cluster_b: ClusterTarget

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.region.strip():
            raise ValueError("multi-cluster namespace and region must be explicit")
        if any(
            not target.cluster_id.strip() or not target.gpu_context.strip()
            for target in (self.cluster_a, self.cluster_b)
        ):
            raise ValueError("multi-cluster target identity must be explicit")
        if self.cluster_a.cluster_id == self.cluster_b.cluster_id:
            raise ValueError("multi-cluster fixture requires distinct cluster IDs")
        if (
            self.cluster_a.gpu_context == self.cluster_b.gpu_context
            and self.cluster_a.gpu_kubeconfig == self.cluster_b.gpu_kubeconfig
        ):
            raise ValueError("multi-cluster fixture requires distinct GPU contexts")
        for path in (
            self.cpu_kubeconfig,
            self.cluster_a.gpu_kubeconfig,
            self.cluster_b.gpu_kubeconfig,
        ):
            if not path.is_file():
                raise ValueError(f"kubeconfig does not exist: {path}")

    def regional(self, target: ClusterTarget) -> RegionalLiveFixture:
        if target not in (self.cluster_a, self.cluster_b):
            raise ValueError("target is outside the approved cluster pair")
        return RegionalLiveFixture(
            RegionalLiveSettings(
                cpu_kubeconfig=self.cpu_kubeconfig,
                gpu_kubeconfig=target.gpu_kubeconfig,
                gpu_context=target.gpu_context,
                namespace=self.namespace,
                cluster_id=target.cluster_id,
                region=self.region,
            )
        )

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "GPU_A_KUBECONFIG": str(self.cluster_a.gpu_kubeconfig),
            "GPU_A_CONTEXT": self.cluster_a.gpu_context,
            "GPU_A_CLUSTER_ID": self.cluster_a.cluster_id,
            "GPU_B_KUBECONFIG": str(self.cluster_b.gpu_kubeconfig),
            "GPU_B_CONTEXT": self.cluster_b.gpu_context,
            "GPU_B_CLUSTER_ID": self.cluster_b.cluster_id,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "AWS_REGION": self.region,
        }


REGISTRY_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_ids = sys.argv[1:]
store = ApplicationContext.from_environment().store
result = []
for cluster_id in cluster_ids:
    registration = store.get_regional_cluster(cluster_id)
    result.append({
        "cluster_id": registration.cluster_id,
        "enabled": registration.enabled,
        "synthetic": registration.synthetic,
        "allowed_namespaces": registration.allowed_namespaces,
        "hyperpod_cluster_name": registration.hyperpod_cluster_name,
        "eks_cluster_arn": registration.eks_cluster_arn,
        "token_sha256_length": len(registration.token_sha256),
    })
print(json.dumps({"clusters": result}, sort_keys=True))
"""


def registration_snapshot(
    fixture: RegionalLiveFixture,
    settings: MultiClusterSettings,
) -> list[dict[str, Any]]:
    value = fixture.cpu_python(
        REGISTRY_PROBE,
        settings.cluster_a.cluster_id,
        settings.cluster_b.cluster_id,
    )
    clusters = value.get("clusters")
    if (
        not isinstance(clusters, list)
        or len(clusters) != 2
        or any(not isinstance(item, dict) for item in clusters)
        or {item.get("cluster_id") for item in clusters}
        != {settings.cluster_a.cluster_id, settings.cluster_b.cluster_id}
    ):
        raise RegionalFixtureError(
            "multi-cluster registry probe returned a different cluster pair"
        )
    return [dict(item) for item in clusters]


CONTROL_PLANE_APPS = ("gpu-fault-api-ha", "gpu-fault-control-worker")


def container_statuses(pods_document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{pod: {container: {restart_count, last_terminated_reason}}}``.

    A Pod UID comparison misses a container restart -- the UID survives it --
    and ISO-006 / E2E-002 have to say "no control-plane replica restarted and
    none was OOM-killed", which is what ``restartCount`` and
    ``lastState.terminated.reason`` carry.
    """

    ready = ready_pod_records(pods_document)
    if not ready or len(ready) != len(pods_document["items"]):
        raise RegionalFixtureError(
            "control-plane Pod inventory is not completely Ready"
        )
    result: dict[str, dict[str, Any]] = {}
    for item in pods_document["items"]:
        name = str(item.get("metadata", {}).get("name") or "")
        status = item.get("status") or {}
        containers: dict[str, Any] = {}
        for entry in [
            *(status.get("initContainerStatuses") or []),
            *(status.get("containerStatuses") or []),
        ]:
            count = entry.get("restartCount")
            if type(count) is not int or count < 0:
                raise RegionalFixtureError(
                    "control-plane container restart count is unknown"
                )
            terminated = (entry.get("lastState") or {}).get("terminated") or {}
            containers[str(entry.get("name"))] = {
                "restart_count": count,
                "last_terminated_reason": terminated.get("reason"),
            }
        expected = [
            container["name"]
            for container in [
                *(item["spec"].get("initContainers") or []),
                *item["spec"]["containers"],
            ]
        ]
        if len(set(expected)) != len(expected) or set(containers) != set(expected):
            raise RegionalFixtureError(
                "control-plane container inventory is incomplete"
            )
        if name in result:
            raise RegionalFixtureError("control-plane Pod name is duplicated")
        result[name] = {
            "uid": str(item.get("metadata", {}).get("uid") or ""),
            "phase": status.get("phase"),
            "ready": True,
            "containers": containers,
        }
    return result


def control_plane_container_statuses(
    fixture: RegionalLiveFixture,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for app in CONTROL_PLANE_APPS:
        document = json.loads(
            fixture.kubectl("cpu", "get", "pod", "-l", f"app={app}", "-o", "json")
        )
        for name, value in container_statuses(document).items():
            result[name] = {"app": app, **value}
    return result


def container_status_errors(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[str]:
    """Restarts, OOM kills and replaced Pods between two snapshots."""

    errors = []
    if not before or not after:
        return ["control-plane Pod inventory is empty"]
    for name, earlier in before.items():
        later = after.get(name)
        if later is None:
            errors.append(f"{name}: Pod disappeared")
            continue
        if later.get("uid") != earlier.get("uid"):
            errors.append(f"{name}: Pod was replaced")
        if any(
            not snapshot.get("uid")
            or snapshot.get("phase") != "Running"
            or snapshot.get("ready") is not True
            or not isinstance(snapshot.get("containers"), dict)
            or not snapshot["containers"]
            for snapshot in (earlier, later)
        ):
            errors.append(f"{name}: Pod/container readiness is unproven")
        if set(later.get("containers") or {}) != set(earlier.get("containers") or {}):
            errors.append(f"{name}: container set changed")
        for container, state in earlier.get("containers", {}).items():
            current = later.get("containers", {}).get(container) or {}
            counts = (state.get("restart_count"), current.get("restart_count"))
            if any(type(count) is not int or count < 0 for count in counts):
                errors.append(f"{name}/{container}: restartCount is unknown")
            elif counts[0] != counts[1]:
                errors.append(f"{name}/{container}: restartCount changed")
            if current.get("last_terminated_reason") == "OOMKilled":
                errors.append(f"{name}/{container}: OOMKilled")
    for name in set(after) - set(before):
        errors.append(f"{name}: Pod appeared")
    return errors


# Runs inside an API Pod; /metrics on loopback is public (the "metrics" bucket
# resolves to execution-token only for non-loopback peers), so nothing is sent.
METRICS_PROBE = r"""
import json
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8080/metrics", timeout=20) as response:
    text = response.read().decode()
print(json.dumps({"text": text}))
"""

REJECTION_COUNTERS = (
    "gpu_fault_store_io_rejections_total",
    "gpu_fault_ingress_backpressure_rejections_total",
    "gpu_fault_processor_admission_rejections_total",
)


def parse_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    try:
        return [
            (sample.name, dict(sample.labels), float(sample.value))
            for family in text_string_to_metric_families(text)
            for sample in family.samples
        ]
    except ValueError as exc:
        raise RegionalFixtureError("control-plane metrics are malformed") from exc


def cluster_pressure_reading(text: str, cluster_id: str) -> dict[str, Any]:
    """Queue depth for one cluster and the region's rejection counters."""

    depth: float | None = None
    rejections: dict[str, float | None] = {name: None for name in REJECTION_COUNTERS}
    for name, labels, value in parse_metrics(text):
        if (
            name == "gpu_fault_processor_cluster_queue_depth"
            and labels.get("cluster_id") == cluster_id
        ):
            if not math.isfinite(value) or value < 0 or depth is not None:
                raise RegionalFixtureError(
                    "cluster queue depth is invalid or duplicated"
                )
            depth = value
        elif name in rejections:
            if not math.isfinite(value) or value < 0:
                raise RegionalFixtureError("rejection counter is invalid")
            rejections[name] = (rejections[name] or 0) + value
    return {"cluster_queue_depth": depth, "rejections": rejections}


def control_plane_pressure(
    fixture: RegionalLiveFixture,
    cluster_id: str,
) -> dict[str, Any]:
    text = str(fixture.cpu_python(METRICS_PROBE).get("text") or "")
    return cluster_pressure_reading(text, cluster_id)


def registrations_are_distinct_physical_clusters(
    registrations: list[dict[str, Any]],
) -> bool:
    if len(registrations) != 2:
        return False
    eks_arns = {
        str(item.get("eks_cluster_arn") or "").strip() for item in registrations
    }
    hyperpod_names = {
        str(item.get("hyperpod_cluster_name") or "").strip() for item in registrations
    }
    return (
        len({item.get("cluster_id") for item in registrations}) == 2
        and all(
            isinstance(item.get("cluster_id"), str)
            and item["cluster_id"]
            and item.get("enabled") is True
            and item.get("synthetic") is False
            for item in registrations
        )
        and "" not in eks_arns
        and "" not in hyperpod_names
        and len(eks_arns) == 2
        and len(hyperpod_names) == 2
    )
