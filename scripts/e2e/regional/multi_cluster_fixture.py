from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)


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
    if not isinstance(clusters, list):
        raise RegionalFixtureError("multi-cluster registry probe returned no list")
    return [dict(item) for item in clusters if isinstance(item, dict)]


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
        "" not in eks_arns
        and "" not in hyperpod_names
        and len(eks_arns) == 2
        and len(hyperpod_names) == 2
    )
