"""CPU, GPU and Pod Identity prerequisites for the bootstrap resource graph."""

from __future__ import annotations

from gpu_fault.admin.bootstrap_task_inputs import task_input_spec

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.bootstrap_tasks import TaskGraph, TaskSpec


@dataclass(frozen=True)
class ClusterAccessPlan:
    graph: TaskGraph
    cpu_kubeconfig: Path
    gpu_kubeconfig: Path
    fleet_master_file: Path


def create_cluster_access_plan(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    state_dir: Path,
    namespace: str,
    secure_dir: Path,
    site_id: str,
    ensure_pod_identity_agent: Callable[[CommandRunner, ClusterIdentity, str], Any],
    update_kubeconfig: Callable[..., None],
    ensure_namespace: Callable[..., None],
    ensure_base_secrets: Callable[..., Path],
) -> ClusterAccessPlan:
    cpu_kubeconfig = state_dir / "cpu.kubeconfig"
    gpu_kubeconfig = state_dir / "gpu.kubeconfig"
    fleet_master_file = secure_dir / "fleet-master"

    def control_plane() -> dict[str, str]:
        update_kubeconfig(runner, cluster=cpu, path=cpu_kubeconfig)
        ensure_namespace(runner, kubeconfig=cpu_kubeconfig, namespace=namespace)
        actual = ensure_base_secrets(
            runner,
            cpu_kubeconfig=cpu_kubeconfig,
            namespace=namespace,
            secure_dir=secure_dir,
        )
        if actual != fleet_master_file:
            raise BootstrapError("unexpected fleet master path from CPU access")
        return {"fleet_master_file": str(actual)}

    def data_plane() -> dict[str, str]:
        # GPU contexts share one kubeconfig; its writes must remain sequential.
        for cluster in gpu_clusters:
            update_kubeconfig(runner, cluster=cluster, path=gpu_kubeconfig)
        for cluster in gpu_clusters:
            ensure_namespace(
                runner,
                kubeconfig=gpu_kubeconfig,
                namespace=namespace,
                context=cluster.context,
            )
        return {}

    graph = TaskGraph(
        {
            "cpu_access": TaskSpec(
                control_plane,
                input_policy=task_input_spec("cpu_access"),
                revalidate=True,
                stop_on_failure=True,
            ),
            "gpu_access": TaskSpec(
                data_plane,
                input_policy=task_input_spec("gpu_access"),
                revalidate=True,
                stop_on_failure=True,
            ),
            "pod_identity_agent": TaskSpec(
                lambda: ensure_pod_identity_agent(runner, cpu, site_id),
                input_policy=task_input_spec("pod_identity_agent"),
                probe=lambda: ensure_pod_identity_agent(
                    ReadOnlyProbeRunner(runner), cpu, site_id
                ),
                revalidate=True,
                stop_on_failure=True,
            ),
        }
    )
    return ClusterAccessPlan(graph, cpu_kubeconfig, gpu_kubeconfig, fleet_master_file)
