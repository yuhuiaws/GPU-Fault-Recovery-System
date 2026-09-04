from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
    run_parallel,
    safe_name,
)
from gpu_fault.admin.notifications import NotificationRouting


def revalidate_pod_identity_agent(
    runner: CommandRunner,
    state: BootstrapState,
    cpu: ClusterIdentity,
    site_id: str,
    ensure: Callable[[CommandRunner, ClusterIdentity, str], Any],
) -> None:
    run_parallel(
        {"pod_identity_agent": lambda: ensure(runner, cpu, site_id)},
        state=state,
        probes={
            "pod_identity_agent": lambda: ensure(
                ReadOnlyProbeRunner(runner), cpu, site_id
            )
        },
        revalidate=frozenset({"pod_identity_agent"}),
    )


def run_platform_prerequisite_tasks(
    *,
    runner: CommandRunner,
    state: BootstrapState,
    repository_root: Path,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    cpu_kubeconfig: Path,
    gpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    monitoring: dict[str, Any],
    adot_image: str,
    alert_email: str | None,
    release_manifest: Path,
    runtime_image: str,
    aurora: dict[str, Any],
    fleet_master_file: Path,
) -> None:
    from gpu_fault.admin.bootstrap_services import (
        install_aurora_refresh,
        install_monitoring,
        provision_node_action_keys,
    )

    def build_tasks(
        task_runner: CommandRunner,
        probe_only: bool,
    ) -> dict[str, Callable[[], Any]]:
        tasks: dict[str, Callable[[], Any]] = {
            "monitoring_install": lambda: install_monitoring(
                task_runner,
                repository_root=repository_root,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                namespace=namespace,
                site_id=site_id,
                monitoring=monitoring,
                adot_image=adot_image,
                alert_email=alert_email,
                probe_only=probe_only,
            ),
            "aurora_refresh": lambda: install_aurora_refresh(
                task_runner,
                repository_root=repository_root,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                namespace=namespace,
                site_id=site_id,
                release_manifest=release_manifest,
                runtime_image=runtime_image,
                aurora=aurora,
                probe_only=probe_only,
            ),
        }
        for cluster in gpu_clusters:
            cluster_id = safe_name(cluster.hyperpod_name)

            def node_key_task(
                cluster: ClusterIdentity = cluster,
                cluster_id: str = cluster_id,
            ) -> dict[str, str]:
                return provision_node_action_keys(
                    task_runner,
                    repository_root=repository_root,
                    cpu_kubeconfig=cpu_kubeconfig,
                    gpu_kubeconfig=gpu_kubeconfig,
                    namespace=namespace,
                    cluster=cluster,
                    cluster_id=cluster_id,
                    fleet_master_file=fleet_master_file,
                    probe_only=probe_only,
                )

            tasks[f"node_keys:{cluster_id}"] = node_key_task
        return tasks

    tasks = build_tasks(runner, False)
    # These three tasks converge live resources whose desired state no static
    # digest can fully describe: node membership changes on its own and manifests
    # can be edited out of band. Their digests therefore no longer include the
    # release identity, and every completed task is re-proved by a read-only
    # probe that enters ensure only on detected drift.
    run_parallel(
        tasks,
        state=state,
        probes=build_tasks(ReadOnlyProbeRunner(runner), True),
        revalidate=frozenset(tasks),
    )


def run_foundation_tasks(
    *,
    runner: CommandRunner,
    state: BootstrapState,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    state_dir: Path,
    admin_email: str,
    routing: NotificationRouting,
    aurora_capacity: Any,
    ensure_nlb_network: Callable[..., Any],
    ensure_pki: Callable[..., Any],
    ensure_aurora: Callable[..., Any],
) -> dict[str, Any]:
    from gpu_fault.admin.bootstrap_load_balancer import (
        ensure_load_balancer_controller,
    )
    from gpu_fault.admin.bootstrap_services import ensure_executor_role
    from gpu_fault.admin.notification_bootstrap import (
        notification_bootstrap_tasks,
    )

    def build_tasks(
        active_runner: CommandRunner,
        active_state: BootstrapState | None,
    ) -> dict[str, Callable[[], Any]]:
        executor_tasks = {
            f"executor_role:{safe_name(cluster.hyperpod_name)}": (
                lambda cluster=cluster: ensure_executor_role(
                    active_runner,
                    cluster=cluster,
                    namespace=namespace,
                    site_id=site_id,
                )
            )
            for cluster in gpu_clusters
        }
        return {
            "nlb_network": lambda: ensure_nlb_network(
                active_runner,
                cpu=cpu,
                gpu_clusters=gpu_clusters,
                site_id=site_id,
            ),
            "pki": lambda: ensure_pki(
                active_runner,
                cpu=cpu,
                gpu_clusters=gpu_clusters,
                state_dir=state_dir,
                site_id=site_id,
            ),
            "aurora": lambda: ensure_aurora(
                active_runner,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                namespace=namespace,
                site_id=site_id,
                capacity=aurora_capacity,
            ),
            "load_balancer_controller": lambda: ensure_load_balancer_controller(
                active_runner,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                state_dir=state_dir,
                site_id=site_id,
            ),
            **notification_bootstrap_tasks(
                active_runner,
                state=active_state,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                namespace=namespace,
                site_id=site_id,
                admin_email=admin_email,
                routing=routing,
            ),
            **executor_tasks,
        }

    tasks = build_tasks(runner, state)
    probes = build_tasks(ReadOnlyProbeRunner(runner), None)
    executor_tasks = {name for name in tasks if name.startswith("executor_role:")}
    return run_parallel(
        tasks,
        state=state,
        probes=probes,
        revalidate=frozenset(
            {
                "load_balancer_controller",
                "control_plane_role",
                "email_notifications",
                "monitoring_resources",
                *executor_tasks,
            }
        ),
    )
