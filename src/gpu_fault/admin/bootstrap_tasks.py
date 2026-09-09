"""The bootstrap task graph: what each task needs, and what re-proves it.

Bootstrap used to run two flat phases -- the AWS foundation (network, PKI,
Aurora, roles, notification resources) and then the platform prerequisites
(monitoring, the Aurora credential refresh, node keys) -- so every platform task
waited for the slowest foundation task, which is the Aurora instance wait. The
two builders below now describe one graph: each task names the tasks whose
results it reads, and ``run_parallel`` starts it the moment those hold.

What follows the graph does not move. The release rollout needs the control
plane Pods before the NLB Service, TLS and DNS can be reconciled, and the site
document is built when everything here is done, as before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from gpu_fault.admin.grafana import GrafanaSettings
from gpu_fault.admin.notifications import NotificationRouting

FOUNDATION_READY_PHASE = "aws-infrastructure-ready"
PLATFORM_READY_PHASE = "platform-prerequisites-ready"


@dataclass(frozen=True)
class TaskGraph:
    """Ensures, their read-only probes, which are re-proved every run, and
    what each waits for. ``run`` hands all four to ``run_parallel``."""

    tasks: Mapping[str, Callable[[], Any]]
    probes: Mapping[str, Callable[[], Any]]
    revalidate: frozenset[str]
    dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def merged(self, other: TaskGraph) -> TaskGraph:
        return TaskGraph(
            tasks={**self.tasks, **other.tasks},
            probes={**self.probes, **other.probes},
            revalidate=self.revalidate | other.revalidate,
            dependencies={**self.dependencies, **other.dependencies},
        )

    def run(
        self,
        *,
        state: BootstrapState,
        on_complete: Callable[[frozenset[str]], None] | None = None,
    ) -> dict[str, Any]:
        return run_parallel(
            self.tasks,
            state=state,
            probes=self.probes,
            revalidate=self.revalidate,
            dependencies=self.dependencies,
            on_complete=on_complete,
        )


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


def platform_task_graph(
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
    adot_image: str,
    alert_email: str | None,
    release_manifest: Path,
    runtime_image: str,
    fleet_master_file: Path,
    ensure_aurora_ready: Callable[..., Any],
    grafana: GrafanaSettings | None = None,
) -> TaskGraph:
    """The tasks that used to wait for the whole foundation, with what each one
    really needs.

    ``monitoring_install`` (ADOT, AMP rules, Grafana) reads the AMP workspace
    and SNS topic from ``monitoring_resources`` and creates its own writer role;
    ``aurora_ready`` waits for the writer and reader the ``aurora`` task created
    and hands the control plane its Secret; ``aurora_refresh`` renders that
    Secret's ARN into the CronJob and its verify Job needs the writer, so it
    follows ``aurora_ready``; ``node_keys:*`` need only the kubeconfigs and the
    fleet master the preamble made. Each reads its inputs from ``state`` when it
    starts, which is after its dependencies were recorded.

    All of them converge live resources whose desired state no static digest
    can fully describe -- node membership changes on its own, manifests can be
    edited out of band -- so every completed one is re-proved by a read-only
    probe that enters ensure only on detected drift.
    """

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
                monitoring=state.result("monitoring_resources"),
                adot_image=adot_image,
                alert_email=alert_email,
                probe_only=probe_only,
                grafana=grafana,
            ),
            "aurora_ready": lambda: ensure_aurora_ready(
                task_runner,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                namespace=namespace,
                aurora=state.result("aurora"),
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
                # The readiness result carries the master Secret ARN; a
                # checkpoint written before the split carries it on ``aurora``.
                aurora={**state.result("aurora"), **state.result("aurora_ready")},
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
    return TaskGraph(
        tasks=tasks,
        probes=build_tasks(ReadOnlyProbeRunner(runner), True),
        revalidate=frozenset(tasks),
        dependencies={
            "monitoring_install": ("monitoring_resources",),
            "aurora_ready": ("aurora",),
            "aurora_refresh": ("aurora_ready",),
        },
    )


def foundation_task_graph(
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
    archive_s3_uri: str | None = None,
) -> TaskGraph:
    """The AWS resources with no dependency among themselves.

    The NLB network and the PKI describe the same clusters but read nothing
    from each other (the hostname comes from the hosted zone, not the load
    balancer); the Aurora task only creates -- the instance wait is
    ``aurora_ready`` in the platform graph; the roles, the notification
    resources and the load balancer controller need only the Pod Identity
    add-on the preamble ensured. So every task here starts at once.
    """

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
                archive_s3_uri=archive_s3_uri,
            ),
            **executor_tasks,
        }

    tasks = build_tasks(runner, state)
    executor_tasks = {name for name in tasks if name.startswith("executor_role:")}
    return TaskGraph(
        tasks=tasks,
        probes=build_tasks(ReadOnlyProbeRunner(runner), None),
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


def run_bootstrap_tasks(
    *,
    state: BootstrapState,
    foundation: TaskGraph,
    platform: TaskGraph,
) -> dict[str, Any]:
    """Run both graphs as one, marking the two phases as their subsets complete.

    ``aws-infrastructure-ready`` is recorded the moment the last foundation
    task completes (platform tasks may already be running by then) and
    ``platform-prerequisites-ready`` when everything is done, so the two markers
    that checkpoint readers and the acceptance evidence know keep their meaning.
    """

    foundation_names = frozenset(foundation.tasks)
    marked = False

    def mark_phase(completed: frozenset[str]) -> None:
        nonlocal marked
        if not marked and foundation_names <= completed:
            marked = True
            state.phase(FOUNDATION_READY_PHASE)

    results = foundation.merged(platform).run(state=state, on_complete=mark_phase)
    state.phase(PLATFORM_READY_PHASE)
    return results
