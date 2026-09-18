"""The bootstrap task graph: what each task needs, and what re-proves it.

Bootstrap used to run two flat phases -- the AWS foundation (network, PKI,
Aurora, roles, notification resources) and then the platform prerequisites
(monitoring, the Aurora credential refresh, node keys) -- so every platform task
waited for the slowest foundation task, which is the Aurora instance wait. The
two builders below now describe one graph: each task names the tasks whose
results it reads, and ``run_parallel`` starts it the moment those hold.

The site document is built after this graph completes. The release rollout
can then request its NLB Service while the CPU converges, but target health
and DNS publication still require the control-plane Pods.
"""

from __future__ import annotations

from gpu_fault.admin.bootstrap_task_inputs import TaskInputSpec, task_input_spec

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
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
class TaskSpec:
    ensure: Callable[[], Any]
    input_policy: TaskInputSpec
    probe: Callable[[], Any] | None = None
    revalidate: bool = False
    dependencies: tuple[str, ...] = ()
    stop_on_failure: bool = False


@dataclass(frozen=True)
class TaskGraph:
    """Task execution and input policy use the same mandatory declaration."""

    specs: Mapping[str, TaskSpec]

    def __post_init__(self) -> None:
        for name, spec in self.specs.items():
            if spec.input_policy.name != name.partition(":")[0]:
                raise BootstrapError(f"bootstrap task input policy differs: {name}")
            if spec.input_policy.always_revalidate_reason and not spec.revalidate:
                raise BootstrapError(
                    f"bootstrap task must revalidate its inputs: {name}"
                )

    @property
    def tasks(self) -> dict[str, Callable[[], Any]]:
        return {name: spec.ensure for name, spec in self.specs.items()}

    @property
    def dependencies(self) -> dict[str, tuple[str, ...]]:
        return {
            name: spec.dependencies
            for name, spec in self.specs.items()
            if spec.dependencies
        }

    @classmethod
    def from_parts(
        cls,
        *,
        tasks: Mapping[str, Callable[[], Any]],
        probes: Mapping[str, Callable[[], Any]],
        revalidate: frozenset[str],
        dependencies: Mapping[str, tuple[str, ...]],
    ) -> TaskGraph:
        unknown = (set(probes) | set(revalidate) | set(dependencies)) - set(tasks)
        if unknown:
            raise BootstrapError(
                "unknown bootstrap task definitions: " + ", ".join(sorted(unknown))
            )
        return cls(
            {
                name: TaskSpec(
                    ensure=ensure,
                    input_policy=task_input_spec(name),
                    probe=probes.get(name),
                    revalidate=name in revalidate,
                    dependencies=dependencies.get(name, ()),
                    stop_on_failure=name == "release",
                )
                for name, ensure in tasks.items()
            }
        )

    def merged(self, other: TaskGraph) -> TaskGraph:
        duplicates = self.specs.keys() & other.specs.keys()
        if duplicates:
            raise BootstrapError(
                "duplicate bootstrap tasks: " + ", ".join(sorted(duplicates))
            )
        return TaskGraph({**self.specs, **other.specs})

    def after(self, dependencies: Mapping[str, tuple[str, ...]]) -> TaskGraph:
        unknown = set(dependencies) - self.specs.keys()
        if unknown:
            raise BootstrapError(
                "unknown bootstrap dependency owners: " + ", ".join(sorted(unknown))
            )
        return TaskGraph(
            {
                name: replace(
                    spec,
                    dependencies=tuple(
                        dict.fromkeys((*spec.dependencies, *dependencies.get(name, ())))
                    ),
                )
                for name, spec in self.specs.items()
            }
        )

    def run(
        self,
        *,
        state: BootstrapState,
        on_complete: Callable[[frozenset[str]], None] | None = None,
    ) -> dict[str, Any]:
        bindings = self.specs
        if state.value.get("input_sha256"):
            declared = state.value.get("task_input_sha256") or {}
            missing = [
                name
                for name, spec in bindings.items()
                if spec.input_policy.fingerprint is not None
                and not spec.input_policy.requires_release
                and name not in declared
            ]
            if missing:
                raise BootstrapError(
                    "bootstrap task input identities are unbound: "
                    + ", ".join(sorted(missing))
                )
        return run_parallel(
            self.tasks,
            state=state,
            probes={
                name: spec.probe
                for name, spec in self.specs.items()
                if spec.probe is not None
            },
            revalidate=frozenset(
                name for name, spec in self.specs.items() if spec.revalidate
            ),
            dependencies=self.dependencies,
            stop_on_failure=frozenset(
                name for name, spec in self.specs.items() if spec.stop_on_failure
            ),
            on_complete=on_complete,
            input_policies={name: spec.input_policy for name, spec in bindings.items()},
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
    alert_email: str | None,
    release: Callable[[], Mapping[str, Any]],
    fleet_master_file: Path,
    ensure_aurora_ready: Callable[..., Any],
    grafana: GrafanaSettings | None = None,
    custody_runtime_profile: str = "hyperpod-v1",
    existing_site: dict[str, Any] | None = None,
) -> TaskGraph:
    """Prepare platform identities and credentials, not versioned runtimes.

    ``monitoring_install`` retains its registry/checkpoint key and prepares the
    site-scoped writer role and Pod Identity association. ``aurora_refresh``
    prepares its IAM access after ``aurora_ready`` supplies the master Secret
    reference. Neither task consumes a candidate image or waits for the build.
    The release owns ADOT/AMP and refresher manifests on first install too.

    ``release`` joins ``SignedReleaseBuild`` before the graph can finish and
    the site can be handed to the release transaction.

    Live-resource tasks are re-proved by read-only probes that enter ensure
    only on detected drift. The release task always joins the current build.
    """

    from gpu_fault.admin.bootstrap_services import (
        ensure_grafana_dashboards,
        install_aurora_refresh,
        install_monitoring,
        provision_node_action_keys,
    )
    from gpu_fault.admin.node_key_custody_admin_config import load_admin_custody
    from gpu_fault.admin.node_key_custody_admin_probe import AdminNodeKeyContext

    custody = load_admin_custody(state.path.parent)
    if custody is not None and existing_site is not None:
        from gpu_fault.admin.node_key_custody_admin import bootstrap_custody_profile

        custody_runtime_profile = bootstrap_custody_profile(
            state.path.parent, repository_root, existing_site
        )

    def build_tasks(
        task_runner: CommandRunner,
        probe_only: bool,
    ) -> dict[str, Callable[[], Any]]:
        tasks: dict[str, Callable[[], Any]] = {
            "release": release,
            "grafana_install": lambda: {
                "grafana": ensure_grafana_dashboards(
                    task_runner,
                    settings=grafana,
                    cpu=cpu,
                    site_id=site_id,
                    amp_workspace_id=str(
                        state.result("monitoring_resources")["workspace_id"]
                    ),
                    repository_root=repository_root,
                    probe_only=probe_only,
                    admin_email=alert_email,
                )
            },
            "monitoring_install": lambda: install_monitoring(
                task_runner,
                repository_root=repository_root,
                cpu=cpu,
                cpu_kubeconfig=cpu_kubeconfig,
                namespace=namespace,
                site_id=site_id,
                monitoring=state.result("monitoring_resources"),
                adot_image="",
                alert_email=alert_email,
                probe_only=probe_only,
                runtime_managed_by_release=True,
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
                release_manifest=repository_root / "dist/current-release.json",
                runtime_image="",
                # The readiness result carries the master Secret ARN; a
                # checkpoint written before the split carries it on ``aurora``.
                aurora={**state.result("aurora"), **state.result("aurora_ready")},
                probe_only=probe_only,
                runtime_managed_by_release=True,
            ),
        }
        for cluster in gpu_clusters:
            cluster_id = safe_name(cluster.hyperpod_name)

            def node_key_task(
                cluster: ClusterIdentity = cluster,
                cluster_id: str = cluster_id,
            ) -> dict[str, str]:
                custody_arguments: dict[str, Any] = {}
                if custody is not None:
                    from gpu_fault.admin.bootstrap_common import (
                        compute_agent_config_digest,
                    )

                    verified_release = release()
                    digest = (
                        str(verified_release["agent_config_digest"])
                        if custody_runtime_profile == "hyperpod-v1"
                        else compute_agent_config_digest(
                            task_runner,
                            repository_root=repository_root,
                            runtime_profile_version=custody_runtime_profile,
                        )
                    )
                    custody_arguments["custody_context"] = AdminNodeKeyContext(
                        state_dir=state.path.parent,
                        repository_root=repository_root,
                        site_id=site_id,
                        cpu_eks_arn=cpu.eks_arn,
                        cpu_kubeconfig=cpu_kubeconfig,
                        gpu_kubeconfig=gpu_kubeconfig,
                        namespace=namespace,
                        cluster=cluster,
                        cluster_id=cluster_id,
                        release_manifest=Path(str(verified_release["manifest"])),
                        runtime_profile_version=custody_runtime_profile,
                        agent_config_digest=digest,
                    )
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
                    **custody_arguments,
                )

            tasks[f"node_keys:{cluster_id}"] = node_key_task
        return tasks

    tasks = build_tasks(runner, False)
    probes = build_tasks(ReadOnlyProbeRunner(runner), True)
    probes.pop("release")
    dependencies = {
        "grafana_install": ("monitoring_resources",),
        "monitoring_install": ("monitoring_resources",),
        "aurora_ready": ("aurora",),
        "aurora_refresh": ("aurora_ready",),
    }
    if custody is not None:
        previous_key_task = "release"
        for name in tasks:
            if name.startswith("node_keys:"):
                dependencies[name] = (previous_key_task,)
                previous_key_task = name
    return TaskGraph.from_parts(
        tasks=tasks,
        probes=probes,
        revalidate=frozenset(tasks),
        dependencies=dependencies,
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
    balancer); Aurora's serial writer/reader creation is independent of them,
    with final readiness in ``aurora_ready``; the roles, the notification
    resources and the load balancer controller need only the Pod Identity
    add-on the preamble ensured. So every task here starts at once.
    """

    from gpu_fault.admin.bootstrap_load_balancer import (
        ensure_load_balancer_controller,
    )
    from gpu_fault.admin.bootstrap_services import (
        ensure_adot_writer_role,
        ensure_executor_role,
    )
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
        # The data-plane collector's role reads the AMP workspace the
        # ``monitoring_resources`` task created (see ``dependencies`` below).
        adot_writer_tasks = {
            f"adot_writer_role:{safe_name(cluster.hyperpod_name)}": (
                lambda cluster=cluster: ensure_adot_writer_role(
                    active_runner,
                    cluster=cluster,
                    namespace=namespace,
                    site_id=site_id,
                    amp_workspace_id=str(
                        state.result("monitoring_resources")["workspace_id"]
                    ),
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
                state=active_state,
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
            **adot_writer_tasks,
        }

    tasks = build_tasks(runner, state)
    executor_tasks = {name for name in tasks if name.startswith("executor_role:")}
    adot_writer_tasks = {name for name in tasks if name.startswith("adot_writer_role:")}
    return TaskGraph.from_parts(
        tasks=tasks,
        probes=build_tasks(ReadOnlyProbeRunner(runner), None),
        revalidate=frozenset(
            {
                "load_balancer_controller",
                "control_plane_role",
                "email_notifications",
                "monitoring_resources",
                *({"control_record_archive_bucket"} if archive_s3_uri else set()),
                *executor_tasks,
                *adot_writer_tasks,
            }
        ),
        # The writer role's policy names the workspace ``monitoring_resources``
        # creates, and its trust sits on the OIDC provider the executor role
        # ensures for the same cluster: running it concurrently with the
        # executor would race two ``create-open-id-connect-provider`` calls.
        dependencies={
            name: (
                name.replace("adot_writer_role:", "executor_role:", 1),
                "monitoring_resources",
            )
            for name in adot_writer_tasks
        },
    )


def run_bootstrap_tasks(
    *,
    state: BootstrapState,
    foundation: TaskGraph,
    platform: TaskGraph,
    access: TaskGraph | None = None,
) -> dict[str, Any]:
    """Run both graphs as one, marking the two phases as their subsets complete.

    ``aws-infrastructure-ready`` is recorded the moment the last foundation
    task completes (platform tasks may already be running by then) and
    ``platform-prerequisites-ready`` when everything is done, so the two markers
    that checkpoint readers and the acceptance evidence know keep their meaning.
    """

    if access is not None:
        foundation = access.merged(
            foundation.after(
                {
                    "aurora": ("cpu_access",),
                    "load_balancer_controller": ("cpu_access", "pod_identity_agent"),
                    "control_plane_role": ("cpu_access", "pod_identity_agent"),
                    "email_notifications": ("cpu_access",),
                }
            )
        )
        platform = platform.after(
            {
                "monitoring_install": ("cpu_access", "pod_identity_agent"),
                "aurora_ready": ("cpu_access",),
                "aurora_refresh": ("pod_identity_agent",),
                **{
                    name: ("cpu_access", "gpu_access")
                    for name in platform.tasks
                    if name.startswith("node_keys:")
                },
            }
        )
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
