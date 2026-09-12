"""Bring a GPU cluster's prerequisites up beside the control-plane endpoint gate.

Measured live on the 2026-09-12 join-cluster release (four GPU nodes): the GPU
half of the release ran strictly serially, each step waiting for Ready before
the next -- DNS/TLS gate 31 s, DCGM DaemonSet 12 s, ADOT collector plus the two
artifact ConfigMaps 44 s, then the Executor Deployment and the watcher/collector
pair each waited on their own -- and about two minutes of that was stacked
waiting on objects that do not depend on one another.

The one real dependency is the endpoint gate. The cluster executor, the
Completion Watcher and the node-resource collector all mount
``gpu-fault-regional-connection`` (``deploy/dataplane/*.yaml``) and talk to the
region, so none of their Deployments may be applied before the gate has proved
the cluster can resolve the control-plane hostname, trust its certificate and
authenticate to it. DCGM, the ADOT collector and the artifact ConfigMaps never
reach the control plane: they go up while the gate probe is still running, and
the Deployments follow once every step of the stage has finished, so DCGM is
still exporting before the first Executor Pod -- and the node Installer -- rolls.

The steps share one ``Runner``. It keeps no per-call state (each ``run`` times
itself from a local clock and prints one line), and the release already drives
it from several threads at once -- four clusters bootstrap in parallel and a
Deployment wave waits its members together -- so overlapping calls here is
nothing new. A dry run stays serial so the printed plan reads in one fixed order.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gpu_fault_release.regional_release_config import ClusterTarget


@dataclass(frozen=True)
class GpuStageStep:
    """One independent unit of the stage, named for the failure report."""

    name: str
    action: Callable[[], Any]


def _dry_run(release: Any) -> bool:
    # A test double models the release as a bare namespace without a runner; it
    # takes the concurrent path, which is the one worth exercising.
    return bool(getattr(getattr(release, "runner", None), "dry_run", False))


def run_gpu_stage(
    release: Any,
    cluster_id: str,
    steps: Sequence[GpuStageStep],
) -> None:
    """Run every step at once and fail with the first one in declared order.

    A step that fails does not interrupt the others: a ``kubectl apply`` cannot
    be taken back by killing it, and a stage that stopped half-way would leave
    objects no failure report names. Every step runs to its own end; then the
    first failure in the order the steps were declared is re-raised unchanged --
    the orchestrator classifies it by type (a ``ClusterLocalReleaseError``
    pauses one cluster, anything else fails the release) -- and the others are
    printed so the operator sees them too. The rollback snapshot was captured
    before the stage began, so whatever the surviving steps applied is exactly
    what a rollback puts back.
    """

    if not steps:
        return
    if len(steps) == 1 or _dry_run(release):
        for step in steps:
            step.action()
        return
    failures: list[tuple[GpuStageStep, Exception]] = []
    with ThreadPoolExecutor(
        max_workers=len(steps),
        thread_name_prefix=f"gpu-stage-{cluster_id}",
    ) as pool:
        futures = [(step, pool.submit(step.action)) for step in steps]
        for step, future in futures:
            try:
                future.result()
            except Exception as exc:
                failures.append((step, exc))
    if not failures:
        return
    for step, error in failures[1:]:
        print(
            f"{cluster_id}: {step.name} also failed while the stage ran: {error}",
            file=sys.stderr,
            flush=True,
        )
    raise failures[0][1]


def gpu_prerequisite_steps(
    release: Any,
    target: ClusterTarget,
) -> tuple[GpuStageStep, GpuStageStep, GpuStageStep]:
    """The gate and the two exporters every fresh GPU cluster needs before its Deployments."""

    return (
        GpuStageStep(
            "GPU DNS/TLS gate",
            lambda: release._verify_gpu_control_plane_endpoint(target),
        ),
        GpuStageStep(
            "DCGM exporter",
            lambda: release._apply_gpu_dcgm_exporter(target),
        ),
        GpuStageStep(
            "data-plane ADOT collector",
            lambda: release._apply_gpu_adot_collector(target),
        ),
    )


def join_gpu_upload_steps(
    release: Any,
    target: ClusterTarget,
) -> tuple[GpuStageStep, GpuStageStep]:
    """The Executor wheel and the node bundle a joining cluster does not have yet.

    A bootstrap uploads them for every cluster in its own phase; a join is the
    only path that uploads inside the GPU stage, and the two ConfigMaps are
    read by nothing until the Deployments and the Installer mount them.
    """

    kubectl = release._gpu(target)
    return (
        GpuStageStep(
            f"ConfigMap {release.executor_wheel_cm}",
            lambda: release._upload_config_map(
                kubectl,
                release.executor_wheel_cm,
                release.config.executor_wheel.name,
                release.config.executor_wheel,
                release.executor_wheel_sha,
                compress=True,
            ),
        ),
        GpuStageStep(
            f"ConfigMap {release.bundle_cm}",
            lambda: release._upload_config_map(
                kubectl,
                release.bundle_cm,
                release.config.bundle.name,
                release.config.bundle,
                release.bundle_sha,
            ),
        ),
    )


def stage_gpu_prerequisites(
    release: Any,
    target: ClusterTarget,
    *,
    extra: Sequence[GpuStageStep] = (),
) -> None:
    """Gate, DCGM and the ADOT collector together, plus any ``extra`` steps.

    Returns only when every step has finished, so the caller's next line -- the
    Executor, watcher and collector Deployments -- is applied after the gate has
    passed and after DCGM and the collector are Ready.
    """

    run_gpu_stage(
        release,
        target.cluster_id,
        [*gpu_prerequisite_steps(release, target), *extra],
    )


def stage_join_gpu_prerequisites(release: Any, target: ClusterTarget) -> None:
    """A joining cluster's stage: the prerequisites plus its two artifact uploads."""

    stage_gpu_prerequisites(
        release,
        target,
        extra=join_gpu_upload_steps(release, target),
    )
