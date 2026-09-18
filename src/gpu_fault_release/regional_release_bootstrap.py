"""Initial-bootstrap joins and resumable cleanup, without business rollback."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_rendering import DATAPLANE_ADOT_DEPLOYMENT

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease


@contextmanager
def bootstrap_observability(configure: Callable[[], None]) -> Iterator[None]:
    """Overlap control-plane configuration with GPU bootstrap.

    Keep GPU checkpoints on the caller's thread. Both branches must finish before
    validation, failure persistence or cleanup can run, including on interruption.
    The caller publishes expected-collector rules only after this scope succeeds.
    """
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="bootstrap-observability"
    ) as executor:
        future = executor.submit(copy_context().run, configure)
        try:
            yield
        except BaseException as primary:
            try:
                future.result()
            except BaseException as secondary:
                if isinstance(primary, Exception) and not isinstance(
                    secondary, Exception
                ):
                    raise secondary from primary
                primary.add_note(
                    "control-plane observability also failed: "
                    + diagnostic_text(f"{type(secondary).__name__}: {secondary}")
                )
            raise
        else:
            future.result()


def cleanup_bootstrap(release: RegionalRelease) -> None:
    completed = set(release.state.get("bootstrap_cleanup_completed_steps") or [])

    def save(phase: str, **updates: Any) -> None:
        release._save_state(
            phase,
            previous=None,
            resume_phase="bootstrap-started",
            completed_cluster_ids=[],
            bootstrap_cleanup_completed_steps=sorted(completed),
            **updates,
        )

    save("bootstrap-cleanup-started")
    try:
        if "installer-jobs-cancelled" not in completed:
            for target in release.config.clusters:
                release._scale_if_present(
                    release._gpu(target),
                    inventory.GPU_RECONCILER_DEPLOYMENT,
                    0,
                    wait=True,
                )
                release._cancel_active_installer_jobs(target)
            completed.add("installer-jobs-cancelled")
            save("bootstrap-cleanup-progress")
        if "gpu-scaled-down" not in completed:
            for target in release.config.clusters:
                for deployment in (*inventory.DEPLOYMENTS, DATAPLANE_ADOT_DEPLOYMENT):
                    release._scale_if_present(
                        release._gpu(target),
                        deployment,
                        0,
                        wait=True,
                    )
            completed.add("gpu-scaled-down")
            save("bootstrap-cleanup-progress")
        if "cpu-scaled-down" not in completed:
            for deployment in inventory.CPU_DEPLOYMENTS:
                release._scale_if_present(release._cpu(), deployment, 0, wait=True)
            completed.add("cpu-scaled-down")
            save("bootstrap-cleanup-progress")
    except Exception as exc:
        save(
            "bootstrap-cleanup-failed",
            bootstrap_cleanup_failure=f"{type(exc).__name__}: {exc}",
        )
        raise
    save("bootstrap-cleaned")
