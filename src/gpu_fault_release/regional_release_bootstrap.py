"""Initial-bootstrap joins and resumable cleanup, without business rollback."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_diff import AURORA_PREREQUISITE_REPAIR_KEY
from gpu_fault_release.regional_release_progress import BOOTSTRAP_CPU_KEY
from gpu_fault_release.regional_release_rendering import DATAPLANE_ADOT_DEPLOYMENT

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

CLEANUP_PHASES = frozenset(
    {
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-failed",
        "bootstrap-cleaned",
    }
)
# The phases one bootstrap attempt writes itself. Cleanup and ``complete`` are
# not among them: neither proves a CPU this attempt could reuse.
_ATTEMPT_PHASES = frozenset(
    {
        "bootstrap-started",
        "bootstrap-cpu-ready",
        "bootstrap-endpoint-ready",
        "bootstrap-data-plane-progress",
        "bootstrap-failed",
    }
)
_CPU_CHECKPOINTS = frozenset(
    {
        "bootstrap-cpu-ready",
        "bootstrap-endpoint-ready",
        "bootstrap-data-plane-progress",
    }
)


def bootstrap_resume_context(
    state: dict[str, Any],
    release_id: str,
) -> tuple[set[str], bool, bool]:
    """The completed clusters and CPU checkpoint an earlier attempt may lend.

    Only this candidate's own attempt counts, and never across cleanup or a
    ``complete`` state: cleanup leaves 0-replica Deployments, and a
    ``complete`` state bootstrap re-enters has lost its control plane. A
    repair journal that took over another candidate's bootstrap
    (``supersedes`` from a handoff, ``predecessor`` from a takeover) already
    names this candidate, but until adoption removes it nothing in the state
    is this candidate's.

    The phase is the checkpoint an attempt reached; ``resume_phase`` counts
    only in ``bootstrap-failed``, where the failure save recorded it, so a
    killed attempt never resumes from an older attempt's checkpoint. The live
    CPU check may stand in for the schema ensure and CPU apply only after this
    candidate's own attempt ensured its schema here (``BOOTSTRAP_CPU_KEY``),
    and it refuses the 0-replica Deployments cleanup leaves; anything else
    re-applies.
    """
    journal = state.get(AURORA_PREREQUISITE_REPAIR_KEY)
    taken_over = isinstance(journal, dict) and not {
        "supersedes",
        "predecessor",
    }.isdisjoint(journal)
    loaded_phase = str(state.get("phase") or "")
    resume_phase = (
        str(state.get("resume_phase") or loaded_phase)
        if loaded_phase == "bootstrap-failed"
        else loaded_phase
    )
    reusable = (
        state.get("release_id") in {None, release_id}
        and not taken_over
        and loaded_phase in _ATTEMPT_PHASES
    )
    completed_cluster_ids = (
        {str(value) for value in state.get("completed_cluster_ids", []) if value}
        if reusable
        else set()
    )
    cpu_checkpoint = reusable and resume_phase in _CPU_CHECKPOINTS
    check_live_cpu = (
        reusable
        and resume_phase in {"bootstrap-started", "bootstrap-failed"}
        and state.get(BOOTSTRAP_CPU_KEY) == release_id
    )
    return completed_cluster_ids, cpu_checkpoint, check_live_cpu


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
