"""Bounded teardown retries and the recovery of a teardown that gave up.

Every capacity suite cleans up through ``regional_capacity_suite.teardown``
(stop the run's resources, purge its data, deregister its synthetic clusters).
When the bounded retries are spent the run keeps its receipts, ``status.json``
records the cleanup error, and the log names the one command that finishes
the teardown from those receipts: ``regional_capacity_suite.py teardown`` with
the run's ``--suite-id`` and ``--run-dir``, which adopts them only while the
run's ``run.json`` still matches ``resume_identity``.
"""

from __future__ import annotations

import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path

if __package__:
    from . import regional_capacity_registry as capacity_registry
    from .regional_capacity_results import write_status
else:
    import regional_capacity_registry as capacity_registry
    from regional_capacity_results import write_status

SUITE_SCRIPT = Path(__file__).resolve().parent / "regional_capacity_suite.py"
# Waits before teardown attempts 2, 3 and 4. A purge that meets a writer still
# touching the run's rows right after the run (live 2026-10-13: "capacity
# cleanup left run-owned data" once the action executors were gone) is one
# transaction and deletes nothing, so a later attempt starts from the same
# state; together the waits outlast the action suite's 120-s synthetic Agent
# lease, after which the control plane stops writing for the run's nodes.
TEARDOWN_RETRY_DELAYS_SECONDS = (15, 45, 90)


def retry_teardown(once: Callable[[], None], *, attempts: int) -> None:
    """Run one teardown attempt up to ``attempts`` times, waiting
    ``TEARDOWN_RETRY_DELAYS_SECONDS`` between them; raise the last error."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            once()
            return
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                delay = TEARDOWN_RETRY_DELAYS_SECONDS[
                    min(attempt, len(TEARDOWN_RETRY_DELAYS_SECONDS)) - 1
                ]
                detail = " ".join(str(exc).split())[-300:]
                capacity_registry.log(
                    f"teardown attempt {attempt}/{attempts} failed "
                    f"({type(exc).__name__}: {detail}); retrying in {delay} s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"capacity teardown failed after {attempts} attempts: {last_error}"
    ) from last_error


def resume_identity(
    *, suite_id: str, scope: str, identity: dict[str, str]
) -> dict[str, str]:
    """The ``run.json`` fields a later ``run``, ``purge`` or ``teardown`` must
    find unchanged before it adopts a run's receipts; every suite records them,
    so the recovery command can finish any suite's teardown."""

    return {
        "suite_id": suite_id,
        "registry_scope": scope,
        "control_namespace": capacity_registry.CONTROL_NAMESPACE,
        "dataplane_namespace": capacity_registry.NAMESPACE,
        "registry_secret": capacity_registry.REGISTRY_SECRET,
        "connection_secret": capacity_registry.CONNECTION_SECRET,
        **identity,
    }


def teardown_recovery_command(*, run_id: str, artifacts: Path, scope: str) -> str:
    """One shell line that finishes a run's teardown from its kept receipts,
    against the run's control plane, data-plane context, namespaces and
    registry Secrets, with this checkout's interpreter and harness."""

    environment = {
        "GPU_FAULT_CONTROL_KUBECONFIG": capacity_registry.CONTROL_KUBECONFIG,
        "GPU_FAULT_DATAPLANE_CONTEXT": capacity_registry.DATAPLANE_CONTEXT,
        "GPU_FAULT_PERF_CONTROL_NAMESPACE": capacity_registry.CONTROL_NAMESPACE,
        "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": capacity_registry.NAMESPACE,
        "GPU_FAULT_PERF_REGISTRY_SECRET": capacity_registry.REGISTRY_SECRET,
        "GPU_FAULT_PERF_CONNECTION_SECRET": capacity_registry.CONNECTION_SECRET,
    }
    command = [
        sys.executable,
        str(SUITE_SCRIPT),
        "teardown",
        "--suite-id",
        run_id,
        "--run-dir",
        str(artifacts.absolute()),
    ]
    if scope == "live":
        command += [
            "--allow-live-registry",
            "--confirm-live-registry",
            capacity_registry.LIVE_REGISTRY_CONFIRMATION,
        ]
    return " ".join(
        [
            *(f"{name}={shlex.quote(value)}" for name, value in environment.items()),
            shlex.join(command),
        ]
    )


def write_aborted_status(
    artifacts: Path,
    failure: BaseException,
    *,
    cleanup_error: BaseException | None,
    run_id: str,
    scope: str,
) -> None:
    """Mark a failed run aborted. After a cleanup that gave up, ``status.json``
    also keeps the cleanup error and the command that finishes the teardown
    from the run's kept receipts, and the log names the same command."""

    fields: dict[str, str] = {}
    if cleanup_error is not None:
        command = teardown_recovery_command(
            run_id=run_id, artifacts=artifacts, scope=scope
        )
        capacity_registry.log(
            f"cleanup incomplete; the run's receipts stay in {artifacts}"
        )
        capacity_registry.log(f"recovery command: {command}")
        fields = {
            "cleanup_error": f"{type(cleanup_error).__name__}: {cleanup_error}",
            "recovery_command": command,
        }
    write_status(
        artifacts,
        status="aborted",
        reason=f"{type(failure).__name__}: {failure}",
        **fields,
    )
