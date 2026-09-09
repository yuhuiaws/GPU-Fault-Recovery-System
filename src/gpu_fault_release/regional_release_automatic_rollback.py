"""What an upgrade transaction does with its own failure.

Record it, then decide whether the compensating rollback runs. Three things
stop it: the site says ``autoRollback: false``; the operator accepted a schema
change for this transaction (the rollback would be refused at the schema, and
they were told so); the failure is a cluster-local pause, which
``partial-convergence`` handles per cluster. A fourth is decided by the
rollback itself: while a driver / firmware / EFA-driver install is in flight
the previous release's control plane would submit it a second time, so the
rollback refuses before touching anything (``regional_release_store_preflight``)
and the transaction stays in ``failed`` -- the phase resume and
``--supersede-failed-transaction`` already handle -- rather than going to
``rollback-failed``, which would claim a restore was attempted. A store that
cannot answer that check does not stop the automatic rollback (its primary
scenario is a control plane that is down); the gate logs and the rollback runs.
"""

from __future__ import annotations

from typing import Any

from gpu_fault_release.regional_release_config import (
    PartialClusterRolloutError,
    ReleaseError,
)
from gpu_fault_release.regional_release_diff import ReleaseDiff, ReleaseExecutionPlan
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_store_preflight import (
    ALLOW_INFLIGHT_INSTALLS_FLAG,
    InflightInstallsRefused,
)
from gpu_fault_release.regional_release_transaction import (
    raise_automatic_rollback_failure,
    record_upgrade_failure,
)
from gpu_fault_release.regional_schema_change import recorded_acceptance


def recover_failed_upgrade(
    release: Any,
    *,
    error: Exception,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    previous: dict[str, Any],
    completed_phases: set[str],
    completed_clusters: set[str],
    registry_staged: bool,
) -> None:
    """Record the failure; roll back when policy allows; raise what happened.

    Always raises: ``error`` itself when no rollback ran or it succeeded, a
    ``ReleaseError`` chained to it when the rollback was refused or failed.
    """

    record_upgrade_failure(
        release,
        error=error,
        release_diff=diff.as_dict(),
        execution_plan=plan.as_dict(),
        previous=previous,
        completed_phases=completed_phases,
        completed_clusters=completed_clusters,
        registry_staged=registry_staged,
    )
    if (
        not release.config.auto_rollback
        or recorded_acceptance(release.state) is not None
        or isinstance(error, PartialClusterRolloutError)
    ):
        raise error
    try:
        # ``automatic``: a store that cannot answer the in-flight install check
        # does not stop this rollback (a dead control plane dispatches
        # nothing; wedging production in ``failed`` is the worse failure).
        release.rollback(state=previous, automatic=True)
    except InflightInstallsRefused as refusal:
        phase = release.state.get("phase")
        narrate_step("automatic-rollback-refused", phase=phase, reason=str(refusal))
        raise ReleaseError(
            "release upgrade failed and the automatic rollback was refused: "
            f"{refusal}. The transaction stays in phase {phase} with nothing "
            "rolled back; once the installs finish, rerun gpu-fault-admin deploy "
            "to resume it, or run the release engine's rollback with "
            f"{ALLOW_INFLIGHT_INSTALLS_FLAG} consent in the environment to roll "
            f"back anyway (upgrade={type(error).__name__}: {error})"
        ) from error
    except Exception as rollback_error:
        raise_automatic_rollback_failure(
            release,
            upgrade_error=error,
            rollback_error=rollback_error,
            previous=previous,
            release_diff=diff.as_dict(),
            execution_plan=plan.as_dict(),
        )
    raise error
