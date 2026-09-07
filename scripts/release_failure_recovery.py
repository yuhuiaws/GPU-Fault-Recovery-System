from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gpu_fault.admin.rollback_alignment import reconcile_rollback_management


# Mirrors ``regional_schema_change.ACCEPT_SCHEMA_CHANGE_ENV`` (pinned equal by
# tests/regional/test_release_schema_change_acceptance.py).
ACCEPT_SCHEMA_CHANGE_ENV = "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE"


def schema_change_fail_forward(
    deployment: Mapping[str, Any],
    environment: Mapping[str, str],
) -> str | None:
    """Why this release runs fail-forward despite ``autoRollback: true``, or None.

    The release engine refuses a schema change under automatic rollback unless
    the operator accepted it (``--accept-schema-change``); when it did, a verify
    or stability failure must not try the rollback either -- it would be refused
    at the schema, after the components had already been moved back. The engine
    owns the decision; this driver only mirrors it so the failure record says
    ``SKIPPED_POLICY`` with the real reason instead of attempting the rollback.
    With no readable diff the acceptance alone decides, since the engine will
    have applied it to whatever diff it computed.
    """

    if not str(environment.get(ACCEPT_SCHEMA_CHANGE_ENV, "")).strip():
        return None
    next_deploy = deployment.get("next_deploy")
    changed = (
        set(next_deploy.get("changed") or []) if isinstance(next_deploy, dict) else None
    )
    if changed is not None and "database_schema" not in changed:
        return None
    return (
        "schema change accepted for this transaction (--accept-schema-change); "
        "rollback cannot cross a schema version, see the pre-schema snapshot in "
        "gpu-fault-admin status"
    )


class PreparedRelease(Protocol):
    release_id: str
    state_dir: Path


RunReleaseMode = Callable[..., None]
ReadLiveState = Callable[[Path], dict[str, Any]]
UpdatePhase = Callable[..., None]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def align_rolled_back_management(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    live_state: dict[str, Any],
    run_release_mode: RunReleaseMode,
) -> None:
    aligned_root = reconcile_rollback_management(
        site_file,
        live_state,
        site_before=prepared.state_dir / "site.before.yaml",
        source=f"rollback:{prepared.release_id}",
    )
    run_release_mode(
        site_file,
        mode="sync-state",
        root=aligned_root,
        environment=environment,
    )


def align_existing_rollback(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    live_state: dict[str, Any],
    run_release_mode: RunReleaseMode,
) -> dict[str, Any]:
    started_at = _utc_now()
    try:
        align_rolled_back_management(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            live_state=live_state,
            run_release_mode=run_release_mode,
        )
    except Exception as exc:
        return {
            "status": "FAILED",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "PASSED",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "management_state_synced": True,
        "source": "low-level-automatic-rollback",
    }


def rollback_after_failure(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    failure_error: str,
    failed_at: str,
    run_release_mode: RunReleaseMode,
    read_live_state: ReadLiveState,
    update_phase: UpdatePhase,
) -> dict[str, Any]:
    rollback = {"status": "IN_PROGRESS", "started_at": _utc_now()}
    update_phase(
        prepared,
        "FAILED",
        error=failure_error,
        failed_at=failed_at,
        rollback=rollback,
    )
    try:
        run_release_mode(
            site_file,
            mode="rollback",
            root=root,
            environment=environment,
        )
        live_state = read_live_state(site_file)
        align_rolled_back_management(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            live_state=live_state,
            run_release_mode=run_release_mode,
        )
    except Exception as rollback_exc:
        return {
            **rollback,
            "status": "FAILED",
            "completed_at": _utc_now(),
            "error": f"{type(rollback_exc).__name__}: {rollback_exc}",
        }
    return {
        **rollback,
        "status": "PASSED",
        "completed_at": _utc_now(),
        "management_state_synced": True,
    }


def recover_release_failure(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    failure_error: str,
    failed_at: str,
    deployment_succeeded: bool,
    commit_started: bool,
    automatic_rollback: bool,
    run_release_mode: RunReleaseMode,
    read_live_state: ReadLiveState,
    update_phase: UpdatePhase,
    deployment: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    # An accepted schema change (``--accept-schema-change``) runs the release
    # fail-forward whatever the site says: the rollback would be refused at the
    # schema. Decided here, next to the policy it overrides, so the failure
    # record names the acceptance rather than ``spec.autoRollback``.
    fail_forward_reason = schema_change_fail_forward(deployment or {}, environment)
    automatic_rollback = automatic_rollback and fail_forward_reason is None
    failure_state: dict[str, Any] | None = None
    if not deployment_succeeded or commit_started:
        try:
            failure_state = read_live_state(site_file)
        except Exception:
            pass
    rollback_result = (
        failure_state.get("rollback_result")
        if isinstance(failure_state, dict)
        else None
    )
    if (
        isinstance(failure_state, dict)
        and failure_state.get("phase") == "rolled-back"
        and isinstance(rollback_result, dict)
        and rollback_result.get("status") == "PASSED"
    ):
        if failure_state.get("rollback_cleanup_completed") is False:
            try:
                run_release_mode(
                    site_file,
                    mode="rollback",
                    root=root,
                    environment=environment,
                )
                failure_state = read_live_state(site_file)
            except Exception as cleanup_exc:
                return {
                    "status": "CLEANUP_PENDING",
                    "management_state_synced": False,
                    "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                }
            rollback_result = failure_state.get("rollback_result")
            if (
                failure_state.get("phase") != "rolled-back"
                or not isinstance(rollback_result, dict)
                or rollback_result.get("status") != "PASSED"
                or failure_state.get("rollback_cleanup_completed") is not True
            ):
                return {
                    "status": "CLEANUP_PENDING",
                    "management_state_synced": False,
                    "error": "rollback cleanup did not reach a verified terminal state",
                }
        return align_existing_rollback(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            live_state=failure_state,
            run_release_mode=run_release_mode,
        )
    if (
        isinstance(failure_state, dict)
        and failure_state.get("phase") == "complete"
        and failure_state.get("transaction_committed") is True
    ):
        return {
            "status": "SKIPPED_COMMITTED",
            "reason": "release is committed; only post-commit cleanup may retry",
        }
    if (
        deployment_succeeded
        and automatic_rollback
        and not (commit_started and failure_state is None)
    ):
        return rollback_after_failure(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            failure_error=failure_error,
            failed_at=failed_at,
            run_release_mode=run_release_mode,
            read_live_state=read_live_state,
            update_phase=update_phase,
        )
    if deployment_succeeded and not automatic_rollback:
        return {
            "status": "SKIPPED_POLICY",
            "reason": fail_forward_reason or "site spec.autoRollback is false",
        }
    if commit_started and failure_state is None:
        return {
            "status": "SKIPPED_AMBIGUOUS_COMMIT",
            "reason": "live commit state could not be read safely",
        }
    return None
