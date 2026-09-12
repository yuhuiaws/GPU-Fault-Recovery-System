from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gpu_fault.admin.rollback_alignment import reconcile_rollback_management
from gpu_fault.admin.rollback_command import (
    align_rolled_back_management as _align_rolled_back_management,
)


# Mirrors ``regional_schema_change.ACCEPT_SCHEMA_CHANGE_ENV`` (pinned equal by
# tests/regional/test_release_schema_change_acceptance.py).
ACCEPT_SCHEMA_CHANGE_ENV = "GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE"
# Mirrors ``regional_release_store_preflight.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE``
# (pinned equal by tests/admin/test_release_deploy.py). The engine exits with it
# when it refused to roll back because a REMEDIATE_DRIVER /
# UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER step is in flight; the driver
# sees only the exit code, in `_run`'s ``command failed (N): ...`` message.
INFLIGHT_INSTALLS_REFUSED_EXIT_CODE = 3
INFLIGHT_INSTALLS_LEVER_ENV = "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS"
_COMMAND_FAILED = re.compile(r"command failed \((\d+)\)")


def refused_inflight_installs(error: BaseException) -> bool:
    """Whether the engine refused the rollback for an in-flight install."""

    match = _COMMAND_FAILED.search(str(error))
    return bool(match) and int(match.group(1)) == INFLIGHT_INSTALLS_REFUSED_EXIT_CODE


# The engine's in-flight install gate writes its verdict under this key of the
# transaction state before the rollback's first checkpoint
# (``regional_release_orchestration.rollback_release``). The driver copies it
# into its own rollback record so the release history says whether the rollback
# was checked, and on what: "clear", "unchecked" (no Running control-plane Pod
# with a ready container could run the probe -- the automatic rollback proceeds
# over exactly that -- or a kubectl-level failure on a ready Pod that consent
# waved through) or "overridden" (consent over a listed set). A refusal never reaches the state
# (nothing is written); the driver records it from the exit code.
INFLIGHT_INSTALLS_STATE_KEY = "inflight_installs"


def _recorded_inflight_verdict(live_state: Mapping[str, Any] | None) -> dict[str, Any]:
    verdict = (
        live_state.get(INFLIGHT_INSTALLS_STATE_KEY)
        if isinstance(live_state, Mapping)
        else None
    )
    if isinstance(verdict, dict) and "verdict" in verdict:
        return dict(verdict)
    return {
        "checked": False,
        "verdict": "unrecorded",
        "reason": (
            f"the engine state carries no {INFLIGHT_INSTALLS_STATE_KEY} verdict: "
            "the engine predates the field"
        ),
        "steps": [],
    }


def _refused_inflight_verdict() -> dict[str, Any]:
    return {
        "checked": True,
        "verdict": "refused",
        "reason": (
            "the engine's in-flight install check refused the rollback; the "
            "rollback log names the workflows and nodes"
        ),
        "steps": [],
    }


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
    # The same alignment `gpu-fault-admin deploy --rollback` runs after a manual
    # rollback (`gpu_fault.admin.rollback_command`); the reconcile is passed
    # through this module's own name so it stays the seam tests replace here.
    _align_rolled_back_management(
        prepared,
        site_file=site_file,
        root=root,
        environment=environment,
        live_state=live_state,
        run_release_mode=run_release_mode,
        reconcile=reconcile_rollback_management,
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
    live_state: dict[str, Any] | None = None
    try:
        run_release_mode(
            site_file,
            mode="rollback",
            root=root,
            environment=environment,
            # Marks the rollback automatic: the engine's in-flight install
            # check then proceeds (logging) when no Running control-plane Pod
            # with a ready container could run its probe (StoreUnreachable),
            # instead of wedging a release whose control plane is down. A
            # kubectl-level failure on a ready Pod (KubectlFailure) and
            # evidence the probe did return still refuse (exit 3, recorded
            # below).
            arguments=("--automatic",),
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
        if refused_inflight_installs(rollback_exc):
            # The engine refused before touching anything: an install step is
            # in flight and the previous control plane would submit it again.
            # Not a failed rollback -- nothing was rolled back -- and the
            # engine's own state stays where the upgrade failure left it.
            return {
                **rollback,
                "status": "REFUSED_INFLIGHT_INSTALLS",
                "completed_at": _utc_now(),
                "reason": (
                    "the release engine refused the rollback: a REMEDIATE_DRIVER / "
                    "UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER step is in "
                    "flight and the previous control plane would submit it a "
                    "second time; nothing was rolled back (the rollback log names "
                    "the workflows). Rerun once the installs finish, or with "
                    f"{INFLIGHT_INSTALLS_LEVER_ENV}=1 to roll back anyway"
                ),
                "error": f"{type(rollback_exc).__name__}: {rollback_exc}",
                "inflight_installs": _refused_inflight_verdict(),
            }
        if live_state is None:
            # The rollback ran past its first checkpoint before failing (or
            # the state read itself failed); one more read is cheap and the
            # record should still say whether the restore was checked.
            try:
                live_state = read_live_state(site_file)
            except Exception:
                live_state = None
        return {
            **rollback,
            "status": "FAILED",
            "completed_at": _utc_now(),
            "error": f"{type(rollback_exc).__name__}: {rollback_exc}",
            "inflight_installs": _recorded_inflight_verdict(live_state),
        }
    return {
        **rollback,
        "status": "PASSED",
        "completed_at": _utc_now(),
        "management_state_synced": True,
        "inflight_installs": _recorded_inflight_verdict(live_state),
    }


def _no_rollback_baseline(
    failure_state: dict[str, Any] | None,
    *,
    site_file: Path,
    read_live_state: ReadLiveState,
) -> bool:
    """Whether the live release is a first bootstrap with nothing to roll back to.

    Only the ``complete`` state a bootstrap writes carries ``previous: null``;
    every upgrade records the release it replaced. Read on demand: the other
    branches above deliberately do not read the live state after a successful
    deploy, and an unreadable state must keep the rollback decision as it was.
    """

    state = failure_state
    if state is None:
        try:
            state = read_live_state(site_file)
        except Exception:
            return False
    return (
        isinstance(state, dict)
        and state.get("phase") == "complete"
        and "previous" in state
        and state.get("previous") is None
    )


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
        and _no_rollback_baseline(
            failure_state, site_file=site_file, read_live_state=read_live_state
        )
    ):
        # A first bootstrap that failed its verify or stability window: there is
        # no previous release to roll back to, and the engine's rollback would
        # only record a second failure ("previous release state is
        # unavailable") over a complete data plane. Leave it uncommitted for the
        # rerun, which verifies and commits it.
        return {
            "status": "SKIPPED_NO_BASELINE",
            "reason": "first bootstrap has no previous release to roll back to; "
            "rerun deploy to verify and commit it",
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
