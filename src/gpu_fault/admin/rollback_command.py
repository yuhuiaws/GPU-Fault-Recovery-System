"""``gpu-fault-admin deploy --state-dir X --rollback``: one step back.

The only legal target is the ``previous`` snapshot the release engine keeps in
the live release state: the release that was committed before the current one.
There is no release-id argument. After a rollback the state records no earlier
committed release, so a second ``--rollback`` is refused with the reason; going
further back is an ordinary ``deploy`` of the older commit.

The command runs the engine's ``rollback`` mode and then does exactly what the
release-deploy failure driver does after an automatic rollback: rewrite the
management ``site.yaml`` to the rolled-back release
(``reconcile_rollback_management``) and ``sync-state`` the engine against it,
so ``gpu-fault-admin status`` judges the live site by what is actually running.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.release_state import live_release_state
from gpu_fault.admin.rollback_alignment import reconcile_rollback_management
from gpu_fault.admin.site import (
    RenderedSite,
    SiteConfigError,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault_release.regional_admin_commands import (
    BOOTSTRAP_PHASES,
    RESUMABLE_PHASES,
    ROLLBACK_PHASES,
)
from gpu_fault_release.regional_release_orchestration import (
    NON_TRANSACTIONAL_CHANGES,
)
from gpu_fault_release.regional_schema_change import recorded_acceptance

ROLLOUT_SCRIPT = "deploy/control-plane/regional/rollout-regional-release.sh"
RELEASE_DEPLOY_DIR = "release-deploy"
RESULT_KEY = "gpu_fault_admin_rollback"
DEPLOY_HINT = "rerun `gpu-fault-admin deploy --state-dir <state-dir>`"


class RollbackCommandError(RuntimeError):
    """A release-engine command run by the rollback returned non-zero."""


class RolledBackRelease(Protocol):
    """What the management alignment needs to know about the rolled-back release."""

    release_id: str
    state_dir: Path


RunReleaseMode = Callable[..., None]
Reconcile = Callable[..., Path]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _release_id(value: Mapping[str, Any]) -> str:
    return str(value.get("release_id") or "unknown")


def _mid_transaction(state: Mapping[str, Any], phase: str) -> bool:
    if phase in RESUMABLE_PHASES:
        return True
    if phase != "complete":
        return False
    # A complete but uncommitted release, or a committed one whose post-commit
    # cleanup has not finished, is still the deploy's to finish.
    return (
        state.get("transaction_committed") is not True
        or state.get("commit_cleanup_completed") is False
    )


def schema_rollback_refusal(state: Mapping[str, Any]) -> str | None:
    """The engine's own reason a rollback across this diff cannot run.

    Mirrors ``regional_release_orchestration._rollback_context`` so the operator
    reads the refusal before the engine is even started, with the same words
    the engine would use.
    """

    changed = set(((state.get("release_diff") or {}).get("changed") or []))
    unsupported = changed.intersection(NON_TRANSACTIONAL_CHANGES)
    if unsupported:
        return "rollback is not transactional for: " + ", ".join(sorted(unsupported))
    if "database_schema" not in changed:
        return None
    snapshot = (recorded_acceptance(dict(state)) or {}).get("snapshot_id")
    return (
        "rollback across this PostgreSQL schema change is not declared "
        "backward-compatible"
        + (
            f"; restore the database from the pre-schema snapshot {snapshot} "
            "first, then rerun the rollback"
            if snapshot
            else "; the database has to be restored to the previous schema "
            "version by hand before a rollback can run"
        )
    )


def rollback_refusal(state: Mapping[str, Any]) -> str | None:
    """Why ``--rollback`` must not run against this live state, or ``None``.

    Checked in the order an operator would want to hear them: a transaction
    that is still moving comes before the shape of what it would roll back to.
    """

    phase = str(state.get("phase") or "")
    release_id = _release_id(state)
    if phase in BOOTSTRAP_PHASES:
        return (
            f"the site is still bootstrapping (phase {phase}); there is no "
            f"committed release to roll back to. {DEPLOY_HINT} to finish or "
            "clean up the bootstrap"
        )
    if phase in ROLLBACK_PHASES or (
        phase == "rolled-back" and state.get("rollback_cleanup_completed") is False
    ):
        return (
            f"a rollback of release {release_id} is already in progress "
            f"(phase {phase}); {DEPLOY_HINT} to finish it"
        )
    if _mid_transaction(state, phase):
        return (
            f"release {release_id} is mid-transaction (phase {phase or 'unknown'}); "
            f"{DEPLOY_HINT} to resume or finish it, then roll back"
        )
    previous = state.get("previous")
    if phase == "rolled-back":
        return (
            f"release {release_id} was already rolled back; the live site is "
            f"release {_release_id(previous or {})} and the state records no "
            "earlier committed release, so a second rollback is refused. To go "
            "further back, deploy the older commit as an ordinary release"
        )
    if phase != "complete":
        return f"release {release_id} is in phase {phase or 'unknown'}, not complete"
    if not isinstance(previous, dict) or not previous:
        return (
            f"release {release_id} has nothing to roll back to: the state records "
            "no previous release (a first bootstrap, or a NOOP release). To move "
            "to another version, deploy it as an ordinary release"
        )
    return schema_rollback_refusal(state)


def run_release_mode(
    site_file: Path,
    *,
    mode: str,
    root: Path,
    environment: Mapping[str, str],
) -> None:
    """Run one engine mode against ``site_file`` rendered at ``root``."""

    site = load_site(site_file, repository_root=root)
    rollout_environment = {
        **effective_environment(site),
        **environment,
        **site.environment,
        "GPU_FAULT_REPO_ROOT": str(root),
    }
    with materialized_release_config(site) as config:
        completed = subprocess.run(
            [str(root / ROLLOUT_SCRIPT), mode, "--config", str(config)],
            cwd=root,
            env=rollout_environment,
            check=False,
        )
    if completed.returncode:
        raise RollbackCommandError(
            f"release engine {mode} failed ({completed.returncode})"
        )


def align_rolled_back_management(
    prepared: RolledBackRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    live_state: dict[str, Any],
    run_release_mode: RunReleaseMode,
    reconcile: Reconcile | None = None,
) -> None:
    """Point the management site and the engine's state at the rolled-back release.

    Shared with ``scripts/release_failure_recovery.py`` (the automatic-rollback
    driver), which passes its own ``reconcile`` so its tests keep their seam;
    without one, this module's ``reconcile_rollback_management`` is looked up
    at call time for the same reason.
    """

    del root  # the aligned root comes from the reconciled management site
    if reconcile is None:
        reconcile = reconcile_rollback_management
    aligned_root = reconcile(
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


def record_rollback(
    prepared: RolledBackRelease,
    *,
    rollback: Mapping[str, Any],
) -> bool:
    """Note the rollback in the release's ``release-deploy`` record, if there is one.

    The record is the failure driver's; a release deployed from another host
    or before the record existed has none, and that is not a rollback failure.
    """

    state_path = prepared.state_dir / "state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(state, dict):
        return False
    state.update({"phase": "ROLLED_BACK", "rollback": dict(rollback)})
    write_json_atomic(state_path, state)
    return True


def _emit(result: dict[str, Any]) -> None:
    print(json.dumps({RESULT_KEY: result}, sort_keys=True))


def run_rollback(site: RenderedSite, *, state_dir: Path) -> int:
    """Roll the site back to its previous committed release; returns the exit code.

    Refusals (nothing to roll back to, a transaction in flight, a schema the
    engine will not cross) print the reason and return 2 without touching the
    site. An engine failure returns the engine's exit code; a rollback that
    succeeded but whose management alignment failed returns 1 and says so, as
    the failure driver's record would.
    """

    state_dir = state_dir.expanduser().resolve()
    state = live_release_state(site)
    refusal = rollback_refusal(state)
    release_id = _release_id(state)
    target = _release_id(state.get("previous") or {})
    result: dict[str, Any] = {
        "release_id": release_id,
        "rolled_back_to": target,
        "started_at": _utc_now(),
    }
    if refusal:
        print(f"gpu-fault-admin: rollback refused: {refusal}", file=sys.stderr)
        _emit({**result, "status": "REFUSED", "reason": refusal})
        return 2
    environment = effective_environment(site)
    with materialized_release_config(site) as config:
        completed = subprocess.run(
            [
                str(site.repository_root / ROLLOUT_SCRIPT),
                "rollback",
                "--config",
                str(config),
            ],
            cwd=site.repository_root,
            env=environment,
            check=False,
        )
    if completed.returncode:
        _emit(
            {
                **result,
                "status": "FAILED",
                "completed_at": _utc_now(),
                "error": f"release engine rollback failed ({completed.returncode})",
            }
        )
        return int(completed.returncode)
    prepared = SimpleNamespace(
        release_id=release_id,
        state_dir=state_dir / RELEASE_DEPLOY_DIR / release_id,
    )
    try:
        live_state = live_release_state(site)
        align_rolled_back_management(
            prepared,
            site_file=site.source,
            root=site.repository_root,
            environment=environment,
            live_state=live_state,
            run_release_mode=run_release_mode,
        )
    except (
        OSError,
        RollbackCommandError,
        SiteConfigError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        rollback = {
            **result,
            "status": "ALIGNMENT_FAILED",
            "completed_at": _utc_now(),
            "management_state_synced": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        record_rollback(prepared, rollback=rollback)
        print(
            "gpu-fault-admin: the release engine rolled back, but the management "
            f"state could not be aligned: {exc}",
            file=sys.stderr,
        )
        _emit(rollback)
        return 1
    rollback = {
        **result,
        "status": "PASSED",
        "completed_at": _utc_now(),
        "management_state_synced": True,
        "source": "gpu-fault-admin deploy --rollback",
    }
    rollback["release_deploy_record_updated"] = record_rollback(
        prepared, rollback=rollback
    )
    _emit(rollback)
    return 0
