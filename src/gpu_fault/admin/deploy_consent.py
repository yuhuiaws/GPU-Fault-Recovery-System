"""Consent refusals of ``gpu-fault-admin deploy`` in the first minute.

Two release-engine gates need the operator's explicit consent on the command:
``--supersede-failed-transaction`` (a *different* candidate over a transaction
that stopped in ``failed``/``partial-convergence``) and
``--accept-schema-change`` (a candidate that changes the PostgreSQL schema
under ``autoRollback: true``). The engine refuses both before anything moves --
but "before anything moves" in the engine is after the source scan, the release
gates, the build and bootstrap's AWS re-validation. This module makes the same
decisions as soon as the two facts are known: the live release (from the quick
``status`` the deploy already runs) and the candidate (from the release
manifest, the moment ``make release-build``/reuse returns it). The engine's own
checks stay as the last line of defence and say exactly the same words.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release.regional_admin_commands import (
    foreign_candidate_resume_message,
    supersede_requested,
)
from gpu_fault_release.regional_release_orchestration import SUPERSEDABLE_PHASES
from gpu_fault_release.regional_schema_change import (
    recorded_acceptance,
    refusal_message as schema_change_refusal_message,
    requested_acceptance_mode,
)


def consent_refusal(
    *,
    live_phase: str,
    live_release_id: str,
    candidate_release_id: str,
    live_schema_version: int | None = None,
    candidate_schema_version: int | None = None,
    auto_rollback: bool = True,
    acceptance_recorded: bool = False,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """The engine's refusal for this live/candidate pair, or None to proceed.

    Supersede: a terminal failed transaction and a candidate with another id,
    without the flag. Schema: the two schema versions differ, automatic rollback
    is on, no acceptance mode travels in the environment, and the live
    transaction did not already record one (a resumed schema-change transaction
    carries the acceptance it started with). A version that is not known
    (``None``) is not compared -- the engine will.
    """

    env = dict(environment if environment is not None else os.environ)
    if (
        live_phase in SUPERSEDABLE_PHASES
        and candidate_release_id
        and candidate_release_id != live_release_id
        and not supersede_requested(env)
    ):
        return foreign_candidate_resume_message(
            live_release_id=live_release_id,
            phase=live_phase,
            candidate_release_id=candidate_release_id,
        )
    if (
        live_schema_version is not None
        and candidate_schema_version is not None
        and live_schema_version != candidate_schema_version
        and auto_rollback
        and not acceptance_recorded
        and requested_acceptance_mode(env) is None
    ):
        return schema_change_refusal_message()
    return None


def _manifest_schema_version(manifest: Mapping[str, Any]) -> int | None:
    value = manifest.get("database_schema_version")
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    )


def load_release_manifest(path: Path) -> dict[str, Any] | None:
    """``dist/current-release.json`` as a dict, or None when it is not there yet."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"release manifest is invalid: {exc}") from exc
    return value if isinstance(value, dict) else None


def refuse_unconsented_candidate(
    *,
    live_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    auto_rollback: bool,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Raise :class:`BootstrapError` with the engine's wording, or return.

    ``live_state`` is the live release state (or the ``live_release`` block of
    a status report -- both carry ``phase`` and ``release_id``); ``manifest`` is
    the candidate's release manifest.
    """

    live_version = live_state.get("database_schema_version")
    refusal = consent_refusal(
        live_phase=str(live_state.get("phase") or ""),
        live_release_id=str(live_state.get("release_id") or ""),
        candidate_release_id=str(manifest.get("release_id") or ""),
        live_schema_version=(
            int(live_version)
            if isinstance(live_version, int) and not isinstance(live_version, bool)
            else None
        ),
        candidate_schema_version=_manifest_schema_version(manifest),
        auto_rollback=auto_rollback,
        acceptance_recorded=recorded_acceptance(dict(live_state)) is not None,
        environment=environment,
    )
    if refusal:
        raise BootstrapError(refusal)


def refuse_unconsented_release(
    *,
    state_dir: Path,
    manifest_path: Path,
    existing_site: Mapping[str, Any] | None,
) -> None:
    """The inner hop's check, run the moment the release build/reuse returns.

    Nothing to check on a first bootstrap (no ``site.yaml``); a live state that
    cannot be read is left to the engine, which reports the cause itself.
    """

    if existing_site is None:
        return
    from gpu_fault.admin.release_state import live_release_state
    from gpu_fault.admin.site import SiteConfigError, load_site

    manifest = load_release_manifest(manifest_path)
    if manifest is None:
        return
    try:
        live_state = live_release_state(load_site(state_dir / "site.yaml"))
    except SiteConfigError:
        return
    spec = existing_site.get("spec")
    declared = spec.get("autoRollback") if isinstance(spec, Mapping) else None
    refuse_unconsented_candidate(
        live_state=live_state,
        manifest=manifest,
        auto_rollback=declared if isinstance(declared, bool) else True,
    )
