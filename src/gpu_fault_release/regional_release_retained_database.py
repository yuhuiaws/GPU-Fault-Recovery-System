"""A completed keep-uninstall authorizes one new namespace, not a generic bypass."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.installation_lifecycle import retained_handoff
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_probe_job import Checkpoint
from gpu_fault_release.regional_release_store_probe import compatible_retained_schema
from gpu_fault_release.regional_release_store_proof import (
    bootstrap_store_proof,
    database_identity,
)

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease

RETAINED_ORIGIN_KEY = "retained_database_origin"
HANDOFF_KEY = "retained_database_handoff"


def retained_database_proof(
    release: RegionalRelease, checkpoint: Checkpoint
) -> dict[str, Any] | None:
    path = release.config.retained_database_handoff
    origin = release.state.get(RETAINED_ORIGIN_KEY)
    if path is None:
        if origin is not None:
            raise ReleaseError(
                "retained database origin lacks its installation handoff"
            )
        return None
    if not release.config.installation_id:
        raise ReleaseError(
            "retained database handoff lacks its new installation identity"
        )
    identity = database_identity(release)
    try:
        binding = retained_handoff(
            path,
            installation_id=release.config.installation_id,
            identity=identity,
            site_name=release.config.site_name,
        )
    except BootstrapError as exc:
        raise ReleaseError(str(exc)) from exc
    if origin is not None:
        try:
            finished = datetime.fromisoformat(origin["finished_at"])
            valid = (
                origin["safe"] is True
                and origin["database_state"] == "initialized"
                and type(origin["schema_version"]) is int
                and (
                    origin["schema_version"] == identity["schema_version"]
                    or compatible_retained_schema(
                        origin["schema_version"], identity["schema_version"]
                    )
                    and origin.get("schema_ensure_required") is True
                )
                and origin["identity_sha256"] == canonical_sha256(identity)
                and origin[HANDOFF_KEY] == binding
                and origin["blockers"]
                == {"workflow": 0, "remote_command": 0, "observation": 0}
                and re.fullmatch(r"[a-f0-9]{32}", origin["run_id"]) is not None
                and isinstance(origin["job_uid"], str)
                and bool(origin["job_uid"])
                and finished.tzinfo is not None
                and finished <= datetime.now(UTC)
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ReleaseError("retained database origin is missing or unbound")
    proof = bootstrap_store_proof(
        release, checkpoint, require_empty=False, allow_retained_schema_upgrade=True
    )
    if proof["database_state"] != "initialized" or proof[
        "identity_sha256"
    ] != canonical_sha256(identity):
        raise ReleaseError("retained database schema or incarnation changed")
    previous = release.state.get("bootstrap_store_safety") or origin
    if previous is not None and proof["schema_version"] < previous["schema_version"]:
        raise ReleaseError("retained database schema version moved backwards")
    return {**proof, HANDOFF_KEY: binding}
