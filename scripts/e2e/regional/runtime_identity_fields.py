"""Compare runtime identities while allowing named release-state fields to move.

A case whose own sanctioned action writes the release state (BOOT-023's NOOP
re-stamps ``updated_at_epoch``) must still hold every other state field and
every Deployment identity byte-identical; this is the one place that
allowance is expressed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    runtime_identity_errors,
)


def identity_without_state_fields(
    identity: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    if not fields:
        return identity
    value = dict(identity)
    state = value.get("release_state")
    if isinstance(state, dict):
        value["release_state"] = {k: v for k, v in state.items() if k not in fields}
    return value


def verify_runtime_identity_allowing(
    fixture: Any,
    expected: dict[str, Any],
    *,
    evidence_path: Path,
    stage: str,
    mutable_state_fields: tuple[str, ...],
) -> dict[str, Any]:
    """``RegionalLiveFixture.verify_runtime_identity`` with named state fields free to move.

    Everything else is the strict check: Deployment generations, images and
    templates and every other release-state field must be byte-identical, and
    the live identity must still be safe (``runtime_identity_errors``).
    """

    current = fixture.runtime_identity()
    write_json_atomic(evidence_path, current)
    if identity_without_state_fields(
        current, mutable_state_fields
    ) != identity_without_state_fields(expected, mutable_state_fields):
        raise RegionalFixtureError(
            f"{stage} release/runtime deployment identity drifted"
        )
    errors = runtime_identity_errors(current)
    if errors:
        raise RegionalFixtureError(
            f"{stage} runtime identity is unsafe: {'; '.join(errors)}"
        )
    return current
