"""The ``rotate-token`` journal's on-disk vocabulary, shared with the release preflight.

``<state-dir>/rotate-token/<cluster-id>/state.json`` is written by
:mod:`gpu_fault.admin.rotate_token` and read back by the release engine's
preflight (``regional_release_token_rotation_safety``), which must refuse a
deploy while a rotation is unfinished. The engine cannot import the rotation
itself -- the rotation composes the engine -- so the constants, the path and
the "is this journal finished" rule live here, below both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from gpu_fault.admin.bootstrap_common import BootstrapError, safe_name

STATE_SCHEMA_VERSION = 1
STATE_ROOT = "rotate-token"

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_ROLLED_BACK = "ROLLED_BACK"
TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_ROLLED_BACK})

STEP_PREPARED = "PREPARED"
STEP_OVERLAP_PUBLISHED = "REGISTRY_OVERLAP_PUBLISHED"
STEP_SECRET_UPDATED = "CONNECTION_SECRET_UPDATED"
STEP_DATA_PLANE_ROLLED = "DATA_PLANE_ROLLED"
STEP_NODES_ROLLED = "NODES_ROLLED"
STEP_ACCEPTED = "DATA_PLANE_ACCEPTED"
STEP_TOKEN_FILE_WRITTEN = "TOKEN_FILE_WRITTEN"
STEP_CONTROL_PLANE_ROLLED = "CONTROL_PLANE_ROLLED"
STEP_RETIRING_DROPPED = "RETIRING_TOKEN_DROPPED"
ROTATION_STEPS = (
    STEP_PREPARED,
    STEP_OVERLAP_PUBLISHED,
    STEP_SECRET_UPDATED,
    STEP_DATA_PLANE_ROLLED,
    STEP_NODES_ROLLED,
    STEP_ACCEPTED,
    STEP_TOKEN_FILE_WRITTEN,
    STEP_CONTROL_PLANE_ROLLED,
    STEP_RETIRING_DROPPED,
)
ROLLBACK_SECRET_RESTORED = "ROLLBACK_SECRET_RESTORED"
ROLLBACK_DATA_PLANE_ROLLED = "ROLLBACK_DATA_PLANE_ROLLED"
ROLLBACK_NODES_ROLLED = "ROLLBACK_NODES_ROLLED"
ROLLBACK_REGISTRY_RESTORED = "ROLLBACK_REGISTRY_RESTORED"
ROLLBACK_STEPS = (
    ROLLBACK_SECRET_RESTORED,
    ROLLBACK_DATA_PLANE_ROLLED,
    ROLLBACK_NODES_ROLLED,
    ROLLBACK_REGISTRY_RESTORED,
)


def rotation_journal_path(state_dir: Path, cluster_id: str) -> Path:
    return state_dir / STATE_ROOT / safe_name(cluster_id) / "state.json"


def load_rotation_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BootstrapError(f"rotate-token state is not an object: {path}")
    return cast(dict[str, Any], value)


def step_done(state: Mapping[str, Any], step: str) -> bool:
    return step in (state.get("steps") or {})


def step_started(state: Mapping[str, Any], step: str) -> bool:
    return step_done(state, step) or step in (state.get("started_steps") or {})


def last_completed_step(state: Mapping[str, Any]) -> str | None:
    """The furthest forward step the journal has completed, in machine order."""

    done = [step for step in ROTATION_STEPS if step_done(state, step)]
    return done[-1] if done else None


def unfinished_rotation(state: Mapping[str, Any]) -> str | None:
    """Why ``state`` still needs ``rotate-token``, or ``None`` once it is finished.

    The same two conditions ``rotate-token`` itself resumes on: an
    ``IN_PROGRESS`` status, or a terminal status whose pending-token cleanup
    has not completed. Anything else that is not terminal is unknown and
    therefore unfinished too.
    """

    status = state.get("status")
    if status == STATUS_IN_PROGRESS:
        return f"status {STATUS_IN_PROGRESS}"
    if state.get("pending_token_cleanup_completed") is False:
        return f"status {status} with pending token cleanup"
    if status not in TERMINAL_STATUSES:
        return f"unknown status {status!r}"
    return None


def rollback_in_progress(state: Mapping[str, Any]) -> bool:
    return bool(state.get("rollback_started_at")) or any(
        step_started(state, step) for step in ROLLBACK_STEPS
    )


def rollback_permitted(state: Mapping[str, Any]) -> bool:
    """``--rollback`` is refused once the token file write intent is journaled."""

    return not step_started(state, STEP_TOKEN_FILE_WRITTEN)
