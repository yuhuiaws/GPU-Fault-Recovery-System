from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_evidence import site_non_membership_sha256
from gpu_fault.admin.site import RenderedSite


class JoinStateRequest(Protocol):
    @property
    def site(self) -> RenderedSite: ...

    @property
    def gpu_cluster_arn(self) -> str: ...

    @property
    def cluster_id(self) -> str | None: ...

    @property
    def allowed_namespaces(self) -> tuple[str, ...]: ...

    @property
    def state_dir(self) -> Path | None: ...


# Attempts that are over: the drift guard protects an attempt still being
# built against the site it started from. A rolled-back one starts fresh, a
# failed rollback is undone against the site as it is now, and a COMPLETED one
# either still describes a managed cluster (ALREADY_MANAGED) or a cluster the
# site has since dropped (fresh attempt) -- the site moving on is expected
# in every case (live 2026-09-12: remove-cluster + two deploys after a join).
FINISHED_ATTEMPT_PHASES = frozenset({"ROLLED_BACK", "ROLLBACK_FAILED", "COMPLETED"})


def load_join_state(
    request: JoinStateRequest,
) -> tuple[Path, Path, dict[str, Any]]:
    identity = hashlib.sha256(request.gpu_cluster_arn.encode()).hexdigest()[:12]
    state_dir = (
        request.state_dir or request.site.source.parent / "join-cluster" / identity
    ).expanduser()
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    path = state_dir / "state.json"
    expected = {
        "site_id": request.site.release_config["site_name"],
        "gpu_cluster_arn": request.gpu_cluster_arn,
        "requested_cluster_id": request.cluster_id or "",
        "allowed_namespaces": sorted(request.allowed_namespaces),
    }
    if path.exists():
        value = cast(
            dict[str, Any],
            json.loads(path.read_text(encoding="utf-8")),
        )
        for key, item in expected.items():
            if value.get(key) != item:
                raise BootstrapError(f"join-cluster state conflicts on {key}")
        current_non_membership = site_non_membership_sha256(request.site.source)
        recorded_non_membership = str(
            value.get("source_site_non_membership_sha256") or ""
        )
        if (
            recorded_non_membership
            and recorded_non_membership != current_non_membership
            and value.get("phase") not in FINISHED_ATTEMPT_PHASES
        ):
            # An attempt still in flight was built against the old site; a
            # rolled-back one is over (the caller starts a fresh attempt) and a
            # failed rollback is undone against the site as it is now.
            raise BootstrapError(
                "join-cluster source site non-membership fields drifted"
            )
        if not recorded_non_membership:
            value["source_site_non_membership_sha256"] = current_non_membership
            write_json_atomic(path, value)
        return state_dir, path, value
    value = {
        "schema_version": 1,
        **expected,
        "attempt": 1,
        "source_site_sha256": request.site.source_sha256,
        "source_site_non_membership_sha256": site_non_membership_sha256(
            request.site.source
        ),
        "phase": "STARTED",
        "completed_steps": [],
        "evidence": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(path, value)
    return state_dir, path, value


def step_done(state: dict[str, Any], step: str) -> bool:
    return step in set(state.get("completed_steps") or [])


def completed_state_is_current(
    request: JoinStateRequest,
    state: dict[str, Any],
) -> bool:
    discovered = (state.get("evidence") or {}).get("DISCOVERED") or {}
    target = discovered.get("target") or {}
    cluster_id = str(discovered.get("cluster_id") or "")
    eks_arn = str(target.get("eks_arn") or "")
    return any(
        item.get("cluster_id") == cluster_id and item.get("eks_cluster_arn") == eks_arn
        for item in request.site.release_config["clusters"]
    )


def note_join_failure(state: dict[str, Any], error: BaseException) -> None:
    """Record why an attempt failed and after which step, in the state file.

    The cause used to live only in the command log; the state file the
    operator is told to fix and resume from said ROLLED_BACK and nothing else.
    Rollback keeps the key; ``reset_completed_state`` starts the next attempt
    clean.
    """

    completed_at = dict(state.get("step_completed_at") or {})
    state["failure"] = {
        "error": f"{type(error).__name__}: {error}",
        "after_step": (
            max(completed_at, key=completed_at.__getitem__) if completed_at else None
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def reset_completed_state(
    request: JoinStateRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    previous_attempt = int(state.get("attempt") or 1)
    archive = state_dir / f"state.attempt-{previous_attempt:03d}.json"
    if not archive.exists():
        write_json_atomic(archive, dict(state))
    state.clear()
    state.update(
        {
            "schema_version": 1,
            "site_id": request.site.release_config["site_name"],
            "gpu_cluster_arn": request.gpu_cluster_arn,
            "requested_cluster_id": request.cluster_id or "",
            "allowed_namespaces": sorted(request.allowed_namespaces),
            "attempt": previous_attempt + 1,
            "source_site_sha256": request.site.source_sha256,
            "source_site_non_membership_sha256": site_non_membership_sha256(
                request.site.source
            ),
            "phase": "STARTED",
            "completed_steps": [],
            "evidence": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(state_path, state)


def complete_step(
    path: Path,
    state: dict[str, Any],
    step: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    completed = set(state.get("completed_steps") or [])
    completed.add(step)
    state["completed_steps"] = sorted(completed)
    state["phase"] = step
    now = datetime.now(timezone.utc).isoformat()
    state["updated_at"] = now
    # One timestamp per step so an operator can read where a join spent its time.
    state.setdefault("step_completed_at", {})[step] = now
    if evidence is not None:
        state.setdefault("evidence", {})[step] = evidence
    write_json_atomic(path, state)
