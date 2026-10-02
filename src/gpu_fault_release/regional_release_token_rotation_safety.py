"""Refuse a release while a cluster's token rotation journal is unfinished.

The mirror of ``rotate-token``'s own guard (it refuses while a release
transaction is open): a deploy re-renders the registry Secret from the site's
token files and republishes it, which drops the retiring digest a rotation
still had to retire, and it rewrites ``site.yaml`` under the journal. Both
happened live, between ``CONTROL_PLANE_ROLLED`` and ``RETIRING_TOKEN_DROPPED``.
The preflight reads every configured cluster's journal through the managed
state directory the administrator CLI binds into the release config and
names the command that finishes (or, where still allowed, rolls back) the
rotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_fault.admin.rotate_token_journal import (
    last_completed_step,
    load_rotation_state,
    rollback_in_progress,
    rollback_permitted,
    rotation_journal_path,
    unfinished_rotation,
)
from gpu_fault_release.regional_release_config import ReleaseError

UNBOUND_SUMMARY = (
    "no managed state directory is bound to this release config; rotate-token "
    "journals were not checked"
)


def _remedy(state_dir: Path, cluster_arn: str, state: dict[str, Any]) -> str:
    command = (
        f"gpu-fault-admin rotate-token --state-dir {state_dir} "
        f"--gpu-cluster-arn {cluster_arn}"
    )
    if rollback_in_progress(state):
        return f"finish the rollback with {command} --rollback"
    if state.get("status") == "ROLLED_BACK":
        return f"finish its cleanup with {command} --rollback"
    if rollback_permitted(state):
        return f"finish it with {command}, or walk it back with {command} --rollback"
    return f"finish it with {command}"


def token_rotation_snapshot(release: Any) -> dict[str, Any]:
    """Every configured cluster's journal status; raises on an unfinished one."""

    state_dir = release.config.admin_state_dir
    if state_dir is None:
        raise ReleaseError(UNBOUND_SUMMARY)
    clusters: dict[str, str] = {}
    blockers: list[str] = []
    for target in release.config.clusters:
        path = rotation_journal_path(state_dir, target.cluster_id)
        state = load_rotation_state(path)
        if state is None:
            clusters[target.cluster_id] = "absent"
            continue
        clusters[target.cluster_id] = str(state.get("status"))
        reason = unfinished_rotation(state)
        if reason is None:
            continue
        blockers.append(
            f"cluster {target.cluster_id} has an unfinished token rotation "
            f"({reason}, last completed step {last_completed_step(state)}); "
            f"{_remedy(state_dir, str(target.eks_cluster_arn), state)}; "
            f"journal {path}"
        )
    if blockers:
        raise ReleaseError(
            "an unfinished token rotation blocks the release: " + "; ".join(blockers)
        )
    return {"state_dir": str(state_dir), "clusters": clusters}


def token_rotation_check(release: Any) -> tuple[str, Any, str]:
    """``(summary, details, status)`` for the preflight's ``token_rotation`` check.

    Shaped for ``CheckValue(*...)``: the checks module owns that type and
    imports this one, so it cannot be constructed here.
    """

    if getattr(release.config, "admin_state_dir", None) is None:
        return UNBOUND_SUMMARY, None, "WARN"
    return (
        "no unfinished token rotation journal in the managed state directory",
        token_rotation_snapshot(release),
        "PASS",
    )
