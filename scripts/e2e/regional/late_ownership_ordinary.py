"""Explicit binding to the completed ordinary DESTR-015, never sibling inference."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from scripts.e2e.regional import run_destr015_parallel_branch_join as ordinary
from scripts.e2e.regional.acceptance_scope import FORMAL_SCOPE
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied

MAX_EVIDENCE_BYTES = 1024 * 1024


def ordinary_result_root(path: Path) -> Path:
    if (
        path.name != f"{ordinary.CASE_ID}.json"
        or path.parent.name != ordinary.CASE_ID
        or path.parent.parent.name != "cases"
    ):
        raise BoundaryDenied(
            "ordinary DESTR-015 evidence must use its canonical result path"
        )
    return path.parent.parent.parent


def require_external_ordinary(run_dir: Path, path: Path) -> None:
    root = run_dir.resolve()
    previous = ordinary_result_root(path.resolve())
    if root.is_relative_to(previous) or previous.is_relative_to(root):
        raise BoundaryDenied(
            "companion and ordinary run directories must be non-nested"
        )


def read_ordinary_completion(
    path: Path, *, preflight: dict[str, Any], cluster_id: str, nodes: tuple[str, str]
) -> dict[str, Any]:
    ordinary_result_root(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_EVIDENCE_BYTES:
            raise BoundaryDenied(
                "ordinary DESTR-015 evidence is not a bounded regular file"
            )
        with os.fdopen(fd, "rb", closefd=False) as source:
            raw = source.read(MAX_EVIDENCE_BYTES + 1)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        if (
            len(raw) > MAX_EVIDENCE_BYTES
            or len(raw) != info.st_size
            or (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise BoundaryDenied("ordinary DESTR-015 evidence changed while reading")
    finally:
        os.close(fd)
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("case_id") != ordinary.CASE_ID
        or value.get("verdict") != "PASS"
        or value.get("status", "COMPLETED") != "COMPLETED"
        or value.get("errors") != []
        or value.get("error")
        or value.get("evidence_mode", "LIVE") != "LIVE"
        or value.get("subproof")
        or value.get("execution_scope", FORMAL_SCOPE) != FORMAL_SCOPE
        or value.get("formal_sequence_satisfied", True) is not True
        or value.get("release_id") != preflight.get("release_id")
        or value.get("cluster_id") != cluster_id
        or value.get("nodes") != list(nodes)
    ):
        raise BoundaryDenied(
            "ordinary DESTR-015 is not a completed PASS for this target"
        )
    components = value.get("components")
    expected = ordinary.evidence_components(
        {"preflight_identity": ordinary.plan_identity(preflight, nodes=nodes)}
    )["preflight_identity"]
    if (
        not isinstance(components, dict)
        or set(components) != {"preflight_identity", "workflow", "incident", "hosts"}
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in components.values()
        )
        or components.get("preflight_identity") != expected
        or value.get("case_digest") != ordinary.case_digest(components)
    ):
        raise BoundaryDenied(
            "ordinary DESTR-015 release, runtime or Node identity differs"
        )
    cleanup = value.get("cleanup")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("errors") != []
        or (cleanup.get("quiescence") or {}).get("safe_to_delete") is not True
        or (cleanup.get("workload_residual") or {}).get("residual") is not False
        or cleanup.get("workload_cleanup_deferred")
        or cleanup.get("isolation_restore_deferred")
        or cleanup.get("runtime_identity") != preflight.get("runtime_identity")
    ):
        raise BoundaryDenied("ordinary DESTR-015 cleanup is not complete")
    for name in ("prewarm_cleanup", *(f"probe_cleanup_{node}" for node in nodes)):
        residuals = cleanup.get(name)
        if not isinstance(residuals, dict) or any(
            value is not False for value in residuals.values()
        ):
            raise BoundaryDenied(
                "ordinary DESTR-015 has missing or residual cleanup resources"
            )
        if name != "prewarm_cleanup":
            resources = set(residuals) - {"host_script", "creation_unresolved"}
            if (
                len(residuals) != 4
                or len(resources) != 2
                or any(
                    sum(
                        key.startswith(f"{kind}/")
                        and key.count("/") == 1
                        and bool(key[len(kind) + 1 :])
                        for key in resources
                    )
                    != 1
                    for kind in ("pod", "configmap")
                )
            ):
                raise BoundaryDenied(
                    "ordinary DESTR-015 probe cleanup inventory is incomplete"
                )
    restored = cleanup.get("restore_isolated_nodes")
    if not isinstance(restored, dict) or set(restored) != set(nodes):
        raise BoundaryDenied("ordinary DESTR-015 node restoration is incomplete")
    for node in nodes:
        state = restored[node]
        if not isinstance(state, dict) or not (
            state.get("isolated") is False
            or (state.get("isolated") is True and state.get("restore") == "SUCCEEDED")
        ):
            raise BoundaryDenied("ordinary DESTR-015 node restoration is unconfirmed")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "case_id": ordinary.CASE_ID,
        "case_digest": value["case_digest"],
        "release_id": value["release_id"],
        "cluster_id": cluster_id,
        "nodes": list(nodes),
        "cleanup_verified": True,
    }
