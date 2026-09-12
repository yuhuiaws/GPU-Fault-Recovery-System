"""Undo helpers for a rolled-back ``join-cluster`` attempt."""

from __future__ import annotations

import subprocess
from pathlib import Path


def restore_current_context(
    kubeconfig: Path, *, deleted: str, errors: list[str]
) -> None:
    """Point ``current-context`` away from the context the rollback deleted.

    ``aws eks update-kubeconfig --alias`` makes the joined cluster current;
    deleting only the context leaves that pointer dangling and every
    ``kubectl --kubeconfig`` call without ``--context`` failing. The first
    remaining context (another managed cluster) takes over; with none left the
    pointer is unset.
    """

    base = ["kubectl", "--kubeconfig", str(kubeconfig), "config"]
    current = subprocess.run(base + ["current-context"], text=True, capture_output=True)
    if current.returncode or (current.stdout or "").strip() != deleted:
        return
    remaining = subprocess.run(
        base + ["get-contexts", "-o", "name"], text=True, capture_output=True
    )
    names = [
        line.strip()
        for line in (remaining.stdout or "").splitlines()
        if line.strip() and line.strip() != deleted
    ]
    repoint = (
        base + ["use-context", names[0]]
        if names
        else base + ["unset", "current-context"]
    )
    result = subprocess.run(repoint, text=True, capture_output=True)
    if result.returncode:
        errors.append("kube current-context rollback: " + result.stderr.strip())
