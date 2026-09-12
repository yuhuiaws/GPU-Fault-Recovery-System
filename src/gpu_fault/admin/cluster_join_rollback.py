"""Undo helpers for a rolled-back ``join-cluster`` attempt."""

from __future__ import annotations

import subprocess
from pathlib import Path

from gpu_fault.admin.bootstrap_common import BootstrapError


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


def rollback_command(
    arguments: list[str],
    *,
    not_found: tuple[str, ...],
) -> None:
    result = subprocess.run(arguments, text=True, capture_output=True)
    if result.returncode == 0:
        return
    message = (result.stdout or "") + "\n" + (result.stderr or "")
    if any(value in message for value in not_found):
        return
    raise BootstrapError(
        f"rollback command failed ({result.returncode}): "
        f"{' '.join(arguments[:3])}: {result.stderr.strip()}"
    )


def rollback_iam_role(role: object, *, label: str, errors: list[str]) -> None:
    """Delete a role a failed join created (inline policy first); absent is fine."""

    if not isinstance(role, dict) or not role.get("role_arn"):
        return
    role_name = str(role["role_arn"]).rsplit("/", 1)[-1]
    policy = str(role.get("inline_policy_name") or "")
    if policy:
        try:
            rollback_command(
                [
                    "aws",
                    "iam",
                    "delete-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-name",
                    policy,
                ],
                not_found=("NoSuchEntity",),
            )
        except BootstrapError as exc:
            errors.append(f"{label} policy rollback: {exc}")
    try:
        rollback_command(
            ["aws", "iam", "delete-role", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )
    except BootstrapError as exc:
        errors.append(f"{label} role rollback: {exc}")
