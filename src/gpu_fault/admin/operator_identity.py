"""Who is running ``gpu-fault-admin``, for the audit trail of what it writes.

Every mutating administrator path -- a Profile approval, a workflow reconcile
apply -- was attributed by a free-text ``--reference`` alone, so two operators
typing the same ticket number were indistinguishable afterwards. The identity
here is the STS caller ARN, which is what the account's own audit log keys on.

Resolving it must never hold the command: a missing CLI, a slow endpoint or an
unauthenticated shell degrades to a named fallback (``UNKNOWN_IDENTITY`` or the
caller's ``user@host``) that goes on the record as such, so the gap itself is
visible rather than silently filled in.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
from typing import Any, Callable

UNKNOWN_IDENTITY = "unknown-identity"
IDENTITY_TIMEOUT_SECONDS = 10.0
_CALLER_IDENTITY_COMMAND = ("aws", "sts", "get-caller-identity", "--output", "json")


def caller_identity_arn(
    *,
    timeout: float = IDENTITY_TIMEOUT_SECONDS,
    run: Callable[..., Any] = subprocess.run,
) -> str | None:
    """The STS caller ARN of the current shell, or ``None`` if it cannot be read.

    Every failure -- no ``aws`` binary, a timeout, a non-zero exit, output that
    is not the expected JSON -- is ``None``: the write this attributes has to
    happen either way, and the caller substitutes a named fallback.
    """

    try:
        completed = run(
            list(_CALLER_IDENTITY_COMMAND),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    try:
        document = json.loads(getattr(completed, "stdout", "") or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    arn = document.get("Arn")
    if not isinstance(arn, str) or not arn.strip():
        return None
    return arn.strip()


def local_operator_identity() -> str:
    """``user@host`` of the shell running the command: the fallback for paths
    where an unnamed approver is worse than an unverified one."""

    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = os.environ.get("USER") or "unknown-user"
    try:
        host = socket.gethostname() or "unknown-host"
    except OSError:
        host = "unknown-host"
    return f"{user}@{host}"


def resolve_operator_identity(
    *,
    fallback: str | None = None,
    timeout: float = IDENTITY_TIMEOUT_SECONDS,
) -> str:
    """The STS caller ARN, else ``fallback``, else ``UNKNOWN_IDENTITY``."""

    arn = caller_identity_arn(timeout=timeout)
    if arn:
        return arn
    return fallback if fallback else UNKNOWN_IDENTITY
