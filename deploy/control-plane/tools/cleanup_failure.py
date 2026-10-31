"""Why a Kubernetes cleanup command failed, in one bounded line without secrets.

``CleanupClient`` (``cleanup_kubernetes.py``) captures kubectl's output. A
failure used to read ``Kubernetes cleanup command failed (exit 1)`` and nothing
else, so neither the uninstall log nor the journal said why (live: the node
metadata cleanup of a reset stopped on one HyperPod node with only that line).
:class:`CleanupCommandError` now names the command, kubectl's exit status, a
reason class and kubectl's last meaningful stderr line::

    Kubernetes cleanup command failed (exit 1, reason: forbidden): kubectl patch
    node <name>: stderr: Error from server (Forbidden): nodes "<name>" is
    forbidden: ...

The line itself is ``gpu_fault.admin.kubectl_failure.KubectlFailure``, shared
with the admin's node cleanup of ``remove-cluster`` and the join rollback; its
docstring says why no secret can reach it (stdout is never read, stderr is
reduced and redacted, the reason is a fixed word, the command is three words).

The same line reaches the cleanup journal. ``prepare-clean-redeploy.sh`` names a
private file in :data:`FAILURE_NOTE_ENV`; every cleanup invocation empties it on
start and, when it fails, leaves ``<action> <context>: <reason>`` there
(:func:`failure_note`). The script's exit trap appends that note to the
``FAILED`` entry it writes, so the journal no longer says only
``cleanup command exited 1``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence

from cleanup_state import CleanupStateError

from gpu_fault.admin.kubectl_failure import KubectlFailure

FAILURE_NOTE_ENV = "GPU_FAULT_CLEANUP_FAILURE_NOTE"
FAILURE_NOTE_CHARACTERS = 600


class CleanupCommandError(CleanupStateError):
    """A cleanup kubectl call exited non-zero, or ran out of time (no exit status).

    ``failure`` describes it (``reason`` is one word of the shared reason
    classes, ``timeout`` without an exit status; ``detail`` the redacted stderr
    line). Callers that recover from a failed call (the node metadata patch
    retry) catch this type; everything else treats it as the
    :class:`CleanupStateError` it is.
    """

    def __init__(
        self,
        arguments: Sequence[str],
        returncode: int | None,
        stderr: str | None = None,
    ) -> None:
        self.failure = KubectlFailure.of(arguments, returncode, stderr)
        super().__init__(f"Kubernetes cleanup command failed {self.failure}")

    @property
    def returncode(self) -> int | None:
        return self.failure.returncode

    @property
    def reason(self) -> str:
        return self.failure.reason

    @property
    def detail(self) -> str:
        return self.failure.detail


def failure_text(error: BaseException) -> str:
    """What the journal note says about ``error``: ours in full, others by type.

    A :class:`CleanupStateError` (including :class:`CleanupCommandError`) and a
    deadline (``TimeoutError``) carry messages this tooling wrote. Any other
    exception may carry foreign text, so only its type is named.
    """

    if isinstance(error, (CleanupStateError, TimeoutError)):
        return str(error) or type(error).__name__
    return type(error).__name__


def write_note(path: str, text: str) -> None:
    """Replace the note's content. Best effort: the note never masks a failure.

    The file must already exist (the cleanup script creates it with ``mktemp``);
    nothing is created and a symlink is not followed.
    """

    printable = "".join(
        character if " " <= character <= "~" else "?" for character in text
    )
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError:
        return
    try:
        os.write(descriptor, printable[:FAILURE_NOTE_CHARACTERS].encode("ascii"))
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def failure_note(label: str) -> Iterator[None]:
    """Leave ``<label>: <reason>`` for the journal when the body fails.

    The note is emptied first, so it always describes the latest cleanup
    invocation: a later successful call cannot leave an earlier failure behind.
    Without :data:`FAILURE_NOTE_ENV` (a direct call, the uninstall's own
    ``cleanup-owned`` and ``verify-targets`` runs) nothing is written.
    """

    path = os.environ.get(FAILURE_NOTE_ENV, "")
    if path:
        write_note(path, "")
    try:
        yield
    except Exception as error:
        if path:
            write_note(path, f"{label}: {failure_text(error)}")
        raise
