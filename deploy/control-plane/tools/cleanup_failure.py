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

Nothing secret can reach that message:

* stdout is never read on a failure. It can carry object contents (a namespace
  listing returns Secret data); only stderr is used.
* stderr goes through ``sensitive_error_lines``, the reducer the admin CLI
  applies to every sensitive command: one line (an ``ERROR:`` line of our own
  scripts, otherwise kubectl's last meaningful line), passed through the shared
  redactor (credential assignments, Bearer/Basic values, URL userinfo, private
  keys, AWS access key ids), every run of 32 or more token characters masked,
  and at most 200 characters.
* The reason class is matched on the raw stderr, but what lands in the message
  is one word of :data:`FAILURE_REASONS` (or ``unclassified``), never the text.
* The command is named from our own argument vector: at most three positional
  words before ``--``. The values of value-taking options (a ``-p`` patch, a
  ``-f`` file, an ``-o`` template, a selector) are never shown.

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
import re
from collections.abc import Iterator, Sequence

from cleanup_state import CleanupStateError

from gpu_fault.admin.command_log import sensitive_error_lines

# Checked in this order; the first match names the failure. kubectl prints an
# API refusal as ``Error from server (<Reason>): ...``, except Unauthorized
# (``error: You must be logged in to the server (Unauthorized)``) and Invalid
# (``The request is invalid: ...``, which a JSON patch whose ``test`` operation
# fails returns, or ``The <Kind> "<name>" is invalid: ...``). Timeouts come
# before ``unreachable``: "Unable to connect to the server: ... i/o timeout" is
# a timeout first.
FAILURE_REASONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("conflict", re.compile(r"\(Conflict\)|the object has been modified")),
    (
        "not found",
        re.compile(r"\(NotFound\)|the server could not find the requested resource"),
    ),
    ("forbidden", re.compile(r"\(Forbidden\)| is forbidden: ")),
    (
        "unauthorized",
        re.compile(r"\(Unauthorized\)|must be logged in to the server"),
    ),
    (
        "invalid",
        re.compile(
            r"\(Invalid\)|\(BadRequest\)|The request is invalid|\bis invalid\b"
            r"|rejected our request"
        ),
    ),
    (
        "timeout",
        re.compile(
            r"(?i)\((?:Server)?Timeout\)|timed out|timeout exceeded"
            r"|deadline exceeded|i/o timeout|handshake timeout"
        ),
    ),
    (
        "unreachable",
        re.compile(
            r"(?i)unable to connect to the server|connection refused"
            r"|no such host|no route to host|connection reset"
        ),
    ),
    ("throttled", re.compile(r"\(TooManyRequests\)")),
    (
        "server error",
        re.compile(r"\((?:InternalError|ServiceUnavailable)\)|failed calling webhook"),
    ),
)
UNCLASSIFIED = "unclassified"
# kubectl options whose value is the next argument. Their values (a patch, a
# manifest, an output template, a selector, a namespace) are skipped, not shown.
VALUE_OPTIONS = frozenset(
    {
        "-c",
        "-f",
        "-l",
        "-n",
        "-o",
        "-p",
        "--container",
        "--context",
        "--field-selector",
        "--filename",
        "--kubeconfig",
        "--namespace",
        "--output",
        "--patch",
        "--selector",
        "--type",
    }
)
SUMMARY_WORDS = 3
SUMMARY_WORD_CHARACTERS = 64

FAILURE_NOTE_ENV = "GPU_FAULT_CLEANUP_FAILURE_NOTE"
FAILURE_NOTE_CHARACTERS = 600


def failure_reason(stderr: str | None) -> str:
    """The reason class of a failed kubectl call, from its raw stderr."""

    text = stderr or ""
    return next(
        (name for name, pattern in FAILURE_REASONS if pattern.search(text)),
        UNCLASSIFIED,
    )


def command_summary(arguments: Sequence[str]) -> str:
    """The verb, kind and name of a kubectl call; never a payload or a path."""

    words: list[str] = []
    skip = False
    for argument in arguments:
        if skip:
            skip = False
            continue
        if argument == "--":
            break
        if argument in VALUE_OPTIONS:
            skip = True
            continue
        if argument.startswith("-") or not argument.isprintable() or " " in argument:
            continue
        if len(argument) > SUMMARY_WORD_CHARACTERS:
            argument = argument[:SUMMARY_WORD_CHARACTERS] + "..."
        words.append(argument)
        if len(words) == SUMMARY_WORDS:
            break
    return " ".join(words) or "<no arguments>"


class CleanupCommandError(CleanupStateError):
    """A cleanup kubectl call exited non-zero, or ran out of time (no exit status).

    ``reason`` is one word of :data:`FAILURE_REASONS` (``timeout`` without an
    exit status), ``detail`` the redacted stderr line. Callers that recover from
    a failed call (the node metadata patch retry) catch this type; everything
    else treats it as the :class:`CleanupStateError` it is.
    """

    def __init__(
        self,
        arguments: Sequence[str],
        returncode: int | None,
        stderr: str | None = None,
    ) -> None:
        self.returncode = returncode
        self.summary = command_summary(arguments)
        if returncode is None:
            self.reason = "timeout"
            self.detail = "the command did not finish within its time budget"
            status = "no exit status"
        else:
            self.reason = failure_reason(stderr)
            self.detail = "; ".join(sensitive_error_lines(stderr)) or "no error output"
            status = f"exit {returncode}"
        super().__init__(
            f"Kubernetes cleanup command failed ({status}, reason: {self.reason}): "
            f"kubectl {self.summary}: {self.detail}"
        )


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
