"""Why a kubectl call failed, in one bounded line without secrets.

Shared by the reset cleanup tool (``deploy/control-plane/tools/cleanup_failure.py``,
which runs from the source snapshot and imports this package) and the admin's own
node cleanup of ``remove-cluster`` and the join rollback
(``cluster_removal_kubernetes``). A bare exit status ("exit 1") left an operator
nothing to act on; :class:`KubectlFailure` names the call, kubectl's exit
status, a reason class and kubectl's last meaningful stderr line::

    (exit 1, reason: forbidden): kubectl patch node <name>: stderr: Error from
    server (Forbidden): nodes "<name>" is forbidden: ...

Nothing secret can reach that line:

* stdout is never read. It can carry object contents (a namespace listing
  returns Secret data); only stderr is used.
* stderr goes through ``sensitive_error_lines``, the reducer the admin CLI applies
  to every sensitive command: one line (an ``ERROR:`` line of our own scripts,
  otherwise kubectl's last meaningful line), passed through the shared redactor
  (credential assignments, Bearer/Basic values, URL userinfo, private keys, AWS
  access key ids), every run of 32 or more token characters masked, and at most
  200 characters.
* The reason class is matched on the raw stderr, but what lands in the line is
  one word of :data:`FAILURE_REASONS` (or ``unclassified``), never the text.
* The call is named from the caller's own argument vector: at most three
  positional words before ``--``. The values of value-taking options (a ``-p``
  patch, a ``-f`` file, an ``-o`` template, a selector) are never shown.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

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


def failure_reason(stderr: str | None) -> str:
    """The reason class of a failed kubectl call, from its raw stderr."""

    text = stderr or ""
    return next(
        (name for name, pattern in FAILURE_REASONS if pattern.search(text)),
        UNCLASSIFIED,
    )


def command_summary(arguments: Sequence[str]) -> str:
    """The verb, kind and name of a kubectl call; never a payload or a path.

    ``arguments`` are kubectl's own arguments, without the ``kubectl`` binary
    and its connection prefix.
    """

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


@dataclass(frozen=True)
class KubectlFailure:
    """One failed kubectl call: what ran, its exit status, why, and kubectl's line.

    ``returncode`` is ``None`` when the call ran out of time before it exited;
    its reason is then ``timeout``. ``str()`` is the bounded line callers append
    to their own error prefix.
    """

    summary: str
    returncode: int | None
    reason: str
    detail: str

    @classmethod
    def of(
        cls,
        arguments: Sequence[str],
        returncode: int | None,
        stderr: str | None = None,
    ) -> KubectlFailure:
        summary = command_summary(arguments)
        if returncode is None:
            return cls(
                summary,
                None,
                "timeout",
                "the command did not finish within its time budget",
            )
        return cls(
            summary,
            returncode,
            failure_reason(stderr),
            "; ".join(sensitive_error_lines(stderr)) or "no error output",
        )

    def __str__(self) -> str:
        status = (
            "no exit status" if self.returncode is None else f"exit {self.returncode}"
        )
        return (
            f"({status}, reason: {self.reason}): kubectl {self.summary}: {self.detail}"
        )
