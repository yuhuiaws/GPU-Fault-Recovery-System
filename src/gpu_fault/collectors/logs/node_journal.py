"""Where ``journalctl`` starts reading for the node log collector, and how it fails.

One positioning option per invocation. ``journalctl`` takes a time window
(``--since``) or a journal position (``--cursor``/``--after-cursor``), never
both: ``parse_argv`` in ``src/journal/journalctl.c`` (systemd v252, the version
on AL2023 nodes; in v257 the same check and message also count
``--cursor-file=``) exits before anything is read::

    if (!!arg_cursor + !!arg_after_cursor + !!arg_since_set > 1)
            return log_error_errno(SYNTHETIC_ERRNO(EINVAL),
                    "Please specify only one of --since=, --cursor=, and --after-cursor=.");

https://github.com/systemd/systemd/blob/v252/src/journal/journalctl.c -- reproduced
on systemd 255 (``journalctl --since '-1min' --after-cursor "$CURSOR"``: exit
status 1, that one line on stderr). ``--until`` is not part of the check and
bounds both modes (``journalctl --after-cursor "$CURSOR" --until @<now>`` exits 0).

The collector passed ``--since`` and ``--after-cursor`` together as soon as it
knew a cursor, so every resumed poll was refused. The refusal says "cursor" and
was taken for a cursor the journal had lost, which dropped the cursor: the next
poll read by timestamp, learned a cursor, and was refused again. Every other poll
read nothing, the control plane saw it only when the periodic slot happened to
land on a refused poll (``journalctl-failures=17`` in a live collector status),
and a line that landed in a refused poll's window waited for the next poll.

Positioning rules (:func:`journal_command`):

* **resume** -- ``--after-cursor=<cursor> --until=@<now>`` whenever a cursor is
  held: the cursor of the last entry read. journalctl(1) on ``--after-cursor=``:
  "Start showing entries from the location in the journal after the location
  specified by the passed cursor."
  (https://www.freedesktop.org/software/systemd/man/latest/journalctl.html#--after-cursor=)
  Nothing after that entry is skipped and the entry itself is not read again:
  no gap, no duplicate, whatever the timestamps say -- three entries in one
  microsecond resume correctly, a time window could not tell them apart.
* **time window** -- ``--since=@<since> --until=@<now>`` when no cursor is held:
  the first read (``since`` is the five-minute default and is reported as a
  gap), after the 900 s window cap, when the state file's cursor belongs to
  another boot (:func:`gpu_fault.collectors.logs.node_state.resumable_cursor`),
  after journalctl could not seek to the cursor, and after it refused the cursor
  arguments. ``since`` is where the last complete read ended (or the last entry
  a budget-stopped read took), so nothing is skipped; the entry on the boundary
  can be read twice and keeps its ``__CURSOR`` as entry id, which the control
  plane keys the event by (``log_recovery_context.log_event_id``) -- a repeat
  spells the same event.

The cursor string is carried, never parsed: "The cursor string should be
considered opaque and not be parsed by clients."
(https://www.freedesktop.org/software/systemd/man/latest/sd_journal_get_cursor.html)
A journald cursor whose entry has been vacuumed is not an error either: the
seek lands on the closest entry in time
(https://www.freedesktop.org/software/systemd/man/latest/sd_journal_seek_head.html),
which is why the boot is recorded next to the cursor instead of read out of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

#: journalctl's refusal of two positioning options, verbatim (module docstring).
POSITIONING_REFUSAL = (
    "Please specify only one of --since=, --cursor=, and --after-cursor=."
)

#: The classes a non-zero ``journalctl`` exit is counted under, as
#: ``journalctl-failures-<class>`` next to the total ``journalctl-failures``.
#: One class per failure, the first pattern that matches stderr wins; a child
#: ended by a signal is ``killed`` whatever it printed.
FAILURE_CLASSES = (
    "refused-arguments",
    "cursor-rejected",
    "permission-denied",
    "journal-unavailable",
    "killed",
    "other",
)

#: A failure of one of these classes while positioned by cursor drops the
#: cursor; the time window from ``journal_since`` is used instead. A refusal of
#: the argv also stops cursor positioning for the life of the process
#: (:meth:`PeriodicReport.cursor_dropped`): resuming it on the next poll is the
#: alternation that hid the live defect.
CURSOR_DROPPING_FAILURES = frozenset({"refused-arguments", "cursor-rejected"})

_FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # parse_argv() and getopt: the argv itself was refused, nothing was read.
    (
        "refused-arguments",
        re.compile(
            r"specify (?:only|at most) one of|specify either|must be before|"
            r"unrecognized option|invalid option|requires an argument|unknown option",
            re.IGNORECASE,
        ),
    ),
    # sd_journal_seek_cursor() could not parse the string (EINVAL).
    (
        "cursor-rejected",
        re.compile(r"failed to (?:seek to|read) cursor", re.IGNORECASE),
    ),
    (
        "permission-denied",
        re.compile(r"permission denied|operation not permitted", re.IGNORECASE),
    ),
    (
        "journal-unavailable",
        re.compile(
            r"no journal files|failed to open journal|failed to (?:add match|iterate|get)",
            re.IGNORECASE,
        ),
    ),
)


def journal_command(
    since: datetime, until: datetime, cursor: str | None = None
) -> list[str]:
    """One ``journalctl`` invocation, positioned by the cursor or by the window.

    Never both: see the module docstring for the refusal. ``--until`` bounds the
    read in both modes, so a complete read may book the window end as read.
    """

    position = (
        [f"--after-cursor={cursor}"] if cursor else ["--since", f"@{since.timestamp()}"]
    )
    return [
        "journalctl",
        *position,
        "--until",
        f"@{until.timestamp()}",
        "--output=json",
        # Without --all journald replaces any field over 4096 bytes with a
        # null value, so a long driver line arrived as MESSAGE: null, became
        # "", matched no rule and was dropped without a trace -- and the long
        # lines are the ones this collector exists to catch. The collector
        # bounds entries itself (`max_entry_bytes`).
        "--all",
        "--no-pager",
    ]


def journalctl_failure_class(returncode: int, stderr: str) -> str:
    """The class of a failed ``journalctl`` exit: how it ended and what it said.

    The class is a counter key and a decision (:data:`CURSOR_DROPPING_FAILURES`),
    so it is coarse on purpose; the stderr line itself is carried in the batch.
    """

    if returncode < 0:
        return "killed"
    for name, pattern in _FAILURE_PATTERNS:
        if pattern.search(stderr):
            return name
    return "other"


@dataclass
class PeriodicReport:
    """What the next periodic batch has to say about the polls nobody saw.

    A quiet poll is not posted, so its collection errors reached the control
    plane only when the periodic slot landed on it. With every other poll
    failing that was a coin toss per report: the collector status flipped
    between the failure and a clean health summary, ``last_success_at`` kept
    moving, and the erroring state never lasted the alert's window. The report
    carries the count of polls that failed unposted since the last report, so
    a node whose journal reads fail half the time is an erroring node until
    they stop -- and only until then: a clean period reports nothing.

    It also carries the one note that must outlive its poll: journalctl refused
    the cursor arguments, so the collector reads by time window until it is
    restarted. That is a product fault on this node's systemd, and a report
    that stayed clean would hide it exactly the way the live defect was hidden.
    """

    polls: int = 0
    unposted: int = 0
    last_error: str | None = None
    cursor_refused: str | None = None

    def resume_cursor(self, cursor: str | None) -> str | None:
        """The cursor a read may be resumed from: none once cursors were refused."""

        return None if self.cursor_refused else cursor

    def cursor_dropped(self, failure: str, detail: str, at: datetime) -> str:
        """The error line for a cursor dropped on ``failure`` (one of
        :data:`CURSOR_DROPPING_FAILURES`), remembered when it has to outlive the poll."""

        if failure == "refused-arguments":
            self.cursor_refused = (
                f"journalctl refused the cursor arguments at {at.isoformat()} "
                f"({detail}); positioning by time window from now on: nothing is "
                "skipped, an entry on a window boundary can be read twice under "
                "its own id"
            )
            return self.cursor_refused
        return "journal cursor unavailable; next poll replays timestamp window"

    def lines(self) -> list[str]:
        """What a periodic report adds to this poll's own collection errors."""

        lines = []
        if self.unposted:
            lines.append(
                f"{self.unposted} of the {self.polls} polls since the last periodic "
                "report ended in collection errors that were not posted; "
                f"last: {self.last_error}"
            )
        if self.cursor_refused:
            lines.append(self.cursor_refused)
        return lines

    def settle(self, errors: list[str], *, periodic: bool, entries: int) -> None:
        """Book one finished poll: posted on the periodic slot, posted with
        entries, or quiet -- and, if quiet and failed, owed to the next report."""

        if periodic:
            self.polls = 0
            self.unposted = 0
            self.last_error = None
            return
        self.polls += 1
        if errors and not entries:
            self.unposted += 1
            self.last_error = errors[0]
