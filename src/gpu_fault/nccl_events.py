"""Annotate stored error evidence; text is not a hung or exit detector."""

from __future__ import annotations

import re

TIMEOUT = re.compile(
    r"(?:collective|watchdog|worknccl|nccl).{0,160}(?:timed\s*out|timeout)"
    r"|(?:timed\s*out|timeout).{0,160}(?:collective|watchdog|worknccl|nccl)",
    re.I,
)


def event_kind(message: str) -> str | None:
    if not re.search(r"\bnccl\b|worknccl|collective|processgroupnccl", message, re.I):
        return None
    return "COLLECTIVE_TIMEOUT" if TIMEOUT.search(message) else "EXPLICIT_NCCL_ERROR"
