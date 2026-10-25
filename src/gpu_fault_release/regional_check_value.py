"""The value one preflight or verify check returns, and how one opts out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CheckSkipped(RuntimeError):
    """A check that does not apply to this site (SKIP, or FAIL where required)."""


@dataclass(frozen=True)
class CheckValue:
    summary: str
    details: Any = None
    # ``WARN`` is a finding that does not fail the report: something the
    # operator should read, not something the deploy should stop on.
    status: str = "PASS"
