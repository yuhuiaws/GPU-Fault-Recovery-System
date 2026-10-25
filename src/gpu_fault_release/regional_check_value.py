"""The value one preflight or verify check returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckValue:
    summary: str
    details: Any = None
    # ``WARN`` is a finding that does not fail the report: something the
    # operator should read, not something the deploy should stop on.
    status: str = "PASS"
