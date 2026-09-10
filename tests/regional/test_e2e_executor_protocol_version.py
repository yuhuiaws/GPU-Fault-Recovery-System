"""Acceptance probes must claim with the protocol the control plane accepts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "scripts/e2e/regional"
PINNED = re.compile(r"\"executor_protocol_version\":\s*[0-9]+")


def test_no_acceptance_script_pins_a_numeric_executor_protocol_version() -> None:
    """AUTH-016's claim probe carried protocol 2 after the round-trip batch raised
    the control plane's requirement to 3; the claim route answered 503 through
    every phase of a token rotation that was otherwise sound. Scripts take the
    version from gpu_fault.regional_compatibility, never a literal."""

    offenders = sorted(
        f"{path.relative_to(ROOT)}:{match.group(0)}"
        for path in E2E.rglob("*.py")
        for match in PINNED.finditer(path.read_text(encoding="utf-8"))
    )
    assert offenders == [], offenders
