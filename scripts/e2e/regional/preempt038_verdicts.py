"""Pure verdict functions and constants of GF-REGIONAL-PREEMPT-038.

The case proves ARCH-D6 and ARCH-D7 on a real regional deployment: the
periodic raw-evidence sweep deletes an expired row nobody needs, skips an
expired row an open incident of the same cluster still names (by node inside
the incident's window), deletes that row once the incident is RECOVERED, and
logs every sweep as one INFO line naming the kind, the count and the deleted
keys. The audit rows are synthetic (``audit-cluster``) and removed in
``finally``; the sweep itself is the deployed periodic runner.

Every function here judges the audit's JSON and the Pod logs; none touches a
cluster.
"""

from __future__ import annotations

import re
from typing import Any

CASE_ID = "GF-REGIONAL-PREEMPT-038"
CONFIRMATION = "PREEMPT038_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-PREEMPT-037"
CLEANUP_LOG = re.compile(r"cleanup raw_evidence deleted (\d+) rows; keys\[:20\]=(.*)$")


def audit_errors(report: dict[str, Any]) -> list[str]:
    """The three outcomes the sweep must produce, in order."""

    errors: list[str] = []
    if report.get("unrelated_deleted_after_seconds") is None:
        errors.append("the unrelated expired row was never deleted")
    if report.get("pinned_present_after_unrelated_deleted") is not True:
        errors.append(
            "the pinned row did not survive the sweep that removed the unrelated one"
        )
    if report.get("pinned_present_after_extra_wait") is not True:
        errors.append("the pinned row was deleted while its incident was still open")
    if report.get("pinned_deleted_after_recovered_seconds") is None:
        errors.append("the pinned row was not deleted after its incident recovered")
    if report.get("residual_rows") != 0:
        errors.append(f"{report.get('residual_rows')} audit rows remain in the store")
    return errors


def cleanup_lines(logs: str) -> list[dict[str, Any]]:
    lines = []
    for line in logs.splitlines():
        match = CLEANUP_LOG.search(line)
        if match is None:
            continue
        lines.append({"count": int(match.group(1)), "keys": match.group(2)})
    return lines


def log_errors(logs: str, *, unrelated_key: str, pinned_key: str) -> list[str]:
    """ARCH-D7: each deletion is one line naming the deleted keys."""

    lines = cleanup_lines(logs)
    if not lines:
        return ["no 'cleanup raw_evidence deleted' line in the control-plane logs"]
    errors: list[str] = []
    if not any(unrelated_key in item["keys"] for item in lines):
        errors.append(
            f"no cleanup line names the deleted unrelated key {unrelated_key}"
        )
    if not any(pinned_key in item["keys"] for item in lines):
        errors.append(
            f"no cleanup line names the pinned key {pinned_key} after recovery"
        )
    for item in lines:
        if item["count"] <= 0:
            errors.append("a cleanup line reports zero rows; empty sweeps must not log")
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"
