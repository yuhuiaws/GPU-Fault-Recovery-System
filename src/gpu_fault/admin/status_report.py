"""Render administrator status without changing the machine-readable document."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault.admin.ownership_release import status_findings
from gpu_fault.admin.site import RenderedSite
from gpu_fault_release.regional_admin_commands import status_header_lines


def status_document(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        starts = [index for index, line in enumerate(lines) if line.strip() == "{"]
        if not starts:
            return None
        try:
            value = json.loads("\n".join(lines[starts[0] :]))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def print_status_report(
    stdout: str | None,
    *,
    site: RenderedSite | None = None,
    findings: Sequence[str] = (),
) -> None:
    """Five readable lines on stderr, then the engine's stdout byte for byte.

    The JSON document is what scripts parse (`staging_live_evidence`, the
    acceptance recorders), so it is passed through unchanged; the header is the
    part an administrator reads. ``findings`` are administrator-side findings
    the engine's report cannot carry, one more header line each; with ``site``
    the stale ownership claims of the clusters it de-registered are read and
    added (the engine's report covers the clusters the site manages, the stale
    claim is on one it dropped).
    """
    if site is not None:
        findings = [*findings, *status_findings(site)]
    report = status_document(stdout or "")
    log_path = os.environ.get(ADMIN_LOG_ENVIRONMENT, "").strip()
    destination = "stdout" + (f" (also in {log_path})" if log_path else "")
    lines = (
        status_header_lines(report, json_destination=destination)
        if report is not None
        else ["status printed no JSON report; see the output below"]
    )
    lines = [*lines, *(f"finding: {item}" for item in findings)]
    print(
        "\n".join(f"gpu-fault-admin status: {line}" for line in lines), file=sys.stderr
    )
    sys.stderr.flush()
    if stdout:
        sys.stdout.write(stdout)
        sys.stdout.flush()
