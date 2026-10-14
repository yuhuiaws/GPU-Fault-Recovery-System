"""Render administrator status; the JSON document keeps every engine field."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.aurora_placement import placement_findings
from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault.admin.ownership_release import status_findings
from gpu_fault.admin.resource_registry_outcome import FIELD, status_view
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


def _running_release_id(report: dict[str, Any] | None) -> str | None:
    """The release the site runs: configured (after a rollback, the one rolled
    back to), else live -- one and the same once a release is committed."""

    for key in ("configured_release", "live_release"):
        block = (report or {}).get(key)
        value = block.get("release_id") if isinstance(block, dict) else None
        if isinstance(value, str) and value:
            return value
    return None


def _with_registry_sync(stdout: str | None, view: dict[str, Any]) -> str | None:
    """The engine's document plus the deploy-host field, else ``stdout`` as is."""

    try:
        value = json.loads(stdout or "")
    except json.JSONDecodeError:
        return stdout
    if not isinstance(value, dict):
        return stdout
    return json.dumps({**value, FIELD: view}, indent=2, sort_keys=True) + "\n"


def print_status_report(
    stdout: str | None,
    *,
    site: RenderedSite | None = None,
    findings: Sequence[str] = (),
    site_file: Path | None = None,
) -> None:
    """Five readable lines on stderr, then the engine's JSON document.

    The JSON document is what scripts parse (`staging_live_evidence`, the
    acceptance recorders), so every engine field is passed through unchanged;
    the header is the part an administrator reads. ``findings`` are
    administrator-side findings the engine's report cannot carry, one more
    header line each; with ``site`` the stale ownership claims of the clusters
    it de-registered are read and added (the engine's report covers the
    clusters the site manages, the stale claim is on one it dropped), and so is
    an Aurora writer outside the control plane's zones (``aurora_placement``,
    read-only; the remedy is ``aurora-failback``, never run from here).

    With ``site_file`` -- the managed site, whose directory holds the deploy
    host's records -- the installation registry sync record is added as the
    ``installation_registry_sync`` field (when the output is one JSON object)
    and, unless it is a success for the release the site runs, as a finding.
    A finding, not a failing check: ``healthy`` and the exit code stay the
    engine's (``resource_registry_outcome`` says why).
    """
    if site is not None:
        state_dir = site_file.parent if site_file is not None else None
        findings = [
            *findings,
            *status_findings(site),
            *placement_findings(site, state_dir=state_dir),
        ]
    report = status_document(stdout or "")
    if site_file is not None:
        view = status_view(site_file.parent, release_id=_running_release_id(report))
        findings = [*findings, *([view["finding"]] if view["finding"] else [])]
        stdout = _with_registry_sync(stdout, view)
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
