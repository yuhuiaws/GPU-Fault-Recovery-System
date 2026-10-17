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
from gpu_fault.admin.dcgm_declaration import (
    READ_ERRORS,
    STATUS_FIELD,
    declarations_document,
    next_deploy_with_declarations,
    read_dcgm_declarations,
    site_cluster_entries,
    unreadable_findings,
)
from gpu_fault.admin.ownership_release import status_findings
from gpu_fault.admin.rds_ca_bundle_drift import rds_ca_bundle_findings
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


def _with_deploy_host_fields(stdout: str | None, fields: dict[str, Any]) -> str | None:
    """The engine's document plus the deploy-host fields, else ``stdout`` as is."""

    try:
        value = json.loads(stdout or "")
    except json.JSONDecodeError:
        return stdout
    if not isinstance(value, dict):
        return stdout
    return json.dumps({**value, **fields}, indent=2, sort_keys=True) + "\n"


def dcgm_declaration_fields(
    report: dict[str, Any] | None, site_file: Path
) -> tuple[dict[str, Any], list[str]]:
    """``(fields, findings)`` of the plain EKS clusters' DCGM declarations.

    Nothing for a site without one, or for an engine that printed no document
    (the output stays the engine's). Each declaration is read as the next
    deploy's discovery reads it; a pending one rewrites ``next_deploy`` to
    what that deploy will do, in ``report`` too, so the header and every
    reader of the document see the same verdict. The checks are left as they
    are: the change is the next deploy's, not drift.
    """

    if report is None:
        return {}, []
    try:
        declarations = read_dcgm_declarations(site_cluster_entries(site_file))
    except READ_ERRORS as exc:
        return {}, [f"DCGM host engine declarations could not be read: {exc}"]
    if not declarations:
        return {}, []
    fields: dict[str, Any] = {STATUS_FIELD: declarations_document(declarations)}
    engine = report.get("next_deploy")
    if isinstance(engine, dict):
        overlaid = next_deploy_with_declarations(engine, declarations)
        if overlaid != engine:
            fields["next_deploy"] = report["next_deploy"] = overlaid
    return fields, unreadable_findings(declarations)


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
    read-only; the remedy is ``aurora-failback``, never run from here), and so
    are the RDS CA bundle lines (``rds_ca_bundle_drift``, read-only: the
    installed bundle against the release's pinned copy, the bundle AWS
    publishes against that copy, the Aurora instances' CA in both).

    With ``site_file`` -- the managed site, whose directory holds the deploy
    host's records -- the installation registry sync record is added as the
    ``installation_registry_sync`` field (when the output is one JSON object)
    and, unless it is a success for the release the site runs, as a finding.
    A finding, not a failing check: ``healthy`` and the exit code stay the
    engine's (``resource_registry_outcome`` says why). A site with plain EKS
    clusters also gets their DCGM host engine declarations
    (``dcgm_declaration_fields``): a label flipped since the last deploy is
    the next deploy's change, ``next deploy: DATA_PLANE_COMPATIBLE
    (dcgm_hostengine: ...)``, where the engine read the recorded entry only.
    """
    if site is not None:
        state_dir = site_file.parent if site_file is not None else None
        findings = [
            *findings,
            *status_findings(site),
            *placement_findings(site, state_dir=state_dir),
            *rds_ca_bundle_findings(site),
        ]
    report = status_document(stdout or "")
    if site_file is not None:
        fields, declaration_findings = dcgm_declaration_fields(report, site_file)
        view = status_view(site_file.parent, release_id=_running_release_id(report))
        findings = [
            *findings,
            *declaration_findings,
            *([view["finding"]] if view["finding"] else []),
        ]
        stdout = _with_deploy_host_fields(stdout, {**fields, FIELD: view})
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
