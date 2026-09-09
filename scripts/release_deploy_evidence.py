"""The evidence one release records, and the checks that read it back.

A release is only as good as what it can prove afterwards: the manifest it
deployed, the phase record it advances, the verification and stability reports it
writes, and the release-diff shape it expects to find once the rollout is
applied. Those are all pure functions of a path or a report document, so they
live apart from the driver that runs the commands producing them -- which keeps
``release_deploy`` to the orchestration and lets the predicates be read (and
tested) without a cluster.

``ReleaseDeployError`` and ``PreparedRelease`` are defined here because this is
the lower of the two modules and both halves use them; ``release_deploy``
re-exports them, so the error a caller catches and the record it receives are
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.resource_registry import sync_installation_resource_registry
from gpu_fault.admin.site import RenderedSite

RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ReleaseDeployError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedRelease:
    site_file: Path
    release_id: str
    runtime_profile_version: str
    agent_config_digest: str
    profile_change_kind: str
    profile_approval: str | None
    state_dir: Path
    site_changed: bool


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def release_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "dist/current-release.json"
    if not path.is_file():
        raise ReleaseDeployError(
            "dist/current-release.json is missing; the build gate did not produce a release"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseDeployError(f"invalid release manifest {path}: {exc}") from exc
    release_id = str(manifest.get("release_id") or "").strip()
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ReleaseDeployError("release manifest has no valid release_id")
    immutable = root / "dist" / release_id / "release.json"
    if not immutable.is_file():
        raise ReleaseDeployError(
            f"content-addressed release manifest is missing: {immutable}"
        )
    if immutable.read_bytes() != path.read_bytes():
        raise ReleaseDeployError(
            "current-release.json differs from its content-addressed release.json"
        )
    return manifest, release_id


def update_phase(prepared: PreparedRelease, phase: str, **values: Any) -> None:
    state_path = prepared.state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": phase, **values})
    write_json_atomic(state_path, state)


def verification_metadata(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASSED",
        "path": str(path),
        "sha256": sha256_file(path),
        "verified_at": utc_now(),
        "summary": report.get("summary"),
    }


def validate_verification_report(report: dict[str, Any]) -> None:
    if report.get("mode") != "verify":
        raise ReleaseDeployError("verification command returned the wrong report mode")
    if report.get("healthy") is not True:
        raise ReleaseDeployError("verification report is not healthy")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("FAIL") != 0:
        raise ReleaseDeployError("verification report has an invalid summary")
    if not isinstance(report.get("checks"), list):
        raise ReleaseDeployError("verification report has no check evidence")


def validate_stability_report(report: dict[str, Any]) -> None:
    if report.get("mode") != "stability" or report.get("healthy") is not True:
        raise ReleaseDeployError("release stability report is not healthy")
    if int(report.get("window_seconds") or 0) < 120:
        raise ReleaseDeployError("release stability report has an invalid window")


def release_summary_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in ("release_status_error", "next_deploy_error"):
        if report.get(field):
            warnings.append(f"{field}: {report[field]}")
    next_deploy = report.get("next_deploy")
    if not isinstance(next_deploy, dict):
        warnings.append("release summary has no next_deploy classification")
    elif next_deploy.get("kind") != "NOOP":
        warnings.append(
            "release summary reports a non-NOOP next deploy: "
            + json.dumps(next_deploy, sort_keys=True)
        )
    return warnings


def is_clean_noop_diff(report: dict[str, Any]) -> bool:
    next_deploy = report.get("next_deploy")
    changed = next_deploy.get("changed") if isinstance(next_deploy, dict) else None
    return (
        report.get("mode") == "release-diff"
        and isinstance(next_deploy, dict)
        and next_deploy.get("kind") == "NOOP"
        and next_deploy.get("action", "upgrade") == "upgrade"
        and next_deploy.get("resume", False) is False
        and isinstance(changed, list)
        and set(changed).issubset({"release_delivery", "rendered_manifests"})
        and isinstance(report.get("state_sha256"), str)
        and len(str(report["state_sha256"])) == 64
    )


def is_pending_commit_diff(report: dict[str, Any], release_id: str) -> bool:
    """Is this the state a successful deploy of `release_id` leaves behind?

    `rollout deploy` ends at `phase=complete, transaction_committed=False`, which
    `next_deploy` labels `resume: True, pending_commit: True`: every component is
    applied, quick validation passed, and the commit this driver runs after the
    stability window is the only outstanding step. That is emphatically not a
    clean NOOP -- the persisted `changed` set still names everything the
    transaction touched -- so the NOOP test rejected it and the driver threw away
    the read-only verifier evidence on every real deploy.

    Nothing here is a substitute for a probe. It only decides whether the
    evidence still describes this release; `quick_validation_evidence` on the
    other side re-checks the release id, the delivery digest, the site identity,
    a ten-minute freshness bound and the live release-state digest before any
    verifier is skipped, and says why it declined otherwise.
    """

    next_deploy = report.get("next_deploy")
    if not isinstance(next_deploy, dict):
        return False
    return (
        report.get("mode") == "release-diff"
        and next_deploy.get("pending_commit") is True
        and next_deploy.get("action", "upgrade") == "upgrade"
        and next_deploy.get("release_id") == release_id
        and isinstance(report.get("state_sha256"), str)
        and len(str(report["state_sha256"])) == 64
    )


def installation_resource_registry_record(
    site: RenderedSite,
) -> tuple[dict[str, Any], str | None]:
    """Synchronize the installed-resource registry and describe the outcome.

    For the driver's completion step, whose transaction is already committed:
    the registry is a description of what was installed, so a failure to write it
    (AWS discovery, one control-plane POST, a direct-Aurora fallback) is a
    warning on the release record, not a failed release. The bootstrap path in
    `cluster_join` keeps calling `sync_installation_resource_registry` directly
    and fail-closed, because there the registry is the source the join reads.
    Returns ``(record, warning)``; ``warning`` is ``None`` on success.
    """

    try:
        path = sync_installation_resource_registry(site)
    except Exception as exc:
        warning = (
            "installation resource registry not synchronized: "
            f"{type(exc).__name__}: {exc}"
        )
        return {"status": "UNAVAILABLE", "synced_at": utc_now(), "error": warning}, (
            warning
        )
    return {"status": "SYNCED", "path": str(path), "synced_at": utc_now()}, None


def finalize_quick_validation_evidence(
    path: Path,
    *,
    release_id: str,
    collect_diff: Callable[[], dict[str, Any]],
) -> Path | None:
    """Stamp the deploy's quick-validation evidence with the state it proved.

    `rollout deploy` writes the evidence at `path` -- the canonical
    `<state-dir>/quick-validation.json` that `gpu-fault-admin status` reads. It
    becomes reusable only once the live release state is known to be the one the
    probes ran against: a pending commit for `release_id`, or a clean NOOP. That
    digest is written into the file (privately, atomically) and the path is
    returned for `verify`. Evidence that cannot be vouched for is removed rather
    than left where the next `status` would find it, report a fallback reason
    and re-run every probe anyway.
    """

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("release_id") != release_id
        ):
            raise ValueError("quick validation evidence identity is invalid")
        diff = collect_diff()
        if not (is_pending_commit_diff(diff, release_id) or is_clean_noop_diff(diff)):
            raise ValueError(
                "post-deploy release state is neither a pending commit for "
                f"{release_id} nor a clean NOOP"
            )
        value["release_state_sha256"] = diff["state_sha256"]
        value["finalized_at"] = utc_now()
        write_json_atomic(path, value)
        path.chmod(0o600)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            "release-deploy: quick validation evidence was not reusable; "
            f"full verification will run: {exc}",
            file=sys.stderr,
        )
        path.unlink(missing_ok=True)
        return None
    return path
