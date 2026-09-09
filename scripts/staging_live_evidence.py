from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV


class LiveEvidenceError(RuntimeError):
    pass


class RuntimeProfileChangePending(LiveEvidenceError):
    """The site's Runtime Profile template no longer matches the live Profile."""


def runtime_profile_policy_evidence(
    site_file: Path,
    *,
    repository_root: Path,
    runner=subprocess.run,
) -> dict[str, object]:
    """The live Runtime Profile's policy digest, or a refusal to call it unchanged.

    The Profile template (``spec.runtimeProfile.templateSource``) lives outside
    the source repository and outside the site file, so neither the source
    fingerprint nor ``site_sha256`` moves when an operator edits it. Without
    this reading a template-only change classified as ``UNCHANGED`` and skipped
    the release engine -- and with it the ``profile-plan.json`` stop the change
    needed (observed live on 2026-09-07: a new capability claim in the template
    never reached the plan).

    The plan comes from the release engine itself (``release_deploy.py
    --profile-plan-json``), run the way ``gpu-fault-admin deploy`` runs it: this
    interpreter with the snapshot's ``src`` on ``PYTHONPATH``. The deploy-host
    bundle carries only the admin subset of ``gpu_fault``, so the planner cannot
    be imported here; and a second implementation of "unchanged" would drift.
    """

    command = [
        sys.executable,
        str(repository_root / "scripts/release_deploy.py"),
        "--site",
        str(site_file),
        "--profile-plan-json",
    ]
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    completed = runner(
        command,
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise LiveEvidenceError(
            f"Runtime Profile plan failed ({completed.returncode}): "
            + (completed.stderr or "").strip()
        )
    try:
        payload = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        raise LiveEvidenceError("Runtime Profile plan is invalid") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("policy_digest"), str
    ):
        raise LiveEvidenceError("Runtime Profile plan must be a JSON object")
    change_kind = payload.get("change_kind")
    if change_kind != "UNCHANGED":
        raise RuntimeProfileChangePending(
            "Runtime Profile template differs from the live Profile "
            f"({change_kind}); the release engine must plan it"
        )
    return {
        "runtime_profile_sha256": payload.get("live_profile_sha256"),
        "runtime_profile_policy_digest": payload["policy_digest"],
    }


def read_live_status(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    lock_fd: int | None = None,
) -> dict[str, object]:
    """The quick ``gpu-fault-admin status`` report of the managed site, as read.

    This is the one status call a deploy makes. Its ``live_release`` block
    (phase, release id) and ``next_deploy`` decide the consent refusals and the
    join short-circuit in the first minute; :func:`collect_live_deploy_evidence`
    turns the same report into the classification evidence, so the report is
    read once and handed on rather than read twice.
    """

    command = [
        str(venv / "bin/gpu-fault-admin"),
        "status",
        "--state-dir",
        str(state_dir),
    ]
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    environment = (
        {**os.environ, SITE_OPERATION_LOCK_FD_ENV: str(lock_fd)}
        if lock_fd is not None
        else None
    )
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        pass_fds=(lock_fd,) if lock_fd is not None else (),
    )
    if completed.returncode:
        raise LiveEvidenceError(
            f"status failed ({completed.returncode}): "
            + (completed.stderr or "").strip()
        )
    try:
        report = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        raise LiveEvidenceError("live deployment status is invalid") from exc
    if not isinstance(report, dict):
        raise LiveEvidenceError("live deployment status must be a JSON object")
    return report


def collect_live_deploy_evidence(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    lock_fd: int | None = None,
    report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The pre-apply evidence: from ``report`` when the caller already read it."""

    if report is None:
        report = read_live_status(
            repository_root=repository_root,
            state_dir=state_dir,
            venv=venv,
            lock_fd=lock_fd,
        )
    if report.get("mode") != "status" or report.get("healthy") is not True:
        raise LiveEvidenceError(
            "live deployment is not a healthy committed NOOP release"
        )
    release_id, state_sha256 = committed_noop_identity(report)
    site_file = state_dir / "site.yaml"
    return _live_evidence(
        site_file,
        release_id=release_id,
        state_sha256=state_sha256,
        **runtime_profile_policy_evidence(site_file, repository_root=repository_root),
    )


def committed_noop_identity(summary: Mapping[str, object]) -> tuple[str, str]:
    """``(release_id, state_sha256)`` of a committed release a deploy would NOOP.

    ``summary`` is a release summary -- the ``release-summary`` document itself
    or the ``status`` document that embeds one; both carry ``live_release``,
    ``configured_release`` and ``next_deploy``.
    """

    live = summary.get("live_release")
    configured = summary.get("configured_release")
    next_deploy = summary.get("next_deploy")
    if (
        not isinstance(live, Mapping)
        or live.get("phase") != "complete"
        or live.get("transaction_committed") is not True
        or not isinstance(configured, Mapping)
        or configured.get("release_id") != live.get("release_id")
        or not isinstance(next_deploy, Mapping)
        or next_deploy.get("kind") != "NOOP"
    ):
        raise LiveEvidenceError(
            "live deployment is not a healthy committed NOOP release"
        )
    release_id = live.get("release_id")
    state_sha256 = live.get("state_sha256")
    if not isinstance(release_id, str) or not isinstance(state_sha256, str):
        raise LiveEvidenceError("live deployment identity is incomplete")
    return release_id, state_sha256


def _live_evidence(
    site_file: Path,
    *,
    release_id: str,
    state_sha256: str,
    runtime_profile_sha256: object,
    runtime_profile_policy_digest: object,
) -> dict[str, object]:
    # One shape for both readings. ``successful_source_live_matches`` compares
    # the stored dictionary with the next run's live reading key for key, so
    # the record written from the release driver's files and the record read
    # from ``status`` must agree on every field or every deploy reclassifies.
    return {
        "site_sha256": hashlib.sha256(site_file.read_bytes()).hexdigest(),
        "runtime_profile_sha256": runtime_profile_sha256,
        "runtime_profile_policy_digest": runtime_profile_policy_digest,
        "release_id": release_id,
        "state_sha256": state_sha256,
        "phase": "complete",
        "transaction_committed": True,
        "next_deploy_kind": "NOOP",
    }


RELEASE_DEPLOY_DIR = "release-deploy"
RELEASE_SUMMARY_REPORT = "release-summary.json"
COMPLETED_RELEASE_PHASE = "COMPLETED"


def latest_release_record(state_dir: Path) -> Path:
    """The newest ``release-deploy/<release_id>/state.json`` the driver wrote."""

    records = sorted(
        (state_dir / RELEASE_DEPLOY_DIR).glob("*/state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not records:
        raise LiveEvidenceError(f"no release record under {state_dir}")
    return records[0]


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveEvidenceError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise LiveEvidenceError(f"{label} must be a JSON object: {path}")
    return value


def live_evidence_from_release_record(
    *,
    state_dir: Path,
    record: Path | None = None,
) -> dict[str, object]:
    """The success record, from the files the release driver just wrote.

    ``scripts/release_deploy.py`` ends a release by committing the transaction,
    collecting ``rollout release-summary`` into ``release-summary.json`` and
    marking ``state.json`` COMPLETED. That summary is the same document a
    ``gpu-fault-admin status`` would collect seconds later -- ``live_release``,
    ``configured_release``, ``next_deploy`` -- and the live Profile digest it
    would plan is the one the driver just published, so the success record is
    assembled here instead of paying for a second full report (about 45
    seconds) to read back what was just written. The next run compares this
    dictionary with its own live reading, so the fields are the same and come
    from the same sources: the live state's ``runtime_profile_sha256`` is the
    configured Profile's ``source_sha256`` the engine wrote at commit, and the
    policy digest is the plan's.
    """

    record = record if record is not None else latest_release_record(state_dir)
    state = _json_object(record, "release record")
    phase = state.get("phase")
    if phase != COMPLETED_RELEASE_PHASE:
        raise LiveEvidenceError(f"release record {record} is {phase}, not COMPLETED")
    verification = state.get("verification")
    if not isinstance(verification, Mapping) or verification.get("status") != "PASSED":
        raise LiveEvidenceError(f"release record {record} has no passed verification")
    summary_meta = state.get("release_summary")
    if not isinstance(summary_meta, Mapping) or str(
        summary_meta.get("status") or ""
    ) not in ("AVAILABLE", "AVAILABLE_WITH_WARNINGS"):
        raise LiveEvidenceError(f"release record {record} has no release summary")
    summary_path = record.parent / RELEASE_SUMMARY_REPORT
    try:
        digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LiveEvidenceError(
            f"release summary is unreadable: {summary_path}"
        ) from exc
    if digest != summary_meta.get("sha256"):
        raise LiveEvidenceError(f"release summary does not match its record: {record}")
    summary = _json_object(summary_path, "release summary")
    if summary.get("mode") != "release-summary":
        raise LiveEvidenceError(f"release summary has the wrong mode: {summary_path}")
    release_id, state_sha256 = committed_noop_identity(summary)
    if state.get("release_id") != release_id:
        raise LiveEvidenceError(
            f"release record and summary disagree on the release: {record}"
        )
    profile_change = state.get("profile_change")
    configured_profile = summary.get("configured_runtime_profile")
    if not isinstance(profile_change, Mapping) or not isinstance(
        configured_profile, Mapping
    ):
        raise LiveEvidenceError(
            f"release record has no Runtime Profile identity: {record}"
        )
    policy_digest = profile_change.get("policy_digest")
    profile_sha256 = configured_profile.get("source_sha256")
    if not isinstance(policy_digest, str) or not isinstance(profile_sha256, str):
        raise LiveEvidenceError(
            f"release record Runtime Profile identity is incomplete: {record}"
        )
    return _live_evidence(
        state_dir / "site.yaml",
        release_id=release_id,
        state_sha256=state_sha256,
        runtime_profile_sha256=profile_sha256,
        runtime_profile_policy_digest=policy_digest,
    )


def successful_source_live_matches(
    previous: Mapping[str, object] | None,
    live_evidence: Mapping[str, object] | None,
) -> bool:
    if previous is None or live_evidence is None:
        return False
    stored = previous.get("live")
    return isinstance(stored, Mapping) and dict(stored) == dict(live_evidence)
