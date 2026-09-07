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


def collect_live_deploy_evidence(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    lock_fd: int | None = None,
) -> dict[str, object]:
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
    live = report.get("live_release")
    configured = report.get("configured_release")
    next_deploy = report.get("next_deploy")
    if (
        report.get("mode") != "status"
        or report.get("healthy") is not True
        or not isinstance(live, dict)
        or live.get("phase") != "complete"
        or live.get("transaction_committed") is not True
        or not isinstance(configured, dict)
        or configured.get("release_id") != live.get("release_id")
        or not isinstance(next_deploy, dict)
        or next_deploy.get("kind") != "NOOP"
    ):
        raise LiveEvidenceError(
            "live deployment is not a healthy committed NOOP release"
        )
    release_id = live.get("release_id")
    state_sha256 = live.get("state_sha256")
    if not isinstance(release_id, str) or not isinstance(state_sha256, str):
        raise LiveEvidenceError("live deployment identity is incomplete")
    site_file = state_dir / "site.yaml"
    return {
        "site_sha256": hashlib.sha256(site_file.read_bytes()).hexdigest(),
        **runtime_profile_policy_evidence(site_file, repository_root=repository_root),
        "release_id": release_id,
        "state_sha256": state_sha256,
        "phase": "complete",
        "transaction_committed": True,
        "next_deploy_kind": "NOOP",
    }


def successful_source_live_matches(
    previous: Mapping[str, object] | None,
    live_evidence: Mapping[str, object] | None,
) -> bool:
    if previous is None or live_evidence is None:
        return False
    stored = previous.get("live")
    return isinstance(stored, Mapping) and dict(stored) == dict(live_evidence)
