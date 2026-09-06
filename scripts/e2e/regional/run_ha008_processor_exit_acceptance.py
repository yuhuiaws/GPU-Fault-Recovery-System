#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .acceptance_scope import scoped_case_evidence
else:
    from acceptance_scope import scoped_case_evidence


ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts/e2e/regional/run_ha008_processor_exit_probe.py"
CASE_ID = "GF-REGIONAL-HA-008"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n"
    )
    path.chmod(0o600)


def _write_text(path: Path, value: str) -> None:
    path.write_text(value)
    path.chmod(0o600)


def _source_environment() -> dict[str, str]:
    env = dict(os.environ)
    source = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    return env


def _sqlite_store(path: Path):
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("gpu_fault.store").SqliteStore(str(path))


def _run_branch(evidence_dir: Path, name: str, *, fail_release: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"gpu-fault-ha008-{name}-") as directory:
        root = Path(directory)
        database = root / "processor.db"
        claim_file = root / "claim.json"
        command = [
            sys.executable,
            str(PROBE),
            "--database",
            str(database),
            "--claim-file",
            str(claim_file),
        ]
        if fail_release:
            command.append("--fail-release")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_source_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        _write_text(evidence_dir / f"{name}.stdout", completed.stdout)
        _write_text(evidence_dir / f"{name}.stderr", completed.stderr)
        if not claim_file.is_file():
            raise RuntimeError(
                f"{name}: child exited before writing claim metadata; "
                f"returncode={completed.returncode}"
            )
        claim = json.loads(claim_file.read_text())
        claim_file.unlink()
        store = _sqlite_store(database)
        try:
            request = store.get_processor_request(claim["request_id"])
            status_after_exit = request.status.value
            lease_expires_at = request.lease_expires_at
            if lease_expires_at is not None:
                delay = (lease_expires_at - datetime.now(timezone.utc)).total_seconds()
                if delay > 0:
                    time.sleep(delay + 0.2)
            # F-D7 charges the deadline to the request: the release carries a
            # retry backoff, so the takeover has to wait it out like a real
            # second owner would.
            if request.not_before is not None:
                delay = (
                    request.not_before - datetime.now(timezone.utc)
                ).total_seconds()
                if delay > 0:
                    time.sleep(delay + 0.2)
            claimed = store.claim_active_processor_requests(
                f"ha008-takeover-{name}",
                now=datetime.now(timezone.utc),
                lease_duration=timedelta(seconds=30),
                limit=1,
            )
            if len(claimed) != 1:
                raise RuntimeError(f"{name}: second owner could not claim request")
            takeover = claimed[0]
            stale_rejected = False
            try:
                store.complete_active_processor_request(
                    claim["request_id"],
                    claim["owner_id"],
                    int(claim["lane_epoch"]),
                    claim["lease_token"],
                    response_status=200,
                    response_content_type="application/json",
                    response_body_base64=base64.b64encode(b'{"late":true}').decode(),
                )
            except ValueError:
                stale_rejected = True
            store.complete_active_processor_request(
                takeover.request_id,
                str(takeover.lease_owner),
                int(takeover.leader_epoch),
                str(takeover.lease_token),
                response_status=200,
                response_content_type="application/json",
                response_body_base64=base64.b64encode(b'{"completed":true}').decode(),
            )
            final = store.get_processor_request(takeover.request_id)
        finally:
            store.close()
        return {
            "branch": name,
            "fail_release": fail_release,
            "exit_code": completed.returncode,
            "request_id": claim["request_id"],
            "first_owner": claim["owner_id"],
            "first_lane_epoch": claim["lane_epoch"],
            "status_after_exit": status_after_exit,
            "second_owner": takeover.lease_owner,
            "second_lane_epoch": takeover.leader_epoch,
            "stale_result_rejected": stale_rejected,
            "final_status": final.status.value,
            "final_response_status": final.response_status,
            "release_failure_logged": (
                "HA008 injected release failure"
                in f"{completed.stdout}\n{completed.stderr}"
            ),
        }


def run_acceptance(evidence_dir: Path) -> dict:
    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    normal = _run_branch(evidence_dir, "release-ok", fail_release=False)
    failed = _run_branch(evidence_dir, "release-fail", fail_release=True)
    errors = []
    for branch in (normal, failed):
        if branch["exit_code"] != 70:
            errors.append(f"{branch['branch']} exit code is not 70")
        if not branch["stale_result_rejected"]:
            errors.append(f"{branch['branch']} accepted the old result")
        if branch["final_status"] != "COMPLETED":
            errors.append(f"{branch['branch']} was not completed by takeover")
        if branch["final_response_status"] != 200:
            errors.append(f"{branch['branch']} final response is not 200")
        if branch["second_owner"] == branch["first_owner"]:
            errors.append(f"{branch['branch']} did not change owner")
    if normal["status_after_exit"] != "PENDING":
        errors.append("normal branch did not release the request before exit")
    if failed["status_after_exit"] != "LEASED":
        errors.append("release-failure branch did not preserve the leased request")
    if not failed["release_failure_logged"]:
        errors.append("release-failure branch did not log the injected error")
    result = {
        "case_id": CASE_ID,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "branches": [normal, failed],
        "sensitive_temp_files_removed": True,
    }
    _write_json(evidence_dir / f"{CASE_ID}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated HA-008 fatal-exit acceptance fixture."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="optional private evidence directory",
    )
    args = parser.parse_args()
    os.umask(0o077)
    if args.run_dir is not None:
        result = run_acceptance(args.run_dir)
    else:
        with tempfile.TemporaryDirectory(
            prefix="gpu-fault-ha008-acceptance-"
        ) as directory:
            result = run_acceptance(Path(directory))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
