#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for source_path in (ROOT, ROOT / "src"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

if __package__:
    from .acceptance_runner_common import write_json_atomic
    from .ha_evidence import isolated_chain
    from .regional_case_contract import case_evidence_path
else:
    from acceptance_runner_common import write_json_atomic
    from ha_evidence import isolated_chain
    from regional_case_contract import case_evidence_path


PROBE = ROOT / "scripts/e2e/regional/run_ha008_processor_exit_probe.py"
CASE_ID = "GF-REGIONAL-HA-008"
# What the processor itself emits (gpu_fault.processor.coordinator), at ERROR:
# LOGGER.exception(...) when the post-deadline release raises, and the
# deadline-exceeded record in every case. The probe's injected exception text is
# not evidence -- it would appear in the traceback whether or not the processor
# logged anything of its own.
PROCESSOR_LOGGER = "gpu_fault.processor.coordinator"
RELEASE_FAILURE_MESSAGE = "could not release request after its execution deadline"
DEADLINE_EXCEEDED_MESSAGE = "processor request execution deadline exceeded"


def _processor_error_logged(output: str, message: str) -> bool:
    return any(
        line.startswith(f"ERROR {PROCESSOR_LOGGER} ") and message in line
        for line in output.splitlines()
    )


def release_failure_logged(output: str) -> bool:
    """True if the processor logged its own release-failure ERROR line."""

    return _processor_error_logged(output, RELEASE_FAILURE_MESSAGE)


def deadline_exceeded_logged(output: str) -> bool:
    return _processor_error_logged(output, DEADLINE_EXCEEDED_MESSAGE)


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
        finally:
            store.close()
        takeover_process = subprocess.run(
            [*command[:6], "--takeover", "--owner", f"ha008-takeover-{name}"],
            cwd=ROOT,
            env=_source_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        _write_text(evidence_dir / f"{name}-takeover.stdout", takeover_process.stdout)
        _write_text(evidence_dir / f"{name}-takeover.stderr", takeover_process.stderr)
        if takeover_process.returncode:
            raise RuntimeError(f"{name}: replacement process failed")
        takeover = json.loads(takeover_process.stdout.strip().splitlines()[-1])
        claim_file.unlink()
        return {
            "branch": name,
            "fail_release": fail_release,
            "exit_code": completed.returncode,
            "request_id": claim["request_id"],
            "first_owner": claim["owner_id"],
            "first_lane_epoch": claim["lane_epoch"],
            "first_process_id": claim["process_id"],
            "status_after_exit": status_after_exit,
            **takeover,
            "release_failure_logged": release_failure_logged(
                f"{completed.stdout}\n{completed.stderr}"
            ),
            "deadline_exceeded_logged": deadline_exceeded_logged(
                f"{completed.stdout}\n{completed.stderr}"
            ),
        }


def run_acceptance(evidence_dir: Path) -> dict:
    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        normal = _run_branch(evidence_dir, "release-ok", fail_release=False)
        failed = _run_branch(evidence_dir, "release-fail", fail_release=True)
    except Exception as exc:
        result = {
            "case_id": CASE_ID,
            "verdict": "FAIL",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "branches": [],
            "sensitive_temp_files_removed": False,
        }
        write_json_atomic(evidence_dir / f"{CASE_ID}.json", result)
        return result
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
        if not branch.get("second_process_id") or branch["second_process_id"] in {
            branch.get("first_process_id"),
            os.getpid(),
        }:
            errors.append(f"{branch['branch']} was not taken over by another process")
        if not branch["deadline_exceeded_logged"]:
            errors.append(
                f"{branch['branch']} processor did not log the deadline-exceeded error"
            )
    if normal["status_after_exit"] != "PENDING":
        errors.append("normal branch did not release the request before exit")
    if normal["release_failure_logged"]:
        errors.append("normal branch logged a release failure it should not have")
    if failed["status_after_exit"] != "LEASED":
        errors.append("release-failure branch did not preserve the leased request")
    if not failed["release_failure_logged"]:
        errors.append(
            "release-failure branch: processor did not log its own "
            f"'{RELEASE_FAILURE_MESSAGE}' ERROR line"
        )
    result = {
        "case_id": CASE_ID,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "branches": [normal, failed],
        "sensitive_temp_files_removed": True,
    }
    write_json_atomic(evidence_dir / f"{CASE_ID}.json", result)
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
    parser.add_argument("--release-id", default="")
    parser.add_argument("--cluster-id", default="")
    args = parser.parse_args()
    os.umask(0o077)
    if args.run_dir is not None:
        path = case_evidence_path(args.run_dir, CASE_ID)
        try:
            binding = isolated_chain(args, CASE_ID)
            result = {**run_acceptance(path.parent), **binding}
        except Exception as exc:
            result = {
                "case_id": CASE_ID,
                "verdict": "FAIL",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        write_json_atomic(path, result)
    else:
        with tempfile.TemporaryDirectory(
            prefix="gpu-fault-ha008-acceptance-"
        ) as directory:
            result = run_acceptance(Path(directory))
            result.update(
                validation_scope="isolated-source", formal_sequence_satisfied=False
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
