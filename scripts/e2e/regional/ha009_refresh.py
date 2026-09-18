"""UID-owned refresh Job execution and detached fallback for HA009."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shlex
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.ha_probe_resources import OwnedProbeResources


class CaseError(RuntimeError):
    pass


def secret_versions(
    aws: Callable[..., dict[str, Any]], secret_arn: str
) -> dict[str, Any]:
    """Read staged version history without accepting ambiguous pagination."""
    items: list[dict[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    for _ in range(100):
        arguments = ["--secret-id", secret_arn, "--max-results", "100"]
        if token:
            arguments += ["--next-token", token]
        value = aws("secretsmanager", "list-secret-version-ids", *arguments)
        page = value.get("Versions")
        if not isinstance(page, list) or any(
            not isinstance(item, dict) for item in page
        ):
            raise CaseError("secret version page is malformed")
        items.extend(page)
        token = value.get("NextToken")
        if token is None or token == "":
            break
        if not isinstance(token, str) or not token.strip() or token in seen_tokens:
            raise CaseError("secret version pagination token is invalid or repeated")
        seen_tokens.add(token)
    else:
        raise CaseError("secret version pagination did not terminate")
    if any(
        not isinstance(item.get("VersionId"), str)
        or not item["VersionId"].strip()
        or not isinstance(item.get("VersionStages"), list)
        or any(
            not isinstance(stage, str) or not stage for stage in item["VersionStages"]
        )
        for item in items
    ):
        raise CaseError("secret version identity or stages are malformed")
    versions = [
        {
            "version_id": str(item.get("VersionId") or ""),
            "stages": sorted(str(stage) for stage in item.get("VersionStages", [])),
            "created_at": str(item.get("CreatedDate") or ""),
        }
        for item in items
    ]
    if len({item["version_id"] for item in versions}) != len(versions):
        raise CaseError("secret version history repeats a version")
    stages = {
        stage: item["version_id"] for item in versions for stage in item["stages"]
    }
    if len(stages) != sum(len(item["stages"]) for item in versions):
        raise CaseError("secret version stage ownership is ambiguous")
    return {"versions": versions, "stages": stages}


def start_refresh_watchdog(
    case_dir: Path,
    manifest: dict[str, Any],
    resources: OwnedProbeResources,
    *,
    resume_command: Callable[[str, str], list[str]],
    delay_seconds: int,
) -> subprocess.Popen[str]:
    """Resume an owned Job after the delay unless its process group is disarmed."""
    job_name = manifest["metadata"]["name"]
    manifest["spec"]["suspend"] = True
    resources.create(manifest)
    job = resources.owned("Job", job_name)
    if job is None:
        raise CaseError("Aurora refresh watchdog Job disappeared")
    log_path = case_dir / "refresh-watchdog.log"
    handle = log_path.open("w", encoding="utf-8")
    os.chmod(log_path, 0o600)
    script = f"sleep {int(delay_seconds)}; exec " + shlex.join(
        resume_command(job_name, job["metadata"]["uid"])
    )
    process = subprocess.Popen(
        ["/bin/bash", "-c", script],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    write_json_atomic(
        case_dir / "refresh-watchdog.json",
        {"pid": process.pid, "job": job_name, "delay_seconds": int(delay_seconds)},
    )
    return process


def stop_refresh_watchdog(process: subprocess.Popen[str] | None) -> dict[str, Any]:
    """Kill the watchdog's process group; never raise from cleanup."""
    if process is None:
        return {"armed": False}
    if process.poll() is not None:
        return {"armed": True, "fired": True, "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except Exception as exc:
        return {
            "armed": True,
            "fired": False,
            "stop_error": type(exc).__name__,
        }
    return {"armed": True, "fired": False, "disarmed": True}


def run_refresh_job(
    case_dir: Path,
    manifest: dict[str, Any],
    resources: OwnedProbeResources,
    *,
    control: Callable[..., str],
    timeout_seconds: int,
) -> dict[str, Any]:
    name = manifest["metadata"]["name"]
    resources.create(manifest)
    control(
        "wait",
        "--for=condition=complete",
        f"job/{name}",
        f"--timeout={timeout_seconds - 100}s",
        check=False,
        timeout=timeout_seconds,
    )
    job = json.loads(control("get", "job", name, "-o", "json"))
    succeeded = int(job.get("status", {}).get("succeeded", 0)) == 1
    if resources.owned("Job", name) is None:
        raise CaseError("Aurora refresh Job disappeared")
    logs = control("logs", f"job/{name}", timeout=120)
    if not succeeded:
        raise CaseError("Aurora refresh Job did not succeed")
    outcomes = set(
        re.findall(r"\brotated=(True|False)\s+restarted=(True|False)\b", logs)
    )
    if len(outcomes) != 1:
        raise CaseError("Aurora refresh Job has no unambiguous safe outcome")
    rotated, restarted = outcomes.pop()
    return {
        "name": name,
        "succeeded": succeeded,
        "logs": [f"rotated={rotated} restarted={restarted}"],
        "log_sha256": hashlib.sha256(logs.encode()).hexdigest(),
    }
