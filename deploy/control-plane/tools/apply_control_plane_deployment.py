"""Apply one final CPU Pod template, including out-of-band legacy env removal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from gpu_fault.admin.diagnostics import diagnostic_text
from pathlib import Path
from typing import Any

LEGACY_NOTIFICATION_ENV = frozenset(
    {"GPU_FAULT_ALLOW_EMAIL", "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"}
)
LEGACY_INGRESS_ENV = frozenset(
    {
        "GPU_FAULT_PROCESSOR_WORKERS",
        "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
        "GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS",
        "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
        "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
        "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
    }
)


def containers(document: dict[str, Any]) -> list[dict[str, Any]]:
    return list(document["spec"]["template"]["spec"]["containers"])


def apply_deployment(
    desired: dict[str, Any], live: dict[str, Any], command: list[str]
) -> None:
    name = desired["metadata"]["name"]
    if desired.get("kind") != "Deployment" or name not in {
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    }:
        raise ValueError("expected a CPU role Deployment")
    existing: dict[str, Any] = next(
        (item for item in live["items"] if item["metadata"]["name"] == name), {}
    )
    legacy = LEGACY_NOTIFICATION_ENV | (
        LEGACY_INGRESS_ENV if name == "gpu-fault-api-ha" else frozenset()
    )
    # Verbatim rollback may deliberately carry an old env name. Its supplied
    # snapshot wins; normal desired templates no longer declare these names.
    removals = {
        container["name"]: legacy - {env["name"] for env in container.get("env", [])}
        for container in containers(desired)
    }
    cleanup = bool(existing) and any(
        env["name"] in removals.get(container["name"], frozenset())
        for container in containers(existing)
        for env in container.get("env", [])
    )
    if not cleanup:
        result = subprocess.run(
            [*command, "apply", "-f", "-"],
            input=json.dumps(desired),
            text=True,
            check=False,
            capture_output=True,
        )
        if result.returncode:
            raise ValueError(
                f"CPU Deployment apply failed: {diagnostic_text(result.stderr)}"
            )
        print(result.stdout, end="")
        return
    uid = existing["metadata"].get("uid")
    if not uid:
        raise ValueError("legacy CPU Deployment has no UID")
    for _attempt in range(3):
        # Ask Kubernetes to calculate its normal three-way apply and admission
        # defaults, then remove only the legacy entries in that final object.
        # Dry-run persists nothing; replace's resourceVersion is the CAS guard.
        preview = subprocess.run(
            [*command, "apply", "--dry-run=server", "-f", "-", "-o", "json"],
            input=json.dumps(desired),
            text=True,
            check=False,
            capture_output=True,
        )
        if preview.returncode:
            raise ValueError(
                f"CPU Deployment dry-run failed: {diagnostic_text(preview.stderr)}"
            )
        merged = json.loads(preview.stdout)
        metadata = merged["metadata"]
        if (
            merged.get("kind") != "Deployment"
            or metadata.get("uid") != uid
            or metadata.get("name") != name
            or metadata.get("namespace") != desired["metadata"]["namespace"]
            or not metadata.get("resourceVersion")
        ):
            raise ValueError("CPU Deployment identity changed during legacy cleanup")
        for container in containers(merged):
            rejected = removals.get(container["name"], frozenset())
            if "env" in container:
                container["env"] = [
                    env for env in container["env"] if env["name"] not in rejected
                ]
        merged.pop("status", None)
        metadata.pop("managedFields", None)
        result = subprocess.run(
            [*command, "replace", "-f", "-"],
            input=json.dumps(merged),
            text=True,
            check=False,
            capture_output=True,
        )
        if not result.returncode:
            print(result.stdout, end="")
            return
        if "(Conflict)" not in result.stderr:
            raise ValueError(
                f"CPU Deployment replacement failed: {diagnostic_text(result.stderr)}"
            )
    raise ValueError("CPU Deployment changed repeatedly during legacy cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("scoped kubectl command is required")
    try:
        apply_deployment(
            json.load(sys.stdin), json.loads(args.live.read_text()), command
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"CPU role apply refused: {exc}") from None


if __name__ == "__main__":
    main()
