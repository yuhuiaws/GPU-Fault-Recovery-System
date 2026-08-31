from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager


STATE_ROOT = Path("/var/lib/gpu-fault/acceptance")
QUIESCE_ROOT = Path("/var/lib/gpu-fault/quiesce")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


class ProbeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return completed


def safe_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def evidence_path(run_id: str) -> Path:
    return STATE_ROOT / f"preempt012-{safe_id(run_id)}.json"


def service_states() -> dict[str, str]:
    result = {}
    for service in (
        "nvidia-fabricmanager",
        "nvidia-dcgm",
        "nvidia-persistenced",
        "gpu-fault-gpu-persistence",
        "gpu-fault-metrics-collector",
        "gpu-fault-host-collector",
        "kubelet",
    ):
        completed = run(["systemctl", "is-active", service], check=False)
        result[service] = completed.stdout.strip() or "unknown"
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def cycle(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    incident_id = f"preempt012-{run_id}"
    manager = GpuServiceQuiesceManager(
        state_dir=str(QUIESCE_ROOT),
        failsafe_seconds=arguments.failsafe_seconds,
        retry_seconds=30,
        settle_seconds=2,
        restore_settle_seconds=10,
    )
    result: dict[str, Any] = {
        "run_id": run_id,
        "incident_id": incident_id,
        "started_at": utc_now(),
        "services_before": service_states(),
        "status": "RUNNING",
    }
    path = evidence_path(run_id)
    write_json(path, result)
    try:
        quiesce = manager.quiesce(
            incident_id=incident_id,
            workflow_request_id=f"workflow-{run_id}",
        )
        result["quiesced_at"] = utc_now()
        result["quiesce"] = quiesce
        result["services_quiesced"] = service_states()
        result["status"] = "QUIESCED"
        write_json(path, result)
        time.sleep(arguments.hold_seconds)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "FAILED"
    finally:
        try:
            result["restore"] = manager.restore(incident_id=incident_id)
        except Exception as exc:
            result["restore_error"] = f"{type(exc).__name__}: {exc}"
            result["status"] = "FAILED"
        result["restored_at"] = utc_now()
        result["services_after"] = service_states()
        result["quiesce_state_files"] = sorted(
            path.name for path in QUIESCE_ROOT.glob("quiesce-*.json")
        )
        if result["status"] != "FAILED":
            result["status"] = "COMPLETED"
        write_json(path, result)


def arm(arguments: argparse.Namespace) -> dict[str, Any]:
    run_id = safe_id(arguments.run_id)
    unit = f"gpu-fault-preempt012-{run_id}"
    run(["systemctl", "stop", unit + ".timer"], check=False)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            f"--on-active={arguments.delay_seconds}s",
            "--timer-property=AccuracySec=1s",
            "--property=Type=oneshot",
            "/opt/gpu-fault/venv/bin/python",
            arguments.probe_script,
            "cycle",
            "--run-id",
            run_id,
            "--hold-seconds",
            str(arguments.hold_seconds),
            "--failsafe-seconds",
            str(arguments.failsafe_seconds),
        ]
    )
    timer = run(
        [
            "systemctl",
            "show",
            unit + ".timer",
            "--property=ActiveState",
            "--property=NextElapseUSecRealtime",
        ]
    ).stdout
    if "ActiveState=active" not in timer:
        raise ProbeError("preemption cycle timer is not active")
    return {"run_id": run_id, "unit": unit, "timer_active": True}


def read_cycle(arguments: argparse.Namespace) -> dict[str, Any]:
    path = evidence_path(arguments.run_id)
    if not path.is_file():
        return {"run_id": arguments.run_id, "status": "PENDING"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeError("cycle evidence is invalid")
    return value


def cleanup(arguments: argparse.Namespace) -> dict[str, Any]:
    run_id = safe_id(arguments.run_id)
    incident_id = f"preempt012-{run_id}"
    manager = GpuServiceQuiesceManager(
        state_dir=str(QUIESCE_ROOT),
        restore_settle_seconds=10,
    )
    restore = manager.restore(incident_id=incident_id)
    unit = f"gpu-fault-preempt012-{run_id}"
    run(["systemctl", "stop", unit + ".timer"], check=False)
    run(
        ["systemctl", "reset-failed", unit + ".service", unit + ".timer"],
        check=False,
    )
    evidence_path(run_id).unlink(missing_ok=True)
    return {
        "restore": restore,
        "services": service_states(),
        "evidence_exists": evidence_path(run_id).exists(),
        "quiesce_state_files": sorted(
            path.name for path in QUIESCE_ROOT.glob("quiesce-*.json")
        ),
    }


def snapshot() -> dict[str, Any]:
    return {
        "services": service_states(),
        "quiesce_state_files": sorted(
            path.name for path in QUIESCE_ROOT.glob("quiesce-*.json")
        ),
        "gpu_count": len(
            [
                line
                for line in run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                    ]
                ).stdout.splitlines()
                if line.strip()
            ]
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot")
    arm_parser = commands.add_parser("arm")
    arm_parser.add_argument("--run-id", required=True)
    arm_parser.add_argument("--probe-script", required=True)
    arm_parser.add_argument("--delay-seconds", type=int, default=5)
    arm_parser.add_argument("--hold-seconds", type=int, default=45)
    arm_parser.add_argument("--failsafe-seconds", type=int, default=180)
    cycle_parser = commands.add_parser("cycle")
    cycle_parser.add_argument("--run-id", required=True)
    cycle_parser.add_argument("--hold-seconds", type=int, required=True)
    cycle_parser.add_argument("--failsafe-seconds", type=int, required=True)
    read_parser = commands.add_parser("read")
    read_parser.add_argument("--run-id", required=True)
    cleanup_parser = commands.add_parser("cleanup")
    cleanup_parser.add_argument("--run-id", required=True)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "snapshot":
            result = snapshot()
        elif arguments.command == "arm":
            result = arm(arguments)
        elif arguments.command == "cycle":
            cycle(arguments)
            return 0
        elif arguments.command == "read":
            result = read_cycle(arguments)
        else:
            result = cleanup(arguments)
    except (OSError, ProbeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
