#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import shlex
import subprocess
import time
from typing import Any


ALLOWED_SERVICES = {
    "gpu-fault-node-agent.service",
    "kubelet.service",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProbeError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def service_name(value: str) -> str:
    if value not in ALLOWED_SERVICES:
        raise ProbeError("service is not in the warm-spare probe allowlist")
    return value


def service_snapshot(service: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            service,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=InvocationID",
            "--property=ExecMainStartTimestampMonotonic",
        ]
    )
    result = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def restore_unit(run_id: str, service: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{service}".encode()).hexdigest()[:16]
    return f"gpu-fault-warm-spare-restore-{digest}"


def snapshot(arguments: argparse.Namespace) -> None:
    service = service_name(arguments.service)
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "state": service_snapshot(service),
        }
    )


def stop_with_failsafe(arguments: argparse.Namespace) -> None:
    service = service_name(arguments.service)
    run_id = safe_id(arguments.run_id, "run ID")
    unit = restore_unit(run_id, service)
    before = service_snapshot(service)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{service} is not active at baseline")
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--on-active={arguments.restore_seconds}s",
            "/bin/systemctl",
            "start",
            service,
        ]
    )
    run(["systemctl", "stop", service], timeout=120)
    after = service_snapshot(service)
    if after.get("ActiveState") == "active":
        raise ProbeError(f"{service} did not stop")
    emit(
        {
            "service": service,
            "restore_unit": unit + ".timer",
            "restore_seconds": arguments.restore_seconds,
            "before": before,
            "after": after,
        }
    )


def restore_service(arguments: argparse.Namespace) -> None:
    service = service_name(arguments.service)
    run_id = safe_id(arguments.run_id, "run ID")
    unit = restore_unit(run_id, service)
    before = service_snapshot(service)
    run(["systemctl", "start", service], timeout=120)
    run(["systemctl", "stop", unit + ".timer"], check=False)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    deadline = time.monotonic() + 120
    after = service_snapshot(service)
    while after.get("ActiveState") != "active" and time.monotonic() < deadline:
        time.sleep(2)
        after = service_snapshot(service)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{service} is not active after restore")
    emit(
        {
            "service": service,
            "restore_unit": unit + ".timer",
            "before": before,
            "after": after,
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    show = commands.add_parser("snapshot")
    show.add_argument("--service", choices=sorted(ALLOWED_SERVICES), required=True)
    show.set_defaults(handler=snapshot)

    stop = commands.add_parser("stop-with-failsafe")
    stop.add_argument("--service", choices=sorted(ALLOWED_SERVICES), required=True)
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--restore-seconds", type=int, default=180)
    stop.set_defaults(handler=stop_with_failsafe)

    restore = commands.add_parser("restore-service")
    restore.add_argument("--service", choices=sorted(ALLOWED_SERVICES), required=True)
    restore.add_argument("--run-id", required=True)
    restore.set_defaults(handler=restore_service)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        restore_seconds = getattr(arguments, "restore_seconds", 180)
        if not 60 <= restore_seconds <= 600:
            raise ProbeError("restore seconds is outside 60..600")
        arguments.handler(arguments)
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
