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


def stop_unit(run_id: str, service: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{service}".encode()).hexdigest()[:16]
    return f"gpu-fault-warm-spare-stop-{digest}"


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
    delay = arguments.stop_delay_seconds
    before = service_snapshot(service)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{service} is not active at baseline")
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--on-active={arguments.restore_seconds + delay}s",
            "/bin/systemctl",
            "start",
            service,
        ]
    )
    result: dict[str, Any] = {
        "service": service,
        "restore_unit": unit + ".timer",
        "restore_seconds": arguments.restore_seconds,
        "stop_delay_seconds": delay,
        "before": before,
    }
    if delay:
        # Stopping kubelet also kills the transport this probe is answering
        # over: `kubectl exec` reaches the container through kubelet, so a
        # synchronous `systemctl stop kubelet` cannot report back. The caller
        # then only regains the channel once the failsafe timer restarts
        # kubelet, by which time the NotReady window it wanted to observe is
        # already closed. Hand the stop to systemd instead, answer while the
        # channel is still up, and let the caller watch the node conditions
        # for the transition -- that is the assertion it actually needs.
        scheduled = stop_unit(run_id, service)
        run(
            [
                "systemd-run",
                "--unit",
                scheduled,
                f"--on-active={delay}s",
                "/bin/systemctl",
                "stop",
                service,
            ]
        )
        result["stop_unit"] = scheduled + ".timer"
        result["scheduled"] = True
        emit(result)
        return
    run(["systemctl", "stop", service], timeout=120)
    after = service_snapshot(service)
    if after.get("ActiveState") == "active":
        raise ProbeError(f"{service} did not stop")
    result["scheduled"] = False
    result["after"] = after
    emit(result)


def restore_service(arguments: argparse.Namespace) -> None:
    service = service_name(arguments.service)
    run_id = safe_id(arguments.run_id, "run ID")
    unit = restore_unit(run_id, service)
    scheduled = stop_unit(run_id, service)
    before = service_snapshot(service)
    # Disarm the delayed stop before starting the service: a stop timer that
    # has not fired yet would otherwise take the service back down after the
    # restore reported success.
    run(["systemctl", "stop", scheduled + ".timer"], check=False)
    run(["systemctl", "reset-failed", scheduled + ".service"], check=False)
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
            "stop_unit": scheduled + ".timer",
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
    stop.add_argument("--stop-delay-seconds", type=int, default=0)
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
        stop_delay_seconds = getattr(arguments, "stop_delay_seconds", 0)
        if not 0 <= stop_delay_seconds <= 120:
            raise ProbeError("stop delay seconds is outside 0..120")
        arguments.handler(arguments)
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
