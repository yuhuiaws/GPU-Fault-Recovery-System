#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
from typing import Any


FABRIC_MANAGER_UNIT = "nvidia-fabricmanager.service"
LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_BDF = re.compile(r"^0000:[0-9a-f]{2}:[0-9a-f]{2}$")


class ProbeError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    # Bounded: `nvidia-smi` blocks on a wedged driver and `systemctl restart`
    # on a unit whose stop hangs, and the host fixture's own exec timeout then
    # kills kubectl while this process lingers on the node.
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


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def service_snapshot() -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            FABRIC_MANAGER_UNIT,
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=NRestarts",
            "--property=InvocationID",
            "--property=ExecMainStartTimestamp",
            "--property=ExecMainStartTimestampMonotonic",
        ]
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def compute_clients() -> list[dict[str, str]]:
    completed = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ]
    )
    result = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) != 3 or not values[1].isdigit():
            continue
        result.append(
            {
                "gpu_uuid": values[0],
                "pid": values[1],
                "process_name": values[2],
            }
        )
    return result


def first_gpu_bdf() -> str:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    first = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    match = re.search(
        r"(?:[0-9A-Fa-f]{8}:)?([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.[0-7]",
        first,
    )
    if match is None:
        raise ProbeError(f"cannot parse GPU PCI BDF: {first!r}")
    return f"0000:{match.group(1).lower()}:{match.group(2).lower()}"


def ledger_rows() -> list[dict[str, Any]]:
    if not LEDGER.is_file():
        return []
    connection = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT command_id, completed_at, attempt, state,
                   operation, started_at
            FROM results
            WHERE operation='RESTART_FABRIC_MANAGER'
            ORDER BY completed_at, command_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "command_id": row[0],
            "completed_at": row[1],
            "attempt": row[2],
            "state": row[3],
            "operation": row[4],
            "started_at": row[5],
        }
        for row in rows
    ]


def gpu_fault_timers() -> list[str]:
    completed = run(
        [
            "systemctl",
            "list-timers",
            "--all",
            "--no-legend",
            "--no-pager",
        ]
    )
    units = {
        match.group(0)
        for line in completed.stdout.splitlines()
        for match in re.finditer(r"gpu-fault-[A-Za-z0-9_.@-]+\.timer", line)
    }
    return sorted(units)


def journal_summary(since_epoch: float | None) -> dict[str, Any]:
    if since_epoch is None:
        return {
            "entry_count": 0,
            "started_count": 0,
            "stopped_count": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "messages": [],
        }
    completed = run(
        [
            "journalctl",
            "-u",
            FABRIC_MANAGER_UNIT,
            "--since",
            f"@{since_epoch:.6f}",
            "--output=json",
            "--no-pager",
        ]
    )
    messages = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = str(value.get("MESSAGE") or "")
        if not message:
            continue
        messages.append(
            {
                "message": message[:300],
                "realtime_timestamp": value.get("__REALTIME_TIMESTAMP"),
                "invocation_id": value.get("_SYSTEMD_INVOCATION_ID"),
            }
        )
    encoded = json.dumps(messages, sort_keys=True).encode()
    lowered = [item["message"].lower() for item in messages]
    return {
        "entry_count": len(messages),
        "started_count": sum(
            item.startswith("started nvidia-fabricmanager.service") for item in lowered
        ),
        "stopped_count": sum(
            item.startswith("stopped nvidia-fabricmanager.service") for item in lowered
        ),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "messages": messages[-20:],
    }


def snapshot(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "kmsg_exists": Path("/dev/kmsg").exists(),
            "kmsg_writable": os.access("/dev/kmsg", os.W_OK),
            "gpu_pci_bdf": first_gpu_bdf(),
            "compute_clients": compute_clients(),
            "fabric_manager": service_snapshot(),
            "ledger": ledger_rows(),
            "gpu_fault_timers": gpu_fault_timers(),
            "journal": journal_summary(arguments.since_epoch),
        }
    )


def write_xid45(arguments: argparse.Namespace) -> None:
    marker = safe_id(arguments.marker, "marker")
    drill_id = safe_id(arguments.drill_id, "drill ID")
    if SAFE_BDF.fullmatch(arguments.pci_bdf) is None:
        raise ProbeError("unsafe PCI BDF")
    message = (
        f"<3>gpu-fault DESTR-010 marker={marker} drill_id={drill_id} "
        f"NVRM: Xid (PCI:{arguments.pci_bdf}): 45, "
        "Preemptive Channel Cleanup, solo acceptance event\n"
    ).encode()
    descriptor = os.open("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)
    try:
        written = os.write(descriptor, message)
    finally:
        os.close(descriptor)
    emit(
        {
            "marker": marker,
            "drill_id": drill_id,
            "pci_bdf": arguments.pci_bdf,
            "bytes_written": written,
        }
    )


def ensure_fabric_manager_active(_arguments: argparse.Namespace) -> None:
    before = service_snapshot()
    restarted = before.get("ActiveState") != "active"
    if restarted:
        run(["systemctl", "restart", FABRIC_MANAGER_UNIT])
    after = service_snapshot()
    if after.get("ActiveState") != "active":
        raise ProbeError("Fabric Manager is not active after recovery")
    emit({"restarted": restarted, "before": before, "after": after})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    snapshot_command = commands.add_parser("snapshot")
    snapshot_command.add_argument("--since-epoch", type=float)
    snapshot_command.set_defaults(handler=snapshot)

    xid45 = commands.add_parser("write-xid45")
    xid45.add_argument("--marker", required=True)
    xid45.add_argument("--drill-id", required=True)
    xid45.add_argument("--pci-bdf", required=True)
    xid45.set_defaults(handler=write_xid45)

    recover = commands.add_parser("ensure-fabric-manager-active")
    recover.set_defaults(handler=ensure_fabric_manager_active)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.handler(arguments)
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
