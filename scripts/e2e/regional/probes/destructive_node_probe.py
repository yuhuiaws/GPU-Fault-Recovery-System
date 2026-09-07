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
import sys
import time
from typing import Any


LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
QUIESCE_STATE_DIR = Path("/var/lib/gpu-fault/quiesce")
SAMPLER_DIR = Path("/var/log/gpu-fault-acceptance")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_BDF = re.compile(r"^0000:[0-9a-f]{2}:[0-9a-f]{2}$")
NODE_ACTION_OPERATIONS = {
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
}
ALLOWED_XIDS = {46, 48, 62, 63, 79, 109}
SERVICE_UNITS = (
    "kubelet.service",
    "nvidia-fabricmanager.service",
    "nvidia-persistenced.service",
    "nvidia-dcgm.service",
    "dcgm.service",
)


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
        timeout=timeout,
        check=False,
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


def gpu_inventory() -> list[dict[str, str]]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ]
    )
    result = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4:
            continue
        match = re.search(
            r"(?:[0-9A-Fa-f]{8}:)?([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.[0-7]",
            values[2],
        )
        if match is None:
            continue
        result.append(
            {
                "index": values[0],
                "uuid": values[1],
                "pci_bdf": f"0000:{match.group(1).lower()}:{match.group(2).lower()}",
                "name": values[3],
            }
        )
    if not result:
        raise ProbeError("nvidia-smi returned no GPU inventory")
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
        if len(values) == 3 and values[1].isdigit():
            result.append(
                {
                    "gpu_uuid": values[0],
                    "pid": values[1],
                    "process_name": values[2],
                }
            )
    return result


def service_snapshot() -> dict[str, dict[str, str]]:
    result = {}
    for unit in SERVICE_UNITS:
        completed = run(
            [
                "systemctl",
                "show",
                unit,
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=InvocationID",
                "--property=ExecMainStartTimestampMonotonic",
            ],
            check=False,
        )
        if completed.returncode:
            continue
        values = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        if values.get("LoadState") != "not-found":
            result[unit] = values
    return result


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
            WHERE operation IN (
                'QUIESCE_GPU_SERVICES',
                'VERIFY_NO_GPU_CLIENTS',
                'RESET_GPU',
                'RESET_ALL_GPUS_NVSWITCHES',
                'RESTORE_GPU_SERVICES',
                'VALIDATE_GPU'
            )
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


def quiesce_states() -> list[dict[str, Any]]:
    if not QUIESCE_STATE_DIR.is_dir():
        return []
    result = []
    for path in sorted(QUIESCE_STATE_DIR.glob("quiesce-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {"phase": "UNREADABLE"}
        result.append(
            {
                "name": path.name,
                "phase": value.get("phase"),
                "active_services": value.get("active_services", []),
                "timer_unit": value.get("timer_unit"),
            }
        )
    return result


def counts_as_target_reset(message: str, target: str) -> bool:
    """A kernel line that reports a reset of the target device.

    The lines this probe itself writes to /dev/kmsg start with `gpu-fault ` and
    carry the case's marker; their text is whatever the case chose, and the
    XID 46 drill happens to say "reset". Counting them made every single-shot
    drill read as "one reset" and the two-line XID 63/48 drill as "none", so
    the count measured injection text, never the kernel. Injected lines stay
    in the journal excerpt as evidence; only kernel-origin lines count.
    """

    lowered = message.lower()
    if lowered.startswith("gpu-fault "):
        return False
    return bool(target) and target in lowered and "reset" in lowered


def kernel_reset_journal(
    since_epoch: float | None,
    pci_bdf: str | None,
) -> dict[str, Any]:
    if since_epoch is None:
        return {
            "entry_count": 0,
            "target_reset_count": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "messages": [],
        }
    completed = run(
        [
            "journalctl",
            "-k",
            "--since",
            f"@{since_epoch:.6f}",
            "--output=json",
            "--no-pager",
        ]
    )
    messages = []
    target = (pci_bdf or "").lower()
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = str(value.get("MESSAGE") or "")
        lowered = message.lower()
        if not any(
            token in lowered
            for token in (
                "gpu reset",
                "resetting gpu",
                "reset gpu",
                "xid (pci:",
            )
        ):
            continue
        messages.append(
            {
                "message": message[:500],
                "realtime_timestamp": value.get("__REALTIME_TIMESTAMP"),
            }
        )
    encoded = json.dumps(messages, sort_keys=True).encode()
    return {
        "entry_count": len(messages),
        "target_reset_count": sum(
            counts_as_target_reset(item["message"], target) for item in messages
        ),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "messages": messages[-30:],
    }


def sampler_paths(run_id: str) -> tuple[str, Path]:
    safe = safe_id(run_id, "run ID")
    digest = hashlib.sha256(safe.encode()).hexdigest()[:16]
    return (
        f"gpu-fault-reset-sampler-{digest}",
        SAMPLER_DIR / f"reset-sampler-{digest}.ndjson",
    )


def sampler_summary(run_id: str) -> dict[str, Any]:
    unit, path = sampler_paths(run_id)
    samples = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                samples.append(value)
    counts = [
        int(item["gpu_count"])
        for item in samples
        if isinstance(item.get("gpu_count"), int)
    ]
    encoded = path.read_bytes() if path.is_file() else b""
    active = (
        run(
            ["systemctl", "is-active", unit + ".service"],
            check=False,
        ).stdout.strip()
        == "active"
    )
    return {
        "unit": unit + ".service",
        "path": str(path),
        "active": active,
        "sample_count": len(samples),
        "min_gpu_count": min(counts) if counts else None,
        "max_gpu_count": max(counts) if counts else None,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "first": samples[0] if samples else None,
        "last": samples[-1] if samples else None,
    }


def snapshot(arguments: argparse.Namespace) -> None:
    inventory = gpu_inventory()
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "kmsg_exists": Path("/dev/kmsg").exists(),
            "kmsg_writable": os.access("/dev/kmsg", os.W_OK),
            "gpu_inventory": inventory,
            "compute_clients": compute_clients(),
            "services": service_snapshot(),
            "ledger": ledger_rows(),
            "gpu_fault_timers": gpu_fault_timers(),
            "quiesce_states": quiesce_states(),
            "kernel_reset_journal": kernel_reset_journal(
                arguments.since_epoch,
                arguments.pci_bdf,
            ),
            "sampler": (
                sampler_summary(arguments.run_id) if arguments.run_id else None
            ),
        }
    )


def write_xid(
    arguments: argparse.Namespace,
    *,
    xid: int,
    description: str,
) -> None:
    marker = safe_id(arguments.marker, "marker")
    drill_id = safe_id(arguments.drill_id, "drill ID")
    if SAFE_BDF.fullmatch(arguments.pci_bdf) is None:
        raise ProbeError("unsafe PCI BDF")
    message = (
        f"<3>gpu-fault {arguments.case_id} marker={marker} drill_id={drill_id} "
        f"NVRM: Xid (PCI:{arguments.pci_bdf}): {xid}, "
        f"{description}\n"
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
            "xid": xid,
            "bytes_written": written,
        }
    )


def write_xid46(arguments: argparse.Namespace) -> None:
    write_xid(
        arguments,
        xid=46,
        description="GPU stopped processing, reset acceptance event",
    )


def write_xid79(arguments: argparse.Namespace) -> None:
    write_xid(
        arguments,
        xid=79,
        description="GPU has fallen off the bus, reboot acceptance event",
    )


def write_generic_xid(arguments: argparse.Namespace) -> None:
    xid = int(arguments.xid)
    if xid not in ALLOWED_XIDS:
        raise ProbeError("XID is not allowlisted")
    write_xid(
        arguments,
        xid=xid,
        description=arguments.description,
    )


def gpu_sample() -> dict[str, Any]:
    """One `nvidia-smi -L` reading for the detached sampler.

    During the reset this sampler exists to observe, `nvidia-smi` blocks on the
    driver and the 10s bound trips. An uncaught ``TimeoutExpired`` here used to
    kill the sampler at exactly that moment, so the trace ended where the
    interesting part began. A timed-out reading is itself the evidence: record
    it and keep sampling.
    """

    observed_at = datetime.now(timezone.utc).isoformat()
    try:
        completed = run(["nvidia-smi", "-L"], check=False, timeout=10)
    except subprocess.TimeoutExpired:
        return {
            "observed_at": observed_at,
            "returncode": None,
            "gpu_count": None,
            "timed_out": True,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith("GPU ")
    ]
    return {
        "observed_at": observed_at,
        "returncode": completed.returncode,
        "gpu_count": len(lines),
        "timed_out": False,
        "sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }


def sample_gpus(arguments: argparse.Namespace) -> None:
    path = Path(arguments.output)
    if path.parent != SAMPLER_DIR or not path.name.startswith("reset-sampler-"):
        raise ProbeError("unsafe sampler output path")
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    deadline = time.monotonic() + arguments.duration_seconds
    while time.monotonic() < deadline:
        payload = gpu_sample()
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        time.sleep(arguments.interval_seconds)


def start_sampler(arguments: argparse.Namespace) -> None:
    unit, path = sampler_paths(arguments.run_id)
    if Path(arguments.probe_script).resolve() != Path(__file__).resolve():
        raise ProbeError("sampler probe script identity mismatch")
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    # A prior run that raised before stop-reset-sampler leaves its transient
    # unit loaded on the node. The unit name is a deterministic digest of the
    # run ID, so systemd-run then refuses with "already loaded or has a
    # fragment file". Clear any leftover of the same name first, exactly as
    # stop_sampler does, so a sampler start is idempotent across reruns.
    run(["systemctl", "stop", unit + ".service"], check=False)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            "--property=RuntimeMaxSec=1800",
            "--property=KillMode=control-group",
            # The interpreter this probe is already running under: the host
            # fixture chroots into the release's venv, and a hard-coded path
            # to another venv either does not exist or runs another release.
            sys.executable,
            str(Path(__file__).resolve()),
            "sample-gpus",
            "--output",
            str(path),
            "--duration-seconds",
            str(arguments.duration_seconds),
            "--interval-seconds",
            str(arguments.interval_seconds),
        ]
    )
    deadline = time.monotonic() + 30
    summary = sampler_summary(arguments.run_id)
    while summary["sample_count"] < 2 and time.monotonic() < deadline:
        time.sleep(1)
        summary = sampler_summary(arguments.run_id)
    if summary["sample_count"] < 2:
        # The unit may well be running and merely slow to produce its first
        # two lines; left alone it lives for RuntimeMaxSec=1800 and blocks the
        # next start of this run ID with "already loaded".
        run(["systemctl", "stop", unit + ".service"], check=False)
        run(["systemctl", "reset-failed", unit + ".service"], check=False)
        raise ProbeError(
            "detached GPU sampler did not start "
            f"(samples={summary['sample_count']}, active={summary['active']}); "
            "unit stopped"
        )
    emit(summary)


def stop_sampler(arguments: argparse.Namespace) -> None:
    unit, path = sampler_paths(arguments.run_id)
    run(["systemctl", "stop", unit + ".service"], check=False)
    summary = sampler_summary(arguments.run_id)
    path.unlink(missing_ok=True)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    emit(summary)


def restore_quiesce(arguments: argparse.Namespace) -> None:
    incident_id = safe_id(arguments.incident_id, "incident ID")
    digest = hashlib.sha256(incident_id.encode()).hexdigest()[:20]
    state_path = QUIESCE_STATE_DIR / f"quiesce-{digest}.json"
    from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager

    result = GpuServiceQuiesceManager.restore_state_file(state_path)
    emit(result)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    snapshot_command = commands.add_parser("snapshot")
    snapshot_command.add_argument("--since-epoch", type=float)
    snapshot_command.add_argument("--pci-bdf")
    snapshot_command.add_argument("--run-id")
    snapshot_command.set_defaults(handler=snapshot)

    for name, handler, case_id in (
        ("write-xid46", write_xid46, "GF-REGIONAL-DESTR-001"),
        ("write-xid79", write_xid79, "GF-REGIONAL-DESTR-002"),
    ):
        command = commands.add_parser(name)
        command.add_argument("--marker", required=True)
        command.add_argument("--drill-id", required=True)
        command.add_argument("--pci-bdf", required=True)
        command.set_defaults(handler=handler, case_id=case_id)

    generic = commands.add_parser("write-xid")
    generic.add_argument("--xid", type=int, choices=sorted(ALLOWED_XIDS), required=True)
    generic.add_argument("--marker", required=True)
    generic.add_argument("--drill-id", required=True)
    generic.add_argument("--pci-bdf", required=True)
    generic.add_argument(
        "--description",
        default="regional destructive collector acceptance event",
    )
    generic.add_argument("--case-id", default="GF-REGIONAL-COLLECT")
    generic.set_defaults(handler=write_generic_xid)

    start = commands.add_parser("start-reset-sampler")
    start.add_argument("--run-id", required=True)
    start.add_argument("--probe-script", required=True)
    start.add_argument("--duration-seconds", type=int, default=1800)
    start.add_argument("--interval-seconds", type=float, default=0.25)
    start.set_defaults(handler=start_sampler)

    stop = commands.add_parser("stop-reset-sampler")
    stop.add_argument("--run-id", required=True)
    stop.set_defaults(handler=stop_sampler)

    sample = commands.add_parser("sample-gpus")
    sample.add_argument("--output", required=True)
    sample.add_argument("--duration-seconds", type=int, required=True)
    sample.add_argument("--interval-seconds", type=float, required=True)
    sample.set_defaults(handler=sample_gpus)

    restore = commands.add_parser("restore-quiesce")
    restore.add_argument("--incident-id", required=True)
    restore.set_defaults(handler=restore_quiesce)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        if (
            getattr(arguments, "duration_seconds", 1) < 1
            or getattr(arguments, "duration_seconds", 1) > 1800
        ):
            raise ProbeError("sampler duration is outside 1..1800 seconds")
        if (
            getattr(arguments, "interval_seconds", 1.0) < 0.1
            or getattr(arguments, "interval_seconds", 1.0) > 5.0
        ):
            raise ProbeError("sampler interval is outside 0.1..5 seconds")
        arguments.handler(arguments)
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
