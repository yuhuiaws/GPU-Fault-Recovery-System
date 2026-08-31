from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any


SERVICES = (
    "gpu-fault-kernel-collector.service",
    "gpu-fault-metrics-collector.service",
    "gpu-fault-host-collector.service",
    "gpu-fault-fabric-manager-collector.service",
)
OUTBOXES = {
    name: Path(f"/var/lib/gpu-fault/outbox/{name}.ndjson")
    for name in ("kernel", "dcgm", "host", "fabric-manager")
}
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ToolError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode:
        raise ToolError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def validate_tag(value: str) -> str:
    if SAFE_TAG.fullmatch(value) is None:
        raise ToolError("unsafe firewall tag")
    return value


def validate_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ToolError(f"unsafe {label}")
    return value


def firewall() -> str:
    path = shutil.which("iptables")
    if path is None:
        raise ToolError("iptables is not installed on the host")
    return path


def service_snapshot() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for service in SERVICES:
        completed = run(
            [
                "systemctl",
                "show",
                service,
                "--property=ActiveState",
                "--property=SubState",
                "--property=NRestarts",
                "--property=MainPID",
            ]
        )
        values: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        result[service] = values
    return result


def read_outbox(path: Path, test_ids: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "parent_exists": path.parent.is_dir(),
            "parent_writable": os.access(path.parent, os.W_OK),
            "line_count": 0,
            "malformed_count": 0,
            "replayable_count": 0,
            "path_counts": {},
            "error_counts": {},
            "matching": [],
        }
    matching = []
    malformed = 0
    replayable = 0
    path_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(record, dict):
            malformed += 1
            continue
        if record.get("replayable") is True:
            replayable += 1
        request_path = str(record.get("path") or "")
        path_counts[request_path] = path_counts.get(request_path, 0) + 1
        error = str(record.get("error") or "")
        error_counts[error] = error_counts.get(error, 0) + 1
        payload = record.get("payload")
        encoded = json.dumps(payload, sort_keys=True, default=str)
        matched = [test_id for test_id in test_ids if test_id in encoded]
        if matched:
            matching.append(
                {
                    "test_ids": matched,
                    "path": record.get("path"),
                    "record_id": (
                        payload.get("record_id") if isinstance(payload, dict) else None
                    ),
                    "replayable": record.get("replayable"),
                    "failed_at": record.get("failed_at"),
                }
            )
    return {
        "exists": True,
        "parent_exists": path.parent.is_dir(),
        "parent_writable": os.access(path.parent, os.W_OK),
        "line_count": len(lines),
        "malformed_count": malformed,
        "replayable_count": replayable,
        "path_counts": dict(sorted(path_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "matching": matching,
    }


def outbox_snapshot(test_ids: tuple[str, ...]) -> dict[str, Any]:
    return {name: read_outbox(path, test_ids) for name, path in OUTBOXES.items()}


def resolve_ipv4(hostname: str) -> list[str]:
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(
            hostname,
            443,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
    }
    if not addresses:
        raise ToolError(f"no IPv4 addresses resolved for {hostname}")
    return sorted(str(item) for item in addresses)


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
        raise ToolError(f"cannot parse GPU PCI BDF: {first!r}")
    return f"0000:{match.group(1).lower()}:{match.group(2).lower()}"


def exact_rule(iptables: str, ip: str, tag: str, operation: str) -> list[str]:
    return [
        iptables,
        operation,
        "OUTPUT",
        "-p",
        "tcp",
        "-d",
        ip,
        "--dport",
        "443",
        "-m",
        "comment",
        "--comment",
        tag,
        "-j",
        "REJECT",
        "--reject-with",
        "tcp-reset",
    ]


def rule_present(iptables: str, ip: str, tag: str) -> bool:
    return run(exact_rule(iptables, ip, tag, "-C"), check=False).returncode == 0


def tagged_rules(tag: str) -> list[str]:
    completed = run([firewall(), "-S", "OUTPUT"])
    return [line for line in completed.stdout.splitlines() if tag in line]


def timer_snapshot(tag: str) -> dict[str, str]:
    unit = f"{tag}-rollback.timer"
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=NextElapseUSecRealtime",
        ],
        check=False,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    result["returncode"] = str(completed.returncode)
    return result


def connectivity(ips: list[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for ip in ips:
        try:
            with socket.create_connection((ip, 443), timeout=3):
                result[ip] = True
        except OSError:
            result[ip] = False
    return result


def preflight(args: argparse.Namespace) -> None:
    validate_tag(args.tag_prefix)
    if shutil.which("systemd-run") is None:
        raise ToolError("systemd-run is not installed on the host")
    if shutil.which("nvidia-smi") is None:
        raise ToolError("nvidia-smi is not installed on the host")
    ips = resolve_ipv4(args.endpoint_host)
    emit(
        {
            "services": service_snapshot(),
            "outboxes": outbox_snapshot(tuple(args.test_id)),
            "endpoint_host": args.endpoint_host,
            "endpoint_ipv4": ips,
            "connectivity": connectivity(ips),
            "gpu_pci_bdf": first_gpu_bdf(),
            "kmsg_exists": Path("/dev/kmsg").exists(),
            "kmsg_writable": os.access("/dev/kmsg", os.W_OK),
            "iptables": firewall(),
            "systemd_run": shutil.which("systemd-run"),
            "existing_tagged_rules": tagged_rules(args.tag_prefix),
        }
    )


def cleanup_script(tag: str, ips: list[str]) -> Path:
    iptables = firewall()
    path = Path(f"/run/{tag}-cleanup.sh")
    lines = ["#!/bin/bash", "set +e"]
    for ip in ips:
        check = shlex.join(exact_rule(iptables, ip, tag, "-C"))
        delete = shlex.join(exact_rule(iptables, ip, tag, "-D"))
        lines.extend(
            [
                f"while {check} >/dev/null 2>&1; do",
                f"  {delete} >/dev/null 2>&1 || break",
                "done",
            ]
        )
    lines.append(f"rm -f {shlex.quote(str(path))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def arm(args: argparse.Namespace) -> None:
    tag = validate_tag(args.tag)
    ips = [str(ipaddress.ip_address(value)) for value in args.ip]
    if not ips:
        raise ToolError("at least one firewall destination is required")
    if tagged_rules(tag):
        raise ToolError("tagged firewall rules already exist")
    unit = f"{tag}-rollback"
    run(["systemctl", "stop", f"{unit}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    script = cleanup_script(tag, ips)
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            f"--on-active={args.ttl_seconds}s",
            "--timer-property=AccuracySec=1s",
            "/bin/bash",
            str(script),
        ]
    )
    timer = timer_snapshot(tag)
    if timer.get("ActiveState") != "active":
        raise ToolError(f"automatic rollback timer is not active: {timer}")
    emit({"tag": tag, "ips": ips, "timer": timer, "cleanup_script": str(script)})


def block(args: argparse.Namespace) -> None:
    tag = validate_tag(args.tag)
    iptables = firewall()
    ips = [str(ipaddress.ip_address(value)) for value in args.ip]
    for ip in ips:
        if not rule_present(iptables, ip, tag):
            run(exact_rule(iptables, ip, tag, "-I"))
    emit(
        {
            "tag": tag,
            "rules": tagged_rules(tag),
            "connectivity": connectivity(ips),
        }
    )


def cleanup(args: argparse.Namespace) -> None:
    tag = validate_tag(args.tag)
    iptables = firewall()
    ips = [str(ipaddress.ip_address(value)) for value in args.ip]
    for ip in ips:
        while rule_present(iptables, ip, tag):
            run(exact_rule(iptables, ip, tag, "-D"))
    unit = f"{tag}-rollback"
    run(["systemctl", "stop", f"{unit}.timer"], check=False)
    run(
        ["systemctl", "reset-failed", f"{unit}.service", f"{unit}.timer"],
        check=False,
    )
    Path(f"/run/{tag}-cleanup.sh").unlink(missing_ok=True)
    emit(
        {
            "tag": tag,
            "rules": tagged_rules(tag),
            "timer": timer_snapshot(tag),
            "connectivity": connectivity(ips),
        }
    )


def write_kmsg(args: argparse.Namespace) -> None:
    test_id = validate_id(args.test_id, "test ID")
    drill_id = validate_id(args.drill_id, "drill ID")
    if re.fullmatch(r"0000:[0-9a-f]{2}:[0-9a-f]{2}", args.pci_bdf) is None:
        raise ToolError("unsafe PCI BDF")
    message = (
        f"<6>gpu-fault NET-001 test_id={test_id} "
        f"drill_id={drill_id} "
        f"NVRM: Xid (PCI:{args.pci_bdf}): 63, "
        "monitor-only row remapping acceptance event\n"
    ).encode()
    descriptor = os.open("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)
    try:
        written = os.write(descriptor, message)
    finally:
        os.close(descriptor)
    emit({"test_id": test_id, "drill_id": drill_id, "bytes_written": written})


def snapshot(args: argparse.Namespace) -> None:
    tag = validate_tag(args.tag) if args.tag else None
    emit(
        {
            "services": service_snapshot(),
            "outboxes": outbox_snapshot(tuple(args.test_id)),
            "rules": tagged_rules(tag) if tag else [],
            "timer": timer_snapshot(tag) if tag else {},
        }
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)

    preflight_command = commands.add_parser("preflight")
    preflight_command.add_argument("--endpoint-host", required=True)
    preflight_command.add_argument("--tag-prefix", required=True)
    preflight_command.add_argument("--test-id", action="append", default=[])
    preflight_command.set_defaults(handler=preflight)

    arm_command = commands.add_parser("arm")
    arm_command.add_argument("--tag", required=True)
    arm_command.add_argument("--ttl-seconds", type=int, required=True)
    arm_command.add_argument("--ip", action="append", required=True)
    arm_command.set_defaults(handler=arm)

    block_command = commands.add_parser("block")
    block_command.add_argument("--tag", required=True)
    block_command.add_argument("--ip", action="append", required=True)
    block_command.set_defaults(handler=block)

    cleanup_command = commands.add_parser("cleanup")
    cleanup_command.add_argument("--tag", required=True)
    cleanup_command.add_argument("--ip", action="append", required=True)
    cleanup_command.set_defaults(handler=cleanup)

    write_command = commands.add_parser("write-kmsg")
    write_command.add_argument("--test-id", required=True)
    write_command.add_argument("--drill-id", required=True)
    write_command.add_argument("--pci-bdf", required=True)
    write_command.set_defaults(handler=write_kmsg)

    snapshot_command = commands.add_parser("snapshot")
    snapshot_command.add_argument("--tag")
    snapshot_command.add_argument("--test-id", action="append", default=[])
    snapshot_command.set_defaults(handler=snapshot)

    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ToolError, subprocess.SubprocessError) as exc:
        emit({"error": str(exc), "command": args.command})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
