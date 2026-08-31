#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import shlex
import subprocess
from typing import Any


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProbeError(RuntimeError):
    pass


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
        timeout=120,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def safe_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def normalized_cidrs(values: list[str]) -> list[str]:
    try:
        networks = {ipaddress.ip_network(value, strict=False) for value in values}
    except ValueError as exc:
        raise ProbeError("invalid control-plane CIDR") from exc
    if any(network.prefixlen == 0 for network in networks):
        raise ProbeError("default-route CIDR is prohibited")
    return sorted(str(network) for network in networks)


def chain_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12].upper()
    return f"GFISO{digest}"


def restore_unit(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f"gpu-fault-network-restore-{digest}"


def remove_chain(chain: str) -> None:
    run(["iptables", "-D", "OUTPUT", "-j", chain], check=False)
    run(["iptables", "-F", chain], check=False)
    run(["iptables", "-X", chain], check=False)


def block(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    cidrs = normalized_cidrs(arguments.control_plane_cidr)
    chain = chain_name(run_id)
    remove_chain(chain)
    unit = restore_unit(run_id)
    script = (
        f"iptables -D OUTPUT -j {chain} || true; "
        f"iptables -F {chain} || true; iptables -X {chain} || true"
    )
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--on-active={arguments.restore_seconds}s",
            "/bin/bash",
            "-ceu",
            script,
        ]
    )
    try:
        run(["iptables", "-N", chain])
        for cidr in cidrs:
            run(
                [
                    "iptables",
                    "-A",
                    chain,
                    "-p",
                    "tcp",
                    "-d",
                    cidr,
                    "--dport",
                    "443",
                    "-m",
                    "comment",
                    "--comment",
                    run_id,
                    "-j",
                    "REJECT",
                ]
            )
        run(["iptables", "-I", "OUTPUT", "1", "-j", chain])
    except Exception:
        remove_chain(chain)
        run(["systemctl", "stop", unit + ".timer"], check=False)
        run(["systemctl", "reset-failed", unit + ".service"], check=False)
        raise
    emit(
        {
            "run_id": run_id,
            "chain": chain,
            "cidrs": cidrs,
            "restore_unit": unit + ".timer",
            "restore_seconds": arguments.restore_seconds,
        }
    )


def unblock(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    chain = chain_name(run_id)
    remove_chain(chain)
    unit = restore_unit(run_id)
    run(["systemctl", "stop", unit + ".timer"], check=False)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    emit({"run_id": run_id, "chain": chain, "blocked": False})


def status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    chain = chain_name(run_id)
    completed = run(["iptables", "-S", chain], check=False)
    emit(
        {
            "run_id": run_id,
            "chain": chain,
            "blocked": completed.returncode == 0,
            "rules": completed.stdout.splitlines(),
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    add = commands.add_parser("block")
    add.add_argument("--run-id", required=True)
    add.add_argument("--control-plane-cidr", action="append", required=True)
    add.add_argument("--restore-seconds", type=int, default=2100)
    add.set_defaults(handler=block)

    remove = commands.add_parser("unblock")
    remove.add_argument("--run-id", required=True)
    remove.set_defaults(handler=unblock)

    show = commands.add_parser("status")
    show.add_argument("--run-id", required=True)
    show.set_defaults(handler=status)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        seconds = getattr(arguments, "restore_seconds", 2100)
        if not 60 <= seconds <= 3600:
            raise ProbeError("restore seconds is outside 60..3600")
        arguments.handler(arguments)
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
