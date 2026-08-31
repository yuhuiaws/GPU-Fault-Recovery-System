from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


SAFE_MARKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_BDF = re.compile(r"^(?:0000:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")


class ProbeError(RuntimeError):
    pass


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return completed


def gpu_bdf() -> str:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ]
    ).stdout
    first = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if SAFE_BDF.fullmatch(first) is None:
        raise ProbeError(f"cannot parse GPU PCI BDF: {first!r}")
    return first.lower()


def snapshot() -> dict[str, Any]:
    service = run(
        [
            "systemctl",
            "show",
            "gpu-fault-kernel-collector.service",
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
        ]
    ).stdout
    values = {}
    for line in service.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "gpu_bdf": gpu_bdf(),
        "kernel_collector": values,
        "kmsg_writable": os.access("/dev/kmsg", os.W_OK),
    }


def write_xid11(marker: str, pci_bdf: str) -> dict[str, Any]:
    if SAFE_MARKER.fullmatch(marker) is None:
        raise ProbeError("unsafe marker")
    if SAFE_BDF.fullmatch(pci_bdf) is None:
        raise ProbeError("unsafe PCI BDF")
    short = pci_bdf.lower().removesuffix(".0")
    message = (
        f"<3>NVRM: Xid (PCI:{short}): 11, Ch 00000001, "
        f"regional E2E workload restart marker={marker}\n"
    ).encode()
    descriptor = os.open("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)
    try:
        written = os.write(descriptor, message)
    finally:
        os.close(descriptor)
    return {"marker": marker, "pci_bdf": pci_bdf, "bytes_written": written}


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot")
    write = commands.add_parser("write-xid11")
    write.add_argument("--marker", required=True)
    write.add_argument("--pci-bdf", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "snapshot":
            result = snapshot()
        else:
            result = write_xid11(arguments.marker, arguments.pci_bdf)
    except (OSError, ProbeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
