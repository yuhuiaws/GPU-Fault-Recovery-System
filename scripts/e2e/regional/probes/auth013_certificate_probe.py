#!/usr/bin/env python3
"""AUTH-013 host probe: is the certificate-expiry alert armed on this node?

Runs under ``chroot /host`` through ``HostProbeFixture``. The only expiry
alerting the deploy ships is the per-node ``gpu-fault-certificate-check``
systemd timer, which runs ``check-control-plane-certificate`` with the
``GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS`` value from
``/etc/gpu-fault/collector.env``. That file also carries the cluster token, so
this probe reads exactly one key from it and prints an integer and unit
states -- never the file, never another key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

ENV_FILE = Path("/etc/gpu-fault/collector.env")
THRESHOLD_KEY = "GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS"
SAFE_UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")


class ProbeError(RuntimeError):
    pass


def threshold_seconds(text: str) -> int | None:
    """The configured minimum validity, or None when the key is absent."""

    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == THRESHOLD_KEY:
            digits = value.strip().strip("'\"")
            return int(digits) if digits.isdigit() else None
    return None


def systemctl_state(unit: str, *query: str) -> str:
    completed = subprocess.run(
        ["systemctl", *query, unit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    return completed.stdout.strip() or completed.stderr.strip()[:80]


def scan(unit: str) -> dict[str, Any]:
    if SAFE_UNIT.fullmatch(unit) is None:
        raise ProbeError("unsafe unit name")
    seconds = None
    env_exists = ENV_FILE.is_file()
    if env_exists:
        seconds = threshold_seconds(ENV_FILE.read_text(encoding="utf-8"))
    enabled = systemctl_state(unit, "is-enabled")
    active = systemctl_state(unit, "is-active")
    service = unit.removesuffix(".timer") + ".service"
    return {
        "unit": unit,
        "env_file_exists": env_exists,
        "min_validity_seconds": seconds,
        "timer_enabled_state": enabled,
        "timer_enabled": enabled == "enabled",
        "timer_active_state": active,
        "timer_active": active == "active",
        "last_service_result": systemctl_state(
            service, "show", "-p", "Result", "--value"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timer", default="gpu-fault-certificate-check.timer")
    arguments = parser.parse_args()
    try:
        result = scan(arguments.timer)
    except (OSError, ProbeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
