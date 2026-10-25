"""Verify the inherited service cgroup envelope instead of claiming RLIMIT_RSS."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def service_limits(
    proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup")
) -> dict[str, Any]:
    try:
        lines = (proc_root / "self/cgroup").read_text().splitlines()
        relative = next(line[3:] for line in lines if line.startswith("0::"))
        if relative == "/" or ".." in relative.split("/"):
            raise ValueError("no delegated service cgroup")
        directory = cgroup_root / relative.lstrip("/")
        quota, period = (directory / "cpu.max").read_text().split()
        memory = int((directory / "memory.max").read_text())
        cores = int(quota) / int(period)
        return {
            "verified": 0 < cores <= 2 and 0 < memory <= 1024 * 1024 * 1024,
            "cpu_cores": cores,
            "memory_bytes": memory,
            "scope": "AGENT_AND_ALL_DESCENDANTS",
        }
    except (OSError, ValueError, StopIteration, ZeroDivisionError):
        return {"verified": False, "reason": "BOUNDED_CGROUP_V2_ENVELOPE_UNAVAILABLE"}


#: What one observation request may weigh on the read-only worker's stdin. The
#: supervisor refuses a larger request before it starts the worker, and the
#: worker refuses it again on its own side of the pipe.
OBSERVATION_REQUEST_BUDGET_BYTES = 262144
