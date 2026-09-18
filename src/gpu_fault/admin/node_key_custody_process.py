"""Conservative process-start and stable projected-key evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from gpu_fault.admin.node_key_custody_crypto import private_command_environment
from gpu_fault.admin.node_key_custody_models import CustodyError
from gpu_fault.admin.node_key_custody_pods import cpu_role_container

NANOSECONDS = 1_000_000_000
_STARTED = re.compile(
    r"(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z|[+-]\d{2}:\d{2})"
)

PROJECTED_KEY_PROBE = r"""
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
def process_ticks():
    fields = Path("/proc/1/stat").read_text().rpartition(")")[2].split()
    return int(fields[19])
def file_identity(value):
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
expected = json.load(sys.stdin)
path = Path(os.environ["GPU_FAULT_NODE_ACTION_KEYS_DIR"]) / expected["node_id"]
ticks = process_ticks()
hz = os.sysconf("SC_CLK_TCK")
realtime_before = time.time_ns()
boottime = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
realtime_after = time.time_ns()
with path.open("rb") as handle:
    before = os.fstat(handle.fileno())
    material = handle.read(4097)
    after = os.fstat(handle.fileno())
current = path.stat()
if (
    not stat.S_ISREG(before.st_mode) or not 32 <= len(material) <= 4096
    or file_identity(before) != file_identity(after)
    or file_identity(before) != file_identity(current)
    or process_ticks() != ticks
):
    raise RuntimeError("projected key or process identity is unstable")
print(json.dumps({
    "sha256": hashlib.sha256(material).hexdigest(),
    "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
    "process_start_ticks": ticks, "ticks_per_second": hz,
    "realtime_before_ns": realtime_before, "boottime_ns": boottime,
    "realtime_after_ns": realtime_after,
}))
"""


def started_interval(value: str) -> tuple[int, int]:
    match = _STARTED.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise CustodyError("credential consumer start timestamp is invalid")
    try:
        stamp = datetime.fromisoformat(
            match["seconds"] + match["zone"].replace("Z", "+00:00")
        )
        elapsed = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
        fraction = match["fraction"] or ""
        lower = (elapsed.days * 86400 + elapsed.seconds) * NANOSECONDS + int(
            fraction.ljust(9, "0")
        )
        return lower, lower + 10 ** (9 - len(fraction))
    except ValueError:
        raise CustodyError("credential consumer start timestamp is invalid") from None


def projection_precedes_start(
    value: dict[str, Any], *, started_at: str, expected_digest: str
) -> bool:
    names = (
        "mtime_ns",
        "ctime_ns",
        "process_start_ticks",
        "ticks_per_second",
        "realtime_before_ns",
        "boottime_ns",
        "realtime_after_ns",
    )
    if any(type(value.get(name)) is not int or value[name] < 0 for name in names):
        raise CustodyError("node key projection clock evidence is malformed")
    hz = value["ticks_per_second"]
    before, after, boot = (
        value["realtime_before_ns"],
        value["realtime_after_ns"],
        value["boottime_ns"],
    )
    if not 1 <= hz <= 1_000_000 or not 0 <= after - before <= 50_000_000:
        raise CustodyError("node key projection clock sample is inconsistent")
    ticks = value["process_start_ticks"]
    lower_ticks = ticks * NANOSECONDS // hz
    upper_ticks = ((ticks + 1) * NANOSECONDS + hz - 1) // hz
    if lower_ticks > boot:
        raise CustodyError("node key projection process start is in the future")
    lower, upper = before - boot + lower_ticks, after - boot + upper_ticks
    cri_lower, cri_upper = started_interval(started_at)
    if upper < cri_lower or lower >= cri_upper:
        raise CustodyError(
            "kernel process start differs from the bound container clock"
        )
    # /proc start ticks are truncated; only their lower bound authorizes reuse.
    # Kubernetes seconds are used for identity/clock consistency, never to
    # round a later file timestamp back into a permissible interval.
    return bool(
        value.get("sha256") == expected_digest
        and max(value["mtime_ns"], value["ctime_ns"]) <= lower
    )


def projected_key_precedes_process(
    *,
    plane: str,
    pod: dict[str, Any],
    namespace: str,
    node_id: str,
    expected_digest: str,
    kubectl: Sequence[str],
    run: Callable[..., str],
    read_pod: Callable[[str], dict[str, Any]],
) -> bool:
    container = (
        cpu_role_container(
            (pod.get("metadata") or {}).get("labels", {}).get("app", ""),
            pod.get("spec") or {},
        )
        if plane == "cpu"
        else "executor"
    )
    statuses = [
        row for row in pod["status"]["containerStatuses"] if row["name"] == container
    ]
    if len(statuses) != 1 or not statuses[0].get("containerID"):
        raise CustodyError("node key activation container identity is incomplete")
    selected = statuses[0]
    started = selected.get("state", {}).get("running", {}).get("startedAt")
    raw = run(
        [
            *kubectl,
            "-n",
            namespace,
            "exec",
            "-i",
            pod["metadata"]["name"],
            "-c",
            container,
            "--",
            "/opt/gpu-fault/control-plane/bin/python"
            if plane == "cpu"
            else "/opt/gpu-fault/executor/bin/python",
            "-c",
            PROJECTED_KEY_PROBE,
        ],
        input_text=json.dumps({"node_id": node_id}),
        capture=True,
        sensitive=True,
        timeout_seconds=30,
        env=private_command_environment(),
    )
    current = read_pod(pod["metadata"]["name"])
    current_statuses = [
        row
        for row in current.get("status", {}).get("containerStatuses", [])
        if row.get("name") == container
    ]
    if (
        current.get("metadata", {}).get("uid") != pod["metadata"]["uid"]
        or current.get("metadata", {}).get("deletionTimestamp")
        or len(current_statuses) != 1
        or current_statuses[0].get("containerID") != selected["containerID"]
        or current_statuses[0].get("state") != selected.get("state")
        or current_statuses[0].get("ready") is not True
    ):
        raise CustodyError("credential consumer restarted during the key proof")
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
    except (ValueError, TypeError):
        raise CustodyError("node key projection proof is malformed") from None
    return projection_precedes_start(
        value, started_at=started, expected_digest=expected_digest
    )
