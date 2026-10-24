"""Process-local elapsed time and bounded, identity-bound kernel thread evidence."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

from gpu_fault.node_agent.operations.hung_summary_io import proc_sample, read_bounded

MAX_THREADS = 64


def wait_channel(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and value not in {"0", "0x0", "-", "unknown"} else None


def process_sample(
    root: Path, pid: int, monotonic: Callable[[], float]
) -> dict[str, Any]:
    result = proc_sample(root, pid)
    result["sample_monotonic"] = monotonic()
    result["wchan"] = wait_channel(result.get("wchan"))
    threads: dict[str, dict[str, Any]] = {}
    scanned = 0
    complete = True
    try:
        with os.scandir(root / str(pid) / "task") as entries:
            for directory in entries:
                if not directory.name.isdigit():
                    continue
                scanned += 1
                if scanned > MAX_THREADS:
                    complete = False
                    break
                try:
                    path = Path(directory.path)
                    text = read_bounded(path / "stat", 8192)
                    fields = text[text.rfind(")") + 2 :].split()
                    start = int(fields[19])
                    if start <= 0:
                        raise ValueError("invalid thread start")
                    try:
                        wchan = wait_channel(read_bounded(path / "wchan", 1024))
                    except OSError:
                        wchan = None
                    namespace_ids = [int(directory.name)]
                    try:
                        for line in read_bounded(path / "status").splitlines():
                            if line.startswith("NSpid:"):
                                values = [int(value) for value in line.split()[1:]]
                                if (
                                    values
                                    and values[0] == int(directory.name)
                                    and all(value > 0 for value in values)
                                ):
                                    namespace_ids = values[:32]
                    except (OSError, ValueError):
                        pass
                    threads[directory.name] = {
                        "start_ticks": start,
                        "state": fields[0],
                        "wchan": wchan,
                        "namespace_ids": namespace_ids,
                    }
                except (OSError, ValueError, IndexError):
                    complete = False
    except OSError:
        complete = False
    result["threads"] = threads
    result["scanned_thread_count"] = min(scanned, MAX_THREADS)
    result["thread_scan_complete"] = complete
    return result


def process_window(
    before: dict[str, Any], after: dict[str, Any], clock_ticks: int
) -> dict[str, Any]:
    elapsed = after.get("sample_monotonic", 0) - before.get("sample_monotonic", 0)
    identity = before.get("start_ticks")
    if (
        type(identity) is not int
        or identity <= 0
        or identity != after.get("start_ticks")
        or not math.isfinite(elapsed)
        or not 0 < elapsed <= 30
        or clock_ticks <= 0
    ):
        return {"status": "UNAVAILABLE", "reason": "IDENTITY_OR_INTERVAL_UNKNOWN"}
    deltas = {
        key + "_delta": after[key] - before[key]
        for key in (
            "cpu_ticks",
            "read_bytes",
            "write_bytes",
            "wchar",
            "voluntary_ctxt_switches",
        )
        if type(before.get(key)) is int
        and type(after.get(key)) is int
        and 0 <= before[key] <= after[key] < 2**64
    }
    stable = []
    for tid, old in before.get("threads", {}).items():
        new = after.get("threads", {}).get(tid, {})
        if not new or old["start_ticks"] != new.get("start_ticks"):
            continue
        repeated_d = old.get("state") == new.get("state") == "D"
        same_wait = (
            old.get("state") in {"S", "D"}
            and new.get("state") in {"S", "D"}
            and wait_channel(old.get("wchan")) is not None
            and old["wchan"] == new.get("wchan")
        )
        if repeated_d or same_wait:
            stable.append(
                {
                    "tid": int(tid),
                    "start_ticks": old["start_ticks"],
                    "state_before": old["state"],
                    "state_after": new["state"],
                    "wchan": new.get("wchan"),
                    "repeated_d": repeated_d,
                    "same_wait": same_wait,
                    "observed_span_seconds": elapsed,
                }
            )
    return {
        "status": "COLLECTED",
        "elapsed_seconds": elapsed,
        "clock_ticks_per_second": clock_ticks,
        **deltas,
        "cpu_cores": deltas["cpu_ticks_delta"] / clock_ticks / elapsed
        if "cpu_ticks_delta" in deltas
        else None,
        "wait_threads": stable,
        "thread_scan_complete": before.get("thread_scan_complete") is True
        and after.get("thread_scan_complete") is True,
        "wchan_before": wait_channel(before.get("wchan")),
        "wchan_after": wait_channel(after.get("wchan")),
        "wchan_unchanged": wait_channel(before.get("wchan")) is not None
        and before.get("wchan") == after.get("wchan"),
    }
