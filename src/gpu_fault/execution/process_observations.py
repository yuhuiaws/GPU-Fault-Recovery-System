"""Process-local observations are diagnostic leads, never a fault probability."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def process_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    identity = row.get("process_identity") or {}
    if not isinstance(identity, dict):
        return None
    node, pid, start = (
        row.get("node_id"),
        row.get("pid"),
        identity.get("start_time_ticks"),
    )
    if (
        identity.get("verified") is True
        and isinstance(node, str)
        and node
        and type(pid) is int
        and pid > 0
        and type(start) is int
        and start > 0
    ):
        return node, pid, start
    return None


def number(value: Any) -> float | None:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def observation_window(
    samples: list[dict[str, Any]],
) -> tuple[datetime, datetime] | None:
    try:
        stamps = [datetime.fromisoformat(item["observed_at"]) for item in samples]
        if len(stamps) < 3 or any(item.tzinfo is None for item in stamps):
            return None
        if any(left >= right for left, right in zip(stamps, stamps[1:])):
            return None
        if not 1 <= (stamps[-1] - stamps[0]).total_seconds() <= 30:
            return None
        return stamps[0], stamps[-1]
    except (ValueError, TypeError, KeyError):
        return None


def stack_observations(row: dict[str, Any]) -> dict[str, Any]:
    samples = (row.get("stack_series") or {}).get("samples") or []
    valid = [
        item
        for item in samples
        if item.get("status") == "COLLECTED"
        and item.get("returncode") == 0
        and not item.get("error")
    ]
    result: dict[str, Any] = {
        "requested_samples": 3,
        "valid_samples": len(valid),
        "missing_samples": max(0, 3 - len(valid)),
        "stable_threads": [],
        "errors": [
            item.get("error", item.get("status"))
            for item in samples
            if item not in valid
        ],
        "observations": [],
        "thread_coverage_complete": len(valid) == 3
        and all(not item.get("threads_truncated") for item in valid),
    }
    span = observation_window(valid)
    proc_samples = (row.get("process_series") or {}).get("samples") or []
    if (
        len(valid) != 3
        or span is None
        or len(proc_samples) != 3
        or any(
            sample.get("start_ticks") != row["process_identity"]["start_time_ticks"]
            for sample in proc_samples
        )
    ):
        return result
    maps = [
        {item["thread_key"]: item for item in sample.get("threads", [])}
        for sample in valid
    ]
    for key in maps[0].keys() & maps[1].keys() & maps[2].keys():
        threads = [items[key] for items in maps]
        native = threads[0].get("host_thread_id", threads[0].get("os_thread_id"))
        if native is None or any(
            item.get("host_thread_id", item.get("os_thread_id")) != native
            for item in threads
        ):
            continue
        kernels = [
            (item.get("threads") or {}).get(str(native), {}) for item in proc_samples
        ]
        starts = [item.get("start_ticks") for item in kernels]
        if (
            not starts
            or type(starts[0]) is not int
            or starts[0] <= 0
            or len(set(starts)) != 1
        ):
            continue
        signatures = {item.get("signature") for item in threads}
        roles = {item.get("role") for item in threads}
        if len(signatures) != 1 or None in signatures or len(roles) != 1:
            continue
        thread = {
            **threads[0],
            "thread_start_ticks": starts[0],
            "observed_span_seconds": (span[1] - span[0]).total_seconds(),
        }
        if not (row.get("comparison_context") or {}).get("verified"):
            thread["role"] = "MAIN" if native == row["pid"] else "UNKNOWN"
        result["stable_threads"].append(thread)
        if thread["role"] in {"MAIN", "TRAINING", "COMMUNICATION"}:
            result["observations"].append(
                {
                    "code": "REPEATED_DISTRIBUTED_CALL_CONTEXT"
                    if all(item.get("distributed_call_context") for item in threads)
                    else "REPEATED_CALL_LOCATION",
                    "thread_id": native,
                    "role": thread["role"],
                    "observed_span_seconds": thread["observed_span_seconds"],
                    "candidate": all(
                        item.get("distributed_call_context") for item in threads
                    ),
                    "qualification": "Python location is not proof of native NCCL blocking",
                }
            )
    return result


def behavior_observations(row: dict[str, Any]) -> dict[str, Any]:
    series = row.get("process_series") or {}
    samples = series.get("samples") or []
    windows = series.get("windows") or []
    if any(
        sample.get("start_ticks") != row["process_identity"]["start_time_ticks"]
        for sample in samples
    ):
        windows = []
    valid = [
        item
        for item in windows
        if item.get("status") == "COLLECTED"
        and number(item.get("elapsed_seconds")) is not None
        and item["elapsed_seconds"] >= 0.5
    ]
    span = observation_window(samples)
    cpu_windows = [item for item in valid if number(item.get("cpu_cores")) is not None]
    seconds = sum(item["elapsed_seconds"] for item in cpu_windows)
    cpu = (
        sum(item["cpu_cores"] * item["elapsed_seconds"] for item in cpu_windows)
        / seconds
        if seconds
        else None
    )
    low_fraction = (
        sum(item["elapsed_seconds"] for item in cpu_windows if item["cpu_cores"] < 0.05)
        / seconds
        if seconds
        else None
    )
    result: dict[str, Any] = {
        "valid_windows": len(valid),
        "cpu_valid_windows": len(cpu_windows),
        "cpu_cores": cpu,
        "cpu_low_observed_fraction": low_fraction,
        "observations": [],
        "missing": [],
        "io_windows": [
            {
                key: item.get(key)
                for key in (
                    "elapsed_seconds",
                    "read_bytes_delta",
                    "write_bytes_delta",
                    "wchar_delta",
                )
            }
            for item in valid
        ],
        "gpu_scope": "DEVICE_NOT_PID",
        "io_valid_windows": sum(
            all(
                item.get(key) is not None
                for key in ("read_bytes_delta", "write_bytes_delta", "wchar_delta")
            )
            for item in valid
        ),
    }
    if len(cpu_windows) < 2:
        result["missing"].append("CPU_WINDOW_COVERAGE")
    if result["io_valid_windows"] < 2:
        result["missing"].append("IO_WINDOW_COVERAGE")
    if len(valid) < 2 or not all(item.get("thread_scan_complete") for item in valid):
        result["missing"].append("THREAD_SCAN_COVERAGE")
    gpu = row.get("gpu_series") or {}
    gpu_samples = gpu.get("samples") or []
    devices = row.get("gpu_uuids") or []
    gpu_valid = (
        gpu.get("mapping_verified") is True
        and bool(devices)
        and len(gpu_samples) == 3
        and all(
            set(sample.get("devices") or {}) == set(devices)
            and all(
                number(value.get("utilization_gpu_percent")) is not None
                and value["utilization_gpu_percent"] <= 100
                for value in sample["devices"].values()
            )
            for sample in gpu_samples
        )
    )
    result["gpu_mapping_verified"] = gpu.get("mapping_verified") is True
    result["gpu_shared"] = gpu.get("shared_with_other_processes")
    result["gpu_idle_sample_fraction"] = (
        sum(
            all(
                value["utilization_gpu_percent"] == 0
                for value in sample["devices"].values()
            )
            for sample in gpu_samples
        )
        / 3
        if gpu_valid
        else None
    )
    if not gpu_valid:
        result["missing"].append("GPU_MAPPING_OR_SAMPLE_COVERAGE")
    if len(valid) != 2 or span is None:
        result["missing"].append("REPEATED_PROCESS_WINDOW")
        return result
    elapsed = (span[1] - span[0]).total_seconds()
    waits = [
        {
            (item["tid"], item["start_ticks"]): item
            for item in window.get("wait_threads", [])
        }
        for window in valid
    ]
    context = row.get("comparison_context") or {}
    roles = (
        (context.get("host_thread_roles") or context.get("thread_roles") or {})
        if context.get("verified") is True
        else {}
    )
    for identity in waits[0].keys() & waits[1].keys():
        items = [wait[identity] for wait in waits]
        repeated_d = all(item.get("repeated_d") for item in items)
        same_wait = (
            all(item.get("same_wait") for item in items)
            and len({item.get("wchan") for item in items}) == 1
        )
        if repeated_d or same_wait:
            role = roles.get(
                str(identity[0]), "MAIN" if identity[0] == row["pid"] else "UNKNOWN"
            )
            result["observations"].append(
                {
                    "code": "REPEATED_D_STATE"
                    if repeated_d
                    else "REPEATED_THREAD_WAIT",
                    "thread_id": identity[0],
                    "thread_start_ticks": identity[1],
                    "wchan": items[-1].get("wchan"),
                    "role": role,
                    "observed_span_seconds": elapsed,
                    "candidate": (repeated_d and role != "BACKGROUND")
                    or role in {"TRAINING", "COMMUNICATION", "MAIN"},
                    "qualification": "Repeated observations, not continuous blocking time or NCCL root cause",
                }
            )
    if cpu is not None:
        result["observations"].append(
            {
                "code": "CPU_LOW_ACTIVITY" if low_fraction == 1 else "CPU_ACTIVITY",
                "cpu_cores": cpu,
                "observed_span_seconds": elapsed,
                "candidate": False,
            }
        )
    if any(
        (item.get("write_bytes_delta") or 0) > 0 or (item.get("wchar_delta") or 0) > 0
        for item in valid
    ):
        result["observations"].append(
            {
                "code": "HOST_WRITES_OBSERVED",
                "candidate": False,
                "qualification": "May be checkpoint, logging or other I/O",
            }
        )
    if gpu_valid and result["gpu_idle_sample_fraction"] == 1:
        result["observations"].append(
            {
                "code": "GPU_DEVICE_IDLE_WITH_LIVE_PROCESS",
                "candidate": False,
                "shared": result["gpu_shared"],
                "scope": "DEVICE_NOT_PID",
            }
        )
    return result
