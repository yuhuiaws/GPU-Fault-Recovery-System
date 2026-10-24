"""Bounded py-spy 0.4.1 JSON parsing with thread identity and explicit coverage."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MAX_THREADS = 32
MAX_FRAMES = 16
MAX_SUMMARY_BYTES = 8192
CALLS = frozenset(
    {
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "all_to_all",
        "all_to_all_single",
        "broadcast",
        "barrier",
        "send",
        "recv",
        "isend",
        "irecv",
        "batch_isend_irecv",
    }
)


def stack_summary(
    text: str,
    pid: int,
    roles: dict[str, str] | None = None,
    kernel_threads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "status": "INVALID",
        "signature": None,
        "frames": [],
        "threads": [],
        "collective_frames": False,
    }
    try:
        if len(text.encode()) > 1_048_576:
            raise ValueError("stack JSON exceeds 1 MiB")
        payload = json.loads(text)
        if not isinstance(payload, list) or not payload or len(payload) > 4096:
            raise ValueError("expected a nonempty py-spy thread array")
        threads = [parse_thread(item, pid, roles or {}) for item in payload]
        if kernel_threads is not None:
            bind_native_threads(threads, kernel_threads, pid)
        identifiers = [item["thread_key"] for item in threads]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate thread identity")
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        return {**failure, "error": str(exc)[:512]}
    priority = {
        "TRAINING": 0,
        "MAIN": 1,
        "COMMUNICATION": 2,
        "UNKNOWN": 3,
        "BACKGROUND": 4,
    }
    threads.sort(key=lambda item: (priority[item["role"]], item["thread_key"]))
    selected = threads[:MAX_THREADS]
    while selected and len(json.dumps(selected).encode()) > MAX_SUMMARY_BYTES:
        selected.pop()
    if not selected:
        return {
            **failure,
            "error": "STACK_SUMMARY_BUDGET",
            "observed_thread_count": len(threads),
        }
    main = next((item for item in selected if item["role"] == "TRAINING"), None)
    if main is None:
        main = next((item for item in selected if item["role"] == "MAIN"), None)
    return {
        "status": "COLLECTED",
        "threads": selected,
        "observed_thread_count": len(threads),
        "retained_thread_count": len(selected),
        "threads_truncated": len(selected) != len(threads),
        "signature": main["signature"] if main else None,
        "frames": main["frames"] if main else [],
        "collective_frames": bool(main and main["distributed_call_context"]),
        "training_thread_declared": bool(main and main["role"] == "TRAINING"),
        "scope": "PYTHON_ONLY_NOT_CUDA_OR_NCCL_EXECUTION_STATE",
    }


def bind_native_threads(
    threads: list[dict[str, Any]], kernels: dict[str, dict[str, Any]], pid: int
) -> None:
    mapping: dict[int, list[int]] = {}
    for tid, kernel in kernels.items():
        inner = (kernel.get("namespace_ids") or [int(tid)])[-1]
        mapping.setdefault(inner, []).append(int(tid))
    for thread in threads:
        native = thread.get("os_thread_id")
        matches = mapping.get(native, []) if type(native) is int else []
        host = matches[0] if len(matches) == 1 else None
        thread["host_thread_id"] = host
        if thread["role_source"] != "APPLICATION":
            thread["role"] = "MAIN" if host == pid else "UNKNOWN"


def parse_thread(item: Any, pid: int, roles: dict[str, str]) -> dict[str, Any]:
    if (
        not isinstance(item, dict)
        or type(item.get("pid")) is not int
        or item["pid"] != pid
    ):
        raise ValueError("stack PID does not match requested process")
    python_tid, native_tid = item.get("thread_id"), item.get("os_thread_id")
    if (
        type(python_tid) is not int
        or python_tid <= 0
        or (native_tid is not None and (type(native_tid) is not int or native_tid <= 0))
    ):
        raise ValueError("invalid thread identity")
    raw = item.get("frames")
    if not isinstance(raw, list) or len(raw) > 4096:
        raise ValueError("invalid frame array")
    frames = []
    for frame in raw[:MAX_FRAMES]:
        if (
            not isinstance(frame, dict)
            or any(not isinstance(frame.get(key), str) for key in ("name", "filename"))
            or type(frame.get("line")) is not int
        ):
            raise ValueError("malformed stack frame")
        frames.append(
            {
                "function": frame["name"][:256],
                "file": frame["filename"][:512],
                "line": frame["line"],
            }
        )
    role = roles.get(str(native_tid), "MAIN" if native_tid == pid else "UNKNOWN")
    if role not in {"TRAINING", "COMMUNICATION", "BACKGROUND", "MAIN", "UNKNOWN"}:
        role = "UNKNOWN"
    call_context = any(
        frame["function"] in CALLS
        and "/torch/distributed/" in frame["file"].replace("\\", "/")
        for frame in frames[:3]
    )
    return {
        "thread_key": f"{python_tid}/{native_tid}",
        "thread_id": python_tid,
        "os_thread_id": native_tid,
        "name": str(item.get("thread_name") or "")[:128],
        "role": role,
        "role_source": "APPLICATION"
        if str(native_tid) in roles
        else "OS_MAIN_OR_UNKNOWN",
        "active": item.get("active") if type(item.get("active")) is bool else None,
        "owns_gil": item.get("owns_gil")
        if type(item.get("owns_gil")) is bool
        else None,
        "frames": frames,
        "frames_truncated": len(raw) > len(frames),
        "signature": hashlib.sha256(
            json.dumps(frames, sort_keys=True).encode()
        ).hexdigest()
        if frames
        else None,
        "distributed_call_context": call_context,
    }
