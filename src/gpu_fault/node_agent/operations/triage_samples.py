"""Bounded repeated process/stack observations; GPU samples stay device-scoped."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from gpu_fault.node_agent.diagnostic_process import run_probe
from gpu_fault.node_agent.operations.hung_summary_io import (
    PROGRESS_ENV,
    application_sample,
    read_bounded,
)
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    workload_process,
)
from gpu_fault.node_agent.operations.process_samples import (
    process_sample,
    process_window,
)
from gpu_fault.node_agent.operations.stack_samples import stack_summary


def scoped(agent: Any, command: Any, pid: int, start: Any) -> bool:
    paths = command.parameters.get("workload_cgroup_paths_by_node", {}).get(
        command.node_id, []
    )
    return (
        type(start) is int
        and start > 0
        and process_start_ticks(agent.proc_root, pid) == start
        and workload_process(agent.proc_root, pid, paths)
    )


def process_context(
    agent: Any, command: Any, row: dict[str, Any], start: Any
) -> dict[str, Any]:
    if not scoped(agent, command, row["pid"], start):
        return {}
    snapshot = application_sample(
        agent,
        {
            **row,
            "world_size": None,
            "progress_path": (row.get("_environment") or {}).get(PROGRESS_ENV),
        },
        command,
        start,
    )
    if snapshot is None or snapshot.diagnostic_context is None:
        return {}
    context = snapshot.diagnostic_context.model_dump(mode="json")
    members = context.pop("members")
    return {
        **context,
        "membership_sha256": hashlib.sha256(
            json.dumps(sorted(members), separators=(",", ":")).encode()
        ).hexdigest(),
        "member_count": len(members),
        "membership_verified": True,
        "job_id": snapshot.job_id,
        "attempt_id": snapshot.attempt_id,
        "declared_rank": snapshot.rank,
        "source": "PROCESS_BOUND_SNAPSHOT",
    }


def capture_stack(
    agent: Any,
    command: Any,
    row: dict[str, Any],
    start: Any,
    workdir: Path,
    index: int,
    context: dict[str, Any],
    kernel_threads: dict[str, dict[str, Any]],
    deadline: float,
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    began = agent.now().isoformat()
    remaining = deadline - monotonic()
    if not scoped(agent, command, row["pid"], start):
        return {
            "status": "UNAVAILABLE",
            "error": "PROCESS_IDENTITY_UNPROVEN",
            "observed_at": began,
        }
    if not agent.python_stack_tool or remaining < 1:
        return {
            "status": "UNAVAILABLE",
            "error": "TOOL_DISABLED_OR_BUDGET_EXHAUSTED",
            "observed_at": began,
        }
    try:
        capture = agent._capture_python_stack_sample(
            workdir,
            row["pid"],
            index,
            3,
            min(2, int(remaining)),
            structured=True,
        )
        if capture.get("returncode") != 0 or capture.get("error"):
            return {
                "status": "FAILED",
                "error": capture.get("error") or "PYSPY_NONZERO_EXIT",
                "returncode": capture.get("returncode"),
                "observed_at": began,
            }
        summary = stack_summary(
            read_bounded(workdir / capture["file"], 1_048_576),
            row["pid"],
            context.get("thread_roles") or {},
            kernel_threads,
        )
        if not scoped(agent, command, row["pid"], start):
            return {
                "status": "UNAVAILABLE",
                "error": "PROCESS_CHANGED_DURING_STACK",
                "observed_at": began,
            }
        return {
            **summary,
            "observed_at": began,
            "finished_at": agent.now().isoformat(),
            "returncode": 0,
        }
    except Exception as exc:
        return {"status": "FAILED", "error": str(exc)[:512], "observed_at": began}


def current_gpu_clients(
    agent: Any, deadline: float, monotonic: Callable[[], float]
) -> dict[int, list[str]] | None:
    remaining = deadline - monotonic()
    if remaining <= 0:
        return None
    try:
        result = run_probe(
            agent.runner,
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=min(2, remaining),
        )
        if result.returncode:
            return None
        rows: dict[int, list[str]] = {}
        lines = result.stdout.splitlines()
        if len(lines) > 4096:
            return None
        for line in lines:
            fields = [value.strip() for value in line.split(",", 2)]
            if (
                len(fields) != 3
                or not fields[0].isdigit()
                or not fields[1].startswith("GPU-")
            ):
                return None
            rows.setdefault(int(fields[0]), []).append(fields[1])
        return rows
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def collect_process_series(
    agent: Any,
    command: Any,
    rows: list[dict[str, Any]],
    identities: dict[int, dict[str, Any]],
    initial_clients: dict[int, list[str]] | None,
    deadline: float,
    monotonic: Callable[[], float],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    contexts = {
        row["pid"]: process_context(
            agent, command, row, identities[row["pid"]]["start_time_ticks"]
        )
        for row in rows
    }
    process_samples: dict[int, list[dict[str, Any]]] = {row["pid"]: [] for row in rows}
    stacks: dict[int, list[dict[str, Any]]] = {row["pid"]: [] for row in rows}
    gpu_samples: list[dict[str, Any]] = []
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    with tempfile.TemporaryDirectory(prefix="gpu-fault-triage-") as temporary:
        for index in range(3):
            if monotonic() >= deadline:
                break
            if index:
                agent.sleep(min(1, max(0, deadline - monotonic())))
                if monotonic() >= deadline:
                    break
            for row in rows:
                scope_verified = scoped(
                    agent,
                    command,
                    row["pid"],
                    identities[row["pid"]]["start_time_ticks"],
                )
                sample = (
                    process_sample(agent.proc_root, row["pid"], monotonic)
                    if scope_verified
                    or not command.parameters.get("require_workload_identity")
                    else {
                        "error": "PROCESS_SCOPE_CHANGED",
                        "sample_monotonic": monotonic(),
                        "threads": {},
                        "thread_scan_complete": False,
                    }
                )
                sample["workload_scope_verified"] = scope_verified
                sample["observed_at"] = agent.now().isoformat()
                process_samples[row["pid"]].append(sample)
            remaining = deadline - monotonic()
            gpu_samples.append(
                {
                    "observed_at": agent.now().isoformat(),
                    "devices": agent._gpu_triage_snapshot(
                        timeout_seconds=min(2, remaining)
                    )
                    if remaining > 0
                    else {},
                }
            )
            if rows:
                with ThreadPoolExecutor(
                    max_workers=min(8, len(rows)), thread_name_prefix="triage-stack"
                ) as pool:
                    futures = {
                        row["pid"]: pool.submit(
                            capture_stack,
                            agent,
                            command,
                            row,
                            identities[row["pid"]]["start_time_ticks"],
                            Path(temporary),
                            index + 1,
                            contexts[row["pid"]],
                            process_samples[row["pid"]][-1].get("threads", {}),
                            deadline,
                            monotonic,
                        )
                        for row in rows
                    }
                    for pid, future in futures.items():
                        stacks[pid].append(future.result())
    clients = current_gpu_clients(agent, deadline, monotonic) if rows else None
    result = []
    for row in rows:
        pid = row["pid"]
        samples = process_samples[pid]
        windows = [
            process_window(left, right, clock_ticks)
            for left, right in zip(samples, samples[1:])
        ]
        verified = scoped(agent, command, pid, identities[pid]["start_time_ticks"])
        context = contexts[pid]
        context = {
            **context,
            "verified": bool(
                context
                and context
                == process_context(
                    agent, command, row, identities[pid]["start_time_ticks"]
                )
            ),
        }
        roles = context.get("thread_roles") or {}
        context["host_thread_roles"] = {
            tid: roles[str((value.get("namespace_ids") or [int(tid)])[-1])]
            for tid, value in (samples[0].get("threads", {}) if samples else {}).items()
            if str((value.get("namespace_ids") or [int(tid)])[-1]) in roles
        }
        owned = sorted(set((initial_clients or {}).get(pid, [])))
        mapping_valid = (
            initial_clients is not None
            and clients is not None
            and owned == sorted(set(clients.get(pid, [])))
            and bool(owned)
        )
        shared = any(
            other_pid != pid and set(owned).intersection(devices)
            for mapping in (initial_clients or {}, clients or {})
            for other_pid, devices in mapping.items()
        )
        result.append(
            {
                "node_id": command.node_id,
                "pid": pid,
                "rank": row.get("rank"),
                "gpu_uuid": row.get("gpu_uuid"),
                "gpu_uuids": owned,
                "local_rank": row.get("local_rank"),
                "process_name": row.get("process_name"),
                "process_identity": {**identities[pid], "verified": verified},
                "comparison_context": context,
                "process_series": {
                    "version": "process-series/v1",
                    "requested_samples": 3,
                    "samples": samples,
                    "windows": windows,
                },
                "stack_series": {
                    "version": "stack-series/v1",
                    "requested_samples": 3,
                    "samples": stacks[pid],
                },
                "gpu_series": {
                    "scope": "DEVICE_NOT_PID",
                    "mapping_verified": mapping_valid,
                    "shared_with_other_processes": shared,
                    "samples": [
                        {
                            **item,
                            "devices": {
                                gpu: value
                                for gpu, value in item["devices"].items()
                                if gpu in owned
                            },
                        }
                        for item in gpu_samples
                    ],
                },
                "python_stack": stacks[pid][-1] if stacks[pid] else {},
                "proc": windows[-1] if windows else {"status": "UNAVAILABLE"},
                "gpu": gpu_samples[-1]["devices"].get(row.get("gpu_uuid"), {})
                if gpu_samples
                else {},
            }
        )
    return result
