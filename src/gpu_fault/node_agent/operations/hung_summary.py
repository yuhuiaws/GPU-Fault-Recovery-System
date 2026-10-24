"""Two passive samples of all discoverable workload ranks on one node."""

from __future__ import annotations

import json
import math
import subprocess
import time
from typing import Any

from gpu_fault.node_agent.diagnostic_process import run_probe
from gpu_fault.node_agent.operations.collection_budget import (
    collection_lease,
    publish_partial,
)
from gpu_fault.node_agent.operations.hung_summary_io import (
    SUMMARY_VERSION,
    application_sample,
    discover,
    proc_sample,
    rdma_sample,
)
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    namespace_pids,
    workload_process,
)
from gpu_fault.node_agent.operations.process_samples import wait_channel
from gpu_fault.node_agent.protocol import NodeActionCommand
from gpu_fault.node_agent.operations.network_bindings import task_rdma_bindings


def gpu_sample(
    agent: Any,
) -> tuple[dict[str, dict[str, float]], dict[int, list[str]] | None]:
    devices: dict[str, dict[str, float]] = {}
    processes: dict[int, list[str]] = {}
    inventory_available = False
    queries = (
        "--query-gpu=uuid,utilization.gpu,utilization.memory",
        "--query-compute-apps=pid,gpu_uuid",
    )
    for query in queries:
        try:
            result = run_probe(
                agent.runner,
                ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode:
                continue
            lines = result.stdout.splitlines()
            if query == queries[1]:
                inventory_available = len(lines) <= 4096
            for line in lines[:4096]:
                fields = [item.strip() for item in line.split(",")]
                if query == queries[0] and len(fields) == 3:
                    try:
                        gpu, memory = float(fields[1]), float(fields[2])
                    except ValueError:
                        continue
                    if (
                        math.isfinite(gpu)
                        and math.isfinite(memory)
                        and 0 <= gpu <= 100
                        and 0 <= memory <= 100
                    ):
                        devices[fields[0]] = {
                            "utilization_gpu_percent": gpu,
                            "utilization_memory_percent": memory,
                        }
                elif (
                    query == queries[1]
                    and len(fields) == 2
                    and fields[0].isdigit()
                    and fields[1].startswith("GPU-")
                ):
                    processes.setdefault(int(fields[0]), []).append(fields[1])
                elif query == queries[1]:
                    inventory_available = False
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return devices, processes if inventory_available else None


def paired_rank(
    agent: Any,
    command: NodeActionCommand,
    row: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    application: Any,
    paths: list[str],
    groups: dict[str, Any],
) -> dict[str, Any]:
    start = first.get("start_ticks")
    valid = bool(
        start
        and start == second.get("start_ticks")
        and second.get("state") not in {"Z", "X"}
        and workload_process(agent.proc_root, row["pid"], paths)
    )
    result = {
        "pid": row["pid"],
        "rank": row["rank"],
        "world_size": row["world_size"],
        "process_identity": {
            "verified": valid,
            "start_time_ticks": start,
            "namespace_pids": namespace_pids(agent.proc_root, row["pid"]),
        },
        "proc": {},
        "communication_status": "UNAVAILABLE",
        "communication": [],
    }
    if not valid:
        result["error"] = "PROCESS_CHANGED_OR_UNREADABLE"
        return result
    result["proc"] = {
        "state": second.get("state"),
        "state_before": first.get("state"),
        "wchan": wait_channel(second.get("wchan")),
        "wchan_unchanged": bool(
            wait_channel(first.get("wchan"))
            and first.get("wchan") == second.get("wchan")
        ),
        **{
            key + "_delta": second[key] - first[key]
            for key in (
                "cpu_ticks",
                "write_bytes",
                "wchar",
                "read_bytes",
                "voluntary_ctxt_switches",
            )
            if key in first and key in second and second[key] >= first[key]
        },
    }
    current = application_sample(agent, row, command, start)
    if (
        application is None
        or current is None
        or current.observed_at < application.observed_at
    ):
        return result
    result["world_size"] = current.world_size
    result["rank"] = current.rank
    if current.process_groups and application.process_groups:
        result["participant_basis"] = "PROCESS_BOUND_COMMUNICATION"
    result["communication_status"] = "AVAILABLE"
    before_groups = {item.descriptor()[0]: item for item in application.process_groups}
    communication = []
    for group in current.process_groups:
        key, descriptor = group.descriptor()
        previous = before_groups.get(key)
        if previous is None:
            continue
        groups[key] = descriptor
        communication.append(
            {
                "group": key,
                "before": [previous.enqueued, previous.started, previous.completed],
                "after": [group.enqueued, group.started, group.completed],
                "first_observed_at": application.observed_at.isoformat(),
                "second_observed_at": current.observed_at.isoformat(),
                "clock_uncertainty_seconds": command.parameters.get(
                    "clock_uncertainty_seconds"
                ),
            }
        )
    result["communication"] = communication
    return result


def collect_hung_summary(agent: Any, command: NodeActionCommand) -> dict[str, Any]:
    with collection_lease(agent, command) as refused:
        if refused:
            return {"status": refused, "ranks": [], "scan_complete": False}
        return _collect_hung_summary(agent, command)


def _collect_hung_summary(agent: Any, command: NodeActionCommand) -> dict[str, Any]:
    parameters = command.parameters
    scope = parameters.get("workload_cgroup_paths_by_node", {}).get(command.node_id, [])
    epoch = parameters.get("snapshot_epoch")
    if not epoch or not parameters.get("job_id") or not parameters.get("attempt_id"):
        raise ValueError("hung summary requires an explicit snapshot epoch and attempt")
    base = {
        "summary_version": SUMMARY_VERSION,
        "snapshot_epoch": epoch,
        "node_id": command.node_id,
        "job_id": parameters["job_id"],
        "attempt_id": parameters["attempt_id"],
        "ranks": [],
        "groups": {},
        "gpu": {},
        "rdma": {},
        "read_only": True,
        "attachments": 0,
    }
    if not scope or not all(
        isinstance(path, str) and path.startswith("/") and path != "/" for path in scope
    ):
        return {**base, "status": "SCOPE_UNKNOWN", "scan_complete": False}
    started = agent.now()
    rows, coverage = discover(agent.proc_root, scope)
    first = {row["pid"]: proc_sample(agent.proc_root, row["pid"]) for row in rows}
    application = {
        row["pid"]: application_sample(
            agent, row, command, first[row["pid"]].get("start_ticks")
        )
        for row in rows
    }
    rdma_before = rdma_sample(agent.infiniband_root)
    monotonic_start = time.monotonic()
    agent.sleep(1)
    second = {row["pid"]: proc_sample(agent.proc_root, row["pid"]) for row in rows}
    rdma_after = rdma_sample(agent.infiniband_root)
    interval = time.monotonic() - monotonic_start
    groups: dict[str, Any] = {}
    ranks = [
        paired_rank(
            agent,
            command,
            row,
            first[row["pid"]],
            second[row["pid"]],
            application[row["pid"]],
            scope,
            groups,
        )
        for row in rows
    ]
    publish_partial(
        agent,
        {**base, **coverage, "status": "COLLECTING", "ranks": ranks, "groups": groups},
    )
    gpu, gpu_by_pid = gpu_sample(agent)
    for row in ranks:
        identity = row["process_identity"]
        identity["verified"] = bool(
            identity["verified"]
            and identity["start_time_ticks"]
            == process_start_ticks(agent.proc_root, row["pid"])
            and workload_process(agent.proc_root, row["pid"], scope)
        )
        row["gpu_uuids"] = (gpu_by_pid or {}).get(row["pid"], [])
        # A child may inherit RANK and even the progress-file path. Only the
        # process-bound registration or its own CUDA context establishes candidacy.
        row.setdefault(
            "participant_basis",
            "GPU_CONTEXT"
            if row["gpu_uuids"]
            else "NO_OBSERVED_GPU_OR_COMMUNICATION"
            if gpu_by_pid is not None
            else "UNKNOWN",
        )
    groups = compact_groups(groups, ranks)
    rdma = {}
    for port in rdma_before.keys() | rdma_after.keys():
        before, after = rdma_before.get(port, {}), rdma_after.get(port, {})
        valid = bool(
            interval > 0
            and before.get("source") == after.get("source")
            and all(
                key in before and key in after and after[key] >= before[key]
                for key in ("rx", "tx")
            )
        )
        rdma[port] = {
            "valid": valid,
            "source": after.get("source"),
            "rx_bytes_delta": after["rx"] - before["rx"] if valid else None,
            "tx_bytes_delta": after["tx"] - before["tx"] if valid else None,
        }
    result = {
        **base,
        **coverage,
        "status": "COLLECTED",
        "ranks": ranks,
        "participant_inventory_available": gpu_by_pid is not None,
        "groups": groups,
        "gpu": {
            key: value
            for key, value in gpu.items()
            if not command.gpu_uuids or key in command.gpu_uuids
        },
        "rdma": rdma,
        "sample_started_at": started.isoformat(),
        "sample_finished_at": agent.now().isoformat(),
        "sample_interval_seconds": interval,
        "training_fabric_scope": task_rdma_bindings(agent, ranks, scope),
    }
    from gpu_fault.node_agent.operations.communication_cache import (
        cached_communication,
        apply_cached_flights,
    )

    result.update(cached_communication(agent, command))
    apply_cached_flights(result)
    if len(json.dumps(result).encode()) > 1_048_576:
        return {**base, "status": "RESULT_SIZE_LIMIT", "scan_complete": False}
    return result


def compact_groups(
    groups: dict[str, Any],
    ranks: list[dict[str, Any]],
) -> dict[str, Any]:
    references = {key: str(index) for index, key in enumerate(sorted(groups))}
    for rank in ranks:
        for item in rank.get("communication", []):
            item["group"] = references[item["group"]]
    return {
        references[key]: {**value, "group_key": key} for key, value in groups.items()
    }
