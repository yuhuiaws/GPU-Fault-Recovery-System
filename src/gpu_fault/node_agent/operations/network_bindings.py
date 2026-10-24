"""Map task processes to active RDMA resources, not a healthy management NIC."""

from __future__ import annotations

import json
import re
from subprocess import TimeoutExpired
from typing import Any

from gpu_fault.node_agent.diagnostic_process import run_probe
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    workload_process,
)


def task_rdma_bindings(
    agent: Any, rows: list[dict[str, Any]], paths: list[str] | None = None
) -> dict[str, Any]:
    participants = {
        row["pid"]: row["process_identity"]["start_time_ticks"]
        for row in rows
        if row.get("gpu_uuids") and row.get("process_identity", {}).get("verified")
    }
    unknown = {
        "basis": "UNKNOWN",
        "reason": "TASK_RDMA_RESOURCES_UNAVAILABLE",
        "network_interfaces": [],
        "rdma_ports": [],
    }
    if not participants:
        return unknown
    try:
        response = run_probe(
            agent.runner,
            ["rdma", "-j", "resource", "show", "qp"],
            timeout=2,
            capture_output=True,
            text=True,
            check=False,
        )
        resources = json.loads(response.stdout) if response.returncode == 0 else None
        if not isinstance(resources, list) or len(resources) > 4096:
            return unknown
        ports: dict[int, set[str]] = {}
        for resource in resources:
            if not isinstance(resource, dict):
                return unknown
            pid, device, port = (
                resource.get("pid"),
                resource.get("ifname"),
                resource.get("port"),
            )
            if (
                pid not in participants
                or resource.get("state") not in {"RTS", "RTR"}
                or not isinstance(pid, int)
                or not isinstance(device, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", device)
                or type(port) is not int
                or port <= 0
            ):
                continue
            if not (agent.infiniband_root / device / "ports" / str(port)).is_dir():
                continue
            ports.setdefault(pid, set()).add(f"{device}/{port}")
        complete = set(ports) == set(participants) and all(
            process_start_ticks(agent.proc_root, pid) == start
            and (paths is None or workload_process(agent.proc_root, pid, paths))
            for pid, start in participants.items()
        )
        return {
            "basis": "TASK_DEVICE_BINDING" if complete else "UNKNOWN",
            "source": "LIVE_PROCESS_RDMA_QP",
            "network_interfaces": [],
            "rdma_ports": sorted(
                {port for values in ports.values() for port in values}
            ),
            "covered_pids": sorted(ports),
            "expected_pids": sorted(participants),
            "coverage_complete": complete,
            "nccl_transport_exclusive": False,
        }
    except (OSError, ValueError, TimeoutError, TimeoutExpired):
        return unknown
