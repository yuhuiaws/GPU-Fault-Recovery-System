"""Map task processes to active RDMA resources, not a healthy management NIC."""

from __future__ import annotations

import json
import re
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from gpu_fault.node_agent.diagnostic_process import run_probe
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    workload_process,
)


def qp_owner(
    proc_root: Path,
    participants: dict[int, Any],
    owner: Any,
    cache: dict[int, int | None],
) -> int | None:
    """The participant process whose thread created a QP, or ``None``.

    ``rdma -j resource show qp`` prints the *thread* that created each user QP,
    not its process. Linux (https://github.com/torvalds/linux/blob/master/
    drivers/infiniband/core/): ``nldev.c`` ``fill_res_name_pid`` emits
    "pid = task_pid_vnr(res->task);", ``verbs.c`` ``create_qp`` calls
    "rdma_restrack_set_name(&qp->res, uattrs ? NULL : caller);" and
    ``restrack.c`` ``rdma_restrack_set_name`` then
    "rdma_restrack_attach_task(res, current);" -- the calling thread. NCCL runs
    the net plugin's listen/connect/accept on its proxy thread
    (https://github.com/NVIDIA/nccl/blob/master/src/transport/net.cc:
    ``recvProxySetup``, ``sendProxyConnect``, ``recvProxyConnect`` call
    ``proxyState->ncclNet->listen/connect/accept``), and aws-ofi-nccl keys its
    endpoints -- whose ``fi_enable`` creates the libfabric EFA QPs -- by that
    thread (https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp:
    "ep = device->get_ep(domain_key, nccl_net_ofi_gettid());"). So a training
    rank's QPs name a proxy thread id, never the rank's PID. Live (HyperPod,
    GF-GENERAL-LIVE-011, 2026-10-24): no QP matched a rank PID, every node
    reported ``basis=UNKNOWN`` and the hung VALIDATE_FABRIC had no task scope.

    A thread belongs to a participant when it is listed under that process's
    ``/proc/<pid>/task``; anything else -- another job's thread, a kernel QP
    without a pid -- is not the task's.
    """

    if type(owner) is not int or owner <= 0:
        return None
    if owner in participants:
        return owner
    if owner not in cache:
        cache[owner] = next(
            (
                pid
                for pid in participants
                if (proc_root / str(pid) / "task" / str(owner)).is_dir()
            ),
            None,
        )
    return cache[owner]


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
        owners: dict[int, int | None] = {}
        for resource in resources:
            if not isinstance(resource, dict):
                return unknown
            device, port = resource.get("ifname"), resource.get("port")
            pid = qp_owner(
                Path(agent.proc_root), participants, resource.get("pid"), owners
            )
            if (
                pid is None
                or resource.get("state") not in {"RTS", "RTR"}
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
            **({} if complete else {"reason": "TASK_RDMA_COVERAGE_INCOMPLETE"}),
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
