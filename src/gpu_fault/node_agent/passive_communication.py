"""Passive local histories and a single task-elected RAS query. No dump triggers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.communication_monitoring import CommunicationPlan, CommunicationTarget
from gpu_fault.node_agent.communication_history import CommunicationHistory
from gpu_fault.node_agent.operations.hung_summary import gpu_sample
from gpu_fault.node_agent.operations.hung_summary_io import (
    application_sample,
    discover,
    proc_sample,
    read_bounded,
)
from gpu_fault.node_agent.operations.network_bindings import task_rdma_bindings
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    workload_process,
)
from gpu_fault.node_agent.operations.workload_scope import target_workload_scope
from gpu_fault.ras_observation import query_ras
from gpu_fault.node_agent.operations.collection_budget import publish_partial


def listening_owner(
    agent: Any, rows: list[dict[str, Any]]
) -> tuple[int, str, int] | None:
    for row in rows:
        if not row["gpu_uuids"] or not row["process_identity"]["verified"]:
            continue
        pid = row["pid"]
        environment = agent._process_environment(pid)
        address = environment.get("NCCL_RAS_ADDR", "127.0.0.1:28028")
        host, _, port = address.rpartition(":")
        host = host.strip("[]")
        if host not in {"127.0.0.1", "::1", "0.0.0.0", "::"} or not port.isdecimal():
            continue
        try:
            inodes: set[str] = set()
            for name in ("tcp", "tcp6"):
                if (name == "tcp6") != (":" in host):
                    continue
                try:
                    lines = read_bounded(
                        agent.proc_root / str(pid) / "net" / name, 262144
                    ).splitlines()[1:]
                except FileNotFoundError:
                    continue
                inodes.update(
                    line.split()[9]
                    for line in lines
                    if len(line.split()) > 9
                    and line.split()[3] == "0A"
                    and line.split()[1].split(":")[0]
                    in (
                        {
                            "00000000000000000000000000000000",
                            "00000000000000000000000001000000",
                        }
                        if name == "tcp6"
                        else {"00000000", "0100007F"}
                    )
                    and int(line.split()[1].split(":")[-1], 16) == int(port)
                )
            for index, fd in enumerate((agent.proc_root / str(pid) / "fd").iterdir()):
                if index >= 4096:
                    break
                link = os.readlink(fd)
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return pid, "::1" if ":" in host else "127.0.0.1", int(port)
        except (OSError, ValueError):
            continue
    return None


def read_ras(
    agent: Any, rows: list[dict[str, Any]], coordinator: bool, paths: list[str]
) -> dict[str, Any]:
    if not coordinator:
        return {"status": "NOT_COORDINATOR"}
    owner = listening_owner(agent, rows)
    if owner is None:
        return {"status": "UNAVAILABLE", "reason": "NO_SCOPED_RAS_LISTENER"}
    pid, host, port = owner
    start = process_start_ticks(agent.proc_root, pid)
    expected = next(
        row["process_identity"]["start_time_ticks"] for row in rows if row["pid"] == pid
    )
    if start != expected or not workload_process(agent.proc_root, pid, paths):
        return {"status": "UNAVAILABLE", "reason": "RAS_OWNER_IDENTITY_CHANGED"}
    try:
        with (agent.proc_root / str(pid) / "ns/net").open("rb") as namespace:
            if (
                os.fstat(namespace.fileno()).st_ino
                != os.stat("/proc/self/ns/net").st_ino
            ):
                os.setns(namespace.fileno(), 0)
        result = query_ras(host, port)
        if process_start_ticks(agent.proc_root, pid) != start or not workload_process(
            agent.proc_root, pid, paths
        ):
            raise ValueError("RAS owner process changed")
        return {
            **result,
            "status": "COLLECTED",
            "listener_pid": pid,
            "listener_start_ticks": start,
        }
    except (OSError, ValueError, TimeoutError) as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc)[:256]}


def local_flight(agent: Any, row: dict[str, Any]) -> dict[str, Any]:
    if not row["gpu_uuids"] or not row["process_identity"]["verified"]:
        return {"status": "UNAVAILABLE"}
    environment = agent._process_environment(row["pid"])
    candidates = agent._flight_dump_candidates(environment, row["rank"], pid=row["pid"])
    fresh = []
    for path in candidates:
        info = agent._dump_file_stat(path)
        if (
            info
            and info.st_size <= 2 * 1024 * 1024
            and 0 <= time.time() - info.st_mtime <= 60
        ):
            fresh.append(str(path))
    if not fresh:
        return {"status": "UNAVAILABLE", "reason": "NO_FRESH_BOUNDED_FR_FILE"}
    value: dict[str, Any] = agent._finish_flight_recorder(
        {"status": "triggered", "dump_candidates": fresh}
    )
    value["source"] = "EXISTING_PROCESS_NAMESPACE_FILE"
    value["dump_triggered"] = False
    if value.get("dump_file"):
        value["source_mtime_ns"] = Path(value["dump_file"]).stat().st_mtime_ns
    # Histories keep status watermarks. Full operation semantics remain in the
    # bounded incident dump, not in every routine sample.
    for group in value.get("last_entries_by_pg", []):
        group.pop("semantic_entries", None)
    value.pop("unfinished_entries", None)
    value.pop("last_entry", None)
    return value


def sample_communication(
    agent: Any, raw_target: dict[str, Any], raw_plan: dict[str, Any]
) -> dict[str, Any]:
    target, plan = (
        CommunicationTarget.model_validate(raw_target),
        CommunicationPlan.model_validate(raw_plan),
    )
    started = agent.now()
    # A managed Pod's target names its containers by runtime id (v2 lease).
    paths, workload_scope = target_workload_scope(
        agent.proc_root, target.cgroup_paths, target.container_ids
    )
    if not paths:
        # No container found here: missing evidence, never an empty job.
        return {
            "status": "SCOPE_UNKNOWN",
            "process_count": 0,
            "observed_at": started.isoformat(),
            "scan_complete": False,
            "workload_scope": workload_scope,
            "dump_triggers": 0,
        }
    rows, coverage = discover(agent.proc_root, paths)
    if workload_scope.get("complete") is False:
        # The processes of a container this node could not find are not here.
        coverage["scan_complete"] = False
    devices, clients = gpu_sample(agent)
    command = SimpleNamespace(
        parameters={"job_id": target.job_id, "attempt_id": target.attempt_id}
    )
    samples = []
    for row in rows:
        proc = proc_sample(agent.proc_root, row["pid"])
        start = proc.get("start_ticks")
        valid = bool(start and proc.get("state") not in {"Z", "X"})
        point: dict[str, Any] = {
            **{name: row[name] for name in ("pid", "rank", "world_size")},
            "process_identity": {"verified": valid, "start_time_ticks": start},
            "gpu_uuids": (clients or {}).get(row["pid"], []),
            "proc": proc,
        }
        application = application_sample(agent, row, command, start)
        point["application"] = (
            application.model_dump(mode="json", exclude={"step", "phase"})
            if application
            else None
        )
        point["flight_recorder"] = local_flight(agent, point)
        point["process_identity"]["verified"] = (
            valid
            and process_start_ticks(agent.proc_root, row["pid"]) == start
            and workload_process(agent.proc_root, row["pid"], paths)
        )
        samples.append(point)
    bindings = task_rdma_bindings(agent, samples, paths)
    publish_partial(
        agent,
        {"status": "COLLECTING", "processes": samples, "fabric_binding": bindings},
    )
    ras = read_ras(agent, samples, target.ras_coordinator, paths)
    document = {
        "version": "communication-sample/v1",
        "job_id": target.job_id,
        "attempt_id": target.attempt_id,
        "boot_id": plan.boot_id,
        "node_id": plan.node_id,
        "node_addresses": plan.node_addresses,
        "observed_at": started.isoformat(),
        "finished_at": agent.now().isoformat(),
        "processes": samples,
        "gpu": {
            gpu: value
            for gpu, value in devices.items()
            if gpu in {gpu for row in samples for gpu in row["gpu_uuids"]}
        },
        "fabric_binding": bindings,
        "ras": ras,
        **coverage,
        "workload_scope": workload_scope,
        "participant_inventory_available": clients is not None,
    }
    history = CommunicationHistory(
        agent.diagnostic_output_dir / "communication-history.db"
    )
    history.append(
        history.scope(target.job_id, target.attempt_id, plan.boot_id),
        int(started.timestamp() // 30),
        started.timestamp(),
        document,
    )
    return {
        "status": "SAMPLED",
        "process_count": len(samples),
        "observed_at": started.isoformat(),
        "ras_status": ras["status"],
        "fabric_binding_status": bindings["basis"],
        "scan_complete": coverage["scan_complete"],
        "workload_scope": workload_scope,
        "dump_triggers": 0,
    }
