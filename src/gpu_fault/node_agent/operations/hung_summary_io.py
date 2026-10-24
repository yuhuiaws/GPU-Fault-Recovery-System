"""Bounded passive reads for full-task hung surveys. No ptrace or dump triggers."""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path
from typing import Any

from gpu_fault.hung_summary_codec import SUMMARY_VERSION as SUMMARY_VERSION
from gpu_fault.node_agent.operations.collection_budget import bounded_read
from gpu_fault.node_agent.operations.flight_recorder import (
    FlightRecorderOperationsMixin,
)
from gpu_fault.node_agent.operations.process_identity import workload_process
from gpu_fault.rank_progress import RankProgressSnapshot

MAX_PROCESSES = 256
MAX_SCAN_PROCESSES = 16384
PROGRESS_ENV = "GPU_FAULT_RANK_PROGRESS_FILE"


def read_bounded(path: Path, limit: int = 65536) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("snapshot source is not a regular file")
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("snapshot source exceeds size limit")
    return value.decode("utf-8", errors="replace")


def proc_sample(root: Path, pid: int) -> dict[str, Any]:
    directory = root / str(pid)
    value: dict[str, Any] = {}
    try:
        text = read_bounded(directory / "stat", 8192)
        fields = text[text.rfind(")") + 2 :].split()
        value.update(
            state=fields[0],
            cpu_ticks=int(fields[11]) + int(fields[12]),
            start_ticks=int(fields[19]),
        )
        if value["cpu_ticks"] < 0 or value["start_ticks"] <= 0:
            raise ValueError("invalid process counters")
    except (OSError, ValueError, IndexError):
        return {"error": "PROC_STAT_UNAVAILABLE"}
    for name in ("io", "status"):
        try:
            for line in read_bounded(directory / name).splitlines():
                key, _, raw = line.partition(":")
                if key in {
                    "read_bytes",
                    "write_bytes",
                    "wchar",
                    "voluntary_ctxt_switches",
                }:
                    value[key] = int(raw.strip())
        except (OSError, ValueError):
            value[name + "_error"] = "UNAVAILABLE"
    try:
        value["wchan"] = read_bounded(directory / "wchan", 1024).strip()
    except (OSError, ValueError):
        value["wchan"] = None
    return value


def discover(
    root: Path, paths: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    incomplete = False
    try:
        for directory in root.iterdir():
            if not directory.name.isdigit():
                continue
            scanned += 1
            if scanned > MAX_SCAN_PROCESSES:
                incomplete = True
                break
            pid = int(directory.name)
            try:
                # A vanished process is normal; unreadable live membership is not
                # proof that it belongs to another workload.
                read_bounded(directory / "cgroup")
                if not workload_process(root, pid, paths):
                    continue
                try:
                    environment = dict(
                        item.split("=", 1)
                        for item in read_bounded(directory / "environ").split("\0")
                        if "=" in item
                    )
                except (OSError, ValueError):
                    environment = {}
                rank, world = None, None
                try:
                    rank = int(environment["RANK"])
                    if rank < 0:
                        rank = None
                except (KeyError, ValueError):
                    pass
                try:
                    world = int(environment["WORLD_SIZE"])
                    if not 0 < world <= 65536 or rank is not None and rank >= world:
                        rank, world = None, None
                except (KeyError, ValueError):
                    pass
                if len(rows) >= MAX_PROCESSES:
                    incomplete = True
                    break
                rows.append(
                    {
                        "pid": pid,
                        "rank": rank,
                        "world_size": world,
                        "progress_path": environment.get(PROGRESS_ENV),
                    }
                )
            except OSError as exc:
                incomplete = incomplete or exc.errno not in {errno.ENOENT, errno.ESRCH}
            except ValueError:
                incomplete = True
    except OSError:
        incomplete = True
    return sorted(
        rows, key=lambda row: (row["rank"] is None, row["rank"] or 0, row["pid"])
    ), {
        "scan_complete": not incomplete,
        "scanned_processes": min(scanned, MAX_SCAN_PROCESSES),
        "discovered_rank_processes": sum(row["rank"] is not None for row in rows),
        "discovered_processes": len(rows),
        "unranked_processes": sum(row["rank"] is None for row in rows),
        "max_rank_processes": MAX_PROCESSES,
    }


def application_sample(
    agent: Any, row: dict[str, Any], command: Any, start_ticks: int | None
) -> RankProgressSnapshot | None:
    if not row.get("progress_path") or start_ticks is None:
        return None
    try:
        path = FlightRecorderOperationsMixin.process_namespace_path_for_root(
            agent.proc_root, row["pid"], Path(row["progress_path"])
        )
        document = json.loads(bounded_read(lambda: read_bounded(path, 262144)))
        if not isinstance(document, dict):
            return None
        # This reader consumes communication counters only, including when
        # optional training-step/phase fields are absent or invalid.
        document.pop("step", None)
        document.pop("phase", None)
        value = RankProgressSnapshot.model_validate(document)
        namespace_pids = {row["pid"]}
        for line in read_bounded(
            agent.proc_root / str(row["pid"]) / "status"
        ).splitlines():
            if line.startswith("NSpid:"):
                namespace_pids.update(int(item) for item in line.split()[1:])
        if (
            value.job_id != command.parameters["job_id"]
            or value.attempt_id != command.parameters["attempt_id"]
            or row.get("rank") is not None
            and value.rank != row["rank"]
            or value.process_start_ticks != start_ticks
            or value.process_pid not in namespace_pids
            or row.get("world_size") is not None
            and value.world_size != row["world_size"]
            or not 0 <= (agent.now() - value.observed_at).total_seconds() <= 30
        ):
            return None
        return value
    except (OSError, ValueError, KeyError):
        return None


def rdma_sample(root: Path) -> dict[str, dict[str, Any]]:
    ports: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/ports/*"))[:128]:
        key = f"{path.parent.parent.name}/{path.name}"
        try:
            counters = path / "hw_counters"
            names = (
                ("rx_bytes", "tx_bytes")
                if (counters / "rx_bytes").exists() or (counters / "tx_bytes").exists()
                else ("recv_bytes", "send_bytes")
            )
            values = [int(read_bounded(counters / name, 128).strip()) for name in names]
            if any(value < 0 for value in values):
                raise ValueError("negative counter")
            ports[key] = {"rx": values[0], "tx": values[1], "source": list(names)}
        except (OSError, ValueError):
            ports[key] = {"error": "INCOMPLETE_RDMA_COUNTER_PAIR"}
    return ports
