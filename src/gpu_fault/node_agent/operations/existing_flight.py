"""Preserve an error's existing FR before running any new active sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.operations.flight_evidence import bundle_raw_flights
from gpu_fault.node_agent.operations.hung_summary_io import discover
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    workload_process,
)


def preserve_existing(agent: Any, command: Any) -> list[dict[str, Any]]:
    paths = command.parameters.get("workload_cgroup_paths_by_node", {}).get(
        command.node_id, []
    )
    rows, _ = discover(agent.proc_root, paths)
    result = []
    for row in rows[:16]:
        start = process_start_ticks(agent.proc_root, row["pid"])
        environment = agent._process_environment(row["pid"])
        candidates = agent._flight_dump_candidates(
            environment, row["rank"], pid=row["pid"]
        )
        key = hashlib.sha256(command.command_id.encode()).hexdigest()[:24]
        captured = agent._finish_flight_recorder(
            {
                "status": "triggered",
                "dump_candidates": [str(path) for path in candidates],
                "evidence_path": str(
                    Path(agent.diagnostic_output_dir)
                    / f"nccl-flight-{key}-{row['pid']}-existing.dump"
                ),
            }
        )
        result.append(
            {
                "pid": row["pid"],
                "rank": row["rank"],
                "process_start_ticks": start,
                "identity_unchanged": start is not None
                and start == process_start_ticks(agent.proc_root, row["pid"])
                and workload_process(agent.proc_root, row["pid"], paths),
                "capture_mode": "EXISTING_FILE_NO_TRIGGER",
                "raw_evidence": captured.get("raw_evidence"),
                "status": captured.get("status"),
                "parse_error": captured.get("parse_error"),
            }
        )
    return result


def prepare_flight_manifest(
    agent: Any, command: Any, work_dir: Path, manifest: dict[str, Any]
) -> None:
    if command.parameters.get("preserve_existing_fr"):
        records = preserve_existing(agent, command)
        manifest["existing_flight_recorders"] = records
        raw = [
            {
                **item["raw_evidence"],
                "sample": "existing",
                "rank": item["rank"],
                "pid": item["pid"],
                "process_start_ticks": item["process_start_ticks"],
            }
            for item in records
            if item["identity_unchanged"]
            and (item.get("raw_evidence") or {}).get("status") == "PRESERVED"
        ]
        command = command.model_copy(
            update={
                "parameters": {
                    **command.parameters,
                    "raw_flights_by_node": {command.node_id: raw},
                }
            }
        )
    manifest["raw_flight_recorders"] = bundle_raw_flights(agent, command, work_dir)
