"""A second fresh FR dump of the same processes, within one triage budget."""

from __future__ import annotations

import math
from typing import Any, Callable

from gpu_fault.node_agent.operations.flight_evidence import request_flight
from gpu_fault.node_agent.operations.process_identity import (
    process_start_ticks,
    stamp_process_identity,
    workload_process,
)


def triage_timeout(parameters: dict[str, Any]) -> float:
    value = float(parameters.get("triage_timeout_seconds", 20))
    if not math.isfinite(value):
        raise ValueError("triage timeout must be finite")
    return min(25.0, max(2.0, value))


def snapshot(recorder: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": recorder.get("status"),
        "parse_error": recorder.get("parse_error"),
        "raw_evidence": recorder.get("raw_evidence"),
        "process_groups_truncated": recorder.get("process_groups_truncated", False),
        "groups": [
            {
                key: entry[key]
                for key in (
                    "pg_name",
                    "pg_members",
                    "pg_rank_order",
                    "pg_member_count",
                    "pg_membership_sha256",
                    "progress",
                    "semantic_entries",
                )
                if key in entry
            }
            for entry in recorder.get("last_entries_by_pg", [])
        ],
    }


def collect_second_flight(
    agent: Any,
    command: Any,
    processes: list[dict[str, Any]],
    ranks: list[dict[str, Any]],
    identities: dict[int, dict[str, Any]],
    deadline: float,
    monotonic: Callable[[], float],
) -> float:
    interval = float(command.parameters.get("triage_fr_sample_interval_seconds", 2))
    if not math.isfinite(interval) or not 1 <= interval <= 10:
        raise ValueError("triage_fr_sample_interval_seconds must be in [1, 10]")
    first_at = agent.now().isoformat()
    first_time = monotonic()
    by_pid = {row["pid"]: row for row in processes}
    eligible = []
    for row in ranks:
        recorder = row.get("flight_recorder") or {}
        if recorder.get("status") != "dumped" or recorder.get("parse_error"):
            continue
        row["flight_recorder_series"] = {
            "version": "fr-series/v1",
            "status": "BUDGET_EXHAUSTED",
            "first_observed_at": first_at,
            "process_start_ticks": identities[row["pid"]]["start_time_ticks"],
            "clock_uncertainty_seconds": command.parameters.get(
                "clock_uncertainty_seconds"
            ),
            "first": snapshot(recorder),
        }
        eligible.append(row)
    if not eligible or first_time + interval + 1 >= deadline:
        return 0
    agent.sleep(interval)
    second_time = monotonic()
    if second_time - first_time < interval or second_time >= deadline:
        return 0
    paths = command.parameters.get("workload_cgroup_paths_by_node", {}).get(
        command.node_id, []
    )
    requests, second_rows = {}, []
    for row in eligible:
        pid = row["pid"]
        series = row["flight_recorder_series"]
        expected = identities[pid]["start_time_ticks"]
        if (
            type(expected) is not int
            or expected <= 0
            or process_start_ticks(agent.proc_root, pid) != expected
            or not workload_process(agent.proc_root, pid, paths)
        ):
            series["status"] = "PROCESS_IDENTITY_UNPROVEN"
            continue
        series["second_requested_at"] = agent.now().isoformat()
        series["interval_seconds"] = second_time - first_time
        requests[pid] = request_flight(agent, command, by_pid[pid], "second")
        second_rows.append(
            {
                "pid": pid,
                "flight_recorder": agent._finish_flight_recorder(requests[pid]),
            }
        )
    waited = agent._await_flight_dumps(requests, second_rows, deadline)
    for second in second_rows:
        row = next(item for item in eligible if item["pid"] == second["pid"])
        series = row["flight_recorder_series"]
        recorder = second["flight_recorder"]
        series["second"] = snapshot(recorder)
        series["second_observed_at"] = agent.now().isoformat()
        valid = (
            recorder.get("status") == "dumped"
            and not recorder.get("parse_error")
            and process_start_ticks(agent.proc_root, row["pid"])
            == series["process_start_ticks"]
            and workload_process(agent.proc_root, row["pid"], paths)
        )
        series["status"] = "PAIRED" if valid else "SECOND_UNAVAILABLE"
        if valid:
            row["flight_recorder"] = recorder
    return float(waited)


def finish_flight_series(
    agent: Any,
    command: Any,
    processes: list[dict[str, Any]],
    ranks: list[dict[str, Any]],
    identities: dict[int, dict[str, Any]],
    requests: dict[int, dict[str, Any]],
    started: float,
    budget: float,
    monotonic: Callable[[], float],
) -> float:
    waited = agent._await_flight_dumps(requests, ranks, started + budget / 2)
    waited += collect_second_flight(
        agent,
        command,
        processes,
        ranks,
        identities,
        started + budget,
        monotonic,
    )
    stamp_process_identity(
        agent.proc_root,
        ranks,
        identities,
        command.parameters.get("workload_cgroup_paths_by_node", {}).get(
            command.node_id, []
        ),
    )
    return float(waited)
