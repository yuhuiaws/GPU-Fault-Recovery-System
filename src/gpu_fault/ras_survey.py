"""RAS-local rank IDs do not replace the task's global rank or PID generation."""

from __future__ import annotations

from typing import Any


def ras_survey(accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aliases: dict[str, set[str]] = {}
    for node, value in accepted.items():
        for alias in value.get("node_addresses", []):
            aliases.setdefault(alias, set()).add(node)
    groups, leads = [], []
    for value in accepted.values():
        report = value.get("cached_ras") or {}
        if report.get("status") != "PAIRED":
            continue
        groups.extend(report.get("groups", []))
        for group in report.get("groups", []):
            for candidate in group.get("candidate_members", []):
                nodes = aliases.get(candidate["host"], set())
                if len(nodes) != 1:
                    continue
                node = next(iter(nodes))
                matches = [
                    row
                    for row in accepted[node].get("ranks", [])
                    if row.get("process_identity", {}).get("verified") is True
                    and candidate["pid"]
                    in row["process_identity"].get("namespace_pids", [row["pid"]])
                    and row.get("gpu_uuids")
                ]
                if len(matches) != 1:
                    continue
                row = matches[0]
                leads.append(
                    {
                        "node_id": node,
                        "pid": row["pid"],
                        "rank": row.get("rank"),
                        "start_time_ticks": row["process_identity"]["start_time_ticks"],
                        "communicator_identity": group["identity"],
                        "communicator_rank": candidate["communicator_rank"],
                        "reason": candidate["reason"],
                        "confidence": "WEAK",
                    }
                )
    return {
        "ras_process_groups": groups[:256],
        "ras_process_leads": leads[:256],
        "ras_lead_nodes": sorted({row["node_id"] for row in leads}),
        "ras_completion_progress_available": False,
    }
