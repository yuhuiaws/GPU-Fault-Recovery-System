"""Coverage and diagnostic priorities from passive rank surveys."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any

from gpu_fault.communication_window import common_window
from gpu_fault.hung_participants import inventory
from gpu_fault.ras_survey import ras_survey
from gpu_fault.hung_summary_codec import (
    MAX_SURVEY_RAW_BYTES,
    SUMMARY_VERSION,
    decode_summary,
)


def survey_analysis(
    parameters: dict[str, Any], nodes: list[str], details: dict[str, Any]
) -> dict[str, Any]:
    accepted: dict[str, dict[str, Any]] = {}
    rejected = dict(details.get("node_failures") or {})
    processes: list[dict[str, Any]] = []
    starts: list[datetime] = []
    node_times: dict[str, datetime] = {}
    groups: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    decoded_bytes = 0
    for node in nodes:
        if decoded_bytes >= MAX_SURVEY_RAW_BYTES:
            rejected[node] = "DECODED_SURVEY_BUDGET"
            continue
        try:
            value = decode_summary((details.get("node_results") or {}).get(node, {}))
        except ValueError:
            rejected[node] = "INVALID_SUMMARY_ENVELOPE"
            continue
        decoded_bytes += len(json.dumps(value, separators=(",", ":")).encode())
        if decoded_bytes > MAX_SURVEY_RAW_BYTES:
            rejected[node] = "DECODED_SURVEY_BUDGET"
            continue
        if not isinstance(value, dict) or any(
            (
                value.get("summary_version") != SUMMARY_VERSION,
                value.get("snapshot_epoch") != parameters.get("snapshot_epoch"),
                value.get("job_id") != parameters.get("job_id"),
                value.get("attempt_id") != parameters.get("attempt_id"),
                value.get("node_id") != node,
                value.get("status") != "COLLECTED",
            )
        ):
            rejected.setdefault(node, "MISSING_OR_INCOMPATIBLE_SUMMARY")
            continue
        try:
            stamp = datetime.fromisoformat(value["sample_started_at"])
            end = datetime.fromisoformat(value["sample_finished_at"])
            if (
                stamp.tzinfo is None
                or end.tzinfo is None
                or not 0 <= (end - stamp).total_seconds() <= 30
            ):
                raise ValueError("invalid sample window")
        except (KeyError, TypeError, ValueError):
            rejected[node] = "INVALID_SAMPLE_TIME"
            continue
        starts.append(stamp)
        node_times[node] = stamp
        accepted[node] = value
        references = {}
        for local_key, raw_group in (value.get("groups") or {}).items():
            if not isinstance(raw_group, dict):
                continue
            key = str(raw_group.get("group_key", local_key))
            references[local_key] = key
            group = {
                name: item for name, item in raw_group.items() if name != "group_key"
            }
            if key in groups and groups[key] != group:
                conflicts.add(key)
            groups[key] = group
        for row in value.get("ranks") or []:
            if isinstance(row, dict):
                processes.append(
                    {
                        **row,
                        "node_id": node,
                        "communication": [
                            {
                                **item,
                                "group": references.get(
                                    item.get("group"), item.get("group")
                                ),
                            }
                            for item in row.get("communication", [])
                            if isinstance(item, dict)
                        ],
                    }
                )
    ranks = [
        row for row in processes if type(row.get("rank")) is int and row["rank"] >= 0
    ]
    verified = [
        row
        for row in processes
        if (row.get("process_identity") or {}).get("verified") is True
        and type((row.get("proc") or {}).get("cpu_ticks_delta")) is int
    ]
    identity_inventory = inventory(processes)
    valid = identity_inventory["valid"]
    duplicates = identity_inventory["duplicates"]
    expected = identity_inventory["expected"]
    present = {row["rank"] for row in valid}
    complete = bool(
        expected
        and present == set(range(expected))
        and not duplicates
        and not identity_inventory["unknown_rank_count"]
        and not identity_inventory["unknown_count"]
        and len(accepted) == len(nodes)
        and all(value.get("scan_complete") is True for value in accepted.values())
    )
    skew = (max(starts) - min(starts)).total_seconds() if starts else None
    coherent = skew is not None and skew <= 30
    group_coverage, candidates, candidate_groups = compare_groups(
        valid,
        groups,
        conflicts,
        {row["rank"]: node_times[row["node_id"]] for row in valid},
    )
    priorities = Counter({node: 0 for node in accepted})
    ras = ras_survey(accepted)
    priorities.update(dict.fromkeys(ras["ras_lead_nodes"], 50))
    leads = [
        {
            "node_id": row["node_id"],
            "pid": row.get("pid"),
            "rank": row.get("rank"),
            "reason": "D_STATE_IN_BOTH_LOCAL_SAMPLES",
            "confidence": "WEAK",
        }
        for row in verified
        if row.get("proc", {}).get("state") == "D"
        and row.get("proc", {}).get("state_before") == "D"
    ]
    for row in verified:
        if row.get("rank") in candidates:
            priorities[row["node_id"]] += 100
        if (
            row.get("proc", {}).get("state")
            == row.get("proc", {}).get("state_before")
            == "D"
        ):
            priorities[row["node_id"]] += 5
    related = sorted(
        {
            row["node_id"]
            for row in valid
            if any(
                item.get("group") in candidate_groups
                for item in row.get("communication", [])
            )
        }
    )
    return {
        "version": "hung-survey-analysis/v1",
        **ras,
        "expected_node_count": len(nodes),
        "responding_node_count": len(accepted),
        "node_coverage_fraction": len(accepted) / len(nodes) if nodes else 0,
        "missing_nodes": rejected,
        "observed_process_count": len(processes),
        "verified_process_count": len(verified),
        "unranked_process_count": len(processes) - len(ranks),
        "nodes_with_incomplete_scan": sorted(
            node
            for node, value in accepted.items()
            if value.get("scan_complete") is not True
        ),
        "valid_rank_count": len(valid),
        "expected_rank_count": expected,
        "rank_inventory_basis": "PROCESS_IDENTITY_GPU_CONTEXT_OR_BOUND_COMMUNICATION",
        "participant_process_count": len(identity_inventory["participants"]),
        "auxiliary_process_count": identity_inventory["auxiliary_count"],
        "unknown_participant_process_count": identity_inventory["unknown_count"],
        "participant_without_rank_count": identity_inventory["unknown_rank_count"],
        "participant_coverage_fraction": (
            min(1, len(identity_inventory["participants"]) / expected)
            if expected
            else None
        ),
        "rank_inventory_complete": complete,
        "duplicate_ranks": duplicates,
        "missing_ranks": sorted(set(range(expected)) - present) if expected else None,
        "rank_coverage_fraction": (
            len(present.intersection(range(expected))) / expected if expected else None
        ),
        "rank_node_map": {str(row["rank"]): row["node_id"] for row in valid},
        "communication_progress_rank_count": sum(
            bool(row.get("communication")) for row in valid
        ),
        "sample_start_skew_seconds": skew,
        "cross_node_time_comparable": coherent,
        "process_groups": group_coverage,
        **communication_coverage(group_coverage),
        "candidate_ranks": sorted(candidates),
        "candidate_nodes": sorted(
            {row["node_id"] for row in valid if row["rank"] in candidates}
        ),
        "related_nodes": related,
        "node_priority": sorted(accepted, key=lambda node: (-priorities[node], node)),
        "diagnostic_lead_nodes": sorted({row["node_id"] for row in leads}),
        "process_leads": leads[:256],
        "process_leads_truncated": len(leads) > 256,
        "diagnosis": "UNDETERMINED",
        "root_cause_confirmed": False,
        **device_coverage(accepted),
    }


def communication_coverage(groups: dict[str, Any]) -> dict[str, Any]:
    expected = sum(group["expected_members"] or 0 for group in groups.values())
    observed = sum(
        min(group["observed_members"], group["expected_members"] or 0)
        for group in groups.values()
        if group["time_comparable"]
    )
    return {
        "observed_communication_groups_complete": bool(groups)
        and all(group["complete"] for group in groups.values()),
        "communication_group_inventory_complete": False,
        "communication_coverage_basis": "DECLARED_OBSERVED_GROUPS_ONLY",
        "communication_time_series_coverage_fraction": observed / expected
        if expected
        else None,
    }


def device_coverage(accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "rdma_scope": "NODE_SHARED_COUNTERS",
        "gpu_observed_device_count": sum(
            len(value.get("gpu", {})) for value in accepted.values()
        ),
        "nodes_without_gpu_metrics": sorted(
            node for node, value in accepted.items() if not value.get("gpu")
        ),
        "rdma_valid_port_count": sum(
            pair.get("valid") is True
            for value in accepted.values()
            for pair in value.get("rdma", {}).values()
        ),
        "nodes_without_valid_rdma": sorted(
            node
            for node, value in accepted.items()
            if not any(
                pair.get("valid") is True for pair in value.get("rdma", {}).values()
            )
        ),
    }


def compare_groups(
    ranks: list[dict[str, Any]],
    descriptors: dict[str, Any],
    conflicts: set[str],
    sample_times: dict[int, datetime],
) -> tuple[dict[str, Any], set[int], set[str]]:
    rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for rank in ranks:
        for group in rank.get("communication") or []:
            if isinstance(group, dict) and isinstance(group.get("group"), str):
                before, after = group.get("before"), group.get("after")
                if all(
                    isinstance(value, list)
                    and len(value) == 3
                    and all(type(item) is int and item >= -1 for item in value)
                    for value in (before, after)
                ):
                    rows.setdefault(group["group"], []).append(
                        (rank["rank"], {**group, "node_id": rank["node_id"]})
                    )
    coverage: dict[str, Any] = {}
    candidates: set[int] = set()
    candidate_groups: set[str] = set()
    for key, descriptor in descriptors.items():
        values = rows.get(key, [])
        members = {rank for rank, _ in values}
        expected = (
            descriptor.get("member_count") if isinstance(descriptor, dict) else None
        )
        full = (
            key not in conflicts
            and type(expected) is int
            and len(members) == len(values) == expected
            and descriptor.get("membership_sha256")
            == hashlib.sha256(
                json.dumps(sorted(members), separators=(",", ":")).encode()
            ).hexdigest()
        )
        times = [sample_times[rank] for rank in members if rank in sample_times]
        skew = (max(times) - min(times)).total_seconds() if times else None
        window = common_window([item for _, item in values])
        coherent = window["time_comparable"]
        coverage[key] = {
            "expected_members": expected,
            "observed_members": len(members),
            "complete": full,
            "sample_start_skew_seconds": skew,
            "time_comparable": coherent,
            "member_coverage_fraction": min(1, len(members) / expected)
            if expected
            else None,
            **window,
        }
        if not full or not coherent or len(values) < 3:
            continue
        first, first_count = Counter(
            item["before"][0] for _, item in values
        ).most_common(1)[0]
        last, last_count = Counter(item["after"][0] for _, item in values).most_common(
            1
        )[0]
        if first_count <= len(values) / 2 or last_count <= len(values) / 2:
            continue
        behind = {
            rank
            for rank, item in values
            if 0 <= item["before"][0] < first
            and 0 <= item["after"][0] < last
            and item["after"] == item["before"]
        }
        candidates.update(behind)
        if behind:
            candidate_groups.add(key)
    return coverage, candidates, candidate_groups
