"""Rank outliers are diagnostic leads, not proof of a collective's root cause."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from gpu_fault.execution.flight_semantics import flight_semantics
from gpu_fault.execution.flight_temporal import temporal_verdict
from gpu_fault.execution.process_diagnosis import process_diagnosis
from gpu_fault.flight_progress import is_p2p


def classify_hung_signals(
    signals: list[dict[str, Any]],
    *,
    undetermined_nodes: list[str],
    efa_zero_pending_at_by_node: dict[str, datetime] | None = None,
) -> dict[str, Any]:
    ranked = [
        signal
        for signal in signals
        if type(signal.get("rank")) is int and signal["rank"] >= 0
    ]
    counts = Counter(signal["rank"] for signal in ranked)
    duplicates = sorted(rank for rank, count in counts.items() if count > 1)
    ranked = [signal for signal in ranked if signal["rank"] not in duplicates]
    times = efa_zero_pending_at_by_node or {}
    nodes = {signal["rank"]: str(signal.get("node_id", "")) for signal in ranked}

    def order(ranks: list[int]) -> list[int]:
        return sorted(
            set(ranks),
            key=lambda rank: (
                times.get(
                    nodes.get(rank, ""), datetime.max.replace(tzinfo=timezone.utc)
                ),
                rank,
            ),
        )

    temporal = temporal_verdict(ranked)
    snapshot = _flight_verdict(ranked)
    verdict = dict(
        temporal
        if temporal is not None
        else snapshot
        if snapshot is not None
        else _verdict(
            "no communication root cause identified; inspect process observations and coverage",
        )
    )
    processes = process_diagnosis(signals, set(verdict.get("culprit_ranks", [])))
    processes["candidate_processes"].sort(
        key=lambda item: (
            times.get(item["node_id"], datetime.max.replace(tzinfo=timezone.utc)),
            item["node_id"],
            item["pid"],
        )
    )
    if (
        verdict["classification"] == "UNDETERMINED"
        and processes["candidate_process_count"]
    ):
        verdict.update(classification="WEAK", confidence="WEAK")
    candidates = order(verdict.get("culprit_ranks", []))
    verdict.update(
        {
            "culprit_ranks": candidates,
            "candidate_count": len(candidates),
            "control_ranks": sorted(
                set(verdict.get("control_ranks", [])) - set(candidates)
            )[:1],
            "undetermined_nodes": sorted(set(undetermined_nodes)),
            "duplicate_ranks": duplicates,
            "observed_rank_count": len(ranked),
            "root_cause_confirmed": False,
            "scope": "collected_processes_only",
            "evidence_branches": {
                "fr_temporal": temporal,
                "fr_snapshot": snapshot,
                "fr_semantics": flight_semantics(ranked),
                "process_observations": processes,
                "snapshot_authoritative": temporal is None,
            },
            "candidate_processes": processes["candidate_processes"],
            "candidate_process_count": processes["candidate_process_count"],
            "candidate_processes_truncated": processes["candidate_processes_truncated"],
            "flight_recorder_missing_count": sum(
                not (signal.get("flight_recorder") or {}).get("last_entry")
                and not (signal.get("flight_recorder") or {}).get("last_entries_by_pg")
                for signal in signals
            ),
        }
    )
    verdict["flight_recorder_unavailable"] = (
        verdict["flight_recorder_missing_count"] > len(signals) / 2
    )
    verdict["targeted_capture_eligible"] = bool(
        verdict["classification"] == "PLAUSIBLE"
        and 0 < len(candidates) <= 3
        and not undetermined_nodes
        and not duplicates
        and verdict.get("membership_verified", False)
        and verdict.get("temporal_verified", False)
    )
    verdict.setdefault("temporal_verified", False)
    return verdict


def _verdict(reason: str, **values: Any) -> dict[str, Any]:
    return {
        "classification": "UNDETERMINED",
        "confidence": "NONE",
        "culprit_ranks": [],
        "control_ranks": [],
        "reason": reason,
        **values,
    }


def _flight_verdict(ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    groups: dict[
        tuple[str, tuple[int, ...], str], list[tuple[int, dict[str, Any]]]
    ] = {}
    for signal in ranked:
        flight = signal.get("flight_recorder") or {}
        if flight.get("parse_error") or flight.get("process_groups_truncated"):
            continue
        entries = flight.get("last_entries_by_pg")
        if not isinstance(entries, list):
            entries = [flight.get("last_entry")]
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or is_p2p(entry):
                continue
            name = entry.get("pg_name")
            if (
                not name
                or name == ":"
                or type(entry.get("collective_seq_id")) is not int
            ):
                continue
            members = entry.get("pg_members")
            membership = (
                tuple(sorted(set(members)))
                if isinstance(members, list)
                and members
                and all(type(rank) is int and rank >= 0 for rank in members)
                else ()
            )
            key = (
                str(name),
                membership,
                str(entry.get("pg_membership_sha256") or "") if not membership else "",
            )
            if key in seen or membership and signal["rank"] not in membership:
                continue
            seen.add(key)
            groups.setdefault(key, []).append((signal["rank"], entry))
    findings = []
    for (name, members, membership_digest), rows in sorted(groups.items()):
        if len(rows) < 3:
            continue
        sequence, count = Counter(
            entry["collective_seq_id"] for _, entry in rows
        ).most_common(1)[0]
        if count <= len(rows) / 2:
            continue
        behind = {rank for rank, entry in rows if entry["collective_seq_id"] < sequence}
        at_mode = [
            (rank, entry)
            for rank, entry in rows
            if entry["collective_seq_id"] == sequence
        ]
        started = {rank for rank, entry in at_mode if _collective_started(entry)}
        if len(started) > len(at_mode) / 2:
            behind.update(rank for rank, _ in at_mode if rank not in started)
        if not behind:
            continue
        findings.append(
            {
                "pg_name": name,
                "mode_collective_seq_id": sequence,
                "culprit_ranks": sorted(behind),
                "control_ranks": sorted(
                    rank for rank, _ in at_mode if rank not in behind
                )[:1],
                "observed_members": sorted(rank for rank, _ in rows),
                "expected_members": list(members),
                "membership_sha256": membership_digest or None,
                "expected_member_count": len(members)
                or rows[0][1].get("pg_member_count"),
                "membership_verified": bool(
                    members and set(members) == {rank for rank, _ in rows}
                ),
            }
        )
    if not findings:
        return None
    candidates = sorted(
        {rank for finding in findings for rank in finding["culprit_ranks"]}
    )
    return _verdict(
        "single-snapshot collective progress outliers; causal direction is unproven",
        classification="PLAUSIBLE",
        confidence="PLAUSIBLE",
        culprit_ranks=candidates,
        control_ranks=[
            rank for finding in findings for rank in finding["control_ranks"]
        ],
        membership_verified=all(finding["membership_verified"] for finding in findings),
        process_groups=findings,
        pg_name=findings[0]["pg_name"] if len(findings) == 1 else None,
        mode_collective_seq_id=findings[0]["mode_collective_seq_id"]
        if len(findings) == 1
        else None,
    )


def _collective_started(entry: dict[str, Any]) -> bool:
    return bool(entry.get("time_discovered_started")) or str(
        entry.get("state") or ""
    ).lower() in {"started", "completed"}
