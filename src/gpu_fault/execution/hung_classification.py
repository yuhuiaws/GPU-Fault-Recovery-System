from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from statistics import median
from typing import Any, Callable


def classify_hung_signals(
    signals: list[dict[str, Any]],
    *,
    undetermined_nodes: list[str],
    efa_zero_pending_at_by_node: dict[str, datetime] | None = None,
) -> dict[str, Any]:
    ranked = [signal for signal in signals if isinstance(signal.get("rank"), int)]
    if not ranked:
        return _verdict(
            "UNDETERMINED",
            "NONE",
            undetermined_nodes,
            "no rank signals were collected",
        )
    efa_times = efa_zero_pending_at_by_node or {}
    order = _rank_order(ranked, efa_times)
    flight, undetermined_ranks = _flight_verdict(
        ranked,
        undetermined_nodes,
        order,
    )
    if flight is not None:
        return flight
    total = len(ranked)
    if undetermined_ranks and len(undetermined_ranks) <= max(1.0, total * 0.05):
        control = next(
            (
                signal["rank"]
                for signal in ranked
                if signal["rank"] not in undetermined_ranks
            ),
            None,
        )
        return {
            **_verdict(
                "PLAUSIBLE",
                "PLAUSIBLE",
                undetermined_nodes,
                "minority ranks produced no flight recorder dump",
            ),
            "culprit_ranks": order(undetermined_ranks),
            "control_ranks": [control] if control is not None else [],
        }
    stack = _stack_verdict(
        ranked,
        undetermined_nodes,
        order,
    )
    if stack is not None:
        return stack
    return _behavior_verdict(
        ranked,
        undetermined_nodes,
        order,
        efa_times,
        flight_recorder_unavailable=(len(undetermined_ranks) > total / 2),
    )


def _rank_order(
    ranked: list[dict[str, Any]],
    efa_times: dict[str, datetime],
) -> Callable[[set[int] | list[int]], list[int]]:
    node_by_rank = {
        signal["rank"]: str(signal.get("node_id") or "") for signal in ranked
    }

    def order(ranks: set[int] | list[int]) -> list[int]:
        return sorted(
            set(ranks),
            key=lambda rank: (
                efa_times.get(
                    node_by_rank.get(rank, ""),
                    datetime.max.replace(tzinfo=timezone.utc),
                ),
                rank,
            ),
        )

    return order


def _flight_verdict(
    ranked: list[dict[str, Any]],
    undetermined_nodes: list[str],
    order: Callable[[set[int] | list[int]], list[int]],
) -> tuple[dict[str, Any] | None, list[int]]:
    by_pg: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    undetermined = []
    for signal in ranked:
        last = (signal.get("flight_recorder") or {}).get("last_entry")
        if (
            not isinstance(last, dict)
            or not str(last.get("pg_name") or "")
            or not isinstance(last.get("collective_seq_id"), int)
        ):
            undetermined.append(signal["rank"])
            continue
        by_pg.setdefault(str(last["pg_name"]), []).append((signal["rank"], last))
    for pg_name, values in by_pg.items():
        verdict = _process_group(
            pg_name,
            values,
            undetermined_nodes,
            order,
        )
        if verdict is not None:
            return verdict, undetermined
    return None, undetermined


def _process_group(
    pg_name: str,
    values: list[tuple[int, dict[str, Any]]],
    undetermined_nodes: list[str],
    order: Callable[[set[int] | list[int]], list[int]],
) -> dict[str, Any] | None:
    mode = Counter(entry["collective_seq_id"] for _rank, entry in values).most_common(
        1
    )[0][0]
    behind = {rank for rank, entry in values if entry["collective_seq_id"] < mode}
    at_mode = [
        (rank, entry) for rank, entry in values if entry["collective_seq_id"] == mode
    ]
    not_started = {rank for rank, entry in at_mode if not _collective_started(entry)}
    lagging = behind | not_started if len(not_started) < len(at_mode) else behind
    if not lagging:
        return None
    if len(lagging) > max(1.0, len(values) * 0.05):
        return {
            **_verdict(
                "FABRIC_SUSPECTED",
                "NONE",
                undetermined_nodes,
                "more than five percent of ranks lag",
            ),
            "pg_name": pg_name,
            "mode_collective_seq_id": mode,
        }
    controls = [
        rank
        for rank, entry in values
        if rank not in lagging and entry["collective_seq_id"] == mode
    ][:1]
    return {
        **_verdict(
            "CONFIRMED",
            "CONFIRMED",
            undetermined_nodes,
            "minority ranks lag the collective sequence",
        ),
        "culprit_ranks": order(lagging),
        "control_ranks": controls,
        "pg_name": pg_name,
        "mode_collective_seq_id": mode,
    }


def _collective_started(entry: dict[str, Any]) -> bool:
    return bool(entry.get("time_discovered_started")) or str(
        entry.get("state") or ""
    ).lower() in {"started", "completed"}


def _stack_verdict(
    ranked: list[dict[str, Any]],
    undetermined_nodes: list[str],
    order: Callable[[set[int] | list[int]], list[int]],
) -> dict[str, Any] | None:
    rows = [
        (
            signal["rank"],
            (signal.get("python_stack") or {}).get("signature"),
            bool((signal.get("python_stack") or {}).get("collective_frames")),
        )
        for signal in ranked
        if (signal.get("python_stack") or {}).get("signature")
    ]
    if not rows:
        return None
    signature, count = Counter(row[1] for row in rows).most_common(1)[0]
    mode_rows = [row for row in rows if row[1] == signature]
    if count < 0.8 * len(ranked):
        return None
    if not any(row[2] for row in mode_rows):
        return _verdict(
            "NOT_COLLECTIVE_HANG",
            "NONE",
            undetermined_nodes,
            "majority stack is not collective communication",
        )
    outliers = [rank for rank, value, _ in rows if value != signature]
    if not 0 < len(outliers) <= 3:
        return None
    return {
        **_verdict(
            "PLAUSIBLE",
            "PLAUSIBLE",
            undetermined_nodes,
            "minority Python stack signature",
        ),
        "culprit_ranks": order(outliers),
        "control_ranks": [mode_rows[0][0]],
    }


def _behavior_verdict(
    ranked: list[dict[str, Any]],
    undetermined_nodes: list[str],
    order: Callable[[set[int] | list[int]], list[int]],
    efa_times: dict[str, datetime],
    *,
    flight_recorder_unavailable: bool,
) -> dict[str, Any]:
    cpu_values = [
        int((signal.get("proc") or {}).get("cpu_ticks_delta", 0)) for signal in ranked
    ]
    gpu_values = [
        float((signal.get("gpu") or {}).get("utilization_gpu_percent", 0))
        for signal in ranked
    ]
    cpu_mode = median(cpu_values) if cpu_values else 0
    gpu_mode = median(gpu_values) if gpu_values else 0
    scores = [
        (_behavior_score(signal, cpu_mode, gpu_mode), signal["rank"])
        for signal in ranked
    ]
    baseline = median([score for score, _rank in scores])
    weak = [(score, rank) for score, rank in scores if score > 0 and score > baseline]
    if weak:
        node_by_rank = {
            signal["rank"]: str(signal.get("node_id") or "") for signal in ranked
        }
        weak.sort(
            key=lambda item: (
                -item[0],
                efa_times.get(
                    node_by_rank.get(item[1], ""),
                    datetime.max.replace(tzinfo=timezone.utc),
                ),
                item[1],
            )
        )
        culprits = order([rank for _score, rank in weak[:3]])
        control = next(
            (signal["rank"] for signal in ranked if signal["rank"] not in culprits),
            None,
        )
        reason = "CPU/GPU behavior differs from rank majority"
        if flight_recorder_unavailable:
            reason += " without flight recorder dumps"
        result = {
            **_verdict("WEAK", "WEAK", undetermined_nodes, reason),
            "culprit_ranks": culprits,
            "control_ranks": [control] if control is not None else [],
        }
        if flight_recorder_unavailable:
            result["flight_recorder_unavailable"] = True
        return result
    if flight_recorder_unavailable:
        return {
            **_verdict(
                "UNDETERMINED",
                "NONE",
                undetermined_nodes,
                "most ranks produced no flight recorder dump and "
                "no rank stood out from the CPU/GPU baseline",
            ),
            "flight_recorder_unavailable": True,
        }
    return _verdict(
        "FABRIC_SUSPECTED",
        "NONE",
        undetermined_nodes,
        "no isolated rank culprit was identified",
    )


def _behavior_score(
    signal: dict[str, Any],
    cpu_mode: float,
    gpu_mode: float,
) -> int:
    proc = signal.get("proc") or {}
    gpu = signal.get("gpu") or {}
    states = proc.get("thread_states") or {}
    score = 0
    if cpu_mode > 0 and proc.get("cpu_ticks_delta", 0) < cpu_mode * 0.2:
        score += 1
    if int(states.get("D", 0)) > 0:
        score += 1
    if proc.get("voluntary_ctxt_switches_delta") == 0 and proc.get("wchan_unchanged"):
        score += 1
    if gpu_mode >= 50 and float(gpu.get("utilization_gpu_percent", 0)) == 0:
        score += 1
    return score


def _verdict(
    classification: str,
    confidence: str,
    undetermined_nodes: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "classification": classification,
        "confidence": confidence,
        "culprit_ranks": [],
        "control_ranks": [],
        "undetermined_nodes": sorted(undetermined_nodes),
        "reason": reason,
    }
