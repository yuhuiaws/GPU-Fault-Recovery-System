"""A hung triage result over the worker's result line is trimmed, never lost.

Live (GF-GENERAL-LIVE-011 ``fr_enabled``, HyperPod, 2026-10-25 23:05Z): the
triage found all eight ranks of each node, triggered two flight recorder dumps
per rank and read them, and the result weighed about 1.2 MB -- 64 operation
summaries per process group serialized four times a rank (``last_entry``,
``last_entries_by_pg``, the series' ``first`` and ``second``) on top of the
~290 KB of process and stack series an FR-less eight-rank triage weighs. The
worker's result line is bounded at 1 MiB (``OBSERVATION_RESULT_BUDGET_BYTES``),
so the whole document was replaced by ``{"status": "RESULT_SIZE_LIMIT",
"ranks": []}``, both nodes read ``ObservationIncomplete: RESULT_SIZE_LIMIT`` and
the decision saw no process at all. Nothing about the node mattered: any
eight-GPU node with the recorder on and more than ~55 entries in its ring
buffer overflows the same way.

The ladder below shrinks the heaviest, most redundant evidence first and stops
as soon as the document fits. Every rank keeps its identity, status, progress
watermarks and raw-dump reference at every step; what was removed is named in
``evidence_trimmed`` so the record says why a field is short.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from gpu_fault.node_agent.observation_limits import OBSERVATION_RESULT_BUDGET_BYTES

#: What a COLLECT_HUNG_TRIAGE result may weigh before the Agent wraps it: the
#: worker's result line less 64 KiB for the wire wrapper, the command id the
#: executor stamps on the result and the runtime's own request-trim notes.
TRIAGE_RESULT_BUDGET_BYTES = OBSERVATION_RESULT_BUDGET_BYTES - 64 * 1024
#: ``semantic_entries`` kept per process group while the FR evidence shrinks.
#: The reader sorts pending operations first, so the head of the list is what
#: the control plane's semantics matcher compares across ranks.
SEMANTIC_ENTRY_STEPS = (32, 16, 8)
TRIMMED_MARKER = "trimmed"


def encoded_size(document: Any) -> int:
    """Bytes of the compact JSON the worker and the control plane codec weigh."""

    return len(json.dumps(document, separators=(",", ":")).encode())


def _series_snapshot(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The series snapshot that carries the same dump as ``flight_recorder``.

    A PAIRED series' ``second`` is the recorder the row keeps; otherwise the
    row keeps the first dump and ``first`` is its snapshot.
    """

    series = row.get("flight_recorder_series")
    if not isinstance(series, dict):
        return None
    which = "second" if series.get("status") == "PAIRED" else "first"
    snapshot = series.get(which)
    return (which, snapshot) if isinstance(snapshot, dict) else None


def _groups(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    groups = (snapshot or {}).get("groups")
    return [group for group in groups or [] if isinstance(group, dict)]


def _recorder_groups(row: dict[str, Any]) -> list[dict[str, Any]]:
    recorder = row.get("flight_recorder")
    if not isinstance(recorder, dict):
        return []
    groups = recorder.get("last_entries_by_pg")
    return [group for group in groups or [] if isinstance(group, dict)]


def drop_semantic_entries_carried_by_the_series(ranks: list[dict[str, Any]]) -> int:
    """Lossless: a recorder's operation list the series already carries."""

    changed = 0
    for row in ranks:
        located = _series_snapshot(row)
        if located is None:
            continue
        which, snapshot = located
        carried = {
            group.get("pg_name"): group.get("semantic_entries")
            for group in _groups(snapshot)
            if "semantic_entries" in group
        }
        touched = False
        for group in _recorder_groups(row):
            entries = group.get("semantic_entries")
            if entries is None or entries != carried.get(group.get("pg_name")):
                continue
            del group["semantic_entries"]
            group["semantic_entries_in"] = f"flight_recorder_series.{which}"
            touched = True
        changed += touched
    return changed


def drop_unfinished_entries(ranks: list[dict[str, Any]]) -> int:
    """``unfinished_entries`` go; ``unfinished_entry_count`` stays."""

    changed = 0
    for row in ranks:
        recorder = row.get("flight_recorder")
        if isinstance(recorder, dict) and recorder.get("unfinished_entries"):
            recorder["unfinished_entries"] = []
            recorder["unfinished_entries_trimmed"] = True
            changed += 1
    return changed


def _series_groups(row: dict[str, Any]) -> list[dict[str, Any]]:
    series = row.get("flight_recorder_series")
    if not isinstance(series, dict):
        return []
    return [
        group for which in ("first", "second") for group in _groups(series.get(which))
    ]


def limit_semantic_entries(keep: int) -> Callable[[list[dict[str, Any]]], int]:
    """Keep the first ``keep`` operations of every list; pending ones lead."""

    def step(ranks: list[dict[str, Any]]) -> int:
        changed = 0
        for row in ranks:
            touched = False
            for group in [*_series_groups(row), *_recorder_groups(row)]:
                entries = group.get("semantic_entries")
                if not isinstance(entries, list) or len(entries) <= keep:
                    continue
                group.setdefault("semantic_entries_total", len(entries))
                group["semantic_entries"] = entries[:keep]
                group["semantic_entries_truncated"] = True
                touched = True
            changed += touched
        return changed

    return step


def drop_semantic_entries(ranks: list[dict[str, Any]]) -> int:
    """No operation lists at all; ``progress`` and the group identity stay."""

    changed = 0
    for row in ranks:
        touched = False
        for group in [*_series_groups(row), *_recorder_groups(row)]:
            entries = group.get("semantic_entries")
            if not isinstance(entries, list):
                continue
            group.setdefault("semantic_entries_total", len(entries))
            del group["semantic_entries"]
            group["semantic_entries_trimmed"] = True
            touched = True
        changed += touched
    return changed


def keep_last_sample(series_key: str) -> Callable[[list[dict[str, Any]]], int]:
    """Only the newest sample of a per-process series; its windows stay."""

    def step(ranks: list[dict[str, Any]]) -> int:
        changed = 0
        for row in ranks:
            series = row.get(series_key)
            if not isinstance(series, dict):
                continue
            samples = series.get("samples")
            if not isinstance(samples, list) or len(samples) <= 1:
                continue
            series["samples_trimmed"] = len(samples) - 1
            series["samples"] = samples[-1:]
            changed += 1
        return changed

    return step


#: Heaviest and most redundant first; process evidence last.
LADDER: tuple[tuple[str, Callable[[list[dict[str, Any]]], int]], ...] = (
    (
        "flight_recorder.semantic_entries_in_series",
        drop_semantic_entries_carried_by_the_series,
    ),
    ("flight_recorder.unfinished_entries", drop_unfinished_entries),
    *(
        (f"semantic_entries:{keep}", limit_semantic_entries(keep))
        for keep in SEMANTIC_ENTRY_STEPS
    ),
    ("semantic_entries", drop_semantic_entries),
    ("stack_series.samples:last", keep_last_sample("stack_series")),
    ("process_series.samples:last", keep_last_sample("process_series")),
)


def bounded_triage_result(
    result: dict[str, Any], budget: int = TRIAGE_RESULT_BUDGET_BYTES
) -> dict[str, Any]:
    """``result`` within ``budget`` where the ladder can get it, annotated.

    The ladder is walked in place and stops at the first step that brings the
    document under the budget. ``evidence_trimmed`` names the budget, the size
    before and after, every step that changed a rank and whether the document
    fits; a document still over the budget after the whole ladder is returned
    as it is, for the worker's stand-in to name its size.
    """

    before = encoded_size(result)
    if before <= budget:
        return result
    ranks = [row for row in result.get("ranks") or [] if isinstance(row, dict)]
    steps: list[dict[str, Any]] = []
    size = before
    for name, step in LADDER:
        changed = step(ranks)
        if not changed:
            continue
        size = encoded_size(result)
        steps.append({"step": name, "ranks": changed, "bytes": size})
        if size <= budget:
            break
    note: dict[str, Any] = {
        "budget_bytes": budget,
        "bytes_before": before,
        "bytes_after": size,
        "within_budget": size <= budget,
        "steps": steps,
    }
    result["evidence_trimmed"] = note
    # Measured with the note in place: that is the document the worker sends.
    after = encoded_size(result)
    note["bytes_after"] = after
    note["within_budget"] = after <= budget
    return result
