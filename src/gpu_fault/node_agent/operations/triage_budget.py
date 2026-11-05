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
``evidence_trimmed`` so the record says why a field is short. That note weighs
itself: its ``bytes_after`` is the size of the document as sent, the note
included, and only a document that fits with its note stops the ladder. It
used to record the size measured while the note still held the size before
it, one byte short whenever the number gained a digit, so a document a byte
or two over the budget could read ``within_budget: true``.

The ladder has a floor. A document still over the budget after every sample
list was cut used to go back as it was, and the worker replaced the whole of it
by the ``RESULT_SIZE_LIMIT`` stand-in with ``ranks: []`` -- every rank identity,
status, watermark and raw-dump reference lost in that last resort. The
``skeleton`` step reduces every rank row to exactly those fields (pid, rank,
process identity, role and status, the recorder's status with its dump path,
size and preserved-copy reference, the group identities with their ``progress``
watermarks and the newest operation's position, the series' status, timestamps
and counts, the sample series' counts and the stack's status) and leaves the
node-level fields whole: a rank of the live shape (two 64-operation dumps of a
16-member group, three stack and process samples) weighs 5.7 KB at the floor,
so 64 such ranks come to 369 KB against the 960 KiB budget and more than 170 of
them fit. Only a document whose skeleton alone is over the line reaches the
stand-in, and the stand-in names its ranks.
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
#: The result field that names what the ladder cut and what the result weighs.
TRIMMED_NOTE_KEY = "evidence_trimmed"
#: The floor of the ladder; the name ``evidence_trimmed.steps`` records for it.
SKELETON_STEP = "skeleton"
#: Row fields the skeleton keeps: identity, role and status. ``process_identity``
#: stays whole (verified, start ticks, namespace pids, container id).
SKELETON_ROW_KEYS = (
    "node_id",
    "pid",
    "rank",
    "local_rank",
    "world_size",
    "gpu_uuid",
    "gpu_uuids",
    "process_name",
    "process_identity",
    "status",
    "error",
    "role",
    "participant_basis",
    "communication_status",
)
#: Recorder fields the skeleton keeps: the status, where the dump is and how
#: big, the parse error, the counts and the preserved copy's reference (path,
#: digest, bytes) -- never an operation list.
SKELETON_RECORDER_KEYS = (
    "status",
    "pipe",
    "pipe_candidates",
    "dump_file",
    "dump_size_bytes",
    "dump_source",
    "dump_prefix_variable",
    "dump_candidates",
    "source_mtime_ns",
    "parse_error",
    "error",
    "trace_buffer",
    "entry_count",
    "process_groups_truncated",
    "unfinished_entry_count",
    "unfinished_entries_truncated",
    "raw_evidence",
)
#: An operation-group row (``last_entry``, ``last_entries_by_pg``, the series'
#: ``groups``) keeps its identity, the newest operation's position and state
#: and the ``progress`` watermarks; operation lists, shapes and dtypes go.
SKELETON_ENTRY_KEYS = (
    "pg_name",
    "pg_id",
    "pg_config_key",
    "pg_members",
    "pg_rank_order",
    "pg_member_count",
    "pg_membership_sha256",
    "pg_members_compacted",
    "collective_seq_id",
    "record_id",
    "op_id",
    "p2p_seq_id",
    "is_p2p",
    "state",
    "profiling_name",
    "retired",
    "time_created",
    "time_discovered_started",
    "time_discovered_completed",
    "timeout_ms",
    "progress",
    "semantic_entries_total",
)
SKELETON_SERIES_KEYS = (
    "version",
    "status",
    "first_observed_at",
    "process_start_ticks",
    "clock_uncertainty_seconds",
    "second_requested_at",
    "interval_seconds",
    "second_observed_at",
)
SKELETON_SNAPSHOT_KEYS = (
    "status",
    "parse_error",
    "raw_evidence",
    "process_groups_truncated",
)
#: What a sample series keeps besides its emptied lists and their counts.
SKELETON_SAMPLES_KEYS = (
    "version",
    "requested_samples",
    "scope",
    "mapping_verified",
    "shared_with_other_processes",
)
#: What the last stack sample and the last process window keep.
SKELETON_STATUS_KEYS = ("status", "error", "returncode")


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


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _pick(document: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: document[key] for key in keys if key in document}


def _series_totals(row: dict[str, Any]) -> dict[Any, int]:
    """Operation counts per group the series still knows, for a recorder group
    whose list became a ``semantic_entries_in`` pointer at the lossless step."""

    totals: dict[Any, int] = {}
    for group in _series_groups(row):
        entries = group.get("semantic_entries")
        total = group.get("semantic_entries_total")
        if type(total) is int:
            totals[group.get("pg_name")] = total
        elif isinstance(entries, list):
            totals[group.get("pg_name")] = len(entries)
    return totals


def _skeleton_entry(entry: dict[str, Any], totals: dict[Any, int]) -> dict[str, Any]:
    kept = _pick(entry, SKELETON_ENTRY_KEYS)
    entries = entry.get("semantic_entries")
    if isinstance(entries, list):
        kept.setdefault("semantic_entries_total", len(entries))
        kept["semantic_entries_trimmed"] = True
    elif entry.get("semantic_entries_in") or entry.get("semantic_entries_trimmed"):
        if "semantic_entries_total" not in kept and entry.get("pg_name") in totals:
            kept["semantic_entries_total"] = totals[entry["pg_name"]]
        kept["semantic_entries_trimmed"] = True
    return kept


def _skeleton_recorder(
    recorder: dict[str, Any], totals: dict[Any, int]
) -> dict[str, Any]:
    kept = _pick(recorder, SKELETON_RECORDER_KEYS)
    last = recorder.get("last_entry")
    if isinstance(last, dict):
        kept["last_entry"] = _skeleton_entry(last, totals)
    groups = recorder.get("last_entries_by_pg")
    if isinstance(groups, list):
        kept["last_entries_by_pg"] = [
            _skeleton_entry(group, totals)
            for group in groups
            if isinstance(group, dict)
        ]
    if "unfinished_entries" in recorder:
        kept["unfinished_entries"] = []
        if recorder["unfinished_entries"] or recorder.get("unfinished_entries_trimmed"):
            kept["unfinished_entries_trimmed"] = True
    return kept


def _skeleton_series(series: dict[str, Any]) -> dict[str, Any]:
    kept = _pick(series, SKELETON_SERIES_KEYS)
    for which in ("first", "second"):
        snapshot = series.get(which)
        if not isinstance(snapshot, dict):
            continue
        compact = _pick(snapshot, SKELETON_SNAPSHOT_KEYS)
        groups = snapshot.get("groups")
        if isinstance(groups, list):
            compact["groups"] = [
                _skeleton_entry(group, {})
                for group in groups
                if isinstance(group, dict)
            ]
        kept[which] = compact
    return kept


def _skeleton_samples(series: dict[str, Any]) -> dict[str, Any]:
    kept = _pick(series, SKELETON_SAMPLES_KEYS)
    samples = series.get("samples")
    trimmed = series.get("samples_trimmed")
    kept["samples"] = []
    kept["samples_trimmed"] = (trimmed if type(trimmed) is int else 0) + (
        len(samples) if isinstance(samples, list) else 0
    )
    if "windows" in series:
        windows = series["windows"]
        kept["windows"] = []
        kept["windows_trimmed"] = len(windows) if isinstance(windows, list) else 0
    return kept


def _skeleton_row(row: dict[str, Any]) -> dict[str, Any]:
    kept = _pick(row, SKELETON_ROW_KEYS)
    kept["skeleton"] = True
    context = row.get("comparison_context")
    if isinstance(context, dict):
        kept["comparison_context"] = {
            key: value for key, value in context.items() if _scalar(value)
        }
    recorder = row.get("flight_recorder")
    if isinstance(recorder, dict):
        kept["flight_recorder"] = _skeleton_recorder(recorder, _series_totals(row))
    series = row.get("flight_recorder_series")
    if isinstance(series, dict):
        kept["flight_recorder_series"] = _skeleton_series(series)
    for key in ("process_series", "stack_series", "gpu_series"):
        samples = row.get(key)
        if isinstance(samples, dict):
            kept[key] = _skeleton_samples(samples)
    for key in ("python_stack", "proc"):
        document = row.get(key)
        if isinstance(document, dict):
            kept[key] = _pick(document, SKELETON_STATUS_KEYS)
    return kept


def skeleton(ranks: list[dict[str, Any]]) -> int:
    """The floor: every rank row down to identity, status, watermarks and
    references, in place; a row already at the floor is left alone."""

    changed = 0
    for row in ranks:
        if row.get("skeleton") is True:
            continue
        compact = _skeleton_row(row)
        row.clear()
        row.update(compact)
        changed += 1
    return changed


#: Heaviest and most redundant first; process evidence last; the floor below.
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
    (SKELETON_STEP, skeleton),
)


def _settle(
    result: dict[str, Any], note: dict[str, Any], start: int, budget: int
) -> int:
    """Put ``note`` into ``result`` naming the size of the document carrying it.

    ``start`` is the size without the note. Writing a size into the note can
    lengthen the document it describes: the number may gain a digit, and
    ``false`` is a byte longer than ``true``. The document is measured again
    until the size written is the size measured; from the smaller start each
    pass can only lengthen it by those few bytes, so it ends after a pass or
    two. Returns that size.
    """

    note["bytes_after"] = start
    note["within_budget"] = start <= budget
    result[TRIMMED_NOTE_KEY] = note
    while True:
        size = encoded_size(result)
        if size == note["bytes_after"] and note["within_budget"] is (size <= budget):
            return size
        note["bytes_after"] = size
        note["within_budget"] = size <= budget


def bounded_triage_result(
    result: dict[str, Any], budget: int = TRIAGE_RESULT_BUDGET_BYTES
) -> dict[str, Any]:
    """``result`` within ``budget`` where the ladder can get it, annotated.

    The ladder is walked in place and stops at the first step after which the
    document fits with its ``evidence_trimmed`` note -- a step after which it
    fits only without the note does not stop it. Its last step is the skeleton
    floor, which keeps every rank's identity and verdict and nothing else. The
    note names the budget, the size before, every step that changed a rank
    with the size after it (without the note), and the size of the document
    returned (``bytes_after``, the note included) with whether that size fits
    (``within_budget``). A document still over the budget at the floor is
    returned as it is, for the worker's stand-in to name its size and ranks.
    """

    before = encoded_size(result)
    if before <= budget:
        return result
    ranks = [row for row in result.get("ranks") or [] if isinstance(row, dict)]
    steps: list[dict[str, Any]] = []
    note: dict[str, Any] = {
        "budget_bytes": budget,
        "bytes_before": before,
        "bytes_after": before,
        "within_budget": False,
        "steps": steps,
    }
    # A note an earlier pass wrote is replaced, never weighed with the steps.
    result.pop(TRIMMED_NOTE_KEY, None)
    size = before
    for name, step in LADDER:
        changed = step(ranks)
        if not changed:
            continue
        size = encoded_size(result)
        steps.append({"step": name, "ranks": changed, "bytes": size})
        # The note only adds bytes: a document over without it is over with it.
        if size <= budget:
            if _settle(result, note, size, budget) <= budget:
                return result
            # Over by its own note: the next step cuts further.
            del result[TRIMMED_NOTE_KEY]
    _settle(result, note, size, budget)
    return result
