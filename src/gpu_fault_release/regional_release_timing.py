from __future__ import annotations

import time
from copy import deepcopy
from threading import Lock
from typing import Any

ROLLBACK_TIMING_SCHEMA_VERSION = 1


def _now(observed_at_epoch: float | None = None) -> float:
    return observed_at_epoch if observed_at_epoch is not None else time.time()


def initialize_rollback_timing(
    state: dict[str, Any],
    *,
    observed_at_epoch: float | None = None,
) -> dict[str, Any]:
    observed = _now(observed_at_epoch)
    raw = state.get("rollback_timing")
    timing = (
        deepcopy(raw)
        if isinstance(raw, dict)
        and raw.get("schema_version") == ROLLBACK_TIMING_SCHEMA_VERSION
        else {
            "schema_version": ROLLBACK_TIMING_SCHEMA_VERSION,
            "phases": {},
            "clusters": {},
        }
    )
    failure_detected = float(
        timing.get("failure_detected_at_epoch")
        or state.get("failure_detected_at_epoch")
        or state.get("failed_at_epoch")
        or state.get("updated_at_epoch")
        or observed
    )
    timing["failure_detected_at_epoch"] = failure_detected
    timing.setdefault("rollback_started_at_epoch", observed)
    timing["rollback_start_latency_seconds"] = max(
        0.0,
        float(timing["rollback_started_at_epoch"]) - failure_detected,
    )
    return timing


def start_timed_entry(
    timing: dict[str, Any],
    scope: str,
    name: str,
    *,
    observed_at_epoch: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    observed = _now(observed_at_epoch)
    entries = timing.setdefault(scope, {})
    entry = dict(entries.get(name) or {})
    entry.setdefault("started_at_epoch", observed)
    entry["updated_at_epoch"] = observed
    entry["status"] = "STARTED"
    if details:
        entry["details"] = {**dict(entry.get("details") or {}), **details}
    entries[name] = entry


def complete_timed_entry(
    timing: dict[str, Any],
    scope: str,
    name: str,
    *,
    status: str = "COMPLETED",
    observed_at_epoch: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    observed = _now(observed_at_epoch)
    entries = timing.setdefault(scope, {})
    entry = dict(entries.get(name) or {})
    started = float(entry.get("started_at_epoch") or observed)
    entry["started_at_epoch"] = started
    entry["updated_at_epoch"] = observed
    entry["completed_at_epoch"] = observed
    entry["duration_seconds"] = max(0.0, observed - started)
    entry["status"] = status
    if details:
        entry["details"] = {**dict(entry.get("details") or {}), **details}
    entries[name] = entry


def mark_rollback_safe(
    timing: dict[str, Any],
    *,
    observed_at_epoch: float | None = None,
) -> None:
    observed = _now(observed_at_epoch)
    timing["safe_at_epoch"] = observed
    timing["t_safe_seconds"] = max(
        0.0,
        observed - float(timing["failure_detected_at_epoch"]),
    )


def mark_rollback_complete(
    timing: dict[str, Any],
    *,
    observed_at_epoch: float | None = None,
) -> None:
    observed = _now(observed_at_epoch)
    timing["completed_at_epoch"] = observed
    timing["t_full_seconds"] = max(
        0.0,
        observed - float(timing["failure_detected_at_epoch"]),
    )


def _wave_key(wave: tuple[str, ...]) -> str:
    return ",".join(wave)


def record_rollback_wave_event(
    release: Any,
    *,
    cluster_id: str,
    wave: tuple[str, ...],
    event: str,
    observed_at_epoch: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    observed = _now(observed_at_epoch)
    lock = getattr(release, "_rollback_wave_timing_lock", None)
    if lock is None:
        lock = Lock()
        release._rollback_wave_timing_lock = lock
    with lock:
        all_clusters = getattr(release, "_rollback_wave_timings", None)
        if not isinstance(all_clusters, dict):
            all_clusters = {}
            release._rollback_wave_timings = all_clusters
        cluster = all_clusters.setdefault(cluster_id, {})
        key = _wave_key(wave)
        entry = dict(cluster.get(key) or {"nodes": list(wave)})
        entry.setdefault("started_at_epoch", observed)
        entry[f"{event}_at_epoch"] = observed
        entry["updated_at_epoch"] = observed
        if event == "failed":
            entry["status"] = "FAILED"
            entry["completed_at_epoch"] = observed
        elif event == "completed":
            entry["status"] = "COMPLETED"
            entry["completed_at_epoch"] = observed
        else:
            entry["status"] = "STARTED"
        if "completed_at_epoch" in entry:
            entry["duration_seconds"] = max(
                0.0,
                float(entry["completed_at_epoch"]) - float(entry["started_at_epoch"]),
            )
        if details:
            entry["details"] = {**dict(entry.get("details") or {}), **details}
        cluster[key] = entry


def merge_rollback_wave_timings(
    release: Any,
    timing: dict[str, Any],
) -> None:
    lock = getattr(release, "_rollback_wave_timing_lock", None)
    if lock is None:
        return
    with lock:
        waves = deepcopy(getattr(release, "_rollback_wave_timings", {}))
    for cluster_id, entries in waves.items():
        cluster = timing.setdefault("clusters", {}).setdefault(cluster_id, {})
        cluster["waves"] = {
            **dict(cluster.get("waves") or {}),
            **entries,
        }
