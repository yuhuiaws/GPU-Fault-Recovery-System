"""One capacity case run N times in sequence, read as median, min and max.

The single-cluster 1000-node case showed about +/-16 % run-to-run spread on one
build, so one run cannot tell a regression from noise. ``--repeat N`` on the
suite's ``all`` command runs the complete case N times -- each run registers,
loads and tears down on its own, with its own suite id and its own run
directory -- inside one repeat group directory::

    <artifact-root>/<label>/<release>/<stamp>/
        run-1/ ... run-N/      complete run directories
        repeat-summary.json

``repeat-summary.json`` lists every run (artifacts, run id, release, exit code,
job status) and, for each headline metric of ``summary.json``, the per-run
values in run order, the median, min, max and the relative spread
``(max - min) / median`` over the runs that completed. The group stops at the
first run that fails; the summary is rewritten after every run, so a group cut
short still says what it measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

SUMMARY_FILE = "repeat-summary.json"
#: Scalar headline metrics of ``summary.json``: name -> key path.
SCALAR_METRICS: dict[str, tuple[str, ...]] = {
    "requests": ("requests",),
    "throughput_req_s": ("throughput_req_s",),
    "fault_p50_ms": ("fault_p50_ms",),
    "fault_p99_ms": ("fault_p99_ms",),
    "queue_drain_seconds": ("queue_drain_seconds",),
    "queue_depth_at_load_stop": ("queue_depth_at_load_stop",),
    "aurora_capacity_max": ("aurora_capacity_max",),
    "aurora_cpu_max": ("aurora_cpu_max",),
    # The capacity each run started on: a run that began at the floor ACU
    # pays the scale-up inside its window.
    "aurora_capacity_at_step_start": (
        "aurora_measurement",
        "serverless_capacity_acu",
        "at_step_start",
    ),
    "aurora_capacity_at_window_start": (
        "aurora_measurement",
        "serverless_capacity_acu",
        "at_window_start",
    ),
    # The store during the start-burst storm (performance plan §11).
    "pg_buffer_content_share_first_two_minutes": (
        "pg_activity",
        "first_two_minutes",
        "buffer_content_share",
    ),
    "pg_peak_admission_inserts_first_two_minutes": (
        "pg_activity",
        "first_two_minutes",
        "peak_admission_inserts",
    ),
    "mean_in_flight_while_backlog": (
        "processor_timeseries",
        "consumers",
        "mean_in_flight_while_backlog",
    ),
}
#: Per ingress path (``summary.json.paths``); throughput is the path's count
#: over the run's ``wall_seconds_max``.
PATH_METRICS = ("count", "throughput_req_s", "p50_ms", "p95_ms", "p99_ms", "max_ms")


def run_label(args: argparse.Namespace) -> str:
    """The artifact label of one ``all`` run and of its repeat group."""

    return str(args.label or f"{args.case}-{args.clusters}c")


def refusal(args: argparse.Namespace) -> str | None:
    """Why ``--repeat`` cannot run with these options, or ``None``."""

    if args.repeat == 1:
        return None
    if args.repeat < 1:
        return "--repeat must be at least 1"
    if args.command != "all":
        return "--repeat is only valid for all"
    if args.suite_id:
        return "--repeat gives every run its own suite id: drop --suite-id"
    if args.keep_registration or args.no_purge:
        return (
            "--repeat needs every run's own teardown: drop --keep-registration "
            "and --no-purge"
        )
    return None


def _number(document: Any, path: tuple[str, ...]) -> float | None:
    value = document
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def spread(values: list[float]) -> dict[str, Any]:
    """Per-run values in run order, median, min, max and relative spread."""

    median = statistics.median(values)
    low, high = min(values), max(values)
    return {
        "runs": len(values),
        "values": [round(value, 3) for value in values],
        "median": round(median, 3),
        "min": round(low, 3),
        "max": round(high, 3),
        "spread": round((high - low) / median, 3) if median else None,
    }


def _path_metric(summary: dict[str, Any], kind: str, name: str) -> float | None:
    if name != "throughput_req_s":
        return _number(summary, ("paths", kind, name))
    count = _number(summary, ("paths", kind, "count"))
    wall = _number(summary, ("wall_seconds_max",))
    return count / wall if count is not None and wall else None


def metric_sections(
    summaries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``metrics`` and ``paths`` over the completed runs' summaries; a metric
    no run reported is left out."""

    metrics = {}
    for name, path in SCALAR_METRICS.items():
        values = [
            value for item in summaries if (value := _number(item, path)) is not None
        ]
        if values:
            metrics[name] = spread(values)
    kinds = sorted({kind for item in summaries for kind in item.get("paths") or {}})
    paths: dict[str, Any] = {}
    for kind in kinds:
        for name in PATH_METRICS:
            values = [
                value
                for item in summaries
                if (value := _path_metric(item, kind, name)) is not None
            ]
            if values:
                paths.setdefault(kind, {})[name] = spread(values)
    return metrics, paths


def _read(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def run_record(index: int, artifacts: Path, outcome: dict[str, Any]) -> dict[str, Any]:
    """One run's line of the group: where it is, who it was and how it ended."""

    run = _read(artifacts / "run.json")
    return {
        "index": index,
        "artifacts": str(artifacts),
        "run_id": run.get("suite_id"),
        "release_id": run.get("release_id"),
        **outcome,
        "status": _read(artifacts / "status.json").get("status"),
        "job_status": _read(artifacts / "summary.json").get("job_status"),
    }


def write_group(group: Path, repeat: int, runs: list[dict[str, Any]]) -> Path:
    completed = [item for item in runs if item.get("exit_code") == 0]
    metrics, paths = metric_sections(
        [_read(Path(item["artifacts"]) / "summary.json") for item in completed]
    )
    releases = sorted({str(item["release_id"]) for item in runs if item["release_id"]})
    document = {
        "repeat": repeat,
        "group": str(group),
        "complete": len(completed) == repeat,
        "completed_runs": len(completed),
        "runs": runs,
        "release_ids": releases,
        "same_release": len(releases) == 1,
        "metrics": metrics,
        "paths": paths,
    }
    path = group / SUMMARY_FILE
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return path


def run_repeated(
    repeat: int,
    group: Path,
    start: Callable[[Path], int],
    log: Callable[[str], None] = print,
) -> int:
    """Run ``start`` once per run directory of ``group``, in sequence; stop at
    the first non-zero exit (or error) and return it."""

    runs: list[dict[str, Any]] = []
    for index in range(1, repeat + 1):
        artifacts = group / f"run-{index}"
        artifacts.mkdir(parents=True, exist_ok=False)
        log(f"repeat run {index}/{repeat}: {artifacts}")
        try:
            code = start(artifacts)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            runs.append(
                run_record(index, artifacts, {"exit_code": None, "error": error})
            )
            log(f"repeat summary: {write_group(group, repeat, runs)}")
            raise
        runs.append(run_record(index, artifacts, {"exit_code": code}))
        log(f"repeat summary: {write_group(group, repeat, runs)}")
        if code != 0:
            log(f"repeat stopped: run {index} exited {code}")
            return code
    return 0
