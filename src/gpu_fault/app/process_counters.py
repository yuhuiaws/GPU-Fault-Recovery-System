"""Pod-coherent control-loop counters for a multi-process worker (F-L1).

The control-worker container runs ``uvicorn --workers 4``. Every worker
process builds its own ``ApplicationContext``, so a counter kept as an
attribute of the dispatcher, executor or merge service is process-local, and a
``/metrics`` scrape is answered by whichever process accepted that connection.
Observed live on 2026-09-08 (GF-REGIONAL-DESTR-018): the lease holder failed a
workflow for its lifetime and incremented ``lifetime_exceeded_total``; sixteen
scrapes of the same Pod returned seven distinct process fingerprints and the
counter read 0 on all of them. An alert written against such a counter fires
by luck.

Each process publishes its snapshot to ``<dir>/<pid>.json`` -- on every render
and, from a small background thread, every few seconds, so the process that
counted does not have to be the one that answers the scrape. A render returns
the sum over every process whose file belongs to a live PID; a dead PID's file
is removed on sight. The directory defaults to a per-Pod path under
``/dev/shm`` when ``POD_UID`` is set (the Deployment already injects it) and is
otherwise off, so a single-process run and the unit tests keep plain in-memory
counters. ``GPU_FAULT_PROCESS_COUNTERS_DIR`` overrides the path, or disables the
mechanism with ``off``.

Only monotonic counters are aggregated here. Timestamps and gauges rendered
alongside them stay process-local; summing those would be wrong and a maximum
is a different design.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event
from typing import Any

LOGGER = logging.getLogger(__name__)

ENVIRONMENT = "GPU_FAULT_PROCESS_COUNTERS_DIR"
DISABLED_TOKENS = frozenset({"off", "0", "false", "none", "disabled"})
DEFAULT_ROOT = Path("/dev/shm/gpu-fault-process-counters")
PUBLISH_INTERVAL_SECONDS = 5.0
ARCHIVE_WITHHELD_PREFIX = "archive_withheld:"

# (snapshot key, context attribute path, counter attribute). The key is what
# the metric contributor renders from; the path is where the process keeps it.
COUNTER_SOURCES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("dispatch_node_busy_timeouts_total", ("dispatcher",), "node_busy_timeouts_total"),
    (
        "dispatch_failure_handling_abandoned_total",
        ("dispatcher",),
        "failure_handling_abandoned_total",
    ),
    ("dispatch_plan_sync_misses_total", ("dispatcher",), "plan_sync_misses_total"),
    ("dispatch_internal_errors_total", ("dispatcher",), "internal_errors_total"),
    ("dispatch_deferred_total", ("dispatcher",), "deferred_total"),
    (
        "placement_holds_dissolved_total",
        ("dispatcher",),
        "placement_holds_dissolved_total",
    ),
    ("lifetime_exceeded_total", ("workflow_executor",), "lifetime_exceeded_total"),
    (
        "branch_escalation_budget_refusals_total",
        ("workflow_executor",),
        "branch_escalation_budget_refusals_total",
    ),
    (
        "absorbed_record_only_total",
        ("orchestrator", "_workflow_merger"),
        "absorbed_record_only_total",
    ),
    (
        "lifetime_record_only_total",
        ("orchestrator", "_workflow_merger"),
        "lifetime_record_only_total",
    ),
    (
        "withdrawn_record_only_total",
        ("orchestrator", "_workflow_merger"),
        "withdrawn_record_only_total",
    ),
    (
        "escalation_chain_terminated_total",
        ("orchestrator", "_escalation"),
        "escalation_chain_terminated_total",
    ),
    (
        "containment_refused_escalations_total",
        ("orchestrator", "_escalation"),
        "containment_refused_escalations_total",
    ),
    ("placement_holds_opened_total", ("orchestrator",), "placement_holds_opened_total"),
    ("placement_holds_failed_total", ("orchestrator",), "placement_holds_failed_total"),
    (
        "health_signal_clock_regressions_total",
        ("store",),
        "health_signal_clock_regressions_total",
    ),
    ("stale_event_link_repairs", ("store",), "stale_event_link_repairs"),
)


def counters_directory(environ: Mapping[str, str] | None = None) -> Path | None:
    """Where this process shares its counters, or ``None`` when off."""

    values = os.environ if environ is None else environ
    explicit = values.get(ENVIRONMENT, "").strip()
    if explicit:
        if explicit.lower() in DISABLED_TOKENS:
            return None
        return Path(explicit)
    pod_uid = values.get("POD_UID", "").strip()
    if not pod_uid:
        return None
    return DEFAULT_ROOT / pod_uid


def _resolve(root: Any, path: tuple[str, ...]) -> Any:
    value = root
    for name in path:
        if value is None:
            return None
        value = getattr(value, name, None)
    return value


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def control_loop_counter_snapshot(context: Any) -> dict[str, int]:
    """This process's control-loop counters, keyed for the metric renderer."""

    snapshot: dict[str, int] = {}
    for key, path, attribute in COUNTER_SOURCES:
        owner = _resolve(context, path)
        snapshot[key] = _count(getattr(owner, attribute, 0)) if owner is not None else 0
    archiver = getattr(context, "control_record_archiver", None)
    withheld = (
        getattr(archiver, "withheld_total", None) if archiver is not None else None
    )
    for reason, count in (withheld or {}).items():
        snapshot[f"{ARCHIVE_WITHHELD_PREFIX}{reason}"] = _count(count)
    return snapshot


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def publish(
    directory: Path, snapshot: Mapping[str, int], *, pid: int | None = None
) -> None:
    """Write this process's snapshot atomically; a partial file is never read."""

    directory.mkdir(parents=True, exist_ok=True)
    own = os.getpid() if pid is None else pid
    handle, temporary = tempfile.mkstemp(
        prefix=f".{own}-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(snapshot), stream, sort_keys=True)
        os.replace(temporary, directory / f"{own}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def aggregate(
    directory: Path, local: Mapping[str, int], *, pid: int | None = None
) -> dict[str, int]:
    """``local`` plus every other live process's published counters.

    Keys are the union of what any live process published, so a counter this
    process has never seen (a withheld reason another process recorded) still
    reaches the scrape. A file whose PID is gone is stale by definition and is
    removed rather than counted again after a worker restart.
    """

    own = os.getpid() if pid is None else pid
    totals: dict[str, int] = dict(local)
    if not directory.is_dir():
        return totals
    for path in sorted(directory.glob("*.json")):
        try:
            other = int(path.stem)
        except ValueError:
            continue
        if other == own:
            continue
        if not _alive(other):
            path.unlink(missing_ok=True)
            continue
        try:
            published = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Being rewritten, or never completed: the next scrape sees it.
            continue
        if not isinstance(published, dict):
            continue
        for key, value in published.items():
            totals[key] = totals.get(key, 0) + _count(value)
    return totals


def coherent_counters(
    local: Mapping[str, int],
    *,
    directory: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Publish ``local`` and return the Pod-wide totals.

    Falls back to ``local`` unchanged when sharing is off or the directory is
    unusable; a scrape must never fail because of its own bookkeeping.
    """

    target = counters_directory(environ) if directory is None else directory
    if target is None:
        return dict(local)
    try:
        publish(target, local)
        return aggregate(target, local)
    except OSError as error:
        LOGGER.warning(
            "process counters could not be shared through %s: %s", target, error
        )
        return dict(local)


def publish_forever(
    snapshot: Callable[[], Mapping[str, int]],
    stop: Event,
    *,
    directory: Path | None = None,
    interval: float = PUBLISH_INTERVAL_SECONDS,
) -> None:
    """Publish this process's counters until ``stop`` is set.

    Without this, a process publishes only when it answers a scrape, and the
    process that counted may never be asked. Runs in every worker process; it
    is idle when sharing is off.
    """

    target = counters_directory() if directory is None else directory
    if target is None:
        return
    while True:
        try:
            publish(target, snapshot())
        except Exception:  # noqa: BLE001 - a publisher must not die on one write
            LOGGER.exception("process counters could not be published to %s", target)
        if stop.wait(interval):
            return
