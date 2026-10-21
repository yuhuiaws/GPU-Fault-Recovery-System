"""Identity-bound, attributed delivery evidence for the COLLECT-002 power window.

A confirmed DCGM batch carries every GPU of the node and one batch-level
``candidate-confirmed``: it does not say whose candidate confirmed. The case
credits a batch to the loaded GPU only when no other GPU in it carries a
candidate of its own (:func:`delivery_evidence`); the runner reads the Store's
latest values before the cap (:func:`power_reference`) so a counter can be
proven unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from gpu_fault.gpu_metrics import GpuMetricsThresholds
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic


class DeliveryEvidence(TypedDict):
    delivered_at: str | None
    record_ids: list[str]
    errors: list[str]
    attribution: dict[str, Any] | None


_POWER_METRICS = frozenset(
    {"power_limit_w", "power_usage_w", "gpu_utilization_percent"}
)
#: The fields whose candidate the DCGM edge filter raises from the value alone
#: (``DcgmMetricsCollector._candidate_keys``): a duty-cycle counter breaches on
#: a tick whose advance over at least one interval reaches the duty-cycle
#: threshold, a pending/failure flag or the last XID while it is nonzero, and
#: a key confirms only on a tick it breaches.
_DUTY_SUFFIX = "_violation_total_us"
_FLAGS = frozenset(
    {
        "retired_pages_pending",
        "row_remap_failure",
        "row_remap_pending",
        "xid_last_error",
    }
)
#: The collector's duty-cycle threshold in ``collector.env``, and its default.
DUTY_CYCLE_KEY = "GPU_FAULT_DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD"
DUTY_CYCLE_DEFAULT = 0.05
#: Read-only, in a CPU-cluster Pod: the Store's latest value of those fields for
#: every GPU of the node. Every ingested batch updates it, the health summaries
#: too, so read after the quiet delivery it holds the values from before the cap.
POWER_REFERENCE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, node_id, suffix, *flags = sys.argv[1:]
store = ApplicationContext.from_environment().store
samples = []
for item in store.list_gpu_metrics_latest(cluster_id, node_id):
    value = item.model_dump(mode="json")
    sample = value["sample"]
    name = sample["canonical_name"]
    if name.endswith(suffix) or name in flags:
        samples.append({
            "observed_at": value["observed_at"],
            "source": value["source"],
            **{key: sample.get(key) for key in ("canonical_name", "gpu_uuid")},
            **{key: sample.get(key) for key in ("gpu_index", "value")},
        })
print(json.dumps({"samples": samples}, sort_keys=True))
"""
_CANDIDATE_REASONS = frozenset({"candidate-confirmed", "threshold"})
_OTHER_REASONS = frozenset(
    {
        "initial-baseline",
        "health-summary",
        "candidate-recovered",
        "counter-increased",
        "counter-reset",
        "device-lost",
        "xid-changed",
        "filter-disabled",
    }
)


def _timestamp(value: object, field: str) -> datetime:
    try:
        if isinstance(value, datetime):
            stamp = value
        elif isinstance(value, str):
            stamp = datetime.fromisoformat(value)
        else:
            raise ValueError
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise ValueError(f"{field} must be a valid timezone-aware timestamp") from None


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _load_target(
    load_start: Mapping[str, Any],
) -> tuple[str, str, datetime]:
    uuid = load_start.get("gpu_uuid")
    index = load_start.get("gpu_index")
    if (
        not isinstance(uuid, str)
        or not uuid.strip()
        or type(index) is not int
        or index < 0
    ):
        raise ValueError("load_start must identify the loaded GPU UUID and index")
    return (
        uuid,
        str(index),
        _timestamp(load_start.get("started_at"), "load_start.started_at"),
    )


def _power_samples(
    payload: Mapping[str, Any], *, gpu_uuid: str, gpu_index: str
) -> dict[str, float]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("candidate samples must be a nonempty list")
    values: dict[str, float] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("candidate contains a malformed sample")
        name = sample.get("canonical_name")
        if not isinstance(name, str) or not name:
            raise ValueError("candidate sample has no canonical_name")
        if name not in _POWER_METRICS:
            continue
        uuid, index = sample.get("gpu_uuid"), sample.get("gpu_index")
        if (
            not isinstance(uuid, str)
            or not uuid.strip()
            or not isinstance(index, str)
            or not index.isascii()
            or not index.isdecimal()
        ):
            raise ValueError("candidate power sample has no valid GPU UUID/index")
        if uuid != gpu_uuid and index != gpu_index:
            continue
        if uuid != gpu_uuid or index != gpu_index:
            raise ValueError("candidate power sample GPU UUID/index disagree")
        if name in values:
            raise ValueError("candidate repeats a loaded GPU power metric")
        values[name] = _number(sample.get("value"), f"candidate {name}")
    if values and values.keys() != _POWER_METRICS:
        raise ValueError("candidate lacks a required loaded GPU power metric")
    if values and (
        values["power_limit_w"] <= 0
        or values["power_usage_w"] < 0
        or not 0 <= values["gpu_utilization_percent"] <= 100
    ):
        raise ValueError("candidate loaded GPU power metrics are outside valid ranges")
    return values


def _candidate(
    record: Mapping[str, Any],
    *,
    cluster_id: str,
    node_id: str,
    gpu_uuid: str,
    gpu_index: str,
    power_limit: float,
    started_at: datetime,
    observed_until: datetime,
) -> tuple[str, datetime, tuple[datetime, datetime, float, float, float]] | None:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("GPU evidence has no stored payload")
    # Consistently foreign identities are unrelated; partial/mixed identities
    # must reach validation instead of hiding a malformed relevant candidate.
    for key, expected in (("cluster_id", cluster_id), ("node_id", node_id)):
        identity = record.get(key)
        if (
            isinstance(identity, str)
            and identity
            and identity == payload.get(key)
            and identity != expected
        ):
            return None
    batch_id = payload.get("batch_id")
    if (
        payload.get("source") == "NVIDIA_SMI"
        and isinstance(batch_id, str)
        and batch_id.startswith("nvidia-smi-")
        and record.get("record_id") == f"gpu-metrics/{batch_id}"
    ):
        return None
    reasons = payload.get("edge_filter_reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) for reason in reasons
    ):
        raise ValueError("DCGM edge_filter_reasons must be a list of strings")
    if set(reasons) - (_CANDIDATE_REASONS | _OTHER_REASONS):
        raise ValueError("DCGM evidence has an unrecognized edge_filter_reason")
    if not set(reasons) & _CANDIDATE_REASONS:
        return None
    if (
        record.get("kind") != "GPU_METRICS"
        or payload.get("source") != "DCGM_EXPORTER"
        or record.get("cluster_id") != cluster_id
        or payload.get("cluster_id") != cluster_id
        or record.get("node_id") != node_id
        or payload.get("node_id") != node_id
    ):
        raise ValueError("DCGM candidate source/cluster/node identity differs")
    prefix = f"dcgm-{node_id}-"
    if (
        not isinstance(batch_id, str)
        or not batch_id.startswith(prefix)
        or not batch_id[len(prefix) :].isascii()
        or not batch_id[len(prefix) :].isdecimal()
        or record.get("record_id") != f"gpu-metrics/{batch_id}"
    ):
        raise ValueError("DCGM candidate batch_id/record_id binding is invalid")
    if payload.get("collection_errors") != []:
        raise ValueError("DCGM candidate collection_errors are missing or nonempty")
    observed = _timestamp(payload.get("observed_at"), "payload.observed_at")
    collected = _timestamp(payload.get("collected_at"), "payload.collected_at")
    received = _timestamp(payload.get("ingested_at"), "payload.ingested_at")
    stored_observed = _timestamp(record.get("observed_at"), "record.observed_at")
    stored_received = _timestamp(record.get("ingested_at"), "record.ingested_at")
    if stored_observed != observed:
        raise ValueError("DCGM candidate record/payload observed_at differ")
    if (
        not started_at
        <= observed
        <= collected
        <= received
        <= stored_received
        <= observed_until
    ):
        raise ValueError(
            "DCGM candidate timestamps are unordered or outside the load window"
        )
    values = _power_samples(payload, gpu_uuid=gpu_uuid, gpu_index=gpu_index)
    thresholds = GpuMetricsThresholds()
    if (
        not values
        or values["power_limit_w"] != power_limit
        or values["power_usage_w"] < thresholds.power_limit_ratio * power_limit
        or values["gpu_utilization_percent"]
        < thresholds.power_correlation_min_utilization_percent
    ):
        return None
    return (
        f"gpu-metrics/{batch_id}",
        received,
        (
            observed,
            collected,
            values["power_limit_w"],
            values["power_usage_w"],
            values["gpu_utilization_percent"],
        ),
    )


def _watched(sample: object) -> tuple[str, str, float] | None:
    """``(GPU UUID, field, value)`` of a finite duty-cycle counter or flag sample."""

    if not isinstance(sample, Mapping):
        return None
    name, uuid = sample.get("canonical_name"), sample.get("gpu_uuid")
    value = _finite(sample.get("value"))
    if (
        not isinstance(name, str)
        or not (name.endswith(_DUTY_SUFFIX) or name in _FLAGS)
        or not isinstance(uuid, str)
        or not uuid
        or value is None
    ):
        return None
    return uuid, name, value


def _readings(
    records: list[object], reference: list[object], *, cluster_id: str, node_id: str
) -> dict[tuple[str, str], list[tuple[datetime, float]]]:
    """Every value of a watched field per ``(GPU UUID, field)`` and when it was
    sampled: the Store's latest values before the cap and this node's DCGM
    batches in the window, accepted or not."""

    sources: list[tuple[object, list[object]]] = [
        (item.get("observed_at"), [item])
        for item in reference
        if isinstance(item, Mapping) and item.get("source") == "DCGM_EXPORTER"
    ]
    for record in records:
        payload = record.get("payload") if isinstance(record, Mapping) else None
        if (
            isinstance(payload, Mapping)
            and payload.get("source") == "DCGM_EXPORTER"
            and payload.get("cluster_id") == cluster_id
            and payload.get("node_id") == node_id
            and isinstance(payload.get("samples"), list)
        ):
            sources.append((payload.get("observed_at"), payload["samples"]))
    readings: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
    for stamp, samples in sources:
        try:
            observed = _timestamp(stamp, "reading")
        except ValueError:
            continue
        for sample in samples:
            watched = _watched(sample)
            if watched is not None:
                uuid, name, value = watched
                readings.setdefault((uuid, name), []).append((observed, value))
    return readings


def _other_candidates(
    payload: Mapping[str, Any],
    observed: datetime,
    gpu_uuid: str,
    readings: Mapping[tuple[str, str], list[tuple[datetime, float]]],
    floor: float,
) -> list[dict[str, Any]]:
    """The other GPUs of a confirmed batch that carry a candidate of their own.

    A key confirms only on a tick it breaches. A duty-cycle counter breaches
    when its advance since the previous tick, over the time since that value
    was first seen (at least one interval), reaches the threshold. The counters
    only grow, so that advance is at most the one since any earlier reading:
    one below ``floor`` (threshold x interval, in microseconds) cannot have
    breached -- a counter creeping by microseconds on a healthy GPU stays out,
    a capped idle GPU's fifteen seconds per tick does not. A pending/failure
    flag or XID that was set and is unchanged had its own confirmation before
    the window. Another GPU is out of the delivery when its draw is not
    correlated with its limit and no watched field breached; a nonzero field
    without an earlier reading, or one that fell (a reset), cannot be bounded
    and names its GPU. The batch's samples are mappings (``_power_samples``).
    """

    thresholds = GpuMetricsThresholds()
    indexes: dict[str, object] = {}
    gpus: dict[str, dict[str, object]] = {}
    for sample in payload["samples"]:
        uuid = sample.get("gpu_uuid")
        if isinstance(uuid, str) and uuid and uuid != gpu_uuid:
            indexes.setdefault(uuid, sample.get("gpu_index"))
            name = str(sample.get("canonical_name"))
            gpus.setdefault(uuid, {})[name] = sample.get("value")
    named = []
    for uuid, gpu in gpus.items():
        usage, limit, utilization = (
            _finite(gpu.get(name))
            for name in ("power_usage_w", "power_limit_w", "gpu_utilization_percent")
        )
        reasons = []
        if (
            usage is not None
            and limit is not None
            and limit > 0
            and utilization is not None
            and usage >= thresholds.power_limit_ratio * limit
            and utilization >= thresholds.power_correlation_min_utilization_percent
        ):
            reasons.append(
                f"power_limit_correlation: {usage:g} W at a {limit:g} W limit, "
                f"{utilization:g}% utilization"
            )
        for name, value in gpu.items():
            watched = _watched(
                {"canonical_name": name, "gpu_uuid": uuid, "value": value}
            )
            if watched is None or not watched[2]:
                continue
            earlier = max(
                (item for item in readings.get((uuid, name), []) if item[0] < observed),
                default=None,
            )
            if earlier is None:
                reasons.append(f"{name} {watched[2]!r} with no earlier reading")
                continue
            advance = watched[2] - earlier[1]
            if advance < 0:
                reasons.append(f"{name} fell from {earlier[1]!r} to {watched[2]!r}")
            elif advance and (name in _FLAGS or advance >= floor):
                reasons.append(f"{name} {earlier[1]!r} -> {watched[2]!r}")
        if reasons:
            named.append(
                {
                    "gpu_index": indexes[uuid],
                    "gpu_uuid": uuid,
                    "power_limit_w": limit,
                    "candidates": reasons,
                }
            )
    return named


def delivery_evidence(
    records: Iterable[object],
    *,
    cluster_id: str,
    node_id: str,
    load_start: dict[str, Any],
    expected_power_limit_w: float,
    observed_until: datetime,
    reference: Mapping[str, Any] | None = None,
) -> DeliveryEvidence:
    """Select the first raw DCGM power candidate credited to the loaded GPU.

    Pass the validated ``injection["load_start"]``, the recorded cap for that GPU,
    the end of the captured load window and :func:`power_reference`'s answer. No
    clock or I/O is consulted. A batch qualifies when it is the loaded GPU's
    correlated power candidate (never a status row); it is credited when no
    other GPU in it carries a candidate of its own (:func:`_other_candidates`),
    and otherwise listed under ``attribution["ignored"]`` with those GPUs: the
    batch's ``candidate-confirmed`` could be theirs. ``delivered_at`` is the
    earliest credited payload ingestion time, normalized to UTC; ``record_ids``
    are the unique credited IDs in delivery order. Any ``errors`` veto
    acceptance, even when another record supplies a valid timestamp.
    """
    result: DeliveryEvidence = {
        "delivered_at": None,
        "record_ids": [],
        "errors": [],
        "attribution": None,
    }
    try:
        if (
            not isinstance(cluster_id, str)
            or not cluster_id.strip()
            or not isinstance(node_id, str)
            or not node_id.strip()
        ):
            raise ValueError("cluster_id and node_id must be nonempty")
        if not isinstance(load_start, Mapping):
            raise ValueError("load_start must be a GPU load receipt object")
        gpu_uuid, gpu_index, started = _load_target(load_start)
        limit = _number(expected_power_limit_w, "expected_power_limit_w")
        if limit <= 0:
            raise ValueError("expected_power_limit_w must be positive")
        until = _timestamp(observed_until, "observed_until")
        if until < started:
            raise ValueError("observed_until precedes load_start")
    except ValueError as exc:
        result["errors"].append(str(exc))
        return result
    records = list(records)
    accepted: dict[str, datetime] = {}
    signatures: dict[str, tuple[datetime, datetime, float, float, float]] = {}
    batches: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        try:
            if not isinstance(record, Mapping):
                raise ValueError("raw evidence record must be an object")
            payload = record.get("payload")
            record_id = record.get("record_id")
            dcgm_record_id = isinstance(record_id, str) and record_id.startswith(
                "gpu-metrics/dcgm-"
            )
            if record.get("kind") != "GPU_METRICS" and not (
                dcgm_record_id
                or (
                    isinstance(payload, Mapping)
                    and (
                        payload.get("source") == "DCGM_EXPORTER"
                        or str(payload.get("batch_id", "")).startswith("dcgm-")
                    )
                )
            ):
                continue
            candidate = _candidate(
                record,
                cluster_id=cluster_id,
                node_id=node_id,
                gpu_uuid=gpu_uuid,
                gpu_index=gpu_index,
                power_limit=limit,
                started_at=started,
                observed_until=until,
            )
            if candidate is None:
                continue
            record_id, received, signature = candidate
            if record_id in signatures and signatures[record_id] != signature:
                raise ValueError(
                    "duplicate DCGM record_id has conflicting power evidence"
                )
            signatures[record_id] = signature
            accepted[record_id] = min(received, accepted.get(record_id, received))
            batches.setdefault(record_id, record["payload"])
        except ValueError as exc:
            result["errors"].append(f"record[{position}]: {exc}")
    ordered = sorted(accepted, key=lambda record_id: (accepted[record_id], record_id))
    stored = reference if isinstance(reference, Mapping) else {}
    samples = stored.get("samples")
    samples = samples if isinstance(samples, list) else []
    readings = _readings(records, samples, cluster_id=cluster_id, node_id=node_id)
    floor = max(0.0, _finite(stored.get("breach_floor_us")) or 0.0)
    ignored: list[dict[str, Any]] = []
    for record_id in ordered:
        others = _other_candidates(
            batches[record_id], signatures[record_id][0], gpu_uuid, readings, floor
        )
        if not others:
            result["record_ids"].append(record_id)
            continue
        ignored.append(
            {
                "record_id": record_id,
                "delivered_at": accepted[record_id].isoformat(),
                "other_gpus": others,
            }
        )
    credited = result["record_ids"][0] if result["record_ids"] else None
    result["attribution"] = {
        "gpu_index": int(gpu_index),
        "gpu_uuid": gpu_uuid,
        "record_id": credited,
        "ignored": ignored,
        "reference": {
            "readings": len(samples),
            "breach_floor_us": floor,
            "error": stored.get("error") if reference is not None else "not read",
        },
    }
    if credited is not None:
        result["delivered_at"] = accepted[credited].isoformat()
    elif ordered:
        named = sorted(
            {str(gpu["gpu_index"]) for item in ignored for gpu in item["other_gpus"]},
            key=lambda index: (len(index), index),
        )
        result["errors"].append(
            "no confirmed DCGM power candidate in the load window can be credited to "
            f"the loaded GPU {gpu_index}: every one also carries another GPU's "
            f"candidate (GPU {', '.join(named)}), which could be its confirmation "
            "(delivery_evidence.attribution.ignored)"
        )
    else:
        result["errors"].append(
            "no valid accepted DCGM power candidate in the load window"
        )
    return result


def candidate_records(records: Iterable[Any]) -> list[Any]:
    """The window's GPU evidence whose edge reasons name a candidate or threshold."""

    return [
        item
        for item in records
        if any(
            "candidate" in str(reason).lower() or "threshold" in str(reason).lower()
            for reason in item.get("payload", {}).get("edge_filter_reasons") or []
        )
    ]


def power_reference(
    fixture: Any, env: Mapping[str, str], interval: float
) -> dict[str, Any]:
    """The Store's latest violation counters and flags of every GPU, before the cap,
    and the smallest advance the node's collector grades as a candidate in one
    tick (``breach_floor_us``: its duty-cycle threshold x ``interval``; 0, so
    any advance counts, when ``collector.env`` holds no valid threshold).

    Read-only (:data:`POWER_REFERENCE` in a CPU-cluster Pod) and never fatal: a
    failed read is recorded, and another GPU's nonzero field in a confirmed
    batch then has no earlier reading that bounds it.
    """

    try:
        threshold = float(env.get(DUTY_CYCLE_KEY, DUTY_CYCLE_DEFAULT))
    except (TypeError, ValueError):
        threshold = 0.0
    floor = threshold * interval * 1_000_000 if 0 < threshold <= 1 else 0.0
    grading = {"duty_cycle_threshold": threshold, "breach_floor_us": max(0.0, floor)}
    try:
        value = fixture.regional.cpu_python(
            POWER_REFERENCE,
            fixture.regional.settings.cluster_id,
            fixture.node,
            _DUTY_SUFFIX,
            *sorted(_FLAGS),
        )
        samples = value.get("samples") if isinstance(value, Mapping) else None
        if not isinstance(samples, list):
            raise ValueError("the Store's answer has no samples")
    except Exception as exc:  # noqa: BLE001 - recorded; attribution fails closed
        return {"samples": [], "error": f"{type(exc).__name__}: {exc}"[:300], **grading}
    return {"samples": samples, "error": None, **grading}


def _finite(value: object) -> float | None:
    """A finite float, or None: never raises (an int beyond float range is None)."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def load_generator_failure(
    gpu: object, *, elapsed_seconds: float, minimum_elapsed_seconds: float
) -> str | None:
    """Why the load is not reaching the GPU, or None.

    ``gpu`` is the loaded GPU's nvidia-smi reading in the middle of the load
    window (power draw, enforced limit, utilization). Past
    ``minimum_elapsed_seconds`` a GPU below both correlation thresholds -- the
    very thresholds the collector's ``power_limit_correlation`` candidate needs
    -- is not being loaded (live, plain EKS: ``dcgmproftester12`` ran as an
    active unit while the loaded GPU stayed at 0 % and 76 W under its 200 W
    cap; no candidate could exist, and the case failed on the chain instead of
    on the load). A reading without finite power, limit and utilization is not
    judged: nothing is invented.
    """

    if not isinstance(gpu, Mapping) or elapsed_seconds < minimum_elapsed_seconds:
        return None
    utilization = _finite(gpu.get("utilization_percent"))
    draw = _finite(gpu.get("power_draw_w"))
    limit = _finite(gpu.get("power_limit_w"))
    if utilization is None or draw is None or limit is None or limit <= 0:
        return None
    thresholds = GpuMetricsThresholds()
    if (
        utilization >= thresholds.power_correlation_min_utilization_percent
        or draw >= thresholds.power_limit_ratio * limit
    ):
        return None
    return (
        f"GPU load generator is not loading GPU {gpu.get('index')}: "
        f"{utilization:.0f}% utilization and {draw:.0f} W at a {limit:.0f} W cap "
        f"{elapsed_seconds:.0f}s into the load"
    )


def poll_load_window(
    *,
    stamp: Callable[[], str | None],
    snapshot: Callable[[], Mapping[str, Any]],
    quiet_stamp: str,
    gpu_index: int,
    started_at: datetime,
    interval: float,
    load_seconds: float,
    now: Callable[[], datetime],
    clock: Any,
    timeline_path: Path,
) -> dict[str, Any]:
    """Poll the GPU chain through the load window, reading the loaded GPU
    whenever the chain is still silent.

    ``delivered_at`` is the first stamp that differs from ``quiet_stamp``;
    ``timeline`` is written to ``timeline_path`` on every poll; from the
    second poll on, while nothing was delivered, the host snapshot's reading
    of the loaded GPU is recorded (``load_status``) and judged
    (``load_generator_failure``) -- the first named reason ends the window
    early, so the case fails on the load, not on the chain's silence.
    """

    deadline = clock.monotonic() + load_seconds
    timeline: list[dict[str, Any]] = []
    load_status: list[dict[str, Any]] = []
    load_failure: str | None = None
    delivered_at: str | None = None
    polls = 0
    while True:
        current = stamp()
        entry: dict[str, Any] = {
            "observed_at": now().isoformat(),
            "gpu_metrics_stamp": current,
        }
        delivered = current is not None and current != quiet_stamp
        if polls and not delivered:
            reading = snapshot()
            gpu = next(
                (
                    item
                    for item in (reading.get("gpu_power") or [])
                    if isinstance(item, Mapping) and item.get("index") == gpu_index
                ),
                None,
            )
            elapsed = (now() - started_at).total_seconds()
            status = {"elapsed_seconds": round(elapsed, 3), "gpu": gpu}
            load_status.append(status)
            entry["load_status"] = status
            load_failure = load_generator_failure(
                gpu, elapsed_seconds=elapsed, minimum_elapsed_seconds=interval * 2
            )
        timeline.append(entry)
        write_json_atomic(timeline_path, {"entries": timeline})
        if delivered:
            delivered_at = current
            break
        if load_failure is not None or clock.monotonic() >= deadline:
            break
        polls += 1
        clock.sleep(interval)
    return {
        "delivered_at": delivered_at,
        "timeline": timeline,
        "load_status": load_status,
        "load_failure": load_failure,
        "errors": [load_failure] if load_failure else [],
    }


def load_window_fields(window: Mapping[str, Any], restore: object) -> dict[str, Any]:
    """The window's evidence for the case result, with the load unit's own
    account from the restore receipt (whether it was still active, its result
    and exit status, its journal tail) when the receipt carries one."""

    unit = (
        {
            key: restore.get(key)
            for key in (
                "load_unit_active_at_restore",
                "load_unit_result",
                "load_exit_status",
                "load_journal_tail",
            )
            if key in restore
        }
        if isinstance(restore, Mapping)
        else {}
    )
    return {
        "load_status": window["load_status"],
        "load_failure": window["load_failure"],
        "load_unit": unit or None,
    }
