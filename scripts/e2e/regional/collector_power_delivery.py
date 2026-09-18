"""Pure, identity-bound delivery evidence for the COLLECT-002 power window."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, TypedDict

from gpu_fault.gpu_metrics import GpuMetricsThresholds


class DeliveryEvidence(TypedDict):
    delivered_at: str | None
    record_ids: list[str]
    errors: list[str]


_POWER_METRICS = frozenset(
    {"power_limit_w", "power_usage_w", "gpu_utilization_percent"}
)
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


def delivery_evidence(
    records: Iterable[object],
    *,
    cluster_id: str,
    node_id: str,
    load_start: dict[str, Any],
    expected_power_limit_w: float,
    observed_until: datetime,
) -> DeliveryEvidence:
    """Select the first accepted raw DCGM power candidate, never a status row.

    Pass the validated ``injection["load_start"]``, the recorded cap for that GPU
    and the end of the captured load window. No clock or I/O is consulted.
    ``delivered_at`` is the earliest payload ingestion time, normalized to UTC;
    ``record_ids`` are unique qualifying IDs in delivery order. Any ``errors``
    veto acceptance, even when another record supplies a valid timestamp.
    """
    result: DeliveryEvidence = {"delivered_at": None, "record_ids": [], "errors": []}
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
    accepted: dict[str, datetime] = {}
    signatures: dict[str, tuple[datetime, datetime, float, float, float]] = {}
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
        except ValueError as exc:
            result["errors"].append(f"record[{position}]: {exc}")
    ordered = sorted(accepted, key=lambda record_id: (accepted[record_id], record_id))
    result["record_ids"] = ordered
    if ordered:
        result["delivered_at"] = accepted[ordered[0]].isoformat()
    else:
        result["errors"].append(
            "no valid accepted DCGM power candidate in the load window"
        )
    return result
