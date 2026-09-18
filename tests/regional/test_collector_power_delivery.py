from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import permutations
from typing import Any

import pytest

from scripts.e2e.regional.collector_power_delivery import (
    DeliveryEvidence,
    delivery_evidence,
)

START = datetime(2026, 9, 17, 8, 24, 56, 674222, tzinfo=timezone.utc)
UNTIL = START + timedelta(seconds=90)
LOAD_START = {"gpu_index": 0, "gpu_uuid": "GPU-loaded", "started_at": START.isoformat()}


def record(
    *,
    observed_at: str = "2026-09-17T08:25:46.745810Z",
    ingested_at: str = "2026-09-17T08:25:47.684496Z",
) -> dict[str, Any]:
    observed = datetime.fromisoformat(observed_at)
    batch_id = f"dcgm-node-a-{int(observed.timestamp() * 1_000_000)}"
    samples = [
        {
            "canonical_name": name,
            "gpu_uuid": "GPU-loaded",
            "gpu_index": "0",
            "value": value,
        }
        for name, value in (
            ("power_limit_w", 200.0),
            ("power_usage_w", 206.058),
            ("gpu_utilization_percent", 99.0),
        )
    ]
    return {
        "kind": "GPU_METRICS",
        "record_id": f"gpu-metrics/{batch_id}",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": observed_at,
        "ingested_at": (
            datetime.fromisoformat(ingested_at) + timedelta(milliseconds=70)
        ).isoformat(),
        "payload": {
            "batch_id": batch_id,
            "source": "DCGM_EXPORTER",
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "observed_at": observed_at,
            "collected_at": observed_at,
            "ingested_at": ingested_at,
            "edge_filter_reasons": ["candidate-confirmed"],
            "collection_errors": [],
            "samples": samples,
        },
    }


def evidence(records: Iterable[object], **kwargs: Any) -> DeliveryEvidence:
    arguments = {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "load_start": LOAD_START,
        "expected_power_limit_w": 200.0,
        "observed_until": UNTIL,
        **kwargs,
    }
    return delivery_evidence(records, **arguments)


def test_earliest_raw_delivery_survives_later_status_overwrite_and_reordering() -> None:
    early = record()
    late = record(
        observed_at="2026-09-17T08:26:02.662686Z",
        ingested_at="2026-09-17T08:26:02.927958Z",
    )
    status = {
        "collector": "GPU_METRICS",
        "batch_id": late["payload"]["batch_id"],
        "observed_at": late["observed_at"],
    }
    for records in permutations([early, late, deepcopy(early)]):
        result = evidence([status, *records])
        assert result == {
            "delivered_at": "2026-09-17T08:25:47.684496+00:00",
            "record_ids": [early["record_id"], late["record_id"]],
            "errors": [],
        }
        latency = (
            datetime.fromisoformat(result["delivered_at"] or "") - START
        ).total_seconds()
        assert latency == pytest.approx(51.010274)
        assert latency <= 60
        assert (
            datetime.fromisoformat(status["observed_at"]) - START
        ).total_seconds() > 60


def test_delivery_order_uses_payload_ingestion_not_observation_or_envelope() -> None:
    early = record()
    early["payload"]["ingested_at"] = (START + timedelta(seconds=80)).isoformat()
    early["ingested_at"] = (START + timedelta(seconds=81)).isoformat()
    later_observed = record(
        observed_at="2026-09-17T08:26:02.662686Z",
        ingested_at="2026-09-17T08:26:02.927958Z",
    )
    result = evidence([early, later_observed])
    assert result["errors"] == []
    assert result["delivered_at"] == "2026-09-17T08:26:02.927958+00:00"
    assert result["record_ids"] == [later_observed["record_id"], early["record_id"]]


def test_equivalent_timezone_offsets_and_ordered_collection_are_accepted() -> None:
    item = record()
    item["payload"]["collected_at"] = "2026-09-17T09:25:47+01:00"
    item["payload"]["ingested_at"] = "2026-09-17T10:25:47.684496+02:00"
    item["observed_at"] = "2026-09-17T10:25:46.745810+02:00"
    result = evidence([item])
    assert result["delivered_at"] == "2026-09-17T08:25:47.684496+00:00"
    assert result["errors"] == []


def test_same_batch_replay_keeps_first_ingestion_time() -> None:
    first = record()
    replay = deepcopy(first)
    replay["payload"]["ingested_at"] = (START + timedelta(seconds=70)).isoformat()
    replay["ingested_at"] = (START + timedelta(seconds=71)).isoformat()
    for rows in ([replay, first], [first, replay]):
        result = evidence(rows)
        assert result["delivered_at"] == "2026-09-17T08:25:47.684496+00:00"
        assert result["record_ids"] == [first["record_id"]]
        assert result["errors"] == []


def test_conflicting_duplicate_cannot_be_hidden_by_record_order() -> None:
    first = record()
    conflicting = deepcopy(first)
    conflicting["payload"]["samples"][1]["value"] = 207.0
    for rows in ([conflicting, first], [first, conflicting]):
        result = evidence(rows)
        assert any("conflicting" in error for error in result["errors"]), (
            'test_conflicting_duplicate_cannot_be_hidden_by_record_order: expected any("conflicting" in error for error in result["errors"])'
        )


@pytest.mark.parametrize("field", ["cluster_id", "node_id"])
def test_consistently_foreign_scope_cannot_supply_delivery(field: str) -> None:
    foreign = record()
    foreign[field] = foreign["payload"][field] = "other"
    assert evidence([foreign])["delivered_at"] is None
    valid = record()
    assert evidence([foreign, valid]) == evidence([valid])


@pytest.mark.parametrize("field", ["cluster_id", "node_id"])
@pytest.mark.parametrize("location", ["record", "payload"])
def test_partial_identity_mismatch_is_retained_as_error(
    field: str, location: str
) -> None:
    malformed = record()
    target = malformed if location == "record" else malformed["payload"]
    target[field] = "other"
    result = evidence([record(), malformed])
    assert result["delivered_at"] is not None
    assert any("identity" in error for error in result["errors"]), (
        'test_partial_identity_mismatch_is_retained_as_error: expected any("identity" in error for error in result["errors"])'
    )


@pytest.mark.parametrize(
    "location,field,value",
    [
        ("record", "kind", "HOST_METRICS"),
        ("record", "record_id", "gpu-metrics/unrelated"),
        ("record", "record_id", None),
        ("payload", "source", "NVIDIA_SMI"),
        ("payload", "batch_id", "nvidia-smi-node-a-123"),
        ("payload", "batch_id", "dcgm-other-node-123"),
        ("payload", "batch_id", "dcgm-node-a-"),
        ("payload", "batch_id", "dcgm-node-a-invalid"),
        ("payload", "batch_id", None),
        ("payload", "cluster_id", None),
        ("record", "node_id", None),
    ],
)
def test_source_and_record_binding_are_required(
    location: str, field: str, value: object
) -> None:
    item = record()
    target = item if location == "record" else item["payload"]
    target[field] = value
    result = evidence([item])
    assert result["delivered_at"] is None
    assert len(result["errors"]) >= 2


@pytest.mark.parametrize("reason", ["candidate-confirmed", "threshold"])
def test_only_exact_positive_reasons_with_power_correlation_qualify(
    reason: str,
) -> None:
    item = record()
    item["payload"]["edge_filter_reasons"] = [reason, "counter-increased"]
    assert evidence([item])["errors"] == []


@pytest.mark.parametrize(
    "reasons",
    [
        ["candidate"],
        ["candidate:power"],
        ["candidate-confirmed-extra"],
        ["threshold:unrelated"],
        ["candidate-confirmed", "future-new-reason"],
        ["candidate-confirmed", {}],
        "candidate-confirmed",
        None,
    ],
)
def test_unrecognized_or_malformed_reason_is_not_silently_dropped(
    reasons: object,
) -> None:
    item = record()
    item["payload"]["edge_filter_reasons"] = reasons
    result = evidence([record(), item])
    assert result["delivered_at"] is not None
    assert any("edge_filter_reason" in error for error in result["errors"]), (
        'test_unrecognized_or_malformed_reason_is_not_silently_dropped: expected any("edge_filter_reason" in error for error in result["errors"])'
    )


@pytest.mark.parametrize(
    "reasons", [[], ["health-summary"], ["candidate-recovered"], ["filter-disabled"]]
)
def test_non_candidate_reason_cannot_pass_even_with_saturated_power(
    reasons: list[str],
) -> None:
    item = record()
    item["payload"]["edge_filter_reasons"] = reasons
    result = evidence([item])
    assert result["delivered_at"] is None
    assert result["record_ids"] == []
    assert result["errors"] == [
        "no valid accepted DCGM power candidate in the load window"
    ]


def test_other_gpu_event_and_context_history_cannot_supply_loaded_gpu_samples() -> None:
    item = record()
    historical = deepcopy(item["payload"]["samples"])
    for sample in item["payload"]["samples"]:
        sample["gpu_uuid"], sample["gpu_index"] = "GPU-other", "1"
    item["payload"]["samples"].append(
        {"canonical_name": "xid_last_error", "gpu_uuid": "GPU-other", "value": 79}
    )
    item["payload"]["context_history"] = [
        {"observed_at": START.isoformat(), "samples": historical}
    ]
    result = evidence([item])
    assert result["delivered_at"] is None
    assert result["record_ids"] == []
    assert result["errors"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", "GPU-other"),
        ("gpu_uuid", None),
        ("gpu_index", "1"),
        ("gpu_index", None),
        ("gpu_index", 0),
        ("gpu_index", "00"),
    ],
)
def test_loaded_gpu_uuid_and_index_must_agree(field: str, value: object) -> None:
    item = record()
    item["payload"]["samples"][0][field] = value
    result = evidence([item])
    assert result["delivered_at"] is None
    assert any("GPU UUID/index" in error for error in result["errors"]), (
        'test_loaded_gpu_uuid_and_index_must_agree: expected any("GPU UUID/index" in error for error in result["errors"])'
    )


@pytest.mark.parametrize(
    "position,value", [(0, 700.0), (0, 200.00001), (1, 189.999), (2, 79.999)]
)
def test_recorded_cap_and_default_power_correlation_thresholds_are_enforced(
    position: int, value: float
) -> None:
    item = record()
    item["payload"]["samples"][position]["value"] = value
    result = evidence([item])
    assert result["delivered_at"] is None
    assert result["errors"]


def test_exact_default_thresholds_qualify_without_reading_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_DCGM_POWER_LIMIT_RATIO", "1")
    monkeypatch.setenv(
        "GPU_FAULT_DCGM_POWER_CORRELATION_MIN_UTILIZATION_PERCENT", "100"
    )
    item = record()
    item["payload"]["samples"][1]["value"] = 190.0
    item["payload"]["samples"][2]["value"] = 80.0
    assert evidence([item])["errors"] == []


@pytest.mark.parametrize(
    "value", [None, "200", True, float("nan"), float("inf"), -float("inf"), 10**1000]
)
def test_nonfinite_or_non_numeric_sample_is_an_error(value: object) -> None:
    item = record()
    item["payload"]["samples"][0]["value"] = value
    result = evidence([record(), item])
    assert result["delivered_at"] is not None
    assert any("finite number" in error for error in result["errors"]), (
        'test_nonfinite_or_non_numeric_sample_is_an_error: expected any("finite number" in error for error in result["errors"])'
    )


@pytest.mark.parametrize("position,value", [(0, 0), (1, -1), (2, 101), (2, -1)])
def test_invalid_numeric_sample_ranges_fail_closed(position: int, value: int) -> None:
    item = record()
    item["payload"]["samples"][position]["value"] = value
    assert any("valid ranges" in error for error in evidence([item])["errors"]), (
        'test_invalid_numeric_sample_ranges_fail_closed: expected any("valid ranges" in error for error in evidence([item])["errors"])'
    )


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "non-object", "no-name", "empty", "no-list"]
)
def test_malformed_relevant_samples_are_preserved_as_errors(problem: str) -> None:
    item = record()
    samples = item["payload"]["samples"]
    if problem == "missing":
        samples.pop()
    elif problem == "duplicate":
        samples.append(deepcopy(samples[0]))
    elif problem == "non-object":
        samples.append(None)
    elif problem == "no-name":
        samples.append({})
    elif problem == "empty":
        samples.clear()
    else:
        item["payload"]["samples"] = None
    result = evidence([record(), item])
    assert result["delivered_at"] is not None
    assert result["errors"]


@pytest.mark.parametrize(
    "location,field",
    [
        ("payload", "observed_at"),
        ("payload", "collected_at"),
        ("payload", "ingested_at"),
        ("record", "observed_at"),
        ("record", "ingested_at"),
    ],
)
@pytest.mark.parametrize("value", [None, "", "invalid", "2026-09-17T08:25:46", 0])
def test_every_candidate_timestamp_must_be_present_valid_and_aware(
    location: str, field: str, value: object
) -> None:
    item = record()
    target = item if location == "record" else item["payload"]
    target[field] = value
    result = evidence([record(), item])
    assert result["delivered_at"] is not None
    assert any(f"{location}.{field}" in error for error in result["errors"]), (
        'test_every_candidate_timestamp_must_be_present_valid_and_aware: expected any(f"{location}.{field}" in error for error in result["errors"])'
    )


@pytest.mark.parametrize(
    "location,field,seconds",
    [
        ("payload", "observed_at", -1),
        ("payload", "collected_at", -1),
        ("payload", "collected_at", 52),
        ("payload", "ingested_at", 49),
        ("record", "ingested_at", 50),
        ("payload", "observed_at", 91),
        ("payload", "collected_at", 91),
        ("payload", "ingested_at", 91),
        ("record", "ingested_at", 91),
        ("record", "observed_at", 51),
    ],
)
def test_stale_future_unordered_or_mismatched_times_are_errors(
    location: str, field: str, seconds: int
) -> None:
    item = record()
    target = item if location == "record" else item["payload"]
    target[field] = (START + timedelta(seconds=seconds)).isoformat()
    if location == "payload" and field == "observed_at":
        item["observed_at"] = target[field]
    result = evidence([item])
    assert result["delivered_at"] is None
    assert len(result["errors"]) >= 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", None),
        ("gpu_uuid", ""),
        ("gpu_index", True),
        ("gpu_index", "0"),
        ("gpu_index", -1),
        ("started_at", None),
        ("started_at", "2026-09-17T08:24:56"),
        ("started_at", UNTIL + timedelta(seconds=1)),
    ],
)
def test_invalid_load_start_cannot_select_records(field: str, value: object) -> None:
    result = evidence([record()], load_start={**LOAD_START, field: value})
    assert result["delivered_at"] is None
    assert result["record_ids"] == []
    assert result["errors"]


@pytest.mark.parametrize(
    "limit", [None, True, "200", 0, -1, float("nan"), float("inf")]
)
def test_recorded_cap_requires_a_positive_finite_number(limit: object) -> None:
    result = evidence([record()], expected_power_limit_w=limit)
    assert result["delivered_at"] is None
    assert result["record_ids"] == []
    assert result["errors"]


@pytest.mark.parametrize(
    "until", [None, "bad", START.replace(tzinfo=None), START - timedelta(seconds=1)]
)
def test_invalid_capture_boundary_fails_closed(until: object) -> None:
    result = evidence([record()], observed_until=until)
    assert result["delivered_at"] is None
    assert result["errors"]


@pytest.mark.parametrize("field", ["cluster_id", "node_id"])
def test_missing_requested_scope_fails_closed(field: str) -> None:
    result = evidence([record()], **{field: ""})
    assert result["delivered_at"] is None
    assert result["errors"]


@pytest.mark.parametrize("errors", [None, ["scrape failed"], ""])
def test_collection_errors_are_not_accepted(errors: object) -> None:
    item = record()
    item["payload"]["collection_errors"] = errors
    result = evidence([record(), item])
    assert any("collection_errors" in error for error in result["errors"]), (
        'test_collection_errors_are_not_accepted: expected any("collection_errors" in error for error in result["errors"])'
    )


def test_malformed_payload_and_record_do_not_disappear() -> None:
    for malformed in (
        {"kind": "GPU_METRICS", "payload": None},
        {"record_id": "gpu-metrics/dcgm-node-a-123"},
        None,
    ):
        result = evidence([record(), malformed])
        assert result["delivered_at"] is not None
        assert result["errors"]


def test_unrelated_non_dcgm_records_do_not_supply_or_poison_delivery() -> None:
    inventory = {"kind": "GPU_INVENTORY", "payload": {}}
    smi = record()
    smi["payload"]["source"] = "NVIDIA_SMI"
    smi["payload"]["batch_id"] = "nvidia-smi-node-a-123"
    smi["record_id"] = "gpu-metrics/nvidia-smi-node-a-123"
    smi["payload"]["edge_filter_reasons"] = ["inventory"]
    assert evidence([inventory, smi])["delivered_at"] is None
    assert evidence([inventory, smi, record()]) == evidence([record()])


def test_empty_input_has_explicit_no_candidate_error() -> None:
    assert evidence([]) == {
        "delivered_at": None,
        "record_ids": [],
        "errors": ["no valid accepted DCGM power candidate in the load window"],
    }


def test_inputs_are_not_mutated() -> None:
    rows = [record()]
    receipt = deepcopy(LOAD_START)
    before = deepcopy((rows, receipt))
    evidence(rows, load_start=receipt)
    assert (rows, receipt) == before


@pytest.mark.parametrize("load_start", [None, [], "unknown"])
def test_malformed_load_receipt_is_reported(load_start: object) -> None:
    result = evidence([record()], load_start=load_start)
    assert result["delivered_at"] is None
    assert result["errors"] == ["load_start must be a GPU load receipt object"]


@pytest.mark.parametrize(
    "location,field",
    [
        ("payload", "observed_at"),
        ("payload", "collected_at"),
        ("payload", "ingested_at"),
        ("record", "observed_at"),
        ("record", "ingested_at"),
    ],
)
def test_missing_timestamp_key_is_preserved_as_error(location: str, field: str) -> None:
    item = record()
    target = item if location == "record" else item["payload"]
    target.pop(field)
    result = evidence([record(), item])
    assert result["delivered_at"] is not None
    assert any(f"{location}.{field}" in error for error in result["errors"]), (
        'test_missing_timestamp_key_is_preserved_as_error: expected any(f"{location}.{field}" in error for error in result["errors"])'
    )


def test_capture_window_boundaries_are_inclusive() -> None:
    item = record(observed_at=START.isoformat(), ingested_at=START.isoformat())
    item["ingested_at"] = START.isoformat()
    result = evidence([item], observed_until=START)
    assert result["delivered_at"] == START.isoformat()
    assert result["errors"] == []


def test_delivery_ties_have_stable_record_id_order() -> None:
    first = record()
    second = record(observed_at="2026-09-17T08:25:47Z")
    for rows in ([first, second], [second, first]):
        result = evidence(rows)
        assert result["record_ids"] == sorted([first["record_id"], second["record_id"]])
        assert result["errors"] == []
