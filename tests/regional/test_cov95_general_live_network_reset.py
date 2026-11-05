"""GF-GENERAL-LIVE network counter reset: sample reads and the re-arm verdict."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional import general_live_network_reset as reset

AFTER = datetime(2026, 11, 5, 4, 0, tzinfo=timezone.utc)
A, B = "gfrs-veth-a", "gfrs-veth-b"


def batch(observed_at: Any, *samples: dict[str, Any]) -> dict[str, Any]:
    return {"observed_at": observed_at, "batch_id": "host-1", "samples": list(samples)}


def drop(device: str, value: Any) -> dict[str, Any]:
    return {"name": reset.RESET_METRIC, "device": device, "value": value}


def test_drop_samples_read_naive_times_as_utc_and_skip_unreadable_ones() -> None:
    document = {
        "batches": [
            batch("2026-11-05T04:01:00", drop(A, 3)),
            batch("yesterday", drop(A, 5)),
            batch("2026-11-05T04:02:00+00:00", drop(A, "n/a"), drop(B, 0)),
            batch("2026-11-05T03:59:00+00:00", drop(A, 7)),
        ]
    }

    positive = reset.drops_samples(document, [A, B], after=AFTER, positive=True)
    zeros = reset.drops_samples(document, [A, B], after=AFTER, positive=False)

    assert [(item["device"], item["value"]) for item in positive] == [(A, 3)]
    assert [(item["device"], item["value"]) for item in zeros] == [(B, 0)]


def test_host_delivery_needs_the_host_collector_and_its_batch_prefix() -> None:
    later = "2026-11-05T04:05:00+00:00"
    document = {
        "collector_statuses": [
            {"collector": "GPU_DCGM", "batch_id": "host-9", "last_success_at": later},
            {
                "collector": reset.HOST_COLLECTOR,
                "batch_id": "gpu-9",
                "observed_at": later,
            },
            {
                "collector": reset.HOST_COLLECTOR,
                "batch_id": "host-3",
                "observed_at": "x",
            },
            {
                "collector": reset.HOST_COLLECTOR,
                "batch_id": "host-4",
                "observed_at": later,
            },
        ]
    }

    assert reset.host_delivery_after(document, AFTER) == {
        "collector": reset.HOST_COLLECTOR,
        "batch_id": "host-4",
        "observed_at": later,
    }
    assert reset.host_delivery_after({"collector_statuses": []}, AFTER) is None


def rearm(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "rearmed": [{"value": 4}],
        "zeros_after": [{"value": 0}],
        "readings": [{"value": 4}, {"value": 0}],
        "post_reset_drops": 10.0,
        "budget": 60,
        **overrides,
    }
    return reset.rearm_errors(A, B, **arguments)


def test_a_rearm_from_the_new_baseline_passes() -> None:
    assert rearm() == []
    assert rearm(post_reset_drops=None) == []


def test_a_negative_delta_after_the_recreate_is_named() -> None:
    assert rearm(readings=[{"value": -3}, {"value": "x"}]) == [
        f"negative {reset.RESET_METRIC} shipped after the recreate: [-3]"
    ]


def test_a_delta_larger_than_the_new_pair_counted_is_named() -> None:
    assert rearm(rearmed=[{"value": 12}, {"value": None}]) == [
        f"{reset.RESET_METRIC} [12] after the recreate exceeds the 10.0 frames the "
        "oracle counted on the new pair: not a delta from the new baseline"
    ]


def test_a_rearm_without_the_following_zero_is_named() -> None:
    assert rearm(zeros_after=[]) == [
        f"no zero {reset.RESET_METRIC} followed the second burst within 60 s"
    ]
