"""await_metric_recovery: the product's own proof that a held metric recovered."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import general_live_common as common

NODE = "node-a"
KEY = "cl/node-a/cpu_usage_percent/node"
LATER = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _ctx() -> Any:
    def wait_until(accept: Any, **_: Any) -> Any:
        return accept()

    return SimpleNamespace(
        settings=SimpleNamespace(regional=SimpleNamespace(cluster_id="cl"), node=NODE),
        host_summary=300,
        host_interval=15,
        wait_until=wait_until,
    )


def _batch(value: float) -> dict[str, Any]:
    return {
        "observed_at": "2026-10-24T08:23:30+00:00",
        "batch_id": f"b-{value}",
        "samples": [{"name": "cpu_usage_percent", "device": None, "value": value}],
    }


def _run(
    monkeypatch: pytest.MonkeyPatch,
    batches: list[dict[str, Any]],
    signal: dict[str, Any] | None,
) -> dict[str, Any]:
    monkeypatch.setattr(
        common,
        "activity",
        lambda ctx, since, **_: {
            "batches": batches,
            "health_signals": {KEY: signal} if signal else {},
        },
    )
    return common.await_metric_recovery(_ctx(), "cpu_usage_percent", 98.0)


def test_one_fresh_sample_and_a_cleared_signal_prove_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Live 2026-10-24 GF-GENERAL-LIVE-001.recovery: the recovered edge (66.3 %)
    # was the only delivered batch for 14 minutes; the cleared signal is the
    # product's own recovery record.
    result = _run(monkeypatch, [_batch(66.3)], {"active": False, "observed_at": LATER})
    assert result["errors"] == [], result


def test_two_fresh_samples_prove_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, [_batch(40.0), _batch(41.0)], None)["errors"] == []


@pytest.mark.parametrize(
    ("batches", "signal"),
    [
        ([_batch(66.3)], {"active": True, "observed_at": LATER}),
        ([_batch(66.3)], None),
        ([_batch(99.0)], {"active": False, "observed_at": LATER}),
        ([_batch(66.3)], {"active": False, "observed_at": "2026-10-24T08:00:00Z"}),
    ],
)
def test_no_recovery_without_fresh_product_evidence(
    monkeypatch: pytest.MonkeyPatch,
    batches: list[dict[str, Any]],
    signal: dict[str, Any] | None,
) -> None:
    errors = _run(monkeypatch, batches, signal)["errors"]
    assert errors and "did not recover" in errors[0], errors


def test_the_wait_reads_from_the_stop_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # Live 2026-10-24 001.cpu_busy (pass 3): the recovered edge was delivered
    # 8 s before the wait began, while the load was stopping.
    seen: list[Any] = []

    def fake_activity(ctx: Any, since: Any, **_: Any) -> dict[str, Any]:
        seen.append(since)
        return {
            "batches": [_batch(88.0)],
            "health_signals": {KEY: {"active": False, "observed_at": LATER}},
        }

    monkeypatch.setattr(common, "activity", fake_activity)
    stop = datetime(2026, 10, 24, 11, 13, 40, tzinfo=timezone.utc)
    result = common.await_metric_recovery(_ctx(), "cpu_usage_percent", 98.0, since=stop)
    assert result["errors"] == [] and seen == [stop], (result, seen)


def test_without_a_stop_time_the_wait_looks_back_two_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(
        common,
        "activity",
        lambda ctx, since, **_: seen.append(since)
        or {"batches": [], "health_signals": {}},
    )
    before = datetime.now(timezone.utc)
    common.await_metric_recovery(_ctx(), "cpu_usage_percent", 98.0)
    assert seen and seen[0] <= before - timedelta(seconds=29), seen
