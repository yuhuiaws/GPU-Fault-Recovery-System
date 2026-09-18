"""Spool consumption with real in-memory revisions and fake replay transport."""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.channel_registry import GPU_INVENTORY_PATH, GPU_METRICS_PATH
from gpu_fault.processor import ProcessorSpoolSettings
from tests._builders import processor_request
from tests.collectors import _cov95_runtime_collect as common
from tests.processor import _cov95_runtime_processor as support

isolated_runtime = common.isolated_runtime
runtime = support.runtime


def spool_request(processor, identity, *, path=GPU_METRICS_PATH, **values):
    payload = {
        "cluster_id": "cluster-a",
        "node_id": identity,
        "observed_at": common.NOW.isoformat(),
        "edge_filter_reasons": ["health-summary"],
        **values,
    }
    request = processor_request(path, body=json.dumps(payload).encode()).model_copy(
        update={"request_id": identity}
    )
    result = processor.store.try_spool_telemetry_requests(
        [request], now=common.NOW, max_depth=100, max_cluster_depth=100
    )
    assert result == [(request, None)]
    return request


def make_spool(runtime, **settings):
    runtime.wait_limit = 1
    return runtime.make_processor(
        spool=ProcessorSpoolSettings(
            **{
                "telemetry_spool_enabled": True,
                "telemetry_spool_workers": 1,
                "telemetry_spool_replay_batch_max_bytes": 16384,
                "telemetry_spool_max_in_flight_bytes": 16384,
                **settings,
            }
        )
    )


@pytest.mark.parametrize("transport", ["http", "direct"])
@pytest.mark.parametrize("status", [200, 422, 503, None])
def test_spool_response_disposes_each_revision_without_queue_receipts(
    runtime, monkeypatch, transport, status, caplog
):
    processor = make_spool(runtime)
    spool_request(processor, "sample-a")
    requests = []

    def respond(payload):
        requests.append(payload)
        return {
            "results": []
            if status is None
            else [
                {
                    "request_id": "sample-a",
                    "status": status,
                    "body": {"detail": "sample verdict"},
                }
            ]
        }

    if transport == "http":
        monkeypatch.setattr(
            "gpu_fault.processor.telemetry_spool.urlopen",
            lambda request, **kwargs: common.Response(
                respond(json.loads(request.data))
            ),
        )
    else:
        processor.telemetry_spool_replay_handler = respond
    processor.run_telemetry_spool()

    metrics = processor.metrics_snapshot()["spool"]
    retry = status is None or status >= 500
    assert processor.store.telemetry_spool_stats()["depth"] == int(retry)
    assert metrics["completed"] == int(not retry)
    assert metrics["released"] == int(retry)
    assert metrics["http_replay"] == int(transport == "http")
    assert metrics["direct_replay"] == int(transport == "direct")
    assert requests[0]["items"][0]["payload"]["node_id"] == "sample-a"
    assert not processor.spool_consumer_running, (
        "spool stop must clear its liveness signal"
    )
    if status == 422:
        assert "sample verdict" in caplog.text


@pytest.mark.parametrize(
    "failure", ["handler", "invalid-status", "completion", "release"]
)
def test_spool_failures_release_only_uncompleted_rows_and_report_errors(
    runtime, monkeypatch, failure, caplog
):
    processor = make_spool(runtime)
    spool_request(
        processor,
        "stale",
        path=GPU_INVENTORY_PATH,
        observed_at=(common.NOW - timedelta(hours=1)).isoformat(),
    )
    spool_request(processor, "fresh", path=GPU_INVENTORY_PATH)
    completed = processor.store.complete_telemetry_spool

    def complete(items):
        if failure == "completion" and any(
            item.request_id == "fresh" for item in items
        ):
            raise OSError("fake completion unavailable")
        return completed(items)

    def respond(payload):
        if failure == "handler":
            raise OSError("fake handler unavailable")
        return {
            "results": [
                {
                    "request_id": item["request_id"],
                    "status": "invalid"
                    if failure == "invalid-status"
                    else 503
                    if failure == "release"
                    else 200,
                    "body": {},
                }
                for item in payload["items"]
            ]
        }

    def release(*args, **kwargs):
        raise OSError("fake release unavailable")

    monkeypatch.setattr(processor.store, "complete_telemetry_spool", complete)
    if failure == "release":
        monkeypatch.setattr(processor.store, "release_telemetry_spool", release)
    processor.telemetry_spool_replay_handler = respond
    processor.run_telemetry_spool()

    metrics = processor.metrics_snapshot()["spool"]
    assert metrics["stale_completed"] == 1
    assert metrics["completed"] == 1
    assert metrics["released"] == (0 if failure == "release" else 1)
    assert metrics["errors"] == (0 if failure == "release" else 1)
    assert processor.store.telemetry_spool_stats()["depth"] == 1
    assert "failed" in caplog.text


@pytest.mark.parametrize("failure", ["claim", "fault-probe", "submit", "abandon"])
def test_spool_external_boundary_failures_leave_observable_liveness_and_ownership(
    runtime, monkeypatch, failure, caplog
):
    processor = make_spool(runtime)
    spool_request(
        processor, "sample-a", blob="x" * (20000 if failure == "abandon" else 0)
    )

    def fail(*args, **kwargs):
        raise OSError(f"fake {failure} unavailable")

    processor.telemetry_spool_replay_handler = lambda payload: {
        "results": [{"request_id": "sample-a", "status": 200, "body": {}}]
    }
    if failure == "claim":
        monkeypatch.setattr(processor.store, "claim_telemetry_spool", fail)
    elif failure == "fault-probe":
        monkeypatch.setattr(processor.store, "processor_fault_backlog_depth", fail)
    elif failure == "submit":
        runtime.submit_error = OSError("fake submit unavailable")
    else:
        monkeypatch.setattr(processor.store, "abandon_telemetry_spool_claims", fail)

    if failure == "submit":
        with pytest.raises(OSError, match="fake submit"):
            processor.run_telemetry_spool()
    else:
        processor.run_telemetry_spool()
    metrics = processor.metrics_snapshot()["spool"]
    assert not processor.spool_consumer_running, (
        "every exit must clear consumer liveness"
    )
    assert all(pool.closed for pool in runtime.pools), (
        "every created pool must be drained"
    )
    assert processor.store.telemetry_spool_stats()["depth"] == int(
        failure != "fault-probe"
    )
    assert metrics["abandoned"] == int(failure == "submit")
    if failure != "submit":
        assert f"fake {failure} unavailable" in caplog.text


@pytest.mark.parametrize("reason", ["stopped", "oversized"])
def test_claimed_but_unsubmitted_spool_rows_are_abandoned_without_replay(
    runtime, monkeypatch, reason
):
    processor = make_spool(runtime)
    spool_request(
        processor, "sample-a", blob="x" * (20000 if reason == "oversized" else 0)
    )
    original = processor.store.claim_telemetry_spool

    def claim(*args, **kwargs):
        items = original(*args, **kwargs)
        if items and reason == "stopped":
            processor.stop()
        return items

    monkeypatch.setattr(processor.store, "claim_telemetry_spool", claim)
    processor.run_telemetry_spool()

    assert runtime.submitted == []
    assert processor.metrics_snapshot()["spool"]["abandoned"] == 1
    assert processor.store.telemetry_spool_stats()["depth"] == 1
    reclaim = original(
        "next-owner",
        now=runtime.clock.now(),
        lease_duration=timedelta(seconds=30),
        limit=1,
    )
    assert [item.request_id for item in reclaim] == ["sample-a"]


def test_spool_completion_cannot_remove_a_newer_revision(runtime):
    processor = make_spool(runtime)
    spool_request(processor, "sample-a")

    def respond(payload):
        item = payload["items"][0]
        newer = processor_request(
            item["path"], body=json.dumps({**item["payload"], "changed": True}).encode()
        ).model_copy(update={"request_id": "sample-new"})
        processor.store.try_spool_telemetry_requests(
            [newer], now=common.NOW, max_depth=10, max_cluster_depth=10
        )
        processor.stop()
        return {"results": [{"request_id": "sample-a", "status": 200, "body": {}}]}

    processor.telemetry_spool_replay_handler = respond
    processor.run_telemetry_spool()
    metrics = processor.metrics_snapshot()["spool"]
    assert metrics["completed"] == 0
    assert metrics["superseded"] == 1
    assert processor.store.telemetry_spool_stats()["depth"] == 1
    rows = processor.store.claim_telemetry_spool(
        "next-owner", now=common.NOW, lease_duration=timedelta(seconds=30), limit=1
    )
    assert rows[0].request_id == "sample-new"
    assert rows[0].payload["changed"] is True


def test_spool_notification_disconnect_preserves_reconnect_diagnostics(
    runtime, monkeypatch, caplog
):
    processor = make_spool(runtime)

    def listener(stop, notify, state):
        state(True)
        notify("sample-a")
        raise OSError("fake notification disconnected")

    monkeypatch.setattr(
        processor.store, "listen_telemetry_spool_notifications", listener, raising=False
    )
    processor.run_telemetry_spool_notifications()
    metrics = processor.metrics_snapshot()["spool"]
    assert metrics["notifications_enabled"] == 0
    assert metrics["notifications_received"] == 1
    assert metrics["notification_reconnects"] == 1
    assert "fake notification disconnected" in caplog.text


def test_absent_optional_listeners_and_disabled_spool_create_no_workers(runtime):
    processor = runtime.make_processor(SimpleNamespace())
    processor.run_queue_notifications()
    processor.run_telemetry_spool_notifications()
    processor.run_telemetry_spool()
    assert runtime.pools == []
    assert processor.metrics_snapshot()["spool"]["rounds"] == 0
