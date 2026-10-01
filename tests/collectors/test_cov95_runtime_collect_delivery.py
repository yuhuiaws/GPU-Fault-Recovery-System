"""Delivery, receipt and replay outcomes with no external connections."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from gpu_fault.collectors import sinks
from gpu_fault.collectors.outbox_file import OutboxFile
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timeout_seconds": 0}, "HTTP timeout"),
        ({"outbox_replay_batch_size": 0}, "batch size"),
        ({"outbox_replay_budget_seconds": 0}, "replay budget"),
        ({"outbox_replay_background_interval_seconds": -1}, "background"),
        ({"processor_receipt_timeout_seconds": 0}, "receipt timeout"),
        ({"processor_receipt_poll_seconds": 0}, "receipt timeout"),
        ({"gzip_min_bytes": -1}, "gzip threshold"),
    ],
)
def test_sink_refuses_unbounded_or_disabled_delivery_limits(options, message):
    with pytest.raises(ValueError, match=message):
        sinks.HttpEventSink("https://ingest.invalid", **options)


@pytest.mark.parametrize("override", [None, {}, "not-a-delivery"])
def test_unknown_optional_deliver_result_falls_back_to_post(override):
    calls = []
    sink = SimpleNamespace(
        deliver=lambda path, payload: override,
        post=lambda path, payload: calls.append((path, payload)) or {"accepted": True},
    )

    result = sinks.deliver_event(sink, "/events", {"event_id": "event-a"})

    assert result.delivered, "a post fallback must preserve successful delivery"
    assert result.response == {"accepted": True}
    assert calls == [("/events", {"event_id": "event-a"})]


def test_explicit_delivery_result_is_not_posted_twice():
    expected = sinks.DeliveryResult(sinks.DeliveryStatus.BUFFERED)
    sink = SimpleNamespace(deliver=lambda *args: expected, post=support.forbidden)
    assert sinks.deliver_event(sink, "/events", {}) is expected
    with pytest.raises(sinks.CollectorError, match="delivery failed"):
        sinks.DeliveryResult(sinks.DeliveryStatus.FAILED).raise_for_failure()


@pytest.mark.parametrize(
    ("payload", "key"),
    [
        ({"attempt_id": "a"}, None),
        ({"node": {"metadata": {"name": "n"}}}, None),
        ({"node": {"metadata": []}}, None),
        ({"attempt_id": "a", "detected_at": "stamp"}, "a/stamp"),
        ({"node": {"metadata": {"name": "n", "resourceVersion": "8"}}}, "node/n/8"),
    ],
)
def test_single_attempt_posts_idempotency_only_when_identity_is_complete(
    monkeypatch, payload, key
):
    calls = []

    def send(request, *, timeout):
        calls.append(dict(request.header_items()))
        return support.Response(body=b"")

    monkeypatch.setattr(sinks, "urlopen", send)
    sink = sinks.HttpEventSink("https://ingest.invalid")
    assert sink.deliver_once("/events", payload) == {}
    headers = {name.lower(): value for name, value in calls[0].items()}
    assert headers.get("idempotency-key") == key
    assert sink.outbox_unlocked_writes_total == 0
    assert sink.outbox_stats()["depth"] == 0
    with pytest.raises(ValueError, match="negative"):
        sink.wait_for_outbox_replay(-1)


@pytest.mark.parametrize("retry_after", ["not-a-number", None, "-2"])
def test_malformed_retry_after_uses_bounded_jitter(monkeypatch, retry_after):
    calls = []
    waits = []

    def send(request, *, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "busy",
                {"Retry-After": retry_after},
                io.BytesIO(b"busy"),
            )
        return support.Response({"accepted": True})

    monkeypatch.setattr(sinks, "urlopen", send)
    sink = sinks.HttpEventSink(
        "https://ingest.invalid",
        max_attempts=2,
        sleep=waits.append,
        jitter=lambda low, high: high,
    )
    assert sink.post("/events", {"event_id": "event-a"}) == {"accepted": True}
    assert waits == [1]
    assert len(calls) == 2


@pytest.mark.parametrize("outcome", ["network-recovery", "empty-body", "timeout"])
def test_receipt_fallback_path_retries_under_its_clock_budget(monkeypatch, outcome):
    clock = support.Clock()
    requests = []

    def send(request, *, timeout):
        requests.append((request.method, request.full_url))
        if request.method == "POST":
            return support.Response({"processor_request_id": "receipt-a"}, status=202)
        if outcome == "timeout" or (
            outcome == "network-recovery" and len(requests) == 2
        ):
            raise OSError("receipt unavailable")
        return support.Response(
            body=b"" if outcome == "empty-body" else b'{"done":true}'
        )

    monkeypatch.setattr(sinks, "urlopen", send)
    monkeypatch.setattr(sinks, "time", SimpleNamespace(monotonic=clock.monotonic))
    sink = sinks.HttpEventSink(
        "https://ingest.invalid",
        sleep=clock.sleep,
        processor_receipt_timeout_seconds=0.5,
        processor_receipt_poll_seconds=0.25,
    )
    if outcome == "timeout":
        with pytest.raises(sinks.CollectorError, match="receipt timed out"):
            sink.post("/v1/attempts/terminal", {"event_id": "event-a"})
        assert len(requests) == 3
    else:
        expected = {} if outcome == "empty-body" else {"done": True}
        assert sink.post("/v1/attempts/terminal", {"event_id": "event-a"}) == expected
    assert all(
        url.endswith("/v1/processor/requests/receipt-a") for _, url in requests[1:]
    ), "receipt fallback must remain bound to the accepted request ID"


def test_receipt_error_with_unreadable_body_still_closes_response(monkeypatch):
    class BrokenBody(io.BytesIO):
        def read(self, *args):
            raise OSError("body unavailable")

    error_body = BrokenBody()
    requests = []

    def send(request, **kwargs):
        requests.append(request.method)
        if request.method == "POST":
            return support.Response({"processor_request_id": "receipt-a"}, status=202)
        raise HTTPError(request.full_url, 422, "refused", {}, error_body)

    monkeypatch.setattr(sinks, "urlopen", send)
    sink = sinks.HttpEventSink("https://ingest.invalid")
    with pytest.raises(sinks.CollectorError, match="completed with HTTP 422"):
        sink.post("/v1/attempts/terminal", {"event_id": "event-a"})
    assert error_body.closed, "even an unreadable receipt response must be closed"
    assert requests == ["POST", "GET"]


def test_replay_failure_cannot_revoke_delivered_event(monkeypatch, tmp_path, caplog):
    path = tmp_path / "outbox.ndjson"
    clock = support.Clock()
    sink = sinks.HttpEventSink(
        "https://ingest.invalid", outbox_path=str(path), outbox_replay_batch_size=1
    )
    assert sink.buffer_for_replay("/events", {"event_id": "old-1"}), (
        "first record must persist"
    )
    assert sink.buffer_for_replay("/events", {"event_id": "old-2"}), (
        "second record must persist"
    )
    real_read = OutboxFile.read
    reads = []
    waits = []

    def read(outbox):
        reads.append(outbox.path)
        if len(reads) == 3:
            raise OSError("background read unavailable")
        return real_read(outbox)

    def send(request, **kwargs):
        if json.loads(request.data)["event_id"] == "old-1":
            waits.append(sink.wait_for_outbox_replay())
        return support.Response({"accepted": True})

    monkeypatch.setattr(OutboxFile, "read", read)
    monkeypatch.setattr(sinks, "urlopen", send)
    monkeypatch.setattr(
        sinks, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    monkeypatch.setattr(
        sinks, "Thread", lambda *, target, **kwargs: SimpleNamespace(start=target)
    )

    assert sink.post("/events", {"event_id": "fresh"}) == {"accepted": True}
    assert waits == [False], "inline replay is active before a worker exists"
    assert sink.wait_for_outbox_replay(), "failed worker must clear replay ownership"
    # Inline batch read, its reconciling re-read, then the worker's one round;
    # the read that failed is not retried on the backoff schedule.
    assert len(reads) == 3, reads
    assert sink.outbox_stats()["replayable"] == 1
    assert "background read unavailable" in caplog.text


def test_replay_diagnostic_cache_is_bounded_and_rearms_evicted_causes(
    monkeypatch, tmp_path, caplog
):
    def fail_read(outbox):
        raise OSError("outbox unavailable")

    monkeypatch.setattr(OutboxFile, "read", fail_read)
    monkeypatch.setattr(sinks, "urlopen", lambda *args, **kwargs: support.Response())
    first = sinks.HttpEventSink(
        "https://ingest.invalid", outbox_path=str(tmp_path / "first.ndjson")
    )
    assert first.post("/events", {}) == {}
    assert first.post("/events", {}) == {}
    for index in range(sinks.REPLAY_FAILURE_TEXTS_REMEMBERED):
        sink = sinks.HttpEventSink(
            "https://ingest.invalid", outbox_path=str(tmp_path / f"{index}.ndjson")
        )
        assert sink.post("/events", {}) == {}
    assert first.post("/events", {}) == {}
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == sinks.REPLAY_FAILURE_TEXTS_REMEMBERED + 2


@pytest.mark.parametrize("installed", [False, True])
def test_optional_sqs_transport_is_constructed_only_through_fake_sdk(
    monkeypatch, installed
):
    calls = []
    client = SimpleNamespace(send_message=lambda **kwargs: calls.append(kwargs) or {})
    sdk = SimpleNamespace(
        client=lambda name: client if name == "sqs" else support.forbidden()
    )
    monkeypatch.setitem(sys.modules, "boto3", sdk if installed else None)
    if not installed:
        with pytest.raises(sinks.CollectorError, match="SQS support"):
            sinks.SqsEventSink("https://sqs.invalid/test")
        assert calls == []
        return
    sink = sinks.SqsEventSink("https://sqs.invalid/test")
    assert sink.post("/v1/collector-events/nvidia-kernel", {"event_id": "event-a"}) == {
        "message_id": None
    }
    assert json.loads(calls[0]["MessageBody"])["payload"] == {"event_id": "event-a"}
