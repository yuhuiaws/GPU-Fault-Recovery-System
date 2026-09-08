from __future__ import annotations

from gpu_fault.collectors.sinks import DeliveryStatus

from ._support import (
    CollectorError,
    HttpEventSink,
    RecordingSink,
    SqsEventSink,
    SqsHmaConsumer,
    _LocalControlPlane,
    io,
    json,
    logging,
    pytest,
    sink_from_environment,
)


def test_sink_reads_the_configured_http_timeout(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_COLLECTOR_HTTP_TIMEOUT_SECONDS", "35")

    sink = sink_from_environment()

    assert sink.timeout_seconds == 35


def test_http_event_sink_reuses_one_connection_and_compresses() -> None:
    with _LocalControlPlane() as server:
        sink = HttpEventSink(server.url, max_attempts=1)
        payload = {"cluster_id": "cluster-a", "blob": "x" * 8192}

        for _ in range(3):
            assert sink.post("/v1/collector-events/kernel", payload) == {"ok": True}
        sink.post("/v1/collector-events/kernel", {"cluster_id": "c"})

        assert len(server.connections) == 1
        assert [item[0] for item in server.requests] == ["gzip", "gzip", "gzip", None]
        assert server.requests[0][1] == payload


def test_http_event_sink_retries_stale_keep_alive_connection() -> None:
    with _LocalControlPlane(drop_after_request=2) as server:
        sink = HttpEventSink(server.url, max_attempts=1)

        for index in range(3):
            sink.post(
                "/v1/collector-events/kernel",
                {"cluster_id": "c", "event_id": f"event-{index}"},
            )

        assert len(server.requests) == 3
        assert len(server.connections) == 2


def test_http_event_sink_raises_on_rejected_event(tmp_path) -> None:
    outbox = tmp_path / "outbox.ndjson"
    with _LocalControlPlane(status=400) as server:
        sink = HttpEventSink(server.url, max_attempts=1, outbox_path=str(outbox))

        with pytest.raises(CollectorError) as error:
            sink.post("/v1/collector-events/kernel", {"cluster_id": "c"})

    assert error.value.status_code == 400
    assert "replayable" in outbox.read_text()


def test_http_event_sink_uses_retry_after_with_jitter(monkeypatch) -> None:
    attempts = 0
    sleeps = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"accepted":true}'

    def urlopen_with_throttle(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            from email.message import Message
            from urllib.error import HTTPError

            headers = Message()
            headers["Retry-After"] = "4"
            raise HTTPError("https://control/collector", 429, "busy", headers, None)
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_with_throttle)
    sink = HttpEventSink(
        "https://control",
        sleep=sleeps.append,
        jitter=lambda low, high: (low + high) / 2,
    )

    result = sink.post(
        "/v1/collector-events/gpu-metrics",
        {"cluster_id": "cluster-a", "batch_id": "batch-a"},
    )

    assert result == {"accepted": True}
    assert sleeps == [4.5]
    assert sleeps[0] >= 4


def test_http_event_sink_retries_collector_record_id(monkeypatch) -> None:
    attempts = 0
    keys = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"accepted":true}'

    def flaky(request, **_kwargs):
        nonlocal attempts
        attempts += 1
        keys.append(request.get_header("Idempotency-key"))
        if attempts == 1:
            raise OSError("connection reset")
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", flaky)
    sink = HttpEventSink(
        "https://control", sleep=lambda _seconds: None, jitter=lambda _low, _high: 0
    )

    result = sink.post(
        "/v1/collector-events/nvidia-kernel",
        {"cluster_id": "cluster-a", "record_id": "kmsg-boot-a-3778"},
    )

    assert result == {"accepted": True}
    assert attempts == 2
    assert keys == ["kmsg-boot-a-3778"] * 2


def test_http_event_sink_retries_attempt_observation_with_unique_time(
    monkeypatch,
) -> None:
    attempts = 0
    keys = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"accepted":true}'

    def flaky(request, **_kwargs):
        nonlocal attempts
        attempts += 1
        keys.append(request.get_header("Idempotency-key"))
        if attempts == 1:
            raise OSError("connection reset")
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", flaky)
    sink = HttpEventSink(
        "https://control", sleep=lambda _seconds: None, jitter=lambda _low, _high: 0
    )
    payload = {
        "cluster_id": "cluster-a",
        "attempt_id": "attempt-a",
        "observed_at": "2026-08-20T15:07:23Z",
    }

    assert sink.post("/v1/workload-observations", payload) == {"accepted": True}
    assert keys == ["attempt-a/2026-08-20T15:07:23Z", "attempt-a/2026-08-20T15:07:23Z"]

    sink.post(
        "/v1/workload-observations", {**payload, "observed_at": "2026-08-20T15:07:53Z"}
    )
    assert keys[-1] == "attempt-a/2026-08-20T15:07:53Z"


def test_http_event_sink_buffers_and_replays_retryable_failures(
    monkeypatch, tmp_path
) -> None:
    calls = []
    available = [False]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"accepted":true}'

    def urlopen_probe(request, **_kwargs):
        calls.append(json.loads(request.data))
        if not available[0]:
            raise OSError("network unavailable")
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink("https://control", max_attempts=1, outbox_path=str(outbox))
    with pytest.raises(CollectorError) as captured:
        sink.post("/events", {"sequence": 1})
    assert captured.value.buffered is True, "retryable failure was not buffered"
    assert captured.value.replayable is True, "buffered failure is not replayable"
    buffered = json.loads(outbox.read_text())
    assert buffered["replayable"] is True

    available[0] = True
    sink.post("/events", {"sequence": 2})

    # The live event goes out first and the backlog is replayed behind
    # it, so a fresh fault is never queued behind stale records.
    assert [item["sequence"] for item in calls] == [1, 2, 1]
    assert outbox.read_text() == ""


def test_http_event_sink_replay_never_delays_the_live_event(
    monkeypatch, tmp_path
) -> None:
    """A backlog must not push the current fault behind it.

    Replaying first meant a node returning from a partition spent
    ``outbox_replay_batch_size`` records x ``max_attempts`` timeouts --
    plus a processor receipt poll on the two receipt paths -- before its
    own new XID was even attempted. Replay now runs after delivery,
    under a wall-clock budget, one attempt per record and no receipt
    poll.
    """

    calls: list[dict] = []
    elapsed = [0.0]

    class Response:
        def __init__(self, body: bytes, status: int = 200) -> None:
            self._body = body
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self._body

    def urlopen_probe(request, **_kwargs):
        payload = json.loads(request.data)
        calls.append(payload)
        if payload.get("sequence") == "live":
            return Response(b'{"accepted":true}')
        # Every stale record costs a full client timeout.
        elapsed[0] += 10.0
        raise OSError("network unavailable")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    monkeypatch.setattr("gpu_fault.collectors.sinks.time.monotonic", lambda: elapsed[0])

    outbox = tmp_path / "outbox.ndjson"
    outbox.write_text(
        "".join(
            json.dumps(
                {
                    "path": "/v1/attempts/failure-detected",
                    "payload": {"sequence": index},
                    "replayable": True,
                    "error": "seeded",
                    "failed_at": "2026-08-16T00:00:00+00:00",
                }
            )
            + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_budget_seconds=5,
        sleep=lambda _seconds: None,
    )

    sink.post("/v1/gpu-events/kernel", {"sequence": "live"})

    assert calls[0]["sequence"] == "live"
    # 5s budget / 10s per stale record: the first one is attempted, the
    # rest stay buffered instead of blocking the collector loop.
    assert len(calls) == 2
    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert len(remaining) == 10


def test_http_event_sink_dead_letters_permanent_rejection(
    monkeypatch, tmp_path
) -> None:
    from email.message import Message
    from urllib.error import HTTPError

    monkeypatch.setattr(
        "gpu_fault.collectors.sinks.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HTTPError(
                "https://control/events",
                400,
                "bad request",
                Message(),
                io.BytesIO(b"invalid schema"),
            )
        ),
    )
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink("https://control", outbox_path=str(outbox))

    with pytest.raises(CollectorError):
        sink.post("/events", {"sequence": 1})

    buffered = json.loads(outbox.read_text())
    assert buffered["replayable"] is False
    assert "HTTP 400" in buffered["error"]


def test_http_event_sink_polls_attempt_processor_receipt(monkeypatch) -> None:
    calls = []
    sleeps = []

    class Response:
        def __init__(self, status, body):
            self.status = status
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.body

    responses = iter(
        [
            Response(
                202,
                b'{"accepted":true,"processor_request_id":"p-1",'
                b'"status_url":"/v1/processor/requests/p-1"}',
            ),
            Response(202, b'{"status":"PENDING"}'),
            Response(200, b'{"status":"NO_ACTION"}'),
        ]
    )

    def urlopen_probe(request, **_kwargs):
        calls.append((request.method, request.full_url))
        return next(responses)

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control", sleep=sleeps.append)

    result = sink.post("/v1/attempts/terminal", {"cluster_id": "cluster-a"})

    assert result == {"status": "NO_ACTION"}
    assert calls == [
        ("POST", "https://control/v1/attempts/terminal"),
        ("GET", "https://control/v1/processor/requests/p-1"),
        ("GET", "https://control/v1/processor/requests/p-1"),
    ]
    assert sleeps == [0.25]


def test_http_event_sink_does_not_poll_observation_receipt(monkeypatch) -> None:
    calls = []

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"accepted":true,"processor_request_id":"p-1",'
                b'"status_url":"/v1/processor/requests/p-1"}'
            )

    def urlopen_probe(request, **_kwargs):
        calls.append(request.method)
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control")

    result = sink.post("/v1/workload-observations", {"cluster_id": "cluster-a"})

    assert result["processor_request_id"] == "p-1"
    assert calls == ["POST"]


def test_http_event_sink_propagates_processor_receipt_rejection(monkeypatch) -> None:
    from email.message import Message
    from urllib.error import HTTPError

    class Accepted:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"accepted":true,"processor_request_id":"p-1",'
                b'"status_url":"/v1/processor/requests/p-1"}'
            )

    calls = [0]

    def urlopen_probe(*_args, **_kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return Accepted()
        raise HTTPError(
            "https://control/v1/processor/requests/p-1",
            409,
            "pending containment",
            Message(),
            io.BytesIO(b'{"detail":"containment pending"}'),
        )

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control")

    with pytest.raises(CollectorError) as captured:
        sink.post("/v1/attempts/terminal", {"cluster_id": "cluster-a"})

    assert captured.value.status_code == 409
    assert "containment pending" in str(captured.value)


def test_a_receipt_path_retry_after_a_lost_202_still_polls_the_receipt(
    monkeypatch,
) -> None:
    """The client half of F-E4: the retry after a transport timeout carries the
    same ``Idempotency-Key`` and polls whatever receipt the retry returns. With
    the control plane deriving ``request_id`` from that key, the receipt is the
    original request's, not a second one's."""

    calls: list[tuple[str, str, str | None]] = []

    class Response:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.body

    responses = iter(
        [
            Response(
                202,
                b'{"accepted":true,"processor_request_id":"p-same",'
                b'"status_url":"/v1/processor/requests/p-same"}',
            ),
            Response(200, b'{"status":"CONTAINED"}'),
        ]
    )

    def urlopen_probe(request, **_kwargs):
        calls.append(
            (request.method, request.full_url, request.get_header("Idempotency-key"))
        )
        if len(calls) == 1:
            raise TimeoutError("read timed out before the 202 arrived")
        return next(responses)

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control", sleep=lambda _delay: None)

    result = sink.post(
        "/v1/attempts/failure-detected",
        {"cluster_id": "cluster-a", "event_id": "evt-1"},
    )

    assert result == {"status": "CONTAINED"}
    assert [call[0] for call in calls] == ["POST", "POST", "GET"]
    assert calls[0][2] == calls[1][2] == "evt-1"
    assert calls[2][1] == "https://control/v1/processor/requests/p-same"


def test_sqs_sink_and_consumer_private_delivery() -> None:
    class FakeSqs:
        def __init__(self) -> None:
            self.messages = []
            self.deleted = []

        def send_message(self, **kwargs):
            self.messages.append(
                {"Body": kwargs["MessageBody"], "ReceiptHandle": "receipt-1"}
            )
            return {"MessageId": "message-1"}

        def receive_message(self, **_kwargs):
            return {"Messages": list(self.messages)}

        def delete_message(self, **kwargs):
            self.deleted.append(kwargs["ReceiptHandle"])

    sqs = FakeSqs()
    queue = SqsEventSink("https://sqs/queue", sqs)
    queue.post("/v1/provider-events/hyperpod-hma/cloudwatch", {"node_id": "worker-1"})
    sink = RecordingSink()

    delivered = SqsHmaConsumer(sink, "https://sqs/queue", sqs).run_once(
        wait_time_seconds=0
    )

    assert delivered.delivered == 1, f"the queued event was not delivered: {delivered}"
    assert sink.requests[0][1]["node_id"] == "worker-1"
    assert sqs.deleted == ["receipt-1"]


class _StubResponse:
    """A 2xx whose body the test chooses."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def test_a_garbage_2xx_body_is_retried_like_a_network_failure(
    monkeypatch, caplog
) -> None:
    """F10: an unparseable 2xx is an unknown outcome, not a verdict.

    ``json.JSONDecodeError`` is a ``ValueError``, so it used to escape
    ``deliver_event`` (which only maps ``CollectorError``) and tore down the
    kernel collector's ``/dev/kmsg`` reader mid-batch. Failing the post
    outright was no better: a garbage 200 would then get a harsher verdict
    than a 503, leaving no persistent record of the event at all.
    """

    bodies = iter([b"<html>gateway maintenance</html>", b'{"accepted":true}'])
    keys: list[str | None] = []

    def urlopen_probe(request, **_kwargs):
        keys.append(request.get_header("Idempotency-key"))
        return _StubResponse(next(bodies))

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink(
        "https://control", sleep=lambda _seconds: None, jitter=lambda _low, _high: 0
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        result = sink.post(
            "/v1/collector-events/kernel", {"cluster_id": "c", "event_id": "e-1"}
        )

    assert result == {"accepted": True}, "the retried attempt was not delivered"
    assert keys == ["e-1", "e-1"], "the retry did not carry the same dedup key"
    warnings = [item for item in caplog.records if item.levelno == logging.WARNING]
    assert len(warnings) == 1, "an unparseable body must be logged exactly once"
    assert "non-JSON-object" in warnings[0].getMessage(), warnings[0].getMessage()


def test_a_persistently_garbage_2xx_body_is_buffered_as_replayable(
    monkeypatch, tmp_path
) -> None:
    """A JSON array is as unusable as HTML, and the event must survive it."""

    attempts = 0

    def urlopen_probe(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return _StubResponse(b"[]")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink(
        "https://control",
        max_attempts=2,
        outbox_path=str(outbox),
        sleep=lambda _seconds: None,
        jitter=lambda _low, _high: 0,
    )

    with pytest.raises(CollectorError) as captured:
        sink.post("/v1/collector-events/kernel", {"cluster_id": "c", "event_id": "e-1"})

    assert attempts == 2, "the unparseable body did not walk the retry ladder"
    assert captured.value.buffered is True, "the event was dropped instead of buffered"
    assert captured.value.replayable is True, "the buffered event is not replayable"
    record = json.loads(outbox.read_text())
    assert record["replayable"] is True, record
    assert record["payload"]["event_id"] == "e-1", record


def test_deliver_maps_a_garbage_2xx_body_to_buffered(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "gpu_fault.collectors.sinks.urlopen",
        lambda *_args, **_kwargs: _StubResponse(b"<html>oops</html>"),
    )
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink("https://control", max_attempts=1, outbox_path=str(outbox))

    result = sink.deliver(
        "/v1/collector-events/kernel", {"cluster_id": "c", "event_id": "e-1"}
    )

    assert result.status is DeliveryStatus.BUFFERED, result
    assert isinstance(result.error, CollectorError), (
        "a raw ValueError leaked out of the sink"
    )
    assert outbox.exists(), "the unparseable 2xx left no persistent record"


def test_receipt_poll_retries_an_unreadable_receipt(monkeypatch) -> None:
    """F11's other half: an unreadable receipt is not a completed request."""

    calls: list[str] = []
    sleeps: list[float] = []
    bodies = iter(
        [
            b'{"accepted":true,"processor_request_id":"p-1",'
            b'"status_url":"/v1/processor/requests/p-1"}',
            b"<html>gateway maintenance</html>",
            b'{"status":"CONTAINED"}',
        ]
    )

    def urlopen_probe(request, **_kwargs):
        calls.append(request.get_method())
        body = next(bodies)
        return _StubResponse(body, status=202 if len(calls) == 1 else 200)

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control", sleep=sleeps.append)

    result = sink.post(
        "/v1/attempts/failure-detected", {"cluster_id": "cluster-a", "event_id": "e-1"}
    )

    assert result == {"status": "CONTAINED"}, result
    assert calls == ["POST", "GET", "GET"], "the unreadable receipt was terminal"
    assert sleeps == [0.25], sleeps


def test_http_event_sink_caps_retry_after(monkeypatch) -> None:
    """F6: an ALB may answer 429 with ``Retry-After: 3600``."""

    attempts = 0
    sleeps: list[float] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"accepted":true}'

    def urlopen_with_throttle(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            from email.message import Message
            from urllib.error import HTTPError

            headers = Message()
            headers["Retry-After"] = "3600"
            raise HTTPError("https://control/collector", 429, "busy", headers, None)
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_with_throttle)
    sink = HttpEventSink(
        "https://control",
        sleep=sleeps.append,
        jitter=lambda low, high: (low + high) / 2,
    )

    result = sink.post(
        "/v1/collector-events/gpu-metrics",
        {"cluster_id": "cluster-a", "batch_id": "batch-a"},
    )

    assert result == {"accepted": True}, result
    assert sleeps == [30.5], "Retry-After was not capped before the jitter was added"
    assert sleeps[0] >= 30, "the cap must still back the retry off"


@pytest.mark.parametrize(
    ("payload", "expected_key"),
    [
        ({"snapshot_id": "snap-1"}, "snap-1"),
        ({"log_event_id": "cw-1"}, "cw-1"),
        (
            {"node": {"metadata": {"name": "worker-1", "resourceVersion": "42"}}},
            "node/worker-1/42",
        ),
    ],
    ids=["gpu-inventory-snapshot", "cloudwatch-log-event", "hma-node-event"],
)
def test_http_event_sink_retries_every_keyed_collector_payload(
    monkeypatch, payload, expected_key
) -> None:
    """F5: without a key these paths got one attempt and no pool retry."""

    attempts = 0
    keys = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"accepted":true}'

    def flaky(request, **_kwargs):
        nonlocal attempts
        attempts += 1
        keys.append(request.get_header("Idempotency-key"))
        if attempts == 1:
            raise OSError("connection reset by an idle keep-alive peer")
        return Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", flaky)
    sink = HttpEventSink(
        "https://control", sleep=lambda _seconds: None, jitter=lambda _low, _high: 0
    )

    result = sink.post(
        "/v1/collector-events/gpu-inventory", {"cluster_id": "cluster-a", **payload}
    )

    assert result == {"accepted": True}, result
    assert attempts == 2, "the payload was sent once and dropped into the outbox"
    assert keys == [expected_key] * 2, "the retry did not carry a stable dedup key"


def test_receipt_poll_retries_a_retryable_status_until_it_succeeds(monkeypatch) -> None:
    """F11: a 503 from the status URL is capacity, not the request's verdict."""

    from email.message import Message
    from urllib.error import HTTPError

    calls: list[str] = []
    sleeps: list[float] = []

    class Response:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.body

    def urlopen_probe(request, **_kwargs):
        calls.append(request.get_method())
        if len(calls) == 1:
            return Response(
                202,
                b'{"accepted":true,"processor_request_id":"p-1",'
                b'"status_url":"/v1/processor/requests/p-1"}',
            )
        if len(calls) == 2:
            raise HTTPError(
                "https://control/v1/processor/requests/p-1",
                503,
                "store at capacity",
                Message(),
                io.BytesIO(b'{"detail":"store at capacity"}'),
            )
        return Response(200, b'{"status":"CONTAINED"}')

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control", sleep=sleeps.append)

    result = sink.post(
        "/v1/attempts/failure-detected", {"cluster_id": "cluster-a", "event_id": "e-1"}
    )

    assert result == {"status": "CONTAINED"}, result
    assert calls == ["POST", "GET", "GET"], "the retryable poll failure was terminal"
    assert sleeps == [0.25], sleeps


def test_receipt_poll_get_drops_the_post_body_headers(monkeypatch) -> None:
    """F12: the bodiless GET reused the POST's body and idempotency headers."""

    seen: list[dict[str, str]] = []

    class Response:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.body

    def urlopen_probe(request, **_kwargs):
        seen.append({key.lower(): value for key, value in request.header_items()})
        if len(seen) == 1:
            return Response(
                202,
                b'{"accepted":true,"processor_request_id":"p-1",'
                b'"status_url":"/v1/processor/requests/p-1"}',
            )
        return Response(200, b'{"status":"CONTAINED"}')

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", urlopen_probe)
    sink = HttpEventSink("https://control", bearer_token="token-1", gzip_min_bytes=1)

    result = sink.post(
        "/v1/attempts/failure-detected", {"cluster_id": "cluster-a", "event_id": "e-1"}
    )

    assert result == {"status": "CONTAINED"}, result
    post_headers, get_headers = seen
    assert post_headers["content-encoding"] == "gzip", post_headers
    assert post_headers["idempotency-key"] == "e-1", post_headers
    assert "content-type" not in get_headers, get_headers
    assert "content-encoding" not in get_headers, get_headers
    assert "idempotency-key" not in get_headers, get_headers
    assert get_headers["authorization"] == "Bearer token-1", (
        "the receipt GET lost its bearer token"
    )
    assert get_headers["x-gpu-fault-cluster-id"] == "cluster-a", (
        "the receipt GET lost the cluster id header"
    )
