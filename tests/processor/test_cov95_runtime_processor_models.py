"""Request routing, sanitized rejection evidence and bounded completion signals."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    GPU_METRICS_PATH,
    NVIDIA_KERNEL_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.processor import ProcessorCoordinator
from gpu_fault.processor.completion_signals import ProcessorCompletionSignals
from gpu_fault.processor.models import processor_partition_id
from gpu_fault.processor.rejected_events import (
    is_fault_layer_request,
    record_rejected_event_status,
    record_replay_completion,
    response_detail,
)
from tests._builders import processor_request
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"edge_filter_reasons":"health-summary"}',
        b'{"edge_filter_reasons":["threshold:gpu_temperature_c"]}',
    ],
)
def test_unreadable_or_nonroutine_payload_keeps_its_own_spool_identity(body):
    request = processor_request(GPU_METRICS_PATH, body=body)
    assert not request.is_routine_telemetry(), (
        "unconfirmed data cannot be classified as routine"
    )
    assert not request.spoolable(), (
        "nonroutine evidence must remain in the ordered queue"
    )
    assert request.spool_key().endswith(":" + request.request_id), (
        "independent evidence must not coalesce under another event's key"
    )
    assert request.response_body() == b""
    assert request != object()


def test_observation_routing_ignores_non_object_container_metadata():
    request = processor_request(
        WORKLOAD_OBSERVATIONS_PATH,
        body=json.dumps(
            {
                "containers": [
                    None,
                    "bad",
                    {"node_id": None},
                    {"node_id": "node-a"},
                    {"node_id": "node-a"},
                ]
            }
        ).encode(),
    )
    assert request.correlation_scope_keys == ['["cluster-a","node","node-a"]']
    assert request.claim_priority(set()) == 50
    assert request.claim_priority(set(request.correlation_scope_keys)) == -1


@pytest.mark.parametrize(
    "collector,suffix", [("NVIDIA_KERNEL", "nvidia_kernel"), (None, "unknown")]
)
def test_health_summaries_are_scoped_to_the_reporting_collector(collector, suffix):
    request = processor_request(
        COLLECTOR_HEALTH_PATH,
        body=json.dumps(
            {
                "node_id": "node-a",
                "collector": collector,
                "edge_filter_reasons": ["health-summary"],
            }
        ).encode(),
    )
    assert request.ordering_key() == "cluster-a:node:node-a:collector-health-" + suffix
    assert not is_fault_layer_request(request), (
        "liveness summaries are not fault-layer evidence"
    )


def test_routine_claim_priority_ages_only_after_the_explicit_cutoff():
    request = processor_request(
        GPU_METRICS_PATH,
        body=b'{"node_id":"node-a","edge_filter_reasons":["health-summary"]}',
    ).model_copy(update={"created_at": support.NOW})
    assert request.claim_priority(set()) == 100
    assert request.claim_priority(set(), support.NOW - timedelta(microseconds=1)) == 100
    assert request.claim_priority(set(), support.NOW) == 49
    with pytest.raises(ValueError, match="partition count"):
        processor_partition_id(None, 0)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"not-json", ""),
        (b"\xff", ""),
        (b"", ""),
        (b"null", ""),
        (b'{"detail":"refused"}', "refused"),
        (
            b'{"detail":[17,"bad",{"loc":["body",0],"msg":"wrong","type":"invalid","input":"not-logged"}]}',
            "17; bad; body.0 wrong invalid",
        ),
        (b'{"detail":[{"loc":null,"msg":"wrong"}]}', "wrong"),
    ],
)
def test_rejection_details_are_bounded_diagnostics_not_payload_echoes(body, expected):
    assert response_detail(body) == expected
    assert "not-logged" not in response_detail(body)
    assert len(response_detail(json.dumps({"detail": "x" * 1000}).encode())) == 200


@pytest.mark.parametrize(
    ("path", "payload", "cluster", "recorded"),
    [
        (NVIDIA_KERNEL_PATH, {"node_id": "node-a"}, "cluster-a", True),
        (
            NVIDIA_KERNEL_PATH,
            {"node_id": "node-a", "cluster_id": "cluster-b"},
            None,
            True,
        ),
        (NVIDIA_KERNEL_PATH, {"node_id": ""}, "cluster-a", False),
        (NVIDIA_KERNEL_PATH, {"node_id": None}, "cluster-a", False),
        (NVIDIA_KERNEL_PATH, {"node_id": "node-a"}, None, False),
        ("/unregistered", {"node_id": "node-a"}, "cluster-a", False),
    ],
)
def test_rejection_status_requires_known_channel_and_explicit_node_cluster_binding(
    path, payload, cluster, recorded
):
    statuses = []
    store = SimpleNamespace(
        save_collector_status=lambda status: statuses.append(status) or True
    )
    request = processor_request(
        path, body=json.dumps(payload).encode(), cluster_id=cluster
    )
    assert (
        record_rejected_event_status(store, request, status=422, detail="invalid")
        is recorded
    )
    assert len(statuses) == int(recorded)
    if recorded:
        assert statuses[0].cluster_id == (cluster or payload["cluster_id"])
        assert statuses[0].node_id == "node-a"
        assert statuses[0].sample_count == 0


def test_rejection_status_write_failure_keeps_the_fault_rejection_visible(caplog):
    def unavailable(status):
        raise OSError("fake status unavailable")

    processor = ProcessorCoordinator(
        SimpleNamespace(save_collector_status=unavailable),
        owner_id="owner-a",
        internal_token="test-replay",
        active_consumers=True,
    )
    request = processor_request(NVIDIA_KERNEL_PATH, body=b'{"node_id":"node-a"}')
    record_replay_completion(
        processor, request, status=422, body=b'{"detail":"invalid evidence"}'
    )
    metrics = processor.metrics_snapshot()
    assert metrics["fault_rejections_total"] == 1
    assert metrics["completions_by_path_status"][NVIDIA_KERNEL_PATH] == {"4xx": 1}
    assert "fake status unavailable" in caplog.text
    assert not is_fault_layer_request(processor_request("/unregistered")), (
        "unknown routes must not be attributed to a collector"
    )


@pytest.mark.parametrize("body", [b"not-json", b"\xff", b"[]"])
def test_malformed_fault_payload_is_counted_without_inventing_collector_identity(body):
    statuses = []
    processor = ProcessorCoordinator(
        SimpleNamespace(save_collector_status=lambda status: statuses.append(status)),
        owner_id="owner-a",
        internal_token="test-replay",
        active_consumers=True,
    )
    request = processor_request(NVIDIA_KERNEL_PATH, body=body)
    record_replay_completion(processor, request, status=422, body=b"")
    assert processor.metrics_snapshot()["fault_rejections_total"] == 1
    assert statuses == []


def test_completion_waiters_refuse_over_capacity_and_release_on_exception():
    signals = ProcessorCompletionSignals(max_waiters=1)
    with signals.waiting("outside-loop") as missing:
        assert missing is None

    async def scenario():
        with signals.waiting("request-a") as first:
            assert first is not None
            with signals.waiting("request-b") as second:
                assert second is None
                assert signals.pending_waiters == 1
            signals.signal("request-a")
            await asyncio.sleep(0)
            assert first.is_set(), "completion must wake the matching registered waiter"
            with pytest.raises(ValueError, match="private caller stopped"):
                with signals.waiting("request-c"):
                    raise ValueError("private caller stopped")
        assert signals.pending_waiters == 0

    asyncio.run(scenario())
    assert signals.refused_total == 2
    with pytest.raises(ValueError, match="positive"):
        ProcessorCompletionSignals(max_waiters=0)


def test_completion_signal_tolerates_a_loop_that_closed_before_handoff(monkeypatch):
    signals = ProcessorCompletionSignals()

    async def scenario():
        loop = asyncio.get_running_loop()
        with signals.waiting("request-a") as event:

            def closed(callback):
                raise RuntimeError("fake closed event loop")

            with monkeypatch.context() as patch:
                patch.setattr(loop, "call_soon_threadsafe", closed)
                signals.signal("request-a")
            assert not event.is_set(), (
                "failed handoff must not claim the waiter was signalled"
            )
        assert signals.pending_waiters == 0

    asyncio.run(scenario())
