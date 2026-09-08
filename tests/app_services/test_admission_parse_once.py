"""``from_http`` parses the body once and pre-fills the caches every later
reader consults, so the event loop and the store thread never parse it again.

A-2 (CP-16). ``from_http`` already computed ``channel.priority(payload)`` from
the payload the decode pool handed it, then threw the result away; the first
``queue_priority()`` call in ``_enqueue`` -- on the event loop -- decoded the
base64 body and ran ``json.loads`` over it a second time (53 ms for a 5 MiB
node-logs batch), and ``ordering_key()`` did it a third time on the store
thread. Measured on the event loop that was 50-80 ms per large request, enough
to stall every concurrent request on the process, including the readiness
probe.
"""

from __future__ import annotations

import json

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NODE_LOG_PATH
from gpu_fault.processor import models
from gpu_fault.processor.models import ProcessorRequest


@pytest.fixture
def count_json_loads(monkeypatch):
    calls: list[int] = []
    real_loads = json.loads

    def counting_loads(*args, **kwargs):
        calls.append(1)
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(models.json, "loads", counting_loads)
    return calls


def _routine_metrics_body() -> bytes:
    return json.dumps(
        {
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "batch_id": "batch-1",
            "observed_at": "2026-09-08T00:00:00Z",
            "edge_filter_reasons": ["health-summary"],
            "samples": [],
        }
    ).encode()


def test_a_parsed_payload_is_never_parsed_again(count_json_loads):
    body = _routine_metrics_body()
    payload = json.loads(body)
    del count_json_loads[:]

    item = ProcessorRequest.from_http(
        method="POST",
        path=GPU_METRICS_PATH,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="cluster-a",
        parsed_payload=payload,
    )
    priority = item.queue_priority()
    key = item.ordering_key()
    spool_key = item.spool_key()
    coalescable = item.coalescable()

    assert priority == 100
    assert key == "cluster-a:node:node-a:gpu-metrics-summary"
    assert spool_key == f"{GPU_METRICS_PATH}|{key}"
    assert coalescable is True
    assert count_json_loads == [], (
        f"the body was parsed {len(count_json_loads)} more time(s) after from_http"
    )


def test_the_prefilled_caches_agree_with_a_cold_computation(count_json_loads):
    body = _routine_metrics_body()
    payload = json.loads(body)

    warm = ProcessorRequest.from_http(
        method="POST",
        path=GPU_METRICS_PATH,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="cluster-a",
        parsed_payload=payload,
    )
    cold = ProcessorRequest(
        method="POST",
        path=GPU_METRICS_PATH,
        body_base64=warm.body_base64,
        cluster_id="cluster-a",
    )

    assert warm.queue_priority() == cold.queue_priority()
    assert warm.ordering_key() == cold.ordering_key()
    assert warm.spool_key() == cold.spool_key()


def test_the_caches_survive_the_idempotent_request_id_copy(count_json_loads):
    body = _routine_metrics_body()
    item = ProcessorRequest.from_http(
        method="POST",
        path=NODE_LOG_PATH,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="cluster-a",
        parsed_payload=json.loads(body),
    )
    del count_json_loads[:]

    copied = item.model_copy(update={"request_id": "processor-idem-abc"})
    copied.queue_priority()
    copied.ordering_key()

    assert count_json_loads == []


def test_without_a_parsed_payload_from_http_parses_exactly_once(count_json_loads):
    body = _routine_metrics_body()

    item = ProcessorRequest.from_http(
        method="POST",
        path=GPU_METRICS_PATH,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="cluster-a",
    )
    item.queue_priority()
    item.ordering_key()

    assert len(count_json_loads) == 1


def test_an_unreadable_body_does_not_poison_the_caches():
    """A body that is not an object is "nobody vouched for it", not routine.

    ``_json_payload`` returns ``None`` for it and the tier stays in the middle;
    ``from_http`` must not pre-fill a cache computed from the ``{}`` it
    substitutes for its own field lookups.
    """

    item = ProcessorRequest.from_http(
        method="POST",
        path=GPU_METRICS_PATH,
        query="",
        body=b"[1, 2, 3]",
        content_type="application/json",
        cluster_id="cluster-a",
    )
    cold = ProcessorRequest(
        method="POST",
        path=GPU_METRICS_PATH,
        body_base64=item.body_base64,
        cluster_id="cluster-a",
    )

    assert item.queue_priority() == cold.queue_priority()
    assert item.ordering_key() == cold.ordering_key()


def test_a_prefilled_request_still_equals_its_persisted_copy():
    """The caches are private state, not identity.

    Pydantic's ``__eq__`` compares ``__pydantic_private__`` too, so a request
    built by ``from_http`` (caches filled) compared unequal to the same row
    read back from a store (caches empty), which broke round-trip assertions
    such as ``store.get_processor_request(id) == request``.
    """

    body = _routine_metrics_body()
    item = ProcessorRequest.from_http(
        method="POST",
        path=GPU_METRICS_PATH,
        query="",
        body=body,
        content_type="application/json",
        cluster_id="cluster-a",
        parsed_payload=json.loads(body),
    )

    persisted = ProcessorRequest.model_validate(item.model_dump())

    assert persisted == item
    assert item == persisted
    assert item != item.model_copy(update={"request_id": "other"})
