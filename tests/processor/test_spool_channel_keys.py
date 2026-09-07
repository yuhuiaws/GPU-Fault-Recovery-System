"""The telemetry spool's primary key names its channel, and its consumer tries
every spoolable path before it sleeps.

FINAL-建议汇总 F-E6 (P1-35D, P2-36G, P1-35C). ``spool_key()`` was the queue's
ordering key for a latest-wins sample, and three edge-filtered channels share
one node lane whenever their reasons are routine but not ``health-summary``
(``filter-disabled``, ``baseline:*``, the nvidia-smi fallback): a host sample
overwrote the gpu-metrics sample for the same node. Dormant only because the
edge filter is on. And the consumer stopped probing paths after three distinct
empties while the schedule has four - the fourth path waited out a fallback
sleep whenever the first three were idle.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from threading import Event, Thread

from gpu_fault.channel_registry import (
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    SPOOLABLE_CHANNEL_PATHS,
    TELEMETRY_SPOOL_PATH_SCHEDULE,
)
from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorSpoolSettings,
)
from tests._builders import build_store, copy_model, processor_request

CLUSTER = "cluster-a"
NODE = "node-a"
TOKEN = "telemetry-spool-token-" + "x" * 40
OWNER = "pod-spool:1"


def _request(path: str, payload: dict, request_id: str):
    return copy_model(
        processor_request(path, body=json.dumps(payload).encode(), cluster_id=CLUSTER),
        request_id=request_id,
    )


def _routine_payload(**extra) -> dict:
    return {
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "edge_filter_reasons": ["filter-disabled"],
        "samples": [],
        **extra,
    }


def _spool(store, requests):
    return store.try_spool_telemetry_requests(
        requests, max_depth=100, max_cluster_depth=100
    )


def test_spool_primary_key_includes_channel() -> None:
    requests = [
        _request(GPU_METRICS_PATH, _routine_payload(batch_id="gpu"), "req-gpu"),
        _request(
            HOST_TELEMETRY_PATH,
            _routine_payload(batch_id="host", collection_errors=[]),
            "req-host",
        ),
        _request(
            NODE_LOG_PATH,
            _routine_payload(
                collected_at=datetime.now(timezone.utc).isoformat(), entries=[]
            ),
            "req-logs",
        ),
    ]
    for request in requests:
        assert request.queue_priority() == 100, request.path
        assert request.spoolable() is True, request.path
    # The three channels share the node lane in the queue; that is the
    # collision the spool key has to survive.
    assert len({request.ordering_key() for request in requests}) == 1

    keys = [request.spool_key() for request in requests]
    assert len(set(keys)) == 3
    for request, key in zip(requests, keys, strict=True):
        assert request.path in key

    store = build_store()
    outcomes = _spool(store, requests)
    assert [decision for _, decision in outcomes] == [None, None, None]
    assert store.telemetry_spool_stats()["depth"] == 3


def test_a_channel_still_coalesces_onto_its_own_key() -> None:
    store = build_store()
    first = _request(GPU_METRICS_PATH, _routine_payload(batch_id="b1"), "req-1")
    second = _request(GPU_METRICS_PATH, _routine_payload(batch_id="b2"), "req-2")

    assert first.spool_key() == second.spool_key()
    assert _spool(store, [first]) == [(first, None)]
    assert _spool(store, [second]) == [(second, "coalesced")]
    assert store.telemetry_spool_stats()["depth"] == 1


def test_spoolable_path_count_is_derived_from_registry() -> None:
    distinct = frozenset(TELEMETRY_SPOOL_PATH_SCHEDULE)
    assert distinct == SPOOLABLE_CHANNEL_PATHS
    assert len(distinct) == 4
    assert len(TELEMETRY_SPOOL_PATH_SCHEDULE) > len(distinct)


def test_the_last_spool_path_is_tried_before_the_fallback_sleep(monkeypatch) -> None:
    """Only node-logs has work; it is the fourth distinct path in the schedule."""

    assert TELEMETRY_SPOOL_PATH_SCHEDULE[-1] == NODE_LOG_PATH
    store = build_store()
    _spool(
        store,
        [
            _request(
                NODE_LOG_PATH,
                {
                    "cluster_id": CLUSTER,
                    "node_id": NODE,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "entries": [],
                    "edge_filter_reasons": ["health-summary"],
                },
                "req-node-log",
            )
        ],
    )
    assert store.telemetry_spool_stats()["depth"] == 1
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True,
            telemetry_spool_workers=1,
            telemetry_spool_notification_fallback_seconds=5,
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.01),
    )
    replayed = Event()

    def replay(items):
        store.complete_telemetry_spool(items)
        replayed.set()

    monkeypatch.setattr(processor, "_replay_telemetry_spool", replay)
    thread = Thread(target=processor.run_telemetry_spool)
    started = time.monotonic()
    thread.start()
    try:
        assert replayed.wait(timeout=1.5), "node-logs waited out the fallback sleep"
        assert time.monotonic() - started < 1.5
    finally:
        processor.stop()
        thread.join(timeout=5)
    assert not thread.is_alive(), "spool consumer did not stop"
