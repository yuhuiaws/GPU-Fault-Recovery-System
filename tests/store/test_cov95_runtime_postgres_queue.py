from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store.contracts import WakeupChannel
from gpu_fault.store.shared.errors import StaleFencingTokenError
from tests._builders import processor_request
from tests.store import _cov95_runtime_postgres as postgres
from tests.store._cov95_runtime_models import CLUSTER, NOW, observation
from tests.store._cov95_runtime_queue import (
    FAULT_PATH,
    HOST_PATH,
    assert_rejected_spool_duplicates_are_not_acknowledged,
    sample,
)
from tests.store._postgres_processor_claim_support import postgres_store_instance


@pytest.fixture
def store(monkeypatch):
    postgres.validated_url()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
    yield from postgres_store_instance()


@pytest.mark.parametrize("reason", ["global", "cluster"])
def test_spool_rejection_does_not_acknowledge_a_discarded_duplicate(store, reason):
    assert_rejected_spool_duplicates_are_not_acknowledged(store, reason)


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        ({"max_depth": 0, "max_cluster_depth": 10}, "global"),
        (
            {"max_depth": 1, "max_cluster_depth": 10, "reserved_fault_depth": 1},
            "global_reserved",
        ),
        ({"max_depth": 10, "max_cluster_depth": 0}, "cluster"),
        (
            {
                "max_depth": 10,
                "max_cluster_depth": 1,
                "reserved_cluster_fault_depth": 1,
            },
            "cluster_reserved",
        ),
    ],
)
def test_routine_admission_preserves_global_and_cluster_reservations(
    store, batch, limits, reason
):
    request = sample("routine-rejected")
    if batch:
        result = store.try_enqueue_processor_requests_batch([request], **limits)[0]
    else:
        result = store.try_enqueue_processor_request(request, **limits)
    assert result == (None, reason), (
        "the reserved capacity must not admit a routine row"
    )
    assert store.processor_queue_stats()["depth"] == 0, (
        "a rejected admission must not consume a queue counter"
    )


def test_duplicate_admission_returns_committed_row_without_charging_capacity(store):
    request = sample("same-request", path=FAULT_PATH)
    accepted = store.try_enqueue_processor_request(
        request, max_depth=1, max_cluster_depth=1
    )
    assert accepted == (request, None), "the first request must fit the one-row budget"
    assert (
        store.try_enqueue_processor_request(request, max_depth=0, max_cluster_depth=0)
        == accepted
    ), "a retry of a committed request does not require new capacity"
    assert store.try_enqueue_processor_requests_batch(
        [request, request], max_depth=0, max_cluster_depth=0
    ) == [accepted, accepted]
    assert (
        store.try_enqueue_processor_requests_batch([], max_depth=0, max_cluster_depth=0)
        == []
    )
    assert store.processor_queue_stats()["depth"] == 1


def test_routine_batch_coalesces_to_a_single_pending_row(store):
    requests = [sample("routine-first"), sample("routine-last", value=2)]
    results = store.try_enqueue_processor_requests_batch(
        requests, max_depth=10, max_cluster_depth=10
    )
    assert [reason for _, reason in results] == [None, "coalesced"]
    assert {row.request_id for row, _ in results} == {"routine-first"}
    assert store.processor_queue_stats()["depth"] == 1
    newer = sample("routine-newer", value=3)
    result = store.try_enqueue_processor_requests_batch(
        [newer], max_depth=10, max_cluster_depth=10
    )
    assert result[0][1] == "coalesced"
    assert result[0][0].request_id == "routine-first"
    assert store.processor_queue_stats()["depth"] == 1


def test_spool_coalescing_keeps_the_newest_payload_and_fences_old_completion(store):
    first, newer = sample("spool-first"), sample("spool-newer", value=2)
    assert store.try_spool_telemetry_requests(
        [first, newer], max_depth=10, max_cluster_depth=10, now=NOW
    ) == [(first, "coalesced"), (newer, None)]
    (claimed,) = store.claim_telemetry_spool(
        "owner-first", now=NOW, lease_duration=timedelta(seconds=30), limit=1
    )
    assert claimed.request_id == newer.request_id
    assert claimed.payload["value"] == 2
    latest = sample("spool-latest", value=3)
    assert store.try_spool_telemetry_requests(
        [latest], max_depth=10, max_cluster_depth=10, now=NOW
    ) == [(latest, "coalesced")]
    assert store.complete_telemetry_spool([claimed]) == 0, (
        "a late completion must not delete the replacement revision"
    )
    (replacement,) = store.claim_telemetry_spool(
        "owner-next", now=NOW, lease_duration=timedelta(seconds=30), limit=1
    )
    assert replacement.revision > claimed.revision, "a replacement needs a fresh fence"
    assert replacement.payload["value"] == 3
    assert store.complete_telemetry_spool([replacement]) == 1


def test_spool_empty_batches_limits_and_path_filters_do_not_claim_work(store):
    assert (
        store.try_spool_telemetry_requests(
            [], max_depth=1, max_cluster_depth=1, now=NOW
        )
        == []
    )
    assert store.complete_telemetry_spool([]) == 0
    assert store.abandon_telemetry_spool_claims([], now=NOW) == 0
    assert store.release_telemetry_spool([], now=NOW) == (0, 0)
    request = sample("spool-one")
    store.try_spool_telemetry_requests(
        [request], max_depth=10, max_cluster_depth=10, now=NOW
    )
    claim = {"now": NOW, "lease_duration": timedelta(seconds=30), "limit": 1}
    assert store.claim_telemetry_spool("owner", **{**claim, "limit": 0}) == []
    assert store.claim_telemetry_spool("owner", max_bytes=0, **claim) == []
    assert store.claim_telemetry_spool("owner", path="/other", **claim) == []
    (large,) = store.claim_telemetry_spool(
        "owner", path=HOST_PATH, max_bytes=1, **claim
    )
    assert large.payload_bytes > 1, "the first oversized sample must make progress"
    assert store.abandon_telemetry_spool_claims([large], now=NOW) == 1
    (retried,) = store.claim_telemetry_spool("owner", **claim)
    assert retried.attempts == large.attempts, "abandoning must refund the attempt"
    assert store.release_telemetry_spool([retried], now=NOW, max_attempts=1) == (0, 1)
    assert store.telemetry_spool_depths() == {"depth": 0, "by_cluster": {}}


def test_spool_cached_admissions_still_enforce_a_cluster_cap(store):
    first = sample("first")
    store.try_spool_telemetry_requests(
        [first], max_depth=10, max_cluster_depth=1, now=NOW
    )
    assert store.try_spool_telemetry_requests(
        [sample("second", node="other-node")],
        max_depth=10,
        max_cluster_depth=1,
        now=NOW,
    ) == [(None, "cluster")]
    stats = store.telemetry_spool_stats(now=NOW)
    assert stats["depth"] == 1
    assert store.telemetry_spool_depths()["by_cluster"] == {CLUSTER: 1}


class ListenerConnection:
    def __init__(self, *, read_only=False, payloads=("repeat", "repeat", "stop")):
        self.autocommit = True
        self.closed = False
        self.read_only = read_only
        self.statements = []
        self.delivered = []
        self.payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def execute(self, query):
        self.statements.append(query)
        return self

    def fetchone(self):
        return False, "on" if self.read_only else "off"

    def notifies(self, **kwargs):
        for payload in self.payloads:
            self.delivered.append(payload)
            yield SimpleNamespace(payload=payload)


def test_listener_deduplicates_notifications_and_closes_its_fake_transport(
    store, monkeypatch
):
    import psycopg

    connection = ListenerConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    stop = Event()
    notifications = []
    states = []

    def notified(payload):
        notifications.append(payload)
        if payload == "stop":
            stop.set()

    store.listen_telemetry_spool_notifications(
        stop, notified, states.append, writer_check_seconds=0
    )
    assert notifications == ["repeat", "stop"], (
        "one wakeup batch must suppress duplicates without losing later payloads"
    )
    assert states == [True, False], (
        "the listener must report its final disconnected state"
    )
    assert connection.closed, "the owned listener connection must be closed"
    assert any("pg_is_in_recovery" in query for query in connection.statements), (
        "a listener must recheck that its connection still reaches a writer"
    )


def test_listener_refuses_a_demoted_fake_connection_before_reading_notifications(
    store, monkeypatch
):
    import psycopg

    connection = ListenerConnection(read_only=True)
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    stop = Event()
    notifications = []
    states = []

    def state_changed(connected):
        states.append(connected)
        if not connected:
            stop.set()

    store.listen_telemetry_spool_notifications(
        stop, notifications.append, state_changed, writer_check_seconds=0
    )
    assert states == [True, False], "demotion must clear readiness"
    assert notifications == [], "a read-only connection cannot provide writer wakeups"
    assert connection.delivered == [], "the listener must check writer status first"
    assert connection.closed, "the demoted fake connection must be closed"


def test_listener_connect_failure_stops_cleanly_on_shutdown(store, monkeypatch):
    import psycopg

    stop = Event()
    states = []

    def failed_connection(*args, **kwargs):
        raise psycopg.OperationalError("synthetic listener connection loss")

    def state_changed(connected):
        states.append(connected)
        stop.set()

    monkeypatch.setattr(psycopg, "connect", failed_connection)
    store.listen_telemetry_spool_notifications(
        stop,
        lambda value: pytest.fail("disconnected listener delivered a wakeup"),
        state_changed,
    )
    assert states == [False], (
        "failed connect must remain disconnected and honor shutdown"
    )


@pytest.mark.parametrize("queue_mode", ["legacy", "dual", "dedicated"])
def test_observation_completion_keeps_queue_counters_and_response_in_sync(
    store, monkeypatch, queue_mode
):
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE", queue_mode)
    with postgres.peer_store("legacy") as active:
        request = processor_request(
            "/v1/workload-observations",
            cluster_id=CLUSTER,
            body=observation().model_dump_json().encode(),
        )
        active.enqueue_processor_request(request)
        (claimed,) = active.claim_active_processor_requests(
            "owner",
            now=datetime.now(timezone.utc),
            lease_duration=timedelta(seconds=30),
            limit=1,
        )
        completed = active.complete_active_processor_request(
            request.request_id,
            "owner",
            claimed.leader_epoch,
            claimed.lease_token,
            path=request.path,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )
        assert completed.status is ProcessorRequestStatus.COMPLETED
        assert completed.response_body_base64 == "e30="
        assert active.get_processor_request(request.request_id) == completed
        assert active.processor_queue_stats()["depth"] == 0, (
            "completion must decrement its counter in the same transaction"
        )


def test_missing_observation_completion_is_fenced_instead_of_acknowledged(store):
    with pytest.raises(StaleFencingTokenError, match="fencing"):
        store.complete_active_processor_request(
            "absent",
            "owner",
            1,
            "synthetic-lease",
            path="/v1/workload-observations",
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )
    assert store.processor_queue_stats()["depth"] == 0


def test_closed_store_propagates_observation_completion_flush_failure(store):
    from psycopg_pool import PoolClosed

    store.close()
    with pytest.raises(PoolClosed, match="closed"):
        store.complete_active_processor_request(
            "absent",
            "owner",
            1,
            "synthetic-lease",
            path="/v1/workload-observations",
            response_status=200,
            response_content_type=None,
            response_body_base64="e30=",
        )


@pytest.mark.parametrize("invalid", ["owner", "epoch", "token", "expired"])
def test_active_request_renewal_refuses_wrong_or_expired_lease_without_mutation(
    store, invalid
):
    request = sample("lease-request", path=FAULT_PATH)
    store.enqueue_processor_request(request)
    at = datetime.now(timezone.utc)
    (claimed,) = store.claim_active_processor_requests(
        "owner",
        now=at - timedelta(seconds=2) if invalid == "expired" else at,
        lease_duration=timedelta(seconds=1 if invalid == "expired" else 30),
        limit=1,
    )
    assert (
        store.renew_active_processor_request(
            request.request_id,
            "other" if invalid == "owner" else "owner",
            claimed.leader_epoch + (invalid == "epoch"),
            "wrong" if invalid == "token" else claimed.lease_token,
            lease_duration=timedelta(seconds=30),
        )
        is False
    ), "a stale or different owner must not extend the stored lease"
    assert store.get_processor_request(request.request_id) == claimed


def test_current_request_renewal_extends_the_current_owner_only(store):
    request = sample("renew-request", path=FAULT_PATH)
    store.enqueue_processor_request(request)
    (claimed,) = store.claim_active_processor_requests(
        "owner",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=10),
        limit=1,
    )
    assert (
        store.renew_active_processor_request(
            request.request_id,
            "owner",
            claimed.leader_epoch,
            claimed.lease_token,
            lease_duration=timedelta(seconds=30),
        )
        is True
    ), "the current lease must remain renewable"
    current = store.get_processor_request(request.request_id)
    assert current.lease_expires_at > claimed.lease_expires_at
    assert current.lease_token == claimed.lease_token


@pytest.mark.parametrize("with_state", [False, True])
def test_wakeup_listener_ignores_malformed_and_duplicate_payloads(
    store, monkeypatch, with_state
):
    import psycopg

    payloads = (
        "not-json",
        "[]",
        "1",
        '{"node":"one"}',
        '{"node":"one"}',
        '{"stop":true}',
    )
    connection = ListenerConnection(payloads=payloads)
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    stop = Event()
    received, states = [], []

    def notified(payload):
        received.append(payload)
        if payload.get("stop"):
            stop.set()

    store.run_wakeup_listener(
        WakeupChannel.WORKFLOW_DISPATCH,
        stop,
        notified,
        on_state=states.append if with_state else None,
    )
    assert received == [{"node": "one"}, {"stop": True}], (
        "malformed or repeated hints cannot become duplicate workflow wakeups"
    )
    assert states == ([True, False] if with_state else [])
    assert connection.closed, "the fake wakeup transport must close on shutdown"


@pytest.mark.parametrize("kind", ["shards", "stall", "timeout"])
def test_listener_invalid_budgets_fail_before_opening_transport(
    store, monkeypatch, kind
):
    import psycopg

    def unexpected_connection(*args, **kwargs):
        pytest.fail("invalid listener configuration attempted a connection")

    monkeypatch.setattr(psycopg, "connect", unexpected_connection)
    with pytest.raises(ValueError, match="positive"):
        if kind == "timeout":
            store.run_wakeup_listener(
                WakeupChannel.REMOTE_COMMAND,
                Event(),
                lambda value: None,
                timeout_seconds=0,
            )
        else:
            store.listen_processor_queue_notifications(
                Event(),
                "owner",
                0 if kind == "shards" else 1,
                lambda value: None,
                lambda ready, shard: None,
                stall_seconds=0 if kind == "stall" else 1,
            )
