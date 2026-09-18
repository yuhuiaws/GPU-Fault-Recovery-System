from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from gpu_fault.channel_registry import CHANNEL_REGISTRY
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryHead,
    RegionalRegistryMember,
    RegionalRegistryRevision,
)
from gpu_fault.regional_registry_runtime import (
    RegionalRegistryRuntime,
    active_registry_member_ids,
    regional_cluster_request_allowed,
    registry_revision_converged,
)
from gpu_fault.store import InMemoryStore

NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def registration(cluster_id: str) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        hyperpod_cluster_name=f"hyperpod-{cluster_id}",
        eks_cluster_arn=(f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}"),
        token_sha256="a" * 64,
        allowed_namespaces=["training"],
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        created_at=NOW,
        updated_at=NOW,
    )


def runtime(
    store: InMemoryStore, clock: list[datetime], *, member_id: str = "pod-a/process-a"
) -> RegionalRegistryRuntime:
    return RegionalRegistryRuntime.bootstrap(
        store,
        member_id=member_id,
        service_role="ingress",
        release_id="release-a",
        poll_seconds=1,
        stale_seconds=5,
        now=lambda: clock[0],
    )


def test_runtime_bootstrap_publishes_generation_one_and_acks() -> None:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]

    loaded = runtime(store, clock)

    assert loaded.is_ready(), loaded.status()
    assert loaded.snapshot().generation == 1
    assert list(loaded.snapshot().registrations) == ["cluster-a"]
    assert store.get_regional_registry_head().generation == 1
    members = store.list_regional_registry_members()
    assert len(members) == 1
    assert members[0].ready, members[0]
    assert members[0].generation == 1


def test_runtime_atomically_switches_to_published_revision() -> None:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]
    loaded = runtime(store, clock)
    original = loaded.snapshot()
    second = RegionalRegistryRevision.build(
        generation=2,
        registrations=[registration("cluster-a"), registration("cluster-b")],
        previous_generation=1,
        required_member_ids=["pod-a/process-a"],
        reason="join cluster-b",
        created_at=NOW + timedelta(seconds=1),
    )
    store.publish_regional_registry_revision(second, expected_generation=1)
    clock[0] += timedelta(seconds=1)

    assert list(original.registrations) == ["cluster-a"]
    assert loaded.refresh_once(), loaded.status()
    assert loaded.snapshot().generation == 2
    assert list(loaded.snapshot().registrations) == ["cluster-a", "cluster-b"]
    with pytest.raises(TypeError):
        loaded.snapshot().registrations["cluster-c"] = registration("cluster-c")  # type: ignore[index]


def test_runtime_keeps_previous_snapshot_and_fails_readiness_on_bad_head() -> None:
    class BadHeadStore:
        def __init__(self, target: InMemoryStore) -> None:
            self.target = target

        def get_regional_registry_head(self):
            return RegionalRegistryHead(
                generation=1,
                content_sha256="f" * 64,
                updated_at=NOW + timedelta(seconds=1),
            )

        def __getattr__(self, name: str):
            return getattr(self.target, name)

    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]
    loaded = runtime(store, clock)
    previous = loaded.snapshot()
    loaded.store = BadHeadStore(store)
    clock[0] += timedelta(seconds=1)

    assert not loaded.refresh_once(), loaded.status()
    assert loaded.snapshot() is previous
    assert not loaded.is_ready(), loaded.status()
    assert "digest mismatch" in str(loaded.status()["error"])
    member = store.list_regional_registry_members()[0]
    assert not member.ready, member
    assert member.generation == 1


def test_runtime_fails_readiness_when_head_watch_becomes_stale() -> None:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]
    loaded = runtime(store, clock)

    clock[0] += timedelta(seconds=6)

    assert not loaded.is_ready(), loaded.status()
    assert loaded.refresh_once(), loaded.status()
    assert loaded.is_ready(), loaded.status()


def test_revision_barrier_requires_every_named_member_at_same_digest() -> None:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]
    first = runtime(store, clock, member_id="pod-a/process-a")
    second = runtime(store, clock, member_id="pod-b/process-b")
    revision = RegionalRegistryRevision.build(
        generation=2,
        registrations=[registration("cluster-a")],
        previous_generation=1,
        required_member_ids=["pod-a/process-a", "pod-b/process-b"],
        reason="same content new generation",
        created_at=NOW + timedelta(seconds=1),
    )
    store.publish_regional_registry_revision(revision, expected_generation=1)
    clock[0] += timedelta(seconds=1)

    first.refresh_once()
    members = store.list_regional_registry_members()
    assert not registry_revision_converged(
        revision, members, observed_at=clock[0], stale_seconds=5
    ), members
    second.refresh_once()
    members = store.list_regional_registry_members()
    assert registry_revision_converged(
        revision, members, observed_at=clock[0], stale_seconds=5
    ), members
    assert active_registry_member_ids(
        members, observed_at=clock[0], stale_seconds=5
    ) == ["pod-a/process-a", "pod-b/process-b"]


@pytest.mark.parametrize(
    ("state", "path", "method", "allowed"),
    [
        (
            RegionalClusterLifecycle.PENDING,
            "/v1/regional/executors/readiness",
            "POST",
            True,
        ),
        (
            RegionalClusterLifecycle.PENDING,
            "/v1/regional/executors/claim",
            "POST",
            False,
        ),
        # A join waits for collector readiness before activation; the events
        # that readiness is built from must get through while PENDING.
        (
            RegionalClusterLifecycle.PENDING,
            "/v1/collector-events/nvidia-kernel",
            "POST",
            True,
        ),
        (
            RegionalClusterLifecycle.FAILED,
            "/v1/collector-events/nvidia-kernel",
            "POST",
            False,
        ),
        (
            RegionalClusterLifecycle.DRAINING,
            "/v1/regional/executors/command-a/result",
            "POST",
            True,
        ),
        (
            RegionalClusterLifecycle.DRAINING,
            "/v1/collector-events/nvidia-kernel",
            "POST",
            False,
        ),
        (
            RegionalClusterLifecycle.REVOKED,
            "/v1/regional/executors/readiness",
            "POST",
            False,
        ),
        (
            RegionalClusterLifecycle.FAILED,
            "/v1/regional/executors/claim",
            "POST",
            False,
        ),
        (RegionalClusterLifecycle.FAILED, "/v1/fleet/agents/heartbeat", "POST", True),
        (
            RegionalClusterLifecycle.ROLLED_BACK,
            "/v1/fleet/agents/heartbeat",
            "POST",
            False,
        ),
    ],
)
def test_cluster_lifecycle_route_policy(
    state: RegionalClusterLifecycle, path: str, method: str, allowed: bool
) -> None:
    item = registration("cluster-a").model_copy(update={"lifecycle_state": state})

    assert regional_cluster_request_allowed(item, path=path, method=method) is allowed


@pytest.mark.parametrize("path", sorted(CHANNEL_REGISTRY))
@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_pending_ingestion_is_limited_to_registered_post_channels(
    path: str, method: str
):
    item = registration("cluster-a").model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
    )

    assert regional_cluster_request_allowed(item, path=path, method=method) is (
        method == "POST"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/v1/collector-events/unregistered",
        "/v1/collector-events/nvidia-kernel/claim",
        "/v1/regional/executors/claim",
        "/v1/regional/executors/hyperpod-submissions",
        "/v1/attempts/terminal",
    ],
)
def test_pending_readiness_does_not_authorize_unregistered_or_action_posts(path: str):
    item = registration("cluster-a").model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
    )

    assert not regional_cluster_request_allowed(item, path=path, method="POST"), (
        "PENDING ingestion must not authorize unregistered channels or action posts"
    )


# --- A-7: one transient refresh error must not fail readiness -----------------


class _FlakyOnceStore:
    """Aurora closes one connection; the head has not moved."""

    def __init__(self, target: InMemoryStore) -> None:
        self.target = target
        self.failures_left = 1

    def get_regional_registry_head(self):
        if self.failures_left:
            self.failures_left -= 1
            raise ConnectionError("Remote end closed connection without response")
        return self.target.get_regional_registry_head()

    def __getattr__(self, name: str):
        return getattr(self.target, name)


def test_a_transient_refresh_error_keeps_a_fresh_snapshot_ready() -> None:
    """``is_ready`` required ``_last_error is None``, so one closed connection
    (a ~1/min baseline on this control plane) made ``/healthz`` and every
    cluster-token request answer 503 for up to a second, per process, and
    ``GPU_FAULT_REGISTRY_STALE_SECONDS`` never applied to the error path."""

    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]
    loaded = runtime(store, clock)
    loaded.store = _FlakyOnceStore(store)
    clock[0] += timedelta(seconds=1)

    assert not loaded.refresh_once(), loaded.status()
    assert loaded.status()["error"], "the error must still be reported"
    assert loaded.is_ready(), "a fresh snapshot with an unchanged head is ready"

    # Past the stale window the error path fails closed like the silent one.
    clock[0] += timedelta(seconds=5)
    assert not loaded.is_ready(), loaded.status()

    # The next successful refresh clears the error.
    clock[0] = NOW + timedelta(seconds=2)
    assert loaded.refresh_once(), loaded.status()
    assert loaded.is_ready()
    assert loaded.status()["error"] is None


def test_a_head_digest_mismatch_still_fails_readiness_at_once() -> None:
    """A-7 narrows the error rule to transient errors; a divergent head is not
    one -- serving a snapshot the head disagrees with is what the check is for."""

    class BadHeadStore:
        def __init__(self, target: InMemoryStore) -> None:
            self.target = target

        def get_regional_registry_head(self):
            return RegionalRegistryHead(
                generation=1,
                content_sha256="f" * 64,
                updated_at=NOW + timedelta(seconds=1),
            )

        def __getattr__(self, name: str):
            return getattr(self.target, name)

    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    clock = [NOW]
    loaded = runtime(store, clock)
    loaded.store = BadHeadStore(store)
    clock[0] += timedelta(seconds=1)

    assert not loaded.refresh_once(), loaded.status()
    assert not loaded.is_ready(), loaded.status()


def test_refresh_success_cannot_extend_readiness_past_a_persisted_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    clock = [NOW]
    loaded = runtime(store, clock)
    save_member = store.save_regional_registry_member

    def unavailable(_member: RegionalRegistryMember) -> None:
        raise ConnectionError("synthetic heartbeat write outage")

    monkeypatch.setattr(store, "save_regional_registry_member", unavailable)
    clock[0] += timedelta(seconds=5)
    assert not loaded.refresh_once(), "the heartbeat write must fail"
    assert loaded.is_ready(), "the last durable heartbeat is valid at its boundary"
    clock[0] += timedelta(microseconds=1)
    assert not loaded.refresh_once(), "successful reads do not repair a failed write"
    assert loaded.status()["last_successful_refresh"] == clock[0]
    assert (
        active_registry_member_ids(
            store.list_regional_registry_members(),
            observed_at=clock[0],
            stale_seconds=5,
        )
        == []
    ), "the persisted row is now outside the fleet's liveness window"
    assert not loaded.is_ready(), (
        "a process removed from the convergence barrier must stop serving traffic"
    )

    monkeypatch.setattr(store, "save_regional_registry_member", save_member)
    assert loaded.refresh_once(), "a successful durable heartbeat repairs readiness"
    assert loaded.is_ready(), "validated read and persisted heartbeat are both fresh"


def test_throttled_heartbeat_and_later_refresh_share_a_safe_readiness_deadline() -> (
    None
):
    store = InMemoryStore()
    clock = [NOW]
    loaded = runtime(store, clock)
    clock[0] += timedelta(seconds=1)

    assert loaded.refresh_once(), "the unchanged snapshot is still refreshed"
    (persisted,) = store.list_regional_registry_members()
    assert persisted.last_seen_at == NOW, "the unchanged heartbeat is throttled"
    assert loaded.status()["last_successful_refresh"] == clock[0]

    clock[0] = NOW + timedelta(seconds=5, microseconds=1)
    assert not loaded.is_ready(), (
        "a newer local refresh cannot outlive the published heartbeat lease"
    )


def test_registry_readiness_fails_closed_if_its_clock_moves_before_the_heartbeat() -> (
    None
):
    store = InMemoryStore()
    clock = [NOW]
    loaded = runtime(store, clock)
    clock[0] -= timedelta(microseconds=1)

    assert not loaded.is_ready(), "future refresh and heartbeat times are not fresh"


def test_a_loaded_snapshot_without_a_persisted_member_never_authorizes_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    clock = [NOW]
    runtime(store, clock)
    loaded = RegionalRegistryRuntime(
        store,
        member_id="unpublished/process",
        service_role="ingress",
        release_id="release-a",
        now=lambda: clock[0],
    )

    def unavailable(_member: RegionalRegistryMember) -> None:
        raise ConnectionError("synthetic initial heartbeat write outage")

    monkeypatch.setattr(store, "save_regional_registry_member", unavailable)

    assert not loaded.refresh_once(), "the first heartbeat never persisted"
    assert loaded.snapshot().generation == 1, "the validated head is still cached"
    assert not loaded.is_ready(), "a cached head is not a published membership lease"


def test_a_delayed_refresh_failure_cannot_backdate_a_newer_durable_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    loaded = runtime(store, [NOW])
    clock = ContextVar("registry-clock", default=NOW)
    loaded.now = clock.get
    read_head = store.get_regional_registry_head
    earlier = NOW + timedelta(seconds=1)
    later = NOW + timedelta(seconds=2)
    reading = Event()
    release_read = Event()

    def delayed_head() -> RegionalRegistryHead:
        if clock.get() == earlier:
            reading.set()
            assert release_read.wait(5), "the test must release its delayed read"
            raise ConnectionError("synthetic delayed head read failure")
        return read_head()

    monkeypatch.setattr(store, "get_regional_registry_head", delayed_head)

    def refresh() -> bool:
        clock.set(earlier)
        return loaded.refresh_once()

    with ThreadPoolExecutor(max_workers=1) as pool:
        delayed = pool.submit(refresh)
        try:
            assert reading.wait(5), "the earlier refresh must reach its read"
            clock.set(later)
            assert loaded.refresh_once(), "the newer refresh persists its heartbeat"
        finally:
            release_read.set()
        assert delayed.result(timeout=5) is False, (
            "the old read failure remains visible"
        )

    (member,) = store.list_regional_registry_members()
    assert member.last_seen_at == later, "a delayed failure cannot backdate the row"
    assert member.ready is True, "the newer successful ACK remains authoritative"
    assert loaded.is_ready(), "the newer validated refresh and heartbeat stay usable"


def test_concurrent_refreshes_serialize_durable_heartbeat_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    loaded = runtime(store, [NOW])
    clock = ContextVar("registry-clock", default=NOW)
    loaded.now = clock.get
    read_head = store.get_regional_registry_head
    save_member = store.save_regional_registry_member
    earlier = NOW + timedelta(seconds=2)
    later = NOW + timedelta(seconds=4)
    first_writing = Event()
    second_reading = Event()
    second_writing = Event()
    release_write = Event()

    def head() -> RegionalRegistryHead:
        if clock.get() == later:
            second_reading.set()
        return read_head()

    def delayed_write(member: RegionalRegistryMember) -> RegionalRegistryMember:
        if member.last_seen_at == earlier:
            first_writing.set()
            assert release_write.wait(5), "the test must release its heartbeat write"
        if member.last_seen_at == later:
            second_writing.set()
        return save_member(member)

    monkeypatch.setattr(store, "get_regional_registry_head", head)
    monkeypatch.setattr(store, "save_regional_registry_member", delayed_write)

    def refresh(observed_at: datetime) -> bool:
        clock.set(observed_at)
        return loaded.refresh_once()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(refresh, earlier)
        try:
            assert first_writing.wait(5), "the first heartbeat must enter the store"
            second = pool.submit(refresh, later)
            assert second_reading.wait(5), "the second refresh must run concurrently"
            overlapped = second_writing.wait(0.1)
        finally:
            release_write.set()
        assert first.result(timeout=5), "the first owned refresh must finish"
        assert second.result(timeout=5), "the second owned refresh must finish"

    assert not overlapped, "writes and their acknowledgements must remain serialized"
    (member,) = store.list_regional_registry_members()
    assert member.last_seen_at == later, "the latest durable heartbeat must win"
    observed_at = later + timedelta(seconds=5)
    assert loaded.is_ready(observed_at), (
        "traffic and the durable lease share a deadline"
    )
    assert not loaded.is_ready(observed_at + timedelta(microseconds=1)), (
        "the process must stop serving when its durable heartbeat expires"
    )
