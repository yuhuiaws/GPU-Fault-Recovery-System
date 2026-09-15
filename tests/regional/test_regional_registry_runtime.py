from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
    member_serves_revision,
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


def test_a_required_member_that_left_the_fleet_does_not_block_convergence() -> None:
    """A publish-time straggler that terminates without acking must not deadlock.

    required_member_ids is a snapshot of active_registry_member_ids at publish,
    so a control-plane process still heartbeating while it drained under a
    rolling restart can be captured in it and then terminate without ever
    re-reading the new head. Once it goes stale it has left the fleet by the
    registry's own liveness rule and serves no traffic, so convergence must
    ignore it rather than run its whole window out -- while a member that is
    still heartbeating and merely behind must still hold the barrier.
    """

    revision = RegionalRegistryRevision.build(
        generation=5,
        registrations=[registration("cluster-a")],
        previous_generation=4,
        required_member_ids=["live/process", "drained/process"],
        reason="join cluster-a",
        created_at=NOW + timedelta(seconds=1),
    )
    now = NOW + timedelta(seconds=120)

    def member(
        member_id: str, *, generation: int, last_seen: datetime
    ) -> RegionalRegistryMember:
        at_revision = generation == revision.generation
        return RegionalRegistryMember(
            member_id=member_id,
            service_role="control-worker",
            release_id="release-a",
            generation=generation,
            content_sha256=revision.content_sha256 if at_revision else "b" * 64,
            ready=True,
            started_at=NOW,
            last_seen_at=last_seen,
        )

    live = member("live/process", generation=5, last_seen=now)
    # Last heartbeat 100s back, past the 90s window: gone, never acked gen 5.
    drained = member(
        "drained/process", generation=4, last_seen=now - timedelta(seconds=100)
    )

    assert "drained/process" in revision.required_member_ids
    assert active_registry_member_ids(
        [live, drained], observed_at=now, stale_seconds=90
    ) == ["live/process"]
    assert member_serves_revision(live, revision), live
    assert not member_serves_revision(drained, revision), drained
    # Converges on the live member alone; the departed straggler is skipped.
    assert registry_revision_converged(
        revision, [live, drained], observed_at=now, stale_seconds=90
    ), (live, drained)
    # A member still heartbeating but merely behind keeps holding the barrier.
    behind = member("live/process", generation=4, last_seen=now)
    assert not registry_revision_converged(
        revision, [behind, drained], observed_at=now, stale_seconds=90
    ), (behind, drained)


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
