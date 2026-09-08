"""The registry heartbeat writes when something changed or a third of the
stale window has passed -- not once a second per process.

Control-plane review 2026-09-08, F-5. ``RegionalRegistryRuntime.refresh_once``
ran every ``GPU_FAULT_REGISTRY_POLL_SECONDS`` (1 s) and upserted the process's
``regional_registry_member`` row unconditionally: ~36 processes x 1 upsert/s on
the hot ``gpu_fault_objects`` table, ~3M same-row updates a day, for a row whose
content changes only on a registry publish or a readiness flip. With
``GPU_FAULT_REGISTRY_STALE_SECONDS=90`` a heartbeat every 30 s keeps every
member well inside the stale window and cuts the write rate 30x.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalRegistryHead,
    RegionalRegistryMember,
    RegionalRegistryRevision,
    regional_registry_content_sha256,
)
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime

T0 = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
CLUSTER_A = RegionalClusterRegistration(
    cluster_id="cluster-a",
    region="us-west-2",
    hyperpod_cluster_name="hp-a",
    eks_cluster_arn="arn:aws:eks:us-west-2:1:cluster/a",
    token_sha256="a" * 64,
    agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
)


class _Clock:
    def __init__(self) -> None:
        self.now = T0

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)

    def __call__(self) -> datetime:
        return self.now


class _RegistryStore:
    def __init__(self) -> None:
        self.generation = 1
        self.registrations: list[RegionalClusterRegistration] = []
        self.head_error: Exception | None = None
        self.saved: list[RegionalRegistryMember] = []

    @property
    def digest(self) -> str:
        return regional_registry_content_sha256(self.registrations)

    def get_regional_registry_head(self) -> RegionalRegistryHead:
        if self.head_error is not None:
            raise self.head_error
        return RegionalRegistryHead(
            generation=self.generation, content_sha256=self.digest
        )

    def get_regional_registry_revision(self, generation: int):
        return RegionalRegistryRevision(
            generation=generation,
            content_sha256=self.digest,
            registrations=list(self.registrations),
            previous_generation=generation - 1 if generation > 1 else None,
            reason="test",
        )

    def save_regional_registry_member(self, member: RegionalRegistryMember):
        self.saved.append(member)
        return member


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def store() -> _RegistryStore:
    return _RegistryStore()


@pytest.fixture
def runtime(store, clock) -> RegionalRegistryRuntime:
    return RegionalRegistryRuntime(
        store,
        member_id="pod-a:1",
        service_role="worker",
        release_id="release-1",
        poll_seconds=1.0,
        stale_seconds=90.0,
        now=clock,
    )


def test_an_unchanged_member_is_written_once_per_third_of_the_stale_window(
    runtime, store, clock
) -> None:
    assert runtime.refresh_once(), "the first refresh must succeed"
    assert len(store.saved) == 1

    for _ in range(29):
        clock.advance(1.0)
        assert runtime.refresh_once(), (
            "an unchanged member still refreshes successfully"
        )
    assert len(store.saved) == 1, "an unchanged heartbeat was re-written every poll"

    clock.advance(1.0)  # t = 30 s = stale_seconds / 3
    assert runtime.refresh_once(), "the refresh at stale_seconds / 3 must succeed"
    assert len(store.saved) == 2
    assert store.saved[-1].last_seen_at == clock.now
    assert store.saved[-1].ready is True


def test_a_readiness_flip_is_written_immediately_in_both_directions(
    runtime, store, clock
) -> None:
    runtime.refresh_once()
    clock.advance(1.0)
    store.head_error = RuntimeError("registry unreachable")

    assert not runtime.refresh_once(), "an unreachable registry must report not ready"

    assert len(store.saved) == 2
    assert store.saved[-1].ready is False
    assert "registry unreachable" in (store.saved[-1].error or "")

    clock.advance(1.0)
    assert not runtime.refresh_once(), "the registry is still unreachable"
    assert len(store.saved) == 2, "the same failure was re-written every poll"

    clock.advance(1.0)
    store.head_error = None
    assert runtime.refresh_once(), "a recovered registry must report ready again"
    assert len(store.saved) == 3 and store.saved[-1].ready is True


def test_a_new_registry_generation_is_written_immediately(
    runtime, store, clock
) -> None:
    runtime.refresh_once()
    clock.advance(1.0)
    store.generation = 2
    store.registrations = [CLUSTER_A]

    assert runtime.refresh_once(), "a new generation must refresh successfully"

    assert len(store.saved) == 2
    assert store.saved[-1].generation == 2
    assert store.saved[-1].content_sha256 == store.digest


def test_the_heartbeat_interval_keeps_a_member_inside_the_stale_window() -> None:
    assert RegionalRegistryRuntime.heartbeat_interval_seconds(90.0) == 30.0
    # Never slower than the poll, never so slow the member looks stale.
    assert RegionalRegistryRuntime.heartbeat_interval_seconds(2.0) == pytest.approx(
        2.0 / 3
    )
