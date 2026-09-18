from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timedelta
from threading import Event
from typing import Any, Mapping

import pytest

from gpu_fault import regional_registry_runtime as registry_runtime
from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalRegistryHead,
    RegionalRegistryMember,
    RegionalRegistryRevision,
)
from gpu_fault.regional_registry_runtime import (
    RegionalRegistryRuntime,
    RegionalRegistrySnapshot,
)
from gpu_fault.store import InMemoryStore
from tests.regional.test_regional_registry_runtime import NOW, registration, runtime


def prepared_runtime() -> tuple[
    InMemoryStore, RegionalRegistryRuntime, ContextVar[datetime]
]:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a"))
    loaded = runtime(store, [NOW])
    clock = ContextVar("registry-refresh-clock", default=NOW)
    loaded.now = clock.get
    return store, loaded, clock


def publish_next(store: InMemoryStore) -> RegionalRegistryRevision:
    revision = RegionalRegistryRevision.build(
        generation=2,
        registrations=[registration("cluster-b")],
        previous_generation=1,
        required_member_ids=["pod-a/process-a"],
        reason="synthetic overlapping refresh",
        created_at=NOW + timedelta(seconds=2),
    )
    store.publish_regional_registry_revision(revision, expected_generation=1)
    return revision


def pause_old_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    older: ContextVar[bool],
    paused: Event,
    resume: Event,
) -> None:
    def snapshot(
        generation: int,
        content_sha256: str,
        registrations: Mapping[str, RegionalClusterRegistration],
    ) -> RegionalRegistrySnapshot:
        candidate = RegionalRegistrySnapshot(
            generation=generation,
            content_sha256=content_sha256,
            registrations=registrations,
        )
        if older.get():
            paused.set()
            assert resume.wait(5), "the test must release its old snapshot candidate"
        return candidate

    monkeypatch.setattr(registry_runtime, "RegionalRegistrySnapshot", snapshot)


@pytest.mark.parametrize("new_generation", [False, True], ids=["same", "new"])
@pytest.mark.parametrize(
    "old_second", [1, 2, 3], ids=["increasing-clock", "equal-clock", "reversed-clock"]
)
def test_old_snapshot_apply_cannot_overwrite_a_newer_durable_ack(
    monkeypatch: pytest.MonkeyPatch, new_generation: bool, old_second: int
) -> None:
    store, loaded, clock = prepared_runtime()
    older = ContextVar("older-refresh", default=False)
    paused = Event()
    resume = Event()
    pause_old_snapshot(monkeypatch, older=older, paused=paused, resume=resume)

    def old_refresh() -> bool:
        older.set(True)
        clock.set(NOW + timedelta(seconds=old_second))
        return loaded.refresh_once(raise_on_failure=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(old_refresh)
        try:
            assert paused.wait(5), (
                "the old candidate must pause immediately before apply"
            )
            if new_generation:
                publish_next(store)
            clock.set(NOW + timedelta(seconds=2))
            assert loaded.refresh_once(raise_on_failure=True), (
                "the newer refresh must complete while the old candidate is paused"
            )
            current_snapshot = loaded.snapshot()
            current_status = loaded.status()
            current_members = store.list_regional_registry_members()
            assert current_members[0].generation == (2 if new_generation else 1)
            assert loaded.is_ready(), (
                "the newer durable ACK must authorize its snapshot"
            )
        finally:
            resume.set()
        old_result = first.result(timeout=5)

    assert loaded.snapshot() is current_snapshot, (
        "an old candidate cannot replace the snapshot proved by the newer ACK"
    )
    assert loaded.status() == current_status, (
        "old completion cannot rewind runtime state"
    )
    assert store.list_regional_registry_members() == current_members
    assert loaded.get("cluster-b" if new_generation else "cluster-a") is not None
    assert old_result is False, "a superseded snapshot candidate was not applied"


@pytest.mark.parametrize("raise_on_failure", [False, True], ids=["return", "raise"])
@pytest.mark.parametrize(
    "failure", ["head-io", "head-invalid", "revision-io", "revision-invalid"]
)
def test_old_read_failure_cannot_poison_newer_target_or_integrity_state(
    monkeypatch: pytest.MonkeyPatch, failure: str, raise_on_failure: bool
) -> None:
    store, loaded, clock = prepared_runtime()
    older = ContextVar("older-refresh", default=False)
    paused = Event()
    resume = Event()
    read_head = store.get_regional_registry_head
    read_revision = store.get_regional_registry_revision

    def fail_old_read() -> None:
        paused.set()
        assert resume.wait(5), "the test must release its delayed old read"
        if failure.endswith("invalid"):
            raise ValueError("synthetic invalid old registry data")
        raise ConnectionError("synthetic delayed old registry read failure")

    def head() -> RegionalRegistryHead:
        if older.get() and failure.startswith("head"):
            fail_old_read()
        return read_head()

    def revision(generation: int) -> RegionalRegistryRevision:
        if older.get() and failure.startswith("revision"):
            fail_old_read()
        return read_revision(generation)

    monkeypatch.setattr(store, "get_regional_registry_head", head)
    monkeypatch.setattr(store, "get_regional_registry_revision", revision)

    def old_refresh() -> bool:
        older.set(True)
        clock.set(NOW + timedelta(seconds=1))
        return loaded.refresh_once(raise_on_failure=raise_on_failure)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(old_refresh)
        try:
            assert paused.wait(5), (
                "the old read must be in flight before the newer read"
            )
            publish_next(store)
            clock.set(NOW + timedelta(seconds=2))
            assert loaded.refresh_once(raise_on_failure=True), (
                "the newer ACK must persist"
            )
            current_snapshot = loaded.snapshot()
            current_status = loaded.status()
            current_members = store.list_regional_registry_members()
        finally:
            resume.set()
        if raise_on_failure:
            error_type = ValueError if failure.endswith("invalid") else ConnectionError
            with pytest.raises(error_type):
                first.result(timeout=5)
        else:
            assert first.result(timeout=5) is False, (
                "the old call still reports failure"
            )

    assert loaded.snapshot() is current_snapshot
    assert loaded.status() == current_status, (
        "a superseded failure cannot rewrite the newer target, error or integrity state"
    )
    assert store.list_regional_registry_members() == current_members
    assert loaded.is_ready(), "the newer validated revision remains ready"


def test_old_revision_read_is_not_misclassified_as_a_sequential_backend_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, loaded, clock = prepared_runtime()
    older = ContextVar("older-refresh", default=False)
    paused = Event()
    resume = Event()
    read_revision = store.get_regional_registry_revision

    def revision(generation: int) -> RegionalRegistryRevision:
        candidate = read_revision(generation)
        if older.get():
            paused.set()
            assert resume.wait(5), "the test must release its old revision read"
        return candidate

    monkeypatch.setattr(store, "get_regional_registry_revision", revision)

    def old_refresh() -> bool:
        older.set(True)
        clock.set(NOW + timedelta(seconds=1))
        return loaded.refresh_once(raise_on_failure=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(old_refresh)
        try:
            assert paused.wait(5), "the old revision read must be in flight"
            publish_next(store)
            clock.set(NOW + timedelta(seconds=2))
            assert loaded.refresh_once(raise_on_failure=True), (
                "the newer head must apply"
            )
            current = loaded.status()
        finally:
            resume.set()
        assert first.result(timeout=5) is False, "only the obsolete read is discarded"

    assert loaded.status() == current, "the newer state must not be marked as regressed"
    assert loaded.is_ready(), "the backend did not regress in this overlapping read"


@pytest.mark.parametrize(
    "failure",
    [
        "head-invalid",
        "same-head-integrity",
        "new-head-integrity",
        "new-head-unavailable",
    ],
)
def test_old_success_cannot_clear_a_newer_failed_refresh(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store, loaded, clock = prepared_runtime()
    older = ContextVar("older-refresh", default=False)
    paused = Event()
    resume = Event()
    pause_old_snapshot(monkeypatch, older=older, paused=paused, resume=resume)
    read_head = store.get_regional_registry_head
    read_revision = store.get_regional_registry_revision

    def invalid_head() -> RegionalRegistryHead:
        raise ValueError("synthetic invalid newer head")

    def failed_revision(generation: int) -> RegionalRegistryRevision:
        if failure == "new-head-unavailable":
            raise ConnectionError("synthetic new head revision unavailable")
        return RegionalRegistryRevision.build(
            generation=generation,
            registrations=[registration("cluster-c")],
            previous_generation=None,
            required_member_ids=[],
            reason="synthetic inconsistent same-head revision",
            created_at=NOW,
        )

    def old_refresh() -> bool:
        older.set(True)
        clock.set(NOW + timedelta(seconds=1))
        return loaded.refresh_once(raise_on_failure=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(old_refresh)
        try:
            assert paused.wait(5), "the old valid candidate must pause before apply"
            if failure.startswith("new-head"):
                publish_next(store)
            if failure == "head-invalid":
                monkeypatch.setattr(store, "get_regional_registry_head", invalid_head)
            monkeypatch.setattr(
                store, "get_regional_registry_revision", failed_revision
            )
            clock.set(NOW + timedelta(seconds=2))
            assert loaded.refresh_once() is False, "the newer read must fail closed"
            failed_snapshot = loaded.snapshot()
            failed_status = loaded.status()
            failed_members = store.list_regional_registry_members()
            assert not loaded.is_ready(), (
                "the newer error invalidates traffic readiness"
            )
        finally:
            resume.set()
        old_result = first.result(timeout=5)

    assert loaded.snapshot() is failed_snapshot
    assert loaded.status() == failed_status, "old success cannot heal a newer failure"
    assert store.list_regional_registry_members() == failed_members
    assert not loaded.is_ready(), "newer target/integrity rejection must remain latched"
    assert old_result is False, "the superseded old candidate was not applied"

    monkeypatch.setattr(store, "get_regional_registry_head", read_head)
    monkeypatch.setattr(store, "get_regional_registry_revision", read_revision)
    assert loaded.refresh_once(raise_on_failure=True), (
        "a new validated read may recover"
    )
    assert loaded.is_ready(), "only the fresh validated recovery clears the rejection"


@pytest.mark.parametrize("old_failed", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("old_second", [2, 3], ids=["equal-clock", "reversed-clock"])
def test_superseded_heartbeat_cannot_overwrite_a_newer_persisted_ack(
    monkeypatch: pytest.MonkeyPatch, old_failed: bool, old_second: int
) -> None:
    store, loaded, clock = prepared_runtime()
    older = ContextVar("older-refresh", default=False)
    paused = Event()
    resume = Event()
    read_head = store.get_regional_registry_head

    def head() -> RegionalRegistryHead:
        if older.get() and old_failed:
            raise ConnectionError("synthetic old refresh failure before heartbeat")
        return read_head()

    def member(**kwargs: Any) -> RegionalRegistryMember:
        candidate = RegionalRegistryMember(**kwargs)
        if older.get():
            paused.set()
            assert resume.wait(5), "the test must release its old heartbeat candidate"
        return candidate

    monkeypatch.setattr(store, "get_regional_registry_head", head)
    monkeypatch.setattr(registry_runtime, "RegionalRegistryMember", member)

    def old_refresh() -> bool:
        older.set(True)
        clock.set(NOW + timedelta(seconds=old_second))
        return loaded.refresh_once(raise_on_failure=not old_failed)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(old_refresh)
        try:
            assert paused.wait(5), (
                "the old state applies before its heartbeat is paused"
            )
            publish_next(store)
            clock.set(NOW + timedelta(seconds=2))
            assert loaded.refresh_once(raise_on_failure=True), (
                "the newer ACK must persist"
            )
            current = loaded.status()
            members = store.list_regional_registry_members()
            assert members[0].generation == 2
            assert members[0].ready is True
        finally:
            resume.set()
        assert first.result(timeout=5) is (not old_failed)

    assert loaded.status() == current, "the newer runtime state remains authoritative"
    assert store.list_regional_registry_members() == members, (
        "request ordering, not equal or reversed timestamps, protects the newer ACK"
    )
    assert loaded.is_ready(), "the retained ACK still proves the current snapshot"


@pytest.mark.parametrize("raise_on_failure", [False, True], ids=["return", "raise"])
def test_sequential_backend_generation_regression_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, raise_on_failure: bool
) -> None:
    store, loaded, clock = prepared_runtime()
    old_head = store.get_regional_registry_head()
    read_head = store.get_regional_registry_head
    publish_next(store)
    clock.set(NOW + timedelta(seconds=2))
    assert loaded.refresh_once(raise_on_failure=True), "generation two must be current"
    current_snapshot = loaded.snapshot()
    monkeypatch.setattr(store, "get_regional_registry_head", lambda: old_head)

    if raise_on_failure:
        with pytest.raises(RuntimeError, match="generation moved backward"):
            loaded.refresh_once(raise_on_failure=True)
    else:
        assert loaded.refresh_once() is False, (
            "a real later read of generation one fails"
        )

    assert loaded.snapshot() is current_snapshot, (
        "the last validated snapshot is retained"
    )
    assert loaded.status()["target_generation"] == 1
    assert "generation moved backward" in str(loaded.status()["error"])
    assert not loaded.is_ready(), "equal clock values do not hide a backend regression"
    (member,) = store.list_regional_registry_members()
    assert member.generation == 2
    assert member.ready is False

    monkeypatch.setattr(store, "get_regional_registry_head", read_head)
    assert loaded.refresh_once(raise_on_failure=True), (
        "the actual current head recovers"
    )
    assert loaded.is_ready(), "the restored backend must be validated before reuse"
