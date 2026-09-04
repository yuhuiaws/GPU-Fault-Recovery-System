from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event, RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping

from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryMember,
    RegionalRegistryRevision,
)
from gpu_fault.store import NotFoundError


@dataclass(frozen=True)
class RegionalRegistrySnapshot:
    generation: int
    content_sha256: str
    registrations: Mapping[str, RegionalClusterRegistration]

    def get(self, cluster_id: str) -> RegionalClusterRegistration | None:
        return self.registrations.get(cluster_id)


class RegionalRegistryRuntime:
    """Process-local immutable snapshot backed by a durable registry head."""

    def __init__(
        self,
        store: Any,
        *,
        member_id: str,
        service_role: str,
        release_id: str,
        poll_seconds: float = 1.0,
        stale_seconds: float = 10.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("regional registry poll interval must be positive")
        if stale_seconds <= poll_seconds:
            raise ValueError(
                "regional registry stale threshold must exceed poll interval"
            )
        self.store = store
        self.member_id = member_id
        self.service_role = service_role
        self.release_id = release_id
        self.poll_seconds = poll_seconds
        self.stale_seconds = stale_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.started_at = self.now()
        self._lock = RLock()
        self._snapshot: RegionalRegistrySnapshot | None = None
        self._target_generation = 0
        self._target_content_sha256 = "0" * 64
        self._last_successful_refresh: datetime | None = None
        self._last_error: str | None = "regional registry has not loaded"

    @classmethod
    def bootstrap(
        cls,
        store: Any,
        *,
        member_id: str,
        service_role: str,
        release_id: str,
        poll_seconds: float = 1.0,
        stale_seconds: float = 10.0,
        now: Callable[[], datetime] | None = None,
    ) -> RegionalRegistryRuntime:
        observed = now or (lambda: datetime.now(timezone.utc))
        try:
            store.get_regional_registry_head()
        except NotFoundError:
            registrations = store.list_regional_clusters()
            revision = RegionalRegistryRevision.build(
                generation=1,
                registrations=registrations,
                previous_generation=None,
                required_member_ids=[],
                reason="startup bootstrap from validated regional registry",
                created_at=observed(),
            )
            try:
                store.publish_regional_registry_revision(
                    revision,
                    expected_generation=0,
                )
            except ValueError:
                # Another process may have won the generation-1 bootstrap.
                store.get_regional_registry_head()
        runtime = cls(
            store,
            member_id=member_id,
            service_role=service_role,
            release_id=release_id,
            poll_seconds=poll_seconds,
            stale_seconds=stale_seconds,
            now=observed,
        )
        runtime.refresh_once(raise_on_failure=True)
        return runtime

    def snapshot(self) -> RegionalRegistrySnapshot:
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError("regional registry snapshot is unavailable")
            return self._snapshot

    def get(self, cluster_id: str) -> RegionalClusterRegistration | None:
        return self.snapshot().get(cluster_id)

    def is_ready(self, observed_at: datetime | None = None) -> bool:
        observed = observed_at or self.now()
        with self._lock:
            return (
                self._snapshot is not None
                and self._last_error is None
                and self._snapshot.generation == self._target_generation
                and self._snapshot.content_sha256 == self._target_content_sha256
                and self._last_successful_refresh is not None
                and observed - self._last_successful_refresh
                <= timedelta(seconds=self.stale_seconds)
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot
            return {
                "member_id": self.member_id,
                "service_role": self.service_role,
                "release_id": self.release_id,
                "generation": snapshot.generation if snapshot is not None else 0,
                "content_sha256": (
                    snapshot.content_sha256 if snapshot is not None else "0" * 64
                ),
                "target_generation": self._target_generation,
                "target_content_sha256": self._target_content_sha256,
                "ready": self.is_ready(),
                "last_successful_refresh": self._last_successful_refresh,
                "error": self._last_error,
            }

    def refresh_once(self, *, raise_on_failure: bool = False) -> bool:
        observed = self.now()
        try:
            head = self.store.get_regional_registry_head()
            revision = self.store.get_regional_registry_revision(head.generation)
            if revision.generation != head.generation:
                raise RuntimeError("regional registry head generation mismatch")
            if revision.content_sha256 != head.content_sha256:
                raise RuntimeError("regional registry head digest mismatch")
            current = self._snapshot
            if current is not None and revision.generation < current.generation:
                raise RuntimeError("regional registry generation moved backward")
            registrations = MappingProxyType(
                {
                    item.cluster_id: item
                    for item in sorted(
                        revision.registrations,
                        key=lambda value: value.cluster_id,
                    )
                }
            )
            snapshot = RegionalRegistrySnapshot(
                generation=revision.generation,
                content_sha256=revision.content_sha256,
                registrations=registrations,
            )
            with self._lock:
                self._target_generation = head.generation
                self._target_content_sha256 = head.content_sha256
                self._snapshot = snapshot
                self._last_successful_refresh = observed
                self._last_error = None
            self.store.save_regional_registry_member(
                self._member(snapshot, ready=True, observed_at=observed)
            )
            return True
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                current_snapshot = self._snapshot
            try:
                self.store.save_regional_registry_member(
                    self._member(
                        current_snapshot,
                        ready=False,
                        observed_at=observed,
                    )
                )
            except Exception:
                pass
            if raise_on_failure:
                raise
            return False

    def run(self, stop: Event) -> None:
        while not stop.is_set():
            self.refresh_once()
            if stop.wait(self.poll_seconds):
                break

    def _member(
        self,
        snapshot: RegionalRegistrySnapshot | None,
        *,
        ready: bool,
        observed_at: datetime,
    ) -> RegionalRegistryMember:
        return RegionalRegistryMember(
            member_id=self.member_id,
            service_role=self.service_role,
            release_id=self.release_id,
            generation=snapshot.generation if snapshot is not None else 0,
            content_sha256=(
                snapshot.content_sha256 if snapshot is not None else "0" * 64
            ),
            ready=ready,
            error=None if ready else self._last_error,
            started_at=self.started_at,
            last_seen_at=observed_at,
        )


def regional_cluster_request_allowed(
    registration: RegionalClusterRegistration,
    *,
    path: str,
    method: str,
) -> bool:
    state = registration.lifecycle_state
    if state is RegionalClusterLifecycle.ACTIVE:
        return True
    if state in {
        RegionalClusterLifecycle.REVOKED,
        RegionalClusterLifecycle.ROLLED_BACK,
    }:
        return False
    if method in {"GET", "HEAD"}:
        return True
    bootstrap_paths = {
        "/v1/fleet/agents/heartbeat",
        "/v1/fleet/readiness",
        "/v1/regional/executors/readiness",
    }
    if path in bootstrap_paths:
        return True
    if state in {
        RegionalClusterLifecycle.PENDING,
        RegionalClusterLifecycle.FAILED,
    }:
        return False
    return path == "/v1/regional/executors/hyperpod-submissions/outcome" or (
        path.startswith("/v1/regional/executors/")
        and path.endswith(("/renew", "/result"))
    )


def active_registry_member_ids(
    members: list[RegionalRegistryMember],
    *,
    observed_at: datetime,
    stale_seconds: float,
) -> list[str]:
    threshold = observed_at - timedelta(seconds=stale_seconds)
    return sorted(
        member.member_id for member in members if member.last_seen_at >= threshold
    )


def registry_revision_converged(
    revision: RegionalRegistryRevision,
    members: list[RegionalRegistryMember],
    *,
    observed_at: datetime,
    stale_seconds: float,
) -> bool:
    threshold = observed_at - timedelta(seconds=stale_seconds)
    by_id = {member.member_id: member for member in members}
    return all(
        (member := by_id.get(member_id)) is not None
        and member.ready
        and member.generation == revision.generation
        and member.content_sha256 == revision.content_sha256
        and member.last_seen_at >= threshold
        for member_id in revision.required_member_ids
    )
