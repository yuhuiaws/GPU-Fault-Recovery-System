from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event, RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping

from gpu_fault.channel_registry import COLLECTOR_EVENT_PREFIX
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryMember,
    RegionalRegistryRevision,
)
from gpu_fault.regional_registry import regional_registry_config_sha256
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

# Exponential backoff for a start-up that finds Aurora unavailable: a writer
# failover takes 30-60 s and a Pod that crashes into a restart loop meanwhile
# only needs Aurora again when it comes back.
STARTUP_RETRY_INITIAL_SECONDS = 0.5
STARTUP_RETRY_MAX_INTERVAL_SECONDS = 10.0


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
        # Digest of the GPU_FAULT_REGIONAL_CLUSTERS_JSON Secret this process was
        # started with; None when the caller did not supply one.
        self.secret_config_sha256: str | None = None
        self._secret_drift_logged: bool | None = None
        # The member row as last written and when: the heartbeat is re-written
        # only when its content changes or a third of the stale window has
        # passed, not on every 1 s poll (control-plane review 2026-09-08, F-5).
        self._last_heartbeat_at: datetime | None = None
        self._last_heartbeat_fingerprint: tuple[object, ...] | None = None

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
        retry_budget_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        secret_config_sha256: str | None = None,
    ) -> RegionalRegistryRuntime:
        """Load the durable head, retrying store failures within a budget.

        ``retry_budget_seconds`` bounds how long start-up waits for the store
        (Aurora) to answer before the failure propagates and the process exits;
        zero keeps the old fail-immediately behaviour for tests and tools.
        ``secret_config_sha256`` is the configured Secret's digest; a mismatch
        with the durable head is logged and reported as ``secret_drift``.
        """

        observed = now or (lambda: datetime.now(timezone.utc))
        deadline = observed() + timedelta(seconds=retry_budget_seconds)
        interval = STARTUP_RETRY_INITIAL_SECONDS
        attempt = 0
        while True:
            attempt += 1
            try:
                runtime = cls._bootstrap_once(
                    store,
                    member_id=member_id,
                    service_role=service_role,
                    release_id=release_id,
                    poll_seconds=poll_seconds,
                    stale_seconds=stale_seconds,
                    now=observed,
                )
                runtime.secret_config_sha256 = secret_config_sha256
                runtime._log_secret_drift()
                return runtime
            except ValueError:
                # Configuration errors (poll/stale) are not store outages.
                raise
            except Exception as exc:
                remaining = (deadline - observed()).total_seconds()
                if remaining <= 0:
                    raise
                wait = min(interval, STARTUP_RETRY_MAX_INTERVAL_SECONDS, remaining)
                LOGGER.warning(
                    "regional registry bootstrap attempt %d failed (%s: %s); "
                    "retrying in %.1fs with %.0fs of start-up budget left",
                    attempt,
                    type(exc).__name__,
                    exc,
                    wait,
                    remaining,
                )
                sleep(wait)
                interval = min(interval * 2, STARTUP_RETRY_MAX_INTERVAL_SECONDS)

    @classmethod
    def _bootstrap_once(
        cls,
        store: Any,
        *,
        member_id: str,
        service_role: str,
        release_id: str,
        poll_seconds: float,
        stale_seconds: float,
        now: Callable[[], datetime],
    ) -> RegionalRegistryRuntime:
        observed = now
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
            # ``_last_error`` is no longer a criterion (A-7): one closed
            # connection (~1/min baseline) failed readiness for the whole
            # process until the next 1 s refresh, so
            # GPU_FAULT_REGISTRY_STALE_SECONDS never applied to the error
            # path. A divergent head is not transient: refresh_once moves the
            # target to it, and the comparison below fails at once.
            return (
                self._snapshot is not None
                and self._snapshot.generation == self._target_generation
                and self._snapshot.content_sha256 == self._target_content_sha256
                and self._last_successful_refresh is not None
                and observed - self._last_successful_refresh
                <= timedelta(seconds=self.stale_seconds)
            )

    def durable_config_sha256(self) -> str | None:
        """Configured-view digest of the snapshot this process serves."""

        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        return regional_registry_config_sha256(snapshot.registrations.values())

    def secret_drift(self) -> bool:
        """Whether the start-up Secret and the durable head disagree.

        The durable head wins (join/remove publish there); drift means the
        Secret is stale and a release that only rewrote the Secret would not
        reach the running registry (review H3).
        """

        if self.secret_config_sha256 is None:
            return False
        durable = self.durable_config_sha256()
        return durable is not None and durable != self.secret_config_sha256

    def _log_secret_drift(self) -> None:
        drift = self.secret_drift()
        if drift and self._secret_drift_logged is not True:
            LOGGER.warning(
                "regional registry Secret drifts from the durable head: "
                "secret_config_sha256=%s durable_config_sha256=%s; the durable "
                "head is authoritative, republish the registry from the release "
                "config to reconcile",
                self.secret_config_sha256,
                self.durable_config_sha256(),
            )
        self._secret_drift_logged = drift

    def status(self) -> dict[str, Any]:
        secret_drift = self.secret_drift()
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
                "secret_config_sha256": self.secret_config_sha256,
                "secret_drift": secret_drift,
            }

    def refresh_once(self, *, raise_on_failure: bool = False) -> bool:
        observed = self.now()
        head = None
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
            self._heartbeat(self._member(snapshot, ready=True, observed_at=observed))
            self._log_secret_drift()
            return True
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                if head is not None:
                    # The head was read and could not be served (digest or
                    # generation mismatch, or the revision read failed):
                    # fail readiness now rather than serve a snapshot the
                    # head disagrees with.
                    self._target_generation = head.generation
                    self._target_content_sha256 = head.content_sha256
                current_snapshot = self._snapshot
            try:
                self._heartbeat(
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

    @staticmethod
    def heartbeat_interval_seconds(stale_seconds: float) -> float:
        """How long an unchanged member row may go without being re-written.

        A third of the stale window: two heartbeats can be lost to a slow
        store before ``active_registry_member_ids`` stops counting the
        process, and with the production ``GPU_FAULT_REGISTRY_STALE_SECONDS=90``
        it turns ~36 upserts/s across the fleet into ~1.2/s (F-5).
        """

        return stale_seconds / 3.0

    def _heartbeat(self, member: RegionalRegistryMember) -> None:
        """Write the member row if its content changed or the heartbeat is due."""

        fingerprint: tuple[object, ...] = (
            member.generation,
            member.content_sha256,
            member.ready,
            member.error,
        )
        with self._lock:
            last_written = self._last_heartbeat_at
            unchanged = fingerprint == self._last_heartbeat_fingerprint
        if (
            unchanged
            and last_written is not None
            and (member.last_seen_at - last_written).total_seconds()
            < self.heartbeat_interval_seconds(self.stale_seconds)
        ):
            return
        self.store.save_regional_registry_member(member)
        with self._lock:
            self._last_heartbeat_at = member.last_seen_at
            self._last_heartbeat_fingerprint = fingerprint

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
    if state is RegionalClusterLifecycle.PENDING:
        # A join verifies collector readiness before it activates the
        # cluster, and readiness is built from these events; claims stay
        # refused, so telemetry admitted here cannot start a node action.
        return path.startswith(COLLECTOR_EVENT_PREFIX)
    if state is RegionalClusterLifecycle.FAILED:
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
