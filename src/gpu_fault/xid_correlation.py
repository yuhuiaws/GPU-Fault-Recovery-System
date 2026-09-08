from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Callable
from uuid import uuid4

from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    GpuFaultPolicyEngine,
    XidCorrelationRecord,
    XidCorrelationStatus,
    XidEvent,
    parse_xid154_action,
)
from gpu_fault.models import RecoveryAction, Severity


LOGGER = logging.getLogger(__name__)


class XidCorrelationCoordinator:
    """Durably closes XID 45 companion windows across HA replicas."""

    def __init__(
        self,
        store,
        policy: GpuFaultPolicyEngine,
        finalizer: Callable[[XidEvent, FaultPolicyDecision], FaultPolicyDecision],
        *,
        owner: str | None = None,
        poll_interval_seconds: float = 1,
        lease_seconds: int = 30,
        batch_size: int = 100,
        retained_windows: int = 8,
        now: Callable[[], datetime] = (lambda: datetime.now(timezone.utc)),
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("XID correlation poll interval must be positive")
        if lease_seconds < 5:
            raise ValueError("XID correlation lease must be at least five seconds")
        if batch_size < 1:
            raise ValueError("XID correlation batch size must be positive")
        if retained_windows < 2:
            raise ValueError(
                "XID event retention must keep at least two companion windows"
            )
        self.retained_windows = retained_windows
        self.store = store
        self.policy = policy
        self.finalizer = finalizer
        self.owner = owner or f"xid-correlation-{uuid4()}"
        self.poll_interval_seconds = poll_interval_seconds
        self.lease_duration = timedelta(seconds=lease_seconds)
        self.batch_size = batch_size
        self.now = now
        self._stop = Event()

    def ingest(self, event: XidEvent) -> FaultPolicyDecision:
        event = self.prepare_xid154(event)
        event = self.prepare_xid74(event)
        inserted = self.store.save_xid_event_if_absent(
            event, retain_from=self._retain_from(event)
        )
        existing = self.store.get_xid_policy_decision(event.event_id)
        if not inserted and existing is not None:
            return existing.model_copy(update={"duplicate": True})

        candidates = self._candidates(event)
        decision = self.policy.evaluate_xid(
            event,
            companion_events=candidates,
            xid45_window_closed=False,
            xid154_window_closed=False,
        )
        if decision.disposition is not ActionDisposition.PENDING_CORRELATION:
            self.store.save_xid_policy_decision(decision)
            return decision

        try:
            correlation = self.store.get_xid_correlation(event.event_id)
        except KeyError:
            correlation = XidCorrelationRecord(
                event_id=event.event_id,
                deadline=(
                    self.now()
                    + timedelta(seconds=(self.policy.policy.companion_window_seconds))
                ),
            )
            self.store.save_xid_correlation_if_absent(correlation)
            correlation = self.store.get_xid_correlation(event.event_id)

        if (
            correlation.status is XidCorrelationStatus.FINALIZED
            and existing is not None
        ):
            return existing.model_copy(update={"duplicate": True})
        self.store.save_xid_policy_decision(decision)
        return decision

    @staticmethod
    def prepare_xid154(event: XidEvent) -> XidEvent:
        # ``xid_154_action`` is server-owned: it is derived solely from the
        # NVIDIA XID 154 log line, never from a client-supplied field. A
        # client that pre-set it must not be able to short-circuit (or steer)
        # this derivation, so the value is always recomputed here and any
        # inbound value is discarded (security review H-10).
        if event.xid != 154:
            if event.xid_154_action is not None:
                return event.model_copy(update={"xid_154_action": None})
            return event
        action = parse_xid154_action(event.raw_message)
        if action is event.xid_154_action:
            return event
        return event.model_copy(update={"xid_154_action": action})

    def prepare_xid74(self, event: XidEvent) -> XidEvent:
        if event.xid != 74:
            return event
        counts = self.store.record_xid74_occurrences(event)
        return event.model_copy(update={"nvlink_occurrence_counts": counts})

    def _retain_from(self, event: XidEvent) -> datetime:
        """Oldest event still useful for correlating this node.

        Persisted XID events are only ever read back as companion
        candidates, so anything older than a multiple of the companion
        window can never influence a verdict and would otherwise grow
        the table without bound.
        """
        window = timedelta(seconds=self.policy.policy.companion_window_seconds)
        return event.observed_at - window * self.retained_windows

    def _candidates(self, event: XidEvent) -> list[XidEvent]:
        window = timedelta(seconds=self.policy.policy.companion_window_seconds)
        return self.store.list_xid_events(
            event.cluster_id,
            event.node_id,
            observed_after=event.observed_at - window,
            observed_before=event.observed_at + window,
        )

    def _prefetch_candidates(
        self, events: list[XidEvent]
    ) -> dict[tuple[str, str], list[XidEvent]]:
        """Candidate windows for the whole batch in one query.

        ``_candidates`` per event meant one round trip per claimed
        correlation, on top of one per event load - so a full batch cost
        200 round trips before any policy ran, and the pass routinely
        overran its poll interval. The union comes back per node; each
        event still narrows to its own window below.
        """
        window = timedelta(seconds=self.policy.policy.companion_window_seconds)
        scopes = [
            (
                event.cluster_id,
                event.node_id,
                event.observed_at - window,
                event.observed_at + window,
            )
            for event in events
        ]
        if not scopes:
            return {}
        return self.store.list_xid_events_for_scopes(scopes)

    def run_once(self) -> int:
        now = self.now()
        claimed = self.store.claim_due_xid_correlations(
            owner=self.owner,
            now=now,
            lease_duration=self.lease_duration,
            limit=self.batch_size,
        )
        completed = 0
        claimed_events = []
        loaded = self.store.get_xid_events(
            [correlation.event_id for correlation in claimed]
        )
        for correlation in claimed:
            event = loaded.get(correlation.event_id)
            if event is None:
                LOGGER.error(
                    "XID correlation event load failed: %s",
                    correlation.event_id,
                )
                continue
            claimed_events.append((correlation, event))
        prefetched = self._prefetch_candidates(
            [event for _correlation, event in claimed_events]
        )
        window = timedelta(seconds=self.policy.policy.companion_window_seconds)
        claimed_events.sort(
            key=lambda item: (
                item[1].xid == 154,
                item[0].deadline,
                item[0].event_id,
            )
        )
        for correlation, event in claimed_events:
            try:
                candidates = [
                    item
                    for item in prefetched.get((event.cluster_id, event.node_id), ())
                    if event.observed_at - window
                    <= item.observed_at
                    <= event.observed_at + window
                ]
                decision = self.policy.evaluate_xid(
                    event,
                    companion_events=candidates,
                    xid45_window_closed=True,
                    xid154_window_closed=True,
                )
                if decision.disposition is ActionDisposition.PENDING_CORRELATION:
                    LOGGER.error(
                        "closed XID correlation remained pending; "
                        "quarantining fail-closed: event=%s",
                        event.event_id,
                    )
                    decision = decision.model_copy(
                        update={
                            "disposition": (ActionDisposition.BLOCKED_MISSING_EVIDENCE),
                            "action": None,
                            "safety_action": (RecoveryAction.QUARANTINE),
                            "severity": Severity.CRITICAL,
                            "requires_operator": True,
                            "reasons": [
                                *decision.reasons,
                                "companion correlation window closed "
                                "without a terminal policy decision",
                            ],
                            "marker": decision.marker.model_copy(
                                update={
                                    "severity": Severity.CRITICAL,
                                    "recommended_action": (RecoveryAction.QUARANTINE),
                                    "site_safety_action": (
                                        RecoveryAction.QUARANTINE.value
                                    ),
                                    "action_disposition": ("SITE_SAFETY"),
                                }
                            ),
                        }
                    )
                finalized = self.finalizer(event, decision)
                self.store.save_xid_policy_decision(finalized)
                self.store.complete_xid_correlation(
                    event.event_id,
                    owner=self.owner,
                    now=self.now(),
                )
                completed += 1
            except Exception:
                LOGGER.exception(
                    "XID correlation finalization failed: %s",
                    correlation.event_id,
                )
        return completed

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.poll_interval_seconds)

    def stop(self) -> None:
        self._stop.set()
