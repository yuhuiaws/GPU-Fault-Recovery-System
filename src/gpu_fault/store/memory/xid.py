from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from gpu_fault.models import (
    HealthSignalState,
    XidMetricBaseline,
)
from gpu_fault.policy import (
    FaultPolicyDecision,
    Nvlink74BitOccurrenceState,
    XidCorrelationRecord,
    XidCorrelationStatus,
    XidEvent,
)
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.store.shared.health_signals import (
    latched_health_signal_state,
    sample_disposition,
    signal_clock,
)

# One NVLink bit's identity: the scope from `_xid74_scope` plus the register
# index and bit position inside it.
_Xid74StateKey = tuple[str, str, int, int, int]


class MemoryXidMixin:
    # Attributes supplied by the composed concrete implementation.
    _health_signal_states: dict[str, HealthSignalState]
    _xid_correlation_events: dict[str, XidEvent]
    _xid_correlations: dict[str, XidCorrelationRecord]
    _xid_policy_decisions: dict[str, FaultPolicyDecision]

    # The rules, from SharedXidSignalMixin.
    _next_health_signal_state: Callable[..., tuple[HealthSignalState, bool]]
    _note_health_signal_clock_regression: Callable[..., None]
    _xid74_populated_bits: Callable[[XidEvent], list[tuple[int, int]]]
    _xid74_scope: Callable[[XidEvent], tuple[str, str, int] | None]

    _lock: Any
    _xid74_counted_events: set[tuple[str, _Xid74StateKey]]
    _xid74_occurrence_states: dict[_Xid74StateKey, Nvlink74BitOccurrenceState]
    _xid_metric_baselines: dict[tuple[str, str, str], XidMetricBaseline]

    def save_xid_event_if_absent(
        self, event: XidEvent, *, retain_from: datetime | None = None
    ) -> bool:
        with self._lock:
            if event.event_id in self._xid_correlation_events:
                return False
            self._xid_correlation_events[event.event_id] = event
            if retain_from is not None:
                for key, item in list(self._xid_correlation_events.items()):
                    if (
                        item.cluster_id == event.cluster_id
                        and item.node_id == event.node_id
                        and item.observed_at < retain_from
                    ):
                        self._xid_correlation_events.pop(key, None)
            return True

    def record_xid74_occurrences(self, event: XidEvent) -> dict[str, int]:
        scope = self._xid74_scope(event)
        if event.xid != 74 or scope is None:
            return {}
        with self._lock:
            counts: dict[str, int] = {}
            for (
                register_index,
                bit,
            ) in self._xid74_populated_bits(event):
                state_key = (*scope, register_index, bit)
                event_key = (event.event_id, state_key)
                state = self._xid74_occurrence_states.get(state_key)
                if event_key not in self._xid74_counted_events:
                    state = (
                        state.incremented(
                            event.event_id,
                            event.observed_at,
                        )
                        if state is not None
                        else Nvlink74BitOccurrenceState(
                            cluster_id=scope[0],
                            gpu_identity=scope[1],
                            link_id=scope[2],
                            register_index=register_index,
                            bit=bit,
                            count=1,
                            first_observed_at=event.observed_at,
                            last_observed_at=event.observed_at,
                            last_event_id=event.event_id,
                        )
                    )
                    self._xid74_occurrence_states[state_key] = state
                    self._xid74_counted_events.add(event_key)
                if state is not None:
                    counts[f"register{register_index + 1}.bit{bit}"] = state.count
            return counts

    def get_xid_event(self, event_id: str) -> XidEvent:
        with self._lock:
            try:
                return self._xid_correlation_events[event_id]
            except KeyError as exc:
                raise NotFoundError(event_id) from exc

    def list_xid_events(
        self,
        cluster_id: str,
        node_id: str,
        *,
        observed_after: datetime | None = None,
        observed_before: datetime | None = None,
    ) -> list[XidEvent]:
        with self._lock:
            return [
                item
                for item in self._xid_correlation_events.values()
                if item.cluster_id == cluster_id
                and item.node_id == node_id
                and (observed_after is None or item.observed_at >= observed_after)
                and (observed_before is None or item.observed_at <= observed_before)
            ]

    def save_xid_policy_decision(self, decision: FaultPolicyDecision) -> None:
        with self._lock:
            self._xid_policy_decisions[decision.event_id] = decision

    def get_xid_policy_decision(self, event_id: str) -> FaultPolicyDecision | None:
        with self._lock:
            return self._xid_policy_decisions.get(event_id)

    def save_xid_correlation_if_absent(self, correlation: XidCorrelationRecord) -> bool:
        with self._lock:
            if correlation.event_id in self._xid_correlations:
                return False
            self._xid_correlations[correlation.event_id] = correlation
            return True

    def get_xid_correlation(self, event_id: str) -> XidCorrelationRecord:
        with self._lock:
            try:
                return self._xid_correlations[event_id]
            except KeyError as exc:
                raise NotFoundError(event_id) from exc

    def claim_due_xid_correlations(
        self,
        *,
        owner: str,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[XidCorrelationRecord]:
        with self._lock:
            due = sorted(
                (
                    item
                    for item in self._xid_correlations.values()
                    if item.status is XidCorrelationStatus.PENDING
                    and item.deadline <= now
                    and (item.lease_expires_at is None or item.lease_expires_at <= now)
                ),
                key=lambda item: (
                    item.deadline,
                    item.event_id,
                ),
            )[:limit]
            claimed: list[XidCorrelationRecord] = []
            for item in due:
                value = item.model_copy(
                    update={
                        "lease_owner": owner,
                        "lease_expires_at": now + lease_duration,
                    }
                )
                self._xid_correlations[item.event_id] = value
                claimed.append(value)
            return claimed

    def complete_xid_correlation(
        self, event_id: str, *, owner: str, now: datetime
    ) -> XidCorrelationRecord:
        with self._lock:
            current = self.get_xid_correlation(event_id)
            if current.lease_owner != owner:
                raise ValueError("XID correlation lease owner changed")
            completed = current.model_copy(
                update={
                    "status": XidCorrelationStatus.FINALIZED,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "finalized_at": now,
                }
            )
            self._xid_correlations[event_id] = completed
            return completed

    def claim_health_signal_transitions(
        self,
        items: Sequence[tuple[str, bool, datetime, float]],
        *,
        received_at: datetime | None = None,
    ) -> list[bool]:
        with self._lock:
            results: list[bool] = []
            for (
                signal_key,
                active,
                observed_at,
                minimum_active_seconds,
            ) in items:
                previous = self._health_signal_states.get(signal_key)
                accept, regressed = sample_disposition(
                    previous, observed_at, received_at
                )
                if regressed:
                    self._note_health_signal_clock_regression(signal_key, observed_at)
                if not accept:
                    results.append(False)
                    continue
                if previous is None and not active:
                    results.append(False)
                    continue
                state, emit = self._next_health_signal_state(
                    signal_key,
                    active,
                    observed_at,
                    previous,
                    minimum_active_seconds,
                    clock=signal_clock(observed_at, received_at),
                )
                self._health_signal_states[signal_key] = state
                results.append(emit)
            return results

    def claim_health_signal_transition(
        self,
        signal_key: str,
        active: bool,
        observed_at: datetime,
        minimum_active_seconds: float = 0,
    ) -> bool:
        # The single form is the training-progress path. Like the plural form
        # it only decides to emit; ``TrainingHealthService.mark_notified``
        # latches once the finding's incident has been written (P0-38B).
        return self.claim_health_signal_transitions(
            [
                (
                    signal_key,
                    active,
                    observed_at,
                    minimum_active_seconds,
                )
            ]
        )[0]

    def get_health_signal_state(self, signal_key: str) -> HealthSignalState | None:
        with self._lock:
            return self._health_signal_states.get(signal_key)

    def mark_health_signal_notified(
        self, signal_key: str, *, notified_at: datetime
    ) -> None:
        with self._lock:
            latched = latched_health_signal_state(
                self._health_signal_states.get(signal_key), notified_at
            )
            if latched is not None:
                self._health_signal_states[signal_key] = latched

    def observe_xid_metric(
        self,
        cluster_id: str,
        node_id: str,
        gpu_key: str,
        xid: int,
        observed_at: datetime,
    ) -> bool:
        """Persist one last-XID observation and claim a fresh transition."""
        key = (cluster_id, node_id, gpu_key)
        baseline = XidMetricBaseline(
            cluster_id=cluster_id,
            node_id=node_id,
            gpu_key=gpu_key,
            xid=xid,
            observed_at=observed_at,
        )
        with self._lock:
            previous = self._xid_metric_baselines.get(key)
            if previous is not None and observed_at <= previous.observed_at:
                return False
            self._xid_metric_baselines[key] = baseline
            return previous is not None and xid > 0 and xid != previous.xid
