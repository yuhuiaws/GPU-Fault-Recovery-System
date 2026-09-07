"""XID and host-health-signal code shared across the stores.

:class:`SharedXidSignalMixin` is the pure part -- how one health signal
advances, how an XID 74 event maps onto NVLink bits -- and every store
composes it. :class:`SharedXidMixin` persists those results through the
key/value primitives and is composed by SQLite and PostgreSQL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, cast

from gpu_fault.models import HealthSignalState
from gpu_fault.policy import (
    FaultPolicyDecision,
    Nvlink74BitOccurrenceState,
    XidCorrelationRecord,
    XidCorrelationStatus,
    XidEvent,
)
from gpu_fault.store.shared.health_signals import previous_clock
from gpu_fault.store.shared.primitives import (
    GetLink,
    GetOptionalRecord,
    GetRecord,
    LinkRecord,
    PutRecord,
    StateKey,
    StatementGuard,
    StateTransaction,
)

LOGGER = logging.getLogger(__name__)


class SharedXidSignalMixin:
    """The rules; no rows."""

    # Node clocks that stepped backwards while the control plane's moved on
    # (F-M2). The sample is still judged; this only says it happened.
    health_signal_clock_regressions_total: int = 0

    def _note_health_signal_clock_regression(
        self, signal_key: str, observed_at: datetime
    ) -> None:
        self.health_signal_clock_regressions_total += 1
        LOGGER.warning(
            "health signal %s reported a node time (%s) not after its previous "
            "sample; judged on control-plane time instead",
            signal_key,
            observed_at.isoformat(),
        )

    @staticmethod
    def _next_health_signal_state(
        signal_key: str,
        active: bool,
        observed_at: datetime,
        previous: HealthSignalState | None,
        minimum_active_seconds: float,
        *,
        clock: datetime | None = None,
    ) -> tuple[HealthSignalState, bool]:
        """Advance one signal. ``clock`` is the time durations are measured on
        (the control plane's receive time when the caller has it); it defaults
        to the node's ``observed_at`` for callers without one.

        The ``notified`` latch is carried over, never set here (P0-38B): the
        claim decides to emit, the deliverer latches through
        ``mark_health_signal_notified`` once the incident has committed. Until
        then a still-active signal emits again on its next sample."""

        minimum_active_seconds = max(0.0, minimum_active_seconds)
        now_on_clock = clock if clock is not None else observed_at
        if not active:
            return (
                HealthSignalState(
                    signal_key=signal_key,
                    active=False,
                    observed_at=observed_at,
                    active_since=None,
                    notified=False,
                    clock_at=now_on_clock,
                ),
                False,
            )
        active_since = (
            previous.active_since or previous_clock(previous)
            if previous is not None and previous.active
            else now_on_clock
        )
        previously_notified = (
            (previous.notified if previous.notified is not None else previous.active)
            if previous is not None
            else False
        )
        duration = (now_on_clock - active_since).total_seconds()
        emit = not previously_notified and duration >= minimum_active_seconds
        return (
            HealthSignalState(
                signal_key=signal_key,
                active=True,
                observed_at=observed_at,
                active_since=active_since,
                notified=previously_notified,
                clock_at=now_on_clock,
            ),
            emit,
        )

    @staticmethod
    def _xid74_scope(event: XidEvent) -> tuple[str, str, int] | None:
        gpu_key = (
            f"uuid:{event.gpu_uuid}"
            if event.gpu_uuid
            else (
                f"node-pci:{event.node_id}:{event.pci_bdf.lower()}"
                if event.pci_bdf
                else None
            )
        )
        if gpu_key is None or event.nvlink_link_id is None:
            return None
        return (
            event.cluster_id,
            gpu_key,
            event.nvlink_link_id,
        )

    @staticmethod
    def _xid74_populated_bits(event: XidEvent) -> list[tuple[int, int]]:
        return [
            (register_index, bit)
            for register_index, value in enumerate(event.registers)
            for bit in range(value.bit_length())
            if value & (1 << bit)
        ]


class SharedXidMixin:
    """The rules persisted through the key/value primitives."""

    # Attributes supplied by the composed concrete implementation.
    _xid74_populated_bits: Callable[..., Any]
    _xid74_scope: Callable[..., Any]

    _get: GetRecord
    _get_link: GetLink
    _get_optional: GetOptionalRecord
    _link: LinkRecord
    _put: PutRecord
    _state_key: StateKey
    _state_transaction: StateTransaction
    _statement_guard: StatementGuard

    def record_xid74_occurrences(self, event: XidEvent) -> dict[str, int]:
        scope = self._xid74_scope(event)
        if event.xid != 74 or scope is None:
            return {}
        scope_key = self._state_key(scope)
        counts: dict[str, int] = {}
        with self._state_transaction(f"xid74-occurrence/{scope_key}"):
            for (
                register_index,
                bit,
            ) in self._xid74_populated_bits(event):
                state_parts = (*scope, register_index, bit)
                state_key = self._state_key(state_parts)
                event_key = self._state_key((event.event_id, *state_parts))
                state = cast(
                    "Nvlink74BitOccurrenceState | None",
                    self._get_optional("xid74_occurrence_state", state_key),
                )
                if self._get_link("xid74_occurrence_event", event_key) is None:
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
                    self._put(
                        "xid74_occurrence_state",
                        state_key,
                        state,
                    )
                    self._link(
                        "xid74_occurrence_event",
                        event_key,
                        state_key,
                    )
                if state is not None:
                    counts[f"register{register_index + 1}.bit{bit}"] = state.count
        return counts

    def get_xid_event(self, event_id: str) -> XidEvent:
        with self._statement_guard():
            return cast("XidEvent", self._get("xid_correlation_event", event_id))

    def save_xid_policy_decision(self, decision: FaultPolicyDecision) -> None:
        with self._statement_guard():
            self._put(
                "xid_policy_decision",
                decision.event_id,
                decision,
            )

    def get_xid_policy_decision(self, event_id: str) -> FaultPolicyDecision | None:
        with self._statement_guard():
            return cast(
                "FaultPolicyDecision | None",
                self._get_optional("xid_policy_decision", event_id),
            )

    def get_xid_correlation(self, event_id: str) -> XidCorrelationRecord:
        with self._statement_guard():
            return cast("XidCorrelationRecord", self._get("xid_correlation", event_id))

    def complete_xid_correlation(
        self, event_id: str, *, owner: str, now: datetime
    ) -> XidCorrelationRecord:
        with self._state_transaction(f"xid-correlation/{event_id}"):
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
            self._put("xid_correlation", event_id, completed)
            return completed

    @staticmethod
    def _xid_metric_key(cluster_id: str, node_id: str, gpu_key: str) -> str:
        return json.dumps(
            [cluster_id, node_id, gpu_key],
            ensure_ascii=True,
            separators=(",", ":"),
        )
