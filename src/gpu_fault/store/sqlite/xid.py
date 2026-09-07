from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, ContextManager, Sequence, cast

from gpu_fault.models import HealthSignalState, XidMetricBaseline
from gpu_fault.policy import (
    XidCorrelationRecord,
    XidCorrelationStatus,
    XidEvent,
)
from gpu_fault.store.shared.health_signals import (
    latched_health_signal_state,
    sample_disposition,
    signal_clock,
)
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


class SqliteXidMixin:
    # Attributes supplied by the composed concrete implementation. The row
    # readers stay `Any` because the model class is resolved at run time from a
    # string kind; every caller below states the row type with a `cast`.
    _db: Any
    _get_optional: Callable[[str, str], Any]
    _list: Callable[[str], list[Any]]
    _lock: Any
    _next_health_signal_state: Callable[..., tuple[HealthSignalState, bool]]
    _note_health_signal_clock_regression: Callable[..., None]
    _put: Callable[..., None]
    _state_transaction: Callable[[str], ContextManager[None]]
    _xid_metric_key: Callable[[str, str, str], str]

    def save_xid_event_if_absent(
        self, event: XidEvent, *, retain_from: datetime | None = None
    ) -> bool:
        with self._lock:
            cursor = self._db.execute(
                """
                INSERT OR IGNORE INTO objects(kind, key, payload)
                VALUES ('xid_correlation_event', ?, ?)
                """,
                (event.event_id, event.model_dump_json()),
            )
            inserted: bool = cursor.rowcount == 1
            if inserted and retain_from is not None:
                self._db.execute(
                    """
                    DELETE FROM objects
                    WHERE kind='xid_correlation_event'
                      AND json_extract(payload, '$.cluster_id')=?
                      AND json_extract(payload, '$.node_id')=?
                      AND json_extract(payload, '$.observed_at')<?
                    """,
                    (
                        event.cluster_id,
                        event.node_id,
                        _utc_text(retain_from),
                    ),
                )
            return inserted

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
                for item in self._list("xid_correlation_event")
                if item.cluster_id == cluster_id
                and item.node_id == node_id
                and (observed_after is None or item.observed_at >= observed_after)
                and (observed_before is None or item.observed_at <= observed_before)
            ]

    def save_xid_correlation_if_absent(self, correlation: XidCorrelationRecord) -> bool:
        with self._lock:
            cursor = self._db.execute(
                """
                INSERT OR IGNORE INTO objects(kind, key, payload)
                VALUES ('xid_correlation', ?, ?)
                """,
                (
                    correlation.event_id,
                    correlation.model_dump_json(),
                ),
            )
            saved: bool = cursor.rowcount == 1
            return saved

    def claim_due_xid_correlations(
        self,
        *,
        owner: str,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[XidCorrelationRecord]:
        with self._state_transaction("xid-correlation-claims"):
            due = sorted(
                (
                    item
                    for item in self._list("xid_correlation")
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
                self._put("xid_correlation", item.event_id, value)
                claimed.append(value)
            return claimed

    def observe_xid_metric(
        self,
        cluster_id: str,
        node_id: str,
        gpu_key: str,
        xid: int,
        observed_at: datetime,
    ) -> bool:
        key = self._xid_metric_key(cluster_id, node_id, gpu_key)
        baseline = XidMetricBaseline(
            cluster_id=cluster_id,
            node_id=node_id,
            gpu_key=gpu_key,
            xid=xid,
            observed_at=observed_at,
        )
        with self._state_transaction(f"xid_metric_baseline/{key}"):
            previous = cast(
                "XidMetricBaseline | None",
                self._get_optional("xid_metric_baseline", key),
            )
            if previous is not None and observed_at <= previous.observed_at:
                return False
            self._put("xid_metric_baseline", key, baseline)
            return previous is not None and xid > 0 and xid != previous.xid

    def claim_health_signal_transitions(
        self,
        items: Sequence[tuple[str, bool, datetime, float]],
        *,
        received_at: datetime | None = None,
    ) -> list[bool]:
        with self._state_transaction("health_signal_state/claims"):
            results: list[bool] = []
            for (
                signal_key,
                active,
                observed_at,
                minimum_active_seconds,
            ) in items:
                previous = cast(
                    "HealthSignalState | None",
                    self._get_optional("health_signal_state", signal_key),
                )
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
                self._put(
                    "health_signal_state",
                    signal_key,
                    state,
                )
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
        return cast(
            "HealthSignalState | None",
            self._get_optional("health_signal_state", signal_key),
        )

    def mark_health_signal_notified(
        self, signal_key: str, *, notified_at: datetime
    ) -> None:
        with self._state_transaction(f"health_signal_state/{signal_key}"):
            latched = latched_health_signal_state(
                cast(
                    "HealthSignalState | None",
                    self._get_optional("health_signal_state", signal_key),
                ),
                notified_at,
            )
            if latched is not None:
                self._put("health_signal_state", signal_key, latched)
