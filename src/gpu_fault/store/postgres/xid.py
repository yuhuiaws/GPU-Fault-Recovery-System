from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Sequence

from gpu_fault.models import HealthSignalState, XidMetricBaseline
from gpu_fault.policy import (
    XidCorrelationRecord,
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


class PostgresXidMixin:
    # Attributes supplied by the composed concrete implementation. `_decode` and
    # `_get_optional` stay `Any` because the model class is resolved at run time
    # from a string kind.
    _db: Any
    _decode: Callable[[str, Any], Any]
    _get_optional: Callable[[str, str], Any]
    _next_health_signal_state: Callable[..., tuple[HealthSignalState, bool]]
    _note_health_signal_clock_regression: Callable[..., None]
    _put: Callable[..., None]
    _state_transaction: Callable[..., Any]
    _xid_metric_key: Callable[[str, str, str], str]

    def claim_due_xid_correlations(
        self,
        *,
        owner: str,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[XidCorrelationRecord]:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH candidates AS (
                        SELECT key
                        FROM gpu_fault_objects
                        WHERE kind='xid_correlation'
                          AND payload->>'status'='PENDING'
                          AND payload->>'deadline' <= %s
                          AND (
                              payload->>'lease_expires_at' IS NULL
                              OR payload->>'lease_expires_at' <= %s
                          )
                        ORDER BY payload->>'deadline', key
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE gpu_fault_objects AS objects
                    SET payload=objects.payload || jsonb_build_object(
                        'lease_owner', %s::text,
                        'lease_expires_at', %s::text
                    )
                    FROM candidates
                    WHERE objects.kind='xid_correlation'
                      AND objects.key=candidates.key
                    RETURNING objects.payload
                    """,
                    (
                        _utc_text(now),
                        _utc_text(now),
                        limit,
                        owner,
                        _utc_text(now + lease_duration),
                    ),
                )
                rows = cursor.fetchall()
        return [self._decode("xid_correlation", row[0]) for row in rows]

    def save_xid_event_if_absent(
        self, event: XidEvent, *, retain_from: datetime | None = None
    ) -> bool:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    VALUES ('xid_correlation_event', %s, %s::jsonb)
                    ON CONFLICT(kind, key) DO NOTHING
                    RETURNING key
                    """,
                    (
                        event.event_id,
                        event.model_dump_json(),
                    ),
                )
                inserted = cursor.fetchone() is not None
                if inserted and retain_from is not None:
                    cursor.execute(
                        """
                        DELETE FROM gpu_fault_objects
                        WHERE kind='xid_correlation_event'
                          AND payload->>'cluster_id'=%s
                          AND payload->>'node_id'=%s
                          AND payload->>'observed_at'<%s
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
        clauses = [
            "kind='xid_correlation_event'",
            "payload->>'cluster_id'=%s",
            "payload->>'node_id'=%s",
        ]
        parameters = [cluster_id, node_id]
        if observed_after is not None:
            clauses.append("payload->>'observed_at'>=%s")
            parameters.append(_utc_text(observed_after))
        if observed_before is not None:
            clauses.append("payload->>'observed_at'<=%s")
            parameters.append(_utc_text(observed_before))
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM gpu_fault_objects WHERE " + " AND ".join(clauses),
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("xid_correlation_event", row[0]) for row in rows]

    def get_xid_events(self, event_ids: Iterable[str]) -> dict[str, XidEvent]:
        keys = list(dict.fromkeys(event_ids))
        if not keys:
            return {}
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT key, payload FROM gpu_fault_objects
                WHERE kind='xid_correlation_event'
                  AND key = ANY(%s)
                """,
                (keys,),
            )
            rows = cursor.fetchall()
        return {row[0]: self._decode("xid_correlation_event", row[1]) for row in rows}

    def list_xid_events_for_scopes(
        self,
        scopes: Iterable[tuple[str, str, datetime | None, datetime | None]],
    ) -> dict[tuple[str, str], list[XidEvent]]:
        # One statement for the whole claimed batch instead of one per
        # event. Each scope keeps its own time bounds so
        # gpu_fault_xid_correlation_lookup still drives every branch;
        # collapsing them into one min/max window would have widened the
        # scan on the busiest node in the batch.
        groups: list[tuple[str, str]] = []
        clauses: list[str] = []
        parameters: list[object] = []
        for cluster_id, node_id, after, before in scopes:
            scope = (cluster_id, node_id)
            if scope not in groups:
                groups.append(scope)
            branch = [
                "payload->>'cluster_id'=%s",
                "payload->>'node_id'=%s",
            ]
            parameters.extend([cluster_id, node_id])
            if after is not None:
                branch.append("payload->>'observed_at'>=%s")
                parameters.append(_utc_text(after))
            if before is not None:
                branch.append("payload->>'observed_at'<=%s")
                parameters.append(_utc_text(before))
            clauses.append("(" + " AND ".join(branch) + ")")
        if not clauses:
            return {}
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM gpu_fault_objects "
                "WHERE kind='xid_correlation_event' AND (" + " OR ".join(clauses) + ")",
                parameters,
            )
            rows = cursor.fetchall()
        found: dict[tuple[str, str], dict[str, XidEvent]] = {
            scope: {} for scope in groups
        }
        for row in rows:
            item = self._decode("xid_correlation_event", row[0])
            bucket = found.get((item.cluster_id, item.node_id))
            if bucket is None:
                continue
            bucket[item.event_id] = item
        return {scope: list(bucket.values()) for scope, bucket in found.items()}

    def save_xid_correlation_if_absent(self, correlation: XidCorrelationRecord) -> bool:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    VALUES ('xid_correlation', %s, %s::jsonb)
                    ON CONFLICT(kind, key) DO NOTHING
                    RETURNING key
                    """,
                    (
                        correlation.event_id,
                        correlation.model_dump_json(),
                    ),
                )
                return cursor.fetchone() is not None

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
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (f"xid_metric_baseline/{key}",),
                )
            previous = self._get_optional("xid_metric_baseline", key)
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
        if not items:
            return []
        keys = [item[0] for item in items]
        unique_keys = list(dict.fromkeys(keys))
        # One transaction for read, decision and write (G-5). On the autocommit
        # pool the FOR UPDATE below released its row locks before the caller
        # saw the rows, and a key with no row yet has nothing to lock at all,
        # so two workers deciding the same signal both read "inactive" and both
        # emitted, or the slower one overwrote the fresher state. The advisory
        # locks are the ones ``mark_health_signal_notified`` takes through
        # ``_state_transaction`` (same key text), taken in sorted order so two
        # batches sharing keys cannot deadlock.
        with self._db.transaction():
            with self._db.cursor() as cursor:
                for signal_key in sorted(unique_keys):
                    cursor.execute(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtextextended(%s, 0)
                        )
                        """,
                        (f"health_signal_state/{signal_key}",),
                    )
                cursor.execute(
                    """
                    SELECT key, payload
                    FROM gpu_fault_objects
                    WHERE kind='health_signal_state'
                      AND key=ANY(%s)
                    FOR UPDATE
                    """,
                    (unique_keys,),
                )
                current_by_key = {
                    key: self._decode("health_signal_state", payload)
                    for key, payload in cursor.fetchall()
                }
            return self._decide_health_signal_transitions(
                items, current_by_key, received_at=received_at
            )

    def _decide_health_signal_transitions(
        self,
        items: Sequence[tuple[str, bool, datetime, float]],
        current_by_key: dict[str, HealthSignalState],
        *,
        received_at: datetime | None,
    ) -> list[bool]:
        """The decision and the write, inside the caller's locked transaction."""

        results: list[bool] = []
        final_by_key: dict[str, HealthSignalState] = {}
        for (
            signal_key,
            active,
            observed_at,
            minimum_active_seconds,
        ) in items:
            previous = current_by_key.get(signal_key)
            accept, regressed = sample_disposition(previous, observed_at, received_at)
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
            current_by_key[signal_key] = state
            final_by_key[signal_key] = state
            results.append(emit)
        if final_by_key:
            final_keys = list(final_by_key)
            payloads = [final_by_key[key].model_dump_json() for key in final_keys]
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    SELECT
                        'health_signal_state',
                        batch.key,
                        batch.payload::jsonb
                    FROM unnest(
                        %s::text[], %s::text[]
                    ) AS batch(key, payload)
                    ON CONFLICT(kind, key) DO UPDATE SET
                        payload=excluded.payload
                    """,
                    (final_keys, payloads),
                )
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
        state = self._get_optional("health_signal_state", signal_key)
        return state if isinstance(state, HealthSignalState) else None

    def mark_health_signal_notified(
        self, signal_key: str, *, notified_at: datetime
    ) -> None:
        with self._state_transaction(f"health_signal_state/{signal_key}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT payload
                    FROM gpu_fault_objects
                    WHERE kind='health_signal_state' AND key=%s
                    FOR UPDATE
                    """,
                    (signal_key,),
                )
                row = cursor.fetchone()
            current = (
                self._decode("health_signal_state", row[0]) if row is not None else None
            )
            latched = latched_health_signal_state(current, notified_at)
            if latched is not None:
                self._put("health_signal_state", signal_key, latched)
