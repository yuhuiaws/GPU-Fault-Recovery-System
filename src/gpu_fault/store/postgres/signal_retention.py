"""The health-signal retention's two primitives on PostgreSQL.

Both statements are bounded by ``limit`` and touch only the
``gpu_fault_objects`` primary key ``(kind, key)``: the page is a key-range
scan, the delete a set of key lookups. Neither waits on a lock held by
anybody else, so the sweep can stall neither the ingest path nor itself:

* every candidate's advisory lock -- the text ``<kind>/<key>`` that
  ``claim_health_signal_transitions`` (``take_advisory_xact_locks``),
  ``mark_health_signal_notified`` and ``observe_efa_traffic``
  (``_state_transaction``) hold for their whole read-decide-write -- is only
  *tried* (``pg_try_advisory_xact_lock``); a busy key is a row being written
  right now and is skipped;
* the row must still hold exactly the payload that was judged (a committed
  write since the page read changed it), and a row lock held by anybody is
  skipped (``SKIP LOCKED``).

A claim that waits for one of these locks after the delete committed sees no
row and opens a fresh window from its own sample; nothing it read before can
be written back, because it reads only under the lock the delete took.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any, Callable, Protocol

from gpu_fault.store.shared.signal_retention import SignalPage, SignalRow
from gpu_fault.store.shared.time import utc_text as _utc_text

# A health signal's clock is ``clock_at`` (control-plane receive time) when it
# has one; an EFA traffic state has none and falls through to ``observed_at``.
# Stored timestamps are fixed-width UTC text (``datetime_json_text``) and are
# compared as text like every other sweep; the caller re-judges each row it
# gets back on the decoded model.
_SIGNAL_STATE_PAGE = """
    SELECT
        key,
        CASE
            WHEN COALESCE(payload->>'clock_at', payload->>'observed_at') < %s
            THEN payload::text
        END
    FROM gpu_fault_objects
    WHERE kind = %s AND key > %s
    ORDER BY key
    LIMIT %s
"""

_DELETE_UNCHANGED_SIGNAL_STATES = """
    WITH unlocked AS MATERIALIZED (
        SELECT batch.key, batch.payload
        FROM unnest(%s::text[], %s::text[]) AS batch(key, payload)
        WHERE pg_try_advisory_xact_lock(
            hashtextextended(%s::text || '/' || batch.key, 0)
        )
    ),
    unchanged AS MATERIALIZED (
        SELECT objects.key
        FROM gpu_fault_objects AS objects
        JOIN unlocked ON unlocked.key = objects.key
        WHERE objects.kind = %s
          AND objects.payload = unlocked.payload::jsonb
        FOR UPDATE OF objects SKIP LOCKED
    )
    DELETE FROM gpu_fault_objects AS objects
    USING unchanged
    WHERE objects.kind = %s AND objects.key = unchanged.key
    RETURNING objects.key
"""


class _Database(Protocol):
    """The two ``PooledPostgresDatabase`` entry points the sweep uses."""

    def cursor(self) -> AbstractContextManager[Any]: ...

    def transaction(self) -> AbstractContextManager[None]: ...


class PostgresSignalRetentionMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: _Database
    _decode: Callable[[str, Any], Any]

    def _signal_state_page(
        self,
        kind: str,
        *,
        after_key: str,
        limit: int,
        stale_before: datetime,
    ) -> SignalPage:
        with self._db.cursor() as cursor:
            cursor.execute(
                _SIGNAL_STATE_PAGE, (_utc_text(stale_before), kind, after_key, limit)
            )
            rows = cursor.fetchall()
        return SignalPage(
            rows=tuple(
                SignalRow(key, self._decode(kind, payload), payload)
                for key, payload in rows
                if payload is not None
            ),
            scanned=len(rows),
            last_key=rows[-1][0] if rows else None,
        )

    def _delete_unchanged_signal_states(
        self, kind: str, victims: Sequence[SignalRow]
    ) -> list[str]:
        """Delete the victims nobody holds and nobody changed, in one statement."""

        if not victims:
            return []
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    _DELETE_UNCHANGED_SIGNAL_STATES,
                    (
                        [row.key for row in victims],
                        [str(row.token) for row in victims],
                        kind,
                        kind,
                        kind,
                    ),
                )
                return sorted(row[0] for row in cursor.fetchall())
