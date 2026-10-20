"""The health-signal retention's two primitives over the in-memory dicts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from threading import RLock
from typing import Any

from gpu_fault.store.shared.signal_retention import (
    SignalPage,
    SignalRow,
    signal_state_clock,
)


class MemorySignalRetentionMixin:
    # Attributes supplied by the composed concrete implementation.
    _efa_traffic_states: dict[str, Any]
    _health_signal_states: dict[str, Any]
    _lock: RLock

    def _signal_state_rows(self, kind: str) -> dict[str, Any]:
        return {
            "health_signal_state": self._health_signal_states,
            "efa_traffic_state": self._efa_traffic_states,
        }[kind]

    def _signal_state_page(
        self,
        kind: str,
        *,
        after_key: str,
        limit: int,
        stale_before: datetime,
    ) -> SignalPage:
        with self._lock:
            rows = self._signal_state_rows(kind)
            keys = sorted(key for key in rows if key > after_key)[:limit]
            page = [(key, rows[key]) for key in keys]
        return SignalPage(
            rows=tuple(
                SignalRow(key, state, state)
                for key, state in page
                if signal_state_clock(state) < stale_before
            ),
            scanned=len(page),
            last_key=keys[-1] if keys else None,
        )

    def _delete_unchanged_signal_states(
        self, kind: str, victims: Sequence[SignalRow]
    ) -> list[str]:
        """Delete each row still holding the very state that was judged."""

        with self._lock:
            rows = self._signal_state_rows(kind)
            deleted = [row.key for row in victims if rows.get(row.key) is row.token]
            for key in deleted:
                del rows[key]
        return deleted
