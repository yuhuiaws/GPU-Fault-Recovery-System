from typing import Any, Callable
from datetime import datetime

from gpu_fault.models import (
    EfaTrafficAdminAction,
    EfaTrafficAdminDecision,
    EfaTrafficState,
)
from gpu_fault.store.shared.errors import NotFoundError


class SqliteEfaTrafficMixin:
    # Attributes supplied by the composed concrete implementation.
    _apply_efa_traffic_admin_action: Callable[..., Any]
    _efa_traffic_admin_decision_id: Callable[..., Any]
    _next_efa_traffic_state: Callable[..., Any]

    _get_optional: Callable[..., Any]
    _lock: Any
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def observe_efa_traffic(self, **parameters):
        state_key = parameters["state_key"]
        with self._state_transaction(f"efa_traffic_state/{state_key}"):
            previous = self._get_optional("efa_traffic_state", state_key)
            state, emit = self._next_efa_traffic_state(previous, **parameters)
            if state is not previous:
                self._put("efa_traffic_state", state_key, state)
            return state, emit

    def get_efa_traffic_state(self, state_key: str) -> EfaTrafficState:
        with self._lock:
            state = self._get_optional("efa_traffic_state", state_key)
            if state is None:
                raise NotFoundError(state_key)
            return state

    def apply_efa_traffic_admin_action(
        self,
        *,
        state_key: str,
        event_id: str,
        action: EfaTrafficAdminAction,
        operator: str,
        reason: str,
        decided_at: datetime,
    ) -> EfaTrafficAdminDecision:
        decision_id = self._efa_traffic_admin_decision_id(event_id, action)
        with self._state_transaction(f"efa_traffic_state/{state_key}"):
            existing = self._get_optional("efa_traffic_admin_decision", decision_id)
            if existing is not None:
                return existing
            state = self._get_optional("efa_traffic_state", state_key)
            if state is None:
                raise NotFoundError(state_key)
            updated, decision = self._apply_efa_traffic_admin_action(
                state,
                event_id=event_id,
                action=action,
                operator=operator,
                reason=reason,
                decided_at=decided_at,
            )
            self._put("efa_traffic_state", state_key, updated)
            self._put(
                "efa_traffic_admin_decision",
                decision.decision_id,
                decision,
            )
            return decision
