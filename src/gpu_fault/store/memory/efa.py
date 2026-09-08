from datetime import datetime
from typing import Any, Callable

from gpu_fault.models import (
    EfaTrafficAdminAction,
    EfaTrafficAdminDecision,
    EfaTrafficState,
)
from gpu_fault.store.shared.errors import NotFoundError


class MemoryEfaTrafficMixin:
    # Attributes supplied by the composed concrete implementation.
    _efa_traffic_admin_decisions: Any
    _efa_traffic_states: Any

    _lock: Any

    # The state machine, from SharedEfaTrafficRulesMixin.
    _apply_efa_traffic_admin_action: Callable[
        ..., tuple[EfaTrafficState, EfaTrafficAdminDecision]
    ]
    _efa_traffic_admin_decision_id: Callable[[str, EfaTrafficAdminAction], str]
    _next_efa_traffic_state: Callable[..., tuple[EfaTrafficState, bool]]

    def observe_efa_traffic(self, **parameters):
        state_key = parameters["state_key"]
        with self._lock:
            state, emit = self._next_efa_traffic_state(
                self._efa_traffic_states.get(state_key),
                **parameters,
            )
            self._efa_traffic_states[state_key] = state
            return state, emit

    def get_efa_traffic_state(self, state_key: str) -> EfaTrafficState:
        with self._lock:
            state = self._efa_traffic_states.get(state_key)
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
        with self._lock:
            existing = self._efa_traffic_admin_decisions.get(decision_id)
            if existing is not None:
                return existing
            state = self._efa_traffic_states.get(state_key)
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
            self._efa_traffic_states[state_key] = updated
            self._efa_traffic_admin_decisions[decision.decision_id] = decision
            return decision
