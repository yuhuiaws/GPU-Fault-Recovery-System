from typing import Any
from datetime import datetime

from gpu_fault.efa_traffic_state import (
    apply_efa_traffic_admin_action as _apply_efa_action,
)
from gpu_fault.efa_traffic_state import (
    efa_traffic_admin_decision_id as _efa_decision_id,
)
from gpu_fault.efa_traffic_state import (
    efa_traffic_state_key as _efa_state_key,
)
from gpu_fault.efa_traffic_state import (
    next_efa_traffic_state as _next_efa_state,
)
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

    _next_efa_traffic_state = staticmethod(_next_efa_state)
    _efa_traffic_admin_decision_id = staticmethod(_efa_decision_id)
    _apply_efa_traffic_admin_action = staticmethod(_apply_efa_action)

    @staticmethod
    def efa_traffic_state_key(
        cluster_id: str,
        node_id: str,
        job_id: str,
        attempt_id: str,
    ) -> str:
        return str(_efa_state_key(cluster_id, node_id, job_id, attempt_id))

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
