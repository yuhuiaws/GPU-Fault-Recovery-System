"""EFA traffic state: the pure rules every store applies, and the
key/value templates that persist their result."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

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
from gpu_fault.store.shared.primitives import (
    GetOptionalRecord,
    PutRecord,
    StatementGuard,
    StateTransaction,
)


class SharedEfaTrafficRulesMixin:
    """The state machine itself; composed by all three stores."""

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


class SharedEfaTrafficMixin:
    """The rules applied to one persisted row; the key/value stores."""

    # Attributes supplied by the composed concrete implementation.
    _apply_efa_traffic_admin_action: Callable[
        ..., tuple[EfaTrafficState, EfaTrafficAdminDecision]
    ]
    _efa_traffic_admin_decision_id: Callable[[str, EfaTrafficAdminAction], str]
    _next_efa_traffic_state: Callable[..., tuple[EfaTrafficState, bool]]

    _get_optional: GetOptionalRecord
    _put: PutRecord
    _state_transaction: StateTransaction
    _statement_guard: StatementGuard

    def observe_efa_traffic(self, **parameters):
        state_key = parameters["state_key"]
        with self._state_transaction(f"efa_traffic_state/{state_key}"):
            previous = self._get_optional("efa_traffic_state", state_key)
            state, emit = self._next_efa_traffic_state(previous, **parameters)
            if state is not previous:
                self._put("efa_traffic_state", state_key, state)
            return state, emit

    def get_efa_traffic_state(self, state_key: str) -> EfaTrafficState:
        with self._statement_guard():
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
