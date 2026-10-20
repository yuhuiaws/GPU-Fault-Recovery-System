from datetime import datetime
from typing import Any, Callable

from gpu_fault.efa_traffic_state import (
    continued_efa_state,
    efa_state_belongs_to,
    legacy_efa_state_due,
    legacy_efa_traffic_state_key,
)
from gpu_fault.models import (
    EfaTrafficAdminAction,
    EfaTrafficAdminDecision,
    EfaTrafficState,
)
from gpu_fault.store.shared.efa import efa_attempt_identity
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict, NotFoundError


class MemoryEfaTrafficMixin:
    """``SharedEfaTrafficMixin`` over dicts: one row per node, and the
    per-attempt row of a release before batch 145 consulted the same way."""

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
        owner = (parameters["job_id"], parameters["attempt_id"])
        with self._lock:
            node = self._efa_traffic_states.get(state_key)
            legacy_key = (
                legacy_efa_traffic_state_key(state_key, *owner)
                if legacy_efa_state_due(node, *owner, parameters["observed_at"])
                else None
            )
            legacy = (
                None if legacy_key is None else self._efa_traffic_states.get(legacy_key)
            )
            previous = continued_efa_state(node, legacy, *owner)
            state, emit = self._next_efa_traffic_state(previous, **parameters)
            if state is not previous:
                self._efa_traffic_states[state_key] = state
                if legacy_key is not None:
                    self._efa_traffic_states.pop(legacy_key, None)
            return state, emit

    def get_efa_traffic_state(
        self,
        state_key: str,
        *,
        job_id: str | None = None,
        attempt_id: str | None = None,
    ) -> EfaTrafficState:
        owner = efa_attempt_identity(job_id, attempt_id)
        with self._lock:
            state = self._efa_traffic_states.get(state_key)
            if owner is not None:
                state = continued_efa_state(
                    state,
                    self._efa_traffic_states.get(
                        legacy_efa_traffic_state_key(state_key, *owner)
                    ),
                    *owner,
                )
                if state is not None and not efa_state_belongs_to(state, *owner):
                    state = None
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
        job_id: str | None = None,
        attempt_id: str | None = None,
    ) -> EfaTrafficAdminDecision:
        decision_id = self._efa_traffic_admin_decision_id(event_id, action)
        owner = efa_attempt_identity(job_id, attempt_id)
        legacy_key = (
            None if owner is None else legacy_efa_traffic_state_key(state_key, *owner)
        )
        with self._lock:
            existing = self._efa_traffic_admin_decisions.get(decision_id)
            if existing is not None:
                return existing
            state = self._efa_traffic_states.get(state_key)
            if owner is not None and legacy_key is not None:
                if state is not None and not efa_state_belongs_to(state, *owner):
                    raise EfaTrafficAdminConflict(
                        "the node's EFA traffic state now belongs to attempt "
                        f"{state.job_id}/{state.attempt_id}"
                    )
                state = continued_efa_state(
                    state, self._efa_traffic_states.get(legacy_key), *owner
                )
            if state is None:
                raise NotFoundError(state_key)
            updated, decision = self._apply_efa_traffic_admin_action(
                state.model_copy(update={"state_key": state_key}),
                event_id=event_id,
                action=action,
                operator=operator,
                reason=reason,
                decided_at=decided_at,
            )
            if legacy_key is not None:
                self._efa_traffic_states.pop(legacy_key, None)
            self._efa_traffic_states[state_key] = updated
            self._efa_traffic_admin_decisions[decision.decision_id] = decision
            return decision
