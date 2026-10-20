from datetime import datetime
from typing import Any, Callable

from gpu_fault.efa_traffic_state import (
    efa_state_belongs_to,
    legacy_efa_traffic_state_key,
)
from gpu_fault.models import (
    EfaTrafficAdminAction,
    EfaTrafficAdminDecision,
    EfaTrafficState,
)
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict, NotFoundError


class MemoryEfaTrafficMixin:
    """``SharedEfaTrafficMixin`` over dicts: one row per node, and the
    per-attempt row of a release before batch 145 read once and dropped."""

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
            previous = self._efa_traffic_states.get(state_key)
            legacy_key = None
            if previous is None or (
                not efa_state_belongs_to(previous, *owner)
                and parameters["observed_at"] > previous.observed_at
            ):
                legacy_key = legacy_efa_traffic_state_key(state_key, *owner)
                previous = self._efa_traffic_states.get(legacy_key, previous)
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
        with self._lock:
            state = self._efa_traffic_states.get(state_key)
            if job_id is not None and attempt_id is not None:
                if state is None or not efa_state_belongs_to(state, job_id, attempt_id):
                    state = self._efa_traffic_states.get(
                        legacy_efa_traffic_state_key(state_key, job_id, attempt_id)
                    )
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
        legacy_key = (
            legacy_efa_traffic_state_key(state_key, job_id, attempt_id)
            if job_id is not None and attempt_id is not None
            else None
        )
        with self._lock:
            existing = self._efa_traffic_admin_decisions.get(decision_id)
            if existing is not None:
                return existing
            state = self._efa_traffic_states.get(state_key)
            if legacy_key is not None and (
                state is None or not efa_state_belongs_to(state, job_id, attempt_id)
            ):
                legacy = self._efa_traffic_states.get(legacy_key)
                if legacy is None and state is not None:
                    raise EfaTrafficAdminConflict(
                        "the node's EFA traffic state now belongs to attempt "
                        f"{state.job_id}/{state.attempt_id}"
                    )
                state = (
                    None
                    if legacy is None
                    else legacy.model_copy(update={"state_key": state_key})
                )
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
            if legacy_key is not None:
                self._efa_traffic_states.pop(legacy_key, None)
            self._efa_traffic_states[state_key] = updated
            self._efa_traffic_admin_decisions[decision.decision_id] = decision
            return decision
