"""EFA traffic state: the pure rules every store applies, and the
key/value templates that persist their result."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from typing import Callable

from gpu_fault.efa_traffic_state import (
    apply_efa_traffic_admin_action as _apply_efa_action,
)
from gpu_fault.efa_traffic_state import (
    efa_traffic_admin_decision_id as _efa_decision_id,
)
from gpu_fault.efa_traffic_state import (
    continued_efa_state,
    efa_state_belongs_to,
    legacy_efa_state_due,
    legacy_efa_traffic_state_key,
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
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict, NotFoundError
from gpu_fault.store.shared.primitives import (
    DeleteRecord,
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
    def efa_traffic_state_key(cluster_id: str, node_id: str) -> str:
        return str(_efa_state_key(cluster_id, node_id))


class SharedEfaTrafficMixin:
    """The rules applied to one persisted row; the key/value stores.

    One row per node (``cluster/node``) holds the state of the attempt that
    owns it. A release before batch 145 kept a row per attempt
    (``cluster/node/job/attempt``). That row is consulted, under its own lock
    (a writer of the older release holds it), when the node row is not the
    attempt's or is far behind the sample (``legacy_efa_state_due``); the
    newer of the two continues and the per-attempt row is deleted once the
    node row is written. So an upgrade does not restart a running attempt at
    WARMUP, a rollback and a new upgrade do not resume a node row the older
    release left behind, and no per-attempt row stays for a rollback to
    resume from.
    """

    # Attributes supplied by the composed concrete implementation.
    _apply_efa_traffic_admin_action: Callable[
        ..., tuple[EfaTrafficState, EfaTrafficAdminDecision]
    ]
    _efa_traffic_admin_decision_id: Callable[[str, EfaTrafficAdminAction], str]
    _next_efa_traffic_state: Callable[..., tuple[EfaTrafficState, bool]]

    _delete: DeleteRecord
    _get_optional: GetOptionalRecord
    _put: PutRecord
    _state_transaction: StateTransaction
    _statement_guard: StatementGuard

    def observe_efa_traffic(self, **parameters):
        state_key = parameters["state_key"]
        owner = (parameters["job_id"], parameters["attempt_id"])
        with self._state_transaction(f"efa_traffic_state/{state_key}"):
            node = self._get_optional("efa_traffic_state", state_key)
            legacy_key = (
                legacy_efa_traffic_state_key(state_key, *owner)
                if legacy_efa_state_due(node, *owner, parameters["observed_at"])
                else None
            )
            with (
                nullcontext()
                if legacy_key is None
                else self._state_transaction(f"efa_traffic_state/{legacy_key}")
            ):
                legacy = (
                    None
                    if legacy_key is None
                    else self._get_optional("efa_traffic_state", legacy_key)
                )
                previous = continued_efa_state(node, legacy, *owner)
                state, emit = self._next_efa_traffic_state(previous, **parameters)
                if state is not previous:
                    self._put("efa_traffic_state", state_key, state)
                    if legacy_key is not None and legacy is not None:
                        self._delete("efa_traffic_state", legacy_key)
                return state, emit

    def get_efa_traffic_state(
        self,
        state_key: str,
        *,
        job_id: str | None = None,
        attempt_id: str | None = None,
    ) -> EfaTrafficState:
        """The node's row; with ``job_id`` and ``attempt_id``, that attempt's state.

        The attempt's state is the node row while it is the attempt's, or the
        attempt's per-attempt row of the release before when that is newer;
        a node row another attempt took over is not it (``NotFoundError``).
        """

        owner = efa_attempt_identity(job_id, attempt_id)
        with self._statement_guard():
            state = self._get_optional("efa_traffic_state", state_key)
            if owner is not None:
                state = continued_efa_state(
                    state,
                    self._get_optional(
                        "efa_traffic_state",
                        legacy_efa_traffic_state_key(state_key, *owner),
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
        """Apply an administrator's SPIKE decision to the named attempt's state.

        With ``job_id``/``attempt_id`` the state must be that attempt's: a node
        row another attempt took over is a conflict (the spike's attempt left
        the node), and a newer state in the attempt's per-attempt row moves to
        the node row with the decision.
        """

        decision_id = self._efa_traffic_admin_decision_id(event_id, action)
        owner = efa_attempt_identity(job_id, attempt_id)
        legacy_key = (
            None if owner is None else legacy_efa_traffic_state_key(state_key, *owner)
        )
        with (
            self._state_transaction(f"efa_traffic_state/{state_key}"),
            nullcontext()
            if legacy_key is None
            else self._state_transaction(f"efa_traffic_state/{legacy_key}"),
        ):
            existing = self._get_optional("efa_traffic_admin_decision", decision_id)
            if existing is not None:
                return existing
            state = self._get_optional("efa_traffic_state", state_key)
            if owner is not None and legacy_key is not None:
                if state is not None and not efa_state_belongs_to(state, *owner):
                    raise EfaTrafficAdminConflict(
                        "the node's EFA traffic state now belongs to attempt "
                        f"{state.job_id}/{state.attempt_id}"
                    )
                state = continued_efa_state(
                    state, self._get_optional("efa_traffic_state", legacy_key), *owner
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
                self._delete("efa_traffic_state", legacy_key)
            self._put("efa_traffic_state", state_key, updated)
            self._put(
                "efa_traffic_admin_decision",
                decision.decision_id,
                decision,
            )
            return decision


def efa_attempt_identity(
    job_id: str | None, attempt_id: str | None
) -> tuple[str, str] | None:
    """``(job_id, attempt_id)``, or None for neither; one alone is a caller bug."""

    if job_id is None and attempt_id is None:
        return None
    if job_id is None or attempt_id is None:
        raise ValueError("an EFA traffic attempt is named by job_id and attempt_id")
    return job_id, attempt_id
