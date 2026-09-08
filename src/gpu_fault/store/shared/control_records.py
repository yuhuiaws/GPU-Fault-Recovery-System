"""Control-record templates shared by the key/value stores.

Decisions, markers, diagnostics, plans, profiles, restart budgets and the
HyperPod identity/submission records are single rows keyed by their own id.
Every method here is one ``_get``/``_put`` or a short ``_state_transaction``
over them, and runs unchanged on SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.models import (
    CompletionDecision,
    EffectiveRuntimeProfile,
    NodeMarker,
    RecoveryPlan,
    RestartBudgetState,
    TerminalEvent,
)
from gpu_fault.store.shared.errors import NotFoundError, StaleWriteError
from gpu_fault.store.shared.primitives import (
    GetLink,
    GetOptionalRecord,
    GetRecord,
    PutRecord,
    StateKey,
    StatementGuard,
    StateTransaction,
    state_key,
)
from gpu_fault.store.shared.record_guards import record_matches_expected


class SharedControlRecordMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: GetRecord
    _get_link: GetLink
    _get_optional: GetOptionalRecord
    _put: PutRecord
    _state_key: StateKey
    _state_transaction: StateTransaction
    _statement_guard: StatementGuard

    @staticmethod
    def _restart_budget_key(cluster_id: str, job_id: str) -> str:
        return state_key((cluster_id, job_id))

    def reserve_job_restart(
        self,
        cluster_id: str,
        job_id: str,
        budget: int,
        reservation_id: str,
    ) -> tuple[RestartBudgetState, bool]:
        key = self._restart_budget_key(cluster_id, job_id)
        with self._state_transaction(f"restart_budget/{key}"):
            state = self._get_optional("restart_budget", key)
            if state is None:
                state = RestartBudgetState(
                    cluster_id=cluster_id,
                    job_id=job_id,
                    budget=budget,
                )
            elif state.budget != budget:
                raise ValueError(
                    "restart budget is immutable for "
                    f"{cluster_id}/{job_id}: configured={state.budget}, "
                    f"received={budget}"
                )
            if reservation_id in state.reservation_ids:
                return state, True
            if state.restart_count >= state.budget:
                self._put("restart_budget", key, state)
                return state, False
            state = state.model_copy(
                update={
                    "restart_count": state.restart_count + 1,
                    "reservation_ids": [
                        *state.reservation_ids,
                        reservation_id,
                    ],
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._put("restart_budget", key, state)
            return state, True

    def get_restart_budget(self, cluster_id: str, job_id: str) -> RestartBudgetState:
        return self._get(
            "restart_budget",
            self._restart_budget_key(cluster_id, job_id),
        )

    def release_job_restart(
        self,
        cluster_id: str,
        job_id: str,
        reservation_id: str,
    ) -> RestartBudgetState:
        key = self._restart_budget_key(cluster_id, job_id)
        with self._state_transaction(f"restart_budget/{key}"):
            state = self._get_optional("restart_budget", key)
            if state is None:
                raise NotFoundError(f"{cluster_id}/{job_id}")
            if reservation_id not in state.reservation_ids:
                return state
            reservations = [
                item for item in state.reservation_ids if item != reservation_id
            ]
            state = state.model_copy(
                update={
                    "restart_count": len(reservations),
                    "reservation_ids": reservations,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._put("restart_budget", key, state)
            return state

    def get_event_by_attempt(self, cluster_id: str, attempt_id: str) -> TerminalEvent:
        lookup = self._state_key((cluster_id, attempt_id))
        key = self._get_link("attempt_event", lookup)
        if key is None:
            raise NotFoundError(f"{cluster_id}/{attempt_id}")
        return self._get("event", key)

    def save_decision(self, decision: CompletionDecision) -> None:
        with self._statement_guard():
            self._put("decision", decision.event_key, decision)

    def get_decision_by_event(self, event_key: str) -> CompletionDecision | None:
        return self._get_optional("decision", event_key)

    def add_marker(self, marker: NodeMarker) -> None:
        with self._statement_guard():
            self._put("marker", marker.marker_id, marker)

    def save_plan(
        self,
        plan: RecoveryPlan,
        *,
        expected: RecoveryPlan | None = None,
    ) -> None:
        """See ``CompletionStore.save_plan`` (architecture review, item D2).

        Without ``expected`` the write is the single-statement upsert it always
        was. With it, the read-compare-write runs inside one write transaction
        so the compare-and-set holds on both key/value backends.
        """

        if expected is None:
            with self._statement_guard():
                self._put("plan", plan.plan_id, plan)
            return
        with self._state_transaction(f"plan/{plan.plan_id}"):
            if not record_matches_expected(
                self._get_optional("plan", plan.plan_id), expected
            ):
                raise StaleWriteError(f"plan/{plan.plan_id} changed since it was read")
            self._put("plan", plan.plan_id, plan)

    def get_plan(self, plan_id: str) -> RecoveryPlan:
        return self._get("plan", plan_id)

    def save_profile(self, profile: EffectiveRuntimeProfile) -> None:
        with self._statement_guard():
            self._put("profile", profile.profile_version, profile)

    def get_profile(self, version: str) -> EffectiveRuntimeProfile:
        return self._get("profile", version)

    @staticmethod
    def _hyperpod_identity_key(cluster_name: str, node_logical_id: str) -> str:
        return f"{cluster_name}/{node_logical_id}"

    def save_hyperpod_node_identity(self, identity):
        key = self._hyperpod_identity_key(
            identity.cluster_name,
            identity.node_logical_id,
        )
        with self._state_transaction(f"hyperpod_node_identity/{key}"):
            existing = self._get_optional("hyperpod_node_identity", key)
            if existing is not None and existing.observed_at > identity.observed_at:
                return existing
            self._put(
                "hyperpod_node_identity",
                key,
                identity,
            )
            return identity

    def get_hyperpod_node_identity(self, cluster_name: str, node_logical_id: str):
        return self._get(
            "hyperpod_node_identity",
            self._hyperpod_identity_key(cluster_name, node_logical_id),
        )

    @staticmethod
    def _hyperpod_submission_key(cluster_name: str, idempotency_key: str) -> str:
        return f"{cluster_name}/{idempotency_key}"

    def reserve_hyperpod_submission(self, record):
        key = self._hyperpod_submission_key(record.cluster_name, record.idempotency_key)
        with self._state_transaction(f"hyperpod_submission/{key}"):
            existing = self._get_optional("hyperpod_submission", key)
            if existing is not None:
                return existing, False
            self._put("hyperpod_submission", key, record)
            return record, True

    def save_hyperpod_submission(self, record) -> None:
        key = self._hyperpod_submission_key(record.cluster_name, record.idempotency_key)
        with self._statement_guard():
            self._put("hyperpod_submission", key, record)

    def get_hyperpod_submission(self, cluster_name: str, idempotency_key: str):
        return self._get(
            "hyperpod_submission",
            self._hyperpod_submission_key(cluster_name, idempotency_key),
        )
