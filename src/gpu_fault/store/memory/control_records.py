from __future__ import annotations

from typing import Any

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    CompletionDecision,
    DiagnosticRequest,
    EffectiveRuntimeProfile,
    NodeMarker,
    RecoveryAction,
    RecoveryPlan,
    RestartBudgetState,
    TerminalEvent,
    TriageReport,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.installation_resources import InstallationResource
from gpu_fault.store.shared.errors import NotFoundError


class MemoryControlRecordMixin:
    # Attributes supplied by the composed concrete implementation.
    _decisions: Any
    _diagnostics: Any
    _hyperpod_node_identities: Any
    _hyperpod_submissions: Any
    _installation_resources: dict[str, InstallationResource]
    _markers: Any
    _profiles: Any
    _raw_evidence: Any
    _restart_budgets: Any
    _triage_reports: Any

    _attempt_event_keys: Any
    _events: Any
    _gpu_finding_history: Any
    _lock: Any
    _plans: Any
    _workflows: Any

    def save_raw_evidence(self, record, *, max_records_per_node: int) -> None:
        with self._lock:
            storage_key = (
                record.cluster_id,
                record.node_id,
                record.record_id,
            )
            self._raw_evidence[storage_key] = record
            now = datetime.now(timezone.utc)
            matching = sorted(
                (
                    item
                    for item in self._raw_evidence.values()
                    if item.expires_at > now
                    if item.cluster_id == record.cluster_id
                    and item.node_id == record.node_id
                ),
                key=lambda item: (
                    item.observed_at,
                    item.record_id,
                ),
                reverse=True,
            )
            for item in matching[max_records_per_node:]:
                self._raw_evidence.pop(
                    (
                        item.cluster_id,
                        item.node_id,
                        item.record_id,
                    ),
                    None,
                )

    def list_raw_evidence(
        self,
        cluster_id: str,
        *,
        node_id: str | None = None,
        attempt_id: str | None = None,
        kind=None,
        limit: int = 100,
    ):
        now = datetime.now(timezone.utc)
        with self._lock:
            result = [
                item
                for item in self._raw_evidence.values()
                if item.cluster_id == cluster_id
                and item.expires_at > now
                and (node_id is None or item.node_id == node_id)
                and (attempt_id is None or attempt_id in item.attempt_ids)
                and (kind is None or item.kind == kind)
            ]
        return sorted(
            result,
            key=lambda item: (
                item.observed_at,
                item.record_id,
            ),
            reverse=True,
        )[:limit]

    def cleanup_expired_raw_evidence(
        self,
        *,
        now: datetime | None = None,
        limit: int = 1000,
    ) -> int:
        observed = now or datetime.now(timezone.utc)
        with self._lock:
            expired = sorted(
                (
                    (key, item)
                    for key, item in self._raw_evidence.items()
                    if item.expires_at <= observed
                ),
                key=lambda value: (
                    value[1].expires_at,
                    value[1].record_id,
                ),
            )[:limit]
            for key, _ in expired:
                self._raw_evidence.pop(key, None)
        return len(expired)

    def cleanup_hot_state(
        self,
        *,
        now: datetime | None = None,
        finding_history_retention: timedelta = timedelta(days=30),
        limit: int = 1000,
        **_kwargs,
    ) -> dict[str, int]:
        observed = now or datetime.now(timezone.utc)
        cutoff = observed - finding_history_retention
        with self._lock:
            expired = sorted(
                (
                    (key, finding)
                    for key, finding in self._gpu_finding_history.items()
                    if finding.observed_at <= cutoff
                ),
                key=lambda value: (
                    value[1].observed_at,
                    value[0],
                ),
            )[:limit]
            for key, _finding in expired:
                self._gpu_finding_history.pop(key, None)
        return {"gpu_finding_history": len(expired)}

    def save_event_if_absent(self, event: TerminalEvent) -> bool:
        with self._lock:
            if event.event_key in self._events:
                return False
            self._events[event.event_key] = event
            self._attempt_event_keys[(event.cluster_id, event.attempt_id)] = (
                event.event_key
            )
            return True

    def reserve_job_restart(
        self,
        cluster_id: str,
        job_id: str,
        budget: int,
        reservation_id: str,
    ) -> tuple[RestartBudgetState, bool]:
        key = (cluster_id, job_id)
        with self._lock:
            state = self._restart_budgets.get(key)
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
            self._restart_budgets[key] = state
            return state, True

    def get_restart_budget(self, cluster_id: str, job_id: str) -> RestartBudgetState:
        with self._lock:
            state = self._restart_budgets.get((cluster_id, job_id))
            if state is None:
                raise NotFoundError(f"{cluster_id}/{job_id}")
            return state

    def release_job_restart(
        self,
        cluster_id: str,
        job_id: str,
        reservation_id: str,
    ) -> RestartBudgetState:
        key = (cluster_id, job_id)
        with self._lock:
            state = self._restart_budgets.get(key)
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
            self._restart_budgets[key] = state
            return state

    def get_event_by_attempt(self, cluster_id: str, attempt_id: str) -> TerminalEvent:
        with self._lock:
            key = self._attempt_event_keys.get((cluster_id, attempt_id))
            if not key:
                raise NotFoundError(f"{cluster_id}/{attempt_id}")
            return self._events[key]

    def save_decision(self, decision: CompletionDecision) -> None:
        with self._lock:
            self._decisions[decision.event_key] = decision

    def get_decision_by_event(self, event_key: str) -> CompletionDecision | None:
        with self._lock:
            return self._decisions.get(event_key)

    def get_decision_by_attempt(
        self, cluster_id: str, attempt_id: str
    ) -> CompletionDecision:
        event = self.get_event_by_attempt(cluster_id, attempt_id)
        decision = self.get_decision_by_event(event.event_key)
        if decision is None:
            raise NotFoundError(f"{cluster_id}/{attempt_id}")
        return decision

    def add_marker(self, marker: NodeMarker) -> None:
        with self._lock:
            self._markers[marker.marker_id] = marker

    def list_markers(self) -> list[NodeMarker]:
        with self._lock:
            return list(self._markers.values())

    def list_markers_for_incident(self, incident_id: str) -> list[NodeMarker]:
        with self._lock:
            return sorted(
                (
                    marker
                    for marker in self._markers.values()
                    if marker.incident_id == incident_id
                ),
                key=lambda marker: (
                    marker.observed_at,
                    marker.marker_id,
                ),
            )

    def list_recent_markers_for_nodes(
        self,
        node_ids: set[str],
        observed_after: datetime,
        *,
        source_boot_id: str | None = None,
        limit: int = 1000,
    ) -> list[NodeMarker]:
        if not node_ids or limit < 1:
            return []
        with self._lock:
            return sorted(
                (
                    marker
                    for marker in self._markers.values()
                    if marker.active
                    and (
                        marker.observed_at >= observed_after
                        or (
                            source_boot_id is not None
                            and marker.source_boot_id == source_boot_id
                        )
                    )
                    and set(marker.scope.node_ids).intersection(node_ids)
                ),
                key=lambda marker: (
                    marker.observed_at,
                    marker.marker_id,
                ),
                reverse=True,
            )[:limit]

    def list_active_markers_for_nodes(
        self,
        node_ids: set[str],
        actions: set[RecoveryAction],
    ) -> list[NodeMarker]:
        if not node_ids or not actions:
            return []
        with self._lock:
            return sorted(
                (
                    marker
                    for marker in self._markers.values()
                    if marker.active
                    and marker.trusted
                    and marker.recommended_action in actions
                    and set(marker.scope.node_ids).intersection(node_ids)
                ),
                key=lambda marker: marker.observed_at,
                reverse=True,
            )

    def save_diagnostic(self, request: DiagnosticRequest) -> None:
        with self._lock:
            self._diagnostics[request.request_id] = request

    def get_diagnostic(self, request_id: str) -> DiagnosticRequest:
        with self._lock:
            request = self._diagnostics.get(request_id)
            if request is None:
                raise NotFoundError(request_id)
            return request

    def save_triage_report(self, report: TriageReport) -> None:
        with self._lock:
            self._triage_reports[report.request_id] = report

    def save_plan(self, plan: RecoveryPlan) -> None:
        with self._lock:
            self._plans[plan.plan_id] = plan

    def get_plan(self, plan_id: str) -> RecoveryPlan:
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise NotFoundError(plan_id)
            return plan

    def save_profile(self, profile: EffectiveRuntimeProfile) -> None:
        with self._lock:
            self._profiles[profile.profile_version] = profile

    def get_profile(self, version: str) -> EffectiveRuntimeProfile:
        with self._lock:
            profile = self._profiles.get(version)
            if profile is None:
                raise NotFoundError(version)
            return profile

    def save_installation_resource(
        self,
        resource: InstallationResource,
    ) -> InstallationResource:
        key = f"{resource.site_id}/{resource.resource_key}"
        with self._lock:
            existing = self._installation_resources.get(key)
            if (
                existing is not None
                and existing.immutable_identity() != resource.immutable_identity()
            ):
                raise ValueError("installation resource identity cannot change")
            self._installation_resources[key] = resource
            return resource

    def get_installation_resource(
        self,
        site_id: str,
        resource_key: str,
    ) -> InstallationResource:
        with self._lock:
            resource = self._installation_resources.get(f"{site_id}/{resource_key}")
            if resource is None:
                raise NotFoundError(resource_key)
            return resource

    def list_installation_resources(
        self,
        site_id: str | None = None,
    ) -> list[InstallationResource]:
        with self._lock:
            return sorted(
                (
                    item
                    for item in self._installation_resources.values()
                    if site_id is None or item.site_id == site_id
                ),
                key=lambda item: (item.site_id, item.resource_key),
            )

    def get_preempting_successor(
        self, predecessor_workflow_id: str
    ) -> WorkflowRequest | None:
        with self._lock:
            matches = sorted(
                (
                    workflow
                    for workflow in self._workflows.values()
                    if workflow.predecessor_workflow_id == predecessor_workflow_id
                    and workflow.preempt_predecessor
                    and workflow.status
                    in {
                        WorkflowStatus.PENDING,
                        WorkflowStatus.SAFETY_PENDING,
                    }
                ),
                key=lambda item: (
                    item.created_at,
                    item.request_id,
                ),
            )
            return matches[0] if matches else None

    def save_hyperpod_node_identity(self, identity):
        with self._lock:
            key = (
                identity.cluster_name,
                identity.node_logical_id,
            )
            existing = self._hyperpod_node_identities.get(key)
            if existing is not None and existing.observed_at > identity.observed_at:
                return existing
            self._hyperpod_node_identities[key] = identity
            return identity

    def get_hyperpod_node_identity(self, cluster_name: str, node_logical_id: str):
        with self._lock:
            identity = self._hyperpod_node_identities.get(
                (cluster_name, node_logical_id)
            )
            if identity is None:
                raise NotFoundError(f"{cluster_name}/{node_logical_id}")
            return identity

    def list_hyperpod_node_identities(self, cluster_name: str | None = None):
        with self._lock:
            identities = list(self._hyperpod_node_identities.values())
        if cluster_name:
            identities = [
                item for item in identities if item.cluster_name == cluster_name
            ]
        return sorted(
            identities,
            key=lambda item: (
                item.cluster_name,
                item.node_logical_id,
            ),
        )

    @staticmethod
    def _hyperpod_submission_key(cluster_name: str, idempotency_key: str) -> str:
        return f"{cluster_name}/{idempotency_key}"

    def reserve_hyperpod_submission(self, record):
        """Claim an idempotency key before calling the provider API.

        Returns (stored_record, reserved). ``reserved`` is True only for
        the caller that created the record, so exactly one caller may
        invoke BatchReboot/BatchReplaceClusterNodes for a given key even
        across control-plane replicas and executor restarts.
        """

        key = self._hyperpod_submission_key(record.cluster_name, record.idempotency_key)
        with self._lock:
            existing = self._hyperpod_submissions.get(key)
            if existing is not None:
                return existing, False
            self._hyperpod_submissions[key] = record
            return record, True

    def save_hyperpod_submission(self, record) -> None:
        key = self._hyperpod_submission_key(record.cluster_name, record.idempotency_key)
        with self._lock:
            self._hyperpod_submissions[key] = record

    def get_hyperpod_submission(self, cluster_name: str, idempotency_key: str):
        key = self._hyperpod_submission_key(cluster_name, idempotency_key)
        with self._lock:
            record = self._hyperpod_submissions.get(key)
            if record is None:
                raise NotFoundError(key)
            return record
