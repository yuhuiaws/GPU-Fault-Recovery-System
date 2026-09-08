from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.installation_resources import InstallationResource
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    DiagnosticRequest,
    EffectiveRuntimeProfile,
    FaultIncident,
    NodeMarker,
    RecoveryAction,
    RecoveryPlan,
    RestartBudgetState,
    TerminalEvent,
    TriageReport,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.attempt_observation_support import (
    AttemptObservationTerminalSupport,
    reconcile_terminal_attempt_observations,
    terminalize_attempt_observation,
)
from gpu_fault.store.shared.cleanup_log import log_cleanup
from gpu_fault.store.shared.errors import NotFoundError, StaleWriteError
from gpu_fault.store.shared.evidence_pins import (
    EVIDENCE_PINNING_INCIDENT_STATES,
    evidence_pinned,
)
from gpu_fault.store.shared.record_guards import record_matches_expected


class MemoryControlRecordMixin(AttemptObservationTerminalSupport):
    # Attributes supplied by the composed concrete implementation.
    _decisions: Any
    _incident_by_event: Any
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
    _incidents: dict[str, FaultIncident]
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
            # Records an open incident still needs are skipped, not deleted
            # (architecture review 2026-09-07, item D6; ``evidence_pins``).
            pinning = [
                incident
                for incident in self._incidents.values()
                if incident.state in EVIDENCE_PINNING_INCIDENT_STATES
            ]
            expired = sorted(
                (
                    (key, item)
                    for key, item in self._raw_evidence.items()
                    if item.expires_at <= observed
                    and not evidence_pinned(item, pinning)
                ),
                key=lambda value: (
                    value[1].expires_at,
                    value[1].record_id,
                ),
            )[:limit]
            for key, _ in expired:
                self._raw_evidence.pop(key, None)
        return log_cleanup("raw_evidence", [item.record_id for _, item in expired])

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
            terminalized = reconcile_terminal_attempt_observations(self, limit)
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
        return {
            "gpu_finding_history": log_cleanup(
                "gpu_finding_history", [key for key, _finding in expired]
            ),
            "attempt_observation_terminalized": terminalized,
        }

    def cleanup_inactive_markers(self, *, older_than: datetime, limit: int) -> int:
        with self._lock:
            marker_ids = [
                marker.marker_id
                for marker in sorted(
                    self._markers.values(),
                    key=lambda item: (item.observed_at, item.marker_id),
                )
                if not marker.active
                and marker.observed_at <= older_than
                and marker.incident_id not in self._incidents
            ][:limit]
            for marker_id in marker_ids:
                del self._markers[marker_id]
        return log_cleanup("marker", marker_ids)

    def cleanup_completion_records(self, *, older_than: datetime, limit: int) -> int:
        with self._lock:
            live_event_ids = {
                incident.event_id for incident in self._incidents.values()
            }
            candidates = []
            for decision in self._decisions.values():
                if decision.status is DecisionStatus.PENDING_TRIAGE:
                    continue
                event = self._events.get(decision.event_key)
                if event is None or event.ended_at > older_than:
                    continue
                linked = self._incident_by_event.get(decision.event_key)
                if linked is not None and linked in self._incidents:
                    continue
                if decision.event_key in live_event_ids:
                    continue
                plan = (
                    self._plans.get(decision.recovery_plan_id)
                    if decision.recovery_plan_id
                    else None
                )
                if plan is not None and plan.incident_id in self._incidents:
                    continue
                candidates.append((event.ended_at, decision, event, plan))
            candidates.sort(key=lambda item: (item[0], item[1].event_key))
            removed = []
            for _ended_at, decision, event, plan in candidates[:limit]:
                key = decision.event_key
                self._decisions.pop(key, None)
                self._events.pop(key, None)
                self._attempt_event_keys.pop((event.cluster_id, event.attempt_id), None)
                if decision.diagnostic_request_id:
                    self._diagnostics.pop(decision.diagnostic_request_id, None)
                    self._triage_reports.pop(decision.diagnostic_request_id, None)
                if plan is not None:
                    self._plans.pop(plan.plan_id, None)
                removed.append(key)
        return log_cleanup("completion_decision", removed)

    def save_event_if_absent(self, event: TerminalEvent) -> bool:
        with self._lock:
            existing = self._events.get(event.event_key)
            inserted = existing is None
            terminal = existing or event
            if inserted:
                self._events[event.event_key] = event
                self._attempt_event_keys[(event.cluster_id, event.attempt_id)] = (
                    event.event_key
                )
            terminalize_attempt_observation(self, terminal)
            return inserted

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

    @contextmanager
    def _completion_transaction(self, event_key: str) -> Iterator[None]:
        with self._lock:
            yield

    def completion_transaction(self, event_key: str) -> AbstractContextManager[None]:
        # One process, one RLock: re-entrant, so the nested store writes the
        # completion service makes inside it take the same lock again.
        return self._completion_transaction(event_key)

    def save_decision(self, decision: CompletionDecision) -> None:
        with self._lock:
            self._decisions[decision.event_key] = decision

    def get_decision_by_event(self, event_key: str) -> CompletionDecision | None:
        with self._lock:
            return self._decisions.get(event_key)

    def _decision_age_key(self, decision: CompletionDecision) -> datetime | None:
        # Diagnostic ``created_at``, else the terminal event's ``ended_at``;
        # ``None`` (neither row) is never older than a cutoff (F-7).
        if decision.diagnostic_request_id:
            request = self._diagnostics.get(decision.diagnostic_request_id)
            if request is not None:
                created_at: datetime | None = request.created_at
                return created_at
        event = self._events.get(decision.event_key)
        ended_at: datetime | None = event.ended_at if event is not None else None
        return ended_at

    def list_decisions_by_status(
        self,
        status: DecisionStatus,
        *,
        older_than: datetime | None = None,
        limit: int = 100,
    ) -> list[CompletionDecision]:
        if limit < 1:
            return []
        with self._lock:
            candidates = [
                (self._decision_age_key(decision), decision)
                for decision in self._decisions.values()
                if decision.status is status
            ]
        if older_than is not None:
            candidates = [
                (created_at, decision)
                for created_at, decision in candidates
                if created_at is not None and created_at <= older_than
            ]
        floor = datetime.min.replace(tzinfo=timezone.utc)
        candidates.sort(
            key=lambda item: (item[0] or floor, item[1].event_key),
        )
        return [decision for _created_at, decision in candidates[:limit]]

    def decision_status_counts(self) -> dict[DecisionStatus, int]:
        with self._lock:
            counts = {status: 0 for status in DecisionStatus}
            for decision in self._decisions.values():
                counts[decision.status] += 1
            return counts

    def count_completion_events_without_decision(self) -> int:
        with self._lock:
            return sum(1 for key in self._events if key not in self._decisions)

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
                    and marker.trusted
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

    def list_markers_in_scope_window(
        self,
        *,
        node_ids: set[str],
        gpu_uuids: set[str],
        fabric_partitions: set[str],
        observed_from: datetime,
        observed_to: datetime,
        limit: int = 1000,
    ) -> list[NodeMarker]:
        """Actionable markers whose scope touches an allocation, newest first.

        The terminal-event correlator asks this once per completed training
        attempt. It matches on GPU UUID and fabric partition as well as node ID,
        because a marker raised by a fabric-level fault names the partition, not
        the nodes attached to it.
        """
        if limit < 1 or not (node_ids or gpu_uuids or fabric_partitions):
            return []
        with self._lock:
            return sorted(
                (
                    marker
                    for marker in self._markers.values()
                    if marker.active
                    and marker.trusted
                    and marker.recommended_action is not None
                    and observed_from <= marker.observed_at <= observed_to
                    and (
                        set(marker.scope.node_ids).intersection(node_ids)
                        or set(marker.scope.gpu_uuids).intersection(gpu_uuids)
                        or set(marker.scope.fabric_partitions).intersection(
                            fabric_partitions
                        )
                    )
                ),
                key=lambda marker: (marker.observed_at, marker.marker_id),
                reverse=True,
            )[:limit]

    def list_active_markers_for_nodes(
        self,
        node_ids: set[str],
        actions: set[RecoveryAction],
        cluster_id: str | None = None,
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
                    # Tenant scope (H-14): when a cluster is given, a marker
                    # matches only if it is stamped with that same cluster.
                    # A legacy marker with no cluster_id is never matched to a
                    # specific cluster, so a colliding node_id in another
                    # tenant cannot read it.
                    and (cluster_id is None or marker.cluster_id == cluster_id)
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

    def save_plan(
        self,
        plan: RecoveryPlan,
        *,
        expected: RecoveryPlan | None = None,
    ) -> None:
        """See ``CompletionStore.save_plan`` (architecture review, item D2)."""

        with self._lock:
            if expected is not None and not record_matches_expected(
                self._plans.get(plan.plan_id), expected
            ):
                raise StaleWriteError(f"plan/{plan.plan_id} changed since it was read")
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
