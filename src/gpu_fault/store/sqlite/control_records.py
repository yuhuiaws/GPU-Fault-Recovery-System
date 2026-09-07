from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, cast

from gpu_fault.installation_resources import InstallationResource
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    DiagnosticRequest,
    NodeMarker,
    RecoveryAction,
    TerminalEvent,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.attempt_observation_support import (
    AttemptObservationTerminalSupport,
    reconcile_terminal_attempt_observations,
    terminalize_attempt_observation,
)
from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


class SqliteControlRecordMixin(AttemptObservationTerminalSupport):
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _delete: Callable[..., Any]
    _get: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _link: Callable[..., Any]
    _list: Callable[..., Any]
    _put: Callable[..., Any]
    _state_key: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def cleanup_hot_state(
        self,
        *,
        now: datetime | None = None,
        finding_history_retention: timedelta = timedelta(days=30),
        limit: int = 1000,
        **_kwargs,
    ) -> dict[str, int]:
        observed = now or datetime.now(timezone.utc)
        cutoff = _utc_text(observed - finding_history_retention)
        with self._state_transaction("hot_state/cleanup"):
            terminalized = reconcile_terminal_attempt_observations(self, limit)
            cursor = self._db.execute(
                """
                DELETE FROM objects
                WHERE kind='gpu_finding_history'
                  AND key IN (
                      SELECT key
                      FROM objects
                      WHERE kind='gpu_finding_history'
                        AND json_extract(
                            payload, '$.observed_at'
                        ) <= ?
                      ORDER BY json_extract(
                          payload, '$.observed_at'
                      ), key
                      LIMIT ?
                  )
                """,
                (cutoff, limit),
            )
            deleted = cursor.rowcount
        return {
            "gpu_finding_history": deleted,
            "attempt_observation_terminalized": terminalized,
        }

    def save_raw_evidence(self, record, *, max_records_per_node: int) -> None:
        storage_key = self._state_key(
            (
                record.cluster_id,
                record.node_id,
                record.record_id,
            )
        )
        with self._state_transaction(f"raw_evidence/{storage_key}"):
            self._put("raw_evidence", storage_key, record)
            now = datetime.now(timezone.utc)
            records = self._list("raw_evidence")
            matching = sorted(
                (
                    item
                    for item in records
                    if item.expires_at > now
                    and item.cluster_id == record.cluster_id
                    and item.node_id == record.node_id
                ),
                key=lambda item: (
                    item.observed_at,
                    item.record_id,
                ),
                reverse=True,
            )
            for item in matching[max_records_per_node:]:
                self._delete(
                    "raw_evidence",
                    self._state_key(
                        (
                            item.cluster_id,
                            item.node_id,
                            item.record_id,
                        )
                    ),
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
        result = [
            item
            for item in self._list("raw_evidence")
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
        expired = sorted(
            (
                item
                for item in self._list("raw_evidence")
                if item.expires_at <= observed
            ),
            key=lambda item: (
                item.expires_at,
                item.record_id,
            ),
        )[:limit]
        if not expired:
            return 0
        with self._state_transaction("raw_evidence/cleanup"):
            for item in expired:
                self._delete(
                    "raw_evidence",
                    self._state_key(
                        (
                            item.cluster_id,
                            item.node_id,
                            item.record_id,
                        )
                    ),
                )
        return len(expired)

    def save_event_if_absent(self, event: TerminalEvent) -> bool:
        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        with self._state_transaction(f"terminal_event/{storage_key}"):
            cursor = self._db.execute(
                """
                INSERT OR IGNORE INTO objects(kind, key, payload)
                VALUES ('event', ?, ?)
                """,
                (event.event_key, event.model_dump_json()),
            )
            if cursor.rowcount:
                self._link(
                    "attempt_event",
                    storage_key,
                    event.event_key,
                )
            terminal = event if cursor.rowcount else self._get("event", event.event_key)
            terminalize_attempt_observation(self, terminal)
            return bool(cursor.rowcount)

    def completion_transaction(self, event_key: str) -> AbstractContextManager[None]:
        # One transaction under the process lock (F-G2 (3)): the writes the
        # completion service nests inside (event row, incident + workflow, plan,
        # decision) become savepoints of it and commit or roll back as one.
        return cast(
            AbstractContextManager[None],
            self._state_transaction(f"completion/{event_key}"),
        )

    def list_decisions_by_status(
        self,
        status: DecisionStatus,
        *,
        older_than: datetime | None = None,
        limit: int = 100,
    ) -> list[CompletionDecision]:
        if limit < 1:
            return []
        rows = self._db.execute(
            """
            SELECT decision.payload, diagnostic.payload
            FROM objects AS decision
            LEFT JOIN objects AS diagnostic
              ON diagnostic.kind='diagnostic'
             AND diagnostic.key=json_extract(
                 decision.payload, '$.diagnostic_request_id'
             )
            WHERE decision.kind='decision'
              AND json_extract(decision.payload, '$.status')=?
            """,
            (status.value,),
        ).fetchall()
        floor = datetime.min.replace(tzinfo=timezone.utc)
        candidates: list[tuple[datetime | None, CompletionDecision]] = []
        for decision_payload, diagnostic_payload in rows:
            created_at = (
                DiagnosticRequest.model_validate_json(diagnostic_payload).created_at
                if diagnostic_payload is not None
                else None
            )
            if older_than is not None and created_at is not None:
                if created_at > older_than:
                    continue
            candidates.append(
                (created_at, CompletionDecision.model_validate_json(decision_payload))
            )
        candidates.sort(key=lambda item: (item[0] or floor, item[1].event_key))
        return [decision for _created_at, decision in candidates[:limit]]

    def decision_status_counts(self) -> dict[DecisionStatus, int]:
        rows = self._db.execute(
            """
            SELECT json_extract(payload, '$.status'), count(*)
            FROM objects
            WHERE kind='decision'
            GROUP BY 1
            """
        ).fetchall()
        counts = {status: 0 for status in DecisionStatus}
        for status, count in rows:
            counts[DecisionStatus(status)] = int(count)
        return counts

    def count_completion_events_without_decision(self) -> int:
        row = self._db.execute(
            """
            SELECT count(*)
            FROM objects AS event
            WHERE event.kind='event'
              AND NOT EXISTS (
                  SELECT 1 FROM objects AS decision
                  WHERE decision.kind='decision'
                    AND decision.key=event.key
              )
            """
        ).fetchone()
        return int(row[0])

    def list_markers(self) -> list[NodeMarker]:
        return self._list("marker")

    def list_markers_for_incident(self, incident_id: str) -> list[NodeMarker]:
        rows = self._db.execute(
            """
            SELECT payload
            FROM objects
            WHERE kind='marker'
              AND json_extract(payload, '$.incident_id')=?
            ORDER BY json_extract(payload, '$.observed_at'), key
            """,
            (incident_id,),
        ).fetchall()
        return [NodeMarker.model_validate_json(row[0]) for row in rows]

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
        return sorted(
            (
                marker
                for marker in self._list("marker")
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
        return sorted(
            (
                marker
                for marker in self._list("marker")
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
    ) -> list[NodeMarker]:
        if not node_ids or not actions:
            return []
        return sorted(
            (
                marker
                for marker in self._list("marker")
                if marker.active
                and marker.trusted
                and marker.recommended_action in actions
                and set(marker.scope.node_ids).intersection(node_ids)
            ),
            key=lambda marker: marker.observed_at,
            reverse=True,
        )

    def save_installation_resource(
        self,
        resource: InstallationResource,
    ) -> InstallationResource:
        key = f"{resource.site_id}/{resource.resource_key}"
        with self._state_transaction(f"installation_resource/{key}"):
            existing = self._get_optional("installation_resource", key)
            if (
                existing is not None
                and existing.immutable_identity() != resource.immutable_identity()
            ):
                raise ValueError("installation resource identity cannot change")
            self._put("installation_resource", key, resource)
        return resource

    def get_installation_resource(
        self,
        site_id: str,
        resource_key: str,
    ) -> InstallationResource:
        return cast(
            InstallationResource,
            self._get("installation_resource", f"{site_id}/{resource_key}"),
        )

    def list_installation_resources(
        self,
        site_id: str | None = None,
    ) -> list[InstallationResource]:
        resources = cast(
            list[InstallationResource],
            self._list("installation_resource"),
        )
        return sorted(
            (item for item in resources if site_id is None or item.site_id == site_id),
            key=lambda item: (item.site_id, item.resource_key),
        )

    def has_workflow_successor(self, predecessor_workflow_id: str) -> bool:
        return any(
            workflow.predecessor_workflow_id == predecessor_workflow_id
            for workflow in self._list("workflow")
        )

    def get_preempting_successor(
        self, predecessor_workflow_id: str
    ) -> WorkflowRequest | None:
        matches = [
            workflow
            for workflow in self._list("workflow")
            if workflow.predecessor_workflow_id == predecessor_workflow_id
            and workflow.preempt_predecessor
            and workflow.status
            in {
                WorkflowStatus.PENDING,
                WorkflowStatus.SAFETY_PENDING,
            }
        ]
        matches.sort(
            key=lambda item: (
                item.created_at,
                item.request_id,
            )
        )
        return matches[0] if matches else None

    def list_hyperpod_node_identities(self, cluster_name: str | None = None):
        identities = self._list("hyperpod_node_identity")
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
