from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Protocol

from gpu_fault.models import FaultIncident, RecoveryPlan
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.store.shared.record_guards import stale_incident_versions


class _Database(Protocol):
    """The two ``PooledPostgresDatabase`` entry points this mixin uses."""

    def cursor(self) -> AbstractContextManager[Any]: ...

    def transaction(self) -> AbstractContextManager[None]: ...


class PostgresRecordCasMixin:
    """Compare-and-set writers for the rows without a version field.

    ``PostgresStore`` inherits ``save_incident``, ``save_plan`` and
    ``link_event_to_incident`` from the SQLite layer, where the process lock
    serializes them; here that lock is a ``nullcontext`` and each is one or
    two autocommit statements (architecture review 2026-09-07, items D1 and
    D2). The overrides below give the incident the version guard
    ``save_workflow`` has, give the plan the ``expected=`` compare-and-set of
    ``_put``, and run the two-statement event link inside one transaction.
    """

    # Attributes supplied by the composed concrete implementation.
    _db: _Database
    _decode: Callable[..., Any]
    _get_for_update: Callable[..., Any]
    _link: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def save_incident(
        self,
        incident: FaultIncident,
        *,
        expected: FaultIncident | None = None,
    ) -> None:
        """See ``WorkflowStore.save_incident``.

        Both statements -- the incident row and its event link -- share one
        transaction, so a reader never sees the link before the incident. With
        ``expected`` the row write is the full-payload compare-and-set of
        ``_put``; without it, the guarded UPDATE / INSERT pair of
        ``save_workflow``, matching only while the stored generation is not ahead
        of the caller's (``stale_incident_versions``), with the same race
        analysis.
        """

        with self._db.transaction():
            if expected is not None:
                self._put("incident", incident.incident_id, incident, expected=expected)
            else:
                self._save_incident_guarded(incident)
            self._link(
                "incident_by_event",
                incident.event_id,
                incident.incident_id,
            )

    def _save_incident_guarded(self, incident: FaultIncident) -> None:
        payload = incident.model_dump_json()
        with self._db.cursor() as cursor:
            # COALESCE to the model default: rows written before the field
            # existed decode with the default, and must compare as such.
            cursor.execute(
                """
                UPDATE gpu_fault_objects
                SET payload=%s::jsonb
                WHERE kind='incident' AND key=%s
                  AND coalesce(payload->>'fencing_token', '1')::int <= %s
                """,
                (payload, incident.incident_id, incident.fencing_token),
            )
            if cursor.rowcount == 1:
                return
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES ('incident', %s, %s::jsonb)
                ON CONFLICT(kind, key) DO NOTHING
                RETURNING key
                """,
                (incident.incident_id, payload),
            )
            if cursor.fetchone() is not None:
                return
            # Neither statement landed: re-read only to name what moved.
            cursor.execute(
                """
                SELECT payload FROM gpu_fault_objects
                WHERE kind='incident' AND key=%s
                """,
                (incident.incident_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise StaleWriteError(
                f"incident/{incident.incident_id} changed since it was read"
            )
        stored: FaultIncident = self._decode("incident", row[0])
        stale = stale_incident_versions(stored, incident)
        if stale is None:
            stale = StaleWriteError(
                f"incident/{incident.incident_id} changed since it was read"
            )
        raise stale

    def save_plan(
        self,
        plan: RecoveryPlan,
        *,
        expected: RecoveryPlan | None = None,
    ) -> None:
        """See ``CompletionStore.save_plan``: one upsert, or one CAS."""

        self._put("plan", plan.plan_id, plan, expected=expected)

    def link_event_to_incident(self, event_id: str, incident_id: str) -> None:
        """Existence check and link in one transaction (item D2).

        The inherited form read the incident and wrote the link as two
        autocommit statements; the row lock here keeps the incident from
        being replaced between them. Raises ``NotFoundError`` like the parent.
        """

        with self._state_transaction(f"incident/{incident_id}"):
            self._get_for_update("incident", incident_id)
            self._link("incident_by_event", event_id, incident_id)
