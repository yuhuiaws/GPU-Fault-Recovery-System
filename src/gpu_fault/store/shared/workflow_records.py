"""Incident and workflow rows read whole, and the event -> incident link,
shared by the key/value stores. The lease-guarded writers stay dialect-side
(they hold a row lock); these are the plain lookups around them."""

from __future__ import annotations

from gpu_fault.models import FaultIncident, WorkflowRequest
from gpu_fault.store.shared.primitives import (
    GetLink,
    GetOptionalRecord,
    GetRecord,
    LinkRecord,
    StatementGuard,
)


class SharedWorkflowRecordMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: GetRecord
    _get_link: GetLink
    _get_optional: GetOptionalRecord
    _link: LinkRecord
    _statement_guard: StatementGuard

    def get_incident(self, incident_id: str) -> FaultIncident:
        return self._get("incident", incident_id)

    def get_incident_by_event(self, event_id: str) -> FaultIncident | None:
        incident_id = self._get_link("incident_by_event", event_id)
        return self._get_optional("incident", incident_id) if incident_id else None

    def link_event_to_incident(self, event_id: str, incident_id: str) -> None:
        self.get_incident(incident_id)
        with self._statement_guard():
            self._link("incident_by_event", event_id, incident_id)

    def get_workflow(self, request_id: str) -> WorkflowRequest:
        return self._get("workflow", request_id)
