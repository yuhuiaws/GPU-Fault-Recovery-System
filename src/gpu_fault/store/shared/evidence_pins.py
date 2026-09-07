"""Which raw evidence records an open incident still needs.

Architecture review 2026-09-07, item D6. ``RawEvidenceRecord`` rows expire
after ``EvidenceService.retention`` (24 h) while incidents live forever
(control-record retention is off), so the evidence an operator opens an
ESCALATED or QUARANTINED incident to look at was gone long before they looked.

An incident never stores a pointer to a record: ``NodeMarker.raw_evidence_ref``
and ``NodeHealthFinding.evidence_ref`` carry the collector's own URIs
(``prometheus://``, ``nvidia-smi://``, ``journal://``), and the ingest names
records ``gpu-metrics/<batch_id>`` and the like -- the two never met. What binds
a record to an incident is the capture itself: the record names the attempt
that was running (``attempt_ids``) and the node it came from, and the incident
names its attempt and its nodes. The pin is therefore

* the attempt: an open incident of the same cluster names an attempt the record
  carries -- bounded by the attempt's lifetime, since a restarted attempt gets a
  new id; or
* the node within a window: an open incident of the same cluster names the
  record's node and the record was observed between ``created_at - window``
  and ``updated_at + window`` -- a fixed slice around the incident's lifetime,
  so a node with a long-open incident does not pin its whole telemetry stream.

A RECOVERED incident pins nothing; every other state is one an operator may
still open. The per-node cap in ``save_raw_evidence`` is left as the hard
storage bound and does not consult pins.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

from gpu_fault.models import FaultIncident, IncidentState
from gpu_fault.telemetry import RawEvidenceRecord

EVIDENCE_PINNING_INCIDENT_STATES: frozenset[IncidentState] = frozenset(
    IncidentState
) - {IncidentState.RECOVERED}

EVIDENCE_PIN_WINDOW = timedelta(hours=1)


def pinning_incident_state_values() -> list[str]:
    """The non-terminal state literals, for the SQL backends' ``IN`` lists."""

    return sorted(state.value for state in EVIDENCE_PINNING_INCIDENT_STATES)


def evidence_pinned_by(
    record: RawEvidenceRecord,
    incident: FaultIncident,
    *,
    window: timedelta = EVIDENCE_PIN_WINDOW,
) -> bool:
    if (
        incident.state not in EVIDENCE_PINNING_INCIDENT_STATES
        or incident.cluster_id != record.cluster_id
    ):
        return False
    if incident.attempt_id is not None and incident.attempt_id in record.attempt_ids:
        return True
    return (
        record.node_id in incident.node_ids
        and incident.created_at - window
        <= record.observed_at
        <= incident.updated_at + window
    )


def evidence_pinned(
    record: RawEvidenceRecord,
    incidents: Iterable[FaultIncident],
    *,
    window: timedelta = EVIDENCE_PIN_WINDOW,
) -> bool:
    """Whether any of ``incidents`` still needs ``record`` (memory / SQLite)."""

    return any(
        evidence_pinned_by(record, incident, window=window) for incident in incidents
    )
