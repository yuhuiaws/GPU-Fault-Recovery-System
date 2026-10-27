"""The labels of an incident that holds a node-health finding (b213).

``FaultIncident.event_type`` is what a record is, not what every finding in it
is. The health family mints a finding's incident ``NODE_HEALTH``
(``families/health.py``). When the product routes more evidence into a record
instead of opening a second one, it re-labels that record: a critical
quarantine joining a node's in-flight isolation makes it ``NODE_HEALTH_GROUP``
(``families/drain.py`` ``ingest_terminal_node_quarantine``), a reboot or
replacement finding of a running attempt opens one under that label itself
(``families/node_lifecycle.py``), a device-resource finding joining the node's
chain makes it ``NODE_RESOURCE_GROUP``
(``ingest_grouped_node_resource_finding``), and a finding of a managed attempt
is opened or folded as ``GPU_FAULT_GROUP`` (``families/grouped_health.py``) --
the label the attempt's XID/SXID records carry too (``families/faults.py``,
``families/grouped_faults.py``, ``orchestration/ingest/sxid.py``).

A reader asking "which incident holds node N's finding F" therefore takes every
label of :data:`NODE_HEALTH_INCIDENT_EVENT_TYPES`, and does not read a grouped
record's ``event_id``, ``policy_source`` or ``node_ids`` as F's: a record F was
*folded into* keeps the identity of the finding that minted it (an XID's
recovery, say), a later fold may overwrite even the policy source of the record
F minted, and a record-only fold does not add F's node at all. F's own identity
stays where ingestion wrote it: the event link (``get_incident_by_event``) names
the record, and F's marker -- ``marker-<event id>``, written before routing and
re-pointed at the record it was routed to (``app/ingest/node_health.py``) --
keeps F's policy source (``event_source``) and node (``scope.node_ids``).
"""

from __future__ import annotations

from typing import Any

from gpu_fault.models import FaultIncident, NodeMarker

NODE_HEALTH_EVENT_TYPE = "NODE_HEALTH"
NODE_HEALTH_GROUP_EVENT_TYPE = "NODE_HEALTH_GROUP"
NODE_RESOURCE_GROUP_EVENT_TYPE = "NODE_RESOURCE_GROUP"
GPU_FAULT_GROUP_EVENT_TYPE = "GPU_FAULT_GROUP"
#: The labels the grouping routes give a record they fold other findings into.
#: A finding ingestion only links to a record without re-labelling it (a
#: same-signal host-resource re-arm, a transient inventory finding a running
#: chain covers, a finding of a node the product is rebooting --
#: ``orchestration/reboot_window.py``) is that record's evidence; readers that
#: follow folded findings look at grouped records only.
GROUPED_INCIDENT_EVENT_TYPES = frozenset(
    {
        NODE_HEALTH_GROUP_EVENT_TYPE,
        NODE_RESOURCE_GROUP_EVENT_TYPE,
        GPU_FAULT_GROUP_EVENT_TYPE,
    }
)
#: Every label an incident holding a node-health finding carries.
NODE_HEALTH_INCIDENT_EVENT_TYPES = frozenset(
    {NODE_HEALTH_EVENT_TYPE, *GROUPED_INCIDENT_EVENT_TYPES}
)


def finding_marker_id(event_id: str) -> str:
    """The id ``NodeHealthFinding.marker`` gives finding ``event_id``'s marker."""

    return f"marker-{event_id}"


def routed_finding_markers(store: Any, incident: FaultIncident) -> list[NodeMarker]:
    """The markers of the findings ingestion routed to ``incident``.

    One indexed read (``list_markers_for_incident``). Only markers stamped
    with the incident's cluster count: a marker without one is never matched
    to a specific cluster (H-14).
    """

    markers: list[NodeMarker] = store.list_markers_for_incident(incident.incident_id)
    return [marker for marker in markers if marker.cluster_id == incident.cluster_id]
