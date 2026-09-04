"""Which node fault markers block reuse of a node, and how to read them cheaply.

Two independent paths decide whether a node may take on work: the local
warm-spare selector in :mod:`gpu_fault.hyperpod_spares` and the remote
spare-health endpoint in :mod:`gpu_fault.app.routes.regional`. They answered the
same question -- "is there a live, trusted marker asking this node to stop taking
work, and has the incident behind it already been recovered?" -- with two
byte-identical private helpers. A divergence between them would mean a spare the
local selector refuses is handed out by the remote check, or the reverse, and
neither side would report a disagreement. The decision therefore lives here once.

Both paths also read the whole marker table and filtered in Python, even though
they only ever care about the aliases of one candidate node. The marker table
grows with every observation on every node in the fleet, so that cost is
proportional to fleet history rather than to the question being asked. The
lookups here push node scope and the blocking-action set into the store, which
answers them with an indexed, bounded read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from gpu_fault.models import NodeMarker, RecoveryAction, WorkflowStatus

#: Recommended actions that mean "this node must not take on work".
#: A marker whose action is outside this set is observational (for
#: example ``RUN_DIAGNOSTICS`` on a TCP retransmission blip) and must
#: not disqualify an otherwise healthy warm spare -- warm-spare
#: failover is the only supported node replacement path, so treating
#: advisory noise as disqualifying makes real replacements fail.
SPARE_BLOCKING_ACTIONS = frozenset(
    {
        RecoveryAction.QUARANTINE,
        RecoveryAction.REPLACE_NODE,
        RecoveryAction.REBOOT_NODE,
        RecoveryAction.RESET_GPU,
        RecoveryAction.DRAIN,
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.ESCALATE_OPERATOR,
    }
)


@runtime_checkable
class MarkerLookupStore(Protocol):
    """The narrow slice of the Store these lookups need.

    Spelled out rather than taking the full Store so a caller can see that
    nothing here writes, and so a test fake stays three methods wide.
    """

    def list_active_markers_for_nodes(
        self,
        node_ids: set[str],
        actions: set[RecoveryAction],
    ) -> list[NodeMarker]: ...

    def get_incident(self, incident_id: str) -> Any: ...

    def get_workflow(self, request_id: str) -> Any: ...


def marker_disqualifies_spare(marker: NodeMarker, now: datetime) -> bool:
    """Whether ``marker`` means a node is unfit to be a warm spare.

    Only markers that actually ask for the node to stop taking work
    disqualify it. Advisory markers (``MONITOR_ONLY`` dispositions such
    as ``RUN_DIAGNOSTICS``) are recorded continuously on healthy nodes
    and never produce a workflow, so the ``SUCCEEDED``-workflow escape
    below can never clear them -- counting them would leave every node
    permanently ineligible.
    """
    if not marker.active or not marker.trusted:
        return False
    if marker.expires_at <= now:
        return False
    if marker.recommended_action not in SPARE_BLOCKING_ACTIONS:
        return False
    return True


def describe_blocking_marker(marker: NodeMarker) -> str:
    """Operator-readable identity of a disqualifying marker.

    The bare "a marker exists" reason left no way to decide whether a
    spare could be released by hand; the caller has the only copy of
    this record, so it has to be named in the failure reason.
    """
    action = (
        marker.recommended_action.value
        if marker.recommended_action is not None
        else "unknown"
    )
    detail = f"{marker.marker_id} ({marker.severity.value}/{action}"
    if marker.raw_reason:
        detail += f": {marker.raw_reason}"
    return detail + ")"


def marker_blocks_spare(
    store: MarkerLookupStore,
    marker: NodeMarker,
    *,
    now: datetime | None = None,
) -> bool:
    """Whether ``marker`` still blocks node reuse, given recovery history.

    A live blocking marker is cleared by the successful recovery of the incident
    that raised it: the marker window outlives the workflow, so keeping the node
    out after a ``SUCCEEDED`` workflow would strand a healthy spare.

    An incident that has not been attached to a workflow yet cannot have been
    recovered, so it blocks. That case is tested explicitly rather than being
    absorbed by an exception handler: ``workflow_request_id`` is legitimately
    ``None`` for a fresh incident, and letting a ``TypeError`` stand in for it
    would have swallowed a genuine argument-type defect in the same branch.
    """

    moment = now if now is not None else datetime.now(timezone.utc)
    if not marker_disqualifies_spare(marker, moment):
        return False
    if not marker.incident_id:
        return True
    try:
        incident = store.get_incident(marker.incident_id)
    except KeyError:
        # The marker outlived its incident record. Nothing proves recovery
        # happened, so the node stays out: fail closed.
        return True
    workflow_request_id = getattr(incident, "workflow_request_id", None)
    if workflow_request_id is None:
        return True
    try:
        workflow = store.get_workflow(workflow_request_id)
    except KeyError:
        return True
    return workflow.status is not WorkflowStatus.SUCCEEDED


def blocking_spare_markers(
    store: MarkerLookupStore,
    node_ids: set[str],
    *,
    observed_after: datetime | None = None,
    now: datetime | None = None,
) -> list[NodeMarker]:
    """Live blocking markers on ``node_ids``, newest first.

    ``observed_after`` is how a caller asks "did anything go wrong *since* this
    moment", which is what a spare-health re-check after a repair wants; markers
    from before the repair are the reason the check is running.
    """

    if not node_ids:
        return []
    moment = now if now is not None else datetime.now(timezone.utc)
    candidates = store.list_active_markers_for_nodes(
        node_ids,
        set(SPARE_BLOCKING_ACTIONS),
    )
    return [
        marker
        for marker in candidates
        if (observed_after is None or marker.observed_at > observed_after)
        and marker_blocks_spare(store, marker, now=moment)
    ]
