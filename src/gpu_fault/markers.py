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

The one writer here, :func:`retire_markers_for_incident`, is the counterpart of
the reads: a marker's ``active`` flag had no writer on any success path, so a
repaired incident's markers stayed live until their TTL and kept matching
terminal events and disqualifying spares (F-G6).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from gpu_fault.models import NodeMarker, RecoveryAction, WorkflowStatus
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
)
from gpu_fault.recovery_actions import RECOVERY_ACTION_PROFILES

LOGGER = logging.getLogger(__name__)

#: Recommended actions that mean "this node must not take on work".
#: A marker whose action is outside this set is observational (for
#: example ``RUN_DIAGNOSTICS`` on a TCP retransmission blip) and must
#: not disqualify an otherwise healthy warm spare -- warm-spare
#: failover is the only supported node replacement path, so treating
#: advisory noise as disqualifying makes real replacements fail.
#: Derived from the same table the passive compiler uses, so an action
#: cannot be plannable here and invisible there (F-G5).
SPARE_BLOCKING_ACTIONS: frozenset[RecoveryAction] = frozenset(
    action
    for action, profile in RECOVERY_ACTION_PROFILES.items()
    if profile.blocks_spare
)

#: Recommended actions that only observe a node: evidence capture,
#: diagnostics, validation. Derived from the operation registry -- the
#: action compiles to an operation that is neither destructive nor
#: node-wide -- so a new action cannot be advisory here and mutating there.
#: A marker recommending one of these says "look at this node", never "this
#: node is being repaired": it must not make its incident the owner of a
#: failed attempt's recovery (the terminal is decided as if the marker
#: were absent), and it must not hold a restart behind the incident's
#: ``RECOVERED`` gate.
DIAGNOSTIC_ACTIONS: frozenset[RecoveryAction] = frozenset(
    action
    for action, profile in RECOVERY_ACTION_PROFILES.items()
    if profile.operation is not None
    and profile.operation not in DESTRUCTIVE_OPERATIONS
    and profile.operation not in NODE_WIDE_RECOVERY_OPERATIONS
)


def marker_is_diagnostic(marker: NodeMarker) -> bool:
    """Whether ``marker`` observes the node rather than asking to change it.

    The completion service used to let any matching marker make its incident
    the owner of a failed attempt's recovery, planning a restart gated on that
    incident reaching ``RECOVERED``. For a WARNING ``RUN_DIAGNOSTICS`` marker
    (CPU at 98 % for two minutes) that gate guarded nothing: the diagnostic
    workflow never repaired anything, and the marker outlived the incident
    until its TTL, so the job could not restart for an hour. A diagnostic
    marker that names a stored incident is therefore skipped there and the
    terminal is decided without it (a budgeted restart when nothing else
    matches); the marker itself stays live, it is still a valid observation.
    A marker without a stored incident is not affected:
    it still plans the diagnostic it asks for through ``from_marker``.

    A marker with no recommended action is treated as the strongest action
    (``QUARANTINE``) everywhere else, so it is not diagnostic here either.
    """
    return marker.recommended_action in DIAGNOSTIC_ACTIONS


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
        cluster_id: str | None = None,
    ) -> list[NodeMarker]: ...

    def get_incident(self, incident_id: str) -> Any: ...

    def get_workflow(self, request_id: str) -> Any: ...


@runtime_checkable
class MarkerRetirementStore(Protocol):
    """The two store methods retiring an incident's markers needs."""

    def list_markers_for_incident(self, incident_id: str) -> list[NodeMarker]: ...

    def add_marker(self, marker: NodeMarker) -> None: ...


#: Upper bounds on the retirement text written onto a marker. ``model_copy``
#: does not validate, so the bound is applied here rather than on the model:
#: a field constraint would let an over-long reason be written and then refuse
#: to load the row.
MARKER_RETIRED_TEXT_LIMIT = 512


def _bounded_text(value: str | None) -> str | None:
    if value is None or len(value) <= MARKER_RETIRED_TEXT_LIMIT:
        return value
    return value[: MARKER_RETIRED_TEXT_LIMIT - 1] + "…"


def retire_markers_for_incident(
    store: MarkerRetirementStore,
    incident_id: str,
    *,
    reason: str,
    retired_by: str | None = None,
    now: datetime | None = None,
) -> int:
    """Set ``active=False`` on every live marker of a recovered incident.

    ``add_marker`` upserts by ``marker_id`` in every store, so this rewrites
    the marker in place. Returns how many markers were retired. Safe to call
    repeatedly: an already-retired marker is skipped, so the first retirement's
    ``retired_at`` / ``retired_reason`` / ``retired_by`` are what the marker
    keeps (I4). ``retired_by`` names the path that retired it (the completion
    service, the spare-health controller); ``now`` is the retirement time.
    """
    if not incident_id:
        return 0
    retired_at = now if now is not None else datetime.now(timezone.utc)
    retired = 0
    for marker in store.list_markers_for_incident(incident_id):
        if not marker.active:
            continue
        store.add_marker(
            marker.model_copy(
                update={
                    "active": False,
                    "retired_at": retired_at,
                    "retired_reason": _bounded_text(reason),
                    "retired_by": _bounded_text(retired_by),
                }
            )
        )
        retired += 1
    if retired:
        LOGGER.info(
            "retired %s marker(s) of incident %s: %s", retired, incident_id, reason
        )
    return retired


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
    cluster_id: str | None = None,
    observed_after: datetime | None = None,
    now: datetime | None = None,
) -> list[NodeMarker]:
    """Live blocking markers on ``node_ids``, newest first.

    ``observed_after`` is how a caller asks "did anything go wrong *since* this
    moment", which is what a spare-health re-check after a repair wants; markers
    from before the repair are the reason the check is running.

    ``cluster_id`` scopes the read to one tenant (H-14): ``node_ids`` collide
    across clusters, so without it a spare in one cluster could be blocked by a
    same-named node's marker in another. It is keyword-only and optional so
    existing callers keep compiling, but a caller that omits it gets an
    unscoped read -- every caller that resolves a real cluster must pass it.
    """

    if not node_ids:
        return []
    moment = now if now is not None else datetime.now(timezone.utc)
    candidates = store.list_active_markers_for_nodes(
        node_ids,
        set(SPARE_BLOCKING_ACTIONS),
        cluster_id,
    )
    return [
        marker
        for marker in candidates
        if (observed_after is None or marker.observed_at > observed_after)
        and marker_blocks_spare(store, marker, now=moment)
    ]
