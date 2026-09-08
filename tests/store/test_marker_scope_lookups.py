"""What a marker question is allowed to read.

The marker table grows with every observation on every node, so the two hot
questions asked of it -- "does anything correlate with this finished attempt?"
and "does anything block this one candidate spare?" -- must be answered by a
scoped read rather than by scanning the table and filtering in Python. These
tests pin the scoping, the predicates that were previously spread across three
call sites, and the fact that the callers no longer reach for the full scan.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import gpu_fault
from gpu_fault.app import ApplicationContext
from gpu_fault.markers import blocking_spare_markers, marker_blocks_spare
from gpu_fault.models import (
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    TerminalEvent,
    WorkflowStatus,
)
from tests._builders import build_store, copy_model, fault_incident, workflow_request

NOW = datetime(2026, 7, 19, 22, 0, tzinfo=timezone.utc)


def marker(
    *,
    marker_id: str,
    scope: MarkerScope,
    observed_at: datetime = NOW,
    incident_id: str = "inc-existing",
    action: RecoveryAction | None = RecoveryAction.REBOOT_NODE,
    trusted: bool = True,
    active: bool = True,
    expires_at: datetime | None = None,
    cluster_id: str | None = None,
) -> NodeMarker:
    return NodeMarker(
        marker_id=marker_id,
        source="test-agent",
        cluster_id=cluster_id,
        trusted=trusted,
        active=active,
        incident_id=incident_id,
        observed_at=observed_at,
        expires_at=expires_at or observed_at + timedelta(minutes=30),
        scope=scope,
        severity=Severity.CRITICAL,
        recommended_action=action,
        action_owner="simulated-runtime",
        mapping_version="test-v1",
    )


def _refuse_full_scan(store) -> None:
    def explode() -> list[NodeMarker]:
        raise AssertionError(
            "a scoped marker question must not scan the whole marker table"
        )

    store.list_markers = explode


def test_scope_window_matches_gpu_and_fabric_scoped_markers() -> None:
    """A fabric fault names the partition, not the nodes attached to it."""

    store = build_store()
    for candidate in (
        marker(marker_id="by-node", scope=MarkerScope(node_ids=["node-a"])),
        marker(marker_id="by-gpu", scope=MarkerScope(gpu_uuids=["GPU-b"])),
        marker(
            marker_id="by-fabric", scope=MarkerScope(fabric_partitions=["fabric-a"])
        ),
        marker(marker_id="elsewhere", scope=MarkerScope(node_ids=["node-z"])),
    ):
        store.add_marker(candidate)

    found = store.list_markers_in_scope_window(
        node_ids={"node-a", "node-b"},
        gpu_uuids={"GPU-a", "GPU-b"},
        fabric_partitions={"fabric-a"},
        observed_from=NOW - timedelta(minutes=10),
        observed_to=NOW + timedelta(minutes=10),
    )

    assert {item.marker_id for item in found} == {"by-node", "by-gpu", "by-fabric"}


def test_scope_window_excludes_markers_that_ask_for_nothing() -> None:
    """Untrusted, retired and observation-only markers are not correlations."""

    store = build_store()
    scope = MarkerScope(node_ids=["node-a"])
    for candidate in (
        marker(marker_id="untrusted", scope=scope, trusted=False),
        marker(marker_id="retired", scope=scope, active=False),
        marker(marker_id="advisory", scope=scope, action=None),
        marker(marker_id="actionable", scope=scope),
    ):
        store.add_marker(candidate)

    found = store.list_markers_in_scope_window(
        node_ids={"node-a"},
        gpu_uuids=set(),
        fabric_partitions=set(),
        observed_from=NOW - timedelta(minutes=10),
        observed_to=NOW + timedelta(minutes=10),
    )

    assert [item.marker_id for item in found] == ["actionable"]


def test_scope_window_is_bounded_and_newest_first() -> None:
    store = build_store()
    scope = MarkerScope(node_ids=["node-a"])
    for index in range(5):
        store.add_marker(
            marker(
                marker_id=f"marker-{index}",
                scope=scope,
                observed_at=NOW - timedelta(minutes=index),
            )
        )
    store.add_marker(
        marker(marker_id="too-old", scope=scope, observed_at=NOW - timedelta(hours=2))
    )

    found = store.list_markers_in_scope_window(
        node_ids={"node-a"},
        gpu_uuids=set(),
        fabric_partitions=set(),
        observed_from=NOW - timedelta(minutes=10),
        observed_to=NOW + timedelta(minutes=10),
        limit=2,
    )

    assert [item.marker_id for item in found] == ["marker-0", "marker-1"]


def test_scope_window_with_no_scope_reads_nothing() -> None:
    """An event with no allocation cannot correlate, so it must not read."""

    store = build_store()
    _refuse_full_scan(store)

    assert (
        store.list_markers_in_scope_window(
            node_ids=set(),
            gpu_uuids=set(),
            fabric_partitions=set(),
            observed_from=NOW,
            observed_to=NOW,
        )
        == []
    )


def test_terminal_correlation_uses_the_scoped_read(
    failed_event: TerminalEvent, ended_at: datetime
) -> None:
    """The correlator must find a GPU-scoped marker without a full scan.

    The action is workload-scoped because the planner refuses a node-scoped
    action from a marker that names no node -- which is exactly the marker shape
    that a node-only store lookup would have dropped.
    """

    store = build_store()
    store.add_marker(
        marker(
            marker_id="gpu-scoped",
            scope=MarkerScope(gpu_uuids=["GPU-b"]),
            action=RecoveryAction.RESTART_WORKLOAD,
            observed_at=ended_at - timedelta(seconds=10),
        )
    )
    _refuse_full_scan(store)
    context = ApplicationContext(store=store)

    decision = context.completion.handle_terminal(failed_event)

    assert decision.matched_marker_ids == ["gpu-scoped"]


def test_expired_marker_does_not_correlate_with_a_later_event(
    failed_event: TerminalEvent, ended_at: datetime
) -> None:
    """Expiry is relative to the event, so it stays a caller-side test."""

    store = build_store()
    store.add_marker(
        marker(
            marker_id="expired",
            scope=MarkerScope(node_ids=["node-a"]),
            observed_at=ended_at - timedelta(minutes=5),
            expires_at=ended_at - timedelta(minutes=1),
        )
    )
    _refuse_full_scan(store)
    context = ApplicationContext(store=store)

    decision = context.completion.handle_terminal(failed_event)

    assert decision.matched_marker_ids == []


def _seed_incident_workflow(store, *, status: WorkflowStatus | None) -> str:
    incident = fault_incident(
        incident_id="inc-existing", event_id="event-1", node_ids=["node-a"]
    )
    if status is None:
        store.save_incident(copy_model(incident, workflow_request_id=None))
        return "inc-existing"
    store.save_incident(copy_model(incident, workflow_request_id="wf-1"))
    store.save_workflow(
        workflow_request(
            request_id="wf-1", incident_id=incident.incident_id, status=status
        )
    )
    return "inc-existing"


@pytest.mark.parametrize(
    ("status", "blocks"),
    [(WorkflowStatus.SUCCEEDED, False), (WorkflowStatus.BLOCKED, True), (None, True)],
)
def test_recovery_history_decides_whether_a_marker_still_blocks(
    status: WorkflowStatus | None, blocks: bool
) -> None:
    """A recovered incident releases its node; an unrecovered one does not.

    ``None`` is the incident that has not been attached to a workflow yet. It
    used to be handled by catching ``TypeError`` from ``get_workflow(None)``,
    which would have hidden a real argument-type defect in the same branch.
    """

    store = build_store()
    _seed_incident_workflow(store, status=status)
    blocking = marker(marker_id="blocking", scope=MarkerScope(node_ids=["node-a"]))
    store.add_marker(blocking)

    assert marker_blocks_spare(store, blocking, now=NOW) is blocks, (
        f"a marker whose incident workflow is {status} must "
        f"{'block' if blocks else 'release'} the node"
    )


def test_blocking_spare_markers_uses_the_scoped_read() -> None:
    store = build_store()
    _seed_incident_workflow(store, status=WorkflowStatus.BLOCKED)
    store.add_marker(
        marker(marker_id="blocking", scope=MarkerScope(node_ids=["node-a"]))
    )
    store.add_marker(
        marker(
            marker_id="advisory",
            scope=MarkerScope(node_ids=["node-a"]),
            action=RecoveryAction.RUN_DIAGNOSTICS,
        )
    )
    store.add_marker(
        marker(marker_id="other-node", scope=MarkerScope(node_ids=["node-z"]))
    )
    _refuse_full_scan(store)

    found = blocking_spare_markers(store, {"node-a"}, now=NOW)

    assert [item.marker_id for item in found] == ["blocking"]


def test_blocking_spare_markers_honours_observed_after() -> None:
    """A re-check after a repair only cares about what happened since."""

    store = build_store()
    _seed_incident_workflow(store, status=WorkflowStatus.BLOCKED)
    store.add_marker(
        marker(
            marker_id="before-repair",
            scope=MarkerScope(node_ids=["node-a"]),
            observed_at=NOW - timedelta(hours=1),
        )
    )
    store.add_marker(
        marker(
            marker_id="after-repair",
            scope=MarkerScope(node_ids=["node-a"]),
            observed_at=NOW - timedelta(minutes=1),
        )
    )
    _refuse_full_scan(store)

    found = blocking_spare_markers(
        store, {"node-a"}, observed_after=NOW - timedelta(minutes=30), now=NOW
    )

    assert [item.marker_id for item in found] == ["after-repair"]


def test_blocking_spare_markers_with_no_aliases_reads_nothing() -> None:
    store = build_store()
    _refuse_full_scan(store)

    assert blocking_spare_markers(store, set(), now=NOW) == []


def test_markers_do_not_cross_tenant_boundaries() -> None:
    """H-14: a node_id colliding across clusters must not leak markers.

    Two tenants both run a ``node-a``. A blocking marker raised in ``alpha``
    must be invisible to a scoped read for ``beta`` and visible to ``alpha``.
    """

    store = build_store()
    _seed_incident_workflow(store, status=WorkflowStatus.BLOCKED)
    store.add_marker(
        marker(
            marker_id="alpha-blocking",
            scope=MarkerScope(node_ids=["node-a"]),
            cluster_id="alpha",
        )
    )
    _refuse_full_scan(store)

    other_tenant = blocking_spare_markers(store, {"node-a"}, cluster_id="beta", now=NOW)
    own_tenant = blocking_spare_markers(store, {"node-a"}, cluster_id="alpha", now=NOW)

    assert other_tenant == []
    assert [item.marker_id for item in own_tenant] == ["alpha-blocking"]


def test_scoped_read_never_matches_a_legacy_marker_without_a_cluster() -> None:
    """A marker with no stamped cluster is never claimed by a specific tenant.

    Legacy rows predate the tenant stamp; matching one to any cluster that
    asked would re-open the cross-tenant read, so a scoped query drops it.
    """

    store = build_store()
    _seed_incident_workflow(store, status=WorkflowStatus.BLOCKED)
    store.add_marker(
        marker(
            marker_id="legacy", scope=MarkerScope(node_ids=["node-a"]), cluster_id=None
        )
    )

    scoped = blocking_spare_markers(store, {"node-a"}, cluster_id="alpha", now=NOW)
    unscoped = blocking_spare_markers(store, {"node-a"}, now=NOW)

    assert scoped == []
    assert [item.marker_id for item in unscoped] == ["legacy"]


def test_no_production_module_reads_the_whole_marker_table() -> None:
    """``list_markers()`` is a fixture affordance, not a control-plane read.

    The tests above prove the two scoped questions do not fall back to the full
    scan, but they can only speak for the call sites they exercise. The method is
    still on ``CompletionStore``, because the acceptance fixtures assert over the
    whole table against a fixture-sized store, and nothing stops a future request
    path from reaching for it. This is the boundary those docstrings claim.
    """

    source_root = Path(gpu_fault.__file__).resolve().parent
    full_scan = re.compile(r"\.list_markers\(\s*\)")
    offenders = sorted(
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
        if full_scan.search(path.read_text(encoding="utf-8"))
    )

    assert offenders == []
