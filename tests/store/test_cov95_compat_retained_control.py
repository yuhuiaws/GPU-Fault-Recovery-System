from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpu_fault.models import MarkerScope, NodeMarker, RecoveryAction, Severity
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from tests._builders import fault_incident
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)


def marker(name, *, at=NOW, active=True, trusted=True, nodes=None, **values):
    return NodeMarker(
        marker_id=name,
        source="local-compatibility",
        cluster_id="cluster-local",
        incident_id=f"incident/{name}",
        observed_at=at,
        expires_at=at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-local"] if nodes is None else nodes),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RESET_GPU,
        mapping_version="compat-v1",
        active=active,
        trusted=trusted,
        **values,
    )


def test_inactive_marker_retention_preserves_incident_pins_and_live_markers(
    compat_store,
):
    cutoff = NOW - timedelta(days=1)
    records = [
        marker("pinned", at=cutoff - timedelta(days=1), active=False),
        marker("live", at=cutoff - timedelta(days=1)),
        marker("a-retired", at=cutoff, active=False),
        marker("b-retired", at=cutoff, active=False),
        marker("recent", at=cutoff + timedelta(microseconds=1), active=False),
    ]
    compat_store.save_incident(
        fault_incident("incident/pinned", "event-pinned", cluster_id="cluster-local")
    )
    for item in records:
        compat_store.add_marker(item)

    assert compat_store.cleanup_inactive_markers(older_than=cutoff, limit=1) == 1, (
        "retention must skip old pinned and active markers before spending its limit"
    )
    assert compat_store.list_markers_for_incident("incident/a-retired") == [], (
        "equal-age retired markers must be removed in marker-ID order"
    )
    assert compat_store.list_markers_for_incident("incident/b-retired") == [
        records[3]
    ], "a one-row cleanup must leave the next eligible marker intact"
    assert compat_store.cleanup_inactive_markers(older_than=cutoff, limit=10) == 1, (
        "only the remaining unpinned retired marker is eligible"
    )
    assert {item.marker_id for item in compat_store.list_markers()} == {
        "pinned",
        "live",
        "recent",
    }, "cleanup must retain active, pinned and newer marker records"
    for item in (records[0], records[1], records[4]):
        assert compat_store.list_markers_for_incident(item.incident_id) == [item], (
            f"cleanup rewrote retained marker {item.marker_id}"
        )
    assert compat_store.cleanup_inactive_markers(older_than=cutoff, limit=10) == 0, (
        "repeating retention must not consume preserved incident evidence"
    )


def test_recent_node_markers_use_trust_activity_and_boot_identity_before_limit(
    compat_store,
):
    records = [
        marker("a-recent"),
        marker("b-recent"),
        marker("same-boot", at=NOW - timedelta(hours=1), source_boot_id="boot-current"),
        marker("old-boot", at=NOW - timedelta(hours=1), source_boot_id="boot-previous"),
        marker("untrusted", trusted=False, at=NOW + timedelta(seconds=1)),
        marker("inactive", active=False, at=NOW + timedelta(seconds=1)),
        marker("other-node", nodes=["node-other"], at=NOW + timedelta(seconds=1)),
    ]
    for item in records:
        compat_store.add_marker(item)
    cutoff = NOW - timedelta(minutes=5)

    assert compat_store.list_recent_markers_for_nodes(
        {"node-local"}, cutoff, limit=2
    ) == [records[1], records[0]], (
        "ineligible markers must not hide recent trusted evidence behind the read limit"
    )
    assert compat_store.list_recent_markers_for_nodes(
        {"node-local"}, cutoff, source_boot_id="boot-current"
    ) == [records[1], records[0], records[2]], (
        "same-boot evidence must survive a node-clock window without admitting other boots"
    )
    assert compat_store.list_recent_markers_for_nodes(set(), cutoff) == [], (
        "an empty node scope must read no marker rows"
    )
    assert (
        compat_store.list_recent_markers_for_nodes({"node-local"}, cutoff, limit=0)
        == []
    ), "a zero scan budget must not return even the first marker"


def test_active_node_marker_reads_require_nonempty_nodes_and_actions(compat_store):
    active = marker("active")
    advisory = marker("advisory").model_copy(
        update={"recommended_action": RecoveryAction.RUN_DIAGNOSTICS}
    )
    for item in (active, advisory, marker("untrusted", trusted=False)):
        compat_store.add_marker(item)

    assert compat_store.list_active_markers_for_nodes(
        {"node-local"}, {RecoveryAction.RESET_GPU}, cluster_id="cluster-local"
    ) == [active], (
        "only trusted active markers for the requested action may be returned"
    )
    assert (
        compat_store.list_active_markers_for_nodes(
            set(), {RecoveryAction.RESET_GPU}, cluster_id="cluster-local"
        )
        == []
    ), "an empty node set must not expand the active-marker query"
    assert (
        compat_store.list_active_markers_for_nodes(
            {"node-local"}, set(), cluster_id="cluster-local"
        )
        == []
    ), "an empty action set must not expand the active-marker query"


def test_raw_evidence_capacity_is_per_node_and_keeps_the_newest_live_records(
    compat_store,
):
    at = datetime.now(UTC)

    def evidence(name, *, cluster="cluster-local", node="node-local", seconds=0):
        return RawEvidenceRecord(
            record_id=name,
            cluster_id=cluster,
            node_id=node,
            kind=EvidenceKind.GPU_METRICS,
            observed_at=at + timedelta(seconds=seconds),
            ingested_at=at,
            expires_at=at + timedelta(hours=1),
            attempt_ids=["attempt-local"],
            payload={"record": name},
        )

    unrelated = [
        evidence("foreign-cluster", cluster="cluster-foreign"),
        evidence("other-node", node="node-other"),
    ]
    for item in unrelated:
        compat_store.save_raw_evidence(item, max_records_per_node=2)
    records = [
        evidence("oldest"),
        evidence("a-latest", seconds=1),
        evidence("b-latest", seconds=1),
    ]
    for item in records:
        compat_store.save_raw_evidence(item, max_records_per_node=2)

    assert compat_store.list_raw_evidence(
        "cluster-local", node_id="node-local", attempt_id="attempt-local"
    ) == [records[2], records[1]], (
        "node capacity must retain the two newest records in stable order"
    )
    for item in unrelated:
        assert compat_store.list_raw_evidence(
            item.cluster_id, node_id=item.node_id
        ) == [item], (
            "one node's capacity enforcement must not trim another node or cluster"
        )
    compat_store.save_raw_evidence(records[-1], max_records_per_node=1)
    assert compat_store.list_raw_evidence("cluster-local", node_id="node-local") == [
        records[-1]
    ], "reducing local capacity must trim only older live records for that same node"
