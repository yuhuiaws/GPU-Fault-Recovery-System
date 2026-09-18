from __future__ import annotations

from datetime import timedelta

from gpu_fault.models import MarkerScope, NodeMarker, RecoveryAction, Severity
from tests.store._cov95_runtime_models import CLUSTER, NODE, NOW

SCOPE_FIELDS = ("node_ids", "gpu_uuids", "fabric_partitions")


def marker(key, *, seconds=0, **updates):
    return NodeMarker(
        marker_id=key,
        cluster_id=CLUSTER,
        source="synthetic-store-contract",
        trusted=True,
        incident_id="incident-example",
        observed_at=NOW + timedelta(seconds=seconds),
        expires_at=NOW + timedelta(days=1),
        scope=MarkerScope(node_ids=[NODE]),
        severity=Severity.WARNING,
        recommended_action=RecoveryAction.QUARANTINE,
        mapping_version="simulated-v1",
    ).model_copy(update=updates)


def assert_cluster_filter_precedes_limit(store, scope_field, *, own_present):
    scope = MarkerScope(**{scope_field: ["shared-alias"]})
    own = marker("own", scope=scope, recommended_action=RecoveryAction.REBOOT_NODE)
    if own_present:
        store.add_marker(own)
    for index in range(1, 6):
        store.add_marker(
            marker(
                f"foreign-{index}",
                seconds=index,
                scope=scope,
                cluster_id="foreign-cluster",
            )
        )
        store.add_marker(
            marker(f"unbound-{index}", seconds=10 + index, scope=scope, cluster_id=None)
        )
    query = {
        "node_ids": set(),
        "gpu_uuids": set(),
        "fabric_partitions": set(),
        "observed_from": NOW,
        "observed_to": NOW + timedelta(seconds=20),
        "limit": 1,
    }
    query[scope_field] = {"shared-alias"}
    expected = [own] if own_present else []
    assert (
        store.list_markers_in_scope_window(cluster_id=CLUSTER, **query) == expected
    ), (
        "foreign or unbound markers must neither cross tenants nor consume the result limit"
    )
    assert (
        store.list_markers_in_scope_window(cluster_id="missing-cluster", **query) == []
    ), "a cluster with no marker cannot borrow another cluster's recovery evidence"
    unscoped = store.list_markers_in_scope_window(**query)
    assert [value.marker_id for value in unscoped] == ["unbound-5"], (
        "omitting cluster scope must preserve the existing unscoped caller contract"
    )
    assert store.list_markers_in_scope_window(cluster_id=None, **query) == unscoped, (
        "explicit None and omitted cluster scope must remain equivalent"
    )
