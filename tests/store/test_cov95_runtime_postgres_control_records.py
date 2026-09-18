from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceStatus,
)
from gpu_fault.managed_recovery import HyperPodNodeIdentity
from gpu_fault.models import MarkerScope, RecoveryAction
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from tests.store import _cov95_runtime_postgres as postgres
from tests.store._cov95_runtime_markers import (
    SCOPE_FIELDS,
    assert_cluster_filter_precedes_limit,
    marker,
)
from tests.store._cov95_runtime_models import (
    CLUSTER,
    NODE,
    NOW,
    ingestion,
    metric,
    metric_key,
    observation,
    progress,
)
from tests.store._postgres_processor_claim_support import postgres_store_instance


@pytest.fixture
def store(monkeypatch):
    postgres.validated_url()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
    yield from postgres_store_instance()


def resource(site="site-example", key="resource-example", **updates):
    return InstallationResource(
        site_id=site,
        resource_key=key,
        provider="fake",
        resource_type="local-test",
        resource_id=f"owned/{key}",
        ownership=InstallationResourceOwnership.REUSED,
        delete_policy=InstallationResourceDeletePolicy.PRESERVE,
        created_at=NOW,
        updated_at=NOW,
    ).model_copy(update=updates)


def test_installation_resources_preserve_identity_and_allow_status_updates(store):
    first = resource()
    assert store.list_installation_resources() == [], "the registry must start empty"
    with pytest.raises(NotFoundError):
        store.get_installation_resource(first.site_id, first.resource_key)
    assert store.save_installation_resource(first) == first
    changed = first.model_copy(
        update={
            "status": InstallationResourceStatus.DELETE_PENDING,
            "attributes": {"phase": "test-only"},
            "updated_at": NOW + timedelta(seconds=1),
        }
    )
    assert store.save_installation_resource(changed) == changed
    with pytest.raises(ValueError, match="identity cannot change"):
        store.save_installation_resource(
            changed.model_copy(update={"resource_id": "other"})
        )
    other = resource("another-site", "second-resource")
    store.save_installation_resource(other)
    assert store.get_installation_resource(first.site_id, first.resource_key) == changed
    assert store.list_installation_resources(first.site_id) == [changed]
    assert store.list_installation_resources("absent-site") == []
    assert store.list_installation_resources() == [other, changed]


def test_node_identity_lists_are_scoped_ordered_and_preserve_newer_generations(store):
    first = HyperPodNodeIdentity(
        cluster_name=CLUSTER,
        node_logical_id=NODE,
        instance_id="i-synthetic",
        status="InService",
        observed_at=NOW,
    )
    other = first.model_copy(update={"cluster_name": "another-cluster"})
    updated = first.model_copy(
        update={"observed_at": NOW + timedelta(seconds=1), "generation": 2}
    )
    store.save_hyperpod_node_identity(first)
    store.save_hyperpod_node_identity(other)
    assert store.save_hyperpod_node_identity(updated) == updated
    assert store.save_hyperpod_node_identity(first) == updated
    assert store.list_hyperpod_node_identities(CLUSTER) == [updated]
    assert store.list_hyperpod_node_identities() == [other, updated]
    assert store.list_hyperpod_node_identities("missing") == []


def test_active_marker_reads_exclude_untrusted_inactive_and_other_tenants(store):
    values = [
        marker("older"),
        marker("newer", seconds=1),
        marker("untrusted", trusted=False),
        marker("inactive", active=False),
        marker("foreign", cluster_id="foreign-cluster"),
        marker("legacy", cluster_id=None),
    ]
    for value in values:
        store.add_marker(value)
    actions = {RecoveryAction.QUARANTINE}
    assert store.list_active_markers_for_nodes(set(), actions) == []
    assert store.list_active_markers_for_nodes({NODE}, set()) == []
    scoped = store.list_active_markers_for_nodes({NODE}, actions, cluster_id=CLUSTER)
    assert [value.marker_id for value in scoped] == ["newer", "older"], (
        "a cluster read must not trust a colliding node ID from another tenant"
    )
    assert {
        value.marker_id
        for value in store.list_active_markers_for_nodes({NODE}, actions)
    } == {"older", "newer", "foreign", "legacy"}
    assert store.list_recent_markers_for_nodes(set(), NOW) == []
    assert store.list_recent_markers_for_nodes({NODE}, NOW, limit=0) == []
    assert store.list_recent_markers_for_nodes({NODE}, NOW, limit=1) == [values[1]]


@pytest.mark.parametrize("scope_field", ["node_ids", "gpu_uuids", "fabric_partitions"])
def test_marker_window_uses_inclusive_time_and_the_requested_scope(store, scope_field):
    scope = MarkerScope(**{scope_field: ["scope-example"]})
    values = [
        marker("before", seconds=-1, scope=scope),
        marker("start", scope=scope),
        marker("end", seconds=2, scope=scope),
        marker("after", seconds=3, scope=scope),
        marker("no-action", recommended_action=None, scope=scope),
        marker("untrusted", trusted=False, scope=scope),
    ]
    for value in values:
        store.add_marker(value)
    query = dict(
        node_ids=set(),
        gpu_uuids=set(),
        fabric_partitions=set(),
        observed_from=NOW,
        observed_to=NOW + timedelta(seconds=2),
    )
    assert store.list_markers_in_scope_window(**query) == []
    query[scope_field] = {"scope-example"}
    assert store.list_markers_in_scope_window(**query, limit=0) == []
    assert [
        value.marker_id
        for value in store.list_markers_in_scope_window(**query, limit=2)
    ] == ["end", "start"], "the window must include both boundary observations"


@pytest.mark.parametrize("scope_field", SCOPE_FIELDS)
@pytest.mark.parametrize(
    "own_present", [False, True], ids=["no-own-marker", "own-marker"]
)
def test_cluster_bound_marker_window_filters_before_limit(
    store, scope_field, own_present
):
    assert_cluster_filter_precedes_limit(store, scope_field, own_present=own_present)


def test_raw_evidence_reads_filter_attempt_kind_node_and_expiry(store):
    now = datetime.now(timezone.utc)
    first = RawEvidenceRecord(
        record_id="evidence-one",
        cluster_id=CLUSTER,
        node_id=NODE,
        kind=EvidenceKind.GPU_METRICS,
        observed_at=now,
        ingested_at=now,
        expires_at=now + timedelta(hours=1),
        attempt_ids=["attempt-example"],
        payload={"synthetic": True},
    )
    second = first.model_copy(
        update={
            "record_id": "evidence-two",
            "kind": EvidenceKind.HOST_TELEMETRY,
            "observed_at": now + timedelta(seconds=1),
            "attempt_ids": ["attempt-other"],
        }
    )
    expired = first.model_copy(
        update={
            "record_id": "evidence-expired",
            "expires_at": now - timedelta(seconds=1),
        }
    )
    for record in (first, second, expired):
        store.save_raw_evidence(record, max_records_per_node=10)
    assert store.list_raw_evidence(CLUSTER) == [second, first]
    assert store.list_raw_evidence(
        CLUSTER, node_id=NODE, attempt_id="attempt-example"
    ) == [first]
    assert store.list_raw_evidence(CLUSTER, kind=EvidenceKind.HOST_TELEMETRY) == [
        second
    ]
    assert store.list_raw_evidence(CLUSTER, attempt_id="missing") == []
    assert store.list_raw_evidence(CLUSTER, limit=1) == [second]
    assert store.list_raw_evidence(CLUSTER, node_id="missing") == []


def test_hot_backfill_refuses_unsafe_purge_and_preserves_newer_native_state(store):
    sample = metric()
    store.observe_gpu_metric(metric_key(sample), sample)
    store.save_gpu_metrics_batch((CLUSTER, NODE, "batch-example"), ingestion())
    store.observe_training_progress(progress())
    store.save_attempt_observation(observation())
    expected_kinds = {
        "gpu_metric_latest",
        "gpu_metrics_batch",
        "training_progress",
        "attempt_observation",
    }
    before = store.hot_state_migration_status()
    assert set(before) == expected_kinds, "every hot-state family must be inspected"
    assert all(item["missing_or_mismatched"] == 1 for item in before.values()), (
        "all seeded legacy rows must initially lack a native twin"
    )
    with pytest.raises(RuntimeError, match="backfill is incomplete"):
        store.purge_legacy_hot_state()
    assert store.hot_state_migration_status() == before, (
        "refused purge must not delete data"
    )
    assert store.backfill_hot_state_tables() == {kind: 1 for kind in expected_kinds}
    assert store.hot_state_backfill_gaps() == {kind: False for kind in expected_kinds}
    newer = metric(75.0, 2)
    with postgres.peer_store("dedicated") as native:
        native.observe_gpu_metric(metric_key(newer), newer)
        native.observe_training_progress(progress(2, step=2))
        native.save_attempt_observation(observation(2))
        store.backfill_hot_state_tables()
        assert native.list_gpu_metrics_latest(CLUSTER, NODE) == [newer]
        assert native.list_training_progress(CLUSTER) == [progress(2, step=2)]
        assert native.list_attempt_observations(CLUSTER) == [observation(2)]
        assert store.purge_legacy_hot_state() == {kind: 1 for kind in expected_kinds}
        assert store.purge_legacy_hot_state() == {kind: 0 for kind in expected_kinds}
        assert (
            native.get_gpu_metrics_batch((CLUSTER, NODE, "batch-example"))
            == ingestion()
        )
        assert native.list_gpu_metrics_latest(CLUSTER, NODE) == [newer]
    assert store.list_gpu_metrics_latest(CLUSTER, NODE) == [], (
        "finalized legacy records must be absent while native state remains authoritative"
    )
