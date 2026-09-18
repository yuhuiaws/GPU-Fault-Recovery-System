from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.markers import (
    blocking_spare_markers,
    marker_blocks_spare,
    retire_markers_for_incident,
)
from gpu_fault.models import MarkerScope, NodeMarker, RecoveryAction, WorkflowStatus
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident, workflow_request
from tests.hyperpod._cov95_runtime_spare_health import NODE, HealthHarness

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


@pytest.fixture
def marker():
    return NodeMarker(
        marker_id="spare-marker",
        source="unit-observation",
        cluster_id="cluster-a",
        trusted=True,
        incident_id="spare-incident",
        observed_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=1),
        scope={"node_ids": ["node-a"]},
        severity="critical",
        recommended_action=RecoveryAction.REBOOT_NODE,
        mapping_version="unit-v1",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"active": False},
        {"trusted": False},
        {"expires_at": NOW},
        {"recommended_action": RecoveryAction.RUN_DIAGNOSTICS},
    ],
)
def test_inactive_untrusted_expired_and_advisory_markers_do_not_disqualify_a_spare(
    marker, change
):
    store = InMemoryStore()
    candidate = marker.model_copy(update=change)
    assert marker_blocks_spare(store, candidate, now=NOW) is False, candidate
    assert store.list_workflows() == [], "marker interpretation must remain read-only"


@pytest.mark.parametrize("history", ["no-incident-id", "unlinked", "missing-workflow"])
def test_absent_recovery_proof_keeps_a_live_fault_marker_blocking(marker, history):
    store = InMemoryStore()
    if history == "no-incident-id":
        marker = marker.model_copy(update={"incident_id": ""})
    else:
        store.save_incident(
            fault_incident(
                marker.incident_id,
                "source-event",
                workflow_request_id=None if history == "unlinked" else "gone-workflow",
            )
        )
    assert marker_blocks_spare(store, marker, now=NOW) is True, (
        "missing history cannot authorize reuse of a faulted node",
        history,
    )


@pytest.mark.parametrize(
    "binding",
    ["matching", "foreign-cluster", "unbound-marker", "foreign-workflow", "old-fence"],
)
def test_only_matching_recovery_identity_can_clear_a_live_spare_marker(marker, binding):
    store = InMemoryStore()
    incident = fault_incident(
        marker.incident_id,
        "source-event",
        cluster_id="cluster-b" if binding == "foreign-cluster" else "cluster-a",
        workflow_request_id="recovery-workflow",
    )
    workflow = workflow_request(
        "recovery-workflow",
        incident.incident_id,
        WorkflowStatus.SUCCEEDED,
        fencing_token=1,
    )
    store.save_incident_and_workflow(incident, workflow)
    if binding == "unbound-marker":
        marker = marker.model_copy(update={"cluster_id": None})
    elif binding == "foreign-workflow":
        other = fault_incident(
            "other-incident", "other-event", workflow_request_id="other-workflow"
        )
        store.save_incident_and_workflow(
            other,
            workflow_request(
                "other-workflow",
                other.incident_id,
                WorkflowStatus.SUCCEEDED,
                fencing_token=1,
            ),
        )
        current = store.get_incident(incident.incident_id)
        store.save_incident(
            current.model_copy(update={"workflow_request_id": "other-workflow"}),
            expected=current,
        )
    elif binding == "old-fence":
        current = store.get_incident(incident.incident_id)
        store.save_incident(
            current.model_copy(update={"fencing_token": current.fencing_token + 1}),
            expected=current,
        )
    before = store.get_incident(incident.incident_id)
    if binding in {"foreign-cluster", "foreign-workflow", "old-fence"}:
        with pytest.raises(ValueError, match="spare marker.*identity|fencing"):
            marker_blocks_spare(store, marker, now=NOW)
    else:
        assert marker_blocks_spare(store, marker, now=NOW) is (
            binding == "unbound-marker"
        ), (
            "only matching recovery proof may release a warm spare",
            binding,
            before,
            store.get_workflow(before.workflow_request_id),
        )
    assert store.get_incident(incident.incident_id) == before, (
        "proof validation must not repair or overwrite unrelated records"
    )


def test_spare_marker_query_ignores_an_entirely_foreign_marker(marker):
    store = InMemoryStore()
    store.add_marker(marker.model_copy(update={"cluster_id": "other-cluster"}))
    assert (
        blocking_spare_markers(store, {"node-a"}, cluster_id="cluster-a", now=NOW) == []
    ), "same-named foreign nodes must not enter this cluster's spare query"


def test_spare_scan_cannot_turn_contradictory_marker_identity_into_hardware_recovery(
    marker,
):
    h = HealthHarness(failure_threshold=1)
    foreign = fault_incident(
        "foreign-incident",
        "foreign-event",
        cluster_id="other-cluster",
        node_ids=[NODE],
        workflow_request_id="foreign-workflow",
    )
    h.store.save_incident_and_workflow(
        foreign,
        workflow_request(
            "foreign-workflow",
            foreign.incident_id,
            WorkflowStatus.SUCCEEDED,
            fencing_token=1,
        ),
    )
    h.store.add_marker(
        marker.model_copy(
            update={
                "cluster_id": "hp-cluster",
                "incident_id": foreign.incident_id,
                "scope": MarkerScope(node_ids=[NODE]),
                "observed_at": h.clock() - timedelta(seconds=1),
                "expires_at": h.clock() + timedelta(minutes=1),
            }
        )
    )
    before = h.store.list_workflows()
    result = h.scan()
    assert result["state"] == "SUSPECT", (
        "a contradictory pointer is not evidence authorizing a hardware reboot",
        result,
    )
    assert result["reasons"] == ["spare health reconciliation failed"], result
    assert h.store.list_workflows() == before, (
        "identity refusal must not create a recovery workflow"
    )
    assert h.core.patches == [], "identity refusal must not alter scheduler state"


def test_marker_retirement_replay_preserves_its_original_audit(marker):
    store = InMemoryStore()
    store.add_marker(marker)
    assert (
        retire_markers_for_incident(
            store,
            marker.incident_id,
            reason="validated restoration",
            retired_by="restoring-workflow",
            now=NOW,
        )
        == 1
    ), "the first retirement must retire exactly its one active marker"
    [retired] = store.list_markers_for_incident(marker.incident_id)
    assert retired.active is False and retired.retired_at == NOW, retired
    assert (
        retire_markers_for_incident(
            store,
            marker.incident_id,
            reason="later replay",
            retired_by="different-workflow",
            now=NOW + timedelta(minutes=1),
        )
        == 0
    ), "retirement replay must not rewrite already-retired evidence"
    assert store.list_markers_for_incident(marker.incident_id) == [retired], (
        "the original retirement actor, reason and time must remain intact"
    )
