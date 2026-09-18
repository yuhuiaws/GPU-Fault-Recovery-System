from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import MarkerScope
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import fault_incident
from tests.orchestration._cov95_orch_extra_faults import policy_decision, xid
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)
from tests.orchestration._cov95_orch_extra_support import NOW, memory_store


@pytest.mark.parametrize(
    ("case", "matched"),
    [
        ("same", True),
        ("legacy-match", True),
        ("legacy-missing", False),
        ("legacy-foreign", False),
        ("unclassified-same", True),
        ("unclassified-different", False),
        ("same-source-unscoped", True),
        ("different-source-unscoped", False),
        ("different-reason-unscoped", False),
        ("same-source-expired", False),
        ("monotonic-over-wall-clock", True),
        ("naive-source-clock", True),
        ("foreign-tenant", False),
        ("self-marker", False),
        ("disjoint-gpu", False),
    ],
)
def test_provider_correlation_obeys_tenant_scope_identity_and_the_applicable_clock(
    case, matched
):
    store = memory_store()
    event = xid(source_boot_id="unit-boot", source_monotonic_us=11_000_000)
    decision = policy_decision(event)
    original = decision.model_copy(deep=True)
    updates = {
        "marker_id": "unit-candidate",
        "incident_id": "inc-candidate",
        "observed_at": NOW,
        "trusted": True,
        "active": True,
    }
    if case.startswith("legacy-"):
        updates["cluster_id"] = None
    if case.startswith("unclassified-"):
        updates["fault_class"] = None
        if case.endswith("different"):
            updates["raw_reason"] = "different unit condition"
    if case.endswith("unscoped"):
        updates["scope"] = MarkerScope(node_ids=["node-0"])
        updates["correlation_keys"] = []
        if case == "different-source-unscoped":
            updates["event_source"] = "DCGM"
        if case == "different-reason-unscoped":
            updates["raw_reason"] = "another unit condition"
    if case == "same-source-expired":
        updates["source_monotonic_us"] = None
        updates["source_event_time"] = NOW - timedelta(seconds=31)
    elif case == "monotonic-over-wall-clock":
        updates["source_monotonic_us"] = 10_000_000
        updates["source_event_time"] = NOW - timedelta(days=1)
    elif case == "naive-source-clock":
        updates["source_monotonic_us"] = None
        updates["source_event_time"] = NOW.replace(tzinfo=None)
    elif case == "foreign-tenant":
        updates["cluster_id"] = "cluster-b"
    elif case == "self-marker":
        updates["marker_id"] = decision.marker.marker_id
    elif case == "disjoint-gpu":
        updates["scope"] = MarkerScope(node_ids=["node-0"], gpu_uuids=["GPU-other"])
    candidate = decision.marker.model_copy(deep=True, update=updates)
    store.add_marker(candidate)
    if case != "legacy-missing":
        store.save_incident(
            fault_incident(
                "inc-candidate",
                "candidate-event",
                cluster_id="cluster-b" if case == "legacy-foreign" else "cluster-a",
                node_ids=["node-0"],
            )
        )

    result = IncidentOrchestrator(store).correlate_provider_event(
        decision, cluster_id="cluster-a"
    )

    assert result.duplicate is matched
    assert result.marker.incident_id == (
        "inc-candidate" if matched else decision.marker.incident_id
    )
    assert decision == original, "correlation mutated the original policy decision"
    assert store.list_markers() == [candidate]
    assert store.list_workflows() == [], "a correlation decision dispatched a workflow"
