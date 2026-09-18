from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.hyperpod_spares import SPARE_RESERVED_AT_ANNOTATION
from gpu_fault.spare_health import SpareReservationReclaimer
from tests._builders import fault_incident
from tests.hyperpod._cov95_runtime_spare_health import NODE, HealthHarness
from tests.hyperpod.test_hyperpod_spares import pod


@pytest.mark.parametrize("pointer", [None, "missing-workflow"])
def test_known_reservation_owner_without_a_workflow_is_reported_for_reclamation(
    pointer,
):
    h = HealthHarness()
    incident = fault_incident(
        "reservation-owner", "owner-event", workflow_request_id=pointer
    )
    h.store.save_incident(incident)
    reclaimer = SpareReservationReclaimer(
        h.coordinator, h.store, now=h.clock, ttl_seconds=60
    )
    reason = reclaimer.reason(NODE, h.node, incident.incident_id)
    assert reason == "incident reservation-owner has no workflow", reason
    assert h.core.patches == [], "the decision alone must not mutate a reservation"
    assert h.store.get_incident(incident.incident_id) == incident, incident


def test_active_gpu_pods_prevent_reclaim_even_after_the_owner_reservation_expires(
    monkeypatch,
):
    h = HealthHarness()
    h.annotations[SPARE_RESERVED_AT_ANNOTATION] = (
        h.clock() - timedelta(hours=1)
    ).isoformat()
    h.core.pod_batches = [[pod("training-rank", limits={"nvidia.com/gpu": "1"})]]

    def unexpected(_incident_id):
        raise AssertionError("an occupied spare must be held before reading its owner")

    monkeypatch.setattr(h.store, "get_incident", unexpected)
    reclaimer = SpareReservationReclaimer(
        h.coordinator, h.store, now=h.clock, ttl_seconds=60
    )
    assert reclaimer.reason(NODE, h.node, "old-owner") is None, (
        "expiry cannot authorize reclaiming an active training node"
    )
    assert h.core.patches == [], "reservation review must not touch the active node"
