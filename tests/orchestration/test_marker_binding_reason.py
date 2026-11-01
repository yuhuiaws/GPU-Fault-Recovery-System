"""A decision names the incident its marker was associated with, and no other.

``bind_marker_incident`` moves a finalized marker to the incident that owns the
execution and keeps the provider association as a reason. When the store
merged an event that was never associated -- a second GPU's XID on an idle
node joins the node's open reset group -- the marker still carried the event's
own ``inc-<event_id>``, an incident that is never written, and the reason named
it as an association (batch 237; reported by batch 236).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.app import default_simulated_profile
from gpu_fault.orchestration.provider_correlation import bind_marker_incident
from gpu_fault.policy import GpuFaultPolicyEngine, XidEvent
from gpu_fault.store import InMemoryStore, NotFoundError
from tests._builders import asgi_client, build_context

NOW = datetime.now(timezone.utc) - timedelta(minutes=1)
ASSOCIATION = "Provider marker association with incident"


def _xid(event_id: str, gpu: str, seconds: int) -> dict:
    return {
        "event_id": event_id,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": (NOW + timedelta(seconds=seconds)).isoformat(),
        "xid": 48,
        "gpu_uuid": gpu,
        "product": "H100",
        "driver_branch": 575,
        "cuda_version": "12.9",
        "runtime_profile_version": "simulated-v1",
        "workload_state": "IDLE",
    }


def test_a_store_merged_event_names_no_unwritten_incident() -> None:
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    context = build_context(store=store)

    async def scenario() -> list[dict]:
        async with asgi_client(context) as client:
            bodies = []
            for event_id, gpu, seconds in (
                ("first", "GPU-a", 0),
                ("second", "GPU-b", 2),
            ):
                response = await client.post(
                    "/v1/gpu-events/xid", json=_xid(event_id, gpu, seconds)
                )
                assert response.status_code == 200, response.text
                bodies.append(response.json())
            return bodies

    first, second = asyncio.run(scenario())

    assert second["incident_id"] == first["incident_id"], "the node group merged them"
    assert not second["duplicate"] and second["correlated_event_id"] is None
    named = [reason for reason in second["reasons"] if ASSOCIATION in reason]
    assert named == [], f"the decision names an association that never was: {named}"
    try:
        store.get_incident("inc-second")
    except NotFoundError:
        pass
    else:  # pragma: no cover - the merge wrote no own incident
        raise AssertionError("inc-second exists")


def test_an_association_is_still_named() -> None:
    decision = GpuFaultPolicyEngine().evaluate_xid(
        XidEvent(**{**_xid("assoc", "GPU-a", 0), "observed_at": NOW})
    )
    associated = decision.model_copy(
        update={
            "duplicate": True,
            "marker": decision.marker.model_copy(update={"incident_id": "inc-written"}),
        }
    )

    bound = bind_marker_incident(associated, "inc-owner")
    unassociated = bind_marker_incident(decision, "inc-owner")

    assert bound.marker.incident_id == "inc-owner"
    assert any(f"{ASSOCIATION} inc-written" in reason for reason in bound.reasons), (
        bound.reasons
    )
    assert unassociated.marker.incident_id == "inc-owner"
    assert not any(ASSOCIATION in reason for reason in unassociated.reasons), (
        unassociated.reasons
    )
