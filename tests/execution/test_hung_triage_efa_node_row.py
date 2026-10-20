"""The hung triage reads a node's EFA zero time only for the incident's attempt.

Batch 145: the EFA traffic state is one row per node owned by the attempt in
its payload. ``_hung_efa_zero_pending_times`` orders weak hang candidates by
when each node's traffic went to zero; it must read that time from the node
row only while the row still belongs to the incident's attempt. Once a
replacement attempt took the node over, the node has no zero time for this
incident -- as when the per-attempt row was missing before.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.efa_traffic_state import (
    efa_traffic_state_key,
    legacy_efa_traffic_state_key,
)
from gpu_fault.models import IncidentState
from tests._builders import active_workflow_executor, build_store, fault_incident

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"


def _sample(
    store, node_id: str, at: datetime, bps: float, *, attempt_id: str, key=None
):
    return store.observe_efa_traffic(
        state_key=key or efa_traffic_state_key(CLUSTER, node_id),
        cluster_id=CLUSTER,
        node_id=node_id,
        job_id="job-a",
        attempt_id=attempt_id,
        observed_at=at,
        bytes_per_second=bps,
        minimum_active_bps=100.0,
        spike_ratio=4.0,
        drop_ratio=0.25,
        zero_bps=1.0,
        zero_warning_seconds=60.0,
        zero_hung_seconds=600.0,
        baseline_alpha=0.2,
        startup_grace_seconds=120.0,
        spike_event_id=f"{node_id}-spike",
    )[0]


def _zero_streak(store, node_id: str, *, attempt_id: str = "attempt-a", key=None):
    for seconds in (0, 130):
        _sample(
            store,
            node_id,
            NOW + timedelta(seconds=seconds),
            1000.0,
            attempt_id=attempt_id,
            key=key,
        )
    return _sample(
        store,
        node_id,
        NOW + timedelta(seconds=150),
        0.0,
        attempt_id=attempt_id,
        key=key,
    )


def test_only_the_incidents_attempt_lends_its_zero_time() -> None:
    store = build_store()
    own = _zero_streak(store, "node-a")
    _zero_streak(store, "node-b")
    # A replacement attempt took node-b over: its row is that attempt's now.
    _sample(
        store, "node-b", NOW + timedelta(seconds=200), 1000.0, attempt_id="attempt-b"
    )
    # node-c still has the state a release before batch 145 kept per attempt.
    legacy = _zero_streak(
        store,
        "node-c",
        key=legacy_efa_traffic_state_key(
            efa_traffic_state_key(CLUSTER, "node-c"), "job-a", "attempt-a"
        ),
    )
    incident = fault_incident(
        "incident-hung",
        "event-hung",
        "NODE_HEALTH",
        node_ids=["node-a", "node-b", "node-c", "node-d"],
        job_id="job-a",
        attempt_id="attempt-a",
        policy_version="v1",
        policy_source="SITE_EFA_TRAFFIC",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
    )
    executor = active_workflow_executor(store, [], frozenset())

    times = executor._hung_efa_zero_pending_times(
        incident, {"node-a", "node-b", "node-c", "node-d"}
    )

    assert times == {"node-a": own.zero_since, "node-c": legacy.zero_since}
    assert own.zero_since == NOW + timedelta(seconds=150)
