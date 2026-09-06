"""Restart safety parameters are matched on the control plane's clock and
read a bounded slice of observations (F-B8).

The RESTART_WORKLOAD step needs the attempt's GPU count and restart budget
from the watcher's latest observation. The match window compared the node's
``observed_at`` with the watcher's control-plane ``observed_at``; a node
clock a few minutes off emptied the window, the step got ``source_gpu_count
= 0`` and an unapprovable WAITING. The lookup also listed every observation
the cluster ever had, inside the merge lock.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.models import FaultIncident, WorkflowOperation, WorkloadState
from gpu_fault.policy import XidEvent
from tests._builders import build_context
from tests.orchestration._cross_fault_support import NOW, WORKLOAD_ID, observation


def _event(**overrides) -> XidEvent:
    values = dict(
        event_id="xid-clock",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW
        - timedelta(minutes=40),  # the node's clock is 40 minutes behind
        ingested_at=NOW,
        xid=79,
        gpu_uuid="GPU-a",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[WORKLOAD_ID],
        job_id="train",
        attempt_id="train-a001",
    )
    values.update(overrides)
    return XidEvent(**values)


def test_the_restart_context_is_matched_on_the_receive_time_not_the_node_clock():
    context = build_context()
    context.store.save_attempt_observation(observation())
    builder = context.orchestrator._builder
    incident = FaultIncident(
        incident_id="inc-clock",
        event_id="xid-clock",
        event_type="XID",
        cluster_id="cluster-a",
        node_ids=["node-a"],
        policy_version="610",
        policy_source="NVIDIA",
    )

    parameters = builder.step_parameters(
        WorkflowOperation.RESTART_WORKLOAD, _event(), incident
    )

    assert parameters["source_attempt_id"] == "train-a001"
    assert parameters["source_gpu_count"] == 1
    assert parameters["restart_budget"] == 3


def test_the_observation_lookup_is_bounded_and_newest_first():
    context = build_context()
    context.store.save_attempt_observation(observation())
    calls: list[dict] = []
    original = context.store.list_attempt_observation_states

    def spy(cluster_id=None, **kwargs):
        calls.append(kwargs)
        return original(cluster_id, **kwargs)

    context.store.list_attempt_observation_states = spy  # type: ignore[method-assign]

    context.orchestrator._builder.restart_step_parameters(
        "cluster-a", [WORKLOAD_ID], job_id="train", observed_at=NOW, node_id="node-a"
    )

    assert calls and calls[0].get("limit"), "the scan must be bounded"
    assert calls[0].get("newest_first") is True, "the newest attempts must come first"
