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
from gpu_fault.policy import SxidClassification, SxidEvent, SxidLinkScope, XidEvent
from tests._builders import attempt_observation, build_context, container_observation
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


def _incident(**overrides) -> FaultIncident:
    values = dict(
        incident_id="inc-xid-fallback",
        event_id="xid-fallback",
        event_type="XID",
        cluster_id="cluster-a",
        node_ids=["node-a"],
        policy_version="610",
        policy_source="NVIDIA",
    )
    values.update(overrides)
    return FaultIncident(**values)


def _sxid_event(**overrides) -> SxidEvent:
    values = dict(
        event_id="sxid-fallback",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW,
        ingested_at=NOW,
        sxid=11001,
        classification=SxidClassification.FATAL,
        classification_source="NVIDIA_FABRIC_MANAGER_CATALOG",
        link_scope=SxidLinkScope.TRUNK,
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[WORKLOAD_ID],
        job_id="train",
        attempt_id="train-a001",
        participating_gpu_uuids=["GPU-a", "GPU-b"],
    )
    values.update(overrides)
    return SxidEvent(**values)


def test_an_xid_restart_counts_the_events_own_gpu_when_no_observation_matches():
    """source_gpu_count must not be 0 on the XID path either.

    The XID kills the process, the watcher records the Pod as terminated
    before the event is ingested, and the [-30 s, +120 s] window holds no
    live container on the node. The step then carried source_gpu_count=0,
    _restart_guard read 0 != declared target as GPU_COUNT_CHANGED and
    waited for an approval the notification says cannot be produced --
    the restart never ran without a manual resubmit. The regional
    executor has no store to recover the count from.
    """
    context = build_context()
    builder = context.orchestrator._builder

    parameters = builder.step_parameters(
        WorkflowOperation.RESTART_WORKLOAD,
        _event(event_id="xid-fallback", xid=13, gpu_uuid="GPU-a"),
        _incident(),
    )

    assert parameters["source_gpu_count"] == 1
    assert parameters["source_attempt_id"] == "train-a001"


def test_the_compiled_xid_restart_step_carries_the_fallback_gpu_count():
    context = build_context()
    event = _event(event_id="xid-fallback", xid=13, gpu_uuid="GPU-a")
    decision = context.policy.evaluate_xid(event)

    _, workflow = context.orchestrator.ingest(event, decision)

    restart_steps = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    ]
    assert restart_steps, "XID 13 on an active workload must plan a restart"
    assert restart_steps[0].parameters["source_gpu_count"] == 1


def test_an_sxid_restart_counts_the_participating_gpus_when_no_observation_matches():
    context = build_context()
    builder = context.orchestrator._builder

    parameters = builder.step_parameters(
        WorkflowOperation.RESTART_WORKLOAD,
        _sxid_event(),
        _incident(incident_id="inc-sxid-fallback", event_id="sxid-fallback"),
    )

    assert parameters["source_gpu_count"] == 2


def test_the_restart_falls_back_to_the_incidents_gpus_when_the_event_names_none():
    context = build_context()
    builder = context.orchestrator._builder

    parameters = builder.step_parameters(
        WorkflowOperation.RESTART_WORKLOAD,
        _event(event_id="xid-fallback", xid=13, gpu_uuid=None),
        _incident(gpu_uuids=["GPU-x", "GPU-y", "GPU-z"]),
    )

    assert parameters["source_gpu_count"] == 3


def test_a_matching_observation_still_wins_over_the_event_gpu():
    """Regression guard: the fallback only fills an empty observation set."""
    context = build_context()
    context.store.save_attempt_observation(
        attempt_observation(
            "train",
            "train-a001",
            NOW,
            containers=[
                container_observation(
                    "pod-train-a001",
                    "trainer-train-a001",
                    0,
                    "node-a",
                    gpu_uuids=["GPU-a", "GPU-b"],
                )
            ],
            workload_ids=[WORKLOAD_ID],
            restart_budget=3,
        )
    )
    builder = context.orchestrator._builder

    parameters = builder.step_parameters(
        WorkflowOperation.RESTART_WORKLOAD,
        _event(event_id="xid-fallback", xid=13, gpu_uuid="GPU-z"),
        _incident(gpu_uuids=["GPU-z"]),
    )

    assert parameters["source_gpu_count"] == 2
