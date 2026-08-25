from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import WorkflowOperation, WorkloadState
from gpu_fault.orchestrator import IncidentOrchestrator
from gpu_fault.policy import (
    GpuFaultPolicyEngine,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
)
from gpu_fault.store import SqliteStore
from gpu_fault.watcher import AttemptObservation
from tests._builders import (
    attempt_observation,
    build_sxid_event,
    container_observation,
    copy_model,
)

NOW = datetime.now(timezone.utc)


def observation(attempt_id: str = "train-a001") -> AttemptObservation:
    return attempt_observation(
        "train",
        attempt_id,
        NOW,
        expected_critical_ranks=2,
        containers=[
            container_observation(
                f"pod-{rank}",
                f"trainer-{rank}",
                rank,
                f"node-{rank}",
                gpu_uuids=[f"GPU-{rank}-0", f"GPU-{rank}-1"],
            )
            for rank in range(2)
        ],
        workload_ids=["training/pytorchjob/train"],
        restart_budget=3,
    )


def sxid(rank: int, *, event_id: str | None = None) -> SxidEvent:
    return build_sxid_event(
        event_id or f"sxid-node-{rank}",
        NOW + timedelta(seconds=1),
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        node_id=f"node-{rank}",
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        product="H200",
        fabric_partition=f"cluster-a/node-{rank}/local-nvswitch",
        participating_gpu_uuids=[f"GPU-{rank}-0", f"GPU-{rank}-1"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train"],
    )


def ingest(orchestrator: IncidentOrchestrator, event: SxidEvent):
    decision = GpuFaultPolicyEngine().evaluate_sxid(event)
    return orchestrator.ingest(event, decision)


def test_concurrent_node_sxids_merge_into_one_attempt_workflow(tmp_path) -> None:
    path = str(tmp_path / "multi-node-sxid.db")
    seed = SqliteStore(path)
    seed.save_profile(default_simulated_profile())
    seed.save_attempt_observation(observation())
    seed.close()
    stores = [SqliteStore(path), SqliteStore(path)]
    orchestrators = [
        IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)
        for store in stores
    ]

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda item: ingest(item[0], sxid(item[1])),
                    zip(orchestrators, [0, 1], strict=True),
                )
            )

        assert len({incident.incident_id for incident, _ in results}) == 1
        assert len({workflow.request_id for _, workflow in results}) == 1
        workflow = stores[0].get_workflow(results[0][1].request_id)
        incident = stores[0].get_incident(workflow.incident_id)
        assert incident.event_type == "GPU_FAULT_GROUP"
        assert incident.node_ids == ["node-0", "node-1"]
        assert workflow.not_before is not None
        assert len(stores[0].list_workflows()) == 1

        reset = next(
            step
            for step in workflow.official_steps
            if step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
        )
        assert reset.node_ids == ["node-0", "node-1"]
        assert reset.parameters["gpu_uuids_by_node"] == {
            "node-0": ["GPU-0-0", "GPU-0-1"],
            "node-1": ["GPU-1-0", "GPU-1-1"],
        }
        assert reset.parameters["fabric_partitions_by_node"] == {
            "node-0": "cluster-a/node-0/local-nvswitch",
            "node-1": "cluster-a/node-1/local-nvswitch",
        }
        assert reset.parameters["sxids_by_node"] == {
            "node-0": [11001],
            "node-1": [11001],
        }

        restart_steps = [
            step
            for step in workflow.official_steps
            if step.operation is WorkflowOperation.RESTART_WORKLOAD
        ]
        assert len(restart_steps) == 1
        assert restart_steps[0].node_ids == ["node-0", "node-1"]
        assert restart_steps[0].parameters == {
            "cluster_id": "cluster-a",
            "job_id": "train",
            "source_attempt_id": "train-a001",
            "source_gpu_count": 4,
            "restart_budget": 3,
        }
    finally:
        for store in stores:
            store.close()


def test_concurrent_access_sxids_merge_into_one_attempt_reset(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "access-sxid.db"))
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(observation())
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)
    try:
        events = [
            copy_model(
                sxid(rank),
                event_id=f"access-sxid-{rank}",
                link_scope=SxidLinkScope.ACCESS,
                link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
                fabric_partition=None,
            )
            for rank in range(2)
        ]
        results = [ingest(orchestrator, event) for event in events]

        assert len({incident.incident_id for incident, _ in results}) == 1
        workflow = store.get_workflow(results[-1][1].request_id)
        reset = next(
            step
            for step in workflow.official_steps
            if step.operation is WorkflowOperation.RESET_GPU
        )
        assert reset.node_ids == ["node-0", "node-1"]
        assert reset.parameters["gpu_uuids_by_node"] == {
            "node-0": ["GPU-0-0", "GPU-0-1"],
            "node-1": ["GPU-1-0", "GPU-1-1"],
        }
        assert not any(
            step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            for step in workflow.official_steps
        )
        assert (
            len(
                [
                    step
                    for step in workflow.official_steps
                    if step.operation is WorkflowOperation.RESTART_WORKLOAD
                ]
            )
            == 1
        )
    finally:
        store.close()


def test_sxids_from_different_attempts_do_not_merge(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "sxid-attempts.db"))
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(observation("train-a001"))
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    try:
        first_incident, first_workflow = ingest(
            orchestrator, sxid(0, event_id="sxid-attempt-1")
        )
        store.save_attempt_observation(
            copy_model(
                observation("train-a002"), observed_at=NOW + timedelta(seconds=2)
            )
        )
        second_event = copy_model(
            sxid(1, event_id="sxid-attempt-2"), observed_at=NOW + timedelta(seconds=3)
        )
        second_incident, second_workflow = ingest(orchestrator, second_event)

        assert first_incident.incident_id != second_incident.incident_id
        assert first_workflow.request_id != second_workflow.request_id
    finally:
        store.close()


def test_unclaimed_workflow_still_merges_after_aggregation_deadline(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "sxid-deadline.db"))
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(observation())
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    try:
        first_incident, first_workflow = ingest(
            orchestrator, sxid(0, event_id="sxid-before-deadline")
        )
        store.save_workflow(
            copy_model(
                first_workflow,
                not_before=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
        )
        second_incident, second_workflow = ingest(
            orchestrator, sxid(1, event_id="sxid-after-deadline")
        )

        assert first_incident.incident_id == second_incident.incident_id
        assert first_workflow.request_id == second_workflow.request_id
    finally:
        store.close()
