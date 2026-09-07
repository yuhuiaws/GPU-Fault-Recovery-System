from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from gpu_fault.app import default_simulated_profile
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, WorkloadState
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.store import SqliteStore
from gpu_fault.watcher import AttemptObservation
from tests._builders import (
    attempt_observation,
    container_observation,
    copy_model,
    node_health_finding,
)

NOW = datetime(2026, 7, 23, 14, 0, tzinfo=timezone.utc)


def _observation() -> AttemptObservation:
    return attempt_observation(
        "train-multi",
        "train-multi-a001",
        NOW,
        cluster_id="hp-cluster",
        expected_critical_ranks=3,
        containers=[
            container_observation(
                f"pod-{rank}",
                f"worker-{rank}",
                rank,
                f"node-{rank}",
                gpu_uuids=[f"GPU-{rank}"],
            )
            for rank in range(3)
        ],
        workload_ids=["training/job/train-multi"],
        restart_budget=2,
    )


def _finding(rank: int) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-node-{rank}",
        f"event-node-{rank}",
        cluster_id="hp-cluster",
        node_id=f"node-{rank}",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason=f"simulated unrecoverable GPU fault on node-{rank}",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=[f"GPU-{rank}"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/train-multi"],
    )


def test_sqlite_concurrent_collectors_merge_without_losing_nodes(tmp_path) -> None:
    path = str(tmp_path / "multi-node.db")
    seed = SqliteStore(path)
    seed.save_profile(default_simulated_profile())
    seed.save_attempt_observation(_observation())
    seed.close()
    first = SqliteStore(path)
    second = SqliteStore(path)
    orchestrators = [
        IncidentOrchestrator(first, multi_node_aggregation_window_seconds=30),
        IncidentOrchestrator(second, multi_node_aggregation_window_seconds=30),
    ]

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda item: item[0].ingest_node_health(_finding(item[1])),
                    zip(orchestrators, [0, 1], strict=True),
                )
            )

        incident_ids = {incident.incident_id for incident, _ in results}
        workflow_ids = {workflow.request_id for _, workflow in results}
        stored_incident = first.get_incident(incident_ids.pop())
        stored_workflow = first.get_workflow(workflow_ids.pop())
        assert not incident_ids
        assert not workflow_ids
        assert stored_incident.node_ids == ["node-0", "node-1"]
        assert len(first.list_workflows()) == 1
        replace = next(
            step
            for step in stored_workflow.official_steps
            if step.operation.value == "REPLACE_NODE"
        )
        restart = next(
            step
            for step in stored_workflow.official_steps
            if step.operation.value == "RESTART_WORKLOAD"
        )
        assert replace.node_ids == ["node-0", "node-1"]
        assert restart.node_ids == ["node-0", "node-1", "node-2"]
    finally:
        first.close()
        second.close()


def test_fault_after_aggregation_deadline_starts_new_workflow(tmp_path) -> None:
    store = SqliteStore(str(tmp_path / "expired-group.db"))
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(_observation())
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    try:
        first_incident, first_workflow = orchestrator.ingest_node_health(_finding(0))
        store.save_workflow(
            copy_model(
                first_workflow,
                not_before=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
        )

        second_incident, second_workflow = orchestrator.ingest_node_health(_finding(1))

        assert second_incident.incident_id != (first_incident.incident_id)
        assert second_workflow.request_id != first_workflow.request_id
        assert len(store.list_workflows()) == 2
    finally:
        store.close()
