from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    Severity,
    WorkflowRequest,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.store import SqliteStore
from tests._builders import (
    build_store,
    fault_incident,
    node_health_finding,
    workflow_request,
)


def _builder(event_id: str, suffix: str):
    def build() -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        incident_id = f"incident-{suffix}"
        workflow_id = f"workflow-{suffix}"
        return (
            fault_incident(
                incident_id,
                event_id,
                "TRAINING_ATTEMPT_FAILURE_DETECTED",
                policy_version="passive-containment-v1",
                policy_source="completion-watcher",
                effective_action=RecoveryAction.STOP_WORKLOAD,
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                created_at=now,
                updated_at=now,
            ),
            workflow_request(
                workflow_id,
                incident_id,
                fencing_token=1,
                created_at=now,
                updated_at=now,
            ),
        )

    return build


def test_in_memory_incident_workflow_create_is_atomic() -> None:
    store = build_store()
    event_id = "cluster-a/attempt-a/failure"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda suffix: (
                    store.create_incident_workflow_if_absent(
                        event_id, _builder(event_id, suffix)
                    )
                ),
                ["a", "b"],
            )
        )

    assert len({item[0].incident_id for item in results}) == 1
    assert len({item[1].request_id for item in results}) == 1
    assert sorted(item[2] for item in results) == [False, True]


def test_sqlite_incident_workflow_create_is_atomic_across_instances(tmp_path) -> None:
    path = str(tmp_path / "passive-containment.db")
    first = SqliteStore(path)
    second = SqliteStore(path)
    event_id = "cluster-a/attempt-a/failure"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda item: (
                        item[0].create_incident_workflow_if_absent(
                            event_id, _builder(event_id, item[1])
                        )
                    ),
                    [(first, "a"), (second, "b")],
                )
            )

        assert len({item[0].incident_id for item in results}) == 1
        assert len({item[1].request_id for item in results}) == 1
        assert sorted(item[2] for item in results) == [False, True]
    finally:
        first.close()
        second.close()


def test_duplicate_create_does_not_overwrite_workflow_state() -> None:
    store = build_store()
    event_id = "cluster-a/attempt-a/failure"
    _, workflow, created = store.create_incident_workflow_if_absent(
        event_id, _builder(event_id, "first")
    )
    store.save_workflow(workflow.model_copy(update={"status": WorkflowStatus.RUNNING}))

    _, duplicate_workflow, duplicate_created = store.create_incident_workflow_if_absent(
        event_id, _builder(event_id, "second")
    )

    assert created
    assert not duplicate_created
    assert duplicate_workflow.status is WorkflowStatus.RUNNING


def test_non_group_node_health_serializes_across_store_instances(tmp_path) -> None:
    barrier = Barrier(2)

    class BarrierSqliteStore(SqliteStore):
        def create_incident_workflow_if_absent(
            self, event_id, builder, *, serialization_key=None
        ):
            barrier.wait(timeout=5)
            return super().create_incident_workflow_if_absent(
                event_id, builder, serialization_key=serialization_key
            )

    path = str(tmp_path / "node-health-atomicity.db")
    stores = [BarrierSqliteStore(path), BarrierSqliteStore(path)]
    contexts = [ApplicationContext(store=store) for store in stores]

    def ingest(item):
        index, context = item
        finding = node_health_finding(
            f"node-health-{index}",
            f"node-health-{index}",
            observed_at=datetime.now(timezone.utc),
            category=NodeHealthCategory.SYSTEM_LOG,
            severity=Severity.CRITICAL,
            reason=f"fatal storage finding {index}",
            recommended_action=RecoveryAction.DRAIN,
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        )
        return context.orchestrator.ingest_node_health(finding)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(ingest, enumerate(contexts)))

        workflows = [item[1] for item in results]
        roots = [item for item in workflows if item.predecessor_workflow_id is None]
        successors = [
            item for item in workflows if item.predecessor_workflow_id is not None
        ]
        assert len(roots) == 1
        assert len(successors) == 1
        assert successors[0].predecessor_workflow_id == roots[0].request_id
        assert any(
            "serialized behind in-flight node-exclusive workflow" in reason
            for reason in results[workflows.index(successors[0])][0].reasons
        )
    finally:
        for store in stores:
            store.close()


def test_active_workflow_scope_query_filters_cluster_node_and_job(tmp_path) -> None:
    stores = [build_store(), SqliteStore(str(tmp_path / "active-scope.db"))]
    try:
        for index, store in enumerate(stores):
            for suffix, cluster, node, job, status in [
                ("match", "cluster-a", "node-a", "job-a", WorkflowStatus.RUNNING),
                ("node-b", "cluster-a", "node-b", "job-a", WorkflowStatus.PENDING),
                ("terminal", "cluster-a", "node-a", "job-a", WorkflowStatus.SUCCEEDED),
                (
                    "other-cluster",
                    "cluster-b",
                    "node-a",
                    "job-a",
                    WorkflowStatus.RUNNING,
                ),
            ]:
                incident, workflow = _builder(
                    f"event-{index}-{suffix}", f"{index}-{suffix}"
                )()
                incident = incident.model_copy(
                    update={"cluster_id": cluster, "node_ids": [node], "job_id": job}
                )
                workflow = workflow.model_copy(update={"status": status})
                store.save_incident(incident)
                store.save_workflow(workflow)

            matches = store.list_active_workflow_incidents(
                "cluster-a", node_ids={"node-a"}, job_id="job-a"
            )

            assert [item[0].incident_id for item in matches] == [
                f"incident-{index}-match"
            ]
    finally:
        stores[1].close()
