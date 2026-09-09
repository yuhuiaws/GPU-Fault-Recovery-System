"""The historical source_gpu_count lookup reads a bounded, newest-first
slice of workflows.

When neither attempt observation lookup knows the GPU count, the guard
falls back to the step parameters of earlier RESTART_WORKLOAD steps for
the same attempt. That fallback listed every workflow the store held
(limit=10000) and walked every official step, inside the executor's
restart path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

_BOUND = 256


class RecordingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.list_workflow_calls: list[dict] = []

    def list_workflows(self, statuses=None, **kwargs):
        self.list_workflow_calls.append(dict(kwargs))
        return super().list_workflows(statuses, **kwargs)


class UnusedApi:
    pass


def _restart_workflow(
    store: InMemoryStore,
    request_id: str,
    *,
    attempt_id: str,
    source_gpu_count: int,
    updated_at: datetime,
) -> None:
    incident = fault_incident(f"incident-{request_id}", f"event-{request_id}")
    store.save_incident(incident)
    store.save_workflow(
        workflow_request(
            request_id,
            incident.incident_id,
            WorkflowStatus.SUCCEEDED,
            1,
            official_steps=[
                workflow_step(
                    WorkflowOperation.RESTART_WORKLOAD,
                    "gpu-fault-kubernetes-adapter",
                    workload_ids=["training/job/training-job"],
                    parameters={
                        "cluster_id": "cluster-a",
                        "job_id": "train-1",
                        "source_attempt_id": attempt_id,
                        "source_gpu_count": source_gpu_count,
                        "restart_budget": 1,
                    },
                )
            ],
            created_at=updated_at,
            updated_at=updated_at,
        )
    )


def _adapter(store: InMemoryStore) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )


def test_the_historical_count_is_found_within_a_bounded_newest_first_scan() -> None:
    store = RecordingStore()
    _restart_workflow(
        store,
        "workflow-unrelated",
        attempt_id="attempt-other",
        source_gpu_count=16,
        updated_at=NOW - timedelta(hours=2),
    )
    _restart_workflow(
        store,
        "workflow-earlier",
        attempt_id="attempt-x",
        source_gpu_count=8,
        updated_at=NOW - timedelta(hours=1),
    )

    count = _adapter(store)._observed_source_gpu_count(
        "cluster-a", "train-1", "attempt-x"
    )

    assert count == 8
    assert store.list_workflow_calls, "the fallback must consult the store"
    for call in store.list_workflow_calls:
        assert 0 < call.get("limit", 0) <= _BOUND, call
        assert call.get("newest_first") is True, call


def test_the_newest_matching_restart_step_wins() -> None:
    """The most recent record is the authoritative one; the older, larger
    value must not be preferred just because it is larger."""
    store = RecordingStore()
    _restart_workflow(
        store,
        "workflow-older",
        attempt_id="attempt-x",
        source_gpu_count=8,
        updated_at=NOW - timedelta(hours=2),
    )
    _restart_workflow(
        store,
        "workflow-newer",
        attempt_id="attempt-x",
        source_gpu_count=4,
        updated_at=NOW - timedelta(hours=1),
    )

    count = _adapter(store)._observed_source_gpu_count(
        "cluster-a", "train-1", "attempt-x"
    )

    assert count == 4


def test_no_history_still_reports_unknown() -> None:
    store = RecordingStore()
    _restart_workflow(
        store,
        "workflow-unrelated",
        attempt_id="attempt-other",
        source_gpu_count=16,
        updated_at=NOW - timedelta(hours=2),
    )

    count = _adapter(store)._observed_source_gpu_count(
        "cluster-a", "train-1", "attempt-x"
    )

    assert count == 0
