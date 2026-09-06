"""The dispatcher owns the process-lifetime ``filtered`` totals (F-L1, §69).

Each ``run_once`` reports the rows it set aside by reason; the sum over the
process used to be kept on the application context by the dispatch thread,
so a dispatcher driven any other way exported nothing. The dispatcher now
accumulates its own ``filtered_total`` and /metrics reads it from there.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import build_store, fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


class _RecordingExecutor:
    def __init__(self) -> None:
        self.requested: list[str] = []
        self.config = SimpleNamespace(executor_id="executor-recording")

    def execute(self, request_id, request):
        self.requested.append(request_id)
        raise WorkflowLeaseError("recorded only")


def _pending(store, request_id: str, created_at: datetime) -> None:
    incident_id = f"inc-{request_id}"
    store.save_incident_and_workflow(
        fault_incident(
            incident_id,
            f"event-{request_id}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=request_id,
            fencing_token=1,
            created_at=created_at,
            updated_at=created_at,
        ),
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.PENDING,
            fencing_token=1,
            official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
            created_at=created_at,
            updated_at=created_at,
        ),
    )


def test_run_once_accumulates_filtered_reasons_across_ticks() -> None:
    store = build_store()
    for index in range(3):
        _pending(store, f"wf-{index}", NOW - timedelta(hours=3 - index))
    dispatcher = WorkflowDispatcher(
        store,
        _RecordingExecutor(),  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, max_workers=1, batch_size=1),
    )

    assert dispatcher.filtered_total == {}, "nothing is counted before a tick"

    first = dispatcher.run_once()
    second = dispatcher.run_once()

    assert first.filtered["batch_limit"] == 2
    assert second.filtered["batch_limit"] == 2
    assert dispatcher.filtered_total["batch_limit"] == 4, (
        "the total is the sum over every tick, not the last report"
    )
