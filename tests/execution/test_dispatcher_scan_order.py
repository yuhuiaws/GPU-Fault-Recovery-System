"""The dispatcher walks its horizon in pages, in eligibility order, and holds
a predecessor whose preempting successor is already on its way.

FINAL-建议汇总 F-A2 (a)(c) (P1-73D, P2-59F, P2-39F) and F-C1. The scan used to
re-read the oldest rows with a four-times-larger LIMIT each time it came up
short; it now passes the last row of a page back as ``after`` and reads the
next page. The order is ``dispatch_eligible_at`` (max of ``created_at`` and
``not_before``), which no merge rewrites, and the batch cut follows it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import (
    build_store,
    fault_incident,
    processor_request,
    workflow_request,
    workflow_step,
)

# Yesterday, so every seeded not_before lies in the past of the real clock the
# dispatcher reads.
NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
OP = WorkflowOperation.FREEZE_EVIDENCE


class _PagingStore:
    """Records the window of every dispatch scan without changing its answer."""

    def __init__(self, store) -> None:
        self._store = store
        self.pages: list[tuple[int, str | None]] = []

    def list_workflows(self, *args, **kwargs):
        if kwargs.get("dispatchable_at") is not None:
            after = kwargs.get("after")
            self.pages.append(
                (kwargs["limit"], None if after is None else after.request_id)
            )
        return self._store.list_workflows(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


class _RecordingExecutor:
    def __init__(self) -> None:
        self.requested: list[str] = []
        self.config = SimpleNamespace(executor_id="executor-recording")

    def execute(self, request_id, request):
        self.requested.append(request_id)
        raise WorkflowLeaseError("recorded only")


def _dispatcher(store, executor, **config) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        executor,  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, max_workers=1, **config),
    )


def _pending(store, request_id: str, **values):
    incident_id = f"inc-{request_id}"
    created_at = values.pop("created_at", NOW)
    incident = fault_incident(
        incident_id,
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        fencing_token=1,
        created_at=created_at,
        updated_at=created_at,
    )
    workflow = workflow_request(
        request_id,
        incident_id,
        status=values.pop("status", WorkflowStatus.PENDING),
        fencing_token=1,
        official_steps=[workflow_step(OP)],
        created_at=created_at,
        updated_at=values.pop("updated_at", created_at),
        **values,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


def _block_node_a_in_the_processor_queue(store) -> None:
    """One pending fault request on node-a keeps every open aggregation window
    on that node closed (the ``processor_queue`` gate, Python-side)."""

    store.enqueue_processor_request(
        processor_request(
            "/v1/gpu-events/xid", body=b'{"cluster_id":"cluster-a","node_id":"node-a"}'
        )
    )


# ---------------------------------------------------------------- F-A2 (c)


def test_scan_walks_the_horizon_in_pages_with_a_cursor() -> None:
    """250 rows a Python-side gate (an open aggregation window with the node's
    fault still in the processor queue) holds back, then the one eligible
    row: the scan reaches it in three pages of the same size, never
    re-reading."""

    inner = build_store()
    _block_node_a_in_the_processor_queue(inner)
    _pending(inner, "wf-successor", status=WorkflowStatus.RUNNING)
    for index in range(250):
        _pending(
            inner,
            f"wf-held-{index:03d}",
            aggregation_max_deadline=datetime.now(timezone.utc) + timedelta(hours=1),
            created_at=NOW - timedelta(days=2),
        )
    _pending(inner, "wf-eligible")
    store = _PagingStore(inner)
    executor = _RecordingExecutor()
    dispatcher = _dispatcher(store, executor, batch_size=5)

    report = dispatcher.run_once()

    assert "wf-eligible" in executor.requested
    assert report.filtered["processor_queue"] == 250
    assert report.horizon_exhausted is False, "251 rows are well inside the ceiling"
    limits = [limit for limit, _ in store.pages]
    cursors = [after for _, after in store.pages]
    assert limits == [100, 100, 100], "one page size, three pages"
    assert cursors[0] is None
    assert cursors[1] == "wf-held-099", "page two starts after the last row of page one"
    assert cursors[2] == "wf-held-199"


def test_the_horizon_ceiling_is_a_total_of_rows_walked_not_one_window() -> None:
    inner = build_store()
    _block_node_a_in_the_processor_queue(inner)
    _pending(inner, "wf-successor", status=WorkflowStatus.RUNNING)
    for index in range(120):
        _pending(
            inner,
            f"wf-held-{index:03d}",
            aggregation_max_deadline=datetime.now(timezone.utc) + timedelta(hours=1),
            created_at=NOW - timedelta(days=2),
        )
    store = _PagingStore(inner)
    dispatcher = _dispatcher(store, _RecordingExecutor(), batch_size=5)
    dispatcher.MAX_SCAN_ROWS = 100

    report = dispatcher.run_once()

    assert report.horizon_exhausted is True
    assert len(store.pages) == 1, "the ceiling stops the walk after the first page"


# ---------------------------------------------------------------- F-A2 (a)


def test_the_batch_is_cut_in_eligibility_order_not_merge_order() -> None:
    """The row merged into most recently is the *oldest* fault; under the
    ``updated_at`` key it sorted last and lost the batch cut every tick."""

    store = build_store()
    _pending(store, "wf-merged", created_at=NOW - timedelta(hours=3), updated_at=NOW)
    _pending(store, "wf-quiet", created_at=NOW - timedelta(hours=1))
    _pending(
        store,
        "wf-deferred",
        created_at=NOW - timedelta(hours=5),
        not_before=NOW - timedelta(minutes=30),
    )
    executor = _RecordingExecutor()
    dispatcher = _dispatcher(store, executor, batch_size=1)

    report = dispatcher.run_once()

    assert executor.requested == ["wf-merged"]
    assert report.filtered["batch_limit"] == 2, (
        "rows past the batch cut are reported, not silently dropped"
    )
    assert report.scanned == 1


# ---------------------------------------------------------------- F-C1


def test_a_stamped_pending_predecessor_is_observed_not_held() -> None:
    """The merge stamps the predecessor; the executor supersedes it at its
    first step. Holding it here parked the successor too, whose predecessor
    gate waits for this row to end."""

    store = build_store()
    _pending(store, "wf-old", preemption_pending_by_workflow_id="wf-new")
    _pending(store, "wf-new", status=WorkflowStatus.RUNNING)
    executor = _RecordingExecutor()
    dispatcher = _dispatcher(store, executor, batch_size=10)

    report = dispatcher.run_once()

    assert "wf-old" in executor.requested, "the predecessor still reaches its boundary"
    assert "preemption_pending" not in report.filtered
    assert dispatcher.preemption_pending_seen_total == 1


@pytest.mark.parametrize("successor_status", [WorkflowStatus.SUPERSEDED, None])
def test_a_stale_stamp_is_not_counted(successor_status) -> None:
    store = build_store()
    _pending(store, "wf-old", preemption_pending_by_workflow_id="wf-new")
    if successor_status is not None:
        _pending(store, "wf-new", status=successor_status)
    executor = _RecordingExecutor()
    dispatcher = _dispatcher(store, executor, batch_size=10)

    dispatcher.run_once()

    assert "wf-old" in executor.requested
    assert dispatcher.preemption_pending_seen_total == 0
