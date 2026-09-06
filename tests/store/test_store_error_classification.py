"""One SQLSTATE table, three named questions, and a typed stale-token error.

FINAL-建议汇总 F-J1 / F-J2 (P0-73B, P1-74C, P0-71A, P0-73A, P1-58E, P1-76E,
P0-58A). Three callers used to ask "is this store error transient?" with two
different classifiers that matched class names, so a serialization failure, a
deadlock or a statement timeout was "not transient" and the dispatcher wrote
the workflow BLOCKED. The callers ask three different questions:

* the pool: has this *connection* died? (discard it)
* ingress: is the *writer* unavailable? (answer 503, ask the caller to retry)
* dispatcher / processor: should this *operation* be retried? (never BLOCK)

and the answers nest: ``connection_is_lost ⊆ writer_unavailable ⊆
operation_should_retry``. Separately, a stale fencing token raised a bare
``ValueError`` in the store while every caller caught ``WorkflowLeaseError``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.transient_errors import transient_store_error
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.orchestrator import WorkflowFencingError
from gpu_fault.store import SqliteStore
from gpu_fault.store.postgres.pool import PooledPostgresDatabase
from gpu_fault.store.shared.errors import (
    StaleFencingTokenError,
    WorkflowLeaseError,
    connection_is_lost,
    operation_should_retry,
    writer_unavailable,
)
from tests._builders import (
    build_store,
    fault_incident,
    processor_request,
    workflow_request,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

psycopg = pytest.importorskip("psycopg")
psycopg_errors = pytest.importorskip("psycopg.errors")
psycopg_pool = pytest.importorskip("psycopg_pool")

NOW = datetime(2026, 9, 5, 16, 0, tzinfo=timezone.utc)


def _pg(sqlstate: str) -> Exception:
    return psycopg_errors.lookup(sqlstate)(f"synthetic {sqlstate}")


# (error, connection_is_lost, writer_unavailable, operation_should_retry)
CLASSIFICATION = [
    (_pg("40001"), False, False, True),  # serialization_failure
    (_pg("40P01"), False, False, True),  # deadlock_detected
    (_pg("57014"), False, False, True),  # query_canceled (statement_timeout)
    (_pg("55P03"), False, False, True),  # lock_not_available
    (_pg("53300"), False, True, True),  # too_many_connections
    (_pg("25006"), True, True, True),  # read_only_sql_transaction: demoted writer
    (_pg("57P01"), True, True, True),  # admin_shutdown
    (_pg("57P03"), True, True, True),  # cannot_connect_now
    (_pg("08006"), True, True, True),  # connection_failure
    (_pg("08003"), True, True, True),  # connection_does_not_exist
    (psycopg.OperationalError("socket closed"), True, True, True),
    (psycopg_pool.PoolTimeout("no connection within 2s"), False, True, True),
    (_pg("23505"), False, False, False),  # unique_violation
    (_pg("22P02"), False, False, False),  # invalid_text_representation
    (ValueError("not a database error"), False, False, False),
]


@pytest.mark.parametrize(
    ("error", "lost", "writer", "retry"),
    CLASSIFICATION,
    ids=[type(case[0]).__name__ for case in CLASSIFICATION],
)
def test_the_three_questions_have_the_table_answers(
    error, lost: bool, writer: bool, retry: bool
) -> None:
    assert connection_is_lost(error) is lost
    assert writer_unavailable(error) is writer
    assert operation_should_retry(error) is retry


@pytest.mark.parametrize("error", [case[0] for case in CLASSIFICATION])
def test_the_predicates_nest(error) -> None:
    if connection_is_lost(error):
        assert writer_unavailable(error), (
            "expected writer_unavailable(error) to be true"
        )
    if writer_unavailable(error):
        assert operation_should_retry(error), (
            "expected operation_should_retry(error) to be true"
        )


@pytest.mark.parametrize("error", [case[0] for case in CLASSIFICATION])
def test_the_dispatcher_classifier_is_the_shared_one(error) -> None:
    assert transient_store_error(error) is operation_should_retry(error)


def test_classification_follows_the_cause_chain() -> None:
    wrapped = RuntimeError("store call failed")
    wrapped.__cause__ = _pg("40001")

    assert operation_should_retry(wrapped), (
        "expected operation_should_retry(wrapped) to be true"
    )
    assert not writer_unavailable(wrapped), (
        "expected writer_unavailable(wrapped) to be false"
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "errors.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _seed_pending_workflow(store, *, token: int = 3) -> None:
    store.save_incident_and_workflow(
        fault_incident(
            "inc-fence",
            "event-fence",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="wf-fence",
            fencing_token=token,
            created_at=NOW,
            updated_at=NOW,
        ),
        workflow_request(
            "wf-fence",
            "inc-fence",
            status=WorkflowStatus.PENDING,
            fencing_token=token,
            created_at=NOW,
            updated_at=NOW,
        ),
    )


def test_claiming_with_a_stale_fencing_token_is_a_lease_error(store) -> None:
    _seed_pending_workflow(store, token=3)

    with pytest.raises(StaleFencingTokenError) as raised:
        store.claim_workflow(
            "wf-fence", "executor-a", 2, lease_duration=timedelta(seconds=30)
        )

    assert isinstance(raised.value, WorkflowLeaseError), (
        "expected isinstance(raised.value, WorkflowLeaseError) to be true"
    )
    assert store.get_workflow("wf-fence").execution_owner_id is None


def test_completing_a_processor_request_with_a_stale_lane_token_is_a_lease_error(
    store,
) -> None:
    request = store.enqueue_processor_request(
        processor_request(
            "/v1/gpu-events/xid", body=b'{"node_id":"node-a"}', cluster_id="c"
        )
    )
    claimed = store.claim_active_processor_requests(
        "worker-a",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=30),
        limit=1,
    )[0]

    with pytest.raises(StaleFencingTokenError) as raised:
        store.complete_active_processor_request(
            request.request_id,
            "worker-a",
            claimed.leader_epoch,
            "not-the-lease-token",
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )

    assert isinstance(raised.value, WorkflowLeaseError), (
        "expected isinstance(raised.value, WorkflowLeaseError) to be true"
    )


class _RaisingExecutor:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.config = SimpleNamespace(executor_id="executor-raising")
        self.calls = 0

    def execute(self, request_id, request):
        self.calls += 1
        raise self.error


def _dispatcher(store, error: Exception) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        _RaisingExecutor(error),  # type: ignore[arg-type]
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(WorkflowFencingError("stale fencing token"), id="fencing"),
        pytest.param(_pg("40001"), id="serialization_failure"),
        pytest.param(_pg("40P01"), id="deadlock"),
        pytest.param(_pg("57014"), id="statement_timeout"),
    ],
)
def test_dispatcher_leaves_the_workflow_pending_on_a_retryable_error(error) -> None:
    """Neither a fence nor a retryable store error is "this workflow is broken"."""

    store = build_store()
    _seed_pending_workflow(store)
    dispatcher = _dispatcher(store, error)

    report = dispatcher.run_once()

    current = store.get_workflow("wf-fence")
    assert current.status is WorkflowStatus.PENDING
    assert current.blocked_reasons == []
    assert store.get_incident("inc-fence").state is IncidentState.ACTION_PENDING
    assert (report.waiting, report.failed) == (1, 0)


def test_dispatcher_counts_a_genuine_internal_error_without_blocking() -> None:
    """A KeyError is neither a lease/fencing signal nor a transient store
    error, so it is not swallowed as "waiting" -- but under F-B4 (3) it no
    longer writes the record BLOCKED either: the row is released with a
    backoff and the tick reports it."""

    store = build_store()
    _seed_pending_workflow(store)
    dispatcher = _dispatcher(store, KeyError("missing adapter parameter"))

    report = dispatcher.run_once()

    current = store.get_workflow("wf-fence")
    assert current.status is WorkflowStatus.PENDING
    assert current.not_before is not None
    assert report.internal_errors == 1
    assert report.waiting == 0
    assert [failure.workflow_request_id for failure in report.failures] == ["wf-fence"]


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakePool:
    def __init__(self) -> None:
        self.connections: list[_FakeConnection] = []

    @contextmanager
    def connection(self):
        connection = _FakeConnection()
        self.connections.append(connection)
        yield connection


@pytest.mark.parametrize(
    ("error", "discarded"),
    [
        pytest.param(_pg("40001"), False, id="serialization_failure-kept"),
        pytest.param(_pg("57014"), False, id="statement_timeout-kept"),
        pytest.param(_pg("40P01"), False, id="deadlock-kept"),
        pytest.param(_pg("08006"), True, id="connection_failure-discarded"),
        pytest.param(
            psycopg.OperationalError("gone"), True, id="operational-discarded"
        ),
    ],
)
def test_pool_discards_a_connection_only_when_it_is_lost(error, discarded) -> None:
    pool = _FakePool()
    database = PooledPostgresDatabase(pool)

    with pytest.raises(type(error)):
        with database._connection():
            raise error

    assert pool.connections[0].closed is discarded
