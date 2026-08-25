"""Real-Postgres coverage for the claim, lane and remote command SQL.

``test_processor_leadership.py`` parametrizes only the memory and sqlite
stores, so the PostgresStore overrides of these paths - which are the ones
that actually run in regional mode - had no test at all. Everything here
asserts a contract that lives in SQL: the bounded claim window still
honours priority, lane exclusivity and the observation/fault interlock;
lane retention only retires idle lanes; and the batched remote command
claim keeps the fencing rules of the per-command version it replaced.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import Barrier, Thread

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import (
    COMPLETION_MARKER,
    POSTGRES_URL,
    REQUEST_LEASE,
    _claim_completions,
    _command,
    _counter_depth,
    _fault,
    _recording_cursor,
    _reload,
    _request,
    _telemetry,
    _workflow_state,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


def test_partitioned_priority_shards_have_a_total_lock_order(store) -> None:
    import psycopg

    store.finalize_processor_counter_shards()
    scopes = [
        f"cluster-{cluster}"
        for cluster in ("a", "b", "c")
        for _priority in (0, 50, 100)
        for _shard in range(16)
    ]
    priorities = [
        priority
        for _cluster in ("a", "b", "c")
        for priority in (0, 50, 100)
        for _shard in range(16)
    ]
    shards = [
        shard
        for _cluster in ("a", "b", "c")
        for _priority in (0, 50, 100)
        for shard in range(16)
    ]
    with store._db.transaction():
        with store._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT gpu_fault_processor_priority_count_apply(
                    %s, %s, %s, %s
                )
                """,
                (scopes, priorities, shards, [10_000] * len(scopes)),
            )

    barrier = Barrier(8)
    failures: list[BaseException] = []

    def update(worker: int) -> None:
        connection = psycopg.connect(POSTGRES_URL, autocommit=False)
        try:
            barrier.wait(timeout=10)
            for iteration in range(40):
                indexes = [
                    index
                    for index in range(len(scopes))
                    if (index + worker + iteration) % 5 != 0
                ]
                if (worker + iteration) % 2:
                    indexes.reverse()
                with connection.transaction():
                    connection.execute(
                        """
                        SELECT
                            gpu_fault_processor_priority_count_apply(
                                %s, %s, %s, %s
                            )
                        """,
                        (
                            [scopes[index] for index in indexes],
                            [priorities[index] for index in indexes],
                            [shards[index] for index in indexes],
                            [1 if (index + iteration) % 2 else -1 for index in indexes],
                        ),
                    )
        except BaseException as exc:
            failures.append(exc)
        finally:
            connection.close()

    threads = [Thread(target=update, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []


def test_counter_shard_finalize_requires_an_empty_queue(store) -> None:
    store.enqueue_processor_request(_fault("node-active"))

    with pytest.raises(RuntimeError, match="requires an empty queue"):
        store.finalize_processor_counter_shards()


def test_multi_cluster_batches_do_not_deadlock_on_the_counters(store) -> None:
    """Concurrent batches spanning the same clusters must not deadlock.

    The statement trigger carries one delta per cluster per statement
    and applies them with a set-based UPDATE, which locks the counter
    rows in whatever order the join produces. Two writers whose batches
    cover the same clusters then take the same two row locks in
    opposite orders, and Postgres aborts one of them - during the
    32-cluster burst that killed roughly three completion batches out
    of four and stalled the whole ingest path. The counter function has
    to claim its rows in cluster_id order first.

    The filler rows below are what makes this reproduce: on a table
    small enough to seq-scan, the join hashes the delta array against
    one physical scan of the counters and every transaction ends up
    taking the locks in the same order, so the pre-fix function looks
    innocent. A regional deployment has thousands of counter rows, the
    planner switches to a nested loop that follows the array, and the
    order becomes per-statement. Verified: with the ordered claim
    removed from gpu_fault_processor_queue_count_apply this body raises
    DeadlockDetected, and it passes with it.
    """
    import random
    import threading

    import psycopg

    clusters = [f"cluster-{index:02d}" for index in range(16)]
    failures: list[BaseException] = []
    workers = 8
    rounds = 40
    barrier = threading.Barrier(workers)

    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_processor_queue_counts(
                    cluster_id, incomplete_count, updated_at
                )
                SELECT 'filler-' || generated, 0, now()
                FROM generate_series(1, 20000) AS generated
                ON CONFLICT (cluster_id) DO NOTHING
                """
            )
            cursor.execute("ANALYZE gpu_fault_processor_queue_counts")

    def writer(worker: int) -> None:
        instance = PostgresStore(POSTGRES_URL)
        try:
            barrier.wait(timeout=30)
            for round_index in range(rounds):
                # A different permutation per statement, so the lock
                # order of two overlapping batches differs.
                scope = list(clusters)
                random.Random(worker * 1000 + round_index).shuffle(scope)
                batch = [
                    _telemetry(
                        f"node-{worker}-{round_index}-{index}", cluster_id=cluster
                    )
                    for index, cluster in enumerate(scope)
                ]
                instance.try_enqueue_processor_requests_batch(
                    batch, max_depth=100_000, max_cluster_depth=100_000
                )
                claimed = instance.claim_active_processor_requests(
                    f"pod-{worker}",
                    now=datetime.now(timezone.utc),
                    lease_duration=REQUEST_LEASE,
                    limit=len(scope),
                )
                for entry in claimed:
                    instance.complete_active_processor_request(
                        entry.request_id,
                        f"pod-{worker}",
                        entry.leader_epoch,
                        entry.lease_token,
                        response_status=200,
                        response_content_type="application/json",
                        response_body_base64="e30=",
                    )
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)
        finally:
            instance.close()

    threads = [
        threading.Thread(target=writer, args=(worker,)) for worker in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)

    assert not [
        failure for failure in failures if "deadlock" in str(failure).lower()
    ], failures
    assert not failures, failures
    assert store.processor_queue_count_status()["mismatched_clusters"] == 0


def test_remote_commands_are_leased_in_one_batch(store) -> None:
    """One transaction, and still FIFO by created_at.

    The inherited implementation opened a transaction per candidate; the
    override leases the whole batch at once, so the ordering and the
    per-command lease token both have to survive the collapse.
    """
    commands = [
        _command(store, f"remote-{index}", request_id=f"workflow-{index}")
        for index in range(5)
    ]

    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=3, lease_seconds=60
    )

    assert [item.command_id for item in claimed] == [
        command.command_id for command in commands[:3]
    ]
    assert all(item.status is RemoteCommandStatus.LEASED for item in claimed)
    assert len({item.lease_token for item in claimed}) == 3
    # The leases are visible to the next claimer, which must move on to
    # the untouched commands rather than re-hand out the leased ones.
    rest = store.claim_remote_commands(
        "cluster-a", "executor-b", limit=5, lease_seconds=60
    )
    assert [item.command_id for item in rest] == [
        command.command_id for command in commands[3:]
    ]


def test_remote_claim_skips_stale_fencing_and_other_clusters(store) -> None:
    stale = _command(store, "remote-stale", request_id="workflow-stale", token=3)
    _workflow_state(store, request_id="workflow-stale", fencing_token=4)
    fresh = _command(store, "remote-fresh", request_id="workflow-fresh", token=3)

    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=10, lease_seconds=60
    )

    assert [item.command_id for item in claimed] == [fresh.command_id]
    assert stale.command_id not in {item.command_id for item in claimed}
    assert (
        store.claim_remote_commands(
            "cluster-b", "executor-a", limit=10, lease_seconds=60
        )
        == []
    )


def test_remote_claim_honours_advertised_owners(store) -> None:
    command = _command(store, "remote-owner", request_id="workflow-owner")

    assert (
        store.claim_remote_commands(
            "cluster-a",
            "executor-a",
            limit=10,
            lease_seconds=60,
            execution_owners={"someone-else"},
        )
        == []
    )
    claimed = store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=10,
        lease_seconds=60,
        execution_owners={"owner-a"},
    )
    assert [item.command_id for item in claimed] == [command.command_id]


def test_expired_remote_lease_is_reclaimed(store) -> None:
    command = _command(store, "remote-expire", request_id="workflow-expire")
    first = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=-5
    )
    assert [item.command_id for item in first] == [command.command_id]

    second = store.claim_remote_commands(
        "cluster-a", "executor-b", limit=1, lease_seconds=60
    )

    assert [item.command_id for item in second] == [command.command_id]
    assert second[0].lease_owner == "executor-b"
    assert second[0].lease_token != first[0].lease_token


def test_workflow_cancellation_only_touches_its_own_commands(store) -> None:
    # Leased first, and on its own, so the claim below can only pick it:
    # the batch claim is FIFO by created_at.
    leased = _command(store, "remote-leased", request_id="workflow-leased")
    store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=600, execution_owners=None
    )
    mine = _command(store, "remote-mine", request_id="workflow-mine")
    other = _command(store, "remote-other", request_id="workflow-other")

    result = store.cancel_remote_commands_for_workflow(
        mine.workflow_request_id, reason="workflow timed out"
    )

    assert result == {"cancelled": 1, "cancellation_requested": 0}
    assert _reload(store, mine.command_id).status is RemoteCommandStatus.FAILED
    assert _reload(store, other.command_id).status is RemoteCommandStatus.PENDING
    leased_now = _reload(store, leased.command_id)
    assert leased_now.status is RemoteCommandStatus.LEASED
    assert leased_now.cancellation_requested_at is None


def test_leased_command_cancellation_is_requested_not_forced(store) -> None:
    command = _command(store, "remote-running", request_id="workflow-running")
    store.claim_remote_commands("cluster-a", "executor-a", limit=1, lease_seconds=600)

    result = store.cancel_remote_commands_for_workflow(
        command.workflow_request_id, reason="workflow timed out"
    )

    assert result == {"cancelled": 0, "cancellation_requested": 1}
    updated = _reload(store, command.command_id)
    assert updated.status is RemoteCommandStatus.LEASED
    assert updated.cancellation_reason == "workflow timed out"


def test_admission_group_is_one_insert_statement(store) -> None:
    """A group is written by one statement, not one per request.

    The count trigger is ``FOR EACH STATEMENT``, so this is what decides
    how long the cluster's counter row stays locked - a per-row loop held
    it across sixty-four inserts, and on a 50-cluster burst that convoy
    put ~200 backends on this table's tuple locks with the longest waiter
    at 43s, at 6% database CPU.
    """
    statements: list[str] = []
    original = store._db.cursor

    class _Recording:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, query, parameters=None):
            if "INTO gpu_fault_processor_queue " in query:
                statements.append(query)
            return self._cursor.execute(query, parameters)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    import contextlib

    @contextlib.contextmanager
    def recording_cursor(*args, **kwargs):
        with original(*args, **kwargs) as cursor:
            yield _Recording(cursor)

    store._db.cursor = recording_cursor
    try:
        requests = [_telemetry(f"node-{index:03d}") for index in range(24)]
        results = store.try_enqueue_processor_requests_batch(
            requests, max_depth=65536, max_cluster_depth=1024
        )
    finally:
        store._db.cursor = original

    assert all(result[1] is None for result in results)
    assert len(statements) == 1
    assert _counter_depth("cluster-a") == 24
    assert store.processor_queue_stats()["depth"] == 24, (
        "counter table and the depth gauge must still agree"
    )


def test_admission_group_rejects_duplicate_request_ids(store) -> None:
    """ON CONFLICT DO UPDATE cannot touch one key twice in a statement.

    The caller keys requests by base id, so this is unreachable through
    the API; asserting it here keeps a future caller from turning it into
    a runtime ``cardinality violation`` under load.
    """
    request = _telemetry("node-000")
    with pytest.raises(ValueError, match="duplicate request ids"):
        store._put_processor_queue_many([request, request])


def test_concurrent_admissions_to_one_cluster_do_not_serialize(store) -> None:
    """Two groups for the same cluster must not queue behind each other.

    The depth read used to be ``FOR UPDATE``, so the second group waited
    out everything the first one still had to do: its decision loop, its
    row writes and its commit. Here the first group is stalled between
    the depth read and the write - exactly the window that used to be
    inside the lock - and the second group must get through anyway.

    The write itself does take the counter row (the count trigger's own
    ``FOR UPDATE``), and that is fine: it is now the last thing the
    transaction does before it commits.
    """
    import threading

    other = PostgresStore(POSTGRES_URL)
    started = threading.Event()
    release = threading.Event()
    # Create the counter row in a committed transaction first: the
    # ON CONFLICT DO NOTHING that seeds it would otherwise leave an
    # uncommitted row that the second group has to wait for, which is a
    # once-per-cluster cost and not what this test is about.
    store.enqueue_processor_request(_telemetry("node-warm"))
    try:
        inner = store._processor_queue_row

        def stalling(request):
            started.set()
            assert release.wait(30)
            return inner(request)

        store._processor_queue_row = stalling
        slow = threading.Thread(
            target=store.try_enqueue_processor_requests_batch,
            args=([_telemetry("node-slow")],),
            kwargs={"max_depth": 65536, "max_cluster_depth": 1024},
            daemon=True,
        )
        slow.start()
        assert started.wait(30), "first group never reached its write"

        began = time.monotonic()
        results = other.try_enqueue_processor_requests_batch(
            [_telemetry("node-fast")], max_depth=65536, max_cluster_depth=1024
        )
        elapsed = time.monotonic() - began

        assert results[0][1] is None
        assert elapsed < 5, (
            "second group waited for the first group's transaction; "
            "the counter row is being locked across application work"
        )
    finally:
        release.set()
        slow.join(timeout=30)
        store._processor_queue_row = inner
        other.close()


def test_cluster_cap_stays_exact_at_the_boundary(store) -> None:
    """The unlocked depth read must not let a cluster exceed its cap.

    Within the guard band the code re-reads the counters ``FOR UPDATE``,
    so the cap is still enforced row-exactly - that band is what keeps
    telemetry out of the depth reserved for faults.
    """
    for index in range(8):
        accepted = store.try_enqueue_processor_requests_batch(
            [_telemetry(f"node-{index:03d}")],
            max_depth=65536,
            max_cluster_depth=6,
            reserved_cluster_fault_depth=2,
            global_admission_guard=4,
        )
        if index < 4:
            assert accepted[0][1] is None, index
        else:
            assert accepted[0][1] == "cluster_reserved", index

    assert _counter_depth("cluster-a") == 4

    fault = store.try_enqueue_processor_requests_batch(
        [
            _request(
                "/v1/collector-events/nvidia-kernel",
                body=b'{"node_id":"node-fault","lines":[]}',
            )
        ],
        max_depth=65536,
        max_cluster_depth=6,
        reserved_cluster_fault_depth=2,
        global_admission_guard=4,
    )
    assert fault[0][1] is None, "the reserve must still admit faults"


def test_concurrent_fault_admissions_do_not_serialize(store) -> None:
    """Faults do not go through the batcher, so they need the same fix.

    ``queue_priority() == 100`` (the three telemetry paths) is what the
    admission batcher takes; faults are priority 0 and go one at a time
    through ``try_enqueue_processor_request``. That path used to read the
    cluster's counter row ``FOR UPDATE`` and hold it through the persist
    and the commit, so concurrent faults for one cluster - and the
    completion batches whose trigger wants the same rows - formed a
    convoy: 78 backends stalled on that one statement during a
    50-cluster burst, waiting up to 48.7s.
    """
    import threading

    other = PostgresStore(POSTGRES_URL)
    started = threading.Event()
    release = threading.Event()
    # Committed counter row first: the once-per-cluster seeding insert is
    # a different (and legitimate) wait.
    store.enqueue_processor_request(_telemetry("node-warm"))
    try:
        inner = store._put_processor_queue

        def stalling(request, **kwargs):
            started.set()
            assert release.wait(30)
            return inner(request, **kwargs)

        store._put_processor_queue = stalling
        slow = threading.Thread(
            target=store.try_enqueue_processor_request,
            args=(_fault("node-slow"),),
            kwargs={"max_depth": 65536, "max_cluster_depth": 1024},
            daemon=True,
        )
        slow.start()
        assert started.wait(30), "the first fault never reached its write"

        began = time.monotonic()
        _, reason = other.try_enqueue_processor_request(
            _fault("node-fast"), max_depth=65536, max_cluster_depth=1024
        )
        elapsed = time.monotonic() - began

        assert reason is None
        assert elapsed < 5, (
            "the second fault waited for the first one's transaction; "
            "the counter row is being locked across application work"
        )
    finally:
        release.set()
        slow.join(timeout=30)
        store._put_processor_queue = inner
        other.close()


def test_fault_path_cluster_cap_stays_exact_at_the_boundary(store) -> None:
    """Dropping the lock must not let the single-request path overshoot.

    Inside the guard band the depth is re-read ``FOR UPDATE``, so the
    cluster cap and the fault reserve are still row-exact.
    """
    for index in range(8):
        _, reason = store.try_enqueue_processor_request(
            _telemetry(f"node-{index:03d}"),
            max_depth=65536,
            max_cluster_depth=6,
            reserved_cluster_fault_depth=2,
            global_admission_guard=4,
        )
        assert reason == (None if index < 4 else "cluster_reserved"), (index, reason)

    assert _counter_depth("cluster-a") == 4

    for index in range(2):
        _, reason = store.try_enqueue_processor_request(
            _fault(f"node-fault-{index}"),
            max_depth=65536,
            max_cluster_depth=6,
            reserved_cluster_fault_depth=2,
            global_admission_guard=4,
        )
        assert reason is None, "the reserve must still admit faults"

    _, reason = store.try_enqueue_processor_request(
        _fault("node-fault-over"),
        max_depth=65536,
        max_cluster_depth=6,
        reserved_cluster_fault_depth=2,
        global_admission_guard=4,
    )
    assert reason == "cluster", "the cluster cap must stay exact"
    assert _counter_depth("cluster-a") == 6


def test_completion_batch_is_one_statement_per_cluster(store) -> None:
    """A flush is split by cluster, because the trigger locks by cluster.

    A claim window is filled from the whole region, so one flush of 64
    carries work for as many clusters as it found, and the statement
    trigger takes the counter row of every one of them until the
    transaction commits. Two flushes that overlap on one cluster then
    serialize on all of them - a 50-cluster burst drained at ~57 rows/s
    with 115 backends blocked on this table and the longest waiter at
    40.1s, while Aurora sat at 3.8% CPU and 28.5 ACU.
    """
    clusters = [f"cluster-{index:02d}" for index in range(4)]
    for cluster in clusters:
        for index in range(3):
            store.enqueue_processor_request(
                _telemetry(f"node-{cluster}-{index}", cluster_id=cluster)
            )

    claimed, completions = _claim_completions(store, "pod-a", limit=12)
    assert len(claimed) == 12

    statements: list[str] = []
    original = _recording_cursor(store, COMPLETION_MARKER, statements)
    try:
        results = store.complete_active_processor_requests_batch(completions)
    finally:
        store._db.cursor = original

    assert len(statements) == len(clusters)
    assert all(result is not None for result in results)
    assert all(result.status is ProcessorRequestStatus.COMPLETED for result in results)
    # The split must not lose a decrement: the counter table is still the
    # oracle the depth gauge and the admission caps read.
    for cluster in clusters:
        assert _counter_depth(cluster) == 0, cluster
    assert store.processor_queue_count_status() == {
        "expected_total": 0,
        "counter_total": 0,
        "mismatched_clusters": 0,
        "ready": True,
    }


def test_single_cluster_completion_batch_stays_one_statement(store) -> None:
    """One cluster keeps the single-statement shape it always had.

    Every completion in a single-cluster deployment lands in one group,
    so the split must not turn a flush of 12 into 12 statements.
    """
    for index in range(12):
        store.enqueue_processor_request(_telemetry(f"node-{index:03d}"))
    _, completions = _claim_completions(store, "pod-a", limit=12)

    statements: list[str] = []
    original = _recording_cursor(store, COMPLETION_MARKER, statements)
    try:
        results = store.complete_active_processor_requests_batch(completions)
    finally:
        store._db.cursor = original

    assert len(statements) == 1
    assert len(results) == 12
    assert all(result is not None for result in results)


def test_completion_holds_one_cluster_counter_row_at_a_time(store) -> None:
    """The held counter-row footprint of a flush must be one row.

    This is the shape that produced the convoy: the completion statement
    locks a counter row per cluster in the batch and holds them all to
    the commit, so any other flush touching any of those clusters waits
    for this one to finish - and every flush in the fleet touches most
    of them. Stalling a flush after its first statement and probing the
    counter rows from another connection with ``FOR UPDATE NOWAIT``
    counts what is actually held: one row now, one per cluster before.

    Row locks are invisible in ``pg_locks`` until they are contended, so
    the probe has to try to take them.
    """
    import threading

    import psycopg

    clusters = [f"cluster-{index:02d}" for index in range(4)]
    for cluster in clusters:
        store.enqueue_processor_request(
            _telemetry(f"node-{cluster}", cluster_id=cluster)
        )
    _, completions = _claim_completions(store, "pod-a", limit=4)

    started = threading.Event()
    release = threading.Event()
    import contextlib

    original = store._db.cursor

    class _Stalling:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, query, parameters=None):
            result = self._cursor.execute(query, parameters)
            if COMPLETION_MARKER in query and not started.is_set():
                started.set()
                assert release.wait(30)
            return result

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    @contextlib.contextmanager
    def stalling(*args, **kwargs):
        with original(*args, **kwargs) as cursor:
            yield _Stalling(cursor)

    store._db.cursor = stalling
    flush = threading.Thread(
        target=store.complete_active_processor_requests_batch,
        args=(completions,),
        daemon=True,
    )
    try:
        flush.start()
        assert started.wait(30), "the flush never reached its statement"

        locked = []
        with psycopg.connect(POSTGRES_URL, autocommit=True) as probe:
            for cluster in clusters:
                try:
                    with probe.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT incomplete_count
                            FROM gpu_fault_processor_queue_counts
                            WHERE cluster_id=%s
                            FOR UPDATE NOWAIT
                            """,
                            (cluster,),
                        )
                        cursor.fetchall()
                except psycopg.errors.LockNotAvailable:
                    locked.append(cluster)
    finally:
        release.set()
        flush.join(timeout=30)
        store._db.cursor = original

    assert locked == [clusters[0]], (
        "an in-flight completion is holding the counter rows of "
        f"{len(locked)} clusters; every other flush covering any of "
        "them has to wait for this transaction to commit"
    )
    assert store.processor_queue_count_status()["mismatched_clusters"] == 0


def test_completion_batch_still_fences_each_request(store) -> None:
    """Splitting by cluster must not weaken the per-request fencing.

    The lease is validated inside the statement, and each group is now a
    statement of its own; a request whose lease was stolen still has to
    come back unfinished while its siblings - in the same group and in
    other groups - complete.
    """
    for cluster in ("cluster-a", "cluster-b"):
        for index in range(2):
            store.enqueue_processor_request(
                _telemetry(f"node-{cluster}-{index}", cluster_id=cluster)
            )
    claimed, completions = _claim_completions(store, "pod-a", limit=4)
    assert len(claimed) == 4

    stolen = completions[0]
    stolen["lease_token"] = "not-the-lease-token"

    results = store.complete_active_processor_requests_batch(completions)

    assert results[0] is None
    assert all(result is not None for result in results[1:])
    assert (
        store.get_processor_request(stolen["request_id"]).status
        is ProcessorRequestStatus.LEASED
    ), "a rejected completion must leave the request claimable again"
    for item in completions[1:]:
        assert (
            store.get_processor_request(item["request_id"]).status
            is ProcessorRequestStatus.COMPLETED
        )


def test_release_leaves_a_completed_request_completed(store) -> None:
    """Releasing a COMPLETED request must not reopen it.

    Completion does not clear ``lease_owner``/``leader_epoch``/
    ``lease_token``, so the fencing check alone accepts a release of a
    request that is already done and puts it back to PENDING - it is
    claimed again and its side effects run a second time. The callers do
    exactly that: a flush that comes back missing one request raises, and
    the handler then releases every request it had submitted, including
    the ones the same flush committed. Splitting the flush per cluster
    makes it partially applied, so this is now reachable.
    """
    store.enqueue_processor_request(_telemetry("node-a"))
    claimed, completions = _claim_completions(store, "pod-a", limit=1)
    entry = claimed[0]
    assert store.complete_active_processor_requests_batch(completions)[0] is not None

    store.release_active_processor_request(
        entry.request_id, "pod-a", entry.leader_epoch, entry.lease_token
    )

    stored = store.get_processor_request(entry.request_id)
    assert stored.status is ProcessorRequestStatus.COMPLETED
    assert stored.response_status == 200
    assert _counter_depth("cluster-a") == 0
    assert (
        store.claim_active_processor_requests(
            "pod-b",
            now=datetime.now(timezone.utc),
            lease_duration=REQUEST_LEASE,
            limit=4,
        )
        == []
    ), "the released request must not become claimable work again"
