"""Planner discipline for the processor's hot transactions.

During a 969 req/s burst the queue's heap was read 580k-906k blocks a second
while it held 1,215-5,796 pages: 300-600 full heap scans a second, next to
~190 flush transactions inserting into the same pages, and the queue INSERT
spent 90% of its active time in ``LWLock:BufferContent`` (the completion CTE
71%). ``pg_class`` said ``reltuples 138 / relpages 45``: the planner saw a tiny
table and chose plans that read the whole backlog -- a heap scan of every
eligible row followed by a sort for the claim window, sequential scans for the
correlated ``EXISTS`` probes (retry barrier, strict predecessor,
fault/observation interlock) and for the completion's row locking -- and kept
them while the burst grew the heap, because autoanalyze is asynchronous and a
burst lasts about a minute. ``gpu_fault_processor_queue_counts`` was summed by
sequential scan ~300 times a second for the same reason.

Every hot transaction (claim, lease renewal/release/reclaim, completion,
admission) therefore starts with ``SET LOCAL enable_seqscan = off``, scoped to
that transaction. The claim window's ``ORDER BY (priority, created_at,
request_id) LIMIT window`` is exactly what ``gpu_fault_processor_queue_
priority_claim`` / ``_path_priority_claim`` store, so once the heap scan is
priced out the planner walks the index in claim order and stops at the window;
the lane probes walk ``gpu_fault_processor_queue_lane``, the depth reads the
counts primary key. Measured on 5,000 PENDING rows
(``tests/store/test_postgres_claim_planner_discipline.py``, numbers recorded as
test properties): the path-filtered sub-window -- the shape 7 of the 8
sub-claims per cycle use -- read 2,323 shared blocks, touched all 5,000 rows
and ran 24 ms by default, under stale and fresh statistics alike; with the
setting it read 769 blocks, touched the 72-row window and ran 4 ms. The
unfiltered and excluded-path windows already walked the index (772 -> 769
blocks, 4.5 -> 4.0 ms), and a two-row table went from 3.9 to 1.0 ms.

``enable_sort = off`` was measured and rejected: it also yields the ordered
walk, but the claim statement then executes in 1.3-1.7 s on a two-row table
and on 5,000 rows alike (its remaining sorts -- the ordered lane insert, the
``DISTINCT ON`` and the final ordering -- are driven onto pathological
alternatives), where the default takes a few milliseconds. The setting here is
a cost penalty, not a prohibition: where no index can serve a statement the
planner still scans, so nothing fails.

``plan_cache_mode = force_custom_plan`` was evaluated and not adopted: psycopg
prepares a statement after five executions, and a generic plan built under
``enable_seqscan = off`` cannot regress to a heap scan; the only parameters
whose values vary are time bounds, limits and the owner, which do not change
the shape of the plan. Forcing custom plans would re-plan the claim's
multi-CTE statement on every one of the several hundred rounds a second the
worker tier issues. Should a generic plan ever prove worse, the tuple below is
the one place to add it.
"""

from __future__ import annotations

from typing import Any

#: ``(setting, value)`` pairs applied with ``SET LOCAL`` at the start of every
#: claim, lease renewal/release/reclaim, completion and admission transaction.
HOT_PATH_PLANNER_SETTINGS: tuple[tuple[str, str], ...] = (("enable_seqscan", "off"),)


def hot_path_planner_statements() -> tuple[str, ...]:
    """The exact statements ``pin_hot_path_plans`` issues (tests pin them)."""

    return tuple(
        f"SET LOCAL {name} = {value}" for name, value in HOT_PATH_PLANNER_SETTINGS
    )


def pin_hot_path_plans(db: Any) -> None:
    """Apply the hot-path planner settings to the transaction ``db`` is in.

    ``SET LOCAL`` lasts until the surrounding transaction commits or rolls
    back, so the caller must already be inside ``db.transaction()``; ``db`` is
    anything with a ``cursor()`` context manager (the store's pool handle or a
    psycopg connection).
    """

    with db.cursor() as cursor:
        for statement in hot_path_planner_statements():
            cursor.execute(statement)
