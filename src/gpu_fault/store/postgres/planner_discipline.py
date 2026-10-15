"""Planner discipline for the processor's claim transactions.

During a 969 req/s burst the queue's heap was read 580k-906k blocks a second
while it held 1,215-5,796 pages: 300-600 full heap scans a second, next to
~190 flush transactions inserting into the same pages, and the queue INSERT
spent 90% of its active time in ``LWLock:BufferContent`` (the completion CTE
71%). ``pg_class`` said ``reltuples 138 / relpages 45``: the planner saw a tiny
table and chose plans that read the whole backlog for the claim window -- a
heap scan of every eligible row followed by a sort -- and kept them while the
burst grew the heap, because autoanalyze is asynchronous and a burst lasts
about a minute.

The claim transactions (``claim_active_processor_requests`` and the legacy
``claim_processor_requests``) therefore start with ``SET LOCAL enable_seqscan
= off``, scoped to that transaction. The window's ``ORDER BY (priority,
created_at, request_id) LIMIT window`` is exactly what ``gpu_fault_processor_
queue_priority_claim`` / ``_path_priority_claim`` store, so once the heap scan
is priced out the planner walks the index in claim order and stops at the
window, and the lane probes walk ``gpu_fault_processor_queue_lane``. Measured
on 5,000 PENDING rows with the planner's statistics fixed by the test
(``tests/store/test_postgres_claim_planner_discipline.py``, numbers recorded as
test properties): with the burst's statistics -- 138 rows analysed, then 4,862
more the planner knows nothing about -- the path-filtered sub-window (the
shape 7 of the 8 sub-claims per cycle use) read 2,316 shared blocks, touched
all 5,000 rows and ran 24.6 ms by default, and so did it under fresh
statistics; with the setting it read 765 blocks, touched the 72-row window and
ran 3.9 ms. In the release that carried it, the claim CTE's slow-statement
maximum over a 13-minute burst window fell from 22.5-25.6 s to 3.9 s.

The setting is deliberately NOT applied to the admission, completion or lease
transactions. It was, in one release, and the admission flush became twice as
slow (``INSERT INTO gpu_fault_processor_queue`` slow statements: average 2.95
-> 4.99 s, maximum 9.1 -> 16.8 s over the same window; the completion CTE
appeared four times as often): a SET LOCAL governs every statement the
transaction runs, including the statement-level triggers the INSERT and the
completion fire -- the per-writer counter apply and the edge-triggered NOTIFY
over the ``added`` / ``removed`` transition tables, which have no indexes, so
pricing the sequential scan out only degrades their plans into per-row
probes. Those statements are keyed by request id, ordering key or cluster id
and need no help; the per-cluster depth reads stay on the default planner.

The setting is a cost penalty, not a prohibition: where no index can serve a
statement the planner still scans, so nothing fails. ``enable_sort = off`` was
measured and rejected: it also yields the ordered walk, but the claim
statement then executes in 1.3-1.7 s on a two-row table and on 5,000 rows
alike (its remaining sorts -- the ordered lane insert, the ``DISTINCT ON`` and
the final ordering -- are driven onto pathological alternatives; most of those
seconds are JIT compilation, which the disabled sorts' penalty cost switches
on). The setting cannot settle a window the planner expects to come back
short, a never-analysed table included: with a row estimate below the LIMIT
it reads every candidate. The window's predicates are spelled to keep that
estimate up instead (processor_claims.py ``_CLAIM_ELIGIBLE_SQL``), and the
statistics of a drained queue, which tell the planner the table is empty, are
refreshed by the first claim that fills its window (queue_statistics.py);
``enable_bitmapscan = off`` is not applied, because it prices the interlock's
scope GIN probes out (GIN indexes are bitmap-only).

``plan_cache_mode = force_custom_plan`` was evaluated and not adopted: psycopg
prepares a statement after five executions, and a generic plan built under
``enable_seqscan = off`` cannot regress to a heap scan; the only parameters
whose values vary are time bounds, limits and the owner, which do not change
the shape of the plan. Forcing custom plans would re-plan the claim's
multi-CTE statement on every one of the several hundred rounds a second the
worker tier issues. Should a generic plan ever prove worse, the tuple below is
the one place to add it.

``enable_material = off`` is the second setting. Once the window's predicates
keep its row estimate up (processor_claims.py ``_CLAIM_ELIGIBLE_SQL``), the
planner priced the retry-barrier anti-join as one Materialize over a full
index scan of the queue -- every deferred STRICT row, computed once per claim
-- instead of a probe of the lane index for each candidate the window's LIMIT
lets it walk: a fault claim on a 3,200-row burst queue read 3,800 shared
blocks with Materialize and 1,000 without it. Nothing else in the claim
statement needs a Materialize node; priced out, the planner keeps the hash
anti-join where every candidate must be read anyway (a backlog whose lanes are
all leased).
"""

from __future__ import annotations

from typing import Any

#: ``(setting, value)`` pairs applied with ``SET LOCAL`` at the start of the
#: claim transactions -- and only those (see the module docstring).
HOT_PATH_PLANNER_SETTINGS: tuple[tuple[str, str], ...] = (
    ("enable_seqscan", "off"),
    ("enable_material", "off"),
)


def hot_path_planner_statements() -> tuple[str, ...]:
    """The exact statements ``pin_hot_path_plans`` issues (tests pin them)."""

    return tuple(
        f"SET LOCAL {name} = {value}" for name, value in HOT_PATH_PLANNER_SETTINGS
    )


def pin_hot_path_plans(db: Any) -> None:
    """Apply the claim planner settings to the transaction ``db`` is in.

    ``SET LOCAL`` lasts until the surrounding transaction commits or rolls
    back, so the caller must already be inside ``db.transaction()``; ``db`` is
    anything with a ``cursor()`` context manager (the store's pool handle or a
    psycopg connection).
    """

    with db.cursor() as cursor:
        for statement in hot_path_planner_statements():
            # Always two positional arguments: every store cursor and every
            # cursor double accepts ``(statement, parameters)``, not all accept
            # a bare statement.
            cursor.execute(statement, ())
