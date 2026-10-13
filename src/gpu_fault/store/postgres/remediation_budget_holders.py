"""The PostgreSQL reads behind remediation budget occupancy.

The counting rule lives in ``gpu_fault.store.shared.remediation_budgets``:
RUNNING workflows listing a scope, whatever their lease, plus a SUPERSEDED
predecessor whose preempting successor has not claimed yet (its hold is
reserved for that successor). ``budget_candidates`` returns the rows that rule
needs for one claim -- the RUNNING holders of any requested scope and both rows
of every such handoff -- so ``budget_occupancy`` decides on exactly what the
in-memory and SQLite stores decide on. ``occupancy_counts`` is the same rule
as one aggregate: the dispatcher reads it once per tick (starvation guard).
"""

from __future__ import annotations

from typing import Any

# Preemption handoffs still in progress: the successor flagged to preempt its
# predecessor waits (PENDING / SAFETY_PENDING, the partial index
# ``gpu_fault_workflows_preempting_successor``) and the predecessor already
# gave way to it at a step boundary.
_HANDOFFS = """
    FROM gpu_fault_workflow_records successor
    JOIN gpu_fault_workflow_records predecessor
      ON predecessor.kind='workflow'
     AND predecessor.key=successor.predecessor_workflow_id
    WHERE successor.kind='workflow'
      AND successor.preempt_predecessor='true'
      AND successor.status IN ('PENDING', 'SAFETY_PENDING')
      AND predecessor.status='SUPERSEDED'
      AND predecessor.payload->>'preempted_by_workflow_id'=successor.key
"""
_CLAIMS = "coalesce({row}.payload->'remediation_budget_claims', '[]'::jsonb)"


def budget_candidates(cursor: Any, claimant: str, scopes: list[str]) -> list[Any]:
    """Raw payloads of every row that may hold one of ``scopes`` against
    ``claimant`` (the claimant's own RUNNING row is left out)."""

    cursor.execute(
        f"""
        SELECT w.payload FROM gpu_fault_workflow_records w
        WHERE w.kind='workflow' AND w.key<>%s AND w.status='RUNNING'
          AND {_CLAIMS.format(row="w")} ?| %s
        UNION ALL
        SELECT pair.payload FROM (
            SELECT predecessor.payload AS predecessor_payload,
                   successor.payload AS successor_payload
            {_HANDOFFS}
              AND {_CLAIMS.format(row="predecessor")} ?| %s
        ) handoff
        CROSS JOIN LATERAL (
            VALUES (handoff.predecessor_payload), (handoff.successor_payload)
        ) AS pair(payload)
        """,
        (claimant, scopes, scopes),
    )
    return [row[0] for row in cursor.fetchall()]


def occupancy_counts(cursor: Any) -> dict[str, int]:
    """Holders per scope over the whole region, one statement."""

    cursor.execute(
        f"""
        SELECT claim.scope, count(DISTINCT holder.key)
        FROM (
            SELECT w.key, {_CLAIMS.format(row="w")} AS claims
            FROM gpu_fault_workflow_records w
            WHERE w.kind='workflow' AND w.status='RUNNING'
            UNION ALL
            SELECT predecessor.key, {_CLAIMS.format(row="predecessor")}
            {_HANDOFFS}
        ) holder
        CROSS JOIN LATERAL jsonb_array_elements_text(
            CASE WHEN jsonb_typeof(holder.claims)='array'
                 THEN holder.claims ELSE '[]'::jsonb END
        ) AS claim(scope)
        GROUP BY claim.scope
        """
    )
    return {str(scope): int(count) for scope, count in cursor.fetchall()}
