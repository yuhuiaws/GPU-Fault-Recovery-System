"""The PostgreSQL reads behind remediation budget occupancy.

The counting rule lives in ``gpu_fault.store.shared.remediation_budgets``:
RUNNING workflows listing a scope, whatever their lease, a BLOCKED record that
still occupies its node with an unresolved node action, plus a SUPERSEDED
predecessor whose preempting successor has not claimed yet (its hold is
reserved for that successor). ``budget_candidates`` returns the rows that rule
needs for one claim -- the RUNNING and node-occupying BLOCKED rows listing any
requested scope and both rows of every such handoff -- so ``budget_occupancy``
decides on exactly what the in-memory and SQLite stores decide on.
``occupancy_counts`` is the same rule for the whole region: one aggregate over
the RUNNING rows and the handoffs, plus the node-occupying BLOCKED rows decoded
for the rule's node-action check (an operator queue, not a fleet-sized set);
the dispatcher reads it once per tick (starvation guard).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from gpu_fault.models import WorkflowRequest
from gpu_fault.store.shared.remediation_budgets import held_while_blocked

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
# BLOCKED rows that still occupy their node (``workflow_is_open``; mirrors
# ``_OPEN_PREDECESSOR_SQL``). Which of them hold is the shared rule's call: an
# unresolved node action cannot be judged from a column.
_OCCUPYING_BLOCKED = (
    "w.status='BLOCKED' AND w.blocked_kind IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')"
)


def budget_candidates(cursor: Any, claimant: str, scopes: list[str]) -> list[Any]:
    """Raw payloads of every row that may hold one of ``scopes`` against
    ``claimant`` (the claimant's own row is left out)."""

    cursor.execute(
        f"""
        SELECT w.payload FROM gpu_fault_workflow_records w
        WHERE w.kind='workflow' AND w.key<>%s
          AND (w.status='RUNNING' OR ({_OCCUPYING_BLOCKED}))
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


def occupancy_counts(
    cursor: Any, decode: Callable[[str, Any], WorkflowRequest]
) -> dict[str, int]:
    """Holders per scope over the whole region."""

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
    counts: Counter[str] = Counter(
        {str(scope): int(count) for scope, count in cursor.fetchall()}
    )
    cursor.execute(
        f"""
        SELECT w.payload FROM gpu_fault_workflow_records w
        WHERE w.kind='workflow' AND {_OCCUPYING_BLOCKED}
          AND {_CLAIMS.format(row="w")} <> '[]'::jsonb
        """
    )
    for (payload,) in cursor.fetchall():
        blocked = decode("workflow", payload)
        if held_while_blocked(blocked):
            counts.update(set(blocked.remediation_budget_claims))
    return dict(counts)
