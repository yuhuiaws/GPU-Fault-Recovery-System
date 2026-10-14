"""The PostgreSQL reads behind remediation budget occupancy.

The counting rule lives in ``gpu_fault.store.shared.remediation_budgets``:
RUNNING workflows listing a scope, whatever their lease, a BLOCKED record that
still occupies its node with a node action that may still be running, plus a SUPERSEDED
predecessor whose preempting successor has not claimed yet (its hold is
reserved for that successor). Both reads apply it in two parts: the RUNNING
holders are counted in SQL -- decoding every holder for every admission made a
burst of N admissions cost N^2 / 2 decodes under the region scope's lock
(18.8 s for 350 in one tick, over the 15-s dispatch lease) -- and the
node-occupying BLOCKED rows and the handoff pairs, an operator queue and a
moment of a preemption rather than fleet-sized sets, are decoded for the
shared rule. ``claim_occupancy`` is one claim's view (its requested scopes,
its own hold left out); ``occupancy_counts`` the region's, which the dispatcher
reads once per tick (starvation guard, over-limit scopes).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from gpu_fault.models import WorkflowRequest
from gpu_fault.store.shared.remediation_budgets import (
    budget_occupancy,
    held_while_blocked,
)

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
# ``_OPEN_PREDECESSOR_SQL``). Which of them hold is the shared rule's call:
# whether a node action may still be running cannot be judged from a column.
_OCCUPYING_BLOCKED = (
    "w.status='BLOCKED' AND w.blocked_kind IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')"
)


def claim_occupancy(
    cursor: Any,
    claimant: str,
    scopes: list[str],
    decode: Callable[[str, Any], WorkflowRequest],
) -> Counter[str]:
    """Holders of each of ``scopes`` against ``claimant``, its own hold and the
    one reserved for it left out."""

    cursor.execute(
        f"""
        SELECT claim.scope, count(DISTINCT w.key)
        FROM gpu_fault_workflow_records w
        CROSS JOIN LATERAL jsonb_array_elements_text(
            CASE WHEN jsonb_typeof({_CLAIMS.format(row="w")})='array'
                 THEN {_CLAIMS.format(row="w")} ELSE '[]'::jsonb END
        ) AS claim(scope)
        WHERE w.kind='workflow' AND w.key<>%s AND w.status='RUNNING'
          AND {_CLAIMS.format(row="w")} ?| %s
          AND claim.scope = ANY(%s)
        GROUP BY claim.scope
        """,
        (claimant, scopes, scopes),
    )
    counts: Counter[str] = Counter(
        {str(scope): int(count) for scope, count in cursor.fetchall()}
    )
    cursor.execute(
        f"""
        SELECT w.payload FROM gpu_fault_workflow_records w
        WHERE w.kind='workflow' AND w.key<>%s AND {_OCCUPYING_BLOCKED}
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
    counts.update(
        budget_occupancy(
            [decode("workflow", row[0]) for row in cursor.fetchall()], claimant=claimant
        )
    )
    return counts


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
