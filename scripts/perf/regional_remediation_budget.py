"""``remediation-budget.json``: the budget evidence of one action capacity run.

The performance acceptance plan (§11) requires every action round under a
remediation budget to keep one document with the five budget tiers, the peak
active claims per scope, the waits, the longest wait, the workflows that never
got to run and the claims released by a lapsed lease. The runner already holds
every one of them, so it writes the document instead of leaving each reader to
recombine ``run.json`` and ``summary.json``:

* the five tiers and the tier the run measured: ``run.json.remediation_budget``
  (read from a control-worker Pod before the run);
* the peak active claims: the occupancy verdict over ``history.json``
  (``budget_occupancy_check``) -- per scope kind, the most RUNNING workflows
  holding one scope of that kind in any sample, the count the store admits
  against; a kind no workflow claimed is absent;
* the waits, from the run's own workflow rows at the end (``database_details``):
  the claim attempts the budget refused (the product's per-workflow
  ``remediation_budget_wait_count``), the workflows refused at least once, the
  longest wait (creation to first remote command of such a workflow: the
  budget wait plus the dispatcher's pickup) and the workflows with no remote
  command at all;
* lease-expiry releases: ``not-applicable``. Claims are held until the
  workflow is terminal, whatever its lease says (详细设计-v2 §9.5), so a lapsed
  lease releases nothing to count.

A field the release's workflow rows do not carry is ``unsupported``, never a
zero (§11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT = "remediation-budget.json"
UNSUPPORTED = "unsupported"
NOT_APPLICABLE = "not-applicable"
LEASE_EXPIRY_REASON = (
    "claims are held until the workflow is terminal, whatever its lease says: "
    "a lapsed lease releases nothing"
)
WAIT_FIELDS = ("refused_claims", "waited_workflows", "longest_wait_seconds")


def remediation_budget_artifact(
    *,
    run_id: str,
    tier: dict[str, Any],
    occupancy: dict[str, Any],
    waits: dict[str, Any] | None,
) -> dict[str, Any]:
    """The document; ``waits`` is ``database_details``' ``remediation_budget_waits``."""

    waits = waits or {}
    limits = tier.get("limits")
    peaks = occupancy.get("peak_holders")
    waits_recorded = waits.get("recorded") is True
    never_executed = waits.get("never_executed_workflows")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "tier": tier.get("tier") or UNSUPPORTED,
        "limits": dict(limits) if isinstance(limits, dict) and limits else UNSUPPORTED,
        "peak_active_claims": (
            {str(kind): int(count) for kind, count in sorted(peaks.items())}
            if waits.get("claims_recorded") is True and isinstance(peaks, dict)
            else UNSUPPORTED
        ),
        "occupancy": {
            "passed": occupancy.get("passed"),
            "samples": occupancy.get("samples"),
            "limits": occupancy.get("limits"),
            "violations": list(occupancy.get("violations") or []),
        },
        "waits": {
            name: waits.get(name) if waits_recorded else UNSUPPORTED
            for name in WAIT_FIELDS
        },
        "never_executed_workflows": (
            never_executed if type(never_executed) is int else UNSUPPORTED
        ),
        "lease_expiry_releases": NOT_APPLICABLE,
        "lease_expiry_releases_reason": LEASE_EXPIRY_REASON,
    }


def write_remediation_budget(
    artifacts: Path,
    run_id: str,
    tier: dict[str, Any] | None,
    occupancy: dict[str, Any],
    details: dict[str, Any],
) -> dict[str, Any]:
    """Write ``<artifacts>/remediation-budget.json`` and return it."""

    document = remediation_budget_artifact(
        run_id=run_id,
        tier=tier or {},
        occupancy=occupancy,
        waits=details.get("remediation_budget_waits"),
    )
    (artifacts / ARTIFACT).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    return document
