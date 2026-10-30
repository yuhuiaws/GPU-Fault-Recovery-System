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

The tier classification (``budget_tier``) and the occupancy verdict
(``budget_occupancy_check``) live here too; the action suite imports them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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


def budget_tier(
    budget: dict[str, int | None],
    *,
    clusters: int,
    workflows_per_cluster: int,
    executor_workers: int,
) -> dict:
    """Classify the live budget against the fixed request model (§8.2).

    Raw capacity tier: region >= clusters x workflows, cluster and resource
    class >= workflows per cluster -- every workflow may run at once, so each
    executor must reach one in-flight command per workflow. Production budget tier:
    a lower limit throttles the workflows on purpose, so an executor only has
    to make progress (floor 1); the budget waits are the measurement. Live
    2026-09-22: under 20/5/2 every executor completed its 40 commands but
    exited on the raw-tier floor of 8, failing a run that had behaved.

    Unbounded: none of the three is set -- the shipped state since they were
    retired. No fixed number caps the model; each GPU cluster's remediation
    storm breaker does, and it trips once max(3, ceil(20% of the cluster's
    registered GPU nodes)) nodes of two or more workflows enter destructive
    remediation within 600 s. The suite registers a healthy fleet beside the
    model's nodes (``regional_synthetic_fleet``: 256 GPU nodes a cluster by
    default, threshold 52), so the model stays below it; the breaker itself is
    the storm breaker step's (perf acceptance plan §8.5).
    """

    region = budget.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION")
    per_cluster = budget.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER")
    per_class = budget.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS")
    limits = [
        (region, clusters * workflows_per_cluster),
        (per_cluster, workflows_per_cluster),
        (per_class, workflows_per_cluster),
    ]
    if all(limit is None for limit, _ in limits):
        tier = "unbounded"
    elif all(limit is None or limit >= needed for limit, needed in limits):
        tier = "raw-capacity"
    else:
        tier = "production-budget"
    return {
        "tier": tier,
        "limits": dict(budget),
        "required_for_raw_capacity": {
            "region": clusters * workflows_per_cluster,
            "per_cluster": workflows_per_cluster,
            "per_resource_class": workflows_per_cluster,
        },
        # The executor's concurrency floor is progress-only in every tier: the
        # per-node and per-failure-domain limits serialize a cluster's workflows
        # on purpose, so peak concurrency is evidence (max_concurrent_commands
        # per executor), not a pass criterion. Live 2026-09-22: floors of 8 and
        # then 4 each failed a run whose 128 workflows had all SUCCEEDED.
        "min_concurrent_commands": 1,
    }


# The scope kinds the occupancy verdict bounds, and the limit each one reads
# from ``run.json.remediation_budget.limits``. Node and failure-domain peaks are
# recorded as evidence only.
BUDGET_OCCUPANCY_LIMITS = {
    "region": "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION",
    "cluster": "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER",
    "class": "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS",
}
# ``execute_capacity_run``'s result when the workflows completed but a sample
# held more RUNNING workflows on one scope than its limit.
BUDGET_LIMIT_EXCEEDED = 3


def budget_occupancy_check(history: list[dict], remediation_budget: dict) -> dict:
    """RUNNING-with-claims against the run's recorded limits (性能压测验收方案 §8.2).

    A claim is held until its workflow ends (详细设计-v2 §9.5), so every
    ``history.json`` sample must stay within the limits, not only the
    admissions: per sample, ``budget_holders_max`` is the number of RUNNING
    workflows listing the busiest scope of each kind -- the region, the busiest
    cluster, the busiest (cluster, resource class). A limit that is unset is
    not checked; a history without any such sample fails, since it proves
    nothing.
    """

    limits = remediation_budget.get("limits") or {}
    peaks: dict[str, int] = {}
    samples = [
        item["budget_holders_max"] for item in history if "budget_holders_max" in item
    ]
    for holders in samples:
        for kind, count in holders.items():
            peaks[kind] = max(peaks.get(kind, 0), int(count))
    bounded = {kind: limits.get(name) for kind, name in BUDGET_OCCUPANCY_LIMITS.items()}
    violations = [
        f"{kind} scope held by {peaks[kind]} RUNNING workflows, limit {limit}"
        for kind, limit in bounded.items()
        if limit is not None and peaks.get(kind, 0) > int(limit)
    ]
    if not samples:
        violations.append("history.json has no remediation budget occupancy sample")
    return {
        "passed": not violations,
        "samples": len(samples),
        "peak_holders": peaks,
        "limits": bounded,
        "violations": violations,
    }


def live_budget_preflight(control: Callable[..., str]) -> dict[str, Any]:
    """Every Running control worker: per node and failure domain 1 and none of
    the retired fixed limits (perf plan §4.3; the integrated suite's
    ``remediation-budget-preflight.json``). A worker that still carries one runs
    an older release or a leftover override."""

    expected = {
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": None,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": None,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": None,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": "1",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": "1",
    }
    pods = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-control-worker",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    if not pods:
        raise RuntimeError("no Running control-worker Pods for budget check")
    actual = {}
    names = sorted(expected)
    for pod in pods:
        values = control(
            "exec",
            pod,
            "--",
            "python3",
            "-c",
            (
                "import json,os; print(json.dumps({"
                + ",".join(f"{name!r}:os.environ.get({name!r})" for name in names)
                + "},sort_keys=True))"
            ),
        )
        actual[pod] = json.loads(values.splitlines()[-1])
    mismatched = {pod: values for pod, values in actual.items() if values != expected}
    if mismatched:
        raise RuntimeError(
            "live remediation budget does not match formal matrix: "
            + json.dumps(mismatched, sort_keys=True)
        )
    return {"expected": expected, "pods": actual}
