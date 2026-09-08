"""Pure verdict functions and constants of GF-REGIONAL-DESTR-023.

Split out of ``run_destr023_idle_cluster_reset.py`` so the runner stays a
driver: everything here is unit-tested against synthetic coverage probes,
Pod listings and store snapshots and touches no cluster. DESTR-024 imports
the coverage half of this module; the two cases are the two sides of one
premise.

The premise: ``telemetry.WorkloadTopologyService.resolve`` calls a node IDLE
only when the cluster has *coverage* younger than
``GPU_FAULT_WORKLOAD_CONTEXT_FRESHNESS_SECONDS`` (600 s shipped) -- an attempt
observation of any attempt, or, since CP-8, the completion watcher's scan
heartbeat (``POST /v1/workload-coverage``, one per full reconcile pass even
with zero managed attempts). Without either the node is UNKNOWN and
``workflow_builder.compile_steps`` refuses every node-mutating plan with
``node workload state is UNKNOWN`` (ARCH-B2).

Before the heartbeat, every idle-node destructive case that passed did so
because a leftover training job kept the observation feed alive; DESTR-016
attempt 4 on 2026-09-08 found a truly idle cluster and was BLOCKED. This case
therefore has to prove IDLE *without* any observation younger than the
window: ``fresh_coverage_errors`` refuses a run in which an attempt
observation could explain the IDLE on its own.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional import run_destr001_gpu_reset as reset_case

CASE_ID = "GF-REGIONAL-DESTR-023"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-001"
CONFIRMATION = "DESTR023_EXECUTE"

# ``deploy/dataplane/completion-watcher.yaml``: the singleton Deployment that
# publishes attempt observations and, since CP-8, the coverage heartbeat.
WATCHER_DEPLOYMENT = "gpu-fault-completion-watcher"
WATCHER_APP_LABEL = "gpu-fault-completion-watcher"
# ``completion_controller.MANAGED_LABEL``: the Pods the watcher observes.
MANAGED_LABEL = "gpu-fault.io/managed"
# The campaign lever that kept coverage alive before the heartbeat existed
# (memory: idle-cluster-workload-state-unknown). Its presence would make IDLE
# unattributable, so the case refuses to run next to it.
COVERAGE_CANARY_JOB = "gpu-fault-coverage-canary"
# ``orchestration/workflow_builder.py::WORKLOAD_STATE_UNKNOWN_REASON``,
# pinned by test so the two cannot drift apart silently.
WORKLOAD_STATE_UNKNOWN_REASON = "node workload state is UNKNOWN"
IDLE = "IDLE"
UNKNOWN = "UNKNOWN"
RESET_OPERATIONS = tuple(reset_case.EXPECTED_STEPS)

# Waiting for old observations to age out is bounded: the shipped window is
# 600 s and a deployment that raised it past this is a site decision the
# case will not sit through inside a maintenance window.
MAX_EXPIRY_WAIT_SECONDS = 1200
EXPIRY_MARGIN_SECONDS = 30
# The probe is re-read on this cadence while the case waits for the feed to
# settle; the heartbeat arrives once per watcher reconcile pass, so a tighter
# loop only reads the same record again.
COVERAGE_POLL_SECONDS = 15
SETTLE_BUDGET_SECONDS = 180


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def freshness_seconds(coverage: dict[str, Any]) -> float:
    return float(coverage.get("freshness_seconds") or 0.0)


def heartbeat_scanned_at(coverage: dict[str, Any]) -> datetime | None:
    heartbeat = coverage.get("heartbeat") or {}
    return _parse(heartbeat.get("scanned_at"))


def _age(coverage: dict[str, Any], key: str) -> float | None:
    value = coverage.get(key)
    return None if value is None else float(value)


def coverage_supported_errors(coverage: dict[str, Any]) -> list[str]:
    """The control plane must carry the heartbeat at all.

    ``store.get_workload_coverage`` does not exist on a release before CP-8;
    the probe reports that as ``heartbeat_supported=false`` and the case has
    to say "deploy first" rather than "the watcher is dead".
    """

    if not coverage.get("heartbeat_supported"):
        return [
            "control plane has no workload coverage heartbeat "
            "(store.get_workload_coverage missing); deploy the CP-8 release first"
        ]
    if freshness_seconds(coverage) <= 0:
        return ["coverage probe reports no freshness window"]
    return []


def fresh_coverage_errors(
    coverage: dict[str, Any],
    *,
    require_stale_observations: bool = True,
) -> list[str]:
    """IDLE, and IDLE *because of the heartbeat*.

    ``require_stale_observations`` is the DESTR-023 premise: no attempt
    observation of the cluster may be younger than the freshness window,
    otherwise a leftover job would explain the IDLE and the heartbeat would
    have proven nothing. DESTR-024's recovery check passes ``False``: there
    the question is only whether the watcher is back.
    """

    errors = coverage_supported_errors(coverage)
    if errors:
        return errors
    window = freshness_seconds(coverage)
    heartbeat_age = _age(coverage, "heartbeat_age_seconds")
    if coverage.get("heartbeat") is None or heartbeat_age is None:
        errors.append("cluster has no coverage heartbeat")
    elif heartbeat_age > window:
        errors.append(
            f"coverage heartbeat is stale (age={heartbeat_age:.0f}s > {window:.0f}s)"
        )
    observation_age = _age(coverage, "latest_observation_age_seconds")
    if (
        require_stale_observations
        and observation_age is not None
        and observation_age <= window
    ):
        errors.append(
            "an attempt observation is younger than the freshness window "
            f"(age={observation_age:.0f}s); IDLE is not attributable to the heartbeat"
        )
    if coverage.get("workload_state") != IDLE:
        errors.append(
            f"target node workload state is {coverage.get('workload_state')}, not IDLE"
        )
    return errors


def stale_coverage_errors(coverage: dict[str, Any]) -> list[str]:
    """UNKNOWN, with nothing fresh that could have made it IDLE."""

    errors = coverage_supported_errors(coverage)
    if errors:
        return errors
    window = freshness_seconds(coverage)
    heartbeat_age = _age(coverage, "heartbeat_age_seconds")
    if coverage.get("heartbeat") is not None and heartbeat_age is not None:
        if heartbeat_age <= window:
            errors.append(
                f"coverage heartbeat is still fresh (age={heartbeat_age:.0f}s <= "
                f"{window:.0f}s)"
            )
    observation_age = _age(coverage, "latest_observation_age_seconds")
    if observation_age is not None and observation_age <= window:
        errors.append(
            "an attempt observation is still younger than the freshness window "
            f"(age={observation_age:.0f}s)"
        )
    if coverage.get("workload_state") != UNKNOWN:
        errors.append(
            f"target node workload state is {coverage.get('workload_state')}, "
            "not UNKNOWN"
        )
    return errors


def _wait_until_older(age: float | None, window: float, margin: int) -> int:
    if age is None:
        return 0
    return max(0, math.ceil(window - age + margin))


def expiry_wait_seconds(
    coverage: dict[str, Any],
    *,
    include_heartbeat: bool = False,
    margin: int = EXPIRY_MARGIN_SECONDS,
) -> int:
    """Seconds until every observation (and, optionally, the heartbeat) has
    aged past the freshness window, plus ``margin``.

    Raises when the deployed window is too long for a maintenance window to
    wait out -- that is a site configuration to reason about, not a case to
    sit through.
    """

    window = freshness_seconds(coverage)
    if window > MAX_EXPIRY_WAIT_SECONDS:
        raise ValueError(
            f"freshness window {window:.0f}s exceeds the {MAX_EXPIRY_WAIT_SECONDS}s "
            "this case is willing to wait"
        )
    wait = _wait_until_older(
        _age(coverage, "latest_observation_age_seconds"), window, margin
    )
    if include_heartbeat:
        wait = max(
            wait,
            _wait_until_older(_age(coverage, "heartbeat_age_seconds"), window, margin),
        )
    return wait


def managed_pod_summary(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The managed Pods of the cluster, whatever their phase.

    A Succeeded or Failed managed Pod is still an attempt the watcher
    observes (a terminal attempt proves the feed is alive), so the idle
    premise excludes every phase, not just Running.
    """

    result = []
    for item in items:
        metadata = item.get("metadata") or {}
        labels = metadata.get("labels") or {}
        if str(labels.get(MANAGED_LABEL, "")).lower() != "true":
            continue
        result.append(
            {
                "namespace": metadata.get("namespace"),
                "name": metadata.get("name"),
                "node": (item.get("spec") or {}).get("nodeName"),
                "phase": (item.get("status") or {}).get("phase"),
            }
        )
    return sorted(result, key=lambda row: (str(row["namespace"]), str(row["name"])))


def idle_cluster_errors(
    managed_pods: list[dict[str, Any]],
    canary_jobs: list[str],
) -> list[str]:
    errors = []
    if managed_pods:
        errors.append(
            f"cluster has {len(managed_pods)} managed Pod(s); the idle premise "
            "requires none"
        )
    if canary_jobs:
        errors.append(
            "a coverage canary Job is present "
            f"({', '.join(sorted(canary_jobs))}); delete it before this case"
        )
    return errors


def deployment_summary(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata") or {}
    spec = value.get("spec") or {}
    return {
        "uid": metadata.get("uid"),
        "generation": metadata.get("generation"),
        "replicas": int(spec.get("replicas") or 0),
        "ready_replicas": int((value.get("status") or {}).get("readyReplicas") or 0),
        "strategy": ((spec.get("strategy") or {}).get("type")),
    }


def watcher_errors(summary: dict[str, Any], *, expected_replicas: int = 1) -> list[str]:
    errors = []
    if int(summary.get("replicas") or 0) != expected_replicas:
        errors.append(
            f"{WATCHER_DEPLOYMENT} has replicas={summary.get('replicas')}, "
            f"expected {expected_replicas}"
        )
    if int(summary.get("ready_replicas") or 0) != expected_replicas:
        errors.append(
            f"{WATCHER_DEPLOYMENT} has ready_replicas={summary.get('ready_replicas')}, "
            f"expected {expected_replicas}"
        )
    return errors


def blocked_by_unknown_errors(state: dict[str, Any]) -> list[str]:
    """The one failure this case exists to catch, named before anything else."""

    workflow = state.get("workflow") or {}
    errors = []
    if WORKLOAD_STATE_UNKNOWN_REASON in (workflow.get("blocked_reasons") or []):
        errors.append(
            "workflow was BLOCKED by UNKNOWN workload state despite fresh "
            "heartbeat coverage"
        )
    if workflow.get("status") == "BLOCKED":
        errors.append("reset workflow is BLOCKED")
    return errors


def reset_errors(state: dict[str, Any]) -> list[str]:
    """The DESTR-001 reset contract, preceded by the UNKNOWN-block check."""

    return [*blocked_by_unknown_errors(state), *reset_case.workflow_errors(state)]
