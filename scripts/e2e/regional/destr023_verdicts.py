"""Pure verdict functions and constants of GF-REGIONAL-DESTR-023.

Split out of ``run_destr023_idle_cluster_reset.py`` so the runner stays a
driver: everything here is unit-tested against synthetic coverage probes,
Pod listings and store snapshots and touches no cluster. DESTR-024 imports
the coverage half of this module; the two cases are the two sides of one
premise.

The premise: ``telemetry.WorkloadTopologyService.resolve`` calls a node IDLE
only when the cluster has *coverage* younger than
``GPU_FAULT_WORKLOAD_CONTEXT_FRESHNESS_SECONDS`` (600 s shipped) -- an attempt
observation of any attempt, or the completion watcher's coverage heartbeat
(``telemetry.WorkloadCoverageHeartbeat`` on ``POST /v1/attempts/coverage``:
one statement per *completed* full pass that saw nothing running, at most one
per 120 s; a busy cluster proves the feed alive through its observations
instead). The resolver reads a heartbeat as coverage only when it is fresh
*and* ``watched_pods == watched_attempts == 0``; a heartbeat that saw running
work leaves the node UNKNOWN (fail closed). Without either the node is UNKNOWN
and ``workflow_builder.compile_steps`` refuses every node-mutating plan with
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
# publishes attempt observations and the idle-cluster coverage heartbeat.
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
# settle; the heartbeat arrives at most once per 120 s (watcher built-in), so
# a tighter loop only reads the same record again.
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
        return None
    return parsed.astimezone(timezone.utc)


def freshness_seconds(coverage: dict[str, Any]) -> float:
    value = _finite_number(coverage.get("freshness_seconds"))
    return value if value is not None else 0.0


def heartbeat_observed_at(coverage: dict[str, Any]) -> datetime | None:
    heartbeat = coverage.get("heartbeat")
    if not isinstance(heartbeat, dict):
        return None
    return _parse(heartbeat.get("observed_at"))


def heartbeat_saw_work(coverage: dict[str, Any]) -> bool:
    """A heartbeat that counted running Pods or attempts is not coverage.

    The resolver reads only ``watched_pods == watched_attempts == 0`` as an
    idle statement; anything else is a watcher that saw work and leaves the
    node UNKNOWN. The watcher does not publish such a heartbeat at all, so a
    non-zero count in the stored row is itself a finding.
    """

    heartbeat = coverage.get("heartbeat") or {}
    return bool(
        int(heartbeat.get("watched_pods") or 0)
        or int(heartbeat.get("watched_attempts") or 0)
    )


def _age(coverage: dict[str, Any], key: str) -> float | None:
    return _finite_number(coverage.get(key))


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def coverage_supported_errors(coverage: dict[str, Any]) -> list[str]:
    """The control plane must carry the heartbeat at all.

    ``store.get_workload_coverage_heartbeat`` does not exist on a release
    before the heartbeat shipped; the probe reports that as
    ``heartbeat_supported=false`` and the case has to say "deploy first"
    rather than "the watcher is dead".
    """

    if coverage.get("heartbeat_supported") is not True:
        return [
            "control plane has no workload coverage heartbeat "
            "(store.get_workload_coverage_heartbeat missing); deploy the "
            "heartbeat release first"
        ]
    if freshness_seconds(coverage) <= 0:
        return ["coverage probe reports no freshness window"]
    errors = []
    probed = _parse(coverage.get("probed_at"))
    if probed is None:
        errors.append("coverage probe has no valid probed_at timestamp")
    count = coverage.get("observation_count")
    if type(count) is not int or count < 0:
        errors.append("coverage probe has no valid observation_count")
    for key in (
        "heartbeat",
        "heartbeat_age_seconds",
        "latest_observation_at",
        "latest_observation_age_seconds",
    ):
        if key not in coverage:
            errors.append(f"coverage probe is missing {key}")
    for key in ("heartbeat_age_seconds", "latest_observation_age_seconds"):
        if coverage.get(key) is not None and _age(coverage, key) is None:
            errors.append(f"coverage probe reports an unusable {key}")
    heartbeat = coverage.get("heartbeat")
    if heartbeat is not None:
        if not isinstance(heartbeat, dict):
            return [*errors, "coverage heartbeat is not an object"]
        observed = heartbeat_observed_at(coverage)
        if observed is None:
            errors.append("coverage heartbeat has no valid observed_at timestamp")
        if observed is not None and probed is not None and observed > probed:
            errors.append("coverage heartbeat is dated after the probe")
        age = _age(coverage, "heartbeat_age_seconds")
        if age is None:
            errors.append("coverage heartbeat age is unknown")
        elif (
            observed is not None
            and probed is not None
            and not math.isclose((probed - observed).total_seconds(), age, abs_tol=0.01)
        ):
            errors.append("coverage heartbeat age disagrees with its timestamp")
        for key in ("watched_pods", "watched_attempts"):
            value = heartbeat.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"coverage heartbeat has no valid {key} count")
    elif coverage.get("heartbeat_age_seconds") is not None:
        errors.append("coverage heartbeat age exists without a heartbeat")
    latest = _parse(coverage.get("latest_observation_at"))
    age = _age(coverage, "latest_observation_age_seconds")
    if type(count) is int and count > 0:
        if latest is None or age is None:
            errors.append(
                "attempt observations exist but their timestamp or age is unknown"
            )
        elif probed is not None and (
            latest > probed
            or not math.isclose((probed - latest).total_seconds(), age, abs_tol=0.01)
        ):
            errors.append("attempt observation age disagrees with its timestamp")
    elif count == 0 and (
        coverage.get("latest_observation_at") is not None
        or coverage.get("latest_observation_age_seconds") is not None
    ):
        errors.append("zero attempt observations have a latest timestamp or age")
    return errors


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
    elif heartbeat_saw_work(coverage):
        errors.append(
            "coverage heartbeat saw running work (watched_pods/watched_attempts "
            "non-zero); the resolver does not read it as coverage"
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

    errors = coverage_supported_errors(coverage)
    if errors:
        raise ValueError("; ".join(errors))
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
    status = value.get("status") or {}
    return {
        "uid": metadata.get("uid"),
        "resource_version": metadata.get("resourceVersion"),
        "generation": metadata.get("generation"),
        "deleting": bool(metadata.get("deletionTimestamp")),
        "observed_generation": status.get("observedGeneration"),
        "replicas": spec.get("replicas"),
        "status_replicas": status.get("replicas", 0),
        "updated_replicas": status.get("updatedReplicas", 0),
        "ready_replicas": status.get("readyReplicas", 0),
        "available_replicas": status.get("availableReplicas", 0),
        "unavailable_replicas": status.get("unavailableReplicas", 0),
        "strategy": ((spec.get("strategy") or {}).get("type")),
    }


def watcher_errors(summary: dict[str, Any], *, expected_replicas: int = 1) -> list[str]:
    errors = []
    if not summary.get("uid") or summary.get("deleting") is not False:
        errors.append("watcher Deployment identity is missing or terminating")
    generation = summary.get("generation")
    observed = summary.get("observed_generation")
    if (
        type(generation) is not int
        or generation < 1
        or type(observed) is not int
        or observed != generation
    ):
        errors.append("watcher Deployment generation is not fully observed")
    for key in (
        "replicas",
        "status_replicas",
        "updated_replicas",
        "ready_replicas",
        "available_replicas",
        "unavailable_replicas",
    ):
        expected = 0 if key == "unavailable_replicas" else expected_replicas
        if type(summary.get(key)) is not int or summary[key] != expected:
            errors.append(
                f"{WATCHER_DEPLOYMENT} has {key}={summary.get(key)}, expected {expected}"
            )
    if "pod_count" in summary and (
        summary["pod_count"] != expected_replicas
        or len(summary.get("ready_pods") or []) != expected_replicas
    ):
        errors.append("watcher Pod population is not exactly Ready")
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
