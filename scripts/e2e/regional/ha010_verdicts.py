"""Pure verdict functions and constants of GF-REGIONAL-HA-010.

The case proves ARCH-H1/H3 on a real control plane: an Aurora writer failover
with no workflow in flight makes CPU Pods *NotReady* at most, never restarts
them. ``/livez`` is process-local and stays 200 on every Pod for the whole
window; ``/healthz`` may answer 503 while the registry refresh is stale but
never any other 5xx (the pre-fix 503 branch raised ``TypeError`` and the probe
saw a 500), and it is 200 again within the registry stale window plus one
refresh after RDS reports ``available``. Container restart counts do not move.
A replacement Pod deleted into the outage (H-2) comes up through the bounded
start-up retry instead of CrashLoopBackOff. ``regional_registry.secret_drift``
is false before and after: the Secret and the durable head agree.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import processor_queue_backlog

CASE_ID = "GF-REGIONAL-HA-010"
PREDECESSOR_CASE_ID = "GF-REGIONAL-HA-001"
CONFIRMATION = "HA010_AURORA_BLACKOUT_LIVENESS"
# The Deployments whose Pods read Aurora; `cleanup-inventory.json` lists the
# same three as CPU runtime deployments (ingress + consumer phases).
CPU_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
ROLLED_DEPLOYMENT = "gpu-fault-api-ha"
MINIMUM_ROLLED_READY_REPLICAS = 3
STALE_SECONDS_VARIABLE = "GPU_FAULT_REGISTRY_STALE_SECONDS"
STARTUP_RETRY_VARIABLE = "GPU_FAULT_STARTUP_STORE_RETRY_SECONDS"
DEFAULT_STALE_SECONDS = 90.0
DEFAULT_STARTUP_RETRY_SECONDS = 120.0
# One poll+refresh cycle after the stale window: readiness is allowed to lag
# the writer by exactly the threshold the manifest sets, plus one refresh.
READINESS_RECOVERY_MARGIN_SECONDS = 30.0
REPLACEMENT_READY_MARGIN_SECONDS = 180.0
STARTUP_RETRY_LOG_MARKER = "regional registry bootstrap attempt"
CRASH_LOOP_REASON = "CrashLoopBackOff"
OBSERVE_SECONDS_DEFAULT = 420
OBSERVE_SECONDS_BOUNDS = (120, 900)
FAILOVER_TIMEOUT_SECONDS = 900
SAMPLE_INTERVAL_SECONDS = 2.0


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def positive_seconds(value: Any, *, default: float, label: str) -> float:
    """A live env value as seconds, or the shipped default when unset."""

    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{label} is not a number: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive: {value!r}")
    return parsed


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    pods: dict[str, list[dict[str, Any]]],
    rds: dict[str, Any],
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    healthz: dict[str, dict[str, Any]],
    predecessor_valid: bool,
    scaled_to_zero: frozenset[str] = frozenset(),
) -> list[str]:
    """Refuse a run that could not prove what the case claims.

    ``pods`` maps each Deployment to its Pod records (``name``, ``ready``,
    ``restarts``, ``port``); ``healthz`` maps Pod name to the ``/healthz``
    payload read before the failover. ``scaled_to_zero`` names the Deployments
    whose ``spec.replicas`` is 0 on the site -- the telemetry spool tier is
    scaled away wherever spool admission is off -- which therefore have no
    Pod to sample and are not a missing tier.
    """

    errors: list[str] = []
    if not predecessor_valid:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    for deployment in CPU_DEPLOYMENTS:
        records = pods.get(deployment) or []
        if not records:
            if deployment not in scaled_to_zero:
                errors.append(f"{deployment} has no Pods")
            continue
        not_ready = sorted(
            str(item.get("name")) for item in records if not item.get("ready")
        )
        if not_ready:
            errors.append(f"{deployment} has Pods that are not Ready: {not_ready}")
        if any(not isinstance(item.get("port"), int) for item in records):
            errors.append(f"{deployment} has a Pod without an http containerPort")
    ready_rolled = [
        item for item in pods.get(ROLLED_DEPLOYMENT) or [] if item.get("ready")
    ]
    if len(ready_rolled) < MINIMUM_ROLLED_READY_REPLICAS:
        errors.append(
            f"{ROLLED_DEPLOYMENT} has {len(ready_rolled)} Ready replicas; "
            f"deleting one needs at least {MINIMUM_ROLLED_READY_REPLICAS}"
        )
    if rds.get("status") != "available":
        errors.append("Aurora cluster is not available")
    if not rds.get("writer") or len(rds.get("members") or []) < 2:
        errors.append("Aurora cluster has no failover-capable reader")
    if processor_queue_backlog(queue):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty; a workflow is in flight")
    errors.extend(registry_errors(healthz, label="preflight"))
    return errors


# --------------------------------------------------------------------------- #
# Registry (ARCH-H3)
# --------------------------------------------------------------------------- #
def registry_errors(
    healthz: dict[str, dict[str, Any]],
    *,
    label: str,
) -> list[str]:
    """Every Pod's ``/healthz`` reports a ready registry with no Secret drift."""

    errors: list[str] = []
    if not healthz:
        errors.append(f"{label}: no /healthz payload was read from any Pod")
    for pod, payload in sorted(healthz.items()):
        if payload.get("http_status") != 200:
            errors.append(
                f"{label}: {pod} /healthz returned {payload.get('http_status')}"
            )
        registry = (payload.get("payload") or {}).get("regional_registry") or {}
        if registry.get("ready") is not True:
            errors.append(f"{label}: {pod} regional registry is not ready")
        if registry.get("secret_drift") is not False:
            errors.append(
                f"{label}: {pod} reports secret_drift="
                f"{registry.get('secret_drift')!r}; the Secret and the durable "
                "head must agree"
            )
        if not registry.get("secret_config_sha256"):
            errors.append(f"{label}: {pod} reports no secret_config_sha256")
    return errors


# --------------------------------------------------------------------------- #
# Probe timelines (ARCH-H1)
# --------------------------------------------------------------------------- #
def sampler_errors(
    samples: list[dict[str, Any]],
    *,
    pod: str,
    stale_seconds: float,
    rds_available_at: datetime,
    failover_requested_at: datetime,
) -> list[str]:
    """One Pod's ``/livez`` and ``/healthz`` timeline across the failover.

    * ``/livez`` answers 200 on every sample; a non-200 or an unreachable
      sample means the kubelet would have restarted the container.
    * ``/healthz`` may answer 503 (readiness) but never another 5xx; the
      pre-fix 503 branch crashed into a 500.
    * ``/healthz`` is 200 again no later than ``stale_seconds`` plus one
      refresh after RDS reported ``available``.
    * The timeline must actually cover the failover: samples both before the
      request and after the recovery deadline.
    """

    errors: list[str] = []
    if not samples:
        return [f"{pod}: the probe recorded no samples"]
    recovery_deadline = rds_available_at.timestamp() + (
        stale_seconds + READINESS_RECOVERY_MARGIN_SECONDS
    )
    times = [float(item.get("t") or 0) for item in samples]
    if min(times) > failover_requested_at.timestamp():
        errors.append(f"{pod}: the probe started after the failover was requested")
    if max(times) < recovery_deadline:
        errors.append(
            f"{pod}: the probe stopped before the readiness recovery deadline"
        )
    for item in samples:
        moment = float(item.get("t") or 0)
        livez = item.get("livez")
        if livez != 200:
            errors.append(
                f"{pod}: /livez answered {livez!r} at {moment:.0f}; liveness must "
                "never fail during an Aurora outage"
            )
        healthz = item.get("healthz")
        if isinstance(healthz, int) and healthz >= 500 and healthz != 503:
            errors.append(
                f"{pod}: /healthz answered {healthz} at {moment:.0f}; only 503 "
                "is a readiness answer"
            )
        if healthz is None and livez == 200:
            errors.append(
                f"{pod}: /healthz was unreachable at {moment:.0f} while /livez "
                "answered; the readiness handler crashed rather than refused"
            )
    late = [item for item in samples if float(item.get("t") or 0) >= recovery_deadline]
    if late and any(item.get("healthz") != 200 for item in late):
        errors.append(
            f"{pod}: /healthz was not 200 within {stale_seconds:g}s + "
            f"{READINESS_RECOVERY_MARGIN_SECONDS:g}s after RDS became available"
        )
    final = late[-1] if late else samples[-1]
    if final.get("registry_ready") is not True:
        errors.append(f"{pod}: the last sample does not report a ready registry")
    if final.get("secret_drift") is not False:
        errors.append(f"{pod}: the last sample reports secret_drift")
    return errors


def readiness_outage_seconds(samples: list[dict[str, Any]]) -> float:
    """How long ``/healthz`` was non-200 in total (informational evidence)."""

    total = 0.0
    previous: float | None = None
    for item in sorted(samples, key=lambda entry: float(entry.get("t") or 0)):
        moment = float(item.get("t") or 0)
        if previous is not None and item.get("healthz") != 200:
            total += moment - previous
        previous = moment
    return round(total, 3)


# --------------------------------------------------------------------------- #
# Pods
# --------------------------------------------------------------------------- #
def restart_errors(
    before: dict[str, list[dict[str, Any]]],
    after: dict[str, list[dict[str, Any]]],
    *,
    deleted_pod: str,
) -> list[str]:
    """No pre-existing container restarted; the deleted Pod is gone."""

    errors: list[str] = []
    for deployment in CPU_DEPLOYMENTS:
        previous = {
            str(item.get("name")): item for item in before.get(deployment) or []
        }
        current = {str(item.get("name")): item for item in after.get(deployment) or []}
        for name, record in sorted(previous.items()):
            if name == deleted_pod:
                if name in current:
                    errors.append(f"{deployment}: deleted Pod {name} still exists")
                continue
            if name not in current:
                errors.append(
                    f"{deployment}: Pod {name} disappeared during the failover"
                )
                continue
            if current[name].get("uid") != record.get("uid"):
                errors.append(f"{deployment}: Pod {name} was recreated")
            delta = int(current[name].get("restarts") or 0) - int(
                record.get("restarts") or 0
            )
            if delta:
                errors.append(
                    f"{deployment}: Pod {name} restartCount moved by {delta}; "
                    "an Aurora outage must not restart control-plane containers"
                )
            if not current[name].get("ready"):
                errors.append(f"{deployment}: Pod {name} is not Ready afterwards")
        if len(current) != len(previous):
            errors.append(
                f"{deployment}: {len(current)} Pods afterwards, {len(previous)} before"
            )
    return errors


def replacement_errors(
    replacement: dict[str, Any],
    *,
    deleted_pod: str,
    deleted_at: datetime,
    budget_seconds: float,
) -> list[str]:
    """The Pod deleted into the outage came back through bounded retry.

    ``replacement`` is the runner's record: ``name``, ``uid``, ``ready``,
    ``restarts``, ``waiting_reasons`` (every ``state.waiting.reason`` and
    ``lastState`` reason seen), ``ready_at``.
    """

    errors: list[str] = []
    name = str(replacement.get("name") or "")
    if not name or name == deleted_pod:
        errors.append("no replacement Pod distinct from the deleted one was found")
    if replacement.get("ready") is not True:
        errors.append(f"replacement Pod {name} never became Ready")
    if int(replacement.get("restarts") or 0):
        errors.append(
            f"replacement Pod {name} restarted {replacement.get('restarts')} "
            "times; start-up must wait for Aurora, not crash"
        )
    reasons = {str(item) for item in replacement.get("waiting_reasons") or []}
    if CRASH_LOOP_REASON in reasons:
        errors.append(f"replacement Pod {name} entered {CRASH_LOOP_REASON}")
    ready_at = parse_time(replacement.get("ready_at"))
    if ready_at is None:
        errors.append(f"replacement Pod {name} has no ready_at")
    elif (ready_at - deleted_at).total_seconds() > budget_seconds:
        errors.append(
            f"replacement Pod {name} took "
            f"{(ready_at - deleted_at).total_seconds():.0f}s to become Ready; "
            f"budget is {budget_seconds:g}s"
        )
    return errors


def startup_retry_observed(log_lines: list[str]) -> bool:
    """Whether the replacement logged the ARCH-H1 bounded start-up retry."""

    return any(STARTUP_RETRY_LOG_MARKER in line for line in log_lines)


def relevant_log_lines(text: str, *, limit: int = 50) -> list[str]:
    tokens = (
        STARTUP_RETRY_LOG_MARKER,
        "operationalerror",
        "connection",
        "traceback",
        "read-only transaction",
    )
    relevant = [
        line[:500]
        for line in text.splitlines()
        if any(token in line.lower() for token in tokens)
    ]
    return relevant[-limit:]


def readiness_recovery_at(
    samples: list[dict[str, Any]],
    *,
    rds_available_at: datetime,
) -> float | None:
    """Epoch of the first ``/healthz`` 200 at or after RDS became available."""

    for item in sorted(samples, key=lambda entry: float(entry.get("t") or 0)):
        moment = float(item.get("t") or 0)
        if moment >= rds_available_at.timestamp() and item.get("healthz") == 200:
            return moment
    return None
