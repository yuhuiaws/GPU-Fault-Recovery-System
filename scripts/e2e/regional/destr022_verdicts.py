"""Pure verdict functions and constants of GF-REGIONAL-DESTR-022.

The case proves ARCH-A4b/A4c on a real cluster: the regional cluster
executor's ``SpareReservationSweep`` (storeless, so only the
``gpu-fault.io/spare-reserved-at`` TTL can decide) reclaims a warm-spare
reservation whose owner can no longer be asked. The runner writes a
reservation for a synthetic incident that exists nowhere, back-dated two days
past the one-day TTL, onto the one declared spare -- and leaves the spare
cordoned, so nothing can ever schedule onto it. Within one sweep interval the
executor must release the reservation (annotations gone, pool state back to
AVAILABLE, node still unschedulable), log the reclaim naming node and incident,
and move ``spare_reservations_reclaimed_total`` in the claim-state breadcrumb,
the executor Pod's only counter surface.

Nothing physical happens: no incident, no workflow, no provider call. The
verdicts here judge plain evidence documents so the unit suite can run them
against fabricated evidence once on the intended run and once per way the
run can be wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

CASE_ID = "GF-REGIONAL-DESTR-022"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-008"
CONFIRMATION = "DESTR022_RECLAIM_STALE_SPARE_RESERVATION"

SPARE_LABEL = "gpu-fault.io/spare"
HYPERPOD_HEALTH_LABEL = "sagemaker.amazonaws.com/node-health-status"
SPARE_RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
SPARE_RESERVED_AT_ANNOTATION = "gpu-fault.io/spare-reserved-at"
SPARE_POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
TRACKED_ANNOTATIONS = (
    SPARE_RESERVATION_ANNOTATION,
    SPARE_RESERVED_AT_ANNOTATION,
    SPARE_POOL_STATE_ANNOTATION,
)
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
OWNERSHIP_ANNOTATIONS = (
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
)
POOL_STATE_ALLOCATED = "ALLOCATED"
POOL_STATE_AVAILABLE = "AVAILABLE"
# The executor sweeps every 300 s (``SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS``)
# and reclaims once the reservation is older than 86400 s. A two-day back-date
# leaves a full day of margin over the TTL, so clock skew cannot keep the
# reservation alive.
STALE_RESERVATION_DAYS = 2
SWEEP_INTERVAL_SECONDS = 300
RECLAIM_TIMEOUT_SECONDS = SWEEP_INTERVAL_SECONDS + 180
MIN_RECLAIM_TIMEOUT_SECONDS = 120
MAX_RECLAIM_TIMEOUT_SECONDS = 900
RECLAIM_COUNTER = "spare_reservations_reclaimed_total"
RECLAIM_LOG_PREFIX = "reclaimed stale spare reservation:"
SYNTHETIC_INCIDENT_PREFIX = "acceptance-stale-"


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


def stale_reserved_at(now: datetime, *, days: int = STALE_RESERVATION_DAYS) -> str:
    """The ``reserved-at`` value the runner writes: well past the TTL."""

    if days < 2:
        raise ValueError("a stale reservation must be back-dated at least two days")
    return (now.astimezone(timezone.utc) - timedelta(days=days)).isoformat()


def synthetic_incident_id(run_id: str) -> str:
    return f"{SYNTHETIC_INCIDENT_PREFIX}{run_id}"


def counter_of(breadcrumb: dict[str, Any] | None) -> int | None:
    """``counters.spare_reservations_reclaimed_total`` of one claim-state file."""

    counters = (breadcrumb or {}).get("counters")
    if not isinstance(counters, dict) or RECLAIM_COUNTER not in counters:
        return None
    try:
        return int(counters[RECLAIM_COUNTER])
    except (TypeError, ValueError):
        return None


def breadcrumb_claimed_at(breadcrumb: dict[str, Any] | None) -> datetime | None:
    return parse_time((breadcrumb or {}).get("last_successful_claim_at"))


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def spare_refusals(spare: dict[str, Any]) -> list[str]:
    """Why this node cannot carry the synthetic reservation."""

    errors: list[str] = []
    if spare.get("ready") != "True":
        errors.append("spare node is not Ready")
    if not spare.get("unschedulable"):
        errors.append("spare node is not cordoned; a declared spare must be")
    labels = spare.get("labels") or {}
    if labels.get(SPARE_LABEL) != "true":
        errors.append("spare node is not declared by the spare label")
    if labels.get(HYPERPOD_HEALTH_LABEL) != "Schedulable":
        errors.append("spare HyperPod health label is not Schedulable")
    annotations = spare.get("annotations") or {}
    if annotations.get(SPARE_RESERVATION_ANNOTATION):
        errors.append("spare node already carries a reservation")
    if annotations.get(SPARE_RESERVED_AT_ANNOTATION):
        errors.append("spare node already carries a reserved-at timestamp")
    if annotations.get(SPARE_POOL_STATE_ANNOTATION) not in {
        None,
        POOL_STATE_AVAILABLE,
    }:
        errors.append("spare pool state is not AVAILABLE")
    if any(item.get("key") == QUARANTINE_TAINT for item in spare.get("taints") or []):
        errors.append("spare node carries the quarantine taint")
    if any(annotations.get(key) for key in OWNERSHIP_ANNOTATIONS):
        errors.append("spare node carries workflow ownership annotations")
    return errors


def preflight_errors(
    *,
    spare: dict[str, Any],
    spare_node: str,
    declared_spares: list[str],
    workloads: list[dict[str, Any]],
    executor_env: list[dict[str, Any]],
    executor_probes: list[dict[str, Any]],
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    incident_lookup: dict[str, Any],
    predecessor_valid: bool,
    tests_passed: bool,
) -> list[str]:
    errors = spare_refusals(spare)
    if declared_spares != [spare_node]:
        errors.append(
            f"declared spare set {declared_spares} is not exactly [{spare_node}]"
        )
    if workloads:
        errors.append("spare node has a non-system workload")
    if not executor_env:
        errors.append("no Ready cluster executor Pod was found")
    for item in executor_env:
        if item.get("spare_failover") != "true":
            errors.append(
                f"executor {item.get('pod')} runs with "
                "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER != true; the spare "
                "coordinator and therefore the reservation sweep are not "
                "constructed on this deployment (a deployment fact, not a case "
                "defect)"
            )
        if item.get("allow_replace") != "false":
            errors.append(
                f"executor {item.get('pod')} does not pin "
                "GPU_FAULT_ALLOW_HYPERPOD_REPLACE=false"
            )
    if not executor_probes:
        errors.append("no executor claim-state breadcrumb was read")
    for probe in executor_probes:
        if counter_of(probe.get("claim_state")) is None:
            errors.append(
                f"executor {probe.get('pod')} breadcrumb carries no "
                f"counters.{RECLAIM_COUNTER}; the deployed executor predates "
                "ARCH-A4b (a deployment fact, not a case defect)"
            )
    if int(queue.get("depth") or 0):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if incident_lookup.get("found") is not False:
        errors.append("the synthetic incident id already exists in the store")
    if not predecessor_valid:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    if not tests_passed:
        errors.append("focused regression tests failed")
    return errors


# --------------------------------------------------------------------------- #
# Injection and reclaim
# --------------------------------------------------------------------------- #
def injection_errors(
    snapshot: dict[str, Any],
    *,
    incident_id: str,
    reserved_at: str,
) -> list[str]:
    """The reservation landed as written, and the spare stayed cordoned."""

    errors: list[str] = []
    annotations = snapshot.get("annotations") or {}
    if annotations.get(SPARE_RESERVATION_ANNOTATION) != incident_id:
        errors.append("the synthetic reservation annotation was not written")
    if annotations.get(SPARE_RESERVED_AT_ANNOTATION) != reserved_at:
        errors.append("the back-dated reserved-at annotation was not written")
    if annotations.get(SPARE_POOL_STATE_ANNOTATION) != POOL_STATE_ALLOCATED:
        errors.append("the pool state was not written as ALLOCATED")
    if not snapshot.get("unschedulable"):
        errors.append(
            "the spare was uncordoned by the injection; it must stay cordoned"
        )
    return errors


def reclaim_errors(snapshot: dict[str, Any], *, incident_id: str) -> list[str]:
    """What the executor's release leaves on the node."""

    errors: list[str] = []
    annotations = snapshot.get("annotations") or {}
    reservation = annotations.get(SPARE_RESERVATION_ANNOTATION)
    if reservation:
        who = (
            "the synthetic"
            if reservation == incident_id
            else f"another ({reservation})"
        )
        errors.append(f"{who} reservation is still on the spare after the sweep window")
    if annotations.get(SPARE_RESERVED_AT_ANNOTATION):
        errors.append("reserved-at annotation was not cleared with the reservation")
    if annotations.get(SPARE_POOL_STATE_ANNOTATION) != POOL_STATE_AVAILABLE:
        errors.append(
            f"pool state is {annotations.get(SPARE_POOL_STATE_ANNOTATION)!r}, "
            f"not {POOL_STATE_AVAILABLE}"
        )
    if not snapshot.get("unschedulable"):
        errors.append("the spare is schedulable; a reclaimed spare must stay cordoned")
    return errors


def reclaimed(snapshot: dict[str, Any]) -> bool:
    """Whether a timeline sample already shows the release."""

    annotations = snapshot.get("annotations") or {}
    return (
        not annotations.get(SPARE_RESERVATION_ANNOTATION)
        and annotations.get(SPARE_POOL_STATE_ANNOTATION) == POOL_STATE_AVAILABLE
    )


def timeline_errors(timeline: list[dict[str, Any]], *, incident_id: str) -> list[str]:
    """Every sample keeps the spare cordoned; the last one shows the reclaim."""

    errors: list[str] = []
    if not timeline:
        return ["no reclaim timeline was recorded"]
    for index, sample in enumerate(timeline):
        if not (sample.get("snapshot") or {}).get("unschedulable"):
            errors.append(f"timeline sample {index} shows the spare schedulable")
    errors.extend(
        reclaim_errors(timeline[-1].get("snapshot") or {}, incident_id=incident_id)
    )
    return errors


def log_errors(lines: list[str], *, node: str, incident_id: str) -> list[str]:
    """The executor logged exactly this reclaim, and no other."""

    reclaims = [line for line in lines if RECLAIM_LOG_PREFIX in line]
    ours = [
        line
        for line in reclaims
        if f"node={node} " in line and f"incident={incident_id} " in line
    ]
    errors: list[str] = []
    if not ours:
        errors.append(
            f"executor logs carry no '{RECLAIM_LOG_PREFIX} node={node} "
            f"incident={incident_id}' line"
        )
    if len(ours) > 1:
        errors.append(
            f"executor logs reclaimed the synthetic reservation {len(ours)} times"
        )
    if any(line not in ours for line in reclaims):
        errors.append(
            "executor logs reclaimed a reservation this case did not write: "
            + "; ".join(line[:200] for line in reclaims if line not in ours)
        )
    if any("sweep failed" in line for line in lines):
        errors.append("the spare reservation sweep reported a failure")
    return errors


def counter_errors(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    reclaimed_at: datetime,
) -> list[str]:
    """``spare_reservations_reclaimed_total`` across the executor replicas.

    Each replica runs its own sweep and its own counter, and the breadcrumb is
    rewritten on every successful claim round-trip (``run_once``), which the
    claim loop makes every ``poll_seconds`` even on an empty queue. The sweep
    runs right after ``run_once``, so a replica whose breadcrumb was rewritten
    after the reclaim moment must show the increment. A replica whose
    breadcrumb predates the reclaim (claims failing, Pod restarted) cannot be
    judged for the increment and is only required not to go backwards.
    """

    errors: list[str] = []
    before_by_pod = {str(item.get("pod")): item for item in before}
    after_by_pod = {str(item.get("pod")): item for item in after}
    if set(before_by_pod) != set(after_by_pod):
        errors.append(
            "the executor replica set changed during the case: "
            f"{sorted(before_by_pod)} -> {sorted(after_by_pod)}"
        )
    increments = 0
    judged = 0
    for pod, first in before_by_pod.items():
        second = after_by_pod.get(pod)
        if second is None:
            continue
        start = counter_of(first.get("claim_state"))
        end = counter_of(second.get("claim_state"))
        if start is None or end is None:
            errors.append(f"executor {pod} breadcrumb lost counters.{RECLAIM_COUNTER}")
            continue
        if end < start:
            errors.append(
                f"executor {pod} {RECLAIM_COUNTER} went backwards: {start} -> {end}"
            )
            continue
        claimed = breadcrumb_claimed_at(second.get("claim_state"))
        if claimed is not None and claimed > reclaimed_at:
            judged += 1
            increments += end - start
    if judged and increments < 1:
        errors.append(
            f"no executor replica moved {RECLAIM_COUNTER} although its breadcrumb "
            "was rewritten after the reclaim"
        )
    if increments > 1:
        errors.append(
            f"{RECLAIM_COUNTER} moved by {increments} across replicas for one "
            "reservation"
        )
    return errors


# --------------------------------------------------------------------------- #
# Store, node and cleanup
# --------------------------------------------------------------------------- #
def store_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The synthetic incident never existed; the sweep created nothing."""

    errors: list[str] = []
    if before.get("found") is not False:
        errors.append("the synthetic incident existed before the injection")
    if after.get("found") is not False:
        errors.append("a control record appeared for the synthetic incident")
    if after.get("workflows"):
        errors.append("a workflow appeared for the synthetic incident")
    return errors


def final_errors(baseline: dict[str, Any], final: dict[str, Any]) -> list[str]:
    """The tracked annotations and cordon are back to what was declared."""

    errors: list[str] = []
    base_annotations = baseline.get("annotations") or {}
    final_annotations = final.get("annotations") or {}
    for key in TRACKED_ANNOTATIONS:
        if base_annotations.get(key) != final_annotations.get(key):
            errors.append(
                f"{key} is {final_annotations.get(key)!r} after cleanup, baseline "
                f"{base_annotations.get(key)!r}"
            )
    if bool(final.get("unschedulable")) != bool(baseline.get("unschedulable")):
        errors.append("spare cordon state differs from the declared baseline")
    if (final.get("labels") or {}).get(SPARE_LABEL) != (
        baseline.get("labels") or {}
    ).get(SPARE_LABEL):
        errors.append("spare label changed during the case")
    if final.get("taints") != baseline.get("taints"):
        errors.append("spare taints changed during the case")
    return errors


def other_nodes_errors(
    before: list[dict[str, Any]], after: list[dict[str, Any]], *, spare_node: str
) -> list[str]:
    """No other GPU node's scheduling state moved."""

    def view(nodes: list[dict[str, Any]]) -> dict[str, tuple[Any, Any]]:
        return {
            str(item.get("name")): (item.get("unschedulable"), item.get("taints"))
            for item in nodes
            if item.get("name") != spare_node
        }

    if view(before) != view(after):
        return ["another GPU node's scheduling state changed during the case"]
    return []
