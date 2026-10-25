"""GF-GENERAL-LIVE-029 ``worker_restart``: the completion watcher is replaced
while a restart-budget claim is open.

The claim is the executor's: when it claims the attempt's recovery workflow,
``restart_budget_preflight.reserve_restart_budgets`` reserves the pending
RESTART_WORKLOAD step on the job's durable ledger (``<workflow>/<index>/
RESTART_WORKLOAD``), and the reservation stays until the step is terminal (a
successful restart keeps it as the consumed budget; an unattempted one is
released). The window this variant uses is that open claim: the reservation
exists and the restart step has no terminal execution yet -- the adapter has
submitted the replica and WAITS for it to run.

Inside that window the runner deletes the one completion-watcher Pod, fenced on
its UID and resourceVersion (``DeleteOptions.preconditions``), with no grace, so
the Deployment (``replicas: 1``, Recreate) replaces it. The replacement restores
its attempt state from the ``-outbox-active`` ConfigMap, replays the write-ahead
outbox and re-observes every attempt from the Pods; the product's claim is that
none of that re-decides the terminal, re-reserves or releases the restart, or
re-sends what was sent. The verdict reads the product's records only: the
ledger (count 1 == one reservation, belonging to the recovery workflow), the
recovery workflow (one, with one successful RESTART_WORKLOAD), the containment
(one successful STOP_WORKLOADS), the decision, the notifications of the
attempt's incidents (none twice), the watcher's observation of exactly one
restart attempt, and a coverage heartbeat from the replacement.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_network_isolation import (
    WATCHER_APP,
    single_action_errors,
    watcher_pod,
)
from scripts.e2e.regional.general_live_verdicts import TERMINAL_WORKFLOWS, executions
from scripts.e2e.regional.regional_commands import RegionalFixtureError

#: The Deployment's startupProbe gives a cold watcher ten minutes.
REPLACEMENT_TIMEOUT_SECONDS = 600
REPLACEMENT_POLL_SECONDS = 10
#: How often the records are read while waiting for the claim to open; the
#: restart's WAITING phase lasts as long as the replica takes to run.
CLAIM_POLL_SECONDS = 5
OPEN = "open"
CLOSED = "closed"
NONE = "none"


def claim_state(records: Mapping[str, Any], attempt_id: str) -> str:
    """``open`` while a reservation exists and no RESTART_WORKLOAD ended,
    ``closed`` once one did, ``none`` before the claim."""

    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    statuses = [
        str(item.get("status"))
        for recovery in entry.get("recovery") or []
        for item in executions(dict(recovery.get("workflow") or {}), "RESTART_WORKLOAD")
    ]
    if any(status in {"SUCCEEDED", "FAILED"} for status in statuses):
        return CLOSED
    budget = records.get("restart_budget") or {}
    if (budget.get("reservation_ids") or []) and entry.get("recovery"):
        return OPEN
    return NONE


def wait_open_claim(
    ctx: VariantContext,
    read: Callable[[], dict[str, Any]],
    attempt_id: str,
    *,
    timeout_seconds: float,
    name: str = "open-claim",
) -> dict[str, Any]:
    """The records at the first poll that shows the claim open.

    A claim that is already closed when first seen means the restart completed
    before the watcher could be replaced: the condition was never established
    and the variant is INCONCLUSIVE, not a verdict on the product.
    """

    last: dict[str, Any] = {}
    seen_closed = False

    def accept() -> dict[str, Any] | None:
        nonlocal last, seen_closed
        last = read()
        state = claim_state(last, attempt_id)
        if state == CLOSED:
            seen_closed = True
            return last
        return last if state == OPEN else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=CLAIM_POLL_SECONDS,
        name=name,
    )
    if seen_closed:
        raise VariantInconclusive(
            "the restart claim closed before the watcher could be replaced "
            "(RESTART_WORKLOAD already ended when first read)"
        )
    if value is None:
        raise VariantInconclusive(
            f"no open restart claim appeared within {timeout_seconds:.0f} s "
            f"(claim state {claim_state(last, attempt_id)})"
        )
    return value


def recovery_settled(records: Mapping[str, Any], attempt_id: str) -> bool:
    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    recovery = entry.get("recovery") or []
    return bool(recovery) and all(
        str((item.get("workflow") or {}).get("status")) in TERMINAL_WORKFLOWS
        for item in recovery
    )


def delete_watcher_pod(regional: Any, pod: Mapping[str, Any]) -> None:
    """Delete ``pod`` fenced on its UID and resourceVersion, no grace."""

    if not pod.get("uid") or not pod.get("resource_version"):
        raise RegionalFixtureError("the watcher Pod's UID/resourceVersion is unknown")
    namespace = regional.settings.namespace
    require_action_time(120)
    regional.kubectl(
        "gpu",
        "delete",
        "--raw",
        f"/api/v1/namespaces/{namespace}/pods/{pod['name']}",
        "-f",
        "-",
        input_text=json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {
                    "uid": pod["uid"],
                    "resourceVersion": pod["resource_version"],
                },
                "propagationPolicy": "Foreground",
                "gracePeriodSeconds": 0,
            }
        ),
    )


def wait_replacement(
    ctx: VariantContext, old_uid: str, *, timeout_seconds: float
) -> dict[str, Any]:
    """The Ready replacement watcher Pod (a new UID) once the old one is gone."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        try:
            pod = watcher_pod(ctx.regional)
        except Exception as exc:  # noqa: BLE001 - the singleton is mid-replacement
            last = {"error": f"{type(exc).__name__}: {exc}"}
            return None
        last = pod
        return pod if pod["uid"] != old_uid else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=REPLACEMENT_POLL_SECONDS,
        name="watcher-replacement",
    )
    if value is None:
        raise RegionalFixtureError(
            f"no Ready replacement {WATCHER_APP} Pod within {timeout_seconds:.0f} s: {last}"
        )
    return value


def restart_watcher(ctx: VariantContext) -> dict[str, Any]:
    """Replace the watcher: the record of the deleted Pod and its successor."""

    pod = watcher_pod(ctx.regional)
    deleted_at = ctx.clock()
    delete_watcher_pod(ctx.regional, pod)
    replacement = wait_replacement(
        ctx, str(pod["uid"]), timeout_seconds=REPLACEMENT_TIMEOUT_SECONDS
    )
    document = {
        "deleted": pod,
        "deleted_at": deleted_at.isoformat(),
        "replacement": replacement,
        "ready_at": ctx.clock().isoformat(),
    }
    ctx.evidence.write("watcher-restart", document)
    return document


# --------------------------------------------------------------------------- #
# graders (pure)
# --------------------------------------------------------------------------- #
def replacement_errors(document: Mapping[str, Any]) -> list[str]:
    deleted = document.get("deleted") or {}
    replacement = document.get("replacement") or {}
    errors = []
    if not replacement.get("uid") or replacement.get("uid") == deleted.get("uid"):
        errors.append("the watcher Pod was not replaced by a new UID")
    if not replacement.get("ready"):
        errors.append("the replacement watcher Pod is not Ready")
    return errors


def worker_restart_errors(
    entry: Mapping[str, Any], records: Mapping[str, Any]
) -> list[str]:
    """The restart happened exactly once and the ledger kept exactly its claim."""

    errors = single_action_errors(entry, records)
    decision = entry.get("decision") or {}
    if decision and decision.get("status") != "PLAN_CREATED":
        errors.append(
            f"the terminal decision was {decision.get('status')}, expected a budgeted restart"
        )
    recovery = entry.get("recovery") or []
    if len(recovery) != 1:
        errors.append(f"expected one recovery workflow, found {len(recovery)}")
    workflows = [dict(item.get("workflow") or {}) for item in recovery]
    restarts = sum(
        1
        for workflow in workflows
        for item in executions(workflow, "RESTART_WORKLOAD")
        if item.get("status") == "SUCCEEDED"
    )
    if restarts != 1:
        errors.append(f"RESTART_WORKLOAD succeeded {restarts} times, expected once")
    for workflow in workflows:
        if workflow.get("status") != "SUCCEEDED":
            errors.append(
                f"recovery workflow {workflow.get('request_id')} ended {workflow.get('status')}"
            )
    budget = records.get("restart_budget")
    if not isinstance(budget, dict):
        errors.append("no restart-budget ledger row for the job")
        return errors
    count = int(budget.get("restart_count", 0))
    reservations = [str(item) for item in budget.get("reservation_ids") or []]
    if count == 0 or not reservations:
        errors.append(
            "the restart reservation was lost (ledger shows no consumed restart)"
        )
    elif count > 1 or len(reservations) > 1:
        errors.append(
            f"the ledger shows {count} restarts / {len(reservations)} reservations: "
            "a double decrement"
        )
    request_ids = {str(workflow.get("request_id")) for workflow in workflows}
    if reservations and not any(
        reservation.startswith(f"{request_id}/")
        for request_id in request_ids
        for reservation in reservations
    ):
        errors.append(
            f"the ledger's reservation {reservations} does not belong to the recovery "
            f"workflow {sorted(request_ids)}"
        )
    return errors


def attempt_once_errors(
    observations: Sequence[Mapping[str, Any]], *, job_id: str, source_attempt_id: str
) -> list[str]:
    """The watcher observed exactly one restart attempt of the job."""

    attempts = sorted(
        {
            str(item.get("attempt_id"))
            for item in observations
            if item.get("job_id") == job_id and item.get("attempt_id")
        }
    )
    if source_attempt_id not in attempts:
        return [f"the watcher never published an observation of {source_attempt_id}"]
    restarted = [item for item in attempts if item != source_attempt_id]
    if not restarted:
        return ["no restart attempt of the job was observed by the watcher"]
    if len(restarted) > 1:
        return [f"{len(restarted)} restart attempts were observed: {restarted}"]
    return []


__all__ = [
    "CLOSED",
    "NONE",
    "OPEN",
    "REPLACEMENT_TIMEOUT_SECONDS",
    "attempt_once_errors",
    "claim_state",
    "delete_watcher_pod",
    "recovery_settled",
    "replacement_errors",
    "restart_watcher",
    "wait_open_claim",
    "wait_replacement",
    "worker_restart_errors",
]
