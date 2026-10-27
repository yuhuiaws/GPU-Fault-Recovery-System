"""GF-GENERAL-LIVE-029 ``worker_restart``: the completion watcher is replaced
while a restart-budget claim is open.

The claim is the executor's: when it claims the attempt's recovery workflow,
``restart_budget_preflight.reserve_restart_budgets`` reserves the pending
RESTART_WORKLOAD step on the job's durable ledger (``<workflow>/<index>/
RESTART_WORKLOAD``), and the reservation stays until the step is terminal (a
successful restart keeps it as the consumed budget; an unattempted one is
released). The window this variant uses is that open claim: the reservation
exists and the restart step has no terminal execution yet.

How long the claim stays open is the product's control loop, not the
replica's start. The claim mints the data-plane command in the tick that
reserved (``regional.py``), the cluster executor is woken by the command's
NOTIFY, creates the restart copy and returns SUCCEEDED in that same call --
``adapters/kubernetes/workload_operations._set_workloads`` reads no Pod of the
copy; it waits neither for scheduling nor for Running -- and the command's
terminal status wakes the dispatcher (``dispatcher_wakeups
.WAKING_REMOTE_COMMAND_STATUSES``), whose executor records the step
SUCCEEDED. Live that is one to a few seconds (the HyperPod run of 2026-10-27
opened and closed it between two records reads 10 s apart), shorter than one
records read (a ``kubectl exec`` probe, about 5 s), so no poll interval and no
start barrier in the job's Pod template can widen it or hit it on purpose from
the harness. The runner therefore polls as before, dates the window from the
product's own records -- the recovery workflow's first CLAIM event to the
terminal RESTART_WORKLOAD record's ``updated_at`` (``claim_window``) -- and
judges its own delete against it (``delete_timing_errors``): a claim already
closed at the first read, or a delete the settled records place outside the
window, is INCONCLUSIVE with the dated window in the detail, never a verdict
on the product. A deterministic window needs the product to hold the step
after the submission (a WAITING with ``restart_submitted: True`` until the
copy's Pods run); only then would a start barrier on the restart copy set the
window's length.

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
restart attempt, and a write from the replacement (its coverage heartbeat once
the cluster is idle, or its observation of the restart attempt).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
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
#: The pause between two records reads while waiting for the claim to open. A
#: read is a ``kubectl exec`` probe of about 5 s, and the claim itself is open
#: for one control-loop round trip (see the module docstring), so no value here
#: reaches the window; the poll is kept at the store's comfortable rate and the
#: window is dated from the product's records instead.
CLAIM_POLL_SECONDS = 5
OPEN = "open"
CLOSED = "closed"
NONE = "none"
#: The step statuses that end a RESTART_WORKLOAD claim.
TERMINAL_STEPS = frozenset({"SUCCEEDED", "FAILED"})


def claim_state(records: Mapping[str, Any], attempt_id: str) -> str:
    """``open`` while a reservation exists and no RESTART_WORKLOAD ended,
    ``closed`` once one did, ``none`` before the claim."""

    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    statuses = [
        str(item.get("status"))
        for recovery in entry.get("recovery") or []
        for item in executions(dict(recovery.get("workflow") or {}), "RESTART_WORKLOAD")
    ]
    if any(status in TERMINAL_STEPS for status in statuses):
        return CLOSED
    budget = records.get("restart_budget") or {}
    if (budget.get("reservation_ids") or []) and entry.get("recovery"):
        return OPEN
    return NONE


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def claim_window(records: Mapping[str, Any], attempt_id: str) -> dict[str, Any]:
    """The open claim as the product's own records date it.

    ``reserved_at`` is the recovery workflow's first CLAIM event: the claim
    preparation reserves the restart in that tick
    (``restart_budget_preflight.reserve_restart_budgets``). ``dispatched_at``
    is the first STEP_ATTEMPT event of RESTART_WORKLOAD, the step WAITING on
    the data-plane command. ``closed_at`` is the terminal RESTART_WORKLOAD
    record's ``updated_at`` -- one record per step identity, replaced on every
    attempt (``step_bounds.record_attempt``), so the terminal record's update
    is the moment the claim closed. ``open_seconds`` is closed minus reserved.
    A field the records cannot date is None.
    """

    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    claims: list[datetime] = []
    dispatches: list[datetime] = []
    closes: list[datetime] = []
    for recovery in entry.get("recovery") or []:
        workflow = dict(recovery.get("workflow") or {})
        for event in workflow.get("events") or []:
            at = _parse_time(event.get("at"))
            if at is None:
                continue
            if event.get("kind") == "CLAIM":
                claims.append(at)
            elif (
                event.get("kind") == "STEP_ATTEMPT"
                and event.get("operation") == "RESTART_WORKLOAD"
            ):
                dispatches.append(at)
        for item in executions(workflow, "RESTART_WORKLOAD"):
            if item.get("status") in TERMINAL_STEPS:
                at = _parse_time(item.get("updated_at"))
                if at is not None:
                    closes.append(at)
    reserved = min(claims, default=None)
    closed = max(closes, default=None)
    budget = records.get("restart_budget")
    return {
        "reserved_at": _iso(reserved),
        "dispatched_at": _iso(min(dispatches, default=None)),
        "closed_at": _iso(closed),
        "open_seconds": (
            round((closed - reserved).total_seconds(), 3)
            if reserved is not None and closed is not None
            else None
        ),
        "ledger_updated_at": (
            budget.get("updated_at") if isinstance(budget, dict) else None
        ),
    }


def _window_text(window: Mapping[str, Any]) -> str:
    if window.get("reserved_at") is None or window.get("closed_at") is None:
        return "the product's records do not date the claim"
    return (
        f"the product's records date the claim open from {window['reserved_at']} "
        f"to {window['closed_at']}, {window['open_seconds']} s"
    )


def delete_timing_errors(
    window: Mapping[str, Any], document: Mapping[str, Any]
) -> list[str]:
    """Why the fenced delete did not land inside the product-dated window.

    The delete's API round trip is bracketed by ``deleted_at`` (the runner's
    clock before the request) and ``deletion_returned_at`` (after it); the
    Pod went at some instant in between. The condition the variant claims --
    the watcher replaced while the claim was open -- holds only when that whole
    bracket lies inside the window; a bracket that straddles a boundary is as
    undecided as one outside it, and says so.
    """

    reserved = _parse_time(window.get("reserved_at"))
    closed = _parse_time(window.get("closed_at"))
    if reserved is None or closed is None:
        return [
            "the product's records do not date the claim (no CLAIM event on the "
            "recovery workflow or no terminal RESTART_WORKLOAD record)"
        ]
    before = _parse_time(document.get("deleted_at"))
    after = _parse_time(document.get("deletion_returned_at")) or before
    if before is None or after is None:
        return ["the restart document does not date the delete"]
    bracket = f"the delete request ran {before.isoformat()} -> {after.isoformat()}"
    if after < reserved:
        return [
            f"{bracket}, {(reserved - after).total_seconds():.1f} s before the "
            f"claim opened; {_window_text(window)}"
        ]
    if before > closed:
        return [
            f"{bracket}, {(before - closed).total_seconds():.1f} s after the "
            f"claim closed; {_window_text(window)}"
        ]
    if before < reserved or after > closed:
        return [f"{bracket} and straddles the window's edge; {_window_text(window)}"]
    return []


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
    and the variant is INCONCLUSIVE, not a verdict on the product. The detail
    then carries the window as the product's records date it and how long each
    records read took, so the record says by how much the harness missed.
    """

    last: dict[str, Any] = {}
    seen_closed = False
    read_seconds: list[float] = []

    def accept() -> dict[str, Any] | None:
        nonlocal last, seen_closed
        started = ctx.clock()
        last = read()
        read_seconds.append(round((ctx.clock() - started).total_seconds(), 3))
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
        window = claim_window(last, attempt_id)
        ctx.evidence.write(
            f"{name}-window",
            {**window, "read_seconds": read_seconds, "first_read_state": CLOSED},
        )
        longest = max(read_seconds, default=0.0)
        raise VariantInconclusive(
            "the restart claim closed before the watcher could be replaced "
            f"(RESTART_WORKLOAD already ended when first read; {_window_text(window)}; "
            f"a records read takes up to {longest:.1f} s plus {CLAIM_POLL_SECONDS} s "
            "between reads)"
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
    # The Pod went between these two readings; ``delete_timing_errors`` judges
    # the whole bracket against the product-dated claim window.
    returned_at = ctx.clock()
    replacement = wait_replacement(
        ctx, str(pod["uid"]), timeout_seconds=REPLACEMENT_TIMEOUT_SECONDS
    )
    document = {
        "deleted": pod,
        "deleted_at": deleted_at.isoformat(),
        "deletion_returned_at": returned_at.isoformat(),
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
    "CLAIM_POLL_SECONDS",
    "CLOSED",
    "NONE",
    "OPEN",
    "REPLACEMENT_TIMEOUT_SECONDS",
    "TERMINAL_STEPS",
    "attempt_once_errors",
    "claim_state",
    "claim_window",
    "delete_timing_errors",
    "delete_watcher_pod",
    "recovery_settled",
    "replacement_errors",
    "restart_watcher",
    "wait_open_claim",
    "wait_replacement",
    "worker_restart_errors",
]
