"""GF-GENERAL-LIVE-028/029: what the product did with a failed attempt's restart.

A plain failure (no trusted marker) is decided as one budgeted restart:
PLAN_CREATED and a recovery workflow of one RESTART_WORKLOAD, chained behind
the passive containment's FREEZE_EVIDENCE/STOP_WORKLOADS (``service._decide``
-> ``planner.without_hardware_evidence``). The decision is recorded long
before the restart runs -- the recovery workflow is dispatched once the
containment ended and its budget is reserved at its claim
(``execution/restart_budget_preflight.reserve_restart_budgets``) -- so a
variant that graded the first read showing a decision graded a restart that
had not happened: a ledger row at ``restart_count`` 0 (a reservation not yet
taken, or released unattempted) passed "consumed once", and an empty recovery
history passed "stop then restart".

The variants now wait for the outcome -- every recovery workflow of the
attempt terminal -- within the product's own step caps, and then read what the
product recorded. Within budget: RESTART_WORKLOAD SUCCEEDED once, recording
the copy's ``restart_attempt_id``, and the ledger holds exactly the one
reservation of that recovery workflow. With the budget spent the preflight
withholds the restart (``withhold_exhausted_restart``): one FAILED
RESTART_WORKLOAD whose details carry ``reason`` RESTART_BUDGET_EXHAUSTED,
``restart_count``, ``restart_budget`` and the ``notification_id`` of the one
BUDGET_EXHAUSTED notice (deduplication key
``<cluster>/<job>/restart-budget-exhausted/<budget>``,
``notifications/restart_guard.py``); no reservation is taken and the recovery
workflow ends FAILED. An outcome that did not settle within the bound is
INCONCLUSIVE, never graded.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpu_fault.execution.restart_budget_preflight import BUDGET_EXHAUSTED_REASON
from scripts.e2e.regional.general_live_common import STORE_POLL_SECONDS
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_terminal_replay import restart_attempt_of
from scripts.e2e.regional.general_live_verdicts import executions, operations
from scripts.e2e.regional.general_live_watcher_restart import (
    recovery_settled,
    settle_bound,
    worker_restart_errors,
)

#: The deduplication-key segment of the product's budget-exhausted notice.
BUDGET_NOTICE = "/restart-budget-exhausted/"


def outcome_budget_seconds() -> float:
    """How long after the decision the recovery may take to settle: the
    product's caps of one WAITING STOP_WORKLOADS and RESTART_WORKLOAD, a
    dispatcher poll per claim, the watcher poll and the margin --
    ``settle_bound`` without the watcher replacement only 029
    ``worker_restart`` pays."""

    bound = settle_bound()
    return bound["total_seconds"] - bound["replacement_timeout_seconds"]


def wait_restart_outcome(
    ctx: VariantContext,
    read: Callable[[], dict[str, Any]],
    attempt_id: str,
    *,
    name: str,
) -> dict[str, Any]:
    """The attempt's records once every recovery workflow is terminal.

    INCONCLUSIVE when the attempt has no recovery workflow, or one still
    running, at the end of ``outcome_budget_seconds``: the restart was neither
    seen to run nor seen to be withheld.
    """

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = read()
        return last if recovery_settled(last, attempt_id) else None

    budget = outcome_budget_seconds()
    value = ctx.wait_until(
        accept, timeout_seconds=budget, poll_seconds=STORE_POLL_SECONDS, name=name
    )
    if value is None:
        entry = (last.get("attempts") or {}).get(attempt_id) or {}
        decision = (entry.get("decision") or {}).get("status")
        statuses = [
            f"{(item.get('workflow') or {}).get('request_id')} "
            f"{(item.get('workflow') or {}).get('status')}"
            for item in entry.get("recovery") or []
        ]
        raise VariantInconclusive(
            f"the recovery of {attempt_id} did not settle within {budget:.0f} s "
            f"of its decision {decision} (recovery workflows "
            f"{statuses or 'none'}): what the product did with the restart was "
            "not observed"
        )
    return value


def budget_errors(records: dict[str, Any], *, consumed: int) -> list[str]:
    """The job's restart ledger shows exactly ``consumed`` restarts.

    ``restart_count`` equals the reservation ids and ``consumed``: a count of
    0 is not "consumed once", it is a reservation never taken or released
    without a restart. No row is right only when nothing was consumed.
    """

    budget = records.get("restart_budget")
    if budget is None:
        return [] if consumed == 0 else ["no restart-budget ledger row for the job"]
    errors = []
    count = int(budget.get("restart_count", 0))
    reservations = budget.get("reservation_ids") or []
    if count < 0:
        errors.append("restart_count went negative")
    if count != len(reservations):
        errors.append("restart_count does not match the reservation ids")
    if count != consumed:
        errors.append(f"restart_count is {count}, expected exactly {consumed}")
    return errors


def restart_once_errors(entry: dict[str, Any], records: dict[str, Any]) -> list[str]:
    """One budgeted restart, shown: PLAN_CREATED, one recovery workflow that
    SUCCEEDED with one RESTART_WORKLOAD SUCCEEDED that recorded the copy's
    attempt id, one containment STOP, and the ledger holding exactly that
    workflow's one reservation (``worker_restart_errors``)."""

    errors = list(worker_restart_errors(entry, records))
    if restart_attempt_of(entry) is None:
        errors.append("the product's RESTART_WORKLOAD recorded no restart_attempt_id")
    return errors


def restart_runs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Every RESTART_WORKLOAD execution of the attempt's recovery workflows."""

    return [
        execution
        for item in entry.get("recovery") or []
        for execution in executions(item.get("workflow") or {}, "RESTART_WORKLOAD")
    ]


def withheld_errors(entry: dict[str, Any], *, budget: int) -> list[str]:
    """The exhausted budget withheld the restart -- and only the restart.

    A recovery that planned no RESTART_WORKLOAD never reached the budget gate
    (a profile that cannot restart escalates instead): INCONCLUSIVE. Otherwise
    exactly one FAILED execution for RESTART_BUDGET_EXHAUSTED at the job's
    budget, no restart and no new attempt; the containment's STOP_WORKLOADS
    still SUCCEEDED (the chain beside the restart ran).
    """

    if not any(
        "RESTART_WORKLOAD" in operations(item.get("workflow") or {})
        for item in entry.get("recovery") or []
    ):
        raise VariantInconclusive(
            "no recovery workflow of the attempt planned RESTART_WORKLOAD: the "
            "restart budget gate was never reached"
        )
    runs = restart_runs(entry)
    errors = []
    if any(item.get("status") == "SUCCEEDED" for item in runs):
        errors.append("RESTART_WORKLOAD succeeded despite the exhausted budget")
    withheld = [
        item
        for item in runs
        if item.get("status") == "FAILED"
        and (item.get("details") or {}).get("reason") == BUDGET_EXHAUSTED_REASON
    ]
    if len(withheld) != 1:
        errors.append(
            f"expected one RESTART_WORKLOAD withheld for {BUDGET_EXHAUSTED_REASON}, "
            "found "
            + (
                ", ".join(
                    f"{item.get('status')} {(item.get('details') or {}).get('reason')}"
                    for item in runs
                )
                or "no execution"
            )
        )
    else:
        details = withheld[0].get("details") or {}
        if (details.get("restart_budget"), details.get("restart_count")) != (
            budget,
            budget,
        ):
            errors.append(
                f"the withheld restart read budget {details.get('restart_budget')} "
                f"with {details.get('restart_count')} used, expected {budget} of "
                f"{budget}"
            )
    if restart_attempt_of(entry) is not None:
        errors.append("a restart attempt was recorded despite the exhausted budget")
    stops = [
        execution
        for workflow in entry.get("containment_workflows") or []
        for execution in executions(workflow, "STOP_WORKLOADS")
        if execution.get("status") == "SUCCEEDED"
    ]
    if not stops:
        errors.append("the containment's STOP_WORKLOADS never succeeded")
    return errors


def budget_notices(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The attempt's incidents' budget-exhausted notifications."""

    return [
        item
        for item in entry.get("notifications") or []
        if BUDGET_NOTICE in str(item.get("deduplication_key") or "")
    ]


def budget_notice_errors(
    entry: dict[str, Any], *, cluster_id: str, job_id: str, budget: int
) -> list[str]:
    """Exactly one budget-exhausted notice, the job's own, and the one the
    withheld RESTART_WORKLOAD names (``details.notification_id``)."""

    notices = budget_notices(entry)
    key = f"{cluster_id}/{job_id}{BUDGET_NOTICE}{budget}"
    if len(notices) != 1:
        return [
            f"{len(notices)} budget-exhausted notifications for the attempt's "
            f"incidents, expected exactly one ({key}): "
            f"{[item.get('deduplication_key') for item in notices]}"
        ]
    errors = []
    notice = notices[0]
    if notice.get("deduplication_key") != key:
        errors.append(
            f"the budget notification is keyed {notice.get('deduplication_key')}, "
            f"expected {key}"
        )
    named = {
        (item.get("details") or {}).get("notification_id")
        for item in restart_runs(entry)
        if (item.get("details") or {}).get("reason") == BUDGET_EXHAUSTED_REASON
    }
    if notice.get("notification_id") not in named:
        errors.append(
            f"the withheld RESTART_WORKLOAD names notification {sorted(map(str, named))}, "
            f"not the attempt's budget notice {notice.get('notification_id')}"
        )
    return errors


__all__ = [
    "BUDGET_NOTICE",
    "budget_errors",
    "budget_notice_errors",
    "budget_notices",
    "outcome_budget_seconds",
    "restart_once_errors",
    "restart_runs",
    "wait_restart_outcome",
    "withheld_errors",
]
