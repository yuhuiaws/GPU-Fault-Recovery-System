"""The product's own recorded terminal, redelivered (028 ``old_terminal``,
029 ``duplicate_event``).

``TERMINAL_REPLAY_PROBE`` runs in an API Pod under the product's interpreter:
it reads the stored TerminalEvent of a *test* attempt (``gl28``/``gl29``
prefixes only) and hands it to ``CompletionService.handle_terminal`` -- the
entry the ``/v1/attempts/terminal`` route calls for the watcher -- so the
product receives the SAME terminal a second time. Nothing is hand-made: the
event is the product's own row. The verdict helpers read the product's
completion records before and after: a redelivery must come back as the
original decision flagged ``duplicate=true`` (``replay_errors``), leave the
attempt's decision row, restart workflows, budget ledger and notifications as
they were (``unchanged_errors``) and, while a restart copy runs, touch nothing
of the new attempt (``untouched_errors``).
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_verdicts import operations

#: The replay probe's test-attempt prefixes: every 028 / 029 managed job id
#: starts with one of them, and the probe refuses anything else.
REPLAY_ATTEMPT_PREFIXES = ("gl28", "gl29")
#: One completion-watcher pass, so the first delivery's records are complete
#: before the redelivery and whatever it caused has landed after it.
WATCHER_PASS_SECONDS = 60
#: Runs in an API Pod under the product's interpreter: redeliver the product's
#: own recorded TerminalEvent of a test attempt through the completion
#: service, exactly as the ``/v1/attempts/terminal`` route does for the
#: watcher. Refuses an attempt outside the test prefixes and one with no
#: recorded terminal (nothing to replay). One JSON object on the last line.
TERMINAL_REPLAY_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

PREFIXES = ("gl28", "gl29")


def main(argv):
    cluster_id, attempt_id = argv[:2]
    if not attempt_id.startswith(PREFIXES):
        raise ValueError(
            f"refusing to replay a terminal outside the test prefixes: {attempt_id}"
        )
    context = ApplicationContext.from_environment()
    store = context.store
    try:
        event = store.get_event_by_attempt(cluster_id, attempt_id)
    except NotFoundError:
        return {"attempt_id": attempt_id, "refused": "no recorded terminal event"}
    before = store.get_decision_by_event(event.event_key)
    decision = context.completion.handle_terminal(event)
    return {
        "attempt_id": attempt_id,
        "event_key": event.event_key,
        "terminal_status": event.terminal_status.value,
        "ended_at": event.ended_at.isoformat(),
        "before": before.model_dump(mode="json") if before is not None else None,
        "decision": decision.model_dump(mode="json"),
    }


print(json.dumps(main(sys.argv[1:]), sort_keys=True, default=str))
"""
#: The decision fields a redelivery must return unchanged.
DECISION_IDENTITY = ("status", "recovery_plan_id", "event_key")


def redeliver_terminal(ctx: VariantContext, attempt_id: str) -> dict[str, Any]:
    """Hand the product its own recorded terminal of ``attempt_id`` once more.

    The probe's answer (the stored decision before, the decision the service
    returned) is evidence ``terminal-replay``; a refusal (no recorded terminal)
    is INCONCLUSIVE -- there was nothing to redeliver.
    """

    replay = ctx.regional.cpu_python(
        TERMINAL_REPLAY_PROBE,
        ctx.settings.regional.cluster_id,
        attempt_id,
        attempts=1,
        timeout=240,
    )
    ctx.evidence.write("terminal-replay", replay)
    if replay.get("refused"):
        raise VariantInconclusive(
            f"the terminal of {attempt_id} could not be replayed: {replay['refused']}"
        )
    return replay


def attempt_entry(records: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    return (records.get("attempts") or {}).get(attempt_id) or {}


def restart_workflows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The attempt's recovery workflows that plan RESTART_WORKLOAD."""

    return [
        item
        for item in entry.get("recovery") or []
        if "RESTART_WORKLOAD" in operations(item.get("workflow") or {})
    ]


def restart_attempt_of(entry: dict[str, Any]) -> str | None:
    """The new attempt id the product's RESTART_WORKLOAD recorded, if any."""

    for item in entry.get("recovery") or []:
        workflow = item.get("workflow") or {}
        for execution in workflow.get("step_executions") or []:
            if (
                execution.get("operation") == "RESTART_WORKLOAD"
                and execution.get("status") == "SUCCEEDED"
            ):
                value = (execution.get("details") or {}).get("restart_attempt_id")
                if value:
                    return str(value)
    return None


def replay_errors(replay: dict[str, Any], original: dict[str, Any]) -> list[str]:
    """The redelivered terminal came back as the original decision, flagged duplicate."""

    decision = replay.get("decision") or {}
    errors = []
    if decision.get("duplicate") is not True:
        errors.append(
            "the replayed terminal was decided anew (duplicate is not true): "
            f"{decision.get('status')} {decision.get('reason')}"
        )
    for key in DECISION_IDENTITY:
        if decision.get(key) != original.get(key):
            errors.append(
                f"the replayed decision's {key} is {decision.get(key)!r}, the "
                f"original's {original.get(key)!r}"
            )
    return errors


def untouched_errors(entry: dict[str, Any], attempt_id: str) -> list[str]:
    """The new attempt has no terminal, decision or containment of its own."""

    return [
        f"the new attempt {attempt_id} has a {name} after the replay: "
        f"{str(entry.get(name))[:160]}"
        for name in ("event", "decision", "containment_incident")
        if entry.get(name)
    ]


def _notification_ids(entry: dict[str, Any]) -> list[str]:
    return sorted(
        str(item.get("notification_id"))
        for item in entry.get("notifications") or []
        if isinstance(item, dict)
    )


def unchanged_errors(
    before: dict[str, Any], after: dict[str, Any], attempt_id: str
) -> list[str]:
    """The redelivery changed nothing the product keeps about the attempt.

    ``before``/``after`` are the ``attempt_records`` documents read around the
    redelivery (one watcher pass apart): the stored decision row keeps its
    identity and is not rewritten as a duplicate, the attempt has the same
    RESTART_WORKLOAD workflows, the job's restart ledger has the same count and
    reservation ids (one deduction, no second), and the attempt's incidents
    carry no new notification.
    """

    old, new = attempt_entry(before, attempt_id), attempt_entry(after, attempt_id)
    errors: list[str] = []
    old_decision, new_decision = old.get("decision") or {}, new.get("decision") or {}
    for key in DECISION_IDENTITY:
        if new_decision.get(key) != old_decision.get(key):
            errors.append(
                f"the stored decision's {key} changed from {old_decision.get(key)!r} "
                f"to {new_decision.get(key)!r} on the redelivery"
            )
    if new_decision.get("duplicate") is True:
        errors.append(
            "the stored decision row is flagged duplicate after the redelivery: the "
            "original was overwritten"
        )
    restarts_before, restarts_after = restart_workflows(old), restart_workflows(new)
    if len(restarts_after) != len(restarts_before):
        errors.append(
            f"the attempt has {len(restarts_after)} RESTART_WORKLOAD workflows after "
            f"the redelivery, {len(restarts_before)} before"
        )
    ledger_before = before.get("restart_budget") or {}
    ledger_after = after.get("restart_budget") or {}
    for key in ("restart_count", "reservation_ids"):
        if ledger_after.get(key) != ledger_before.get(key):
            errors.append(
                f"the restart ledger's {key} changed from {ledger_before.get(key)!r} "
                f"to {ledger_after.get(key)!r} on the redelivery"
            )
    fresh = sorted(set(_notification_ids(new)) - set(_notification_ids(old)))
    if fresh:
        errors.append(
            f"the redelivery produced {len(fresh)} new notification(s) for the "
            f"attempt's incidents: {fresh[:4]}"
        )
    return errors


__all__ = [
    "DECISION_IDENTITY",
    "REPLAY_ATTEMPT_PREFIXES",
    "TERMINAL_REPLAY_PROBE",
    "WATCHER_PASS_SECONDS",
    "attempt_entry",
    "redeliver_terminal",
    "replay_errors",
    "restart_attempt_of",
    "restart_workflows",
    "unchanged_errors",
    "untouched_errors",
]
