"""The completion notification a GPU reset owes, judged from the incident's records.

Since the compound node command (性能 C) the RESET_GPU step of a reset chain rides
inside a carrier headed by QUIESCE_GPU_SERVICES, and the control plane creates
the batched step's GPU_RESET_COMPLETED record -- keyed by the step's own
idempotency key ``<workflow>/<step_index>/RESET_GPU`` -- when that carrier
reaches a terminal state. A verdict that read workflow and host evidence alone
passed resets that produced no completion record at all, so every reset case
also requires exactly one such record, keyed by that step and by nothing else.

What the record's delivery result must say depends on the incident. Every
destructive case injects its XID with a drill id, and a site does not deliver
drill mail (``GPU_FAULT_NOTIFICATION_DELIVER_DRILLS`` stays off), so a drill
incident's record is SKIPPED by the drill policy -- the policy reason is the
proof that the record was created and judged. An incident without a drill id
must be SENT with a provider message id; only that branch waits (bounded) for
the asynchronous dispatcher. Delivery of the compound-path mail itself is
proven by NOTIFY-001's drill, which mails drills on purpose in an isolated Store.

The records come from the store probe (``state["notifications"]``: one
``{"notification", "result"}`` pair per notification of the incident).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scripts.e2e.regional.remote_command_shapes import command_operations

RESET_NOTIFICATION_MARKER = "/gpu-reset/"
# The switch the drill policy names in its SKIPPED reason
# (``AdvisoryNotificationService.suppresses``); the reason is the proof that
# the record went through the delivery decision rather than being left unsent.
DRILL_POLICY_MARKER = "GPU_FAULT_NOTIFICATION_DELIVER_DRILLS"
# The outbox dispatcher claims and sends within seconds of the terminal
# report; the budget exists for the store read cadence (~6 s per probe), not
# for a mail that never comes -- that is a FAIL, reached at the deadline.
DELIVERY_WAIT_SECONDS = 120
DELIVERY_POLL_SECONDS = 5


def reset_step_idempotency_key(commands: list[dict[str, Any]]) -> str | None:
    """The RESET_GPU step's own idempotency key from the one command carrying it.

    Head or passenger, the key is ``<workflow>/<index>/RESET_GPU`` and it is
    what both the standalone executor path and the compound path key the record
    by. ``None`` when no single command carries the reset or the key is absent.
    """

    carriers = [item for item in commands if "RESET_GPU" in command_operations(item)]
    if len(carriers) != 1:
        return None
    command = carriers[0]
    keys: list[Any] = []
    if (command.get("step") or {}).get("operation") == "RESET_GPU":
        keys.append(command.get("idempotency_key"))
    keys.extend(
        (entry or {}).get("idempotency_key")
        for entry in command.get("batched_steps") or []
        if ((entry or {}).get("step") or {}).get("operation") == "RESET_GPU"
    )
    if len(keys) != 1 or not isinstance(keys[0], str) or not keys[0]:
        return None
    return keys[0]


def expected_status(drill_id: str | None) -> str:
    """SKIPPED by the drill policy for a drill incident, SENT for a real one."""

    return "SKIPPED" if drill_id else "SENT"


def _deduplication_key(entry: dict[str, Any]) -> str:
    return str((entry.get("notification") or {}).get("deduplication_key") or "")


def reset_notifications(notifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The GPU reset completion records among an incident's notifications."""

    return [
        entry
        for entry in notifications
        if isinstance(entry, dict)
        and RESET_NOTIFICATION_MARKER in _deduplication_key(entry)
    ]


def _partition(
    notifications: list[dict[str, Any]], operation_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """The records keyed by ``operation_id`` and the keys of every other one."""

    suffix = RESET_NOTIFICATION_MARKER + operation_id
    matching, foreign = [], []
    for entry in reset_notifications(notifications):
        key = _deduplication_key(entry)
        if key.endswith(suffix):
            matching.append(entry)
        else:
            foreign.append(key)
    return matching, foreign


def _delivery_errors(
    notification: dict[str, Any], result: dict[str, Any], drill_id: str | None
) -> list[str]:
    status = result.get("status")
    reason = str(result.get("reason") or "")
    described = f"{status or 'unsent'}" + (f": {reason}" if reason else "")
    if drill_id:
        errors = []
        if notification.get("drill_id") != drill_id:
            errors.append(
                "GPU reset completion notification is not labelled with the "
                f"incident's drill {drill_id}"
            )
        if status != "SKIPPED" or DRILL_POLICY_MARKER not in reason:
            errors.append(
                f"drill incident's GPU reset completion notification is {described}, "
                f"not SKIPPED by the drill policy ({DRILL_POLICY_MARKER})"
            )
        return errors
    if status != "SENT":
        return [f"GPU reset completion notification is {described}, not SENT"]
    if not result.get("provider_message_id"):
        return ["GPU reset completion notification has no provider message id"]
    return []


def reset_notification_errors(
    notifications: list[dict[str, Any]],
    *,
    operation_id: str | None,
    drill_id: str | None = None,
) -> list[str]:
    """Why the incident's records do not prove the reset was recorded once.

    Exactly one GPU reset completion keyed by the RESET_GPU step's own
    idempotency key and none keyed by anything else (the head in particular),
    category ACTION_COMPLETED, with the delivery result the incident calls for:
    SKIPPED by the drill policy for a drill incident, SENT with a provider
    message id otherwise. Any other status or reason is named as such.
    """

    if not operation_id:
        return ["RESET_GPU step idempotency key is not identifiable from the commands"]
    matching, foreign = _partition(notifications, operation_id)
    errors = [
        f"GPU reset completion notification keyed by {key!r} instead of the "
        f"RESET_GPU step's own idempotency key {operation_id}"
        for key in foreign
    ]
    if len(matching) != 1:
        errors.append(
            "expected exactly one GPU reset completion notification for "
            f"{operation_id}, found {len(matching)}"
        )
        return errors
    notification = matching[0].get("notification") or {}
    result = matching[0].get("result") or {}
    if notification.get("category") != "ACTION_COMPLETED":
        errors.append("GPU reset completion notification is not ACTION_COMPLETED")
    errors.extend(_delivery_errors(notification, result, drill_id))
    return errors


def delivery_pending(
    notifications: list[dict[str, Any]], *, operation_id: str | None = None
) -> bool:
    """Whether another store read could still turn a SENT verdict into a PASS.

    The dispatcher has not produced the record yet, or produced it and has not
    sent it yet (no result, QUEUED, or a FAILED attempt the outbox retries).
    Everything else is final: a second record, one keyed by another step or a
    SKIPPED result does not change by waiting, and the run must not spend the
    budget on it.
    """

    mails = reset_notifications(notifications)
    if operation_id is not None:
        mails, foreign = _partition(notifications, operation_id)
        if foreign:
            return False
    if not mails:
        return True
    if len(mails) != 1:
        return False
    status = (mails[0].get("result") or {}).get("status")
    return status in {None, "QUEUED", "FAILED"}


def wait_for_reset_notification(
    state: dict[str, Any],
    *,
    operation_id: str | None,
    snapshot: Callable[[], dict[str, Any]],
    timeout_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    drill_id: str | None = None,
    poll_seconds: float = DELIVERY_POLL_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The incident's records once the verdict is settled, or at the deadline.

    The terminal workflow state is judged first, so a record that already has
    its result costs no extra store read. Only a real incident's SENT is worth
    waiting for: every later read is a fresh ``snapshot()``, and the loop stops
    early on a defect no later read can cure (``delivery_pending``). A drill
    incident's record is SKIPPED synchronously when the carrier completes, so
    it is judged once.
    """

    notifications = list(state.get("notifications") or [])
    errors = reset_notification_errors(
        notifications, operation_id=operation_id, drill_id=drill_id
    )
    deadline = monotonic() + timeout_seconds
    while (
        errors
        and operation_id
        and not drill_id
        and delivery_pending(notifications, operation_id=operation_id)
        and monotonic() < deadline
    ):
        sleep(poll_seconds)
        notifications = list(snapshot().get("notifications") or [])
        errors = reset_notification_errors(
            notifications, operation_id=operation_id, drill_id=drill_id
        )
    return notifications, errors


def reset_notification_evidence(
    regional: Any,
    state: dict[str, Any],
    *,
    node: str,
    marker: str,
    observed_after: Any,
    wait: bool,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> tuple[dict[str, Any], list[str]]:
    """Judge the reset's completion record and return ``(evidence, errors)``.

    The incident's ``drill_id`` selects the branch: a drill incident (every
    kmsg-injected case) owes a record SKIPPED by the drill policy and is judged
    once; a real incident owes a SENT record and gets the delivery budget --
    but only on an otherwise passing run (``wait``). A run that already failed
    its workflow or host contract is judged once from the terminal state; the
    verdict is FAIL either way.
    """

    operation_id = reset_step_idempotency_key(state.get("commands") or [])
    drill_id = str((state.get("incident") or {}).get("drill_id") or "") or None
    budget = DELIVERY_WAIT_SECONDS if wait and not drill_id else 0

    def snapshot() -> dict[str, Any]:
        value: dict[str, Any] = regional.store_snapshot(
            node=node, marker=marker, observed_after=observed_after, queue_attempts=1
        )
        return value

    notifications, errors = wait_for_reset_notification(
        state,
        operation_id=operation_id,
        drill_id=drill_id,
        snapshot=snapshot,
        timeout_seconds=budget,
        sleep=sleep,
        monotonic=monotonic,
    )
    return {
        "expected_operation_id": operation_id,
        "drill": bool(drill_id),
        "drill_id": drill_id,
        "expected_status": expected_status(drill_id),
        "delivery_wait_seconds": budget,
        "reset_notifications": reset_notifications(notifications),
        "notification_count": len(notifications),
        "errors": errors,
    }, errors
