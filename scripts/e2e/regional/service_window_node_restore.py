"""Undo of a service window that stopped kubelet, awaited through the node.

The DESTR-008 service window (``destr008_service_window.ServiceWindowController``
over ``probes/warm_spare_node_probe.py``) stops ``kubelet.service`` from an
on-node systemd timer and restores it from an independent on-node restore unit
at the binding's ``restore_at``; that unit's authority ends at ``expires_at``
(``restore_at`` + ``RECOVERY_SECONDS``). Every request the controller makes to
the node -- ``restore-service`` and ``cleanup`` included -- travels over
``kubectl exec`` into the host-probe Pod, and the API server hands an exec to
the node's own kubelet. While kubelet is stopped that channel is gone: the exec
returns no output and the host fixture raises ``HostProbeMissingResponseError``,
which says nothing about the node.

Live 2026-10-27 (plain EKS, GF-GENERAL-LIVE-030 ``not_ready``): the explicit
undo was issued about 80 s into the stop, got the missing response, and the
variant graded "the candidate is not Ready after the release (Unknown)" and
"cleanup did not prove the node clean" at 14:19:20Z -- eight minutes before the
restore unit fired at ``restore_at`` (14:27:22Z) and the node was Ready again
(14:27:23Z). The node-side deadman had worked; the harness had judged the
hand-back before it could.

``restore_through_node`` is the undo for such a window. It still attempts the
explicit undo first -- with kubelet up (the stop timer not yet fired, the
restore unit already done; HyperPod and EKS alike) that cancels or closes the
window at once. A missing or unavailable transport while the node is not Ready
is recorded as the expected shape, not a failure: the undo is deferred to the
on-node restore unit, the node is polled through the API until it posts Ready
again, with a deadline past ``expires_at``, and only then is the explicit undo
repeated over the restored channel -- the window's own CLOSED report (service
running, stop and restore units quiescent, ``restore_reason`` naming who
restored it) is the proof the service came back. A node that does not return
by the deadline fails the undo with ``NodeRestoreOverdue``, whose message names
``restore_at``, ``expires_at``, the wait deadline and the last node condition
observed, and whose ``report`` carries the observations made so far.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional.host_probe_fixture import (
    HostProbeMissingResponseError,
    HostProbeTransportError,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError

#: After the window's ``expires_at`` -- the last instant the on-node restore
#: unit may still start the service -- how long the node is given to post Ready
#: through the restarted kubelet (one node-status period is 10 s; the lifecycle
#: controller clears ``Unknown`` on the first status it receives).
NODE_READY_MARGIN_SECONDS = 120
#: Transport failures that say nothing about the node while kubelet is down:
#: no response came back over the exec channel kubelet itself serves.
DEFERRABLE_ERRORS = (HostProbeMissingResponseError, HostProbeTransportError)
POLL_SECONDS = 5.0


class NodeRestoreOverdue(RegionalFixtureError):
    """The node did not post Ready by the wait deadline; ``report`` has the trail."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def window_deadlines(
    binding: Mapping[str, Any] | None,
    *,
    margin_seconds: int = NODE_READY_MARGIN_SECONDS,
) -> dict[str, datetime] | None:
    """``restore_at`` / ``expires_at`` / ``ready_by`` of a bound window.

    ``binding`` is the controller's durable host binding or the probe's own
    report (both carry the epoch seconds ``restore_at`` and ``expires_at``);
    ``None`` or one without them -- no stop was ever bound -- gives ``None``.
    """

    if not isinstance(binding, Mapping):
        return None
    try:
        restore_at = _epoch(binding["restore_at"])
        expires_at = _epoch(binding["expires_at"])
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    return {
        "restore_at": restore_at,
        "expires_at": expires_at,
        "ready_by": expires_at + timedelta(seconds=margin_seconds),
    }


def restore_unit_name(binding: Mapping[str, Any] | None) -> str:
    """The on-node restore unit of a bound window, or ``""`` when unbound."""

    from scripts.e2e.regional.probes import warm_spare_node_probe as probe

    if not isinstance(binding, Mapping):
        return ""
    try:
        return str(probe.ServiceWindow(dict(binding)).units["restore-service"])
    except (probe.ProbeError, KeyError, TypeError, ValueError):
        return ""


def restore_through_node(
    *,
    service: str,
    node: str,
    undo: Callable[[], Any],
    node_snapshot: Callable[[], Mapping[str, Any]],
    binding: Mapping[str, Any] | None,
    restore_unit: str = "",
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    poll_seconds: float = POLL_SECONDS,
    ready_timeout_seconds: int = 300,
    margin_seconds: int = NODE_READY_MARGIN_SECONDS,
) -> dict[str, Any]:
    """Undo ``service``'s window on ``node`` without reading through its kubelet.

    ``undo`` is the explicit undo over the probe channel (the window's
    ``cleanup``); ``node_snapshot`` reads the node through the API (``ready`` is
    the Ready condition's status string); ``binding`` is the window's durable
    host binding (``restore_at``/``expires_at``), ``None`` when no stop was
    bound. ``clock``/``sleep`` are the variant's, so a test runs on a virtual
    clock. Returns the undo report; raises ``NodeRestoreOverdue`` when the node
    is not Ready by the deadline, and re-raises any rejected host action.
    """

    deadlines = window_deadlines(binding, margin_seconds=margin_seconds)
    report: dict[str, Any] = {
        "service": service,
        "node": node,
        "restore_unit": restore_unit or restore_unit_name(binding) or None,
        **{key: value.isoformat() for key, value in (deadlines or {}).items()},
    }
    attempt: dict[str, Any] = {"at": clock().isoformat()}
    report["explicit_undo"] = attempt
    try:
        attempt["result"] = undo()
    except DEFERRABLE_ERRORS as exc:
        # No answer came over the channel kubelet serves: expected while the
        # node is not Ready. The on-node restore unit is the restore now.
        attempt.update(
            outcome="deferred",
            error=f"{type(exc).__name__}: {exc}",
            node_ready=_condition(node_snapshot),
            deferred_to=report["restore_unit"] or "the on-node restore unit",
        )
    else:
        attempt["outcome"] = "restored"
    deferred = attempt["outcome"] == "deferred"
    if deferred and deadlines is not None:
        ready_by = deadlines["ready_by"]
    else:
        ready_by = clock() + timedelta(seconds=ready_timeout_seconds)
    report["ready_by"] = ready_by.isoformat()
    report["node_ready"] = _await_ready(
        node_snapshot,
        clock=clock,
        sleep=sleep,
        poll_seconds=poll_seconds,
        ready_by=ready_by,
        report=report,
    )
    if deferred:
        # The channel is back with kubelet: the window's CLOSED report over it
        # is the proof of the restoration. A failure now is a real one.
        report["confirmation"] = undo()
    return report


def _await_ready(
    node_snapshot: Callable[[], Mapping[str, Any]],
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    poll_seconds: float,
    ready_by: datetime,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Poll the node until Ready=True; transitions only in the timeline."""

    timeline: list[dict[str, Any]] = []
    polls = 0
    while True:
        now = clock()
        condition = _condition(node_snapshot)
        polls += 1
        if not timeline or timeline[-1]["ready"] != condition:
            timeline.append({"observed_at": now.isoformat(), "ready": condition})
        observation = {
            "ready": condition,
            "observed_at": now.isoformat(),
            "polls": polls,
            "timeline": timeline,
        }
        if condition == "True":
            return observation
        if now >= ready_by:
            report["node_ready"] = observation
            raise NodeRestoreOverdue(_overdue_message(report, ready_by), report)
        sleep(poll_seconds)


def _overdue_message(report: Mapping[str, Any], ready_by: datetime) -> str:
    last = report["node_ready"]
    parts = [
        f"{report['service']} on {report['node']} was not restored through the "
        f"node by {ready_by.isoformat()}"
    ]
    if report.get("restore_at"):
        parts.append(
            f"the on-node restore unit {report.get('restore_unit') or '(unnamed)'} "
            f"was due at {report['restore_at']} and its authority expired at "
            f"{report['expires_at']}"
        )
    else:
        parts.append(
            "no service window binding was recorded, so no on-node restore was due"
        )
    parts.append(
        f"last observed node condition Ready={last['ready']} at "
        f"{last['observed_at']} after {last['polls']} reads"
    )
    attempt = report.get("explicit_undo") or {}
    if attempt.get("outcome") == "deferred":
        parts.append(
            f"the explicit undo at {attempt.get('at')} got {attempt.get('error')} "
            f"while the node was Ready={attempt.get('node_ready')}"
        )
    return "; ".join(parts)


def _condition(node_snapshot: Callable[[], Mapping[str, Any]]) -> str:
    """The node's Ready status as a string; an unreadable node is named, not raised."""

    try:
        snapshot = node_snapshot()
    except Exception as exc:  # noqa: BLE001 - one unreadable poll is recorded, not fatal
        return f"unreadable ({type(exc).__name__})"
    return str(snapshot.get("ready"))


def _epoch(value: Any) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"epoch seconds expected, got {type(value).__name__}")
    return datetime.fromtimestamp(value, timezone.utc)
