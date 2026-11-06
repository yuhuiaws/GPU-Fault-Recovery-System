"""DESTR-010's first phase on a graded node: the observe-only default (batch 131).

A solo XID 45 on a graded NVSwitch node takes no node action by default
(``gpu_fault.orchestration.xid45_policy``). Before the case opens its opt-in
window (``xid45_opt_in``) and proves batch 128's full GPU + NVSwitch reset
chain, it injects one solo XID 45 under the default and proves the default:
the decision is MONITOR_ONLY / NO_ACTION with RESTART_FM still the official
action; the incident is RECOVERED with no workflow, cites NVIDIA's GPU debug
guidelines and says how to opt in; its one notification is the XID
investigatory notification, SENT or kept back by the drill policy as this
phase's drill (``<run_id>-observe``); and nothing reached the node -- Fabric
Manager's MainPID and InvocationID, the Agent ledger, the gpu-fault timers and
the node's scheduling state are exactly the baseline's.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.reset_notification_evidence import (
    DELIVERY_WAIT_SECONDS,
    wait_for_delivery,
)

#: After the decision lands, how long the phase keeps watching for a workflow
#: or a node action that must not come.
SETTLE_SECONDS = 60
POLL_SECONDS = 5


def finalized(state: dict[str, Any]) -> bool:
    """The XID's window closed and the control plane recorded its incident."""

    decision = state.get("decision") or {}
    return bool(
        decision
        and decision.get("disposition") != "PENDING_CORRELATION"
        and state.get("incident")
    )


def wait_finalized(
    regional: Any,
    *,
    node: str,
    marker: str,
    injected_at: datetime,
    timeout_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = regional.store_snapshot(
            node=node, marker=marker, observed_after=injected_at, queue_attempts=1
        )
        if finalized(state):
            return state
        sleep(POLL_SECONDS)
    raise RegionalFixtureError(f"the solo XID 45 was never finalized: {state}")


def notification_errors(
    state: dict[str, Any], *, drill_id: str | None = None
) -> list[str]:
    """Exactly the XID investigatory notice, no completion notice; the notice
    SENT or kept back by the drill policy as ``drill_id``
    (``fabric_restart_shape.notification_delivery_errors``)."""

    items = shapes.normalize_notifications(state.get("notifications") or [])
    investigatory = shapes.shape_notifications(items, shapes.OBSERVE_ONLY)
    if len(investigatory) != 1 or len(items) != 1:
        return [f"expected only the XID investigatory notification, got {len(items)}"]
    return shapes.notification_delivery_errors(
        investigatory[0], drill_id=drill_id, label="XID investigatory"
    )


def investigatory_delivery_pending(notifications: list[Any]) -> bool:
    """Whether a later read could still settle the investigatory notice's result."""

    items = shapes.normalize_notifications(notifications)
    return len(items) == 1 and shapes.notification_delivery_pending(
        shapes.shape_notifications(items, shapes.OBSERVE_ONLY)
    )


def run_phase(
    regional: Any,
    fixture: Any,
    case_dir: Path,
    *,
    node: str,
    run_id: str,
    attempt: int,
    baseline_host: dict[str, Any],
    baseline_node: dict[str, Any],
    graded_gpus: list[str],
    companion_window_seconds: int,
    node_state_errors: Callable[..., list[str]],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Inject one solo XID 45 under the default and judge it; never raises
    for a verdict, only for a case that cannot continue."""

    phase_dir = case_dir / "observe-only"
    phase_dir.mkdir(parents=True, exist_ok=True)
    marker = f"destr010-observe-{int(time.time())}-a{attempt}"
    drill_id = f"{run_id}-observe"
    injected_at = datetime.now(timezone.utc)
    injection = fixture.execute(
        "write-xid45",
        "--marker",
        marker,
        "--drill-id",
        drill_id,
        "--pci-bdf",
        baseline_host["gpu_pci_bdf"],
    )
    write_json_atomic(phase_dir / "injection.json", injection)
    wait_finalized(
        regional,
        node=node,
        marker=marker,
        injected_at=injected_at,
        timeout_seconds=companion_window_seconds + 300,
        sleep=sleep,
    )
    # Give a workflow or a node action that must not come the time to come.
    sleep(SETTLE_SECONDS)

    def snapshot() -> dict[str, Any]:
        value: dict[str, Any] = regional.store_snapshot(
            node=node, marker=marker, observed_after=injected_at, queue_attempts=1
        )
        return value

    state = snapshot()
    # The notice's result follows the decision: at once when the drill policy
    # keeps it back, from the dispatcher when the site mails it.
    state["notifications"], _errors = wait_for_delivery(
        list(state.get("notifications") or []),
        judge=lambda items: notification_errors(
            {"notifications": items}, drill_id=drill_id
        ),
        pending=investigatory_delivery_pending,
        snapshot=snapshot,
        timeout_seconds=DELIVERY_WAIT_SECONDS,
        sleep=sleep,
        monotonic=monotonic,
    )
    write_json_atomic(phase_dir / "store.json", state)
    host = fixture.execute("snapshot", "--since-epoch", str(injected_at.timestamp()))
    write_json_atomic(phase_dir / "host.json", host)
    errors = shapes.observe_only_errors(state, graded_gpus)
    errors.extend(notification_errors(state, drill_id=drill_id))
    if (state.get("event") or {}).get("xid") != 45:
        errors.append("matched event is not XID 45")
    if not str((state.get("event") or {}).get("evidence_ref") or "").startswith(
        "kmsg://"
    ):
        errors.append("XID evidence is not backed by kmsg://")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote commands are open after the observe-only XID 45")
    errors.extend(shapes.observe_only_host_errors(baseline_host, host))
    errors.extend(
        node_state_errors(
            baseline_node, regional.node_snapshot(node), stage="observe-only"
        )
    )
    report = {
        "marker": marker,
        "drill_id": drill_id,
        "injected_at": injected_at.isoformat(),
        "incident_id": (state.get("incident") or {}).get("incident_id"),
        "decision": {
            key: (state.get("decision") or {}).get(key)
            for key in ("official_action", "disposition", "action")
        },
        "policy_reference": (state.get("incident") or {}).get("policy_reference"),
        "errors": [f"observe-only: {item}" for item in errors],
    }
    write_json_atomic(phase_dir / "verdict.json", report)
    return report


def wait_outside_companion_window(
    injected_at: datetime,
    companion_window_seconds: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """The chain's XID 45 must not land inside the first one's window."""

    clear_at = injected_at + timedelta(seconds=companion_window_seconds + 5)
    remaining = (clear_at - datetime.now(timezone.utc)).total_seconds()
    if remaining > 0:
        sleep(remaining)
