"""Which clock a health-signal sample is judged on (F-M2).

Durations ("active for N seconds") and ordering run on the control plane's
receive time when the caller supplies it; the node's ``observed_at`` is kept
for display only. A node clock that steps backwards is reported, not obeyed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import hashlib
import json
from typing import TYPE_CHECKING

from gpu_fault.models import HealthSignalState

if TYPE_CHECKING:
    from gpu_fault.host_health import NodeHealthFinding


def signal_clock(observed_at: datetime, received_at: datetime | None) -> datetime:
    return received_at if received_at is not None else observed_at


def previous_clock(previous: HealthSignalState) -> datetime:
    return previous.clock_at if previous.clock_at is not None else previous.observed_at


def sample_disposition(
    previous: HealthSignalState | None,
    observed_at: datetime,
    received_at: datetime | None,
) -> tuple[bool, bool]:
    """``(accept, node_clock_regressed)`` for one sample against its state.

    A sample is dropped only when it is not newer on the clock the state
    machine runs on (a replayed or out-of-order delivery). A node timestamp
    that went backwards while the receive time moved forward is a clock
    regression: accepted, and counted by the caller.
    """

    if previous is None:
        return True, False
    accept = signal_clock(observed_at, received_at) > previous_clock(previous)
    regressed = received_at is not None and observed_at <= previous.observed_at
    return accept, regressed


def latched_health_signal_state(
    state: HealthSignalState | None,
    notified_at: datetime,
    *,
    semantic_fingerprint: str | None = None,
) -> HealthSignalState | None:
    """The state with its ``notified`` latch set, or None when there is nothing
    to latch (P0-38B).

    The claim no longer latches; the deliverer does, after the commit that
    carried the incident. Nothing is latched for a missing or inactive signal,
    for one already latched, or for an activation that began after
    ``notified_at``: that is a new episode which still owes a notification.
    """

    if state is None or not state.active or state.notified is True:
        return None
    if state.active_since is not None and state.active_since > notified_at:
        return None
    if state.semantic_fingerprint is not None or semantic_fingerprint is not None:
        if (
            semantic_fingerprint != state.semantic_fingerprint
            or state.semantic_since is None
            or not state.semantic_since <= notified_at <= previous_clock(state)
        ):
            return None
    return state.model_copy(update={"notified": True})


def signal_semantic_fingerprints(
    count: int, fingerprints: Sequence[str | None] | None
) -> tuple[str | None, ...]:
    """Validate the whole batch before any backend can persist a partial claim."""
    if fingerprints is None:
        return (None,) * count
    values = tuple(fingerprints)
    if len(values) != count or any(
        value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 256)
        for value in values
    ):
        raise ValueError("health signal semantics must match the claim batch")
    return values


def finding_health_signal_fingerprint(finding: NodeHealthFinding) -> str:
    """Stable policy meaning; reading values, timestamps and prose are not identity."""
    semantics = (
        finding.recommended_action.value,
        finding.severity.value,
        finding.diagnostic_parameters.get("failure_mode"),
    )
    return hashlib.sha256(
        json.dumps(semantics, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finding_health_signal_key(finding: NodeHealthFinding) -> str:
    """The health-signal key ``HostHealthService.evaluate_metrics`` claimed for
    a finding: ``cluster/node/metric/device`` plus the sustained rule id when
    the finding came from one (it travels as ``diagnostic_parameters["signal"]``).
    """

    parts = [
        finding.cluster_id,
        finding.node_id,
        str(finding.metric_name),
        finding.device or "node",
    ]
    rule_id = finding.diagnostic_parameters.get("signal")
    if rule_id:
        parts.append(str(rule_id))
    return "/".join(parts)
