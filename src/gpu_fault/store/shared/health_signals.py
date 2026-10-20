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
import re
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


def health_signal_key(
    cluster_id: str,
    node_id: str,
    metric_name: str,
    device: str | None,
    rule_id: str | None = None,
) -> str:
    """``cluster/node/metric/device``, plus the sustained rule id if any."""

    parts = [cluster_id, node_id, metric_name, device or "node"]
    if rule_id:
        parts.append(rule_id)
    return "/".join(parts)


#: The ``active_workload_only`` rules of ``NodeHealthPolicy``: the only rules
#: whose windows are claimed under :func:`attempt_signal_key` (pinned against
#: ``SUSTAINED_METRIC_RULES`` by a test). Retention recognises such a key by
#: this segment, three from the end.
ATTEMPT_SCOPED_RULE_IDS = frozenset({"LOW_CPU_UTILIZATION", "LOW_GPU_UTILIZATION"})

_TRAINING_SIGNAL_KEY = re.compile(
    r"(?P<cluster>[^/]+)/(?P<attempt>[^/]+)/rank-(?P<rank>[0-9]+)/training-[a-z-]+"
)


def attempt_signal_key(signal_key: str, job_id: str, attempt_id: str) -> str:
    """The window of an active-workload rule for one managed attempt.

    Low CPU/GPU utilisation is timed per attempt: a job that replaces another
    on the node starts a fresh window and a fresh notification latch instead of
    inheriting the previous attempt's low time (round 10, COLLECT-017 ->
    E2E-002). Rows accumulate per attempt, like training-health states; the
    health-signal retention removes them once the attempt has ended
    (``store/shared/signal_retention.py``).
    """

    return "/".join([signal_key, job_id, attempt_id])


def training_signal_key(cluster_id: str, attempt_id: str, rank: int, kind: str) -> str:
    """``cluster/attempt/rank-N/training-<kind>``: one training-health signal."""

    return f"{cluster_id}/{attempt_id}/rank-{rank}/training-{kind}"


def signal_key_attempt(signal_key: str) -> tuple[str | None, str] | None:
    """``(job_id, attempt_id)`` of an attempt-scoped key, else None.

    Two shapes carry an attempt: :func:`attempt_signal_key` (``job_id`` and
    ``attempt_id`` follow an attempt-scoped rule id) and
    :func:`training_signal_key` (no job, no node). Kubernetes names and label
    values never contain ``/``, so the segments are unambiguous; a device can
    (a mount point), which is why the rule id is matched from the end.
    """

    training = _TRAINING_SIGNAL_KEY.fullmatch(signal_key)
    if training is not None:
        return None, training["attempt"]
    parts = signal_key.split("/")
    if len(parts) >= 7 and parts[-3] in ATTEMPT_SCOPED_RULE_IDS:
        return parts[-2], parts[-1]
    return None


def finding_health_signal_key(finding: NodeHealthFinding) -> str:
    """The health-signal key ``HostHealthService.evaluate_metrics`` claimed for
    a finding: ``cluster/node/metric/device`` plus the sustained rule id when
    the finding came from one (it travels as ``diagnostic_parameters["signal"]``)
    and, for a rule timed per managed attempt, the attempt the finding names.
    """

    rule_id = finding.diagnostic_parameters.get("signal")
    key = health_signal_key(
        finding.cluster_id,
        finding.node_id,
        str(finding.metric_name),
        finding.device,
        str(rule_id) if rule_id else None,
    )
    if rule_id and finding.job_id and finding.attempt_id:
        return attempt_signal_key(key, finding.job_id, finding.attempt_id)
    return key
