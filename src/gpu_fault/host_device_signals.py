"""Device-scoped host health signals whose device stopped reporting (batch 160).

Live GF-GENERAL-LIVE-006 (plain EKS, 2026-10-25, read-only replay of the
product's records): ``disk_io_await_ms`` on a test dm-delay device went
active and NOTIFIED at 15:03Z, then the device was torn down. The host
collector emits ``disk_io_*`` only for block devices present in
``/proc/diskstats`` with completed I/O, so no below-threshold sample ever
reached the control plane and ``gpu-eks-cluster/<node>/disk_io_await_ms/dm-0``
stayed active and notified for two hours. At 17:04Z a new dm-0 crossed the
threshold again; ``_next_health_signal_state`` (``store/shared/xid.py``) saw
the ``notified`` latch of the same key, emitted nothing, and a genuinely new
sustained fault on a same-named device went unreported.

Two layers close that. Both only ever *clear* a signal -- neither invents a
fault, and a cleared signal needs its full sustain window again before it
notifies -- and a device that keeps reporting at or above its threshold is
carried by every delivered batch and is affected by neither:

* The host collector reports a block device that was in ``/proc/diskstats``
  on its previous read and is not now (or whose monotonic counters restarted,
  which the kernel documents as reattachment or reinitialisation) with one
  ``device_state=GONE`` sample per ``disk_io_*`` metric. Its value is 0 but it
  is not a reading: :func:`device_gone` grades it as a clear and keeps it out
  of the latest metrics, so neither VALIDATE_HOST nor anything else sees a
  healthy 0 ms for a device that is not there.
* The control plane ages out a device-scoped health signal of a node that is
  still reporting while no sample for that device arrived for
  ``GPU_FAULT_HEALTH_SIGNAL_DEVICE_AGE_OUT_SUMMARIES`` (default 3, ``0`` off,
  at least 2 otherwise) host health summaries
  (``GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS``, default 300 s: fifteen minutes
  by default) on *both* the control plane's receive clock and the node's own
  clock, so a backlog the node delivers after an outage, where only the
  receive clock jumped, ages nothing out. It runs before the batch's own
  claim and also re-arms a device the batch carries again after such a gap:
  when the node itself went silent for the window -- a reboot, a dead
  collector, a replacement node under the same name; a collector start
  delivers every present device -- the device's first sample after it opens a
  new episode instead of being judged against the old latch (that used to
  depend on whether the device was in the node's first batch or a later one).
  Node-scoped signals (device ``node``) and attempt-scoped low-utilisation
  windows (the retention sweep's business) are never aged out; only the
  node's own collector speaks for the node's devices. The age-out is the
  cleared state itself -- ``active=False`` at the batch's time (a microsecond
  before it for a device the batch carries), written through the ordinary
  claim -- one INFO log line naming the keys, and
  ``device_signal_age_outs_total`` on the policy. (No new
  field on ``HealthSignalState``: a row with an unknown field cannot be decoded
  by the previous release after a rollback, and the claim path reads these
  rows on every sample.)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from gpu_fault.collector_kinds import COLLECTOR_KINDS, CollectorKind
from gpu_fault.store.shared.health_signals import signal_clock, signal_key_attempt

if TYPE_CHECKING:
    from gpu_fault.host_health import (
        HostMetricSample,
        HostTelemetryBatch,
        NodeHealthPolicy,
    )
    from gpu_fault.host_low_utilization import Transition

LOGGER = logging.getLogger("gpu_fault.host_health")

#: The sample label the host collector sets when the device a sample names is
#: no longer on the node. Its value is never a reading.
DEVICE_STATE_LABEL = "device_state"
DEVICE_STATE_GONE = "GONE"
DEVICE_AGE_OUT_SUMMARIES_ENV = "GPU_FAULT_HEALTH_SIGNAL_DEVICE_AGE_OUT_SUMMARIES"
#: Fewer summaries than this would age out a device whose one jittered summary
#: ran long (``cadence.jittered_next_phase``: up to ``1.5 x`` the interval).
MINIMUM_DEVICE_AGE_OUT_SUMMARIES = 2
#: How far before a batch a key it carries again is cleared.
_BEFORE = timedelta(microseconds=1)


def device_gone(sample: HostMetricSample) -> bool:
    """Whether ``sample`` says its device is gone rather than reads a value."""

    return sample.labels.get(DEVICE_STATE_LABEL) == DEVICE_STATE_GONE


def device_signal_age_out_seconds(environ: Mapping[str, str] | None = None) -> float:
    """How long a device-scoped signal may go without a sample while its node
    reports before it is aged out: the configured number of host health
    summaries times the host collector's summary period; ``0`` when off."""

    env = os.environ if environ is None else environ
    # Spelled out for the environment reference generator (name and default).
    summaries = int(env.get("GPU_FAULT_HEALTH_SIGNAL_DEVICE_AGE_OUT_SUMMARIES", "3"))
    if summaries != 0 and summaries < MINIMUM_DEVICE_AGE_OUT_SUMMARIES:
        raise ValueError(
            f"{DEVICE_AGE_OUT_SUMMARIES_ENV} must be 0 (off) or at least "
            f"{MINIMUM_DEVICE_AGE_OUT_SUMMARIES}"
        )
    cadence = COLLECTOR_KINDS[CollectorKind.HOST_TELEMETRY].cadence
    assert cadence is not None, "the host telemetry kind posts health summaries"
    period = float(env.get(cadence.env, str(cadence.default)))
    if period <= 0:
        raise ValueError(f"{cadence.env} must be positive")
    return summaries * period


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _device_scoped(policy: NodeHealthPolicy, prefix: str, signal_key: str) -> bool:
    """A host-telemetry rule's signal for a named device, not timed per attempt."""

    metric, _, rest = signal_key[len(prefix) :].partition("/")
    if not rest or rest == "node" or rest.startswith("node/"):
        return False
    if (
        metric not in policy.METRIC_RULES
        and metric not in policy.SUSTAINED_METRIC_RULES
    ):
        return False
    return signal_key_attempt(signal_key) is None


def age_out_vanished_device_signals(
    policy: NodeHealthPolicy,
    batch: HostTelemetryBatch,
    transitions: list[Transition],
) -> list[str]:
    """Clear the active device-scoped signals of ``batch``'s node that carried
    no sample for the age-out window on both clocks; the cleared keys.

    Runs before ``transitions`` -- the claims this batch is about to make --
    are written. A device the batch carries that reported within the window
    is reporting and is never aged out, whatever its reading; one it carries
    again after the window is cleared a microsecond before the batch, so its
    sample is judged as the first of a new episode.
    """

    stale_after = policy.device_signal_age_out_seconds
    if stale_after <= 0 or batch.producer != "node":
        return []
    # A store without the reader (a narrow double) ages nothing out, like the
    # retention sweep's evidence readers.
    stale_states = getattr(policy.store, "stale_health_signal_states", None)
    if stale_states is None:
        return []
    window = timedelta(seconds=stale_after)
    prefix = f"{batch.cluster_id}/{batch.node_id}/"
    carried = {signal_key for signal_key, *_ in transitions}
    received = _utc(signal_clock(batch.observed_at, batch.received_at))
    observed = _utc(batch.observed_at)
    absent: list[Transition] = []
    returned: list[Transition] = []
    # Inclusive: a device last heard from exactly one window ago has been
    # missing for the window (the store's bound is exclusive, like retention's).
    for state in stale_states(prefix, stale_before=received - window + _BEFORE):
        if (
            not state.active
            or not _device_scoped(policy, prefix, state.signal_key)
            or observed - _utc(state.observed_at) < window
        ):
            continue
        if state.signal_key in carried:
            returned.append((state.signal_key, False, batch.observed_at - _BEFORE, 0.0))
        else:
            absent.append((state.signal_key, False, batch.observed_at, 0.0))
    if absent:
        policy.store.claim_health_signal_transitions(
            absent, received_at=batch.received_at
        )
    if returned:
        # Strictly older than the batch, or the store would drop the batch's
        # own sample of the key as a replay (``sample_disposition``).
        policy.store.claim_health_signal_transitions(
            returned,
            received_at=(
                batch.received_at - _BEFORE if batch.received_at is not None else None
            ),
        )
    keys = [signal_key for signal_key, *_ in (*absent, *returned)]
    if not keys:
        return []
    policy.device_signal_age_outs_total += len(keys)
    LOGGER.info(
        "aged out %d device-scoped health signal(s) of %s/%s: no sample for their "
        "device in %.0f s on the node's and the receive clock; a new crossing "
        "opens a new episode: %s",
        len(keys),
        batch.cluster_id,
        batch.node_id,
        stale_after,
        ", ".join(keys),
    )
    return keys
