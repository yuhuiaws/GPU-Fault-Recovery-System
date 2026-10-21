"""COLLECT-002's latency bound: one term per stage between the load and the Store.

The case measures to the Store's ``ingested_at`` of the first accepted power
candidate from the load's observed onset (:func:`load_onset`): after the
load-start receipt the throttle probe reads the loaded GPU (read-only
``nvidia-smi``, the NVML values DCGM samples) until one read reaches the
collector's ``power_limit_correlation`` thresholds, and the latency runs from
the start of the last read still below them. ``dcgmproftester``'s own start-up
between the receipt and its first kernel -- it execs, starts an embedded DCGM
over every GPU, forks a CUDA worker that creates its context, loads its PTX
module and cuBLAS, and only then runs its GEMMs -- lies before that read: no
source bounds it, and it is not part of the product's path. Live on a healthy
H200 node the first anomalous sample came 33.1 s after the receipt against the
31 s the sampling allows, every product stage after it nominal. Without the
onset reads (an older probe, a failed read, a load that never reached the
thresholds) the latency runs from the receipt and the start-up is listed as
the one stage without an allowance. The stages after the onset, read off the
pinned sources -- dcgm-exporter ``4.6.1-4.8.4``, its go-dcgm ``v1.4611.2`` and
DCGM 4.6.1 (the exporter's embedded engine, and the nearest public source of
the node host engines) -- are the same in the exporter's embedded and host
(``DCGM_REMOTE_HOSTENGINE_INFO``) modes:

1. NVML. DCGM reads the power through ``nvmlDeviceGetPowerUsage``, a 1-s average
   on Ampere and newer, and the utilization over a sample period of at most
   1 s: a reading trails the load by up to a second. Measured from the onset
   read the same values cross the thresholds before the first read at them
   ends: the second covers that gap up to 1 s, and a wider gap adds its excess
   (``load_onset_excess_seconds``).
2. DCGM watch. The exporter watches its fields every collect interval
   (``updateFreq = -c x 1000`` us, ``maxKeepAge`` 600 s) and the cache manager
   samples a field only once its age reaches that period; a host engine with
   more watchers samples at the shortest of their periods, never longer. The
   power limit the case lowers with ``nvidia-smi -pl`` is polled the same way:
   DCGM appends a limit on change only when the limit is set through DCGM.
   Up to one exporter collect interval.
3. The exporter adds no sampler of its own: ``/metrics`` gathers on request
   (overlapping scrapes are coalesced into one gather) and reads DCGM's cached
   latest values (``dcgmEntityGetLatestValues``: no live flag, no field
   update; only a profiling-watch re-arm forces one, and the product exports
   no profiling field).
4. Collector. One scrape per tick. The loop sleeps the interval after each
   tick's work, so ticks are the interval plus that work apart; the cadence
   jitter does not touch this tick (it moves the health summary, the inventory
   and the start-up only). Up to one tick to the first anomalous sample, then
   ``confirmation - 1`` ticks to the sample that confirms it.
5. Delivery: the confirming tick's own work from its sample stamp to the Store.

The sampling terms come from the live configuration; the onset gap, the ticks'
work and the delivery are measured in the case itself.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from gpu_fault.dcgm_exporter_cadence import DCGM_EXPORTER_COLLECT_INTERVAL_MS

#: What dcgm-exporter samples at when neither ``-c``/``--collect-interval`` nor
#: ``DCGM_EXPORTER_INTERVAL`` is set (its ``pkg/cmd/app.go`` default), and the
#: documented assumption when the node runs no readable product exporter Pod:
#: above the product's rendered period, so it can over-allow the product's
#: exporter but never under-allow it, and the result names the assumption.
EXPORTER_DEFAULT_INTERVAL_MS = 30000
#: ``nvmlDeviceGetPowerUsage`` averages over 1 s; utilization samples at most 1 s.
NVML_AVERAGE_SECONDS = 1.0
EXPORTER_APP = "gpu-fault-dcgm-exporter"
INTERVAL_FLAGS = frozenset({"-c", "--c", "-collect-interval", "--collect-interval"})
INTERVAL_ENV = "DCGM_EXPORTER_INTERVAL"
REMOTE_FLAGS = frozenset(
    {"-r", "--r", "-remote-hostengine-info", "--remote-hostengine-info"}
)
REMOTE_ENV = "DCGM_REMOTE_HOSTENGINE_INFO"
#: Without the probe's onset reads, the one stage between the receipt and the
#: Store the bound has no term for.
WITHOUT_ALLOWANCE = (
    "dcgmproftester start-up between the receipt and its first kernel "
    "(embedded DCGM, CUDA worker and cuBLAS start): no source bounds it and, "
    "without the probe's onset reads, the case cannot observe it"
)
ONSET = "load-onset"
RECEIPT = "load-start-receipt"


def _flag_value(tokens: Sequence[str], names: frozenset[str]) -> str | None:
    """The last value given to one of ``names`` (Go ``flag`` syntax), if any."""

    value: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            break
        name, equals, inline = token.partition("=")
        if name in names:
            if equals:
                value = inline
            elif index + 1 < len(tokens):
                index += 1
                value = tokens[index]
        index += 1
    return value


def _container_interval(container: Mapping[str, Any]) -> tuple[int | None, str, bool]:
    """``(milliseconds, source, host_mode)`` of one exporter container.

    urfave/cli order: the flag, then ``DCGM_EXPORTER_INTERVAL``, then the
    exporter default. ``None`` milliseconds: the value cannot be resolved.
    """

    tokens = [str(item) for item in (container.get("command") or [])]
    tokens += [str(item) for item in (container.get("args") or [])]
    env = {
        str(item.get("name")): item
        for item in container.get("env") or []
        if isinstance(item, Mapping)
    }
    host = _flag_value(tokens, REMOTE_FLAGS) is not None or REMOTE_ENV in env
    flag = _flag_value(tokens, INTERVAL_FLAGS)
    if flag is not None:
        value, source = flag, "pod-args"
    elif INTERVAL_ENV in env:
        value, source = env[INTERVAL_ENV].get("value"), "pod-env"
    else:
        return EXPORTER_DEFAULT_INTERVAL_MS, "exporter-default", host
    text = str(value) if value is not None else ""
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        return None, f"{source}-unresolved", host
    return int(text), source, host


def _assumed(source: str, detail: str) -> dict[str, Any]:
    return {
        "collect_interval_ms": EXPORTER_DEFAULT_INTERVAL_MS,
        "source": source,
        "detail": detail,
        "dcgm_hostengine": "unknown",
        "pods": [],
        "rendered_collect_interval_ms": DCGM_EXPORTER_COLLECT_INTERVAL_MS,
    }


def exporter_interval(pods: Iterable[Any]) -> dict[str, Any]:
    """The collect interval and DCGM mode of the node's running exporter Pods.

    Several running Pods (a roll in flight) take the longest interval: the
    bound has to cover whichever one serves the scrape.
    """

    running = [
        pod
        for pod in pods
        if isinstance(pod, Mapping)
        and (pod.get("status") or {}).get("phase") == "Running"
    ]
    readings = []
    for pod in running:
        containers = (pod.get("spec") or {}).get("containers") or []
        exporter = [item for item in containers if item.get("name") == "exporter"]
        for container in exporter or containers[:1]:
            readings.append(_container_interval(container))
    if not readings:
        return _assumed("no-exporter-pod", "no running exporter Pod on the node")
    unresolved = sorted({source for value, source, _ in readings if value is None})
    if unresolved:
        return _assumed(unresolved[0], "the exporter's interval could not be read")
    modes = {"host" if host else "embedded" for _, _, host in readings}
    return {
        "collect_interval_ms": max(int(value or 0) for value, _, _ in readings),
        "source": ",".join(sorted({source for _, source, _ in readings})),
        "dcgm_hostengine": modes.pop() if len(modes) == 1 else "mixed",
        "pods": sorted(str((pod.get("metadata") or {}).get("name")) for pod in running),
        "rendered_collect_interval_ms": DCGM_EXPORTER_COLLECT_INTERVAL_MS,
    }


def read_exporter(fixture: Any) -> dict[str, Any]:
    """The node's exporter Pods through the case's GPU-cluster kubectl.

    Read-only and never fatal: a failed read takes the documented default and
    says why, so the result names a bound it could not measure.
    """

    try:
        listing = json.loads(
            fixture.regional.kubectl(
                "gpu",
                "get",
                "pod",
                "-l",
                f"app={EXPORTER_APP}",
                "--field-selector",
                f"spec.nodeName={fixture.node}",
                "-o",
                "json",
                timeout=60,
            )
        )
        items = listing.get("items") if isinstance(listing, Mapping) else None
        if not isinstance(items, list):
            raise ValueError("the Pod listing has no items")
    except Exception as exc:  # noqa: BLE001 - recorded; the documented default applies
        return _assumed("unreadable", f"{type(exc).__name__}: {exc}"[:300])
    return exporter_interval(items)


def _stamp(value: object) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else None


def load_onset(injection: Mapping[str, Any], started_at: datetime) -> dict[str, Any]:
    """Where the latency is measured from: the probe's onset reads, else the receipt.

    ``injection["load_onset"]`` holds the start of the last read of the loaded
    GPU still below the correlation thresholds (``idle_read_at``; ``None`` when
    the first read after the receipt already reached them) and the end of the
    first read at both (``loaded_read_at``). NVML -- and so DCGM and the
    collector -- crossed the thresholds between the two, so the latency runs
    from the idle read (else the receipt) and their gap is the first stage's
    width. Reads that are missing, never reached the thresholds or are out of
    order leave the receipt as the reference, and the start-up unpriced.
    """

    reads = injection.get("load_onset")
    receipt: dict[str, Any] = {
        "source": RECEIPT,
        "measured_from": started_at.isoformat(),
        "window_seconds": None,
        "excluded_seconds": 0.0,
        "reads": reads,
    }
    if not isinstance(reads, Mapping):
        return {**receipt, "detail": "the throttle receipt carries no onset reads"}
    idle_read, loaded_read = reads.get("idle_read_at"), reads.get("loaded_read_at")
    if loaded_read is None:
        why = reads.get("error") or "the probe's wait ended first"
        return {**receipt, "detail": f"no read reached the thresholds: {why}"}
    idle = started_at if idle_read is None else _stamp(idle_read)
    loaded = _stamp(loaded_read)
    if idle is None or loaded is None or not started_at <= idle < loaded:
        return {**receipt, "detail": "the onset reads are malformed or out of order"}
    return {
        "source": ONSET,
        "measured_from": idle.isoformat(),
        "window_seconds": round((loaded - idle).total_seconds(), 3),
        "excluded_seconds": round((idle - started_at).total_seconds(), 3),
        "reads": dict(reads),
    }


def measured_latency(delivered_at: str, bound: Mapping[str, Any]) -> float:
    """Seconds from the bound's reference (onset read or receipt) to the Store."""

    start = _stamp(bound["load_onset"]["measured_from"])
    end = _stamp(delivered_at)
    if start is None or end is None:
        raise ValueError("the latency's reference or delivery stamp is invalid")
    return (end - start).total_seconds()


def collector_window(
    quiet_stamp: str, confirming_at: datetime | None, interval: int
) -> dict[str, Any]:
    """The collector ticks from the quiet delivery to the confirming sample.

    Both stamps are ticks of the same loop and every tick is at least the
    interval after the previous one, so ``floor(seconds / interval)`` is the
    tick count and the remainder is the ticks' summed work: the path's ticks
    are among them. Valid while that work stays under one interval; a loop that
    lost a whole interval in the window reads as one tick more, which shrinks
    the allowance rather than widening it.
    """

    quiet = _stamp(quiet_stamp)
    if confirming_at is None or quiet is None:
        return {"seconds": None, "ticks": None, "overrun_seconds": 0.0}
    seconds = (confirming_at - quiet).total_seconds()
    ticks = math.floor((seconds + 1e-6) / interval) if seconds > 0 else 0
    overrun = max(0.0, seconds - ticks * interval) if ticks else 0.0
    return {
        "quiet_stamp": quiet_stamp,
        "confirming_sample_at": confirming_at.isoformat(),
        "seconds": round(seconds, 3),
        "ticks": ticks,
        "overrun_seconds": round(overrun, 3),
    }


def allowed_latency(
    interval: int,
    confirmation: int,
    exporter: Mapping[str, Any],
    quiet_stamp: str,
    records: Iterable[Any],
    delivery: Mapping[str, Any],
    onset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The bound, its components in seconds and what each was read from.

    ``exporter`` is :func:`read_exporter`'s answer, ``quiet_stamp`` the
    collector's last delivery before the load, ``records`` the window's raw
    evidence, ``delivery`` its ``collector_power_delivery.delivery_evidence``
    and ``onset`` :func:`load_onset`'s reference (the receipt when absent).
    """

    record_ids = list(delivery.get("record_ids") or [])
    payload: Mapping[str, Any] = next(
        (
            item.get("payload") or {}
            for item in records
            if isinstance(item, Mapping)
            and record_ids
            and item.get("record_id") == record_ids[0]
        ),
        {},
    )
    sampled = _stamp(payload.get("observed_at")) if payload else None
    delivered = _stamp(delivery.get("delivered_at"))
    window = collector_window(quiet_stamp, sampled, interval)
    transport = 0.0
    if delivered is not None and sampled is not None:
        transport = round((delivered - sampled).total_seconds(), 3)
    components = {
        "nvml_average_seconds": NVML_AVERAGE_SECONDS,
        "dcgm_watch_seconds": int(exporter["collect_interval_ms"]) / 1000,
        "collector_phase_seconds": float(interval),
        "confirmation_seconds": float(interval * (confirmation - 1)),
        "collector_tick_overrun_seconds": float(window["overrun_seconds"]),
        "delivery_seconds": transport,
    }
    observed = False
    if onset is not None and onset.get("source") == ONSET:
        observed = True
        excess = float(onset["window_seconds"]) - NVML_AVERAGE_SECONDS
        components["load_onset_excess_seconds"] = round(max(0.0, excess), 3)
    return {
        "total_seconds": round(sum(components.values()), 3),
        "components": components,
        "exporter": dict(exporter),
        "collector": {
            "interval_seconds": interval,
            "confirmation_samples": confirmation,
            "window": window,
        },
        "load_onset": dict(onset) if onset is not None else None,
        "stages_without_allowance": [] if observed else [WITHOUT_ALLOWANCE],
    }


def latency_errors(
    latency: float, bound: Mapping[str, Any], *, interval: int
) -> list[str]:
    """Why a measured latency breaks the bound, naming every component."""

    components = bound["components"]
    errors = []
    if components["delivery_seconds"] > interval:
        errors.append(
            f"the confirming batch took {components['delivery_seconds']:.1f}s from "
            f"its sample to the Store, more than one {interval}s collector "
            "interval: a slow or retried delivery, not sampling"
        )
    if not 0 <= latency <= bound["total_seconds"]:
        onset = bound.get("load_onset") or {}
        reference = {ONSET: "load onset read"}.get(onset.get("source"), "receipt")
        errors.append(
            f"the anomaly took {latency:.1f}s to deliver, more than the "
            f"{bound['total_seconds']:.1f}s bound ("
            + ", ".join(
                f"{name.removesuffix('_seconds')} {value:g}s"
                for name, value in components.items()
            )
            + f"), measured from the {reference}"
        )
    return errors
