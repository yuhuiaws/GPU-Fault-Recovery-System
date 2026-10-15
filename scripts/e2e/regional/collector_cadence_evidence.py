"""COLLECT-001's arithmetic: what a suppressed delivery chain looks like on the node.

A healthy node's DCGM and host chains post one health summary per summary period
on a per-node stable phase that the cadence jitter (``gpu_fault.collectors.cadence``)
moves by ``ratio/2`` of the period either way, so two consecutive summaries are
``summary x (1 +- ratio)`` apart plus the collection tick that notices one is due.
The window and every bound below follow the node's live interval and summary
period (``collector.env``) and the collectors' jitter ratio, and reduce to the
unjittered ``2 x summary + 4 x interval`` window and ``summary - 2 x interval``
gap at ratio 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from gpu_fault.collectors.cadence import CadenceJitter


def jitter_ratio(env: Mapping[str, str]) -> float:
    """The jitter ratio the node's collectors run with.

    The installer never writes ``GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO`` into
    ``collector.env`` (and the probe reads only keys it writes), so the collectors
    fall back to the product default; reading it through ``CadenceJitter`` keeps
    the harness on that same default and validation.
    """

    return CadenceJitter.from_environment(env).ratio


def observation_seconds(chains: Mapping[str, Mapping[str, int]], ratio: float) -> int:
    """Two jittered summary cycles of the slowest chain, ticks at either end.

    Consecutive summaries can be ``summary x (1 + ratio)`` apart, so a window of
    ``2 x summary`` could hold a single healthy delivery.
    """

    return max(
        math.ceil(item["summary"] * 2 * (1 + ratio)) + item["interval"] * 4
        for item in chains.values()
    )


def summary_window(*, interval: int, summary: int, ratio: float) -> float:
    """The shortest gap two consecutive health summaries keep on a healthy node."""

    return summary * (1 - ratio) - interval * 2


def chain_verdict(
    kind: str,
    deliveries: Mapping[str, str],
    *,
    interval: int,
    summary: int,
    ratio: float,
    duration: int,
    edge_reasons: Mapping[str, list[str]],
) -> tuple[dict[str, Any], list[str]]:
    """One chain's observation and its errors.

    ``deliveries`` maps every in-window ``observed_at`` to its ``batch_id``;
    ``edge_reasons`` maps the batches that became raw evidence to their
    ``edge_filter_reasons``. Ingestion persists every batch except a pure
    periodic ``health-summary``, so a delivery without evidence was one.
    """

    stamps = sorted(
        (datetime.fromisoformat(stamp.replace("Z", "+00:00")), batch_id)
        for stamp, batch_id in deliveries.items()
    )
    gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(stamps, stamps[1:])]
    window = summary_window(interval=interval, summary=summary, ratio=ratio)
    # What the sampling loop would deliver with the filter off, against the most
    # summaries that fit at their shortest spacing: an order of magnitude apart,
    # which is what makes the criterion decidable.
    sampled = duration // interval
    allowed = math.floor(duration / max(window, interval)) + 1
    observed: dict[str, Any] = {
        "interval_seconds": interval,
        "summary_seconds": summary,
        "jitter_ratio": ratio,
        "summary_window_seconds": window,
        "collection_samples_in_window": sampled,
        "summary_deliveries_in_window": duration // summary,
        "deliveries_allowed": allowed,
        "delivered": len(stamps),
        "observed_at": [stamp.isoformat() for stamp, _ in stamps],
        "gaps_seconds": gaps,
    }
    if len(stamps) < 2:
        return observed, [f"{kind} delivered fewer than two batches in the window"]
    errors = []
    if len(stamps) > allowed:
        errors.append(
            f"{kind} delivered {len(stamps)} batches, closer to the "
            f"{sampled}-sample collection cadence than to the {allowed} "
            "the summary cadence allows"
        )
    if max(gaps) < window:
        errors.append(
            f"{kind} never went a full summary window without delivering, "
            f"so no suppression was observed: gaps {gaps}"
        )
    # A pure summary inside the summary window of the summary before it is the
    # schedule re-arming onto the phase it just served, not an edge: an edge
    # would have been persisted as evidence.
    repeated = [
        f"{right[0].isoformat()} after {gap:.1f} s"
        for left, right, gap in zip(stamps, stamps[1:], gaps)
        if gap < window
        and right[1] not in edge_reasons
        and "health-summary" in edge_reasons.get(left[1], ["health-summary"])
    ]
    if repeated:
        errors.append(
            f"{kind} delivered its health summary again inside one "
            f"{window:.0f} s summary window: {repeated}"
        )
    return observed, errors
