"""DESTR-015 clock calibration: the pings inside the arm and collect sessions
tighten the clock-offset bound without relaxing the drift model.

Synthetic captures only. The runner's monotonic clock runs a fixed ``OFFSET``
ahead of the host's; each arm and collect request pays ``WIDE_TRIP`` (exec
setup, interpreter start, the handler's work) and the host samples its clock
at the end of it, while each ping pays ``PING_TRIP`` and is answered mid-trip.
Every expected bound is recomputed here from the samples with the unchanged
drift model (200 ppm of the span plus 50 ms on each side).
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import destr015_verdicts as verdicts
from scripts.e2e.regional.destr015_physical_evidence import (
    guaranteed_interval,
    physical_overlap_errors,
)
from tests.regional.test_acceptance_physical_interval_alignment import (
    BASE,
    SECOND,
    interval_fixture,
)
from tests.regional.test_destr015_detached_witness import both_witnesses

OFFSET = 950 * SECOND
WIDE_TRIP = 2_300_000_000
REPLY = 20_000_000
PING_TRIP = 20_000_000
PINGS = 8
ARM_HOST = 50 * SECOND
COLLECT_HOST = 150 * SECOND
CLOCK_KEYS = ("sent_ns", "received_ns", "monotonic_ns")
SERIAL = ["the actual reset process intervals do not prove parallel execution"]
UNORDERED = [
    "physical reset overlap is unproven: BoundaryDenied: "
    "physical interval clock exchange is unordered"
]


def drift(span: int) -> int:
    """The unchanged drift allowance on each side of the offset bound."""

    return span * 200 // 1_000_000 + 50_000_000


def request(kind: str, host: int) -> dict[str, Any]:
    """One arm or collect exchange whose host clock was sampled ``REPLY``
    before the response reached the runner."""

    received = OFFSET + host + REPLY
    return {
        "kind": kind,
        "sent_ns": received - WIDE_TRIP,
        "received_ns": received,
        "monotonic_ns": host,
    }


def pings(kind: str, after: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """``count`` back-to-back pings after the ``after`` exchange."""

    items = []
    sent = after["received_ns"]
    for sequence in range(count):
        items.append(
            {
                "kind": f"{kind}-ping",
                "sequence": sequence,
                "sent_ns": sent,
                "received_ns": sent + PING_TRIP,
                "monotonic_ns": sent - OFFSET + PING_TRIP // 2,
            }
        )
        sent += PING_TRIP
    return items


def calibrated(
    collect_host: int = COLLECT_HOST,
    *,
    arm_pings: int = PINGS,
    collect_pings: int = PINGS,
) -> list[dict[str, Any]]:
    """The exchanges one witness records, in recorded order."""

    arm, collect = request("arm", ARM_HOST), request("collect", collect_host)
    return [
        arm,
        *pings("arm", arm, arm_pings),
        collect,
        *pings("collect", collect, collect_pings),
    ]


def span_of(exchanges: list[dict[str, Any]]) -> int:
    return int(exchanges[-1]["received_ns"] - exchanges[0]["sent_ns"])


def offset_bound(exchanges: list[dict[str, Any]]) -> tuple[int, int]:
    """The offset interval the samples admit: the tightest send and receive
    stamps, each widened by the drift allowance."""

    allowance = drift(span_of(exchanges))
    lower = max(item["sent_ns"] - item["monotonic_ns"] for item in exchanges)
    upper = min(item["received_ns"] - item["monotonic_ns"] for item in exchanges)
    return lower - allowance, upper + allowance


def width(exchanges: list[dict[str, Any]]) -> int:
    lower, upper = offset_bound(exchanges)
    return upper - lower


def midway() -> int:
    """An overlap the pings prove and the request round trips alone cannot."""

    tight = width(calibrated())
    return (tight + width(calibrated(arm_pings=0, collect_pings=0))) // 2


def clocks_of(exchanges: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [{key: item[key] for key in CLOCK_KEYS} for item in exchanges]


def place_resets(actions: dict[str, dict[str, Any]], overlap: int) -> None:
    """node-a resets over [5 s, 15 s], node-b over [15 s - overlap, 25 s]."""

    actions["node-a"].update(started_ns=BASE + 5 * SECOND, ended_ns=BASE + 15 * SECOND)
    actions["node-b"].update(
        started_ns=BASE + 15 * SECOND - overlap, ended_ns=BASE + 25 * SECOND
    )


def overlap_errors(exchanges: list[dict[str, Any]], overlap: int) -> list[str]:
    """Grade two nodes calibrated by the same samples whose resets truly
    overlap by ``overlap`` nanoseconds."""

    captures, scopes, workflow, hosts = interval_fixture()
    for capture in captures.values():
        capture["clock_exchanges"] = clocks_of(exchanges)
    place_resets(
        {node: capture["end"]["actions"][0] for node, capture in captures.items()},
        overlap,
    )
    return physical_overlap_errors(
        captures, scopes=scopes, workflow=workflow, hosts=hosts
    )


@pytest.mark.parametrize("collect_host", [COLLECT_HOST, 770 * SECOND])
def test_pings_prove_an_overlap_the_request_round_trips_cannot(
    collect_host: int,
) -> None:
    pinged = calibrated(collect_host)
    bare = calibrated(collect_host, arm_pings=0, collect_pings=0)
    tight, wide = width(pinged), width(bare)
    assert tight == PING_TRIP + 2 * drift(span_of(pinged)), "the ping trip bounds"
    assert wide == WIDE_TRIP + 2 * drift(span_of(bare)), "the request trip bounds"
    overlap = (tight + wide) // 2
    assert overlap_errors(pinged, overlap) == [], "proven with the pings"
    assert overlap_errors(bare, overlap) == SERIAL, "not from the requests alone"
    assert overlap_errors(pinged, tight) == SERIAL, "the bound itself is not proof"
    assert overlap_errors(pinged, tight + 1) == [], "anything above it is"


def test_the_binding_and_the_verdict_summary_report_the_bound_the_proof_used() -> None:
    pinged = calibrated()
    binding = verdicts.clock_binding(pinged)
    assert binding["bounded"] is True and binding["exchanges"] == 2 + 2 * PINGS
    assert (binding["offset_min_ns"], binding["offset_max_ns"]) == offset_bound(pinged)
    assert binding["uncertainty_ns"] == width(pinged)
    assert binding["drift_allowance_ns"] == drift(span_of(pinged))
    assert verdicts.clock_calibration(pinged) == {
        "samples": 2 + 2 * PINGS,
        "min_round_trip_ms": PING_TRIP / 1e6,
        "offset_uncertainty_ms": round(width(pinged) / 1e6, 3),
    }
    bare = calibrated(arm_pings=0, collect_pings=0)
    assert verdicts.clock_calibration(bare) == {
        "samples": 2,
        "min_round_trip_ms": WIDE_TRIP / 1e6,
        "offset_uncertainty_ms": round(width(bare) / 1e6, 3),
    }
    assert verdicts.clock_calibration(bare[:1]) == {
        "samples": 1,
        "min_round_trip_ms": WIDE_TRIP / 1e6,
        "offset_uncertainty_ms": None,
    }, "a single exchange bounds nothing"
    assert verdicts.clock_calibration([]) == {
        "samples": 0,
        "min_round_trip_ms": None,
        "offset_uncertainty_ms": None,
    }


@pytest.mark.parametrize(
    ("arm_pings", "collect_pings"), [(PINGS, 0), (0, PINGS), (1, 0), (0, 1), (3, 2)]
)
def test_a_short_series_proves_with_the_pings_it_has(
    arm_pings: int, collect_pings: int
) -> None:
    short = calibrated(arm_pings=arm_pings, collect_pings=collect_pings)
    tight = width(short)
    assert tight == PING_TRIP + 2 * drift(span_of(short)), "one ping is enough"
    wide = width(calibrated(arm_pings=0, collect_pings=0))
    assert overlap_errors(short, (tight + wide) // 2) == []
    assert overlap_errors(short, tight) == SERIAL


def test_without_pings_the_proof_falls_back_to_the_wide_bound() -> None:
    bare = calibrated(arm_pings=0, collect_pings=0)
    wide = width(bare)
    assert overlap_errors(bare, wide + 1) == [], "a larger overlap is still proven"
    assert overlap_errors(bare, wide) == SERIAL, "never on a fabricated sample"


@pytest.mark.parametrize(
    "defect", ["sent-before-previous", "received-before-sent", "no-host-clock"]
)
def test_an_unordered_ping_refuses_the_proof(defect: str) -> None:
    pinged = calibrated()
    previous, ping = pinged[2], pinged[3]
    assert ping["kind"] == "arm-ping", ping
    if defect == "sent-before-previous":
        ping["sent_ns"] = previous["received_ns"] - 1
    elif defect == "received-before-sent":
        ping["received_ns"] = ping["sent_ns"] - 1
    else:
        ping["monotonic_ns"] = 0
    assert overlap_errors(pinged, midway()) == UNORDERED, defect


def test_the_inner_interval_charges_the_whole_offset_bound() -> None:
    pinged = calibrated()
    captures, scopes, workflow, hosts = interval_fixture()
    capture = captures["node-a"]
    capture["clock_exchanges"] = clocks_of(pinged)
    inner = guaranteed_interval(
        capture,
        scope=scopes["node-a"],
        workflow=workflow,
        host=hosts["node-a"]["after"],
        baseline=hosts["node-a"]["before"],
    )
    action, wall = capture["end"]["actions"][0], BASE - 50 * SECOND
    assert capture["end"]["wall_minus_monotonic_min_ns"] == wall
    true_start = action["started_ns"] - wall + OFFSET
    true_end = action["ended_ns"] - wall + OFFSET
    lower, upper = offset_bound(pinged)
    assert inner == (true_start + upper - OFFSET, true_end - (OFFSET - lower))
    shrink = PING_TRIP // 2 + drift(span_of(pinged))
    assert inner == (true_start + shrink, true_end - shrink), "symmetric pings"


def test_detached_records_carry_their_pings_into_the_proof() -> None:
    pinged, bare = calibrated(), calibrated(arm_pings=0, collect_pings=0)

    def graded(exchanges: list[dict[str, Any]]) -> list[str]:
        witnesses, workflow, hosts = both_witnesses()
        for item in witnesses.values():
            item["exchanges"] = [dict(entry) for entry in exchanges]
        place_resets(
            {
                node: item["collection"]["final"]["actions"][0]
                for node, item in witnesses.items()
            },
            midway(),
        )
        return verdicts.physical_witness_errors(
            witnesses, workflow=workflow, hosts=hosts
        )

    assert graded(pinged) == [], "the recorded pings reach the arithmetic"
    assert graded(bare) == SERIAL
    foreign = [
        {**pinged[1], "kind": "status"},
        {**pinged[1], "kind": "disarm-ping", "sent_ns": 0},
    ]
    mixed = [*pinged[:4], *foreign, *pinged[4:]]
    assert graded(mixed) == [], "foreign kinds never enter the calibration"
    records = both_witnesses()[0]["node-a"]
    capture = verdicts.witness_capture(records["receipt"], records["collection"], mixed)
    assert capture["clock_exchanges"] == clocks_of(pinged), "folded in recorded order"
