"""COLLECT-002 credits a delivery to the loaded GPU only when no other GPU in it
carries a candidate of its own.

A confirmed DCGM batch holds every GPU of the node and one batch-level
``candidate-confirmed``; it never names the key that confirmed. With every GPU
capped (the probe's design until now) the capped idle GPUs' power-violation
counters advance about one second per second and confirm their own duty-cycle
candidates around the loaded GPU's: live, plain EKS confirmed them one tick
after the loaded GPU, HyperPod on the same tick. The batches look the same
whichever key confirmed, so an idle GPU's confirmation could be timed as the
load's -- a slow pipeline read as a fast one, a false PASS. The case now reads
the Store's latest counters before the cap and credits a batch only when every
other GPU's counters and flags are zero or unchanged since their last earlier
reading and its draw is not correlated with its limit; the probe caps only the
loaded GPU, so on a healthy node the idle GPUs never qualify.
"""

from __future__ import annotations

import io
import json
import runpy
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource
from scripts.e2e.regional import collector_power_delivery as power
from scripts.e2e.regional import run_collector_acceptance as runner
from tests.regional._collector_power_receipt import candidate_record, load_receipt
from tests.regional.test_collect002_load_onset import EXPORTER_POD

UTC = timezone.utc
LOADED = "GPU-test"
IDLE = [f"GPU-0000000{index}-0000-0000-0000-000000000000" for index in range(8)]
VIOLATION = "power_violation_total_us"

#: GPU index -> (enforced limit W, draw W, utilization %, power violation us):
#: the two confirmed batches of a live HyperPod attempt (H200, host DCGM, every
#: GPU capped at 200 W, GPU 0 loaded). GPUs 1, 3, 4 and 6 count about 15 s of
#: violation per 15-s sample while idle; 2, 5 and 7 never move.
HP_K = {
    0: (200.0, 208.68, 99.0, 35445933.568),
    1: (200.0, 74.953, 0.0, 51411043.552),
    2: (200.0, 76.213, 0.0, 0.0),
    3: (200.0, 74.423, 0.0, 49357632.576),
    4: (200.0, 74.11, 0.0, 48972001.536),
    5: (200.0, 74.884, 0.0, 0.0),
    6: (200.0, 76.93, 0.0, 46618376.352),
    7: (200.0, 76.862, 0.0, 0.0),
}
HP_NEXT = {
    0: (200.0, 208.824, 99.0, 50345921.888),
    1: (200.0, 74.928, 0.0, 66511081.952),
    2: (200.0, 76.335, 0.0, 0.0),
    3: (200.0, 74.435, 0.0, 64257605.888),
    4: (200.0, 74.081, 0.0, 64072043.008),
    5: (200.0, 74.902, 0.0, 0.0),
    6: (200.0, 76.916, 0.0, 61618377.024),
    7: (200.0, 76.88, 0.0, 0.0),
}
#: The same of a live plain-EKS attempt (embedded DCGM, every GPU capped):
#: there the idle GPUs 1, 2, 3 and 7 confirmed one tick after the loaded GPU.
EKS_K = {
    0: (200.0, 203.835, 99.0, 32757659.968),
    1: (200.0, 74.246, 0.0, 42176095.328),
    2: (200.0, 74.164, 0.0, 41748499.488),
    3: (200.0, 72.711, 0.0, 41222776.864),
    4: (200.0, 73.817, 0.0, 0.0),
    5: (200.0, 72.302, 0.0, 0.0),
    6: (200.0, 74.314, 0.0, 0.0),
    7: (200.0, 72.942, 0.0, 38037390.944),
}
EKS_NEXT = {
    0: (200.0, 201.881, 99.0, 47857660.48),
    1: (200.0, 74.228, 0.0, 57076062.72),
    2: (200.0, 74.188, 0.0, 56748488.32),
    3: (200.0, 72.663, 0.0, 56222786.976),
    4: (200.0, 73.807, 0.0, 0.0),
    5: (200.0, 72.445, 0.0, 0.0),
    6: (200.0, 74.376, 0.0, 0.0),
    7: (200.0, 72.942, 0.0, 53037396.896),
}


def stamp(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


#: Each attempt's GPU-chain stamps: opening and quiet deliveries, the request,
#: the load-start receipt, the onset reads (HyperPod only), the two confirmed
#: batches (sample, payload ingest, stored ingest) and the stamp the poll saw.
HP = SimpleNamespace(
    opening=stamp("2026-10-21T06:45:12.278582"),
    quiet=stamp("2026-10-21T06:50:13.637691"),
    requested=stamp("2026-10-21T06:50:17.607184"),
    receipt=stamp("2026-10-21T06:50:31.517414"),
    onset={
        "idle_read_at": stamp("2026-10-21T06:50:42.898427").isoformat(),
        "loaded_read_at": stamp("2026-10-21T06:50:43.298592").isoformat(),
    },
    k=(
        stamp("2026-10-21T06:51:28.934062"),
        stamp("2026-10-21T06:51:29.001154"),
        stamp("2026-10-21T06:51:29.066772"),
    ),
    next=(
        stamp("2026-10-21T06:51:43.972562"),
        stamp("2026-10-21T06:51:44.242810"),
        stamp("2026-10-21T06:51:44.311853"),
    ),
    batches=(HP_K, HP_NEXT),
)
EKS = SimpleNamespace(
    opening=stamp("2026-10-21T06:09:31.885378"),
    quiet=stamp("2026-10-21T06:14:48.451323"),
    requested=stamp("2026-10-21T06:15:03.761640"),
    receipt=stamp("2026-10-21T06:15:15.514116"),
    onset=None,
    k=(
        stamp("2026-10-21T06:16:03.760716"),
        stamp("2026-10-21T06:16:03.820844"),
        stamp("2026-10-21T06:16:03.884096"),
    ),
    next=(
        stamp("2026-10-21T06:16:18.800303"),
        stamp("2026-10-21T06:16:18.867325"),
        stamp("2026-10-21T06:16:18.930633"),
    ),
    batches=(EKS_K, EKS_NEXT),
)


def uuid_of(index: int) -> str:
    return LOADED if index == 0 else IDLE[index]


def batch(
    stamps: Sequence[datetime],
    gpus: dict[int, tuple[float, float, float, float]],
    **extra: dict[str, float],
) -> dict[str, Any]:
    """A confirmed DCGM batch of eight GPUs; ``extra`` adds fields per index."""

    observed, ingested, stored = stamps
    record = candidate_record(observed)
    record["payload"]["ingested_at"] = ingested.isoformat()
    record["ingested_at"] = stored.isoformat()
    names = ("power_limit_w", "power_usage_w", "gpu_utilization_percent", VIOLATION)
    record["payload"]["samples"] = [
        {
            "canonical_name": name,
            "gpu_uuid": uuid_of(index),
            "gpu_index": str(index),
            "value": value,
        }
        for index, values in gpus.items()
        for name, value in [
            *zip(names, values, strict=True),
            ("thermal_violation_total_us", 0.0),
            *extra.get(f"gpu{index}", {}).items(),
        ]
    ]
    return record


def uncapped(
    gpus: dict[int, tuple[float, float, float, float]], counters: dict[int, float]
) -> dict[int, tuple[float, float, float, float]]:
    """The batch as it reads when only the loaded GPU is capped: the idle GPUs
    keep their 700 W default and their counters stand still."""

    return {
        index: values if index == 0 else (700.0, values[1], 0.0, counters[index])
        for index, values in gpus.items()
    }


def reference(
    observed: datetime, counters: dict[int, float], *, error: str | None = None
) -> dict[str, Any]:
    """The Store's latest counters read after the quiet delivery, before the cap."""

    return {
        "samples": [
            {
                "observed_at": observed.isoformat(),
                "source": "DCGM_EXPORTER",
                "canonical_name": name,
                "gpu_uuid": uuid_of(index),
                "gpu_index": str(index),
                "value": value if name == VIOLATION else 0.0,
            }
            for index, value in counters.items()
            for name in (VIOLATION, "thermal_violation_total_us")
        ],
        "error": error,
    }


ZERO = dict.fromkeys(range(8), 0.0)
#: Counters left over by an earlier all-GPU cap on the same boot: they stay put.
RESIDUAL = {index: HP_NEXT[index][3] for index in range(8)}
#: The collector's default duty-cycle threshold (0.05) over a 15-s interval, in us.
FLOOR = 750_000.0


def evidence(
    records: list[dict[str, Any]], stored: dict[str, Any] | None
) -> power.DeliveryEvidence:
    return power.delivery_evidence(
        records,
        cluster_id="cluster-a",
        node_id="node-a",
        load_start=load_receipt("run-a", HP.receipt)["load_start"],
        expected_power_limit_w=200.0,
        observed_until=HP.next[2] + timedelta(seconds=30),
        reference=stored,
    )


def run_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    live: SimpleNamespace,
    records: list[dict[str, Any]],
    stored: dict[str, Any] | Exception,
    *,
    polled: datetime,
) -> dict[str, Any]:
    """``run_collect002`` on an attempt's stamps; the Store read is the reference."""

    current = [live.requested]
    monkeypatch.setattr(
        runner,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: current[0], fromisoformat=datetime.fromisoformat
        ),
    )
    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    monkeypatch.setattr(runner, "recent_evidence", lambda *a, **k: records)
    stamps = iter([live.opening, live.quiet, polled])

    def chain_stamp(_fixture: object) -> str:
        value = next(stamps)
        if value == polled:
            current[0] = polled + timedelta(seconds=29.9)
        return value.isoformat()

    monkeypatch.setattr(runner, "gpu_metrics_stamp", chain_stamp)

    def cpu_python(script: str, *arguments: str, **options: object) -> Any:
        assert script == power.POWER_REFERENCE, "the only Store read left"
        assert arguments[:3] == ("cluster-a", "node-a", "_violation_total_us")
        if isinstance(stored, Exception):
            raise stored
        return stored

    def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, Any]:
        if command == "throttle-gpu":
            receipt = load_receipt(
                arguments[arguments.index("--run-id") + 1], live.receipt
            )
            current[0] = live.receipt + timedelta(seconds=12.0)
            return {
                **receipt,
                **({} if live.onset is None else {"load_onset": live.onset}),
                "mutation_started": True,
                "timer_armed": True,
                "load_unit_active": True,
            }
        current[0] += timedelta(seconds=10)
        return {
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }

    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            kubectl=lambda *a, **k: json.dumps({"items": [EXPORTER_POD]}),
            cpu_python=cpu_python,
        ),
        execute=execute,
        snapshot=lambda: {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "3",
            },
            "gpu_power": [
                {"index": 0, "power_limit_w": 700.0, "power_default_limit_w": 700.0}
            ],
        },
    )
    return runner.run_collect002(fixture, tmp_path, 1)


def attribution_error(gpus: str) -> str:
    return (
        "no confirmed DCGM power candidate in the load window can be credited to "
        "the loaded GPU 0: every one also carries another GPU's candidate "
        f"(GPU {gpus}), which could be its confirmation "
        "(delivery_evidence.attribution.ignored)"
    )


@pytest.mark.parametrize("live", [HP, EKS], ids=["hyperpod", "plain-eks"])
def test_the_live_all_gpu_cap_batches_are_no_longer_credited_to_the_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, live: SimpleNamespace
) -> None:
    """Both live attempts passed on these batches. On HyperPod the capped idle
    GPUs confirmed on the loaded GPU's own tick, on plain EKS one tick later;
    yet the same two batches are what the case reads had the idle GPUs confirmed
    first and the loaded GPU a tick later -- then the batch it timed was theirs
    (plain EKS: 48.3 s credited, the load's own 63.4 s past the 61.4-s bound)."""

    records = [batch(live.k, live.batches[0]), batch(live.next, live.batches[1])]

    result = run_attempt(
        monkeypatch,
        tmp_path,
        live,
        records,
        reference(live.quiet, ZERO),
        polled=live.next[0],
    )

    idle = [str(index) for index, item in live.batches[0].items() if index and item[3]]
    assert result["verdict"] == "FAIL"
    assert result["errors"] == [attribution_error(", ".join(idle))]
    assert result["delivered_at"] is None and result["delivery_latency_seconds"] is None
    attribution = result["delivery_evidence"]["attribution"]
    assert attribution["gpu_index"] == 0 and attribution["gpu_uuid"] == LOADED
    assert attribution["record_id"] is None
    assert [item["record_id"] for item in attribution["ignored"]] == [
        record["record_id"] for record in records
    ]
    first = attribution["ignored"][0]
    assert first["delivered_at"] == live.k[1].isoformat()
    assert [gpu["gpu_index"] for gpu in first["other_gpus"]] == idle
    assert {gpu["power_limit_w"] for gpu in first["other_gpus"]} == {200.0}, (
        "every one of them was capped"
    )
    assert first["other_gpus"][0]["candidates"] == [
        f"{VIOLATION} 0.0 -> {live.batches[0][int(idle[0])][3]!r}"
    ]
    assert attribution["reference"] == {
        "readings": 16,
        "breach_floor_us": FLOOR,
        "error": None,
    }


def test_with_only_the_loaded_gpu_capped_its_own_confirmation_is_credited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The HyperPod attempt as it reads with the probe's single cap: the idle
    GPUs keep 700 W and their counters -- zero, or left over by an earlier
    all-GPU cap -- equal the Store's values from before the cap."""

    k = batch(HP.k, uncapped(HP_K, RESIDUAL))
    later = batch(HP.next, uncapped(HP_NEXT, RESIDUAL))

    result = run_attempt(
        monkeypatch,
        tmp_path,
        HP,
        [k, later],
        reference(HP.quiet, RESIDUAL),
        polled=HP.next[0],
    )

    assert result["verdict"] == "PASS", result["errors"]
    assert result["delivered_at"] == HP.k[1].isoformat()
    assert result["delivery_latency_seconds"] == pytest.approx(46.102727)
    assert result["allowed_latency_seconds"] == 61.363
    delivery = result["delivery_evidence"]
    assert delivery["record_ids"] == [k["record_id"], later["record_id"]]
    assert delivery["attribution"] == {
        "gpu_index": 0,
        "gpu_uuid": LOADED,
        "record_id": k["record_id"],
        "ignored": [],
        "reference": {"readings": 16, "breach_floor_us": FLOOR, "error": None},
    }


def test_a_loaded_gpu_that_confirms_a_tick_late_still_fails_on_the_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Single cap, the idle GPUs out of the way: the plain-EKS stamps with the
    loaded GPU's confirmation only in the later batch. The credited delivery is
    63.4 s from the receipt, past the 61.4-s bound: a slow chain is not hidden."""

    later = batch(EKS.next, uncapped(EKS_NEXT, ZERO))

    result = run_attempt(
        monkeypatch,
        tmp_path,
        EKS,
        [later],
        reference(EKS.quiet, ZERO),
        polled=EKS.next[0],
    )

    assert result["verdict"] == "FAIL"
    assert result["delivery_latency_seconds"] == pytest.approx(63.353209)
    assert result["delivery_evidence"]["attribution"]["record_id"] == later["record_id"]
    assert len(result["errors"]) == 1, result["errors"]
    assert result["errors"][0].startswith("the anomaly took 63.4s"), result["errors"]
    assert result["errors"][0].endswith("measured from the receipt"), result["errors"]


def test_an_unreadable_reference_is_recorded_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the Store's earlier values a nonzero idle counter in the only
    confirmed batch cannot be proven unchanged: the case fails, naming why."""

    k = batch(HP.k, uncapped(HP_K, RESIDUAL))

    result = run_attempt(
        monkeypatch, tmp_path, HP, [k], RuntimeError("exec refused"), polled=HP.k[0]
    )

    assert result["verdict"] == "FAIL"
    assert result["errors"] == [attribution_error("1, 3, 4, 6")]
    attribution = result["delivery_evidence"]["attribution"]
    assert attribution["reference"] == {
        "readings": 0,
        "breach_floor_us": FLOOR,
        "error": "RuntimeError: exec refused",
    }
    assert attribution["ignored"][0]["other_gpus"][0]["candidates"] == [
        f"{VIOLATION} {RESIDUAL[1]!r} with no earlier reading"
    ]


def test_a_later_batch_is_credited_once_an_earlier_one_proves_the_idle_gpu_still():
    """GPU 3's counter moved between the Store's reading and the first batch, so
    that batch's confirmation could be GPU 3's; in the next batch it stands
    still, so that one is credited -- later, never earlier, than the truth."""

    moved = {**RESIDUAL, 3: RESIDUAL[3] + 2_000_000.0}
    first = batch(HP.k, uncapped(HP_K, moved))
    second = batch(HP.next, uncapped(HP_NEXT, moved))

    result = evidence([second, first], reference(HP.quiet, RESIDUAL))

    assert result["errors"] == []
    assert result["record_ids"] == [second["record_id"]]
    assert result["delivered_at"] == HP.next[1].isoformat()
    ignored = result["attribution"]["ignored"]
    assert [item["record_id"] for item in ignored] == [first["record_id"]]
    assert ignored[0]["other_gpus"] == [
        {
            "gpu_index": "3",
            "gpu_uuid": IDLE[3],
            "power_limit_w": 700.0,
            "candidates": [f"{VIOLATION} {RESIDUAL[3]!r} -> {moved[3]!r}"],
        }
    ]


def test_an_earlier_batch_in_the_window_is_a_reading_too() -> None:
    """No reference at all: the first batch is unproven, the second is credited
    because the first holds the same counters (they only grow)."""

    first = batch(HP.k, uncapped(HP_K, RESIDUAL))
    second = batch(HP.next, uncapped(HP_NEXT, RESIDUAL))

    result = evidence([first, second], None)

    assert result["record_ids"] == [second["record_id"]]
    assert result["attribution"]["reference"] == {
        "readings": 0,
        "breach_floor_us": 0.0,
        "error": "not read",
    }
    assert [
        gpu["gpu_index"] for gpu in result["attribution"]["ignored"][0]["other_gpus"]
    ] == ["1", "3", "4", "6"]


@pytest.mark.parametrize(
    ("advance", "named"), [(120.0, False), (FLOOR - 1, False), (FLOOR + 1, True)]
)
def test_an_advance_below_the_collectors_threshold_cannot_have_breached(
    advance: float, named: bool
) -> None:
    """A healthy GPU's violation counter may creep by microseconds: an advance
    under the duty-cycle threshold over one interval cannot have breached."""

    moved = {**RESIDUAL, 1: RESIDUAL[1] + advance}
    record = batch(HP.k, uncapped(HP_K, moved))
    graded = {**reference(HP.quiet, RESIDUAL), "breach_floor_us": FLOOR}

    result = evidence([record], graded)

    assert (result["delivered_at"] is None) is named
    assert result["attribution"]["reference"]["breach_floor_us"] == FLOOR
    ungraded = evidence([record], reference(HP.quiet, RESIDUAL))
    assert ungraded["attribution"]["ignored"][0]["other_gpus"][0]["candidates"] == [
        f"{VIOLATION} {RESIDUAL[1]!r} -> {moved[1]!r}"
    ], "without the node's threshold any advance names its GPU"


def test_a_counter_that_fell_cannot_be_bounded() -> None:
    """A reset (GPU reset, driver reload) restarts the counter, which may then
    have advanced on this very tick: the GPU is named."""

    fell = {**RESIDUAL, 4: 1_500_000.0}
    record = batch(HP.k, uncapped(HP_K, fell))

    result = evidence(
        [record], {**reference(HP.quiet, RESIDUAL), "breach_floor_us": FLOOR}
    )

    assert result["attribution"]["ignored"][0]["other_gpus"] == [
        {
            "gpu_index": "4",
            "gpu_uuid": IDLE[4],
            "power_limit_w": 700.0,
            "candidates": [f"{VIOLATION} fell from {RESIDUAL[4]!r} to 1500000.0"],
        }
    ]


def test_another_gpu_drawing_at_its_own_limit_names_itself() -> None:
    """A second loaded GPU (someone else's job) has a correlation candidate."""

    gpus = uncapped(HP_K, RESIDUAL)
    gpus[5] = (700.0, 690.0, 97.0, RESIDUAL[5])

    result = evidence([batch(HP.k, gpus)], reference(HP.quiet, RESIDUAL))

    assert result["delivered_at"] is None
    assert result["errors"] == [attribution_error("5")]
    assert result["attribution"]["ignored"][0]["other_gpus"][0]["candidates"] == [
        "power_limit_correlation: 690 W at a 700 W limit, 97% utilization"
    ]


@pytest.mark.parametrize(
    ("before", "now", "named"),
    [(0.0, 1.0, True), (1.0, 1.0, False), (None, 1.0, True), (None, 0.0, False)],
)
def test_a_pending_flag_names_its_gpu_unless_it_was_already_set(
    before: float | None, now: float, named: bool
) -> None:
    """``row_remap_pending`` confirms on its own while nonzero: newly set, or
    set without an earlier reading, it could be the batch's confirmation."""

    stored = reference(HP.quiet, RESIDUAL)
    if before is not None:
        stored["samples"].append(
            {
                **stored["samples"][2],
                "canonical_name": "row_remap_pending",
                "value": before,
            }
        )
    record = batch(HP.k, uncapped(HP_K, RESIDUAL), gpu1={"row_remap_pending": now})

    result = evidence([record], stored)

    assert (result["delivered_at"] is None) is named
    if named:
        expected = (
            "row_remap_pending 1.0 with no earlier reading"
            if before is None
            else "row_remap_pending 0.0 -> 1.0"
        )
        assert result["attribution"]["ignored"][0]["other_gpus"][0]["candidates"] == [
            expected
        ]


def test_readings_from_elsewhere_or_from_later_prove_nothing() -> None:
    """Only this node's DCGM readings from before the batch count: an
    nvidia-smi reading, a later reading, an unparseable stamp, another node's
    batch and a node-level sample leave GPU 1 unproven."""

    stored = reference(HP.quiet, {1: RESIDUAL[1]})
    smi, late, broken = (dict(stored["samples"][0]) for _ in range(3))
    smi["source"] = "NVIDIA_SMI"
    late["observed_at"] = (HP.k[0] + timedelta(seconds=1)).isoformat()
    broken["observed_at"] = "soon"
    summaries = [batch((HP.quiet,) * 3, uncapped(HP_K, RESIDUAL)) for _ in range(2)]
    summaries[0]["payload"]["node_id"] = "node-b"
    summaries[1]["payload"]["samples"] = [None]
    for summary in summaries:
        summary["payload"]["edge_filter_reasons"] = ["health-summary"]
    gpus = uncapped(HP_K, RESIDUAL)
    record = batch(HP.k, {0: gpus[0], 1: gpus[1]})
    record["payload"]["samples"].append(
        {"canonical_name": "fabric_manager_status", "value": 3.0}
    )

    result = evidence(
        [record, *summaries, "not a record"],
        {"samples": [smi, late, broken, "not a sample"], "error": None},
    )

    assert result["errors"] == [
        "record[3]: raw evidence record must be an object",
        attribution_error("1"),
    ]
    assert result["attribution"]["ignored"][0]["other_gpus"][0]["candidates"] == [
        f"{VIOLATION} {RESIDUAL[1]!r} with no earlier reading"
    ]
    assert result["attribution"]["reference"]["readings"] == 4


def test_the_reference_is_read_once_through_the_store_and_never_raises() -> None:
    calls: list[tuple[str, ...]] = []
    answer = reference(HP.quiet, ZERO)

    def cpu_python(script: str, *arguments: str) -> Any:
        assert script == power.POWER_REFERENCE
        calls.append(arguments)
        return answer

    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"), cpu_python=cpu_python
        ),
    )

    assert power.power_reference(fixture, {}, 15) == {
        "samples": answer["samples"],
        "error": None,
        "duty_cycle_threshold": 0.05,
        "breach_floor_us": FLOOR,
    }
    assert calls == [
        (
            "cluster-a",
            "node-a",
            "_violation_total_us",
            "retired_pages_pending",
            "row_remap_failure",
            "row_remap_pending",
            "xid_last_error",
        )
    ]
    for broken, error in (
        ({"records": []}, "ValueError: the Store's answer has no samples"),
        ("[]", "ValueError: the Store's answer has no samples"),
    ):
        fixture.regional.cpu_python = lambda *a, value=broken: value
        assert power.power_reference(fixture, {}, 15) == {
            "samples": [],
            "error": error,
            "duty_cycle_threshold": 0.05,
            "breach_floor_us": FLOOR,
        }
    fixture.regional = SimpleNamespace()
    failed = power.power_reference(fixture, {}, 15)
    assert failed["samples"] == [] and failed["error"].startswith("AttributeError")


@pytest.mark.parametrize(
    ("value", "threshold", "floor"),
    [
        ("0.2", 0.2, 6_000_000.0),
        ("1", 1.0, 30_000_000.0),
        ("0", 0.0, 0.0),
        ("1.5", 1.5, 0.0),
        ("nan", None, 0.0),
        ("five percent", 0.0, 0.0),
    ],
)
def test_the_breach_floor_follows_the_nodes_threshold_or_counts_any_advance(
    value: str, threshold: float | None, floor: float
) -> None:
    """``collector.env``'s duty-cycle threshold x the interval; a value the
    collector itself would refuse leaves no floor: every advance names its GPU."""

    fixture = SimpleNamespace(node="node-a", regional=SimpleNamespace())
    graded = power.power_reference(
        fixture, {power.DUTY_CYCLE_KEY: value, "GPU_FAULT_OTHER": "1"}, 30
    )

    assert graded["breach_floor_us"] == floor
    if threshold is not None:
        assert graded["duty_cycle_threshold"] == threshold


def test_the_reference_script_reads_only_the_watched_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Pod-side script, run here against an in-memory Store answer."""

    observed = HP.quiet
    latest = [
        GpuMetricLatest(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=observed,
            source=GpuMetricSource.DCGM_EXPORTER,
            sample=GpuMetricSample(
                metric_name=name.upper(),
                canonical_name=name,
                value=value,
                gpu_index="1",
                gpu_uuid=IDLE[1],
            ),
        )
        for name, value in (
            (VIOLATION, 5.5),
            ("thermal_violation_total_us", 0.0),
            ("row_remap_pending", 0.0),
            ("power_usage_w", 74.0),
        )
    ]
    calls: list[tuple[str, str]] = []

    def list_latest(cluster_id: str, node_id: str) -> list[GpuMetricLatest]:
        calls.append((cluster_id, node_id))
        return latest

    store = SimpleNamespace(list_gpu_metrics_latest=list_latest)
    app = ModuleType("gpu_fault.app")
    app.ApplicationContext = SimpleNamespace(  # type: ignore[attr-defined]
        from_environment=lambda: SimpleNamespace(store=store)
    )
    monkeypatch.setitem(sys.modules, "gpu_fault.app", app)
    script = tmp_path / "reference.py"
    script.write_text(power.POWER_REFERENCE)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "cluster-a",
            "node-a",
            "_violation_total_us",
            "row_remap_pending",
        ],
    )
    output = io.StringIO()
    with redirect_stdout(output):
        runpy.run_path(str(script), run_name="__main__")

    assert calls == [("cluster-a", "node-a")]
    assert json.loads(output.getvalue()) == {
        "samples": [
            {
                "observed_at": observed.isoformat().replace("+00:00", "Z"),
                "source": "DCGM_EXPORTER",
                "canonical_name": name,
                "gpu_uuid": IDLE[1],
                "gpu_index": "1",
                "value": value,
            }
            for name, value in (
                (VIOLATION, 5.5),
                ("thermal_violation_total_us", 0.0),
                ("row_remap_pending", 0.0),
            )
        ]
    }
