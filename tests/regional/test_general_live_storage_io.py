"""GF-GENERAL-LIVE-006 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_storage_io as storage_io
from scripts.e2e.regional.general_live_common import STORE_POLL_SECONDS
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    store_script,
    workflow,
)
from tests.regional._general_live_storage_fakes import (
    SPECS,
    node_probe,
    probes,
    storage_probe,
    survey,
    window_probe,
)


def test_sustained_util_passes(tmp_path: Path) -> None:
    # Without dm_delay the plain linear device path is attempted and the scripted
    # util samples carry it through; with dm_delay it runs on a 20 ms delay.
    spec = SPECS["sustained_util"]
    node = node_probe(survey(dm_delay=False))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"util_percent": 100.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    inc = incident(
        "inc-io",
        "disk_io_util_percent",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io",
        "inc-io",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="disk_io_util_percent",
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    doc = {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                "2026-10-24T00:02:00+00:00", "disk_io_util_percent", 99.0, device="dm-7"
            )
        ]
        + [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "disk_io_util_percent",
                2.0,
                device="dm-7",
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node, storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_short_flush_passes(tmp_path: Path) -> None:
    spec = SPECS["short_flush"]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def _reference_iostat_result(tmp_path: Path, standard_await_ms: float):
    spec = SPECS["reference_iostat"]
    node = node_probe(survey())
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": standard_await_ms}}}
            for _ in range(30)
        ],
        "next_offset": 30,
        "total": 30,
    }
    inc = incident(
        "inc-io",
        "disk_io_await_ms",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io", "inc-io", ["FREEZE_EVIDENCE", "VALIDATE_HOST"], status="RUNNING"
    )
    doc = {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                f"2026-10-24T00:0{i}:00+00:00", "disk_io_await_ms", 200.0, device="dm-7"
            )
            for i in range(4)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    # The detection poll reads the loaded records; once the writer stopped the
    # product delivers the recovered edge (the signal-clearing proof).
    import itertools

    recovered = {
        **doc,
        "batches": [
            host_batch(
                f"2026-10-24T00:1{i}:00+00:00", "disk_io_await_ms", 0.3, device="dm-7"
            )
            for i in range(2)
        ],
    }
    polls = itertools.count(1)

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        return doc if next(polls) == 1 else recovered

    regional = FakeRegional(cpu_python=product)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node, storage_probe(), window_probe()),
    )
    return spec, spec.execute(ctx)


def test_reference_iostat_fails_on_divergent_await(tmp_path: Path) -> None:
    # The product reports 200 ms while the oracle's standard await of the same
    # device and interval is 10 ms: the spec FAIL, whatever the registry expects.
    spec, result = _reference_iostat_result(tmp_path, standard_await_ms=10.0)
    assert result["verdict"] == "FAIL"
    assert result["diverged"] is True


def test_reference_iostat_expects_pass_and_passes_on_matching_await(
    tmp_path: Path,
) -> None:
    # Batch 172: the host collector reports iostat's await (fields 4+8 over
    # 1+5), so the registry no longer carries the weighted-await product gap and
    # a product value within 10 % / 5 ms of the oracle's is the expected PASS.
    spec, result = _reference_iostat_result(tmp_path, standard_await_ms=198.0)
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec
    assert result["verdict"] == "PASS", result["errors"]
    assert result["diverged"] is False
    assert result["product_await_ms"] == [200.0] * 4


def test_zero_completed_io_and_counter_reset_and_recovery(tmp_path: Path) -> None:
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    for variant in ("zero_completed_io", "counter_reset"):
        spec = SPECS[variant]
        regional = FakeRegional(cpu_python=store_script({"activity": empty}))
        ctx = make_context(
            spec,
            tmp_path=tmp_path,
            regional=regional,
            probes=probes(node_probe(survey()), storage_probe(), window_probe()),
        )
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])
    # recovery
    spec = SPECS["recovery"]
    inc = incident(
        "inc-io",
        "disk_io_util_percent",
        device="dm-7",
        state="RECOVERED",
        workflow="wf-io",
    )
    doc = {
        "incidents": [inc],
        "workflows": [
            workflow(
                "wf-io",
                "inc-io",
                ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
                status="SUCCEEDED",
            )
        ],
        "batches": [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "disk_io_util_percent",
                1.0,
                device="dm-7",
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    # The recovery variant times its detection from the oracle's crossing.
    node = node_probe(survey())
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"util_percent": 100.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node, storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["crossing"]["value"] == 100.0, result["crossing"]


def test_sustained_latency_requires_dm_delay(tmp_path: Path) -> None:
    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["sustained_latency"]
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    settings = make_settings(spec)
    settings = type(settings)(**{**settings.__dict__, "approvals": ()})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings,
        probes=probes(
            node_probe(survey(dm_delay=False)), storage_probe(), window_probe()
        ),
    )
    with pytest.raises(VariantNotRun) as caught:
        spec.execute(ctx)
    assert caught.value.reason == "CAPABILITY_ABSENT"


def _dm_create_arguments(store: FakeProbe) -> tuple[str, ...]:
    created = [call for call in store.calls if call[0] == "dm-create"]
    assert len(created) == 1, store.calls
    return created[0]


@pytest.mark.parametrize(
    ("variant", "loaded", "delay", "load"),
    [
        ("sustained_util", True, "20", False),
        ("sustained_util", False, "0", False),
        ("sustained_latency", True, "200", False),
        ("sustained_latency", False, "200", True),
    ],
    ids=["util-delay", "util-linear", "latency-loaded", "latency-loads"],
)
def test_dm_io_load_runs_natively_with_or_without_a_loaded_dm_delay(
    tmp_path: Path, variant: str, loaded: bool, delay: str, load: bool
) -> None:
    # The 2026-10-24/25 TimeoutExpired was the probe Pod's dmsetup waiting for a
    # udev cookie it could never get (see the storage probe), not the delay
    # target, so neither variant defers on dm_delay any more: util uses a 20 ms
    # delay when one is loaded, latency a 200 ms one -- loaded for the run
    # (KERNEL_MODULE_LOAD, the variant's approval) when the node lacks it.
    spec = SPECS[variant]
    store = storage_probe()
    metric = (
        "disk_io_util_percent" if variant == "sustained_util" else "disk_io_await_ms"
    )
    detected = {"incidents": [incident("inc-io", metric, device="dm-7")]}
    regional = FakeRegional(cpu_python=store_script({"activity": detected}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(dm_delay=loaded)), store, window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)  # no scripted oracle samples: the injection is unproven
    arguments = _dm_create_arguments(store)
    assert arguments[arguments.index("--delay-ms") + 1] == delay, arguments
    assert ("--load-module" in arguments) is load, arguments
    assert [call[0] for call in store.calls[:2]] == ["dm-create", "dm-status"], (
        store.calls
    )


@pytest.mark.parametrize(
    ("loaded", "approved", "delay", "load"),
    [(True, False, "200", False), (False, True, "200", True)],
    ids=["delay-loaded", "delay-loaded-for-the-run"],
)
def test_reference_iostat_lifts_the_awaits_with_dm_delay(
    tmp_path: Path, loaded: bool, approved: bool, delay: str, load: bool
) -> None:
    """The weighted/standard comparison needs awaits above the 5 ms tolerance
    floor: dm-delay 200 ms, loaded for the run under the optional
    KERNEL_MODULE_LOAD when the node lacks it (before, only a dm_delay someone
    else had loaded counted, and this site never has one)."""

    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["reference_iostat"]
    assert spec.optional_approvals == ("KERNEL_MODULE_LOAD",), spec.optional_approvals
    store = storage_probe()
    settings = make_settings(spec)
    approvals = ("KERNEL_MODULE_LOAD",) if approved else ()
    settings = type(settings)(**{**settings.__dict__, "approvals": approvals})
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings,
        probes=probes(node_probe(survey(dm_delay=loaded)), store, window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)  # no scripted oracle samples
    arguments = _dm_create_arguments(store)
    assert arguments[arguments.index("--delay-ms") + 1] == delay, arguments
    assert ("--load-module" in arguments) is load, arguments


def test_reference_iostat_is_inconclusive_under_the_tolerance_floor(
    tmp_path: Path,
) -> None:
    """No dm_delay and no approval to load it: a linear device on NVMe reads
    sub-millisecond awaits that no formula difference can lift past the 5 ms
    floor -- 2026-10-25 on ip-10-0-43-5 the product's 0.30-0.44 ms against the
    oracle's 0.33-0.76 ms read PASS on an expected-FAIL variant."""

    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["reference_iostat"]
    store = storage_probe()
    node = node_probe(survey(dm_delay=False))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": 0.45}}} for _ in range(30)
        ],
        "next_offset": 30,
        "total": 30,
    }
    doc = {
        "incidents": [incident("inc-io", "disk_io_await_ms", device="dm-7")],
        "workflows": [],
        "batches": [
            host_batch(
                f"2026-10-24T00:0{i}:00+00:00", "disk_io_await_ms", 0.3, device="dm-7"
            )
            for i in range(4)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    settings = make_settings(spec)
    settings = type(settings)(**{**settings.__dict__, "approvals": ()})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": doc})),
        settings=settings,
        probes=probes(node, store, window_probe()),
    )
    with pytest.raises(VariantInconclusive) as caught:
        spec.execute(ctx)
    assert "tolerance floor" in str(caught.value), str(caught.value)
    arguments = _dm_create_arguments(store)
    assert arguments[arguments.index("--delay-ms") + 1] == "0", arguments


def _latency_run(tmp_path: Path, store: FakeProbe) -> dict[str, Any]:
    spec = SPECS["sustained_latency"]
    node = node_probe(survey(dm_delay=True))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": 210.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    inc = incident(
        "inc-io",
        "disk_io_await_ms",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io",
        "inc-io",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="disk_io_await_ms",
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    doc = {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                "2026-10-24T00:02:00+00:00", "disk_io_await_ms", 210.0, device="dm-7"
            )
        ]
        + [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00", "disk_io_await_ms", 0.4, device="dm-7"
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node, store, window_probe()),
    )
    return spec.execute(ctx)


def test_sustained_latency_removes_the_delay_and_proves_recovery_on_paced_io(
    tmp_path: Path,
) -> None:
    """Live 2026-10-25 14:47Z (ip-10-0-43-5, R13e): after ``io-stop`` the
    dm-delay device saw no completed I/O for the whole 690 s recovery wait,
    so the product -- by its zero-completed-I/O rule -- emitted no await
    sample and the signal could never clear; with the 200 ms table still
    loaded any I/O would have read above the 100 ms threshold anyway. The
    recovery proof now takes the delay out of the data table and keeps a
    paced single writer on the same dm-N while the product samples it."""

    store = storage_probe()
    result = _latency_run(tmp_path, store)
    assert result["verdict"] == "PASS", result["errors"]
    names = [call[0] for call in store.calls]
    first_stop = names.index("io-stop")
    assert names[first_stop:] == ["io-stop", "dm-retarget", "io-start", "io-stop"], (
        names
    )
    retarget = store.calls[first_stop + 1]
    assert retarget[retarget.index("--target") + 1] == "data", retarget
    assert retarget[retarget.index("--delay-ms") + 1] == "0", retarget
    writer = store.calls[first_stop + 2]
    assert writer[writer.index("--threads") + 1] == "1", writer
    assert writer[writer.index("--pace-ms") + 1] == "100", writer
    assert "--direct" in writer, writer
    assert int(writer[writer.index("--seconds") + 1]) >= 690 + 60, writer
    assert result["recovery_load"]["delay_removed"]["target"] == "data", result
    assert result["recovery"]["fresh_samples"], result["recovery"]


def test_sustained_util_recovery_needs_no_verification_load(tmp_path: Path) -> None:
    """Util falls to 0 % without I/O (field 10 over elapsed time), so the util
    variant keeps the idle recovery wait: no table swap, no second writer."""

    spec = SPECS["sustained_util"]
    store = storage_probe()
    node = node_probe(survey(dm_delay=False))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"util_percent": 100.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    doc = {
        "incidents": [incident("inc-io", "disk_io_util_percent", device="dm-7")],
        "workflows": [],
        "batches": [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "disk_io_util_percent",
                2.0,
                device="dm-7",
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": doc})),
        probes=probes(node, store, window_probe()),
    )
    spec.execute(ctx)
    names = [call[0] for call in store.calls]
    assert "dm-retarget" not in names and names.count("io-start") == 1, names


# --------------------------------------------------------------------------- #
# 006 detection budget on the product's cadence (live 2026-10-25 17:04Z, eksb)
# --------------------------------------------------------------------------- #
def _poll_clock(ctx: Any) -> Any:
    """One store poll per clock reading, as live (a read every 20 s)."""

    import itertools
    from datetime import datetime, timedelta, timezone

    ticks = itertools.count()
    start = datetime(2026, 10, 25, 17, 3, 12, tzinfo=timezone.utc)
    ctx.clock = lambda: start + timedelta(seconds=STORE_POLL_SECONDS * next(ticks))
    return ctx


def _await_oracle_reads(*phases: list[float | None]) -> list[dict[str, Any]]:
    """Every ``oracle-read`` serves the dm-7 readings since the previous read."""

    entries: list[dict[str, Any]] = []
    total = 0
    for values in phases:
        total += len(values)
        entries.append(
            {
                "records": [
                    {"diskstats": {"dm-7": {"standard_await_ms": value}}}
                    for value in values
                ],
                "next_offset": total,
                "total": total,
            }
        )
    return entries


#: The live shape: the writer's first interval has no completion yet, the
#: crossing arrives on the second oracle read (210 ms, the dm-delay table) and
#: holds; the product's finding needs an active sample >= 120 s after the
#: first and comes with the first health summary after that -- up to 435 s
#: after the edge -- which the old 510-s budget from the load start only just
#: covered. Here the records arrive 30 store polls (600 s) after the crossing.
AWAIT_READS = _await_oracle_reads([None] * 2, [210.0] * 80, [210.0] * 10)
AWAIT_MINTED_AT_POLL = 30


def _latency_doc() -> dict[str, Any]:
    inc = incident(
        "inc-io",
        "disk_io_await_ms",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io",
        "inc-io",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="disk_io_await_ms",
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                "2026-10-25T17:08:30+00:00", "disk_io_await_ms", 210.5, device="dm-7"
            )
        ]
        + [
            host_batch(
                f"2026-10-25T17:2{i}:00+00:00", "disk_io_await_ms", 0.2, device="dm-7"
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def _empty_doc() -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [
            host_batch(
                f"2026-10-25T17:2{i}:00+00:00", "disk_io_await_ms", 0.2, device="dm-7"
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def _latency_cadence_run(
    tmp_path: Path, reads: list[dict[str, Any]], minted_at_poll: int | None
) -> tuple[dict[str, Any], Any, FakeProbe, FakeProbe]:
    import itertools

    spec = SPECS["sustained_latency"]
    polls = itertools.count(1)

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        poll = next(polls)
        if minted_at_poll is not None and poll >= minted_at_poll:
            return _latency_doc()
        return _empty_doc()

    node = node_probe(survey(dm_delay=True))
    node.script["oracle-read"] = reads
    store = storage_probe()
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=product),
        probes=probes(node, store, window_probe()),
    )
    result = spec.execute(_poll_clock(ctx))
    return result, ctx, node, store


def test_io_detection_hold_covers_the_products_two_summaries(tmp_path: Path) -> None:
    """The budget is the product's cadence, read from the node's collector env:
    the rule's sustain + two health summaries + one collector interval + the
    ingest/dispatch/jitter margin, as 003's ``page_cache_hold``."""

    from gpu_fault.host_health import NodeHealthPolicy

    spec = SPECS["sustained_latency"]
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": _empty_doc()})),
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    ctx.survey = survey()
    assert storage_io.IO_SUSTAIN == NodeHealthPolicy.DISK_IO_SUSTAIN_SECONDS
    assert storage_io.io_detection_hold(ctx) == 120 + 2 * 300 + 15 + 180
    assert storage_io.io_detection_hold(ctx) > storage_io.IO_BUDGET, (
        "the budget counted from the load start that the cadence can outrun"
    )
    for variant in ("sustained_util", "sustained_latency", "recovery"):
        minutes = SPECS[variant].max_minutes
        assert minutes * 60 >= (
            storage_io.IO_RAMP_BUDGET
            + storage_io.io_detection_hold(ctx)
            + 2 * 300
            + 6 * 15
            + storage_io.IO_BUDGET
            + 240
        ), (variant, minutes)


def test_sustained_latency_times_detection_from_the_oracles_crossing(
    tmp_path: Path,
) -> None:
    result, ctx, node, store = _latency_cadence_run(
        tmp_path, AWAIT_READS, AWAIT_MINTED_AT_POLL
    )
    assert result["verdict"] == "PASS", result["errors"]
    import json

    ramp = json.loads(
        (ctx.case_dir / "disk_io_await_ms-ramp-timeline.json").read_text()
    )
    assert [entry["accepted"] for entry in ramp["entries"]] == [False, True]
    load = json.loads((ctx.case_dir / "io-load.json").read_text())
    assert load["detection_budget_seconds"] == storage_io.io_detection_hold(ctx)
    assert load["ramp_budget_seconds"] == storage_io.IO_RAMP_BUDGET
    assert result["crossing"]["value"] == 210.0, result["crossing"]
    # The writer and the oracle outlive the ramp and the detection budget.
    writer = next(call for call in store.calls if call[0] == "io-start")
    assert int(writer[writer.index("--seconds") + 1]) >= (
        storage_io.IO_RAMP_BUDGET + storage_io.io_detection_hold(ctx)
    ), writer
    oracle = next(call for call in node.calls if call[0] == "oracle-start")
    assert int(oracle[oracle.index("--seconds") + 1]) >= (
        storage_io.IO_RAMP_BUDGET + storage_io.io_detection_hold(ctx)
    ), oracle


def test_sustained_latency_fails_when_the_product_misses_two_summaries(
    tmp_path: Path,
) -> None:
    budget = storage_io.IO_SUSTAIN + 2 * 300 + 15 + 180
    late = budget // STORE_POLL_SECONDS + 4
    result, ctx, _node, store = _latency_cadence_run(tmp_path, AWAIT_READS, late)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["chain"] == [
        "no disk_io_await_ms on dm-7 finding reached an incident"
    ], result["stages"]
    # Nothing to settle must not poll the store for another budget (live
    # 2026-10-25: 8.5 minutes of settle polls after the chain FAIL), and the
    # device's await signal is still given its recovery proof (hygiene: a
    # signal left active and notified masks the next run's finding).
    assert not (ctx.case_dir / "disk_io_await_ms-settled-timeline.json").exists(), (
        "nothing to settle must not poll the store"
    )
    names = [call[0] for call in store.calls]
    first_stop = names.index("io-stop")
    assert names[first_stop:] == ["io-stop", "dm-retarget", "io-start", "io-stop"], (
        names
    )


def test_sustained_latency_is_inconclusive_when_the_ramp_runs_out(
    tmp_path: Path,
) -> None:
    with pytest.raises(VariantInconclusive, match="condition not established"):
        _latency_cadence_run(tmp_path, _await_oracle_reads([None] * 3), None)


def test_an_unsustained_oracle_is_inconclusive_only_after_the_signal_recovery(
    tmp_path: Path,
) -> None:
    """The oracle crossed but did not hold: INCONCLUSIVE, raised after the
    delay is out and the paced writer gave the product its recovery samples --
    a device destroyed with an active, notified await signal poisons the next
    run on the node (live 2026-10-25 17:04Z, eksb)."""

    reads = _await_oracle_reads([None], [210.0] * 3 + [50.0] * 80, [50.0] * 10)
    with pytest.raises(VariantInconclusive, match="never sustained"):
        _latency_cadence_run(tmp_path, reads, AWAIT_MINTED_AT_POLL)


def test_reference_iostat_clears_the_await_signal_before_teardown(
    tmp_path: Path,
) -> None:
    """Live 2026-10-25 eksb: ``reference_iostat`` (14:56-15:05Z, dm-delay 200 ms)
    minted an await incident at 15:03:10Z, stopped its writer and destroyed the
    device with no below-threshold await sample ever delivered, so the
    device-scoped signal stayed active and notified until 17:14:22Z; the next
    ``sustained_latency`` on dm-0 (17:04Z) had its finding withheld by that
    latch (store/shared/xid.py _next_health_signal_state: previously_notified)
    -- "no disk_io_await_ms on dm-0 finding reached an incident". Every
    await-breaching variant now leaves the signal cleared."""

    spec = SPECS["reference_iostat"]
    node = node_probe(survey())
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": 212.0}}} for _ in range(30)
        ],
        "next_offset": 30,
        "total": 30,
    }
    doc = {
        "incidents": [incident("inc-io", "disk_io_await_ms", device="dm-7")],
        "workflows": [],
        "batches": [
            host_batch(
                "2026-10-25T15:03:09+00:00", "disk_io_await_ms", 212.6, device="dm-7"
            )
        ]
        + [
            host_batch(
                f"2026-10-25T15:1{i}:00+00:00", "disk_io_await_ms", 0.3, device="dm-7"
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    import itertools

    loaded = {**doc, "batches": doc["batches"][:1]}
    recovered = {**doc, "batches": doc["batches"][1:]}
    polls = itertools.count(1)

    def product(_script: str, *_arguments: str, **_: Any) -> dict[str, Any]:
        return loaded if next(polls) == 1 else recovered

    store = storage_probe()
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=product),
        probes=probes(node, store, window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["signal_cleared"] == []
    assert result["recovery"]["fresh_samples"], result["recovery"]
    names = [call[0] for call in store.calls]
    first_stop = names.index("io-stop")
    assert names[first_stop:] == ["io-stop", "dm-retarget", "io-start", "io-stop"], (
        names
    )
    retarget = store.calls[first_stop + 1]
    assert retarget[retarget.index("--delay-ms") + 1] == "0", retarget


def test_reference_iostat_clears_the_signal_before_an_inconclusive_reading(
    tmp_path: Path,
) -> None:
    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["reference_iostat"]
    store = storage_probe()
    settings = make_settings(spec)
    settings = type(settings)(**{**settings.__dict__, "approvals": ()})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": _empty_doc()})),
        settings=settings,
        probes=probes(node_probe(survey(dm_delay=False)), store, window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)  # no await samples to compare
    names = [call[0] for call in store.calls]
    first_stop = names.index("io-stop")
    # A linear device (delay 0) needs no table swap, only the paced writer.
    assert names[first_stop:] == ["io-stop", "io-start", "io-stop"], names
