"""GF-GENERAL-LIVE-019 ``pending_supported`` / ``post_reset``: the flag has to
be read on the node's confirmation count of consecutive scrapes before the
collector delivers it at all.

Live 2026-10-27 04:05-04:18Z (HyperPod, the hp lane's fault node, R13g,
harness int/r13c-live23) ``pending_supported`` FAILed "no row_remap_pending ...
finding reached an incident" with the node untouched. The plan (S36) served
``row_remap_pending=1`` for ONE scrape after three pass-through reads -- bound
drawn from the control plane alone (``gpu_memory_policy.memory_decision`` mints
the flag on one reading > 0, no delta, confirmation or sustain) -- and the
replay journal shows exactly that: read 5 served 1.0, reads 6-43 were the
device's own, no incident, no error record. The batch never left the node: the
collector's edge filter delivers a changed ``row_remap_pending`` only once the
flag has been a "breached" candidate on ``GPU_FAULT_DCGM_EDGE_CONFIRMATION_
SAMPLES`` consecutive scrapes (``collectors/gpu/dcgm.py`` ``_candidate_keys``
/ ``_should_deliver`` ``candidate-confirmed``; 3 on the live nodes, per the
case's ``gpu-survey.json`` ``collector_env``). The flag is a gauge, not a
counter (``_is_counter``: ``*_total``) and not a state gauge
(``STATE_CHANGE_REASONS``: XID, fabric state), so one read of it is no
delivery edge and the batch is suppressed. 024 ``wrong_scope_source`` mints on
one read only because the DBE counter in the same batch is a
``counter-increased`` edge (eksb 2026-10-26 07:48Z, read 4).

These tests drive the harness's plan through the real field tool, the real
collector delivery filter and the real ingest: the shipped shape (one flag
read per confirmation sample, nothing after) delivers the confirmation read,
mints the WARNING RESET_GPU finding on it and clears it on the first real
reading; the one-read shape delivers nothing (the live failure); a spare read
rides the restarted collector's initial baseline to the post-reset
VALIDATE_GPU; and a health summary on a pre-confirmation read mints early,
which is the delivery the arm's summary-phase clearance keeps those reads away
from. No product code changes with this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService
from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.general_live_gpu_replay import baseline, plan, put, step
from tests.collectors._support import DcgmMetricsCollector, RecordingSink, context
from tools import gpu_dcgm_field_replay as tool

#: The cluster of ``tests.collectors._support.context()`` and the collector's
#: node id: what ``next_stable_phase`` hashes for the health-summary phase.
CLUSTER = "hp-cluster"
NODE = "worker-1"
GPU = "GPU-80f702c5-74ea-2200-2582-f0c795d4f919"
BOOT = "aa20cf7c-1b08-4929-8c02-0fb234dd9f7c"
PENDING = "row_remap_pending"
#: The real exporter's GPU_IDLE bit: not a candidate.
IDLE = 1
INTERVAL = 15
SUMMARY = 300
CONFIRMATION = 3
#: Scrapes before the arm: the restarted collector's first scrape is its
#: initial baseline, delivered whatever it carries.
PRE_ARM = 1
POST_PLAN = 3


def exporter_text(pending: float) -> str:
    """One scrape of an idle H200 whose row-remap-pending flag is ``pending``."""

    labels = f'gpu="0",UUID="{GPU}",modelName="NVIDIA H200"'
    return "\n".join(
        [
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 30",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_FAILURE{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} {pending:g}",
            f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY{{{labels}}} 0",
            f"DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{{labels}}} {IDLE}",
            f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
            "# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter",
            f"DCGM_FI_DEV_THERMAL_VIOLATION{{{labels}}} 0",
            "",
        ]
    )


def pending_plan(reads: int) -> dict[str, Any]:
    """The harness's plan for a node confirming on ``reads`` samples."""

    return plan("memory", baseline(), step(reads, put(PENDING, [GPU], 1)))


def collector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    times: list[datetime],
    *,
    jitter_ratio: str = "0",
) -> tuple[DcgmMetricsCollector, RecordingSink]:
    """The real collector of the live nodes' settings (3 confirmation
    samples, 300 s summary) scraping at ``times``, its summary phase
    unjittered so the test's reads land where they are placed."""

    path = tmp_path / "boot"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BOOT + "\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", jitter_ratio)
    sink = RecordingSink()
    clock = iter(times)
    dcgm = DcgmMetricsCollector(
        sink,
        context(),
        node_id=NODE,
        now=lambda: next(clock),
        health_summary_seconds=SUMMARY,
        edge_confirmation_samples=CONFIRMATION,
    )
    return dcgm, sink


def scrape(replay: tool.Replay, dcgm: DcgmMetricsCollector) -> float:
    """One collector scrape through the armed field tool over the real
    (healthy) exporter text; returns the flag the tool served."""

    output, receipt = replay.render(exporter_text(0).encode())
    replay.reads += 1
    dcgm.collect_text(output.decode())
    served = (receipt.get("served") or {}).get(f"{GPU}/{PENDING}")
    return 0.0 if served is None else float(served)


def shipped(sink: RecordingSink) -> list[tuple[list[str], float]]:
    batches = [
        GpuMetricBatch.model_validate(payload) for _path, payload in sink.requests
    ]
    return [
        (
            batch.edge_filter_reasons,
            next(
                sample.value
                for sample in batch.samples
                if sample.canonical_name == PENDING and sample.gpu_uuid == GPU
            ),
        )
        for batch in batches
    ]


def batches(sink: RecordingSink) -> list[GpuMetricBatch]:
    return [GpuMetricBatch.model_validate(payload) for _path, payload in sink.requests]


def active(service: GpuMetricsService) -> list[str]:
    """What ``_validate_active_findings`` reads for the GPU on its next attempt."""

    return sorted(
        finding.canonical_name
        for finding in service.findings(CLUSTER, NODE)
        if finding.gpu_uuid == GPU
    )


def quiet_start() -> datetime:
    """A first scrape 20 s past the node's summary phase: the next summary is
    280 s away, so a plan of a dozen scrapes meets none (ratio 0)."""

    phase = next_stable_phase(
        datetime(2026, 10, 27, 4, 7, tzinfo=timezone.utc),
        cluster_id=CLUSTER,
        node_id=NODE,
        channel="gpu-metrics",
        interval_seconds=SUMMARY,
    )
    return phase + timedelta(seconds=20)


def run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    document: dict[str, Any],
    *,
    start: datetime,
    scrapes: int,
) -> tuple[list[float], RecordingSink, tool.Replay]:
    """``scrapes`` collector scrapes at ``INTERVAL`` from ``start``: one before
    the arm, then the armed plan, then the device's own readings."""

    times = [start + timedelta(seconds=INTERVAL * index) for index in range(scrapes)]
    dcgm, sink = collector(monkeypatch, tmp_path, times)
    replay = tool.Replay(tool.validate_plan(document), 600)
    served = []
    for index in range(scrapes):
        if index == PRE_ARM:
            replay.arm()
        served.append(scrape(replay, dcgm))
    return served, sink, replay


def test_the_confirmation_read_is_the_delivery_and_the_real_reading_clears_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = mem.pending_fault_reads(CONFIRMATION)
    document = pending_plan(reads)
    mem.require_pending_plan(document, gpu=GPU, reads=reads, variant="post_reset")
    served, sink, replay = run(
        monkeypatch,
        tmp_path,
        document,
        start=quiet_start(),
        scrapes=PRE_ARM + 3 + reads + POST_PLAN,
    )
    assert served == [0.0] * 4 + [1.0] * 3 + [0.0] * 3, served
    assert replay.active() is None, "the plan ran out by itself"
    assert not any(batch.collection_errors for batch in batches(sink)), (
        "every scrape carries the core fields"
    )
    assert shipped(sink) == [
        (["initial-baseline"], 0.0),
        (["candidate-confirmed"], 1.0),
        (["candidate-recovered"], 0.0),
    ], "the filter ships the third flag read and the first real one, nothing else"

    service = GpuMetricsService()
    first_batch, confirmed_batch, cleared_batch = batches(sink)
    first = service.ingest(first_batch)
    assert not first.findings, "the pass-through baseline breaches nothing"
    assert active(service) == []
    confirmed = service.ingest(confirmed_batch)
    (finding,) = confirmed.new_findings
    assert finding.canonical_name == PENDING and finding.gpu_uuid == GPU
    assert finding.automatic_action == "RESET_GPU", finding
    assert finding.official_action == "RESET_GPU_IN_SERVICE_WINDOW", finding
    assert active(service) == [PENDING], "the incident's finding, minted"
    cleared = service.ingest(cleared_batch)
    assert not cleared.findings and not cleared.new_findings
    assert active(service) == [], (
        "the device's real reading clears it: the batch VALIDATE_GPU re-checks on"
    )


def test_one_read_of_the_flag_is_no_delivery_edge_the_live_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S36's bound, live 2026-10-27 04:08Z: the fifth read served the flag and
    the collector shipped nothing but its baseline; the control plane never saw
    a reading > 0 and no finding, incident or chain followed."""

    served, sink, _replay = run(
        monkeypatch,
        tmp_path,
        pending_plan(1),
        start=quiet_start(),
        scrapes=PRE_ARM + 3 + 1 + 6,
    )
    assert served == [0.0] * 4 + [1.0] + [0.0] * 6, served
    assert shipped(sink) == [(["initial-baseline"], 0.0)], (
        "a gauge flip on one scrape is no edge: suppressed"
    )
    service = GpuMetricsService()
    for batch in batches(sink):
        assert not service.ingest(batch).findings, "no reading > 0 ever arrived"
    assert active(service) == [], "no finding, no incident, no chain"
    with pytest.raises(ValueError, match="serves 1 read"):
        mem.require_pending_plan(
            pending_plan(1), gpu=GPU, reads=CONFIRMATION, variant="pending_supported"
        )


def test_a_spare_flag_read_rides_the_restarted_collectors_baseline_to_the_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Why exactly the count: a fourth flag read on a three-sample node is no
    edge of its own (a confirmed candidate is no new edge), so if the quiesce
    stops the collector right after the mint it stays in the tool, and the
    collector RESTORE_GPU_SERVICES restarts ships it as its initial baseline --
    the batch the post-reset VALIDATE_GPU grades, with the finding still
    active. With the exact count the restarted collector's first scrape is
    the device's own reading and the finding clears."""

    start = quiet_start()
    for reads, expected in ((CONFIRMATION + 1, [PENDING]), (CONFIRMATION, [])):
        times = [
            start + timedelta(seconds=INTERVAL * index) for index in range(PRE_ARM + 6)
        ]
        dcgm, sink = collector(monkeypatch, tmp_path / f"r{reads}", times)
        replay = tool.Replay(tool.validate_plan(pending_plan(reads)), 600)
        for index in range(PRE_ARM + 3 + CONFIRMATION):
            if index == PRE_ARM:
                replay.arm()
            scrape(replay, dcgm)
        assert [reasons for reasons, _ in shipped(sink)] == [
            ["initial-baseline"],
            ["candidate-confirmed"],
        ], reads
        service = GpuMetricsService()
        for batch in batches(sink):
            service.ingest(batch)
        assert active(service) == [PENDING], "minted on the confirmation"
        # The quiesce: the collector and exporter stop ~20 s after the mint
        # (nothing scrapes), RESET_GPU, then RESTORE_GPU_SERVICES restarts
        # the collector with no memory of the plan's reads.
        restarted, fresh = collector(
            monkeypatch,
            tmp_path / f"r{reads}-restart",
            [times[-1] + timedelta(seconds=120 + INTERVAL * k) for k in range(2)],
        )
        value = scrape(replay, restarted)
        assert value == (1.0 if reads > CONFIRMATION else 0.0), reads
        ((reasons, delivered),) = shipped(fresh)
        assert reasons == ["initial-baseline"] and delivered == value
        service.ingest(batches(fresh)[0])
        assert active(service) == expected, (
            f"{reads} flag reads on a {CONFIRMATION}-sample node: the post-reset "
            f"VALIDATE_GPU reads {active(service)}"
        )


def test_a_health_summary_on_a_pre_confirmation_read_mints_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one delivery an idle node makes by itself: a summary that lands on
    the first flag read ships it, the control plane mints on it, and the two
    flag reads behind it are the spare of the previous test once the quiesce
    stops the collector. The arm's clearance moves such a start past the
    phase's window, so the first flag read of the moved plan is 40 s or more
    past the phase and no summary can reach a read before the confirmation."""

    phase = quiet_start() - timedelta(seconds=20)
    # Read 5 -- the first flag read -- scraped on the phase itself.
    start = phase - timedelta(seconds=INTERVAL * 4)
    served, sink, _replay = run(
        monkeypatch,
        tmp_path,
        pending_plan(CONFIRMATION),
        start=start,
        scrapes=PRE_ARM + 3 + CONFIRMATION + 1,
    )
    assert served == [0.0] * 4 + [1.0] * 3 + [0.0], served
    assert shipped(sink) == [
        (["initial-baseline"], 0.0),
        (["health-summary"], 1.0),
        (["candidate-confirmed"], 1.0),
        (["candidate-recovered"], 0.0),
    ], "the summary ships the first flag read by itself"
    service = GpuMetricsService()
    service.ingest(batches(sink)[0])
    early = service.ingest(batches(sink)[1])
    assert [item.canonical_name for item in early.new_findings] == [PENDING], (
        "minted on the first flag read, two flag reads still ahead"
    )

    armed_at = start + timedelta(seconds=1)
    clearance = mem.summary_clear_wait_seconds(
        armed_at,
        cluster_id=CLUSTER,
        node_id=NODE,
        interval=INTERVAL,
        health_summary=SUMMARY,
        passthrough_reads=3,
        fault_reads=CONFIRMATION,
    )
    assert clearance["wait_seconds"] > 0, clearance
    moved = datetime.fromisoformat(clearance["fault_window"][0])
    assert moved >= phase + timedelta(seconds=mem.ARM_SKEW_SECONDS + 2 * INTERVAL), (
        "the first flag read of the moved plan is past the phase's delivery window"
    )
