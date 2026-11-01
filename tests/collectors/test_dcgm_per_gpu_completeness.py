"""DCGM field completeness is judged GPU by GPU (GF-GENERAL-LIVE-025).

The collector used to compute the scrape's required-field check over the union
of every GPU's fields, so a required field one GPU stopped reporting was hidden
by the GPUs that still reported it (the spec's ``one_gpu_missing``: "其他GPU有
同名字段不能覆盖缺失GPU"). And an incomplete scrape was no delivery edge: it
reached the control plane's ``dcgm_field_completeness`` finding only on the next
health summary, so the finding opened late, escalated only when two summaries
both happened to be incomplete, and the first complete scrape afterwards (or
after a failed scrape) was suppressed as an unchanged healthy batch, leaving the
finding active.
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu import dcgm
from gpu_fault.collectors.gpu.completeness import completeness_errors
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.dcgm_fields import NVLINK5_LINK_RECOVERY_EXPORTER_NAMES
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample
from gpu_fault.gpu_temperature_policy import TLIMIT_READING
from gpu_fault.models import RecoveryAction, WorkflowOperation
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import EvidenceKind
from tests._builders import asgi_client

CLUSTER = "completeness-cluster"
NODE = "completeness-node"
SUMMARY_SECONDS = 86_400
#: Just past the node's stable summary phase, so no health summary falls inside
#: a test and every delivery is an edge the test names.
START = next_stable_phase(
    datetime(2026, 10, 24, tzinfo=timezone.utc),
    cluster_id=CLUSTER,
    node_id=NODE,
    channel="gpu-metrics",
    interval_seconds=SUMMARY_SECONDS,
) + timedelta(seconds=1)
GPUS = {"GPU-a": ("0", "00000000:97:00.0"), "GPU-b": ("1", "00000000:A8:00.0")}
VALUES = {
    "DCGM_FI_DEV_GPU_TEMP": 40,
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": 700,
    # ECC enabled: a healthy GPU (ECC disabled is graded since batch 233).
    "DCGM_FI_DEV_ECC_CURRENT": 1,
    "DCGM_FI_DEV_ECC_MODE": 1,
    "DCGM_FI_DEV_ECC_PENDING": 1,
}
PCIE = "DCGM_FI_DEV_PCIE_REPLAY_COUNTER"
ROW_REMAP_FAILURE = "DCGM_FI_DEV_ROW_REMAP_FAILURE"
INVALID = "invalid or ambiguous GPU observation"
#: Every exporter name that answers for the CRC group: the per-class counters,
#: the DL aggregate and (batch 235) the NVLink5 link-recovery counters.
NVLINK_CRC = (
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL",
    "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC",
    *NVLINK5_LINK_RECOVERY_EXPORTER_NAMES,
)
MISSING = "DCGM exporter is missing required fields"
COMPLETENESS_KEY = (CLUSTER, NODE, "node", "dcgm_field_completeness")
WINDOW_SECONDS = 180
DIAGNOSTIC_PLAN = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    WorkflowOperation.VALIDATE_HOST,
]


def scrape(
    *,
    omit: dict[str, tuple[str, ...]] | None = None,
    values: dict[str, dict[str, float]] | None = None,
) -> str:
    """Every mapped field for both GPUs, minus ``omit`` (by exporter name).

    The T.Limit margin is never required and is left out: a zero margin is a
    temperature candidate, which would add edges no test here is about.
    """

    lines = []
    for uuid, (index, bus) in GPUS.items():
        labels = (
            f'gpu="{index}",UUID="{uuid}",pci_bus_id="{bus}",modelName="NVIDIA H200"'
        )
        for name, (canonical, _unit) in DCGM_METRICS.items():
            if canonical == TLIMIT_READING or name in (omit or {}).get(uuid, ()):
                continue
            value = (values or {}).get(uuid, {}).get(name, VALUES.get(name, 0))
            lines.append(f"{name}{{{labels}}} {value}")
    return "\n".join(lines) + "\n"


class Sink:
    def __init__(self) -> None:
        self.batches: list[GpuMetricBatch] = []

    def post(self, path: str, payload: dict) -> dict:
        if path == GPU_METRICS_PATH:
            self.batches.append(GpuMetricBatch.model_validate(payload))
        return {"accepted": True}


def collector(sink: Sink | None = None, **options) -> DcgmMetricsCollector:
    return DcgmMetricsCollector(
        sink if sink is not None else Sink(),
        CollectorContext(cluster_id=CLUSTER, runtime_profile_version="simulated-v1"),
        node_id=NODE,
        health_summary_seconds=SUMMARY_SECONDS,
        **options,
    )


def test_a_field_one_gpu_stopped_reporting_names_that_gpu() -> None:
    """The live ``one_gpu_missing`` shape: GPU-a still reports the PCIe replay
    counter, so the union-based check used to report nothing at all."""

    batch = collector().collect_text(scrape(omit={"GPU-b": (PCIE,)}), observed_at=START)

    assert batch.collection_errors == [f"{MISSING} on GPU-b: pcie_replay_total"], (
        "a sibling GPU's field must not stand in for the GPU that lacks it: "
        f"{batch.collection_errors}"
    )


def test_a_field_no_gpu_reports_keeps_the_batch_wide_message() -> None:
    batch = collector().collect_text(
        scrape(omit={"GPU-a": (PCIE,), "GPU-b": (PCIE,)}), observed_at=START
    )

    assert batch.collection_errors == [f"{MISSING}: pcie_replay_total"], (
        "a scrape-wide gap keeps the message operators and runners already read: "
        f"{batch.collection_errors}"
    )


def test_a_scrape_wide_gap_and_a_one_gpu_gap_are_both_reported() -> None:
    batch = collector().collect_text(
        scrape(omit={"GPU-a": (PCIE,), "GPU-b": (PCIE, *NVLINK_CRC)}), observed_at=START
    )

    assert batch.collection_errors == [
        f"{MISSING}: pcie_replay_total",
        f"{MISSING} on GPU-b: nvlink_crc",
    ], f"each gap is reported once, at its own scope: {batch.collection_errors}"


def test_each_gpu_may_answer_a_field_group_with_its_own_exporter_alias() -> None:
    batch = collector().collect_text(
        scrape(omit={"GPU-a": NVLINK_CRC[:2], "GPU-b": NVLINK_CRC[2:]}),
        observed_at=START,
    )

    assert batch.collection_errors == [], (
        "a GPU that reports one of a group's exporter names has the group: "
        f"{batch.collection_errors}"
    )


@pytest.mark.parametrize("unreadable", ["9223372036854775794", "NaN"])
def test_an_unreadable_reading_on_one_gpu_is_missing_never_healthy(
    unreadable: str,
) -> None:
    batch = collector().collect_text(
        scrape(values={"GPU-b": {PCIE: unreadable}}), observed_at=START
    )

    assert batch.collection_errors == [f"{MISSING} on GPU-b: pcie_replay_total"], (
        "a blank or NaN slot is missing evidence, not a reading: "
        f"{batch.collection_errors}"
    )
    assert not any(
        sample.gpu_uuid == "GPU-b" and sample.canonical_name == "pcie_replay_total"
        for sample in batch.samples
    ), "the unreadable slot must not become a value"


def test_cached_nvidia_smi_limits_do_not_invent_an_incomplete_gpu() -> None:
    """The nvidia-smi temperature limits are cached for the process's life: a
    replaced GPU's old UUID is not a GPU that stopped reporting DCGM fields."""

    stale_limit = GpuMetricSample(
        metric_name="gpu_slowdown_temperature_c",
        canonical_name="gpu_slowdown_temperature_c",
        value=87,
        gpu_index="0",
        gpu_uuid="GPU-replaced",
    )

    batch = collector().collect_text(
        scrape(), observed_at=START, extra_samples=[stale_limit]
    )

    assert batch.collection_errors == [], (
        f"only what the exporter served is judged: {batch.collection_errors}"
    )
    assert stale_limit.canonical_name in {
        sample.canonical_name for sample in batch.samples
    }, "the limits are still delivered"


def test_samples_that_name_no_gpu_are_judged_as_one_union() -> None:
    """An exporter that labels no GPU leaves nothing to judge per GPU; a sample
    without identity never stands in for a GPU that has one."""

    def sample(name: str, gpu_uuid: str | None = None) -> GpuMetricSample:
        return GpuMetricSample(
            metric_name=name, canonical_name=name, value=0, gpu_uuid=gpu_uuid
        )

    everything_else = (
        "clock_throttle_reasons,ecc_dbe_volatile_total,pcie_replay_total,"
        "power_limit_w,row_remap_failure,row_remap_pending,"
        "nvlink_crc,nvlink_recovery,nvlink_replay"
    )
    assert completeness_errors([sample("gpu_temperature_c")]) == [
        f"{MISSING}: {everything_else}"
    ], "with no GPU identity at all, the union is judged"
    assert completeness_errors(
        [sample("gpu_temperature_c", "GPU-a"), sample("pcie_replay_total")]
    ) == [f"{MISSING}: {everything_else}"], (
        "a node-level sample cannot supply GPU-a's PCIe replay counter"
    )


def test_completeness_changes_are_delivery_edges() -> None:
    """Delivered at once, unchanged repeats filtered, re-delivered once when
    it has lasted the control plane's sustain window, and the recovery at once."""

    sink = Sink()
    reader = collector(sink)
    incomplete = scrape(omit={"GPU-b": (PCIE,)})
    plan = [(0, scrape())]
    plan += [(offset, incomplete) for offset in range(15, 256, 15)]
    plan += [(270, scrape()), (285, scrape())]

    delivered = []
    for offset, text in plan:
        before = len(sink.batches)
        reader.collect_text(text, observed_at=START + timedelta(seconds=offset))
        if len(sink.batches) > before:
            delivered.append((offset, sink.batches[-1].edge_filter_reasons))

    assert delivered == [
        (0, ["initial-baseline"]),
        (15, ["fields-incomplete"]),
        (15 + WINDOW_SECONDS, ["fields-incomplete-sustained"]),
        (270, ["collection-recovered"]),
    ], f"completeness deliveries: {delivered}"


def test_a_different_incompleteness_is_a_new_edge() -> None:
    sink = Sink()
    reader = collector(sink)
    reader.collect_text(scrape(), observed_at=START)
    reader.collect_text(
        scrape(omit={"GPU-b": (PCIE,)}), observed_at=START + timedelta(seconds=15)
    )
    reader.collect_text(
        scrape(omit={"GPU-a": (PCIE,)}), observed_at=START + timedelta(seconds=30)
    )

    assert [batch.collection_errors for batch in sink.batches] == [
        [],
        [f"{MISSING} on GPU-b: pcie_replay_total"],
        [f"{MISSING} on GPU-a: pcie_replay_total"],
    ], "the GPU that lacks the field changed, so the evidence did"
    assert sink.batches[-1].edge_filter_reasons == ["fields-incomplete"], sink.batches[
        -1
    ].edge_filter_reasons


class _Response:
    def __init__(self, body: str) -> None:
        self.body = body.encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args) -> None:
        return None


class _StopLoop(BaseException):
    pass


def test_the_first_clean_scrape_after_a_failed_one_is_delivered(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Error-only batches are sent every failing tick; the scrape that ends
    them used to be filtered as unchanged, so the node's NO_ACTION finding
    stayed active until the next health summary."""

    request = tmp_path / "gpu.request"
    request.write_text("trigger\n", encoding="ascii")
    clock = [START]
    scrapes = iter([scrape(), None, None, scrape(), scrape()])
    ticks: list[float] = []

    def exporter(*_args, **_kwargs) -> _Response:
        text = next(scrapes)
        if text is None:
            raise OSError("connection refused")
        return _Response(text)

    def sleep(seconds: float) -> None:
        ticks.append(seconds)
        clock[0] += timedelta(seconds=seconds)
        if len(ticks) == 5:
            raise _StopLoop

    def no_nvidia_smi(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="absent")

    monkeypatch.setattr(dcgm, "urlopen", exporter)
    monkeypatch.setattr(dcgm, "time", SimpleNamespace(sleep=sleep))
    sink = Sink()
    reader = collector(
        sink,
        now=lambda: clock[0],
        runner=no_nvidia_smi,
        inventory_interval_seconds=3600,
        startup_grace_seconds=0,
        force_snapshot_path=str(request),
    )

    with pytest.raises(_StopLoop):
        reader.run()

    assert [batch.edge_filter_reasons for batch in sink.batches] == [
        ["initial-baseline"],
        ["collection-error"],
        ["collection-error"],
        ["collection-recovered"],
    ], [batch.edge_filter_reasons for batch in sink.batches]
    assert sink.batches[-1].collection_errors == [], "the recovery is a clean batch"


def _post(context: ApplicationContext, batch: GpuMetricBatch) -> None:
    async def run():
        async with asgi_client(context) as client:
            return await client.post(
                GPU_METRICS_PATH, json=batch.model_dump(mode="json")
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text


def _incidents(context: ApplicationContext) -> int:
    return sum(context.store.incident_state_counts().values())


def test_one_gpu_missing_reaches_the_bounded_diagnostic_and_clears() -> None:
    """Collector to control plane: the finding names the GPU and the group at
    once, asks for the bounded snapshot only once the window has passed --
    one incident, no reset or reboot -- and the recovery clears it."""

    sink = Sink()
    reader = collector(sink)
    incomplete = scrape(omit={"GPU-b": (PCIE,)})
    for offset in range(0, 286, 15):
        text = incomplete if 15 <= offset <= 255 else scrape()
        reader.collect_text(text, observed_at=START + timedelta(seconds=offset))
    baseline, first, sustained, recovered = sink.batches
    context = ApplicationContext(store=InMemoryStore())

    _post(context, baseline)
    _post(context, first)

    state = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is not None, "the gap is booked at once"
    finding = state.finding
    assert finding.policy_source == "SITE_DCGM_FIELD_COMPLETENESS", finding
    assert finding.automatic_action == RecoveryAction.NO_ACTION.value, finding
    assert "GPU-b" in finding.reason and "pcie_replay_total" in finding.reason, (
        f"the evidence names the GPU and the field group: {finding.reason}"
    )
    assert "GPU-a" not in finding.reason, finding.reason
    assert _incidents(context) == 0, "inside the window nothing opens"

    _post(context, sustained)

    state = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is not None, "the gap persists"
    assert state.finding.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value, (
        state.finding
    )
    assert _incidents(context) == 1, "the window opens exactly one incident"
    (workflow,) = context.store.list_workflows(limit=10)
    assert [step.operation for step in workflow.official_steps] == DIAGNOSTIC_PLAN, (
        "a bounded snapshot and an observation check, no hardware action: "
        f"{[step.operation for step in workflow.official_steps]}"
    )

    _post(context, recovered)

    state = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is None, "the recovery clears it"
    assert _incidents(context) == 1, "and opens nothing new"


CLOCK_LEGACY = "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"
CLOCK_CURRENT = "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"
CLOCK = "clock_throttle_reasons"


def test_two_exporter_names_of_one_field_that_disagree_drop_the_whole_slot() -> None:
    """GF-GENERAL-LIVE-025 ``conflicting_alias`` as served live (2026-10-26
    15:32Z, HyperPod, H200): both GPUs publish the clock event reasons under the
    legacy name (reading 1); on GPU-b the current name arrives beside it with
    64 (``0x40``, HW slowdown) while GPU-a keeps its one name.

    Two names of one canonical field with equal values and labels are one
    reading (``deduplicate_dcgm_aliases``); disagreeing, they are one ``(device,
    canonical)`` slot seen twice, which the control plane rejects whole as
    "invalid or ambiguous" (``normalize_gpu_observation``): neither the injected
    value nor the published reading is accepted, the slot's latest row stays the
    earlier reading and its time, no finding opens on the slot (the injected
    ``0x40`` would have been a thermal-limiting candidate), and GPU-a's reading
    of the same field in the same scrape is accepted. The node forwards both
    samples and delivers the scrape on its own edge: the alias's clock-reason
    candidate is confirmed over three scrapes before the edge ships, which is
    why the live product's first rejection (15:33:01Z) trailed the first
    conflicting read (~15:32:30Z) by two scrapes. A scrape with one name again
    is the recovery edge and advances the slot with its real value.
    """

    one_name = {"GPU-a": (CLOCK_CURRENT,), "GPU-b": (CLOCK_CURRENT,)}
    legacy_one = {"GPU-a": {CLOCK_LEGACY: 1}, "GPU-b": {CLOCK_LEGACY: 1}}
    healthy = scrape(omit=one_name, values=legacy_one)
    conflicting = scrape(
        omit={"GPU-a": (CLOCK_CURRENT,)},
        values={
            "GPU-a": {CLOCK_LEGACY: 1},
            "GPU-b": {CLOCK_LEGACY: 1, CLOCK_CURRENT: 64},
        },
    )
    sink = Sink()
    reader = collector(sink)
    delivered = []
    plan = [(0, healthy), (15, healthy), (30, healthy), (45, healthy)]
    plan += [(offset, conflicting) for offset in (60, 75, 90)]
    for offset, text in plan:
        before = len(sink.batches)
        reader.collect_text(text, observed_at=START + timedelta(seconds=offset))
        if len(sink.batches) > before:
            delivered.append((offset, sink.batches[-1].edge_filter_reasons))
    assert delivered == [(0, ["initial-baseline"]), (90, ["candidate-confirmed"])], (
        "the pass-through scrapes are no edge; the conflicting scrape ships once "
        f"the alias's clock-reason candidate is confirmed: {delivered}"
    )
    baseline, served = sink.batches
    served_at = START + timedelta(seconds=90)
    assert served.collection_errors == [], "the node reports no completeness gap"
    assert {
        (sample.metric_name, sample.value)
        for sample in served.samples
        if sample.gpu_uuid == "GPU-b" and sample.canonical_name == CLOCK
    } == {(CLOCK_LEGACY, 1.0), (CLOCK_CURRENT, 64.0)}, (
        "the node forwards both names of the disagreeing slot; the control plane judges"
    )

    context = ApplicationContext(store=InMemoryStore())
    _post(context, baseline)
    _post(context, served)

    latest = _latest(context)
    assert latest[("GPU-b", CLOCK)] == (1.0, START), (
        "the ambiguous slot keeps the earlier accepted reading and its time: "
        f"{latest[('GPU-b', CLOCK)]}"
    )
    assert latest[("GPU-a", CLOCK)] == (1.0, served_at), (
        f"GPU-a's one-name reading of the same scrape is accepted: {latest[('GPU-a', CLOCK)]}"
    )
    assert _slot_finding(context, "GPU-b", CLOCK) is None, (
        "the injected 0x40 never becomes a thermal-limiting finding"
    )
    (evidence,) = [
        item
        for item in context.store.list_raw_evidence(
            CLUSTER, node_id=NODE, kind=EvidenceKind.GPU_METRICS, limit=10
        )
        if item.observed_at == served_at
    ]
    assert evidence.payload["collection_errors"] == [f"{INVALID}: GPU-b/{CLOCK}"], (
        f"the stored batch names the ambiguous slot: {evidence.payload['collection_errors']}"
    )
    assert not [
        sample
        for sample in evidence.payload["samples"]
        if sample["gpu_uuid"] == "GPU-b" and sample["canonical_name"] == CLOCK
    ], "the stored batch holds neither reading of the ambiguous slot"
    assert _incidents(context) == 0

    recovered_at = START + timedelta(seconds=105)
    reader.collect_text(healthy, observed_at=recovered_at)
    recovered = sink.batches[-1]
    assert recovered.observed_at == recovered_at and "candidate-recovered" in (
        recovered.edge_filter_reasons
    ), f"one name again is the recovery edge: {recovered.edge_filter_reasons}"
    _post(context, recovered)
    assert _latest(context)[("GPU-b", CLOCK)] == (1.0, recovered_at), (
        "the real reading is accepted again at the recovery"
    )
    assert _incidents(context) == 0


def _latest(
    context: ApplicationContext,
) -> dict[tuple[str, str], tuple[float, datetime]]:
    return {
        (str(item.sample.gpu_uuid), item.sample.canonical_name): (
            item.sample.value,
            item.observed_at,
        )
        for item in context.store.list_gpu_metrics_latest(CLUSTER, NODE)
    }


def _slot_finding(context: ApplicationContext, gpu: str, name: str):
    state = context.store.get_gpu_finding_state((CLUSTER, NODE, gpu, name))
    return None if state is None else state.finding


def test_a_nan_and_a_negative_reading_keep_the_earlier_accepted_value() -> None:
    """GF-GENERAL-LIVE-025 ``nan_negative`` as served live (2026-10-26, H200):
    GPU-b's row remap failure flag reads the Prometheus literal ``NaN`` and its
    PCIe replay counter ``-1.0`` while GPU-a stays healthy.

    Neither reading may become a value. The NaN slot is gone at the node's
    parser and reads as GPU-b's missing required field; the negative counter is
    forwarded (a counter reset is a delivery edge) and rejected by the control
    plane as an invalid observation. The product's latest accepted sample of
    both slots stays the earlier healthy one, timestamped by the scrape that
    carried it -- retained, not converted -- while GPU-a's slots advance. No
    finding on either slot, no incident; the clean scrape afterwards is the
    recovery edge and advances both slots with their real value.
    """

    sink = Sink()
    reader = collector(sink)
    reader.collect_text(scrape(), observed_at=START)
    served_at = START + timedelta(seconds=15)
    reader.collect_text(
        scrape(values={"GPU-b": {PCIE: "-1.0", ROW_REMAP_FAILURE: "NaN"}}),
        observed_at=served_at,
    )
    baseline, served = sink.batches
    assert {"fields-incomplete", "counter-reset"} <= set(served.edge_filter_reasons), (
        f"the invalid scrape is delivered on its own edges: {served.edge_filter_reasons}"
    )
    assert served.collection_errors == [f"{MISSING} on GPU-b: row_remap_failure"], (
        f"the NaN slot is GPU-b's missing field, not a reading: {served.collection_errors}"
    )
    assert [
        sample.value
        for sample in served.samples
        if sample.gpu_uuid == "GPU-b" and sample.canonical_name == "pcie_replay_total"
    ] == [-1.0], (
        "the node forwards the negative counter as served; the control plane judges it"
    )

    context = ApplicationContext(store=InMemoryStore())
    _post(context, baseline)
    _post(context, served)

    latest = _latest(context)
    for name in ("pcie_replay_total", "row_remap_failure"):
        assert latest[("GPU-b", name)] == (0.0, START), (
            f"GPU-b/{name} keeps the earlier accepted sample and its time, it is not "
            f"converted to a healthy value at the invalid scrape: {latest[('GPU-b', name)]}"
        )
        assert latest[("GPU-a", name)] == (0.0, served_at), (
            f"GPU-a's valid readings of the same scrape are accepted: {latest[('GPU-a', name)]}"
        )
        assert _slot_finding(context, "GPU-b", name) is None, (
            f"an invalid reading is no fault on GPU-b/{name}"
        )
    (evidence,) = [
        item
        for item in context.store.list_raw_evidence(
            CLUSTER, node_id=NODE, kind=EvidenceKind.GPU_METRICS, limit=10
        )
        if item.observed_at == served_at
    ]
    assert evidence.payload["collection_errors"] == [
        f"{MISSING} on GPU-b: row_remap_failure",
        f"{INVALID}: GPU-b/pcie_replay_total",
    ], (
        f"the stored batch names both rejections: {evidence.payload['collection_errors']}"
    )
    assert not [
        sample
        for sample in evidence.payload["samples"]
        if sample["gpu_uuid"] == "GPU-b"
        and sample["canonical_name"] in {"pcie_replay_total", "row_remap_failure"}
    ], "the stored batch holds no accepted sample of either rejected slot"
    completeness = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert completeness is not None and completeness.finding is not None
    assert completeness.finding.automatic_action == RecoveryAction.NO_ACTION.value
    assert "GPU-b/pcie_replay_total" in completeness.finding.reason, (
        completeness.finding.reason
    )
    assert "GPU-b: row_remap_failure" in completeness.finding.reason, (
        completeness.finding.reason
    )
    assert _incidents(context) == 0, "inside the sustain window nothing opens"

    recovered_at = START + timedelta(seconds=30)
    reader.collect_text(scrape(), observed_at=recovered_at)
    recovered = sink.batches[-1]
    assert "collection-recovered" in recovered.edge_filter_reasons, (
        recovered.edge_filter_reasons
    )
    _post(context, recovered)

    latest = _latest(context)
    for name in ("pcie_replay_total", "row_remap_failure"):
        assert latest[("GPU-b", name)] == (0.0, recovered_at), (
            f"the real reading is accepted again at the recovery: {latest[('GPU-b', name)]}"
        )
        assert _slot_finding(context, "GPU-b", name) is None, (
            "the negative-to-zero step is no counter fault"
        )
    completeness = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert completeness is not None and completeness.finding is None, (
        "the clean scrape clears the completeness finding"
    )
    assert _incidents(context) == 0
