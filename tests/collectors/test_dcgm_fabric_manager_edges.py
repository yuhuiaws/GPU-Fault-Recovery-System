"""The DCGM collector reports how long each GPU has held its NVLink fabric state.

Batch 127: ``DCGM_FI_DEV_FABRIC_MANAGER_STATUS`` (170) and
``DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE`` (171) are exported. The collector maps
them, adds ``fabric_manager_status_held_seconds`` beside every GPU whose state
is not settled (measured from its own first observation of the value), delivers
every state change at once and the crossing of the window once confirmed, so
the control plane grades a node whose fabric never trained within minutes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.gpu_fabric_manager import (
    FABRIC_MANAGER_ERROR_CODE,
    FABRIC_MANAGER_HELD_SECONDS,
    FABRIC_MANAGER_STATUS,
    FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS,
)
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService

CLUSTER = "fabric-cluster"
NODE = "fabric-node"
SUMMARY_SECONDS = 86_400
GPUS = [f"GPU-{index}" for index in range(8)]
START = next_stable_phase(
    datetime(2026, 10, 16, 20, 33, tzinfo=timezone.utc),
    cluster_id=CLUSTER,
    node_id=NODE,
    channel="gpu-metrics",
    interval_seconds=SUMMARY_SECONDS,
) + timedelta(seconds=1)
DEFAULTS = {"gpu_temperature_c": 40, "memory_temperature_c": 45, "power_limit_w": 700}
#: What dcgm-exporter would never emit (it drops blank values) but a raw DCGM
#: sentinel looks like, should one ever reach the text.
DCGM_INT64_BLANK = 9223372036854775792


def exporter_text(status: int, *, error_code: int | None = None) -> str:
    lines = []
    for index, uuid in enumerate(GPUS):
        labels = f'gpu="{index}",UUID="{uuid}",modelName="NVIDIA H200"'
        for name, (canonical, _unit) in DCGM_METRICS.items():
            if canonical == FABRIC_MANAGER_STATUS:
                value: float = status
            elif canonical == FABRIC_MANAGER_ERROR_CODE:
                if error_code is None:
                    continue
                value = error_code
            else:
                value = DEFAULTS.get(canonical, 0)
            lines.append(f"{name}{{{labels}}} {value}\n")
    return "".join(lines)


class RecordingSink:
    def __init__(self) -> None:
        self.service = GpuMetricsService()
        self.batches: list[GpuMetricBatch] = []
        self.results: list = []

    def post(self, path: str, payload: dict) -> dict:
        assert path == GPU_METRICS_PATH, "fabric state rides on GPU metrics"
        batch = GpuMetricBatch.model_validate(payload)
        self.batches.append(batch)
        result = self.service.ingest(batch)
        self.results.append(result)
        return result.model_dump(mode="json")


class Node:
    def __init__(self, sink: RecordingSink | None = None, start: datetime = START):
        self.sink = sink or RecordingSink()
        self.collector = DcgmMetricsCollector(
            self.sink,
            CollectorContext(cluster_id=CLUSTER),
            node_id=NODE,
            health_summary_seconds=SUMMARY_SECONDS,
            edge_confirmation_samples=3,
        )
        self.at = start

    def tick(self, status: int, **kwargs) -> GpuMetricBatch:
        batch = self.collector.collect_text(
            exporter_text(status, **kwargs), observed_at=self.at
        )
        self.at += timedelta(seconds=15)
        return batch


def _held(batch: GpuMetricBatch) -> dict[str, float]:
    return {
        str(sample.gpu_uuid): sample.value
        for sample in batch.samples
        if sample.canonical_name == FABRIC_MANAGER_HELD_SECONDS
    }


def _fabric_findings(results) -> list:
    return [
        finding
        for result in results
        for finding in result.new_findings
        if finding.canonical_name == FABRIC_MANAGER_STATUS
    ]


def test_both_fields_are_mapped_and_a_blank_error_code_is_skipped() -> None:
    node = Node()

    batch = node.tick(3, error_code=0)
    names = {sample.canonical_name for sample in batch.samples}
    assert {FABRIC_MANAGER_STATUS, FABRIC_MANAGER_ERROR_CODE} <= names
    assert _held(batch) == {}, "a settled GPU carries no held companion"

    blank = Node().collector.collect_text(
        exporter_text(2, error_code=DCGM_INT64_BLANK), observed_at=START
    )
    assert FABRIC_MANAGER_ERROR_CODE not in {
        sample.canonical_name for sample in blank.samples
    }, "a blank sentinel is not a value"


def test_the_held_clock_counts_from_the_first_observation_of_the_value() -> None:
    node = Node()

    first = node.tick(2)
    second = node.tick(2)
    third = node.tick(2)
    changed = node.tick(1)

    assert set(_held(first)) == set(GPUS)
    assert set(_held(first).values()) == {0.0}, "nothing remembered yet"
    assert set(_held(second).values()) == {15.0}
    assert set(_held(third).values()) == {30.0}
    assert set(_held(changed).values()) == {0.0}, "a new state starts a new clock"


def test_a_restarted_collector_starts_the_clock_again() -> None:
    node = Node()
    for _ in range(4):
        node.tick(2)

    restarted = Node(start=node.at)

    assert set(_held(restarted.tick(2)).values()) == {0.0}, (
        "after a quiesce, a restore or a reboot the window starts over"
    )


def test_a_state_change_is_delivered_at_once() -> None:
    node = Node()
    node.tick(2)
    node.tick(2)
    delivered = len(node.sink.batches)

    node.tick(3)

    assert len(node.sink.batches) == delivered + 1, (
        "VALIDATE_GPU waits on the change to Success; it must not wait for the "
        "next health summary"
    )
    assert "fabric-manager-status-changed" in node.sink.batches[-1].edge_filter_reasons


def test_an_untrained_fabric_reaches_the_control_plane_as_one_finding() -> None:
    node = Node()
    ticks_to_window = FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS // 15

    for _ in range(ticks_to_window):
        node.tick(2)
    assert _fabric_findings(node.sink.results) == [], (
        "boot-time training inside the window is never a finding"
    )
    for _ in range(3):
        node.tick(2)

    last = node.sink.batches[-1]
    assert "candidate-confirmed" in last.edge_filter_reasons, (
        "the crossing is delivered once confirmed, not at the next health summary"
    )
    (finding,) = _fabric_findings(node.sink.results)
    assert finding.affected_gpu_uuids == GPUS
    assert finding.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"

    node.tick(3)
    assert "candidate-recovered" in node.sink.batches[-1].edge_filter_reasons
    assert all(
        item.canonical_name != FABRIC_MANAGER_STATUS
        for item in node.sink.service.findings(CLUSTER, NODE)
    ), "the fabric trained: the finding clears"
