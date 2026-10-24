"""Environmental signals must survive aliases, edge filtering and CSV fallback."""

from subprocess import CompletedProcess

import pytest

from gpu_fault.collectors.gpu.dcgm import parse_dcgm_samples
from gpu_fault.collectors.gpu.nvidia_smi import NvidiaSmiMetricsCollector
from gpu_fault.gpu_metrics import GpuMetricBatch
from gpu_fault.gpu_observation import normalize_gpu_observation
from tests.execution._raw_health_e2e_support import RawHealthRig


@pytest.mark.parametrize(
    "values,expected", [([128, 128], 1), ([128, 0], 0), ([128, 128, 128], 0)]
)
def test_aliases_only_deduplicate_one_identical_pair(values, expected):
    names = ["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"]
    text = "".join(
        f'{names[min(index, 1)]}{{UUID="GPU-a"}} {value}\n'
        for index, value in enumerate(values)
    )
    batch = normalize_gpu_observation(
        GpuMetricBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            source="DCGM_EXPORTER",
            observed_at="2026-10-23T00:00:00Z",
            samples=parse_dcgm_samples(text),
        )
    )
    assert len(batch.samples) == expected, batch
    assert bool(batch.collection_errors) is (expected == 0), batch


@pytest.mark.parametrize(
    "values", [{"gpu_temperature_c": 95}, {"clock_throttle_reasons": 0x80}]
)
def test_constant_environmental_samples_reach_control_plane_confirmation(
    tmp_path, monkeypatch, values
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:
        rig.exporter.edge_filter_enabled = True
        rig.gpu({})
        counts = []
        for _ in range(3):
            rig.advance(15)
            result = rig.gpu(values)
            counts.append(
                max(item["confirmation_count"] for item in result["findings"])
            )
        assert counts == [1, 2, 3], counts
        assert result["findings"][0]["automatic_action"] == "DRAIN"
    finally:
        rig.store.close()


def test_a_clear_between_brakes_is_delivered_even_before_edge_confirmation(
    tmp_path, monkeypatch
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:
        rig.exporter.edge_filter_enabled = True
        rig.gpu({})
        for mask in (0x80, 0, 0x80, 0, 0x80):
            rig.advance(15)
            result = rig.gpu({"clock_throttle_reasons": mask})
            assert all(
                item["automatic_action"] != "DRAIN" for item in result["findings"]
            ), result
        assert result["findings"][0]["confirmation_count"] == 1
    finally:
        rig.store.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_nvidia_smi_collects_power_brake_hex_and_optional_environment_fields(
    tmp_path, monkeypatch, legacy
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    values = {
        "index": "0",
        "uuid": "GPU-a",
        "name": "NVIDIA H100",
        "pci.bus_id": "0000:81:00.0",
        "temperature.gpu": "40",
        "temperature.memory": "45",
        "power.draw": "300",
        "power.limit": "700",
        "clocks_event_reasons.active": "0x0000000000000080",
        "clocks_throttle_reasons.active": "0x0000000000000080",
    }
    commands = []

    def runner(argv, **_):
        commands.append(argv)
        if "-q" in argv:
            return CompletedProcess(argv, 0, "<nvidia_smi_log/>", "")
        query = next(
            item.split("=", 1)[1] for item in argv if item.startswith("--query-gpu=")
        )
        if legacy and "clocks_event_reasons.active" in query:
            return CompletedProcess(argv, 1, "", "invalid field")
        return CompletedProcess(
            argv,
            0,
            ",".join(values.get(name, "0") for name in query.split(",")) + "\n",
            "",
        )

    try:
        collector = NvidiaSmiMetricsCollector(
            rig.sink,
            rig.raw_context,
            node_id="node-a",
            runner=runner,
            now=lambda: rig.clock,
        )
        batch = collector.collect_once()
        by_name = {item.canonical_name: item for item in batch.samples}
        assert by_name["clock_throttle_reasons"].value == 128
        assert by_name["memory_temperature_c"].value == 45
        assert by_name["power_limit_w"].value == 700
        assert by_name["clock_throttle_reasons"].labels["source_boot_id"], (
            "confirmations need boot identity"
        )
        assert all(
            "--power-limit" not in argument
            for command in commands
            for argument in command
        ), commands
    finally:
        rig.store.close()
