"""COLLECT-002's throttle caps only the GPU it loads.

Capping every GPU made the idle GPUs' power-violation counters advance with
the cap and confirm duty-cycle candidates of their own around the loaded GPU's
confirmation, in batches that do not say whose candidate confirmed. Only the
loaded GPU's correlated throttle is the case's anomaly, so only that GPU is
capped; every other GPU must read its baseline limit before the load starts,
and the restore still returns any GPU found off its baseline (an operation
recorded under the all-GPU design included).
"""

from __future__ import annotations

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional.test_collector_power_safety import PowerHost
from tests.regional.test_collector_power_safety import power_host as power_host_fixture

power_host = power_host_fixture


@pytest.mark.parametrize(
    ("index", "limits"), [(0, [200.0, 700.0]), (1, [700.0, 250.0])]
)
def test_the_throttle_caps_only_the_loaded_gpu(
    power_host: PowerHost, index: int, limits: list[float]
) -> None:
    host = power_host

    probe.throttle_gpu(host.arguments(gpu_index=index))

    gpu = host.gpus[index]
    assert host.power_writes() == [
        [str(probe.POWER_SMI), "-i", gpu["uuid"], "-pl", str(gpu["power_min_limit_w"])]
    ]
    assert [item["power_limit_w"] for item in host.gpus] == limits
    receipt = host.emitted[-1]
    assert [item["power_limit_w"] for item in receipt["gpu_power"]] == limits
    assert probe.power_caps(host.record()) == limits
    assert host.executions[-1][-2:] == ["-i", str(index)], "the load runs on it"
    assert receipt["load_onset"]["loaded_read_at"] is not None


def test_a_gpu_moved_off_its_baseline_during_the_cap_refuses_the_load(
    power_host: PowerHost,
) -> None:
    """Another writer lowers the idle GPU while the loaded one is capped: the
    readback names it and no load starts; the restore puts both back."""

    host = power_host

    def foreign_cap(command: list[str]) -> None:
        if "-pl" in command:
            host.gpus[1]["power_limit_w"] = host.gpus[1]["power_min_limit_w"]

    host.after_call = foreign_cap
    with pytest.raises(probe.ProbeError, match="cap readback does not match"):
        probe.throttle_gpu(host.arguments())
    assert [call[-1] for call in host.power_writes()] == ["200.0"]
    assert host.record()["load_intended"] is False and not host.executions
    host.after_call = None
    host.calls.clear()

    probe.restore_gpu_power_limit(host.arguments())

    assert [call[2] for call in host.power_writes()] == [
        host.gpus[0]["uuid"],
        host.gpus[1]["uuid"],
    ]
    assert [item["power_limit_w"] for item in host.gpus] == [700.0, 700.0]
    assert host.emitted[-1]["cleanup_verified"] is True


@pytest.mark.parametrize(
    ("loaded", "idle"),
    [(200.0, 250.0), (700.0, 700.0)],
    ids=["idle-gpu-capped", "loaded-gpu-restored"],
)
def test_the_load_starts_only_on_the_single_verified_cap(
    power_host: PowerHost, loaded: float, idle: float
) -> None:
    host = power_host
    probe.throttle_gpu(host.arguments())
    assert [call[-1] for call in host.power_writes()] == ["200.0"]
    host.gpus[0]["power_limit_w"], host.gpus[1]["power_limit_w"] = loaded, idle

    with pytest.raises(probe.ProbeError, match="requires the verified power cap"):
        probe.power_load_only(host.arguments())


def test_the_restore_returns_every_gpu_found_at_its_minimum(
    power_host: PowerHost,
) -> None:
    """An idle GPU at its minimum -- as an operation of the all-GPU design left
    it -- is still within the operation's limits and is restored and verified."""

    host = power_host
    probe.throttle_gpu(host.arguments())
    assert [call[-1] for call in host.power_writes()] == ["200.0"]
    host.gpus[1]["power_limit_w"] = host.gpus[1]["power_min_limit_w"]
    host.calls.clear()

    probe.restore_gpu_power_limit(host.arguments())

    assert [call[-1] for call in host.power_writes()] == ["700.0", "700.0"]
    assert [item["power_limit_w"] for item in host.gpus] == [700.0, 700.0]
    receipt = host.emitted[-1]
    assert receipt["restored"] is True and receipt["cleanup_verified"] is True
