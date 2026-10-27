"""The host collector's device-plugin liveness contributor
(``*_device_plugin_unresponsive``, GF-GENERAL-LIVE-014 hung plugin).

Live 2026-10-27 (HyperPod, ``gpu_plugin``): the NVIDIA plugin frozen for four
minutes and ``nvidia.com/gpu`` allocatable 8 throughout -- kubelet holds the
plugin's ``ListAndWatch`` stream open without a deadline, so the control-plane
allocatable rule never fires. The node probes the plugin's own socket instead.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_fault.collectors.host.collector import build_from_environment
from gpu_fault.collectors.host.device_plugin_probe import ProbeResult, ProbeState
from gpu_fault.device_plugin_liveness import (
    DEFAULT_CONSECUTIVE_SAMPLES,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    DEFAULT_SOCKET_DIR,
    parse_socket_patterns,
)

from ._support import NOW, HostTelemetryCollector, RecordingSink, context, timedelta

GPU = "gpu_device_plugin_unresponsive"
EFA = "efa_device_plugin_unresponsive"


def make_socket(path: Path) -> None:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.close()


class ScriptedProbe:
    """Answers per socket basename; records every call and its timeout."""

    def __init__(self, **states: ProbeState) -> None:
        self.states = states
        self.calls: list[tuple[str, float]] = []

    def __call__(self, path: Path, timeout: float) -> ProbeResult:
        self.calls.append((path.name, timeout))
        state = self.states.get(path.name, ProbeState.RESPONDED)
        return ProbeResult(
            state,
            latency_ms=timeout * 1000.0 if state is ProbeState.TIMEOUT else 1.5,
            detail=f"scripted {state.value}",
            transport_answered=state is not ProbeState.TIMEOUT,
        )


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    # A unix socket path is limited to 108 bytes (``sun_path``); pytest's
    # tmp_path carries the test name and overflows it.
    path = Path(tempfile.mkdtemp(prefix="gfrs-dp-"))
    make_socket(path / "kubelet.sock")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def collector(
    socket_dir: Path | None, probe, monkeypatch: pytest.MonkeyPatch, **kwargs
) -> HostTelemetryCollector:
    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_device_plugins",))
    return HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        node_instance_type="p5en.48xlarge",
        device_plugin_socket_dir=str(socket_dir) if socket_dir else None,
        device_plugin_probe=probe,
        **kwargs,
    )


def by_name(batch) -> dict:
    return {sample.name: sample for sample in batch.samples}


def test_a_hung_gpu_plugin_opens_the_finding_on_the_second_tick_and_clears(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_socket(socket_dir / "nvidia-gpu.sock")
    probe = ScriptedProbe(**{"nvidia-gpu.sock": ProbeState.TIMEOUT})
    host = collector(socket_dir, probe, monkeypatch)

    first = host.collect_once()
    sample = by_name(first)[GPU]
    assert sample.value == 0, "one timed-out tick is a suspicion, not a finding"
    assert sample.labels["probe_state"] == "TIMEOUT"
    assert sample.labels["consecutive_timeout_samples"] == "1"
    assert sample.labels["required_consecutive_samples"] == "2"
    assert sample.labels["failure_mode"] == "HEALTHY"
    assert first.edge_filter_reasons == ["baseline"]

    host.now = lambda: NOW + timedelta(seconds=15)
    second = host.collect_once()
    sample = by_name(second)[GPU]
    assert sample.value == 1
    assert sample.labels == {
        "resource": "GPU",
        "resource_name": "nvidia.com/gpu",
        "node_instance_type": "p5en.48xlarge",
        "sockets": "nvidia-gpu.sock",
        "probe_state": "TIMEOUT",
        "probe_states": "nvidia-gpu.sock=TIMEOUT",
        "probe_detail": "nvidia-gpu.sock: scripted TIMEOUT",
        "transport_answered": "false",
        "latency_ms": "5000.0",
        "timeout_seconds": "5",
        "consecutive_timeout_samples": "2",
        "required_consecutive_samples": "2",
        "failure_mode": "DEVICE_PLUGIN_UNRESPONSIVE",
        "expected_count": "8",
    }
    assert f"threshold:{GPU}" in second.edge_filter_reasons, second.edge_filter_reasons
    efa = by_name(second)[EFA]
    assert efa.value == 0 and efa.labels["probe_state"] == "SOCKET_MISSING", efa
    assert efa.labels["expected_count"] == "16"
    assert [name for name, _ in probe.calls] == ["nvidia-gpu.sock"] * 2, (
        "kubelet.sock is kubelet's registration server, never probed"
    )
    assert {timeout for _, timeout in probe.calls} == {5.0}, (
        "every probe is bounded by the configured timeout"
    )

    probe.states["nvidia-gpu.sock"] = ProbeState.RESPONDED
    host.now = lambda: NOW + timedelta(seconds=30)
    third = host.collect_once()
    sample = by_name(third)[GPU]
    assert sample.value == 0
    assert sample.labels["probe_state"] == "RESPONDED"
    assert sample.labels["consecutive_timeout_samples"] == "0"
    assert "recovered" in third.edge_filter_reasons, third.edge_filter_reasons


def test_the_efa_family_matches_the_plugin_socket_by_the_resource_word(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_socket(socket_dir / "aws-efa-device-plugin.sock")
    make_socket(socket_dir / "nvidia-gpu.sock")
    make_socket(socket_dir / "other-vendor.sock")
    probe = ScriptedProbe(**{"aws-efa-device-plugin.sock": ProbeState.TIMEOUT})
    host = collector(
        socket_dir, probe, monkeypatch, device_plugin_unresponsive_samples=1
    )

    batch = host.collect_once()

    efa = by_name(batch)[EFA]
    assert efa.value == 1
    assert efa.labels["resource_name"] == "vpc.amazonaws.com/efa"
    assert efa.labels["sockets"] == "aws-efa-device-plugin.sock"
    assert efa.labels["failure_mode"] == "DEVICE_PLUGIN_UNRESPONSIVE"
    assert by_name(batch)[GPU].value == 0
    assert sorted(name for name, _ in probe.calls) == [
        "aws-efa-device-plugin.sock",
        "nvidia-gpu.sock",
    ], "a socket of no known family is not probed"


def test_a_hang_on_one_of_several_gpu_sockets_is_the_family_hung(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NVIDIA plugin serves one socket per resource name (MIG ``mixed``)."""

    for name in ("nvidia-gpu.sock", "nvidia-mig-1g.10gb.sock"):
        make_socket(socket_dir / name)
    probe = ScriptedProbe(**{"nvidia-mig-1g.10gb.sock": ProbeState.TIMEOUT})
    host = collector(
        socket_dir, probe, monkeypatch, device_plugin_unresponsive_samples=1
    )

    sample = by_name(host.collect_once())[GPU]

    assert sample.value == 1
    assert sample.labels["probe_state"] == "TIMEOUT"
    assert sample.labels["probe_states"] == (
        "nvidia-gpu.sock=RESPONDED,nvidia-mig-1g.10gb.sock=TIMEOUT"
    )


def test_a_gone_or_refusing_plugin_is_never_a_hang_and_resets_the_count(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kubelet sees a plugin that exited (its ListAndWatch stream ended) and the
    allocatable rule owns that; a stale socket, a refused connection or no
    socket at all must not open this finding, and must not keep a count alive."""

    make_socket(socket_dir / "nvidia-gpu.sock")
    probe = ScriptedProbe(**{"nvidia-gpu.sock": ProbeState.TIMEOUT})
    host = collector(socket_dir, probe, monkeypatch)
    assert (
        by_name(host.collect_once())[GPU].labels["consecutive_timeout_samples"] == "1"
    )

    probe.states["nvidia-gpu.sock"] = ProbeState.CONNECT_REFUSED
    host.now = lambda: NOW + timedelta(seconds=15)
    sample = by_name(host.collect_once())[GPU]
    assert sample.value == 0
    assert sample.labels["probe_state"] == "CONNECT_REFUSED"
    assert sample.labels["consecutive_timeout_samples"] == "0"

    (socket_dir / "nvidia-gpu.sock").unlink()
    probe.states["nvidia-gpu.sock"] = ProbeState.TIMEOUT
    host.now = lambda: NOW + timedelta(seconds=30)
    sample = by_name(host.collect_once())[GPU]
    assert sample.value == 0
    assert sample.labels["probe_state"] == "SOCKET_MISSING"
    assert sample.labels["sockets"] == "none"
    assert sample.labels["probe_detail"] == "no socket matched nvidia-*.sock"


def test_a_node_without_the_kubelet_directory_is_not_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = ScriptedProbe()
    host = collector(tmp_path / "absent", probe, monkeypatch)

    batch = host.collect_once()

    assert not [sample for sample in batch.samples if sample.name in {GPU, EFA}]
    assert batch.collection_errors == []
    assert probe.calls == []


def test_a_disabled_probe_emits_nothing(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_socket(socket_dir / "nvidia-gpu.sock")
    probe = ScriptedProbe()
    host = collector(None, probe, monkeypatch)

    batch = host.collect_once()

    assert not [sample for sample in batch.samples if sample.name in {GPU, EFA}]
    assert probe.calls == []


def test_an_unreadable_directory_is_a_collection_error_not_a_verdict(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = collector(socket_dir, ScriptedProbe(), monkeypatch)

    def refuse(_self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", refuse)

    batch = host.collect_once()

    assert not [sample for sample in batch.samples if sample.name in {GPU, EFA}]
    assert len(batch.collection_errors) == 1
    assert batch.collection_errors[0].startswith("_device_plugins: "), (
        batch.collection_errors
    )
    assert "unreadable" in batch.collection_errors[0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device_plugin_probe_timeout_seconds": 0}, "probe timeout must be positive"),
        (
            {"device_plugin_unresponsive_samples": 0},
            "unresponsive samples must be positive",
        ),
    ],
)
def test_knobs_outside_their_range_refuse_to_start(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        collector(socket_dir, ScriptedProbe(), monkeypatch, **kwargs)


def test_the_factory_reads_the_documented_knobs(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR", str(socket_dir))
    monkeypatch.setenv("GPU_FAULT_DEVICE_PLUGIN_PROBE_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("GPU_FAULT_DEVICE_PLUGIN_UNRESPONSIVE_CONSECUTIVE_SAMPLES", "3")
    monkeypatch.setenv(
        "GPU_FAULT_DEVICE_PLUGIN_SOCKET_PATTERNS",
        "efa=aws-efa-*.sock, gpu=nvidia-gpu.sock,gpu=nvidia-mig-*.sock",
    )
    arguments = argparse.Namespace(node_id="worker-1", interval_seconds=15)

    host = build_from_environment(RecordingSink(), context(), arguments)

    assert host.device_plugin_socket_dir == socket_dir
    assert host.device_plugin_probe_timeout_seconds == 2.5
    assert host.device_plugin_unresponsive_samples == 3
    assert host.device_plugin_socket_patterns == {
        "gpu": ("nvidia-gpu.sock", "nvidia-mig-*.sock"),
        "efa": ("aws-efa-*.sock",),
    }
    assert "_device_plugins" in host.CONTRIBUTORS

    monkeypatch.setenv("GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR", "")
    assert (
        build_from_environment(
            RecordingSink(), context(), arguments
        ).device_plugin_socket_dir
        is None
    ), "an empty directory disables the probe"

    for name in (
        "GPU_FAULT_DEVICE_PLUGIN_SOCKET_DIR",
        "GPU_FAULT_DEVICE_PLUGIN_PROBE_TIMEOUT_SECONDS",
        "GPU_FAULT_DEVICE_PLUGIN_UNRESPONSIVE_CONSECUTIVE_SAMPLES",
        "GPU_FAULT_DEVICE_PLUGIN_SOCKET_PATTERNS",
    ):
        monkeypatch.delenv(name)
    unset = build_from_environment(RecordingSink(), context(), arguments)
    assert unset.device_plugin_socket_dir == Path(DEFAULT_SOCKET_DIR)
    assert unset.device_plugin_probe_timeout_seconds == DEFAULT_PROBE_TIMEOUT_SECONDS
    assert unset.device_plugin_unresponsive_samples == DEFAULT_CONSECUTIVE_SAMPLES
    assert unset.device_plugin_socket_patterns == parse_socket_patterns(None), (
        "the factory's literal defaults (the documented ones) are the constants"
    )


def test_the_defaults_name_the_kubelet_directory_and_both_plugins() -> None:
    assert parse_socket_patterns(None) == {
        "gpu": ("nvidia-*.sock",),
        "efa": ("*efa*.sock",),
    }
    assert parse_socket_patterns(" ") == parse_socket_patterns(None)
    assert parse_socket_patterns("efa=efa.sock") == {
        "gpu": ("nvidia-*.sock",),
        "efa": ("efa.sock",),
    }, "a family left out keeps its default"


@pytest.mark.parametrize("text", ["nvidia-*.sock", "tpu=foo.sock", "gpu=", "=x"])
def test_malformed_socket_patterns_refuse_to_start(text: str) -> None:
    with pytest.raises(ValueError, match="GPU_FAULT_DEVICE_PLUGIN_SOCKET_PATTERNS"):
        parse_socket_patterns(text)
