"""GF-ENV-LIVE host controller: what it reads from the node and the proxy.

``probes/env_replay_proxy.py`` runs on the node. These tests run its read-only
side in-process over ``EnvNode`` (``_cov95_glive_env_replay``): ``collector.env``,
``systemctl show`` / ``list-units``, ``/proc`` (environment, exec, stop state),
the loopback client against the proxy and the exporter (the probe's own
``build_opener`` answered by the fake), ``nvidia-smi`` and the product's
temperature-limit parser, and the failure line ``main`` prints.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.collectors.gpu import discovery
from scripts.e2e.regional.probes import env_replay_proxy as probe
from tests.regional._cov95_glive_env_replay import (
    ENDPOINT,
    METRICS,
    REAL,
    RUN,
    EnvNode,
    env_node,
    proxy_up,
)

UUID0 = "GPU-11111111-2222-3333-4444-555555555555"
UUID1 = "GPU-66666666-7777-8888-9999-000000000000"
LIMITS_XML = f"""<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:53:00.0">
    <product_name>NVIDIA H200</product_name>
    <uuid>{UUID0}</uuid>
    <minor_number>0</minor_number>
    <pci><pci_bus_id>00000000:53:00.0</pci_bus_id></pci>
    <temperature>
      <gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>
      <gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>
    </temperature>
  </gpu>
  <gpu id="00000000:64:00.0">
    <product_name>NVIDIA A100-SXM4-80GB</product_name>
    <minor_number>1</minor_number>
    <temperature>
      <gpu_temp_max_threshold>92 C</gpu_temp_max_threshold>
    </temperature>
  </gpu>
</nvidia_smi_log>
"""
EXPORTER = (
    "# TYPE DCGM_FI_DEV_GPU_TEMP_LIMIT gauge\n"
    f'DCGM_FI_DEV_GPU_TEMP_LIMIT{{gpu="0",UUID="{UUID0}"}} 48.5\n'
    f'DCGM_FI_DEV_GPU_TEMP_LIMIT{{gpu="1",UUID="{UUID1}"}} -3\n'
    'DCGM_FI_DEV_GPU_TEMP_LIMIT{gpu="2"} 40\n'
    f'DCGM_FI_DEV_GPU_TEMP{{gpu="0",UUID="{UUID0}"}} 41\n'
)


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EnvNode:
    return env_node(monkeypatch, tmp_path)


# --------------------------------------------------------------------------- #
# the command helper, collector.env, systemd reads
# --------------------------------------------------------------------------- #
def test_run_raises_on_a_checked_failure_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def fake_run(command: list[str], **options: Any) -> Any:
        seen.append(options)
        return subprocess.CompletedProcess(command, 5, "", "Unit not loaded.\n")

    fake = SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE)
    monkeypatch.setattr(probe, "subprocess", fake)
    with pytest.raises(probe.ProbeError) as caught:
        probe.run(["systemctl", "restart", METRICS])
    assert str(caught.value) == "command failed (5): systemctl: Unit not loaded."
    assert probe.run(["systemctl", "show", METRICS], check=False).returncode == 5
    assert seen[0] == {
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": 120,
        "check": False,
    }


def test_parse_env_requires_collector_env_and_skips_comments(node: EnvNode) -> None:
    probe.COLLECTOR_ENV.write_text(
        f"# installer\n\n{ENDPOINT}='{REAL}'\nnot an assignment\nGPU_FAULT_X=1\n"
    )
    assert probe.parse_env() == {ENDPOINT: REAL, "GPU_FAULT_X": "1"}
    probe.COLLECTOR_ENV.unlink()
    with pytest.raises(probe.ProbeError, match="collector env file does not exist"):
        probe.baseline_endpoint()


def test_unit_state_keeps_assignments_and_the_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "ActiveState=active\nWarning: unit file changed on disk\nMainPID=7\n"
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, output, ""),
    )
    assert probe.unit_state(METRICS) == {
        "ActiveState": "active",
        "MainPID": "7",
        "returncode": "0",
    }


def test_a_node_without_runs_has_no_windows_pauses_or_records(node: EnvNode) -> None:
    assert probe.open_windows() == [], "no acceptance root at all"
    assert probe.pause_labels(RUN) == [], "no run directory"
    root = probe.run_dir(RUN)
    root.mkdir(parents=True)
    probe.pause_path(RUN).write_text("{not json")
    assert probe.read_pause(RUN) == {}, "an unreadable pause record is no pause"
    probe.pause_path(RUN, "recount").write_text("[1]")
    assert probe.read_pause(RUN, "recount") == {}, "a record must be an object"
    assert probe.pause_labels(RUN) == ["", "recount"]
    assert probe.open_windows() == [probe.digest(RUN)]


def test_unit_is_stopped_reads_the_main_process_state(node: EnvNode) -> None:
    assert probe.unit_is_stopped(METRICS) is False, "running (S)"
    node.signal("100", "T")
    assert probe.unit_is_stopped(METRICS) is True, "SIGSTOPped (T)"
    node.unit(METRICS, MainPID="555")
    assert probe.unit_is_stopped(METRICS) is False, "no /proc entry"
    node.unit(METRICS, MainPID="0")
    assert probe.unit_is_stopped(METRICS) is False, "no main process"


def test_dropin_path_lives_in_the_metrics_units_runtime_drop_in_directory(
    node: EnvNode, tmp_path: Path
) -> None:
    directory = tmp_path / "run-root" / "systemd" / "system" / f"{METRICS}.d"
    assert probe.dropin_dir() == directory
    assert probe.dropin_path(RUN) == (
        directory / f"gpu-fault-env-replay-{probe.digest(RUN)}.conf"
    )


def test_has_execed_is_false_for_a_process_that_is_gone(node: EnvNode) -> None:
    assert probe.has_execed("100") is True
    node.process("101", {}, execed=False)
    assert probe.has_execed("101") is False, "systemd's pre-exec child"
    assert probe.has_execed("4242") is False, "no such process"


def test_settled_environment_returns_what_it_saw_when_the_window_closes(
    node: EnvNode,
) -> None:
    node.collector("101", {"GPU_FAULT_METRICS_MODE": "dcgm"}, execed=False)
    state, environment = probe.settled_environment(METRICS, seconds=3)
    assert state["MainPID"] == "101"
    assert environment == {"GPU_FAULT_METRICS_MODE": "dcgm"}
    assert node.clock.sleeps == [0.5] * 6, "polled until the window closed"


# --------------------------------------------------------------------------- #
# loopback HTTP: the proxy's /status, the exporter's field 153
# --------------------------------------------------------------------------- #
def test_proxy_status_reads_the_tool_over_loopback_only(node: EnvNode) -> None:
    status_url = "http://127.0.0.1:9411/status"
    node.http[status_url] = ConnectionRefusedError(111, "Connection refused")
    with pytest.raises(probe.ProbeError, match=f"proxy GET {status_url} failed"):
        probe.proxy_status(9411)
    node.http[status_url] = (503, "starting\n")
    with pytest.raises(probe.ProbeError, match="proxy /status returned 503"):
        probe.proxy_status(9411)
    node.http[status_url] = (200, "[1, 2]")
    with pytest.raises(probe.ProbeError, match="did not return an object"):
        probe.proxy_status(9411)
    del node.http[status_url]
    proxy_up(node, reads=4)
    assert probe.proxy_status(9411) == {"armed": False, "reads": 4}
    assert set(node.requests) == {("GET", status_url)}


def test_current_tlimit_readings_take_field_153_per_gpu_from_the_exporter(
    node: EnvNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    node.exporter = EXPORTER
    assert probe.current_tlimit_readings() == {UUID0: 48.5, UUID1: -3.0}, (
        "a sample without a UUID and other fields are skipped"
    )
    node.http[REAL] = (500, "")
    assert probe.current_tlimit_readings() == {}, "an exporter error reads nothing"
    node.http[REAL] = ConnectionResetError(104, "Connection reset by peer")
    assert probe.current_tlimit_readings() == {}, "an unreachable exporter"
    requests = len(node.requests)
    monkeypatch.setattr(probe, "DEFAULT_UPSTREAM", "http://10.1.2.3:9400/metrics")
    assert probe.current_tlimit_readings() == {}, "never off the loopback"
    assert len(node.requests) == requests, "no connection was opened"


def test_wait_first_read_waits_out_a_proxy_that_does_not_answer(node: EnvNode) -> None:
    result = probe.wait_first_read(9411, 0, 3)
    assert result == {
        "observed": False,
        "reads_before": 0,
        "reads_after": 0,
        "waited_seconds": 3.0,
        "budget_seconds": 3,
    }


def test_status_command_reports_the_proxy_and_the_effective_endpoint(
    node: EnvNode,
) -> None:
    proxy_up(node, reads=2)
    assert probe.main(["status", "--run-id", RUN]) == 0
    assert node.emitted[-1] == {
        "run_id": RUN,
        "status": {"armed": False, "reads": 2},
        "effective_endpoint": REAL,
    }


def test_a_failure_inside_the_standard_library_names_the_probe_frame(
    node: EnvNode, capsys: pytest.CaptureFixture[str]
) -> None:
    node.http["http://127.0.0.1:9411/status"] = (200, "not json")
    assert probe.main(["status", "--run-id", RUN]) == 1
    failure = node.emitted[-1]
    assert failure["error_class"] == "JSONDecodeError"
    assert failure["error_site"].startswith("proxy_status:"), failure
    assert failure["command"] == "status"
    err = capsys.readouterr().err
    assert err.startswith("JSONDecodeError: Expecting value"), err


# --------------------------------------------------------------------------- #
# preflight inputs: limits through the product's parser, the GPU inventory
# --------------------------------------------------------------------------- #
def limits_query(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The product's limit query over a scripted ``nvidia-smi -q -x``."""

    commands: list[list[str]] = []
    query = discovery.query_nvidia_temperature_limits

    def runner(command: list[str], **_: Any) -> Any:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, LIMITS_XML, "")

    monkeypatch.setattr(
        discovery, "query_nvidia_temperature_limits", lambda: query(runner=runner)
    )
    return commands


def test_device_limits_go_through_the_products_xml_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = limits_query(monkeypatch)
    assert probe.device_limits() == [
        {
            "gpu_uuid": None,
            "gpu_index": "1",
            "pci_bdf": None,
            "product": "NVIDIA A100-SXM4-80GB",
            "limits": {"gpu_shutdown_temperature_c": 92.0},
        },
        {
            "gpu_uuid": UUID0,
            "gpu_index": "0",
            "pci_bdf": "00000000:53:00.0",
            "product": "NVIDIA H200",
            "limits": {"gpu_shutdown_tlimit_c": -8.0, "gpu_slowdown_tlimit_c": -2.0},
        },
    ], "a GPU without a UUID is keyed by its index"
    assert commands == [["nvidia-smi", "-q", "-x"]]


def test_gpu_inventory_keeps_only_complete_rows(node: EnvNode) -> None:
    node.inventory = (
        f"0, {UUID0}, 00000000:53:00.0, NVIDIA H200\n"
        "1, GPU-b, 00000000:64:00.0, NVIDIA H200, extra\nshort,row\n"
    )
    assert probe.gpu_inventory() == [
        {
            "index": "0",
            "uuid": UUID0,
            "pci_bus_id": "00000000:53:00.0",
            "name": "NVIDIA H200",
        },
        {
            "index": "1",
            "uuid": "GPU-b",
            "pci_bus_id": "00000000:64:00.0",
            "name": "NVIDIA H200, extra",
        },
    ]
    assert node.commands[-1] == [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader",
    ]


def test_preflight_reads_the_node_without_changing_it(
    node: EnvNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    node.exporter = EXPORTER
    node.inventory = f"0, {UUID0}, 00000000:53:00.0, NVIDIA H200\n"
    (node.proc / "sys" / "kernel" / "random").mkdir(parents=True)
    (node.proc / "sys" / "kernel" / "random" / "boot_id").write_text("boot-7\n")
    limits_query(monkeypatch)
    node.unit("gpu-fault-env-replay-0123456789abcdef.service", MainPID="9")
    assert probe.main(["preflight"]) == 0
    report = node.emitted[-1]
    assert report["boot_id"] == "boot-7"
    assert report["effective_endpoint"] == REAL == report["baseline_endpoint"]
    assert report["current_tlimit_readings"] == {UUID0: 48.5, UUID1: -3.0}
    assert [gpu["uuid"] for gpu in report["gpu_inventory"]] == [UUID0]
    assert [unit["unit"] for unit in report["leftover_units"]] == [
        "gpu-fault-env-replay-0123456789abcdef.service"
    ], "an earlier run's proxy is named before anything is deployed"
    assert len(report["device_limits"]) == 2
    verbs = {c[1] if c[0] == "systemctl" else c[0] for c in node.commands}
    assert verbs == {"show", "list-units", "nvidia-smi"}, "reads only"


def test_the_probe_file_runs_as_a_script_and_reports_a_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["env_replay_proxy.py", "status", "--run-id", "bad run id"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert caught.value.code == 1
    out, err = capsys.readouterr()
    failure = json.loads(out.splitlines()[-1])
    assert failure["error_message"] == "unsafe run ID"
    assert failure["command"] == "status"
    assert failure["error_site"].startswith("safe_id:"), failure
    assert err == "ProbeError: unsafe run ID\n"
