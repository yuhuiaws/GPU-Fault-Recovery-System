"""DCGM field-replay host controller: what it reads and reports.

``probes/general_live_dcgm_probe.py`` runs on the node. These tests run its
read-only side in-process -- the survey, the real exporter's scrape through the
product's own parser, the journal's boot list, the GPU inventory and limits,
``/proc`` reads, the loopback client and the audit journal -- over the recorded
systemd sandbox (``_dcgm_probe_fakes``) plus scripted journal, ``nvidia-smi``,
exporter and ``/proc`` answers (``_cov95_glive_dcgm``).
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import ProxyHandler

import pytest

from gpu_fault.collectors.gpu import discovery
from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._cov95_glive_dcgm import BDF0, BOOT_ID, SCRAPE, UUID0, Node, node
from tests.regional._dcgm_probe_fakes import DIGEST, ENDPOINT, REAL, RUN, sandbox

LIMITS_XML = f"""<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="{BDF0}">
    <product_name>NVIDIA H200</product_name>
    <uuid>{UUID0}</uuid>
    <minor_number>0</minor_number>
    <pci><pci_bus_id>{BDF0}</pci_bus_id></pci>
    <temperature>
      <gpu_temp>41 C</gpu_temp>
      <gpu_temp_tlimit>48 C</gpu_temp_tlimit>
      <gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>
      <gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>
      <gpu_temp_max_gpu_tlimit_threshold>0 C</gpu_temp_max_gpu_tlimit_threshold>
    </temperature>
  </gpu>
  <gpu id="00000000:64:00.0">
    <product_name>NVIDIA A100-SXM4-80GB</product_name>
    <minor_number>1</minor_number>
    <temperature>
      <gpu_temp_max_threshold>92 C</gpu_temp_max_threshold>
      <gpu_temp_slow_threshold>89 C</gpu_temp_slow_threshold>
    </temperature>
  </gpu>
</nvidia_smi_log>
"""
UNDASHED = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Node:
    return node(monkeypatch, tmp_path)


@pytest.fixture
def proc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Only ``/proc`` faked: the probe's own ``/proc`` readers run for real."""

    root = tmp_path / "proc"

    def path(value: Any, *parts: Any) -> Path:
        item = Path(value, *parts)
        return (
            root / item.relative_to("/proc") if item.is_relative_to("/proc") else item
        )

    monkeypatch.setattr(probe, "Path", path)
    return root


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


# --------------------------------------------------------------------------- #
# output, commands, collector settings
# --------------------------------------------------------------------------- #
def test_emit_prints_one_sorted_compact_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe.emit({"z": 1, "a": Path("/run/gfrs-live")})
    assert capsys.readouterr().out == '{"a":"/run/gfrs-live","z":1}\n'


def test_run_fails_on_a_nonzero_exit_only_when_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_run(command: list[str], **options: Any) -> Any:
        seen.append(options)
        return subprocess.CompletedProcess(command, 4, "", " Unit not found. \n")

    fake = SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE)
    monkeypatch.setattr(probe, "subprocess", fake)
    with pytest.raises(probe.ProbeError) as caught:
        probe.run(["systemctl", "restart", probe.METRICS_UNIT])
    assert str(caught.value) == "command failed (4): systemctl: Unit not found."
    assert (caught.value.reason, caught.value.detail) == ("", None)
    completed = probe.run(["systemctl", "show", "x"], check=False, timeout=7)
    assert completed.returncode == 4
    common = {"text": True, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    assert seen == [
        {**common, "timeout": 120, "check": False},
        {**common, "timeout": 7, "check": False},
    ]


def test_collector_env_reads_only_the_documented_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox(monkeypatch, tmp_path)
    probe.COLLECTOR_ENV.write_text(
        "# written by the node bootstrap\n\n"
        "GPU_FAULT_METRICS_MODE = 'dcgm'\n"
        'GPU_FAULT_EXPECTED_GPU_COUNT="8"\n'
        "AWS_SECRET_ACCESS_KEY=never-read-out\n"
        "a line without an assignment\n"
    )
    assert probe.collector_env() == {
        "GPU_FAULT_METRICS_MODE": "dcgm",
        "GPU_FAULT_EXPECTED_GPU_COUNT": "8",
    }
    assert probe.baseline_endpoint() == probe.DEFAULT_UPSTREAM, "no endpoint set"
    probe.COLLECTOR_ENV.unlink()
    assert probe.collector_env() == {}, "a node without collector.env"
    assert probe.baseline_endpoint() == probe.DEFAULT_UPSTREAM


def test_unit_state_keeps_assignments_and_the_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "ActiveState=active\nWarning: unit changed on disk\n\nMainPID=42\n"
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, output, ""),
    )
    assert probe.unit_state(probe.METRICS_UNIT) == {
        "ActiveState": "active",
        "MainPID": "42",
        "returncode": "0",
    }


# --------------------------------------------------------------------------- #
# /proc: the collector's real environment, and whether it has exec'd
# --------------------------------------------------------------------------- #
def test_process_environment_needs_a_live_pid_and_a_readable_environ(
    proc: Path,
) -> None:
    (proc / "321").mkdir(parents=True)
    (proc / "321" / "environ").write_bytes(
        b"GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9412/metrics\0"
        b"AWS_SECRET_ACCESS_KEY=x\0PATH=/usr/bin\0"
    )
    assert probe.process_environment("321") == {
        ENDPOINT: "http://127.0.0.1:9412/metrics",
        "PATH": "/usr/bin",
    }
    for pid in ("", "0", "12a"):
        assert probe.process_environment(pid) == {}, f"no process for {pid!r}"
    assert probe.process_environment("999") == {}, "a process that exited"


def test_has_execed_tells_the_service_from_systemds_pre_exec_child(
    proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    links = {
        "/proc/10/exe": "/opt/gpu-fault/current/venv/bin/python3.12",
        "/proc/11/exe": "/usr/lib/systemd/systemd",
        "/proc/12/exe": "/usr/lib/systemd/systemd",
    }
    real_readlink = os.readlink

    def readlink(path: Any, *args: Any, **kwargs: Any) -> str:
        if not str(path).startswith("/proc/"):
            return real_readlink(path, *args, **kwargs)
        if str(path) not in links:
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return links[str(path)]

    monkeypatch.setattr(probe.os, "readlink", readlink)
    for pid, comm in (("10", "python3"), ("11", "(python3)"), ("12", "python3")):
        (proc / pid).mkdir(parents=True)
        (proc / pid / "comm").write_text(comm + "\n")
    assert probe.has_execed("10") is True
    assert probe.has_execed("11") is False, "systemd names the child (python3)"
    assert probe.has_execed("12") is False, "the child still runs systemd's exe"
    assert probe.has_execed("13") is False, "no such process"


def test_wait_active_returns_the_last_state_when_no_main_process_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    fake.unit(probe.METRICS_UNIT, MainPID="0")
    state = probe.wait_active(probe.METRICS_UNIT, seconds=5)
    assert (state["ActiveState"], state["MainPID"]) == ("active", "0")
    assert probe.time.slept == 5.0, "the whole window was waited out"


# --------------------------------------------------------------------------- #
# the loopback client and the tool's /status
# --------------------------------------------------------------------------- #
def test_loopback_speaks_only_to_loopback_and_never_through_a_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[Any, ...]] = []
    handlers: list[tuple[Any, ...]] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self, size: int) -> bytes:
            requests.append(("read", size))
            return b"body"

    class Opener:
        def open(self, request: Any, timeout: int) -> Response:
            requests.append(
                (
                    request.get_method(),
                    dict(request.header_items()),
                    request.data,
                    timeout,
                )
            )
            if request.full_url.endswith("/refused"):
                raise ConnectionRefusedError(111, "Connection refused")
            return Response()

    def build_opener(*given: Any) -> Opener:
        handlers.append(given)
        return Opener()

    monkeypatch.setattr(probe, "build_opener", build_opener)
    with pytest.raises(probe.ProbeError, match="only loopback http is allowed"):
        probe.loopback("http://10.0.0.5:9400/metrics")
    assert handlers == [], "a non-loopback URL never opens a connection"
    assert probe.loopback("http://127.0.0.1:9412/status") == (200, b"body")
    reply = probe.loopback("http://127.0.0.1:9412/arm", method="POST", timeout=5)
    assert reply == (200, b"body")
    ack = {"X-gfrs-replay-ack": probe.ACKNOWLEDGEMENT}
    assert requests == [
        ("GET", {}, None, 10),
        ("read", 4 * 1024 * 1024),
        ("POST", ack, b"", 5),
        ("read", 4 * 1024 * 1024),
    ]
    for given in handlers:
        assert len(given) == 1 and isinstance(given[0], ProxyHandler), given
        assert given[0].proxies == {}, "no proxy environment is ever used"
    with pytest.raises(probe.ProbeError, match="loopback GET failed: .*refused"):
        probe.loopback("http://127.0.0.1:9413/refused")


def test_tool_status_refuses_an_error_status_and_a_non_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter([(503, b"busy"), (200, b"[1, 2]"), (200, b'{"armed": false}')])
    urls: list[str] = []

    def loopback(url: str, **_: Any) -> tuple[int, bytes]:
        urls.append(url)
        return next(answers)

    monkeypatch.setattr(probe, "loopback", loopback)
    with pytest.raises(probe.ProbeError, match="replay /status returned 503"):
        probe.tool_status(9412)
    with pytest.raises(probe.ProbeError, match="did not return an object"):
        probe.tool_status(9412)
    assert probe.tool_status(9412) == {"armed": False}
    assert set(urls) == {"http://127.0.0.1:9412/status"}


# --------------------------------------------------------------------------- #
# survey inputs: inventory, scrape, boots, limits, leftovers
# --------------------------------------------------------------------------- #
def test_gpu_inventory_keeps_only_complete_csv_rows(host: Node) -> None:
    host.inventory = (
        f"0, {UUID0}, {BDF0}, NVIDIA H200\n"
        "1, GPU-b, 00000000:64:00.0, NVIDIA H200, with, commas\n"
        "not,a,row\n\n"
    )
    assert probe.gpu_inventory() == [
        {"index": "0", "uuid": UUID0, "pci_bus_id": BDF0, "name": "NVIDIA H200"},
        {
            "index": "1",
            "uuid": "GPU-b",
            "pci_bus_id": "00000000:64:00.0",
            "name": "NVIDIA H200, with, commas",
        },
    ]
    assert host.fake.commands[-1] == [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader",
    ]


def test_exporter_readings_parse_the_scrape_with_the_products_parser(
    host: Node,
) -> None:
    body = SCRAPE.encode()
    host.answers[REAL] = (200, body)
    readings = probe.exporter_readings(REAL)
    assert readings["url"] == REAL and readings["sample_count"] == 3
    assert readings["sha256"] == hashlib.sha256(body).hexdigest()
    assert readings["gpus"] == {
        UUID0: {
            "gpu_index": "0",
            "pci_bdf": BDF0,
            "model": "NVIDIA H200",
            "values": {"gpu_temperature_c": 41.0, "power_usage_w": 312.5},
            "metric_names": {
                "gpu_temperature_c": ["DCGM_FI_DEV_GPU_TEMP"],
                "power_usage_w": ["DCGM_FI_DEV_POWER_USAGE"],
            },
        }
    }, "the sample without a UUID is counted but belongs to no GPU"
    missing = readings["batch_missing_groups"]
    assert missing and "gpu_temperature_c" not in missing, missing
    assert readings["per_gpu_missing_groups"] == {UUID0: missing}
    host.answers[REAL] = probe.ProbeError("loopback GET failed: refused")
    assert probe.exporter_readings(REAL) == {"error": "loopback GET failed: refused"}
    host.answers[REAL] = (500, b"")
    assert probe.exporter_readings(REAL) == {"error": "exporter returned 500"}


def test_recorded_run_id_trusts_only_a_record_that_names_this_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox(monkeypatch, tmp_path)
    root = probe.run_dir(RUN)
    root.mkdir(parents=True)
    marker, repoint = root / probe.RUN_MARKER, root / "repoint.json"
    marker.write_text("[]")
    repoint.write_text(json.dumps({"run_id": RUN}))
    assert probe.recorded_run_id(DIGEST) == RUN, "the marker is no object"
    marker.write_text("{not json")
    repoint.write_text(json.dumps({"run_id": "gl019-another-run-a1-cafebabe"}))
    assert probe.recorded_run_id(DIGEST) is None, "another run's id"
    marker.write_text(json.dumps({"run_id": "Bad Run"}))
    assert probe.recorded_run_id(DIGEST) is None, "an unsafe id"


def test_residue_names_no_run_for_unit_names_outside_the_pattern(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    fake.unit("gfrs-live-notadigest-proxy.service")
    fake.unit(f"gfrs-live-{DIGEST}-other.service")
    residue = probe.residue()
    assert residue["units"] == [
        f"gfrs-live-{DIGEST}-other.service",
        "gfrs-live-notadigest-proxy.service",
    ]
    assert residue["runs"] == {}, "neither name is a proxy or restore unit"
    assert (residue["port"], residue["port_answers"]) == (None, None)


def test_prior_boots_lists_earlier_boots_newest_first(host: Node) -> None:
    current = BOOT_ID.replace("-", "")
    host.boots = (
        "IDX BOOT ID                          FIRST ENTRY  LAST ENTRY\n"
        f" -2 {UNDASHED} Mon 2026-11-02 10:00:00 UTC\n"
        f" -1 {'b' * 32} Tue 2026-11-03 10:00:00 UTC\n"
        f"  0 {current} Wed 2026-11-04 10:00:00 UTC\n"
        f" -1 {'b' * 32} listed twice\n"
    )
    assert probe.prior_boots() == [
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    ], "the current boot and a repeated id are left out"
    host.boots = "".join(f" -{n} {n:032x} x\n" for n in range(1, 8))
    assert [boot[-1] for boot in probe.prior_boots()] == ["7", "6", "5", "4", "3"]


def test_temperature_limits_go_through_the_products_xml_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = limits_query(monkeypatch)
    assert probe.temperature_limits() == [
        {
            "gpu_uuid": None,
            "gpu_index": "1",
            "pci_bdf": None,
            "product": "NVIDIA A100-SXM4-80GB",
            "limits": {
                "gpu_slowdown_temperature_c": 89.0,
                "gpu_shutdown_temperature_c": 92.0,
            },
        },
        {
            "gpu_uuid": UUID0,
            "gpu_index": "0",
            "pci_bdf": BDF0,
            "product": "NVIDIA H200",
            "limits": {
                "gpu_shutdown_tlimit_c": -8.0,
                "gpu_slowdown_tlimit_c": -2.0,
                "gpu_max_operating_tlimit_c": 0.0,
            },
        },
    ], "a GPU without a UUID is keyed by its index"
    assert commands == [["nvidia-smi", "-q", "-x"]]


def test_survey_reports_the_collector_scrape_inventory_and_residue(
    host: Node, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits_query(monkeypatch)
    host.boots = f" -1 {UNDASHED} Mon\n"
    host.answers[REAL] = (200, SCRAPE.encode())
    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}",
    )
    assert probe.main(["survey", "--run-id", RUN, "--limits"]) == 0
    report = host.fake.emitted[-1]
    assert report["boot_id"] == BOOT_ID
    assert report["prior_boot_ids"] == ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]
    assert report["collector_env"] == {ENDPOINT: REAL, "GPU_FAULT_METRICS_MODE": "dcgm"}
    assert report["metrics_unit"]["MainPID"] == "100"
    assert report["process_environment"] == {ENDPOINT: REAL}
    assert report["baseline_endpoint"] == REAL
    assert [gpu["uuid"] for gpu in report["gpu_inventory"]] == [UUID0]
    assert report["exporter"]["gpus"][UUID0]["values"]["gpu_temperature_c"] == 41.0
    assert (report["residue"]["port"], report["residue"]["port_answers"]) == (
        9412,
        False,
    ), "no tool answers the replay port"
    assert report["tools"] == {
        "systemd-run": True,
        "systemctl": True,
        "nvidia-smi": False,
    }
    assert report["venv_python"] is True
    assert [gpu["gpu_index"] for gpu in report["temperature_limits"]] == ["1", "0"]
    assert probe.main(["survey", "--run-id", RUN]) == 0
    assert "temperature_limits" not in host.fake.emitted[-1], "only with --limits"


# --------------------------------------------------------------------------- #
# the audit journal and the script entry point
# --------------------------------------------------------------------------- #
def test_audit_read_pages_the_replay_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    root = probe.run_dir(RUN)
    root.mkdir(parents=True)
    (root / "replay.jsonl").write_text('{"read": 1}\n{"read": 2}\nnot json at all\n')
    assert probe.main(["audit-read", "--run-id", RUN, "--offset", "1"]) == 0
    assert fake.emitted[-1] == {
        "run_id": RUN,
        "journal": "replay.jsonl",
        "records": [{"read": 2}, {"unparsed": "not json at all"}],
        "next_offset": 3,
        "total": 3,
    }
    argv = ["audit-read", "--run-id", RUN, "--journal", "inventory.jsonl"]
    assert probe.main(argv) == 0
    assert (fake.emitted[-1]["records"], fake.emitted[-1]["total"]) == ([], 0)
    assert probe.main(["audit-read", "--run-id", RUN, "--offset", "-1"]) == 1
    assert fake.emitted[-1]["error_message"] == "offset is outside 0..1000000"


def test_the_probe_file_runs_as_a_script_and_reports_a_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["general_live_dcgm_probe.py", "survey", "--run-id", "Bad Run"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert caught.value.code == 1
    failure = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert failure["error"] == "ProbeError: unsafe run ID"
    assert failure["command"] == "survey" and failure["error_reason"] is None
    assert failure["error_site"].startswith("checked_run_id:"), (
        "the innermost frame of this file names the site"
    )
