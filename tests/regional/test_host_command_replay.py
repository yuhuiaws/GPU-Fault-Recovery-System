"""Contract tests for the reviewed pass-through command wrapper of the host replay.

Nothing here touches systemd or a node: ``tools/host_command_replay.py`` is
validated as a library (the per-tool, per-field plan allow-list) and run as a
standalone ``python -I`` process against stub real binaries placed on a PATH
that also carries the run's ``bin/`` shim directory, to prove it forwards
untouched calls, rewrites only the planned field, never invents a successful
answer for a failed real query, honours the call bound and the deadline, and
journals what it served.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import host_command_replay as wrapper

TOOL = Path(wrapper.__file__)
UUID_A = "GPU-11111111-2222-3333-4444-555555555555"
UUID_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# --------------------------------------------------------------------------- #
# the allow-list
# --------------------------------------------------------------------------- #
def _plan(tool: str, *rules: dict) -> dict:
    return {"tool": tool, "not_after": time.time() + 600, "rules": list(rules)}


def test_catalog_names_the_spec_tools_and_the_collector_units_only() -> None:
    assert set(wrapper.TOOLS) == {"smartctl", "ipmitool", "ethtool", "nvidia-smi"}, (
        "the wrapper catalogue is exactly the spec's command sources"
    )
    assert wrapper.ALLOWED_UNITS == (
        "gpu-fault-host-collector.service",
        "gpu-fault-kernel-collector.service",
    ), "the host collector, and the kernel collector for its version query only"
    assert wrapper.NODE_AGENT_UNIT not in wrapper.ALLOWED_UNITS, (
        "the Node Agent is never in the allow-list"
    )
    with pytest.raises(wrapper.PlanError, match="Node Agent"):
        wrapper.checked_unit(wrapper.NODE_AGENT_UNIT)
    with pytest.raises(wrapper.PlanError, match="may not carry a wrapper"):
        wrapper.checked_unit("gpu-fault-metrics-collector.service")
    with pytest.raises(wrapper.PlanError, match="never in a replay allow-list"):
        wrapper.checked_tool("py-spy")
    with pytest.raises(wrapper.PlanError, match="not a replayable tool"):
        wrapper.checked_tool("journalctl")


def test_kernel_collector_carries_the_driver_version_query_only() -> None:
    """KERNEL_COLLECTOR_SOURCE_REPLAY: one unit, one tool, one query."""

    assert wrapper.UNIT_QUERIES == {
        wrapper.KERNEL_UNIT: {"nvidia-smi": frozenset({"driver_version"})}
    }
    assert wrapper.unit_queries(wrapper.HOST_UNIT, "nvidia-smi") is None, (
        "the host collector may carry every catalogued query"
    )
    assert wrapper.unit_queries(wrapper.KERNEL_UNIT, "smartctl") == frozenset()
    version = {
        "query": "driver_version",
        "action": "exit",
        "exit_code": 1,
        "stderr": "withheld\n",
    }
    checked = wrapper.checked_unit_plan(
        wrapper.KERNEL_UNIT, "nvidia-smi", _plan("nvidia-smi", version)
    )
    assert [rule["query"] for rule in checked["rules"]] == ["driver_version"]
    passthrough = {"query": "driver_version", "action": "passthrough"}
    wrapper.checked_unit_plan(
        wrapper.KERNEL_UNIT, "nvidia-smi", _plan("nvidia-smi", passthrough)
    )
    with pytest.raises(wrapper.PlanError, match="may rewrite only"):
        wrapper.checked_unit_plan(
            wrapper.KERNEL_UNIT,
            "nvidia-smi",
            _plan(
                "nvidia-smi",
                version,
                {"query": "utilization", "action": "drop_row", "uuid": UUID_A},
            ),
        )
    with pytest.raises(wrapper.PlanError, match="may rewrite only"):
        wrapper.checked_unit_plan(
            wrapper.KERNEL_UNIT,
            "nvidia-smi",
            _plan(
                "nvidia-smi", {"query": "topology", "action": "exit", "exit_code": 2}
            ),
        )
    with pytest.raises(wrapper.PlanError, match="may not carry a smartctl wrapper"):
        wrapper.checked_unit_plan(
            wrapper.KERNEL_UNIT,
            "smartctl",
            _plan("smartctl", {"query": "scan", "action": "exit", "exit_code": 2}),
        )
    with pytest.raises(wrapper.PlanError, match="Node Agent"):
        wrapper.checked_unit_plan(
            wrapper.NODE_AGENT_UNIT, "nvidia-smi", _plan("nvidia-smi", version)
        )
    # The host collector keeps every reviewed field (the 012 row drop included).
    host = wrapper.checked_unit_plan(
        wrapper.HOST_UNIT,
        "nvidia-smi",
        _plan(
            "nvidia-smi",
            version,
            {"query": "utilization", "action": "drop_row", "uuid": UUID_A},
        ),
    )
    assert len(host["rules"]) == 2
    # A malformed plan is still a plan error before the unit is considered.
    with pytest.raises(wrapper.PlanError, match="exactly the keys"):
        wrapper.checked_unit_plan(wrapper.KERNEL_UNIT, "nvidia-smi", {"tool": "x"})


def test_validate_plan_accepts_each_reviewed_field() -> None:
    checked = wrapper.validate_plan(
        "smartctl",
        _plan(
            "smartctl",
            {
                "query": "health",
                "action": "set_json",
                "device": "/dev/nvme0",
                "field": "smart_status.passed",
                "value": False,
            },
            {"query": "health", "action": "delete_json", "field": "smart_status"},
            {"query": "scan", "action": "replace_output", "stdout": "", "exit_code": 1},
            {"query": "health", "action": "exit", "exit_code": 2, "stderr": "boom"},
        ),
    )
    assert len(checked["rules"]) == 4, checked
    wrapper.validate_plan(
        "ethtool",
        _plan(
            "ethtool",
            {
                "query": "stats",
                "interface": "rdmap79s0",
                "action": "set_counter",
                "counter": "rx_drops",
                "value": "5000",
                "max_calls": 3,
            },
            {
                "query": "stats",
                "action": "delete_counter",
                "counter": "rnr_nak_retry_err",
            },
        ),
    )
    wrapper.validate_plan(
        "nvidia-smi",
        _plan(
            "nvidia-smi",
            {
                "query": "utilization",
                "action": "drop_row",
                "uuid": UUID_A,
                "max_calls": 1,
            },
            {
                "query": "utilization",
                "action": "set_column",
                "uuid": UUID_B,
                "column": "utilization.gpu",
                "value": "0",
            },
            {"query": "driver_version", "action": "exit", "exit_code": 9},
        ),
    )
    wrapper.validate_plan(
        "ipmitool",
        _plan(
            "ipmitool",
            {
                "query": "sensor",
                "action": "replace_output",
                "stdout": "CPU Temp | 40 | degrees C | cr | ...\n",
            },
        ),
    )


@pytest.mark.parametrize(
    "tool, plan_rule, match",
    [
        ("smartctl", {"query": "sensor", "action": "exit", "exit_code": 1}, "no query"),
        (
            "smartctl",
            {"query": "scan", "action": "set_json", "field": "x", "value": 1},
            "allows only",
        ),
        (
            "smartctl",
            {
                "query": "health",
                "action": "set_json",
                "field": "temperature.current",
                "value": 1,
            },
            "not allow-listed",
        ),
        (
            "smartctl",
            {
                "query": "health",
                "action": "set_json",
                "field": "smart_status.passed",
                "value": "false",
            },
            "takes a boolean",
        ),
        (
            "smartctl",
            {"query": "health", "action": "delete_json", "field": "smartctl"},
            "may not be deleted",
        ),
        (
            "smartctl",
            {"query": "health", "action": "exit", "exit_code": 0},
            "exit_code",
        ),
        (
            "smartctl",
            {"query": "health", "action": "exit", "exit_code": 1, "stdout": "x"},
            "unexpected keys",
        ),
        (
            "smartctl",
            {"query": "health", "action": "exit", "exit_code": 1, "device": "nvme0"},
            "not allow-listed",
        ),
        (
            "ethtool",
            {
                "query": "stats",
                "action": "set_counter",
                "counter": "rx drops",
                "value": "1",
            },
            "counter",
        ),
        (
            "ethtool",
            {
                "query": "stats",
                "action": "set_counter",
                "counter": "rx_drops",
                "value": "1e9",
            },
            "value",
        ),
        (
            "ethtool",
            {"query": "stats", "action": "passthrough", "delay_seconds": 121},
            "delay_seconds",
        ),
        (
            "ethtool",
            {"query": "stats", "action": "passthrough", "max_calls": 1001},
            "max_calls",
        ),
        (
            "nvidia-smi",
            {"query": "utilization", "action": "drop_row", "uuid": "GPU-1"},
            "uuid",
        ),
        (
            "nvidia-smi",
            {
                "query": "utilization",
                "action": "set_column",
                "uuid": UUID_A,
                "column": "uuid",
                "value": "0",
            },
            "column",
        ),
        (
            "nvidia-smi",
            {
                "query": "utilization",
                "action": "set_column",
                "uuid": UUID_A,
                "column": "utilization.gpu",
                "value": "1000",
            },
            "value",
        ),
        (
            "nvidia-smi",
            {"query": "compute_apps", "action": "drop_row", "uuid": UUID_A},
            "allows only",
        ),
        (
            "ipmitool",
            {
                "query": "sensor",
                "action": "replace_output",
                "stdout": "x" * (64 * 1024 + 1),
            },
            "at most",
        ),
    ],
)
def test_validate_plan_rejects_fields_outside_the_allow_list(
    tool: str, plan_rule: dict, match: str
) -> None:
    with pytest.raises(wrapper.PlanError, match=match):
        wrapper.validate_plan(tool, _plan(tool, plan_rule))


def test_validate_plan_shape_and_bounds() -> None:
    with pytest.raises(wrapper.PlanError, match="exactly the keys"):
        wrapper.validate_plan("smartctl", {"tool": "smartctl", "rules": []})
    with pytest.raises(wrapper.PlanError, match="not 'ethtool'"):
        wrapper.validate_plan(
            "ethtool", _plan("smartctl", {"query": "scan", "action": "passthrough"})
        )
    with pytest.raises(wrapper.PlanError, match="1..16 rules"):
        wrapper.validate_plan("smartctl", _plan("smartctl"))
    too_many = [{"query": "scan", "action": "passthrough"}] * (wrapper.MAX_RULES + 1)
    with pytest.raises(wrapper.PlanError, match="1..16 rules"):
        wrapper.validate_plan("smartctl", _plan("smartctl", *too_many))
    with pytest.raises(wrapper.PlanError, match="epoch"):
        wrapper.validate_plan(
            "smartctl",
            {
                "tool": "smartctl",
                "not_after": 0,
                "rules": [{"query": "scan", "action": "passthrough"}],
            },
        )


def test_match_query_recognises_only_the_product_queries() -> None:
    assert wrapper.match_query("smartctl", ["--scan-open"]) == "scan"
    assert wrapper.match_query("smartctl", ["-H", "-j", "/dev/nvme0"]) == "health"
    assert wrapper.match_query("smartctl", ["-H", "/dev/nvme0"]) is None, (
        "a non-JSON health query is not the product's and passes straight through"
    )
    assert wrapper.match_query("ethtool", ["-S", "eth0"]) == "stats"
    assert wrapper.match_query("ethtool", ["-S", "eth0", "--extra"]) is None
    assert (
        wrapper.match_query(
            "nvidia-smi",
            list(wrapper.TOOLS["nvidia-smi"]["queries"]["utilization"]["argv"]),
        )
        == "utilization"
    )
    assert (
        wrapper.match_query(
            "nvidia-smi", ["--query-gpu=index,uuid,name", "--format=csv,noheader"]
        )
        is None
    ), "product discovery is never intercepted"
    assert wrapper.match_query("nvidia-smi", ["topo", "-m"]) == "topology"
    assert wrapper.match_query("ipmitool", ["sensor"]) == "sensor"
    assert wrapper.match_query("ipmitool", ["sdr"]) is None


# --------------------------------------------------------------------------- #
# pure rewriting
# --------------------------------------------------------------------------- #
def test_apply_rule_edits_only_the_named_field() -> None:
    payload = {
        "smartctl": {"exit_status": 0},
        "smart_status": {"passed": True},
        "model_name": "X",
    }
    out, code, _err, reason = wrapper.apply_rule(
        {
            "query": "health",
            "action": "set_json",
            "field": "smart_status.passed",
            "value": False,
        },
        json.dumps(payload),
        0,
        "",
    )
    assert reason is None and code == 0, reason
    served = json.loads(out)
    assert served["smart_status"]["passed"] is False and served["model_name"] == "X", (
        served
    )
    out, _code, _err, reason = wrapper.apply_rule(
        {"query": "health", "action": "delete_json", "field": "smart_status"},
        json.dumps(payload),
        0,
        "",
    )
    assert "smart_status" not in json.loads(out) and reason is None
    _out, _code, _err, reason = wrapper.apply_rule(
        {
            "query": "health",
            "action": "set_json",
            "field": "smart_status.passed",
            "value": False,
        },
        "not json",
        0,
        "",
    )
    assert reason == "the real output is not JSON"
    out, code, _err, reason = wrapper.apply_rule(
        {
            "query": "health",
            "action": "set_json",
            "field": "smart_status.passed",
            "value": False,
        },
        json.dumps(payload),
        2,
        "failed",
    )
    assert reason is not None and code == 2 and json.loads(out) == payload, (
        "a failed real query is passed through as it failed"
    )


def test_apply_rule_counters_and_rows() -> None:
    stats = "NIC statistics:\n     rx_drops: 7\n     rnr_nak_retry_err: 1\n"
    out, _code, _err, reason = wrapper.apply_rule(
        {
            "query": "stats",
            "action": "set_counter",
            "counter": "rx_drops",
            "value": "5000",
        },
        stats,
        0,
        "",
    )
    assert (
        reason is None
        and "     rx_drops: 5000\n" in out
        and "rnr_nak_retry_err: 1" in out
    ), out
    out, _code, _err, _reason = wrapper.apply_rule(
        {"query": "stats", "action": "delete_counter", "counter": "rx_drops"},
        stats,
        0,
        "",
    )
    assert "rx_drops" not in out and "rnr_nak_retry_err: 1" in out, out
    out, _code, _err, _reason = wrapper.apply_rule(
        {
            "query": "stats",
            "action": "set_counter",
            "counter": "tx_drops",
            "value": "3",
        },
        stats,
        0,
        "",
    )
    assert out.endswith("     tx_drops: 3\n"), "a counter the netdev lacks is appended"
    rows = f"{UUID_A}, 0\n{UUID_B}, 97\n"
    out, _code, _err, reason = wrapper.apply_rule(
        {"query": "utilization", "action": "drop_row", "uuid": UUID_A}, rows, 0, ""
    )
    assert reason is None and out == f"{UUID_B}, 97\n", out
    out, _code, _err, reason = wrapper.apply_rule(
        {
            "query": "utilization",
            "action": "set_column",
            "uuid": UUID_B,
            "column": "utilization.gpu",
            "value": "0",
        },
        rows,
        0,
        "",
    )
    assert reason is None and out == f"{UUID_A}, 0\n{UUID_B}, 0\n", out
    _out, _code, _err, reason = wrapper.apply_rule(
        {
            "query": "utilization",
            "action": "drop_row",
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
        },
        rows,
        0,
        "",
    )
    assert reason is not None and reason.startswith("no row for"), reason
    out, code, err, reason = wrapper.apply_rule(
        {
            "query": "scan",
            "action": "replace_output",
            "stdout": "# scan_smart_devices: nope\n",
            "exit_code": 0,
        },
        "/dev/nvme0 -d nvme\n",
        0,
        "",
    )
    assert (out, code, err, reason) == ("# scan_smart_devices: nope\n", 0, "", None)
    out, code, err, reason = wrapper.apply_rule(
        {"query": "scan", "action": "exit", "exit_code": 4, "stderr": "no"}, "x", 0, ""
    )
    assert (out, code, err, reason) == ("", 4, "no", None)


# --------------------------------------------------------------------------- #
# the wrapper process
# --------------------------------------------------------------------------- #
def _stub(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n" + body, encoding="utf-8"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _install(
    tmp_path: Path, tool: str, plan: dict | None, *, real_body: str | None
) -> dict[str, Path]:
    """A run dir with the wrapper, a shim and (optionally) a stub real binary."""

    run_dir = tmp_path / "run"
    fixtures = run_dir / "fixtures"
    fixtures.mkdir(parents=True)
    installed = fixtures / "host_command_replay.py"
    installed.write_bytes(TOOL.read_bytes())
    if plan is not None:
        (fixtures / f"plan-{tool}.json").write_text(json.dumps(plan), encoding="utf-8")
    shim_dir = run_dir / "bin"
    shim_dir.mkdir()
    shim = shim_dir / tool
    shim.write_text(
        "#!/bin/sh\n"
        f"{wrapper.TOOL_ENV}={tool} {wrapper.DIR_ENV}={run_dir} "
        f'exec {sys.executable} -I {installed} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    real_dir = tmp_path / "realbin"
    real_dir.mkdir()
    if real_body is not None:
        _stub(real_dir, tool, real_body)
    return {
        "run_dir": run_dir,
        "shim_dir": shim_dir,
        "real_dir": real_dir,
        "argv_log": tmp_path / "argv.json",
    }


def _call(
    paths: dict[str, Path], tool: str, *argv: str
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{paths['shim_dir']}{os.pathsep}{paths['real_dir']}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "GFRS_TEST_ARGV_LOG": str(paths["argv_log"]),
    }
    return subprocess.run(
        [str(paths["shim_dir"] / tool), *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


REAL_SMARTCTL = """
open(os.environ["GFRS_TEST_ARGV_LOG"], "w").write(json.dumps(sys.argv[1:]))
if sys.argv[1:] == ["--scan-open"]:
    print("/dev/nvme0 -d nvme # /dev/nvme0, NVMe device")
    sys.exit(0)
if sys.argv[1:3] == ["-H", "-j"]:
    print(json.dumps({"smartctl": {"exit_status": 0}, "smart_status": {"passed": True}, "device": {"name": sys.argv[3]}}))
    sys.exit(0)
print("REAL OTHER")
sys.exit(0)
"""


def _audit(paths: dict[str, Path], tool: str) -> list[dict]:
    path = paths["run_dir"] / "audit" / f"{tool}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_wrapper_passes_uncatalogued_calls_straight_to_the_real_binary(
    tmp_path: Path,
) -> None:
    plan = _plan("smartctl", {"query": "health", "action": "exit", "exit_code": 2})
    paths = _install(tmp_path, "smartctl", plan, real_body=REAL_SMARTCTL)
    completed = _call(paths, "smartctl", "--version")
    assert completed.returncode == 0 and completed.stdout == "REAL OTHER\n", completed
    assert json.loads(paths["argv_log"].read_text()) == ["--version"], (
        "every flag reaches the real binary"
    )
    assert _audit(paths, "smartctl") == [], "an uncatalogued call is not journaled"


def test_wrapper_rewrites_only_the_planned_json_field(tmp_path: Path) -> None:
    plan = _plan(
        "smartctl",
        {
            "query": "health",
            "device": "/dev/nvme0",
            "action": "set_json",
            "field": "smart_status.passed",
            "value": False,
        },
    )
    paths = _install(tmp_path, "smartctl", plan, real_body=REAL_SMARTCTL)
    served = _call(paths, "smartctl", "-H", "-j", "/dev/nvme0")
    assert served.returncode == 0, served.stderr
    payload = json.loads(served.stdout)
    assert (
        payload["smart_status"]["passed"] is False
        and payload["device"]["name"] == "/dev/nvme0"
    ), payload
    other = _call(paths, "smartctl", "-H", "-j", "/dev/nvme1")
    assert json.loads(other.stdout)["smart_status"]["passed"] is True, (
        "another device passes through"
    )
    scan = _call(paths, "smartctl", "--scan-open")
    assert scan.stdout.startswith("/dev/nvme0 -d nvme"), (
        "the scan has no rule and passes through"
    )
    entries = _audit(paths, "smartctl")
    assert [item["query"] for item in entries] == ["health", "health", "scan"], entries
    assert entries[0]["applied_rules"] == [0] and entries[1]["applied_rules"] == [], (
        entries
    )
    assert all(
        item["proof_scope"] == "live-replay"
        and item["physical_fault_injected"] is False
        for item in entries
    ), "every journal entry claims only the live-replay scope"


def test_wrapper_honours_max_calls_and_deadline(tmp_path: Path) -> None:
    plan = _plan(
        "smartctl",
        {
            "query": "scan",
            "action": "exit",
            "exit_code": 3,
            "stderr": "scan down",
            "max_calls": 1,
        },
    )
    paths = _install(tmp_path, "smartctl", plan, real_body=REAL_SMARTCTL)
    first = _call(paths, "smartctl", "--scan-open")
    assert (first.returncode, first.stderr) == (3, "scan down"), first
    second = _call(paths, "smartctl", "--scan-open")
    assert second.returncode == 0 and second.stdout.startswith("/dev/nvme0"), (
        "the bounded rule applies once"
    )
    calls = json.loads((paths["run_dir"] / "audit" / "calls.json").read_text())
    assert calls == {"smartctl/0": 1}, calls
    expired = dict(plan, not_after=time.time() - 1)
    (paths["run_dir"] / "fixtures" / "plan-smartctl.json").write_text(
        json.dumps(expired)
    )
    (paths["run_dir"] / "audit" / "calls.json").unlink()
    third = _call(paths, "smartctl", "--scan-open")
    assert third.returncode == 0 and third.stdout.startswith("/dev/nvme0"), (
        "past the deadline everything passes through"
    )
    assert "deadline" in " ".join(_audit(paths, "smartctl")[-1]["unchanged_reason"])


@pytest.mark.allows_cluster_binaries("nvidia-smi")
def test_wrapper_never_invents_success_for_a_failed_real_query(tmp_path: Path) -> None:
    failing = """
print("Unable to determine the device handle for GPU 0000:53:00.0", file=sys.stderr)
sys.exit(15)
"""
    plan = _plan(
        "nvidia-smi", {"query": "utilization", "action": "drop_row", "uuid": UUID_A}
    )
    paths = _install(tmp_path, "nvidia-smi", plan, real_body=failing)
    served = _call(
        paths,
        "nvidia-smi",
        *wrapper.TOOLS["nvidia-smi"]["queries"]["utilization"]["argv"],
    )
    assert served.returncode == 15 and "device handle" in served.stderr, served
    entry = _audit(paths, "nvidia-smi")[-1]
    assert entry["applied_rules"] == [] and any(
        "failed" in item for item in entry["unchanged_reason"]
    ), entry


@pytest.mark.allows_cluster_binaries("nvidia-smi")
def test_wrapper_drops_one_gpu_row_once(tmp_path: Path) -> None:
    body = f"""
print("{UUID_A}, 0")
print("{UUID_B}, 42")
"""
    plan = _plan(
        "nvidia-smi",
        {"query": "utilization", "action": "drop_row", "uuid": UUID_A, "max_calls": 1},
    )
    paths = _install(tmp_path, "nvidia-smi", plan, real_body=body)
    argv = wrapper.TOOLS["nvidia-smi"]["queries"]["utilization"]["argv"]
    first = _call(paths, "nvidia-smi", *argv)
    assert first.stdout == f"{UUID_B}, 42\n", first
    second = _call(paths, "nvidia-smi", *argv)
    assert second.stdout == f"{UUID_A}, 0\n{UUID_B}, 42\n", (
        "the second sample is the real one"
    )
    discovery = _call(
        paths, "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"
    )
    assert (
        discovery.returncode == 0 and discovery.stdout == f"{UUID_A}, 0\n{UUID_B}, 42\n"
    ), "a discovery query is not the catalogued one and is forwarded untouched"


def test_wrapper_serves_reviewed_sensor_table_without_a_real_ipmitool(
    tmp_path: Path,
) -> None:
    table = "CPU Temp         | 45.000     | degrees C  | cr    | na | na | 0 | 90 | 95 | na\n"
    plan = _plan(
        "ipmitool", {"query": "sensor", "action": "replace_output", "stdout": table}
    )
    paths = _install(tmp_path, "ipmitool", plan, real_body=None)
    served = _call(paths, "ipmitool", "sensor")
    assert (served.returncode, served.stdout) == (0, table), served
    other = _call(paths, "ipmitool", "sdr")
    assert other.returncode == 127 and "command not found" in other.stderr, (
        "without a real binary every other call answers as a missing tool"
    )
    edit_plan = _plan(
        "ethtool",
        {
            "query": "stats",
            "action": "set_counter",
            "counter": "rx_drops",
            "value": "1",
        },
    )
    edit_paths = _install(tmp_path / "e", "ethtool", edit_plan, real_body=None)
    edited = _call(edit_paths, "ethtool", "-S", "eth0")
    assert edited.returncode == 127, (
        "an edit rule never fabricates output when there is no real binary"
    )


def test_wrapper_edits_an_ethtool_counter_and_applies_a_bounded_delay(
    tmp_path: Path,
) -> None:
    body = """
print("NIC statistics:")
print("     rx_drops: 7")
print("     rnr_nak_retry_err: 2")
"""
    plan = _plan(
        "ethtool",
        {
            "query": "stats",
            "interface": "rdmap79s0",
            "action": "set_counter",
            "counter": "rx_drops",
            "value": "5000",
        },
        {
            "query": "stats",
            "interface": "rdmap79s0",
            "action": "passthrough",
            "delay_seconds": 1,
        },
    )
    paths = _install(tmp_path, "ethtool", plan, real_body=body)
    started = time.monotonic()
    served = _call(paths, "ethtool", "-S", "rdmap79s0")
    assert time.monotonic() - started >= 1.0, "the planned delay is applied"
    assert (
        served.stdout
        == "NIC statistics:\n     rx_drops: 5000\n     rnr_nak_retry_err: 2\n"
    ), served
    other = _call(paths, "ethtool", "-S", "rdmap80s0")
    assert "rx_drops: 7" in other.stdout, "another interface passes through"


def test_wrapper_refuses_to_run_unconfigured_and_ignores_replay_root_binaries(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", str(TOOL), "--scan-open"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert completed.returncode == 125 and "not configured" in completed.stderr, (
        completed
    )
    fake_root = tmp_path / "run" / "gfrs-live" / "x" / "bin"
    assert (
        wrapper.real_binary("smartctl", f"{fake_root}{os.pathsep}/nonexistent") is None
    ), "a candidate below the replay root or absent is never the real binary"
    real_dir = tmp_path / "real"
    _stub(real_dir, "smartctl", "print('x')\n")
    assert wrapper.real_binary(
        "smartctl",
        f"{tmp_path / 'own'}{os.pathsep}{real_dir}",
        own_dir=str(tmp_path / "own"),
    ) == str(real_dir / "smartctl")
