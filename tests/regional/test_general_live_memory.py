"""GF-GENERAL-LIVE-002 injector/verdict contract over scripted product records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.host_diagnostics import diagnostic_probe
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from scripts.e2e.regional import general_live_memory as mem
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests._builders import build_store, host_telemetry_batch
from tests.collectors import _cov95_runtime_collect as cov_support
from tests.collectors import _cov95_runtime_collect_host as cov_host
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    make_settings,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in mem.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey_response(swap_total: int = 0) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"systemd-run": True, "systemctl": True},
        "modules": {},
        "backing": [],
        "memory": {"MemTotal": 2_000_000_000_000, "SwapTotal": swap_total},
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def oracle_records(value: float, count: int = 40) -> dict[str, Any]:
    records = [{"memory_used_percent": value} for _ in range(count)]
    return {"records": records, "next_offset": count, "total": count}


def mem_node_probe(used: float = 97.0) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_response(),
            "memory-start": {"unit": "gfrs-x-memory"},
            "memory-stop": {"stopped": {"ActiveState": "inactive"}},
            "oracle-start": {"unit": "gfrs-x-oracle"},
            "oracle-stop": {"stopped": {}},
            "oracle-read": [
                oracle_records(used),
                {"records": [], "next_offset": 40, "total": 40},
            ],
        }
    )


def chain_activity(metric: str = "memory_used_percent") -> dict[str, Any]:
    inc = incident("inc-1", metric, state="ACTION_PENDING")
    wf = workflow(
        "wf-1",
        "inc-1",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric=metric,
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    batches = [host_batch("2026-10-24T00:02:00+00:00", metric, 99.0)]
    batches += [
        host_batch(f"2026-10-24T00:2{index}:00+00:00", metric, 50.0)
        for index in range(3)
    ]
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": batches,
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def test_memory_pressure_passes_the_diagnostic_chain(tmp_path: Path) -> None:
    spec = SPECS["memory_pressure"]
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert (ctx.case_dir / "oracle.jsonl").is_file(), (
        "(ctx.case_dir / 'oracle.jsonl').is_file()"
    )


def test_memory_pressure_needs_the_approval(tmp_path: Path) -> None:
    spec = SPECS["memory_pressure"]
    settings = make_settings(spec, node=NODE)
    object.__setattr__(settings, "approvals", ())
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
        settings=settings,
    )
    with pytest.raises(VariantNotRun, match="HOST_MEMORY_PRESSURE"):
        spec.execute(ctx)


def test_memory_pressure_is_inconclusive_when_the_oracle_never_crossed(
    tmp_path: Path,
) -> None:
    spec = SPECS["memory_pressure"]
    probe = mem_node_probe()
    probe.script["oracle-read"] = [oracle_records(70.0)]
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    # Live 2026-10-24: a 2 TiB host was still at 76 % when detection timed
    # out; the ramp is now waited for (and named) before detection is timed.
    with pytest.raises(VariantInconclusive, match="did not raise memory_used"):
        spec.execute(ctx)


def test_memory_pressure_fails_when_a_hardware_step_is_planned(tmp_path: Path) -> None:
    spec = SPECS["memory_pressure"]
    document = chain_activity()
    document["workflows"][0]["official_steps"].append(
        {"operation": "RESET_GPU", "parameters": {}}
    )
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("RESET_GPU" in error for error in result["errors"]), result["errors"]


def test_memory_pressure_fails_when_validate_host_succeeded_while_loaded(
    tmp_path: Path,
) -> None:
    spec = SPECS["memory_pressure"]
    document = chain_activity()
    document["workflows"][0]["step_executions"] = [
        {"operation": "VALIDATE_HOST", "status": "SUCCEEDED", "details": {}}
    ]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("succeeded while the fault" in error for error in result["errors"]), (
        result["errors"]
    )


def test_recovery_passes_and_settles(tmp_path: Path) -> None:
    spec = SPECS["recovery"]
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": mem_node_probe()},
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_no_swap_passes_on_a_swapless_node(tmp_path: Path) -> None:
    spec = SPECS["no_swap"]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [host_batch("2026-10-24T00:01:00+00:00", "swap_used_percent", 0.0)],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [],
    }
    probe = FakeProbe({"survey": survey_response(swap_total=0)})
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_no_swap_blocks_when_the_node_has_swap(tmp_path: Path) -> None:
    spec = SPECS["no_swap"]
    probe = FakeProbe({"survey": survey_response(swap_total=8_000_000_000)})
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    with pytest.raises(VariantBlocked, match="swapless"):
        spec.execute(ctx)


# --------------------------------------------------------------------------- #
# cgroup_only: an OOM inside a small test cgroup is not host pressure
# --------------------------------------------------------------------------- #
def oom_status(
    result: str = "oom-kill", state: str = "failed", used: float = 60.0
) -> dict[str, Any]:
    return {
        "unit": {
            "ActiveState": state,
            "Result": result,
            "ExecMainCode": "killed",
            "ExecMainStatus": "9",
        },
        "memory": {"memory_used_percent": used},
    }


def cgroup_node_probe(
    statuses: list[dict[str, Any]] | None = None, used: float = 60.0
) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_response(),
            "memory-start": {
                "unit": "gfrs-x-memory",
                "memory_max_bytes": 2 << 30,
                "cgroup_oom": True,
            },
            "memory-status": statuses or [oom_status("", "active"), oom_status()],
            "memory-stop": {"stopped": {"ActiveState": "inactive"}},
            "oracle-start": {"unit": "gfrs-x-oracle"},
            "oracle-stop": {"stopped": {}},
            "oracle-read": [
                oracle_records(used),
                {"records": [], "next_offset": 40, "total": 40},
            ],
        }
    )


def quiet_activity(
    used: float = 60.0,
    incidents: list[dict[str, Any]] | None = None,
    workflows: list[dict[str, Any]] | None = None,
    batches: int = 3,
) -> dict[str, Any]:
    return {
        "incidents": incidents or [],
        "workflows": workflows or [],
        "batches": [
            host_batch(f"2026-10-24T00:0{index}:00+00:00", "memory_used_percent", used)
            for index in range(batches)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "metric_latest": [],
    }


def log_incident(action: str, workflow_id: str = "wf-log") -> dict[str, Any]:
    return {
        "incident_id": "inc-log",
        "event_id": "log-oom-abc",
        "event_type": "NODE_HEALTH",
        "state": "ACTION_PENDING",
        "effective_action": action,
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow_id,
        "node_ids": [NODE],
    }


def cgroup_context(
    tmp_path: Path, document: dict[str, Any], probe: FakeProbe | None = None
) -> Any:
    spec = SPECS["cgroup_only"]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": probe or cgroup_node_probe()},
    )


def test_cgroup_only_passes_without_a_memory_finding(tmp_path: Path) -> None:
    probe = cgroup_node_probe()
    ctx = cgroup_context(tmp_path, quiet_activity(), probe)
    result = SPECS["cgroup_only"].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    starts = [call for call in probe.calls if call[0] == "memory-start"]
    assert len(starts) == mem.CGROUP_OOM_REPEATS
    assert all(
        "--cgroup-oom-bytes" in call and "--approved" in call for call in starts
    ), starts
    assert all(
        call[call.index("--cgroup-oom-bytes") + 1] == str(mem.CGROUP_OOM_CAP_BYTES)
        for call in starts
    ), "all((call[call.index('--cgroup-oom-bytes') + 1] == str(mem.CGROUP_OOM_"
    assert "--target-used-percent" not in starts[0]
    stops = [call for call in probe.calls if call[0] == "memory-stop"]
    assert len(stops) == mem.CGROUP_OOM_REPEATS, "every OOM-killed unit is reset"
    assert "memory" not in ctx.cleanup.pending()
    assert len(result["cgroup_oom_kills"]) == mem.CGROUP_OOM_REPEATS
    assert all(item["result"] == "oom-kill" for item in result["cgroup_oom_kills"]), (
        "all((item['result'] == 'oom-kill' for item in result['cgroup_oom_kills"
    )
    assert result["host_used_percent"]["max"] < mem.USED_THRESHOLD
    assert round(result["cgroup_cap_percent_of_host"], 2) == 0.11
    assert (ctx.case_dir / "oracle.jsonl").is_file(), "oracle evidence written"


def test_cgroup_only_needs_the_approval(tmp_path: Path) -> None:
    spec = SPECS["cgroup_only"]
    settings = make_settings(spec, node=NODE)
    object.__setattr__(settings, "approvals", ())
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": quiet_activity()})),
        probes={f"node@{NODE}": cgroup_node_probe()},
        settings=settings,
    )
    with pytest.raises(VariantNotRun, match="HOST_MEMORY_PRESSURE"):
        spec.execute(ctx)


def test_cgroup_only_is_inconclusive_when_the_unit_was_not_oom_killed(
    tmp_path: Path,
) -> None:
    probe = cgroup_node_probe(statuses=[oom_status("success", "inactive")])
    ctx = cgroup_context(tmp_path, quiet_activity(), probe)
    with pytest.raises(VariantInconclusive, match="instead of oom-kill"):
        SPECS["cgroup_only"].execute(ctx)
    assert [call[0] for call in probe.calls if call[0] == "memory-stop"] == [
        "memory-stop"
    ], "the unit is reset before the verdict is abandoned"
    assert "memory" not in ctx.cleanup.pending()


def test_cgroup_only_is_inconclusive_when_the_unit_never_ended(tmp_path: Path) -> None:
    probe = cgroup_node_probe(statuses=[oom_status("", "active")])
    ctx = cgroup_context(tmp_path, quiet_activity(), probe)
    with pytest.raises(VariantInconclusive, match="still running"):
        SPECS["cgroup_only"].execute(ctx)


def test_cgroup_only_is_inconclusive_when_the_host_itself_was_under_pressure(
    tmp_path: Path,
) -> None:
    ctx = cgroup_context(tmp_path, quiet_activity(), cgroup_node_probe(used=96.0))
    with pytest.raises(VariantInconclusive, match="under pressure"):
        SPECS["cgroup_only"].execute(ctx)


def test_cgroup_only_is_inconclusive_without_a_fresh_host_sample(
    tmp_path: Path,
) -> None:
    ctx = cgroup_context(tmp_path, quiet_activity(batches=0))
    with pytest.raises(VariantInconclusive, match="silent channel"):
        SPECS["cgroup_only"].execute(ctx)


def test_cgroup_only_fails_when_the_product_minted_a_memory_finding(
    tmp_path: Path,
) -> None:
    ctx = cgroup_context(tmp_path, chain_activity())
    result = SPECS["cgroup_only"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "unexpected memory_used_percent incident" in e for e in result["errors"]
    ), result["errors"]


def test_cgroup_only_fails_when_an_oom_log_line_isolated_the_node(
    tmp_path: Path,
) -> None:
    document = quiet_activity(
        incidents=[log_incident("QUARANTINE")],
        workflows=[
            workflow(
                "wf-log",
                "inc-log",
                ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
            )
        ],
    )
    result = SPECS["cgroup_only"].execute(cgroup_context(tmp_path, document))
    assert result["verdict"] == "FAIL"
    assert any("graded QUARANTINE" in e for e in result["errors"]), result["errors"]
    assert any("isolation operations" in e for e in result["errors"]), result["errors"]


def test_cgroup_only_accepts_a_diagnostic_only_oom_log_finding(tmp_path: Path) -> None:
    document = quiet_activity(
        incidents=[log_incident("RUN_DIAGNOSTICS")],
        workflows=[workflow("wf-log", "inc-log", ["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    result = SPECS["cgroup_only"].execute(cgroup_context(tmp_path, document))
    assert result["verdict"] == "PASS", result["errors"]
    stopped = quiet_activity(
        incidents=[log_incident("RUN_DIAGNOSTICS")],
        workflows=[
            workflow("wf-log", "inc-log", ["FREEZE_EVIDENCE", "STOP_WORKLOADS"])
        ],
    )
    result = SPECS["cgroup_only"].execute(cgroup_context(tmp_path, stopped))
    assert result["verdict"] == "FAIL"
    assert any("stopped a workload" in e for e in result["errors"]), result["errors"]


def test_cgroup_only_spec_reuses_the_memory_approval() -> None:
    spec = SPECS["cgroup_only"]
    assert spec.implemented and spec.approvals == ("HOST_MEMORY_PRESSURE",)
    assert spec.lanes == ("eks", "hp") and spec.proof_mode == "native"


# --------------------------------------------------------------------------- #
# the real probe's memory-start behind the fake (live 2026-10-25: the variant
# asked for a 1980 s hold and the probe's 1800 s cap refused it before any
# product evidence, three times on two 2 TiB nodes)
# --------------------------------------------------------------------------- #
def real_memory_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A ``memory-start`` that runs the real probe function over a p5en-shaped host."""

    import subprocess

    from scripts.e2e.regional.probes import general_live_node_probe as nodep

    gib = 1024**3
    monkeypatch.setattr(nodep, "STATE_ROOT", tmp_path / "node-state")
    monkeypatch.setattr(
        nodep,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(nodep, "arm_deadman", lambda run_id, kind, seconds, *a: kind)
    monkeypatch.setattr(
        nodep,
        "meminfo",
        lambda: {"MemTotal": 2000 * gib, "MemAvailable": 1928 * gib, "SwapTotal": 0.0},
    )
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(nodep, "emit", emitted.append)

    def start(arguments: tuple[str, ...]) -> dict[str, Any]:
        parsed = nodep.parser().parse_args(list(arguments))
        parsed.handler(parsed)  # raises ProbeError exactly as on the node
        return dict(emitted[-1])

    return start


@pytest.mark.parametrize("variant", ["memory_pressure", "recovery", "cgroup_only"])
def test_memory_variants_ask_the_real_probe_for_an_accepted_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    probe = mem_node_probe() if variant != "cgroup_only" else cgroup_node_probe()
    probe.script["memory-start"] = real_memory_start(monkeypatch, tmp_path)
    document = chain_activity() if variant != "cgroup_only" else quiet_activity()
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        SPECS[variant],
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": probe},
    )
    result = SPECS[variant].execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    starts = [call for call in probe.calls if call[0] == "memory-start"]
    assert starts, "the variant started the memory unit through the real probe"
    seconds = [int(call[call.index("--seconds") + 1]) for call in starts]
    assert all(value <= mem.PROBE_MAX_SECONDS for value in seconds), seconds
    if variant == "memory_pressure":
        assert seconds == [1980], "the 14-minute ramp plus detection hold"
    if variant == "recovery":
        assert seconds == [1800], "exactly the old cap: recovery never failed live"


def test_memory_pressure_fails_fast_when_the_hold_exceeds_the_probe_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    monkeypatch.setattr(mem, "PROBE_MAX_SECONDS", 1800)  # the live cap of 2026-10-25
    probe = mem_node_probe()
    regional = FakeRegional(cpu_python=store_script({"activity": chain_activity()}))
    ctx = make_context(
        SPECS["memory_pressure"],
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": probe},
    )
    with pytest.raises(
        RegionalFixtureError, match="1980 s memory hold exceeds .* 1800 s"
    ):
        SPECS["memory_pressure"].execute(ctx)
    assert [call[0] for call in probe.calls] == ["survey"], (
        "refused before the oracle or the memory unit touched the node"
    )


def test_low_available_memory_hold_fits_the_probe_cap() -> None:
    from scripts.e2e.regional import general_live_utilization as util

    requested = util.MEMORY_RAMP_BUDGET + (util.MEMORY_DURATION + 300 + 120) + 120
    assert requested <= mem.PROBE_MAX_SECONDS, requested


# --------------------------------------------------------------------------- #
# read_failure / swap_pressure: raw /proc/meminfo replay (HOST_SOURCE_REPLAY)
# --------------------------------------------------------------------------- #

isolated_runtime = cov_support.isolated_runtime
host_case = cov_host.host_case

SWAP_FIXTURE_PERCENT = (
    100.0 * (mem.SWAP_TOTAL_KB - mem.SWAP_FREE_KB) / mem.SWAP_TOTAL_KB
)
MEMINFO_TEXT = (
    "MemTotal:       2113929216 kB\n"
    "MemFree:        2000000000 kB\n"
    "MemAvailable:   2050000000 kB\n"
    "Buffers:           1000000 kB\n"
    "Cached:           40000000 kB\n"
    "SwapCached:              0 kB\n"
    "SwapTotal:               0 kB\n"
    "SwapFree:                0 kB\n"
    "Shmem:              500000 kB\n"
    "SReclaimable:      2000000 kB\n"
)


def _replay_probe(**extra: Any) -> FakeProbe:
    script: dict[str, Any] = {
        "build-meminfo-fixture": {
            "name": "meminfo",
            "path": "/run/gfrs-live/source-x/fixtures/meminfo",
            "swap_used_percent": SWAP_FIXTURE_PERCENT,
            "memory_used_percent": 3.0,
        },
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
            "verification": {
                "ok": True,
                "checks": [
                    {
                        "kind": "inaccessible",
                        "dest": "/proc/meminfo",
                        "unit_readable": False,
                        "host_readable": True,
                        "ok": True,
                    }
                ],
            },
        },
        "status": {"fixtures": {}},
        "close-window": {"clean": True},
    }
    script.update(extra)
    return FakeProbe(script)


def _replay_ctx(
    variant: str,
    *,
    docs: list[dict[str, Any]],
    tmp_path: Path,
    replay: FakeProbe | None = None,
) -> Any:
    spec = SPECS[variant]
    node = FakeProbe({"survey": survey_response()})
    regional = FakeRegional(cpu_python=store_script({"activity": docs}))
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={
            f"source-replay@{NODE}": replay or _replay_probe(),
            f"node@{NODE}": node,
        },
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )


def _empty_doc(**extra: Any) -> dict[str, Any]:
    return {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "health_signals": {},
        "metric_latest": [],
        "notifications": [],
        "collector_statuses": [],
        **extra,
    }


def _unreadable_batch(
    observed_at: str, samples: list[dict[str, Any]] = ()
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": ["baseline", "collection-error:_memory"],
        "collection_errors": [
            "_memory: PermissionError: [Errno 13] Permission denied: '/proc/meminfo'"
        ],
        "samples": list(samples),
    }


def test_read_failure_is_unknown_and_recovers(tmp_path: Path) -> None:
    window = _empty_doc(batches=[_unreadable_batch("2026-10-24T00:00:20+00:00")])
    recovered = _empty_doc(
        batches=[host_batch("2026-10-24T00:04:00+00:00", "memory_used_percent", 3.0)]
    )
    replay = _replay_probe()
    ctx = _replay_ctx(
        "read_failure", docs=[window, recovered], tmp_path=tmp_path, replay=replay
    )
    result = mem.read_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay" and result["batches_in_window"] == 1
    opened = next(call for call in replay.calls if call[0] == "open-window")
    assert ("--inaccessible", "/proc/meminfo") == tuple(
        opened[opened.index("--inaccessible") : opened.index("--inaccessible") + 2]
    ), opened
    assert "--bind" not in opened and "--wrapper" not in opened, opened
    assert any(call[0] == "close-window" for call in replay.calls), replay.calls


def test_read_failure_fails_on_a_healthy_value_or_a_hardware_step(
    tmp_path: Path,
) -> None:
    sample = {"name": "memory_used_percent", "device": None, "value": 3.0, "labels": {}}
    window = _empty_doc(
        batches=[_unreadable_batch("2026-10-24T00:00:20+00:00", [sample])],
        workflows=[
            workflow(
                "wf-x",
                "inc-x",
                ["MARK_UNSCHEDULABLE", "RESTART_NODE"],
                status="RUNNING",
            )
        ],
    )
    ctx = _replay_ctx("read_failure", docs=[window, window], tmp_path=tmp_path)
    result = mem.read_failure(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert "memory_used_percent=3.0" in joined and "RESTART_NODE" in joined, joined


def test_read_failure_is_inconclusive_when_the_unit_could_still_read(
    tmp_path: Path,
) -> None:
    replay = _replay_probe(
        **{
            "open-window": {
                "after": {"ActiveState": "active"},
                "verification": {
                    "ok": True,
                    "checks": [
                        {"kind": "inaccessible", "dest": "/proc/meminfo", "ok": False}
                    ],
                },
            }
        }
    )
    ctx = _replay_ctx(
        "read_failure", docs=[_empty_doc()], tmp_path=tmp_path, replay=replay
    )
    with pytest.raises(
        VariantInconclusive, match="unreadable inside the host collector unit"
    ):
        mem.read_failure(ctx)


def _swap_docs(*, validation_status: str = "FAILED") -> list[dict[str, Any]]:
    inc = incident("inc-swap", "swap_used_percent", state="RECOVERED")
    inc["reasons"] = [
        "host swap pressure",
        "diagnostic inconclusive: validation failed",
    ]
    executions = [
        {
            "operation": "VALIDATE_HOST",
            "status": validation_status,
            "details": {
                "host_diagnostic_failure": True,
                "node_failures": {NODE: ["host_diagnostic:swap_used_percent"]},
            }
            if validation_status == "FAILED"
            else {},
        }
    ]
    wf = workflow(
        "wf-swap",
        "inc-swap",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status=validation_status,
        probe_metric="swap_used_percent",
        executions=executions,
    )
    detection = _empty_doc(
        incidents=[inc],
        workflows=[wf],
        batches=[
            host_batch(
                "2026-10-24T00:00:20+00:00", "swap_used_percent", SWAP_FIXTURE_PERCENT
            ),
            host_batch("2026-10-24T00:00:20+00:00", "memory_used_percent", 3.0),
        ],
    )
    recovery = _empty_doc(
        incidents=[inc],
        workflows=[wf],
        batches=[
            host_batch("2026-10-24T00:03:00+00:00", "swap_used_percent", 0.0),
            host_batch("2026-10-24T00:03:15+00:00", "swap_used_percent", 0.0),
        ],
    )
    return [detection, detection, recovery]


def test_swap_pressure_passes_the_diagnostic_chain_from_a_raw_meminfo_replay(
    tmp_path: Path,
) -> None:
    replay = _replay_probe()
    ctx = _replay_ctx(
        "swap_pressure", docs=_swap_docs(), tmp_path=tmp_path, replay=replay
    )
    result = mem.swap_pressure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    built = next(call for call in replay.calls if call[0] == "build-meminfo-fixture")
    assert built[1:] == (
        "--source",
        "/proc/meminfo",
        "--set",
        f"SwapTotal={mem.SWAP_TOTAL_KB}",
        "--set",
        f"SwapFree={mem.SWAP_FREE_KB}",
        "--run-id",
        ctx.run_id,
    ), built
    opened = next(call for call in replay.calls if call[0] == "open-window")
    assert opened[1:7] == (
        "--mode",
        "bind",
        "--restore-seconds",
        str(SPECS["swap_pressure"].max_minutes * 60),
        "--bind",
        "meminfo:/proc/meminfo",
    ), opened
    assert result["fixture_swap_used_percent"] == SWAP_FIXTURE_PERCENT
    assert result["recovery"]["errors"] == [] and result["proof_scope"] == "live-replay"


def test_swap_pressure_fails_when_validate_host_succeeded_while_held(
    tmp_path: Path,
) -> None:
    ctx = _replay_ctx(
        "swap_pressure",
        docs=_swap_docs(validation_status="SUCCEEDED"),
        tmp_path=tmp_path,
    )
    result = mem.swap_pressure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "succeeded while the fault was still held" in e for e in result["errors"]
    ), result["errors"]


def test_swap_pressure_is_inconclusive_when_the_fixture_or_samples_do_not_match(
    tmp_path: Path,
) -> None:
    low = _replay_probe(
        **{
            "build-meminfo-fixture": {
                "swap_used_percent": 50.0,
                "memory_used_percent": 3.0,
            }
        }
    )
    ctx = _replay_ctx("swap_pressure", docs=_swap_docs(), tmp_path=tmp_path, replay=low)
    with pytest.raises(VariantInconclusive, match="below the 80.0% rule"):
        mem.swap_pressure(ctx)
    docs = _swap_docs()
    for doc in docs:
        for batch in doc["batches"]:
            for sample in batch["samples"]:
                if sample["name"] == "swap_used_percent" and sample["value"] > 80:
                    sample["value"] = 99.0
    ctx = _replay_ctx("swap_pressure", docs=docs, tmp_path=tmp_path)
    with pytest.raises(VariantInconclusive, match="never matched the fixture"):
        mem.swap_pressure(ctx)


def test_read_failure_and_swap_pressure_specs() -> None:
    for name in ("read_failure", "swap_pressure"):
        spec = SPECS[name]
        assert spec.implemented and spec.proof_mode == "live-replay"
        assert spec.lanes == ("eks", "hp") and spec.approvals == ("HOST_SOURCE_REPLAY",)
    assert mem.SWAP_THRESHOLD == NodeHealthPolicy.METRIC_RULES["swap_used_percent"][0]


def test_swap_rule_is_pinned_against_the_product() -> None:
    threshold, category, severity, action, _reason = NodeHealthPolicy.METRIC_RULES[
        "swap_used_percent"
    ]
    assert (threshold, severity, action) == (
        80.0,
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
    )
    assert category.value == "MEMORY"
    assert "swap_used_percent" not in NodeHealthPolicy.METRIC_RULE_SUSTAIN_SECONDS, (
        "swap has no sustain: the first delivered breach mints the finding"
    )
    batch = host_telemetry_batch(
        "swap-1",
        datetime(2026, 10, 24, tzinfo=timezone.utc),
        [
            HostMetricSample(
                name="swap_used_percent", value=SWAP_FIXTURE_PERCENT, unit="percent"
            )
        ],
        runtime_profile_version="simulated-v1",
    )
    (finding,) = NodeHealthPolicy(build_store()).evaluate_metrics(batch)
    assert finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS
    probe_parameters = diagnostic_probe(finding)
    assert (
        probe_parameters is not None
        and probe_parameters["metric"] == "swap_used_percent"
    )
    assert (
        probe_parameters["threshold"] == 80.0
        and probe_parameters["comparison"] == "gte"
    )


def test_the_real_memory_contributor_reads_the_probe_built_meminfo(
    host_case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``build-meminfo-fixture`` output -> ``system_metrics._memory`` -> swap 87.5 %."""

    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    source = tmp_path / "meminfo.real"
    source.write_text(MEMINFO_TEXT)
    code = probe.main(
        [
            "build-meminfo-fixture",
            "--run-id",
            "gl002-swap-pressure-a1-deadbeef",
            "--source",
            str(source),
            "--set",
            f"SwapTotal={mem.SWAP_TOTAL_KB}",
            "--set",
            f"SwapFree={mem.SWAP_FREE_KB}",
        ]
    )
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0, receipt
    assert receipt["swap_used_percent"] == SWAP_FIXTURE_PERCENT
    fixture = Path(receipt["path"]).read_text()
    assert "SwapTotal:" in fixture and fixture.count("\n") == MEMINFO_TEXT.count("\n")
    cov_host.write_file(host_case.roots.proc / "meminfo", fixture)
    batch = host_case.build("_memory").collect_once()
    values = cov_host.values(batch)
    assert values[("swap_used_percent", None)] == SWAP_FIXTURE_PERCENT, values
    assert (
        abs(values[("memory_used_percent", None)] - receipt["memory_used_percent"])
        < 1e-6
    )
    assert values[("memory_used_percent", None)] < mem.USED_THRESHOLD
    graded = NodeHealthPolicy(build_store()).evaluate_metrics(batch)
    assert [item.metric_name for item in graded] == ["swap_used_percent"], graded
