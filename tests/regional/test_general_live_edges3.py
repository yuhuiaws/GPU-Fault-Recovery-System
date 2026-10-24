"""Direct unit coverage of collector/utilization/storage/runner helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_collectors as col
from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional import general_live_utilization as util
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

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
EMPTY = {
    "incidents": [],
    "workflows": [],
    "batches": [],
    "markers": [],
    "commands": [],
    "health_signals": {},
    "notifications": [],
    "collector_statuses": [],
    "metric_latest": [],
}


# --------------------------------------------------------------------------- #
# collectors helpers
# --------------------------------------------------------------------------- #
def _obs(channel: str, state: str) -> dict[str, Any]:
    return {
        "name": "collector_observation_valid",
        "device": channel,
        "value": 1 if state == "HEALTHY" else 0,
        "labels": {"state": state, "reason": "x"},
    }


def test_collector_observation_helpers() -> None:
    assert col._observation_unknown({"metric_latest": []}, "GPU_METRICS")
    assert col._observation_unknown(
        {"metric_latest": [_obs("GPU_METRICS", "HEALTHY")]}, "GPU_METRICS"
    )
    assert (
        col._observation_unknown(
            {"metric_latest": [_obs("GPU_METRICS", "UNKNOWN")]}, "GPU_METRICS"
        )
        == []
    )
    assert col._snapshot_only({"workflows": [workflow("w", "i", ["RESET_GPU"])]})
    assert (
        col._snapshot_only(
            {"workflows": [workflow("w", "i", ["TRIGGER_HEALTH_SNAPSHOT"])]}
        )
        == []
    )
    many = {
        "workflows": [
            workflow("a", "i", ["TRIGGER_HEALTH_SNAPSHOT"]),
            workflow("b", "i", ["TRIGGER_HEALTH_SNAPSHOT"]),
        ]
    }
    assert col._one_snapshot_per_hour(many)
    assert (
        col._one_snapshot_per_hour(
            {"workflows": [workflow("a", "i", ["TRIGGER_HEALTH_SNAPSHOT"])]}
        )
        == []
    )
    assert col._silent_advisory({"notifications": []}, "GPU_METRICS")
    assert (
        col._silent_advisory(
            {
                "notifications": [
                    {"deduplication_key": "collector-silent/c/n/GPU_METRICS/1"}
                ]
            },
            "GPU_METRICS",
        )
        == []
    )


def test_collector_cleared_error_and_channel_success(tmp_path: Path) -> None:
    doc = {
        "collector_statuses": [
            {
                "collector": "GPU_METRICS",
                "last_error_at": "2026-10-24T02:00:00+00:00",
                "last_success_at": "2026-10-24T01:00:00+00:00",
            }
        ]
    }

    class Ctx:
        def __init__(self) -> None:
            self.result = doc

    # _cleared_error reads activity() -> patch activity on the module
    spec = next(s for s in col.VARIANTS if s.variant == "exporter_timeout")
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    assert col._cleared_error(
        ctx,
        __import__("datetime").datetime(
            2026, 10, 24, tzinfo=__import__("datetime").timezone.utc
        ),
    )


def test_post_refresh_without_sustained_error_fails(tmp_path: Path) -> None:
    spec = next(s for s in col.VARIANTS if s.variant == "post_refresh_success")
    survey = {
        "cpu_count": 8,
        "tools": {},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "exporter_processes": [{"pid": 1, "start_ticks": "5"}],
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }
    node = FakeProbe(
        {"survey": survey, "exporter-pause": {"stopped": True}, "exporter-resume": {}}
    )
    fresh = {
        **EMPTY,
        "metric_latest": [_obs("GPU_METRICS", "HEALTHY")],
        "collector_statuses": [
            {
                "collector": "GPU_METRICS",
                "last_success_at": "2026-10-24T05:00:00+00:00",
                "last_error_at": None,
            }
        ],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": fresh}))
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node},
        settings=settings,
    )
    result = spec.execute(ctx)
    # No sustained collection error was ever observed -> the no_stale_pass stage fails.
    assert result["verdict"] == "FAIL"
    assert any("no sustained collection error" in e for e in result["errors"])


# --------------------------------------------------------------------------- #
# utilization helpers and low_available_memory inconclusive
# --------------------------------------------------------------------------- #
def test_utilization_helpers() -> None:
    assert util._attribution_errors([], {}, "a1") == [
        "no low-utilisation finding reached an incident"
    ]
    assert util._attribution_errors([{"attempt_id": "other"}], {}, "a1")
    assert util._attribution_errors([{"attempt_id": None}], {}, "a1") == []
    hot = {"batches": [host_batch("t", "memory_used_percent", 96.0)]}
    assert util._memory_used_low(hot)
    cool = {"batches": [host_batch("t", "memory_used_percent", 10.0)]}
    assert util._memory_used_low(cool) == []


def test_low_available_memory_inconclusive(tmp_path: Path) -> None:
    spec = next(s for s in util.VARIANTS if s.variant == "low_available_memory")
    survey = {
        "cpu_count": 8,
        "tools": {"systemd-run": True},
        "modules": {},
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }
    node = FakeProbe(
        {
            "survey": survey,
            "memory-start": {},
            "memory-stop": {},
            "oracle-start": {},
            "oracle-stop": {},
            "oracle-read": {
                "records": [{"memory_available_percent": 50.0} for _ in range(20)],
                "next_offset": 20,
                "total": 20,
            },
        }
    )
    inc = incident("m", "memory_available_percent", rule_id="memory")
    doc = {
        **EMPTY,
        "incidents": [inc],
        "workflows": [workflow("w", "inc-memory_available_percent", ["VALIDATE_HOST"])],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(
        **{
            **settings.__dict__,
            "approvals": ("HOST_MEMORY_PRESSURE",),
            "site_file": tmp_path / "s.yaml",
        }
    )
    (tmp_path / "s.yaml").write_text("x\n")
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node},
        settings=settings,
    )
    with pytest.raises(util.VariantInconclusive):
        spec.execute(ctx)


# --------------------------------------------------------------------------- #
# storage helpers
# --------------------------------------------------------------------------- #
def test_storage_await_divergence_and_dm_delay() -> None:
    assert storage._await_divergence([], [1.0]) is False
    assert storage._await_divergence([200.0], [10.0]) is True
    assert storage._await_divergence([11.0], [10.0]) is False


def test_storage_below_threshold_inconclusive(tmp_path: Path) -> None:
    spec = next(s for s in storage.VARIANTS if s.variant == "below_threshold")
    survey = {
        "cpu_count": 8,
        "tools": {n: True for n in ("losetup", "mkfs.ext4", "mount", "fallocate")},
        "modules": {},
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "ext4"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            }
        ],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }
    node = FakeProbe({"survey": survey})
    store = FakeProbe(
        {
            "fs-create": {"mountpoint": "/mnt", "statvfs": {"used_percent": 20.0}},
            "fs-fill": {"statvfs": {"used_percent": 50.0}},
            "fs-destroy": {},
            "fs-unfill": {},
        }
    )
    win = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )
    regional = FakeRegional(cpu_python=store_script({"activity": EMPTY}))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "site_file": tmp_path / "s.yaml"})
    (tmp_path / "s.yaml").write_text("x\n")
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node, f"storage@{NODE}": store, f"window@{NODE}": win},
        settings=settings,
    )
    with pytest.raises(storage.VariantInconclusive):
        spec.execute(ctx)


# --------------------------------------------------------------------------- #
# run_general_live main() plan for a deferred variant and FAIL exit
# --------------------------------------------------------------------------- #
def test_main_plan_deferred_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional import run_general_live as runner

    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda settings: FakeRegional())
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda s, spec, r: {
            "errors": ["needs a fixture"],
            "planned_verdict": "NOT_RUN(FIXTURE_NOT_READY)",
            "release_id": "r",
            "cluster_id": "c",
        },
    )
    monkeypatch.setattr(runner, "build_plan", lambda **kw: {"case_id": kw["case_id"]})
    kc = tmp_path / "kc"
    kc.write_text("apiVersion: v1\n")
    argv = [
        "run_general_live.py",
        "--run-dir",
        str(tmp_path / "run"),
        "--case",
        "GF-GENERAL-LIVE-001",
        "--variant",
        "invalid_delta",
        "--cpu-kubeconfig",
        str(kc),
        "--gpu-kubeconfig",
        str(kc),
        "--gpu-context",
        "c",
        "--cluster-id",
        "x",
        "--region",
        "us-west-2",
        "--fault-node",
        NODE,
        "--host-probe-image",
        "a@sha256:" + "0" * 64,
        "--plan",
    ]
    monkeypatch.setattr("sys.argv", argv)
    # A deferred variant's plan returns 0 even though the preflight has errors.
    assert runner.main() == 0


def test_main_execute_fail_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.e2e.regional import run_general_live as runner

    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda settings: FakeRegional())
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda s, spec, r: {
            "errors": [],
            "planned_verdict": None,
            "release_id": "r",
            "cluster_id": "c",
        },
    )
    monkeypatch.setattr(
        runner,
        "authorize_execution",
        lambda *a, **k: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        + __import__("datetime").timedelta(hours=1),
    )
    monkeypatch.setattr(runner, "execute_variant", lambda *a, **k: {"verdict": "FAIL"})
    kc = tmp_path / "kc"
    kc.write_text("apiVersion: v1\n")
    argv = [
        "run_general_live.py",
        "--run-dir",
        str(tmp_path / "run"),
        "--case",
        "GF-GENERAL-LIVE-001",
        "--variant",
        "cpu_busy",
        "--cpu-kubeconfig",
        str(kc),
        "--gpu-kubeconfig",
        str(kc),
        "--gpu-context",
        "c",
        "--cluster-id",
        "x",
        "--region",
        "us-west-2",
        "--fault-node",
        NODE,
        "--host-probe-image",
        "a@sha256:" + "0" * 64,
        "--execute",
        "--confirm",
        "GENERALLIVE001_EXECUTE",
        "--maintenance-window-end",
        "2026-11-07T00:00:00Z",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert runner.main() == 1
