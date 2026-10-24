"""GF-GENERAL-LIVE-004/006 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    host_batch,
    incident,
    make_context,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in storage.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey(**modules: bool) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            name: True
            for name in (
                "losetup",
                "mkfs.ext4",
                "mount",
                "fallocate",
                "dmsetup",
                "blockdev",
                "systemd-run",
            )
        },
        "modules": {
            "dm_mod": {"loaded": True, "available": True},
            "dm_delay": {"loaded": modules.get("dm_delay", True), "available": True},
        },
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


def storage_probe(**overrides: Any) -> FakeProbe:
    script = {
        "fs-create": {
            "name": "fs",
            "mountpoint": "/mnt",
            "statvfs": {"used_percent": 20.0},
        },
        "fs-fill": {"statvfs": {"used_percent": 99.0}},
        "fs-unfill": {"statvfs": {"used_percent": 20.0}},
        "fs-inodes": {"enospc": True, "created": 256},
        "fs-remount": {"statvfs": {"read_only": True, "used_percent": 20.0}},
        "fs-umount": {"directory_present": False, "statvfs": {"used_percent": None}},
        "fs-mount": {"mounted": "/mnt"},
        "fs-destroy": {"destroyed": True},
        "dm-create": {
            "name": "io",
            "dm": "gfrs-x-io",
            "device": "dm-7",
            "mountpoint": "/mnt",
        },
        "dm-suspend": {"suspended": "gfrs-x-io"},
        "dm-resume": {"resumed": "gfrs-x-io"},
        "dm-recreate": {"device_before": "dm-7", "device_after": "dm-8"},
        "dm-destroy": {"destroyed": True},
        "io-start": {"unit": "gfrs-x-io-io"},
        "io-stop": {"stopped": "gfrs-x-io-io"},
    }
    script.update(overrides)
    return FakeProbe(script)


def window_probe() -> FakeProbe:
    return FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}, "appended": {}},
            "close-window": {"after": {"ActiveState": "active"}},
            "start-unit": {"unit": "gpu-fault-host-collector.service"},
        }
    )


def node_probe(survey_value: dict[str, Any], **extra: Any) -> FakeProbe:
    script = {
        "survey": survey_value,
        "oracle-start": {"unit": "oracle"},
        "oracle-stop": {},
        "oracle-read": {"records": [], "next_offset": 0, "total": 0},
    }
    script.update(extra)
    return FakeProbe(script)


def quarantine_activity(
    mount: str, metric: str = "filesystem_used_percent"
) -> dict[str, Any]:
    inc = incident(
        "inc-fs", metric, device=mount, state="QUARANTINED", workflow="wf-fs"
    )
    wf = workflow(
        "wf-fs",
        "inc-fs",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="SUCCEEDED",
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch("2026-10-24T00:02:00+00:00", metric, 99.0, device=mount)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


def probes(
    ctx_node: FakeProbe, store: FakeProbe, win: FakeProbe
) -> dict[str, FakeProbe]:
    return {f"node@{NODE}": ctx_node, f"storage@{NODE}": store, f"window@{NODE}": win}


def restore_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        storage,
        "restore_incidents",
        lambda ctx, incidents, reason: [{"status": "SUCCEEDED"}],
    )


def test_capacity_full_quarantines_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore_patch(monkeypatch)
    spec = SPECS["capacity_full"]
    regional = FakeRegional(cpu_python=store_script({"activity": None}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    mount = storage._fs_mount(ctx)
    regional.cpu_python_responder = store_script(
        {"activity": quarantine_activity(mount)}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_capacity_full_inconclusive_below_98(tmp_path: Path) -> None:
    spec = SPECS["capacity_full"]
    store = storage_probe(**{"fs-fill": {"statvfs": {"used_percent": 80.0}}})
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), store, window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_below_threshold_passes_without_quarantine(tmp_path: Path) -> None:
    spec = SPECS["below_threshold"]
    store = storage_probe(**{"fs-fill": {"statvfs": {"used_percent": 97.0}}})
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), store, window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


@pytest.mark.parametrize("variant", ["read_only", "required_mount_missing"])
def test_byte_only_gap_is_fail_when_undetected(tmp_path: Path, variant: str) -> None:
    spec = SPECS[variant]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert spec.expected_outcome == "FAIL"
    assert result["verdict"] == "FAIL"
    assert result["detected"] is False


@pytest.mark.parametrize(
    ("fired", "verdict"),
    [
        ("filesystem_inodes_used_percent", "PASS"),
        ("filesystem_used_percent", "FAIL"),
        (None, "FAIL"),
    ],
    ids=["inode-signal", "bytes-signal", "undetected"],
)
def test_inode_full_passes_only_on_the_inode_quarantine_and_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fired: str | None, verdict: str
) -> None:
    restored: list[Any] = []
    monkeypatch.setattr(
        storage,
        "restore_incidents",
        lambda ctx, incidents, reason: restored.append(incidents)
        or [{"status": "SUCCEEDED"}],
    )
    spec = SPECS["inode_full"]
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec
    assert storage.INODE_METRIC == "filesystem_inodes_used_percent", spec
    regional = FakeRegional(cpu_python=store_script({"activity": None}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    mount = storage._fs_mount(ctx)
    activity = (
        quarantine_activity(mount, fired)
        if fired
        else {
            "incidents": [],
            "workflows": [],
            "batches": [],
            "markers": [],
            "commands": [],
            "health_signals": {},
            "notifications": [],
        }
    )
    regional.cpu_python_responder = store_script({"activity": activity})
    result = spec.execute(ctx)
    assert result["verdict"] == verdict, result["errors"]
    if fired == storage.INODE_METRIC:
        assert [item["incident_id"] for item in restored[0]] == ["inc-fs"], (
            "the inode quarantine must be restored before the variant ends"
        )


def test_sustained_util_passes(tmp_path: Path) -> None:
    spec = SPECS["sustained_util"]
    node = node_probe(survey())
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"util_percent": 100.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    inc = incident(
        "inc-io",
        "disk_io_util_percent",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io",
        "inc-io",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="disk_io_util_percent",
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    doc = {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                "2026-10-24T00:02:00+00:00", "disk_io_util_percent", 99.0, device="dm-7"
            )
        ]
        + [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "disk_io_util_percent",
                2.0,
                device="dm-7",
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node, storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_short_flush_passes(tmp_path: Path) -> None:
    spec = SPECS["short_flush"]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_reference_iostat_fails_on_divergent_await(tmp_path: Path) -> None:
    spec = SPECS["reference_iostat"]
    node = node_probe(survey())
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": 10.0}}} for _ in range(30)
        ],
        "next_offset": 30,
        "total": 30,
    }
    inc = incident(
        "inc-io",
        "disk_io_await_ms",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io", "inc-io", ["FREEZE_EVIDENCE", "VALIDATE_HOST"], status="RUNNING"
    )
    doc = {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                f"2026-10-24T00:0{i}:00+00:00", "disk_io_await_ms", 200.0, device="dm-7"
            )
            for i in range(4)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node, storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["diverged"] is True


def test_zero_completed_io_and_counter_reset_and_recovery(tmp_path: Path) -> None:
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    for variant in ("zero_completed_io", "counter_reset"):
        spec = SPECS[variant]
        regional = FakeRegional(cpu_python=store_script({"activity": empty}))
        ctx = make_context(
            spec,
            tmp_path=tmp_path,
            regional=regional,
            probes=probes(node_probe(survey()), storage_probe(), window_probe()),
        )
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])
    # recovery
    spec = SPECS["recovery"]
    inc = incident(
        "inc-io",
        "disk_io_util_percent",
        device="dm-7",
        state="RECOVERED",
        workflow="wf-io",
    )
    doc = {
        "incidents": [inc],
        "workflows": [
            workflow(
                "wf-io",
                "inc-io",
                ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
                status="SUCCEEDED",
            )
        ],
        "batches": [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "disk_io_util_percent",
                1.0,
                device="dm-7",
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_sustained_latency_requires_dm_delay(tmp_path: Path) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantNotRun
    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["sustained_latency"]
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    settings = make_settings(spec)
    settings = type(settings)(**{**settings.__dict__, "approvals": ()})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings,
        probes=probes(
            node_probe(survey(dm_delay=False)), storage_probe(), window_probe()
        ),
    )
    with pytest.raises(VariantNotRun):
        spec.execute(ctx)


def test_probe_error_is_deferred() -> None:
    assert SPECS["probe_error"].deferred.startswith("FIXTURE_NOT_READY"), (
        "SPECS['probe_error'].deferred.startswith('FIXTURE_NOT_READY')"
    )
