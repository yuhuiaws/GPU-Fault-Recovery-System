"""Edge and negative branches of the GF-GENERAL-LIVE category modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_collectors as col
from scripts.e2e.regional import general_live_host as host
from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
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
}


def host_survey(cpu_count: int = 8) -> dict[str, Any]:
    return {
        "cpu_count": cpu_count,
        "tools": {"systemd-run": True},
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


def host_node(oracle: dict[str, Any]) -> FakeProbe:
    return FakeProbe(
        {
            "survey": host_survey(),
            "cpu-start": {},
            "cpu-stop": {},
            "oracle-start": {},
            "oracle-stop": {},
            "oracle-read": oracle,
        }
    )


def ctx_for(
    module, variant: str, tmp_path: Path, *, probes: dict[str, FakeProbe], activity: Any
) -> Any:
    spec = next(s for s in module.VARIANTS if s.variant == variant)
    regional = FakeRegional(cpu_python=store_script({"activity": activity}))
    settings = make_settings(spec, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "site_file": tmp_path / "s.yaml"})
    (tmp_path / "s.yaml").write_text("x\n")
    return spec, make_context(
        spec, tmp_path=tmp_path, regional=regional, probes=probes, settings=settings
    )


def test_host_cpu_busy_no_cpus(tmp_path: Path) -> None:
    spec = next(s for s in host.VARIANTS if s.variant == "cpu_busy")
    probe = FakeProbe({"survey": host_survey(cpu_count=0)})
    regional = FakeRegional(cpu_python=store_script({"activity": EMPTY}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={f"node@{NODE}": probe}
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_host_load_average_inconclusive(tmp_path: Path) -> None:
    oracle = {
        "records": [{"load1_per_cpu": 0.1} for _ in range(10)],
        "next_offset": 10,
        "total": 10,
    }
    inc = incident("inc", "load1_per_cpu")
    doc = {
        **EMPTY,
        "incidents": [inc],
        "workflows": [workflow("wf", "inc-load1_per_cpu", ["VALIDATE_HOST"])],
    }
    spec, ctx = ctx_for(
        host,
        "load_average",
        tmp_path,
        probes={f"node@{NODE}": host_node(oracle)},
        activity=doc,
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_host_recovery_without_incident_fails(tmp_path: Path) -> None:
    # cpu stays busy on the oracle, but no incident is ever minted.
    oracle = {
        "records": [{"cpu_busy_percent": 99.0} for _ in range(40)],
        "next_offset": 40,
        "total": 40,
    }
    doc = {
        **EMPTY,
        "batches": [
            host_batch(f"2026-10-24T00:2{i}:00+00:00", "cpu_usage_percent", 1.0)
            for i in range(3)
        ],
    }
    spec, ctx = ctx_for(
        host,
        "recovery",
        tmp_path,
        probes={f"node@{NODE}": host_node(oracle)},
        activity=doc,
    )
    # detection never finds an incident (wait budget elapses), so recovery stage
    # reports "no cpu finding to recover from".
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("recover from" in e for e in result["errors"]), (
        "any('recover from' in e for e in result['errors'])"
    )


def test_host_await_recovery_timeout(tmp_path: Path) -> None:
    oracle = {
        "records": [{"cpu_busy_percent": 99.0} for _ in range(40)],
        "next_offset": 40,
        "total": 40,
    }
    # Chain is fine but no fresh below-threshold samples ever appear.
    inc = incident("inc", "cpu_usage_percent")
    wf = workflow(
        "wf",
        "inc-cpu_usage_percent",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="cpu_usage_percent",
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    doc = {
        **EMPTY,
        "incidents": [inc],
        "workflows": [wf],
        "batches": [host_batch("2026-10-24T00:02:00+00:00", "cpu_usage_percent", 99.0)],
    }
    spec, ctx = ctx_for(
        host,
        "cpu_busy",
        tmp_path,
        probes={f"node@{NODE}": host_node(oracle)},
        activity=doc,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("did not recover" in e for e in result["errors"]), result["errors"]


def storage_survey() -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            n: True
            for n in (
                "losetup",
                "mkfs.ext4",
                "mount",
                "fallocate",
                "dmsetup",
                "blockdev",
            )
        },
        "modules": {
            "dm_mod": {"loaded": True, "available": True},
            "dm_delay": {"loaded": True, "available": True},
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


def storage_probes(tmp_path: Path, **fs_overrides: Any) -> dict[str, FakeProbe]:
    node = FakeProbe(
        {
            "survey": storage_survey(),
            "oracle-start": {},
            "oracle-stop": {},
            "oracle-read": {"records": [], "next_offset": 0, "total": 0},
        }
    )
    store_script_map = {
        "fs-create": {
            "name": "fs",
            "mountpoint": "/mnt/gfrs-live/x/fs",
            "statvfs": {"used_percent": 20.0},
        },
        "fs-fill": {"statvfs": {"used_percent": 99.0}},
        "fs-unfill": {},
        "fs-inodes": {"enospc": True},
        "fs-remount": {"statvfs": {"read_only": True}},
        "fs-umount": {"directory_present": False, "statvfs": {"used_percent": None}},
        "fs-mount": {},
        "fs-destroy": {},
        "dm-create": {
            "name": "io",
            "dm": "d",
            "device": "dm-9",
            "mountpoint": "/mnt/gfrs-live/x/io",
        },
        "dm-suspend": {},
        "dm-resume": {},
        "dm-recreate": {"device_before": "dm-9", "device_after": "dm-10"},
        "dm-destroy": {},
        "io-start": {},
        "io-stop": {},
    }
    store_script_map.update(fs_overrides)
    return {
        f"node@{NODE}": node,
        f"storage@{NODE}": FakeProbe(store_script_map),
        f"window@{NODE}": FakeProbe(
            {
                "open-window": {"after": {"ActiveState": "active"}},
                "close-window": {"after": {"ActiveState": "active"}},
            }
        ),
    }


def test_storage_read_only_gap_is_fail(tmp_path: Path) -> None:
    spec, ctx = ctx_for(
        storage, "read_only", tmp_path, probes=storage_probes(tmp_path), activity=EMPTY
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL" and result["detected"] is False


def test_storage_required_mount_missing_gap_is_fail(tmp_path: Path) -> None:
    spec, ctx = ctx_for(
        storage,
        "required_mount_missing",
        tmp_path,
        probes=storage_probes(tmp_path),
        activity=EMPTY,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"


def test_storage_sustained_util_inconclusive(tmp_path: Path) -> None:
    probes = storage_probes(tmp_path)
    # The inconclusive (oracle never sustains) path is only reached on the plain
    # linear device; with dm_delay present the variant defers instead.
    probes[f"node@{NODE}"].script["survey"]["modules"]["dm_delay"]["loaded"] = False
    probes[f"node@{NODE}"].script["oracle-read"] = {
        "records": [{"diskstats": {"dm-9": {"util_percent": 1.0}}} for _ in range(40)],
        "next_offset": 40,
        "total": 40,
    }
    doc = {
        **EMPTY,
        "incidents": [incident("i", "disk_io_util_percent", device="dm-9")],
        "workflows": [workflow("w", "inc-disk_io_util_percent", ["VALIDATE_HOST"])],
    }
    spec, ctx = ctx_for(
        storage, "sustained_util", tmp_path, probes=probes, activity=doc
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_storage_zero_completed_io_flags_pseudo_rate(tmp_path: Path) -> None:
    doc = {
        **EMPTY,
        "batches": [host_batch("t", "disk_io_await_ms", 50.0, device="dm-9")],
    }
    spec, ctx = ctx_for(
        storage,
        "zero_completed_io",
        tmp_path,
        probes=storage_probes(tmp_path),
        activity=doc,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any("await sample was emitted" in e for e in result["errors"]), (
        "any('await sample was emitted' in e for e in result['errors'])"
    )


def net_survey() -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {"ip": True, "tc": True, "systemd-run": True, "systemctl": True},
        "modules": {
            m: {"loaded": True, "available": True}
            for m in ("veth", "dummy", "sch_netem")
        },
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
            "GPU_FAULT_REQUIRED_INTERFACES": "",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def test_network_unknown_state_without_error_fails(tmp_path: Path) -> None:
    node = FakeProbe({"survey": net_survey()})
    netp = FakeProbe(
        {
            "create-dummy": {"name": "gfrsaaaaaad", "link": {"operstate": "unknown"}},
            "cleanup": {},
        }
    )
    win = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )
    spec = next(s for s in net.VARIANTS if s.variant == "unknown_state")
    regional = FakeRegional(cpu_python=store_script({"activity": EMPTY}))
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node, f"net@{NODE}": netp, f"window@{NODE}": win},
        settings=settings,
    )
    result = spec.execute(ctx)
    # No collection error observed -> the collection_error stage fails.
    assert result["verdict"] == "FAIL"
    assert any("collection error" in e for e in result["errors"]), (
        "any('collection error' in e for e in result['errors'])"
    )


def test_network_unknown_state_inconclusive_if_not_unknown(tmp_path: Path) -> None:
    node = FakeProbe({"survey": net_survey()})
    netp = FakeProbe(
        {
            "create-dummy": {"name": "gfrsaaaaaad", "link": {"operstate": "up"}},
            "cleanup": {},
        }
    )
    win = FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )
    spec = next(s for s in net.VARIANTS if s.variant == "unknown_state")
    regional = FakeRegional(cpu_python=store_script({"activity": EMPTY}))
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node, f"net@{NODE}": netp, f"window@{NODE}": win},
        settings=settings,
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_collectors_exporter_timeout_inconclusive_if_not_stopped(
    tmp_path: Path,
) -> None:
    spec = next(s for s in col.VARIANTS if s.variant == "exporter_timeout")
    node = FakeProbe(
        {
            "survey": {
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
            },
            "exporter-pause": {"stopped": False},
        }
    )
    regional = FakeRegional(cpu_python=store_script({"activity": EMPTY}))
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node},
        settings=settings,
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)
