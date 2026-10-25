"""GF-GENERAL-LIVE-004/006 injector/verdict contract over scripted records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
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


def survey(log_collector: str = "inactive", **modules: bool) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            name: True
            for name in (
                "losetup",
                "mkfs.ext4",
                "mkfs.vfat",
                "mount",
                "fallocate",
                "dmsetup",
                "blockdev",
                "systemd-run",
                "nsenter",
                "modprobe",
            )
        },
        "modules": {
            "dm_mod": {"loaded": True, "available": True},
            "dm_delay": {"loaded": modules.get("dm_delay", True), "available": True},
            "vfat": {"loaded": modules.get("vfat", True), "available": True},
        },
        "units": {storage.LOG_COLLECTOR_UNIT: {"ActiveState": log_collector}},
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


def retarget_answer(arguments: tuple[str, ...]) -> dict[str, Any]:
    """``dm-retarget``: EIO on the error table, a healthy reading on the data one."""

    error = "error" in arguments
    return {
        "name": "err",
        "target": "error" if error else "data",
        "attempts": 1,
        "mounted_on_host": True,
        "faulted": error,
        "statvfs": {"error": "OSError: 5"} if error else {"used_percent": 1.0},
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
        # dm-create only starts the setup unit; dm-status reports the device.
        "dm-create": {"name": "io", "dm": "gfrs-x-io", "phase": "setup"},
        "dm-status": {
            "name": "io",
            "phase": "ready",
            "dm": "gfrs-x-io",
            "device": "dm-7",
            "mountpoint": "/mnt",
        },
        "cleanup": {"clean": True, "residue": {}},
        "dm-suspend": {"suspended": "gfrs-x-io"},
        "dm-resume": {"resumed": "gfrs-x-io"},
        "dm-recreate": {"device_before": "dm-7", "device_after": "dm-8"},
        "dm-retarget": retarget_answer,
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


def unavailable_activity(mount: str, mode: str | None = None) -> dict[str, Any]:
    labels = {"failure_mode": mode} if mode else {}
    inc = incident(
        "inc-fs",
        "filesystem_unavailable",
        device=mount,
        state="ACTION_PENDING",
        workflow="wf-fs",
    )
    wf = workflow(
        "wf-fs",
        "inc-fs",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="filesystem_unavailable",
        executions=[
            {
                "operation": "VALIDATE_HOST",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            {
                **host_batch(
                    "2026-10-24T00:02:00+00:00",
                    "filesystem_unavailable",
                    1.0,
                    device=mount,
                ),
                "samples": [
                    {
                        "name": "filesystem_unavailable",
                        "device": mount,
                        "value": 1.0,
                        "labels": labels,
                    }
                ],
            }
        ]
        + [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00",
                "filesystem_unavailable",
                0.0,
                device=mount,
            )
            for i in range(3)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }


@pytest.mark.parametrize("variant", ["read_only", "required_mount_missing"])
def test_unusable_filesystem_validates_and_readmits(
    tmp_path: Path, variant: str
) -> None:
    # Batch 163 reports a read-only or vanished configured mount as
    # filesystem_unavailable and runs the VALIDATE_HOST diagnostic, so both
    # variants now expect PASS live.
    spec = SPECS[variant]
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec
    regional = FakeRegional(cpu_python=store_script({"activity": None}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    mount = storage._fs_mount(ctx)
    regional.cpu_python_responder = store_script(
        {"activity": unavailable_activity(mount)}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


@pytest.mark.parametrize("variant", ["read_only", "required_mount_missing"])
def test_unusable_filesystem_fails_when_undetected(
    tmp_path: Path, variant: str
) -> None:
    # A product that still skipped the unusable mount (no filesystem_unavailable
    # finding) makes the runner FAIL the variant.
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
    assert result["verdict"] == "FAIL", result
    assert any("filesystem_unavailable" in error for error in result["errors"]), result


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
    # Without dm_delay the plain linear device path is attempted and the scripted
    # util samples carry it through; with dm_delay it runs on a 20 ms delay.
    spec = SPECS["sustained_util"]
    node = node_probe(survey(dm_delay=False))
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


def _reference_iostat_result(tmp_path: Path, standard_await_ms: float):
    spec = SPECS["reference_iostat"]
    node = node_probe(survey())
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": standard_await_ms}}}
            for _ in range(30)
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
    return spec, spec.execute(ctx)


def test_reference_iostat_fails_on_divergent_await(tmp_path: Path) -> None:
    # The product reports 200 ms while the oracle's standard await of the same
    # device and interval is 10 ms: the spec FAIL, whatever the registry expects.
    spec, result = _reference_iostat_result(tmp_path, standard_await_ms=10.0)
    assert result["verdict"] == "FAIL"
    assert result["diverged"] is True


def test_reference_iostat_expects_pass_and_passes_on_matching_await(
    tmp_path: Path,
) -> None:
    # Batch 172: the host collector reports iostat's await (fields 4+8 over
    # 1+5), so the registry no longer carries the weighted-await product gap and
    # a product value within 10 % / 5 ms of the oracle's is the expected PASS.
    spec, result = _reference_iostat_result(tmp_path, standard_await_ms=198.0)
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec
    assert result["verdict"] == "PASS", result["errors"]
    assert result["diverged"] is False
    assert result["product_await_ms"] == [200.0] * 4


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
    with pytest.raises(VariantNotRun) as caught:
        spec.execute(ctx)
    assert caught.value.reason == "CAPABILITY_ABSENT"


def _dm_create_arguments(store: FakeProbe) -> tuple[str, ...]:
    created = [call for call in store.calls if call[0] == "dm-create"]
    assert len(created) == 1, store.calls
    return created[0]


@pytest.mark.parametrize(
    ("variant", "loaded", "delay", "load"),
    [
        ("sustained_util", True, "20", False),
        ("sustained_util", False, "0", False),
        ("sustained_latency", True, "200", False),
        ("sustained_latency", False, "200", True),
    ],
    ids=["util-delay", "util-linear", "latency-loaded", "latency-loads"],
)
def test_dm_io_load_runs_natively_with_or_without_a_loaded_dm_delay(
    tmp_path: Path, variant: str, loaded: bool, delay: str, load: bool
) -> None:
    # The 2026-10-24/25 TimeoutExpired was the probe Pod's dmsetup waiting for a
    # udev cookie it could never get (see the storage probe), not the delay
    # target, so neither variant defers on dm_delay any more: util uses a 20 ms
    # delay when one is loaded, latency a 200 ms one -- loaded for the run
    # (KERNEL_MODULE_LOAD, the variant's approval) when the node lacks it.
    spec = SPECS[variant]
    store = storage_probe()
    metric = (
        "disk_io_util_percent" if variant == "sustained_util" else "disk_io_await_ms"
    )
    detected = {"incidents": [incident("inc-io", metric, device="dm-7")]}
    regional = FakeRegional(cpu_python=store_script({"activity": detected}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(dm_delay=loaded)), store, window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)  # no scripted oracle samples: the injection is unproven
    arguments = _dm_create_arguments(store)
    assert arguments[arguments.index("--delay-ms") + 1] == delay, arguments
    assert ("--load-module" in arguments) is load, arguments
    assert [call[0] for call in store.calls[:2]] == ["dm-create", "dm-status"], (
        store.calls
    )


# --------------------------------------------------------------------------- #
# 004 probe_error: the FAT16 dm device and its error-table swap
# --------------------------------------------------------------------------- #
ERR_MOUNT = "/mnt"  # the scripted dm-status mountpoint


def probe_error_context(
    tmp_path: Path,
    *,
    activity: dict[str, Any] | None,
    store: FakeProbe | None = None,
    node: FakeProbe | None = None,
) -> tuple[Any, Any]:
    spec = SPECS["probe_error"]
    regional = FakeRegional(cpu_python=store_script({"activity": activity or {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(
            node or node_probe(survey()), store or storage_probe(), window_probe()
        ),
    )
    return spec, ctx


def test_probe_error_is_a_native_variant_behind_kernel_module_load() -> None:
    spec = SPECS["probe_error"]
    assert spec.implemented and not spec.deferred, spec
    assert spec.approvals == ("KERNEL_MODULE_LOAD",), spec.approvals
    assert spec.lanes == ("eks", "hp"), spec.lanes
    assert spec.expected_outcome == "PASS" and not spec.product_gap, spec


def test_probe_error_swaps_the_error_table_under_a_declared_vfat_device(
    tmp_path: Path,
) -> None:
    store = storage_probe()
    win = window_probe()
    spec, ctx = probe_error_context(
        tmp_path, activity=unavailable_activity(ERR_MOUNT, "PROBE_FAILED"), store=store
    )
    ctx.probes[f"window@{NODE}"] = win  # type: ignore[assignment]
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    created = next(call for call in store.calls if call[0] == "dm-create")
    assert ("--fstype", "vfat") == created[
        created.index("--fstype") : created.index("--fstype") + 2
    ]
    assert "--name" in created and created[created.index("--name") + 1] == "err", (
        created
    )
    assert "--load-module" not in created, "vfat is loaded in the survey: no load"
    retargets = [call for call in store.calls if call[0] == "dm-retarget"]
    assert [call[call.index("--target") + 1] for call in retargets] == ["error", "data"]
    opened = next(call for call in win.calls if call[0] == "open-window")
    assert f"GPU_FAULT_FILESYSTEMS={ERR_MOUNT}" in opened, opened
    assert "dm-data" not in ctx.cleanup.pending(), ctx.cleanup.pending()
    assert ctx.cleanup.pending() == ["dm"], ctx.cleanup.pending()


def test_probe_error_loads_vfat_under_the_token_when_the_node_lacks_it(
    tmp_path: Path,
) -> None:
    store = storage_probe()
    spec, ctx = probe_error_context(
        tmp_path,
        activity=unavailable_activity(ERR_MOUNT, "PROBE_FAILED"),
        store=store,
        node=node_probe(survey(vfat=False)),
    )
    assert spec.execute(ctx)["verdict"] == "PASS"
    created = next(call for call in store.calls if call[0] == "dm-create")
    assert "--load-module" in created, created


def test_probe_error_is_inconclusive_when_statvfs_still_answers(tmp_path: Path) -> None:
    store = storage_probe(
        **{
            "dm-retarget": {
                "faulted": False,
                "attempts": 3,
                "mounted_on_host": True,
                "statvfs": {"used_percent": 1.0},
            }
        }
    )
    spec, ctx = probe_error_context(tmp_path, activity={}, store=store)
    with pytest.raises(VariantInconclusive, match="no EIO"):
        spec.execute(ctx)
    # The data-table restore stays registered for the cleanup stack.
    assert "dm-data" in ctx.cleanup.pending(), ctx.cleanup.pending()


def test_probe_error_refuses_an_active_node_log_collector(tmp_path: Path) -> None:
    spec, ctx = probe_error_context(
        tmp_path, activity={}, node=node_probe(survey(log_collector="active"))
    )
    with pytest.raises(VariantNotRun, match="log collector"):
        spec.execute(ctx)
    assert ctx.cleanup.pending() == [], "refused before any mutation"


def test_probe_error_needs_mkfs_vfat(tmp_path: Path) -> None:
    value = survey()
    value["tools"]["mkfs.vfat"] = False
    spec, ctx = probe_error_context(tmp_path, activity={}, node=node_probe(value))
    with pytest.raises(VariantNotRun, match="mkfs.vfat"):
        spec.execute(ctx)


def test_probe_error_fails_when_the_product_never_reports_the_mount(
    tmp_path: Path,
) -> None:
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    spec, ctx = probe_error_context(tmp_path, activity=empty)
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("filesystem_unavailable" in error for error in result["errors"]), result


@pytest.mark.parametrize(
    ("variant", "mode"),
    [("read_only", "READ_ONLY"), ("required_mount_missing", "MOUNT_MISSING")],
)
def test_unusable_filesystem_wrong_failure_mode_label_fails(
    tmp_path: Path, variant: str, mode: str
) -> None:
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=store_script({"activity": None}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), storage_probe(), window_probe()),
    )
    digest = hashlib.sha256(ctx.run_id.encode()).hexdigest()[:12]
    mount = f"/mnt/gfrs-live/{digest}/{storage.TEST_FS}"
    regional.cpu_python_responder = store_script(
        {"activity": unavailable_activity(mount, "PROBE_FAILED")}
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(mode in error for error in result["errors"]), result["errors"]


@pytest.mark.parametrize(
    ("loaded", "approved", "delay", "load"),
    [(True, False, "200", False), (False, True, "200", True)],
    ids=["delay-loaded", "delay-loaded-for-the-run"],
)
def test_reference_iostat_lifts_the_awaits_with_dm_delay(
    tmp_path: Path, loaded: bool, approved: bool, delay: str, load: bool
) -> None:
    """The weighted/standard comparison needs awaits above the 5 ms tolerance
    floor: dm-delay 200 ms, loaded for the run under the optional
    KERNEL_MODULE_LOAD when the node lacks it (before, only a dm_delay someone
    else had loaded counted, and this site never has one)."""

    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["reference_iostat"]
    assert spec.optional_approvals == ("KERNEL_MODULE_LOAD",), spec.optional_approvals
    store = storage_probe()
    settings = make_settings(spec)
    approvals = ("KERNEL_MODULE_LOAD",) if approved else ()
    settings = type(settings)(**{**settings.__dict__, "approvals": approvals})
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        settings=settings,
        probes=probes(node_probe(survey(dm_delay=loaded)), store, window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)  # no scripted oracle samples
    arguments = _dm_create_arguments(store)
    assert arguments[arguments.index("--delay-ms") + 1] == delay, arguments
    assert ("--load-module" in arguments) is load, arguments


def test_reference_iostat_is_inconclusive_under_the_tolerance_floor(
    tmp_path: Path,
) -> None:
    """No dm_delay and no approval to load it: a linear device on NVMe reads
    sub-millisecond awaits that no formula difference can lift past the 5 ms
    floor -- 2026-10-25 on ip-10-0-43-5 the product's 0.30-0.44 ms against the
    oracle's 0.33-0.76 ms read PASS on an expected-FAIL variant."""

    from tests.regional._general_live_fakes import make_settings

    spec = SPECS["reference_iostat"]
    store = storage_probe()
    node = node_probe(survey(dm_delay=False))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": 0.45}}} for _ in range(30)
        ],
        "next_offset": 30,
        "total": 30,
    }
    doc = {
        "incidents": [incident("inc-io", "disk_io_await_ms", device="dm-7")],
        "workflows": [],
        "batches": [
            host_batch(
                f"2026-10-24T00:0{i}:00+00:00", "disk_io_await_ms", 0.3, device="dm-7"
            )
            for i in range(4)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
    }
    settings = make_settings(spec)
    settings = type(settings)(**{**settings.__dict__, "approvals": ()})
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": doc})),
        settings=settings,
        probes=probes(node, store, window_probe()),
    )
    with pytest.raises(VariantInconclusive) as caught:
        spec.execute(ctx)
    assert "tolerance floor" in str(caught.value), str(caught.value)
    arguments = _dm_create_arguments(store)
    assert arguments[arguments.index("--delay-ms") + 1] == "0", arguments


def _latency_run(tmp_path: Path, store: FakeProbe) -> dict[str, Any]:
    spec = SPECS["sustained_latency"]
    node = node_probe(survey(dm_delay=True))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"standard_await_ms": 210.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    inc = incident(
        "inc-io",
        "disk_io_await_ms",
        device="dm-7",
        state="ACTION_PENDING",
        workflow="wf-io",
    )
    wf = workflow(
        "wf-io",
        "inc-io",
        ["FREEZE_EVIDENCE", "VALIDATE_HOST"],
        status="FAILED",
        probe_metric="disk_io_await_ms",
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
                "2026-10-24T00:02:00+00:00", "disk_io_await_ms", 210.0, device="dm-7"
            )
        ]
        + [
            host_batch(
                f"2026-10-24T00:2{i}:00+00:00", "disk_io_await_ms", 0.4, device="dm-7"
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
        probes=probes(node, store, window_probe()),
    )
    return spec.execute(ctx)


def test_sustained_latency_removes_the_delay_and_proves_recovery_on_paced_io(
    tmp_path: Path,
) -> None:
    """Live 2026-10-25 14:47Z (ip-10-0-43-5, R13e): after ``io-stop`` the
    dm-delay device saw no completed I/O for the whole 690 s recovery wait,
    so the product -- by its zero-completed-I/O rule -- emitted no await
    sample and the signal could never clear; with the 200 ms table still
    loaded any I/O would have read above the 100 ms threshold anyway. The
    recovery proof now takes the delay out of the data table and keeps a
    paced single writer on the same dm-N while the product samples it."""

    store = storage_probe()
    result = _latency_run(tmp_path, store)
    assert result["verdict"] == "PASS", result["errors"]
    names = [call[0] for call in store.calls]
    first_stop = names.index("io-stop")
    assert names[first_stop:] == ["io-stop", "dm-retarget", "io-start", "io-stop"], (
        names
    )
    retarget = store.calls[first_stop + 1]
    assert retarget[retarget.index("--target") + 1] == "data", retarget
    assert retarget[retarget.index("--delay-ms") + 1] == "0", retarget
    writer = store.calls[first_stop + 2]
    assert writer[writer.index("--threads") + 1] == "1", writer
    assert writer[writer.index("--pace-ms") + 1] == "100", writer
    assert "--direct" in writer, writer
    assert int(writer[writer.index("--seconds") + 1]) >= 690 + 60, writer
    assert result["recovery_load"]["delay_removed"]["target"] == "data", result
    assert result["recovery"]["fresh_samples"], result["recovery"]


def test_sustained_util_recovery_needs_no_verification_load(tmp_path: Path) -> None:
    """Util falls to 0 % without I/O (field 10 over elapsed time), so the util
    variant keeps the idle recovery wait: no table swap, no second writer."""

    spec = SPECS["sustained_util"]
    store = storage_probe()
    node = node_probe(survey(dm_delay=False))
    node.script["oracle-read"] = {
        "records": [
            {"diskstats": {"dm-7": {"util_percent": 100.0}}} for _ in range(80)
        ],
        "next_offset": 80,
        "total": 80,
    }
    doc = {
        "incidents": [incident("inc-io", "disk_io_util_percent", device="dm-7")],
        "workflows": [],
        "batches": [
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
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=store_script({"activity": doc})),
        probes=probes(node, store, window_probe()),
    )
    spec.execute(ctx)
    names = [call[0] for call in store.calls]
    assert "dm-retarget" not in names and names.count("io-start") == 1, names
