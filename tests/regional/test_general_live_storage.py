"""GF-GENERAL-LIVE-004 injector/verdict contract over scripted records."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional.general_live_deps import DEPS_TIMEOUT, PACKAGE_APPROVAL
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
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
from tests.regional._general_live_storage_fakes import (
    NODE,
    SPECS,
    node_probe,
    probes,
    storage_probe,
    survey,
    window_probe,
)


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


# --------------------------------------------------------------------------- #
# 004 probe_error: the FAT16 dm device and its error-table swap
# --------------------------------------------------------------------------- #
ERR_MOUNT = "/mnt"  # the scripted dm-status mountpoint


DOSFS_NEVRA = "dosfstools-4.2-3.amzn2023.0.2.x86_64"
WITH_INSTALL = ("KERNEL_MODULE_LOAD", PACKAGE_APPROVAL)


def deps_probe(**overrides: Any) -> FakeProbe:
    """The dependency probe: ``dosfstools`` installed for the run, clean removal."""

    ensure = {
        "refused": None,
        "tools": {"mkfs.vfat": True},
        "modules": {},
        "installed": [DOSFS_NEVRA],
        "modules_loaded_by_run": [],
        "transaction": [{"name": "dosfstools", "section": "Installing"}],
        "deadman": "gfrs-abcdef-pdeadman",
    }
    ensure.update(overrides)
    return FakeProbe({"ensure": ensure, "cleanup": {"clean": True}})


def probe_error_context(
    tmp_path: Path,
    *,
    activity: dict[str, Any] | None,
    store: FakeProbe | None = None,
    node: FakeProbe | None = None,
    deps: FakeProbe | None = None,
    approvals: tuple[str, ...] | None = None,
) -> tuple[Any, Any]:
    spec = SPECS["probe_error"]
    regional = FakeRegional(cpu_python=store_script({"activity": activity or {}}))
    settings = make_settings(spec, node=NODE)
    if approvals is not None:
        settings = dataclasses.replace(settings, approvals=approvals)
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={
            **probes(
                node or node_probe(survey()), store or storage_probe(), window_probe()
            ),
            f"deps@{NODE}": deps or deps_probe(),
        },
        settings=settings,
    )
    return spec, ctx


def test_probe_error_is_a_native_variant_behind_kernel_module_load() -> None:
    spec = SPECS["probe_error"]
    assert spec.implemented and not spec.deferred, spec
    assert spec.approvals == ("KERNEL_MODULE_LOAD",), spec.approvals
    assert spec.optional_approvals == (PACKAGE_APPROVAL,), spec.optional_approvals
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
    deps = deps_probe()
    store = storage_probe()
    spec, ctx = probe_error_context(
        tmp_path, activity={}, node=node_probe(value), store=store, deps=deps
    )
    with pytest.raises(VariantNotRun) as raised:
        spec.execute(ctx)
    assert raised.value.reason == "CAPABILITY_ABSENT"
    assert raised.value.detail == (
        "the node lacks mkfs.vfat (TEMP_TEST_PACKAGE_INSTALL installs dosfstools "
        "for one run)"
    )
    assert deps.calls == [] and store.calls == [], "refused before any change"
    assert ctx.cleanup.pending() == []


def test_probe_error_installs_dosfstools_for_the_run_behind_the_token(
    tmp_path: Path,
) -> None:
    value = survey()
    value["tools"]["mkfs.vfat"] = False
    deps = deps_probe()
    store = storage_probe()
    spec, ctx = probe_error_context(
        tmp_path,
        activity=unavailable_activity(ERR_MOUNT, "PROBE_FAILED"),
        node=node_probe(value),
        store=store,
        deps=deps,
        approvals=WITH_INSTALL,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert deps.calls == [
        (
            "ensure",
            "--tool",
            "mkfs.vfat",
            "--deadman-seconds",
            str(spec.max_minutes * 60 + 600 + DEPS_TIMEOUT + 300),
            "--allow-install",
            "--run-id",
            ctx.run_id,
        )
    ]
    created = next(call for call in store.calls if call[0] == "dm-create")
    assert ("--fstype", "vfat") == created[
        created.index("--fstype") : created.index("--fstype") + 2
    ], "the device is built with the tool installed moments before"
    assert result["dependencies"] == {
        "changed": True,
        "installed": [DOSFS_NEVRA],
        "modules_loaded_by_run": [],
        "transaction": [{"name": "dosfstools", "section": "Installing"}],
    }
    # LIFO: the device goes first, the package last.
    assert ctx.cleanup.pending() == ["deps", "dm"], ctx.cleanup.pending()
    assert ctx.cleanup.run() and ctx.cleanup.failed == [], ctx.cleanup.results
    assert deps.calls[-1] == ("cleanup", "--run-id", ctx.run_id)
    assert store.calls[-1][:3] == ("cleanup", "--kind", "dm-err"), store.calls[-1]


def test_probe_error_with_mkfs_vfat_present_installs_nothing(tmp_path: Path) -> None:
    deps = deps_probe()
    spec, ctx = probe_error_context(
        tmp_path,
        activity=unavailable_activity(ERR_MOUNT, "PROBE_FAILED"),
        deps=deps,
        approvals=WITH_INSTALL,
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert deps.calls == [] and result["dependencies"]["changed"] is False
    assert ctx.cleanup.pending() == ["dm"]


def test_probe_error_refused_install_is_fixture_not_ready_before_the_device(
    tmp_path: Path,
) -> None:
    value = survey()
    value["tools"]["mkfs.vfat"] = False
    refused = {
        "reason": "FIXTURE_NOT_READY",
        "detail": "dnf would also upgrading glibc-2.34-52.amzn2023.0.10.x86_64",
    }
    store = storage_probe()
    spec, ctx = probe_error_context(
        tmp_path,
        activity={},
        node=node_probe(value),
        store=store,
        deps=deps_probe(refused=refused, installed=[], tools={"mkfs.vfat": False}),
        approvals=WITH_INSTALL,
    )
    with pytest.raises(VariantNotRun) as raised:
        spec.execute(ctx)
    assert (raised.value.reason, raised.value.detail) == (
        "FIXTURE_NOT_READY",
        refused["detail"],
    )
    assert store.calls == [], "no device once the dependency step refused"
    assert ctx.cleanup.pending() == ["deps"], "a half-finished install is undone"


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
