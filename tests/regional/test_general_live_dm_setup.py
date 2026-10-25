"""GF-GENERAL-LIVE-006: the runner side of the bounded device setup.

The storage probe's ``dm-create`` only starts a transient setup unit; the
runner polls ``dm-status`` on the context clock, gives every 006 probe call the
probe's own budget plus a transport margin as its ``kubectl exec`` timeout,
and registers each undo before the call it undoes. Fake probes and a fake
clock drive it; nothing here touches a node.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.probes import general_live_storage_probe as storage_probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    store_script,
)

SPECS = {spec.variant: spec for spec in storage.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
EMPTY = {
    "incidents": [],
    "workflows": [],
    "batches": [],
    "markers": [],
    "commands": [],
    "health_signals": {},
    "notifications": [],
}
READY = {
    "name": "io",
    "phase": "ready",
    "dm": "gfrs-x-io",
    "device": "dm-7",
    "mountpoint": "/mnt/gfrs-live/x/io",
    "failed_step": None,
}


def survey(*, dm_delay: bool = False) -> dict[str, Any]:
    tools = (
        "losetup",
        "dmsetup",
        "mkfs.ext4",
        "mount",
        "blockdev",
        "fallocate",
        "systemd-run",
        "nsenter",
        "modprobe",
    )
    return {
        "tools": {name: True for name in tools},
        "modules": {
            "dm_mod": {"loaded": True, "available": True},
            "dm_delay": {"loaded": dm_delay, "available": True},
        },
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "xfs"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            }
        ],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {},
    }


def storage_fake(**overrides: Any) -> FakeProbe:
    script: dict[str, Any] = {
        "dm-create": {"name": "io", "phase": "setup"},
        "dm-status": dict(READY),
        "dm-suspend": {"suspended": "gfrs-x-io"},
        "dm-resume": {"resumed": "gfrs-x-io"},
        "dm-recreate": {"device_before": "dm-7", "device_after": "dm-7"},
        "io-start": {"unit": "gfrs-x-io-io"},
        "io-stop": {"stopped": {}},
        "cleanup": {"clean": True, "residue": {}},
    }
    script.update(overrides)
    return FakeProbe(script)


def context(
    tmp_path: Path,
    variant: str,
    store: FakeProbe,
    *,
    activity: dict[str, Any] | None = None,
    dm_delay: bool = False,
) -> Any:
    node = FakeProbe(
        {
            "survey": survey(dm_delay=dm_delay),
            "oracle-read": {"records": [], "next_offset": 0, "total": 0},
        }
    )
    regional = FakeRegional(cpu_python=store_script({"activity": activity or EMPTY}))
    ctx = make_context(
        SPECS[variant],
        tmp_path=tmp_path,
        regional=regional,
        probes={f"node@{NODE}": node, f"storage@{NODE}": store},
    )
    # A 10 s step per clock read: the observation windows end in tens of polls.
    start = datetime(2026, 10, 25, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=10 * next(ticks))
    return ctx


def names(store: FakeProbe, *wanted: str) -> list[str]:
    return [call[0] for call in store.calls if not wanted or call[0] in wanted]


def test_the_undo_comes_first_then_the_setup_is_polled_until_ready(
    tmp_path: Path,
) -> None:
    store = storage_fake()
    ctx = context(tmp_path, "short_flush", store)
    pending: list[list[str]] = []

    def dm_create(arguments: tuple[str, ...]) -> dict[str, Any]:
        pending.append(ctx.cleanup.pending())
        return {"name": "io", "phase": "setup"}

    store.script["dm-create"] = dm_create
    store.script["dm-status"] = [
        {"name": "io", "phase": "setup", "step": "fallocate"},
        {"name": "io", "phase": "setup", "step": "mkfs"},
        dict(READY),
    ]
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert pending == [["dm"]], "the device undo is registered before the create"
    assert names(store)[:5] == [
        "dm-create",
        "dm-status",
        "dm-status",
        "dm-status",
        "io-start",
    ]
    timeline = json.loads((ctx.case_dir / "dm-setup-timeline.json").read_text())
    assert [entry["accepted"] for entry in timeline["entries"]] == [False, False, True]
    assert json.loads((ctx.case_dir / "dm-ready.json").read_text())["device"] == "dm-7"
    assert result["device"] == "dm-7"


def test_a_failed_setup_names_its_step_and_keeps_the_undo(tmp_path: Path) -> None:
    store = storage_fake(
        **{
            "dm-status": {
                "name": "io",
                "phase": "failed",
                "failed_step": "dmsetup-create",
                "setup_error_class": "TimeoutExpired",
                "setup_error_code": "a" * 16,
                "setup_error_site": "build_device:640",
            }
        }
    )
    ctx = context(tmp_path, "short_flush", store)
    expected = (
        f"failed at 'dmsetup-create': TimeoutExpired {'a' * 16} at build_device:640"
    )
    with pytest.raises(RegionalFixtureError) as caught:
        ctx.spec.execute(ctx)
    assert expected in str(caught.value), str(caught.value)
    assert ctx.cleanup.pending() == ["dm"]
    results = ctx.cleanup.run()
    assert [(item["name"], item["ok"]) for item in results] == [("dm", True)]
    assert store.calls[-1] == ("cleanup", "--kind", "dm-io", "--run-id", ctx.run_id)


def test_a_setup_unit_that_died_without_a_record_names_the_step(tmp_path: Path) -> None:
    status = {
        "name": "io",
        "phase": "failed",
        "failed_step": "mkfs",
        "setup_error_class": "SetupUnitEnded",
        "setup_error_code": None,
        "setup_error_site": None,
    }
    store = storage_fake(**{"dm-status": status})
    ctx = context(tmp_path, "short_flush", store)
    with pytest.raises(RegionalFixtureError) as caught:
        ctx.spec.execute(ctx)
    message = str(caught.value)
    assert message.endswith("failed at 'mkfs': SetupUnitEnded"), message
    assert "None" not in message, message


def test_a_setup_that_never_finishes_is_bounded_by_its_unit(tmp_path: Path) -> None:
    store = storage_fake(
        **{"dm-status": {"name": "io", "phase": "setup", "step": "mkfs"}}
    )
    ctx = context(tmp_path, "short_flush", store)
    start = datetime(2026, 10, 25, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=7 * next(ticks))
    with pytest.raises(RegionalFixtureError, match="did not finish within") as caught:
        ctx.spec.execute(ctx)
    assert "'mkfs'" in str(caught.value), str(caught.value)
    polls = names(store, "dm-status")
    assert 2 <= len(polls) <= storage.SETUP_WAIT_SECONDS // 7 + 2, len(polls)
    assert storage.SETUP_WAIT_SECONDS > storage_probe.SETUP_MAX_SECONDS, (
        "the unit's RuntimeMaxSec always ends the setup before the runner gives up"
    )
    assert ctx.cleanup.pending() == ["dm"]


@pytest.mark.parametrize(
    "variant",
    [
        "short_flush",
        "zero_completed_io",
        "counter_reset",
        "recovery",
        "reference_iostat",
        "sustained_util",
        "sustained_latency",
    ],
)
def test_every_006_storage_call_waits_out_the_probe_budget(
    tmp_path: Path, variant: str
) -> None:
    store = storage_fake()
    detected = {
        **EMPTY,
        "incidents": [
            incident("inc-u", "disk_io_util_percent", device="dm-7"),
            incident("inc-a", "disk_io_await_ms", device="dm-7"),
        ],
    }
    activity = EMPTY if variant in {"short_flush", "zero_completed_io"} else detected
    ctx = context(tmp_path, variant, store, activity=activity, dm_delay=True)
    try:
        ctx.spec.execute(ctx)
    except VariantInconclusive:
        pass  # sustained_*: no scripted oracle samples prove the injection
    ctx.cleanup.run()
    used = {subcommand for subcommand, _ in store.timeouts}
    assert {"dm-create", "dm-status", "io-start", "io-stop", "cleanup"} <= used, used
    wrong = [
        (subcommand, timeout)
        for subcommand, timeout in store.timeouts
        if timeout
        != storage_probe.CALL_BUDGET_SECONDS[subcommand]
        + storage.TRANSPORT_MARGIN_SECONDS
    ]
    assert wrong == [], wrong


def test_counter_reset_stops_the_writer_before_the_recreate(tmp_path: Path) -> None:
    store = storage_fake()
    ctx = context(tmp_path, "counter_reset", store)
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert names(store, "io-start", "io-stop", "dm-recreate") == [
        "io-start",
        "io-stop",
        "dm-recreate",
        "io-start",
        "io-stop",
    ]
    assert ctx.cleanup.pending() == ["dm"]


def test_zero_completed_io_registers_the_resume_before_the_suspend(
    tmp_path: Path,
) -> None:
    store = storage_fake()
    ctx = context(tmp_path, "zero_completed_io", store)
    pending: list[list[str]] = []

    def dm_suspend(arguments: tuple[str, ...]) -> dict[str, Any]:
        pending.append(ctx.cleanup.pending())
        return {"suspended": "gfrs-x-io"}

    store.script["dm-suspend"] = dm_suspend
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    # LIFO: the resume runs first, then the writer stops, then the device goes.
    assert pending == [["dm", "oracle-disk", "io", "dm-resume"]]
    assert ctx.cleanup.pending() == ["dm", "oracle-disk"]


def test_the_device_undo_fails_unless_the_node_is_proven_clean(tmp_path: Path) -> None:
    residue = {"device_mapper": ["gfrs-x-io"], "loop_devices": []}
    store = storage_fake(cleanup={"clean": False, "residue": residue})
    ctx = context(tmp_path, "short_flush", store)
    assert ctx.spec.execute(ctx)["verdict"] == "PASS"
    results = ctx.cleanup.run()
    assert [(item["name"], item["ok"]) for item in results] == [("dm", False)]
    assert "left residue" in results[0]["error"], results
    assert ctx.cleanup.failed == results


def test_a_missing_nsenter_is_capability_absent_before_any_change(
    tmp_path: Path,
) -> None:
    from scripts.e2e.regional.general_live_fixture import VariantNotRun

    store = storage_fake()
    ctx = context(tmp_path, "short_flush", store)
    ctx.probes[f"node@{NODE}"].script["survey"]["tools"]["nsenter"] = False
    with pytest.raises(VariantNotRun) as caught:
        ctx.spec.execute(ctx)
    assert caught.value.reason == "CAPABILITY_ABSENT", caught.value
    assert "nsenter" in caught.value.detail, caught.value.detail
    assert store.calls == [] and ctx.cleanup.pending() == [], store.calls
