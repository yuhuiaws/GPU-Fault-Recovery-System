"""The storage probe's device-mapper test devices on a fake node.

Device commands (``dm-create``, ``dm-setup``, ``dm-suspend``/``dm-resume``,
``dm-recreate``, ``dm-retarget``, ``io-start``, ``dm-destroy`` and the
``cleanup`` kinds) run through ``main(argv)`` against ``FakeStorageHost``
(``_cov95_general_live_storage``), whose fake ``/proc`` lets the probe's own
holder sweep find a window Pod's private copy of a mapping, evict it and report
a locked one it cannot. Each test checks the refusal, the JSON report, the
state record and the host commands the probe ran.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_storage_probe as probe
from tests.regional._cov95_general_live_storage import (
    DIGEST,
    NS,
    RUN_ID,
    FakeStorageHost,
    add_ready_device,
    install_host,
)

Cli = Callable[..., tuple[int, dict[str, Any], str]]
DM = f"gfrs-{DIGEST}-io"
RESUME_DEADMAN = f"gfrs-{DIGEST}-sdeadman-resume-io"
WRITER = f"gfrs-{DIGEST}-io-io"


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeStorageHost:
    return install_host(monkeypatch, tmp_path)


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Cli:
    def invoke(*argv: str) -> tuple[int, dict[str, Any], str]:
        code = probe.main(list(argv))
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines, f"the probe prints one JSON object for {argv}: {captured}"
        return code, json.loads(lines[-1]), captured.err

    return invoke


def named(command: str, name: str = "io", *extra: str) -> list[str]:
    return [command, "--run-id", RUN_ID, "--name", name, *extra]


def create(host: FakeStorageHost, *extra: str) -> list[str]:
    return named(
        "dm-create",
        "io",
        "--size-mib",
        "256",
        "--backing",
        str(host.backing),
        "--deadman-seconds",
        "600",
        *extra,
    )


def refused(answer: dict[str, Any], command: str, site: str) -> bool:
    return (
        answer["error"] == "ProbeError"
        and answer["command"] == command
        and str(answer["error_site"]).startswith(f"{site}:")
    )


def hosted(host: FakeStorageHost) -> list[list[str]]:
    """The commands the probe ran in PID 1's namespaces, without the prefix."""

    return [call[len(NS) :] for call in host.calls if call[: len(NS)] == NS]


# --------------------------------------------------------------------------- #
# dm-create refusals
# --------------------------------------------------------------------------- #
def test_dm_create_without_nsenter_is_refused_before_any_change(
    host: FakeStorageHost, cli: Cli
) -> None:
    host.found.discard("nsenter")
    code, answer, err = cli(*create(host))
    assert code == 1 and refused(answer, "dm-create", "dm_create"), answer
    assert "nsenter is missing: device commands cannot reach the host" in err, err
    assert host.calls == [], "no deadman and no record without the host namespaces"


def test_dm_create_refuses_a_second_device_of_the_same_name(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    code, answer, err = cli(*create(host))
    assert code == 1 and refused(answer, "dm-create", "dm_create"), answer
    assert "a test device of this name already exists" in err, err


def test_a_module_that_will_not_load_leaves_a_record_the_cleanup_undoes(
    host: FakeStorageHost, cli: Cli
) -> None:
    host.loadable = False
    code, answer, err = cli(*create(host, "--delay-ms", "50", "--load-module"))
    assert code == 1 and refused(answer, "dm-create", "dm_create"), answer
    assert "dm_delay could not be loaded" in err, err
    state = probe.load_state(RUN_ID)
    assert state["modules"] == ["dm_delay"], "recorded before the load was tried"
    assert state["devices"]["io"]["phase"] == "setup", state["devices"]
    assert ["modprobe", "dm_delay"] in host.calls, host.calls
    assert list(host.started) == [f"gfrs-{DIGEST}-sdeadman-dm-io"], (
        "only the deadman runs; no setup unit starts without the module"
    )

    code, answer, _ = cli("cleanup", "--run-id", RUN_ID, "--kind", "dm-io")
    assert code == 0, answer
    destroyed = answer["actions"][0]["device"]
    assert destroyed["destroyed"] is True and destroyed["detached"] == [], destroyed
    assert {"modules_unloaded": ["dm_delay"]} in answer["actions"], answer["actions"]
    assert probe.load_state(RUN_ID)["modules"] == [], "the module record is cleared"


def test_dm_create_refuses_an_unsupported_file_system(host: FakeStorageHost) -> None:
    arguments = argparse.Namespace(
        run_id=RUN_ID,
        name="io",
        size_mib=256,
        delay_ms=0,
        deadman_seconds=600,
        fstype="xfs",
        load_module=False,
        backing=str(host.backing),
    )
    with pytest.raises(probe.ProbeError, match="unsupported test file system type"):
        probe.dm_create(arguments)
    assert host.calls == [], "nothing runs for an unsupported file system"


# --------------------------------------------------------------------------- #
# commands that need a device in the right phase
# --------------------------------------------------------------------------- #
def test_dm_setup_runs_only_for_a_device_waiting_for_it(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    code, answer, err = cli(*named("dm-setup"))
    assert code == 1 and refused(answer, "dm-setup", "dm_setup"), answer
    assert "the test device is not waiting for its setup" in err, err
    assert host.calls == [], "a finished device is never set up twice"


def test_device_commands_need_a_recorded_device(
    host: FakeStorageHost, cli: Cli
) -> None:
    code, answer, err = cli(*named("dm-resume", "nodev"))
    assert code == 1 and refused(answer, "dm-resume", "checked_device"), answer
    assert "no test device of this name" in err, err
    with pytest.raises(probe.ProbeError, match="no test device of this name"):
        probe.update_device(RUN_ID, "nodev", step="mkfs")


def test_device_commands_without_nsenter_cannot_reach_the_host(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    host.found.discard("nsenter")
    code, answer, err = cli(*named("dm-resume"))
    assert code == 1 and refused(answer, "dm-resume", "dm_resume"), answer
    assert "nsenter is missing" in err, err
    assert host.calls == [], "dmsetup never runs outside PID 1's namespaces"


def test_dm_suspend_never_suspends_without_an_armed_resume_deadman(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    host.timer_starts = False
    code, answer, err = cli(*named("dm-suspend", "io", "--seconds", "90"))
    assert code == 1 and refused(answer, "dm-suspend", "arm_deadman"), answer
    assert f"deadman {RESUME_DEADMAN} is not armed" in err, err
    assert not any(call[:1] == ["dmsetup"] for call in hosted(host)), (
        "the mapping is not suspended when nothing would resume it"
    )


def test_the_resume_deadman_resumes_without_stopping_its_own_service(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    assert cli(*named("dm-suspend", "io", "--seconds", "90"))[0] == 0, "suspended"
    assert ["dmsetup", "suspend", "--nolockfs", "--noflush", DM] in hosted(host)
    # The deadman fired: this cleanup runs inside its service.
    host.main_pids[f"{RESUME_DEADMAN}.service"] = str(os.getpid())
    host.calls.clear()
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID, "--kind", "resume-io")
    assert code == 0, answer
    assert answer["actions"] == [{"resumed_device": "io"}], answer["actions"]
    assert hosted(host) == [["dmsetup", "resume", DM]], host.calls
    assert ["systemctl", "stop", f"{RESUME_DEADMAN}.timer"] in host.calls, "timer"
    assert ["systemctl", "stop", f"{RESUME_DEADMAN}.service"] not in host.calls, (
        "the deadman's own cleanup never stops the service it runs in"
    )
    assert "io" in probe.load_state(RUN_ID)["devices"], "a resume keeps the device"


def test_a_resume_cleanup_without_dmsetup_leaves_the_device_alone(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    host.found.discard("dmsetup")
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID, "--kind", "resume-io")
    assert code == 0 and answer["actions"] == [], answer
    assert hosted(host) == [], "nothing is resumed without dmsetup"


@pytest.mark.parametrize("kind", ["dm-IO", "all;reboot", "fs-"])
def test_cleanup_refuses_an_unsafe_kind(
    host: FakeStorageHost, cli: Cli, kind: str
) -> None:
    code, answer, err = cli("cleanup", "--run-id", RUN_ID, "--kind", kind)
    assert code == 1 and refused(answer, "cleanup", "cleanup"), answer
    assert "unsafe cleanup kind" in err, err
    assert host.calls == [], "an unsafe kind destroys nothing"


# --------------------------------------------------------------------------- #
# dm-recreate / dm-retarget / io-start
# --------------------------------------------------------------------------- #
def test_dm_recreate_of_an_unmounted_device_maps_and_mounts_it_again(
    host: FakeStorageHost, cli: Cli
) -> None:
    record = add_ready_device(host)
    code, answer, _ = cli(*named("dm-recreate"))
    assert code == 0, answer
    assert answer == {"dm": DM, "device_before": "dm-11", "device_after": "dm-12"}, (
        "the recreated mapping is a new dm-N with fresh counters"
    )
    assert hosted(host) == [
        ["dmsetup", "remove", DM],
        ["dmsetup", "create", DM, "--table", "0 4194304 linear /dev/loop0 0"],
        ["mount", "-t", "ext4", record["dm_path"], record["mountpoint"]],
    ], "nothing to unmount first: remove, create, mount"
    assert probe.load_state(RUN_ID)["devices"]["io"]["device"] == "dm-12"


def test_dm_retarget_refuses_a_second_fault_mapping(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host, fstype="vfat", mounted=True)
    host.create_mapping(f"{DM}-fault")
    code, answer, err = cli(*named("dm-retarget", "io", "--target", "error"))
    assert code == 1 and refused(answer, "dm-retarget", "fault_mapping"), answer
    assert "the fault mapping already exists; retarget data first" in err, err
    assert not any(call[1] == "create" for call in hosted(host)), "no second mapping"


def test_dm_retarget_refuses_an_unknown_target(host: FakeStorageHost) -> None:
    add_ready_device(host)
    arguments = argparse.Namespace(
        run_id=RUN_ID, name="io", target="flaky", delay_ms=None
    )
    with pytest.raises(probe.ProbeError, match="retarget target must be data or error"):
        probe.dm_retarget(arguments)


def test_io_start_needs_the_device_mounted_on_the_host(
    host: FakeStorageHost, cli: Cli
) -> None:
    add_ready_device(host)
    code, answer, err = cli(*named("io-start", "io", "--seconds", "30"))
    assert code == 1 and refused(answer, "io-start", "io_start"), answer
    assert "the test device is not mounted on the host" in err, err


def test_io_start_records_its_writer_unit_once(host: FakeStorageHost, cli: Cli) -> None:
    record = add_ready_device(host, mounted=True)
    for _ in range(2):
        code, answer, _ = cli(*named("io-start", "io", "--seconds", "30"))
        assert code == 0 and answer["unit"] == WRITER, answer
    assert answer["state"]["ActiveState"] == "active", answer["state"]
    assert probe.load_state(RUN_ID)["units"] == [WRITER], (
        "a restarted writer is recorded once"
    )
    started = host.started[WRITER]
    assert "--property=Nice=19" in started and "--property=RuntimeMaxSec=30" in started
    assert started[started.index("-c") + 2 :] == [
        str(Path(record["mountpoint"]) / "gfrs-io"),
        "16",
        str(64 * 1024),
        "30",
        "0",
        str(16 << 20),
        "0.0",
    ], "the writer gets path, threads, block, seconds, direct, span and pace"


# --------------------------------------------------------------------------- #
# dm-destroy: unmount, evict foreign copies, remove, detach
# --------------------------------------------------------------------------- #
def test_dm_destroy_evicts_private_copies_and_reports_a_locked_one(
    host: FakeStorageHost, cli: Cli
) -> None:
    record = add_ready_device(host, mounted=True)
    target = record["mountpoint"]
    devno = "253:11"
    window = (("259:1", "/"), (devno, target))
    host.add_process(200, mnt="mnt:[4026532500]", comm="gfrs-window", mounts=window)
    host.add_process(201, mnt="mnt:[4026532500]", comm="gfrs-window", mounts=window)
    host.add_process(300, mnt="mnt:[4026532600]", comm="kubelet", mounts=window[:1])
    host.add_process(
        400,
        mnt="mnt:[4026532700]",
        user="user:[4026532800]",
        comm="sandbox",
        mounts=((devno, "/data"),),
    )
    (host.proc / "500").mkdir()  # a process that exited mid-sweep
    (Path(target) / "gfrs-io.0").write_bytes(b"left by the writer")

    code, answer, _ = cli(*named("dm-destroy"))
    assert code == 0, answer
    holders = sorted(answer["holders"], key=lambda holder: holder["comm"])
    assert holders[0]["pid"] in {200, 201}, "one entry per mount namespace"
    assert [{k: v for k, v in item.items() if k != "pid"} for item in holders] == [
        {
            "comm": "gfrs-window",
            "locked": False,
            "mountpoints": [target],
            "evicted": True,
        },
        {"comm": "sandbox", "locked": True, "mountpoints": ["/data"], "evicted": False},
    ], "the window Pod's copy is evicted; a locked copy stays and is reported"
    assert holders[1]["pid"] == 400, holders
    assert answer["deferred"] == [] and answer["detached"] == ["/dev/loop0"], answer
    assert answer["directory_removed"] is False, (
        "a mount point that still holds files is left, not forced"
    )
    assert Path(target, "gfrs-io.0").is_file(), "nothing inside it is deleted"
    assert answer["directories_removed"] == [str(host.backing / f"gfrs-live-{DIGEST}")]
    assert ["umount", target] in hosted(host), "the host mount goes first"
    assert ["dmsetup", "remove", "--retry", DM] in hosted(host), "then the mapping"
    assert "io" not in probe.load_state(RUN_ID)["devices"], "the record is removed"


def test_unmount_stack_takes_busy_mounts_down_lazily_and_stops_at_eight(
    host: FakeStorageHost,
) -> None:
    target = str(host.root / "stacked")
    host.mount_on_host("/dev/mapper/a", target)
    host.mount_on_host("/dev/mapper/b", target)
    host.busy.add(target)
    assert probe.unmount_stack(target) == 2, "two stacked mounts were taken down"
    assert hosted(host) == [["umount", target], ["umount", "-l", target]] * 2, (
        "a busy mount is detached lazily"
    )
    host.busy.clear()
    for index in range(9):
        host.mount_on_host(f"/dev/mapper/m{index}", target)
    assert probe.unmount_stack(target) == 8, "the sweep is bounded"
    assert [item[1] for item in host.host_mounts] == [target], "one mount is left"


def test_unload_modules_unloads_only_idle_modules_the_run_loaded(
    host: FakeStorageHost, cli: Cli
) -> None:
    (host.sys_module / "dm_delay").mkdir()  # no refcnt file: read as idle
    (host.sys_module / "vfat").mkdir()
    (host.sys_module / "vfat" / "refcnt").write_text("2\n", encoding="utf-8")
    (host.sys_module / "nvme").mkdir()
    state = probe.load_state(RUN_ID)
    state["modules"] = ["dm_delay", "vfat", "nvme"]
    probe.save_state(RUN_ID, state)

    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert code == 0, answer
    assert {"modules_unloaded": ["dm_delay"]} in answer["actions"], answer["actions"]
    assert [call for call in host.calls if call[0] == "modprobe"] == [
        ["modprobe", "-r", "dm_delay"]
    ], "a module in use and one the probe may not unload are left loaded"
    assert probe.load_state(RUN_ID)["modules"] == ["vfat", "nvme"], "still recorded"
    assert answer["residue"]["modules_loaded_by_run"] == ["vfat", "nvme"], answer
    assert answer["clean"] is False, "loaded modules of the run are residue"


# --------------------------------------------------------------------------- #
# the device readers
# --------------------------------------------------------------------------- #
def test_dm_device_name_skips_block_devices_it_cannot_read(
    host: FakeStorageHost,
) -> None:
    (host.sys_block / "dm-1").mkdir()  # no dm/name: not readable
    host.create_mapping("gfrs-other")
    assert probe.dm_device_name("gfrs-other") == "dm-11", "found past dm-1"


def test_direct_read_reads_one_page_with_o_direct_and_always_closes(
    host: FakeStorageHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[str, int]] = []
    reads: list[tuple[int, int, int]] = []
    closed: list[int] = []
    failure: list[OSError] = []

    def preadv(fd: int, buffers: list[Any], offset: int) -> int:
        reads.append((fd, len(buffers[0]), offset))
        if failure:
            raise failure[0]
        return len(buffers[0])

    monkeypatch.setattr(
        probe.os, "open", lambda path, flags: opened.append((path, flags)) or 42
    )
    monkeypatch.setattr(probe.os, "preadv", preadv)
    monkeypatch.setattr(probe.os, "close", closed.append)
    assert probe.direct_read("/dev/mapper/fault", 2048) == "ok", "a served page"
    failure.append(OSError(errno.EIO, "Input/output error"))
    assert probe.direct_read("/dev/mapper/fault", 8) == "OSError: 5", "EIO reported"
    assert opened == [("/dev/mapper/fault", os.O_RDONLY | os.O_DIRECT)] * 2, opened
    page = 512 * probe.PAGE_SECTORS
    assert reads == [(42, page, 2048 * 512), (42, page, 8 * 512)], reads
    assert closed == [42, 42], "the descriptor is closed after an error too"


def test_mount_devno_is_the_major_minor_of_the_path_file_system(
    host: FakeStorageHost, tmp_path: Path
) -> None:
    device = os.stat(tmp_path).st_dev
    assert probe.mount_devno(str(tmp_path)) == f"{os.major(device)}:{os.minor(device)}"
