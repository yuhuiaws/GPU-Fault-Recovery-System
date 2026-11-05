"""The storage probe's test file systems, guards and CLI shell on a fake node.

``probes/general_live_storage_probe.py`` creates loop file systems under
``/mnt/gfrs-live/<digest>``; here ``FakeStorageHost`` (``_cov95_general_live_
storage``) answers its commands and ``statvfs`` readings, so ``main(argv)``
creates, fills, exhausts, remounts, unmounts and destroys a file system and the
test checks the probe's JSON, its state file, the files it leaves in the fake
mount point and the host commands in the order it ran them.
"""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_storage_probe as probe
from tests.regional._cov95_general_live_storage import (
    DIGEST,
    RUN_ID,
    FakeStorageHost,
    add_ready_device,
    install_host,
)

Cli = Callable[..., tuple[int, dict[str, Any], str]]
MIB = 1 << 20


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


def create(cli: Cli, host: FakeStorageHost, name: str = "data", *extra: str) -> Any:
    return cli(
        "fs-create",
        "--run-id",
        RUN_ID,
        "--name",
        name,
        "--size-mib",
        "256",
        "--backing",
        str(host.backing),
        "--deadman-seconds",
        "600",
        *extra,
    )


def named(command: str, name: str = "data", *extra: str) -> list[str]:
    return [command, "--run-id", RUN_ID, "--name", name, *extra]


def refused(answer: dict[str, Any], command: str, site: str) -> bool:
    return (
        answer["error"] == answer["error_class"] == "ProbeError"
        and answer["command"] == command
        and str(answer["error_site"]).startswith(f"{site}:")
        and len(answer["error_code"]) == 16
    )


def mountpoint(name: str = "data") -> str:
    return str(probe.mountpoint(RUN_ID, name))


# --------------------------------------------------------------------------- #
# fs-create
# --------------------------------------------------------------------------- #
def test_fs_create_arms_the_deadman_then_formats_and_mounts_a_loop_file_system(
    host: FakeStorageHost, cli: Cli
) -> None:
    code, answer, err = create(cli, host, "data", "--inodes", "4096")
    assert code == 0, err
    backing = host.backing / f"gfrs-live-{DIGEST}" / "data.img"
    assert answer["backing"] == str(backing) and backing.is_file(), answer
    assert (answer["name"], answer["loop"]) == ("data", "/dev/loop0"), answer
    assert answer["mountpoint"] == mountpoint(), answer
    assert answer["deadman"] == f"gfrs-{DIGEST}-sdeadman-fs-data", answer
    assert answer["statvfs"]["used_percent"] == 50.0, answer["statvfs"]
    deadman = host.started[answer["deadman"]]
    assert "--on-active=600s" in deadman, deadman
    assert deadman[-5:] == ["cleanup", "--run-id", RUN_ID, "--kind", "fs-data"], (
        "the deadman undoes exactly this file system"
    )
    tools = [call for call in host.mutations() if call[0] != "systemctl"]
    assert tools[1:] == [
        ["fallocate", "-l", "256M", str(backing)],
        ["losetup", "--find", "--show", str(backing)],
        [
            "mkfs.ext4",
            "-q",
            "-F",
            "-m",
            "0",
            "-L",
            "gfrs-data",
            "-N",
            "4096",
            "/dev/loop0",
        ],
        ["mount", "-t", "ext4", "/dev/loop0", mountpoint()],
    ], "the deadman (first) is armed before anything is created"
    assert tools[0][0] == "systemd-run", tools[0]
    record = probe.load_state(RUN_ID)["filesystems"]["data"]
    assert record["mounted"] is True and record["loop"] == "/dev/loop0", record

    code, answer, err = create(cli, host, "data")
    assert code == 1 and refused(answer, "fs-create", "fs_create"), answer
    assert "a test file system of this name already exists" in err, err


@pytest.mark.parametrize(
    "free_blocks, total_blocks, message",
    [
        (10, 1000, "has no room for the test file"),
        (0, 0, "has no room for the test file"),
        (250_000, 1_000_000, "would exceed its usage guard"),
    ],
    ids=["no-room", "no-size", "usage-guard"],
)
def test_fs_create_refuses_a_backing_file_system_without_headroom(
    host: FakeStorageHost, cli: Cli, free_blocks: int, total_blocks: int, message: str
) -> None:
    host.filesystems[str(host.backing)] = (total_blocks, free_blocks, 10, 10, 0)
    code, answer, err = create(cli, host)
    assert code == 1 and refused(answer, "fs-create", "backing_root"), answer
    assert message in err, err
    assert host.mutations() == [], "nothing is armed or created on a full root"
    assert host.backing.is_dir(), "the allow-listed root itself may be created"


@pytest.mark.parametrize(
    "position, value, site, message",
    [
        (2, "BAD ID", "checked_run_id", "unsafe run ID"),
        (4, "Data1", "checked_name", "unsafe resource name"),
        (6, "100", "bounded", "size MiB is outside 256..4096"),
        (10, "30", "bounded", "deadman seconds is outside 60..5400"),
    ],
    ids=["run-id", "name", "size", "deadman"],
)
def test_fs_create_validates_every_argument_before_touching_the_host(
    host: FakeStorageHost, cli: Cli, position: int, value: str, site: str, message: str
) -> None:
    argv = [
        "fs-create",
        "--run-id",
        RUN_ID,
        "--name",
        "data",
        "--size-mib",
        "256",
        "--backing",
        str(host.backing),
        "--deadman-seconds",
        "600",
    ]
    argv[position] = value
    code, answer, err = cli(*argv)
    assert code == 1 and refused(answer, "fs-create", site), answer
    assert message in err, err
    assert host.calls == [], "a refused argument runs no command"


def test_an_unexpected_loop_device_keeps_the_record_the_cleanup_undoes(
    host: FakeStorageHost, cli: Cli
) -> None:
    host.losetup_answer = "loop0"
    code, answer, err = create(cli, host)
    assert code == 1 and refused(answer, "fs-create", "loop_attach"), answer
    assert "losetup returned an unexpected device" in err, err
    record = probe.load_state(RUN_ID)["filesystems"]["data"]
    assert (record["loop"], record["mounted"]) == (None, False), record

    host.calls.clear()
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID, "--kind", "fs-data")
    assert code == 0, answer
    assert answer["actions"] == [
        {"filesystem": {"name": "data", "destroyed": True}},
        {"directories_removed": [str(host.backing / f"gfrs-live-{DIGEST}")]},
    ], answer
    assert not any(call[0] == "losetup" and "-d" in call for call in host.calls), (
        "no loop was recorded, so none is detached"
    )
    assert "data" not in probe.load_state(RUN_ID)["filesystems"], "record removed"


# --------------------------------------------------------------------------- #
# fs-fill / fs-inodes / fs-unfill
# --------------------------------------------------------------------------- #
def test_fill_exhaust_and_unfill_a_test_file_system(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    target = Path(mountpoint())
    # 4000 KiB of which 10 % used; four inodes.
    host.filesystems[str(target)] = (1000, 900, 4, 4, 0)
    synced = host.synced

    code, answer, _ = cli(*named("fs-fill", "data", "--percent", "50"))
    assert code == 0 and answer["fill_bytes"] == MIB, answer
    assert (target / "gfrs-fill.bin").is_file(), "the fill file is allocated"
    assert host.calls[-1] == [
        "fallocate",
        "-l",
        str(MIB),
        str(target / "gfrs-fill.bin"),
    ]
    assert host.synced == synced + 2, "the probe syncs before and after the fill"

    host.calls.clear()
    code, answer, _ = cli(*named("fs-fill", "data", "--percent", "5"))
    assert code == 0 and answer["fill_bytes"] == 0, "already above 5 %: no fill"
    assert host.calls == [], "nothing is allocated for an empty fill"
    assert not (target / "gfrs-fill.bin").exists(), "the previous fill is removed"

    code, answer, _ = cli(*named("fs-inodes"))
    assert code == 0, answer
    assert (answer["created"], answer["enospc"]) == (20, False), (
        "the filler stops at the inode count plus its margin"
    )
    assert len(list((target / "gfrs-inodes").iterdir())) == 20, "twenty empty files"

    assert cli(*named("fs-fill", "data", "--percent", "50"))[0] == 0, "refilled"
    code, answer, _ = cli(*named("fs-unfill"))
    assert code == 0 and answer["removed"] == 21, answer
    assert not (target / "gfrs-inodes").exists(), "the inode directory is removed"
    assert not (target / "gfrs-fill.bin").exists(), "the fill file is removed"
    assert cli(*named("fs-unfill"))[1]["removed"] == 0, "unfill again is a no-op"


@pytest.mark.parametrize("percent", ["0", "100.5"])
def test_fs_fill_refuses_a_percent_outside_one_to_a_hundred(
    host: FakeStorageHost, cli: Cli, percent: str
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    code, answer, err = cli(*named("fs-fill", "data", "--percent", percent))
    assert code == 1 and refused(answer, "fs-fill", "fs_fill"), answer
    assert "fill percent is outside 1..100" in err, err


def test_fs_fill_reports_a_failed_allocation(host: FakeStorageHost, cli: Cli) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    host.failing["fallocate"] = 1
    code, answer, err = cli(*named("fs-fill", "data", "--percent", "90"))
    assert code == 1 and refused(answer, "fs-fill", "fs_fill"), answer
    assert "fallocate of the fill file failed" in err, err


def test_fs_inodes_stops_at_the_first_file_it_cannot_create(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    directory = Path(mountpoint()) / "gfrs-inodes"
    directory.mkdir()
    (directory / "f0000002").write_text("", encoding="utf-8")
    code, answer, _ = cli(*named("fs-inodes"))
    assert code == 0, answer
    assert (answer["created"], answer["enospc"]) == (2, False), (
        "a failure other than ENOSPC ends the filler without claiming exhaustion"
    )


def test_fs_commands_need_a_recorded_file_system(
    host: FakeStorageHost, cli: Cli
) -> None:
    code, answer, err = cli(*named("fs-unfill", "nothing"))
    assert code == 1 and refused(answer, "fs-unfill", "checked_filesystem"), answer
    assert "no test file system of this name" in err, err


# --------------------------------------------------------------------------- #
# fs-remount / fs-umount / fs-mount
# --------------------------------------------------------------------------- #
def test_remount_unmount_and_mount_round_trip(host: FakeStorageHost, cli: Cli) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    target = mountpoint()
    code, answer, _ = cli(*named("fs-remount", "data", "--mode", "ro"))
    assert code == 0 and answer["mode"] == "ro", answer
    assert host.calls[-1] == ["mount", "-o", "remount,ro", target], host.calls[-1]

    code, answer, _ = cli(*named("fs-umount"))
    assert code == 0, answer
    assert (answer["unmounted"], answer["directory_present"]) == (target, True), answer
    assert probe.load_state(RUN_ID)["filesystems"]["data"]["mounted"] is False

    code, answer, _ = cli(*named("fs-mount"))
    assert code == 0 and answer["mounted"] == target, answer
    assert host.calls[-1] == ["mount", "-t", "ext4", "/dev/loop0", target], (
        "a file system record mounts its loop device"
    )
    assert probe.load_state(RUN_ID)["filesystems"]["data"]["mounted"] is True

    code, answer, _ = cli(*named("fs-umount", "data", "--rmdir"))
    assert code == 0 and answer["directory_present"] is False, answer
    assert answer["statvfs"] == {"path": target, "error": "FileNotFoundError: 2"}, (
        "a removed mount point reads back as the statvfs error"
    )


def test_fs_remount_refuses_an_unknown_mode_and_reports_a_failed_mount(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    code, answer, err = cli(*named("fs-remount", "data", "--mode", "noexec"))
    assert code == 1 and refused(answer, "fs-remount", "fs_remount"), answer
    assert "remount mode must be ro or rw" in err, err

    host.failing["mount"] = 32
    code, answer, err = cli(*named("fs-remount", "data", "--mode", "rw"))
    assert code == 1 and refused(answer, "fs-remount", "fs_remount"), answer
    assert f"command failed (32): mount -o remount,rw {mountpoint()}" in err, err
    assert "mount: refused by the fake host" in err, "the tool's stderr is carried"


def test_fs_mount_and_umount_address_a_device_record_by_its_mapping(
    host: FakeStorageHost, cli: Cli
) -> None:
    record = add_ready_device(host, "io", fstype="vfat")
    code, answer, _ = cli(*named("fs-mount", "io"))
    assert code == 0 and answer["mounted"] == record["mountpoint"], answer
    assert host.calls[-1] == [
        "mount",
        "-t",
        "vfat",
        "-o",
        "errors=continue,noatime",
        record["dm_path"],
        record["mountpoint"],
    ], "a device mounts its mapping with the FAT carrier's options"
    assert probe.load_state(RUN_ID)["devices"]["io"]["mounted"] is True, (
        "a device record is updated in the devices bucket"
    )
    code, answer, _ = cli(*named("fs-umount", "io"))
    assert code == 0 and answer["unmounted"] == record["mountpoint"], answer
    assert probe.load_state(RUN_ID)["devices"]["io"]["mounted"] is False, (
        "and unmounted there again"
    )


def test_mount_command_refuses_an_unsupported_file_system() -> None:
    with pytest.raises(probe.ProbeError, match="unsupported test file system type"):
        probe.mount_command("xfs", "/dev/loop0", "/mnt/gfrs-live/x/data")


# --------------------------------------------------------------------------- #
# fs-destroy and cleanup of file systems
# --------------------------------------------------------------------------- #
def test_fs_destroy_unmounts_detaches_and_prunes_the_run_directories(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    target = mountpoint()
    host.calls.clear()
    code, answer, _ = cli(*named("fs-destroy"))
    deadman = f"gfrs-{DIGEST}-sdeadman-fs-data"
    assert code == 0, answer
    assert answer == {
        "name": "data",
        "destroyed": True,
        "directories_removed": [
            str(host.backing / f"gfrs-live-{DIGEST}"),
            str(probe.MOUNT_ROOT / DIGEST),
        ],
    }, answer
    assert host.mutations() == [
        ["umount", target],
        ["losetup", "-d", "/dev/loop0"],
        ["systemctl", "stop", f"{deadman}.timer"],
        ["systemctl", "stop", f"{deadman}.service"],
        ["systemctl", "reset-failed", f"{deadman}.service", f"{deadman}.timer"],
    ], host.mutations()
    assert host.own_mounts == [], "the probe's own mount is gone"
    assert not Path(target).exists(), "the mount point is removed"

    code, answer, _ = cli(*named("fs-destroy"))
    assert answer == {"name": "data", "absent": True, "directories_removed": []}, (
        "destroying again finds nothing"
    )


def test_fs_destroy_keeps_the_run_directories_while_another_file_system_is_left(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host, "data")[0] == 0, "first file system"
    assert create(cli, host, "logs")[0] == 0, "second file system"
    code, answer, _ = cli(*named("fs-destroy", "data"))
    assert code == 0 and answer["directories_removed"] == [], answer
    assert list(probe.load_state(RUN_ID)["filesystems"]) == ["logs"], "logs is left"


def test_cleanup_of_one_file_system_leaves_the_rest_until_all(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host, "data")[0] == 0, "first file system"
    assert create(cli, host, "logs")[0] == 0, "second file system"
    add_ready_device(host, "io")
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID, "--kind", "fs-data")
    assert code == 0, answer
    assert answer["actions"] == [{"filesystem": {"name": "data", "destroyed": True}}], (
        "only the named file system goes; no device and no module is touched"
    )
    state = probe.load_state(RUN_ID)
    assert list(state["filesystems"]) == ["logs"] and "io" in state["devices"], state
    assert answer["clean"] is False, answer["residue"]
    assert not any(call[0] == "modprobe" for call in host.calls), "no module unload"

    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert code == 0 and answer["clean"] is True, answer["residue"]
    kinds = [next(iter(action)) for action in answer["actions"]]
    assert kinds == ["device", "filesystem", "directories_removed"], answer["actions"]


# --------------------------------------------------------------------------- #
# residue
# --------------------------------------------------------------------------- #
def test_residue_lists_what_the_run_still_holds(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    record = add_ready_device(host, "io", mounted=True)
    host.services[f"gfrs-{DIGEST}-io-io"] = "active"
    code, answer, _ = cli("residue", "--run-id", RUN_ID)
    assert code == 0 and answer["clean"] is False, answer
    residue = answer["residue"]
    assert residue["mounts"] == [record["mountpoint"]], "PID 1's table is read"
    assert residue["device_mapper"] == [record["dm"]], residue
    assert residue["loop_devices"] == ["/dev/loop0", "/dev/loop1"], residue
    assert residue["backing_files"] == sorted(
        str(host.backing / f"gfrs-live-{DIGEST}" / f"{name}.img")
        for name in ("data", "io")
    ), residue
    assert residue["units"] == [f"gfrs-{DIGEST}-io-io.service"], residue
    assert residue["directories"] == sorted(
        [str(host.backing / f"gfrs-live-{DIGEST}"), str(probe.MOUNT_ROOT / DIGEST)]
    ), residue


def test_residue_falls_back_to_the_probe_mount_table_without_device_tools(
    host: FakeStorageHost, cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    monkeypatch.setattr(probe, "HOST_MOUNTINFO", host.root / "unreadable")
    host.found.clear()
    host.calls.clear()
    code, answer, _ = cli("residue", "--run-id", RUN_ID)
    residue = answer["residue"]
    assert residue["mounts"] == [mountpoint()], (
        "an unreadable host table falls back to the probe's own mounts"
    )
    assert (residue["device_mapper"], residue["loop_devices"]) == ([], []), residue
    assert [call[0] for call in host.calls] == ["systemctl"], (
        "without dmsetup and losetup only the unit listing runs"
    )


def test_residue_treats_an_unparsable_loop_listing_as_none(
    host: FakeStorageHost, cli: Cli
) -> None:
    assert create(cli, host)[0] == 0, "file system created"
    host.losetup_listing = "losetup: not json"
    code, answer, _ = cli("residue", "--run-id", RUN_ID)
    assert code == 0 and answer["residue"]["loop_devices"] == [], answer


def test_a_state_file_of_another_run_is_refused(
    host: FakeStorageHost, cli: Cli
) -> None:
    path = probe.state_dir(RUN_ID) / "storage-state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": "gl004-other-run"}), encoding="utf-8")
    code, answer, err = cli("residue", "--run-id", RUN_ID)
    assert code == 1 and refused(answer, "residue", "load_state"), answer
    assert "storage probe state identity mismatch" in err, err


# --------------------------------------------------------------------------- #
# the call budget and the CLI shell
# --------------------------------------------------------------------------- #
def test_a_call_past_its_budget_fails_before_the_next_command(
    host: FakeStorageHost, cli: Cli
) -> None:
    host.clock.step = 1000.0  # every clock reading is 1000 s later
    code, answer, err = cli(*named("io-stop", "io"))
    assert code == 1, answer
    assert (answer["error"], answer["error_class"]) == ("BudgetExhausted",) * 2
    assert str(answer["error_site"]).startswith("stop_unit:"), answer
    assert "used up its time budget" in err, err
    assert host.calls == [], "no command starts once the budget is spent"
    assert probe.DEADLINE == [], "the deadline is cleared when the call ends"


def test_budgeted_commands_are_clamped_to_what_is_left(
    host: FakeStorageHost, cli: Cli
) -> None:
    code, answer, _ = cli(*named("io-stop", "io"))
    assert code == 0 and answer["stopped"]["ActiveState"] == "inactive", answer
    assert host.timeouts[0] == probe.UNIT_STOP_SECONDS, (
        "a stop inside a 120 s budget keeps its own 60 s bound"
    )


def test_utc_now_is_an_aware_utc_timestamp() -> None:
    stamp = datetime.fromisoformat(probe.utc_now())
    assert stamp.utcoffset() == timedelta(0), stamp


def test_running_the_file_as_a_script_exits_with_the_cli_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unsafe run ID is refused before the probe reads or writes the host.
    monkeypatch.setattr(sys, "argv", [probe.__file__, "residue", "--run-id", "BAD ID"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert raised.value.code == 1, "a refused command exits 1"
    captured = capsys.readouterr()
    answer = json.loads(captured.out.strip().splitlines()[-1])
    assert answer["command"] == "residue" and answer["error"] == "ProbeError", answer
    assert str(answer["error_site"]).startswith("checked_run_id:"), answer
    assert "ProbeError: unsafe run ID" in captured.err, captured.err
