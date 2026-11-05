"""The shared-mount probe (GF-GENERAL-LIVE-005) on the host shapes its contract
tests leave out.

``test_general_live_shared_probe.py`` proves the mount/unmount/cleanup cycle on
its ``FakeHost``; this module reuses that host (``host`` fixture, ``mount`` /
``call`` / ``target``) and bends it: a missing ``mount.nfs4``, a second mount of
one name, a mount point already mounted, an NFS module that will not load, a
mount the host table never shows, a deadman that does not arm (with stray
``systemctl`` output), an export that will not unmount even lazily, a forgotten
mount whose directory is gone, a remount of a live mount, an out-of-range stat
bound, the deadman's own cleanup, an unsafe cleanup kind and run id, a state of
another run, the module guard -- plus the real ``run``/``emit``/``tool_present``
wrappers, ``host_command`` without ``nsenter``, the reading's JSON line, the
mount-table parser and the script's ``__main__``.
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_shared_probe as sharedp
from tests.regional import test_general_live_shared_probe as shared_tests

#: The FakeHost world of test_general_live_shared_probe, reused as it is.
host = shared_tests.host
FakeHost = shared_tests.FakeHost
RUN_ID = shared_tests.RUN_ID
mount = shared_tests.mount
call = shared_tests.call
target = shared_tests.target


def _error(host: FakeHost, argv: list[str]) -> dict[str, Any]:
    assert sharedp.main(argv) == 1, host.emitted[-1]
    return host.emitted[-1]


def _answering(host: FakeHost, tool: str, answer: Any) -> None:
    """``tool`` answered by ``answer(args)``; every other command as before."""

    original = host.answer

    def answer_of(inner: list[str]) -> tuple[int, str]:
        if inner[0] == tool:
            return answer(inner[1:])
        return original(inner)

    host.answer = answer_of  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# The real wrappers, the readings and the mount table
# --------------------------------------------------------------------------- #
def test_the_real_wrappers_run_emit_and_find_tools(
    capsys: pytest.CaptureFixture[str],
) -> None:
    failing = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('nope'); sys.exit(3)",
    ]
    with pytest.raises(sharedp.ProbeError, match=r"command failed \(3\): .*: nope"):
        sharedp.run(failing)
    assert sharedp.tool_present(sys.executable) is True
    assert sharedp.tool_present("gfrs-no-such-tool") is False
    sharedp.emit({"b": 1, "a": [1, 2]})
    assert capsys.readouterr().out == '{"a":[1,2],"b":1}\n'
    assert datetime.fromisoformat(sharedp.utc_now()).utcoffset() is not None, (
        "the probe stamps in UTC"
    )


def test_host_commands_need_nsenter_and_a_reading_is_its_last_json_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sharedp, "tool_present", lambda name: name != "nsenter")
    with pytest.raises(sharedp.ProbeError, match="nsenter is missing"):
        sharedp.host_command(["mount"])
    seen: list[tuple[list[str], float]] = []

    def host_command(
        command: list[str], *, check: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        seen.append((command, timeout))
        return subprocess.CompletedProcess(
            command, 0, 'warming up\n{"total_bytes": 4096}\n', ""
        )

    monkeypatch.setattr(sharedp, "host_command", host_command)
    reading = sharedp.host_python(sharedp.STATVFS_SNIPPET, "/mnt/x", timeout=10_000)
    assert reading == {"total_bytes": 4096}
    assert seen[0][1] == float(sharedp.READING_MAX_SECONDS), "the reading is capped"


def test_the_mount_table_parser_skips_what_it_cannot_read(tmp_path: Path) -> None:
    table = tmp_path / "mountinfo"
    table.write_text(
        "22 1 259:1 / / rw,relatime shared:1 - xfs /dev/nvme0n1p1 rw\n"
        "not a mountinfo line\n"
        "40 22 0:40 / /mnt/cut rw -\n"
        "41 22 0:41 / /mnt/with\\040space rw - nfs4 10.0.9.7:/a rw,vers=4.1\n"
    )
    assert [(item["mountpoint"], item["fstype"]) for item in sharedp.mounts(table)] == [
        ("/", "xfs"),
        ("/mnt/with space", "nfs4"),
    ]
    assert sharedp.mounts(tmp_path / "absent") == [], "an unreadable table lists none"


def test_the_script_answers_help_as_a_standalone_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [sharedp.__file__, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(sharedp.__file__, run_name="__main__")
    assert stopped.value.code == 0
    assert "nfs-mount" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# nfs-mount refusals
# --------------------------------------------------------------------------- #
def test_a_mount_without_mount_nfs4_or_twice_of_one_name_is_refused(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sharedp, "tool_present", lambda name: name != "mount.nfs4")
    assert (
        _error(host, mount())["error_message"] == "mount.nfs4 is missing on this node"
    )
    assert host.calls == [], "refused before any host command"
    monkeypatch.setattr(sharedp, "tool_present", lambda name: True)
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    assert _error(host, mount())["error_message"] == (
        "a test shared mount of this name already exists"
    )


def test_a_mount_point_the_host_already_mounts_is_refused(host: FakeHost) -> None:
    host.mounts[target()] = "10.0.9.8:/other"
    host.write_mountinfo()
    assert _error(host, mount())["error_message"] == (
        "the mount point is already mounted on the host"
    )
    assert not any(call[0] == "systemd-run" for call in host.calls), "no deadman armed"


def test_an_nfs_module_that_will_not_load_refuses_after_recording_it(
    host: FakeHost,
) -> None:
    (host.sys_module / sharedp.NFS_MODULE).rmdir()
    _answering(host, "modprobe", lambda args: (1, ""))
    error = _error(host, mount("--load-module"))
    assert error["error_message"] == f"{sharedp.NFS_MODULE} could not be loaded"
    assert sharedp.load_state(RUN_ID)["modules"] == [sharedp.NFS_MODULE], (
        "the attempted load is recorded so the cleanup can undo it"
    )


def test_a_mount_the_host_table_never_shows_is_refused(host: FakeHost) -> None:
    _answering(host, "mount", lambda args: (0, ""))
    assert _error(host, mount())["error_message"] == (
        "the export is not mounted in the host mount table"
    )
    assert sharedp.load_state(RUN_ID)["mounts"]["shared"]["mounted"] is False


def test_a_deadman_that_does_not_arm_refuses_the_mount(host: FakeHost) -> None:
    def show(args: list[str]) -> tuple[int, str]:
        if args[0] == "show":
            return 0, "a stray line\nLoadState=loaded\nActiveState=inactive\n"
        return FakeHost.systemctl(host, args)

    host.systemctl = show  # type: ignore[method-assign]
    error = _error(host, mount())
    assert error["error_message"] == f"deadman {shared_tests.DEADMAN} is not armed"
    assert target() not in host.mounts, "nothing is mounted without the deadman"


# --------------------------------------------------------------------------- #
# nfs-umount / nfs-remount / nfs-stat
# --------------------------------------------------------------------------- #
def test_an_export_that_will_not_unmount_even_lazily_is_an_error(
    host: FakeHost,
) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    _answering(host, "umount", lambda args: (32, "target is busy"))
    error = _error(host, call("nfs-umount", "--name", "shared"))
    assert error["error_message"] == "the test shared mount could not be unmounted"
    assert target() in host.mounts


def test_a_forgotten_mount_whose_directory_is_gone_is_still_forgotten(
    host: FakeHost,
) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    Path(target()).rmdir()
    assert sharedp.main(call("nfs-umount", "--name", "shared", "--forget")) == 0
    receipt = host.emitted[-1]
    assert receipt["forgotten"] is True and receipt["directory_present"] is False
    assert sharedp.load_state(RUN_ID)["mounts"] == {}


def test_a_remount_of_a_live_mount_only_records_it(host: FakeHost) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    mounts_before = sum(
        1 for item in host.calls if item[len(shared_tests.NS) :][:1] == ["mount"]
    )
    assert sharedp.main(call("nfs-remount", "--name", "shared")) == 0
    mounts_after = sum(
        1 for item in host.calls if item[len(shared_tests.NS) :][:1] == ["mount"]
    )
    assert mounts_after == mounts_before, "a live mount is not mounted twice"
    assert host.emitted[-1]["host_mount"]["fstype"] == "nfs4"


def test_a_stat_bound_outside_the_range_is_refused(host: FakeHost) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    error = _error(host, call("nfs-stat", "--name", "shared", "--timeout", "0.5"))
    assert error["error_message"] == "stat timeout is outside 1..120"


# --------------------------------------------------------------------------- #
# cleanup, state and the module guard
# --------------------------------------------------------------------------- #
def test_the_deadmans_own_cleanup_never_stops_the_unit_it_runs_in(
    host: FakeHost,
) -> None:
    assert sharedp.main(mount()) == 0, host.emitted[-1]
    own = f"{shared_tests.DEADMAN}.service"

    def show(args: list[str]) -> tuple[int, str]:
        if args[0] == "show" and args[1] == own:
            return 0, f"LoadState=loaded\nActiveState=active\nMainPID={os.getpid()}\n"
        return FakeHost.systemctl(host, args)

    host.systemctl = show  # type: ignore[method-assign]
    assert sharedp.main(call("cleanup", "--kind", "nfs-shared")) == 0
    assert host.emitted[-1]["clean"] is True, host.emitted[-1]
    stops = [item for item in host.calls if item[:2] == ["systemctl", "stop"]]
    assert ["systemctl", "stop", own] not in stops, stops
    assert ["systemctl", "stop", f"{shared_tests.DEADMAN}.timer"] in stops


def test_an_unsafe_cleanup_kind_run_id_or_foreign_state_is_refused(
    host: FakeHost,
) -> None:
    assert _error(host, call("cleanup", "--kind", "nfs-Shared"))["error_message"] == (
        "unsafe cleanup kind"
    )
    unsafe = ["residue", "--run-id", "Not A Run"]
    assert _error(host, unsafe)["error_message"] == "unsafe run ID"
    sharedp.save_state(RUN_ID, {"run_id": "gl005-other-a1-cafe", "mounts": {}})
    assert _error(host, call("residue"))["error_message"] == (
        "shared probe state identity mismatch"
    )
    assert host.calls == [], "no refusal reached the host"


def test_the_module_is_never_unloaded_under_a_live_mount(host: FakeHost) -> None:
    sharedp.save_state(
        RUN_ID,
        {
            "run_id": RUN_ID,
            "mounts": {"shared": {"mounted": True}},
            "modules": [sharedp.NFS_MODULE],
        },
    )
    assert sharedp.unload_modules(RUN_ID) == []
    assert (host.sys_module / sharedp.NFS_MODULE).is_dir(), "the module stays loaded"
    (host.sys_module / "sunrpc").mkdir()
    (host.sys_module / sharedp.NFS_MODULE).rmdir()
    sharedp.save_state(
        RUN_ID,
        {"run_id": RUN_ID, "mounts": {}, "modules": [sharedp.NFS_MODULE, "sunrpc"]},
    )
    assert sharedp.unload_modules(RUN_ID) == [sharedp.NFS_MODULE], (
        "a module someone else already unloaded is done; another is never touched"
    )
    assert sharedp.load_state(RUN_ID)["modules"] == ["sunrpc"]
    assert not any(item[:1] == ["modprobe"] for item in host.calls), host.calls
    assert sharedp.residue(RUN_ID)["mounts"] == [], "nothing of the run is mounted"
