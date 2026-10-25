"""Host source replay probe: files and tools made unopenable inside the unit.

Why GF-GENERAL-LIVE-002 ``read_failure`` (and 007 ``tool_missing``) cannot use
systemd's ``InaccessiblePaths=`` -- the collector units run as root, and root
reads the mode-0000 node systemd binds there as an *empty* file -- and what the
probe binds instead: its own unopenable node, a driverless 0:0 character
device. The fakes (``FakeHost`` with root semantics, the ``host`` fixture) are
the main probe test module's.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional import test_general_live_source_probe as probe_tests

RUN = probe_tests.RUN
MAIN_PID = probe_tests.MAIN_PID
FakeHost = probe_tests.FakeHost
host = probe_tests.host
unopenable_on_host = probe_tests.unopenable_on_host


def _main(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


def test_root_reads_an_inaccessible_paths_node_as_an_empty_file(
    host: FakeHost, tmp_path: Path
) -> None:
    """Why ``InaccessiblePaths=`` is no read failure for the (root) host collector.

    systemd binds a mode-0000 node over the path; CAP_DAC_OVERRIDE reads the
    regular one as an empty file and CAP_DAC_READ_SEARCH lists the directory one
    as empty. The product would parse an empty ``/proc/meminfo`` or count zero
    GPUs in an empty ``gpus`` -- and the window verification rightly refused
    that on the first live run (``unit_readable`` stayed True).
    """

    directory = tmp_path / "gpus"
    (directory / "0000:53:00.0").mkdir(parents=True)
    host.inaccessible = {"/proc/meminfo", str(directory)}
    unit_file = probe.view_path(MAIN_PID, "/proc/meminfo")
    assert unit_file["readable"] is True and unit_file["value"] == "", unit_file
    unit_dir = probe.view_path(MAIN_PID, str(directory))
    assert unit_dir["readable"] is True and unit_dir["kind"] == "dir", unit_dir
    assert probe.view_path("1", str(directory))["sha256"] != unit_dir["sha256"]


def test_an_inaccessible_file_is_an_unopenable_node_bound_over_it(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """002 ``read_failure``: the unit's ``open("/proc/meminfo")`` must fail.

    The window binds the run's unopenable node (a driverless 0:0 character
    device, mode 0000, under the state dir) read-only over the file instead of
    an ``InaccessiblePaths=`` line; the verification sees ``cat`` fail inside
    the unit and succeed for PID 1, and the close removes the node with the
    state dir.
    """

    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--inaccessible",
        "/proc/meminfo",
        capsys=capsys,
    )
    assert code == 0, opened
    node = probe.unreadable_node_path(RUN)
    assert node.parent == probe.state_dir(RUN) and unopenable_on_host(str(node)), node
    lines = probe.dropin_path(RUN).read_text(encoding="utf-8").splitlines()
    assert lines == ["[Service]", f"BindReadOnlyPaths={node}:/proc/meminfo"], lines
    assert opened["inaccessible"] == ["/proc/meminfo"] and opened["binds"] == []
    (check,) = opened["verification"]["checks"]
    assert check == {
        "kind": "inaccessible",
        "dest": "/proc/meminfo",
        "unit_readable": False,
        "host_readable": True,
        "ok": True,
    }, check
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True
    assert not node.exists() and host.binds == {}


def test_a_hidden_tool_is_unopenable_in_the_unit_only(
    host: FakeHost,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """007 ``tool_missing``: every copy of the tool on the standard PATH gets the
    node bound over it, so ``shutil.which`` inside the unit finds nothing
    executable while the host's binary is untouched."""

    sbin = tmp_path / "usr" / "sbin"
    sbin.mkdir(parents=True)
    smartctl = sbin / "smartctl"
    smartctl.write_text("#!/bin/sh\nexit 0\n")
    smartctl.chmod(0o755)
    monkeypatch.setattr(probe, "STANDARD_PATH", f"{sbin}:/nonexistent/bin")
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--inaccessible",
        "tool:smartctl",
        capsys=capsys,
    )
    assert code == 0, opened
    node = probe.unreadable_node_path(RUN)
    lines = probe.dropin_path(RUN).read_text(encoding="utf-8").splitlines()
    assert lines == ["[Service]", f"BindReadOnlyPaths={node}:{smartctl}"], lines
    (check,) = opened["verification"]["checks"]
    assert check["dest"] == str(smartctl) and check["ok"] is True, check
    assert (
        host.run(
            [
                "nsenter",
                "-t",
                MAIN_PID,
                "-m",
                "--",
                "/usr/bin/test",
                "-x",
                str(smartctl),
            ]
        ).returncode
        == 1
    )
    assert os.access(smartctl, os.X_OK), "the host's binary is never replaced"
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--inaccessible",
        "tool:ipmitool",
        capsys=capsys,
    )
    assert code == 1 and error["error_class"] == "ProbeError", (
        "a tool the host lacks has nothing to hide"
    )
    assert not probe.unreadable_node_path(RUN).exists(), (
        "a refused window creates no node"
    )


def test_unopenable_node_is_a_driverless_character_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real node: ``mknod(S_IFCHR | 0000, makedev(0, 0))`` -- no driver answers
    major 0 (ENXIO) and the state dir's tmpfs is ``nodev`` (EACCES), so the
    open fails for every uid, root's CAP_DAC_OVERRIDE included."""

    made: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        os, "mknod", lambda path, mode, device: made.append((str(path), mode, device))
    )
    probe.make_unopenable_node(tmp_path / "unreadable")
    ((path, mode, device),) = made
    assert path == str(tmp_path / "unreadable")
    assert stat.S_ISCHR(mode) and not mode & 0o777, oct(mode)
    assert device == os.makedev(0, 0)


def test_probe_error_line_carries_class_digest_and_site(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host fixture withholds a failed probe's output and shows only the
    tokens it whitelists for ``error_kind: probe``: the class, a 16-hex digest
    of the message and the ``function:line`` site. The first live
    ``read_failure`` run surfaced none of them ("exit 1; output withheld")."""

    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "x:/etc/passwd",
        capsys=capsys,
    )
    assert code == 1
    assert error["error_kind"] == "probe" and error["error_class"] == "ProbeError"
    assert error["error"] == "ProbeError", "the message itself never leaves the node"
    assert re.fullmatch(r"[a-f0-9]{16}", error["error_code"]), error
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*:[1-9][0-9]*", error["error_site"]), (
        "the site is function:line, the shape host_probe_fixture shows"
    )
    assert error["error_site"].startswith("checked_bind_dest:"), error["error_site"]
