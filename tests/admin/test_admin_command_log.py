from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_fault.admin.command_log import (
    ADMIN_LOG_ENVIRONMENT,
    command_log,
    prune_admin_logs,
)


@pytest.fixture(autouse=True)
def _outermost_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test the fresh outer context it assumes.

    The release gates run *inside* ``gpu-fault-admin deploy``, which exports
    ``GPU_FAULT_ADMIN_LOG`` so that a nested invocation appends to the outer log
    rather than opening its own. Inheriting that here sends ``command_log`` down
    the reuse branch, which installs no tee at all -- so the tests that assert on
    captured output silently assert nothing. That is exactly how three of these
    passed on a developer machine and failed in a real deployment. The nested
    case sets the variable itself.
    """

    monkeypatch.delenv(ADMIN_LOG_ENVIRONMENT, raising=False)


def test_the_log_captures_subprocess_output_not_just_python_prints(
    tmp_path: Path,
) -> None:
    """The output worth reading back is produced by ``kubectl`` and ``aws``.

    A deploy narrates a few lines itself and shells out for everything else, so a
    tee installed on ``sys.stdout`` would record the narration and drop the
    preflight report and the error that ended the run.
    """

    with command_log(tmp_path, command="deploy") as path:
        subprocess.run(
            [sys.executable, "-c", "print('from-a-child-process')"], check=True
        )
        # Written through the descriptors rather than ``print`` because pytest
        # replaces ``sys.stdout`` with a Python-level object that never touches
        # fd 1; in a real run those are the same thing.
        os.write(1, b"from-the-parent\n")
        os.write(2, b"on-stderr\n")

    assert path is not None
    recorded = path.read_text(encoding="utf-8")
    assert "from-a-child-process" in recorded, (
        "subprocess output never reached the log, so the tee is not at the "
        "file-descriptor level"
    )
    assert "from-the-parent" in recorded
    assert "on-stderr" in recorded


def test_the_log_names_itself_and_stays_private(tmp_path: Path) -> None:
    with command_log(tmp_path, command="deploy") as path:
        pass

    assert path is not None
    assert f"logging to {path}" in path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o077 == 0, (
        "the log sits beside cluster tokens and kubeconfigs, so it must not be "
        "group or world readable"
    )
    assert path.parent.stat().st_mode & 0o077 == 0


def test_console_output_still_reaches_the_terminal(tmp_path: Path) -> None:
    """Logging must not silence the run an administrator is watching."""

    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from gpu_fault.admin.command_log import command_log\n"
        "with command_log(Path(%r), command='deploy'):\n"
        "    print('still-on-the-console')\n"
        % (str(Path("src").resolve()), str(tmp_path))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert "still-on-the-console" in completed.stdout


def test_a_nested_invocation_appends_to_the_outer_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One operator action must produce one file.

    ``gpu-fault-admin deploy`` re-enters this same CLI from the deploy-host venv,
    so opening a second log there would split the story of a single deployment
    across two files and hide the half that failed.
    """

    outer = tmp_path / "logs" / "deploy-outer.log"
    outer.parent.mkdir(mode=0o700)
    outer.write_text("outer\n", encoding="utf-8")
    monkeypatch.setenv(ADMIN_LOG_ENVIRONMENT, str(outer))

    with command_log(tmp_path, command="release-deploy") as path:
        pass

    assert path == outer
    assert list(outer.parent.iterdir()) == [outer]


def test_a_command_without_managed_state_logs_nowhere(tmp_path: Path) -> None:
    """Some commands take no ``--state-dir``, and there is no private home then."""

    with command_log(None, command="status") as path:
        pass

    assert path is None
    assert not (tmp_path / "logs").exists(), (
        "a command with no managed state directory still created a log directory"
    )


def test_pruning_keeps_the_newest_logs(tmp_path: Path) -> None:
    directory = tmp_path / "logs"
    directory.mkdir()
    for index in range(6):
        path = directory / f"deploy-{index}.log"
        path.write_text("x", encoding="utf-8")
        os.utime(path, (index, index))
    (directory / "keep-me.json").write_text("{}", encoding="utf-8")

    prune_admin_logs(directory, retained=2)

    assert sorted(item.name for item in directory.iterdir()) == [
        "deploy-4.log",
        "deploy-5.log",
        "keep-me.json",
    ]


def _sized_logs(directory: Path, sizes: list[int]) -> None:
    for index, size in enumerate(sizes):
        path = directory / f"deploy-{index}.log"
        path.write_text("x" * size, encoding="utf-8")
        os.utime(path, (index, index))


def test_pruning_bounds_the_total_size_as_well_as_the_count(tmp_path: Path) -> None:
    """A count bounds nothing when one run is a thousand times another.

    These logs tee whole ``kubectl`` and ``aws`` outputs, so a fleet rollout is
    tens of megabytes while a ``status`` run is kilobytes -- and the directory they
    fill is the one holding the kubeconfigs and cluster tokens the next command
    needs.
    """

    directory = tmp_path / "logs"
    directory.mkdir()
    _sized_logs(directory, [100, 100, 100, 100])

    prune_admin_logs(directory, retained=10, max_bytes=250)

    assert sorted(item.name for item in directory.iterdir()) == [
        "deploy-2.log",
        "deploy-3.log",
    ], "the newest logs are kept until the ceiling is reached"


def test_the_log_being_written_is_never_pruned_for_size(tmp_path: Path) -> None:
    """A single run larger than the ceiling is still the run being diagnosed."""

    directory = tmp_path / "logs"
    directory.mkdir()
    _sized_logs(directory, [100, 4096])

    prune_admin_logs(directory, retained=10, max_bytes=256)

    assert [item.name for item in directory.iterdir()] == ["deploy-1.log"], (
        "the newest file survives its own size; the older one does not"
    )


def test_the_tail_of_the_output_survives_the_shutdown(tmp_path: Path) -> None:
    """Closing the log must not race the copy still in flight.

    The pump threads write the passthrough copy to the saved descriptors, so
    closing those before the pipes have drained drops however much was still
    buffered -- and the end of a run is where the error is.
    """

    lines = [f"line-{index}" for index in range(2000)]
    with command_log(tmp_path, command="deploy") as path:
        os.write(1, ("\n".join(lines) + "\n").encode("utf-8"))

    assert path is not None
    recorded = path.read_text(encoding="utf-8")
    assert recorded.count("line-") == len(lines), (
        f"the log lost part of the output: kept {recorded.count('line-')} of "
        f"{len(lines)} lines"
    )
    assert lines[-1] in recorded
