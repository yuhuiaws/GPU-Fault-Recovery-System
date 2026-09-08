from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import command_log as command_log_module
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.command_log import (
    ADMIN_LOG_ENVIRONMENT,
    child_failure,
    command_log,
    last_output_line,
    prune_admin_logs,
    report_failure,
)
from scripts import staging_deploy


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

    assert path is None, (
        "a nested invocation owns no log; handing it the outer path is what made "
        "it announce the file a second time"
    )
    assert list(outer.parent.iterdir()) == [outer]


def test_a_failure_announces_the_log_once_across_nested_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failed deploy, one ``full output in`` line.

    ``gpu-fault-admin deploy`` re-enters itself from the deploy-host venv, and
    each level used to name the log on its way out: the operator read three
    identical lines under four nested ``command failed`` lines. Only the process
    that opened the log knows the path now, so only it speaks.
    """

    class Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(command="deploy", state_dir=tmp_path)

    def run(_arguments: argparse.Namespace) -> int:
        if os.environ.get(ADMIN_LOG_ENVIRONMENT):
            return 2  # the innermost level: the failure itself
        return admin_cli.main()  # the re-entered CLI, inheriting the log

    monkeypatch.setattr(admin_cli, "parser", Parser)
    monkeypatch.setattr(admin_cli, "enforce_deploy_host_state_dir", lambda _a: None)
    monkeypatch.setattr(admin_cli, "run", run)

    assert admin_cli.main() == 2

    logs = list((tmp_path / "logs" / "mutating").glob("*.log"))
    assert len(logs) == 1, "the nested invocation opened a log of its own"
    recorded = logs[0].read_text(encoding="utf-8")
    assert recorded.count("full output in") == 1, recorded
    assert recorded.count("logging to") == 1, recorded


def test_a_child_that_is_our_own_driver_is_not_wrapped_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The deploy is five wrappers deep; only the innermost knows the cause.

    Each layer used to add ``command failed (N): <interpreter path>``, so the
    tail of a failed deploy was four lines naming a venv python while the cause
    sat hundreds of lines above. A Python module, ``gpu-fault-admin`` or ``make``
    has already said why it stopped; the wrapper passes its status through and
    prints nothing.
    """

    error = child_failure(
        RuntimeError,
        ["/x/bundle-abc/bin/python", "-m", "gpu_fault_release.rollout", "deploy"],
        3,
    )

    assert "command failed" not in str(error)
    assert "gpu_fault_release.rollout" in str(error), (
        "the message that survives a re-wrap should name the module, not python"
    )
    assert report_failure("release-deploy", error) == 3
    assert capsys.readouterr().err == ""
    assert report_failure("x", child_failure(RuntimeError, ["make", "check"], 2)) == 2
    assert "make check" in str(child_failure(RuntimeError, ["make", "check"], 2))


def test_a_foreign_command_gets_one_line_naming_the_step_and_its_last_words(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``kubectl`` and ``aws`` do not always say what they were doing.

    ``command failed (1): kubectl`` named the binary and nothing else; the step
    -- which Job, which namespace -- was in the ``+`` echo far above, and the
    reason was in a stderr the wrapper had swallowed.
    """

    error = child_failure(
        RuntimeError,
        [
            "kubectl",
            "--kubeconfig",
            "/s/cpu.kubeconfig",
            "-n",
            "gpu-fault-system",
            "wait",
            "--for=condition=complete",
            "jobs/gpu-fault-aurora-credential-refresh-verify",
        ],
        1,
        detail=last_output_line("warning: x\nerror: timed out waiting\n\n"),
    )

    assert str(error) == (
        "command failed (1): kubectl --kubeconfig /s/cpu.kubeconfig -n "
        "gpu-fault-system wait ...: error: timed out waiting"
    )
    assert report_failure("gpu-fault-admin", error) == 2
    assert capsys.readouterr().err == "gpu-fault-admin: " + str(error) + "\n"
    # A sensitive command shows its program only; its argv is why it is sensitive.
    secret = child_failure(RuntimeError, ["aws", "sts", "x"], 254, sensitive=True)
    assert str(secret) == "command failed (254): aws"


def test_a_failure_propagates_the_first_cause_through_the_admin_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through ``main``: a real child, a real exit status, one line."""

    class Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(command="deploy", state_dir=None)

    def run(_arguments: argparse.Namespace) -> int:
        CommandRunner().run(
            [sys.executable, "-c", "import sys; print('the cause'); sys.exit(4)"],
            capture=False,
        )
        return 0

    monkeypatch.setattr(admin_cli, "parser", Parser)
    monkeypatch.setattr(admin_cli, "enforce_deploy_host_state_dir", lambda _a: None)
    monkeypatch.setattr(admin_cli, "run", run)

    assert admin_cli.main() == 4, "the child's exit status must pass through"
    assert "command failed" not in capsys.readouterr().err


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


def test_logs_are_kept_per_kind_and_a_mutating_run_is_the_default(
    tmp_path: Path,
) -> None:
    """A hundred ``status`` runs must never evict the one deploy log (I3).

    The rotation counted every command against one ceiling, so the logs an
    operator most needs to read back -- a deploy that failed halfway -- were the
    ones the cheap, frequent read-only commands pushed out. Each kind now has its
    own directory and ceiling. A caller that names no kind is treated as
    mutating: the safe error is keeping a read-only log too long, not losing a
    deploy log.
    """

    with command_log(tmp_path, command="deploy") as deploy_path:
        pass
    with command_log(tmp_path, command="status", kind="readonly") as status_path:
        pass

    assert deploy_path is not None and status_path is not None
    root = tmp_path / "logs"
    assert deploy_path.parent == root / "mutating", deploy_path
    assert status_path.parent == root / "readonly", status_path
    for directory in (deploy_path.parent, status_path.parent):
        assert directory.stat().st_mode & 0o077 == 0, (
            f"{directory} must stay private like the state directory"
        )


def test_readonly_pruning_never_touches_the_mutating_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "logs"
    mutating = root / "mutating"
    readonly = root / "readonly"
    mutating.mkdir(parents=True, mode=0o700)
    readonly.mkdir(mode=0o700)
    _sized_logs(mutating, [100] * 5)
    _sized_logs(readonly, [100] * 5)
    monkeypatch.setitem(
        command_log_module.ADMIN_LOG_LIMITS,
        "readonly",
        command_log_module.LogLimits(retained=2, max_bytes=1 << 20),
    )

    with command_log(tmp_path, command="status", kind="readonly"):
        pass

    assert len(list(mutating.glob("*.log"))) == 5, (
        "opening a read-only log pruned the mutating directory"
    )
    kept = sorted(item.name for item in readonly.glob("*.log"))
    assert len(kept) == 2, f"the read-only ceiling was not applied: {kept}"
    assert "deploy-4.log" in kept, "pruning evicted a newer log before an older one"


def test_the_mutating_ceiling_is_higher_than_the_readonly_one() -> None:
    limits = command_log_module.ADMIN_LOG_LIMITS

    assert set(limits) == {"mutating", "readonly"}
    assert limits["mutating"].retained > limits["readonly"].retained, (
        "mutating logs are the ones worth keeping; their count must not be the "
        "smaller one"
    )
    assert limits["mutating"].max_bytes >= limits["readonly"].max_bytes


def test_an_unknown_kind_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind"):
        with command_log(tmp_path, command="deploy", kind="whatever"):
            pass

    assert not (tmp_path / "logs").exists(), (
        "a refused kind still created a log directory"
    )


def test_a_failed_nested_admin_passes_its_status_through_unwrapped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The nested ``gpu-fault-admin`` has already said why; this layer adds nothing.

    ``staging-deploy: command failed (2): /…/deployer-venv/bin/gpu-fault-admin``
    was the second of four such lines under one failure. A foreign command still
    gets its one line, with the step and its last stderr line, and exits 2.
    """

    monkeypatch.setattr(
        staging_deploy, "parser", lambda: argparse.ArgumentParser(add_help=False)
    )

    def nested_admin_failed(_parsed: argparse.Namespace) -> dict[str, object]:
        raise child_failure(
            staging_deploy.StagingDeployError,
            ["/s/deployer-venv/bin/gpu-fault-admin", "deploy", "--state-dir", "/s"],
            3,
        )

    monkeypatch.setattr(staging_deploy, "deploy", nested_admin_failed)
    assert staging_deploy.main([]) == 3
    assert capsys.readouterr().err == ""

    def foreign_failed(_parsed: argparse.Namespace) -> dict[str, object]:
        raise child_failure(
            staging_deploy.StagingDeployError,
            ["git", "fetch", "origin"],
            128,
            detail=last_output_line("fatal: could not read from remote\n"),
        )

    monkeypatch.setattr(staging_deploy, "deploy", foreign_failed)
    assert staging_deploy.main([]) == 2
    assert capsys.readouterr().err == (
        "staging-deploy: command failed (128): git fetch origin: "
        "fatal: could not read from remote\n"
    )
