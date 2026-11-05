"""The CRI Pod-log replay probe's collector window, driven through ``main(argv)``.

``open-window`` points the node log collector at this run's fixture glob
through a ``/run`` drop-in, ``restart-collector``/``status``/``snapshot`` read
it back and ``close-window`` restores the baseline; the host is ``FakeSystemd``
plus ``tmp_path`` roots (``_cov95_general_live_applog``). Each test checks the
probe's JSON (or its failure line and reason token), the files it wrote and the
systemd calls in order.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_applog_probe as probe
from tests.regional._cov95_general_live_applog import (
    CONTAINER,
    LOG,
    NAMESPACE,
    OLD_UID,
    POD,
    RUN_ID,
    fixture_name,
    install_host,
    record_flags,
)
from tests.regional._cov95_general_live_systemd import unit_properties

BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
Cli = Callable[..., tuple[int, dict[str, Any], str]]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
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


def open_window(cli: Cli, *extra: str) -> dict[str, Any]:
    code, answer, err = cli(
        "open-window",
        "--run-id",
        RUN_ID,
        "--container",
        CONTAINER,
        "--approved",
        *extra,
    )
    assert code == 0, f"the window opens: {err}"
    return answer


def deadman() -> str:
    return probe.deadman_unit(RUN_ID)


def failed(answer: dict[str, Any], command: str, reason: str) -> bool:
    return (
        answer["error_class"] == "ProbeError"
        and answer["error_kind"] == "probe"
        and answer["command"] == command
        and answer["error_reason"] == reason
    )


def old_pod_log(cli: Cli) -> dict[str, Any]:
    """``create-log`` for an old Pod: the probe makes its directory."""

    code, answer, err = cli(
        "create-log",
        "--run-id",
        RUN_ID,
        "--pod-uid",
        OLD_UID,
        "--container",
        CONTAINER,
        "--namespace",
        NAMESPACE,
        "--pod-name",
        POD,
        "--allow-create-dir",
    )
    assert code == 0, f"the old Pod's fixture is created: {err}"
    return answer


def write(cli: Cli, records: list[dict[str, str]], *extra: str) -> Any:
    sha256, encoded = record_flags(records)
    return cli(
        "write",
        "--run-id",
        RUN_ID,
        "--pod-uid",
        OLD_UID,
        "--container",
        CONTAINER,
        "--sha256",
        sha256,
        "--b64",
        encoded,
        *extra,
    )


# --------------------------------------------------------------------------- #
# open-window
# --------------------------------------------------------------------------- #
def test_open_window_points_the_collector_at_this_runs_fixtures_only(
    host: SimpleNamespace, cli: Cli
) -> None:
    answer = open_window(cli, "--initial-tail-bytes", "4096")
    glob = f"{host.pods}/*/{CONTAINER}/{fixture_name()}*"
    assert answer["glob"] == glob, answer
    assert answer["enabled_log_collector"] is True, "the run enabled the collector"
    assert answer["after"]["ActiveState"] == "active", answer["after"]
    assert answer["deadman_timer"]["ActiveState"] == "active", answer
    assert answer["proof_scope"] == "live-replay", answer
    assert answer["boot_id"] == BOOT_ID.read_text(encoding="ascii").strip(), answer

    env_file = Path(answer["env_file"])
    assert env_file.read_text(encoding="utf-8") == (
        f"{probe.PATHS_ENV}={glob}\n{probe.INITIAL_TAIL_ENV}=4096\n"
    ), "the second EnvironmentFile carries the glob and the initial tail"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o644, "readable by the unit"
    assert Path(answer["dropin"]).read_text(encoding="utf-8") == (
        f"[Service]\nEnvironmentFile={env_file}\n"
    ), "the drop-in adds the run's environment file after the installed one"

    name = deadman()
    mutations = host.systemd.mutations()
    systemd_run = mutations[2]
    copy = probe.state_dir(RUN_ID) / "probe.py"
    assert systemd_run[:3] == ["systemd-run", f"--unit={name}", "--on-active=900s"]
    assert systemd_run[-5:] == [
        sys.executable,
        str(copy),
        "close-window",
        "--run-id",
        RUN_ID,
    ]
    assert filecmp.cmp(copy, probe.__file__, shallow=False), "the deadman's copy"
    assert [mutations[:2], mutations[3:]] == [
        [
            ["systemctl", "stop", f"{name}.timer"],
            ["systemctl", "reset-failed", f"{name}.service"],
        ],
        [
            ["systemctl", "enable", "--runtime", LOG],
            ["systemctl", "daemon-reload"],
            ["systemctl", "restart", LOG],
        ],
    ], "deadman first, then the drop-in takes effect on a restart"


def test_open_window_keeps_an_enabled_collector_and_close_restarts_it(
    host: SimpleNamespace, cli: Cli
) -> None:
    host.systemd.units[LOG] = unit_properties(active="active", file_state="enabled")
    answer = open_window(cli)
    assert answer["enabled_log_collector"] is False, answer
    assert Path(answer["env_file"]).read_text(encoding="utf-8") == (
        f"{probe.PATHS_ENV}={answer['glob']}\n"
    ), "no initial tail asked, none set"
    assert not any(call[1] == "enable" for call in host.systemd.mutations()), (
        "an enabled collector is not enabled again"
    )

    host.systemd.calls.clear()
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert closed["removed"] == {"files": [], "directories": [], "absent": []}, closed
    assert closed["after"]["ActiveState"] == "active", "restarted on the baseline"
    assert host.systemd.mutations()[:2] == [
        ["systemctl", "daemon-reload"],
        ["systemctl", "restart", LOG],
    ], "no disable and no stop for a collector the run found enabled"


def already_open(host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    probe.save_state(RUN_ID, {"run_id": RUN_ID})


def untracked_dropin(host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    probe.dropin_path(RUN_ID).parent.mkdir(parents=True)
    probe.dropin_path(RUN_ID).write_text("[Service]\n", encoding="utf-8")


def unit_missing(host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    host.systemd.units[LOG] = unit_properties(load="not-found", file_state="")


def no_pod_logs(host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "POD_LOG_ROOT", host.root / "elsewhere")


def nothing(host: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    return None


@pytest.mark.parametrize(
    "prepare, extra, reason, message",
    [
        (nothing, ["--approved", "--initial-tail-bytes", "-1"], "input_shape", "tail"),
        (
            nothing,
            [
                "--approved",
                "--initial-tail-bytes",
                str(probe.MAX_INITIAL_TAIL_BYTES + 1),
            ],
            "input_shape",
            "initial tail bytes is outside 0..1048576",
        ),
        (
            nothing,
            ["--approved", "--restore-seconds", "30"],
            "input_shape",
            "restore seconds is outside 60..5400",
        ),
        (nothing, [], "not_approved", "ENABLE_NODE_LOG_COLLECTOR approval"),
        (already_open, ["--approved"], "window_open", "already open"),
        (untracked_dropin, ["--approved"], "dropin_untracked", "untracked"),
        (unit_missing, ["--approved"], "unit_missing", "not installed"),
        (no_pod_logs, ["--approved"], "pod_log_root_missing", "does not exist"),
    ],
    ids=[
        "negative-tail",
        "huge-tail",
        "short-restore",
        "unapproved",
        "already-open",
        "untracked-dropin",
        "unit-missing",
        "no-pod-log-root",
    ],
)
def test_open_window_refuses_before_any_host_change(
    host: SimpleNamespace,
    cli: Cli,
    monkeypatch: pytest.MonkeyPatch,
    prepare: Callable[[SimpleNamespace, pytest.MonkeyPatch], None],
    extra: list[str],
    reason: str,
    message: str,
) -> None:
    prepare(host, monkeypatch)
    code, answer, err = cli(
        "open-window", "--run-id", RUN_ID, "--container", CONTAINER, *extra
    )
    assert code == 1 and failed(answer, "open-window", reason), answer
    assert message in answer["error_message"] and message in err, answer
    assert host.systemd.mutations() == [], "a refused window changes nothing"


def test_a_failing_deadman_launch_is_a_command_failure(
    host: SimpleNamespace, cli: Cli
) -> None:
    host.systemd.failing["systemd-run"] = 1
    code, answer, _ = cli(
        "open-window", "--run-id", RUN_ID, "--container", CONTAINER, "--approved"
    )
    assert code == 1 and failed(answer, "open-window", "command_failed"), answer
    assert answer["error_message"].startswith(
        f"command failed (1): systemd-run --unit={deadman()} "
    ), answer
    assert answer["error_message"].endswith("Failed to run systemd-run: unit busy"), (
        "the tool's stderr is carried in the message"
    )
    assert str(answer["error_site"]).startswith("run:"), answer
    assert not probe.dropin_path(RUN_ID).exists(), "no drop-in without a deadman"


def test_a_deadman_timer_that_never_activates_fails_the_window(
    host: SimpleNamespace, cli: Cli
) -> None:
    host.systemd.timer_starts = False
    code, answer, _ = cli(
        "open-window", "--run-id", RUN_ID, "--container", CONTAINER, "--approved"
    )
    assert code == 1 and failed(answer, "open-window", "deadman_inactive"), answer
    assert ["systemctl", "daemon-reload"] not in host.systemd.calls, (
        "the collector is never touched without an armed deadman"
    )


def test_a_collector_that_does_not_come_back_leaves_the_window_to_close(
    host: SimpleNamespace, cli: Cli
) -> None:
    host.systemd.restart_leaves = "failed"
    code, answer, _ = cli(
        "open-window", "--run-id", RUN_ID, "--container", CONTAINER, "--approved"
    )
    assert code == 1 and failed(answer, "open-window", "unit_inactive"), answer
    code, snapshot, _ = cli("snapshot", "--run-id", RUN_ID)
    assert snapshot == {
        "run_id": RUN_ID,
        "window_open": True,
        "residue": {
            "dropin": True,
            "fixture_files": False,
            "created_directories": False,
            "log_collector_enabled_by_run": True,
            "deadman": True,
        },
    }, "the drop-in, the runtime enable and the deadman are what is left"

    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert closed["after"]["ActiveState"] == "inactive", (
        "a collector the run enabled is stopped again, whatever state it was in"
    )
    assert cli("snapshot", "--run-id", RUN_ID)[1]["residue"] == {"window": False}


# --------------------------------------------------------------------------- #
# a window with an old Pod's fixture: status, restart, close
# --------------------------------------------------------------------------- #
def test_status_reads_back_the_fixture_files_and_the_collector_state(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_window(cli)
    created = old_pod_log(cli)
    assert write(cli, [{"message": "step 1 loss 0.5"}, {"message": "step 2"}])[0] == 0
    code, rotated, _ = cli(
        "rotate", "--run-id", RUN_ID, "--pod-uid", OLD_UID, "--container", CONTAINER
    )
    assert code == 0, rotated
    assert write(cli, [{"message": "after rotation"}], "--target", "rotated")[0] == 0

    code, answer, _ = cli("status", "--run-id", RUN_ID)
    assert code == 0, answer
    key = created["key"]
    files = answer["files"][key]
    assert files["writes"] == 3 and files["directory_exists"] is True, files
    assert files["current"]["exists"] is True and files["current"]["size"] == 0
    assert [item["path"] for item in files["rotated"]] == [rotated["rotated"]["path"]]
    assert files["rotated"][0]["size"] > rotated["rotated"]["size"], (
        "the rotated inode kept growing after the rename"
    )
    assert answer["glob"].endswith(f"/{CONTAINER}/{fixture_name()}*"), answer
    assert answer["collector"]["readable"] is False, "no collector state file yet"
    assert answer["unit"]["ActiveState"] == "active", answer["unit"]


def test_restart_collector_reports_both_invocations_and_needs_it_back(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_window(cli)
    code, answer, _ = cli("restart-collector", "--run-id", RUN_ID)
    assert code == 0, answer
    assert answer["before"]["InvocationID"] != answer["after"]["InvocationID"], (
        "the restart starts a new invocation that must reload its state"
    )
    host.systemd.restart_leaves = "failed"
    code, answer, _ = cli("restart-collector", "--run-id", RUN_ID)
    assert code == 1 and failed(answer, "restart-collector", "unit_inactive"), answer


def test_close_window_removes_exactly_what_the_run_made(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_window(cli)
    created = old_pod_log(cli)
    code, rotated, _ = cli(
        "rotate", "--run-id", RUN_ID, "--pod-uid", OLD_UID, "--container", CONTAINER
    )
    current = created["file"]["path"]
    pod_dir, container_dir = created["created_dirs"]
    host.systemd.calls.clear()

    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    name = deadman()
    assert code == 0 and closed["clean"] is True, closed
    assert closed["removed"] == {
        "files": [current, rotated["rotated"]["path"]],
        "directories": [container_dir, pod_dir],
        "absent": [],
    }, "files first, then the directories the run created, innermost first"
    assert closed["dropin_removed"] is True, closed
    assert host.systemd.mutations() == [
        ["systemctl", "disable", "--runtime", LOG],
        ["systemctl", "daemon-reload"],
        ["systemctl", "stop", LOG],
        ["systemctl", "stop", f"{name}.timer"],
        ["systemctl", "stop", f"{name}.service"],
        ["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"],
    ], host.systemd.mutations()
    assert not Path(pod_dir).exists(), "the old Pod's directory is gone again"
    assert not probe.state_dir(RUN_ID).exists(), "the window state is removed"


def test_close_window_of_a_collector_it_found_enabled_needs_it_back(
    host: SimpleNamespace, cli: Cli
) -> None:
    host.systemd.units[LOG] = unit_properties(active="active", file_state="static")
    open_window(cli)
    host.systemd.restart_leaves = "failed"
    code, answer, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 1 and failed(answer, "close-window", "unit_inactive"), answer
    assert "is not active after the window closed" in answer["error_message"]
    assert probe.load_state(RUN_ID), "the window stays tracked for a retry"


def test_close_window_run_by_the_deadman_does_not_stop_its_own_service(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_window(cli)
    name = deadman()
    host.systemd.unit(f"{name}.service")["MainPID"] = str(os.getpid())
    host.systemd.calls.clear()
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert ["systemctl", "stop", f"{name}.service"] not in host.systemd.calls, (
        "the deadman's own close never stops the service it runs in"
    )


def test_close_window_tolerates_the_deadman_removing_the_state_first(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_window(cli)
    state_dir = probe.state_dir(RUN_ID)
    host.systemd.on_stop[f"{deadman()}.service"] = lambda: shutil.rmtree(state_dir)
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert not state_dir.exists(), "the state directory stays gone"


def test_close_window_without_a_tracked_window_sweeps_a_stray_drop_in(
    host: SimpleNamespace, cli: Cli
) -> None:
    probe.probe_copy(RUN_ID)  # a state directory without a window
    dropin = probe.dropin_path(RUN_ID)
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\n", encoding="utf-8")
    expected = {"run_id": RUN_ID, "already_closed": True, "clean": True}
    code, answer, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and answer == expected, answer
    assert not dropin.exists() and not probe.state_dir(RUN_ID).exists(), (
        "the stray drop-in and the untracked state directory are removed"
    )
    assert host.systemd.calls == [], "nothing tracked: the unit is not touched"
    assert cli("close-window", "--run-id", RUN_ID)[:2] == (0, expected), "idempotent"


# --------------------------------------------------------------------------- #
# window identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window", [{"run_id": "gl017-other-run"}, ["a", "list"]])
def test_a_window_of_another_run_is_refused(
    host: SimpleNamespace, cli: Cli, window: object
) -> None:
    path = probe.state_dir(RUN_ID) / "window.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(window), encoding="utf-8")
    code, answer, _ = cli("snapshot", "--run-id", RUN_ID)
    assert code == 1 and failed(answer, "snapshot", "window_identity"), answer


@pytest.mark.parametrize("command", ["status", "restart-collector"])
def test_window_commands_need_an_open_window(
    host: SimpleNamespace, cli: Cli, command: str
) -> None:
    code, answer, _ = cli(command, "--run-id", RUN_ID)
    assert code == 1 and failed(answer, command, "no_window"), answer
    assert host.systemd.calls == [], "nothing is read or restarted without a window"
