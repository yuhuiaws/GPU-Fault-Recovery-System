"""The kernel-log source replay probe driven end to end through ``main(argv)``.

``probes/general_live_kernel_source_probe.py`` runs on a GPU node; here its host
is ``FakeSystemd`` plus ``tmp_path`` roots (``_cov95_general_live_kernel``), so a
test installs a reviewed record set, opens a journal or kmsg window, arms it,
reads it back and closes it with the probe's own CLI and checks what each step
leaves behind: the window state, the drop-in, the deadman, the shadow
``journalctl`` and the systemd calls in the order the probe made them.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import runpy
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_kernel_source_probe as probe
from tests.regional._cov95_general_live_kernel import (
    KERNEL,
    LOG,
    REAL_LINE,
    RUN_ID,
    SYSTEM_PATH,
    install_host,
    records_flags,
    unit_properties,
)

BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
RECORD = {"message": "NVRM: Xid (PCI:0000:53:00): 79, fell off the bus"}
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


def install(cli: Cli, records: list[dict[str, Any]], run_id: str = RUN_ID) -> Any:
    sha256, encoded = records_flags(records)
    return cli(
        "install-records", "--run-id", run_id, "--sha256", sha256, "--b64", encoded
    )


def deadman() -> str:
    return probe.deadman_unit(RUN_ID)


def refused(answer: dict[str, Any], command: str) -> bool:
    return answer == {"error": "ProbeError", "error_kind": "probe", "command": command}


def open_journal(host: SimpleNamespace, cli: Cli) -> dict[str, Any]:
    assert install(cli, [{**RECORD, "unit": "gpu-train.service"}])[0] == 0, (
        "records install"
    )
    code, answer, err = cli(
        "open-window", "--run-id", RUN_ID, "--mode", "journal", "--approved"
    )
    assert code == 0, f"the journal window opens: {err}"
    return answer


def open_kmsg(host: SimpleNamespace, cli: Cli) -> dict[str, Any]:
    assert install(cli, [RECORD])[0] == 0, "records install"
    code, answer, err = cli("open-window", "--run-id", RUN_ID, "--mode", "kmsg")
    assert code == 0, f"the kmsg window opens: {err}"
    return answer


# --------------------------------------------------------------------------- #
# install-records
# --------------------------------------------------------------------------- #
def test_install_records_stores_the_validated_set_owner_only(
    host: SimpleNamespace, cli: Cli
) -> None:
    records = [RECORD, {"message": "user wrote this", "transport": "syslog"}]
    code, answer, _ = install(cli, records)
    assert code == 0, answer
    assert answer == {
        "run_id": RUN_ID,
        "records": 2,
        "sha256": records_flags(records)[0],
        "transports": ["kernel", "syslog"],
    }, answer
    stored = probe.records_path(RUN_ID)
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600, "records are owner-only"
    assert json.loads(stored.read_text(encoding="utf-8")) == probe.validate_records(
        records
    ), "the stored set is the validated set, defaults filled in"

    code, answer, err = install(cli, records)
    assert code == 1 and refused(answer, "install-records"), answer
    assert "already installed for this run" in err, err


@pytest.mark.parametrize(
    "sha256, encoded, message",
    [
        ("abc", "W10=", "a sha256 is required"),
        ("0" * 64, "not base64!", "records are not valid base64"),
        ("0" * 64, records_flags([RECORD])[1], "sha256 does not match"),
    ],
    ids=["malformed-digest", "bad-base64", "digest-mismatch"],
)
def test_install_records_refuses_an_unverifiable_set(
    host: SimpleNamespace, cli: Cli, sha256: str, encoded: str, message: str
) -> None:
    code, answer, err = cli(
        "install-records", "--run-id", RUN_ID, "--sha256", sha256, "--b64", encoded
    )
    assert code == 1 and refused(answer, "install-records"), answer
    assert message in err, err
    assert not probe.records_path(RUN_ID).exists(), "nothing is installed"


def test_validate_records_refuses_a_record_that_is_not_an_object() -> None:
    with pytest.raises(probe.ProbeError, match="must be an object"):
        probe.validate_records(["NVRM: Xid 79"])


# --------------------------------------------------------------------------- #
# journal mode: the shadow journalctl in front of the node log collector
# --------------------------------------------------------------------------- #
def test_open_journal_window_arms_the_deadman_before_shadowing_journalctl(
    host: SimpleNamespace, cli: Cli
) -> None:
    answer = open_journal(host, cli)
    state_dir = probe.state_dir(RUN_ID)
    assert answer["mode"] == "journal" and answer["unit"] == LOG, answer
    assert answer["enabled_log_collector"] is True, "the run enabled the collector"
    assert answer["proof_scope"] == "live-replay", answer
    assert answer["before"]["UnitFileState"] == "disabled", answer["before"]
    assert answer["after"]["ActiveState"] == "active", answer["after"]
    assert answer["deadman_timer"]["ActiveState"] == "active", answer["deadman_timer"]
    assert answer["boot_id"] == BOOT_ID.read_text(encoding="ascii").strip(), answer

    name = deadman()
    mutations = host.systemd.mutations()
    systemd_run = next(call for call in mutations if call[0] == "systemd-run")
    copy = state_dir / "probe.py"
    assert systemd_run[1:3] == [f"--unit={name}", "--on-active=900s"], systemd_run
    assert systemd_run[-7:] == [
        sys.executable,
        str(copy),
        "close-window",
        "--run-id",
        RUN_ID,
        "--mode",
        "journal",
    ], "the deadman closes this window from a private copy of the probe"
    assert filecmp.cmp(copy, probe.__file__, shallow=False), "an exact copy"
    assert stat.S_IMODE(copy.stat().st_mode) == 0o700, "the copy is owner-only"
    assert [call for call in mutations if call[0] == "systemctl"] == [
        ["systemctl", "stop", f"{name}.timer"],
        ["systemctl", "reset-failed", f"{name}.service"],
        ["systemctl", "enable", "--runtime", LOG],
        ["systemctl", "daemon-reload"],
        ["systemctl", "restart", LOG],
    ], mutations
    assert mutations.index(systemd_run) < mutations.index(
        ["systemctl", "daemon-reload"]
    ), "the deadman exists before any unit changes"

    shadow = state_dir / "bin"
    assert Path(answer["dropin"]).read_text(encoding="utf-8") == (
        "[Service]\n"
        f"Environment=PATH={shadow}:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"ReadWritePaths={state_dir}\n"
    ), "the drop-in puts the shadow first on the unit's PATH"
    assert answer["dropin"] == str(probe.dropin_path(LOG, RUN_ID)), answer
    assert os.access(shadow / "journalctl", os.X_OK), "the wrapper is executable"
    assert ("journalctl", SYSTEM_PATH) in host.lookups, (
        "the real journalctl is resolved from the system directories only"
    )
    saved = probe.load_state(RUN_ID)
    assert saved["enabled_log_collector"] is True, saved
    assert saved["after"]["InvocationID"] == f"{1:032x}", saved


def test_armed_journal_window_feeds_the_shadow_journalctl_and_status_counts_it(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_journal(host, cli)
    code, status, _ = cli("status", "--run-id", RUN_ID)
    assert code == 0, status
    assert status["injected"] == {
        "mode": "journal",
        "armed": False,
        "emission_counts": {},
    }, "nothing is emitted before arm"

    code, armed, _ = cli("arm", "--run-id", RUN_ID)
    assert armed == {
        "run_id": RUN_ID,
        "mode": "journal",
        "armed": True,
        "records": 1,
        "proof_scope": "live-replay",
    }, armed
    assert probe.armed_path(RUN_ID).is_file(), "arm leaves the armed marker"

    wrapper = probe.state_dir(RUN_ID) / "bin" / "journalctl"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(wrapper),
            "--output=json",
            "--since",
            "@1",
            "--until",
            "@2",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    lines = [json.loads(line) for line in completed.stdout.splitlines()]
    boot = BOOT_ID.read_text(encoding="ascii").strip()
    assert lines[0] == REAL_LINE, "the host's real journalctl answers first"
    assert lines[1]["MESSAGE"] == RECORD["message"], lines
    assert lines[1]["__CURSOR"] == f"gfrs-replay-{RUN_ID}-0-{boot}", lines[1]
    assert lines[1]["_BOOT_ID"] == boot and lines[1]["_SYSTEMD_UNIT"] == (
        "gpu-train.service"
    ), f"the wrapper carries this node's boot and the record's unit: {lines[1]}"

    code, status, _ = cli("status", "--run-id", RUN_ID)
    assert status["injected"] == {
        "mode": "journal",
        "armed": True,
        "emission_counts": {"0": 1, "_until": "@2"},
    }, status
    assert status["unit"]["ActiveState"] == "active", status["unit"]


def test_close_journal_window_stops_and_disables_what_the_run_enabled(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_journal(host, cli)
    host.systemd.calls.clear()
    code, answer, _ = cli("close-window", "--run-id", RUN_ID)
    name = deadman()
    assert code == 0, answer
    assert answer["after"]["ActiveState"] == "inactive", (
        "a collector the run enabled is stopped again, not restarted"
    )
    assert answer["dropin_removed"] is True and answer["clean"] is True, answer
    assert answer["residue"] == {
        "dropin": False,
        "log_collector_enabled_by_run": False,
        "deadman": False,
    }, answer
    assert host.systemd.mutations() == [
        ["systemctl", "disable", "--runtime", LOG],
        ["systemctl", "daemon-reload"],
        ["systemctl", "stop", LOG],
        ["systemctl", "stop", f"{name}.timer"],
        ["systemctl", "stop", f"{name}.service"],
        ["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"],
    ], host.systemd.mutations()
    assert not probe.state_dir(RUN_ID).exists(), "the window state is removed"
    assert host.systemd.unit(LOG)["UnitFileState"] == "disabled", "back to baseline"


def test_journal_window_keeps_a_log_collector_that_was_already_enabled(
    host: SimpleNamespace, cli: Cli
) -> None:
    host.systemd.units[LOG] = unit_properties(
        active="active", file_state="enabled-runtime", pid="77"
    )
    answer = open_journal(host, cli)
    assert answer["enabled_log_collector"] is False, answer
    assert ["systemctl", "enable", "--runtime", LOG] not in host.systemd.calls, (
        "an already runtime-enabled collector is not enabled again"
    )
    host.systemd.calls.clear()
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert closed["after"]["ActiveState"] == "active", "restarted, not stopped"
    assert ["systemctl", "restart", LOG] in host.systemd.calls, host.systemd.calls
    assert not any(call[1] == "disable" for call in host.systemd.mutations()), (
        "the run never disables a collector it did not enable"
    )


def test_write_wrapper_needs_a_host_journalctl(host: SimpleNamespace) -> None:
    host.found.clear()
    with pytest.raises(probe.ProbeError, match="journalctl is not installed"):
        probe.write_wrapper(RUN_ID)


# --------------------------------------------------------------------------- #
# kmsg mode: the kernel collector repointed at a reviewed fixture
# --------------------------------------------------------------------------- #
def test_kmsg_window_repoints_the_kernel_collector_and_restarts_it_on_close(
    host: SimpleNamespace, cli: Cli
) -> None:
    answer = open_kmsg(host, cli)
    fixture = probe.kmsg_fixture_path(RUN_ID)
    assert answer["enabled_log_collector"] is False, answer
    assert Path(answer["dropin"]).read_text(encoding="utf-8") == (
        f"[Service]\nEnvironment={probe.KMSG_ENV}={fixture}\n"
    ), "only the kernel collector's source path is repointed"
    assert fixture.read_text(encoding="utf-8") == "", "an empty fixture to follow"
    assert stat.S_IMODE(fixture.stat().st_mode) == 0o600, "the fixture is private"
    assert ["systemctl", "restart", KERNEL] in host.systemd.calls, host.systemd.calls
    assert not any(call[1] == "enable" for call in host.systemd.mutations()), (
        "kmsg mode never enables a unit"
    )

    host.systemd.calls.clear()
    code, closed, _ = cli("close-window", "--run-id", RUN_ID, "--mode", "kmsg")
    assert code == 0 and closed["clean"] is True, closed
    assert closed["mode"] == "kmsg" and closed["after"]["ActiveState"] == "active"
    assert host.systemd.mutations()[:2] == [
        ["systemctl", "daemon-reload"],
        ["systemctl", "restart", KERNEL],
    ], "the drop-in goes and the collector comes back on /dev/kmsg"
    assert not Path(answer["dropin"]).exists(), "the drop-in is removed"


def test_a_window_whose_collector_does_not_come_back_stays_tracked_for_the_deadman(
    host: SimpleNamespace, cli: Cli
) -> None:
    install(cli, [RECORD])
    host.systemd.restart_leaves = "failed"
    code, answer, err = cli("open-window", "--run-id", RUN_ID, "--mode", "kmsg")
    assert code == 1 and refused(answer, "open-window"), answer
    assert f"{KERNEL} is not active after the window opened" in err, err
    assert probe.load_state(RUN_ID)["mode"] == "kmsg", "the window stays tracked"

    code, snapshot, _ = cli("snapshot", "--run-id", RUN_ID)
    assert snapshot == {
        "run_id": RUN_ID,
        "window_open": True,
        "residue": {
            "dropin": True,
            "log_collector_enabled_by_run": False,
            "deadman": True,
        },
    }, "the drop-in and the armed deadman are what is left to restore"

    code, answer, err = cli("close-window", "--run-id", RUN_ID)
    assert code == 1 and "is not active after the window closed" in err, err
    assert probe.load_state(RUN_ID), "a failed close keeps the window for a retry"

    host.systemd.restart_leaves = "active"
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert cli("snapshot", "--run-id", RUN_ID)[1] == {
        "run_id": RUN_ID,
        "window_open": False,
        "residue": {"window": False},
    }, "nothing is tracked after a clean close"


# --------------------------------------------------------------------------- #
# open-window refusals
# --------------------------------------------------------------------------- #
def already_open(host: SimpleNamespace, cli: Cli) -> None:
    install(cli, [RECORD])
    probe.save_state(RUN_ID, {"run_id": RUN_ID, "mode": "kmsg"})


def untracked_dropin(host: SimpleNamespace, cli: Cli) -> None:
    install(cli, [RECORD])
    dropin = probe.dropin_path(LOG, RUN_ID)
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\n", encoding="utf-8")


def log_unit_missing(host: SimpleNamespace, cli: Cli) -> None:
    install(cli, [RECORD])
    host.systemd.units[LOG] = unit_properties(load="not-found", file_state="")


def kernel_unit_down(host: SimpleNamespace, cli: Cli) -> None:
    install(cli, [RECORD])
    host.systemd.units[KERNEL] = unit_properties(active="failed")


@pytest.mark.parametrize(
    "prepare, argv, message",
    [
        (lambda host, cli: None, ["--mode", "kmsg"], "install the reviewed records"),
        (already_open, ["--mode", "kmsg"], "already open"),
        (untracked_dropin, ["--mode", "journal", "--approved"], "untracked"),
        (
            lambda host, cli: install(cli, [RECORD]),
            ["--mode", "journal"],
            "ENABLE_NODE_LOG_COLLECTOR approval",
        ),
        (
            log_unit_missing,
            ["--mode", "journal", "--approved"],
            "log collector unit is not installed",
        ),
        (kernel_unit_down, ["--mode", "kmsg"], f"{KERNEL} is not active at baseline"),
        (
            lambda host, cli: install(cli, [RECORD]),
            ["--mode", "kmsg", "--restore-seconds", "59"],
            "restore seconds is outside 60..5400",
        ),
        (
            lambda host, cli: install(cli, [RECORD]),
            ["--mode", "kmsg", "--restore-seconds", "5401"],
            "restore seconds is outside 60..5400",
        ),
    ],
    ids=[
        "no-records",
        "already-open",
        "untracked-dropin",
        "journal-unapproved",
        "log-unit-missing",
        "kernel-unit-down",
        "restore-too-short",
        "restore-too-long",
    ],
)
def test_open_window_refuses_before_any_host_change(
    host: SimpleNamespace,
    cli: Cli,
    prepare: Callable[[SimpleNamespace, Cli], object],
    argv: list[str],
    message: str,
) -> None:
    prepare(host, cli)
    code, answer, err = cli("open-window", "--run-id", RUN_ID, *argv)
    assert code == 1 and refused(answer, "open-window"), answer
    assert message in err, err
    assert host.systemd.mutations() == [], "a refused window changes nothing"


def test_open_window_refuses_an_unknown_mode(host: SimpleNamespace) -> None:
    arguments = argparse.Namespace(
        run_id=RUN_ID, mode="dmesg", restore_seconds=900, approved=True
    )
    with pytest.raises(probe.ProbeError, match="mode must be journal or kmsg"):
        probe.open_window(arguments)


def test_a_failing_deadman_launch_reports_the_command_and_its_stderr(
    host: SimpleNamespace, cli: Cli
) -> None:
    install(cli, [RECORD])
    host.systemd.failing["systemd-run"] = 1
    code, answer, err = cli("open-window", "--run-id", RUN_ID, "--mode", "kmsg")
    assert code == 1 and refused(answer, "open-window"), answer
    assert f"command failed (1): systemd-run --unit={deadman()}" in err, err
    assert "Failed to run systemd-run: unit busy" in err, "stderr is carried"
    assert not probe.dropin_path(KERNEL, RUN_ID).exists(), (
        "no drop-in without a deadman"
    )


def test_a_deadman_timer_that_never_activates_fails_the_window(
    host: SimpleNamespace, cli: Cli
) -> None:
    install(cli, [RECORD])
    host.systemd.timer_starts = False
    code, _, err = cli("open-window", "--run-id", RUN_ID, "--mode", "kmsg")
    assert code == 1 and "deadman timer is not active" in err, err
    assert ["systemctl", "daemon-reload"] not in host.systemd.calls, (
        "the unit is never touched without an active deadman"
    )


# --------------------------------------------------------------------------- #
# close-window edges
# --------------------------------------------------------------------------- #
def test_close_window_run_by_the_deadman_does_not_stop_its_own_service(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_kmsg(host, cli)
    name = deadman()
    host.systemd.unit(f"{name}.service")["MainPID"] = str(os.getpid())
    host.systemd.calls.clear()
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert ["systemctl", "stop", f"{name}.timer"] in host.systemd.calls, "timer stops"
    assert ["systemctl", "stop", f"{name}.service"] not in host.systemd.calls, (
        "the deadman's own close never stops the service it runs in"
    )


def test_close_window_without_a_tracked_window_sweeps_stray_drop_ins(
    host: SimpleNamespace, cli: Cli
) -> None:
    install(cli, [RECORD])
    strays = [probe.dropin_path(unit, RUN_ID) for unit in (LOG, KERNEL)]
    for dropin in strays:
        dropin.parent.mkdir(parents=True)
        dropin.write_text("[Service]\n", encoding="utf-8")
    expected = {"run_id": RUN_ID, "already_closed": True, "clean": True}
    code, answer, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and answer == expected, answer
    assert not any(dropin.exists() for dropin in strays), "stray drop-ins removed"
    assert not probe.state_dir(RUN_ID).exists(), "the installed records go too"
    assert host.systemd.calls == [], "nothing tracked: no unit is touched"
    assert cli("close-window", "--run-id", RUN_ID)[:2] == (0, expected), (
        "closing again is a no-op"
    )


def test_close_window_tolerates_a_state_directory_already_removed(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_kmsg(host, cli)
    state_dir = probe.state_dir(RUN_ID)
    # The deadman's own close-window got there first while this one waited.
    host.systemd.on_stop[f"{deadman()}.service"] = lambda: shutil.rmtree(state_dir)
    code, closed, _ = cli("close-window", "--run-id", RUN_ID)
    assert code == 0 and closed["clean"] is True, closed
    assert not state_dir.exists(), "the state directory stays gone"


# --------------------------------------------------------------------------- #
# snapshot / arm / status need a tracked window of this run
# --------------------------------------------------------------------------- #
def test_snapshot_reports_the_residue_of_an_open_journal_window(
    host: SimpleNamespace, cli: Cli
) -> None:
    open_journal(host, cli)
    code, answer, _ = cli("snapshot", "--run-id", RUN_ID)
    assert code == 0, answer
    assert answer["residue"] == {
        "dropin": True,
        "log_collector_enabled_by_run": True,
        "deadman": True,
    }, answer


@pytest.mark.parametrize("window", [{"run_id": "gl016-other-run"}, ["not", "a", "map"]])
def test_a_window_file_of_another_run_is_refused(
    host: SimpleNamespace, cli: Cli, window: object
) -> None:
    path = probe.state_dir(RUN_ID) / "window.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(window), encoding="utf-8")
    code, answer, err = cli("snapshot", "--run-id", RUN_ID)
    assert code == 1 and refused(answer, "snapshot"), answer
    assert "window identity mismatch" in err, err


@pytest.mark.parametrize("command", ["arm", "status"])
def test_arm_and_status_need_an_open_window(
    host: SimpleNamespace, cli: Cli, command: str
) -> None:
    code, answer, err = cli(command, "--run-id", RUN_ID)
    assert code == 1 and refused(answer, command), answer
    assert "no kernel replay window is open for this run" in err, err


# --------------------------------------------------------------------------- #
# the kmsg consumption oracle on what /proc cannot tell
# --------------------------------------------------------------------------- #
def kmsg_state(host: SimpleNamespace, *, fixture: bool = True) -> dict[str, Any]:
    state = {"run_id": RUN_ID, "mode": "kmsg", "after": {"InvocationID": "c" * 32}}
    probe.save_state(RUN_ID, state)
    if fixture:
        probe.kmsg_fixture_path(RUN_ID).write_text(
            "3,900000001,1000,-;NVRM: Xid 79\n", encoding="utf-8"
        )
    return state


def descriptors(host: SimpleNamespace, pid: str, links: dict[str, Path]) -> Path:
    fd_dir = host.root / "proc" / pid / "fd"
    fd_dir.mkdir(parents=True)
    for name, target in links.items():
        (fd_dir / name).symlink_to(target)
    info = host.root / "proc" / pid / "fdinfo"
    info.mkdir()
    return info


def test_kmsg_consumption_without_a_main_pid_reads_no_descriptors(
    host: SimpleNamespace,
) -> None:
    oracle = probe.kmsg_consumption(RUN_ID, kmsg_state(host), {"MainPID": "0"})
    assert oracle["collector"] == {
        "main_pid": "0",
        "fixture_open": False,
        "offset": None,
    }, oracle
    assert oracle["fixture_lines"] == 1 and oracle["consumed"] is False, oracle


def test_kmsg_consumption_of_a_missing_fixture_or_process_is_not_followed(
    host: SimpleNamespace,
) -> None:
    state = kmsg_state(host, fixture=False)
    missing = probe.kmsg_consumption(RUN_ID, state, {"MainPID": "4242"})
    assert (missing["fixture_lines"], missing["fixture_bytes"]) == (0, 0), missing
    assert missing["collector"]["fixture_open"] is False, missing
    probe.kmsg_fixture_path(RUN_ID).write_text("3,1,1,-;x\n", encoding="utf-8")
    gone = probe.kmsg_consumption(RUN_ID, state, {"MainPID": "4242"})
    assert gone["collector"]["fixture_open"] is False, "no /proc entry, no descriptor"
    assert gone["followed"] is False and gone["consumed"] is False, gone


def test_kmsg_consumption_does_not_mistake_another_open_file_for_the_fixture(
    host: SimpleNamespace,
) -> None:
    state = kmsg_state(host)
    other = host.root / "other.log"
    other.write_text("unrelated\n", encoding="utf-8")
    descriptors(host, "4242", {"3": other})
    oracle = probe.kmsg_consumption(RUN_ID, state, {"MainPID": "4242"})
    assert oracle["collector"] == {
        "main_pid": "4242",
        "fixture_open": False,
        "offset": None,
    }, "a collector holding other files does not hold the fixture"
    assert oracle["followed"] is False, oracle


def test_kmsg_consumption_of_an_open_fixture_without_fdinfo_has_no_offset(
    host: SimpleNamespace,
) -> None:
    state = kmsg_state(host)
    descriptors(host, "4242", {"9": probe.kmsg_fixture_path(RUN_ID)})
    oracle = probe.kmsg_consumption(RUN_ID, state, {"MainPID": "4242"})
    assert oracle["collector"] == {
        "main_pid": "4242",
        "fixture_open": True,
        "offset": None,
    }, "an open fixture without a readable position"
    assert oracle["followed"] is True and oracle["consumed"] is False, oracle


def test_kmsg_consumption_ignores_a_position_that_is_not_a_number(
    host: SimpleNamespace,
) -> None:
    state = kmsg_state(host)
    info = descriptors(host, "4242", {"5": probe.kmsg_fixture_path(RUN_ID)})
    (info / "5").write_text("pos:\tunknown\nflags:\t0100000\n", encoding="ascii")
    oracle = probe.kmsg_consumption(RUN_ID, state, {"MainPID": "4242"})
    assert oracle["collector"]["offset"] is None, oracle
    assert oracle["consumed"] is False, "an unknown position is never consumed"


def test_kmsg_consumption_counts_no_reopens_when_the_journal_fails(
    host: SimpleNamespace,
) -> None:
    host.systemd.journal_returncode = 1
    host.systemd.journal = [probe.STREAM_REOPEN_MARKER]
    oracle = probe.kmsg_consumption(RUN_ID, kmsg_state(host), {"MainPID": "0"})
    assert oracle["stream_reopens"] is None, "an unreadable journal is not a count"
    journal = [call for call in host.systemd.calls if call[0] == "journalctl"]
    assert journal and journal[0][1] == f"_SYSTEMD_INVOCATION_ID={'c' * 32}", journal


# --------------------------------------------------------------------------- #
# read-only listings and the CLI shell
# --------------------------------------------------------------------------- #
def test_preflight_reads_the_boot_journalctl_and_both_collector_units(
    host: SimpleNamespace, cli: Cli
) -> None:
    code, answer, _ = cli("preflight")
    assert code == 0, answer
    assert answer["boot_id"] == BOOT_ID.read_text(encoding="ascii").strip(), answer
    assert answer["journalctl"] is True, answer
    assert answer["units"][LOG]["LoadState"] == "loaded", answer["units"]
    assert answer["units"][KERNEL]["MainPID"] == "4242", answer["units"]
    assert answer["units"][KERNEL]["returncode"] == "0", answer["units"]
    assert "garbage" not in answer["units"][KERNEL], "lines without '=' are skipped"
    host.found.clear()
    assert cli("preflight")[1]["journalctl"] is False, "a host without journalctl"
    assert host.systemd.mutations() == [], "preflight only reads"


def test_rdma_devices_stops_at_the_device_bound(
    host: SimpleNamespace, cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = host.root / "sys" / "class" / "infiniband"
    for name in ("rdmap1s0", "rdmap2s0", "rdmap3s0"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(probe, "INFINIBAND_ROOT", root)
    monkeypatch.setattr(probe, "MAX_RDMA_DEVICES", 2)
    code, answer, _ = cli("rdma-devices")
    assert code == 0, answer
    assert [device["name"] for device in answer["devices"]] == [
        "rdmap1s0",
        "rdmap2s0",
    ], "at most MAX_RDMA_DEVICES devices are listed, in name order"


def test_running_the_file_as_a_script_exits_with_the_cli_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unsafe run ID is refused before the probe reads or writes the host.
    monkeypatch.setattr(sys, "argv", [probe.__file__, "snapshot", "--run-id", "BAD ID"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert raised.value.code == 1, "a refused command exits 1"
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip().splitlines()[-1]) == {
        "error": "ProbeError",
        "error_kind": "probe",
        "command": "snapshot",
    }, captured.out
    assert "ProbeError: unsafe run ID" in captured.err, captured.err
