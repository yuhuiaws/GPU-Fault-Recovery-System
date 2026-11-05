"""The temporary-test-package probe driven through ``main(argv)`` at its boundary.

``probes/general_live_deps_probe.py`` installs, loads and removes exactly the
reviewed test packages and modules of one run. Here the deps tests' AL2023
``FakeNode`` answers it from behind the probe's own ``subprocess``,
``shutil.which`` and ``os.uname`` (``_cov95_general_live_deps.Boundary``), so a
test runs ``ensure``/``cleanup``/``residue`` as the runner does and checks the
JSON, the state record and the node commands, including the failures a live
node can show; the transaction judges are checked on their own.
"""

from __future__ import annotations

import filecmp
import json
import os
import runpy
import stat
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_deps_probe as deps
from tests.regional._cov95_general_live_deps import Boundary, install_boundary
from tests.regional.test_general_live_deps_probe import (
    BASELINE,
    DOSFS_NEVRA,
    KERNEL,
    RUN_ID,
    TC_NEVRA,
    FakeNode,
)

Cli = Callable[..., tuple[int, dict[str, Any], str]]
Build = Callable[..., tuple[FakeNode, Boundary]]
DEADMAN = deps.deadman_unit(RUN_ID)
TC_ROWS = {**BASELINE, TC_NEVRA: ("iproute-tc", "5.10.0", "2.amzn2023.0.5")}


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Build:
    def make(**kwargs: Any) -> tuple[FakeNode, Boundary]:
        node = FakeNode(tmp_path, **kwargs)
        return node, install_boundary(monkeypatch, tmp_path, node)

    return make


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Cli:
    def invoke(*argv: str) -> tuple[int, dict[str, Any], str]:
        code = deps.main(list(argv))
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines, f"the probe prints one JSON object for {argv}: {captured}"
        return code, json.loads(lines[-1]), captured.err

    return invoke


def ensure(cli: Cli, *extra: str) -> tuple[int, dict[str, Any], str]:
    return cli("ensure", "--run-id", RUN_ID, "--deadman-seconds", "2700", *extra)


def record(**values: Any) -> None:
    state = deps.load_state(RUN_ID)
    state.update(values)
    deps.save_state(RUN_ID, state)


def refused(answer: dict[str, Any], command: str) -> bool:
    return answer == {
        "error": "ProbeError",
        "error_kind": "probe",
        "error_class": "ProbeError",
        "command": command,
    }


# --------------------------------------------------------------------------- #
# ensure / cleanup / residue through the CLI
# --------------------------------------------------------------------------- #
def test_ensure_then_cleanup_through_the_cli_in_the_c_locale(
    build: Build, cli: Cli
) -> None:
    node, boundary = build(loaded={"sch_netem"})
    code, answer, err = ensure(
        cli, "--tool", "tc", "--module", "sch_netem", "--allow-install"
    )
    assert code == 0, err
    assert answer["installed"] == [TC_NEVRA] and answer["tools"] == {"tc": True}
    assert (answer["kernel"], answer["refused"], answer["deadman"]) == (
        KERNEL,
        None,
        DEADMAN,
    ), answer
    assert answer["modules_before"]["sch_netem"]["loaded"] is True, answer
    assert [command[1] for command, _ in boundary.environments] == [
        "--assumeno",
        "-y",
    ], "the preview and the install are the only commands with their own locale"
    assert all(
        (env["LC_ALL"], env["LANG"], env.get("PATH"))
        == ("C", "C", os.environ.get("PATH"))
        for _, env in boundary.environments
    ), "dnf runs in the C locale its table parser reads, otherwise as the probe"
    started = node.ran("systemd-run")[0]
    copy = deps.state_dir(RUN_ID) / "deps-probe.py"
    assert started[-5:] == [sys.executable, str(copy), "cleanup", "--run-id", RUN_ID]
    assert node.ran("systemd-run") and node.ran("dnf", "-y")[0][-1] == TC_NEVRA, (
        "the deadman is armed and only the previewed NEVRA is installed"
    )

    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert code == 0 and answer["clean"] is True, answer
    assert answer["report"]["removed"] == [TC_NEVRA], answer["report"]
    assert ["systemctl", "stop", f"{DEADMAN}.service"] in node.calls, (
        "a clean cleanup disarms the deadman"
    )
    code, answer, _ = cli("residue", "--run-id", RUN_ID)
    assert (code, answer["clean"]) == (0, True), answer
    assert answer["residue"] == {
        "packages": [],
        "modules": [],
        "units": [],
        "selinux_modules": [],
    }, answer


def test_cleanup_run_by_the_deadman_does_not_stop_its_own_service(
    build: Build, cli: Cli
) -> None:
    node, boundary = build(installed=TC_ROWS)
    record(installed=[TC_NEVRA])
    boundary.deadman_pid = str(os.getpid())
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert code == 0 and answer["clean"] is True, answer
    assert ["systemctl", "stop", f"{DEADMAN}.timer"] in node.calls, "timer stopped"
    assert ["systemctl", "stop", f"{DEADMAN}.service"] not in node.calls, (
        "the deadman's own cleanup never stops the service it runs in"
    )


def test_a_deadman_that_does_not_arm_stops_the_ensure_before_any_change(
    build: Build, cli: Cli
) -> None:
    node, boundary = build()
    boundary.timer_active = False
    code, answer, err = ensure(cli, "--tool", "tc", "--allow-install")
    assert code == 1 and refused(answer, "ensure"), answer
    assert f"deadman {DEADMAN} is not armed" in err, err
    assert node.ran("dnf", "-y") == [], "nothing is installed without the deadman"


def test_an_rpm_database_error_is_reported_with_the_command(
    build: Build, cli: Cli
) -> None:
    node, boundary = build()
    boundary.rpm_error = True
    code, answer, err = ensure(cli, "--tool", "tc", "--allow-install")
    assert code == 1 and refused(answer, "ensure"), answer
    assert "command failed (1): rpm -qa --qf" in err, err
    assert "rpmdb: BDB0113 lock held" in err, "rpm's own error is carried"
    assert node.ran("systemd-run") == [], "the refusal comes before the deadman"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["residue", "--run-id", "BAD ID"], "unsafe run ID"),
        (
            ["ensure", "--run-id", RUN_ID, "--deadman-seconds", "30", "--tool", "tc"],
            "deadman is outside 60..5400",
        ),
    ],
    ids=["run-id", "deadman"],
)
def test_the_cli_refuses_unsafe_arguments(
    build: Build, cli: Cli, argv: list[str], message: str
) -> None:
    node, _ = build()
    code, answer, err = cli(*argv)
    assert code == 1 and refused(answer, argv[0]), answer
    assert message in err, err
    assert node.calls == [], "a refused argument runs no command"


def test_a_state_file_of_another_run_is_refused(build: Build, cli: Cli) -> None:
    build()
    path = deps.state_dir(RUN_ID) / "deps-state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": "gl009-other-run"}), encoding="utf-8")
    code, answer, err = cli("residue", "--run-id", RUN_ID)
    assert code == 1 and refused(answer, "residue"), answer
    assert "deps probe state identity mismatch" in err, err


# --------------------------------------------------------------------------- #
# ensure decisions before the node changes
# --------------------------------------------------------------------------- #
def test_a_module_built_into_the_kernel_is_neither_loaded_nor_installed(
    build: Build, cli: Cli
) -> None:
    node, _ = build()
    builtin = deps.MODULES_ROOT / KERNEL / "modules.builtin"
    builtin.parent.mkdir(parents=True)
    builtin.write_text(
        "kernel/net/sched/sch_tbf.ko\nkernel/net/sched/sch_netem.ko\n", encoding="utf-8"
    )
    code, answer, _ = ensure(cli, "--module", "sch_netem")
    assert code == 0 and answer["refused"] is None, answer
    assert answer["modules_before"] == {
        "sch_netem": {"loaded": False, "builtin": True, "loadable": True}
    }, answer
    assert node.ran("modprobe") == [] and node.ran("systemd-run") == [], (
        "a built-in module needs nothing: no dry run, no load, no deadman"
    )


def test_ensure_without_dnf_reports_the_capability_absent(
    build: Build, cli: Cli
) -> None:
    node, boundary = build()
    boundary.hidden.add("dnf")
    code, answer, _ = ensure(cli, "--tool", "tc", "--allow-install")
    assert code == 0, answer
    assert answer["refused"] == {
        "reason": "CAPABILITY_ABSENT",
        "detail": "no dnf/rpm on the node for iproute-tc",
    }, answer
    assert node.calls == [], "nothing is queried or armed without a package manager"


def test_a_package_installed_but_unusable_is_never_touched(
    build: Build, cli: Cli
) -> None:
    node, boundary = build(installed=TC_ROWS)
    boundary.hidden.add("tc")  # the package is there, the tool is not usable
    code, answer, _ = ensure(cli, "--tool", "tc", "--allow-install")
    assert code == 0, answer
    assert answer["refused"] == {
        "reason": "FIXTURE_NOT_READY",
        "detail": f"{TC_NEVRA} is installed but still unusable; not touched",
    }, answer
    assert node.ran("dnf") == [] and node.ran("systemd-run") == [], (
        "an existing package is never reinstalled or armed for"
    )


def test_a_failed_dnf_install_names_its_exit_its_error_and_what_is_missing(
    build: Build, cli: Cli
) -> None:
    node, boundary = build()
    boundary.install_error = True
    code, answer, _ = ensure(cli, "--tool", "tc", "--allow-install")
    assert code == 0 and answer["refused"]["reason"] == "FIXTURE_NOT_READY", answer
    assert answer["refused"]["detail"] == (
        "dnf install ended 1; Error: Transaction test error:; "
        f"dnf did not install {TC_NEVRA}; tc still missing after the install"
    ), answer["refused"]
    assert deps.load_state(RUN_ID)["pending"] == [], "the pending record is settled"


def test_a_module_that_will_not_load_is_recorded_and_loaded_on_a_retry(
    build: Build, cli: Cli
) -> None:
    node, boundary = build(loadable={"sch_netem"})
    boundary.modprobe_error = True
    code, answer, _ = ensure(cli, "--module", "sch_netem", "--allow-module-load")
    assert code == 0, answer
    assert answer["refused"]["reason"] == "FIXTURE_NOT_READY", answer
    assert answer["refused"]["detail"].startswith(
        "sch_netem could not be loaded (1): modprobe: FATAL: Module sch_netem"
    ), answer["refused"]
    assert deps.load_state(RUN_ID)["modules"] == ["sch_netem"], (
        "recorded before the load, so the cleanup owns whatever it left"
    )

    boundary.modprobe_error = False
    code, answer, _ = ensure(cli, "--module", "sch_netem", "--allow-module-load")
    assert code == 0 and answer["refused"] is None, answer
    assert answer["modules_loaded_by_run"] == ["sch_netem"], "recorded once"
    assert answer["modules"]["sch_netem"]["loaded"] is True, answer["modules"]


def test_footprint_claims_are_recorded_once(build: Build, cli: Cli) -> None:
    node, _ = build()
    # An earlier attempt of this run already claimed both; one unit is gone.
    record(units=["smartd.service", "gone.service"], selinux_modules=["200 smartmon"])
    code, answer, _ = ensure(cli, "--tool", "smartctl", "--allow-install")
    assert code == 0 and answer["refused"] is None, answer
    footprint = answer["footprint"]
    assert footprint["claimed"] == {
        "units": ["smartd.service", "gone.service"],
        "selinux_modules": ["200 smartmon"],
    }, "nothing is claimed twice"
    assert footprint["quieted"] == ["smartd.service"], (
        "only a unit that was enabled or active is quieted"
    )
    assert node.units["smartd.service"]["enabled"] is False, "the preset is undone"


def test_what_the_node_already_had_is_never_claimed(build: Build, cli: Cli) -> None:
    node, _ = build()
    node.units["smartd.service"] = {"loaded": True, "active": False, "enabled": False}
    node.selinux_modules.add("200 smartmon")
    code, answer, _ = ensure(cli, "--tool", "smartctl", "--allow-install")
    assert code == 0 and answer["refused"] is None, answer
    assert answer["footprint"]["claimed"] == {"units": [], "selinux_modules": []}, (
        "a unit file and a policy module the node had before are not this run's"
    )
    assert answer["footprint"]["quieted"] == [], answer["footprint"]


# --------------------------------------------------------------------------- #
# cleanup edges
# --------------------------------------------------------------------------- #
def test_cleanup_unloads_a_module_without_a_refcount_and_skips_one_already_gone(
    build: Build, cli: Cli
) -> None:
    node, _ = build()
    (node.sys_module / "sch_netem").mkdir()  # no refcnt file: read as unused
    record(modules=["sch_netem"])
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert answer["report"]["unloaded"] == ["sch_netem"], answer["report"]
    assert ["modprobe", "-r", "sch_netem"] in node.calls, node.calls

    record(modules=["sch_netem"])  # recorded again, but no longer loaded
    node.calls.clear()
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert answer["report"]["unloaded"] == [] and node.ran("modprobe") == [], (
        "a module that is gone already is not unloaded again"
    )
    assert answer["clean"] is True, answer


def test_cleanup_reports_a_unit_it_cannot_disable_and_keeps_the_deadman(
    build: Build, cli: Cli
) -> None:
    node, _ = build()
    node.units["smartd.service"] = {"loaded": True, "active": False, "enabled": True}
    node.undisableable = True
    record(units=["smartd.service"])
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert answer["report"]["errors"] == ["smartd.service is still enabled"], answer
    assert answer["residue"]["units"] == ["smartd.service"], answer["residue"]
    assert answer["clean"] is False, answer
    assert ["systemctl", "stop", f"{DEADMAN}.timer"] not in node.calls, (
        "the deadman stays armed to retry"
    )


def test_cleanup_without_dnf_keeps_the_record(build: Build, cli: Cli) -> None:
    node, boundary = build(installed=TC_ROWS)
    boundary.hidden.update({"dnf", "rpm"})
    record(installed=[TC_NEVRA])
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert answer["report"]["errors"] == [f"no dnf/rpm to remove {TC_NEVRA}"], answer
    assert answer["residue"]["packages"] == [TC_NEVRA], "still recorded as residue"
    assert node.ran("dnf") == [], "nothing is removed without dnf"


def test_cleanup_skips_a_package_already_gone_and_reports_a_wider_removal(
    build: Build, cli: Cli
) -> None:
    node, boundary = build(installed=TC_ROWS)
    record(installed=[DOSFS_NEVRA, TC_NEVRA])  # dosfstools went meanwhile
    lost = "bash-5.2.15-1.amzn2023.0.2.x86_64"
    boundary.cascade = lost
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    report = answer["report"]
    assert report["removed"] == [TC_NEVRA], report
    assert report["errors"] == [f"the installed set also lost {lost}"], report
    removals = [call[-1] for call in node.ran("dnf") if "remove" in call]
    assert removals == [TC_NEVRA, TC_NEVRA], "preview and removal of iproute-tc only"


def test_a_removal_dnf_refuses_keeps_the_package_recorded(
    build: Build, cli: Cli
) -> None:
    node, boundary = build(installed=TC_ROWS)
    boundary.remove_error = True
    record(installed=[TC_NEVRA])
    code, answer, _ = cli("cleanup", "--run-id", RUN_ID)
    assert answer["report"]["errors"] == [
        f"dnf remove {TC_NEVRA} ended 1: Error: package is protected"
    ], answer["report"]
    assert deps.load_state(RUN_ID)["installed"] == [TC_NEVRA], "kept on record"
    assert answer["clean"] is False and answer["residue"]["packages"] == [TC_NEVRA]


# --------------------------------------------------------------------------- #
# readers and transaction judges
# --------------------------------------------------------------------------- #
def test_rpm_packages_carries_the_epoch_and_skips_malformed_rows(build: Build) -> None:
    node, boundary = build()
    boundary.rpm_rows = ["vim-enhanced\t2\t9.0.2081\t1.amzn2023\tx86_64\t900", "junk"]
    packages = deps.rpm_packages()
    assert packages["vim-enhanced-2:9.0.2081-1.amzn2023.x86_64"] == {
        "name": "vim-enhanced",
        "tid": "900",
    }, "a non-zero epoch is part of the NEVRA"
    assert sorted(packages) == sorted(
        [*BASELINE, "vim-enhanced-2:9.0.2081-1.amzn2023.x86_64"]
    ), "a row without six fields is skipped"


def test_unit_state_skips_lines_without_a_separator(build: Build) -> None:
    build()
    assert deps.unit_state(f"{DEADMAN}.timer") == {
        "LoadState": "loaded",
        "ActiveState": "active",
    }, "only key=value lines are read"


def test_probe_copy_is_made_once_and_owner_only(build: Build) -> None:
    build()
    first = deps.probe_copy(RUN_ID)
    assert deps.probe_copy(RUN_ID) == first, "the same copy is reused"
    assert filecmp.cmp(first, deps.__file__, shallow=False), "an exact copy"
    assert stat.S_IMODE(first.stat().st_mode) == 0o700, "owner-only"


def test_selinux_modules_are_absent_without_semodule(build: Build) -> None:
    node, boundary = build()
    boundary.hidden.add("semodule")
    assert deps.selinux_module_present("100 smartmon") is False, "no store to read"
    assert node.calls == [], "semodule is not run when it is missing"


def test_first_error_prefers_an_error_or_problem_line() -> None:
    planned = subprocess.CompletedProcess(
        ["dnf"],
        1,
        stdout="Last metadata expiration check: 0:01:02 ago.\nProblem: conflicting\n",
        stderr="",
    )
    assert deps.first_error(planned) == "Problem: conflicting", planned
    quiet = subprocess.CompletedProcess(["dnf"], 1, stdout="", stderr="  lock held  ")
    assert deps.first_error(quiet) == "lock held", "else the tail of the output"


TC = {"section": "Installing", "name": "iproute-tc", "arch": "x86_64"}
TC_ENTRY = {**TC, "version": "5.10.0-2.amzn2023.0.5", "repository": "amazonlinux"}


@pytest.mark.parametrize(
    "stdout, entries, unparsed, wanted, reason",
    [
        ("", [], ["  a row dnf printed oddly"], {"iproute-tc": None}, "unreadable"),
        (
            "Error: Failed to download metadata for repo 'amazonlinux'\n",
            [],
            [],
            {"iproute-tc": None},
            "dnf could not plan the install: Error: Failed to download metadata",
        ),
        (
            "",
            [TC_ENTRY],
            [],
            {"iproute-tc": None, "dosfstools": None},
            "dnf plans no install of dosfstools",
        ),
        (
            "",
            [TC_ENTRY, TC_ENTRY],
            [],
            {"iproute-tc": None},
            "more than one iproute-tc",
        ),
    ],
    ids=["unparsed", "unplannable", "missing", "repeated"],
)
def test_install_refusal_names_why_a_preview_may_not_run(
    stdout: str,
    entries: list[dict[str, str]],
    unparsed: list[str],
    wanted: dict[str, str | None],
    reason: str,
) -> None:
    preview = subprocess.CompletedProcess(["dnf"], 1, stdout=stdout, stderr="")
    refusal = deps.install_refusal(
        preview, {"entries": entries, "unparsed": unparsed}, wanted
    )
    assert refusal is not None and reason in refusal, refusal


def test_install_order_lists_only_the_rows_dnf_planned() -> None:
    wanted: dict[str, str | None] = {"iproute-tc": None, "dosfstools": None}
    assert deps.install_order([TC_ENTRY], wanted) == [TC_ENTRY], (
        "a wanted package dnf did not plan is not invented"
    )


def test_removal_refusal_reads_unparsed_rows_and_a_different_set() -> None:
    unreadable = deps.removal_refusal(
        {"entries": [], "unparsed": ["garbled"]}, [TC_NEVRA]
    )
    assert unreadable == "unreadable dnf removal rows: garbled", unreadable
    nothing = deps.removal_refusal({"entries": [], "unparsed": []}, [TC_NEVRA])
    assert nothing == f"dnf plans to remove nothing instead of {TC_NEVRA}", nothing


def test_installed_by_run_names_a_missing_and_a_lost_package() -> None:
    before = {"bash-5": {"name": "bash", "tid": "1"}}
    ours, problems = deps.installed_by_run(before, {}, [TC_NEVRA])
    assert ours == [], ours
    assert problems == [
        f"dnf did not install {TC_NEVRA}",
        "the installed set lost bash-5 during the install",
    ], problems


def test_unit_problem_names_an_active_unit_first() -> None:
    state = {"loaded": True, "active": True, "enabled": True}
    assert deps.unit_problem("smartd.service", state, " now") == (
        "smartd.service is still active now"
    ), state


def test_utc_now_is_an_aware_utc_timestamp() -> None:
    stamp = datetime.fromisoformat(deps.utc_now())
    assert stamp.utcoffset() == timedelta(0), stamp


def test_running_the_file_as_a_script_exits_with_the_cli_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unsafe run ID is refused before the probe reads or writes the host.
    monkeypatch.setattr(sys, "argv", [deps.__file__, "residue", "--run-id", "BAD ID"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(deps.__file__, run_name="__main__")
    assert raised.value.code == 1, "a refused command exits 1"
    captured = capsys.readouterr()
    assert refused(json.loads(captured.out.strip().splitlines()[-1]), "residue"), (
        captured.out
    )
    assert "ProbeError: unsafe run ID" in captured.err, captured.err


def test_ensure_records_nothing_for_a_tool_that_is_present(
    build: Build, cli: Cli
) -> None:
    node, _ = build(installed=TC_ROWS)
    code, answer, _ = ensure(cli, "--tool", "tc")
    assert code == 0 and answer["refused"] is None, answer
    assert answer["installed"] == [] and node.ran("systemd-run") == [], answer
    assert not deps.state_dir(RUN_ID).exists(), "no state for a no-op ensure"
