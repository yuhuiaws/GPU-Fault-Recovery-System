from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional.site_profile import (
    SITE_PROFILE_ENV,
    SiteProfileError,
    applied_site_profile,
    apply_site_environment,
    install_site_profile,
    load_site_profile,
    profile_argv,
    site_profile_path,
)

ROOT = Path(__file__).resolve().parents[2]


def _profile(tmp_path: Path, document: dict, name: str = "site.yaml") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_command_line_flag_overrides_the_profile_instead_of_adding_to_it() -> None:
    # `--node` is an append action on several runners, so a profile that merged
    # rather than deferred would hand COLLECT-011 three nodes for a case that
    # requires exactly two, and the extra one would be a node nobody approved.
    profile = {"arguments": {"node": ["profile-a", "profile-b"], "region": "us-west-2"}}

    extra = profile_argv(profile, ["--node", "typed-a", "--node", "typed-b"])

    assert extra == ["--region", "us-west-2"], extra
    assert profile_argv(profile, []) == [
        "--node",
        "profile-a",
        "--node",
        "profile-b",
        "--region",
        "us-west-2",
    ]


def test_flag_written_with_an_equals_sign_still_wins() -> None:
    profile = {"arguments": {"namespace": "profile-namespace"}}

    assert profile_argv(profile, ["--namespace=typed"]) == []


def test_ambient_environment_wins_over_the_profile() -> None:
    # The profile is a default, not an override: an operator who exported a
    # different window in this shell must not have it silently replaced.
    environment = {"GPU_FAULT_ACCEPTANCE_WINDOW_END": "2026-09-05T14:00:00Z"}
    profile = {
        "environment": {
            "GPU_FAULT_ACCEPTANCE_WINDOW_END": "2026-01-01T00:00:00Z",
            "AWS_REGION": "us-west-2",
        }
    }

    applied = apply_site_environment(profile, environment)

    assert applied == ["AWS_REGION"], applied
    assert environment["GPU_FAULT_ACCEPTANCE_WINDOW_END"] == "2026-09-05T14:00:00Z"


def test_group_or_world_writable_profile_is_refused(tmp_path: Path) -> None:
    # This file names the node a destructive case reboots. If anyone can write
    # it, anyone can redirect a real reboot.
    path = _profile(tmp_path, {"arguments": {"node": "node-a"}})
    path.chmod(0o666)

    with pytest.raises(SiteProfileError, match="world-writable"):
        load_site_profile(path)


def test_unknown_profile_section_is_refused(tmp_path: Path) -> None:
    # A typo in a section name would otherwise silently drop every value in it
    # and leave the runner falling back to whatever the shell happened to hold.
    path = _profile(tmp_path, {"argument": {"node": "node-a"}})

    with pytest.raises(SiteProfileError, match="unknown sections"):
        load_site_profile(path)


def test_missing_profile_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(SiteProfileError, match="does not exist"):
        load_site_profile(tmp_path / "absent.yaml")


def test_profile_path_prefers_the_flag_over_the_environment(tmp_path: Path) -> None:
    environment = {SITE_PROFILE_ENV: str(tmp_path / "from-env.yaml")}

    chosen = site_profile_path(
        ["--site-profile", str(tmp_path / "typed.yaml")], environment
    )
    assert chosen == (tmp_path / "typed.yaml").resolve()

    assert site_profile_path([], environment) == (tmp_path / "from-env.yaml").resolve()
    assert site_profile_path([], {}) is None


def test_a_runners_own_site_flag_is_not_taken_as_the_profile(tmp_path: Path) -> None:
    """``--site <RegionalSite yaml>`` is a BOOT runner flag, not ``--site-profile``.

    The pre-parser used argparse's default abbreviation matching, so a run with
    ``--site /secure/.../site.yaml`` loaded site.yaml as the profile and failed
    with "unknown sections: apiVersion, kind, metadata, spec" (BOOT-011, live).
    """

    assert site_profile_path(["--site", str(tmp_path / "site.yaml")], {}) is None
    chosen = site_profile_path(
        [
            "--site-profile",
            str(tmp_path / "profile.yaml"),
            "--site",
            str(tmp_path / "site.yaml"),
        ],
        {},
    )
    assert chosen == (tmp_path / "profile.yaml").resolve()


def test_profile_only_supplies_flags_the_runner_actually_defines() -> None:
    # One profile describes the site, not one case, and the cases disagree on
    # flag names: the collector runners take --node, DESTR-003 takes
    # --fault-node/--spare-node. Injecting every key would make every runner
    # missing one of them exit 2 on an unrecognised argument.
    profile = {"arguments": {"node": "node-a", "region": "us-west-2"}}

    accepted = {"--fault-node", "--spare-node", "--region"}
    assert profile_argv(profile, [], accepted=accepted) == ["--region", "us-west-2"]
    assert profile_argv(profile, [], accepted={"--node"}) == ["--node", "node-a"]


def test_bind_lets_a_runner_read_profile_arguments_without_rewriting_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _profile(
        tmp_path,
        {"arguments": {"node": "node-a", "region": "us-west-2", "absent-flag": "x"}},
    )
    monkeypatch.setenv(SITE_PROFILE_ENV, str(path))
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="")
    from scripts.e2e.regional.site_profile import bind_site_profile

    bind_site_profile(parser)
    # Added after the bind: the flag set is read at parse time, not at bind time,
    # so a runner that adds its own flags later is still filled from the profile.
    parser.add_argument("--node", default="")

    arguments = parser.parse_args([])

    assert arguments.region == "us-west-2"
    assert arguments.node == "node-a"


def test_install_records_a_digest_without_touching_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _profile(
        tmp_path,
        {
            "arguments": {"region": "us-west-2", "node": ["node-a"]},
            "environment": {"AWS_REGION": "us-west-2"},
        },
    )
    # setenv, not delenv: `install_site_profile` assigns these keys itself, and
    # monkeypatch records no undo for a delenv of a key that was already absent.
    # Without an undo the profile path escapes into every later test's
    # subprocesses, which then die on a profile under a deleted tmp_path.
    monkeypatch.setenv(SITE_PROFILE_ENV, "")
    monkeypatch.setenv("AWS_REGION", "")
    monkeypatch.setattr(
        sys, "argv", ["runner", "--site-profile", str(path), "--case", "CASE"]
    )

    record = install_site_profile()

    assert record is not None
    assert record["path"] == str(path)
    assert len(record["sha256"]) == 64
    # argv is left alone: at this point nothing knows which flags this runner
    # defines, so the arguments are applied later by `bind_site_profile`.
    assert sys.argv[1:] == ["--site-profile", str(path), "--case", "CASE"]
    # What install does own is the environment the parser defaults read from.
    assert os.environ["AWS_REGION"] == "us-west-2"
    assert applied_site_profile() == record


def test_a_plan_cannot_be_executed_under_a_different_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of recording the profile is that the plan names the target
    # it was approved for. Swapping the file between plan and execute has to be
    # a refusal, not a silent retarget.
    first = _profile(tmp_path, {"arguments": {"node": "node-a"}}, "first.yaml")
    second = _profile(tmp_path, {"arguments": {"node": "node-b"}}, "second.yaml")
    environment = {"KUBECONFIG": "/tmp/gpu.kubeconfig"}
    monkeypatch.setenv(SITE_PROFILE_ENV, str(first))
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", raising=False)
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE", raising=False)
    run_dir = tmp_path / "run"
    live_driver_guard.build_plan(
        run_dir=run_dir,
        case_id="GF-REGIONAL-COLLECT-002",
        attempt=1,
        confirmation="COLLECT002_EXECUTE",
        details={},
        environment=environment,
    )

    monkeypatch.setenv(SITE_PROFILE_ENV, str(second))
    arguments = argparse.Namespace(
        execute=True,
        confirm="COLLECT002_EXECUTE",
        maintenance_window_end="2099-01-01T00:00:00Z",
        run_dir=run_dir,
        attempt=1,
    )

    with pytest.raises(RuntimeError, match="drifted at site_profile"):
        live_driver_guard.authorize_execution(
            arguments,
            case_id="GF-REGIONAL-COLLECT-002",
            confirmation="COLLECT002_EXECUTE",
            environment=environment,
        )


def test_every_live_runner_installs_the_profile_before_parsing() -> None:
    # A runner that forgets the call still parses, still runs, and quietly
    # ignores the operator's single source of site values -- so this is checked
    # for all of them rather than for the ones in use today.
    missing = []
    for path in sorted((ROOT / "scripts/e2e/regional").glob("run_*.py")):
        source = path.read_text(encoding="utf-8")
        if "add_live_arguments(" not in source:
            continue
        if "install_site_profile()" in source:
            continue
        # A runner that hands its ``main`` to the shared spine inherits the
        # call; tests/test_acceptance_runner_main.py pins that the spine makes
        # it before the parser exists.
        if any(
            f"{spine}(CASE)" in source
            for spine in ("run_standard_case", "run_selected_case", "run_plain_case")
        ):
            continue
        # NET-002/003 delegate to net_command_fixture.run_main, which installs
        # the profile before it builds the parser; pinned in
        # tests/regional/test_net002_command_recovery.py.
        if "fixture.run_main(" in source:
            continue
        missing.append(path.name)
    assert not missing, missing


def test_runner_help_still_works_without_a_profile() -> None:
    # The profile hooks both the environment and `parse_args`, so the cheapest
    # regression it could cause is breaking every runner's --help.
    completed = subprocess.run(
        [sys.executable, "scripts/e2e/regional/run_collector_acceptance.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--site-profile" in completed.stdout, completed.stdout


def test_a_profile_may_not_carry_the_per_case_approval_flags(tmp_path: Path) -> None:
    """The profile names the target; ``--execute``/``--confirm``/``--plan``/
    ``--attempt`` are what the operator types for one run after reading its
    plan. A profile carrying them turns every ``--plan`` into an execute."""

    for key in ("confirm", "execute", "plan", "attempt", "--confirm"):
        path = _profile(tmp_path, {"arguments": {"node": "node-a", key: "x"}})
        with pytest.raises(SiteProfileError, match=key.lstrip("-")):
            load_site_profile(path)

    with pytest.raises(SiteProfileError, match="confirm"):
        profile_argv({"arguments": {"confirm": "DESTR001_EXECUTE"}}, [])
    # A key that merely contains a reserved word is not reserved.
    assert profile_argv({"arguments": {"confirm_node": "node-a"}}, []) == [
        "--confirm-node",
        "node-a",
    ]
