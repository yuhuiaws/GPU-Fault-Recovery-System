from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_collect022_fm_cursor_recovery as collect022
from scripts.e2e.regional import run_collector_acceptance as collector
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
    arguments = argparse.Namespace(
        execute=True,
        confirm="COLLECT002_EXECUTE",
        maintenance_window_end="2099-01-01T00:00:00Z",
        run_dir=run_dir,
        attempt=1,
    )
    live_driver_guard.build_plan(
        arguments=arguments,
        preflight_passed=True,
        run_dir=run_dir,
        case_id="GF-REGIONAL-COLLECT-002",
        attempt=1,
        confirmation="COLLECT002_EXECUTE",
        details={},
        environment=environment,
    )

    monkeypatch.setenv(SITE_PROFILE_ENV, str(second))

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
        # Both named CASEs and inline CaseRunner instances inherit the shared
        # spine's profile-before-parser contract.
        tree = ast.parse(source, filename=str(path))
        if any(
            isinstance(node, ast.FunctionDef)
            and node.name == "main"
            and len(node.body) == 1
            and isinstance(statement := node.body[0], ast.Return)
            and isinstance(call := statement.value, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id
            in {"run_standard_case", "run_selected_case", "run_plain_case"}
            for node in tree.body
        ):
            continue
        # NET-002/003 delegate to net_command_fixture.run_main, which installs
        # the profile before it builds the parser; pinned in
        # tests/regional/test_net002_command_recovery.py.
        if "fixture.run_main(" in source:
            continue
        missing.append(path.name)
    assert not missing, missing


def test_collect022_delegates_its_complete_case_to_the_standard_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegate = Mock(return_value=7)
    monkeypatch.setattr(collect022, "run_standard_case", delegate)

    assert collect022.main() == 7

    delegate.assert_called_once()
    case = delegate.call_args.args[0]
    assert isinstance(case, live_driver_guard.CaseRunner), (
        "COLLECT022 must delegate a complete fixed-identity case"
    )
    assert case.case_id == collect022.CASE_ID
    assert case.confirmation == collect022.CONFIRMATION
    for name in (
        "parser",
        "configure",
        "read_only_preflight",
        "plan_details",
        "execute_case",
    ):
        assert getattr(case, name) is getattr(collect022, name), name


def test_collect022_installs_profile_before_constructing_its_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _profile(
        tmp_path,
        {
            "arguments": {"node": "profile-node", "region": "us-west-2"},
            "environment": {"AWS_REGION": "us-west-2"},
        },
    )
    monkeypatch.setenv(SITE_PROFILE_ENV, str(path))
    monkeypatch.setenv("AWS_REGION", "")
    monkeypatch.setattr(
        sys, "argv", ["collect022", "--run-dir", str(tmp_path / "unused")]
    )
    original_parser = collect022.parser
    blocked = Mock(
        side_effect=AssertionError("parsing must not enter the case lifecycle")
    )

    class Parsed(Exception):
        pass

    def parser() -> argparse.ArgumentParser:
        assert os.environ["AWS_REGION"] == "us-west-2", (
            "profile environment must exist before parser defaults are constructed"
        )
        record = applied_site_profile()
        assert record is not None and record["path"] == str(path)
        arguments = original_parser().parse_args()
        assert arguments.node == "profile-node"
        assert arguments.region == "us-west-2"
        raise Parsed

    monkeypatch.setattr(collect022, "parser", parser)
    for name in ("configure", "read_only_preflight", "plan_details", "execute_case"):
        monkeypatch.setattr(collect022, name, blocked)

    with pytest.raises(Parsed):
        collect022.main()

    blocked.assert_not_called()
    assert not (tmp_path / "unused").exists(), "profile tests must not create a run"


def test_runner_help_still_works_without_a_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(SITE_PROFILE_ENV, "")
    monkeypatch.setattr(sys, "argv", ["run_collector_acceptance.py", "--help"])
    blocked = Mock(side_effect=AssertionError("help must not enter the case lifecycle"))
    monkeypatch.setattr(
        collector,
        "CASE",
        replace(
            collector.CASE,
            configure=blocked,
            read_only_preflight=blocked,
            plan_details=blocked,
            execute_case=blocked,
        ),
    )

    with pytest.raises(SystemExit) as caught:
        collector.main()

    assert caught.value.code == 0
    assert "--site-profile" in capsys.readouterr().out
    blocked.assert_not_called()


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
