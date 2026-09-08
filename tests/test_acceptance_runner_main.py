"""The shared ``main`` of the live acceptance runners.

Sixteen runners once carried a byte-identical ``main()``: install the site
profile, parse, restrict the umask, install the abort signals, configure, then
either print a plan (dry run) or authorize and execute. That body now lives
once in ``live_driver_guard.run_standard_case`` and every runner hands it a
``CaseRunner`` naming its own functions. These tests pin the ordering and the
exit codes of that spine, and the last test stops a second copy from growing
back anywhere under ``scripts/e2e``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional.live_driver_guard import (
    CaseRunner,
    CaseSurface,
    PlainCaseRunner,
    add_live_arguments,
    run_plain_case,
    run_selected_case,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "scripts" / "e2e"
REGIONAL = E2E / "regional"
CASE_ID = "GF-REGIONAL-TEST-002"
CONFIRMATION = "TEST002_EXECUTE"
DEADLINE = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

#: Every runner that delegates its ``main`` to the shared spine.
STANDARD_RUNNERS = (
    "run_collect016_training_recovery",
    "run_collect017_efa_plugin",
    "run_destr001_gpu_reset",
    "run_destr002_hyperpod_reboot",
    "run_destr003_warm_spare_failover",
    "run_destr008_warm_spare_shortage",
    "run_destr009_workload_restart",
    "run_destr010_fabric_manager_restart",
    "run_destr012_managed_recovery_guard",
    "run_destr014_branch_exhaustion",
    "run_destr015_parallel_branch_join",
    "run_destr016_preempting_reboot",
    "run_destr017_out_of_band_reboot_fence",
    "run_destr018_lifetime_deadline",
    "run_e2e002_multicluster_fault",
    "run_ha003_aurora_failover_reset",
    "run_ha004_waiting_reclaim_reset",
    "run_iso006_cluster_offline",
    "run_net007_transient_api_outage",
)
#: Runners that select their case at configure time, so the identity comes
#: from the settings rather than from module constants.
SELECTED_RUNNERS = ("run_collector_acceptance", "run_collector_destructive")


@dataclass
class FakeSettings:
    case_id: str = CASE_ID
    confirmation: str = CONFIRMATION

    def environment(self) -> dict[str, str]:
        return {"GPU_FAULT_TEST_ENVIRONMENT": "recorded"}


class Recorder:
    """Plain functions standing in for one case and for the spine's helpers.

    Every call is appended to ``calls`` so a test can assert the order the
    spine invoked them in, not merely that each was reached.
    """

    def __init__(self, *, errors: list[str]) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.errors = errors
        self.settings = FakeSettings()

    def _note(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    @property
    def names(self) -> list[str]:
        return [name for name, _args, _kwargs in self.calls]

    def call(self, name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        matches = [(args, kwargs) for n, args, kwargs in self.calls if n == name]
        assert len(matches) == 1, f"{name} called {len(matches)} times"
        return matches[0]

    # -- the per-case surface -------------------------------------------------

    def parser(self) -> argparse.ArgumentParser:
        self._note("parser")
        parser = argparse.ArgumentParser(prog="fake-runner")
        add_live_arguments(parser, confirmation=CONFIRMATION)
        return parser

    def configure(self, arguments: argparse.Namespace) -> FakeSettings:
        self._note("configure", arguments)
        return self.settings

    def read_only_preflight(
        self, settings: FakeSettings, case_dir: Path
    ) -> dict[str, Any]:
        self._note("read_only_preflight", settings, case_dir)
        return {"errors": list(self.errors), "probe": "seen"}

    def plan_details(
        self, settings: FakeSettings, preflight: dict[str, Any]
    ) -> dict[str, Any]:
        self._note("plan_details", settings, preflight)
        return {"preflight": preflight}

    def execute_case(
        self, settings: FakeSettings, run_dir: Path, attempt: int, deadline: datetime
    ) -> int:
        self._note("execute_case", settings, run_dir, attempt, deadline)
        return 7

    # -- the spine's collaborators, patched into live_driver_guard ------------

    def install_site_profile(self) -> None:
        self._note("install_site_profile")

    def install_abort_signals(self) -> None:
        self._note("install_abort_signals")

    # -- the plain (settings-less) surface -----------------------------------

    def plain_plan_details(self) -> dict[str, Any]:
        self._note("plain_plan_details")
        return {"static": "plan"}

    def run_case(self, run_dir: Path, attempt: int, deadline: datetime) -> int:
        self._note("run_case", run_dir, attempt, deadline)
        return 5

    def build_plan(self, **kwargs: Any) -> dict[str, Any]:
        self._note("build_plan", **kwargs)
        return {
            "case_id": kwargs["case_id"],
            "confirmation": kwargs["confirmation"],
            "details": kwargs["details"],
            "environment": kwargs.get("environment"),
        }

    def authorize_execution(
        self, arguments: argparse.Namespace, **kwargs: Any
    ) -> datetime:
        self._note("authorize_execution", arguments, **kwargs)
        return DEADLINE

    # -- assemble -----------------------------------------------------------

    def standard(self) -> CaseRunner[FakeSettings]:
        return CaseRunner(
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            parser=self.parser,
            configure=self.configure,
            read_only_preflight=self.read_only_preflight,
            plan_details=self.plan_details,
            execute_case=self.execute_case,
        )

    def selected(self) -> CaseSurface[FakeSettings]:
        return CaseSurface(
            parser=self.parser,
            configure=self.configure,
            read_only_preflight=self.read_only_preflight,
            plan_details=self.plan_details,
            execute_case=self.execute_case,
        )

    def plain(self) -> PlainCaseRunner:
        return PlainCaseRunner(
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            parser=self.parser,
            plan_details=self.plain_plan_details,
            run_case=self.run_case,
        )


@pytest.fixture(autouse=True)
def restore_umask() -> Iterator[int]:
    original = os.umask(0o022)
    os.umask(original)
    yield original
    os.umask(original)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    value = Recorder(errors=[])
    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)
    monkeypatch.setattr(
        live_driver_guard, "install_site_profile", value.install_site_profile
    )
    monkeypatch.setattr(
        live_driver_guard, "install_abort_signals", value.install_abort_signals
    )
    monkeypatch.setattr(live_driver_guard, "build_plan", value.build_plan)
    monkeypatch.setattr(
        live_driver_guard, "authorize_execution", value.authorize_execution
    )
    return value


def _argv(monkeypatch: pytest.MonkeyPatch, run_dir: Path, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fake-runner", "--run-dir", str(run_dir), *extra])


def test_dry_run_prints_the_plan_and_exits_zero_on_a_clean_preflight(
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, tmp_path, "--plan")

    result = run_standard_case(recorder.standard())

    assert result == 0
    assert recorder.names == [
        "install_site_profile",
        "parser",
        "install_abort_signals",
        "configure",
        "read_only_preflight",
        "plan_details",
        "build_plan",
    ]
    case_dir = tmp_path / "cases" / CASE_ID
    assert case_dir.is_dir(), "the case directory must exist"
    preflight_args, _ = recorder.call("read_only_preflight")
    assert preflight_args == (recorder.settings, case_dir)
    _, plan_kwargs = recorder.call("build_plan")
    assert plan_kwargs == {
        "run_dir": tmp_path,
        "case_id": CASE_ID,
        "attempt": 1,
        "confirmation": CONFIRMATION,
        "environment": {"GPU_FAULT_TEST_ENVIRONMENT": "recorded"},
        "details": {"preflight": {"errors": [], "probe": "seen"}},
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["case_id"] == CASE_ID
    assert printed["details"] == {"preflight": {"errors": [], "probe": "seen"}}


def test_dry_run_still_prints_the_plan_but_exits_one_on_preflight_errors(
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder.errors.append("target node is not Ready")
    _argv(monkeypatch, tmp_path, "--attempt", "3")

    result = run_standard_case(recorder.standard())

    assert result == 1
    assert "authorize_execution" not in recorder.names
    assert "execute_case" not in recorder.names
    _, plan_kwargs = recorder.call("build_plan")
    assert plan_kwargs["attempt"] == 3
    printed = json.loads(capsys.readouterr().out)
    assert printed["details"]["preflight"]["errors"] == ["target node is not Ready"]


def test_execute_authorizes_then_runs_the_case_with_the_deadline(
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(
        monkeypatch,
        tmp_path,
        "--execute",
        "--confirm",
        CONFIRMATION,
        "--maintenance-window-end",
        "2026-09-07T13:00:00Z",
        "--attempt",
        "2",
    )

    result = run_standard_case(recorder.standard())

    assert result == 7
    assert recorder.names == [
        "install_site_profile",
        "parser",
        "install_abort_signals",
        "configure",
        "authorize_execution",
        "execute_case",
    ]
    authorize_args, authorize_kwargs = recorder.call("authorize_execution")
    assert authorize_args[0].confirm == CONFIRMATION
    assert authorize_kwargs == {
        "case_id": CASE_ID,
        "confirmation": CONFIRMATION,
        "environment": {"GPU_FAULT_TEST_ENVIRONMENT": "recorded"},
    }
    execute_args, _ = recorder.call("execute_case")
    assert execute_args == (recorder.settings, tmp_path, 2, DEADLINE)
    assert capsys.readouterr().out == ""


def test_plain_case_prints_the_static_plan_after_installing_the_profile(
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The settings-less spine (CMD/NET hold cases) has no preflight and no
    # environment snapshot, but it must still install the site profile before
    # the parser exists: tests/regional/test_site_profile.py trusts this.
    _argv(monkeypatch, tmp_path, "--plan", "--attempt", "4")

    result = run_plain_case(recorder.plain())

    assert result == 0
    assert recorder.names == [
        "install_site_profile",
        "parser",
        "plain_plan_details",
        "build_plan",
    ]
    _, plan_kwargs = recorder.call("build_plan")
    assert plan_kwargs == {
        "run_dir": tmp_path,
        "case_id": CASE_ID,
        "attempt": 4,
        "confirmation": CONFIRMATION,
        "details": {"static": "plan"},
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["details"] == {"static": "plan"}, "the plan must be printed"


def test_plain_case_authorizes_then_runs_with_the_deadline(
    recorder: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(
        monkeypatch,
        tmp_path,
        "--execute",
        "--confirm",
        CONFIRMATION,
        "--maintenance-window-end",
        "2026-09-07T13:00:00Z",
        "--attempt",
        "2",
    )

    result = run_plain_case(recorder.plain())

    assert result == 5
    assert recorder.names == [
        "install_site_profile",
        "parser",
        "authorize_execution",
        "run_case",
    ]
    authorize_args, authorize_kwargs = recorder.call("authorize_execution")
    assert authorize_args[0].confirm == CONFIRMATION
    assert authorize_kwargs == {"case_id": CASE_ID, "confirmation": CONFIRMATION}
    run_args, _ = recorder.call("run_case")
    assert run_args == (tmp_path, 2, DEADLINE)
    assert capsys.readouterr().out == "", "execution prints no plan"


def test_the_umask_is_restricted_before_the_case_directory_exists(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _argv(monkeypatch, tmp_path, "--plan")
    os.umask(0o022)

    run_standard_case(recorder.standard())

    applied = os.umask(0o022)
    assert applied == 0o077
    case_dir = tmp_path / "cases" / CASE_ID
    assert case_dir.stat().st_mode & 0o777 == 0o700


def test_a_missing_case_function_fails_when_the_runner_is_declared(
    recorder: Recorder,
) -> None:
    with pytest.raises(TypeError, match="execute_case"):
        CaseRunner(  # type: ignore[call-arg]
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            parser=recorder.parser,
            configure=recorder.configure,
            read_only_preflight=recorder.read_only_preflight,
            plan_details=recorder.plan_details,
        )


def test_selected_case_takes_its_identity_from_the_configured_settings(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder.settings.case_id = "GF-REGIONAL-COLLECT-999"
    recorder.settings.confirmation = "COLLECT999_EXECUTE"
    _argv(monkeypatch, tmp_path, "--plan")

    result = run_selected_case(recorder.selected())

    assert result == 0
    assert (tmp_path / "cases" / "GF-REGIONAL-COLLECT-999").is_dir(), (
        "the selected case must get its own directory"
    )
    _, plan_kwargs = recorder.call("build_plan")
    assert plan_kwargs["case_id"] == "GF-REGIONAL-COLLECT-999"
    assert plan_kwargs["confirmation"] == "COLLECT999_EXECUTE"


def test_selected_case_rejects_a_foreign_confirmation_before_authorizing(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _argv(
        monkeypatch,
        tmp_path,
        "--execute",
        "--confirm",
        "SOME_OTHER_CASE_EXECUTE",
        "--maintenance-window-end",
        "2026-09-07T13:00:00Z",
    )

    with pytest.raises(RegionalFixtureError, match=CONFIRMATION):
        run_selected_case(recorder.selected())

    assert "authorize_execution" not in recorder.names
    assert "execute_case" not in recorder.names


def test_selected_case_executes_with_the_settings_identity(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _argv(
        monkeypatch,
        tmp_path,
        "--execute",
        "--confirm",
        CONFIRMATION,
        "--maintenance-window-end",
        "2026-09-07T13:00:00Z",
    )

    result = run_selected_case(recorder.selected())

    assert result == 7
    assert recorder.names[-2:] == ["authorize_execution", "execute_case"]
    _, authorize_kwargs = recorder.call("authorize_execution")
    assert authorize_kwargs["case_id"] == CASE_ID
    assert authorize_kwargs["confirmation"] == CONFIRMATION
    execute_args, _ = recorder.call("execute_case")
    assert execute_args[3] == DEADLINE


# -- the runners themselves --------------------------------------------------


@pytest.mark.parametrize("name", STANDARD_RUNNERS)
def test_standard_runner_declares_its_own_functions_as_the_case(name: str) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    case = module.CASE
    assert isinstance(case, CaseRunner), "CASE must be a CaseRunner"
    assert case.case_id == module.CASE_ID
    assert case.confirmation == module.CONFIRMATION
    assert case.parser is module.parser
    assert case.configure is module.configure
    assert case.read_only_preflight is module.read_only_preflight
    assert case.plan_details is module.plan_details
    assert case.execute_case is module.execute_case


@pytest.mark.parametrize("name", SELECTED_RUNNERS)
def test_selected_runner_declares_its_own_functions_as_the_case(name: str) -> None:
    module = importlib.import_module(f"scripts.e2e.regional.{name}")
    case = module.CASE
    assert type(case) is CaseSurface
    assert case.parser is module.parser
    assert case.configure is module.configure
    assert case.read_only_preflight is module.read_only_preflight
    assert case.plan_details is module.plan_details
    assert case.execute_case is module.execute_case


@pytest.mark.parametrize("name", (*STANDARD_RUNNERS, *SELECTED_RUNNERS))
def test_runner_answers_help_without_any_cluster_access(
    name: str, tmp_path: Path
) -> None:
    path = REGIONAL / f"{name}.py"
    environment = {
        key: value for key, value in os.environ.items() if key != SITE_PROFILE_ENV
    }
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"usage: {path.name}"), (
        "--help must print the runner's own usage"
    )
    assert "--execute" in result.stdout


def _main_body_digest(path: Path) -> str | None:
    """Digest of ``main``'s body, or None when it only delegates.

    A ``main`` that is a single ``return <call>`` is the shape the fold
    produces; sixteen of those are the point, not the smell.
    """

    tree = ast.parse(path.read_bytes(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "main":
            continue
        body = node.body
        if (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Call)
        ):
            return None
        dumped = "\n".join(ast.dump(statement) for statement in body)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return None


def test_no_two_runners_share_a_main_body() -> None:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(E2E.rglob("run_*.py")):
        if "__pycache__" in path.parts:
            continue
        digest = _main_body_digest(path)
        if digest is not None:
            groups[digest].append(path.relative_to(ROOT).as_posix())
    duplicated = [files for files in groups.values() if len(files) > 1]
    assert duplicated == [], (
        "runners carry an identical main(); fold them onto "
        f"live_driver_guard.run_standard_case: {duplicated}"
    )
