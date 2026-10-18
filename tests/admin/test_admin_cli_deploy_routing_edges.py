"""``gpu-fault-admin`` routing edges: the hidden ``-f`` deploy path and the
``workflow-reconcile`` dispositions that need a state directory.

``deploy -f site.yaml`` is the driver-only shape: it proves custody, refuses a
notification change on the command line, runs the engine's preflight before
the deploy and surfaces either child's failure status unchanged. The incident
close and departed-agent dispositions live under ``--state-dir`` (the lock and
the archive are there), so the ``-f`` shape alone is refused for them too.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin.site import SiteConfigError


def test_release_history_lives_under_the_state_dir_only(tmp_path: Path) -> None:
    assert cli.release_history_environment(argparse.Namespace()) == {}
    assert cli.release_history_environment(argparse.Namespace(state_dir=None)) == {}
    assert cli.release_history_environment(argparse.Namespace(state_dir=tmp_path)) == {
        cli.RELEASE_HISTORY_DIR_ENV: str(tmp_path.resolve() / "history")
    }


def _site_file(tmp_path: Path) -> Path:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("placeholder\n", encoding="utf-8")
    return site_file


@pytest.mark.parametrize(
    ("flags", "problem"),
    [
        (["--close-escalated"], "--close-escalated requires --state-dir"),
        (["--close-quarantined"], "--close-quarantined requires --state-dir"),
        (["--close-incident", "inc-1"], "--close-incident requires --state-dir"),
        (["--retire-departed-agents"], "--retire-departed-agents requires --state-dir"),
    ],
)
def test_dispositions_under_f_alone_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], problem: str
) -> None:
    def never(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no site may be loaded without a state directory")

    monkeypatch.setattr(cli, "load_site", never)
    arguments = cli.parser().parse_args(
        ["workflow-reconcile", "-f", str(_site_file(tmp_path)), *flags]
    )

    with pytest.raises(SiteConfigError, match=problem):
        cli.run(arguments)


def test_several_close_selectors_on_one_request_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_site", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(
        cli, "run_incident_close", lambda *_a, **_k: pytest.fail("must not run")
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _site_file(state_dir)
    arguments = cli.parser().parse_args(
        ["workflow-reconcile", "--state-dir", str(state_dir), "--close-escalated"]
    )
    # The parser keeps the three exclusive; the command still refuses a
    # namespace that carries two, so a driver cannot slip one past it.
    arguments.close_quarantined = True

    with pytest.raises(SiteConfigError, match="not several"):
        cli.run(arguments)


class EngineDriver:
    """``run_driver`` double answering a fixed status per engine verb."""

    def __init__(self, statuses: dict[str, int]) -> None:
        self.statuses = statuses
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        return SimpleNamespace(returncode=self.statuses.get(command[1], 0))

    @property
    def verbs(self) -> list[str]:
        return [command[1] for command in self.commands]


def _deploy_by_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: EngineDriver
) -> tuple[argparse.Namespace, list[str]]:
    site_file = _site_file(tmp_path)
    custody_checks: list[str] = []
    site = SimpleNamespace(audit_summary={"site": "staging"}, repository_root=tmp_path)
    monkeypatch.setattr(cli, "load_site", lambda *_a, **_k: site)
    monkeypatch.setattr(
        cli,
        "assert_site_custody_current",
        lambda _site, _runner: custody_checks.append("custody"),
    )

    @contextmanager
    def materialized(_site: object) -> Iterator[Path]:
        yield tmp_path / "release-config.json"

    monkeypatch.setattr(cli, "materialized_release_config", materialized)
    monkeypatch.setattr(cli, "effective_environment", lambda _site: {})
    monkeypatch.setattr(cli, "run_driver", driver)
    return cli.parser().parse_args(["deploy", "-f", str(site_file)]), custody_checks


def test_deploy_by_file_proves_custody_then_runs_preflight_before_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    driver = EngineDriver({})
    arguments, custody_checks = _deploy_by_file(tmp_path, monkeypatch, driver)

    assert cli.run(arguments) == 0

    assert custody_checks == ["custody"]
    assert driver.verbs == ["preflight", "deploy"]
    assert driver.commands[0][2:3] == ["--for-deploy"]
    assert '"gpu_fault_admin"' in capsys.readouterr().err


def test_a_failed_engine_preflight_stops_the_deploy_with_its_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = EngineDriver({"preflight": 7})
    arguments, _ = _deploy_by_file(tmp_path, monkeypatch, driver)

    assert cli.run(arguments) == 7
    assert driver.verbs == ["preflight"], "the deploy never started"


def test_a_failed_engine_deploy_returns_its_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = EngineDriver({"deploy": 3})
    arguments, _ = _deploy_by_file(tmp_path, monkeypatch, driver)

    assert cli.run(arguments) == 3
    assert driver.verbs == ["preflight", "deploy"]


def test_a_notification_change_on_a_deploy_by_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = EngineDriver({})
    arguments, custody_checks = _deploy_by_file(tmp_path, monkeypatch, driver)
    arguments.alert_email = "ops@example.com"

    with pytest.raises(SiteConfigError, match="applied through IaC"):
        cli.run(arguments)
    assert custody_checks == ["custody"], "custody is proven before the refusal"
    assert driver.commands == [], "the engine is never reached"
