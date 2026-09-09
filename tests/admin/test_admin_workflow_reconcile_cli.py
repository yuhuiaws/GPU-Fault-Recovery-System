"""The ``workflow-reconcile`` command surface after the 2026-09-08 collapse.

One mode, one command: ``--state-dir``, optional ``--workflow-id`` /
``--incident-id`` / ``--max-items`` selectors, ``--reference``, ``--dry-run``.
The removed flags (``--plan``, ``--apply``, ``--mode``, ``--plan-sha256``,
``--blocked-kind``) are refused by the parser, the hidden ``-f`` path no longer
tracebacks on a missing ``--state-dir``, and a partial apply exits 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin.site import SiteConfigError


def _state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("placeholder\n", encoding="utf-8")
    return state_dir


@pytest.fixture
def reconcile_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_reconcile(site, state_dir, **kwargs):
        calls.append({"site": site, "state_dir": state_dir, **kwargs})
        return {
            "mode": "workflow-reconcile-apply",
            "applied_workflow_ids": ["workflow-a"],
            "failed_workflow_ids": [],
            "records_deleted": 0,
        }

    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(cli, "run_workflow_reconcile", fake_reconcile)
    return calls


@pytest.mark.parametrize(
    "removed",
    [
        ["--plan"],
        ["--apply"],
        ["--mode", "restore"],
        ["--mode", "retired-generation"],
        ["--plan-sha256", "a" * 64],
        ["--blocked-kind", "NEEDS_OPERATOR"],
    ],
)
def test_the_removed_flags_are_refused_by_the_parser(
    tmp_path: Path, removed: list[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            ["workflow-reconcile", "--state-dir", str(tmp_path), *removed]
        )


def test_the_command_wires_every_flag_into_one_invocation(
    tmp_path: Path, reconcile_calls: list[dict], capsys: pytest.CaptureFixture[str]
) -> None:
    state_dir = _state_dir(tmp_path)

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(state_dir),
                "--workflow-id",
                "workflow-a",
                "--workflow-id",
                "workflow-b",
                "--reference",
                "CHG-12345",
            ]
        )
    )

    assert exit_code == 0
    (call,) = reconcile_calls
    assert call["state_dir"] == state_dir.resolve()
    assert call["workflow_ids"] == ("workflow-a", "workflow-b")
    assert call["incident_ids"] == ()
    assert call["max_items"] is None
    assert call["reference"] == "CHG-12345"
    assert call["dry_run"] is False
    printed = json.loads(capsys.readouterr().out)
    assert printed["applied_workflow_ids"] == ["workflow-a"]


def test_dry_run_and_the_discovery_selectors_are_passed_through(
    tmp_path: Path, reconcile_calls: list[dict]
) -> None:
    state_dir = _state_dir(tmp_path)

    cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(state_dir),
                "--incident-id",
                "incident-a",
                "--max-items",
                "25",
                "--dry-run",
            ]
        )
    )

    (call,) = reconcile_calls
    assert call["workflow_ids"] == ()
    assert call["incident_ids"] == ("incident-a",)
    assert call["max_items"] == 25
    assert call["reference"] is None
    assert call["dry_run"] is True


def test_a_partial_apply_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "run_workflow_reconcile",
        lambda *_args, **_kwargs: {
            "applied_workflow_ids": [],
            "failed_workflow_ids": ["workflow-a"],
            "failures": {"workflow-a": "StaleWriteError: moved"},
        },
    )

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(_state_dir(tmp_path)),
                "--reference",
                "CHG-1",
            ]
        )
    )

    assert exit_code == 1


def test_the_hidden_site_file_path_without_a_state_dir_is_refused_not_a_traceback(
    tmp_path: Path, reconcile_calls: list[dict]
) -> None:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("placeholder\n", encoding="utf-8")

    with pytest.raises(SiteConfigError, match="requires --state-dir"):
        cli.run(
            cli.parser().parse_args(
                ["workflow-reconcile", "-f", str(site_file), "--reference", "CHG-1"]
            )
        )

    assert reconcile_calls == [], "nothing may run without a state directory"


def test_a_refusal_from_the_reconcile_surfaces_as_a_site_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_fault.admin.bootstrap_common import BootstrapError

    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: SimpleNamespace())

    def refuse(*_args, **_kwargs):
        raise BootstrapError("workflow-reconcile requires --reference")

    monkeypatch.setattr(cli, "run_workflow_reconcile", refuse)

    with pytest.raises(SiteConfigError, match="requires --reference"):
        cli.run(
            cli.parser().parse_args(
                ["workflow-reconcile", "--state-dir", str(_state_dir(tmp_path))]
            )
        )
