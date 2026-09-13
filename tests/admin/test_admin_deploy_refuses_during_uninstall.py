"""`deploy` refuses a site whose uninstall has not completed.

Live 2026-09-13: an uninstall had deleted the namespaces and Aurora and stopped
at a certificate; a `deploy --state-dir` started meanwhile re-created the SNS
topic and re-subscribed the administrator before stopping for the confirmation
link. The uninstall record is the site's word on what it is.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpu_fault.admin import deploy_command
from gpu_fault.admin.site import SiteConfigError


def _deploy(tmp_path) -> int:
    return deploy_command.run_deploy(
        SimpleNamespace(state_dir=tmp_path, rollback=False), hooks=object()
    )


def test_deploy_refuses_while_an_uninstall_is_in_progress(tmp_path) -> None:
    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text(
        json.dumps({"phase": "NON_AURORA_DELETE_IN_PROGRESS"}), encoding="utf-8"
    )

    with pytest.raises(SiteConfigError, match="uninstall of this site is in progress"):
        _deploy(tmp_path)


def test_deploy_refuses_an_unreadable_uninstall_record(tmp_path) -> None:
    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(SiteConfigError, match="phase unreadable"):
        _deploy(tmp_path)


def test_a_completed_uninstall_does_not_block_the_next_bootstrap(tmp_path) -> None:
    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text(
        json.dumps({"phase": "COMPLETED"}), encoding="utf-8"
    )

    deploy_command.refuse_deploy_during_uninstall(tmp_path)


def test_a_completed_uninstall_archives_the_bootstrap_checkpoints_once(
    tmp_path,
) -> None:
    """The deploy after an uninstall must create every resource again.

    Live 2026-09-13: bootstrap-state.json still trusted aurora, the IAM roles
    and the AMP workspace after the uninstall had deleted them, so the deploy
    skipped their creation and failed its preflight on the missing resources.
    """

    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text(
        json.dumps({"phase": "COMPLETED"}), encoding="utf-8"
    )
    (tmp_path / "bootstrap-state.json").write_text(
        json.dumps({"completed_tasks": ["aurora"]}), encoding="utf-8"
    )

    deploy_command.retire_bootstrap_checkpoints_after_uninstall(tmp_path)

    assert not (tmp_path / "bootstrap-state.json").exists(), (
        "the checkpoints must not be trusted after an uninstall"
    )
    archived = list(tmp_path.glob("bootstrap-state.uninstalled-*.json"))
    assert len(archived) == 1, "the checkpoints are archived, not lost"
    assert not (tmp_path / "uninstall" / "state.json").exists(), (
        "the uninstall record is consumed so the next deploy keeps its checkpoints"
    )
    assert list((tmp_path / "uninstall").glob("state.consumed-*.json")), (
        "the consumed record is kept for the audit trail"
    )
    # A second deploy finds no record and leaves the fresh checkpoints alone.
    (tmp_path / "bootstrap-state.json").write_text("{}", encoding="utf-8")
    deploy_command.retire_bootstrap_checkpoints_after_uninstall(tmp_path)
    assert (tmp_path / "bootstrap-state.json").exists(), (
        "only the first deploy after the uninstall archives"
    )


def test_an_unfinished_uninstall_keeps_the_checkpoints(tmp_path) -> None:
    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text(
        json.dumps({"phase": "READY_TO_DELETE_AURORA"}), encoding="utf-8"
    )
    (tmp_path / "bootstrap-state.json").write_text("{}", encoding="utf-8")

    deploy_command.retire_bootstrap_checkpoints_after_uninstall(tmp_path)

    assert (tmp_path / "bootstrap-state.json").exists(), (
        "an uninstall that has not finished decides nothing about the checkpoints"
    )
