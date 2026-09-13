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


def test_a_completed_uninstall_retires_the_site_records_once(tmp_path) -> None:
    """The deploy after an uninstall must start as a new site.

    Live 2026-09-13: bootstrap-state.json still trusted aurora, the IAM roles
    and the AMP workspace after the uninstall had deleted them, and site.yaml
    plus the signed success record made the source deploy classify the
    directory as an installed site; the deploy skipped every creation task and
    failed its preflight on the missing resources.
    """

    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text(
        json.dumps({"phase": "COMPLETED"}), encoding="utf-8"
    )
    for name in deploy_command.RETIRED_AFTER_UNINSTALL:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "admin-config").mkdir()
    (tmp_path / "admin-config" / "desired.json").write_text("{}", encoding="utf-8")

    archive = deploy_command.retire_site_after_uninstall(tmp_path)

    assert archive is not None and archive.is_dir(), "the records are retired, not lost"
    for name in deploy_command.RETIRED_AFTER_UNINSTALL:
        assert not (tmp_path / name).exists(), (
            f"{name} must not describe the site any more"
        )
        assert (archive / name).is_file(), f"{name} is kept for the audit trail"
    assert (tmp_path / "admin-config" / "desired.json").is_file(), (
        "the administrator's desired config survives a keep-uninstall"
    )
    assert not (tmp_path / "uninstall" / "state.json").exists(), (
        "the uninstall record is consumed so the next deploy keeps the fresh records"
    )
    assert list((tmp_path / "uninstall").glob("state.consumed-*.json")), (
        "the consumed record is kept for the audit trail"
    )
    # A second deploy finds no record and leaves the fresh site alone.
    (tmp_path / "site.yaml").write_text("fresh", encoding="utf-8")
    assert deploy_command.retire_site_after_uninstall(tmp_path) is None, (
        "only the first deploy after the uninstall retires"
    )
    assert (tmp_path / "site.yaml").read_text(encoding="utf-8") == "fresh", (
        "the fresh site document is untouched"
    )


def test_an_unfinished_uninstall_keeps_the_checkpoints(tmp_path) -> None:
    (tmp_path / "uninstall").mkdir()
    (tmp_path / "uninstall" / "state.json").write_text(
        json.dumps({"phase": "READY_TO_DELETE_AURORA"}), encoding="utf-8"
    )
    (tmp_path / "bootstrap-state.json").write_text("{}", encoding="utf-8")

    assert deploy_command.retire_site_after_uninstall(tmp_path) is None, (
        "an uninstall that has not finished decides nothing"
    )
    assert (tmp_path / "bootstrap-state.json").exists(), (
        "an uninstall that has not finished decides nothing about the checkpoints"
    )
