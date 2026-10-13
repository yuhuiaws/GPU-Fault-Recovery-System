"""``uninstall`` and ``remove-cluster`` refuse to start over an unsynchronized registry.

Both delete what the Aurora installation registry records -- ``uninstall`` the
whole list, ``remove-cluster`` the removed cluster's rows, after which it drops
that cluster's roles from the bootstrap inventory -- so a resource a failed
registry sync never recorded would be left behind with nothing pointing at it.
While the deploy host's last sync record is a failure, missing or unreadable, a
new attempt refuses before its journal exists (an open journal would make the
remedy, a deploy, refuse or break the attempt's site binding), unless
``--accept-unsynchronized-registry`` is given; the verdict, the flag and the
recorded outcome go into the attempt's journal and result, and a resume keeps
the verdict its attempt started on.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import resource_registry_outcome as outcome
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal import RemoveClusterRequest, remove_cluster
from gpu_fault.admin.deploy_command import (
    refuse_deploy_during_uninstall,
    retire_site_after_uninstall,
)
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall import UninstallRequest, uninstall
from tests.admin._cluster_removal_support import EKS_ARN, RemovalScenario
from tests.admin._registry_sync_support import SYNCED_AT, record_registry_sync
from tests.admin.test_admin_site import site_file
from tests.admin.test_uninstall_lifecycle import Harness

FAILURE = (
    "installation resource registry not synchronized: BootstrapError: direct "
    "Aurora registry synchronization failed: ValueError"
)


def _spoil(state_dir: Path, how: str) -> None:
    """Leave the record ``how``: failed, missing or one of the unreadable shapes."""

    path = outcome.record_path(state_dir)
    if how == "failed":
        record_registry_sync(state_dir, release_id="release-a", error=FAILURE)
    elif how == "missing":
        path.unlink()
    elif how == "not-json":
        path.write_text("{truncated", encoding="utf-8")
    elif how == "unknown-status":
        path.write_text('{"schema_version": 1, "status": "MAYBE"}', encoding="utf-8")
    else:
        path.unlink()
        path.mkdir()


REFUSALS = {
    "failed": f"the last installation registry sync failed at {SYNCED_AT} "
    "\\(release release-a\\)",
    "missing": "no installation registry sync outcome is recorded",
    "not-json": "sync record is unreadable: .* is not valid JSON",
    "unknown-status": "sync record is unreadable: .* unknown status 'MAYBE'",
    "directory": "sync record is unreadable: .* cannot be read",
}


# --- uninstall -------------------------------------------------------------------


@pytest.mark.parametrize("how", sorted(REFUSALS))
def test_uninstall_refuses_before_its_journal_while_the_registry_is_unsynchronized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _spoil(tmp_path, how)

    with pytest.raises(BootstrapError, match=REFUSALS[how]) as refused:
        uninstall(harness.request(), runner=harness)

    message = str(refused.value)
    assert message.startswith("uninstall refuses to start: "), message
    assert f"gpu-fault-admin deploy --state-dir {tmp_path}" in message, message
    assert "`gpu-fault-admin config` re-roll" in message, message
    assert outcome.ACCEPT_FLAG in message, message
    assert not (tmp_path / "uninstall").exists(), "a refusal opens no uninstall"
    assert harness.exports == 0, "the registry must not be exported"
    assert harness.events == [], "nothing may be cleaned up or deleted"
    # The remedy stays available: no uninstall is in progress.
    refuse_deploy_during_uninstall(tmp_path)


def test_the_override_starts_the_uninstall_and_journals_the_recorded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _spoil(tmp_path, "failed")
    request = replace(harness.request(), accept_unsynchronized_registry=True)

    result = uninstall(request, runner=harness)

    evidence = harness.state()[outcome.FIELD]
    assert evidence["status"] == outcome.UNAVAILABLE
    assert evidence["outcome"]["error"] == FAILURE, "the recorded failure is kept"
    assert evidence["outcome"]["synced_at"] == SYNCED_AT
    assert (evidence["accept_flag"], evidence["accepted_unsynchronized"]) == (
        True,
        True,
    )
    assert evidence["record"] == str(outcome.record_path(tmp_path))
    assert result[outcome.FIELD] == evidence, "the result shows the same verdict"
    assert harness.state()["phase"] == "COMPLETED"


def test_a_synchronized_registry_needs_no_flag_and_is_journaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    uninstall(harness.request(), runner=harness)

    evidence = harness.state()[outcome.FIELD]
    assert evidence["status"] == outcome.SYNCED
    assert evidence["outcome"]["release_id"] == "release-a"
    assert (evidence["accept_flag"], evidence["accepted_unsynchronized"]) == (
        False,
        False,
    )


def test_a_completed_uninstall_retires_the_record_with_the_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next deploy archives ``release-deploy/`` with the retired site, so the
    old installation's success cannot vouch for the reinstall's registry."""

    harness = Harness(tmp_path, monkeypatch)
    uninstall(harness.request(), runner=harness)

    archive = retire_site_after_uninstall(tmp_path)

    assert archive is not None
    assert outcome.read_record(tmp_path) is None, "the record left with the site"
    assert outcome.read_record(archive) is not None, "it is kept in the archive"


def test_a_resumed_uninstall_keeps_the_verdict_it_started_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attempt's export fixes its deletion list; a resume is not re-judged.

    Re-judging could only wedge an attempt that already passed: while it is
    open, ``deploy`` refuses, so the record cannot be renewed in between.
    """

    harness = Harness(tmp_path, monkeypatch)
    harness.fail_sync = True
    with pytest.raises(BootstrapError, match="injected registry sync failure"):
        uninstall(harness.request(), runner=harness)
    started = harness.state()[outcome.FIELD]
    _spoil(tmp_path, "failed")

    result = uninstall(harness.request(), runner=harness)

    assert result["delete_policy_residuals"] == 0
    assert harness.state()[outcome.FIELD] == started, "the verdict is not rewritten"
    assert started["status"] == outcome.SYNCED


def test_uninstall_parses_the_flag_into_its_request(tmp_path: Path) -> None:
    site = load_site(site_file(tmp_path))
    base = ["uninstall", "--state-dir", str(tmp_path), "--confirm", "X"]
    plain = admin_cli.parser().parse_args(base)
    flagged = admin_cli.parser().parse_args([*base, outcome.ACCEPT_FLAG])
    legacy = argparse.Namespace(
        cpu_cluster="keep",
        confirm="UNINSTALL_GPU_FAULT",
        aurora_final_snapshot="retain",
        reset_database=False,
    )

    assert (
        plain.accept_unsynchronized_registry,
        flagged.accept_unsynchronized_registry,
    ) == (False, True)
    assert UninstallRequest.from_arguments(
        site, flagged
    ).accept_unsynchronized_registry, "the flag reaches the request"
    assert not UninstallRequest.from_arguments(
        site, legacy
    ).accept_unsynchronized_registry, "an older namespace defaults to off"
    with pytest.raises(BootstrapError, match="must be a boolean"):
        UninstallRequest(
            site, "keep", "UNINSTALL_GPU_FAULT", accept_unsynchronized_registry=1
        )


# --- remove-cluster ----------------------------------------------------------------


@pytest.mark.parametrize("how", sorted(REFUSALS))
def test_remove_cluster_refuses_before_its_journal_while_the_registry_is_unsynchronized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    scenario = RemovalScenario(tmp_path, monkeypatch)
    original = scenario.path.read_bytes()
    _spoil(tmp_path, how)

    with pytest.raises(BootstrapError, match=REFUSALS[how]) as refused:
        remove_cluster(scenario.request())

    message = str(refused.value)
    assert message.startswith("remove-cluster refuses to start: "), message
    assert "drops its roles from the bootstrap inventory" in message, message
    assert f"gpu-fault-admin deploy --state-dir {tmp_path}" in message, message
    assert outcome.ACCEPT_FLAG in message, message
    assert not (tmp_path / "remove-cluster").exists(), "no attempt or journal"
    assert scenario.calls == [], "nothing was read, drained or deleted"
    assert scenario.path.read_bytes() == original, "the site is untouched"


def test_the_override_starts_the_removal_and_journals_the_recorded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = RemovalScenario(tmp_path, monkeypatch)
    _spoil(tmp_path, "failed")

    result = remove_cluster(
        replace(scenario.request(), accept_unsynchronized_registry=True)
    )

    evidence = scenario.state()[outcome.FIELD]
    assert result["phase"] == "COMPLETED"
    assert evidence["status"] == outcome.UNAVAILABLE
    assert evidence["outcome"]["error"] == FAILURE, "the recorded failure is kept"
    assert evidence["accepted_unsynchronized"] is True
    assert result[outcome.FIELD] == evidence, "the result shows the same verdict"


def test_a_resumed_removal_keeps_the_verdict_it_started_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = RemovalScenario(tmp_path, monkeypatch)
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="simulated cleanup failure"):
        remove_cluster(scenario.request())
    started = scenario.state()[outcome.FIELD]
    _spoil(tmp_path, "missing")
    scenario.failure = None

    result = remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED"
    assert scenario.state()[outcome.FIELD] == started
    assert started["status"] == outcome.SYNCED


def test_a_rejoined_cluster_is_a_new_removal_attempt_and_judged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = RemovalScenario(tmp_path, monkeypatch)
    original = scenario.path.read_bytes()
    assert remove_cluster(scenario.request())["phase"] == "COMPLETED"
    scenario.path.write_bytes(original)  # the cluster joined again
    _spoil(tmp_path, "failed")

    with pytest.raises(BootstrapError, match="remove-cluster refuses to start"):
        remove_cluster(scenario.request())

    assert scenario.state()["phase"] == "COMPLETED", "the old tombstone stays"


def test_remove_cluster_parses_the_flag_into_its_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    seen: list[RemoveClusterRequest] = []

    def remove(request: RemoveClusterRequest) -> dict[str, str]:
        seen.append(request)
        return {"phase": "COMPLETED"}

    monkeypatch.setattr(admin_cli, "remove_cluster", remove)
    base = [
        "remove-cluster",
        "--state-dir",
        str(path.parent),
        "--gpu-cluster-arn",
        EKS_ARN,
        "--confirm",
        "REMOVE_GPU_CLUSTER",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    flagged = admin_cli.parser().parse_args([*base, outcome.ACCEPT_FLAG])
    assert admin_cli.run(flagged) == 0

    assert [item.accept_unsynchronized_registry for item in seen] == [False, True]
