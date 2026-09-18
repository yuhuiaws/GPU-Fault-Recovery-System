"""Committed-main uninstall cases must retain the stronger journal safeguards."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.aws_commands import FinalSnapshotPolicy
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_command import retire_site_after_uninstall
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall import uninstall
from tests.admin.test_uninstall_lifecycle import STATE, Harness, registry


def journal_files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "names",
    [
        ("installation-resources-before.json", "kubernetes-cleanup.json"),
        (
            "state.consumed-20260915.json",
            "installation-resources-before.json",
            "installation-resources-before.json.sha256",
            "kubernetes-cleanup.json",
        ),
        ("unbound-operator-note.txt",),
    ],
    ids=["prior-artifacts", "consumed-state", "unbound-file"],
)
def test_unbound_uninstall_files_are_neither_archived_nor_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    directory = tmp_path / "uninstall"
    directory.mkdir()
    for name in names:
        (directory / name).write_text(
            json.dumps({"phase": "COMPLETED", "session": "prior"}), encoding="utf-8"
        )
    before = journal_files(directory)

    with pytest.raises(
        BootstrapError, match="transaction files lack their original journal"
    ):
        uninstall(harness.request(reset=True), runner=harness)

    assert journal_files(directory) == before, (
        "missing state proof must preserve every unbound file at its original path"
    )
    assert harness.exports == harness.syncs == 0, (
        "missing journal proof must fail before registry reads or writes"
    )
    assert harness.events == [], "unbound cleanup evidence cannot authorize deletion"
    assert not (directory / "state.json").exists(), (
        "a new journal cannot legitimize arbitrary leftover artifacts"
    )


@pytest.mark.parametrize("policy", ["retain", "skip"])
def test_reset_reinstall_retires_the_whole_journal_before_fresh_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: FinalSnapshotPolicy
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    request = replace(harness.request(reset=True), final_snapshot_policy=policy)
    uninstall(request, runner=harness)
    original_state = harness.state()
    directory = tmp_path / "uninstall"
    before = journal_files(directory)
    source = harness.site.source.read_bytes()

    archive = retire_site_after_uninstall(tmp_path)

    assert archive is not None, "completed uninstall must retire before reenrollment"
    assert not directory.exists(), "no prior working artifact may remain active"
    assert journal_files(archive / "uninstall") == before, (
        "retirement must preserve the state, registry, cleanup and digest files together"
    )
    harness.site.source.write_bytes(source)
    harness.site.source.chmod(0o600)
    harness.site = load_site(harness.site.source)
    harness.snapshot = registry(harness.site)
    harness.existing = {item.resource_key for item in harness.snapshot.resources}
    harness.namespace_uid = "namespace-new-installation"
    request = replace(harness.request(reset=True), final_snapshot_policy=policy)

    result = uninstall(request, runner=harness)

    current = harness.state()
    cleanup = STATE.read_state(directory / "kubernetes-cleanup.json")
    assert current["phase"] == "COMPLETED", (
        "the fresh cleanup must finish independently"
    )
    assert current["installation_id"] != original_state["installation_id"], (
        "reinstallation requires a new installation generation"
    )
    assert current["attempt_id"] != original_state["attempt_id"], (
        "the new uninstall cannot inherit the retired transaction"
    )
    assert current["registry_site_id"] != original_state["registry_site_id"], (
        "registry snapshots must be scoped to the new generation"
    )
    assert cleanup["namespace_snapshots"]["cpu"]["uid"] == harness.namespace_uid, (
        "the new cleanup must prove the newly installed namespace"
    )
    assert harness.events.count("cleanup") == 2 and harness.exports == 2, (
        "both installations need their own registry export and actual cleanup"
    )
    assert journal_files(archive / "uninstall") == before, (
        "a later cleanup must not modify archived evidence"
    )
    assert (result["aurora_final_snapshot"] is None) == (policy == "skip"), (
        "retirement and reinstall must preserve the selected snapshot branch"
    )


@pytest.mark.parametrize("policy", ["retain", "skip"])
def test_uninstall_retry_keeps_its_original_snapshot_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: FinalSnapshotPolicy
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    original_run = harness.run
    private_configs: list[Path] = []

    def run(arguments: Sequence[str], **kwargs: Any) -> str:
        if "--config" in arguments:
            config = Path(arguments[arguments.index("--config") + 1])
            assert config.stat().st_mode & 0o777 == 0o600, (
                "materialized uninstall config must remain owner-readable only"
            )
            assert config.parent.stat().st_mode & 0o777 == 0o700, (
                "materialized uninstall config needs a private directory"
            )
            private_configs.append(config)
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(harness, "run", run)
    request = replace(harness.request(reset=True), final_snapshot_policy=policy)
    harness.fail_cleanup = BootstrapError("modeled cleanup interruption")
    with pytest.raises(BootstrapError, match="modeled cleanup interruption"):
        uninstall(request, runner=harness)
    assert private_configs and all(
        not path.exists() and not path.parent.exists() for path in private_configs
    ), "failed uninstall must remove its temporary config and private directory"
    before = journal_files(tmp_path / "uninstall")
    events, exports, syncs = list(harness.events), harness.exports, harness.syncs
    config_count = len(private_configs)

    with pytest.raises(BootstrapError, match="conflicts on final_snapshot_policy"):
        uninstall(
            replace(
                request,
                final_snapshot_policy="skip" if policy == "retain" else "retain",
            ),
            runner=harness,
        )

    assert journal_files(tmp_path / "uninstall") == before, (
        "a rejected policy change must not rewrite any original cleanup proof"
    )
    assert harness.events == events and (harness.exports, harness.syncs) == (
        exports,
        syncs,
    ), "policy conflicts must be rejected before cleanup or registry I/O"
    assert len(private_configs) == config_count, (
        "rejected approval must not materialize another private config"
    )

    result = uninstall(request, runner=harness)

    assert harness.state()["phase"] == "COMPLETED", (
        "the unchanged original approval must remain resumable"
    )
    assert harness.state()["final_snapshot_policy"] == policy, (
        "retry cannot change the persisted snapshot authorization"
    )
    assert (result["aurora_final_snapshot"] is None) == (policy == "skip"), (
        "the completed result must distinguish retained and skipped snapshots"
    )
    assert len(private_configs) > config_count and all(
        not path.exists() and not path.parent.exists() for path in private_configs
    ), "successful retry must also clean up every temporary private config"
