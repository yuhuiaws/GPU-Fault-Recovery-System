from __future__ import annotations

from pathlib import Path

import pytest

from scripts import deploy_source_identity, staging_deploy


@pytest.mark.parametrize(
    "relative_path",
    [
        "deploy/control-plane/regional/probes/workflow_safety.py",
        "deploy/control-plane/regional/probes/new_snapshot_probe.py",
        "deploy/control-plane/regional/new_snapshot_driver.sh",
        "deploy/control-plane/tools/new_snapshot_tool.py",
        "deploy/control-plane/tools/new_snapshot_tool.sh",
        "deploy/control-plane/tools/wait-for-kubernetes-job.sh",
    ],
)
def test_snapshot_probes_and_tools_invalidate_deploy_host_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    for pattern in deploy_source_identity.DEPLOY_HOST_ORCHESTRATION_INPUTS:
        path = tmp_path / pattern.replace("*", "fixture")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(
        deploy_source_identity,
        "deploy_host_bundle_identity",
        lambda _root: {"sha256": "b" * 64},
    )
    original = deploy_source_identity.deploy_host_identity(tmp_path)
    changed_file = tmp_path / relative_path
    changed_file.write_text("first revision\n", encoding="utf-8")
    added = deploy_source_identity.deploy_host_identity(tmp_path)
    changed_file.write_text("second revision\n", encoding="utf-8")
    modified = deploy_source_identity.deploy_host_identity(tmp_path)

    assert relative_path not in original["orchestration"]["files"]
    assert relative_path in added["orchestration"]["files"]
    assert len({item["sha256"] for item in (original, added, modified)}) == 3, (
        "adding or changing a snapshot probe/tool did not invalidate deploy-host reuse"
    )
    assert original["bundle"] == added["bundle"] == modified["bundle"]
    assert (
        added["orchestration"]["files"][relative_path]["sha256"]
        != modified["orchestration"]["files"][relative_path]["sha256"]
    )

    source = staging_deploy.SourceCheckout(
        repository_root=tmp_path,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    application = {"sha256": "c" * 64}
    previous = {
        "identities": {"application": application, "deploy_host": original},
        "source": {"fingerprint": source.fingerprint},
    }
    assert (
        staging_deploy.classify_source_deploy(
            previous,
            {"application": application, "deploy_host": modified},
            source=source,
            site_exists=True,
            live_matches=True,
        )
        == "DEPLOY_HOST_ONLY"
    ), "unchanged runtime content hid a changed deployment probe/tool"

    changed_file.unlink()
    assert deploy_source_identity.deploy_host_identity(tmp_path) == original
