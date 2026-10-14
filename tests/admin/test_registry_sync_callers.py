"""A registry API conflict leaves every admin caller failed closed (batch 91).

deploy/config completion, join-cluster, its rollback, remove-cluster and
uninstall all write the registry through ``sync_installation_resource_snapshot``
and treat a ``BootstrapError`` as a stop. The conflict the API now answers with
409 reaches them as ``InstallationRegistryConflict`` -- a ``BootstrapError`` that
names key and field -- and nothing of the refused write lands. Each caller checks
the identities it sends against a fresh read first, so the 409 is the backstop
for a row that changed between that read and the write.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.admin.cluster_join_commit import _sync_registry
from gpu_fault.admin.resource_registry import InstallationRegistryConflict
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)
from scripts import release_deploy_evidence
from tests.admin._registry_pod_support import PodRegistry
from tests.admin.test_admin_site import site_file


def _row(key: str, resource_id: str, **updates: object) -> InstallationResource:
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type="iam_role",
        resource_id=resource_id,
        resource_arn=f"arn:aws:iam::123456789012:role/{resource_id}",
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
    ).model_copy(update=updates)


def test_a_join_commit_fails_closed_on_a_concurrent_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    pod = PodRegistry(monkeypatch)
    unrelated = _row("aws/iam/control-plane/role", "control-plane")
    pod.store.save_installation_resource(unrelated)
    joined = _row("aws/iam/executor/gpu-b/role", "gpu-b-executor")
    concurrent = _row("aws/iam/executor/gpu-b/role", "another-executor")
    pod.before_post = lambda: pod.store.save_installation_resource(concurrent)

    with pytest.raises(InstallationRegistryConflict) as refused:
        _sync_registry(
            site,
            before_path=tmp_path / "installation-resources-before-001.json",
            state_dir=tmp_path,
            resources=[joined],
        )

    assert refused.value.conflicts == [
        ("aws/iam/executor/gpu-b/role", ["resource_id", "resource_arn"])
    ]
    assert pod.store.list_installation_resources("test-site") == [
        unrelated,
        concurrent,
    ], "the refused join delta landed"
    assert not (tmp_path / "installation-resources-after-001.json").exists(), (
        "a refused registry commit wrote the join's after-snapshot"
    )


def test_a_release_completion_records_the_named_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """deploy/config: the completion warning and the durable sync record name
    the key and fields, so ``status`` and the deletion-verb refusals do too."""

    site = load_site(site_file(tmp_path))

    def refuse(_site: object) -> Path:
        raise InstallationRegistryConflict(
            "installation resource identity cannot change",
            [("aws/nlb", ["dependencies"])],
        )

    monkeypatch.setattr(
        release_deploy_evidence, "sync_installation_resource_registry", refuse
    )
    record, warning = release_deploy_evidence.installation_resource_registry_record(
        site, release_id="release-a"
    )
    assert record["status"] == "UNAVAILABLE"
    assert warning is not None
    assert "aws/nlb (dependencies)" in record["error"] == warning
