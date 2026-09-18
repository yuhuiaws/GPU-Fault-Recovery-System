from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cluster_removal_kubernetes as kubernetes
from gpu_fault.admin import cluster_removal_resources as resources
from gpu_fault.admin import operation_lock, profile_approval
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceSnapshot
from tests.admin._aws_cleanup_support import resource
from tests.admin.test_admin_site import site_file


def snapshot(entries, site_id="test-site"):
    value = InstallationResourceSnapshot(site_id=site_id, resources=entries)
    return value.model_copy(update={"source_sha256": value.digest()})


def executor():
    return resource("iam_role", "executor").model_copy(
        update={"resource_key": "aws/iam/executor/gpu-a/role"}
    )


def test_removal_dependency_cycle_has_no_execution_wave():
    first, second = executor(), resource("iam_policy", "example")
    first = first.model_copy(update={"dependencies": [second.resource_key]})
    second = second.model_copy(update={"dependencies": [first.resource_key]})
    with pytest.raises(BootstrapError, match="dependency cycle"):
        resources.deletion_waves([first, second])


@pytest.mark.parametrize("change", ["site", "region", "account"])
def test_removal_plan_rejects_foreign_registry_scope(tmp_path, change):
    site = load_site(site_file(tmp_path))
    item = executor()
    site_id = "test-site"
    if change == "site":
        site_id = "foreign"
        item = item.model_copy(update={"site_id": site_id})
    else:
        item = item.model_copy(
            update={
                "region" if change == "region" else "account_id": "us-west-2"
                if change == "region"
                else "111122223333"
            }
        )
    with pytest.raises(BootstrapError, match="another site|scope or preservation"):
        resources.target_resource_plan(site, "gpu-a", snapshot([item], site_id))


def test_removal_merge_cannot_cross_installation_identities():
    before = snapshot([executor()])
    foreign = snapshot(
        [executor().model_copy(update={"site_id": "foreign"})], "foreign"
    )
    with pytest.raises(BootstrapError, match="different sites"):
        resources.merge_removal_resources(before, before, foreign)


def test_removal_wave_failure_is_reported_without_returning_terminal_snapshot(tmp_path):
    site = load_site(site_file(tmp_path))
    calls = []

    def delete(item):
        calls.append(item.resource_key)
        raise TimeoutError("example delete did not complete")

    cleaner = SimpleNamespace(validate_supported=lambda _items: None, delete=delete)
    with pytest.raises(BootstrapError, match="failed to delete.*did not complete"):
        resources.remove_target_resources(
            site, "gpu-a", snapshot([executor()]), cleaner_factory=lambda _site: cleaner
        )
    assert calls == ["aws/iam/executor/gpu-a/role"]


@pytest.mark.parametrize("bindings", [{}, {"node-a": ""}, {"": "uid-a"}])
def test_annotation_cleanup_empty_or_invalid_bindings_never_query_nodes(bindings):
    calls = []

    def runner(*_args, **_options):
        calls.append("query")

    if bindings:
        with pytest.raises(BootstrapError, match="incomplete UID bindings"):
            kubernetes.clear_installer_annotations(
                runner,
                ["kubectl"],
                hyperpod_name="example",
                node_uids=bindings,
                annotations=["example/key"],
            )
    else:
        kubernetes.clear_installer_annotations(
            runner,
            ["kubectl"],
            hyperpod_name="example",
            node_uids=bindings,
            annotations=["example/key"],
        )
    assert calls == []


@pytest.mark.parametrize("code,body", [(1, "{}"), (0, '{"items":null}')])
def test_annotation_cleanup_requires_successful_structured_inventory(code, body):
    calls = []

    def read(arguments, **_options):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, code, body, "example unavailable")

    with pytest.raises(
        BootstrapError, match="cannot verify nodes|node identity drifted"
    ):
        kubernetes.clear_installer_annotations(
            read,
            ["kubectl"],
            hyperpod_name="example",
            node_uids={"node-a": "uid-a"},
            annotations=["example/key"],
        )
    assert len(calls) == 1


def test_namespace_wait_rejects_recreation_without_any_delete():
    calls = []

    def read(arguments, **_options):
        calls.append(arguments)
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "kind": "Namespace",
                    "metadata": {"name": "example", "uid": "recreated"},
                }
            ),
            "",
        )

    with pytest.raises(BootstrapError, match="recreated during deletion"):
        kubernetes.wait_namespace_absent(
            read, ["kubectl"], "example", "original", timeout_seconds=1
        )
    assert len(calls) == 1
    assert "delete" not in calls[0]


def test_standard_input_is_not_an_inherited_operation_lock(monkeypatch):
    monkeypatch.setenv(operation_lock.SITE_OPERATION_LOCK_FD_ENV, "0")
    assert operation_lock.inherited_lock_pass_fds() == ()


def test_inherited_lock_descriptor_must_name_the_exact_lock_file(tmp_path, monkeypatch):
    (tmp_path / operation_lock.SITE_OPERATION_LOCK).touch(mode=0o600)
    with (tmp_path / "other").open("w") as other:
        monkeypatch.setenv(
            operation_lock.SITE_OPERATION_LOCK_FD_ENV, str(other.fileno())
        )
        assert operation_lock.inherited_site_operation_lock_fd(tmp_path) is None


def test_current_thread_can_reenter_its_proven_inherited_operation_lock(
    tmp_path, monkeypatch
):
    with operation_lock.site_operation_lock(tmp_path, wait=False) as descriptor:
        monkeypatch.setenv(operation_lock.SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
        with operation_lock.site_operation_lock(tmp_path, wait=False) as nested:
            assert nested == descriptor


@pytest.mark.parametrize("digest", ["", "invalid", "../other"])
def test_profile_approval_archive_requires_sha256_identifier(tmp_path, digest):
    with pytest.raises(
        profile_approval.ProfileApprovalError, match="digest is invalid"
    ):
        profile_approval.profile_approval_archive_path(tmp_path, digest)


def test_profile_approval_rejects_invalid_reviewed_digest_before_identity_lookup(
    tmp_path,
):
    with pytest.raises(
        profile_approval.ProfileApprovalError, match="SHA-256 is invalid"
    ):
        profile_approval.approve_profile(
            tmp_path,
            reference="CHG-EXAMPLE",
            expected_plan_sha256="invalid",
            approver_identity="example",
        )
