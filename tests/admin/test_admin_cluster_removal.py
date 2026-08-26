from __future__ import annotations

from datetime import datetime, timezone

import yaml

from gpu_fault import admin_cluster_removal
from gpu_fault.admin_cluster_removal import (
    RemoveClusterRequest,
    _detach_network,
    _remove_target_aws_resources,
    _write_site_without_cluster,
    remove_cluster,
)
from gpu_fault.admin_resource_registry import write_installation_resource_snapshot
from gpu_fault.admin_site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from tests.admin.test_admin_site import site_file


def _resource(
    key: str,
    resource_type: str,
    resource_id: str,
    *,
    policy: InstallationResourceDeletePolicy,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    ownership = (
        InstallationResourceOwnership.CREATED
        if policy is InstallationResourceDeletePolicy.DELETE
        else InstallationResourceOwnership.EXTERNAL
    )
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type=resource_type,
        resource_id=resource_id,
        region="us-east-1",
        account_id="123456789012",
        ownership=ownership,
        delete_policy=policy,
        created_at=now,
        updated_at=now,
    )


def _snapshot() -> InstallationResourceSnapshot:
    value = InstallationResourceSnapshot(
        site_id="test-site",
        resources=[
            _resource(
                "aws/iam/executor/gpu-a/role",
                "iam_role",
                "executor-role",
                policy=InstallationResourceDeletePolicy.DELETE,
            ),
            _resource(
                "aws/iam/executor/gpu-a/oidc-provider",
                "iam_oidc_provider",
                "arn:aws:iam::123456789012:oidc-provider/test",
                policy=InstallationResourceDeletePolicy.PRESERVE,
            ),
            _resource(
                "cluster/gpu-a/eks",
                "gpu_eks",
                "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
                policy=InstallationResourceDeletePolicy.PRESERVE,
            ),
            _resource(
                "aws/route53/vpc-association/gpu-a",
                "route53_vpc_association",
                "Z123:us-east-1:vpc-gpu-a",
                policy=InstallationResourceDeletePolicy.DETACH,
            ),
            _resource(
                "aws/nlb",
                "nlb",
                "gpu-fault-nlb",
                policy=InstallationResourceDeletePolicy.DELETE,
            ),
        ],
    )
    return value.model_copy(update={"source_sha256": value.digest()})


def test_site_update_allows_the_last_gpu_cluster_to_be_removed(tmp_path) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
    )

    _write_site_without_cluster(request, tmp_path / "evidence")
    updated = load_site(path)

    assert updated.release_config["clusters"] == []
    assert (
        updated.release_config["runtime_profile"]["registration_cluster_id"] == "gpu-a"
    )


def test_target_aws_resources_are_deleted_or_detached_without_touching_cpu(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    deleted = []

    class Cleaner:
        def __init__(self, _site):
            pass

        def validate_supported(self, _resources):
            pass

        def delete(self, resource):
            deleted.append(resource.resource_key)

    monkeypatch.setattr(admin_cluster_removal, "ResourceCleaner", Cleaner)
    updated, evidence = _remove_target_aws_resources(
        RemoveClusterRequest(
            site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
        ),
        _snapshot(),
    )
    statuses = {item.resource_key: item.status for item in updated.resources}

    assert deleted == ["aws/iam/executor/gpu-a/role"]
    assert statuses["aws/iam/executor/gpu-a/role"] is InstallationResourceStatus.DELETED
    assert (
        statuses["aws/iam/executor/gpu-a/oidc-provider"]
        is InstallationResourceStatus.DETACHED
    )
    assert statuses["cluster/gpu-a/eks"] is InstallationResourceStatus.DETACHED
    assert (
        statuses["aws/route53/vpc-association/gpu-a"]
        is InstallationResourceStatus.DETACHED
    )
    assert statuses["aws/nlb"] is InstallationResourceStatus.ACTIVE
    assert evidence["deleted"] == ["aws/iam/executor/gpu-a/role"]


def test_cluster_network_detach_removes_only_exclusive_sources(
    tmp_path, monkeypatch
) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["dns"] = {"hostedZoneId": "Z123", "hostname": "api.test.internal"}
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    path.chmod(0o600)
    site = load_site(path)
    commands = []
    waits = []

    def run(arguments, **_kwargs):
        commands.append(arguments)
        return True

    monkeypatch.setattr(admin_cluster_removal, "_idempotent_aws", run)
    monkeypatch.setattr(
        admin_cluster_removal,
        "_wait_vpc_association_absent",
        lambda **kwargs: waits.append(kwargs),
    )

    result = _detach_network(
        RemoveClusterRequest(
            site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
        ),
        target_network={
            "vpc_id": "vpc-gpu-a",
            "nat_eips": ["192.0.2.10", "192.0.2.11"],
        },
        remaining_networks=[{"vpc_id": "vpc-gpu-b", "nat_eips": ["192.0.2.11"]}],
        cpu_vpc_id="vpc-cpu",
    )

    assert result == {
        "revoked_nat_eips": ["192.0.2.10"],
        "detached_vpc_id": "vpc-gpu-a",
    }
    assert len(commands) == 2
    assert "192.0.2.10/32" in commands[0]
    assert "192.0.2.11/32" not in commands[0]
    assert waits[0]["vpc_id"] == "vpc-gpu-a"


def test_remove_cluster_is_resumable_after_site_update(tmp_path, monkeypatch) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    calls = []

    def export_registry(current_site, state_dir):
        calls.append("registry")
        return_snapshot = _snapshot()
        write_installation_resource_snapshot(
            current_site,
            return_snapshot,
            path=state_dir / "installation-resources-before.json",
        )
        return return_snapshot

    def network(_runner, *, region, eks_arn):
        del region
        return {
            "vpc_id": "vpc-cpu" if eks_arn.endswith("/control") else "vpc-gpu",
            "nat_eips": [] if eks_arn.endswith("/control") else ["192.0.2.10"],
        }

    monkeypatch.setattr(admin_cluster_removal, "_export_registry", export_registry)
    monkeypatch.setattr(admin_cluster_removal, "_cluster_network", network)
    monkeypatch.setattr(
        admin_cluster_removal, "_target_nodes", lambda _site, _target: ["node-a"]
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_run_kubernetes_cleanup",
        lambda _request, _target, state_dir: state_dir / "cleanup.json",
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_clear_installer_annotations",
        lambda *_args: calls.append("annotations"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_delete_target_namespace",
        lambda *_args: calls.append("namespace"),
    )
    monkeypatch.setattr(
        admin_cluster_removal, "_remove_node_action_keys", lambda *_args: 1
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_run_control_plane_unregister",
        lambda *_args: calls.append("registry-secret"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_detach_network",
        lambda *_args, **_kwargs: {
            "revoked_nat_eips": ["192.0.2.10"],
            "detached_vpc_id": "vpc-gpu",
        },
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_remove_target_aws_resources",
        lambda _request, snapshot: (snapshot, {"deleted": [], "detached": []}),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "sync_installation_resource_snapshot",
        lambda *_args: calls.append("aurora"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_update_bootstrap_state",
        lambda *_args, **_kwargs: calls.append("bootstrap-state"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_delete_cluster_token",
        lambda *_args: calls.append("token"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_verify_target_absent",
        lambda *_args: calls.append("target-verified"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_verify_remaining_site",
        lambda *_args: calls.append("site-verified"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_sync_release_state",
        lambda *_args: calls.append("release-state"),
    )

    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
    )
    first = remove_cluster(request)
    resumed = remove_cluster(
        RemoveClusterRequest(
            site=load_site(path), cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
        )
    )

    assert first["phase"] == "COMPLETED"
    assert resumed["phase"] == "COMPLETED"
    assert first["remaining_cluster_ids"] == []
    assert load_site(path).release_config["clusters"] == []
    assert calls.count("registry") == 1
    assert calls.count("release-state") == 1
    assert calls.count("site-verified") == 1
