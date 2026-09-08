from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone

import pytest
import yaml

from gpu_fault.admin import cluster_removal as admin_cluster_removal
from gpu_fault.admin.cluster_removal import (
    RemoveClusterRequest,
    _detach_network,
    _parallel_cluster_networks,
    _remove_target_aws_resources,
    _verify_removal_parallel,
    _write_site_without_cluster,
    remove_cluster,
)
from gpu_fault.admin.resource_registry import write_installation_resource_snapshot
from gpu_fault.admin.site import load_site
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
    route53_changes = []

    def run(arguments, **_kwargs):
        commands.append(arguments)
        return True

    monkeypatch.setattr(admin_cluster_removal, "_idempotent_aws", run)
    monkeypatch.setattr(
        admin_cluster_removal,
        "disassociate_vpc_from_hosted_zone",
        lambda **kwargs: route53_changes.append(kwargs)
        or {"changed": True, "change_id": "/change/C123", "change_status": "INSYNC"},
    )
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
        "route53_change_id": "/change/C123",
        "route53_change_status": "INSYNC",
    }
    assert len(commands) == 1
    permissions = json.loads(commands[0][commands[0].index("--ip-permissions") + 1])
    cidrs = {item["CidrIp"] for item in permissions[0]["IpRanges"]}
    assert cidrs == {"192.0.2.10/32"}
    assert route53_changes == [
        {"hosted_zone_id": "Z123", "region": "us-east-1", "vpc_id": "vpc-gpu-a"}
    ]
    assert waits[0]["vpc_id"] == "vpc-gpu-a"


def test_route53_disassociation_waits_for_change_insync(monkeypatch) -> None:
    waits = []
    monkeypatch.setattr(
        admin_cluster_removal.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {"ChangeInfo": {"Id": "/change/C123", "Status": "PENDING"}}
            ),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "wait_route53_change_insync",
        lambda change_id: waits.append(change_id),
    )

    result = admin_cluster_removal.disassociate_vpc_from_hosted_zone(
        hosted_zone_id="Z123", region="us-east-1", vpc_id="vpc-gpu-a"
    )

    assert result == {
        "changed": True,
        "change_id": "/change/C123",
        "change_status": "INSYNC",
    }
    assert waits == ["/change/C123"]


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
        "_request_target_namespace_deletion",
        lambda *_args: calls.append("namespace-request"),
    )
    monkeypatch.setattr(
        admin_cluster_removal,
        "_wait_target_namespace_absent",
        lambda *_args: calls.append("namespace-wait"),
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
        "_verify_removal_parallel",
        lambda *_args: calls.extend(["target-verified", "site-verified"]),
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


def test_remove_network_discovery_is_bounded_parallel(monkeypatch) -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()
    four_started = threading.Event()

    def network(_runner, *, region, eks_arn):
        nonlocal active, maximum
        del region
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_started.set()
        assert four_started.wait(timeout=2), "four network queries did not overlap"
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"vpc_id": eks_arn.rsplit("/", 1)[-1], "nat_eips": []}

    monkeypatch.setattr(admin_cluster_removal, "_cluster_network", network)
    target = {"eks_cluster_arn": "arn:aws:eks:us-east-1:123:cluster/target"}
    remaining = [
        {"eks_cluster_arn": f"arn:aws:eks:us-east-1:123:cluster/gpu-{index}"}
        for index in range(5)
    ]

    target_network, remaining_networks, cpu_network = _parallel_cluster_networks(
        object(),
        region="us-east-1",
        target=target,
        remaining=remaining,
        cpu_eks_arn="arn:aws:eks:us-east-1:123:cluster/cpu",
    )

    assert maximum == 4
    assert target_network["vpc_id"] == "target"
    assert [item["vpc_id"] for item in remaining_networks] == [
        f"gpu-{index}" for index in range(5)
    ]
    assert cpu_network["vpc_id"] == "cpu"


def test_target_resource_deletion_uses_dependency_waves(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))
    now = datetime.now(timezone.utc)

    def resource(key, resource_type, dependencies=()):
        return InstallationResource(
            site_id="test-site",
            resource_key=key,
            resource_type=resource_type,
            resource_id=key,
            region="us-east-1",
            account_id="123456789012",
            ownership=InstallationResourceOwnership.CREATED,
            delete_policy=InstallationResourceDeletePolicy.DELETE,
            dependencies=list(dependencies),
            created_at=now,
            updated_at=now,
        )

    oidc = resource("aws/iam/executor/gpu-a/oidc-provider", "iam_oidc_provider")
    role = resource("aws/iam/executor/gpu-a/role", "iam_role", (oidc.resource_key,))
    associations = [
        resource(
            f"aws/iam/executor/gpu-a/association-{index}",
            "eks_pod_identity_association",
            (role.resource_key,),
        )
        for index in range(2)
    ]
    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=[oidc, role, *associations]
    )
    snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
    calls: list[str] = []
    active = 0
    maximum = 0
    lock = threading.Lock()
    two_started = threading.Event()

    class Cleaner:
        def __init__(self, _site):
            pass

        def validate_supported(self, _resources):
            pass

        def delete(self, item):
            nonlocal active, maximum
            with lock:
                calls.append(item.resource_key)
                active += 1
                maximum = max(maximum, active)
                if (
                    item.resource_key.startswith("aws/iam/executor/gpu-a/association-")
                    and active == 2
                ):
                    two_started.set()
            if item.resource_key.startswith("aws/iam/executor/gpu-a/association-"):
                assert two_started.wait(timeout=2), (
                    "independent association deletes did not overlap"
                )
            time.sleep(0.01)
            with lock:
                active -= 1

    monkeypatch.setattr(admin_cluster_removal, "ResourceCleaner", Cleaner)

    _remove_target_aws_resources(
        RemoveClusterRequest(
            site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
        ),
        snapshot,
    )

    role_index = calls.index(role.resource_key)
    oidc_index = calls.index(oidc.resource_key)
    assert maximum == 2
    assert all(calls.index(item.resource_key) < role_index for item in associations), (
        "dependent IAM role was deleted before its associations"
    )
    assert role_index < oidc_index


def test_remove_final_checks_run_in_parallel(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))
    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
    )
    active = 0
    maximum = 0
    lock = threading.Lock()
    five_started = threading.Event()

    def check(*_args) -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 5:
                five_started.set()
        assert five_started.wait(timeout=2), "final checks did not all overlap"
        time.sleep(0.01)
        with lock:
            active -= 1

    for name in (
        "_verify_target_namespace_absent",
        "_verify_control_registry_absent",
        "_verify_preserved_gpu_eks",
        "_verify_preserved_hyperpod",
        "_verify_remaining_site",
    ):
        monkeypatch.setattr(admin_cluster_removal, name, check)

    _verify_removal_parallel(request, site.release_config["clusters"][0], site)

    assert maximum == 5


def _site_with_clusters(tmp_path):
    return load_site(site_file(tmp_path))


def test_resolve_cluster_id_matches_an_eks_arn_without_aws(tmp_path) -> None:
    site = _site_with_clusters(tmp_path)

    resolved = admin_cluster_removal.resolve_cluster_id(
        site,
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        discover=lambda _arn: pytest.fail("an EKS ARN must not call AWS"),
    )

    assert resolved == "gpu-a"


def test_resolve_cluster_id_resolves_a_hyperpod_arn_through_discovery(tmp_path) -> None:
    site = _site_with_clusters(tmp_path)
    asked: list[str] = []

    def discover(arn: str) -> tuple[str, str]:
        asked.append(arn)
        return "arn:aws:eks:us-east-1:123456789012:cluster/other", "hp-gpu-a"

    resolved = admin_cluster_removal.resolve_cluster_id(
        site,
        "arn:aws:sagemaker:us-east-1:123456789012:cluster/abc123def456",
        discover=discover,
    )

    assert resolved == "gpu-a"
    assert asked == ["arn:aws:sagemaker:us-east-1:123456789012:cluster/abc123def456"]


def test_resolve_cluster_id_refuses_unknown_and_non_cluster_arns(tmp_path) -> None:
    site = _site_with_clusters(tmp_path)

    with pytest.raises(admin_cluster_removal.BootstrapError, match="managed clusters"):
        admin_cluster_removal.resolve_cluster_id(
            site, "arn:aws:eks:us-east-1:123456789012:cluster/gpu-z"
        )
    with pytest.raises(admin_cluster_removal.BootstrapError, match="eks or sagemaker"):
        admin_cluster_removal.resolve_cluster_id(
            site, "arn:aws:iam::123456789012:role/executor"
        )
    with pytest.raises(admin_cluster_removal.BootstrapError, match="AWS discovery"):
        admin_cluster_removal.resolve_cluster_id(
            site, "arn:aws:sagemaker:us-east-1:123456789012:cluster/abc123def456"
        )
