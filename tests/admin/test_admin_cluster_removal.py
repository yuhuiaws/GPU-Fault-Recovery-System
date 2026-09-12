from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone

import pytest
import yaml

from gpu_fault import node_installer_reconciler
from gpu_fault.admin import cluster_removal as admin_cluster_removal
from gpu_fault.admin.bootstrap_common import BootstrapError
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
                "aws/iam/adot-writer/gpu-a/role",
                "iam_role",
                "adot-writer-role",
                policy=InstallationResourceDeletePolicy.DELETE,
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

    # The executor role and the data-plane ADOT writer role go together; the
    # waves run on threads, so only the set is deterministic.
    assert sorted(deleted) == [
        "aws/iam/adot-writer/gpu-a/role",
        "aws/iam/executor/gpu-a/role",
    ], "remove-cluster left the cluster's ADOT writer role behind"
    assert statuses["aws/iam/executor/gpu-a/role"] is InstallationResourceStatus.DELETED
    assert (
        statuses["aws/iam/adot-writer/gpu-a/role"] is InstallationResourceStatus.DELETED
    )
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
    assert sorted(evidence["deleted"]) == [
        "aws/iam/adot-writer/gpu-a/role",
        "aws/iam/executor/gpu-a/role",
    ]


def _without(snapshot: InstallationResourceSnapshot, key: str):
    return snapshot.model_copy(
        update={
            "resources": [
                item for item in snapshot.resources if item.resource_key != key
            ]
        }
    )


def test_target_resources_still_require_exactly_one_executor_role(
    tmp_path, monkeypatch
) -> None:
    """The writer role is optional (no workspace, no role); the executor role is
    not, and a second role under the executor prefix is a registry the removal
    must not act on."""

    site = load_site(site_file(tmp_path))

    class Cleaner:
        def __init__(self, _site):
            pass

    monkeypatch.setattr(admin_cluster_removal, "ResourceCleaner", Cleaner)
    request = RemoveClusterRequest(
        site=site, cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
    )

    with pytest.raises(BootstrapError, match="exactly one target Executor IAM role"):
        _remove_target_aws_resources(
            request, _without(_snapshot(), "aws/iam/executor/gpu-a/role")
        )

    duplicated = _snapshot().model_copy(
        update={
            "resources": [
                *_snapshot().resources,
                _resource(
                    "aws/iam/executor/gpu-a/role-2",
                    "iam_role",
                    "executor-role-2",
                    policy=InstallationResourceDeletePolicy.DELETE,
                ),
            ]
        }
    )
    with pytest.raises(BootstrapError, match="exactly one target Executor IAM role"):
        _remove_target_aws_resources(request, duplicated)


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


def _install_removal_harness(monkeypatch, calls: list) -> None:
    """Every side effect of a removal, recorded in ``calls``."""

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
        "prune_workload_namespace_rbac",
        lambda *_args: (calls.append("workload-rbac"), ["role/gpu-fault-x"])[1],
    )

    def kubernetes_cleanup(_request, _target, state_dir):
        calls.append("kubernetes-cleanup")
        return state_dir / "cleanup.json"

    monkeypatch.setattr(
        admin_cluster_removal, "_run_kubernetes_cleanup", kubernetes_cleanup
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
    monkeypatch.setattr(
        admin_cluster_removal,
        "refresh_failure_domain_map",
        lambda site: calls.append(
            f"failure-domain-map:{len(site.release_config['clusters'])}"
        ),
    )


def test_remove_cluster_is_resumable_after_site_update(tmp_path, monkeypatch) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    calls = []
    _install_removal_harness(monkeypatch, calls)

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
    # The engine's label-owned workload RBAC goes before the fail-closed
    # cleanup, which would otherwise meet it as unregistered and refuse
    # (live 2026-09-12); the evidence records what was deleted.
    assert calls.count("workload-rbac") == 1
    assert calls.index("workload-rbac") < calls.index("kubernetes-cleanup")
    state = json.loads(
        (tmp_path / "remove-cluster" / "gpu-a" / "state.json").read_text("utf-8")
    )
    assert state["evidence"]["KUBERNETES_QUIESCED"][
        "workload_namespace_rbac_deleted"
    ] == ["role/gpu-fault-x"]
    assert calls.count("site-verified") == 1
    # The failure-domain map is re-rendered once, for the remaining (empty)
    # cluster set, before the release state moves on.
    assert calls.count("failure-domain-map:0") == 1
    assert calls.index("failure-domain-map:0") < calls.index("release-state")


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


def _annotations_the_reconciler_writes() -> set[str]:
    """Every ``gpu-fault.io/installer-*`` annotation the reconciler defines."""

    return {
        value
        for name, value in vars(node_installer_reconciler).items()
        if name.startswith("INSTALLER_")
        and name.endswith("_ANNOTATION")
        and isinstance(value, str)
    }


def test_cluster_removal_clears_every_annotation_the_reconciler_writes() -> None:
    """Final review M3 (Task 15 residual).

    Removal cleared five installer annotations; the reconciler writes eleven.
    ``installer-attempts`` and ``installer-retry-after`` survived a removal,
    so a cluster added back later inherited a node's failure count and backed
    off for up to an hour on its first failed install.
    """

    written = _annotations_the_reconciler_writes()
    assert len(written) >= 11, f"the reconciler defines fewer than expected: {written}"
    missing = written - set(admin_cluster_removal.INSTALLER_ANNOTATIONS)
    assert not missing, (
        "cluster removal must clear every installer annotation the reconciler "
        f"writes, or a re-added cluster inherits it; not cleared: {sorted(missing)}"
    )


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


def test_workload_namespace_rbac_prune_selects_by_the_engine_label_only(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    site.release_config["gpu_kubeconfig"] = str(tmp_path / "gpu.kubeconfig")
    seen: list[list[str]] = []

    def run(arguments, **_kwargs):
        seen.append(list(arguments))
        return subprocess.CompletedProcess(
            arguments,
            0,
            'role.rbac.authorization.k8s.io "gpu-fault-cluster-executor" deleted\n'
            'rolebinding.rbac.authorization.k8s.io "gpu-fault-cluster-executor" deleted\n',
            "",
        )

    monkeypatch.setattr(admin_cluster_removal.subprocess, "run", run)
    target = next(
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] == "gpu-a"
    )

    deleted = admin_cluster_removal.prune_workload_namespace_rbac(site, target)

    [command] = seen
    assert command[:3] == ["kubectl", "--kubeconfig", str(tmp_path / "gpu.kubeconfig")]
    assert command[3:5] == ["--context", target["context"]]
    assert command[5:] == [
        "delete",
        "roles,rolebindings",
        "--all-namespaces",
        "-l",
        "gpu-fault.io/workload-namespace-rbac=true",
        "--ignore-not-found",
        "--wait=true",
    ], "selection is by the engine's label across every namespace, nothing else"
    assert len(deleted) == 2


def test_resolve_cluster_id_reaches_an_unfinished_removal_the_site_already_dropped(
    tmp_path,
) -> None:
    """SITE_UPDATED removes the cluster from site.yaml; a rerun after a later
    step failed (live 2026-09-12: sync-state) must still resolve the ARN
    through the removal's DISCOVERED evidence, or "rerun with the same
    arguments" is a promise the last steps cannot keep."""

    site = _site_with_clusters(tmp_path)
    arn = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
    site.release_config["clusters"] = []
    state_dir = site.source.parent / "remove-cluster" / "gpu-a"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "cluster_id": "gpu-a",
                "phase": "SITE_UPDATED",
                "evidence": {
                    "DISCOVERED": {
                        "target": {
                            "cluster_id": "gpu-a",
                            "eks_cluster_arn": arn,
                            "hyperpod_cluster_name": "hp-gpu-a",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert admin_cluster_removal.resolve_cluster_id(site, arn) == "gpu-a"

    (state_dir / "state.json").write_text(
        json.dumps({"cluster_id": "gpu-a", "phase": "COMPLETED", "evidence": {}}),
        encoding="utf-8",
    )
    with pytest.raises(BootstrapError, match="managed clusters: none"):
        admin_cluster_removal.resolve_cluster_id(site, arn)


def test_a_completed_removal_does_not_shadow_a_cluster_that_joined_again(
    tmp_path, monkeypatch
) -> None:
    """Live 2026-09-12: the cluster removed in the morning was joined again,
    and the second remove-cluster found the morning's COMPLETED record, said
    COMPLETED in one second and touched nothing. A finished removal of a
    cluster the site manages again is history; the removal starts over."""

    path = site_file(tmp_path)
    original = path.read_bytes()
    calls = []
    _install_removal_harness(monkeypatch, calls)

    first = remove_cluster(
        RemoveClusterRequest(
            site=load_site(path), cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
        )
    )
    assert first["phase"] == "COMPLETED"
    path.write_bytes(original)  # the cluster joined again

    second = remove_cluster(
        RemoveClusterRequest(
            site=load_site(path), cluster_id="gpu-a", confirmation="REMOVE_GPU_CLUSTER"
        )
    )

    assert second["phase"] == "COMPLETED"
    assert calls.count("registry") == 2, "the second removal must run, not resume"
    assert load_site(path).release_config["clusters"] == [], (
        "the second removal must drop the cluster from the site again"
    )
    archived = list(
        (tmp_path / "remove-cluster" / "gpu-a").glob("state.completed-*.json")
    )
    assert len(archived) == 1, "the morning's record must be kept as history"
