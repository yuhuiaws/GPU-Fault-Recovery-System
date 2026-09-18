from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal import (
    RemoveClusterRequest,
    _removal_identity,
    _remove_node_action_keys,
    _run_kubernetes_cleanup,
    _update_bootstrap_state,
    _verify_control_registry_absent,
    _verify_target_namespace_absent,
    _wait_target_namespace_absent,
    remove_cluster,
    resolve_cluster_id,
    resolve_removal_cluster_id,
)
from gpu_fault.admin.cluster_removal_resources import merge_removal_resources
from gpu_fault.admin.cluster_removal_state import STEP_DEPENDENCIES, canonical_digest
from gpu_fault.admin.node_key_proof import NodeKeyProof
from gpu_fault.admin.site import load_site
from tests.admin._cluster_removal_support import EKS_ARN, HP_ARN, RemovalScenario
from tests.admin._cluster_removal_support import snapshot as _snapshot
from tests.admin.test_admin_site import site_file


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return RemovalScenario(tmp_path, monkeypatch)


@pytest.mark.parametrize("arn", [EKS_ARN, HP_ARN])
def test_public_arn_retry_finishes_a_partial_site_commit(scenario, arn) -> None:
    scenario.failure = "bootstrap"
    with pytest.raises(BootstrapError, match="bootstrap"):
        remove_cluster(scenario.request(arn))
    assert load_site(scenario.path).release_config["clusters"] == []
    assert "SITE_COMMIT_STARTED" in scenario.state()["completed_steps"]
    assert "SITE_UPDATED" not in scenario.state()["completed_steps"]

    site = load_site(scenario.path)
    resolved = resolve_removal_cluster_id(
        site, arn, discover=lambda _arn: (EKS_ARN, "hp-gpu-a")
    )
    scenario.failure = None
    result = remove_cluster(
        RemoveClusterRequest(site, resolved, "REMOVE_GPU_CLUSTER", arn)
    )

    assert result["phase"] == "COMPLETED"
    assert scenario.calls.count("cleanup") == 1
    assert scenario.calls.count("aws-network") == 1
    assert scenario.calls.count("unregister") == 1
    assert scenario.calls.count("bootstrap") == 2
    assert set(scenario.state()["completed_steps"]) == STEP_DEPENDENCIES.keys()


@pytest.mark.parametrize("arn", [EKS_ARN, HP_ARN])
def test_cli_arn_retry_resumes_after_partial_site_commit(
    scenario, monkeypatch, capsys, arn
) -> None:
    from types import SimpleNamespace

    from gpu_fault.admin import cli

    scenario.failure = "bootstrap"
    with pytest.raises(BootstrapError, match="bootstrap"):
        remove_cluster(scenario.request(arn))
    assert load_site(scenario.path).release_config["clusters"] == []
    scenario.failure = None
    discoveries = []
    requests = []
    invoke_removal = cli.remove_cluster

    def discover(_runner, *, cluster_arn, **_kwargs):
        discoveries.append(cluster_arn)
        return SimpleNamespace(eks_arn=EKS_ARN, hyperpod_name="hp-gpu-a")

    def remove(request):
        requests.append(request)
        return invoke_removal(request)

    monkeypatch.setattr(cli, "discover_cluster", discover)
    monkeypatch.setattr(cli, "remove_cluster", remove)
    arguments = cli.parser().parse_args(
        [
            "remove-cluster",
            "--state-dir",
            str(scenario.path.parent),
            "--gpu-cluster-arn",
            arn,
            "--confirm",
            "REMOVE_GPU_CLUSTER",
        ]
    )

    assert cli.run(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "COMPLETED"
    assert output["cluster_id"] == "gpu-a"
    assert output["remaining_cluster_ids"] == []
    assert scenario.state()["phase"] == "COMPLETED"
    assert scenario.calls.count("cleanup") == 1
    assert scenario.calls.count("aws-network") == 1
    assert discoveries == ([HP_ARN] if arn == HP_ARN else [])
    assert requests[0].gpu_cluster_arn == arn


def test_completed_removal_does_not_skip_a_rejoined_member(scenario) -> None:
    remove_cluster(scenario.request())
    previous = scenario.state()["attempt_id"]
    scenario.path.write_bytes(scenario.original)
    target = load_site(scenario.path).release_config["clusters"][0]
    Path(target["token_file"]).write_text("r" * 64)
    scenario.namespace_uid = "gpu-namespace-2"
    scenario.node_uid = "node-uid-2"
    scenario.incarnation = "2026-02-01T00:00:00Z"
    scenario.lifecycle = "ACTIVE"

    result = remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED"
    assert scenario.state()["attempt_id"] != previous
    assert scenario.calls.count("cleanup") == 2
    assert scenario.calls.count("snapshot") == 2
    assert (
        scenario.path.parent / f"remove-cluster/gpu-a/history/{previous}.json"
    ).is_file(), "a new removal attempt must archive the previous completed attempt"
    assert (
        len(list((scenario.path.parent / "remove-cluster/gpu-a/attempts").iterdir()))
        == 2
    )


@pytest.mark.parametrize("changed", ["eks", "context", "cpu", "namespace"])
def test_partial_removal_refuses_site_identity_drift(scenario, changed) -> None:
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        remove_cluster(scenario.request())
    document = yaml.safe_load(scenario.path.read_text())
    if changed == "eks":
        document["spec"]["clusters"][0]["eksClusterArn"] = EKS_ARN + "-other"
    elif changed == "context":
        document["spec"]["clusters"][0]["context"] = "other-context"
    elif changed == "cpu":
        document["spec"]["cpu"]["eksArn"] += "-other"
    else:
        document["spec"]["namespace"] = "another-namespace"
    scenario.path.write_text(yaml.safe_dump(document, sort_keys=False))
    calls = list(scenario.calls)
    scenario.failure = None

    with pytest.raises(BootstrapError, match="identity"):
        remove_cluster(scenario.request())
    assert scenario.calls == calls


@pytest.mark.parametrize(
    "changed", ["eks", "namespace", "node", "token", "release", "registry"]
)
def test_partial_removal_refuses_live_reincarnation(scenario, changed) -> None:
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        remove_cluster(scenario.request())
    if changed == "eks":
        scenario.incarnation = "2026-03-01T00:00:00Z"
    elif changed == "namespace":
        scenario.namespace_uid = "different-namespace"
    elif changed == "node":
        scenario.node_uid = "different-node"
    elif changed == "release":
        scenario.release_identity = "c" * 64
    elif changed == "registry":
        scenario.lifecycle = "ACTIVE"
    else:
        Path(
            load_site(scenario.path).release_config["clusters"][0]["token_file"]
        ).write_text("r" * 64)
    calls = list(scenario.calls)
    scenario.failure = None

    with pytest.raises(BootstrapError, match="drifted|recreated|incarnation|conflicts"):
        remove_cluster(scenario.request())
    assert scenario.calls == calls


def test_namespace_barrier_precedes_credentials_and_aws_cleanup(scenario) -> None:
    scenario.failure = "namespace-wait"
    with pytest.raises(BootstrapError, match="namespace-wait"):
        remove_cluster(scenario.request())
    assert "KUBERNETES_QUIESCED" in scenario.state()["completed_steps"]
    assert "KUBERNETES_REMOVED" not in scenario.state()["completed_steps"]
    assert not {"unregister", "keys", "aws-network", "aurora"}.intersection(
        scenario.calls
    ), "credential or AWS cleanup ran before namespace absence was confirmed"
    assert not any(call.startswith("delete:") for call in scenario.calls), (
        "resource deletion ran before namespace absence was confirmed"
    )
    scenario.failure = None

    remove_cluster(scenario.request())

    assert scenario.calls.count("cleanup") == 1
    assert scenario.calls.index("namespace-wait") < scenario.calls.index("aws-network")
    assert scenario.calls.index("unregister") < scenario.calls.index("keys")


def test_shared_resource_dependency_is_rejected_before_cleanup(scenario) -> None:
    from gpu_fault.installation_resources import InstallationResourceSnapshot

    role = scenario.registry.resources[0]
    preserved = role.model_copy(
        update={
            "resource_key": "aws/iam/preserved/role",
            "resource_id": "preserved-role",
            "dependencies": [role.resource_key],
        }
    )
    snapshot = InstallationResourceSnapshot(
        site_id=scenario.registry.site_id,
        resources=[*scenario.registry.resources, preserved],
    )
    scenario.registry = snapshot.model_copy(update={"source_sha256": snapshot.digest()})

    with pytest.raises(BootstrapError, match="preserved resource"):
        remove_cluster(scenario.request())
    assert scenario.calls == ["snapshot"]


def test_absent_member_without_site_commit_is_not_a_tombstone(scenario) -> None:
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError):
        remove_cluster(scenario.request())
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"]["clusters"] = []
    scenario.path.write_text(yaml.safe_dump(document, sort_keys=False))

    with pytest.raises(BootstrapError, match="partial site commit"):
        resolve_removal_cluster_id(load_site(scenario.path), EKS_ARN)
    with pytest.raises(BootstrapError, match="partial site commit"):
        remove_cluster(scenario.request())


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:eks:us-east-1:111122223333:cluster/gpu-a",
        "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
        "arn:aws-cn:eks:us-east-1:123456789012:cluster/gpu-a",
        "arn:aws:eks:us-east-1:123456789012:nodegroup/gpu-a",
    ],
)
def test_resolver_requires_the_full_cluster_arn(tmp_path, arn) -> None:
    site = load_site(site_file(tmp_path))
    with pytest.raises(BootstrapError):
        resolve_cluster_id(site, arn)


@pytest.mark.parametrize(
    "discovered",
    [
        (EKS_ARN + "-other", "hp-gpu-a"),
        (EKS_ARN, "hp-other"),
        ("arn:aws:eks:us-east-1:111122223333:cluster/gpu-a", "hp-gpu-a"),
    ],
)
def test_hyperpod_name_or_eks_alias_alone_cannot_select_a_target(
    tmp_path, discovered
) -> None:
    site = load_site(site_file(tmp_path))
    with pytest.raises(BootstrapError):
        resolve_cluster_id(site, HP_ARN, discover=lambda _arn: discovered)


def test_tombstone_cannot_resolve_a_different_hyperpod_arn(scenario) -> None:
    remove_cluster(scenario.request())
    with pytest.raises(BootstrapError):
        resolve_removal_cluster_id(
            load_site(scenario.path),
            HP_ARN + "-new",
            discover=lambda _arn: (EKS_ARN, "hp-gpu-a"),
        )


def test_cpu_shared_vpc_remains_in_bootstrap_ownership(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    path = tmp_path / "bootstrap-state.json"
    association = {
        "vpc_id": "vpc-shared",
        "vpc_region": "us-east-1",
        "ownership": "CREATED",
    }
    write_json_atomic(
        path,
        {
            "site_id": "test-site",
            "resources": {"pki": {"vpc_associations": [association]}},
        },
    )
    _update_bootstrap_state(
        RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
        target_network={"vpc_id": "vpc-shared", "nat_eips": ["192.0.2.10"]},
        remaining_networks=[],
        cpu_vpc_id="vpc-shared",
    )
    resources = json.loads(path.read_text())["resources"]
    assert resources["pki"]["vpc_associations"] == [association]
    assert resources["nlb_network"]["gpu_nat_eips"] == ["192.0.2.10"]


def completed(stdout: str = "", *, code: int = 0, stderr: str = ""):
    from subprocess import CompletedProcess

    return CompletedProcess([], code, stdout, stderr)


@pytest.mark.parametrize(
    "stderr",
    [
        'context "not found" does not exist',
        "credential helper not found",
        'Error from server (Forbidden): namespaces "not found" is forbidden',
        'Error from server (NotFound): namespaces "gpu-fault-system" not found',
    ],
)
def test_namespace_verification_never_treats_a_failed_command_as_absence(
    tmp_path, monkeypatch, stderr
) -> None:
    site = load_site(site_file(tmp_path))
    request = RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER")
    target = {
        **site.release_config["clusters"][0],
        "expected_namespace_uid": "namespace-a",
    }
    monkeypatch.setattr(
        removal, "run_command", lambda *_a, **_k: completed(code=1, stderr=stderr)
    )

    with pytest.raises(BootstrapError, match="cannot read namespace"):
        _verify_target_namespace_absent(request, target)
    with pytest.raises(BootstrapError, match="cannot read namespace"):
        _wait_target_namespace_absent(site, target)


@pytest.mark.parametrize(
    "output",
    [
        "{}",
        "[]",
        "null",
        "{invalid",
        '{"kind":"Namespace","metadata":{"name":"other","uid":"uid"}}',
    ],
)
def test_successful_but_invalid_namespace_json_is_not_absence(
    tmp_path, monkeypatch, output
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(removal, "run_command", lambda *_a, **_k: completed(output))
    with pytest.raises(BootstrapError, match="namespace"):
        _verify_target_namespace_absent(
            RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
            site.release_config["clusters"][0],
        )


@pytest.mark.parametrize(
    "data", [None, {}, {"clusters.json": ""}, {"clusters.json": "invalid"}]
)
def test_missing_cpu_registry_data_cannot_prove_revocation(
    tmp_path, monkeypatch, data
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(
        removal,
        "_json_command",
        lambda *_a, **_k: {
            "kind": "Secret",
            "metadata": {
                "name": "gpu-fault-regional-clusters",
                "namespace": "gpu-fault-system",
                "uid": "uid",
            },
            "data": data,
        },
    )
    with pytest.raises(BootstrapError, match="registry"):
        _verify_control_registry_absent(
            RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER")
        )


def test_empty_valid_cpu_registry_proves_the_target_is_absent(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(
        removal,
        "_json_command",
        lambda *_a, **_k: {
            "kind": "Secret",
            "metadata": {
                "name": "gpu-fault-regional-clusters",
                "namespace": "gpu-fault-system",
                "uid": "uid",
            },
            "data": {"clusters.json": base64.b64encode(b"[]").decode()},
        },
    )
    monkeypatch.setattr(
        removal,
        "membership_runtime_snapshot",
        lambda _site: {
            "registry_generation": 1,
            "registry_content_sha256": "a" * 64,
            "live_release_identity_sha256": "b" * 64,
            "registry_cluster_states": {},
        },
    )
    _verify_control_registry_absent(
        RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER")
    )


def test_node_key_removal_uses_the_read_secret_version(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))
    proofs = iter(
        [
            NodeKeyProof("key-uid", "12", {"node-a": "a" * 64}, False),
            NodeKeyProof("key-uid", "13", {}, False),
        ]
    )
    monkeypatch.setattr(removal, "read_node_key_proof", lambda *_a, **_k: next(proofs))
    commands = []
    monkeypatch.setattr(
        removal, "run_command", lambda args, **_k: commands.append(args) or completed()
    )

    assert _remove_node_action_keys(site, ["node-a"]) == 1

    patch = json.loads(commands[0][commands[0].index("-p") + 1])
    assert patch == [
        {"op": "test", "path": "/metadata/uid", "value": "key-uid"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "12"},
        {"op": "remove", "path": "/data/node-a"},
    ]


def test_successful_cleanup_exit_requires_completed_bound_evidence(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    request = RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER")
    directory = tmp_path / "cleanup"
    directory.mkdir()
    monkeypatch.setattr(removal, "run_driver", lambda *_a, **_k: completed())

    with pytest.raises(BootstrapError, match="cleanup evidence"):
        _run_kubernetes_cleanup(request, site.release_config["clusters"][0], directory)


def test_completed_cleanup_evidence_is_bound_to_the_selected_target(
    tmp_path, monkeypatch
) -> None:
    import hashlib

    site = load_site(site_file(tmp_path))
    directory = tmp_path / "cleanup"
    directory.mkdir()
    commands = []

    def driver(arguments, **_kwargs):
        commands.append(arguments)
        path = Path(arguments[arguments.index("--state-file") + 1])
        document = {
            "schema_version": 2,
            "scope": "gpu",
            "mode": "clean",
            "node_mode": "uninstall",
            "phase": "CLEANUP_COMPLETED",
            "status": "COMPLETED",
            "config_sha256": hashlib.sha256(
                json.dumps(site.release_config, indent=2, sort_keys=True).encode()
            ).hexdigest(),
            "targets": {
                "namespace": site.release_config["namespace"],
                "cpu_kubeconfig": site.release_config["cpu_kubeconfig"],
                "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a"}],
            },
            "original_resources": [
                {
                    "scope": "gpu:gpu-a",
                    "context": "gpu-a",
                    "kind": "nodes",
                    "name": "all",
                    "previous": "node-a",
                }
            ],
        }
        document["content_sha256"] = canonical_digest(document)
        write_json_atomic(path, document)
        return completed()

    monkeypatch.setattr(removal, "run_driver", driver)
    request = RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER")
    target = site.release_config["clusters"][0]

    first = _run_kubernetes_cleanup(request, target, directory)
    resumed = _run_kubernetes_cleanup(request, target, directory)

    assert first == resumed
    assert len(commands) == 1


def test_foreign_completed_cleanup_cannot_satisfy_the_barrier(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    directory = tmp_path / "cleanup"
    directory.mkdir()
    evidence = {
        "schema_version": 2,
        "scope": "gpu",
        "mode": "clean",
        "node_mode": "uninstall",
        "phase": "CLEANUP_COMPLETED",
        "status": "COMPLETED",
        "targets": {"namespace": "foreign"},
    }
    evidence["content_sha256"] = canonical_digest(evidence)
    write_json_atomic(directory / "kubernetes-cleanup-001.json", evidence)
    monkeypatch.setattr(
        removal, "run_driver", lambda *_a, **_k: pytest.fail("must not run cleanup")
    )

    with pytest.raises(BootstrapError, match="conflicts"):
        _run_kubernetes_cleanup(
            RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
            site.release_config["clusters"][0],
            directory,
        )


@pytest.mark.parametrize(
    "change", ["hyperpod_arn", "eks_arn", "node_recovery", "context"]
)
def test_live_provider_identity_is_checked_before_cleanup(
    tmp_path, monkeypatch, change
) -> None:
    site = load_site(site_file(tmp_path))
    target = site.release_config["clusters"][0]
    response = {
        "ClusterArn": HP_ARN,
        "ClusterName": "hp-gpu-a",
        "NodeRecovery": "None",
        "Orchestrator": {"Eks": {"ClusterArn": EKS_ARN}},
    }
    if change == "hyperpod_arn":
        response["ClusterArn"] = HP_ARN + "-other"
    elif change == "eks_arn":
        response["Orchestrator"]["Eks"]["ClusterArn"] = EKS_ARN + "-other"
    elif change == "node_recovery":
        response["NodeRecovery"] = "Automatic"

    class Runner:
        def aws_json(self, *_args):
            return response

    monkeypatch.setattr(
        removal, "run_command", lambda *_a, **_k: completed("https://wrong.example")
    )
    with pytest.raises(BootstrapError, match="identity|policy|context"):
        _removal_identity(
            RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER", HP_ARN),
            target,
            Runner(),
            {"eks_created_at": "created", "eks_endpoint": "https://gpu.example"},
            {"eks_created_at": "created", "eks_endpoint": "https://cpu.example"},
        )


def test_completed_journal_does_not_hide_a_recreated_namespace(scenario) -> None:
    remove_cluster(scenario.request())
    scenario.namespace_uid = "recreated"
    previous = list(scenario.calls)

    with pytest.raises(BootstrapError, match="recreated"):
        remove_cluster(scenario.request())
    assert scenario.calls == previous


def test_removed_credential_may_be_absent_after_partial_site_commit(scenario) -> None:
    scenario.failure = "release-state"
    token = Path(scenario.request().site.release_config["clusters"][0]["token_file"])
    with pytest.raises(BootstrapError, match="release-state"):
        remove_cluster(scenario.request())
    assert not token.exists(), (
        "the removed cluster token remained after the partial site commit"
    )
    scenario.failure = None

    assert remove_cluster(scenario.request())["phase"] == "COMPLETED"


def test_registry_merge_preserves_unrelated_current_state() -> None:
    from gpu_fault.installation_resources import (
        InstallationResourceSnapshot,
        InstallationResourceStatus,
    )

    before = _snapshot()
    removed = InstallationResourceSnapshot(
        site_id=before.site_id,
        resources=[
            item.model_copy(update={"status": InstallationResourceStatus.DELETED})
            if item.resource_key == "aws/iam/executor/gpu-a/role"
            else item
            for item in before.resources
        ],
    )
    removed = removed.model_copy(update={"source_sha256": removed.digest()})
    current = InstallationResourceSnapshot(
        site_id=before.site_id,
        resources=[
            item.model_copy(update={"status": InstallationResourceStatus.FAILED})
            if item.resource_key == "aws/nlb"
            else item
            for item in before.resources
        ],
    )
    current = current.model_copy(update={"source_sha256": current.digest()})

    merged = merge_removal_resources(before, removed, current)

    statuses = {item.resource_key: item.status for item in merged.resources}
    assert statuses["aws/nlb"] is InstallationResourceStatus.FAILED
    assert statuses["aws/iam/executor/gpu-a/role"] is InstallationResourceStatus.DELETED
    assert current.resources[-1].status is InstallationResourceStatus.FAILED


def test_registry_merge_refuses_a_recreated_target_resource() -> None:
    from datetime import timedelta

    from gpu_fault.installation_resources import (
        InstallationResourceSnapshot,
        InstallationResourceStatus,
    )

    before = _snapshot()
    removed = InstallationResourceSnapshot(
        site_id=before.site_id,
        resources=[
            item.model_copy(update={"status": InstallationResourceStatus.DELETED})
            if item.resource_key == "aws/iam/executor/gpu-a/role"
            else item
            for item in before.resources
        ],
    )
    removed = removed.model_copy(update={"source_sha256": removed.digest()})
    current = InstallationResourceSnapshot(
        site_id=before.site_id,
        resources=[
            item.model_copy(
                update={"created_at": item.created_at + timedelta(seconds=1)}
            )
            if item.resource_key == "aws/iam/executor/gpu-a/role"
            else item
            for item in before.resources
        ],
    )
    current = current.model_copy(update={"source_sha256": current.digest()})

    with pytest.raises(BootstrapError, match="resource identity changed"):
        merge_removal_resources(before, removed, current)
