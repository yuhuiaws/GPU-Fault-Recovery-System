from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace

import pytest
import yaml

from gpu_fault.admin import aws_commands
from gpu_fault.admin.bootstrap_common import BootstrapError
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_observe as observe
from scripts.e2e.regional.boot032_observe import aws as checked_aws
from tests.regional._cov95_boot032_world import World


@pytest.mark.parametrize(
    ("status", "stdout", "stderr", "expected"),
    [
        (0, '{"cluster": {}}', "", {"cluster": {}}),
        (254, "", "ResourceNotFoundException", None),
        (
            255,
            "",
            "An error occurred (ResourceNotFoundException) when calling DescribeCluster: missing",
            None,
        ),
    ],
)
def test_aws_read_uses_checked_exact_absence_protocol(
    monkeypatch, status, stdout, stderr, expected
):
    calls = []

    def command(arguments):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, status, stdout, stderr)

    monkeypatch.setattr(aws_commands, "bounded_command", command)
    value = checked_aws(
        "us-east-1",
        "eks",
        "describe-cluster",
        "--name",
        "fixture",
        absent=("ResourceNotFoundException",),
    )
    assert value == expected, (
        "only the native service-specific absence protocol may return None"
    )
    assert calls == [
        [
            "aws",
            "eks",
            "describe-cluster",
            "--region",
            "us-east-1",
            "--name",
            "fixture",
            "--output",
            "json",
        ]
    ], "AWS read must preserve the explicit region, target and structured output"


@pytest.mark.parametrize(
    ("status", "stdout", "stderr"),
    [
        (1, "", "ResourceNotFoundException"),
        (254, "", "AccessDenied"),
        (254, "", "credential helper not found"),
        (0, "", ""),
        (0, "[]", ""),
        (0, "{", ""),
    ],
)
def test_aws_unknown_reads_never_become_absence(monkeypatch, status, stdout, stderr):
    monkeypatch.setattr(
        aws_commands,
        "bounded_command",
        lambda arguments: subprocess.CompletedProcess(
            arguments, status, stdout, stderr
        ),
    )
    with pytest.raises(BootstrapError):
        checked_aws(
            "us-east-1",
            "eks",
            "describe-cluster",
            absent=("ResourceNotFoundException",),
        )


@pytest.mark.parametrize(
    "change",
    [
        "document",
        "current-context",
        "cluster-name",
        "server",
        "ca",
        "tls",
        "duplicate-context",
    ],
)
def test_kubeconfig_identity_is_parsed_and_fails_closed(tmp_path, monkeypatch, change):
    world = World(tmp_path, monkeypatch)
    site = world.target
    spec = contract.cluster_specs(site)[0]
    path = contract.kubeconfig(site, "cpu")
    value = yaml.safe_load(path.read_text())
    if change == "document":
        value = []
    elif change == "current-context":
        value["current-context"] = None
    elif change == "cluster-name":
        value["contexts"][0]["context"]["cluster"] = ""
    elif change == "server":
        value["clusters"][0]["cluster"]["server"] = "https://foreign.example.invalid"
    elif change == "ca":
        value["clusters"][0]["cluster"]["certificate-authority-data"] = "Zm9yZWlnbg=="
    elif change == "tls":
        value["clusters"][0]["cluster"]["insecure-skip-tls-verify"] = True
    else:
        value["contexts"].append(copy.deepcopy(value["contexts"][0]))
    path.write_text(yaml.safe_dump(value))
    with pytest.raises((contract.UninstallCaseError, ValueError)):
        observe.kube_target(site, spec, world.cloud[spec["eks_name"]]["eks"]["cluster"])
    assert not world.settings.native_dir.exists(), (
        "bad kubeconfig must be refused without native state"
    )


@pytest.mark.parametrize("tags", [None, {}, [{"Key": "purpose"}]])
def test_unreadable_fixture_tags_do_not_qualify_ordinary_clusters(
    tmp_path, monkeypatch, tags
):
    world = World(tmp_path, monkeypatch)
    spec = contract.cluster_specs(world.target)[0]
    world.cloud[spec["eks_name"]]["tags"] = {"Tags": tags}
    with pytest.raises((ValueError, KeyError)):
        observe.cluster_observation(
            world.target, spec, fixture_id=world.settings.fixture_id
        )
    assert not world.settings.native_dir.exists(), (
        "missing explicit fixture tags must block native entry"
    )


@pytest.mark.parametrize("remaining", ["eks", "hp", "both", "none"])
def test_cpu_delete_observation_uses_aws_without_querying_deleted_kubernetes(
    tmp_path, monkeypatch, remaining
):
    world = World(tmp_path, monkeypatch)
    spec = contract.cluster_specs(world.target)[0]
    before = copy.deepcopy(world.cloud[spec["eks_name"]])
    if remaining not in {"eks", "both"}:
        world.cloud[spec["eks_name"]]["eks"] = None
    if remaining not in {"hp", "both"}:
        world.cloud[spec["eks_name"]]["hp"] = None
    calls = []

    def forbidden(*arguments, **kwargs):
        calls.append(arguments)
        raise PermissionError("CPU kube API unavailable after retirement")

    monkeypatch.setattr(observe, "kubernetes_uid", forbidden)
    result = observe.cluster_observation(
        world.target,
        spec,
        cpu_deletion_started=True,
        fixture_id=world.settings.fixture_id,
    )
    assert calls == [], (
        "CPU absence must come from checked AWS observations, never failed Kube reads"
    )
    if remaining != "both":
        assert result["eks_absent"] is (remaining not in {"eks", "both"}), (
            "EKS absence must be explicit"
        )
        assert result["hyperpod_absent"] is (remaining not in {"hp", "both"}), (
            "HyperPod absence must be explicit"
        )
    if remaining in {"eks", "both"}:
        assert result["eks_created_at"] == before["eks"]["cluster"]["createdAt"], (
            "remaining EKS incarnation must remain bound"
        )
    if remaining in {"hp", "both"}:
        assert result["hyperpod_arn"] == before["hp"]["ClusterArn"], (
            "remaining HyperPod incarnation must remain bound"
        )


@pytest.mark.parametrize("output", ["uid=", "not-jsonpath", "uid=two values"])
def test_successful_kubernetes_transport_still_requires_a_uid(
    tmp_path, monkeypatch, output
):
    world = World(tmp_path, monkeypatch)
    monkeypatch.setattr(world, "command", lambda *_a, **_k: output)
    with pytest.raises(contract.UninstallCaseError, match="UID"):
        observe.kubernetes_uid(
            world.target,
            contract.cluster_specs(world.target)[0],
            "namespace",
            "kube-system",
        )


def test_resource_read_error_after_confirmed_half_deletion_is_not_full_absence(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    spec = contract.cluster_specs(world.target)[0]

    def read(_region, service, *_args, **_kwargs):
        if service == "eks":
            return None
        raise PermissionError("HyperPod observation is unknown")

    monkeypatch.setattr(observe, "aws", read)
    with pytest.raises(PermissionError):
        observe.cluster_observation(world.target, spec, cpu_deletion_started=True)
    assert not world.settings.native_dir.exists(), (
        "one NotFound cannot prove the other provider resource absent"
    )


def test_unregistered_inventory_is_not_accepted_from_successful_collector(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    world.inventory[world.target.metadata_name]["unregistered_resources"] = [
        {"name": "customer-resource"}
    ]
    with pytest.raises(contract.UninstallCaseError, match="unregistered"):
        observe.installed_inventory(world.target)
    assert all(
        "--apply" not in call[1] for call in world.calls if call[0] == "command"
    ), "plan must not rewrite installed registries to erase unknown resources"


def test_registered_cluster_scoped_uid_uses_no_namespace(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    inventory = copy.deepcopy(world.inventory[world.target.metadata_name])
    inventory["cpu"]["resources"].append(
        {"scope": "cluster", "kind": "clusterrole", "name": "fixture-role"}
    )
    result = observe.inventory_uids(world.target, inventory)
    assert result["cpu"][-1] == {
        "identity": ["cluster", "clusterrole", "", "fixture-role"],
        "uid": None,
    }, "cluster-scoped inventory must retain explicit scope and confirmed absence"
    with pytest.raises(contract.UninstallCaseError, match="duplicated"):
        inventory["cpu"]["resources"].append(inventory["cpu"]["resources"][-1])
        observe.inventory_uids(world.target, inventory)


def test_explicit_gpu_config_can_come_from_release_config_without_ambient_lookup(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    expected = contract.kubeconfig(world.target, "gpu")
    alternate = replace(
        world.target,
        release_config={**world.target.release_config, "gpu_kubeconfig": str(expected)},
        environment={},
    )
    assert contract.kubeconfig(alternate, "gpu") == expected, (
        "explicit release config path is a valid native form"
    )
    assert (
        json.loads(
            world.target.source.parent.joinpath("bootstrap-state.json").read_text()
        )["phase"]
        == "site-ready"
    ), "observation must leave original provisioning provenance untouched"
