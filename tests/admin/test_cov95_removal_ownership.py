from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import bootstrap_common
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import cluster_removal_network as network
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.installation_resources import InstallationResourceSnapshot
from tests.admin._cov95_removal_support import RemovalTransport


@pytest.fixture
def transport(tmp_path, monkeypatch):
    return RemovalTransport(tmp_path, monkeypatch)


def replace_resources(transport, resources):
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    transport.snapshot = snapshot.model_copy(
        update={"source_sha256": snapshot.digest()}
    )


def install_command(transport, monkeypatch, transform):
    original = transport.command

    def command(arguments, **options):
        return transform(list(arguments), options, original)

    monkeypatch.setattr(removal, "run_command", command)
    monkeypatch.setattr(bootstrap_common, "run_command", command)
    monkeypatch.setattr(network, "run_command", command)


@pytest.mark.parametrize(
    "field",
    ["cpu", "ca", "master", "gpu", "ambient-gpu", "ambient-multiple", "default-gpu"],
)
def test_removal_never_deletes_token_path_shared_with_preserved_inputs(
    transport, monkeypatch, field
):
    original_token = Path(transport.gpu["token_file"])
    before = original_token.read_bytes()
    document = yaml.safe_load(transport.path.read_text())
    cluster = document["spec"]["clusters"][0]
    if field == "cpu":
        cluster["tokenFile"] = document["spec"]["cpu"]["kubeconfig"]
    elif field == "ca":
        cluster["tokenFile"] = cluster["caFile"]
    elif field == "master":
        cluster["tokenFile"] = cluster["fleetMasterFile"]
    elif field == "gpu":
        document["spec"]["gpuKubeconfig"] = cluster["tokenFile"]
    elif field.startswith("ambient"):
        prefix = (
            str(transport.directory / "other.kubeconfig") + ":"
            if field == "ambient-multiple"
            else ""
        )
        monkeypatch.setenv("KUBECONFIG", prefix + cluster["tokenFile"])
    else:
        home = transport.directory / "owned-home"
        kubeconfig = home / ".kube/config"
        kubeconfig.parent.mkdir(parents=True)
        kubeconfig.write_text("owned example kubeconfig")
        cluster["tokenFile"] = str(kubeconfig)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("KUBECONFIG", raising=False)
    protected_token = Path(cluster["tokenFile"])
    protected_before = protected_token.read_bytes()
    transport.path.write_text(yaml.safe_dump(document))
    with pytest.raises(BootstrapError, match="shared with a preserved resource"):
        transport.remove()
    assert transport.events == []
    assert original_token.read_bytes() == before
    assert protected_token.read_bytes() == protected_before


@pytest.mark.parametrize("kind", ["cpu-overlap", "same-context"])
def test_removal_refuses_target_scope_overlapping_preserved_cluster(transport, kind):
    document = yaml.safe_load(transport.path.read_text())
    cluster = document["spec"]["clusters"][0]
    if kind == "cpu-overlap":
        cluster["eksClusterArn"] = document["spec"]["cpu"]["eksArn"]
    else:
        sibling = copy.deepcopy(cluster)
        sibling.update(
            clusterId="gpu-b",
            eksClusterArn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
            hyperpodClusterName="hp-gpu-b",
        )
        document["spec"]["clusters"].append(sibling)
    transport.path.write_text(yaml.safe_dump(document))
    request = replace(transport.request(), gpu_cluster_arn=None)
    with pytest.raises(BootstrapError, match="overlaps or conflicts"):
        removal.remove_cluster(request, runner=transport)
    assert transport.events == []


@pytest.mark.parametrize("kind", ["missing-role", "foreign-role", "foreign-eks"])
def test_removal_requires_registry_to_bind_preserved_clusters_and_owned_role(
    transport, kind
):
    resources = []
    for resource in transport.snapshot.resources:
        if resource.resource_type == "iam_role":
            if kind == "missing-role":
                continue
            if kind == "foreign-role":
                resource = resource.model_copy(
                    update={"resource_arn": "arn:aws:iam::123456789012:role/foreign"}
                )
        if resource.resource_type == "gpu_eks" and kind == "foreign-eks":
            resource = resource.model_copy(update={"resource_id": "foreign"})
        resources.append(resource)
    replace_resources(transport, resources)
    with pytest.raises(BootstrapError, match="identity is incomplete"):
        transport.remove()
    assert "drain-cluster" not in transport.events


@pytest.mark.parametrize("hint", [None, "foreign", "matching"])
def test_removal_legacy_hyperpod_reference_needs_exact_bootstrap_identity(
    transport, hint
):
    replace_resources(
        transport,
        [
            resource.model_copy(update={"resource_arn": None})
            if resource.resource_type == "gpu_hyperpod"
            else resource
            for resource in transport.snapshot.resources
        ],
    )
    if hint is not None:
        (transport.directory / "bootstrap-state.json").write_text(
            json.dumps(
                {
                    "site_id": "test-site",
                    "resources": {
                        "hyperpod_by_eks": {
                            transport.gpu["eks_cluster_arn"]: transport.hyperpod
                            if hint == "matching"
                            else "foreign"
                        }
                    },
                }
            )
        )
    if hint == "matching":
        assert transport.remove()["phase"] == "COMPLETED"
    else:
        with pytest.raises(BootstrapError, match="HyperPod ARN is missing or drifted"):
            transport.remove()
        assert "drain-cluster" not in transport.events


def test_removal_accepts_legacy_role_name_without_widening_cluster_scope(transport):
    replace_resources(
        transport,
        [
            resource.model_copy(update={"resource_arn": None})
            if resource.resource_type == "iam_role"
            else resource
            for resource in transport.snapshot.resources
        ],
    )
    assert transport.remove()["phase"] == "COMPLETED"
    assert transport.events.count("delete-role") == 1


def test_removal_handles_empty_gpu_inventory_without_key_or_annotation_mutation(
    transport,
):
    transport.nodes = {"items": []}
    assert transport.remove()["phase"] == "COMPLETED"
    assert "remove-keys" not in transport.events
    assert "clear-annotations" not in transport.events


def test_duplicate_node_inventory_cannot_authorize_removal(transport):
    document = json.loads(transport.command(["kubectl", "get", "nodes"]).stdout)
    document["items"].append(copy.deepcopy(document["items"][0]))
    transport.nodes = document
    with pytest.raises(BootstrapError, match="duplicate names"):
        transport.remove()
    assert "drain-cluster" not in transport.events


def test_missing_cpu_namespace_identity_blocks_removal(transport, monkeypatch):
    def response(arguments, options, command):
        if "namespace" in arguments and "--context" not in arguments:
            return transport.result(arguments, "")
        return command(arguments, **options)

    install_command(transport, monkeypatch, response)
    with pytest.raises(
        BootstrapError, match="CPU solution namespace identity is missing"
    ):
        transport.remove()
    assert "drain-cluster" not in transport.events


@pytest.mark.parametrize(
    "kind", ["missing-token", "missing-namespace", "other-member", "target-lifecycle"]
)
def test_removal_requires_current_precleanup_ownership(transport, kind):
    if kind == "missing-token":
        Path(transport.gpu["token_file"]).unlink()
    elif kind == "missing-namespace":
        transport.namespace = False
    elif kind == "other-member":
        transport.states["foreign"] = "ACTIVE"
    else:
        transport.states["gpu-a"] = "PENDING"
    with pytest.raises(
        BootstrapError,
        match="missing before site commit|cleanup barrier|membership lifecycle|lifecycle conflicts",
    ):
        transport.remove()
    assert "drain-cluster" not in transport.events


@pytest.mark.parametrize("field", ["registry_snapshot", "registry_digest"])
def test_removal_resume_rejects_changed_registry_provenance(transport, field):
    transport.failure = "cleanup"
    with pytest.raises(BootstrapError):
        transport.remove()
    state = transport.state()
    state["evidence"]["DISCOVERED"][field] = "different"
    path = transport.directory / "remove-cluster/gpu-a/state.json"
    path.write_text(json.dumps(state))
    prior = list(transport.events)
    transport.failure = None
    with pytest.raises(BootstrapError, match="snapshot identity|provenance changed"):
        transport.remove()
    assert transport.events == prior


def test_removal_resume_rejects_replaced_cluster_token_before_more_cleanup(transport):
    transport.failure = "cleanup"
    with pytest.raises(BootstrapError):
        transport.remove()
    Path(transport.gpu["token_file"]).write_text("example-replaced-token")
    transport.failure = None
    prior = list(transport.events)
    with pytest.raises(BootstrapError, match="credential incarnation changed"):
        transport.remove()
    assert transport.events == prior


@pytest.mark.parametrize("change", ["uid", "key", "failed-patch", "unconfirmed-patch"])
def test_node_key_race_never_records_success_without_bound_deletion(
    transport, monkeypatch, change
):
    def response(arguments, options, command):
        is_key = "gpu-fault-node-action-keys" in arguments
        if is_key and "patch" in arguments and change == "failed-patch":
            return transport.result(arguments, code=1, error="example conflict")
        result = command(arguments, **options)
        if "remove-cluster" in transport.events and is_key:
            if "get" in arguments and change in {"uid", "key"}:
                document = json.loads(result.stdout)
                if change == "uid":
                    document["metadata"]["uid"] = "different-secret"
                else:
                    document["data"]["node-a"] = base64.b64encode(b"d" * 64).decode()
                return transport.result(arguments, document)
            if "patch" in arguments and change == "unconfirmed-patch":
                transport.keys["node-a"] = base64.b64encode(b"a" * 64).decode()
        return result

    install_command(transport, monkeypatch, response)
    with pytest.raises(
        BootstrapError,
        match="incarnation changed|ownership changed|failed to remove|not confirmed",
    ):
        transport.remove()
    assert "CONTROL_REGISTRY_REMOVED" not in transport.state()["completed_steps"]
    assert transport.state()["phase"] != "COMPLETED"


@pytest.mark.parametrize(
    "payload", ["not-base64", b"invalid-json", {}, [{}], [{"cluster_id": "gpu-a"}]]
)
def test_final_registry_data_must_prove_target_absence(transport, payload):
    encoded = (
        payload
        if isinstance(payload, str)
        else base64.b64encode(
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        ).decode()
    )
    transport.registry_secret_override = {
        "kind": "Secret",
        "metadata": {
            "name": "gpu-fault-regional-clusters",
            "namespace": "gpu-fault-system",
            "uid": "registry-uid",
        },
        "data": {"clusters.json": encoded},
    }
    with pytest.raises(BootstrapError, match="final verification failed"):
        transport.remove()
    assert transport.state()["phase"] != "COMPLETED"


@pytest.mark.parametrize("provider", ["eks", "sagemaker"])
def test_preserved_gpu_identity_is_rechecked_after_cleanup(
    transport, monkeypatch, provider
):
    def response(arguments, options, command):
        result = command(arguments, **options)
        if (
            arguments[:3] == ["aws", provider, "describe-cluster"]
            and "remove-cluster" in transport.events
        ):
            document = json.loads(result.stdout)
            if provider == "eks":
                document["cluster"]["createdAt"] = "different-incarnation"
            else:
                document["NodeRecovery"] = "Automatic"
            return transport.result(arguments, document)
        return result

    install_command(transport, monkeypatch, response)
    with pytest.raises(BootstrapError, match="final verification failed"):
        transport.remove()
    assert transport.state()["phase"] != "COMPLETED"
    assert transport.events.count("delete-role") == 1
