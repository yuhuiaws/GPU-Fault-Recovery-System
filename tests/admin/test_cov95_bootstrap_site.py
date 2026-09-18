from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import bootstrap_site as sites
from gpu_fault.admin.bootstrap_common import BootstrapError, BootstrapState
from tests.admin._cov95_join_support import target


def document():
    cpu = replace(target("a"), role="cpu")
    gpu = target("b")
    return {
        "kind": "RegionalSite",
        "metadata": {"name": "example-site"},
        "spec": {
            "awsRegion": cpu.region,
            "cpu": {"eksArn": cpu.eks_arn, "hyperpodClusterName": cpu.hyperpod_name},
            "clusters": [
                {
                    "clusterId": "gpu-b",
                    "eksClusterArn": gpu.eks_arn,
                    "hyperpodClusterName": gpu.hyperpod_name,
                    "context": gpu.context,
                }
            ],
            "release": {"manifest": "example-release"},
            "runtimeProfile": {"version": "example-profile"},
        },
    }


@pytest.mark.parametrize("value", ["false", "true", 0, 1, [], {}])
def test_invalid_existing_rollback_policy_cannot_be_normalized_into_authorization(
    value,
):
    """COV95-ADMIN-002: invalid operator policy must not become generated True."""
    generated, existing = document(), document()
    generated["spec"]["autoRollback"] = True
    existing["spec"]["autoRollback"] = value
    with pytest.raises(BootstrapError, match="autoRollback"):
        sites.preserve_existing_site_contract(generated, existing)


@pytest.mark.parametrize("value", [False, True])
def test_valid_rollback_policy_and_cluster_contract_are_preserved_by_value(value):
    generated, existing = document(), document()
    existing["spec"].update(
        autoRollback=value,
        failureDomainLabels=["example/fabric"],
        retention={"days": 7},
    )
    existing["spec"]["clusters"][0].update(
        context="saved-context",
        allowedNamespaces=["training"],
        agentEndpointAllowedCidrs=["10.0.0.0/24"],
        tokenFile="example-token-reference",
        caFile="example-ca-reference",
        fleetMasterFile="example-master-reference",
        adotIrsaRoleArn="example-role-reference",
    )
    result = sites.preserve_existing_site_contract(generated, existing)
    assert result["spec"]["autoRollback"] is value
    assert result["spec"]["clusters"] == existing["spec"]["clusters"]
    result["spec"]["clusters"][0]["allowedNamespaces"].append("other")
    assert existing["spec"]["clusters"][0]["allowedNamespaces"] == ["training"]
    assert result["spec"]["failureDomainLabels"] == ["example/fabric"]


@pytest.mark.parametrize("existing", [None, {}, {"spec": []}])
def test_no_valid_existing_contract_leaves_generation_unchanged(existing):
    generated = document()
    assert sites.preserve_existing_site_contract(generated, existing) is generated


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"spec": []},
        {"spec": {"clusters": {}}},
        {"spec": {"clusters": [None]}},
    ],
)
def test_existing_context_does_not_infer_missing_identity(value):
    assert sites.existing_gpu_context(value, target()) is None


def test_existing_context_requires_a_nonempty_matching_value():
    value = document()
    assert sites.existing_gpu_context(value, target("b")) == "gpu-b"
    assert sites.existing_gpu_context(value, target("c")) is None
    value["spec"]["clusters"][0]["context"] = ""
    assert sites.existing_gpu_context(value, target("b")) is None


@pytest.mark.parametrize("value", ["invalid: [yaml", "[]", "kind: Other"])
def test_existing_site_file_requires_regional_site_shape(tmp_path, value):
    (tmp_path / "site.yaml").write_text(value)
    with pytest.raises(BootstrapError, match="cannot load|not a RegionalSite"):
        sites.load_existing_site(tmp_path)


@pytest.mark.parametrize(
    "state",
    [
        "invalid",
        [],
        {},
        {"phase": "FAILED"},
        {"phase": "COMPLETED", "verification": []},
        {"phase": "COMPLETED", "verification": {"status": "FAILED"}},
    ],
)
def test_recovery_never_selects_unverified_release_state(tmp_path, state):
    path = tmp_path / "release-deploy/release-a/state.json"
    path.parent.mkdir(parents=True)
    path.write_text(state if isinstance(state, str) else json.dumps(state))
    if state == "invalid":
        with pytest.raises(BootstrapError, match="cannot load release state"):
            sites.load_latest_verified_site_contract(tmp_path)
    else:
        assert sites.load_latest_verified_site_contract(tmp_path) is None


def write_verified_candidate(tmp_path, value):
    directory = tmp_path / "release-deploy/release-a"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(
        json.dumps({"phase": "COMPLETED", "verification": {"status": "PASSED"}})
    )
    path = directory / "site.candidate.yaml"
    if value is not None:
        path.write_text(value if isinstance(value, str) else yaml.safe_dump(value))
    return path


@pytest.mark.parametrize("candidate", [None, "invalid: [yaml", "[]", "kind: Other"])
def test_verified_release_still_requires_a_valid_candidate(tmp_path, candidate):
    write_verified_candidate(tmp_path, candidate)
    with pytest.raises(
        BootstrapError, match="no candidate|cannot load verified|not a RegionalSite"
    ):
        sites.load_latest_verified_site_contract(tmp_path)


@pytest.mark.parametrize("field", ["spec", "metadata", "awsRegion", "cpu"])
def test_contract_recovery_requires_same_site_identity(tmp_path, field):
    existing, verified = document(), document()
    if field == "spec":
        verified["spec"] = []
    elif field == "metadata":
        verified["metadata"]["name"] = "foreign"
    elif field == "awsRegion":
        verified["spec"]["awsRegion"] = "us-west-2"
    else:
        verified["spec"]["cpu"]["eksArn"] = target("c").eks_arn
    write_verified_candidate(tmp_path, verified)
    with pytest.raises(BootstrapError, match="valid spec|differs"):
        sites.recover_verified_site_contract(tmp_path, existing)


def test_recovery_copies_verified_contract_without_overwriting_membership(tmp_path):
    existing, verified = document(), document()
    existing["spec"]["clusters"].extend(
        [None, {"clusterId": "gpu-c", "eksClusterArn": target("c").eks_arn}]
    )
    verified["spec"]["release"]["manifest"] = "verified-release"
    verified["spec"]["runtimeProfile"]["version"] = "verified-profile"
    verified["spec"]["clusters"][0]["context"] = "verified-context"
    write_verified_candidate(tmp_path, verified)
    recovered = sites.recover_verified_site_contract(tmp_path, existing)
    assert recovered["spec"]["release"] == verified["spec"]["release"]
    assert recovered["spec"]["clusters"][0]["context"] == "verified-context"
    assert recovered["spec"]["clusters"][1:] == existing["spec"]["clusters"][1:]
    assert existing["spec"]["clusters"][0]["context"] == "gpu-b"
    recovered["spec"]["release"]["manifest"] = "changed"
    assert verified["spec"]["release"]["manifest"] == "verified-release"


@pytest.mark.parametrize(
    "site",
    [
        {},
        {"spec": []},
        {"spec": {"cpu": {}, "clusters": []}},
        {"spec": {"cpu": {}, "clusters": [None]}},
    ],
)
def test_existing_cluster_validation_refuses_missing_or_changed_cpu(site):
    if not site:
        assert (
            sites.validate_existing_cluster_identity(
                site, cpu=replace(target("a"), role="cpu"), gpu_clusters=[target("b")]
            )
            == []
        )
    else:
        with pytest.raises(BootstrapError, match="cluster identity"):
            sites.validate_existing_cluster_identity(
                site, cpu=replace(target("a"), role="cpu"), gpu_clusters=[target("b")]
            )


def test_existing_member_set_cannot_be_silently_reduced():
    site = document()
    with pytest.raises(BootstrapError, match="command omits"):
        sites.validate_existing_cluster_identity(
            site, cpu=replace(target("a"), role="cpu"), gpu_clusters=[]
        )
    assert sites.validate_existing_cluster_identity(
        site,
        cpu=replace(target("a"), role="cpu"),
        gpu_clusters=[target("b"), target("c")],
    ) == [target("c")]


@pytest.mark.parametrize(
    "stored",
    ["invalid", {"schema_version": 1, "status": "PENDING"}, {"status": "COMPLETE"}],
)
def test_initial_target_checkpoint_cannot_rebind_identity(tmp_path, stored):
    state = BootstrapState(tmp_path / "bootstrap.json", site_id="example-site")
    state.record(sites.INITIAL_DEPLOY_TARGET, stored)
    existing = document()
    if stored == {"status": "COMPLETE"}:
        existing["spec"]["cpu"]["eksArn"] = target("c").eks_arn
    with pytest.raises(BootstrapError, match="invalid|differs"):
        sites.bind_initial_deploy_target(
            state, existing, replace(target("a"), role="cpu"), [target("b")]
        )


def test_initial_target_rejects_duplicates_before_checkpoint_write(tmp_path):
    state = BootstrapState(tmp_path / "bootstrap.json", site_id="example-site")
    with pytest.raises(BootstrapError, match="duplicate GPU"):
        sites.bind_initial_deploy_target(
            state, None, replace(target("a"), role="cpu"), [target("b"), target("b")]
        )
    assert sites.INITIAL_DEPLOY_TARGET not in state.value["resources"]


def test_finalizer_records_site_before_success_and_leaves_extra_gpus_pending(tmp_path):
    state = BootstrapState(tmp_path / "bootstrap.json", site_id="example-site")
    path = tmp_path / "site.yaml"
    writes = []

    def write(destination, value):
        writes.append((destination, copy.deepcopy(value)))

    result = sites.finalize_bootstrap_site(
        path, document(), None, [target("b"), target("c")], state, write
    )
    assert writes == [(path, document())]
    assert result.pending_gpu_cluster_arns == (target("c").eks_arn,)
    assert state.value["phase"] == "site-ready"
    assert state.result("site_file") == str(path)


def test_subnet_discovery_empty_scope_is_noop_and_cidrs_are_sorted():
    calls = []

    def aws(*args):
        calls.append(args)
        return {
            "Subnets": [{"CidrBlock": "10.1.0.0/24"}, {}, {"CidrBlock": "10.0.0.0/24"}]
        }

    runner = SimpleNamespace(aws_json=aws)
    assert sites.discover_subnet_cidrs(runner, region="us-east-1", subnet_ids=()) == ()
    assert calls == []
    assert sites.discover_subnet_cidrs(
        runner, region="us-east-1", subnet_ids=("subnet-a", "subnet-b")
    ) == ("10.0.0.0/24", "10.1.0.0/24")
    assert len(calls) == 1
