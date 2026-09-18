from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import bootstrap_site as sites
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
)
from tests.admin._cov95_join_support import target
from tests.admin.test_cov95_bootstrap_site import document, write_verified_candidate


@pytest.mark.parametrize("existing", [{}, {"spec": []}, {"spec": {"clusters": {}}}])
def test_bootstrap_gpu_scope_requires_an_existing_membership_list(existing):
    with pytest.raises(BootstrapError, match="no valid cluster identity"):
        sites.bootstrap_gpu_scope(existing, [target()])


def test_bootstrap_recovery_without_verified_history_preserves_existing_contract(
    tmp_path,
):
    existing = document()
    assert sites.load_existing_site(tmp_path) is None
    assert sites.recover_verified_site_contract(tmp_path, None) is None
    assert sites.recover_verified_site_contract(tmp_path, existing) is existing
    (tmp_path / "site.yaml").write_text(yaml.safe_dump(existing))
    assert sites.load_existing_site(tmp_path) == existing


def test_initial_pending_target_replay_preserves_the_same_identity(tmp_path):
    state = BootstrapState(tmp_path / "state.json", site_id="example-site")
    cpu = replace(target("a"), role="cpu")
    assert sites.bind_initial_deploy_target(state, None, cpu, [target("b")]) == [
        target("b")
    ]
    prior = copy.deepcopy(state.result(sites.INITIAL_DEPLOY_TARGET))
    sites.bind_initial_deploy_target(state, None, cpu, [target("b")])
    assert state.result(sites.INITIAL_DEPLOY_TARGET) == prior


def test_contract_merge_keeps_unmatched_rows_for_the_subsequent_site_validator():
    generated, existing = document(), document()
    generated["spec"]["clusters"].extend([None, {"clusterId": "new", "context": "new"}])
    merged = sites.preserve_existing_site_contract(generated, existing)
    assert merged["spec"]["clusters"][1:] == [
        None,
        {"clusterId": "new", "context": "new"},
    ]


def test_verified_contract_without_release_fields_does_not_erase_current_values(
    tmp_path,
):
    existing, verified = document(), document()
    verified["spec"]["release"] = None
    verified["spec"].pop("runtimeProfile")
    write_verified_candidate(tmp_path, verified)
    recovered = sites.recover_verified_site_contract(tmp_path, existing)
    assert recovered["spec"]["release"] == existing["spec"]["release"]
    assert recovered["spec"]["runtimeProfile"] == existing["spec"]["runtimeProfile"]


@pytest.mark.parametrize("field", ["cpu", "clusters"])
def test_existing_cluster_validation_requires_structured_cpu_and_gpu_records(field):
    existing = document()
    existing["spec"][field] = None
    with pytest.raises(BootstrapError, match="no valid cluster identity"):
        sites.validate_existing_cluster_identity(
            existing, cpu=replace(target("a"), role="cpu"), gpu_clusters=[target("b")]
        )


@pytest.mark.parametrize("existing", [False, True])
def test_public_scope_discovery_preserves_existing_contexts_and_input_order(
    tmp_path, existing
):
    cpu = replace(target("a"), role="cpu")
    members = [target("b"), target("c")]
    if existing:
        (tmp_path / "site.yaml").write_text(yaml.safe_dump(document()))
    identities = {identity.input_arn: identity for identity in (cpu, *members)}
    calls = []

    def discover(_runner, *, cluster_arn, role, context):
        calls.append(cluster_arn)
        assert identities[cluster_arn].role == role
        return replace(identities[cluster_arn], context=context)

    request = BootstrapRequest(
        cpu.input_arn, tuple(member.input_arn for member in members), tmp_path, tmp_path
    )
    restored, found_cpu, found_gpu = sites.discover_bootstrap_scope(
        request=request,
        runner=SimpleNamespace(),
        discover=discover,
        alias=sites.cluster_alias,
    )
    assert restored == (document() if existing else None)
    assert found_cpu.context == sites.cluster_alias(cpu.eks_arn, "cpu", 0)
    assert [member.eks_arn for member in found_gpu] == [
        member.eks_arn for member in members
    ]
    assert found_gpu[0].context == (
        "gpu-b" if existing else sites.cluster_alias(members[0].eks_arn, "gpu", 1)
    )
    assert len(calls) == 3
