"""Refusal and edge paths of the workload-namespace RBAC pruner.

The happy path and the main ownership drifts live in
``test_admin_cluster_removal_rbac.py``. These tests pin the remaining
fail-closed branches: malformed inventories, incomplete approved
configuration, argv that selects another context, tampered recorded proofs,
anchors that disappear between preparation and deletion, and the
empty-resources proof a labelled-but-foreign inventory produces.
"""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_rbac import (
    delete_recorded_workload_namespace_rbac,
    inspect_workload_namespace_rbac,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    EXECUTOR_SERVICE_ACCOUNT,
    WORKLOAD_NAMESPACE_RBAC_LABEL,
)
from tests.admin.test_admin_cluster_removal_rbac import RbacApi, with_gpu_kubeconfig
from tests.admin.test_admin_site import site_file

KUBECTL = ["kubectl", "--context", "gpu"]
RBAC_API = "rbac.authorization.k8s.io/v1"


def role_item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "apiVersion": RBAC_API,
        "kind": "Role",
        "metadata": {
            "name": "operator-custom",
            "namespace": "training",
            "uid": "foreign-role",
            "resourceVersion": "1",
            "labels": {WORKLOAD_NAMESPACE_RBAC_LABEL: "true"},
        },
        "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
    }
    for key, value in overrides.items():
        if key in item["metadata"]:
            item["metadata"][key] = value
        else:
            item[key] = value
    return item


def listing(items: object) -> Any:
    def run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"kind": "List", "items": items}), ""
        )

    return run


@pytest.fixture
def scenario(tmp_path):
    site = with_gpu_kubeconfig(site_file(tmp_path))
    target = {
        **site.release_config["clusters"][0],
        "expected_namespace_uid": "namespace-original",
        "expected_hyperpod_arn": "arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-original",
    }
    return site, target, RbacApi(site, target)


def inspect(scenario, **overrides: Any) -> dict[str, Any] | None:
    site, target, api = scenario
    return inspect_workload_namespace_rbac(
        overrides.get("config", site.release_config),
        overrides.get("target", target),
        run=overrides.get("run", api),
        kubectl=overrides.get("kubectl", api.kubectl),
    )


def delete(scenario, proof: dict[str, Any], **overrides: Any) -> list[str]:
    site, target, api = scenario
    return delete_recorded_workload_namespace_rbac(
        overrides.get("config", site.release_config),
        overrides.get("target", target),
        proof,
        run=overrides.get("run", api),
        kubectl=overrides.get("kubectl", api.kubectl),
        checkpoint=lambda: api.checkpoints.append(deepcopy(proof)),
    )


def first_key(api: RbacApi, kind: str) -> tuple[str, str, str]:
    return next(key for key in api.objects if key[0] == kind)


def test_listing_item_without_uid_is_an_incomplete_identity():
    with pytest.raises(BootstrapError, match="identity is incomplete"):
        inspect_workload_namespace_rbac(
            {}, {}, run=listing([role_item(uid="")]), kubectl=KUBECTL
        )


@pytest.mark.parametrize(
    "items",
    [
        {"not": "a list"},
        [role_item(kind="ClusterRole")],
        [role_item(labels="not-a-mapping")],
        [role_item(), role_item()],
    ],
    ids=["items-not-list", "cluster-role", "labels-not-mapping", "duplicate"],
)
def test_listing_shape_violations_are_malformed_inventory(items):
    with pytest.raises(BootstrapError, match="inventory is malformed"):
        inspect_workload_namespace_rbac({}, {}, run=listing(items), kubectl=KUBECTL)


def test_labelled_inventory_requires_the_system_namespace():
    with pytest.raises(BootstrapError, match="requires the system namespace"):
        inspect_workload_namespace_rbac(
            {},
            {"allowed_namespaces": ["training"]},
            run=listing([role_item()]),
            kubectl=KUBECTL,
        )


def test_labelled_inventory_requires_approved_cluster_members(scenario):
    _, target, api = scenario
    with pytest.raises(BootstrapError, match="complete approved cluster"):
        inspect(scenario, config={"namespace": api.namespace})
    assert api.requests == []


def test_role_with_malformed_rules_is_refused(scenario):
    _, _, api = scenario
    api.objects[first_key(api, "Role")]["rules"] = "not-a-list"
    with pytest.raises(BootstrapError, match="Role rules are malformed"):
        inspect(scenario)
    assert api.requests == []


def test_binding_with_malformed_subjects_is_refused(scenario):
    _, _, api = scenario
    api.objects[first_key(api, "RoleBinding")]["subjects"] = "not-a-list"
    with pytest.raises(BootstrapError, match="subjects are malformed"):
        inspect(scenario)
    assert api.requests == []


def test_empty_service_account_api_group_matches_the_renderer(scenario):
    _, _, api = scenario
    binding = first_key(api, "RoleBinding")
    subject = api.objects[binding]["subjects"][0]
    assert subject["kind"] == "ServiceAccount"
    subject["apiGroup"] = ""
    proof = inspect(scenario)
    assert proof is not None
    removed = delete(scenario, proof)
    assert len(removed) == len(api.expected)
    assert binding in api.deleted


def test_argv_without_context_is_accepted(scenario):
    _, target, api = scenario
    kubectl = api.kubectl[:3]
    assert "--context" not in kubectl

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert arguments[:3] == kubectl
        return api([*kubectl, "--context", target["context"], *arguments[3:]], **kwargs)

    proof = inspect(scenario, run=run, kubectl=kubectl)
    assert proof is not None
    removed = delete(scenario, proof, run=run, kubectl=kubectl)
    assert len(removed) == len(api.expected)
    assert len(api.deleted) == len(api.expected)


def test_argv_selecting_another_context_is_refused_before_any_call(scenario):
    _, _, api = scenario
    calls: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    with pytest.raises(BootstrapError, match="different context"):
        delete(
            scenario,
            {
                "binding": {},
                "resources": {},
                "anchors": {},
                "documents": [],
                "removed": [],
            },
            run=run,
            kubectl=[*api.kubectl[:3], "--context", "other"],
        )
    assert calls == []


def test_target_allowlist_must_be_a_list(scenario):
    _, target, api = scenario
    with pytest.raises(BootstrapError, match="lacks bound target identity"):
        delete(
            scenario,
            {
                "binding": {},
                "resources": {},
                "anchors": {},
                "documents": [],
                "removed": [],
            },
            target={**target, "allowed_namespaces": "training"},
        )
    assert api.requests == []


def test_target_must_match_the_managed_member(scenario):
    _, target, api = scenario
    with pytest.raises(BootstrapError, match="differs from the managed member"):
        delete(
            scenario,
            {
                "binding": {},
                "resources": {},
                "anchors": {},
                "documents": [],
                "removed": [],
            },
            target={
                **target,
                "eks_cluster_arn": "arn:aws:eks:us-east-1:123456789012:cluster/other",
            },
        )
    assert api.requests == []


@pytest.mark.parametrize(
    "code,stdout,message",
    [(1, "", "cannot read workload RBAC ownership"), (0, "not-json", "invalid JSON")],
)
def test_anchor_reads_fail_closed(scenario, code, stdout, message):
    _, _, api = scenario

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "get" in arguments and arguments[arguments.index("get") + 1] == (
            "serviceaccount"
        ):
            return subprocess.CompletedProcess(arguments, code, stdout, "refused")
        return api(arguments, **kwargs)

    with pytest.raises(BootstrapError, match=message):
        inspect(scenario, run=run)
    assert api.requests == []


def test_binding_to_a_cluster_role_is_not_a_shared_role_binding(scenario):
    _, _, api = scenario
    binding = first_key(api, "RoleBinding")
    foreign = deepcopy(api.objects[binding])
    foreign["metadata"].update(name="operator-cluster", uid="foreign-cluster-binding")
    foreign["metadata"].pop("labels")
    foreign["roleRef"] = {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "ClusterRole",
        "name": "view",
    }
    foreign["subjects"] = [{"kind": "User", "name": "operator"}]
    key = "RoleBinding", binding[1], "operator-cluster"
    api.objects[key] = foreign
    proof = inspect(scenario)
    assert proof is not None
    removed = delete(scenario, proof)
    assert len(removed) == len(api.expected)
    assert api.objects[key] == foreign


@pytest.mark.parametrize("state", ["absent", "terminating"])
def test_workload_namespace_must_be_present_and_live(scenario, state):
    _, _, api = scenario
    key = "Namespace", "", "training"
    if state == "absent":
        api.objects.pop(key)
    else:
        api.objects[key]["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    with pytest.raises(BootstrapError, match="absent or terminating"):
        inspect(scenario)
    assert api.requests == []


def test_system_namespace_recreated_after_inspection_blocks_deletion(scenario):
    _, _, api = scenario
    proof = inspect(scenario)
    assert proof is not None
    api.objects[("Namespace", "", api.namespace)]["metadata"]["uid"] = "replacement"
    with pytest.raises(BootstrapError, match="system namespace was recreated"):
        delete(scenario, proof)
    assert api.requests == []
    assert api.deleted == []


def test_missing_service_account_blocks_inspection(scenario):
    _, _, api = scenario
    api.objects.pop(("ServiceAccount", api.namespace, EXECUTOR_SERVICE_ACCOUNT))
    with pytest.raises(BootstrapError, match="ServiceAccount ownership is unavailable"):
        inspect(scenario)
    assert api.requests == []


def test_unlabelled_counterpart_makes_the_pair_incomplete(scenario):
    _, _, api = scenario
    api.objects[first_key(api, "RoleBinding")]["metadata"].pop("labels")
    with pytest.raises(BootstrapError, match="pair is incomplete or foreign"):
        inspect(scenario)
    assert api.requests == []


def tamper(proof: dict[str, Any], how: str) -> None:
    if how == "missing-section":
        proof.pop("removed")
    elif how == "namespace-uid-type":
        proof["binding"]["system_namespace_uid"] = 123
    elif how == "binding-differs":
        proof["binding"]["cluster_id"] = "gpu-other"
    elif how == "duplicate-document":
        proof["documents"].append(deepcopy(proof["documents"][0]))
    elif how == "document-uid":
        proof["documents"][0]["metadata"]["uid"] = "other-uid"
    elif how == "document-missing":
        proof["documents"].pop()
    elif how == "anchors-without-resources":
        proof["resources"].clear()
        proof["documents"].clear()


@pytest.mark.parametrize(
    "how",
    [
        "missing-section",
        "namespace-uid-type",
        "binding-differs",
        "duplicate-document",
        "document-uid",
        "document-missing",
        "anchors-without-resources",
    ],
)
def test_tampered_recorded_proof_is_invalid(scenario, how):
    _, _, api = scenario
    proof = inspect(scenario)
    assert proof is not None
    tamper(proof, how)
    with pytest.raises(BootstrapError, match="proof is invalid or rebound"):
        delete(scenario, proof)
    assert api.requests == []


def test_recorded_proof_with_half_a_pair_is_incomplete(scenario):
    _, _, api = scenario
    proof = inspect(scenario)
    assert proof is not None
    binding = next(key for key in proof["resources"] if "/RoleBinding/" in key)
    proof["resources"].pop(binding)
    proof["documents"] = [
        item
        for item in proof["documents"]
        if f"{item['metadata']['namespace']}/{item['kind']}/{item['metadata']['name']}"
        != binding
    ]
    with pytest.raises(BootstrapError, match="incomplete pair"):
        delete(scenario, proof)
    assert api.requests == []


def test_key_already_recorded_as_removed_must_not_still_exist(scenario):
    _, _, api = scenario
    proof = inspect(scenario)
    assert proof is not None
    first = min(key for key in proof["resources"] if "/RoleBinding/" in key)
    proof["removed"].append(first)
    with pytest.raises(BootstrapError, match="UID changed"):
        delete(scenario, proof)
    assert api.requests == []


def test_foreign_labelled_role_alone_yields_an_empty_proof(scenario):
    _, _, api = scenario
    for key, item in api.objects.items():
        if key[0] in {"Role", "RoleBinding"}:
            item["metadata"].pop("labels", None)
    manual = api.add_manual_role()
    preserved = deepcopy(api.objects[manual])
    proof = inspect(scenario)
    assert proof is not None
    assert proof["resources"] == {}
    assert proof["anchors"] == {}
    assert proof["documents"] == []
    assert delete(scenario, proof) == []
    assert api.requests == []
    assert api.objects[manual] == preserved
    tampered = deepcopy(proof)
    tampered["anchors"] = {"namespace_uids": {}}
    with pytest.raises(BootstrapError, match="proof is invalid or rebound"):
        delete(scenario, tampered)
    for key, item in api.objects.items():
        if key[0] in {"Role", "RoleBinding"} and key != manual:
            item["metadata"]["labels"] = {WORKLOAD_NAMESPACE_RBAC_LABEL: "true"}
    with pytest.raises(BootstrapError, match="remains after cleanup"):
        delete(scenario, proof)
    assert api.requests == []


def test_system_namespace_recreated_after_last_deletion_is_reported(scenario):
    _, _, api = scenario

    def recreate_after_last_delete(key):
        if key[0] == "Namespace" and len(api.deleted) == len(api.expected):
            api.objects[("Namespace", "", api.namespace)]["metadata"]["uid"] = "new"

    api.before_get = recreate_after_last_delete
    proof = inspect(scenario)
    assert proof is not None
    with pytest.raises(BootstrapError, match="system namespace was recreated"):
        delete(scenario, proof)
    assert len(api.deleted) == len(api.expected)
    assert set(proof["removed"]) == set(proof["resources"])
