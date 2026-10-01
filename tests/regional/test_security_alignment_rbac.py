from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from gpu_fault_release.regional_release_gpu_rollout import (
    render_workload_namespace_rbac,
)
from scripts.e2e.regional.blast_acceptance_base import CheckError
from scripts.e2e.regional.blast_acceptance_cases_2 import (
    EXECUTOR_MANIFEST,
    BlastCasesTwo,
    expected_executor_named_roles,
    expected_executor_role,
)
from scripts.e2e.regional.blast_rbac_scope import (
    BoundRule,
    bound_rules,
    unexpected_grants,
)

SA = "system:serviceaccount:gpu-system:gpu-fault-cluster-executor"


def binding_document(*, subject=None, names=None, scope="foreign", resource="secrets"):
    return {
        "items": [
            {
                "kind": "Role" if scope else "ClusterRole",
                "metadata": {
                    "name": "extra",
                    **({"namespace": scope} if scope else {}),
                },
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": [resource],
                        "verbs": ["get"],
                        **({"resourceNames": names} if names is not None else {}),
                    }
                ],
            },
            {
                "kind": "RoleBinding" if scope else "ClusterRoleBinding",
                "metadata": {
                    "name": "extra-binding",
                    **({"namespace": scope} if scope else {}),
                },
                "subjects": [
                    subject
                    or {
                        "kind": "ServiceAccount",
                        "name": "gpu-fault-cluster-executor",
                        "namespace": "gpu-system",
                    }
                ],
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role" if scope else "ClusterRole",
                    "name": "extra",
                },
            },
        ]
    }


@pytest.mark.parametrize(
    "subject",
    [
        {
            "kind": "ServiceAccount",
            "name": "gpu-fault-cluster-executor",
            "namespace": "gpu-system",
        },
        {"kind": "User", "name": SA},
        {"kind": "Group", "name": "system:authenticated"},
        {"kind": "Group", "name": "system:serviceaccounts"},
        {"kind": "Group", "name": "system:serviceaccounts:gpu-system"},
    ],
)
@pytest.mark.parametrize("scope", ["foreign", None])
def test_named_grants_through_every_applicable_binding_are_visible(subject, scope):
    document = binding_document(subject=subject, names=["credential"], scope=scope)
    errors = unexpected_grants(
        bound_rules(document, SA),
        expected_cluster=expected_executor_role(),
        expected_namespaces={},
    )
    assert errors == [
        {
            "binding": ("RoleBinding" if scope else "ClusterRoleBinding")
            + "/extra-binding",
            "namespace": scope,
            "api_group": "",
            "resource": "secrets",
            "verb": "get",
            "resource_names": ["credential"],
        }
    ]


def test_roles_with_no_applicable_binding_do_not_grant_permissions():
    document = binding_document(
        subject={"kind": "Group", "name": "system:serviceaccounts:other"}
    )
    assert bound_rules(document, SA) == []


@pytest.mark.parametrize(
    "defect",
    ["missing-reference", "pagination", "duplicate", "bad-subject", "bad-reference"],
)
def test_rbac_resolution_cannot_turn_an_incomplete_inventory_into_denial(defect):
    document = binding_document()
    if defect == "missing-reference":
        document["items"].pop(0)
    elif defect == "pagination":
        document["metadata"] = {"continue": "remaining-page"}
    elif defect == "duplicate":
        document["items"].append(copy.deepcopy(document["items"][0]))
    elif defect == "bad-subject":
        document["items"][1]["subjects"] = "malformed"
    else:
        document["items"][1]["roleRef"]["apiGroup"] = "other"
    with pytest.raises(CheckError):
        bound_rules(document, SA)


def test_renderer_workload_writes_and_plugin_exception_are_exact_not_blanket():
    documents = render_workload_namespace_rbac(
        ["training"], system_namespace="gpu-system"
    )["gpu-fault-cluster-executor"]
    expected = {
        item["metadata"]["namespace"]: BlastCasesTwo.normalized_role_rules(item)
        for item in documents
        if item["kind"] == "Role"
    }
    rules = bound_rules({"items": documents}, SA)
    assert (
        unexpected_grants(
            rules,
            expected_cluster=expected_executor_role(),
            expected_namespaces=expected,
        )
        == []
    )
    rules.append(
        BoundRule(
            "RoleBinding/extra",
            "kube-system",
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["patch"]},
        )
    )
    assert unexpected_grants(
        rules, expected_cluster=expected_executor_role(), expected_namespaces=expected
    ) == [
        {
            "binding": "RoleBinding/extra",
            "namespace": "kube-system",
            "api_group": "",
            "resource": "pods",
            "verb": "patch",
            "resource_names": [],
        }
    ]


def test_cpu_named_pod_patch_is_denied_even_without_an_unnamed_permission():
    document = binding_document(names=["specific-pod"], resource="pods")
    document["items"][0]["rules"][0]["verbs"] = ["patch"]
    errors = unexpected_grants(
        bound_rules(document, SA), expected_cluster={}, expected_namespaces={}, cpu=True
    )
    assert len(errors) == 1 and errors[0]["resource_names"] == ["specific-pod"]


def test_rolebinding_to_clusterrole_is_still_namespace_scoped():
    document = binding_document(names=["specific-pod"], resource="pods")
    role = document["items"][0]
    role["kind"] = "ClusterRole"
    role["metadata"].pop("namespace")
    role["rules"][0]["verbs"] = ["delete"]
    document["items"][1]["roleRef"]["kind"] = "ClusterRole"
    errors = unexpected_grants(
        bound_rules(document, SA),
        expected_cluster=expected_executor_role(),
        expected_namespaces={"training": {"core:pods": ["delete"]}},
    )
    assert len(errors) == 1 and errors[0]["namespace"] == "foreign"


NODE_KEY_SECRET = "gpu-fault-node-action-keys"
NAMED_EXPECTATION = {
    "gpu-system": {f"core:secrets@{NODE_KEY_SECRET}": ["get", "patch"]}
}


def node_key_document(*, names, scope="gpu-system", verbs=("get", "patch")):
    document = binding_document(names=names, scope=scope)
    document["items"][0]["rules"][0]["verbs"] = list(verbs)
    return document


def test_manifest_named_expectation_is_exactly_the_node_key_secret_get_patch():
    assert (
        expected_executor_named_roles(system_namespace="gpu-system")
        == NAMED_EXPECTATION
    )


def test_manifest_named_secret_grant_in_the_system_namespace_is_expected():
    errors = unexpected_grants(
        bound_rules(node_key_document(names=[NODE_KEY_SECRET]), SA),
        expected_cluster=expected_executor_role(),
        expected_namespaces={},
        expected_named_namespaces=expected_executor_named_roles(
            system_namespace="gpu-system"
        ),
    )
    assert errors == [], "the manifest's named node-key Secret grant is expected"


@pytest.mark.parametrize(
    ("names", "scope", "verbs", "denied"),
    [
        (None, "gpu-system", ("get", "patch"), ["get", "patch"]),
        ([], "gpu-system", ("get", "patch"), ["get", "patch"]),
        (["other-secret"], "gpu-system", ("get", "patch"), ["get", "patch"]),
        ([NODE_KEY_SECRET, "other-secret"], "gpu-system", ("get",), ["get"]),
        ([NODE_KEY_SECRET], "foreign", ("get", "patch"), ["get", "patch"]),
        ([NODE_KEY_SECRET], None, ("get", "patch"), ["get", "patch"]),
        ([NODE_KEY_SECRET], "gpu-system", ("get", "list"), ["list"]),
        ([NODE_KEY_SECRET], "gpu-system", ("patch", "delete"), ["delete"]),
    ],
    ids=[
        "unnamed",
        "empty-names",
        "other-secret",
        "extra-name",
        "other-namespace",
        "cluster-wide",
        "list",
        "delete",
    ],
)
def test_named_expectation_never_excuses_a_wider_secret_grant(
    names, scope, verbs, denied
):
    errors = unexpected_grants(
        bound_rules(node_key_document(names=names, scope=scope, verbs=verbs), SA),
        expected_cluster=expected_executor_role(),
        expected_namespaces={},
        expected_named_namespaces=NAMED_EXPECTATION,
    )
    assert [error["verb"] for error in errors] == denied
    assert all(error["resource"] == "secrets" for error in errors), errors


def test_named_expectation_cannot_satisfy_an_unnamed_rule_via_the_unnamed_map():
    document = node_key_document(names=None)
    errors = unexpected_grants(
        bound_rules(document, SA),
        expected_cluster=expected_executor_role(),
        expected_namespaces=NAMED_EXPECTATION,
        expected_named_namespaces=NAMED_EXPECTATION,
    )
    assert [error["verb"] for error in errors] == ["get", "patch"]


def test_cpu_policy_rejects_the_named_secret_patch_despite_a_named_expectation():
    errors = unexpected_grants(
        bound_rules(node_key_document(names=[NODE_KEY_SECRET]), SA),
        expected_cluster={},
        expected_namespaces={},
        expected_named_namespaces=NAMED_EXPECTATION,
        cpu=True,
    )
    assert [error["verb"] for error in errors] == ["patch"]


def manifest_variant(tmp_path: Path, mutate) -> Path:
    documents = [
        item
        for item in yaml.safe_load_all(EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
        if isinstance(item, dict)
    ]
    role = next(
        item
        for item in documents
        if item["kind"] == "Role"
        and item["metadata"]["name"] == "gpu-fault-cluster-executor-node-keys"
    )
    mutate(role)
    path = tmp_path / "executor.yaml"
    path.write_text(yaml.safe_dump_all(documents), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "defect", ["unnamed", "list", "delete", "update", "wildcard", "non-resource"]
)
def test_manifest_role_with_unnamed_or_wider_rule_fails_the_audit(tmp_path, defect):
    def mutate(role):
        rule = role["rules"][0]
        if defect == "unnamed":
            rule.pop("resourceNames")
        elif defect == "non-resource":
            rule["nonResourceURLs"] = ["/healthz"]
        elif defect == "wildcard":
            rule["verbs"] = ["*"]
        else:
            rule["verbs"] = ["get", defect]

    manifest = manifest_variant(tmp_path, mutate)
    with pytest.raises(CheckError, match="un-named or non get/patch"):
        expected_executor_named_roles(system_namespace="gpu-system", manifest=manifest)


def test_manifest_named_roles_follow_the_executor_binding_not_the_role_name(tmp_path):
    def unbind(role):
        role["metadata"]["name"] = "renamed-so-the-binding-dangles"

    manifest = manifest_variant(tmp_path, unbind)
    with pytest.raises(CheckError, match="absent or incomplete"):
        expected_executor_named_roles(system_namespace="gpu-system", manifest=manifest)


def test_manifest_without_the_executor_service_account_cannot_name_a_namespace(
    tmp_path,
):
    documents = [
        item
        for item in yaml.safe_load_all(EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
        if isinstance(item, dict) and item["kind"] != "ServiceAccount"
    ]
    path = tmp_path / "executor.yaml"
    path.write_text(yaml.safe_dump_all(documents), encoding="utf-8")
    with pytest.raises(CheckError, match="ServiceAccount namespace"):
        expected_executor_named_roles(system_namespace="gpu-system", manifest=path)
