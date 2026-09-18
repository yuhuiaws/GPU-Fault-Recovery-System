from __future__ import annotations

import copy

import pytest

from gpu_fault_release.regional_release_gpu_rollout import (
    render_workload_namespace_rbac,
)
from scripts.e2e.regional.blast_acceptance_base import CheckError
from scripts.e2e.regional.blast_acceptance_cases_2 import (
    BlastCasesTwo,
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
