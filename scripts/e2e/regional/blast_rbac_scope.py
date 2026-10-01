"""Resolve the audited ServiceAccount's complete Kubernetes RBAC grant union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scripts.e2e.regional.blast_acceptance_base import CheckError

RBAC_API = "rbac.authorization.k8s.io"
RBAC_INVENTORY = "roles,rolebindings,clusterroles,clusterrolebindings"
WRITE_VERBS = frozenset(
    {
        "create",
        "patch",
        "update",
        "delete",
        "deletecollection",
        "*",
        "bind",
        "escalate",
        "impersonate",
        "approve",
        "sign",
    }
)
SENSITIVE_RESOURCES = frozenset(
    {
        "secrets",
        "configmaps",
        "roles",
        "rolebindings",
        "clusterroles",
        "clusterrolebindings",
        "serviceaccounts",
        "certificatesigningrequests",
    }
)
WORKLOAD_RESOURCES = frozenset({"nodes", "pods", "jobs", "pytorchjobs", "jobsets"})
SELF_REVIEW_RESOURCES = {
    "authorization.k8s.io": {
        "selfsubjectaccessreviews",
        "selfsubjectrulesreviews",
    },
    "authentication.k8s.io": {"selfsubjectreviews"},
}


@dataclass(frozen=True)
class BoundRule:
    binding: str
    namespace: str | None
    rule: dict[str, Any]


def _strings(value: Any, *, nonempty: bool = True) -> list[str]:
    if (
        not isinstance(value, list)
        or nonempty
        and not value
        or any(not isinstance(item, str) or not item and nonempty for item in value)
    ):
        raise CheckError("RBAC rule or binding inventory is malformed")
    return value


def bound_rules(document: dict[str, Any], service_account: str) -> list[BoundRule]:
    """Include direct SA, equivalent User, and authenticated/SA Group subjects.

    The effective ClusterRole's rules, including API-server aggregation, are
    read from the same complete inventory. A missing reference cannot be treated
    as denial, even when `auth can-i` on an unnamed resource said no.
    """

    parts = service_account.split(":")
    if len(parts) != 4 or parts[:2] != ["system", "serviceaccount"]:
        raise CheckError("RBAC audit requires an explicit ServiceAccount identity")
    _, _, namespace, name = parts
    items = document.get("items")
    if not isinstance(items, list) or (document.get("metadata") or {}).get("continue"):
        raise CheckError("RBAC binding inventory is incomplete")
    roles: dict[tuple[str, str, str], dict[str, Any]] = {}
    bindings = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise CheckError("RBAC inventory contains an invalid object")
        kind, metadata = item.get("kind"), item.get("metadata") or {}
        if kind not in {"Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding"}:
            raise CheckError("RBAC inventory contains an unexpected resource kind")
        key = (
            kind,
            str(metadata.get("namespace") or ""),
            str(metadata.get("name") or ""),
        )
        if (
            not key[2]
            or (kind in {"Role", "RoleBinding"}) != bool(key[1])
            or metadata.get("deletionTimestamp")
            or key in seen
        ):
            raise CheckError("RBAC resource identity is incomplete or duplicated")
        seen.add(key)
        if kind.endswith("Binding"):
            bindings.append(item)
        else:
            roles[key] = item
    groups = {
        "system:authenticated",
        "system:serviceaccounts",
        "system:serviceaccounts:" + namespace,
    }
    result = []
    for binding in bindings:
        subjects = binding.get("subjects")
        if subjects is None:
            subjects = []
        if not isinstance(subjects, list):
            raise CheckError("RBAC binding subjects are missing")
        matched = False
        scope = binding["metadata"].get("namespace")
        for subject in subjects:
            if not isinstance(subject, dict):
                raise CheckError("RBAC binding subject is invalid")
            matched |= (
                (
                    subject.get("kind") == "ServiceAccount"
                    and subject.get("name") == name
                    and (subject.get("namespace") or scope) == namespace
                )
                or (
                    subject.get("kind") == "User"
                    and subject.get("name") == service_account
                )
                or (subject.get("kind") == "Group" and subject.get("name") in groups)
            )
        if not matched:
            continue
        reference = binding.get("roleRef") or {}
        kind = reference.get("kind")
        if (
            reference.get("apiGroup") != RBAC_API
            or kind not in {"Role", "ClusterRole"}
            or kind == "Role"
            and not scope
        ):
            raise CheckError("RBAC binding role reference is invalid")
        role_key = (
            str(kind),
            str(scope) if kind == "Role" else "",
            str(reference.get("name") or ""),
        )
        role = roles.get(role_key)
        if role is None:
            raise CheckError("RBAC bound role is absent or incomplete")
        rules = role.get("rules")
        if rules is None:
            rules = []
        if not isinstance(rules, list):
            raise CheckError("RBAC bound role is absent or incomplete")
        for rule in rules:
            if not isinstance(rule, dict):
                raise CheckError("RBAC bound rule is malformed")
            result.append(
                BoundRule(
                    binding=binding["kind"] + "/" + binding["metadata"]["name"],
                    namespace=scope,
                    rule=rule,
                )
            )
    return result


def named_grant_key(key: str, name: str) -> str:
    """Key of one named-resource expectation: ``core:secrets@secret-name``."""

    return f"{key}@{name}"


def unexpected_grants(
    grants: list[BoundRule],
    *,
    expected_cluster: dict[str, list[str]],
    expected_namespaces: dict[str, dict[str, list[str]]],
    expected_named_namespaces: Mapping[str, Mapping[str, list[str]]] | None = None,
    cpu: bool = False,
) -> list[dict[str, Any]]:
    """Reject extra actionable or sensitive grants, including named resources.

    ``expected_named_namespaces[namespace][named_grant_key(key, name)]`` lists
    the verbs a *named* rule may hold in one namespace. Such an expectation
    satisfies a rule only when the rule carries ``resourceNames`` and every
    name it lists is expected for the verb; a rule without ``resourceNames``
    is judged solely against the unnamed expectations, so a manifest's
    named Secret grant can never excuse namespace-wide Secret access. The
    ``cpu`` policy ignores both expectation maps.
    """

    named_expectations = expected_named_namespaces or {}
    errors = []
    for grant in grants:
        rule = grant.rule
        verbs = _strings(rule.get("verbs"))
        if rule.get("nonResourceURLs"):
            if not set(verbs) <= {"get", "head"}:
                errors.append({"binding": grant.binding, "non_resource_write": True})
            continue
        groups = _strings(rule.get("apiGroups"), nonempty=False)
        if not groups:
            raise CheckError("RBAC resource rule has no API group")
        resources = _strings(rule.get("resources"))
        names = _strings(rule.get("resourceNames", []), nonempty=False)
        for group in groups:
            for resource in resources:
                base = resource.split("/", 1)[0]
                for verb in verbs:
                    if verb == "create" and resource in SELF_REVIEW_RESOURCES.get(
                        group, set()
                    ):
                        continue
                    relevant = (
                        base in WORKLOAD_RESOURCES | SENSITIVE_RESOURCES
                        or base == "*"
                        or verb in WRITE_VERBS
                    )
                    if not relevant:
                        continue
                    key = f"{group or 'core'}:{resource}"
                    allowed = set(expected_cluster.get(key, []))
                    if grant.namespace is not None:
                        allowed.update(
                            expected_namespaces.get(grant.namespace, {}).get(key, [])
                        )
                    denied = verb not in allowed
                    if denied and names and grant.namespace is not None:
                        named = named_expectations.get(grant.namespace, {})
                        denied = any(
                            verb not in named.get(named_grant_key(key, name), [])
                            for name in names
                        )
                    if cpu:
                        denied = verb in WRITE_VERBS or (
                            resource in {"pods/exec", "pods/attach", "nodes/proxy"}
                            and verb == "get"
                        )
                    if denied:
                        errors.append(
                            {
                                "binding": grant.binding,
                                "namespace": grant.namespace,
                                "api_group": group,
                                "resource": resource,
                                "verb": verb,
                                "resource_names": names,
                            }
                        )
    return errors
