"""Renderer-derived RBAC proof and post-quiescence, UID-bound deletion."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from urllib.parse import quote

from gpu_fault.admin.aws_commands import wait_until
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_kubernetes import Command, namespace_document
from gpu_fault.admin.cluster_removal_state import canonical_digest
from gpu_fault.admin.execution import deadline_scope
from gpu_fault_release.regional_release_gpu_rollout import (
    EXECUTOR_SERVICE_ACCOUNT,
    WATCHER_SERVICE_ACCOUNT,
    WORKLOAD_NAMESPACE_RBAC_LABEL,
    render_workload_namespace_rbac,
)
from gpu_fault_release.regional_deployment_inventory import (
    GPU_EXECUTOR_DEPLOYMENT,
    GPU_WATCHER_DEPLOYMENT,
)

RBAC_API = "rbac.authorization.k8s.io/v1"


def _identity(document: Mapping[str, Any]) -> str:
    metadata = document["metadata"]
    return f"{metadata['namespace']}/{document['kind']}/{metadata['name']}"


def _metadata(document: Any, kind: str, namespace: str, name: str) -> dict[str, Any]:
    metadata = document.get("metadata") if isinstance(document, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(namespace, str)
        or not namespace
        or not isinstance(name, str)
        or not name
        or document.get("kind") != kind
        or metadata.get("namespace") != namespace
        or metadata.get("name") != name
        or any(
            not isinstance(metadata.get(field), str) or not metadata[field]
            for field in ("uid", "resourceVersion")
        )
    ):
        raise BootstrapError("workload RBAC resource identity is incomplete")
    return metadata


def _policy(document: Mapping[str, Any]) -> object:
    if document["kind"] == "Role":
        rules = document.get("rules")
        if not isinstance(rules, list) or any(
            not isinstance(rule, dict) for rule in rules
        ):
            raise BootstrapError("workload Role rules are malformed")
        return sorted(
            json.dumps(rule, sort_keys=True, separators=(",", ":")) for rule in rules
        )
    subjects = document.get("subjects")
    if not isinstance(subjects, list) or any(
        not isinstance(item, dict) for item in subjects
    ):
        raise BootstrapError("workload RoleBinding subjects are malformed")
    normalized = []
    for item in subjects:
        subject = dict(item)
        if subject.get("kind") == "ServiceAccount" and subject.get("apiGroup") == "":
            subject.pop("apiGroup")
        normalized.append(subject)
    return {"roleRef": document.get("roleRef"), "subjects": normalized}


def _list_rbac(run: Command, kubectl: Sequence[str]) -> dict[str, dict[str, Any]]:
    result = run(
        [
            *kubectl,
            "get",
            "roles,rolebindings",
            "--all-namespaces",
            "-o",
            "json",
            "--request-timeout=15s",
        ],
        timeout_seconds=20,
    )
    if result.returncode:
        raise BootstrapError("cannot list workload RBAC ownership")
    try:
        document = json.loads(result.stdout)
        items = document["items"]
        if not isinstance(items, list):
            raise ValueError
        found: dict[str, dict[str, Any]] = {}
        for item in items:
            metadata = item["metadata"]
            if (
                item["kind"] not in {"Role", "RoleBinding"}
                or item.get("apiVersion") != RBAC_API
                or metadata.get("labels") is not None
                and not isinstance(metadata["labels"], dict)
            ):
                raise ValueError
            _metadata(item, item["kind"], metadata["namespace"], metadata["name"])
            key = _identity(item)
            if key in found:
                raise ValueError
            found[key] = item
        return found
    except (KeyError, TypeError, ValueError):
        raise BootstrapError("workload RBAC inventory is malformed") from None


class _WorkloadRbac:
    def __init__(
        self,
        config: Mapping[str, Any],
        target: dict[str, Any],
        run: Command,
        *,
        kubectl: Sequence[str],
    ) -> None:
        if (
            not isinstance(config.get("namespace"), str)
            or not config["namespace"]
            or not isinstance(config.get("clusters"), list)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("cluster_id"), str)
                for item in config["clusters"]
            )
            or any(
                not isinstance(target.get(key), str) or not target[key]
                for key in (
                    "cluster_id",
                    "context",
                    "eks_cluster_arn",
                    "executor_irsa_role_arn",
                )
            )
        ):
            raise BootstrapError(
                "workload RBAC requires complete approved cluster configuration"
            )
        self.target, self.run = target, run
        self.namespace = str(config["namespace"])
        self.kubectl = list(kubectl)
        if "--context" in self.kubectl:
            position = self.kubectl.index("--context")
            if self.kubectl[position + 1 : position + 2] != [str(target["context"])]:
                raise BootstrapError("workload RBAC argv selects a different context")
        allowed = target.get("allowed_namespaces")
        if not isinstance(allowed, list) or any(
            not isinstance(value, str) or not value for value in allowed
        ):
            raise BootstrapError("workload RBAC cleanup lacks bound target identity")
        rendered = render_workload_namespace_rbac(
            allowed, system_namespace=self.namespace
        )
        self.expected = {
            _identity(item): item for group in rendered.values() for item in group
        }
        self.binding = {
            "cluster_id": target["cluster_id"],
            "eks_arn": target["eks_cluster_arn"],
            "hyperpod_arn": target.get("expected_hyperpod_arn"),
            "context": target["context"],
            "system_namespace": self.namespace,
            "system_namespace_uid": target.get("expected_namespace_uid"),
            "executor_role_arn": target["executor_irsa_role_arn"],
            "expected_sha256": canonical_digest(self.expected),
        }
        members = [
            item
            for item in config["clusters"]
            if item["cluster_id"] == target["cluster_id"]
        ]
        if len(members) != 1 or any(
            members[0].get(field) != target.get(field)
            for field in (
                "eks_cluster_arn",
                "context",
                "allowed_namespaces",
                "executor_irsa_role_arn",
            )
        ):
            raise BootstrapError("workload RBAC target differs from the managed member")

    def _get(self, kind: str, namespace: str, name: str) -> dict[str, Any] | None:
        result = self.run(
            [
                *self.kubectl,
                "-n",
                namespace,
                "get",
                kind.lower(),
                name,
                "--ignore-not-found",
                "-o",
                "json",
                "--request-timeout=15s",
            ],
            timeout_seconds=20,
        )
        if result.returncode:
            raise BootstrapError("cannot read workload RBAC ownership")
        if not (result.stdout or "").strip():
            return None
        try:
            document = json.loads(result.stdout)
        except ValueError:
            raise BootstrapError("workload RBAC query returned invalid JSON") from None
        _metadata(document, kind, namespace, name)
        return cast(dict[str, Any], document)

    def _list(self) -> dict[str, dict[str, Any]]:
        return _list_rbac(self.run, self.kubectl)

    def _validate_object(self, key: str, document: dict[str, Any]) -> None:
        expected = self.expected[key]
        metadata = _metadata(
            document,
            expected["kind"],
            expected["metadata"]["namespace"],
            expected["metadata"]["name"],
        )
        if (
            document.get("apiVersion") != RBAC_API
            or (metadata.get("labels") or {}).get(WORKLOAD_NAMESPACE_RBAC_LABEL)
            != "true"
            or metadata.get("ownerReferences")
            or _policy(document) != _policy(expected)
        ):
            raise BootstrapError(
                f"workload RBAC ownership differs from renderer: {key}"
            )

    def _shared_bindings(
        self,
        found: dict[str, dict[str, Any]],
        keys: set[str],
        *,
        uids: Mapping[str, str] | None = None,
        removed: Sequence[str] = (),
    ) -> None:
        roles = {key for key in keys if self.expected[key]["kind"] == "Role"}
        for key, item in found.items():
            if item["kind"] != "RoleBinding":
                continue
            reference = item.get("roleRef") or {}
            role = f"{item['metadata']['namespace']}/Role/{reference.get('name')}"
            if reference.get("kind") == "Role" and role in roles:
                if (
                    key not in keys
                    or key in removed
                    or uids is not None
                    and item["metadata"]["uid"] != uids.get(key)
                ):
                    raise BootstrapError(
                        "workload Role has a foreign binding; cleanup refused"
                    )
                self._validate_object(key, item)

    def _anchors(self, namespaces: set[str], accounts: set[str]) -> dict[str, Any]:
        namespace_uids = {}
        for namespace in sorted(namespaces | {self.namespace}):
            document = namespace_document(self.run, self.kubectl, namespace)
            if document is None or document["metadata"].get("deletionTimestamp"):
                raise BootstrapError("workload RBAC namespace is absent or terminating")
            namespace_uids[namespace] = document["metadata"]["uid"]
        if namespace_uids[self.namespace] != self.binding["system_namespace_uid"]:
            raise BootstrapError("workload RBAC system namespace was recreated")
        anchors = {}
        deployments = {
            EXECUTOR_SERVICE_ACCOUNT: GPU_EXECUTOR_DEPLOYMENT,
            WATCHER_SERVICE_ACCOUNT: GPU_WATCHER_DEPLOYMENT,
        }
        for account in sorted(accounts):
            sa = self._get("ServiceAccount", self.namespace, account)
            deployment = self._get("Deployment", self.namespace, deployments[account])
            if sa is None or deployment is None:
                raise BootstrapError(
                    "workload RBAC ServiceAccount ownership is unavailable"
                )
            role = (sa["metadata"].get("annotations") or {}).get(
                "eks.amazonaws.com/role-arn"
            )
            expected_role = (
                self.binding["executor_role_arn"]
                if account == EXECUTOR_SERVICE_ACCOUNT
                else None
            )
            if (
                sa.get("apiVersion") != "v1"
                or deployment.get("apiVersion") != "apps/v1"
                or role != expected_role
                or sa["metadata"].get("ownerReferences")
                or sa["metadata"].get("deletionTimestamp")
                or deployment["metadata"].get("deletionTimestamp")
                or deployment.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("serviceAccountName")
                != account
            ):
                raise BootstrapError(
                    "workload RBAC ServiceAccount is not the managed identity"
                )
            anchors[account] = {
                "uid": sa["metadata"]["uid"],
                "deployment_uid": deployment["metadata"]["uid"],
            }
        return {"namespace_uids": namespace_uids, "service_accounts": anchors}

    def validate_recorded(self, proof: dict[str, Any]) -> None:
        """Validate source scope and the recorded identities, not a self-hash."""
        try:
            if (
                set(proof)
                != {"binding", "resources", "anchors", "documents", "removed"}
                or not isinstance(proof.get("binding"), dict)
                or not isinstance(proof.get("resources"), dict)
                or set(proof["resources"]) - self.expected.keys()
                or any(
                    not isinstance(uid, str) or not uid
                    for uid in proof["resources"].values()
                )
                or len(set(proof["resources"].values())) != len(proof["resources"])
                or not isinstance(proof.get("removed"), list)
                or any(not isinstance(key, str) for key in proof["removed"])
                or len(set(proof["removed"])) != len(proof["removed"])
                or set(proof["removed"]) - proof["resources"].keys()
                or not isinstance(proof.get("anchors"), dict)
                or not isinstance(proof.get("documents"), list)
            ):
                raise ValueError
            namespace_uid = proof["binding"].get("system_namespace_uid")
            if namespace_uid is not None and (
                not isinstance(namespace_uid, str) or not namespace_uid
            ):
                raise ValueError
            if self.binding["system_namespace_uid"] not in {None, namespace_uid}:
                raise ValueError
            self.binding["system_namespace_uid"] = namespace_uid
            if proof["binding"] != self.binding:
                raise ValueError
            namespaces, accounts = self._required_anchors(set(proof["resources"]))
            if proof["resources"]:
                anchors = proof["anchors"]
                if (
                    not namespace_uid
                    or not isinstance(anchors.get("namespace_uids"), dict)
                    or set(anchors["namespace_uids"]) != namespaces | {self.namespace}
                    or anchors["namespace_uids"].get(self.namespace) != namespace_uid
                    or any(
                        not isinstance(uid, str) or not uid
                        for uid in anchors["namespace_uids"].values()
                    )
                    or not isinstance(anchors.get("service_accounts"), dict)
                    or set(anchors["service_accounts"]) != accounts
                    or any(
                        not isinstance(value, dict)
                        or set(value) != {"uid", "deployment_uid"}
                        or any(
                            not isinstance(uid, str) or not uid
                            for uid in value.values()
                        )
                        for value in anchors["service_accounts"].values()
                    )
                ):
                    raise ValueError
            elif proof["anchors"]:
                raise ValueError
            documents = {}
            for item in proof["documents"]:
                key = _identity(item)
                if key not in proof["resources"] or key in documents:
                    raise ValueError
                self._validate_object(key, item)
                if item["metadata"]["uid"] != proof["resources"][key]:
                    raise ValueError
                documents[key] = item
            if set(documents) != set(proof["resources"]):
                raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError):
            raise BootstrapError(
                "workload RBAC recorded proof is invalid or rebound"
            ) from None

    def _verify_absent(self, state: dict[str, Any]) -> None:
        found = self._list()
        for key in state["resources"]:
            if key in found:
                raise BootstrapError(
                    "removed workload RBAC reappeared; refusing replacement"
                )
        for key in self.expected.keys() & found.keys():
            if (found[key]["metadata"].get("labels") or {}).get(
                WORKLOAD_NAMESPACE_RBAC_LABEL
            ) == "true":
                raise BootstrapError("workload RBAC remains after cleanup")
        original = namespace_document(self.run, self.kubectl, self.namespace)
        if (
            original is not None
            and original["metadata"]["uid"] != self.binding["system_namespace_uid"]
        ):
            raise BootstrapError("workload RBAC system namespace was recreated")

    def inspect(self, found: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Inspect current source-owned pairs without writing files or resources."""
        names = {item["metadata"]["name"] for item in self.expected.values()}
        for key, item in found.items():
            if (
                item["metadata"]["name"] in names
                and key not in self.expected
                and (item["metadata"].get("labels") or {}).get(
                    WORKLOAD_NAMESPACE_RBAC_LABEL
                )
                == "true"
            ):
                raise BootstrapError(
                    "workload RBAC is outside the recorded namespace scope"
                )
        selected = {
            key: item
            for key, item in found.items()
            if key in self.expected
            and (item["metadata"].get("labels") or {}).get(
                WORKLOAD_NAMESPACE_RBAC_LABEL
            )
            == "true"
        }
        for key, item in selected.items():
            self._validate_object(key, item)
            other_kind = "RoleBinding" if item["kind"] == "Role" else "Role"
            counterpart = f"{item['metadata']['namespace']}/{other_kind}/{item['metadata']['name']}"
            if counterpart not in selected:
                raise BootstrapError("workload RBAC pair is incomplete or foreign")
        self._shared_bindings(found, set(selected))
        namespaces, accounts = self._required_anchors(set(selected))
        state = {
            "binding": dict(self.binding),
            "resources": {
                key: item["metadata"]["uid"] for key, item in selected.items()
            },
            "anchors": self._anchors(namespaces, accounts) if selected else {},
            "removed": [],
            "documents": [
                {
                    **self.expected[key],
                    "metadata": {
                        **self.expected[key]["metadata"],
                        "uid": item["metadata"]["uid"],
                        "resourceVersion": item["metadata"]["resourceVersion"],
                    },
                }
                for key, item in sorted(selected.items())
            ],
        }
        if not selected:
            self._verify_absent(state)
        return state

    def _required_anchors(self, keys: set[str]) -> tuple[set[str], set[str]]:
        namespaces: set[str] = set()
        accounts: set[str] = set()
        for key in keys:
            item = self.expected[key]
            metadata = item["metadata"]
            namespaces.add(metadata["namespace"])
            other = "RoleBinding" if item["kind"] == "Role" else "Role"
            if f"{metadata['namespace']}/{other}/{metadata['name']}" not in keys:
                raise BootstrapError("workload RBAC proof has an incomplete pair")
            if item["kind"] == "RoleBinding":
                accounts.update(subject["name"] for subject in item["subjects"])
        return namespaces, accounts

    def _remove(self, state: dict[str, Any], key: str) -> None:
        expected = self.expected[key]
        namespace, name = (
            expected["metadata"]["namespace"],
            expected["metadata"]["name"],
        )
        current = self._get(expected["kind"], namespace, name)
        if current is None:
            return
        if (
            key in state["removed"]
            or current["metadata"]["uid"] != state["resources"][key]
        ):
            raise BootstrapError("workload RBAC UID changed; refusing replacement")
        self._validate_object(key, current)
        anchors = state["anchors"]
        if (
            self._anchors(
                set(anchors["namespace_uids"]), set(anchors["service_accounts"])
            )
            != anchors
        ):
            raise BootstrapError("workload RBAC ownership changed since preparation")
        self._shared_bindings(
            self._list(),
            set(state["resources"]),
            uids=state["resources"],
            removed=state["removed"],
        )
        plural = "roles" if expected["kind"] == "Role" else "rolebindings"
        result = self.run(
            [
                *self.kubectl,
                "delete",
                "--raw",
                f"/apis/{RBAC_API}/namespaces/{quote(namespace, safe='')}/{plural}/{quote(name, safe='')}",
                "-f",
                "-",
            ],
            input_text=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "preconditions": {
                        "uid": state["resources"][key],
                        "resourceVersion": current["metadata"]["resourceVersion"],
                    },
                    "propagationPolicy": "Foreground",
                }
            ),
            timeout_seconds=30,
        )

        def absent() -> bool:
            observed = self._get(expected["kind"], namespace, name)
            if (
                observed is not None
                and observed["metadata"]["uid"] != state["resources"][key]
            ):
                raise BootstrapError("workload RBAC was recreated during deletion")
            return observed is None

        if result.returncode and not absent():
            raise BootstrapError("workload RBAC deletion was not confirmed")
        wait_until(absent, description="workload RBAC deletion", timeout_seconds=120)

    def delete_recorded(
        self, proof: dict[str, Any], checkpoint: Callable[[], None]
    ) -> list[str]:
        self.validate_recorded(proof)
        with deadline_scope("workload RBAC cleanup", 600):
            if set(proof["removed"]) == set(proof["resources"]):
                self._verify_absent(proof)
                return sorted(proof["resources"])
            keys = sorted(
                proof["resources"],
                key=lambda key: (self.expected[key]["kind"] != "RoleBinding", key),
            )
            for key in keys:
                self._remove(proof, key)
                if key not in proof["removed"]:
                    proof["removed"].append(key)
                    checkpoint()
            self._verify_absent(proof)
            return sorted(proof["resources"])


def inspect_workload_namespace_rbac(
    config: Mapping[str, Any],
    target: dict[str, Any],
    *,
    run: Command,
    kubectl: Sequence[str],
) -> dict[str, Any] | None:
    """Return live proof, or None for a successful listing without owned labels."""
    found = _list_rbac(run, kubectl)
    if not any(
        WORKLOAD_NAMESPACE_RBAC_LABEL in (item["metadata"].get("labels") or {})
        for item in found.values()
    ):
        return None
    if not isinstance(target.get("allowed_namespaces"), list):
        raise BootstrapError(
            "workload RBAC inspection requires the namespace allowlist"
        )
    if not isinstance(config.get("namespace"), str) or not config["namespace"]:
        raise BootstrapError("workload RBAC inspection requires the system namespace")
    namespace = str(config["namespace"])
    document = namespace_document(run, kubectl, namespace)
    observed_uid = document["metadata"]["uid"] if document is not None else None
    if target.get("expected_namespace_uid") is not None and (
        target["expected_namespace_uid"] != observed_uid
    ):
        raise BootstrapError("workload RBAC system namespace was recreated")
    selected = {
        **target,
        "expected_namespace_uid": observed_uid,
    }
    return _WorkloadRbac(config, selected, run, kubectl=kubectl).inspect(found)


def delete_recorded_workload_namespace_rbac(
    config: Mapping[str, Any],
    target: dict[str, Any],
    proof: dict[str, Any],
    *,
    run: Command,
    kubectl: Sequence[str],
    checkpoint: Callable[[], None],
) -> list[str]:
    """Delete after caller-verified quiescence, checkpointing confirmed originals."""
    return _WorkloadRbac(config, target, run, kubectl=kubectl).delete_recorded(
        proof, checkpoint
    )
