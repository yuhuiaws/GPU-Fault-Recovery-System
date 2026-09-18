from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.regional.test_installed_resource_registry import (
    COLLECT_MODULE,
    MODULE,
    FakeKubectl,
    write_inventory,
)

NAMESPACE = "gpu-fault-system"


def registry(resources: list[dict], **overrides) -> dict:
    return {
        "schema_version": 1,
        "plane": "cpu",
        "namespace": NAMESPACE,
        "resources": resources,
        **overrides,
    }


def sync(kubectl, inventory: Path, *, apply: bool = True) -> dict:
    return MODULE.synchronize(
        kubectl,
        plane="cpu",
        namespace=NAMESPACE,
        inventory_path=inventory,
        release_id="candidate",
        apply=apply,
    )


def resource(kind: str, name: str, namespace: str | None) -> dict:
    return {
        "kind": kind,
        "metadata": {
            "name": name,
            **({"namespace": namespace} if namespace is not None else {}),
            "uid": f"uid-{name}",
        },
    }


class ScopedDiscovery:
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def run(self, arguments: list[str], **_kwargs):
        if "-A" in arguments:
            items = [item for item in self.items if "namespace" in item["metadata"]]
        elif arguments[1] == "namespace":
            items = [item for item in self.items if item["kind"] == "Namespace"]
        else:
            items = [
                item
                for item in self.items
                if item["kind"] in {"ClusterRole", "ClusterRoleBinding"}
            ]
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"items": items}), ""
        )


def discover(kubectl, registered: set[tuple]) -> list[dict]:
    return COLLECT_MODULE.discover_unregistered(
        kubectl,
        plane="cpu",
        context="fixture-context",
        namespace=NAMESPACE,
        registered=registered,
    )


@pytest.mark.parametrize("legacy_keys", [False, True])
def test_same_name_in_a_legacy_namespace_is_not_registered(legacy_keys: bool) -> None:
    name = "gpu-fault-api"
    kubectl = ScopedDiscovery(
        [
            resource("Deployment", name, NAMESPACE),
            resource("Deployment", name, "gf-regional-old"),
            resource("Deployment", name, "training"),
            resource("ClusterRole", name, None),
            resource("Namespace", "gf-regional-old", None),
        ]
    )
    registered = (
        {("deployment", name)}
        if legacy_keys
        else {("namespaced", "deployment", NAMESPACE, name)}
    )
    found = discover(kubectl, registered)
    assert {
        (item["scope"], item["kind"], item["namespace"], item["name"]) for item in found
    } == {
        ("namespaced", "deployment", "gf-regional-old", name),
        ("cluster", "clusterrole", None, name),
        ("cluster", "namespace", None, "gf-regional-old"),
    }


@pytest.mark.parametrize(
    "registered",
    [
        {("namespaced", "deployment", "gf-regional-old", "gpu-fault-api")},
        {("cluster", "deployment", None, "gpu-fault-api")},
        {("namespaced", "role", NAMESPACE, "gpu-fault-api")},
    ],
)
def test_a_different_scope_namespace_or_kind_cannot_cover_a_resource(
    registered,
) -> None:
    found = discover(
        ScopedDiscovery([resource("Deployment", "gpu-fault-api", NAMESPACE)]),
        registered,
    )
    assert len(found) == 1
    assert found[0]["namespace"] == NAMESPACE


def test_a_same_named_legacy_cronjob_does_not_authorize_its_job() -> None:
    name = "gpu-fault-aurora-credential-refresh"
    current = resource("CronJob", name, NAMESPACE)
    current["apiVersion"] = "batch/v1"
    old = resource("CronJob", name, "gf-regional-old")
    old["apiVersion"] = "batch/v1"
    old["metadata"]["uid"] = "old-parent"
    job = resource("Job", "gpu-fault-store-proof-fixture", "gf-regional-old")
    job["metadata"]["ownerReferences"] = [
        {"apiVersion": "batch/v1", "kind": "CronJob", "name": name, "uid": "old-parent"}
    ]
    found = discover(
        ScopedDiscovery([current, old, job]),
        {("namespaced", "cronjob", NAMESPACE, name)},
    )
    assert {(item["kind"], item["namespace"]) for item in found} == {
        ("cronjob", "gf-regional-old"),
        ("job", "gf-regional-old"),
    }


@pytest.mark.parametrize("missing_parent_version", [False, True])
def test_incomplete_owner_identity_cannot_cover_a_job(
    missing_parent_version: bool,
) -> None:
    name = "gpu-fault-aurora-credential-refresh"
    parent = resource("CronJob", name, NAMESPACE)
    parent["apiVersion"] = "batch/v1"
    job = resource("Job", "gpu-fault-store-proof-fixture", NAMESPACE)
    owner = {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "name": name,
        "uid": parent["metadata"]["uid"],
    }
    if missing_parent_version:
        parent.pop("apiVersion")
        owner.pop("apiVersion")
    else:
        owner["uid"] = []
    job["metadata"]["ownerReferences"] = [owner]
    found = discover(
        ScopedDiscovery([parent, job]), {("namespaced", "cronjob", NAMESPACE, name)}
    )
    assert [item["kind"] for item in found] == ["job"]


@pytest.mark.parametrize(
    ("kind", "name", "scope"),
    [
        ("clusterrole", "cluster-admin", "cluster"),
        ("namespace", "training", "cluster"),
        ("deployment", "gpu-fault-unreviewed", "namespaced"),
    ],
)
@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("apply", [False, True])
def test_restamped_unknown_rows_never_authorize_or_disappear(
    tmp_path: Path, kind: str, name: str, scope: str, present: bool, apply: bool
) -> None:
    entry = MODULE.stamp_provenance(
        {
            "kind": kind,
            "name": name,
            "scope": scope,
            "phase": "support",
            "order": 100,
            "clean": "delete",
            "retired": True,
        }
    )
    assert MODULE.provenance_is_verified(entry), (
        "restamped fixture row must have valid provenance"
    )
    inventory = write_inventory(tmp_path)
    existing = registry(
        [entry], source_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest()
    )
    before = copy.deepcopy(existing)
    kubectl = FakeKubectl(
        present={(kind, name)} if present else set(), existing=existing
    )
    with pytest.raises(
        MODULE.RegistryError, match="index 0.*no current source authorization"
    ):
        sync(kubectl, inventory, apply=apply)
    assert kubectl.applied is None
    assert kubectl.existing == before
    assert len(kubectl.calls) == 1, "unknown rows must not even select resource queries"


@pytest.mark.parametrize(
    "change",
    [
        {"namespace": "gf-regional-old"},
        {"scope": "cluster"},
        {"scope": "cluster", "namespace": NAMESPACE},
    ],
)
def test_retained_identity_cannot_change_namespace_or_scope(
    tmp_path: Path, change
) -> None:
    path = write_inventory(tmp_path)
    entry = json.loads(path.read_text())["cpu"]["resources"][0]
    entry.update(change)
    kubectl = FakeKubectl(
        present={("deployment", "gpu-fault-api")},
        existing=registry([MODULE.stamp_provenance(entry)]),
    )
    with pytest.raises(MODULE.RegistryError):
        sync(kubectl, path)
    assert kubectl.applied is None
    assert len(kubectl.calls) == 1


@pytest.mark.parametrize("document_namespace", [None, "gf-regional-old"])
def test_registry_document_must_bind_the_selected_namespace(
    tmp_path: Path, document_namespace
) -> None:
    kubectl = FakeKubectl(
        present=set(), existing=registry([], namespace=document_namespace)
    )
    with pytest.raises(MODULE.RegistryError, match="registry namespace"):
        sync(kubectl, write_inventory(tmp_path))
    assert kubectl.applied is None


def test_source_fields_replace_mutable_policy_on_a_known_legacy_row(
    tmp_path: Path,
) -> None:
    path = write_inventory(tmp_path)
    expected = json.loads(path.read_text())["cpu"]["resources"][0]
    forged = MODULE.stamp_provenance(
        {**expected, "clean": "reset", "order": 0, "phase": "support", "retired": True}
    )
    kubectl = FakeKubectl(
        present={("deployment", expected["name"])}, existing=registry([forged])
    )
    document = sync(kubectl, path)
    assert document["resources"] == [
        MODULE.stamp_provenance({**expected, "namespace": NAMESPACE})
    ]
    assert kubectl.applied is not None


def test_foreign_source_namespace_is_refused_before_any_query(tmp_path: Path) -> None:
    path = write_inventory(tmp_path)
    source = json.loads(path.read_text())
    source["cpu"]["resources"][0]["namespace"] = "gf-regional-old"
    path.write_text(json.dumps(source))
    kubectl = FakeKubectl(present=set())
    with pytest.raises(MODULE.RegistryError, match="source inventory targets"):
        sync(kubectl, path)
    assert kubectl.calls == []


class ResponseKubectl(FakeKubectl):
    def __init__(self, target: str, code: int, output: str, error: str = "") -> None:
        super().__init__(present=set())
        self.target = target
        self.response = code, output, error

    def run(self, arguments: list[str], **kwargs):
        if "get" in arguments and arguments[arguments.index("get") + 1] == self.target:
            self.calls.append(arguments)
            return subprocess.CompletedProcess(arguments, *self.response)
        return super().run(arguments, **kwargs)


@pytest.mark.parametrize("target", ["configmap", "deployment"])
def test_notfound_in_an_error_is_not_proof_of_absence(
    tmp_path: Path, target: str
) -> None:
    kubectl = ResponseKubectl(
        target, 1, "", "credential helper NotFound: opaque-credential-fixture"
    )
    with pytest.raises(MODULE.RegistryError) as error:
        sync(kubectl, write_inventory(tmp_path))
    assert "opaque-credential-fixture" not in str(error.value)
    assert kubectl.applied is None


@pytest.mark.parametrize(
    "item",
    [
        resource("Deployment", "gpu-fault-api", "gf-regional-old"),
        resource("DaemonSet", "gpu-fault-api", NAMESPACE),
        resource("Deployment", "gpu-fault-api", None),
        {"kind": "Deployment", "metadata": {}},
    ],
)
def test_unbound_list_results_cannot_change_the_registry(
    tmp_path: Path, item: dict
) -> None:
    kubectl = ResponseKubectl("deployment", 0, json.dumps({"items": [item]}))
    with pytest.raises(MODULE.RegistryError):
        sync(kubectl, write_inventory(tmp_path))
    assert kubectl.applied is None


@pytest.mark.parametrize("field", ["namespace", "name", "uid"])
def test_the_registry_configmap_identity_is_verified(
    tmp_path: Path, field: str
) -> None:
    document = resource("ConfigMap", MODULE.REGISTRY_NAME, NAMESPACE)
    document["metadata"][field] = "" if field == "uid" else "different"
    document["data"] = {"inventory.json": json.dumps(registry([]))}
    kubectl = ResponseKubectl("configmap", 0, json.dumps(document))
    with pytest.raises(MODULE.RegistryError, match="identity or absence"):
        sync(kubectl, write_inventory(tmp_path))
    assert kubectl.applied is None


@pytest.mark.parametrize("crd_code", [0, 1])
def test_optional_api_absence_requires_a_successful_crd_probe(
    tmp_path: Path, crd_code: int
) -> None:
    path = write_inventory(tmp_path)
    source = json.loads(path.read_text())
    source["cpu"]["resources"] = [
        {
            "kind": "prometheusrule",
            "name": "gpu-fault-alerts",
            "scope": "namespaced",
            "phase": "support",
            "order": 100,
            "clean": "namespace",
        }
    ]
    path.write_text(json.dumps(source))
    kubectl = ResponseKubectl(
        "customresourcedefinition",
        crd_code,
        "",
        "the server doesn't have a resource type",
    )
    if crd_code:
        with pytest.raises(MODULE.RegistryError, match="optional PrometheusRule API"):
            sync(kubectl, path)
        assert kubectl.applied is None
    else:
        assert sync(kubectl, path, apply=False)["resources"] == []
    assert all("prometheusrule" not in call for call in kubectl.calls), (
        "PrometheusRule queries require confirmed CRD availability"
    )


def test_collect_passes_scoped_identities_and_closes_each_client(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_inventory(tmp_path)
    source = json.loads(path.read_text())
    source["cpu"]["database_pod_preference"] = ["gpu-fault-api"]
    source["gpu"]["resources"] = copy.deepcopy(source["cpu"]["resources"])
    path.write_text(json.dumps(source))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "namespace": NAMESPACE,
                "cpu_kubeconfig": "cpu",
                "clusters": [{"context": "gpu-a"}, {"context": "gpu-b"}],
            }
        )
    )
    clients = []

    class Client(FakeKubectl):
        def __init__(self, *, context, **_kwargs) -> None:
            super().__init__(present={("deployment", "gpu-fault-api")})
            self.context = context
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

        def run(self, arguments: list[str], **kwargs):
            if arguments[:2] == ["get", "roles,rolebindings"]:
                assert "--all-namespaces" in arguments, (
                    "workload RBAC inspection must include every namespace"
                )
                assert kwargs == {"check": False, "timeout_seconds": 20}, (
                    "RBAC inspection must preserve bounded raw query results"
                )
                return subprocess.CompletedProcess(arguments, 0, '{"items":[]}', "")
            if "-A" in arguments or arguments[:2] in [
                ["get", COLLECT_MODULE.CLUSTER_DISCOVERY_KINDS],
                ["get", "namespace"],
            ]:
                items = [resource("Deployment", "gpu-fault-api", NAMESPACE)]
                if self.context:
                    items.append(
                        resource("Deployment", "gpu-fault-api", "gf-regional-old")
                    )
                return ScopedDiscovery(items).run(arguments)
            return super().run(arguments, **kwargs)

    monkeypatch.setattr(COLLECT_MODULE, "Kubectl", Client)
    result = COLLECT_MODULE.collect(config_path, inventory_path=path, apply=False)
    assert {
        (item["context"], item["namespace"], item["name"])
        for item in result["unregistered_resources"]
    } == {
        ("gpu-a", "gf-regional-old", "gpu-fault-api"),
        ("gpu-b", "gf-regional-old", "gpu-fault-api"),
    }
    assert len(result["gpu"]["resources"]) == 1
    assert result["gpu"]["resources"][0]["namespace"] == NAMESPACE
    assert result["cpu"]["database_pod_preference"] == ["gpu-fault-api"]
    assert len(clients) == 3
    assert all(client.closed and client.applied is None for client in clients), (
        "read-only collection must close every client without applying changes"
    )
