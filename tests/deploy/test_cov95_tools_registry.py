from __future__ import annotations

import copy
import json
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from tests.regional.test_installed_resource_registry import COLLECT_MODULE as collect
from tests.regional.test_installed_resource_registry import MODULE as sync
from tests.regional.test_installed_resource_registry import FakeKubectl, write_inventory
from tests.regional.test_installed_resource_registry_safety import registry

NAMESPACE = "gpu-fault-system"


@pytest.mark.parametrize(
    "document,problem",
    [
        ({"users": {}}, None),
        ({"users": []}, None),
        ({"users": "invalid"}, "users are invalid"),
        ({"users": [{}, {}]}, "must have one user"),
        ({"users": ["invalid"]}, "user is invalid"),
        ({"users": [{"user": "invalid"}]}, "user is invalid"),
        ({"users": [{"user": {}}]}, None),
        ({"users": [{"user": {"exec": "invalid"}}]}, "exec credential is invalid"),
    ],
)
def test_registry_client_checks_selected_kubeconfig_before_any_plugin(
    document: dict[str, Any], problem: str | None
) -> None:
    calls = []

    def runner(
        arguments: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, json.dumps(document), "")

    if problem:
        with pytest.raises(sync.RegistryError, match=problem):
            sync.Kubectl(
                kubeconfig=None,
                context="fixture",
                reuse_exec_credential=True,
                runner=runner,
            )
    else:
        with closing(
            sync.Kubectl(
                kubeconfig=None,
                context="fixture",
                reuse_exec_credential=True,
                runner=runner,
            )
        ) as client:
            assert client.prefix == ["kubectl", "--context", "fixture"]
    assert len(calls) == 1
    assert calls[0][3:5] == ["config", "view"]


def test_registry_client_rejects_ambiguous_selection_and_accepts_default_explicit_runner() -> (
    None
):
    with pytest.raises(sync.RegistryError, match="mutually exclusive"):
        sync.Kubectl(kubeconfig="/dev/null", context="fixture")
    client = sync.Kubectl(
        kubeconfig=None,
        context=None,
        runner=lambda *_args, **_kwargs: pytest.fail("construction should not execute"),
    )
    assert client.prefix == ["kubectl"]
    client.close()


@pytest.mark.parametrize(
    "change,problem",
    [
        ({"args": ["eks", "get-token", 1]}, "command is invalid"),
        ({"env": "invalid"}, "environment is invalid"),
        ({"env": [None]}, "environment is invalid"),
        ({"env": [{"name": "", "value": "example"}]}, "environment is invalid"),
        ({"env": [{"name": "EXAMPLE", "value": 1}]}, "environment is invalid"),
    ],
)
def test_invalid_exec_configuration_never_starts_the_plugin(
    change: dict[str, Any], problem: str
) -> None:
    command = {
        "command": "aws",
        "args": ["eks", "get-token", "--cluster-name", "fixture"],
        "apiVersion": "client.authentication.k8s.io/v1",
        **change,
    }
    calls = []

    def runner(
        arguments: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        assert arguments[0] == "kubectl", (
            "invalid exec inputs must stop before the plugin"
        )
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps({"users": [{"name": "fixture", "user": {"exec": command}}]}),
            "",
        )

    with closing(
        sync.Kubectl(
            kubeconfig="/dev/null",
            context=None,
            reuse_exec_credential=True,
            runner=runner,
        )
    ) as client:
        with pytest.raises(sync.RegistryError, match=problem):
            client.run(["get", "deployment"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "value,problem",
    [
        ("{", "not a JSON object"),
        ("[]", "not a JSON object"),
        ("{}", "has no items"),
        ('{"items":[null]}', "invalid item"),
        ('{"items":[{"kind":"","metadata":{"name":"x"}}]}', "invalid identity"),
    ],
)
def test_registry_resource_list_rejects_incomplete_or_untyped_evidence(
    value: str, problem: str
) -> None:
    with pytest.raises(sync.RegistryError, match=problem):
        sync.resource_list(value)


@pytest.mark.parametrize(
    "resource,problem",
    [
        (
            {"scope": "namespaced", "kind": "deployment", "name": "a", "namespace": ""},
            "namespace does not match",
        ),
        (
            {
                "scope": "cluster",
                "kind": "clusterrole",
                "name": "a",
                "namespace": NAMESPACE,
            },
            "namespace does not match",
        ),
        (
            {"scope": "unknown", "kind": "deployment", "name": "a"},
            "identity is invalid",
        ),
    ],
)
def test_registry_resource_scope_is_not_inferred_from_a_name(
    resource: dict[str, Any], problem: str
) -> None:
    with pytest.raises(sync.RegistryError, match=problem):
        sync.resource_identity(resource, NAMESPACE)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("schema", "schema or plane is unsupported"),
        ("plane", "schema or plane is unsupported"),
        ("section", "resources are invalid"),
        ("resources", "resources are invalid"),
        ("resource", "resource is invalid"),
        ("duplicate", "duplicate resource identities"),
    ],
)
def test_invalid_source_inventory_causes_no_live_reads_or_writes(
    tmp_path: Path, fault: str, problem: str
) -> None:
    path = write_inventory(tmp_path)
    source = json.loads(path.read_text())
    plane = "cpu"
    if fault == "schema":
        source["schema_version"] = 2
    elif fault == "plane":
        plane = "unknown"
    elif fault == "section":
        source["cpu"] = []
    elif fault == "resources":
        source["cpu"]["resources"] = {}
    elif fault == "resource":
        source["cpu"]["resources"] = [None]
    else:
        source["cpu"]["resources"].append(copy.deepcopy(source["cpu"]["resources"][0]))
    path.write_text(json.dumps(source))
    client = FakeKubectl(present=set())
    with pytest.raises(sync.RegistryError, match=problem):
        sync.synchronize(
            client,
            plane=plane,
            namespace=NAMESPACE,
            inventory_path=path,
            release_id="example",
            apply=True,
        )
    assert client.calls == []
    assert client.applied is None


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("schema", "schema is unsupported"),
        ("plane", "plane is not cpu"),
        ("resources", "resources are invalid"),
        ("resource", "resource is invalid"),
        ("duplicate", "duplicate resource identities"),
    ],
)
def test_invalid_existing_registry_is_preserved_instead_of_rewritten(
    tmp_path: Path, fault: str, problem: str
) -> None:
    path = write_inventory(tmp_path)
    resource = json.loads(path.read_text())["cpu"]["resources"][0]
    current = registry([resource])
    if fault == "schema":
        current["schema_version"] = 2
    elif fault == "plane":
        current["plane"] = "gpu"
    elif fault == "resources":
        current["resources"] = {}
    elif fault == "resource":
        current["resources"] = [None]
    else:
        current["resources"].append(copy.deepcopy(resource))
    client = FakeKubectl(present=set(), existing=current)
    before = copy.deepcopy(current)
    with pytest.raises(sync.RegistryError, match=problem):
        sync.synchronize(
            client,
            plane="cpu",
            namespace=NAMESPACE,
            inventory_path=path,
            release_id="example",
            apply=True,
        )
    assert client.applied is None
    assert client.existing == before
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "kind,metadata,scope",
    [
        (
            "Namespace",
            {"name": "gpu-fault-wrong", "namespace": NAMESPACE},
            "namespaced",
        ),
        ("Deployment", {"name": "gpu-fault-wrong"}, "namespaced"),
        ("Deployment", {"name": "gpu-fault-wrong"}, "cluster"),
        ("ClusterRole", {"name": "gpu-fault-wrong", "namespace": NAMESPACE}, "cluster"),
        ("ConfigMap", {"name": "gf-regional-old"}, "namespace"),
    ],
)
def test_registry_discovery_rejects_cross_kind_or_namespace_response(
    kind: str, metadata: dict[str, Any], scope: str
) -> None:
    class Client:
        def run(
            self, arguments: list[str], **_kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            actual_scope = (
                "namespaced"
                if "-A" in arguments
                else "namespace"
                if arguments[1] == "namespace"
                else "cluster"
            )
            items = (
                [{"kind": kind, "metadata": metadata}] if actual_scope == scope else []
            )
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps({"items": items}), ""
            )

    with pytest.raises(sync.RegistryError, match="invalid scope or kind"):
        collect.discover_unregistered(
            Client(),
            plane="cpu",
            context="example",
            namespace=NAMESPACE,
            registered=set(),
        )


@pytest.mark.parametrize("dry_run", [False, True])
def test_sync_entrypoint_closes_its_fake_client_and_reports_actual_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    path = write_inventory(tmp_path)
    client = FakeKubectl(present={("deployment", "gpu-fault-api")})
    closed = []
    client.close = lambda: closed.append(True)
    monkeypatch.setattr(sync, "Kubectl", lambda **_kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync",
            "--plane",
            "cpu",
            "--context",
            "example",
            "--inventory",
            str(path),
            *(["--dry-run"] if dry_run else []),
        ],
    )
    assert sync.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "plane": "cpu",
        "resources": 1,
        "registry": sync.REGISTRY_NAME,
        "written": not dry_run,
    }
    assert (client.applied is not None) is (not dry_run)
    assert closed == [True]
