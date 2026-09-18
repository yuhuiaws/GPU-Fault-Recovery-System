from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    ROOT / "deploy" / "control-plane" / "tools" / "sync_installed_resource_registry.py"
)
COLLECT_MODULE = lazy_script_module(
    ROOT
    / "deploy"
    / "control-plane"
    / "tools"
    / "collect_installed_resource_registry.py"
)


class FakeKubectl:
    def __init__(
        self, *, present: set[tuple[str, str]], existing: dict | None = None
    ) -> None:
        self.present = present
        self.existing = existing
        self.applied: dict | None = None
        self.calls: list[list[str]] = []

    def run(
        self, arguments: list[str], *, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments)
        if arguments[0:5] == [
            "-n",
            "gpu-fault-system",
            "get",
            "configmap",
            MODULE.REGISTRY_NAME,
        ]:
            if self.existing is None:
                return subprocess.CompletedProcess(arguments, 0, "", "")
            value = {
                "kind": "ConfigMap",
                "metadata": {
                    "name": MODULE.REGISTRY_NAME,
                    "namespace": "gpu-fault-system",
                    "uid": "registry-uid",
                },
                "data": {"inventory.json": json.dumps(self.existing)},
            }
            return subprocess.CompletedProcess(arguments, 0, json.dumps(value), "")
        if arguments[0] == "apply":
            self.applied = json.loads(input_text or "{}")
            return subprocess.CompletedProcess(arguments, 0, "", "")
        offset = 2 if arguments[0] == "-n" else 0
        kind = arguments[offset + 1]
        items = [
            {
                "kind": kind.title(),
                "metadata": {
                    "name": name,
                    **({"namespace": arguments[1]} if offset else {}),
                },
            }
            for present_kind, name in sorted(self.present)
            if present_kind == kind
        ]
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"items": items}), ""
        )

    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        del timeout_seconds
        completed = self.run(args, check=False)
        return completed.returncode, completed.stdout, completed.stderr


class DiscoveryKubectl:
    def run(
        self, arguments: list[str], *, input_text: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        del input_text, check
        if "-A" in arguments:
            items = [
                {
                    "kind": "Deployment",
                    "metadata": {
                        "namespace": "gf-regional-old",
                        "name": "gpu-fault-legacy-executor",
                    },
                },
                {
                    "kind": "Deployment",
                    "metadata": {
                        "namespace": "training",
                        "name": "gpu-fault-customer-workload",
                    },
                },
                # A workload label alone is not renderer/UID ownership proof.
                {
                    "kind": "Role",
                    "metadata": {
                        "namespace": "gpu-fault-system",
                        "name": "gpu-fault-cluster-executor",
                        "labels": {"gpu-fault.io/workload-namespace-rbac": "true"},
                    },
                },
                {
                    "kind": "RoleBinding",
                    "metadata": {
                        "namespace": "gpu-fault-system",
                        "name": "gpu-fault-completion-watcher",
                        "labels": {"gpu-fault.io/workload-namespace-rbac": "true"},
                    },
                },
                {
                    "kind": "Role",
                    "metadata": {
                        "namespace": "gpu-fault-system",
                        "name": "gpu-fault-hand-made-role",
                    },
                },
            ]
        elif "clusterrole" in arguments[1]:
            items = [
                {"kind": "ClusterRole", "metadata": {"name": "gpu-fault-legacy-role"}}
            ]
        else:
            items = [{"kind": "Namespace", "metadata": {"name": "gf-regional-old"}}]
        return subprocess.CompletedProcess(
            arguments, 0, json.dumps({"items": items}), ""
        )


def write_inventory(tmp_path: Path) -> Path:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cpu": {
                    "resources": [
                        {
                            "kind": "deployment",
                            "name": "gpu-fault-api",
                            "scope": "namespaced",
                            "phase": "ingress",
                            "order": 10,
                            "clean": "delete",
                        },
                        {
                            "kind": "daemonset",
                            "name": "gpu-fault-absent",
                            "scope": "namespaced",
                            "phase": "auxiliary",
                            "order": 20,
                            "clean": "delete",
                        },
                    ]
                },
                "gpu": {"resources": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_first_sync_writes_only_resources_that_exist(tmp_path: Path) -> None:
    kubectl = FakeKubectl(present={("deployment", "gpu-fault-api")})

    document = MODULE.synchronize(
        kubectl,
        plane="cpu",
        namespace="gpu-fault-system",
        inventory_path=write_inventory(tmp_path),
        release_id="release-a",
        apply=True,
    )

    assert [(item["kind"], item["name"]) for item in document["resources"]] == [
        ("deployment", "gpu-fault-api")
    ]
    assert kubectl.applied is not None
    stored = json.loads(kubectl.applied["data"]["inventory.json"])
    assert stored["release_id"] == "release-a"
    assert stored["resources"] == document["resources"]
    discovery = [
        call
        for call in kubectl.calls
        if "get" in call and MODULE.REGISTRY_NAME not in call
    ]
    assert [call[call.index("get") + 1] for call in discovery] == [
        "deployment",
        "daemonset",
    ]


def test_retired_hma_resources_are_discovered_without_an_old_registry() -> None:
    identities = {
        ("deployment", "gpu-fault-hma-watcher"),
        ("deployment", "gpu-fault-hma-cloudwatch-consumer"),
        ("clusterrole", "gpu-fault-hma-watcher"),
        ("clusterrolebinding", "gpu-fault-hma-watcher"),
        ("serviceaccount", "gpu-fault-hma-watcher"),
        ("serviceaccount", "gpu-fault-hma-cloudwatch-consumer"),
    }
    kubectl = FakeKubectl(present=identities)
    document = MODULE.synchronize(
        kubectl,
        plane="gpu",
        namespace="gpu-fault-system",
        inventory_path=MODULE.DEFAULT_INVENTORY,
        release_id="retirement-release",
        apply=False,
    )
    assert {
        (item["kind"], item["name"]) for item in document["resources"]
    } == identities
    assert all(item["retired"] for item in document["resources"]), (
        "discovered HMA resources must be marked retired"
    )
    assert all(MODULE.provenance_is_verified(item) for item in document["resources"]), (
        "discovered HMA resources must have verified provenance"
    )
    assert kubectl.applied is None, "discovery must not delete or modify resources"
    kubectl.present = set()
    kubectl.existing = document
    updated = MODULE.synchronize(
        kubectl,
        plane="gpu",
        namespace="gpu-fault-system",
        inventory_path=MODULE.DEFAULT_INVENTORY,
        release_id="retirement-release",
        apply=False,
    )
    assert updated["resources"] == [], (
        "entries disappear only after resources are absent"
    )


def _legacy_resource() -> dict:
    return {
        "kind": "deployment",
        "name": "gpu-fault-legacy",
        "scope": "namespaced",
        "phase": "producer",
        "order": 90,
        "clean": "delete",
    }


def test_sync_retains_live_legacy_resource_until_deleted(tmp_path: Path) -> None:
    path = write_inventory(tmp_path)
    source = json.loads(path.read_text(encoding="utf-8"))
    source["cpu"]["resources"].append({**_legacy_resource(), "retired": True})
    path.write_text(json.dumps(source), encoding="utf-8")
    legacy = MODULE.stamp_provenance(_legacy_resource())
    kubectl = FakeKubectl(
        present={("deployment", "gpu-fault-api"), ("deployment", "gpu-fault-legacy")},
        existing={
            "schema_version": 1,
            "plane": "cpu",
            "namespace": "gpu-fault-system",
            "resources": [legacy],
        },
    )

    document = MODULE.synchronize(
        kubectl,
        plane="cpu",
        namespace="gpu-fault-system",
        inventory_path=path,
        release_id="release-b",
        apply=False,
    )

    assert {(item["kind"], item["name"]) for item in document["resources"]} == {
        ("deployment", "gpu-fault-api"),
        ("deployment", "gpu-fault-legacy"),
    }
    assert sum(call.count("deployment") for call in kubectl.calls) == 1
    retired = next(
        item for item in document["resources"] if item["name"] == "gpu-fault-legacy"
    )
    assert retired["retired"] is True
    assert retired["namespace"] == "gpu-fault-system"


def test_sync_refuses_unknown_entry_without_provenance(tmp_path: Path) -> None:
    legacy = _legacy_resource()
    assert MODULE.PROVENANCE_KEY not in legacy
    kubectl = FakeKubectl(
        present={("deployment", "gpu-fault-api"), ("deployment", "gpu-fault-legacy")},
        existing={
            "schema_version": 1,
            "plane": "cpu",
            "namespace": "gpu-fault-system",
            "resources": [legacy],
        },
    )

    with pytest.raises(MODULE.RegistryError, match="no current source authorization"):
        MODULE.synchronize(
            kubectl,
            plane="cpu",
            namespace="gpu-fault-system",
            inventory_path=write_inventory(tmp_path),
            release_id="release-b",
            apply=True,
        )
    assert kubectl.applied is None
    assert kubectl.existing["resources"] == [legacy]


def test_sync_refuses_unknown_entry_with_mismatched_provenance(tmp_path: Path) -> None:
    legacy = MODULE.stamp_provenance(_legacy_resource())
    legacy["clean"] = "orphan"  # flip a field without re-stamping the digest
    kubectl = FakeKubectl(
        present={("deployment", "gpu-fault-api"), ("deployment", "gpu-fault-legacy")},
        existing={
            "schema_version": 1,
            "plane": "cpu",
            "namespace": "gpu-fault-system",
            "resources": [legacy],
        },
    )

    with pytest.raises(MODULE.RegistryError, match="no current source authorization"):
        MODULE.synchronize(
            kubectl,
            plane="cpu",
            namespace="gpu-fault-system",
            inventory_path=write_inventory(tmp_path),
            release_id="release-b",
            apply=True,
        )
    assert kubectl.applied is None
    assert kubectl.existing["resources"] == [legacy]


def test_sync_stamps_candidate_provenance(tmp_path: Path) -> None:
    kubectl = FakeKubectl(present={("deployment", "gpu-fault-api")})

    document = MODULE.synchronize(
        kubectl,
        plane="cpu",
        namespace="gpu-fault-system",
        inventory_path=write_inventory(tmp_path),
        release_id="release-a",
        apply=False,
    )

    assert document["resources"], "the live candidate should sync"
    for resource in document["resources"]:
        assert MODULE.provenance_is_verified(resource), resource


def test_kubectl_reuses_one_exec_credential() -> None:
    calls: list[list[str]] = []
    session_files: list[Path] = []

    def runner(arguments: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(arguments)
        if "config" in arguments and "view" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "Config",
                        "current-context": "gpu",
                        "clusters": [
                            {
                                "name": "gpu",
                                "cluster": {
                                    "server": "https://example.invalid",
                                    "certificate-authority-data": "Y2E=",
                                },
                            }
                        ],
                        "contexts": [
                            {
                                "name": "gpu",
                                "context": {"cluster": "gpu", "user": "gpu"},
                            }
                        ],
                        "users": [
                            {
                                "name": "gpu",
                                "user": {
                                    "exec": {
                                        "apiVersion": "client.authentication.k8s.io/v1beta1",
                                        "command": "aws",
                                        "args": ["eks", "get-token"],
                                        "env": [
                                            {"name": "AWS_REGION", "value": "test"}
                                        ],
                                    }
                                },
                            }
                        ],
                    }
                ),
                "",
            )
        if arguments[:2] == ["aws", "eks"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps(
                    {
                        "status": {
                            "token": "private-token",
                            "expirationTimestamp": (
                                datetime.now(timezone.utc) + timedelta(minutes=15)
                            ).isoformat(),
                        }
                    }
                ),
                "",
            )
        path = Path(arguments[arguments.index("--kubeconfig") + 1])
        session_files.append(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["users"][0]["user"] == {"token": "private-token"}
        assert path.stat().st_mode & 0o777 == 0o600
        return subprocess.CompletedProcess(arguments, 0, json.dumps({"items": []}), "")

    kubectl = MODULE.Kubectl(
        kubeconfig=None, context="gpu", reuse_exec_credential=True, runner=runner
    )

    kubectl.run(["get", "nodes", "-o", "json"])
    kubectl.run(["get", "deployments", "-o", "json"])

    assert sum(call[:2] == ["aws", "eks"] for call in calls) == 1
    assert len(session_files) == 2
    assert all("private-token" not in " ".join(call) for call in calls), (
        "reused exec credential leaked into a subprocess argument"
    )
    assert os.environ.get("AWS_REGION") != "test"
    kubectl.close()
    assert all(not path.exists() for path in session_files), (
        "closing kubectl must remove credential session files"
    )


def test_kubectl_refuses_when_config_view_is_not_json() -> None:
    calls: list[list[str]] = []

    def runner(arguments: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    with pytest.raises(MODULE.RegistryError, match="selected kubeconfig"):
        MODULE.Kubectl(
            kubeconfig="/secure/test.kubeconfig",
            context=None,
            reuse_exec_credential=True,
            runner=runner,
        )
    assert len(calls) == 1


def test_deployment_entrypoints_refresh_runtime_registry() -> None:
    control = (
        ROOT
        / "deploy"
        / "control-plane"
        / "tools"
        / "apply-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")
    gpu = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text(
        encoding="utf-8"
    )
    bundle = (ROOT / "deploy/node/build-node-installer-bundle.sh").read_text(
        encoding="utf-8"
    )

    # The apply script runs the verifier itself and then syncs the CPU-plane
    # registry; there is no shell wrapper in between any more.
    assert "verify_control_plane_role_split.py" in control, (
        "the apply script must run the verifier directly"
    )
    assert "verify-control-plane-role-split.sh" not in control, (
        "the shell wrapper was deleted"
    )
    for text in (control, gpu, bundle):
        assert "sync_installed_resource_registry.py" in text


def test_cleanup_discovery_fails_closed_on_legacy_resources() -> None:
    found = COLLECT_MODULE.discover_unregistered(
        DiscoveryKubectl(),
        plane="gpu",
        context="gpu-context",
        namespace="gpu-fault-system",
        registered=set(),
    )

    assert {(item["kind"], item["name"]) for item in found} == {
        ("deployment", "gpu-fault-legacy-executor"),
        ("clusterrole", "gpu-fault-legacy-role"),
        ("namespace", "gf-regional-old"),
        ("role", "gpu-fault-hand-made-role"),
        ("role", "gpu-fault-cluster-executor"),
        ("rolebinding", "gpu-fault-completion-watcher"),
    }, "label-only resources must remain visible until their ownership is proven"


def test_proven_workload_rbac_uses_the_existing_context_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gpu_fault.admin.cluster_removal_rbac as rbac

    config = {"namespace": "gpu-fault-system"}
    target = {"context": "gpu-a", "cluster_id": "gpu-a"}
    proof = {
        "binding": {"context": "gpu-a"},
        "resources": {
            "gf-regional-train/Role/grant": "role-uid",
            "gf-regional-train/RoleBinding/grant": "binding-uid",
        },
        "anchors": {"namespace_uids": {"gf-regional-train": "namespace-uid"}},
        "removed": [],
        "documents": [
            {
                "kind": kind,
                "metadata": {
                    "name": "grant",
                    "namespace": "gf-regional-train",
                    "uid": uid,
                    "resourceVersion": "7",
                },
            }
            for kind, uid in (("Role", "role-uid"), ("RoleBinding", "binding-uid"))
        ],
    }
    calls: list[list[str]] = []

    class Client:
        def run(self, arguments, **kwargs):
            calls.append(arguments)
            assert kwargs["check"] is False, "proof reads must preserve error status"
            return subprocess.CompletedProcess(arguments, 0, '{"items":[]}', "")

    def inspect(actual_config, actual_target, *, run, kubectl):
        assert actual_config == config and actual_target == target, (
            "proof scope changed"
        )
        run([*kubectl, "get", "roles,rolebindings", "-A", "-o", "json"])
        return proof

    monkeypatch.setattr(rbac, "inspect_workload_namespace_rbac", inspect)
    document = {"resources": []}
    COLLECT_MODULE.attach_workload_rbac(Client(), config, target, document)
    assert calls == [["get", "roles,rolebindings", "-A", "-o", "json"]], (
        "proof must use the bound client, not a second context"
    )
    assert document["workload_rbac"] == proof, "per-context proof was not preserved"
    assert {
        (row["scope"], row["kind"], row["namespace"], row["name"])
        for row in document["resources"]
    } == {
        ("namespaced", "role", "gf-regional-train", "grant"),
        ("namespaced", "rolebinding", "gf-regional-train", "grant"),
    }, "dynamic grants did not join the existing resource inventory"
    assert all(
        row["guarded_delete"] == "workload-rbac"
        and row["clean"] == "delete"
        and MODULE.provenance_is_verified(row)
        for row in document["resources"]
    ), "ordinary deletion must not bypass the guarded lifecycle"
    assert all("uid" not in row for row in document["resources"]), (
        "cluster-specific UIDs must stay in per-context proof, not the top-level union"
    )


def test_workload_rbac_inventory_rejects_a_cross_context_proof_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gpu_fault.admin.cluster_removal_rbac as rbac

    def inspect(_config, _target, *, run, kubectl):
        assert kubectl[-1] == "gpu-a", "proof was initialized for the wrong context"
        run(["kubectl", "--context", "gpu-b", "get", "roles"])

    monkeypatch.setattr(rbac, "inspect_workload_namespace_rbac", inspect)
    with pytest.raises(COLLECT_MODULE.RegistryError, match="crossed"):
        COLLECT_MODULE.attach_workload_rbac(
            object(),
            {"namespace": "gpu-fault-system"},
            {"context": "gpu-a"},
            {"resources": []},
        )
