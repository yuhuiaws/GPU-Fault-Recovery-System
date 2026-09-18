from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

MODULE = lazy_script_module(
    Path(__file__).resolve().parents[2]
    / "deploy/control-plane/tools/cleanup_kubernetes.py"
)
NAMESPACE = "gpu-fault-system"
CONTEXT = "gpu:gpu-a"
HOST_SCRIPT = base64.b64encode(b"set -eu\nexit 0\n").decode()


def resource(
    kind: str, name: str, *, namespace: str | None = NAMESPACE
) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1"
        if kind in {"Deployment", "DaemonSet", "ReplicaSet"}
        else "v1",
        "kind": kind,
        "metadata": {
            "name": name,
            "uid": "uid-" + name,
            "resourceVersion": "1",
            **({"namespace": namespace} if namespace else {}),
        },
    }


def document() -> dict[str, Any]:
    return {
        "run_id": "cleanup-fixture",
        "mode": "reset",
        "node_mode": "uninstall",
        "targets": {
            "namespace": NAMESPACE,
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a"}],
        },
        "cluster_uids": {CONTEXT: "uid-kube-system"},
        "namespace_snapshots": {CONTEXT: {"uid": "uid-" + NAMESPACE, "objects": []}},
        "inventory_snapshot": {
            "gpu": {
                "resources": [
                    {
                        "kind": "deployment",
                        "name": "gpu-fault-collector",
                        "scope": "namespaced",
                    }
                ],
                "node_annotations": ["gpu-fault.io/installed"],
                "node_labels": ["gpu-fault.io/managed"],
            }
        },
        "fleet_snapshot": [
            {
                "cluster_id": "gpu-a",
                "node_id": "managed-node",
                "lifecycle_state": "ACTIVE",
                "installed_unit_inventory": {"units": ["gpu-fault-node-agent.service"]},
            }
        ],
    }


class Client:
    def __init__(self) -> None:
        self.nodes = [
            resource("Node", "managed-node", namespace=None),
            resource("Node", "customer-node", namespace=None),
        ]
        self.objects = {
            ("namespace", "kube-system", ""): resource(
                "Namespace", "kube-system", namespace=None
            ),
            ("namespace", NAMESPACE, ""): resource(
                "Namespace", NAMESPACE, namespace=None
            ),
        }
        self.namespaced: list[dict[str, Any]] = []
        self.calls: list[tuple[tuple[str, ...], dict[str, Any] | None]] = []
        self.lose_ack = False
        self.drift_preview = False
        self.replace_uid = False
        self.daemonset_reads = 0

    def get(self, kind: str, name: str, namespace: str = "") -> dict[str, Any] | None:
        if kind == "node":
            return copy.deepcopy(
                next(
                    (node for node in self.nodes if node["metadata"]["name"] == name),
                    None,
                )
            )
        value = self.objects.get((kind, name, namespace))
        if kind == "daemonset" and value is not None:
            self.daemonset_reads += 1
            if self.replace_uid and self.daemonset_reads > 1:
                value["metadata"]["uid"] = "replacement-uid"
        return copy.deepcopy(value)

    def items(self, *arguments: str) -> list[dict[str, Any]]:
        if "pods" in arguments:
            return copy.deepcopy(
                [item for item in self.namespaced if item.get("kind") == "Pod"]
            )
        return copy.deepcopy(self.nodes if "nodes" in arguments else self.namespaced)

    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
        self.calls.append((arguments, payload))
        if arguments[:2] == ("patch", "node"):
            node = next(
                node for node in self.nodes if node["metadata"]["name"] == arguments[2]
            )
            candidate = copy.deepcopy(node)
            for operation in json.loads(arguments[arguments.index("-p") + 1]):
                parts = [
                    part.replace("~1", "/").replace("~0", "~")
                    for part in operation["path"].split("/")[1:]
                ]
                parent = candidate
                for part in parts[:-1]:
                    parent = parent[part]
                key = parts[-1]
                if operation["op"] == "test":
                    if parent[key] != operation["value"]:
                        raise MODULE.CleanupStateError("node patch precondition failed")
                elif operation["op"] == "remove":
                    del parent[key]
                elif operation["op"] == "replace":
                    parent[key] = operation["value"]
                else:
                    raise AssertionError("unexpected node patch operation")
            node.clear()
            node.update(candidate)
        if "api-resources" in arguments:
            return "deployments.apps\npods\nsecrets\n"
        if "create" in arguments:
            assert payload is not None
            value = copy.deepcopy(payload)
            if "--dry-run=server" in arguments:
                if self.drift_preview:
                    value["spec"]["template"]["spec"]["containers"].append(
                        {"name": "injected"}
                    )
                return json.dumps(value)
            value["metadata"].update(uid="owned-uid", generation=1)
            value["status"] = {
                "observedGeneration": 1,
                "desiredNumberScheduled": 1,
                "updatedNumberScheduled": 1,
                "numberReady": 1,
            }
            self.objects[("daemonset", value["metadata"]["name"], NAMESPACE)] = value
            if self.lose_ack:
                raise MODULE.CleanupStateError("injected lost create acknowledgement")
        if "delete" in arguments:
            assert payload is not None
            assert payload["preconditions"]["uid"] in {"owned-uid", "uid-" + NAMESPACE}
            assert payload["propagationPolicy"] == "Foreground"
            name = arguments[arguments.index("--raw") + 1].rsplit("/", 1)[1]
            self.objects.pop(("daemonset", name, NAMESPACE), None)
            self.objects.pop(("namespace", name, ""), None)
        return ""

    def mutations(self) -> list[tuple[str, ...]]:
        return [
            args
            for args, _ in self.calls
            if args[0] in {"create", "delete", "patch"}
            and "--dry-run=server" not in args
        ]


def test_node_cleanup_targets_only_fleet_nodes_and_rejects_uid_replacement() -> None:
    client, state = Client(), document()
    assert MODULE.node_targets(client, state, CONTEXT, "gpu-a") == {
        "managed-node": "uid-managed-node"
    }
    client.nodes[0]["metadata"]["uid"] = "replacement"
    with pytest.raises(MODULE.CleanupStateError, match="changed UID"):
        MODULE.node_targets(client, state, CONTEXT, "gpu-a")
    assert client.mutations() == []


def test_unknown_node_runtime_is_not_silently_excluded() -> None:
    client, state = Client(), document()
    client.nodes[1]["metadata"]["labels"] = {"gpu-fault.io/managed": "true"}
    with pytest.raises(MODULE.CleanupStateError, match="unregistered node runtime"):
        MODULE.node_targets(client, state, CONTEXT, "gpu-a")


def _parked_spare(previous: str) -> tuple[Client, dict[str, Any]]:
    client, state = Client(), document()
    node = client.nodes[0]
    node["spec"] = {"unschedulable": True}
    node["metadata"]["labels"] = {
        "gpu-fault.io/spare": "true",
        "customer.example/retained": "yes",
    }
    node["metadata"]["annotations"] = {"gpu-fault.io/previous-unschedulable": previous}
    inventory = state["inventory_snapshot"]["gpu"]
    inventory["node_labels"].append("gpu-fault.io/spare")
    inventory["node_annotations"].append("gpu-fault.io/previous-unschedulable")
    return client, state


@pytest.mark.parametrize("previous,expected", [("false", False), ("true", True)])
def test_spare_cleanup_restores_only_its_recorded_cordon(
    previous: str, expected: bool
) -> None:
    client, state = _parked_spare(previous)
    untouched = copy.deepcopy(client.nodes[1])

    MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")

    node = client.nodes[0]
    assert node["spec"]["unschedulable"] is expected, (
        "operator cordon was not preserved"
    )
    assert node["metadata"]["labels"] == {"customer.example/retained": "yes"}, (
        "owned labels must be removed without changing customer labels"
    )
    assert not node["metadata"]["annotations"], "spare baseline annotation remains"
    assert client.nodes[1] == untouched, "cleanup changed an unrelated node"
    operations = json.loads(client.mutations()[0][-1])
    assert operations[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-managed-node"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "1"},
    ], "spare restoration lacks UID/resourceVersion preconditions"
    if previous == "false":
        restore = next(
            i
            for i, operation in enumerate(operations)
            if operation["op"] == "replace"
            and operation["path"] == "/spec/unschedulable"
        )
        remove_label = next(
            i
            for i, operation in enumerate(operations)
            if operation["path"] == "/metadata/labels/gpu-fault.io~1spare"
        )
        assert restore < remove_label, "cordon ownership was removed before restoration"

    before = len(client.mutations())
    MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert len(client.mutations()) == before, (
        "completed metadata cleanup was not idempotent"
    )


@pytest.mark.parametrize("previous", [None, "", "unknown", False, {"invalid": True}])
def test_spare_cleanup_refuses_unknown_baselines(previous: Any) -> None:
    client, state = _parked_spare("false")
    if previous is None:
        client.nodes[0]["metadata"]["annotations"].clear()
    else:
        client.nodes[0]["metadata"]["annotations"][
            "gpu-fault.io/previous-unschedulable"
        ] = previous
    before = copy.deepcopy(client.nodes)
    with pytest.raises(MODULE.CleanupStateError, match="scheduling baseline"):
        MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert client.nodes == before, "unknown ownership changed node metadata"
    assert not client.mutations(), "unknown scheduling baseline authorized mutation"


def test_spare_cleanup_refuses_a_new_quarantine() -> None:
    client, state = _parked_spare("false")
    client.nodes[0]["metadata"]["labels"]["gpu-fault.io/quarantined"] = "true"
    with pytest.raises(MODULE.CleanupStateError, match="quarantined spare"):
        MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    assert not client.mutations(), "a quarantined spare was released"


def test_spare_cleanup_cas_refuses_a_late_node_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, state = _parked_spare("false")
    original = client.run

    def race(*arguments: str, **kwargs: Any) -> str:
        if arguments[:2] == ("patch", "node"):
            client.nodes[0]["metadata"]["resourceVersion"] = "2"
        return original(*arguments, **kwargs)

    monkeypatch.setattr(client, "run", race)
    with pytest.raises(MODULE.CleanupStateError, match="precondition failed"):
        MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    node = client.nodes[0]
    assert node["spec"]["unschedulable"] is True, "stale cleanup removed the cordon"
    assert node["metadata"]["labels"]["gpu-fault.io/spare"] == "true", (
        "stale cleanup removed the ownership label"
    )


def test_spare_cleanup_readback_cannot_adopt_a_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, state = _parked_spare("false")
    original = client.get

    def replacement(kind: str, name: str, namespace: str = "") -> dict[str, Any] | None:
        value = original(kind, name, namespace)
        if kind == "node" and value is not None:
            value["metadata"]["uid"] = "replacement"
        return value

    monkeypatch.setattr(client, "get", replacement)
    with pytest.raises(MODULE.CleanupStateError, match="node changed"):
        MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")


def test_failed_host_cleanup_never_creates_the_readiness_marker(tmp_path: Path) -> None:
    manifest = MODULE.node_manifest(
        name="owned",
        namespace=NAMESPACE,
        run_id="fixture",
        nodes=["managed-node"],
        image="fixture/image",
        mode="uninstall",
        host_script_b64=HOST_SCRIPT,
    )
    pod = manifest["spec"]["template"]["spec"]
    assert pod["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchFields"][0]["values"] == ["managed-node"]
    marker = tmp_path / "late-marker"
    for name, text in {
        "chroot": "#!/bin/sh\nexit 19\n",
        "touch": f"#!/bin/sh\nprintf late > '{marker}'\n",
        "sleep": "#!/bin/sh\nexit 0\n",
    }.items():
        executable = tmp_path / name
        executable.write_text(text)
        executable.chmod(0o700)
    container = pod["containers"][0]
    result = subprocess.run(
        [*container["command"], *container["args"]],
        env={**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 19
    assert not marker.exists(), "failed host cleanup must not create a readiness marker"


@pytest.mark.parametrize("lose_ack", [False, True])
def test_owned_node_cleanup_is_foreground_removed_even_after_lost_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lose_ack: bool
) -> None:
    client, state = Client(), document()
    client.lose_ack = lose_ack
    monkeypatch.setattr(MODULE, "CleanupClient", lambda _config, _context: client)
    arguments = {
        "context": CONTEXT,
        "cluster_id": "gpu-a",
        "image": "fixture/image",
        "mode": "uninstall",
        "host_script_b64": HOST_SCRIPT,
    }
    if lose_ack:
        with pytest.raises(
            MODULE.CleanupStateError, match="lost create acknowledgement"
        ):
            MODULE.run_node_cleanup({}, state, tmp_path / "state.json", **arguments)
    else:
        MODULE.run_node_cleanup({}, state, tmp_path / "state.json", **arguments)
    assert state["node_cleanup"][CONTEXT]["status"] == "REMOVED"
    assert not any(key[0] == "daemonset" for key in client.objects), (
        "owned cleanup DaemonSet must be removed even after a lost ACK"
    )
    assert [args[0] for args in client.mutations()] == ["create", "delete"]


@pytest.mark.parametrize("preview", [True, False])
def test_admission_or_uid_drift_never_deletes_an_unowned_daemonset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preview: bool
) -> None:
    client, state = Client(), document()
    client.drift_preview = preview
    client.replace_uid = not preview
    monkeypatch.setattr(MODULE, "CleanupClient", lambda _config, _context: client)
    with pytest.raises(MODULE.CleanupStateError, match="spec differs|was replaced"):
        MODULE.run_node_cleanup(
            {},
            state,
            tmp_path / "state.json",
            context=CONTEXT,
            cluster_id="gpu-a",
            image="fixture/image",
            mode="uninstall",
            host_script_b64=HOST_SCRIPT,
        )
    assert not any(args[0] == "delete" for args in client.mutations()), (
        "admission or UID drift must not delete an unowned DaemonSet"
    )
    if preview:
        assert client.mutations() == []


def test_namespace_capture_refuses_customer_workloads_before_any_mutation() -> None:
    client, state = Client(), document()
    client.namespaced = [resource("Deployment", "customer-training")]
    with pytest.raises(MODULE.CleanupStateError, match="unowned resources"):
        MODULE.capture_namespace(client, state, CONTEXT)
    assert client.mutations() == []


def test_namespace_children_require_exact_parent_uid() -> None:
    client, state = Client(), document()
    parent = resource("Deployment", "gpu-fault-collector")
    child = resource("ReplicaSet", "gpu-fault-collector-1")
    child["metadata"]["ownerReferences"] = [
        {
            "apiVersion": parent["apiVersion"],
            "kind": parent["kind"],
            "name": parent["metadata"]["name"],
            "uid": parent["metadata"]["uid"],
        }
    ]
    client.namespaced = [parent, child]
    MODULE.capture_namespace(client, state, CONTEXT)
    child["metadata"]["ownerReferences"][0]["uid"] = "previous-parent"
    with pytest.raises(MODULE.CleanupStateError, match="unowned resources"):
        MODULE.capture_namespace(client, state, CONTEXT)


@pytest.mark.parametrize("replace_namespace", [False, True])
def test_namespace_deletion_rechecks_ownership_and_namespace_uid(
    replace_namespace: bool,
) -> None:
    client, state = Client(), document()
    MODULE.capture_namespace(client, state, CONTEXT)
    if replace_namespace:
        client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "replacement"
    else:
        client.namespaced.append(resource("Pod", "customer-job"))
    with pytest.raises(MODULE.CleanupStateError, match="replaced|unowned resources"):
        MODULE.delete_namespace(client, state, CONTEXT)
    assert client.mutations() == []


def test_node_metadata_removal_is_uid_and_resource_version_conditional() -> None:
    client, state = Client(), document()
    client.nodes[0]["metadata"]["annotations"] = {"gpu-fault.io/installed": "yes"}
    MODULE.clear_node_metadata(client, state, CONTEXT, "gpu-a")
    calls = client.mutations()
    assert len(calls) == 1
    assert calls[0][:3] == ("patch", "node", "managed-node")
    operations = json.loads(calls[0][-1])
    assert operations[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-managed-node"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "1"},
    ]
    assert operations[2]["path"] == "/metadata/annotations/gpu-fault.io~1installed"


def test_partial_cleanup_checkpoint_does_not_authorize_stopping_restarted_workloads() -> (
    None
):
    client, state = Client(), document()
    state["scope"] = "all"
    state["history"] = [
        {"phase": "GPU_DATA_PLANE_SOURCES_STOPPED", "status": "COMPLETED"}
    ]
    state["inventory_snapshot"]["gpu"]["resources"][0]["phase"] = "producer"
    restarted = resource("Deployment", "gpu-fault-collector")
    restarted["spec"] = {"replicas": 1}
    client.objects[("deployment", "gpu-fault-collector", NAMESPACE)] = restarted
    with pytest.raises(MODULE.CleanupStateError, match="reappeared or restarted"):
        MODULE.verify_quiesced(client, state, CONTEXT)
    assert client.mutations() == []


def test_completed_namespace_checkpoint_is_not_permission_to_delete_a_new_namespace() -> (
    None
):
    client, state = Client(), document()
    state["history"] = [{"phase": "NAMESPACES_DELETED", "status": "COMPLETED"}]
    with pytest.raises(MODULE.CleanupStateError, match="namespace has reappeared"):
        MODULE.verify_quiesced(client, state, CONTEXT)
    assert client.mutations() == []


def test_a_gpu_context_named_cpu_uses_only_the_gpu_kubeconfig() -> None:
    config = {"cpu_kubeconfig": "/private/cpu", "gpu_kubeconfig": "/private/gpu"}
    client = MODULE.CleanupClient(config, "gpu:cpu")
    assert client.prefix == [
        "kubectl",
        "--request-timeout=30s",
        "--kubeconfig",
        "/private/gpu",
        "--context",
        "cpu",
    ]


def test_distinct_cpu_gpu_targets_cannot_alias_the_same_live_cluster() -> None:
    client, state = Client(), document()
    state["cluster_uids"] = {"cpu": "uid-kube-system"}
    with pytest.raises(MODULE.CleanupStateError, match="alias the same"):
        MODULE.verify_cluster(client, state, "gpu:cpu")
    assert client.mutations() == []


def test_partial_cleanup_rejects_a_recreated_namespace_before_resuming() -> None:
    client, state = Client(), document()
    state["scope"] = "all"
    state["history"] = [{"phase": "PREFLIGHT", "status": "COMPLETED"}]
    MODULE.capture_namespace(client, state, CONTEXT)
    client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "replacement"
    with pytest.raises(MODULE.CleanupStateError, match="namespace.*replaced"):
        MODULE.verify_quiesced(client, state, CONTEXT)
    assert client.mutations() == [], (
        "a partial checkpoint must not authorize cleanup in a recreated namespace"
    )


def test_namespace_recapture_cannot_replace_the_original_uid_binding() -> None:
    client, state = Client(), document()
    MODULE.capture_namespace(client, state, CONTEXT)
    original = copy.deepcopy(state["namespace_snapshots"])
    client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "replacement"
    with pytest.raises(MODULE.CleanupStateError, match="namespace.*replaced"):
        MODULE.capture_namespace(client, state, CONTEXT)
    assert state["namespace_snapshots"] == original, (
        "a retried preflight must preserve the namespace incarnation it first observed"
    )


def test_node_cleanup_rejects_namespace_replacement_before_privileged_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, state = Client(), document()
    MODULE.capture_namespace(client, state, CONTEXT)
    client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "replacement"
    monkeypatch.setattr(MODULE, "CleanupClient", lambda _config, _context: client)
    with pytest.raises(MODULE.CleanupStateError, match="namespace.*replaced"):
        MODULE.run_node_cleanup(
            {},
            state,
            tmp_path / "state.json",
            context=CONTEXT,
            cluster_id="gpu-a",
            image="fixture/image",
            mode="uninstall",
            host_script_b64=HOST_SCRIPT,
        )
    assert client.mutations() == [], (
        "namespace replacement must block the privileged DaemonSet before creation"
    )


def test_pending_daemonset_cleanup_cannot_cross_a_namespace_reincarnation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, state = Client(), document()
    manifest = MODULE.node_manifest(
        name="gpu-fault-node-cleanup-fixture",
        namespace=NAMESPACE,
        run_id=state["run_id"],
        nodes=["managed-node"],
        image="fixture/image",
        mode="uninstall",
        host_script_b64=HOST_SCRIPT,
    )
    live = copy.deepcopy(manifest)
    live["metadata"]["uid"] = "owned-uid"
    client.objects[("daemonset", manifest["metadata"]["name"], NAMESPACE)] = live
    client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "replacement"
    state["node_cleanup"] = {
        CONTEXT: {
            "name": manifest["metadata"]["name"],
            "uid": None,
            "manifest": manifest,
            "status": "PLANNED",
        }
    }
    monkeypatch.setattr(MODULE, "CleanupClient", lambda _config, _context: client)
    with pytest.raises(MODULE.CleanupStateError, match="namespace.*replaced"):
        MODULE.cleanup_pending({}, state, tmp_path / "state.json")
    assert client.mutations() == [], (
        "a lost create ACK must not authorize deletion in a replacement namespace"
    )
    assert state["node_cleanup"][CONTEXT]["status"] == "PLANNED", (
        "uncertain temporary cleanup must remain pending for reconciliation"
    )


def test_clean_mode_still_captures_the_namespace_incarnation() -> None:
    client, state = Client(), document()
    state["mode"] = "clean"
    state["scope"] = "gpu"
    state["history"] = [{"phase": "PREFLIGHT", "status": "COMPLETED"}]
    state.pop("namespace_snapshots")
    MODULE.capture_namespace(client, state, CONTEXT)
    client.objects[("namespace", NAMESPACE, "")]["metadata"]["uid"] = "replacement"
    with pytest.raises(MODULE.CleanupStateError, match="namespace.*replaced"):
        MODULE.verify_quiesced(client, state, CONTEXT)
    assert client.mutations() == [], "clean mode must preserve the namespace UID guard"
