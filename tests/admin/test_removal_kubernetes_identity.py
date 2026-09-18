from __future__ import annotations

import copy
import json
import subprocess
import threading
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_kubernetes import (
    clear_installer_annotations,
    request_namespace_deletion,
    wait_namespace_absent,
)

KUBECTL = ["kubectl", "--context", "fixture-gpu"]
NAMESPACE = "gpu-fault-system"
ANNOTATION = "gpu-fault.io/installer-state"


class NamespaceApi:
    def __init__(self) -> None:
        self.uid: str | None = "original-namespace"
        self.before_delete_uid: str | None = None
        self.delete_status = 0
        self.read_status = 0
        self.delete_applied = True
        self.fail_read_after_delete = False
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((arguments, kwargs))
        if "get" in arguments:
            document = (
                {"kind": "Namespace", "metadata": {"name": NAMESPACE, "uid": self.uid}}
                if self.uid
                else None
            )
            return subprocess.CompletedProcess(
                arguments,
                self.read_status,
                json.dumps(document) if document is not None else "",
                "private-fixture-error" if self.read_status else "",
            )
        assert arguments[len(KUBECTL) :][:2] == ["delete", "--raw"]
        options = json.loads(kwargs["input_text"])
        if self.before_delete_uid:
            self.uid = self.before_delete_uid
        if options["preconditions"]["uid"] != self.uid:
            return subprocess.CompletedProcess(arguments, 1, "", "uid conflict")
        if self.delete_applied:
            self.uid = None
        if self.fail_read_after_delete:
            self.read_status = 1
        return subprocess.CompletedProcess(arguments, self.delete_status, "", "")


def test_namespace_deletion_is_bound_to_the_saved_uid() -> None:
    api = NamespaceApi()
    request_namespace_deletion(api, KUBECTL, NAMESPACE, "original-namespace")
    assert api.uid is None
    options = json.loads(api.calls[1][1]["input_text"])
    assert options["preconditions"] == {"uid": "original-namespace"}
    assert options["propagationPolicy"] == "Foreground"


@pytest.mark.parametrize("changed_before_read", [False, True])
def test_recreated_namespace_is_never_deleted(changed_before_read: bool) -> None:
    api = NamespaceApi()
    if changed_before_read:
        api.uid = "replacement-namespace"
    else:
        api.before_delete_uid = "replacement-namespace"
    with pytest.raises(BootstrapError):
        request_namespace_deletion(api, KUBECTL, NAMESPACE, "original-namespace")
    assert api.uid == "replacement-namespace"
    assert sum("delete" in call for call, _ in api.calls) == int(
        not changed_before_read
    )


def test_namespace_delete_lost_ack_needs_a_successful_absence_read() -> None:
    api = NamespaceApi()
    api.delete_status = 1
    request_namespace_deletion(api, KUBECTL, NAMESPACE, "original-namespace")
    assert len(api.calls) == 3
    assert api.uid is None


def test_namespace_read_error_cannot_authorize_deletion() -> None:
    api = NamespaceApi()
    api.read_status = 1
    with pytest.raises(BootstrapError, match="cannot read namespace") as rejected:
        request_namespace_deletion(api, KUBECTL, NAMESPACE, "original-namespace")
    assert len(api.calls) == 1
    assert "private-fixture-error" not in str(rejected.value)


@pytest.mark.parametrize("failed_recheck", [False, True])
def test_failed_delete_does_not_imply_absence(failed_recheck: bool) -> None:
    api = NamespaceApi()
    api.delete_status = 1
    api.delete_applied = False
    api.fail_read_after_delete = failed_recheck
    with pytest.raises(BootstrapError):
        request_namespace_deletion(api, KUBECTL, NAMESPACE, "original-namespace")
    assert api.uid == "original-namespace"


@pytest.mark.parametrize("expected", [None, "", 123])
def test_namespace_cleanup_requires_prior_identity(expected: object) -> None:
    api = NamespaceApi()
    with pytest.raises(BootstrapError, match="proven UID"):
        request_namespace_deletion(api, KUBECTL, NAMESPACE, expected)
    assert api.calls == []


def test_namespace_wait_rejects_a_recreated_object() -> None:
    api = NamespaceApi()
    api.uid = "replacement-namespace"
    with pytest.raises(BootstrapError, match="recreated during deletion"):
        wait_namespace_absent(
            api, KUBECTL, NAMESPACE, "original-namespace", interval_seconds=0
        )


def test_namespace_wait_accepts_only_proven_absence() -> None:
    api = NamespaceApi()
    api.uid = None
    wait_namespace_absent(api, KUBECTL, NAMESPACE, "original-namespace")
    assert len(api.calls) == 1
    api.read_status = 1
    with pytest.raises(BootstrapError, match="cannot read namespace"):
        wait_namespace_absent(api, KUBECTL, NAMESPACE, "original-namespace")


class NodeApi:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {
            "node-a": {
                "kind": "Node",
                "metadata": {
                    "name": "node-a",
                    "uid": "original-node",
                    "resourceVersion": "version-a",
                    "labels": {"sagemaker.amazonaws.com/cluster-name": "gpu-fixture"},
                    "annotations": {
                        ANNOTATION: "installed",
                        "customer.example/keep": "unchanged",
                    },
                },
            }
        }
        self.before_patch_uid: str | None = None
        self.before_patch_version: str | None = None
        self.calls: list[list[str]] = []

    def __call__(
        self, arguments: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments)
        if "get" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps({"items": list(self.nodes.values())}), ""
            )
        assert "patch" in arguments and "--type=json" in arguments
        patch = json.loads(arguments[arguments.index("-p") + 1])
        name = arguments[arguments.index("node") + 1]
        metadata = self.nodes[name]["metadata"]
        if self.before_patch_uid:
            metadata["uid"] = self.before_patch_uid
        if self.before_patch_version:
            metadata["resourceVersion"] = self.before_patch_version
        for operation in patch:
            if operation["op"] == "test":
                key = operation["path"].split("/")[-1]
                if metadata[key] != operation["value"]:
                    return subprocess.CompletedProcess(arguments, 1, "", "conflict")
        for operation in patch:
            if operation["op"] == "remove":
                key = (
                    operation["path"]
                    .split("/")[-1]
                    .replace("~1", "/")
                    .replace("~0", "~")
                )
                metadata["annotations"].pop(key)
        return subprocess.CompletedProcess(arguments, 0, "", "")


def clear(api: NodeApi) -> None:
    clear_installer_annotations(
        api,
        KUBECTL,
        hyperpod_name="gpu-fixture",
        node_uids={"node-a": "original-node"},
        annotations=[ANNOTATION],
    )


def test_node_cleanup_preserves_unowned_annotations() -> None:
    api = NodeApi()
    clear(api)
    assert api.nodes["node-a"]["metadata"]["annotations"] == {
        "customer.example/keep": "unchanged"
    }
    assert len(api.calls) == 2
    patch = json.loads(api.calls[1][api.calls[1].index("-p") + 1])
    assert patch[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "original-node"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "version-a"},
    ]


@pytest.mark.parametrize("mutation", ["uid", "resourceVersion"])
def test_node_recreation_or_concurrent_change_cannot_be_cleared(mutation: str) -> None:
    api = NodeApi()
    if mutation == "uid":
        api.before_patch_uid = "replacement-node"
    else:
        api.before_patch_version = "version-b"
    with pytest.raises(BootstrapError, match="guarded patch"):
        clear(api)
    assert api.nodes["node-a"]["metadata"]["annotations"][ANNOTATION] == "installed"


@pytest.mark.parametrize("change", ["missing", "uid", "label", "extra"])
def test_all_node_identities_are_validated_before_any_patch(change: str) -> None:
    api = NodeApi()
    if change == "missing":
        api.nodes.clear()
    elif change == "extra":
        extra = copy.deepcopy(api.nodes["node-a"])
        extra["metadata"]["name"] = "unexpected-node"
        api.nodes["unexpected-node"] = extra
    elif change == "uid":
        api.nodes["node-a"]["metadata"]["uid"] = "replacement-node"
    else:
        api.nodes["node-a"]["metadata"]["labels"] = {}
    with pytest.raises(BootstrapError, match="identity drifted"):
        clear(api)
    assert len(api.calls) == 1


@pytest.mark.parametrize("annotations", [None, {}])
def test_no_annotation_to_remove_is_a_verified_noop(annotations: object) -> None:
    api = NodeApi()
    api.nodes["node-a"]["metadata"]["annotations"] = annotations
    clear(api)
    assert len(api.calls) == 1


def test_fatal_cleanup_failure_is_not_hidden_by_an_ordinary_failure() -> None:
    class FatalCleanup(BaseException):
        pass

    api = NodeApi()
    second = copy.deepcopy(api.nodes["node-a"])
    second["metadata"].update({"name": "node-b", "uid": "second-node"})
    api.nodes["node-b"] = second
    started = threading.Barrier(2)

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "get" in arguments:
            return api(arguments, **kwargs)
        started.wait(timeout=5)
        if "node-a" in arguments:
            raise BootstrapError("ordinary failure")
        raise FatalCleanup("completion proof lost")

    with pytest.raises(FatalCleanup, match="completion proof lost"):
        clear_installer_annotations(
            run,
            KUBECTL,
            hyperpod_name="gpu-fixture",
            node_uids={"node-a": "original-node", "node-b": "second-node"},
            annotations=[ANNOTATION],
        )
