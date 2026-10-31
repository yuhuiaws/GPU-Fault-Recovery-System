"""remove-cluster's and the join rollback's installer annotation cleanup under node races.

``clear_installer_annotations`` lists the target cluster's nodes once, proves
every identity, then patches each node bound to the UID and resourceVersion of
that list. ``remove-cluster`` runs it right after its node DaemonSets are deleted
-- the moment the kubelet posts Node status -- so it had the same exposure as the
reset's node metadata cleanup (BOOT-029 stage 4): one moved resourceVersion
stopped the whole removal with "node annotation cleanup failed its guarded
patch" and no reason. These cases drive it against a scripted API server that
applies JSON patches the way the real one does, with other writers (a kubelet,
a replacement, an operator) acting between the cleanup's read and its patch.
The nodes are patched in parallel, so the server serialises its requests.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from typing import Any

import pytest

from gpu_fault.admin import node_patch_retry as RETRY
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_kubernetes import (
    NodePatchCommandFailed,
    clear_installer_annotations,
)
from gpu_fault.admin.execution import deadline_scope
from tests.regional.test_cleanup_kubernetes import JSON_PATCH_TEST_FAILED

KUBECTL = ["kubectl", "--kubeconfig", "/fixture/gpu.kubeconfig", "--context", "gpu-a"]
CLUSTER_LABEL = {"sagemaker.amazonaws.com/cluster-name": "gpu-fixture"}
INSTALLED = {
    "gpu-fault.io/installer-state": "Succeeded",
    "gpu-fault.io/installer-version": "1.0.0",
}
ANNOTATIONS = tuple(INSTALLED)
FORBIDDEN = (
    'Error from server (Forbidden): nodes "node-a" is forbidden: User '
    '"system:serviceaccount:ops:remover" cannot patch resource "nodes" in API '
    'group "" at the cluster scope'
)


def bump(node: dict[str, Any]) -> None:
    metadata = node["metadata"]
    metadata["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)


class ApiServer:
    """Node list/get/patch of one GPU cluster, one request at a time.

    ``writers[name]`` holds the writes that land on the node right before each
    successive patch of it; ``refusals[name]`` is the kubectl stderr of a patch
    the server refuses for a reason of its own; a node in ``lost`` has its patch
    applied while kubectl runs out of time; ``reread_failure`` makes every
    single-node read fail with that stderr.
    """

    def __init__(self, *names: str) -> None:
        self.lock = threading.Lock()
        self.nodes: dict[str, dict[str, Any]] = {}
        self.writers: dict[str, list[Writer]] = {}
        self.refusals: dict[str, str] = {}
        self.lost: set[str] = set()
        self.reread_failure = ""
        self.calls: list[list[str]] = []
        for name in names:
            self.nodes[name] = {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {
                    "name": name,
                    "uid": f"uid-{name}",
                    "resourceVersion": "1",
                    "labels": dict(CLUSTER_LABEL),
                    "annotations": {**INSTALLED, "customer.example/keep": "yes"},
                },
            }

    def __call__(
        self, arguments: list[str], **_options: Any
    ) -> subprocess.CompletedProcess[str]:
        with self.lock:
            self.calls.append(list(arguments))
            command = arguments[len(KUBECTL) :]
            if command[:2] == ["get", "nodes"]:
                listing = {"kind": "List", "items": list(self.nodes.values())}
                return self.answer(arguments, 0, json.dumps(listing))
            if command[:2] == ["get", "node"]:
                if self.reread_failure:
                    return self.answer(arguments, 1, stderr=self.reread_failure)
                node = self.nodes.get(command[2])
                return self.answer(arguments, 0, json.dumps(node) if node else "")
            if command[:2] == ["patch", "node"]:
                return self.patch(arguments, command)
        raise AssertionError(f"unexpected kubectl call {command}")

    @staticmethod
    def answer(
        arguments: list[str], code: int, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, code, stdout, stderr)

    def patch(
        self, arguments: list[str], command: list[str]
    ) -> subprocess.CompletedProcess[str]:
        name = command[2]
        pending = self.writers.get(name) or []
        if pending:
            pending.pop(0)(self, name)
        if name in self.refusals:
            return self.answer(arguments, 1, stderr=self.refusals[name])
        node = self.nodes.get(name)
        if node is None:
            missing = f'Error from server (NotFound): nodes "{name}" not found'
            return self.answer(arguments, 1, stderr=missing)
        metadata = node["metadata"]
        operations = json.loads(command[command.index("-p") + 1])
        for operation in operations:
            if operation["op"] == "test":
                key = operation["path"].rsplit("/", 1)[1]
                if metadata.get(key) != operation["value"]:
                    return self.answer(arguments, 1, stderr=JSON_PATCH_TEST_FAILED)
        for operation in operations:
            if operation["op"] == "remove":
                key = operation["path"].rsplit("/", 1)[1]
                del metadata["annotations"][key.replace("~1", "/").replace("~0", "~")]
        bump(node)
        if name in self.lost:
            raise subprocess.TimeoutExpired(arguments, 30)
        return self.answer(arguments, 0)

    def patches(self, name: str) -> list[list[dict[str, Any]]]:
        return [
            json.loads(call[call.index("-p") + 1])
            for call in self.calls
            if call[len(KUBECTL) :][:3] == ["patch", "node", name]
        ]

    def annotations(self, name: str) -> dict[str, str]:
        return dict(self.nodes[name]["metadata"]["annotations"])


Writer = Callable[[ApiServer, str], None]


def kubelet_status(server: ApiServer, name: str) -> None:
    """The kubelet posts Node status: only status and resourceVersion move."""

    node = server.nodes[name]
    node.setdefault("status", {})["conditions"] = [{"type": "Ready", "status": "True"}]
    bump(node)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(RETRY.time, "sleep", recorded.append)
    return recorded


def clear(server: ApiServer, *names: str) -> None:
    clear_installer_annotations(
        server,
        KUBECTL,
        hyperpod_name="gpu-fixture",
        node_uids={name: f"uid-{name}" for name in names or tuple(server.nodes)},
        annotations=ANNOTATIONS,
    )


def versions(patches: list[list[dict[str, Any]]]) -> list[str]:
    return [patch[1]["value"] for patch in patches]


def test_a_kubelet_status_update_between_list_and_patch_is_retried_on_the_fresh_node(
    sleeps: list[float],
) -> None:
    server = ApiServer("node-a")
    server.writers["node-a"] = [kubelet_status]

    clear(server)

    first, second = server.patches("node-a")
    assert versions([first, second]) == ["1", "2"], (
        "the retry must be bound to the resourceVersion of a fresh read"
    )
    assert second[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "uid-node-a",
    }, "the retry keeps the UID precondition"
    assert second[2:] == first[2:], "the fresh Node still carried both annotations"
    assert server.annotations("node-a") == {"customer.example/keep": "yes"}, (
        "only the installer annotations are removed"
    )
    assert sleeps == [RETRY.NODE_PATCH_BACKOFF_SECONDS[0]], (
        "one short back-off before the fresh read"
    )


def test_the_other_nodes_are_still_cleaned_when_one_raced(sleeps: list[float]) -> None:
    server = ApiServer("node-a", "node-b", "node-c")
    server.writers["node-b"] = [kubelet_status]

    clear(server)

    assert {name: len(server.patches(name)) for name in server.nodes} == {
        "node-a": 1,
        "node-b": 2,
        "node-c": 1,
    }, "only the raced node is patched twice"
    for name in server.nodes:
        assert server.annotations(name) == {"customer.example/keep": "yes"}, (
            f"{name} kept an installer annotation"
        )


def test_a_node_replaced_during_the_race_stays_a_hard_error(
    sleeps: list[float],
) -> None:
    server = ApiServer("node-a")

    def replace(api: ApiServer, name: str) -> None:
        api.nodes[name]["metadata"]["uid"] = "uid-replacement"
        bump(api.nodes[name])

    server.writers["node-a"] = [replace]
    with pytest.raises(
        BootstrapError,
        match="node annotation cleanup target changed UID: node-a was replaced",
    ):
        clear(server)
    assert len(server.patches("node-a")) == 1, "nothing is sent to a replacement"
    assert "gpu-fault.io/installer-state" in server.annotations("node-a"), (
        "the replacement's annotations are not ours to remove"
    )


def test_a_node_deleted_during_the_race_stays_a_hard_error(sleeps: list[float]) -> None:
    server = ApiServer("node-a")

    def delete(api: ApiServer, name: str) -> None:
        del api.nodes[name]

    server.writers["node-a"] = [delete]
    with pytest.raises(
        BootstrapError, match="node annotation cleanup target disappeared: node-a"
    ) as raised:
        clear(server)
    assert isinstance(raised.value.__cause__, NodePatchCommandFailed), (
        "the refused patch is the cause"
    )
    assert raised.value.__cause__.failure.reason == "not found", (
        "kubectl's NotFound is classified"
    )
    assert len(server.patches("node-a")) == 1, "a vanished Node is not patched again"


def test_a_node_that_left_the_cluster_selector_during_the_race_is_refused(
    sleeps: list[float],
) -> None:
    server = ApiServer("node-a")

    def relabel(api: ApiServer, name: str) -> None:
        api.nodes[name]["metadata"]["labels"] = {}
        bump(api.nodes[name])

    server.writers["node-a"] = [relabel]
    with pytest.raises(BootstrapError, match="identity drifted during annotation"):
        clear(server)
    assert len(server.patches("node-a")) == 1, "a node outside the scope is left alone"


def test_a_refused_patch_of_an_unchanged_node_fails_with_kubectls_reason(
    sleeps: list[float],
) -> None:
    server = ApiServer("node-a")
    server.refusals["node-a"] = FORBIDDEN

    with pytest.raises(NodePatchCommandFailed) as raised:
        clear(server)

    assert str(raised.value) == (
        "node annotation cleanup failed its guarded patch (exit 1, reason: "
        f"forbidden): kubectl patch node node-a: stderr: {FORBIDDEN}"
    ), str(raised.value)
    assert raised.value.failure.reason == "forbidden", "the reason class is kept"
    assert any("did not move" in note for note in raised.value.__notes__), (
        "the failure says why it was not retried"
    )
    assert len(server.patches("node-a")) == 1, (
        "a failure no concurrent update explains is not retried"
    )
    assert "gpu-fault.io/installer-state" in server.annotations("node-a"), (
        "a refused patch changes nothing"
    )


def test_a_node_that_keeps_changing_fails_after_bounded_attempts_naming_node_and_reason(
    sleeps: list[float],
) -> None:
    server = ApiServer("node-a")
    server.writers["node-a"] = [kubelet_status] * RETRY.NODE_PATCH_ATTEMPTS

    with pytest.raises(BootstrapError) as raised:
        clear(server)

    message = str(raised.value)
    assert message.startswith(
        "node annotation cleanup of node-a did not settle: the Node's resourceVersion"
        f" moved before each of {RETRY.NODE_PATCH_ATTEMPTS} patch attempts"
    ), message
    assert (
        "(exit 1, reason: invalid): kubectl patch node node-a: stderr: " in message
    ), "the last failure's reason is named"
    assert isinstance(raised.value.__cause__, NodePatchCommandFailed), (
        "the last refused patch is the cause"
    )
    assert versions(server.patches("node-a")) == ["1", "2", "3", "4"], (
        "every attempt is bound to the version its own fresh read returned"
    )
    assert sleeps == list(RETRY.NODE_PATCH_BACKOFF_SECONDS), (
        "the back-off is short, bounded, and none follows the last attempt"
    )
    assert "gpu-fault.io/installer-state" in server.annotations("node-a"), (
        "nothing is removed from a Node that never held still"
    )


def test_a_patch_applied_despite_a_lost_acknowledgement_is_not_sent_again(
    sleeps: list[float],
) -> None:
    server = ApiServer("node-a")
    server.lost.add("node-a")

    clear(server)

    assert len(server.patches("node-a")) == 1, (
        "the fresh read shows nothing left to remove: no second patch"
    )
    assert server.annotations("node-a") == {"customer.example/keep": "yes"}, (
        "the applied patch stands"
    )


def test_a_patch_that_ran_out_of_time_on_an_unchanged_node_is_a_timeout(
    sleeps: list[float],
) -> None:
    class Expiring(ApiServer):
        def patch(
            self, arguments: list[str], command: list[str]
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(arguments, 30)

    server = Expiring("node-a")
    with pytest.raises(NodePatchCommandFailed) as raised:
        clear(server)
    assert (raised.value.failure.reason, raised.value.failure.returncode) == (
        "timeout",
        None,
    ), str(raised.value)
    assert "(no exit status, reason: timeout): kubectl patch node node-a: " in str(
        raised.value
    ), str(raised.value)


def test_a_fresh_read_that_fails_says_why(sleeps: list[float]) -> None:
    server = ApiServer("node-a")
    server.writers["node-a"] = [kubelet_status]
    server.reread_failure = (
        "Unable to connect to the server: dial tcp 10.0.0.1:443: connect:"
        " connection refused"
    )

    with pytest.raises(BootstrapError) as raised:
        clear(server)

    assert str(raised.value).startswith(
        "cannot re-read node node-a for annotation cleanup (exit 1, reason: "
        "unreachable): kubectl get node node-a: stderr: Unable to connect"
    ), str(raised.value)


def test_the_failure_message_never_carries_stdout_or_secret_material(
    sleeps: list[float],
) -> None:
    bearer = "eyJhbGciOiJSUzI1NiJ9." + "b" * 40

    class Leaky(ApiServer):
        def patch(
            self, arguments: list[str], command: list[str]
        ) -> subprocess.CompletedProcess[str]:
            stdout = json.dumps({"kind": "Secret", "data": {"key": "c2VjcmV0"}})
            stderr = f"error: Authorization: Bearer {bearer} password=hunter2"
            return self.answer(arguments, 1, stdout, stderr)

    with pytest.raises(NodePatchCommandFailed) as raised:
        clear(Leaky("node-a"))

    message = str(raised.value)
    for secret in (bearer, "hunter2", "c2VjcmV0", '"kind"'):
        assert secret not in message, f"{secret!r} reached the failure message"
    assert "kubectl patch node node-a: stderr: error: Authorization: " in message, (
        message
    )


def test_without_annotations_to_remove_no_node_is_patched(sleeps: list[float]) -> None:
    """The join rollback's ownership proof calls it with no annotations at all."""

    server = ApiServer("node-a", "node-b")
    clear_installer_annotations(
        server,
        KUBECTL,
        hyperpod_name="gpu-fixture",
        node_uids={"node-a": "uid-node-a", "node-b": "uid-node-b"},
        annotations=(),
    )
    assert [call[len(KUBECTL) :][:2] for call in server.calls] == [["get", "nodes"]], (
        "only the identity proof runs"
    )


def test_the_back_off_stays_inside_the_callers_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[float] = []
    monkeypatch.setattr(RETRY.time, "sleep", recorded.append)
    with deadline_scope("fixture cleanup", 0.2):
        RETRY.back_off(3)
    assert len(recorded) == 1 and 0 < recorded[0] <= 0.2, (
        f"the 2 s back-off was not capped by the deadline: {recorded}"
    )
