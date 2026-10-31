"""The reset's node metadata cleanup when another writer moves a Node mid-phase.

Live (BOOT-029 stage 4): ``clear_node_metadata`` listed the nodes, a kubelet
posted one node's status between the list and that node's patch, the patch's
resourceVersion ``test`` refused it (kubectl exit 1) and the uninstall stopped
on ``Kubernetes cleanup command failed (exit 1)``, with no reason in the log or
the journal. These cases drive the cleanup against the scripted API server of
the neighbouring tests, which applies a JSON patch the way the real one does,
with other writers (a kubelet, an operator) acting between the cleanup's read
and its patch, and pin what a failed cleanup command says.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module
from tests.deploy.test_cov95_tools_cleanup_kubernetes import cli_state
from tests.regional.test_cleanup_kubernetes import (
    CONTEXT,
    JSON_PATCH_TEST_FAILED,
    Client,
    document,
    resource,
)

TOOLS = Path(__file__).resolve().parents[2] / "deploy/control-plane/tools"
KUBE = lazy_script_module(TOOLS / "cleanup_kubernetes.py")
FAILURE = lazy_script_module(TOOLS / "cleanup_failure.py")
PATCH = lazy_script_module(TOOLS / "cleanup_node_patch.py")
FORBIDDEN = (
    'Error from server (Forbidden): nodes "managed-node" is forbidden: User '
    '"system:serviceaccount:ops:cleanup" cannot patch resource "nodes" in API '
    'group "" at the cluster scope'
)

Writer = Callable[[dict[str, Any]], None]


def bump(node: dict[str, Any]) -> None:
    metadata = node["metadata"]
    metadata["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)


def kubelet_status(node: dict[str, Any]) -> None:
    """The kubelet posts Node status: only status and resourceVersion move."""

    node.setdefault("status", {})["conditions"] = [{"type": "Ready", "status": "True"}]
    bump(node)


class RacingClient(Client):
    """The scripted API server, with other writers acting before node patches.

    ``writers[name]`` holds the writes that land on the node right before each
    successive patch of it reaches the server; ``refusals[name]`` is the kubectl
    stderr of a patch the server refuses for a reason of its own (RBAC).
    """

    def __init__(self) -> None:
        super().__init__()
        self.writers: dict[str, list[Writer]] = {}
        self.refusals: dict[str, str] = {}

    def node(self, name: str) -> dict[str, Any]:
        return next(node for node in self.nodes if node["metadata"]["name"] == name)

    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
        if arguments[:2] == ("patch", "node"):
            name = arguments[2]
            pending = self.writers.get(name) or []
            if pending:
                pending.pop(0)(self.node(name))
            if name in self.refusals:
                self.calls.append((arguments, payload))
                raise FAILURE.CleanupCommandError(arguments, 1, self.refusals[name])
        return super().run(*arguments, payload=payload)

    def patches(self, name: str | None = None) -> list[list[dict[str, Any]]]:
        return [
            json.loads(arguments[arguments.index("-p") + 1])
            for arguments, _payload in self.calls
            if arguments[:2] == ("patch", "node")
            and (name is None or arguments[2] == name)
        ]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(PATCH.time, "sleep", recorded.append)
    return recorded


def managed(client: RacingClient, name: str = "managed-node") -> dict[str, Any]:
    node = client.node(name)
    node["metadata"]["labels"] = {
        "gpu-fault.io/managed": "true",
        "customer.example/kept": "yes",
    }
    node["metadata"]["annotations"] = {"gpu-fault.io/installed": "yes"}
    return node


def versions(patches: list[list[dict[str, Any]]]) -> list[str]:
    return [patch[1]["value"] for patch in patches]


def clear(client: RacingClient, state: dict[str, Any]) -> None:
    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")


# --- a fleet node ------------------------------------------------------------


def test_a_kubelet_status_update_between_list_and_patch_is_retried_on_the_fresh_node(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = managed(client)
    client.writers["managed-node"] = [kubelet_status]

    clear(client, state)

    first, second = client.patches("managed-node")
    assert versions([first, second]) == ["1", "2"], (
        "the retry must be bound to the resourceVersion of a fresh read"
    )
    assert second[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "uid-managed-node",
    }, "the retry keeps the UID precondition"
    assert second[2:] == first[2:], "the fresh Node still carried the same keys"
    assert node["metadata"]["labels"] == {"customer.example/kept": "yes"}, (
        "our label must be gone and the customer's kept"
    )
    assert node["metadata"]["annotations"] == {}, "our annotation must be gone"
    assert node["status"]["conditions"] == [{"type": "Ready", "status": "True"}], (
        "the kubelet's own write is untouched"
    )
    assert sleeps == [PATCH.NODE_PATCH_BACKOFF_SECONDS[0]], (
        "one short back-off before the fresh read"
    )


def test_the_next_node_is_still_cleaned_after_the_first_one_raced(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    client.nodes.append(resource("Node", "managed-node-2", namespace=None))
    state["fleet_snapshot"].append(
        {**state["fleet_snapshot"][0], "node_id": "managed-node-2"}
    )
    first, second = managed(client), managed(client, "managed-node-2")
    client.writers["managed-node"] = [kubelet_status]

    clear(client, state)

    assert [
        arguments[2]
        for arguments, _payload in client.calls
        if arguments[:2] == ("patch", "node")
    ] == ["managed-node", "managed-node", "managed-node-2"], (
        "the second node is patched once, after the first one's retry"
    )
    for node in (first, second):
        assert node["metadata"]["labels"] == {"customer.example/kept": "yes"}, (
            f"{node['metadata']['name']} kept our label"
        )
    assert sleeps == [PATCH.NODE_PATCH_BACKOFF_SECONDS[0]], "only the raced node waited"


def test_a_node_replaced_during_the_race_stays_a_hard_error(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = managed(client)

    def replace(target: dict[str, Any]) -> None:
        target["metadata"]["uid"] = "uid-replacement"
        bump(target)

    client.writers["managed-node"] = [replace]
    with pytest.raises(
        KUBE.CleanupStateError, match="changed UID: managed-node was replaced"
    ):
        clear(client, state)
    assert len(client.patches()) == 1, "nothing is sent to a replacement Node"
    assert node["metadata"]["labels"]["gpu-fault.io/managed"] == "true", (
        "the replacement's metadata is not ours to remove"
    )


def test_a_refused_patch_of_an_unchanged_node_fails_with_kubectls_reason(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = managed(client)
    client.refusals["managed-node"] = FORBIDDEN

    with pytest.raises(FAILURE.CleanupCommandError) as raised:
        clear(client, state)

    message = str(raised.value)
    assert message == (
        "Kubernetes cleanup command failed (exit 1, reason: forbidden): kubectl "
        f"patch node managed-node: stderr: {FORBIDDEN}"
    ), message
    assert raised.value.reason == "forbidden", "the reason class is kept"
    assert any("did not move" in note for note in raised.value.__notes__), (
        "the failure says why it was not retried"
    )
    assert len(client.patches()) == 1, (
        "a failure no concurrent update explains is not retried"
    )
    assert node["metadata"]["labels"]["gpu-fault.io/managed"] == "true", (
        "a refused patch changes nothing"
    )


def test_a_node_that_keeps_changing_fails_after_bounded_attempts_naming_node_and_reason(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = managed(client)
    client.writers["managed-node"] = [kubelet_status] * PATCH.NODE_PATCH_ATTEMPTS

    with pytest.raises(KUBE.CleanupStateError) as raised:
        clear(client, state)

    message = str(raised.value)
    assert message.startswith(
        "node metadata cleanup of managed-node did not settle: the Node's "
        f"resourceVersion moved before each of {PATCH.NODE_PATCH_ATTEMPTS} patch "
        "attempts"
    ), message
    assert "reason: invalid): kubectl patch node managed-node: stderr: " in message, (
        "the last failure's reason is named"
    )
    assert JSON_PATCH_TEST_FAILED in message, "kubectl's own line is kept"
    assert isinstance(raised.value.__cause__, FAILURE.CleanupCommandError), (
        "the last refused patch is the cause"
    )
    assert versions(client.patches()) == ["1", "2", "3", "4"], (
        "every attempt is bound to the version its own fresh read returned"
    )
    assert sleeps == list(PATCH.NODE_PATCH_BACKOFF_SECONDS), (
        "the back-off is short, bounded, and none follows the last attempt"
    )
    assert node["metadata"]["labels"]["gpu-fault.io/managed"] == "true", (
        "nothing is removed from a Node that never held still"
    )


def test_a_patch_applied_despite_a_lost_acknowledgement_is_not_sent_again(
    sleeps: list[float],
) -> None:
    class LostAcknowledgement(RacingClient):
        def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
            result = super().run(*arguments, payload=payload)
            if arguments[:2] == ("patch", "node"):
                # Applied server-side; the client ran out of time first.
                raise FAILURE.CleanupCommandError(arguments, None)
            return result

    client, state = LostAcknowledgement(), document()
    node = managed(client)

    clear(client, state)

    assert len(client.patches()) == 1, (
        "the fresh read shows nothing left to remove: no second patch"
    )
    assert node["metadata"]["labels"] == {"customer.example/kept": "yes"}, (
        "the applied patch stands"
    )


# --- a declared warm spare ---------------------------------------------------


def parked_spare(client: RacingClient, state: dict[str, Any]) -> dict[str, Any]:
    node = client.node("managed-node")
    node["spec"] = {"unschedulable": True}
    node["metadata"]["labels"] = {
        "gpu-fault.io/spare": "true",
        "customer.example/kept": "yes",
    }
    node["metadata"]["annotations"] = {"gpu-fault.io/previous-unschedulable": "false"}
    inventory = state["inventory_snapshot"]["gpu"]
    inventory["node_labels"].append("gpu-fault.io/spare")
    inventory["node_annotations"].append("gpu-fault.io/previous-unschedulable")
    return node


def baseline_true(node: dict[str, Any]) -> None:
    node["metadata"]["annotations"]["gpu-fault.io/previous-unschedulable"] = "true"
    bump(node)


def quarantined(node: dict[str, Any]) -> None:
    node["metadata"]["labels"]["gpu-fault.io/quarantined"] = "true"
    bump(node)


def baseline_lost(node: dict[str, Any]) -> None:
    del node["metadata"]["annotations"]["gpu-fault.io/previous-unschedulable"]
    bump(node)


def test_a_spare_whose_baseline_changed_during_the_race_keeps_its_cordon(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = parked_spare(client, state)
    client.writers["managed-node"] = [baseline_true]

    clear(client, state)

    stale, fresh = client.patches("managed-node")
    assert {"op": "replace", "path": "/spec/unschedulable", "value": False} in stale, (
        "the first plan uncordoned the spare its baseline said was schedulable"
    )
    assert not any(operation["path"] == "/spec/unschedulable" for operation in fresh), (
        "the fresh baseline says the node was cordoned: no uncordon"
    )
    assert node["spec"]["unschedulable"] is True, "the recorded cordon is kept"
    assert node["metadata"]["labels"] == {"customer.example/kept": "yes"}, (
        "our spare label is gone"
    )
    assert node["metadata"]["annotations"] == {}, "the baseline annotation is gone"


@pytest.mark.parametrize(
    ("writer", "refusal"),
    [
        (quarantined, "quarantined spare cannot be released"),
        (baseline_lost, "no scheduling baseline"),
    ],
)
def test_a_spare_rejudged_on_the_fresh_node_is_refused_without_a_second_patch(
    sleeps: list[float], writer: Writer, refusal: str
) -> None:
    client, state = RacingClient(), document()
    node = parked_spare(client, state)
    client.writers["managed-node"] = [writer]

    with pytest.raises(KUBE.CleanupStateError, match=refusal):
        clear(client, state)

    assert len(client.patches()) == 1, "the refusal precedes any further patch"
    assert node["spec"]["unschedulable"] is True, "the cordon is kept"
    assert node["metadata"]["labels"]["gpu-fault.io/spare"] == "true", (
        "the spare label is kept"
    )


# --- an orphaned installer node ----------------------------------------------


def orphan(client: RacingClient) -> dict[str, Any]:
    node = client.node("customer-node")
    node["metadata"]["annotations"] = {
        "gpu-fault.io/installer-state": "Failed",
        "gpu-fault.io/installer-attempts": "2",
        "customer.example/owner": "team-a",
    }
    return node


def installer_succeeded(node: dict[str, Any]) -> None:
    node["metadata"]["annotations"]["gpu-fault.io/installer-state"] = "Succeeded"
    bump(node)


def test_an_orphaned_installer_node_that_raced_is_cleaned_on_the_fresh_node(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = orphan(client)
    client.writers["customer-node"] = [kubelet_status]

    clear(client, state)

    assert versions(client.patches("customer-node")) == ["1", "2"], (
        "the orphan's retry is bound to the fresh resourceVersion"
    )
    assert node["metadata"]["annotations"] == {"customer.example/owner": "team-a"}, (
        "only the installer annotations are removed"
    )


def test_an_orphan_that_gained_a_runtime_during_the_race_is_refused(
    sleeps: list[float],
) -> None:
    client, state = RacingClient(), document()
    node = orphan(client)
    client.writers["customer-node"] = [installer_succeeded]

    with pytest.raises(KUBE.CleanupStateError, match="unregistered node runtime"):
        clear(client, state)

    assert len(client.patches("customer-node")) == 1, "no second patch"
    assert node["metadata"]["annotations"]["gpu-fault.io/installer-state"] == (
        "Succeeded"
    ), "an installed node keeps its installer record"


# --- what a failed cleanup command says --------------------------------------


def transport(
    monkeypatch: pytest.MonkeyPatch, *, code: int, stdout: str = "", stderr: str = ""
) -> None:
    def run(arguments: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, code, stdout, stderr)

    monkeypatch.setattr(KUBE, "run_command", run)


@pytest.mark.parametrize(
    ("stderr", "reason"),
    [
        (
            'Error from server (Conflict): Operation cannot be fulfilled on nodes "n":'
            " the object has been modified; please apply your changes to the latest"
            " version and try again",
            "conflict",
        ),
        ('Error from server (NotFound): nodes "n" not found', "not found"),
        (FORBIDDEN, "forbidden"),
        ("error: You must be logged in to the server (Unauthorized)", "unauthorized"),
        (JSON_PATCH_TEST_FAILED, "invalid"),
        (
            'The Node "n" is invalid: metadata.labels: Invalid value: "x y": a valid'
            " label must be an empty string or consist of alphanumeric characters",
            "invalid",
        ),
        (
            "Unable to connect to the server: net/http: request canceled"
            " (Client.Timeout exceeded while awaiting headers)",
            "timeout",
        ),
        (
            "Error from server (Timeout): the server was unable to return a response"
            " in the time allotted, but may still be processing the request",
            "timeout",
        ),
        (
            "Unable to connect to the server: dial tcp 10.0.0.1:443: i/o timeout",
            "timeout",
        ),
        (
            "Unable to connect to the server: dial tcp 10.0.0.1:443: connect:"
            " connection refused",
            "unreachable",
        ),
        (
            "Error from server (InternalError): Internal error occurred: failed calling"
            ' webhook "validate.example.io": context canceled',
            "server error",
        ),
        (
            "Error from server (TooManyRequests): the server has received too many"
            " requests and has asked us to try again later",
            "throttled",
        ),
        ("error: something nobody anticipated", "unclassified"),
    ],
)
def test_a_failed_kubectl_call_names_its_reason_class(
    monkeypatch: pytest.MonkeyPatch, stderr: str, reason: str
) -> None:
    transport(monkeypatch, code=1, stderr=f"W1031 warnings.go:70] noise\n{stderr}\n")
    client = KUBE.CleanupClient({"cpu_kubeconfig": "/dev/null"}, "cpu")

    with pytest.raises(FAILURE.CleanupCommandError) as raised:
        client.run("patch", "node", "n", "--type=json", "-p", '[{"op":"test"}]')

    assert raised.value.reason == reason, str(raised.value)
    assert str(raised.value).startswith(
        f"Kubernetes cleanup command failed (exit 1, reason: {reason}): "
        "kubectl patch node n: stderr: "
    ), str(raised.value)
    assert raised.value.returncode == 1, "the exit status is kept"


def test_a_failure_without_stderr_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    transport(monkeypatch, code=2)
    client = KUBE.CleanupClient({}, "gpu:gpu-a")

    with pytest.raises(FAILURE.CleanupCommandError) as raised:
        client.items("get", "nodes")

    assert str(raised.value) == (
        "Kubernetes cleanup command failed (exit 2, reason: unclassified): "
        "kubectl get nodes: no error output"
    ), str(raised.value)


def test_the_failure_message_never_carries_stdout_or_secret_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bearer = "eyJhbGciOiJSUzI1NiJ9." + "a" * 40
    blob = "c2VjcmV0LWJ5dGVzLXRoYXQtbXVzdC1uZXZlci1sZWFr" * 2
    transport(
        monkeypatch,
        code=1,
        stdout=json.dumps(
            {"kind": "Secret", "data": {"token": "c3Rkb3V0LXNlY3JldA=="}}
        ),
        stderr=(
            "Error from server (BadRequest): request rejected: Authorization: Bearer "
            f"{bearer} password=hunter2 data={blob} " + "x" * 500 + "\n"
        ),
    )
    client = KUBE.CleanupClient({"cpu_kubeconfig": "/dev/null"}, "cpu")

    with pytest.raises(FAILURE.CleanupCommandError) as raised:
        client.run("-n", "gpu-fault-system", "get", "secret,configmap", "-o", "json")

    message = str(raised.value)
    for secret in (bearer, "hunter2", blob, "c3Rkb3V0LXNlY3JldA==", '"kind"'):
        assert secret not in message, f"{secret!r} reached the failure message"
    assert "kubectl get secret,configmap: stderr: " in message, message
    assert len(raised.value.detail) <= len("stderr: ") + 200, (
        "the stderr excerpt is bounded"
    )


def test_a_kubectl_call_that_runs_out_of_time_is_a_timeout_without_exit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def expire(arguments: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(arguments, 45)

    monkeypatch.setattr(KUBE, "run_command", expire)
    client = KUBE.CleanupClient({"cpu_kubeconfig": "/dev/null"}, "cpu")

    with pytest.raises(FAILURE.CleanupCommandError) as raised:
        client.run("patch", "node", "n", "--type=json", "-p", "[]")

    assert (raised.value.reason, raised.value.returncode) == ("timeout", None), str(
        raised.value
    )
    assert str(raised.value).startswith(
        "Kubernetes cleanup command failed (no exit status, reason: timeout): "
        "kubectl patch node n: "
    ), str(raised.value)


@pytest.mark.parametrize(
    ("arguments", "summary"),
    [
        (("patch", "node", "n", "--type=json", "-p", "[]"), "patch node n"),
        (
            (
                "-n",
                "ns",
                "get",
                "deployment",
                "api",
                "--ignore-not-found",
                "-o",
                "json",
            ),
            "get deployment api",
        ),
        (("-n", "ns", "exec", "pod-a", "--", "python", "-c", "print(1)"), "exec pod-a"),
        (
            ("delete", "--raw", "/api/v1/namespaces/ns", "-f", "-"),
            "delete /api/v1/namespaces/ns",
        ),
        (
            ("get", ",".join(["kind"] * 40)),
            "get " + ",".join(["kind"] * 40)[:64] + "...",
        ),
        ((), "<no arguments>"),
    ],
)
def test_a_failed_command_is_named_by_verb_kind_and_name_only(
    arguments: tuple[str, ...], summary: str
) -> None:
    assert FAILURE.command_summary(arguments) == summary, "unexpected command summary"


# --- the reason reaches the journal --------------------------------------------


def run_main(monkeypatch: pytest.MonkeyPatch, config: Path, path: Path) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cleanup_kubernetes",
            "--config",
            str(config),
            "--state-file",
            str(path),
            "--context",
            CONTEXT,
            "--cluster-id",
            "gpu-a",
            "clear-node-metadata",
        ],
    )
    return int(KUBE.main())


def test_a_failed_cleanup_invocation_leaves_its_reason_for_the_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    config, path, _state = cli_state(tmp_path)
    note = tmp_path / "failure-note"
    note.write_text("an earlier invocation's failure")
    monkeypatch.setenv(FAILURE.FAILURE_NOTE_ENV, str(note))
    client = RacingClient()
    managed(client)
    client.refusals["managed-node"] = FORBIDDEN
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)

    with pytest.raises(FAILURE.CleanupCommandError):
        run_main(monkeypatch, config, path)

    assert note.read_text() == (
        f"clear-node-metadata {CONTEXT}: Kubernetes cleanup command failed (exit 1,"
        f" reason: forbidden): kubectl patch node managed-node: stderr: {FORBIDDEN}"
    ), note.read_text()

    client.refusals.clear()
    assert run_main(monkeypatch, config, path) == 0, "the resumed cleanup completes"
    assert note.read_text() == "", "a later success leaves no stale reason behind"


def test_the_failure_note_holds_only_text_this_tooling_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    note = tmp_path / "failure-note"
    note.write_text("")
    monkeypatch.setenv(FAILURE.FAILURE_NOTE_ENV, str(note))

    with pytest.raises(KeyError), FAILURE.failure_note("capture cpu"):
        raise KeyError("password=hunter2")

    assert note.read_text() == "capture cpu: KeyError", (
        "a foreign exception is named by its type, never by its text"
    )

    with pytest.raises(KUBE.CleanupStateError), FAILURE.failure_note("capture cpu"):
        raise KUBE.CleanupStateError("node \u2603 " + "y" * 1000)

    text = note.read_text()
    assert text.startswith("capture cpu: node ? yyy"), "non-ASCII is replaced"
    assert len(text) == FAILURE.FAILURE_NOTE_CHARACTERS, "the note is bounded"


def test_the_failure_note_is_never_created_by_the_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "absent"
    monkeypatch.setenv(FAILURE.FAILURE_NOTE_ENV, str(missing))

    with pytest.raises(KUBE.CleanupStateError), FAILURE.failure_note("capture cpu"):
        raise KUBE.CleanupStateError("injected")

    assert not missing.exists(), "only the file the cleanup script created is written"
