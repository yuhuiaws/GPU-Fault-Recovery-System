"""The executor keys the nodes HyperPod replaced after the last deploy.

``NodeActionKeySync`` compares the nodes labelled with this cluster's HyperPod
name against the key names in ``gpu-fault-node-action-keys``, asks the control
plane for the missing ones and merge-patches only those in. Every invariant the
module docstring promises is a test here: only missing keys, never a
rewrite or a delete, one resourceVersion-fenced patch, nothing written on any
refusal or malformed answer, key names but never values in the log.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import pytest

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    NodeActionKeySync,
)
from gpu_fault.cluster_executor.node_key_sync import HYPERPOD_CLUSTER_LABEL
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
)

CLUSTER = "cluster-a"
HYPERPOD = "hp-cluster-a"
NAMESPACE = "gpu-fault-system"
SECRET = "gpu-fault-node-action-keys"
KEY_OLD = "o" * 64
KEY_NEW = "n" * 64


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


class FakeCore:
    """Just the three CoreV1Api calls the sync makes, recorded."""

    def __init__(
        self,
        nodes: list[str],
        secret_keys: dict[str, str] | None,
        *,
        resource_version: str = "17",
        patch_error: Exception | None = None,
    ) -> None:
        self.nodes = nodes
        self.secret = (
            None
            if secret_keys is None
            else {
                "metadata": {"name": SECRET, "resourceVersion": resource_version},
                "data": {name: _b64(value) for name, value in secret_keys.items()},
            }
        )
        self.patch_error = patch_error
        self.list_selectors: list[str] = []
        self.reads: list[tuple[str, str]] = []
        self.patches: list[tuple[str, str, dict[str, Any]]] = []

    def list_node(self, label_selector: str) -> dict[str, Any]:
        self.list_selectors.append(label_selector)
        return {"items": [{"metadata": {"name": name}} for name in self.nodes]}

    def read_namespaced_secret(self, name: str, namespace: str) -> dict[str, Any]:
        self.reads.append((name, namespace))
        if self.secret is None:
            error = RuntimeError('secrets "%s" not found' % name)
            error.status = 404  # type: ignore[attr-defined]
            raise error
        return self.secret

    def patch_namespaced_secret(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        if self.patch_error is not None:
            raise self.patch_error
        self.patches.append((name, namespace, body))
        assert self.secret is not None
        self.secret["data"].update(body["data"])


class FakeKeyClient:
    cluster_id = CLUSTER

    def __init__(
        self, answer: dict[str, str] | None = None, *, error: Exception | None = None
    ) -> None:
        self.answer = answer
        self.error = error
        self.requests: list[list[str]] = []

    def node_action_keys(self, node_ids: list[str]) -> dict[str, str]:
        self.requests.append(list(node_ids))
        if self.error is not None:
            raise self.error
        assert self.answer is not None
        return {node: self.answer[node] for node in node_ids}


def _sync(client: Any, core: FakeCore, **overrides: Any) -> NodeActionKeySync:
    return NodeActionKeySync(
        client, core, hyperpod_cluster=HYPERPOD, namespace=NAMESPACE, **overrides
    )


def test_only_the_missing_keys_are_added_in_one_fenced_merge_patch(caplog) -> None:
    core = FakeCore(["hyperpod-i-old", "hyperpod-i-new"], {"hyperpod-i-old": KEY_OLD})
    client = FakeKeyClient({"hyperpod-i-new": KEY_NEW})

    with caplog.at_level(logging.INFO, logger="gpu_fault.cluster_executor"):
        added = _sync(client, core).run()

    assert added == ["hyperpod-i-new"]
    assert client.requests == [["hyperpod-i-new"]], "the existing node was requested"
    assert core.list_selectors == [f"{HYPERPOD_CLUSTER_LABEL}={HYPERPOD}"]
    assert core.reads == [(SECRET, NAMESPACE)]
    assert core.patches == [
        (
            SECRET,
            NAMESPACE,
            {
                "metadata": {"resourceVersion": "17"},
                "data": {"hyperpod-i-new": _b64(KEY_NEW)},
            },
        )
    ]
    # The old key's bytes are untouched and the new one is stored the way the
    # deploy host stores them (base64 of the utf-8 hex string).
    assert core.secret["data"]["hyperpod-i-old"] == _b64(KEY_OLD)
    assert core.secret["data"]["hyperpod-i-new"] == _b64(KEY_NEW)
    messages = [record.getMessage() for record in caplog.records]
    assert any("hyperpod-i-new" in message for message in messages), messages
    assert not any(KEY_NEW in message or KEY_OLD in message for message in messages), (
        "a key value leaked into a log line"
    )
    assert not any(_b64(KEY_NEW) in message for message in messages), (
        "a base64 key value leaked into a log line"
    )


def test_a_complete_secret_is_a_no_op_without_a_control_plane_call() -> None:
    core = FakeCore(
        ["hyperpod-i-1", "hyperpod-i-2"],
        {"hyperpod-i-1": KEY_OLD, "hyperpod-i-2": KEY_OLD},
    )
    client = FakeKeyClient({})

    assert _sync(client, core).run() == []

    assert client.requests == []
    assert core.patches == []


def test_a_key_without_a_node_is_left_alone_never_deleted() -> None:
    """A node that left the cluster keeps its entry: removal is the deploy host's."""

    core = FakeCore(
        ["hyperpod-i-1"], {"hyperpod-i-1": KEY_OLD, "hyperpod-i-gone": KEY_OLD}
    )

    assert _sync(FakeKeyClient({}), core).run() == []
    assert core.patches == []
    assert set(core.secret["data"]) == {"hyperpod-i-1", "hyperpod-i-gone"}


def test_a_control_plane_refusal_writes_nothing() -> None:
    core = FakeCore(["hyperpod-i-new"], {})
    client = FakeKeyClient(
        error=ClusterExecutorError(
            "regional control plane rejected request (404): unknown node",
            status_code=404,
        )
    )
    sync = _sync(client, core)

    with pytest.raises(ClusterExecutorError):
        sync.run()

    assert core.patches == []
    assert sync.failures_total == 1
    assert sync.added_total == 0


@pytest.mark.parametrize(
    "answer", [{"hyperpod-i-new": "short"}, {"hyperpod-i-new": " " + "n" * 64}]
)
def test_a_malformed_key_is_refused_before_any_patch(answer) -> None:
    core = FakeCore(["hyperpod-i-new"], {})
    sync = _sync(FakeKeyClient(answer), core)

    with pytest.raises(ClusterExecutorError, match="malformed"):
        sync.run()

    assert core.patches == []


def test_an_answer_for_a_different_node_set_is_refused() -> None:
    class ExtraNodeClient(FakeKeyClient):
        def node_action_keys(self, node_ids: list[str]) -> dict[str, str]:
            self.requests.append(list(node_ids))
            return {"hyperpod-i-new": KEY_NEW, "hyperpod-i-other": KEY_NEW}

    core = FakeCore(["hyperpod-i-new"], {})

    with pytest.raises(ClusterExecutorError, match="different node set"):
        _sync(ExtraNodeClient(), core).run()

    assert core.patches == []


def test_an_absent_secret_is_never_created_here() -> None:
    core = FakeCore(["hyperpod-i-new"], None)
    client = FakeKeyClient({"hyperpod-i-new": KEY_NEW})

    with pytest.raises(ClusterExecutorError, match="absent"):
        _sync(client, core).run()

    assert client.requests == [], "no key is fetched for a Secret nobody created"
    assert core.patches == []


def test_a_conflicting_patch_propagates_and_the_next_cycle_retries() -> None:
    conflict = RuntimeError("the object has been modified")
    conflict.status = 409  # type: ignore[attr-defined]
    core = FakeCore(["hyperpod-i-new"], {}, patch_error=conflict)
    sync = _sync(FakeKeyClient({"hyperpod-i-new": KEY_NEW}), core)

    with pytest.raises(RuntimeError):
        sync.run()
    assert sync.added_total == 0

    core.patch_error = None
    # Move the clock past the interval the failed cycle armed: the next cycle
    # is due again through the public pacing rule, not by resetting state.
    clock = {"now": time.monotonic() + sync.interval_seconds + 1.0}
    sync.clock = lambda: clock["now"]
    assert sync.due(), "a full interval after the failed cycle the sync is due"
    assert sync.run() == ["hyperpod-i-new"]
    assert sync.added_total == 1


def test_the_sync_is_paced_by_its_interval() -> None:
    clock = {"now": 1000.0}
    core = FakeCore([], {})
    sync = _sync(
        FakeKeyClient({}), core, interval_seconds=60.0, clock=lambda: clock["now"]
    )

    assert sync.due(), "the first sync runs immediately"
    sync.run()
    assert not sync.due(), "a cycle that just ran is not due again"
    clock["now"] += 59.0
    assert not sync.due(), "inside the interval the sync stays paced"
    clock["now"] += 1.0
    assert sync.due(), "past the interval the sync is due"


def test_more_than_one_batch_of_missing_nodes_is_requested_in_bounded_chunks() -> None:
    nodes = [f"hyperpod-i-{index:03d}" for index in range(70)]
    core = FakeCore(nodes, {})
    client = FakeKeyClient({node: KEY_NEW for node in nodes})

    added = _sync(client, core).run()

    assert added == nodes
    assert [len(request) for request in client.requests] == [64, 6]
    assert len(core.patches) == 1, "still one patch, after every key was validated"


@pytest.mark.parametrize(
    "overrides",
    [
        {"hyperpod_cluster": ""},
        {"namespace": " "},
        {"interval_seconds": 0},
        {"secret_name": ""},
    ],
)
def test_the_sync_refuses_an_unusable_configuration(overrides) -> None:
    settings: dict[str, Any] = {
        "hyperpod_cluster": HYPERPOD,
        "namespace": NAMESPACE,
        **overrides,
    }
    with pytest.raises(ValueError):
        NodeActionKeySync(FakeKeyClient({}), FakeCore([], {}), **settings)


def test_the_executor_runs_the_sync_when_due_and_never_lets_it_raise(caplog) -> None:
    core = FakeCore(["hyperpod-i-new"], {})
    client = FakeKeyClient(
        error=ClusterExecutorError("regional control plane request failed: timeout")
    )
    sync = _sync(client, core)
    executor = ClusterActionExecutor(
        FakeExecutorClient(),
        [],
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        node_key_sync=sync,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        executor.sync_node_action_keys()

    assert sync.failures_total == 1
    assert any(
        "node action key sync failed" in record.getMessage()
        for record in caplog.records
    ), "the wrapper must log the failure"
    # Not due again until the interval elapses: a failing control plane costs
    # one request per cycle, not one per claim.
    executor.sync_node_action_keys()
    assert sync.failures_total == 1


def test_an_executor_without_a_sync_is_unchanged() -> None:
    executor = ClusterActionExecutor(
        FakeExecutorClient(), [], executor_id=EXECUTOR, allowed_namespaces={"training"}
    )
    assert executor.node_key_sync is None
    executor.sync_node_action_keys()
