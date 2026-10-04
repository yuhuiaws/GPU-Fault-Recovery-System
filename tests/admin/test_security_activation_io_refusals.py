"""Refusals and resume paths of the real custody activation adapter.

The adapter talks to both clusters through kubectl; ``ActivationWorld`` answers
in memory. Each test moves one fact in that world -- a naive timestamp, a
missing ingress Pod, a foreign namespace UID, an in-flight installer Job, a
patch that did not land -- and pins that the adapter refuses instead of guessing,
or that a resumed step recognises its own earlier work instead of redoing it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from gpu_fault.admin import node_key_custody_activation_io as adapter
from gpu_fault.admin.node_key_custody_models import CustodyError
from tests.admin._security_activation_io_world import ActivationWorld

Reply = Callable[[tuple[str, ...], dict[str, Any], str], str | None]


@pytest.fixture
def world(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> ActivationWorld:
    return ActivationWorld(tmp_path, monkeypatch)


def intercept(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch, reply: Reply
) -> None:
    """Let ``reply`` rewrite the answer of any command; ``None`` keeps the original."""
    original = world.run

    def run(command: Any, **kwargs: Any) -> str:
        args = tuple(str(item) for item in command)
        result = original(command, **kwargs)
        replaced = reply(args, kwargs, result)
        return result if replaced is None else replaced

    monkeypatch.setattr(world, "run", run)


def _is(args: tuple[str, ...], operation: str, kind: str) -> bool:
    return operation in args and args[args.index(operation) + 1] == kind


def test_a_naive_agent_timestamp_is_an_invalid_time_binding(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def naive(args: tuple[str, ...], kwargs: dict[str, Any], result: str) -> str | None:
        if "exec" in args and args[-1] == adapter.AGENTS_PROBE:
            agents = json.loads(result)
            for value in agents.values():
                stamp = str(value["lease_expires_at"]).replace("Z", "+00:00")
                value["lease_expires_at"] = stamp.split("+")[0]
            return json.dumps(agents)
        return None

    intercept(world, monkeypatch, naive)
    with pytest.raises(CustodyError, match="invalid time binding"):
        world.io.capture()
    assert world.patches == []


def test_the_cpu_probe_requires_a_ready_ingress_pod(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(world.io, "consumer_pods", lambda *_a: [])
    with pytest.raises(CustodyError, match="no Ready CPU ingress"):
        world.io.agents()
    assert not any("exec" in call for call in world.calls), (
        "no probe may be executed without a Pod to run it in"
    )


def test_a_changed_cluster_anchor_stops_the_activation(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def foreign(
        args: tuple[str, ...], kwargs: dict[str, Any], result: str
    ) -> str | None:
        if _is(args, "get", "namespace") and "kube-system" in args:
            value = json.loads(result)
            value["metadata"]["uid"] = "foreign-cluster"
            return json.dumps(value)
        return None

    intercept(world, monkeypatch, foreign)
    with pytest.raises(CustodyError, match="anchor changed"):
        world.io.verify_anchors()


@pytest.mark.parametrize("field", ["uid", "resourceVersion"])
def test_a_deployment_without_complete_identity_is_refused(
    world: ActivationWorld, field: str
) -> None:
    world.records["gpu", "deployment", adapter.GPU_RECONCILER_DEPLOYMENT]["metadata"][
        field
    ] = ""
    with pytest.raises(CustodyError, match="identity is incomplete"):
        world.io.capture()


def test_a_foreign_activation_annotation_is_not_adopted(world: ActivationWorld) -> None:
    world.records["gpu", "deployment", adapter.GPU_EXECUTOR_DEPLOYMENT]["spec"][
        "template"
    ]["metadata"]["annotations"][adapter.ACTIVATION_ANNOTATION] = "foreign-writer"
    with pytest.raises(CustodyError, match="belongs to another writer"):
        world.io.capture()


def test_sibling_keys_require_a_key_source_on_both_planes(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def absent(
        args: tuple[str, ...], kwargs: dict[str, Any], result: str
    ) -> str | None:
        return "" if _is(args, "get", "secret") and "gpu-context" in args else None

    intercept(world, monkeypatch, absent)
    with pytest.raises(CustodyError, match="no key source"):
        world.io.sibling_keys()


@pytest.mark.parametrize(
    "field, value",
    [
        ("release_id", "release-b"),
        ("phase", "rolling"),
        ("transaction_committed", False),
        ("bundle_sha256", "0" * 64),
        ("runtime_profile_version", "profile-b"),
    ],
)
def test_release_state_must_match_the_authorized_release(
    world: ActivationWorld, field: str, value: Any
) -> None:
    record = world.records["cpu", "configmap", "gpu-fault-regional-release-state"]
    state = json.loads(record["data"]["state.json"])
    state[field] = value
    record["data"]["state.json"] = json.dumps(state)
    with pytest.raises(CustodyError, match="committed authorized release"):
        world.io.release_state()


def test_capture_requires_every_agent_to_be_active(world: ActivationWorld) -> None:
    current = world.store.get_agent("cluster-a", "node-a")
    world.store.save_agent(
        current.model_copy(
            update={"lifecycle_state": type(current.lifecycle_state).DRAINING}
        )
    )
    with pytest.raises(CustodyError, match="requires current ACTIVE Agents"):
        world.io.capture()


@pytest.mark.parametrize(
    "jobs, message",
    [
        ({"items": [], "metadata": {"continue": "more"}}, "inventory is incomplete"),
        ({"items": "none"}, "inventory is incomplete"),
        (
            {
                "items": [
                    {
                        "metadata": {"name": "gpu-fault-install-node-a"},
                        "status": {
                            "conditions": [{"type": "Complete", "status": "False"}]
                        },
                    }
                ]
            },
            "in-flight installer Job",
        ),
    ],
)
def test_installer_jobs_block_or_fail_capture(
    world: ActivationWorld,
    monkeypatch: pytest.MonkeyPatch,
    jobs: dict[str, Any],
    message: str,
) -> None:
    def listing(
        args: tuple[str, ...], kwargs: dict[str, Any], result: str
    ) -> str | None:
        return json.dumps(jobs) if _is(args, "get", "jobs") else None

    intercept(world, monkeypatch, listing)
    with pytest.raises(CustodyError, match=message):
        world.io.capture()


def test_a_reconciler_without_its_wave_binding_is_not_the_authorized_one(
    world: ActivationWorld,
) -> None:
    container = world.records["gpu", "deployment", adapter.GPU_RECONCILER_DEPLOYMENT][
        "spec"
    ]["template"]["spec"]["containers"][0]
    container["env"] = [
        item
        for item in container["env"]
        if item["name"] != adapter.INSTALLER_WAVE_CONFIG_MAP_ENV
    ]
    with pytest.raises(CustodyError, match="Reconciler identity differs"):
        world.io.capture()


def test_a_repeated_guard_recognises_its_own_drain(world: ActivationWorld) -> None:
    snapshot = world.io.capture()
    first = world.io.guard(snapshot)
    drains = len([c for c in world.calls if c[-1] == adapter.TRANSITION_PROBE])
    assert first == world.io.guard(snapshot)
    assert (
        len([c for c in world.calls if c[-1] == adapter.TRANSITION_PROBE]) == drains
    ), "an Agent already draining under this transaction must not be drained again"


def test_a_routine_heartbeat_does_not_block_the_drain(world: ActivationWorld) -> None:
    snapshot = world.io.capture()
    current = world.store.get_agent("cluster-a", "node-a")
    world.store.save_agent(
        current.model_copy(
            update={"last_seen_at": datetime.now(timezone.utc) + timedelta(seconds=1)}
        )
    )
    transition = world.io.guard(snapshot)["transition_id"]
    assert transition == "node-key-" + world.authorization.transaction_id
    assert world.store.get_agent("cluster-a", "node-a").lifecycle_state.value == (
        "DRAINING"
    )


def test_an_unacknowledged_drain_is_an_error(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stale(args: tuple[str, ...], kwargs: dict[str, Any], result: str) -> str | None:
        if "exec" in args and args[-1] == adapter.TRANSITION_PROBE:
            value = json.loads(result)
            value["lifecycle_state"] = "ACTIVE"
            return json.dumps(value)
        return None

    intercept(world, monkeypatch, stale)
    with pytest.raises(CustodyError, match="drain was not acknowledged"):
        world.io.guard(world.io.capture())


def test_a_wave_patch_that_did_not_land_is_reported(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = world.run

    def run(command: Any, **kwargs: Any) -> str:
        args = tuple(str(item) for item in command)
        if _is(args, "patch", "configmap"):
            world.calls.append(args)
            return ""
        return original(command, **kwargs)

    monkeypatch.setattr(world, "run", run)
    snapshot = world.io.capture()
    with pytest.raises(CustodyError, match="did not converge"):
        world.io.fence(snapshot)
    assert world.records["gpu", "configmap", "wave"]["data"] == snapshot["wave"]["data"]


def _node_annotations(world: ActivationWorld) -> dict[str, Any]:
    return world.records["gpu", "node", "node-a"]["metadata"]["annotations"]


def _fenced(world: ActivationWorld) -> dict[str, Any]:
    snapshot = world.io.capture()
    world.io.guard(snapshot)
    world.io.fence(snapshot)
    world.bind_provisioned()
    return snapshot


def test_a_resumed_consumer_roll_does_not_repatch_its_own_marker(
    world: ActivationWorld,
) -> None:
    snapshot = _fenced(world)
    marker = world.io.markers("gpu", adapter.GPU_EXECUTOR_DEPLOYMENT)[0]
    world.records["gpu", "deployment", adapter.GPU_EXECUTOR_DEPLOYMENT]["spec"][
        "template"
    ]["metadata"]["annotations"][adapter.ACTIVATION_ANNOTATION] = marker
    assert world.io.refresh_executor(snapshot)["activation_marker"] == marker
    assert ("gpu", "deployment", adapter.GPU_EXECUTOR_DEPLOYMENT) not in world.patches


def test_consumer_pods_still_on_the_old_process_fail_the_roll(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _fenced(world)
    original = world.pods

    def stale(plane: str, name: str) -> dict[str, Any]:
        value = original(plane, name)
        for pod in value["items"]:
            pod["metadata"]["annotations"][adapter.ACTIVATION_ANNOTATION] = "old"
        return value

    monkeypatch.setattr(world, "pods", stale)
    with pytest.raises(CustodyError, match="old consumer process"):
        world.io.refresh_executor(snapshot)


def test_install_recognises_a_node_it_already_activated(world: ActivationWorld) -> None:
    snapshot = _fenced(world)
    annotations = world.records["gpu", "node", "node-a"]["metadata"]["annotations"]
    transaction = world.authorization.transaction_id
    annotations[adapter.INSTALLER_ACTIVATION_ANNOTATION] = transaction
    annotations[adapter.INSTALLER_ACTIVATED_ANNOTATION] = transaction
    assert world.io.install(snapshot) == {
        "node_id": "node-a",
        "activation_id": transaction,
    }
    assert ("gpu", "node", "node-a") not in world.patches, (
        "a Node already carrying this activation must not be patched again"
    )


def test_install_waits_for_the_installer_and_then_converges(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _fenced(world)
    original = world.run
    naps: list[float] = []

    def run(command: Any, **kwargs: Any) -> str:
        result = original(command, **kwargs)
        if _is(tuple(str(item) for item in command), "patch", "node"):
            _node_annotations(world).pop(adapter.INSTALLER_ACTIVATED_ANNOTATION)
        return result

    def nap(seconds: float) -> None:
        naps.append(seconds)
        node = _node_annotations(world)
        node[adapter.INSTALLER_ACTIVATED_ANNOTATION] = node[
            adapter.INSTALLER_ACTIVATION_ANNOTATION
        ]

    monkeypatch.setattr(world, "run", run)
    monkeypatch.setattr(adapter.time, "sleep", nap)
    assert world.io.install(snapshot)["node_id"] == "node-a"
    assert naps == [5], "the wait must poll in bounded steps"


def test_install_stops_when_the_installer_reports_failure(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _fenced(world)
    original = world.run

    def run(command: Any, **kwargs: Any) -> str:
        result = original(command, **kwargs)
        if _is(tuple(str(item) for item in command), "patch", "node"):
            node = _node_annotations(world)
            node.pop(adapter.INSTALLER_ACTIVATED_ANNOTATION)
            node["gpu-fault.io/installer-state"] = "Failed"
        return result

    monkeypatch.setattr(world, "run", run)
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: pytest.fail("must not wait"))
    with pytest.raises(CustodyError, match="installer did not converge"):
        world.io.install(snapshot)


def test_observe_requires_the_captured_agent_membership(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = world.io.capture()
    original = world.io.agents
    monkeypatch.setattr(world.io, "agents", lambda: {"node-a": original()["node-a"]})
    with pytest.raises(CustodyError, match="Agent membership changed"):
        world.io.observe(snapshot)


def test_observe_polls_until_the_new_agent_is_fresh(
    world: ActivationWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = world.io.capture()
    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        current = world.store.get_agent("cluster-a", "node-a")
        now = datetime.now(timezone.utc)
        world.store.save_agent(
            current.model_copy(
                update={
                    "agent_incarnation_id": "new-agent",
                    "generation": current.generation + 1,
                    "last_seen_at": now,
                    "lease_expires_at": now + timedelta(seconds=120),
                }
            )
        )

    monkeypatch.setattr(adapter.time, "sleep", nap)
    observed = world.io.observe(snapshot)
    assert observed["agent_incarnation_id"] == "new-agent"
    assert naps == [5]


def test_a_completed_activation_must_keep_its_release_binding(
    world: ActivationWorld,
) -> None:
    snapshot = world.io.capture()
    world.bind_provisioned()
    snapshot["release"]["uid"] = "replacement"
    with pytest.raises(CustodyError, match="no longer has its release/node binding"):
        world.io.current(snapshot)
