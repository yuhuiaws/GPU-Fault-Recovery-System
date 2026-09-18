from __future__ import annotations

import base64
import copy
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_join_nodes as join_nodes
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_nodes import NodeClaims, read_node_claims
from gpu_fault.admin.cluster_removal import _remove_node_action_keys as remove_keys
from gpu_fault.admin.cluster_removal_keys import verify_node_key_ownership
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from tests.admin.test_admin_cluster_removal_lifecycle import RemovalScenario


def encoded(label: str) -> str:
    return base64.b64encode((label * 8).encode()).decode()


class NodeKeyScenario(RemovalScenario):
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.cpu = self.secret("cpu-keys", {"node-a": encoded("target-fixture")})
        self.gpu = self.secret("gpu-keys", {"node-a": encoded("target-fixture")})
        self.cpu["data"]["node-other"] = encoded("preserved-fixture")
        self.key_patches: list[list[dict[str, Any]]] = []
        self.node_names_by_context = {"gpu-a": ["node-a"]}
        self.agent_owners = {"node-a": "gpu-a"}
        monkeypatch.setattr(removal, "CommandRunner", lambda: self)
        monkeypatch.setattr(removal, "_remove_node_action_keys", remove_keys)
        monkeypatch.setattr(
            removal, "verify_node_key_ownership", verify_node_key_ownership
        )
        monkeypatch.setattr(removal, "run_command", self.command)
        monkeypatch.setattr(
            removal,
            "_json_command",
            lambda arguments, **_kwargs: json.loads(self.command(arguments).stdout),
        )
        monkeypatch.setattr(
            join_nodes,
            "read_node_claims",
            lambda *_a: NodeClaims(
                nodes_by_cluster={"gpu-a": frozenset({"node-a"})},
                agent_owners={"node-a": "gpu-a"},
                cpu_key_names=frozenset(self.cpu["data"]),
            ),
        )

    @staticmethod
    def secret(uid: str, data: dict[str, str]) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "Opaque",
            "metadata": {
                "name": "gpu-fault-node-action-keys",
                "namespace": "gpu-fault-system",
                "uid": uid,
                "resourceVersion": "1",
            },
            "data": data,
        }

    def run(self, arguments: Sequence[str], **kwargs: Any) -> str:
        if "secret" in arguments or "exec" in arguments:
            assert kwargs.get("sensitive") is True, (
                "node-key proof must use private command output"
            )
        if "nodes" in arguments:
            context = arguments[arguments.index("--context") + 1]
            return json.dumps(self.node_names_by_context[context])
        if "pod" in arguments:
            return "fixture-ingress"
        if "exec" in arguments:
            return json.dumps(self.agent_owners)
        if any(argument.startswith("go-template=") for argument in arguments):
            return json.dumps(sorted(self.cpu["data"]))
        return self.command(arguments).stdout

    def command(
        self, arguments: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if "get" in arguments and "secret" in arguments:
            secret = self.gpu if "--context" in arguments else self.cpu
            return subprocess.CompletedProcess(arguments, 0, json.dumps(secret), "")
        if "patch" in arguments and "secret" in arguments:
            patch = json.loads(arguments[arguments.index("-p") + 1])
            self.key_patches.append(patch)
            candidate = copy.deepcopy(self.cpu)
            for operation in patch:
                parent, key = operation["path"].strip("/").split("/")
                if operation["op"] == "test" and (
                    candidate[parent].get(key) != operation["value"]
                ):
                    return subprocess.CompletedProcess(arguments, 1, "", "Conflict")
                if operation["op"] == "remove":
                    candidate[parent].pop(key)
            self.cpu = candidate
            self.cpu["metadata"]["resourceVersion"] = str(
                int(self.cpu["metadata"]["resourceVersion"]) + 1
            )
            return subprocess.CompletedProcess(arguments, 0, "", "")
        pytest.fail("unexpected external command in node-key removal fixture")

    def add_preserved_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        document = yaml.safe_load(self.path.read_text())
        target = dict(document["spec"]["clusters"][0])
        token = self.path.parent / "secure/gpu-b.token"
        token.write_text("b" * 64)
        token.chmod(0o600)
        target.update(
            clusterId="gpu-b",
            context="gpu-b",
            hyperpodClusterName="hp-gpu-b",
            eksClusterArn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
            executorIrsaRoleArn="arn:aws:iam::123456789012:role/executor-b",
            tokenFile=str(token),
        )
        document["spec"]["clusters"].append(target)
        self.path.write_text(yaml.safe_dump(document, sort_keys=False))
        self.node_names_by_context["gpu-b"] = ["node-b"]
        self.agent_owners["node-b"] = "gpu-b"
        self.cpu["data"]["node-b"] = encoded("preserved-b-fixture")
        monkeypatch.setattr(
            removal,
            "membership_runtime_snapshot",
            lambda _site: {
                "registry_generation": 1,
                "registry_content_sha256": "a" * 64,
                "live_release_identity_sha256": self.release_identity,
                "registry_cluster_states": {
                    **({"gpu-a": self.lifecycle} if self.lifecycle is not None else {}),
                    "gpu-b": "ACTIVE",
                },
            },
        )
        monkeypatch.setattr(join_nodes, "read_node_claims", read_node_claims)


@pytest.mark.parametrize("changed", ["value", "uid"])
def test_removal_retains_keys_changed_after_namespace_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    preserved = copy.deepcopy(scenario.cpu["data"])

    def namespace_absent(*_args: Any) -> None:
        scenario.event("namespace-wait")
        if changed == "value":
            scenario.cpu["data"]["node-a"] = encoded("foreign-fixture")
        else:
            scenario.cpu["metadata"]["uid"] = "replacement-cpu-keys"
        scenario.cpu["metadata"]["resourceVersion"] = "2"

    monkeypatch.setattr(removal, "_wait_target_namespace_absent", namespace_absent)

    with pytest.raises(BootstrapError, match="node-key"):
        removal.remove_cluster(scenario.request())

    assert scenario.key_patches == [], (
        "removal deleted a CPU key after its value or Secret incarnation changed"
    )
    assert "node-a" in scenario.cpu["data"], (
        "the replacement owner's same-name key must remain untouched"
    )
    assert scenario.cpu["data"]["node-other"] == preserved["node-other"], (
        "removal changed an unrelated cluster key"
    )


@pytest.mark.parametrize("source", ["nodes", "agents"])
@pytest.mark.parametrize("when", ["discovery", "after-cleanup"])
def test_removal_checks_actual_global_ownership_even_when_key_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, when: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    scenario.add_preserved_cluster(monkeypatch)

    def conflict() -> None:
        if source == "nodes":
            scenario.node_names_by_context["gpu-b"] = ["node-a"]
        else:
            scenario.agent_owners["node-a"] = "gpu-b"

    if when == "discovery":
        conflict()
    else:
        monkeypatch.setattr(
            removal, "_wait_target_namespace_absent", lambda *_a: conflict()
        )

    with pytest.raises(BootstrapError, match="owned by another cluster"):
        removal.remove_cluster(scenario.request())

    assert scenario.key_patches == [], (
        "global ownership conflict did not block deletion"
    )
    assert "unregister" not in scenario.calls, (
        "membership was revoked before resolving the NodeName ownership conflict"
    )
    assert "aws-network" not in scenario.calls, (
        "AWS cleanup began despite conflicting NodeName ownership"
    )
    if when == "discovery":
        assert "drain" not in scenario.calls, (
            "initial global ownership conflict was discovered after mutation"
        )


def test_removal_preserves_other_cluster_keys_and_records_private_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    scenario.add_preserved_cluster(monkeypatch)
    preserved = {
        node: value for node, value in scenario.cpu["data"].items() if node != "node-a"
    }

    result = removal.remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED"
    assert scenario.cpu["data"] == preserved, (
        "removal did not retain the complete unrelated CPU key map"
    )
    binding = scenario.state()["evidence"]["DISCOVERED"]["node_key_ownership"]
    assert set(binding) == {
        "cpu_secret_uid",
        "gpu_secret_uid",
        "expected_key_sha256",
    }, "removal evidence must contain only identities and digests"
    assert binding["cpu_secret_uid"] == "cpu-keys"
    assert set(binding["expected_key_sha256"]) == {"node-a"}
    assert len(binding["expected_key_sha256"]["node-a"]) == 64
    assert scenario.key_patches[0][:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "cpu-keys"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "1"},
    ], "node-key deletion must bind the exact read Secret incarnation and version"

    assert removal.remove_cluster(scenario.request())["phase"] == "COMPLETED"
    assert len(scenario.key_patches) == 1, (
        "completed removal retry submitted another key mutation"
    )


@pytest.mark.parametrize("ack", ["timeout", "failure", "readback-timeout"])
def test_removal_retries_a_committed_key_delete_without_collateral_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ack: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    preserved = {
        node: value for node, value in scenario.cpu["data"].items() if node != "node-a"
    }
    pending_error = True
    original_run = scenario.run

    def command(
        arguments: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        nonlocal pending_error
        result = scenario.command(arguments, **kwargs)
        if "patch" in arguments and pending_error:
            pending_error = False
            if ack == "failure":
                return subprocess.CompletedProcess(arguments, 1, "", "ACK lost")
            if ack == "timeout":
                raise subprocess.TimeoutExpired("fixture-key-delete", 1)
        return result

    def run(arguments: Sequence[str], **kwargs: Any) -> str:
        if ack == "readback-timeout" and scenario.key_patches and pending_error:
            raise TimeoutError("node-key readback timed out")
        return original_run(arguments, **kwargs)

    if ack == "readback-timeout":
        monkeypatch.setattr(scenario, "run", run)
    else:
        monkeypatch.setattr(removal, "run_command", command)

    with pytest.raises(BootstrapError):
        removal.remove_cluster(scenario.request())

    assert scenario.cpu["data"] == preserved, (
        "the simulated lost acknowledgement did not follow a committed deletion"
    )
    assert "CONTROL_REGISTRY_REMOVED" not in scenario.state()["completed_steps"]
    pending_error = False

    result = removal.remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED"
    assert len(scenario.key_patches) == 1, (
        "retry must confirm the prior deletion, not remove other keys"
    )
    assert scenario.cpu["data"] == preserved, "retry changed unrelated key material"


def test_removal_cas_conflict_preserves_a_concurrent_cluster_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    raced = False

    def command(
        arguments: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        nonlocal raced
        if "patch" in arguments and not raced:
            raced = True
            scenario.cpu["data"]["node-concurrent"] = encoded("concurrent-fixture")
            scenario.cpu["metadata"]["resourceVersion"] = "2"
        return scenario.command(arguments, **kwargs)

    monkeypatch.setattr(removal, "run_command", command)

    with pytest.raises(BootstrapError, match="failed to remove"):
        removal.remove_cluster(scenario.request())

    assert "node-a" in scenario.cpu["data"], "a failed CAS still deleted the target key"
    preserved = {
        node: value for node, value in scenario.cpu["data"].items() if node != "node-a"
    }

    assert removal.remove_cluster(scenario.request())["phase"] == "COMPLETED"
    assert scenario.cpu["data"] == preserved, (
        "key deletion retry overwrote a concurrent cluster's added key"
    )


@pytest.mark.parametrize(
    "changed", ["cpu-uid", "gpu-uid", "cpu-key", "gpu-key", "node"]
)
def test_removal_retry_cannot_refresh_its_original_key_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        removal.remove_cluster(scenario.request())
    original = copy.deepcopy(
        scenario.state()["evidence"]["DISCOVERED"]["node_key_ownership"]
    )
    if changed == "node":
        scenario.node_uid = "replacement-node"
    else:
        target = scenario.cpu if changed.startswith("cpu") else scenario.gpu
        if changed.endswith("uid"):
            target["metadata"]["uid"] += "-replacement"
        else:
            target["data"]["node-a"] = encoded("replacement-fixture")
    scenario.failure = None
    calls = list(scenario.calls)

    with pytest.raises(BootstrapError, match="changed|drifted"):
        removal.remove_cluster(scenario.request())

    assert scenario.calls == calls, "identity drift resumed destructive removal phases"
    assert scenario.key_patches == [], "identity drift authorized a node-key deletion"
    assert scenario.state()["evidence"]["DISCOVERED"]["node_key_ownership"] == (
        original
    ), "retry replaced the original ownership proof with a newly observed identity"


def test_removal_rejects_old_inflight_state_without_key_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        removal.remove_cluster(scenario.request())
    path = scenario.path.parent / "remove-cluster/gpu-a/state.json"
    state = json.loads(path.read_text())
    state["evidence"]["DISCOVERED"].pop("node_key_ownership")
    path.write_text(json.dumps(state))
    calls = list(scenario.calls)
    scenario.failure = None

    with pytest.raises(BootstrapError, match="no saved node-key"):
        removal.remove_cluster(scenario.request())

    assert scenario.calls == calls, "an unbound journal resumed removal"
    assert scenario.key_patches == [], "an unbound journal authorized key deletion"


@pytest.mark.parametrize(
    "invalid",
    [
        "cpu-absent",
        "gpu-absent",
        "cpu-unreadable",
        "gpu-unreadable",
        "gpu-pending",
        "gpu-extra",
        "gpu-missing",
        "cpu-missing",
        "cpu-conflict",
    ],
)
def test_removal_requires_complete_readable_key_proof_before_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    original_run = scenario.run

    def run(arguments: Sequence[str], **kwargs: Any) -> str:
        scope = "gpu" if "--context" in arguments else "cpu"
        if invalid == f"{scope}-absent":
            return ""
        if invalid == f"{scope}-unreadable":
            raise BootstrapError("node-key fixture read failed")
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(scenario, "run", run)
    if invalid == "gpu-pending":
        scenario.gpu["metadata"]["annotations"] = {
            "gpu-fault.io/node-action-key-rotation": "fixture-pending"
        }
    elif invalid == "gpu-extra":
        scenario.gpu["data"]["node-retired"] = encoded("retired-fixture")
    elif invalid == "gpu-missing":
        scenario.gpu["data"].clear()
    elif invalid == "cpu-missing":
        scenario.cpu["data"].pop("node-a")
    elif invalid == "cpu-conflict":
        scenario.cpu["data"]["node-a"] = encoded("foreign-fixture")

    with pytest.raises(BootstrapError, match="node-key"):
        removal.remove_cluster(scenario.request())

    assert scenario.calls == ["snapshot"], (
        "removal reached a mutation phase without complete node-key ownership"
    )
    assert scenario.key_patches == [], "invalid proof authorized a key deletion"


@pytest.mark.parametrize("changed", ["value", "uid"])
def test_removal_checks_key_ownership_again_after_unregister(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)

    def unregister(*_args: Any) -> None:
        scenario.event("unregister")
        if changed == "value":
            scenario.cpu["data"]["node-a"] = encoded("foreign-fixture")
        else:
            scenario.cpu["metadata"]["uid"] = "replacement-cpu-keys"

    monkeypatch.setattr(removal, "_run_control_plane_unregister", unregister)

    with pytest.raises(BootstrapError, match="node-key"):
        removal.remove_cluster(scenario.request())

    assert scenario.key_patches == [], (
        "the final key delete trusted ownership captured before unregister"
    )
    assert "node-a" in scenario.cpu["data"], "the replacement key was deleted"


def test_removal_checks_node_incarnation_again_after_namespace_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)

    def namespace_absent(*_args: Any) -> None:
        scenario.node_uid = "replacement-node"

    monkeypatch.setattr(removal, "_wait_target_namespace_absent", namespace_absent)

    with pytest.raises(BootstrapError, match="node incarnation changed"):
        removal.remove_cluster(scenario.request())

    assert scenario.key_patches == [], "a replaced node authorized CPU key deletion"
    assert "unregister" not in scenario.calls, (
        "removal continued after discovering a replaced node"
    )


def test_completed_removal_refuses_reappearing_keys_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    removal.remove_cluster(scenario.request())
    scenario.cpu["data"]["node-a"] = encoded("foreign-fixture")
    calls = list(scenario.calls)

    with pytest.raises(BootstrapError, match="reappeared"):
        removal.remove_cluster(scenario.request())

    assert scenario.calls == calls, "completed removal retry started new mutations"
    assert len(scenario.key_patches) == 1, "completed removal deleted a reappearing key"


@pytest.mark.parametrize("phase", ["cleanup", "unregister"])
def test_removal_supervision_loss_blocks_a_fresh_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)

    def lose_supervision(*_args: Any, **_kwargs: Any) -> None:
        scenario.event(phase)
        raise ProcessSupervisionLost("fixture process ownership is unproven")

    with monkeypatch.context() as failure:
        failure.setattr(
            removal,
            "_run_kubernetes_cleanup"
            if phase == "cleanup"
            else "_run_control_plane_unregister",
            lose_supervision,
        )
        with pytest.raises(ProcessSupervisionLost, match="ownership is unproven"):
            removal.remove_cluster(scenario.request())
    calls = list(scenario.calls)

    # The fixture did not poison the process-local supervisor event, matching
    # a new CLI process whose only previous-run evidence is the on-disk journal.
    with pytest.raises(ProcessSupervisionLost, match="automatic retry is forbidden"):
        removal.remove_cluster(scenario.request())

    assert scenario.calls == calls, (
        "a fresh removal invocation resumed mutations with unproven command ownership"
    )
    assert scenario.state()["phase"] == "SUPERVISION_LOST"
