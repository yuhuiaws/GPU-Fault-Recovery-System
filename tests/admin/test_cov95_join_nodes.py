from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault.admin import cluster_join_nodes as nodes
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinInputs
from gpu_fault.admin.cluster_join_state import complete_step
from gpu_fault.admin.site import load_site
from tests.admin._cov95_join_support import target
from tests.admin.test_admin_site import site_file


class Inventory:
    def __init__(self):
        self.calls = []
        self.names = '["node-a"]'
        self.keys = ""
        self.agents = "{}"
        self.pod = "cpu-fixture"
        self.secret = ""
        self.target = {
            "items": [
                {
                    "kind": "Node",
                    "metadata": {
                        "name": "node-b",
                        "uid": "uid-b",
                        "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-b"},
                    },
                }
            ]
        }

    def run(self, arguments, **options):
        self.calls.append((arguments, options))
        if "get" in arguments and "nodes" in arguments:
            return (
                self.names
                if any(item.startswith("go-template=") for item in arguments)
                else json.dumps(self.target)
            )
        if "get" in arguments and "secret" in arguments:
            return (
                self.keys
                if any(item.startswith("go-template=") for item in arguments)
                else self.secret
            )
        if "get" in arguments and "pod" in arguments:
            return self.pod
        if "exec" in arguments:
            return self.agents
        raise AssertionError("unexpected node inventory command")


@pytest.fixture
def context(tmp_path):
    site = load_site(site_file(tmp_path))
    runner = Inventory()
    inputs = JoinInputs(
        target(),
        "gpu-b",
        {},
        {
            "nodes": ["node-b"],
            "node_uids": {"node-b": "uid-b"},
            "fleet_master_file": site.release_config["clusters"][0][
                "fleet_master_file"
            ],
            "gpu_kubeconfig": str(tmp_path / "fake-kubeconfig"),
        },
    )
    return (
        JoinClusterRequest(site, target().eks_arn),
        inputs,
        runner,
        tmp_path / "state.json",
        {"completed_steps": [], "evidence": {}},
    )


def verify(context, *, claims=None):
    request, inputs, runner, path, state = context
    nodes.verify_join_node_names(
        request,
        inputs,
        claims=claims or nodes.NodeClaims(),
        state_path=path,
        state=state,
        runner=runner,
    )


@pytest.mark.parametrize(
    "field,output",
    [
        ("names", "invalid"),
        ("names", "{}"),
        ("names", '["same","same"]'),
        ("names", '[""]'),
        ("keys", "invalid"),
        ("keys", "{}"),
        ("agents", "invalid"),
        ("agents", "[]"),
        ("agents", '{"node":""}'),
        ("agents", '{"": "cluster"}'),
        ("pod", ""),
    ],
)
def test_read_claims_rejects_unknown_inventory(context, field, output):
    request, _inputs, runner, _path, _state = context
    setattr(runner, field, output)
    with pytest.raises(BootstrapError, match="inventory|ownership"):
        nodes.read_node_claims(request.site, runner)


def test_read_claims_includes_all_planes_and_uses_bounded_private_key_reads(context):
    request, _inputs, runner, _path, _state = context
    runner.keys = '["node-a"]'
    runner.agents = '{"node-a":"gpu-a"}'
    claims = nodes.read_node_claims(request.site, runner)
    assert claims.nodes_by_cluster == {"gpu-a": frozenset({"node-a"})}
    assert claims.cpu_key_names == frozenset({"node-a"})
    assert claims.agent_owners == {"node-a": "gpu-a"}
    assert all(
        0 < options["timeout_seconds"] <= 60 for _arguments, options in runner.calls
    ), "node ownership reads lost their bounded timeout"
    assert all(
        options.get("sensitive")
        for arguments, options in runner.calls
        if "secret" in arguments or "exec" in arguments
    ), "node ownership reads exposed private capture"


@pytest.mark.parametrize(
    "document", [{}, {"items": []}, {"items": [None]}, {"items": [{"metadata": {}}]}]
)
def test_target_inventory_requires_complete_node_list(context, document):
    _request, inputs, runner, _path, _state = context
    runner.target = document
    with pytest.raises(BootstrapError, match="missing or conflicting"):
        nodes.read_target_node_inventory(
            runner, Path(inputs.local["gpu_kubeconfig"]), inputs.target
        )


@pytest.mark.parametrize("kind", ["duplicate", "kind", "name", "uid", "cluster"])
def test_target_inventory_rejects_each_identity_ambiguity(context, kind):
    _request, inputs, runner, _path, _state = context
    item = runner.target["items"][0]
    if kind == "duplicate":
        runner.target["items"].append(copy.deepcopy(item))
    elif kind == "kind":
        item["kind"] = "Pod"
    elif kind == "cluster":
        item["metadata"]["labels"]["sagemaker.amazonaws.com/cluster-name"] = "foreign"
    else:
        item["metadata"][kind] = ""
    with pytest.raises(BootstrapError, match="missing or conflicting"):
        nodes.read_target_node_inventory(
            runner, Path(inputs.local["gpu_kubeconfig"]), inputs.target
        )


@pytest.mark.parametrize(
    "claims",
    [
        nodes.NodeClaims(nodes_by_cluster={"foreign": frozenset({"node-b"})}),
        nodes.NodeClaims(agent_owners={"node-b": "foreign"}),
        nodes.NodeClaims(cpu_key_names=frozenset({"node-b"})),
    ],
)
def test_foreign_name_claims_fail_before_reading_or_writing_keys(context, claims):
    with pytest.raises(BootstrapError, match="conflict"):
        verify(context, claims=claims)
    assert context[2].calls == []
    assert not context[3].exists(), "foreign NodeName claim created ownership evidence"


@pytest.mark.parametrize(
    "updates",
    [
        {"nodes": []},
        {"nodes": "node-b"},
        {"nodes": ["node-b", "node-b"]},
        {"nodes": [None]},
        {"node_uids": []},
        {"node_uids": {}},
        {"node_uids": {"node-b": ""}},
    ],
)
def test_join_ownership_requires_exact_node_uid_set(context, updates):
    context[1].local.update(updates)
    with pytest.raises(BootstrapError, match="inventory"):
        verify(context)
    assert not context[2].calls, "invalid node inventory reached key reads"
    assert not context[3].exists(), "invalid node inventory created ownership evidence"


def key_document(value):
    return json.dumps(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "Opaque",
            "metadata": {
                "name": "gpu-fault-node-action-keys",
                "namespace": "gpu-fault-system",
                "uid": "keys-uid",
                "resourceVersion": "1",
            },
            "data": {"node-b": base64.b64encode(value).decode()},
        }
    )


def test_existing_gpu_key_bytes_are_preserved_in_the_original_ownership_proof(context):
    context[2].secret = key_document(b"b" * 64)
    verify(context)
    proof = context[4]["evidence"]["NODE_NAMES_VERIFIED"]
    assert proof["expected_key_sha256"] == {
        "node-b": hashlib.sha256(b"b" * 64).hexdigest()
    }
    assert proof["node_uids"] == {"node-b": "uid-b"}
    assert (
        json.loads(context[3].read_text())["evidence"]["NODE_NAMES_VERIFIED"] == proof
    )


@pytest.mark.parametrize(
    "drift", ["missing", "empty", "malformed", "gpu-bytes", "none"]
)
def test_resumed_key_proof_cannot_be_replaced_with_new_observed_values(context, drift):
    runner, path, state = context[2:]
    runner.secret = key_document(b"b" * 64)
    verify(context)
    complete_step(path, state, "NODE_KEYS_STARTED")
    saved = state["evidence"]["NODE_NAMES_VERIFIED"]
    if drift == "missing":
        saved.pop("expected_key_sha256")
    elif drift == "empty":
        saved["expected_key_sha256"] = {}
    elif drift == "malformed":
        saved["expected_key_sha256"] = {"node-b": "x" * 64}
    elif drift == "gpu-bytes":
        runner.secret = key_document(b"c" * 64)
    claims = nodes.NodeClaims(cpu_key_names=frozenset({"node-b"}))
    if drift != "none":
        with pytest.raises(BootstrapError, match="digest proof|keys changed"):
            verify(context, claims=claims)
    else:
        verify(context, claims=claims)
        assert state["evidence"]["NODE_NAMES_VERIFIED"]["expected_key_sha256"] == {
            "node-b": hashlib.sha256(b"b" * 64).hexdigest()
        }


def test_same_batch_node_name_cannot_belong_to_two_clusters(context):
    inputs = context[1]
    with pytest.raises(BootstrapError, match="globally conflicting"):
        nodes.assert_batch_node_names_unique(
            [inputs, replace(inputs, cluster_id="gpu-c")]
        )
    nodes.assert_batch_node_names_unique([inputs, inputs])


def test_bound_key_runner_forwards_only_the_verified_mapping(context):
    runner = context[2]
    wrapper = nodes.BoundNodeKeyRunner(runner, {"node-b": "uid-b"})
    environment = {
        "EXAMPLE": "kept",
        nodes.EXPECTED_NODES_ENV: "untrusted",
        "GPU_FAULT_ROTATE_NODE_ACTION_KEY": "foreign",
    }
    wrapper.run(
        ["kubectl", "get", "pod"],
        env=environment,
        capture=False,
        sensitive=True,
        mutate=True,
        timeout_seconds=9,
    )
    arguments, options = runner.calls[-1]
    assert arguments == ["kubectl", "get", "pod"]
    assert json.loads(options["env"][nodes.EXPECTED_NODES_ENV]) == {"node-b": "uid-b"}
    assert options["env"]["GPU_FAULT_ROTATE_NODE_ACTION_KEY"] == ""
    assert options["env"]["EXAMPLE"] == "kept"
    assert environment[nodes.EXPECTED_NODES_ENV] == "untrusted"
    assert options["capture"] is False and options["sensitive"] and options["mutate"]
    assert options["timeout_seconds"] == 9
