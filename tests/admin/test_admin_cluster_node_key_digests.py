"""Every reader of a node-key digest map refuses the same malformed digests.

A node-key digest is what ``hashlib``'s ``hexdigest`` writes: 64 lowercase hex
characters. ``gpu_fault.admin.cluster_node_key_digests`` is the one validator,
and the four readers of a saved digest map -- the join retry
(``verify_join_node_names``), the join rollback (``join_cluster``'s
compensation), the removal (``remove_cluster``) and the CPU key delete both
compensations end in (``_remove_node_action_keys``) -- take the same variants
of their own valid digest below and refuse each where they read it, before
anything is mutated. The join rollback used to accept any 64-character string,
so an uppercase or non-hex digest passed its ownership proof and the undo steps
after it, and only the key delete at the end refused it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_nodes as join_nodes
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster
from gpu_fault.admin.cluster_join_state import complete_step
from gpu_fault.admin.cluster_node_key_digests import (
    NAMED_NODES,
    is_node_key_digest,
    node_key_digests,
)
from gpu_fault.admin.cluster_removal import _remove_node_action_keys as remove_keys
from gpu_fault.admin.node_key_proof import node_key_digest
from tests.admin.test_admin_cluster_join_rollback import GPU_B_ARN, Attempt
from tests.admin.test_admin_cluster_removal_keys import NodeKeyScenario
from tests.admin.test_cov95_join_nodes import context as context
from tests.admin.test_cov95_join_nodes import key_document, verify

#: What turns a valid digest into one no reader may take.
MALFORMED: dict[str, Callable[[str], str]] = {
    "uppercase": str.upper,
    "63-characters": lambda digest: digest[:63],
    "65-characters": lambda digest: digest + "0",
    "non-hex": lambda digest: digest[:63] + "g",
    "trailing-newline": lambda digest: digest + "\n",
}
VARIANTS = pytest.mark.parametrize("variant", list(MALFORMED))
DIGEST = hashlib.sha256(b"node-key-fixture").hexdigest()


def malformed(variant: str, digest: str) -> str:
    value = MALFORMED[variant](digest)
    assert is_node_key_digest(digest) and value != digest, (variant, digest)
    return value


@VARIANTS
def test_only_the_form_hexdigest_writes_is_a_node_key_digest(variant: str) -> None:
    value = malformed(variant, DIGEST)

    assert not is_node_key_digest(value), variant
    with pytest.raises(BootstrapError) as refused:
        node_key_digests(
            {"node-a": DIGEST, "node-b": value},
            ["node-a", "node-b"],
            refusal="fixture refusal",
        )
    message = str(refused.value)
    assert message == (
        "fixture refusal: the node-key digest of node-b is not a SHA-256 hex "
        "digest (64 lowercase hex characters)"
    ), message
    assert DIGEST[:16] not in message, "a refusal never echoes a digest value"


def test_a_digest_map_must_name_exactly_the_proved_nodes() -> None:
    assert node_key_digests({"node-a": DIGEST}, ["node-a"], refusal="r") == {
        "node-a": DIGEST
    }
    assert not is_node_key_digest(None) and not is_node_key_digest(int(DIGEST, 16))
    for value, reason in (
        (None, "no node-key digest map"),
        ([DIGEST], "no node-key digest map"),
        ({}, "the node-key digest map has no digest for node-a"),
        (
            {"node-a": DIGEST, "node-x": DIGEST},
            "the node-key digest map has a digest for unproved node-x",
        ),
        (
            {"node-x": DIGEST},
            "the node-key digest map has no digest for node-a; a digest for "
            "unproved node-x",
        ),
        (
            {"node-a": None},
            "the node-key digest of node-a is not a SHA-256 hex digest "
            "(64 lowercase hex characters)",
        ),
    ):
        with pytest.raises(BootstrapError) as refused:
            node_key_digests(value, ["node-a"], refusal="r")
        assert str(refused.value) == f"r: {reason}", value


def test_a_refusal_names_a_few_nodes_and_counts_the_rest() -> None:
    nodes = [f"node-{index:02d}" for index in range(NAMED_NODES + 3)]

    with pytest.raises(BootstrapError) as refused:
        node_key_digests(dict.fromkeys(nodes, DIGEST.upper()), nodes, refusal="r")

    assert str(refused.value) == (
        f"r: the node-key digest of {', '.join(nodes[:NAMED_NODES])} and 3 more "
        "is not a SHA-256 hex digest (64 lowercase hex characters)"
    ), str(refused.value)


@VARIANTS
def test_a_join_retry_refuses_a_saved_proof_with_a_malformed_digest(
    context: Any,  # noqa: F811 -- the join-nodes fixture imported above
    variant: str,
) -> None:
    runner, path, state = context[2:]
    runner.secret = key_document(b"b" * 64)
    verify(context)
    complete_step(path, state, "NODE_KEYS_STARTED")
    proof = state["evidence"]["NODE_NAMES_VERIFIED"]
    value = malformed(variant, proof["expected_key_sha256"]["node-b"])
    proof["expected_key_sha256"] = {"node-b": value}

    with pytest.raises(
        BootstrapError,
        match="^join retry has no complete node-key digest proof: the node-key "
        "digest of node-b is not",
    ):
        verify(
            context, claims=join_nodes.NodeClaims(cpu_key_names=frozenset({"node-b"}))
        )

    assert state["evidence"]["NODE_NAMES_VERIFIED"]["expected_key_sha256"] == {
        "node-b": value
    }, "a refused proof is never replaced by newly observed digests"


@VARIANTS
def test_a_join_rollback_refuses_a_malformed_digest_before_any_undo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    attempt = Attempt(tmp_path)
    proof = attempt.evidence()["NODE_NAMES_VERIFIED"]
    value = malformed(variant, proof["expected_key_sha256"]["node-b"])
    attempt.extra_evidence = {
        "NODE_NAMES_VERIFIED": {**proof, "expected_key_sha256": {"node-b": value}}
    }
    attempt.install(monkeypatch)
    removed: list[Any] = []
    monkeypatch.setattr(
        admin_cluster_join,
        "_remove_node_action_keys",
        lambda *args, **keywords: removed.append((args, keywords)),
    )

    with pytest.raises(
        BootstrapError,
        match="join rollback has no complete node/key ownership proof: the "
        "node-key digest of node-b is not",
    ):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=SimpleNamespace(dry_run=False),
        )

    state = attempt.state()
    assert state["phase"] == "ROLLBACK_FAILED", state["phase"]
    assert any("node-b" in error for error in state["rollback_errors"]), state
    assert removed == [], "the CPU key delete was reached on an unproven digest"
    assert attempt.rollouts == [], "the registry entry was failed before the proof"
    assert attempt.cleared == [], "installer annotations were cleared"
    assert not attempt.commands.matching("prepare-clean-redeploy.sh"), (
        "the GPU cleanup ran on an unproven key map"
    )
    assert not attempt.commands.matching("delete-role"), "the IAM undo ran"
    assert attempt.membership == [], "membership was undone"
    assert attempt.token_file.is_file(), "the join token is kept for the retry"


@VARIANTS
def test_a_removal_resume_refuses_a_saved_proof_with_a_malformed_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        removal.remove_cluster(scenario.request())
    path = scenario.path.parent / "remove-cluster/gpu-a/state.json"
    state = json.loads(path.read_text())
    binding = state["evidence"]["DISCOVERED"]["node_key_ownership"]
    value = malformed(variant, binding["expected_key_sha256"]["node-a"])
    binding["expected_key_sha256"] = {"node-a": value}
    path.write_text(json.dumps(state))
    calls = list(scenario.calls)
    scenario.failure = None

    with pytest.raises(
        BootstrapError,
        match="^removal has no complete node-key digest proof: the node-key "
        "digest of node-a is not",
    ):
        removal.remove_cluster(scenario.request())

    assert scenario.calls == calls, "a malformed digest resumed the removal"
    assert scenario.key_patches == [], "a malformed digest authorized a key deletion"


@VARIANTS
def test_the_cpu_key_delete_refuses_a_malformed_digest_and_keeps_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    scenario = NodeKeyScenario(tmp_path, monkeypatch)
    site = scenario.request().site
    digest = node_key_digest(scenario.cpu["data"]["node-a"])

    with pytest.raises(
        BootstrapError,
        match="^node-key deletion has an incomplete digest proof: the node-key "
        "digest of node-a is not",
    ):
        remove_keys(
            site, ["node-a"], expected_key_sha256={"node-a": malformed(variant, digest)}
        )

    assert scenario.key_patches == [], "a malformed digest authorized a key deletion"
    assert "node-a" in scenario.cpu["data"], "the key was deleted"
    assert remove_keys(site, ["node-a"], expected_key_sha256={"node-a": digest}) == 1
    assert "node-a" not in scenario.cpu["data"], (
        "the digest the Secret holds still deletes its key"
    )
