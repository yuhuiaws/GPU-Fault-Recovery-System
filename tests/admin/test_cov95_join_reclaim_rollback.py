"""The rollback of a re-attach attempt undoes only what the attempt created.

Live sequence: a ``--reclaim-cluster-token`` attempt passed the token proof,
re-read the committed namespace and the node inventory (local inputs), then
failed on the node-name conflict rule; its rollback failed too ("GPU join has
no recorded namespace ownership") because the attempt rightly had not created
the namespace. A re-attach adopts the committed join's namespace, node keys,
IAM roles, network, token and kubeconfig context; its rollback verifies them
and leaves them standing, finishes as ROLLED_BACK, and the next attempt
resumes from that journal with the token binding and the reclaim proof.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster
from tests.admin.test_admin_cluster_join_rollback import (
    GPU_B_ARN,
    ROLE_ARN,
    Attempt,
    Commands,
    _target,
)

RECLAIM = {
    "cluster_id": "hp-gpu-b",
    "token_sha256": "e" * 64,
    "namespace_uid": "namespace-b",
    "identity": {"attempt": 1, "journal": "state.attempt-001.json"},
}


def _binding(attempt: Attempt) -> dict[str, str]:
    return {
        "path": str(attempt.token_file),
        "cluster_id": "hp-gpu-b",
        "eks_arn": _target().eks_arn,
        "hyperpod_arn": _target().hyperpod_arn,
    }


def _reclaim_attempt(tmp_path: Path, *, namespace_uid: str = "namespace-b") -> Attempt:
    """The live journal: a re-attach that stopped right after its local inputs."""

    attempt = Attempt(tmp_path)
    # The site's own token file is the reclaimed credential.
    attempt.token_file = attempt.site.source.parent / "secure" / "hp-gpu-b.token"
    attempt.token_file.parent.mkdir(mode=0o700, exist_ok=True)
    attempt.token_file.write_text("f" * 64, encoding="utf-8")
    attempt.token_file.chmod(0o600)
    attempt.joined = False
    attempt.released = False
    attempt.prepare_error = BootstrapError(
        "join NodeNames conflict with existing cluster, Agent, or CPU key ownership"
    )
    return _as_reclaim(attempt, namespace_uid=namespace_uid)


def _as_reclaim(attempt: Attempt, *, namespace_uid: str = "namespace-b") -> Attempt:
    attempt.reclaim_binding = _binding(attempt)  # type: ignore[attr-defined]
    local = attempt.evidence()["LOCAL_INPUTS_READY"]
    local.update(
        {
            "token_file": str(attempt.token_file),
            "namespace_creation_started": False,
            "namespace_uid": namespace_uid,
            "reclaimed_cluster_token": True,
            "kubeconfig_alias_replaced": {"context": _target().context},
        }
    )
    attempt.local_override = local  # type: ignore[attr-defined]
    return attempt


def _install_reclaim(
    attempt: Attempt,
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: list[str],
    evidence_keys: tuple[str, ...] = ("PRECHECKED", "DISCOVERED", "LOCAL_INPUTS_READY"),
    extra_evidence: dict[str, Any] | None = None,
    prepare_result: Any = None,
) -> list[dict[str, Any]]:
    """Patch ``_prepare_execution`` to leave the live journal shape behind."""

    attempt.install(monkeypatch)
    seen: list[dict[str, Any]] = []

    def prepare(_request: Any, **keywords: Any) -> Any:
        state = keywords["state"]
        seen.append(json.loads(json.dumps(state)))
        if prepare_result is not None:
            return prepare_result
        evidence = {
            key: value
            for key, value in attempt.evidence().items()
            if key in evidence_keys
        }
        evidence["LOCAL_INPUTS_READY"] = dict(attempt.local_override)  # type: ignore[attr-defined]
        evidence["LOCAL_INPUTS_STARTED"] = dict(evidence["LOCAL_INPUTS_READY"])
        evidence.update(extra_evidence or {})
        state["completed_steps"] = sorted(steps)
        state["evidence"] = evidence
        state["retained_cluster_token"] = dict(attempt.reclaim_binding)  # type: ignore[attr-defined]
        state["reclaimed_cluster_token"] = dict(RECLAIM)
        assert attempt.prepare_error is not None
        raise attempt.prepare_error

    monkeypatch.setattr(admin_cluster_join, "_prepare_execution", prepare)
    return seen


def _request(attempt: Attempt) -> JoinClusterRequest:
    return JoinClusterRequest(
        site=attempt.site,
        gpu_cluster_arn=GPU_B_ARN,
        state_dir=attempt.state_dir,
        reclaim_cluster_token=True,
    )


LIVE_STEPS = ["DISCOVERED", "LOCAL_INPUTS_READY", "LOCAL_INPUTS_STARTED", "PRECHECKED"]


def test_a_reclaim_that_created_nothing_rolls_back_to_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _reclaim_attempt(tmp_path)
    _install_reclaim(attempt, monkeypatch, steps=LIVE_STEPS)

    with pytest.raises(BootstrapError, match="NodeNames conflict"):
        join_cluster(_request(attempt), runner=SimpleNamespace(dry_run=False))

    state = attempt.state()
    assert state["phase"] == "ROLLED_BACK", state.get("rollback_errors")
    assert attempt.token_file.is_file(), "the live cluster's token file was deleted"
    assert state["retained_cluster_token"] == _binding(attempt)
    assert state["reclaimed_cluster_token"] == RECLAIM, (
        "the proof survives the rollback"
    )
    assert not attempt.commands.matching("delete-context"), (
        "the pre-existing kubeconfig context was removed"
    )
    assert not attempt.commands.namespace_deleted, "the adopted namespace was deleted"
    assert not attempt.commands.matching("prepare-clean-redeploy"), (
        "the installed data plane was cleaned"
    )
    assert attempt.rollouts == [], "no registry transition without a registration"
    assert set(state["completed_steps"]) == {"DISCOVERED", "PRECHECKED"}
    assert set(state["evidence"]) == {"DISCOVERED", "PRECHECKED"}


def test_a_reclaim_rollback_refuses_a_namespace_that_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _reclaim_attempt(tmp_path, namespace_uid="namespace-before")
    _install_reclaim(attempt, monkeypatch, steps=LIVE_STEPS)

    with pytest.raises(BootstrapError, match="NodeNames conflict") as failure:
        join_cluster(_request(attempt), runner=SimpleNamespace(dry_run=False))

    assert any(
        "namespace changed before rollback" in note for note in failure.value.__notes__
    ), failure.value.__notes__
    assert attempt.state()["phase"] == "ROLLBACK_STARTED"
    assert attempt.token_file.is_file(), "the live token file was deleted"
    assert not attempt.commands.namespace_deleted, "the adopted namespace was deleted"


def test_a_reclaim_rollback_requires_the_recorded_namespace_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _reclaim_attempt(tmp_path, namespace_uid="")
    _install_reclaim(attempt, monkeypatch, steps=LIVE_STEPS)

    with pytest.raises(BootstrapError) as failure:
        join_cluster(_request(attempt), runner=SimpleNamespace(dry_run=False))

    assert any(
        "no recorded namespace identity" in note for note in failure.value.__notes__
    ), failure.value.__notes__
    assert not attempt.commands.namespace_deleted, "a namespace was deleted unproven"


def test_the_next_attempt_resumes_the_rollback_and_keeps_both_token_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the live journal (ROLLBACK_STARTED, pre-release steps, reclaim proof)
    the retry finishes the rollback, archives the attempt and starts the next
    one with the token binding and the reclaim proof carried forward."""

    attempt = _reclaim_attempt(tmp_path)
    attempt.install(monkeypatch)
    local = dict(attempt.local_override)  # type: ignore[attr-defined]
    journal = {
        "schema_version": 1,
        "site_id": attempt.site.release_config["site_name"],
        "gpu_cluster_arn": GPU_B_ARN,
        "requested_cluster_id": "",
        "allowed_namespaces": ["gpu-fault-system", "training"],
        "attempt": 4,
        "source_site_sha256": attempt.site.source_sha256,
        "phase": "ROLLBACK_STARTED",
        "completed_steps": LIVE_STEPS,
        "evidence": {
            "DISCOVERED": attempt.evidence()["DISCOVERED"],
            "LOCAL_INPUTS_STARTED": local,
            "LOCAL_INPUTS_READY": local,
        },
        "retained_cluster_token": _binding(attempt),
        "reclaimed_cluster_token": dict(RECLAIM),
        "superseded_attempts": [{"attempt": 3, "phase": "BLOCKED_IDENTITY"}],
        "failure": {"error": "BootstrapError: join NodeNames conflict"},
        "updated_at": "2000-01-01T00:00:00+00:00",
    }
    (attempt.state_dir / "state.json").write_text(json.dumps(journal), encoding="utf-8")
    seen: list[dict[str, Any]] = []

    def prepare(_request: Any, **keywords: Any) -> Any:
        seen.append(json.loads(json.dumps(keywords["state"])))
        return {"phase": "RETRY"}

    monkeypatch.setattr(admin_cluster_join, "_prepare_execution", prepare)
    # The resumed rollback re-confirms the recorded target before it acts.
    monkeypatch.setattr(
        admin_cluster_join, "discover_cluster", lambda *_args, **_kwargs: _target()
    )

    result = join_cluster(_request(attempt), runner=SimpleNamespace(dry_run=False))

    assert result == {"phase": "RETRY"}
    assert len(seen) == 1
    fresh = seen[0]
    assert fresh["attempt"] == 5
    assert fresh["phase"] == "STARTED"
    assert fresh["retained_cluster_token"] == _binding(attempt)
    assert fresh["reclaimed_cluster_token"] == RECLAIM
    archive = json.loads(
        (attempt.state_dir / "state.attempt-004.json").read_text(encoding="utf-8")
    )
    assert archive["phase"] == "ROLLED_BACK"
    assert attempt.token_file.is_file(), "the live token file was deleted"
    assert not attempt.commands.matching("delete-context"), "the context was removed"
    assert not attempt.commands.namespace_deleted, "the adopted namespace was deleted"


def test_a_reclaim_rollback_leaves_adopted_roles_and_keys_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _reclaim_attempt(tmp_path)
    attempt.prepare_error = BootstrapError("candidate preflight failed")
    removed: list[tuple[list[str], dict[str, str] | None]] = []
    prerequisites = attempt.evidence()["PREREQUISITES_READY"]
    prerequisites["executor_role"]["pre_existing"] = True
    prerequisites["adot_writer_role"] = {
        "role_arn": ROLE_ARN + "-adot",
        "ownership": "CREATED",
        "inline_policy_name": "GPUFaultDataplaneAmpWriter",
        "pre_existing": True,
    }
    prerequisites["network"].update(
        {"association_created": False, "created_ingress_eips": []}
    )
    verified = attempt.evidence()["NODE_NAMES_VERIFIED"]
    verified["adopted_node_keys"] = {
        "node-b": {"agent_owner": "hp-gpu-b", "key_sha256": "a" * 64}
    }
    _install_reclaim(
        attempt,
        monkeypatch,
        steps=[
            *LIVE_STEPS,
            "NODE_NAMES_VERIFIED",
            "NODE_KEYS_STARTED",
            "PREREQUISITES_READY",
        ],
        extra_evidence={
            "PREREQUISITES_READY": prerequisites,
            "NODE_NAMES_VERIFIED": verified,
        },
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_remove_node_action_keys",
        lambda _site, nodes, *, expected_key_sha256=None: removed.append(
            (list(nodes), expected_key_sha256)
        ),
    )

    with pytest.raises(BootstrapError, match="preflight failed"):
        join_cluster(_request(attempt), runner=SimpleNamespace(dry_run=False))

    state = attempt.state()
    assert state["phase"] == "ROLLED_BACK", state.get("rollback_errors")
    assert not attempt.commands.matching("iam delete-role"), (
        "a pre-existing role was deleted by the attempt that only found it"
    )
    assert not attempt.commands.matching("disassociate-vpc"), (
        "an association the attempt did not create was removed"
    )
    assert not attempt.commands.matching("revoke-security-group-ingress"), (
        "an NLB rule the attempt did not create was revoked"
    )
    assert removed == [], "adopted node keys were removed"
    assert attempt.cleared == [], "adopted nodes were re-annotated"
    assert attempt.token_file.is_file(), "the live token file was deleted"


def test_a_reclaim_rollback_still_removes_the_keys_the_attempt_minted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One adopted node, one node the attempt provisioned itself: only the
    second one's key is withdrawn, under its own digest proof."""

    attempt = _reclaim_attempt(tmp_path)
    attempt.prepare_error = BootstrapError("candidate preflight failed")
    removed: list[tuple[list[str], dict[str, str] | None]] = []
    local = attempt.local_override  # type: ignore[attr-defined]
    local["nodes"] = ["node-b", "node-new"]
    attempt.commands = Commands(
        outputs=[
            (
                "get nodes",
                json.dumps(
                    {
                        "items": [
                            {
                                "kind": "Node",
                                "metadata": {
                                    "name": name,
                                    "uid": f"uid-{name}",
                                    "resourceVersion": "1",
                                    "labels": {
                                        "sagemaker.amazonaws.com/cluster-name": "hp-gpu-b"
                                    },
                                },
                            }
                            for name in ("node-b", "node-new")
                        ]
                    }
                ),
            )
        ]
    )
    local["node_uids"] = {"node-b": "uid-node-b", "node-new": "uid-node-new"}
    prerequisites = attempt.evidence()["PREREQUISITES_READY"]
    prerequisites["executor_role"]["pre_existing"] = True
    prerequisites["network"].update(
        {"association_created": False, "created_ingress_eips": []}
    )
    verified = attempt.evidence()["NODE_NAMES_VERIFIED"]
    verified.update(
        {
            "nodes": ["node-b", "node-new"],
            "node_uids": dict(local["node_uids"]),
            "expected_key_sha256": {"node-b": "a" * 64, "node-new": "b" * 64},
            "adopted_node_keys": {
                "node-b": {"agent_owner": "hp-gpu-b", "key_sha256": "a" * 64}
            },
        }
    )
    _install_reclaim(
        attempt,
        monkeypatch,
        steps=[
            *LIVE_STEPS,
            "NODE_NAMES_VERIFIED",
            "NODE_KEYS_STARTED",
            "PREREQUISITES_READY",
        ],
        extra_evidence={
            "PREREQUISITES_READY": prerequisites,
            "NODE_NAMES_VERIFIED": verified,
        },
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_remove_node_action_keys",
        lambda _site, nodes, *, expected_key_sha256=None: removed.append(
            (list(nodes), expected_key_sha256)
        ),
    )

    with pytest.raises(BootstrapError, match="preflight failed"):
        join_cluster(_request(attempt), runner=SimpleNamespace(dry_run=False))

    assert attempt.state()["phase"] == "ROLLED_BACK"
    assert removed == [(["node-new"], {"node-new": "b" * 64})]


def test_an_ordinary_attempt_rollback_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a reclaim record the rollback still deletes the context and the
    attempt's own token, and roles it created, exactly as before."""

    attempt = Attempt(tmp_path)
    attempt.joined = False
    attempt.released = False
    attempt.run(monkeypatch, error="cluster deploy failed")

    state = attempt.state()
    assert state["phase"] == "ROLLED_BACK"
    assert attempt.commands.matching("delete-context"), (
        "an ordinary rollback drops its context"
    )
    assert attempt.commands.matching("iam delete-role"), (
        "an ordinary rollback deletes its role"
    )
    assert not attempt.token_file.exists(), "an ordinary rollback deletes its token"
    assert "reclaimed_cluster_token" not in state
    assert asdict(_target())["eks_arn"] == GPU_B_ARN
