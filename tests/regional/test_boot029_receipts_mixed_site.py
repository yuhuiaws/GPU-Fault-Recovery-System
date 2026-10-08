"""BOOT-029 receipts for a mixed-provider first deploy: the in-stage joins, the
first-deploy state a resume reads, and the Runtime Profile plan a join stops for."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.profile_approval import profile_plan_sha256
from scripts.e2e.regional import boot029_receipts as receipts

CPU = "arn:aws:eks:us-west-2:123456789012:cluster/cpu"
HYPERPOD_GPU = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-hyperpod"
EKS_GPU = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-eks"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _first_deploy(tmp_path: Path, gpu_arns: list[str], *, phase: str = "site-ready"):
    state = tmp_path / "state"
    record = state / "acceptance/receipt.json"
    receipts.initialize(record, inputs={"cpu_cluster_arn": CPU}, stage=1)
    receipts.start_stage(record, state=state, stage=1)
    write_json_atomic(
        state / "bootstrap-state.json",
        {
            "phase": phase,
            "resources": {
                "initial_deploy_target": {
                    "cpu": {"input_arn": CPU, "eks_arn": CPU, "hyperpod_arn": ""},
                    "gpu_clusters": [
                        {"input_arn": arn, "eks_arn": arn, "hyperpod_arn": ""}
                        for arn in gpu_arns
                    ],
                }
            },
        },
    )
    return record, state


def _join_journal(state: Path, arn: str, *, phase: str = "COMPLETED") -> Path:
    path = state / "join-cluster" / _sha256(arn)[:12] / "state.json"
    write_json_atomic(path, {"phase": phase, "attempt": 1, "gpu_cluster_arn": arn})
    return path


def _plan(**overrides: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": 1,
        "site_identity": {
            "site_name": "s",
            "aws_region": "us-west-2",
            "cpu_eks_arn": CPU,
        },
        "site_identity_sha256": "0" * 64,
        "registration_cluster_id": "gpu-eks",
        "current_version": "hyperpod-v1",
        "desired_version": "eks-v1",
        "current_policy_digest": "1" * 64,
        "policy_digest": "2" * 64,
        "live_profile_sha256": None,
        "active_source_sha256": None,
        "source_sha256": "3" * 64,
        "snapshot_sha256": "4" * 64,
        "change_kind": "CLUSTER_PROFILE_NEW",
        "changes": [
            "cluster gpu-eks (eks-managed) runs its own Runtime Profile eks-v1"
        ],
        "approval_required": True,
    }
    plan["plan_sha256"] = profile_plan_sha256(plan)
    plan.update(overrides)
    return plan


def test_a_mixed_first_deploy_proof_binds_the_deploy_and_every_in_stage_join(
    tmp_path: Path,
) -> None:
    record, state = _first_deploy(tmp_path, [HYPERPOD_GPU])
    journal = _join_journal(state, EKS_GPU)

    proof = receipts.verify_journal(
        record,
        state=state,
        stage=1,
        gpu_arns=[HYPERPOD_GPU],
        cpu_arn=CPU,
        joined_arns=[EKS_GPU],
    )

    assert proof["phase"] == "site-ready", proof
    assert proof["first_deploy_gpu_cluster_count"] == 1, proof
    assert proof["gpu_cluster_count"] == 2, "the run manages both clusters"
    (join,) = proof["joins"]
    assert join["gpu_arn_sha256"] == _sha256(EKS_GPU), join
    assert join["phase"] == "COMPLETED", join
    assert join["journal_sha256"] == _sha256(journal.read_text(encoding="utf-8")), (
        "the join proof binds the journal's bytes"
    )
    serialized = json.dumps(proof)
    assert EKS_GPU not in serialized and HYPERPOD_GPU not in serialized, (
        "ARNs appear as digests only"
    )


def test_a_single_provider_proof_keeps_its_shape(tmp_path: Path) -> None:
    record, state = _first_deploy(tmp_path, [HYPERPOD_GPU])

    proof = receipts.verify_journal(
        record, state=state, stage=1, gpu_arns=[HYPERPOD_GPU], cpu_arn=CPU
    )

    assert proof["gpu_cluster_count"] == 1, proof
    assert proof["first_deploy_gpu_cluster_count"] == 1, proof
    assert "joins" not in proof, "no in-stage join on a single-provider site"


def test_a_missing_or_unfinished_join_journal_fails_the_first_deploy_proof(
    tmp_path: Path,
) -> None:
    record, state = _first_deploy(tmp_path, [HYPERPOD_GPU])
    with pytest.raises(ValueError, match="joined GPU cluster 1/1; found 0"):
        receipts.verify_journal(
            record,
            state=state,
            stage=1,
            gpu_arns=[HYPERPOD_GPU],
            cpu_arn=CPU,
            joined_arns=[EKS_GPU],
        )
    _join_journal(state, EKS_GPU, phase="ROLLED_BACK")
    with pytest.raises(ValueError, match="joined GPU cluster 1/1; found 0"):
        receipts.verify_journal(
            record,
            state=state,
            stage=1,
            gpu_arns=[HYPERPOD_GPU],
            cpu_arn=CPU,
            joined_arns=[EKS_GPU],
        )


def test_the_first_deploy_must_name_the_first_provider_only(tmp_path: Path) -> None:
    """A checkpoint naming both clusters is not the one-provider first deploy
    the mixed sequence performs; the proof is bound to the deploy's own set."""
    record, state = _first_deploy(tmp_path, [HYPERPOD_GPU, EKS_GPU])
    _join_journal(state, EKS_GPU)

    with pytest.raises(ValueError, match="first deploy; found 0"):
        receipts.verify_journal(
            record,
            state=state,
            stage=1,
            gpu_arns=[HYPERPOD_GPU],
            cpu_arn=CPU,
            joined_arns=[EKS_GPU],
        )


def test_first_deploy_state_is_pending_complete_or_conflict(tmp_path: Path) -> None:
    state = tmp_path / "empty"
    state.mkdir()
    assert receipts.first_deploy_state(state, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU]) == (
        "pending"
    ), "an empty directory has everything ahead of it"

    _, unfinished = _first_deploy(tmp_path, [HYPERPOD_GPU], phase="new")
    assert (
        receipts.first_deploy_state(unfinished, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU])
        == "pending"
    ), "a checkpoint short of site-ready still deploys"

    _, done = _first_deploy(tmp_path / "done", [HYPERPOD_GPU])
    assert receipts.first_deploy_state(done, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU]) == (
        "complete"
    ), "site-ready naming exactly these clusters: the deploy is behind us"
    assert (
        receipts.first_deploy_state(done, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU, EKS_GPU])
        == "pending"
    ), "a site-ready checkpoint for another set is not this deploy"

    request = {
        "schema_version": 1,
        "state_dir": str(unfinished),
        "cpu_cluster_arn": CPU,
        "gpu_cluster_arns": [HYPERPOD_GPU, EKS_GPU],
        "admin_email": "ops@example.com",
    }
    (unfinished / "initial-deploy-request.json").write_text(json.dumps(request))
    assert (
        receipts.first_deploy_state(unfinished, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU])
        == "conflict"
    ), "the product refuses to change a recorded first-deploy request"
    assert (
        receipts.first_deploy_state(
            unfinished, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU, EKS_GPU]
        )
        == "pending"
    ), "the same request is no conflict"
    (unfinished / "initial-deploy-request.json").write_text("not json")
    assert (
        receipts.first_deploy_state(unfinished, cpu_arn=CPU, gpu_arns=[HYPERPOD_GPU])
        == "pending"
    ), "an unreadable request is left to the product to judge"


def test_pending_profile_plan_accepts_only_a_verifying_cluster_profile_plan(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    plan_path = state / "release-deploy/profile-plan.json"
    assert receipts.pending_profile_plan(state) is None, "no plan file, no stop"

    write_json_atomic(plan_path, _plan())
    pending = receipts.pending_profile_plan(state)
    assert pending == {
        "plan_sha256": _plan()["plan_sha256"],
        "change_kind": "CLUSTER_PROFILE_NEW",
        "current_version": "hyperpod-v1",
        "desired_version": "eks-v1",
    }, pending

    for broken in (
        _plan(change_kind="SITE_PROFILE_CHANGE"),
        _plan(approval_required=False),
        _plan(plan_sha256="f" * 64),
        {"schema_version": 1},
        [],
    ):
        write_json_atomic(plan_path, broken)
        assert receipts.pending_profile_plan(state) is None, broken
    plan_path.write_text("{not json", encoding="utf-8")
    assert receipts.pending_profile_plan(state) is None, "unreadable plans are no stop"


def test_cli_pending_and_profile_plan_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record, state = _first_deploy(tmp_path, [HYPERPOD_GPU])

    def run(*arguments: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "boot029_receipts",
                *arguments,
                "--record",
                str(record),
                "--state",
                str(state),
            ],
        )
        try:
            receipts.main()
        except SystemExit as raised:
            return int(raised.code or 0)
        return 0

    assert (
        run("pending", "--stage", "1", "--cpu-arn", CPU, "--gpu-arn", HYPERPOD_GPU) == 0
    )
    assert capsys.readouterr().out.strip() == "complete", "the state word is printed"
    assert (
        run("pending", "--stage", "2", "--cpu-arn", CPU, "--gpu-arn", HYPERPOD_GPU) == 2
    ), "pending is a first-deploy question"
    capsys.readouterr()

    assert run("profile-plan", "--stage", "1") == 1
    assert "no CLUSTER_PROFILE_NEW Runtime Profile plan awaits approval" in (
        capsys.readouterr().err
    )
    write_json_atomic(state / "release-deploy/profile-plan.json", _plan())
    assert run("profile-plan", "--stage", "1") == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["plan_sha256"] == _plan()["plan_sha256"], printed
    assert printed["desired_version"] == "eks-v1", printed

    _join_journal(state, EKS_GPU)
    assert (
        run(
            "verify",
            "--stage",
            "1",
            "--cpu-arn",
            CPU,
            "--gpu-arn",
            HYPERPOD_GPU,
            "--joined-gpu-arn",
            EKS_GPU,
        )
        == 0
    )
    proof = json.loads(capsys.readouterr().out)
    assert proof["gpu_cluster_count"] == 2 and len(proof["joins"]) == 1, proof
