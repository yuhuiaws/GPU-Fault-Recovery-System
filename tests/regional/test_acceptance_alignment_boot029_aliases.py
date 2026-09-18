from __future__ import annotations

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import boot029_receipts as receipts

CPU_EKS = "arn:aws:eks:us-west-2:123456789012:cluster/cpu"
CPU_HYPERPOD = "arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu"
GPU_EKS = "arn:aws:eks:us-west-2:123456789012:cluster/gpu"
GPU_HYPERPOD = "arn:aws:sagemaker:us-west-2:123456789012:cluster/gpu"


def case_documents(tmp_path, stage):
    state = tmp_path / "state"
    record = state / "acceptance/receipt.json"
    receipts.initialize(
        record,
        inputs={"cpu_cluster_arn": CPU_HYPERPOD, "gpu_cluster_arn": GPU_HYPERPOD},
        stage=1,
    )
    write_json_atomic(
        state / "bootstrap-state.json",
        {
            "phase": "site-ready",
            "resources": {
                "initial_deploy_target": {
                    "schema_version": 1,
                    "cpu": {
                        "input_arn": CPU_HYPERPOD,
                        "eks_arn": CPU_EKS,
                        "hyperpod_arn": CPU_HYPERPOD,
                    },
                }
            },
        },
    )
    if stage == 2:
        journal = state / "remove-cluster/gpu/state.json"
        value = {
            "phase": "COMPLETED",
            "attempt_id": "a" * 32,
            "target": {"eks_cluster_arn": GPU_EKS, "hyperpod_cluster_name": "gpu"},
            "evidence": {
                "DISCOVERED": {
                    "provider_identity": {
                        "eks_arn": GPU_EKS,
                        "hyperpod_arn": GPU_HYPERPOD,
                    }
                }
            },
        }
    elif stage == 3:
        journal = state / "join-cluster/gpu/state.json"
        value = {"phase": "COMPLETED", "attempt": 1, "gpu_cluster_arn": GPU_HYPERPOD}
    else:
        journal = state / "uninstall/state.json"
        value = {
            "phase": "COMPLETED",
            "attempt_id": "a" * 32,
            "cpu_disposition": "keep",
            "reset_database": True,
            "site_identity": {"cpu_eks_arn": CPU_EKS},
        }
    return record, state, journal, value


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_completion_matches_saved_hyperpod_alias_without_discovery(tmp_path, stage):
    record, state, journal, value = case_documents(tmp_path, stage)
    receipts.start_stage(record, state=state, stage=stage)
    write_json_atomic(journal, value)
    result = receipts.verify_journal(
        record, state=state, stage=stage, gpu_arn=GPU_HYPERPOD, cpu_arn=CPU_HYPERPOD
    )
    assert result["phase"] == "COMPLETED"
    assert result["journal_sha256"] == receipts.digest(journal.read_bytes())
    assert GPU_HYPERPOD not in str(result) and CPU_HYPERPOD not in str(result)


@pytest.mark.parametrize(
    "damage",
    ["missing", "missing-eks", "foreign-eks", "foreign-hyperpod", "wrong-location"],
)
def test_removal_alias_requires_discovery_bound_to_the_same_eks(tmp_path, damage):
    record, state, journal, value = case_documents(tmp_path, 2)
    identity = value["evidence"]["DISCOVERED"]["provider_identity"]
    if damage == "missing-eks":
        identity.pop("eks_arn")
    elif damage == "foreign-eks":
        identity["eks_arn"] = CPU_EKS
    elif damage == "foreign-hyperpod":
        identity["hyperpod_arn"] = CPU_HYPERPOD
    else:
        value["evidence"] = {}
        if damage == "wrong-location":
            value["target"]["hyperpod_arn"] = GPU_HYPERPOD
            value["gpu_cluster_arn"] = GPU_HYPERPOD
    receipts.start_stage(record, state=state, stage=2)
    write_json_atomic(journal, value)
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            record, state=state, stage=2, gpu_arn=GPU_HYPERPOD, cpu_arn=CPU_HYPERPOD
        )


@pytest.mark.parametrize(
    "damage",
    [
        "missing-bootstrap",
        "missing-cpu",
        "foreign-eks",
        "foreign-hyperpod",
        "input-only",
        "foreign-uninstall",
        "symlink",
    ],
)
def test_uninstall_alias_requires_the_initial_target_and_same_cpu(tmp_path, damage):
    record, state, journal, value = case_documents(tmp_path, 4)
    bootstrap_path = state / "bootstrap-state.json"
    bootstrap = receipts.read(bootstrap_path)
    cpu = bootstrap["resources"]["initial_deploy_target"]["cpu"]
    if damage == "missing-cpu":
        bootstrap["resources"]["initial_deploy_target"].pop("cpu")
    elif damage == "foreign-eks":
        cpu["eks_arn"] = GPU_EKS
    elif damage == "foreign-hyperpod":
        cpu["hyperpod_arn"] = GPU_HYPERPOD
    elif damage == "input-only":
        cpu.pop("hyperpod_arn")
    elif damage == "foreign-uninstall":
        value["site_identity"]["cpu_eks_arn"] = GPU_EKS
    write_json_atomic(bootstrap_path, bootstrap)
    if damage == "missing-bootstrap":
        bootstrap_path.unlink()
    elif damage == "symlink":
        saved = bootstrap_path.with_name("other.json")
        bootstrap_path.rename(saved)
        bootstrap_path.symlink_to(saved)
    receipts.start_stage(record, state=state, stage=4)
    write_json_atomic(journal, value)
    with pytest.raises(ValueError):
        receipts.verify_journal(
            record, state=state, stage=4, gpu_arn=GPU_HYPERPOD, cpu_arn=CPU_HYPERPOD
        )


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_alias_match_cannot_reuse_a_preexisting_completed_attempt(tmp_path, stage):
    record, state, journal, value = case_documents(tmp_path, stage)
    write_json_atomic(journal, value)
    receipts.start_stage(record, state=state, stage=stage)
    value["updated_at"] = "mtime-is-not-new-completion"
    write_json_atomic(journal, value)
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            record, state=state, stage=stage, gpu_arn=GPU_HYPERPOD, cpu_arn=CPU_HYPERPOD
        )
    value["attempt" if stage == 3 else "attempt_id"] = 2 if stage == 3 else "b" * 32
    write_json_atomic(journal, value)
    assert (
        receipts.verify_journal(
            record, state=state, stage=stage, gpu_arn=GPU_HYPERPOD, cpu_arn=CPU_HYPERPOD
        )["phase"]
        == "COMPLETED"
    )


def test_alias_match_still_requires_a_unique_current_removal_journal(tmp_path):
    record, state, journal, value = case_documents(tmp_path, 2)
    receipts.start_stage(record, state=state, stage=2)
    write_json_atomic(journal, value)
    write_json_atomic(state / "remove-cluster/other/state.json", value)
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            record, state=state, stage=2, gpu_arn=GPU_HYPERPOD, cpu_arn=CPU_HYPERPOD
        )
