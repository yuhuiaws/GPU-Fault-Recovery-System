from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.site import load_site
from tests.admin._cov95_join_support import JoinScenario, target


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return JoinScenario(tmp_path, monkeypatch)


@pytest.mark.parametrize("batch_mode", [False, True])
def test_public_join_completes_and_replay_preserves_membership(scenario, batch_mode):
    result = scenario.batch() if batch_mode else scenario.join()
    assert result["phase"] == "COMPLETED"
    targets = ("b", "c") if batch_mode else ("b",)
    for suffix in targets:
        state = scenario.state(suffix)
        assert state["phase"] == "COMPLETED"
        assert {
            "LOCAL_INPUTS_STARTED",
            "LOCAL_INPUTS_READY",
            "NODE_NAMES_VERIFIED",
            "NODE_KEYS_STARTED",
            "JOIN_STARTED",
            "JOINED",
            "VERIFIED",
            "SITE_UPDATED",
            "REGISTRY_UPDATED",
            "ACTIVATION_STARTED",
            "ACTIVATED",
            "FINAL_VERIFIED",
        }.issubset(state["completed_steps"]), (
            "completed join omitted a required persistent barrier"
        )
        assert state["evidence"]["FINAL_VERIFIED"]["registry_lifecycle"] == "ACTIVE"
    before = list(scenario.driver_calls)
    replay = scenario.batch() if batch_mode else scenario.join()
    assert replay["phase"] == "COMPLETED"
    assert scenario.driver_calls == before
    assert sorted(scenario.cluster_states) == [
        "gpu-a",
        *("hp-gpu-" + suffix for suffix in targets),
    ]
    assert all(value == "ACTIVE" for value in scenario.cluster_states.values()), (
        "successful join left a non-ACTIVE member"
    )
    assert scenario.gpu_kubeconfig.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "boundary",
    [
        "executor-role:gpu-b",
        "adot-role:gpu-b",
        "node-keys:gpu-b",
        "preflight",
        "join-cluster:hp-gpu-b",
        "verify",
        "registry-sync",
        "activate-cluster:hp-gpu-b",
        "failure-domains",
    ],
)
def test_interrupted_public_join_reuses_journal_and_resumes_forward(scenario, boundary):
    scenario.failure = boundary
    with pytest.raises(BootstrapError, match="modeled join boundary"):
        scenario.join()
    failed = scenario.state()
    assert failed["phase"] == (
        "FAILED_AFTER_ACTIVATION"
        if boundary in {"activate-cluster:hp-gpu-b", "failure-domains"}
        else "FAILED"
    )
    assert not any(
        mode in {"rollback-cluster", "fail-cluster"}
        for mode, _cluster in scenario.driver_calls
    ), "fail-forward policy unexpectedly started compensation"
    scenario.failure = None
    result = scenario.join()
    assert result["phase"] == "COMPLETED"
    assert scenario.state()["attempt"] == 1
    assert scenario.cluster_states["hp-gpu-b"] == "ACTIVE"
    if boundary in {"activate-cluster:hp-gpu-b", "failure-domains"}:
        assert scenario.driver_calls.count(("join-cluster", "hp-gpu-b")) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("region", "us-west-2"),
        ("account_id", "111122223333"),
        ("node_recovery", "Automatic"),
        ("role", "cpu"),
        ("eks_arn", "arn:aws:eks:us-east-1:123456789012:cluster/control"),
        ("hyperpod_name", "hp-gpu-a"),
        ("context", "gpu-a"),
    ],
)
def test_target_identity_failure_precedes_local_inputs(scenario, field, value):
    original = target()
    scenario.targets[original.eks_arn] = replace(original, **{field: value})
    with pytest.raises(BootstrapError):
        scenario.join()
    assert not scenario.gpu_kubeconfig.exists(), (
        "invalid target created a GPU kubeconfig"
    )
    assert not scenario.commands, "invalid target started local preparation commands"
    assert not scenario.driver_calls, "invalid target started a release driver"


def test_existing_managed_target_returns_without_new_candidate(scenario):
    original = target()
    scenario.targets[original.eks_arn] = replace(
        target("a"), input_arn=original.eks_arn, hyperpod_name="hp-gpu-a"
    )
    request = scenario.request(cluster_id="gpu-a")
    result = join.join_cluster(request, runner=scenario)
    assert result["phase"] == "ALREADY_MANAGED"
    assert not scenario.driver_calls, "already managed target started a new rollout"
    assert not scenario.commands, "already managed target repeated local preparation"


def test_existing_target_cannot_change_registered_cluster_id(scenario):
    original = target()
    scenario.targets[original.eks_arn] = replace(
        target("a"), input_arn=original.eks_arn
    )
    with pytest.raises(BootstrapError, match="already managed"):
        scenario.join()
    assert not scenario.commands, "conflicting cluster ID reached local preparation"


@pytest.mark.parametrize("boundary", ["preflight", "verify", "join-cluster:hp-gpu-b"])
def test_batch_failure_keeps_successful_siblings_and_records_target_failure(
    scenario, boundary
):
    scenario.failure = boundary
    with pytest.raises(
        BootstrapError, match="batch join completed with failed clusters"
    ):
        scenario.batch()
    failed = scenario.state("b")
    assert failed["phase"] == "FAILED"
    if boundary == "join-cluster:hp-gpu-b":
        assert scenario.state("c")["phase"] == "COMPLETED"
        assert scenario.cluster_states["hp-gpu-c"] == "ACTIVE"
    else:
        assert scenario.state("c")["phase"] == "FAILED"
    scenario.failure = None
    assert scenario.batch()["phase"] == "COMPLETED"
    assert scenario.cluster_states == {
        "gpu-a": "ACTIVE",
        "hp-gpu-b": "ACTIVE",
        "hp-gpu-c": "ACTIVE",
    }


@pytest.mark.parametrize(
    "boundary", ["executor-role:gpu-b", "node-keys:gpu-b", "adot-role:gpu-b"]
)
def test_batch_preparation_failure_does_not_undo_successful_sibling(scenario, boundary):
    scenario.failure = boundary
    with pytest.raises(
        BootstrapError, match="batch join completed with failed clusters"
    ):
        scenario.batch()
    assert scenario.state("b")["phase"] == "FAILED"
    assert scenario.state("c")["phase"] == "COMPLETED"
    assert scenario.cluster_states["hp-gpu-c"] == "ACTIVE"


@pytest.mark.parametrize("boundary", ["preflight", "join-cluster:hp-gpu-b", "verify"])
def test_batch_interrupt_records_every_uncommitted_attempt(scenario, boundary):
    scenario.failure = boundary
    scenario.failure_error = KeyboardInterrupt("modeled cancellation")
    with pytest.raises(KeyboardInterrupt, match="modeled cancellation"):
        scenario.batch()
    assert scenario.state("b")["phase"] == "FAILED"
    assert scenario.state("c")["phase"] == "FAILED"
    assert not any(
        mode == "activate-cluster" for mode, _cluster in scenario.driver_calls
    ), "interrupted batch activated an uncommitted cluster"


@pytest.mark.parametrize("batch_mode", [False, True])
def test_supervision_loss_is_durable_and_retry_starts_no_commands(scenario, batch_mode):
    scenario.failure = "preflight"
    scenario.failure_error = ProcessSupervisionLost("modeled proof loss")
    run = scenario.batch if batch_mode else scenario.join
    with pytest.raises(ProcessSupervisionLost):
        run()
    assert scenario.state()["phase"] == "SUPERVISION_LOST"
    if batch_mode:
        assert scenario.state("c")["phase"] == "SUPERVISION_LOST"
    count = len(scenario.events)
    scenario.failure = None
    with pytest.raises(ProcessSupervisionLost, match="automatic retry is forbidden"):
        run()
    assert len(scenario.events) == count


@pytest.mark.parametrize("setting", [None, 1, 8])
def test_batch_concurrency_valid_values(scenario, setting):
    scenario.site.release_config["release"]["upgrade_max_parallel_clusters"] = setting
    assert batch.deploy_concurrency(scenario.site) == (setting or 1)


@pytest.mark.parametrize("setting", [True, "2", 0, -1, 9, 1.5])
def test_batch_concurrency_rejects_unbounded_values(scenario, setting):
    scenario.site.release_config["release"]["upgrade_max_parallel_clusters"] = setting
    with pytest.raises(BootstrapError, match="integer within 1..8"):
        batch.deploy_concurrency(scenario.site)
    assert not scenario.events, "invalid parallelism started batch work"


def test_batch_rejects_duplicate_arns_before_discovery(scenario):
    request = scenario.request()
    with pytest.raises(BootstrapError, match="duplicate GPU cluster ARNs"):
        batch.join_clusters((request, request), runner_factory=lambda: scenario)
    assert not scenario.events, "duplicate ARN batch started discovery"


def test_batch_empty_is_noop_and_mixed_sites_are_refused(scenario, tmp_path):
    assert batch.join_clusters(()) == {
        "phase": "COMPLETED",
        "joined": [],
        "already_managed": [],
    }
    copy = tmp_path / "different.yaml"
    copy.write_bytes(scenario.path.read_bytes())
    copy.chmod(0o600)
    request = replace(scenario.request("c"), site=load_site(copy))
    with pytest.raises(BootstrapError, match="same managed site"):
        batch.join_clusters(
            (scenario.request(), request), runner_factory=lambda: scenario
        )
    assert not scenario.events, "mixed-site batch started work"


@pytest.mark.parametrize("kind", ["shared-node", "shared-vpc", "shared-context"])
def test_batch_identity_collisions_block_before_any_rollout(scenario, kind):
    if kind == "shared-node":
        scenario.node_names["c"] = scenario.node_names["b"]
    else:
        field = "vpc_id" if kind == "shared-vpc" else "context"
        scenario.targets[target("c").eks_arn] = replace(
            target("c"), **{field: getattr(target(), field)}
        )
    with pytest.raises(
        BootstrapError, match="failed clusters|conflicting cluster identities"
    ):
        scenario.batch()
    assert not scenario.driver_calls, "colliding batch identities started a rollout"
    assert not any(
        name.startswith(("executor-role:", "node-keys:")) for name in scenario.events
    ), "colliding batch identities started IAM or key preparation"


def test_failed_network_discovery_never_starts_local_preparation(scenario):
    scenario.network_error = BootstrapError("modeled network discovery failure")
    with pytest.raises(BootstrapError, match="read-only preparation"):
        scenario.batch()
    assert not scenario.driver_calls, "failed network discovery started a rollout"
    assert not scenario.gpu_kubeconfig.exists(), (
        "failed network discovery created GPU access"
    )


def test_successful_join_updates_bootstrap_inventory_incrementally(scenario):
    path = scenario.path.parent / "bootstrap-state.json"
    original = {
        "site_id": "test-site",
        "resources": {
            "nlb_network": {"gpu_nat_eips": ["192.0.2.10"]},
            "pki": {
                "hosted_zone_id": "ZEXAMPLE",
                "vpc_associations": [
                    {
                        "vpc_id": "vpc-gpu-a",
                        "vpc_region": "us-east-1",
                        "ownership": "EXTERNAL",
                        "resource_key": "aws/route53/vpc-association/0",
                    }
                ],
            },
        },
        "removed_clusters": {"hp-gpu-b": {}},
        "completed_tasks": ["existing-task"],
    }
    path.write_text(json.dumps(original))
    assert scenario.join()["phase"] == "COMPLETED"
    saved = json.loads(path.read_text())
    assert saved["resources"]["nlb_network"]["gpu_nat_eips"] == [
        "192.0.2.10",
        "192.0.2.20",
    ]
    assert saved["resources"]["pki"] == {
        "hosted_zone_id": "ZEXAMPLE",
        "vpc_associations": [
            *original["resources"]["pki"]["vpc_associations"],
            {
                "vpc_id": "vpc-gpu-b",
                "vpc_region": "us-east-1",
                "ownership": "CREATED",
                "cluster_ids": ["hp-gpu-b"],
                "resource_key": "aws/route53/vpc-association/us-east-1/vpc-gpu-b",
                "native": False,
            },
        ],
    }
    assert "hp-gpu-b" not in saved["removed_clusters"]
    assert "hp-gpu-b" in saved["joined_clusters"]
    assert {
        "existing-task",
        "executor_role:hp-gpu-b",
        "node_keys:hp-gpu-b",
        "adot_writer_role:hp-gpu-b",
    }.issubset(saved["completed_tasks"]), (
        "join dropped prior or newly completed bootstrap tasks"
    )
    assert yaml.safe_load(scenario.path.read_text())["spec"]["gpuKubeconfig"] == str(
        scenario.gpu_kubeconfig
    )
