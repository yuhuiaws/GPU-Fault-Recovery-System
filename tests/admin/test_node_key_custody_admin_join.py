from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone

import pytest
import yaml

from gpu_fault.admin import bootstrap
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster
from gpu_fault.admin.cluster_join_nodes import NodeClaims
from gpu_fault.admin.cluster_join_state import complete_step, load_join_state
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
)
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def joining(tmp_path, monkeypatch):
    world = AdminWorld(tmp_path, monkeypatch)
    world.configure()
    world.access()
    world.release_ready = True
    site = world.site(managed=False)
    site.release_config["aws_region"] = world.gpu.region
    site.release_config["auto_rollback"] = True
    site.source.write_text(
        yaml.safe_dump({"metadata": {"name": world.site_id}, "spec": {"clusters": []}})
    )
    site.source_sha256 = hashlib.sha256(site.source.read_bytes()).hexdigest()
    candidate = world.site(managed=True)
    candidate.source = world.root / "candidate.yaml"
    candidate.source.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": world.site_id},
                "spec": {"clusters": [{"clusterId": world.cluster_id}]},
            }
        )
    )
    candidate.source_sha256 = hashlib.sha256(candidate.source.read_bytes()).hexdigest()
    request = JoinClusterRequest(
        site=site, gpu_cluster_arn=world.gpu.eks_arn, state_dir=world.state_dir / "join"
    )
    state_dir, state_path, state = load_join_state(request)
    complete_step(
        state_path,
        state,
        "DISCOVERED",
        {"target": asdict(world.gpu), "cluster_id": world.cluster_id},
    )
    complete_step(
        state_path,
        state,
        "LOCAL_INPUTS_STARTED",
        {
            "gpu_kubeconfig": str(world.gpu_kubeconfig),
            "namespace_uid": world.namespaces["gpu", world.context().namespace],
            "namespace_creation_started": True,
        },
    )
    complete_step(
        state_path,
        state,
        "LOCAL_INPUTS_READY",
        {
            "gpu_kubeconfig": str(world.gpu_kubeconfig),
            "fleet_master_file": str(world.master_file),
            "namespace_uid": world.namespaces["gpu", world.context().namespace],
            "nodes": [
                item["metadata"]["name"] for item in world.api.state["nodes"]["items"]
            ],
            "node_uids": {
                item["metadata"]["name"]: item["metadata"]["uid"]
                for item in world.api.state["nodes"]["items"]
            },
        },
    )
    complete_step(
        state_path,
        state,
        "PREREQUISITES_READY",
        {
            "network": {"complete": True},
            "executor_role": {"role_arn": "arn:aws:iam::111122223333:role/executor"},
            "node_keys": {"cluster_id": world.cluster_id},
        },
    )
    complete_step(
        state_path,
        state,
        "CANDIDATE_READY",
        {"site_file": str(candidate.source), "cluster": {}},
    )
    monkeypatch.setattr(join, "reload_site_for_mutation", lambda value: value)
    monkeypatch.setattr(
        join,
        "load_site",
        lambda path, **kwargs: candidate if path == candidate.source else site,
    )
    monkeypatch.setattr(join, "discover_cluster", lambda *a, **k: world.gpu)
    monkeypatch.setattr(bootstrap, "discover_cluster", lambda *a, **k: world.gpu)
    monkeypatch.setattr(join, "cached_network_baseline", lambda *a, **k: [])
    monkeypatch.setattr(join, "read_node_claims", lambda *a, **k: NodeClaims())
    monkeypatch.setattr(
        join,
        "wait_collector_readiness",
        lambda *a, **k: {"ready": True, "nodes": ["node-a", "node-b"]},
    )

    @contextmanager
    def materialized(_site):
        yield world.root / "controlled-release.json"

    monkeypatch.setattr(join, "materialized_release_config", materialized)
    monkeypatch.setattr(join, "effective_environment", lambda _: {})
    rollouts = []

    def run_driver(command, **kwargs):
        assert world.helper_calls == 1, "join rollout preceded custody completion"
        rollouts.append(command[1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(join, "run_driver", run_driver)
    monkeypatch.setattr(
        join,
        "membership_runtime_snapshot",
        lambda value: {
            "live_release_state_sha256": "a" * 64,
            "live_release_identity_sha256": "b" * 64,
            "registry_generation": 2,
            "registry_content_sha256": "c" * 64,
            "registry_cluster_states": {world.cluster_id: "PENDING"},
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    def commit(_request, **kwargs):
        assert world.helper_calls == 1
        site.release_config["clusters"] = candidate.release_config["clusters"]
        complete_step(kwargs["state_path"], kwargs["state"], "RELEASE_STATE_UPDATED")

    monkeypatch.setattr(join, "commit_membership", commit)
    return world, request, state_path, rollouts


def test_join_rechecks_old_checkpoints_pauses_and_resumes_without_namespace_recreation(
    joining,
):
    world, request, state_path, rollouts = joining
    uid = world.namespaces["gpu", world.context().namespace]
    with pytest.raises(CustodyPreparationRequired):
        join_cluster(request, runner=world)
    state = json.loads(state_path.read_text())
    assert state["phase"] == "CUSTODY_AWAITING_AUTHORIZATION"
    assert "NODE_KEYS_STARTED" not in state["completed_steps"]
    assert world.helper_calls == 0 and rollouts == []
    chain = world.authorize()
    result = join_cluster(request, runner=world)
    assert result["phase"] == "COMPLETED"
    assert chain.exists() and world.helper_calls == 1
    assert world.namespaces["gpu", world.context().namespace] == uid
    state = json.loads(state_path.read_text())
    assert state["evidence"]["PREREQUISITES_READY"]["node_keys"][
        "custody_chain"
    ] == str(chain)
    assert rollouts == ["join-cluster", "verify"]
    assert join_cluster(request, runner=world)["phase"] == "COMPLETED"
    assert world.helper_calls == 1


def test_join_partial_custody_failure_is_not_rolled_back_or_reconstructed(joining):
    world, request, state_path, rollouts = joining
    with pytest.raises(CustodyPreparationRequired):
        join_cluster(request, runner=world)
    chain = world.authorize()
    world.api.state["events"] = [
        {"on": "cpu:create", "returncode": 1, "lost_ack": True}
    ]
    with pytest.raises(CustodyReconciliationRequired):
        join_cluster(request, runner=world)
    state = json.loads(state_path.read_text())
    assert state["phase"] == "CUSTODY_BLOCKED"
    assert not chain.exists() and rollouts == []
    with pytest.raises(CustodyReconciliationRequired, match="incomplete custody"):
        join_cluster(request, runner=world)
    assert world.helper_calls == 1


def test_custody_batch_stops_on_preparation_then_resumes_through_single_join(
    joining, monkeypatch
):
    from gpu_fault.admin import cluster_batch_join as batch

    world, request, _, rollouts = joining
    monkeypatch.setattr(batch, "reload_site_for_mutation", lambda site: site)
    runners = []

    def runner():
        runners.append(world)
        return world

    later = replace(request, gpu_cluster_arn=request.gpu_cluster_arn + "-later")
    with pytest.raises(CustodyPreparationRequired):
        batch.join_clusters((request, later), runner_factory=runner)
    assert len(runners) == 1, "later cluster was admitted after custody paused"
    assert not rollouts and world.helper_calls == 0
    world.authorize()
    completed = batch.join_clusters((request,), runner_factory=runner)
    assert len(completed["joined"]) == 1 and completed["already_managed"] == []
    assert world.helper_calls == 1
    fresh = replace(request, state_dir=world.state_dir / "join-new-attempt")
    result = batch.join_clusters((fresh,), runner_factory=runner)
    assert result["joined"] == []
    assert result["already_managed"][0]["cluster_id"] == world.cluster_id
    assert world.helper_calls == 1
