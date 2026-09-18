from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_commit as commit
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinExecution, join_cluster
from gpu_fault.admin.cluster_join_nodes import NodeClaims
from gpu_fault.admin.cluster_join_state import load_join_state
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from gpu_fault.store import InMemoryStore
from tests.admin.test_admin_cluster_join import (
    GPU_B_ARN,
    Runner,
    _batch_execution,
    _joined_prerequisites,
    _membership_snapshot,
    _patch_batch_discovery,
    _snapshot,
    _target,
)
from tests.admin.test_admin_cluster_join_rollback import Attempt


def test_membership_rollback_retries_release_sync_after_site_was_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.path.write_bytes(attempt.candidate_path.read_bytes())
    request = JoinClusterRequest(load_site(attempt.path), GPU_B_ARN)
    record = attempt.evidence()
    execution = JoinExecution(
        _target(),
        "hp-gpu-b",
        record["DISCOVERED"],
        record["LOCAL_INPUTS_READY"],
        record["PREREQUISITES_READY"],
        load_site(attempt.candidate_path),
    )
    syncs: list[list[str]] = []
    transitions: list[str] = []

    def sync(site: RenderedSite) -> None:
        syncs.append([item["cluster_id"] for item in site.release_config["clusters"]])
        if len(syncs) == 1:
            raise TimeoutError("release-state synchronization timed out")

    monkeypatch.setattr(join, "_sync_join_release_state", sync)
    monkeypatch.setattr(commit, "_remove_joined_resources", lambda *_a, **_k: None)
    monkeypatch.setattr(
        join, "_run_rollout", lambda _site, mode, **_kwargs: transitions.append(mode)
    )

    with pytest.raises(TimeoutError, match="synchronization timed out"):
        commit.rollback_membership(request, execution=execution, joined=True)

    restored = load_site(attempt.path)
    assert [item["cluster_id"] for item in restored.release_config["clusters"]] == [
        "gpu-a"
    ], "the first rollback did not restore the local membership"
    assert transitions == [], "rollback was declared before release-state converged"

    commit.rollback_membership(
        replace(request, site=restored), execution=execution, joined=True
    )

    assert syncs == [["gpu-a"], ["gpu-a"]], (
        "retry skipped release-state repair because the site was already restored"
    )
    assert transitions == ["rollback-cluster"], (
        "the registry rollback must follow successful release-state repair"
    )


@pytest.mark.parametrize("namespace_uid", [None, "replacement-namespace"])
@pytest.mark.parametrize("activation_started", [False, True])
def test_join_resume_checks_the_prepared_namespace_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace_uid: str | None,
    activation_started: bool,
) -> None:
    attempt = Attempt(tmp_path)
    request = JoinClusterRequest(attempt.site, GPU_B_ARN, state_dir=attempt.state_dir)
    _directory, path, state = load_join_state(request)
    state["evidence"] = attempt.evidence()
    state["completed_steps"] = sorted(state["evidence"])
    if activation_started:
        state["completed_steps"].append("ACTIVATION_STARTED")
    state["phase"] = "CANDIDATE_READY"
    write_json_atomic(path, state)
    mutations: list[str] = []
    monkeypatch.setattr(join, "discover_cluster", lambda *_a, **_k: _target())
    monkeypatch.setattr(join, "_existing_cluster_networks", lambda *_a, **_k: [])
    monkeypatch.setattr(join, "read_node_claims", lambda *_a: NodeClaims())
    monkeypatch.setattr(join, "probe_join_namespace", lambda *_a, **_k: namespace_uid)
    monkeypatch.setattr(
        join, "_run_rollout", lambda _site, mode, **_k: mutations.append(mode)
    )
    monkeypatch.setattr(
        join,
        "wait_collector_readiness",
        lambda *_a: {"ready": True, "nodes": [{"ready": True}]},
    )
    monkeypatch.setattr(join, "membership_runtime_snapshot", _membership_snapshot)
    monkeypatch.setattr(
        join, "commit_membership", lambda *_a, **_k: mutations.append("commit")
    )

    with pytest.raises(join.JoinTargetIdentityError, match="namespace"):
        join_cluster(request, runner=Runner())

    assert mutations == [], "a resumed join mutated an unbound namespace incarnation"


def test_namespace_uid_probe_remains_usable_while_rollback_deletes_the_namespace(
    tmp_path: Path,
) -> None:
    attempt = Attempt(tmp_path)

    class NamespaceRunner(CommandRunner):
        def run(self, _arguments: Any, **_kwargs: Any) -> str:
            return json.dumps(
                {
                    "kind": "Namespace",
                    "metadata": {
                        "name": "gpu-fault-system",
                        "uid": "namespace-b",
                        "deletionTimestamp": "2026-09-12T00:00:00Z",
                    },
                }
            )

    assert (
        join.probe_join_namespace(
            NamespaceRunner(),
            site=attempt.site,
            kubeconfig=tmp_path / "gpu.kubeconfig",
            context="gpu-b",
        )
        == "namespace-b"
    ), (
        "rollback must still identify its terminating namespace before waiting for absence"
    )


@pytest.mark.parametrize("identity_drift", [False, True])
def test_membership_rollback_retires_resources_through_upsert_registry_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_drift: bool
) -> None:
    attempt = Attempt(tmp_path)
    request = JoinClusterRequest(attempt.site, GPU_B_ARN)
    record = attempt.evidence()
    prerequisites = _joined_prerequisites()
    execution = JoinExecution(
        _target(),
        "hp-gpu-b",
        record["DISCOVERED"],
        record["LOCAL_INPUTS_READY"],
        prerequisites,
        load_site(attempt.candidate_path),
    )
    preserved = _snapshot().resources[0]
    owned = preserved.model_copy(
        update={
            "resource_key": "aws/iam/executor/hp-gpu-b/role",
            "resource_type": "iam_role",
            "resource_id": "gpu-b-executor",
            "resource_arn": prerequisites["executor_role"]["role_arn"],
            "attributes": {"inline_policy_name": "GPUFaultRegionalExecutor"},
        }
    )
    if identity_drift:
        owned = owned.model_copy(update={"resource_id": "replacement-role"})
    gpu = preserved.model_copy(
        update={
            "resource_key": "cluster/hp-gpu-b/eks",
            "resource_type": "gpu_eks",
            "resource_id": "gpu-b",
            "resource_arn": GPU_B_ARN,
            "ownership": InstallationResourceOwnership.EXTERNAL,
            "delete_policy": InstallationResourceDeletePolicy.PRESERVE,
        }
    )
    store = InMemoryStore()
    for resource in (preserved, owned, gpu):
        store.save_installation_resource(resource)

    def fetch(_site: RenderedSite) -> InstallationResourceSnapshot:
        snapshot = InstallationResourceSnapshot(
            site_id="test-site",
            resources=store.list_installation_resources("test-site"),
        )
        return snapshot.model_copy(update={"source_sha256": snapshot.digest()})

    def sync(_site: RenderedSite, snapshot: InstallationResourceSnapshot) -> None:
        assert preserved.resource_key not in {
            resource.resource_key for resource in snapshot.resources
        }, "rollback must not replay unrelated registry rows"
        for resource in snapshot.resources:
            store.save_installation_resource(resource)

    monkeypatch.setattr(join, "_fetch_installation_registry", fetch)
    monkeypatch.setattr(join, "_sync_installation_snapshot", sync)
    monkeypatch.setattr(join, "_write_installation_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(join, "_sync_join_release_state", lambda *_a: None)
    monkeypatch.setattr(join, "_run_rollout", lambda *_a, **_k: None)

    if identity_drift:
        with pytest.raises(BootstrapError, match="resource identity changed"):
            commit.rollback_membership(request, execution=execution, joined=True)
        assert store.get_installation_resource("test-site", owned.resource_key) == (
            owned
        ), "rollback changed a replacement resource's status"
        assert store.get_installation_resource("test-site", gpu.resource_key) == gpu, (
            "failed ownership validation partially synchronized resource status"
        )
        return

    commit.rollback_membership(request, execution=execution, joined=True)

    assert (
        store.get_installation_resource("test-site", owned.resource_key).status
        is InstallationResourceStatus.DELETED
    ), "upsert synchronization left the compensated IAM role ACTIVE"
    assert (
        store.get_installation_resource("test-site", gpu.resource_key).status
        is InstallationResourceStatus.DETACHED
    ), "rollback failed to detach the preserved GPU cluster registry record"
    assert store.get_installation_resource("test-site", preserved.resource_key) == (
        preserved
    ), "rollback changed an unrelated installation resource"
    completed = store.list_installation_resources("test-site")
    commit.rollback_membership(request, execution=execution, joined=True)
    assert store.list_installation_resources("test-site") == completed, (
        "rollback retry rewrote already-terminal resource evidence"
    )


@pytest.mark.parametrize("phase", ["prepare", "deploy"])
def test_batch_supervision_loss_drains_started_workers_without_compensation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    attempt = Attempt(tmp_path)
    document = yaml.safe_load(attempt.path.read_text())
    document["spec"]["release"]["upgradeMaxParallelClusters"] = 2
    attempt.path.write_text(yaml.safe_dump(document, sort_keys=False))
    site = load_site(attempt.path)
    requests = tuple(
        JoinClusterRequest(
            site, f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{index}"
        )
        for index in range(2)
    )
    executions = {
        request.gpu_cluster_arn: _batch_execution(tmp_path, site, request, index)
        for index, request in enumerate(requests)
    }
    _patch_batch_discovery(monkeypatch, executions)
    started = threading.Barrier(2)
    finished = threading.Event()
    compensations: list[str] = []
    monkeypatch.setattr(join, "_run_rollout", lambda *_a, **_k: None)
    monkeypatch.setattr(
        join,
        "_rollback",
        lambda request, **_k: compensations.append(request.gpu_cluster_arn),
    )

    def fail_or_finish(cluster_id: str) -> None:
        started.wait(timeout=5)
        if cluster_id == "gpu-0":
            raise ProcessSupervisionLost("fixture supervision proof lost")
        time.sleep(0.05)
        finished.set()
        raise BootstrapError("late worker failed after finishing its work")

    def prepare(request: JoinClusterRequest, **_kwargs: Any) -> None:
        fail_or_finish(executions[request.gpu_cluster_arn].cluster_id)

    def deploy(*, execution: JoinExecution, **_kwargs: Any) -> None:
        fail_or_finish(execution.cluster_id)

    if phase == "prepare":
        monkeypatch.setattr(join, "_prepare_execution", prepare)
    else:
        monkeypatch.setattr(join, "_deploy_cluster", deploy)

    with pytest.raises(ProcessSupervisionLost, match="proof lost"):
        batch.join_clusters(requests, runner_factory=Runner)

    assert finished.is_set(), "fatal batch exit did not drain the other started worker"
    assert compensations == [], "supervision loss started automatic compensation"
    states = [
        json.loads(path.read_text())
        for path in (site.source.parent / "join-cluster").glob("*/state.json")
    ]
    assert len(states) == 2
    assert all(state["phase"] == "SUPERVISION_LOST" for state in states), (
        "batch supervision loss did not fence every unfinished attempt"
    )
