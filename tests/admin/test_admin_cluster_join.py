from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from gpu_fault import admin_cluster_join
from gpu_fault.admin_bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin_cluster_join import (
    JoinClusterRequest,
    JoinExecution,
    join_cluster,
    wait_collector_readiness,
)
from gpu_fault.admin_site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file

GPU_B_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"


def _target() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=GPU_B_ARN,
        role="gpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-gpu-b"),
        hyperpod_name="hp-gpu-b",
        eks_arn=GPU_B_ARN,
        eks_name="gpu-b",
        vpc_id="vpc-gpu-b",
        subnet_ids=("subnet-b",),
        node_recovery="None",
        context="gpu-fault-gpu-2-gpu-b",
    )


def _snapshot() -> InstallationResourceSnapshot:
    now = datetime.now(timezone.utc)
    snapshot = InstallationResourceSnapshot(
        site_id="test-site",
        resources=[
            InstallationResource(
                site_id="test-site",
                resource_key="aws/nlb",
                resource_type="nlb",
                resource_id="gpu-fault-nlb",
                region="us-east-1",
                account_id="123456789012",
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                created_at=now,
                updated_at=now,
            )
        ],
    )
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


class Runner:
    dry_run = False

    def run(self, arguments, **_kwargs):
        if "update-kubeconfig" in arguments:
            path = Path(arguments[arguments.index("--kubeconfig") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("kubeconfig", encoding="utf-8")
        return ""


def test_join_cluster_is_atomic_resumable_and_registers_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    state_dir = tmp_path / "join-state"
    gpu_kubeconfig = tmp_path / "secure/gpu.kubeconfig"
    rollout_calls: list[tuple[str, str | None]] = []
    synced: list[InstallationResourceSnapshot] = []
    release_state_syncs: list[list[str]] = []

    monkeypatch.setattr(
        admin_cluster_join,
        "_run_rollout",
        lambda _site, mode, *, cluster_id=None: rollout_calls.append(
            (mode, cluster_id)
        ),
    )
    monkeypatch.setattr(
        admin_cluster_join, "discover_cluster", lambda *_args, **_kwargs: _target()
    )

    def export_registry(_site, output):
        admin_cluster_join.write_installation_resource_snapshot(
            site, _snapshot(), path=output
        )
        return _snapshot()

    monkeypatch.setattr(
        admin_cluster_join, "fetch_installation_resource_registry", export_registry
    )
    monkeypatch.setattr(
        admin_cluster_join, "_gpu_kubeconfig", lambda _site: gpu_kubeconfig
    )
    monkeypatch.setattr(admin_cluster_join, "ensure_namespace", lambda *_a, **_k: None)

    def base_secrets(_runner, *, secure_dir, **_kwargs):
        path = secure_dir / "fleet-master"
        path.write_text("f" * 64, encoding="utf-8")
        path.chmod(0o600)
        return path

    monkeypatch.setattr(admin_cluster_join, "_ensure_base_secrets", base_secrets)
    monkeypatch.setattr(
        admin_cluster_join,
        "_shared_ca_file",
        lambda current: Path(current.release_config["clusters"][0]["ca_file"]),
    )
    monkeypatch.setattr(
        admin_cluster_join, "_list_nodes", lambda *_args, **_kwargs: ["node-b"]
    )
    prerequisites = {
        "executor_role": {
            "role_arn": "arn:aws:iam::123456789012:role/gpu-b-executor",
            "ownership": "CREATED",
            "inline_policy_name": "GPUFaultRegionalExecutor",
            "oidc_provider_arn": ("arn:aws:iam::123456789012:oidc-provider/issuer-b"),
            "oidc_provider_ownership": "EXTERNAL",
            "cluster_name": "gpu-b",
        },
        "network": {
            "vpc_id": "vpc-gpu-b",
            "nat_eips": ["192.0.2.20"],
            "created_ingress_eips": ["192.0.2.20"],
            "existing_vpc_ids": ["vpc-gpu-a"],
            "hosted_zone_id": "Z123",
            "association_created": True,
        },
        "node_keys": {"cluster_id": "hp-gpu-b"},
    }
    monkeypatch.setattr(
        admin_cluster_join,
        "_parallel_prerequisites",
        lambda *_args, **_kwargs: prerequisites,
    )
    monkeypatch.setattr(
        admin_cluster_join, "_update_bootstrap_state", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_sync_release_state",
        lambda current: release_state_syncs.append(
            [item["cluster_id"] for item in current.release_config["clusters"]]
        ),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "wait_collector_readiness",
        lambda *_args, **_kwargs: {"ready": True, "nodes": [{}, {}, {}, {}]},
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "sync_installation_resource_snapshot",
        lambda _site, snapshot: synced.append(snapshot),
    )

    def fetch_live(_site, output=None):
        del output
        return synced[-1] if synced else _snapshot()

    monkeypatch.setattr(
        admin_cluster_join, "fetch_installation_resource_registry", fetch_live
    )
    # Discovery exports the before snapshot before the first sync.
    before = state_dir / "installation-resources-before-001.json"
    before.parent.mkdir(parents=True, exist_ok=True)
    admin_cluster_join.write_installation_resource_snapshot(
        site, _snapshot(), path=before
    )

    request = JoinClusterRequest(
        site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir
    )
    result = join_cluster(request, runner=Runner())
    resumed = join_cluster(
        JoinClusterRequest(
            site=load_site(path), gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir
        ),
        runner=Runner(),
    )

    updated = load_site(path)
    keys = {item.resource_key for item in synced[-1].resources}
    assert result["phase"] == "COMPLETED"
    assert resumed["phase"] == "COMPLETED"
    assert [item["cluster_id"] for item in updated.release_config["clusters"]] == [
        "gpu-a",
        "hp-gpu-b",
    ]
    assert {
        "cluster/hp-gpu-b/eks",
        "cluster/hp-gpu-b/hyperpod",
        "aws/iam/executor/hp-gpu-b/role",
        "aws/route53/vpc-association/hp-gpu-b",
    }.issubset(keys), "Aurora registry omitted joined cluster resources"
    assert rollout_calls == [
        ("preflight", None),
        ("verify", None),
        ("preflight", None),
        ("join-cluster", "hp-gpu-b"),
        ("verify", None),
        ("verify", None),
    ]
    assert release_state_syncs == [["gpu-a", "hp-gpu-b"]]
    state = json.loads((state_dir / "state.json").read_text())
    assert "RELEASE_STATE_UPDATED" in state["completed_steps"]
    assert yaml.safe_load(path.read_text())["spec"]["gpuKubeconfig"] == str(
        gpu_kubeconfig
    )


def test_join_cluster_rolls_back_before_site_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    state_dir = tmp_path / "join-state"
    rolled_back = []
    monkeypatch.setattr(
        admin_cluster_join,
        "_prepare_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BootstrapError("prerequisite failed")
        ),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_rollback",
        lambda *_args, **_kwargs: rolled_back.append(True),
    )

    with pytest.raises(BootstrapError, match="prerequisite failed"):
        join_cluster(
            JoinClusterRequest(
                site=site,
                gpu_cluster_arn=GPU_B_ARN,
                allowed_namespaces=(),
                state_dir=state_dir,
            ),
            runner=Runner(),
        )

    assert rolled_back == [True]
    assert len(load_site(site.source).release_config["clusters"]) == 1


def test_join_cluster_failure_after_site_commit_is_fail_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    state_dir = tmp_path / "join-state"
    target = _target()
    candidate = state_dir / "candidate.yaml"
    state_dir.mkdir(parents=True)
    document = yaml.safe_load(path.read_text())
    document["spec"]["clusters"].append(
        {
            "clusterId": "hp-gpu-b",
            "context": target.context,
            "region": target.region,
            "hyperpodClusterName": target.hyperpod_name,
            "eksClusterArn": target.eks_arn,
            "executorIrsaRoleArn": "arn:aws:iam::123456789012:role/gpu-b",
            "allowedNamespaces": ["gpu-fault-system", "training"],
            "controlPlaneUrl": "https://control.example",
            "tokenFile": document["spec"]["clusters"][0]["tokenFile"],
            "caFile": document["spec"]["clusters"][0]["caFile"],
            "fleetMasterFile": document["spec"]["clusters"][0]["fleetMasterFile"],
        }
    )
    candidate.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    candidate.chmod(0o600)
    execution = JoinExecution(
        target=target,
        cluster_id="hp-gpu-b",
        discovery={"registry_snapshot": str(state_dir / "before.json")},
        local={},
        prerequisites={},
        candidate=load_site(candidate, repository_root=site.repository_root),
    )
    monkeypatch.setattr(
        admin_cluster_join, "_prepare_execution", lambda *_args, **_kwargs: execution
    )

    def fail_after_commit(request, **_kwargs):
        path.write_bytes(candidate.read_bytes())
        raise BootstrapError("registry unavailable")

    monkeypatch.setattr(admin_cluster_join, "_deploy_and_commit", fail_after_commit)
    monkeypatch.setattr(
        admin_cluster_join,
        "_rollback",
        lambda *_args, **_kwargs: pytest.fail("committed site was rolled back"),
    )

    with pytest.raises(BootstrapError, match="registry unavailable"):
        join_cluster(
            JoinClusterRequest(
                site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir
            ),
            runner=Runner(),
        )

    state = json.loads((state_dir / "state.json").read_text())
    assert state["phase"] == "FAILED_AFTER_COMMIT"
    assert len(load_site(path).release_config["clusters"]) == 2


def test_join_waits_for_collector_freshness(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = iter(
        [
            {"ready": False, "nodes": [{"ready": False}]},
            {"ready": True, "nodes": [{"ready": True}]},
        ]
    )
    monkeypatch.setitem(
        wait_collector_readiness.__globals__,
        "_collector_readiness_report",
        lambda *_args: next(reports),
    )

    result = wait_collector_readiness(
        object(), "gpu-b", timeout_seconds=1, interval_seconds=0
    )

    assert result["ready"] is True


def test_completed_join_state_starts_a_new_attempt_after_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["clusters"] = []
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    site = load_site(path)
    state_dir = tmp_path / "join-state"
    state_dir.mkdir(mode=0o700)
    state = {
        "schema_version": 1,
        "site_id": "test-site",
        "gpu_cluster_arn": GPU_B_ARN,
        "requested_cluster_id": "",
        "allowed_namespaces": [],
        "attempt": 1,
        "source_site_sha256": "old",
        "phase": "COMPLETED",
        "completed_steps": ["PRECHECKED", "DISCOVERED"],
        "evidence": {
            "DISCOVERED": {"cluster_id": "hp-gpu-b", "target": asdict(_target())}
        },
    }
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    observed = {}

    def prepare(_request, **kwargs):
        observed.update(kwargs["state"])
        return {"phase": "NEW_ATTEMPT"}

    monkeypatch.setattr(admin_cluster_join, "_prepare_execution", prepare)

    result = join_cluster(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=GPU_B_ARN,
            allowed_namespaces=(),
            state_dir=state_dir,
        ),
        runner=Runner(),
    )

    assert result == {"phase": "NEW_ATTEMPT"}
    assert observed["attempt"] == 2
    assert observed["completed_steps"] == []
    assert observed["source_site_sha256"] == site.source_sha256
    assert (state_dir / "state.attempt-001.json").is_file(), (
        "previous completed join evidence was not archived"
    )


def test_completed_join_repairs_missing_release_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    state_dir = tmp_path / "join-state"
    state_dir.mkdir(mode=0o700)
    state = {
        "schema_version": 1,
        "site_id": "test-site",
        "gpu_cluster_arn": site.release_config["clusters"][0]["eks_cluster_arn"],
        "requested_cluster_id": "",
        "allowed_namespaces": [],
        "attempt": 1,
        "source_site_sha256": site.source_sha256,
        "phase": "COMPLETED",
        "completed_steps": ["DISCOVERED", "SITE_UPDATED", "FINAL_VERIFIED"],
        "evidence": {
            "DISCOVERED": {
                "cluster_id": "gpu-a",
                "target": {
                    "eks_arn": site.release_config["clusters"][0]["eks_cluster_arn"]
                },
            }
        },
    }
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    synced: list[list[str]] = []
    monkeypatch.setattr(
        admin_cluster_join,
        "_sync_release_state",
        lambda current: synced.append(
            [item["cluster_id"] for item in current.release_config["clusters"]]
        ),
    )

    result = join_cluster(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=site.release_config["clusters"][0]["eks_cluster_arn"],
            allowed_namespaces=(),
            state_dir=state_dir,
        ),
        runner=Runner(),
    )

    updated_state = json.loads((state_dir / "state.json").read_text())
    assert result["phase"] == "COMPLETED"
    assert synced == [["gpu-a"]]
    assert updated_state["phase"] == "COMPLETED"
    assert "RELEASE_STATE_UPDATED" in updated_state["completed_steps"]
