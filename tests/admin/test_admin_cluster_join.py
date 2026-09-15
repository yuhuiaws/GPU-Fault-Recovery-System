from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as admin_cluster_batch_join
from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_commit as admin_cluster_join_commit
from gpu_fault.admin import cluster_join_state as admin_cluster_join_state
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_batch_join import join_clusters
from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinExecution, join_cluster
from gpu_fault.admin.cluster_join_evidence import JoinVerificationExpired
from gpu_fault.admin.resource_registry_dns import vpc_association_resource
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file

JOINED_KEYS = {
    "cluster/hp-gpu-b/eks",
    "cluster/hp-gpu-b/hyperpod",
    "aws/iam/executor/hp-gpu-b/role",
    "aws/iam/adot-writer/hp-gpu-b/role",
    "aws/iam/executor/hp-gpu-b/oidc-provider",
    "aws/route53/vpc-association/hp-gpu-b",
}


def _readiness_evidence(nodes: int = 4) -> dict:
    """What the COLLECTORS_READY gate records: fast kinds waited, slow deferred."""

    return {
        "ready": True,
        "node_count": nodes,
        "nodes": [f"node-{index}" for index in range(nodes)],
        "waited_kinds": ["GPU_INVENTORY", "HOST_TELEMETRY"],
        "deferred_kinds": {"GPU_METRICS": {"verified_as": "scheduled"}},
        "fleet_ready": True,
    }


def _verify_report() -> dict:
    return {
        "mode": "verify",
        "healthy": True,
        "summary": {"PASS": 8, "WARN": 0, "FAIL": 0, "SKIP": 0},
        "checks": [{"name": "control_api", "status": "PASS"}],
        "scope": {"skipped_checks": {"monitoring": "..."}},
    }


GPU_B_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"
ADOT_WRITER_B = (
    "arn:aws:iam::123456789012:role/gpu-fault-test-site-hp-gpu-b-adot-writer"
)


def _membership_snapshot(site) -> dict:
    states = {
        item["cluster_id"]: ("ACTIVE" if item["cluster_id"] == "gpu-a" else "PENDING")
        for item in site.release_config["clusters"]
    }
    return {
        "live_release_state_sha256": "a" * 64,
        "live_release_identity_sha256": "b" * 64,
        "registry_generation": 2,
        "registry_content_sha256": "c" * 64,
        "registry_cluster_states": states,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


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
    def run(self, arguments, **_kwargs):
        if "update-kubeconfig" in arguments:
            path = Path(arguments[arguments.index("--kubeconfig") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("kubeconfig", encoding="utf-8")
        return ""


def _joined_prerequisites() -> dict:
    """What ``_parallel_prerequisites`` records for gpu-b: both per-cluster roles,
    the network it joined and its node keys."""

    provider = "arn:aws:iam::123456789012:oidc-provider/issuer-b"
    return {
        "executor_role": {
            "role_arn": "arn:aws:iam::123456789012:role/gpu-b-executor",
            "ownership": "CREATED",
            "inline_policy_name": "GPUFaultRegionalExecutor",
            "oidc_provider_arn": provider,
            "oidc_provider_ownership": "EXTERNAL",
            "cluster_name": "gpu-b",
        },
        "adot_writer_role": {
            "role_arn": ADOT_WRITER_B,
            "ownership": "CREATED",
            "inline_policy_name": "GPUFaultDataplaneAmpWriter",
            "oidc_provider_arn": provider,
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


def _seed_bootstrap_state(site_path: Path) -> Path:
    """A checkpoint beside site.yaml holding the zone and its CPU VPC association, so
    the join's real ``_update_bootstrap_state`` has something to append to."""

    path = site_path.parent / "bootstrap-state.json"
    path.write_text(
        json.dumps(
            {
                "site_id": "test-site",
                "resources": {
                    "pki": {
                        "hosted_zone_id": "Z123",
                        "vpc_associations": [
                            {
                                "vpc_id": "vpc-cpu",
                                "vpc_region": "us-east-1",
                                "ownership": "CREATED",
                            }
                        ],
                    }
                },
                "completed_tasks": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _assert_association_committed(
    snapshot: InstallationResourceSnapshot, bootstrap_state: Path
) -> None:
    """The join's Route53 association row is the shared registry shape and its
    checkpoint entry names the cluster the registry keys it by (finding A)."""

    association = next(
        item
        for item in snapshot.resources
        if item.resource_key == "aws/route53/vpc-association/hp-gpu-b"
    )
    expected = vpc_association_resource(
        site_id="test-site",
        owner="hp-gpu-b",
        hosted_zone_id="Z123",
        vpc_id="vpc-gpu-b",
        vpc_region="us-east-1",
        region="us-east-1",
        account_id="123456789012",
    )
    timestamps = {"created_at", "updated_at"}
    assert association.model_dump(exclude=timestamps) == expected.model_dump(
        exclude=timestamps
    ), "the join registers its association in a shape of its own"
    recorded = json.loads(bootstrap_state.read_text(encoding="utf-8"))
    assert recorded["resources"]["pki"]["vpc_associations"][-1] == {
        "vpc_id": "vpc-gpu-b",
        "vpc_region": "us-east-1",
        "ownership": "CREATED",
        "cluster_ids": ["hp-gpu-b"],
    }, "the checkpoint entry does not name the cluster the registry keys it by"
    assert recorded["joined_clusters"]["hp-gpu-b"]["vpc_id"] == "vpc-gpu-b"


def _assert_adot_writer_committed(snapshot: InstallationResourceSnapshot, site) -> None:
    """The created ADOT writer role is registered for cleanup and named in the site."""

    adot_row = next(
        item
        for item in snapshot.resources
        if item.resource_key == "aws/iam/adot-writer/hp-gpu-b/role"
    )
    assert adot_row.resource_type == "iam_role"
    assert adot_row.resource_id == "gpu-fault-test-site-hp-gpu-b-adot-writer"
    assert adot_row.delete_policy is InstallationResourceDeletePolicy.DELETE, (
        "remove-cluster would leave the ADOT writer role behind"
    )
    assert adot_row.attributes["inline_policy_name"] == "GPUFaultDataplaneAmpWriter"
    joined = next(
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] == "hp-gpu-b"
    )
    assert joined["adot_irsa_role_arn"] == ADOT_WRITER_B, (
        "the created ADOT writer role did not reach the committed site"
    )


def _assert_commit_evidence_is_scoped(state: dict, state_dir: Path) -> None:
    """Each commit step records the scope it worked at, for the operator."""

    evidence = state["evidence"]
    assert evidence["COLLECTORS_READY"]["deferred_kinds"] == {
        "GPU_METRICS": {"verified_as": "scheduled"}
    }, "the operator cannot see which collector kinds the gate deferred"
    assert evidence["REGISTRY_UPDATED"]["mode"] == "delta"
    assert set(evidence["REGISTRY_UPDATED"]["delta_keys"]) == JOINED_KEYS
    assert evidence["RELEASE_STATE_UPDATED"]["capture_scope"] == "cluster:hp-gpu-b"
    assert evidence["VERIFIED"]["verify"]["skipped_checks"] == ["monitoring"]
    after = admin_cluster_join.load_installation_resource_snapshot(
        state_dir / "installation-resources-after-001.json"
    )
    assert {item.resource_key for item in after.resources} == JOINED_KEYS | {
        "aws/nlb"
    }, "the after snapshot is the before snapshot plus the delta"


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
    monkeypatch.setattr(
        admin_cluster_join, "clear_stale_installer_annotations", lambda *_a: None
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_existing_cluster_networks",
        lambda *_args, **_kwargs: [{"vpc_id": "vpc-gpu-a", "nat_eips": ["192.0.2.10"]}],
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_parallel_prerequisites",
        lambda *_args, **_kwargs: _joined_prerequisites(),
    )
    bootstrap_state = _seed_bootstrap_state(path)
    monkeypatch.setattr(
        admin_cluster_join,
        "sync_cluster_release_state",
        lambda current, *, cluster_id: release_state_syncs.append(
            [item["cluster_id"] for item in current.release_config["clusters"]]
        )
        or {"capture_scope": f"cluster:{cluster_id}"},
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_verify_candidate",
        lambda _candidate, readiness: rollout_calls.append(
            ("verify", ",".join(sorted(readiness)))
        )
        or _verify_report(),
    )
    failure_domain_renders: list[list[str]] = []
    monkeypatch.setattr(
        admin_cluster_join,
        "refresh_failure_domain_map",
        lambda current: failure_domain_renders.append(
            [
                item["cluster_id"]
                for item in load_site(
                    current.source, repository_root=current.repository_root
                ).release_config["clusters"]
            ]
        ),
    )
    readiness_waits: list[list[str] | None] = []
    monkeypatch.setattr(
        admin_cluster_join,
        "wait_join_collector_readiness",
        lambda _site, _cluster_id, *, expected_nodes: readiness_waits.append(
            expected_nodes
        )
        or _readiness_evidence(),
    )
    monkeypatch.setattr(
        admin_cluster_join, "membership_runtime_snapshot", _membership_snapshot
    )
    monkeypatch.setattr(
        admin_cluster_join_commit,
        "validate_verified_membership",
        lambda **_kwargs: _membership_snapshot(site),
    )
    monkeypatch.setattr(
        admin_cluster_join_commit,
        "final_membership_identity",
        lambda _site, **kwargs: {
            **kwargs,
            "live_release_state_sha256": "a" * 64,
            "live_release_identity_sha256": "b" * 64,
            "registry_generation": 3,
            "registry_content_sha256": "c" * 64,
            "registry_lifecycle": "ACTIVE",
        },
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
    assert keys == JOINED_KEYS, (
        "the registry write must be exactly the join's delta, not the whole site"
    )
    assert len(synced) == 1, "a resumed COMPLETED join re-sent the registry delta"
    _assert_adot_writer_committed(synced[-1], updated)
    _assert_association_committed(synced[-1], bootstrap_state)
    assert rollout_calls == [
        ("preflight", None),
        ("join-cluster", "hp-gpu-b"),
        ("verify", "hp-gpu-b"),
        ("activate-cluster", "hp-gpu-b"),
    ], "the join ran a baseline verify the candidate verify already answers"
    assert readiness_waits == [["node-b"]], (
        "the readiness gate must be asked about the cluster's HyperPod nodes"
    )
    assert release_state_syncs == [["gpu-a", "hp-gpu-b"]]
    assert failure_domain_renders == [["gpu-a", "hp-gpu-b"]], (
        "the failure-domain map is re-rendered once, from the committed site"
    )
    state = json.loads((state_dir / "state.json").read_text())
    assert "RELEASE_STATE_UPDATED" in state["completed_steps"]
    token_file = Path(state["evidence"]["LOCAL_INPUTS_READY"]["token_file"])
    assert token_file.parent == site.source.parent / "secure", (
        "the cluster token was written into the disposable join state directory"
    )
    assert token_file.is_file(), "the cluster token file was not written"
    assert state["evidence"]["FINAL_VERIFIED"]["registry_generation"] == 3
    assert state["evidence"]["FINAL_VERIFIED"]["live_release_state_sha256"] == "a" * 64
    _assert_commit_evidence_is_scoped(state, state_dir)
    assert yaml.safe_load(path.read_text())["spec"]["gpuKubeconfig"] == str(
        gpu_kubeconfig
    )


def _prerequisite_stubs(
    monkeypatch: pytest.MonkeyPatch, order: list[str]
) -> threading.Event:
    executor_done = threading.Event()
    lock = threading.Lock()

    def note(name: str) -> None:
        with lock:
            order.append(name)

    def executor_role(_runner, **_kwargs):
        time.sleep(0.05)
        note("executor_role")
        executor_done.set()
        return {"role_arn": "arn:aws:iam::123456789012:role/gpu-b-executor"}

    def adot_writer_role(_runner, **kwargs):
        assert executor_done.is_set(), (
            "the ADOT writer role raced the executor role for the OIDC provider"
        )
        note("adot_writer_role")
        return {
            "role_arn": ADOT_WRITER_B,
            "workspace": kwargs["amp_workspace_id"],
            "namespace": kwargs["namespace"],
            "site_id": kwargs["site_id"],
        }

    monkeypatch.setattr(admin_cluster_join, "ensure_executor_role", executor_role)
    monkeypatch.setattr(admin_cluster_join, "ensure_adot_writer_role", adot_writer_role)
    monkeypatch.setattr(
        admin_cluster_join,
        "_ensure_network",
        lambda _runner, **_kwargs: note("network") or {"vpc_id": "vpc-gpu-b"},
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "provision_node_action_keys",
        lambda _runner, **_kwargs: note("node_keys") or {"cluster_id": "hp-gpu-b"},
    )
    return executor_done


def _prerequisites(site, *, tmp_path: Path, state: dict) -> dict:
    return admin_cluster_join._parallel_prerequisites(
        JoinClusterRequest(site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=tmp_path),
        runner=Runner(),
        target=_target(),
        cluster_id="hp-gpu-b",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        fleet_master_file=tmp_path / "fleet-master",
        state=state,
        state_path=tmp_path / "state.json",
    )


def test_join_prerequisites_create_the_adot_writer_role_after_the_executor_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """join-cluster makes the data-plane collector's role like bootstrap does.

    It runs after the executor role (same OIDC provider; two concurrent
    ``create-open-id-connect-provider`` calls would fail one of them) and reads
    the workspace from the site, so a joined cluster gets its collector without
    the operator hand-making a role.
    """

    site = load_site(site_file(tmp_path))
    order: list[str] = []
    _prerequisite_stubs(monkeypatch, order)
    state: dict = {}

    result = _prerequisites(site, tmp_path=tmp_path, state=state)

    assert result["adot_writer_role"] == {
        "role_arn": ADOT_WRITER_B,
        "workspace": "ws-test",
        "namespace": "gpu-fault-system",
        "site_id": "test-site",
    }, "the writer role was not created from the site's workspace and namespace"
    assert order.index("adot_writer_role") > order.index("executor_role")
    assert state["evidence"]["PREREQUISITES_READY"]["adot_writer_role"]["role_arn"] == (
        ADOT_WRITER_B
    ), "the writer role is not persisted with the other prerequisites"


def test_join_prerequisites_skip_the_adot_writer_role_without_a_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No workspace, no collector, no role -- the same rule the release applies."""

    site = load_site(site_file(tmp_path))
    site.release_config["health"]["amp_workspace_id"] = None
    order: list[str] = []
    _prerequisite_stubs(monkeypatch, order)

    result = _prerequisites(site, tmp_path=tmp_path, state={})

    assert "adot_writer_role" not in result, (
        "a writer role was created for a site with no AMP workspace"
    )
    assert "adot_writer_role" not in order
    assert set(order) == {"executor_role", "network", "node_keys"}


def test_a_resumed_join_does_not_recreate_a_recorded_adot_writer_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    order: list[str] = []
    _prerequisite_stubs(monkeypatch, order)
    recorded = {"role_arn": ADOT_WRITER_B, "inline_policy_name": "x"}
    state = {"evidence": {"PREREQUISITES_READY": {"adot_writer_role": recorded}}}

    result = _prerequisites(site, tmp_path=tmp_path, state=state)

    assert result["adot_writer_role"] == recorded, "a cached prerequisite was redone"
    assert "adot_writer_role" not in order


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


def test_join_cluster_failure_after_site_commit_restores_original_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    original = path.read_bytes()
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

    def rollback(_request, *, state_path, state, **_kwargs):
        path.write_bytes(original)
        path.chmod(0o600)
        state["phase"] = "ROLLED_BACK"
        state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(admin_cluster_join, "_rollback", rollback)

    with pytest.raises(BootstrapError, match="registry unavailable"):
        join_cluster(
            JoinClusterRequest(
                site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir
            ),
            runner=Runner(),
        )

    state = json.loads((state_dir / "state.json").read_text())
    assert state["phase"] == "ROLLED_BACK"
    assert len(load_site(path).release_config["clusters"]) == 1


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


def _batch_execution(
    tmp_path: Path, site, request: JoinClusterRequest, index: int
) -> JoinExecution:
    cluster_id = f"gpu-{index}"
    target = replace(
        _target(),
        input_arn=request.gpu_cluster_arn,
        hyperpod_arn=(f"arn:aws:sagemaker:us-east-1:123456789012:cluster/{cluster_id}"),
        hyperpod_name=cluster_id,
        eks_arn=request.gpu_cluster_arn,
        eks_name=cluster_id,
        vpc_id=f"vpc-{cluster_id}",
        context=f"context-{cluster_id}",
    )
    document = yaml.safe_load(site.source.read_text(encoding="utf-8"))
    cluster = dict(document["spec"]["clusters"][0])
    cluster.update(
        {
            "clusterId": cluster_id,
            "context": target.context,
            "hyperpodClusterName": target.hyperpod_name,
            "eksClusterArn": target.eks_arn,
            "executorIrsaRoleArn": (
                f"arn:aws:iam::123456789012:role/{cluster_id}-executor"
            ),
        }
    )
    document["spec"]["clusters"].append(cluster)
    candidate_path = tmp_path / f"{cluster_id}.yaml"
    candidate_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    candidate_path.chmod(0o600)
    gpu_kubeconfig = tmp_path / "gpu.kubeconfig"
    gpu_kubeconfig.write_text("kubeconfig", encoding="utf-8")
    gpu_kubeconfig.chmod(0o600)
    return JoinExecution(
        target=target,
        cluster_id=cluster_id,
        discovery={"registry_snapshot": str(tmp_path / "registry.json")},
        local={"gpu_kubeconfig": str(gpu_kubeconfig)},
        prerequisites={
            "executor_role": {
                "role_arn": (f"arn:aws:iam::123456789012:role/{cluster_id}-executor"),
                "inline_policy_name": "GPUFaultRegionalExecutor",
            },
            "network": {
                "vpc_id": f"vpc-{cluster_id}",
                "existing_vpc_ids": [],
                "nat_eips": [],
            },
        },
        candidate=load_site(candidate_path, repository_root=site.repository_root),
    )


def _site_allowing_parallel_clusters(tmp_path: Path, count: int):
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["release"]["upgradeMaxParallelClusters"] = count
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_site(path)


def _patch_batch_discovery(
    monkeypatch: pytest.MonkeyPatch, executions, verified: list[str] | None = None
) -> None:
    monkeypatch.setattr(
        admin_cluster_join,
        "_prepare_execution",
        lambda request, **_kwargs: executions[request.gpu_cluster_arn],
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_verify_candidate",
        lambda _candidate, _readiness: (
            verified.append("verify") if verified is not None else None
        )
        or _verify_report(),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_discover_join_target",
        lambda request, _runner: (
            executions[request.gpu_cluster_arn].target,
            executions[request.gpu_cluster_arn].cluster_id,
            None,
        ),
    )
    monkeypatch.setattr(
        admin_cluster_join, "_export_registry", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        admin_cluster_join, "_existing_cluster_networks", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        admin_cluster_batch_join, "membership_runtime_snapshot", _membership_snapshot
    )


def test_batch_join_rolls_as_many_clusters_as_the_release_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cluster parallelism is the release engine's setting, not the batch's.

    ``spec.release.upgradeMaxParallelClusters`` is what a deploy honours; a batch
    join that rolled a different number would be a second, undocumented policy.
    Discovery and the candidate preflight/verify stay shared across the batch.
    """

    site = _site_allowing_parallel_clusters(tmp_path, 4)
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=(f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{index}"),
        )
        for index in range(6)
    )
    executions = {
        request.gpu_cluster_arn: _batch_execution(tmp_path, site, request, index)
        for index, request in enumerate(requests)
    }
    rollout_modes: list[str] = []
    active = 0
    maximum = 0
    lock = threading.Lock()
    four_started = threading.Event()
    committed: list[str] = []

    monkeypatch.setattr(
        admin_cluster_join,
        "_run_rollout",
        lambda _site, mode, **_kwargs: rollout_modes.append(mode),
    )
    _patch_batch_discovery(monkeypatch, executions, rollout_modes)

    def deploy(*, execution, **_kwargs) -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_started.set()
        assert four_started.wait(timeout=2), "four join workers did not start"
        time.sleep(0.01)
        with lock:
            active -= 1

    monkeypatch.setattr(admin_cluster_join, "_deploy_cluster", deploy)
    monkeypatch.setattr(
        admin_cluster_join,
        "_activate_and_commit",
        lambda _request, *, execution, **_kwargs: committed.append(
            execution.cluster_id
        ),
    )

    result = join_clusters(requests, runner_factory=Runner)

    assert maximum == 4, "batch join did not roll upgradeMaxParallelClusters at once"
    assert rollout_modes.count("preflight") == 1
    assert rollout_modes.count("verify") == 1, (
        "the batch ran a baseline verify besides the one candidate verify"
    )
    assert committed == sorted(
        execution.cluster_id for execution in executions.values()
    )
    assert result["joined"] == committed
    assert result["deploy_concurrency"] == 4
    assert [item["cluster_id"] for item in result["clusters"]] == committed
    assert not (tmp_path / "join-cluster/batches").exists(), (
        "the batch wrote its own state tree beside the per-cluster records"
    )


def test_batch_join_rolls_one_cluster_at_a_time_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The site default is one cluster at a time; the batch must not widen it."""

    site = load_site(site_file(tmp_path))
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=(f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{index}"),
        )
        for index in range(3)
    )
    executions = {
        request.gpu_cluster_arn: _batch_execution(tmp_path, site, request, index)
        for index, request in enumerate(requests)
    }
    active = 0
    maximum = 0
    lock = threading.Lock()
    monkeypatch.setattr(
        admin_cluster_join, "_run_rollout", lambda *_args, **_kwargs: None
    )
    _patch_batch_discovery(monkeypatch, executions)

    def deploy(*, execution, **_kwargs) -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    monkeypatch.setattr(admin_cluster_join, "_deploy_cluster", deploy)
    monkeypatch.setattr(
        admin_cluster_join, "_activate_and_commit", lambda *_args, **_kwargs: None
    )

    result = join_clusters(requests, runner_factory=Runner)

    assert maximum == 1, "batch join rolled more clusters than the site allows"
    assert result["deploy_concurrency"] == 1


def test_batch_join_keeps_successful_clusters_when_one_rollout_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=(f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{index}"),
        )
        for index in range(3)
    )
    executions = {
        request.gpu_cluster_arn: _batch_execution(tmp_path, site, request, index)
        for index, request in enumerate(requests)
    }
    committed: list[str] = []
    rolled_back: list[str] = []
    monkeypatch.setattr(
        admin_cluster_join, "_run_rollout", lambda *_args, **_kwargs: None
    )
    _patch_batch_discovery(monkeypatch, executions)

    def deploy(*, execution, **_kwargs) -> None:
        if execution.cluster_id == "gpu-1":
            raise BootstrapError("rollout failed")

    monkeypatch.setattr(admin_cluster_join, "_deploy_cluster", deploy)
    monkeypatch.setattr(
        admin_cluster_join,
        "_record_join_failure",
        lambda attempt, execution: rolled_back.append(
            execution.cluster_id if execution is not None else "unknown"
        ),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_activate_and_commit",
        lambda _request, *, execution, **_kwargs: committed.append(
            execution.cluster_id
        ),
    )

    with pytest.raises(BootstrapError, match="batch join completed") as failure:
        join_clusters(requests, runner_factory=Runner)

    assert rolled_back == ["gpu-1"]
    assert committed == ["gpu-0", "gpu-2"]
    # The summary that used to live in a separate batch state file is in the
    # message the operator reads.
    assert '"joined": ["gpu-0", "gpu-2"]' in str(failure.value)
    assert '"phase": "PARTIAL"' in str(failure.value)
    assert not (tmp_path / "join-cluster/batches").exists(), (
        "the batch wrote its own state tree beside the per-cluster records"
    )


def test_batch_join_re_verifies_expired_evidence_instead_of_rolling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commits are sequential, so a late cluster can outlive the 15-minute window.

    Its data plane is joined and healthy; the only thing wrong is the age of the
    evidence. Re-verifying what is left of the batch is the answer, a rollback of
    a working cluster is not.
    """

    site = load_site(site_file(tmp_path))
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=(f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{index}"),
        )
        for index in range(3)
    )
    executions = {
        request.gpu_cluster_arn: _batch_execution(tmp_path, site, request, index)
        for index, request in enumerate(requests)
    }
    rollout_modes: list[str] = []
    committed: list[str] = []
    rolled_back: list[str] = []
    expired_once: list[str] = []
    monkeypatch.setattr(
        admin_cluster_join,
        "_run_rollout",
        lambda _site, mode, **_kwargs: rollout_modes.append(mode),
    )
    _patch_batch_discovery(monkeypatch, executions, rollout_modes)
    monkeypatch.setattr(admin_cluster_join, "_deploy_cluster", lambda **_kwargs: None)
    monkeypatch.setattr(
        admin_cluster_join,
        "_record_join_failure",
        lambda attempt, execution: rolled_back.append(
            execution.cluster_id if execution is not None else "unknown"
        ),
    )

    def activate(_request, *, execution, state, **_kwargs) -> None:
        if execution.cluster_id == "gpu-1" and not expired_once:
            expired_once.append(execution.cluster_id)
            raise JoinVerificationExpired("join verification evidence expired")
        assert "VERIFIED" in state["completed_steps"], "commit ran without evidence"
        committed.append(execution.cluster_id)

    monkeypatch.setattr(admin_cluster_join, "_activate_and_commit", activate)

    result = join_clusters(requests, runner_factory=Runner)

    assert rolled_back == [], "an expired verification rolled a healthy cluster back"
    assert committed == ["gpu-0", "gpu-1", "gpu-2"]
    assert rollout_modes.count("verify") == 2, (
        "the remaining clusters were not verified again after the window closed"
    )
    assert result["joined"] == committed


def test_join_re_verifies_expired_evidence_instead_of_rolling_back(
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
    rollout_modes: list[str] = []
    rolled_back: list[str] = []
    expired_once: list[bool] = []
    monkeypatch.setattr(
        admin_cluster_join, "_prepare_execution", lambda *_args, **_kwargs: execution
    )
    monkeypatch.setattr(admin_cluster_join, "_deploy_cluster", lambda **_kwargs: None)
    monkeypatch.setattr(
        admin_cluster_join,
        "_verify_candidate",
        lambda _candidate, _readiness: rollout_modes.append("verify")
        or _verify_report(),
    )
    monkeypatch.setattr(
        admin_cluster_join, "membership_runtime_snapshot", _membership_snapshot
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_rollback",
        lambda *_args, **_kwargs: rolled_back.append("rolled back"),
    )

    def activate(_request, *, state, **_kwargs) -> None:
        if not expired_once:
            expired_once.append(True)
            raise JoinVerificationExpired("join verification evidence expired")
        assert "VERIFIED" in state["completed_steps"], "commit ran without evidence"

    monkeypatch.setattr(admin_cluster_join, "_activate_and_commit", activate)

    result = join_cluster(
        JoinClusterRequest(site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir),
        runner=Runner(),
    )

    assert result["phase"] == "COMPLETED"
    assert rolled_back == [], "an expired verification rolled a healthy cluster back"
    assert rollout_modes == ["verify", "verify"]
    state = json.loads((state_dir / "state.json").read_text())
    assert "VERIFIED" in state["completed_steps"]


def test_join_resumes_after_a_crash_between_commit_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process dies after the registry delta is written and before the
    cluster is activated; the resumed attempt neither redoes the finished
    steps nor re-sends the delta, and finishes from where it stopped."""

    path = site_file(tmp_path)
    site = load_site(path)
    state_dir = tmp_path / "join-state"
    gpu_kubeconfig = tmp_path / "secure/gpu.kubeconfig"
    synced: list[InstallationResourceSnapshot] = []
    verifies: list[str] = []
    release_state_syncs: list[str] = []
    activations: list[str] = []

    def rollout(_site, mode, *, cluster_id=None):
        if mode == "activate-cluster":
            activations.append(mode)
            if len(activations) == 1:
                raise KeyboardInterrupt  # the operator's terminal died here

    monkeypatch.setattr(admin_cluster_join, "_run_rollout", rollout)
    monkeypatch.setattr(
        admin_cluster_join, "discover_cluster", lambda *_args, **_kwargs: _target()
    )
    monkeypatch.setattr(
        admin_cluster_join, "_gpu_kubeconfig", lambda _site: gpu_kubeconfig
    )
    monkeypatch.setattr(admin_cluster_join, "ensure_namespace", lambda *_a, **_k: None)

    def base_secrets(_runner, *, secure_dir, **_kwargs):
        secret = secure_dir / "fleet-master"
        secret.write_text("f" * 64, encoding="utf-8")
        secret.chmod(0o600)
        return secret

    monkeypatch.setattr(admin_cluster_join, "_ensure_base_secrets", base_secrets)
    monkeypatch.setattr(
        admin_cluster_join,
        "_shared_ca_file",
        lambda current: Path(current.release_config["clusters"][0]["ca_file"]),
    )
    monkeypatch.setattr(
        admin_cluster_join, "_list_nodes", lambda *_args, **_kwargs: ["node-b"]
    )
    monkeypatch.setattr(
        admin_cluster_join, "clear_stale_installer_annotations", lambda *_a: None
    )
    monkeypatch.setattr(
        admin_cluster_join, "_existing_cluster_networks", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_parallel_prerequisites",
        lambda *_args, **_kwargs: _joined_prerequisites(),
    )
    monkeypatch.setattr(
        admin_cluster_join, "_update_bootstrap_state", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "sync_cluster_release_state",
        lambda _current, *, cluster_id: release_state_syncs.append(cluster_id) or {},
    )
    monkeypatch.setattr(
        admin_cluster_join, "refresh_failure_domain_map", lambda _current: None
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "wait_join_collector_readiness",
        lambda *_args, **_kwargs: _readiness_evidence(1),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "_verify_candidate",
        lambda _candidate, readiness: verifies.append(",".join(readiness))
        or _verify_report(),
    )
    monkeypatch.setattr(
        admin_cluster_join, "membership_runtime_snapshot", _membership_snapshot
    )
    monkeypatch.setattr(
        admin_cluster_join_commit,
        "validate_verified_membership",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        admin_cluster_join_commit,
        "final_membership_identity",
        lambda _site, **kwargs: {**kwargs, "registry_lifecycle": "ACTIVE"},
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "sync_installation_resource_snapshot",
        lambda _site, snapshot: synced.append(snapshot),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "fetch_installation_resource_registry",
        lambda _site, output=None: synced[-1],
    )
    before = state_dir / "installation-resources-before-001.json"
    before.parent.mkdir(parents=True, exist_ok=True)
    admin_cluster_join.write_installation_resource_snapshot(
        site, _snapshot(), path=before
    )
    request = JoinClusterRequest(
        site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir
    )

    with pytest.raises(KeyboardInterrupt):
        join_cluster(request, runner=Runner())

    crashed = json.loads((state_dir / "state.json").read_text())
    assert "REGISTRY_UPDATED" in crashed["completed_steps"]
    assert "ACTIVATION_STARTED" in crashed["completed_steps"]
    assert "ACTIVATED" not in crashed["completed_steps"]
    assert crashed["phase"] != "ROLLED_BACK", "a crash is not a failure to undo"

    result = join_cluster(
        JoinClusterRequest(
            site=load_site(path), gpu_cluster_arn=GPU_B_ARN, state_dir=state_dir
        ),
        runner=Runner(),
    )

    assert result["phase"] == "COMPLETED"
    assert activations == ["activate-cluster", "activate-cluster"]
    assert len(synced) == 1, "the resumed attempt re-sent the registry delta"
    assert release_state_syncs == ["hp-gpu-b"], "sync-state ran again on resume"
    assert verifies == ["hp-gpu-b"], "fresh VERIFIED evidence was verified again"
    state = json.loads((state_dir / "state.json").read_text())
    assert "FINAL_VERIFIED" in state["completed_steps"]
    assert state["evidence"]["FINAL_VERIFIED"]["registry_lifecycle"] == "ACTIVE"


def test_a_noted_join_failure_names_its_cause_and_the_last_completed_step() -> None:
    """The state file the operator is told to fix and resume from carries the
    cause; before, it read ROLLED_BACK and the reason lived only in the log."""

    state = {
        "attempt": 2,
        "completed_steps": ["PRECHECKED", "DISCOVERED", "JOINED"],
        "step_completed_at": {
            "PRECHECKED": "2026-09-15T05:47:47+00:00",
            "DISCOVERED": "2026-09-15T05:47:54+00:00",
            "JOINED": "2026-09-15T05:54:10+00:00",
        },
        "evidence": {},
    }

    admin_cluster_join_state.note_join_failure(
        state, RuntimeError("release component wheels and bundle must exist")
    )

    assert state["failure"]["error"] == (
        "RuntimeError: release component wheels and bundle must exist"
    )
    assert state["failure"]["after_step"] == "JOINED"
    assert state["failure"]["recorded_at"]
