"""Site commit and resource rollback edges of the join's commit module.

``activate_and_commit`` rewrites site.yaml exactly once per attempt, and only
when the file still says what the verified candidate was derived from.
``rollback_membership`` undoes the registry rows of a join that never reached the
site. These tests pin the refusals around both and the bookkeeping that keeps
unrelated registry rows intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_commit as commit
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_join_types import JoinClusterRequest, JoinExecution
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from tests.admin.test_admin_cluster_join_rollback import Attempt

CLUSTER = "hp-gpu-b"
GPU_B_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"
ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-gpu-b-executor"
ADOT_ARN = "arn:aws:iam::123456789012:role/gpu-fault-gpu-b-adot-writer"
PROVIDER_ARN = "arn:aws:iam::123456789012:oidc-provider/oidc.eks/id/B"


class StopAfterSiteCommit(Exception):
    """Raised by the bootstrap-state stub right after site.yaml was rewritten."""


def _identity() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=GPU_B_ARN,
        role="gpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-gpu-b",
        hyperpod_name=CLUSTER,
        eks_arn=GPU_B_ARN,
        eks_name="gpu-b",
        vpc_id="vpc-gpu-b",
        subnet_ids=("subnet-b",),
        node_recovery="None",
        context="gpu-fault-gpu-2-gpu-b",
    )


class Join:
    """A prepared join whose site-level collaborators are recorded, not run."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attempt = Attempt(tmp_path)
        self.site = self.attempt.site
        self.state_dir = self.attempt.state_dir
        candidate = _site_document(self.attempt.candidate_path)
        candidate["spec"]["gpuKubeconfig"] = str(self.state_dir / "gpu.kubeconfig")
        self.attempt.candidate_path.write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        self.synced: list[InstallationResourceSnapshot] = []
        self.written: list[InstallationResourceSnapshot] = []
        self.release_syncs: list[str | None] = []
        self.rollouts: list[tuple[str, str | None]] = []
        self.live: InstallationResourceSnapshot | None = None
        dns = self.site.release_config.get("dns") or {}
        self.network: dict[str, Any] = {
            "vpc_id": "vpc-gpu-b",
            "nat_eips": [],
            "association_created": True,
            "hosted_zone_id": dns.get("hosted_zone_id") or "Z123",
        }
        self.prerequisites: dict[str, Any] = {
            "executor_role": {
                "role_arn": ROLE_ARN,
                "inline_policy_name": "GPUFaultRegionalExecutor",
                "oidc_provider_arn": PROVIDER_ARN,
                "oidc_provider_ownership": "CREATED",
            },
            "network": self.network,
            "adot_writer_role": {"role_arn": ADOT_ARN, "inline_policy_name": "AMP"},
        }
        monkeypatch.setattr(
            admin_cluster_join, "_fetch_installation_registry", self._fetch
        )
        monkeypatch.setattr(
            admin_cluster_join, "_sync_installation_snapshot", self._sync
        )
        monkeypatch.setattr(
            admin_cluster_join, "_write_installation_snapshot", self._write
        )
        monkeypatch.setattr(
            admin_cluster_join,
            "_sync_join_release_state",
            lambda _site, cluster_id=None: self.release_syncs.append(cluster_id),
        )
        monkeypatch.setattr(
            admin_cluster_join,
            "_run_rollout",
            lambda _site, mode, *, cluster_id=None: self.rollouts.append(
                (mode, cluster_id)
            ),
        )

    def _fetch(self, _site: Any) -> InstallationResourceSnapshot:
        if self.live is None:
            raise AssertionError("the test did not provide a live registry")
        return self.live

    def _sync(self, _site: Any, snapshot: InstallationResourceSnapshot) -> None:
        self.synced.append(snapshot)

    def _write(
        self, _site: Any, snapshot: InstallationResourceSnapshot, **_kw: Any
    ) -> Path:
        self.written.append(snapshot)
        return self.state_dir / "registry.json"

    def request(self) -> JoinClusterRequest:
        return JoinClusterRequest(
            site=self.site, gpu_cluster_arn=GPU_B_ARN, state_dir=self.state_dir
        )

    def execution(self) -> JoinExecution:
        return JoinExecution(
            target=_identity(),
            cluster_id=CLUSTER,
            discovery={},
            local={},
            prerequisites=self.prerequisites,
            candidate=load_site(self.attempt.candidate_path),
        )

    def resource(
        self,
        key: str,
        *,
        resource_type: str,
        resource_id: str,
        arn: str | None,
        ownership: InstallationResourceOwnership = InstallationResourceOwnership.CREATED,
        policy: InstallationResourceDeletePolicy = InstallationResourceDeletePolicy.DELETE,
    ) -> InstallationResource:
        return InstallationResource(
            site_id=self.site.registry_site_id,
            resource_key=key,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_arn=arn,
            region=str(self.site.release_config["aws_region"]),
            account_id="123456789012",
            ownership=ownership,
            delete_policy=policy,
        )

    def registry(self, *resources: InstallationResource) -> None:
        self.live = InstallationResourceSnapshot(
            site_id=self.site.registry_site_id, resources=list(resources)
        )

    def unrelated(self) -> InstallationResource:
        return self.resource(
            "cluster/gpu-a/eks",
            resource_type="gpu_eks",
            resource_id="gpu-a",
            arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
            ownership=InstallationResourceOwnership.EXTERNAL,
            policy=InstallationResourceDeletePolicy.PRESERVE,
        )

    def legacy_role(self, **overrides: Any) -> InstallationResource:
        """The executor role as a legacy bootstrap stored it: without an ARN."""
        return self.resource(
            f"aws/iam/executor/{CLUSTER}/role",
            resource_type="iam_role",
            resource_id="gpu-fault-gpu-b-executor",
            arn=None,
            **overrides,
        )


def test_rolling_back_an_uncommitted_join_only_retires_its_own_registry_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.registry(join.unrelated(), join.legacy_role())
    commit.rollback_membership(join.request(), execution=join.execution(), joined=False)
    assert join.release_syncs == [] and join.rollouts == [], (
        "a join that never reached the site has nothing to sync or roll back"
    )
    (delta,) = join.synced
    assert {item.resource_key: item.status for item in delta.resources} == {
        f"aws/iam/executor/{CLUSTER}/role": InstallationResourceStatus.DELETED
    }
    (snapshot,) = join.written
    kept = {item.resource_key: item for item in snapshot.resources}
    assert kept["cluster/gpu-a/eks"] == join.unrelated().model_copy(
        update={
            "created_at": kept["cluster/gpu-a/eks"].created_at,
            "updated_at": kept["cluster/gpu-a/eks"].updated_at,
        }
    ), "rows of other clusters must pass through untouched"
    assert kept[f"aws/iam/executor/{CLUSTER}/role"].resource_arn is None, (
        "a legacy row without an ARN keeps its stored nullable field"
    )


def test_rolling_back_without_any_joined_rows_writes_but_does_not_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.registry(join.unrelated())
    commit.rollback_membership(join.request(), execution=join.execution(), joined=False)
    assert join.synced == [], "an empty delta must not be pushed to the registry"
    assert [item.resource_key for item in join.written[0].resources] == [
        "cluster/gpu-a/eks"
    ]


def test_a_joined_rollback_syncs_release_state_and_rolls_the_cluster_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.registry(join.unrelated())
    commit.rollback_membership(join.request(), execution=join.execution(), joined=True)
    assert join.release_syncs == [None]
    assert join.rollouts == [("rollback-cluster", CLUSTER)]


def test_a_registry_row_whose_identity_changed_blocks_the_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.registry(join.legacy_role(ownership=InstallationResourceOwnership.EXTERNAL))
    with pytest.raises(BootstrapError, match="identity changed before rollback"):
        commit.rollback_membership(
            join.request(), execution=join.execution(), joined=False
        )
    assert join.synced == [] and join.written == []


@pytest.mark.parametrize(
    "field, value", [("vpc_id", "vpc-other"), ("vpc_region", "eu-west-1")]
)
def test_an_association_bound_to_another_vpc_is_not_this_joins_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.network[field] = value
    join.registry(join.unrelated())
    with pytest.raises(BootstrapError, match="association target binding differs"):
        commit.rollback_membership(
            join.request(), execution=join.execution(), joined=False
        )
    assert join.written == []


def _activate(
    join: Join, monkeypatch: pytest.MonkeyPatch, state: dict[str, Any]
) -> None:
    """Run ``activate_and_commit`` up to and including the site.yaml rewrite."""
    monkeypatch.setattr(commit, "validate_verified_membership", lambda **_kw: None)
    monkeypatch.setattr(commit, "require_join_failure_domains", lambda *_a, **_kw: None)

    def stop(*_args: Any, **_kwargs: Any) -> None:
        raise StopAfterSiteCommit

    monkeypatch.setattr(admin_cluster_join, "_update_bootstrap_state", stop)
    with pytest.raises(StopAfterSiteCommit):
        commit.activate_and_commit(
            join.request(),
            execution=join.execution(),
            state_dir=join.state_dir,
            state_path=join.state_dir / "state.json",
            state=state,
        )


def _state() -> dict[str, Any]:
    return {
        "attempt": 1,
        "completed_steps": [],
        "evidence": {"VERIFIED": {"candidate_site_sha256": "c" * 64}},
    }


def _site_document(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_site_commit_appends_the_cluster_and_keeps_one_backup_per_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    backup = join.state_dir / "site.before-001.yaml"
    backup.write_text("# earlier backup of this attempt\n", encoding="utf-8")
    _activate(join, monkeypatch, _state())
    clusters = [
        item["clusterId"]
        for item in _site_document(join.site.source)["spec"]["clusters"]
    ]
    assert clusters == ["gpu-a", CLUSTER]
    assert backup.read_text(encoding="utf-8") == "# earlier backup of this attempt\n", (
        "a resumed attempt must keep the backup taken before its first rewrite"
    )


def test_the_site_commit_accepts_an_identical_cluster_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    candidate = _site_document(join.attempt.candidate_path)
    join.site.source.write_text(
        yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
    )
    join.site = load_site(join.site.source)
    _activate(join, monkeypatch, _state())
    clusters = [
        item["clusterId"]
        for item in _site_document(join.site.source)["spec"]["clusters"]
    ]
    assert clusters == ["gpu-a", CLUSTER], "the identical row must not be duplicated"


def test_the_site_commit_refuses_a_conflicting_cluster_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    candidate = _site_document(join.attempt.candidate_path)
    candidate["spec"]["clusters"][1]["allowedNamespaces"] = ["gpu-fault-system"]
    join.site.source.write_text(
        yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
    )
    join.site = load_site(join.site.source)
    before = join.site.source.read_bytes()
    with pytest.raises(BootstrapError, match="conflicting GPU cluster"):
        _activate(join, monkeypatch, _state())
    assert join.site.source.read_bytes() == before


def test_the_site_commit_refuses_a_candidate_without_the_joined_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    candidate = _site_document(join.attempt.candidate_path)
    candidate["spec"]["clusters"] = candidate["spec"]["clusters"][:1]
    join.attempt.candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
    )
    before = join.site.source.read_bytes()
    with pytest.raises(BootstrapError, match="no unique joined cluster"):
        _activate(join, monkeypatch, _state())
    assert join.site.source.read_bytes() == before


def test_the_site_commit_refuses_when_other_site_fields_moved_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    document = _site_document(join.site.source)
    document["spec"]["release"]["agentConfigDigest"] = "b" * 64
    join.site.source.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    before = join.site.source.read_bytes()
    with pytest.raises(BootstrapError, match="non-membership fields changed"):
        _activate(join, monkeypatch, _state())
    assert join.site.source.read_bytes() == before
    assert not (join.state_dir / "site.before-001.yaml").exists(), (
        "a refused commit must not leave a backup suggesting it rewrote the site"
    )


def test_a_join_without_a_data_plane_writer_role_registers_no_adot_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.prerequisites.pop("adot_writer_role")
    join.registry(join.unrelated())
    commit.rollback_membership(join.request(), execution=join.execution(), joined=False)
    assert join.synced == [], "without joined rows there is no registry delta"
    assert [item.resource_key for item in join.written[0].resources] == [
        "cluster/gpu-a/eks"
    ]


def test_a_data_plane_writer_role_is_retired_with_the_executor_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    join = Join(tmp_path, monkeypatch)
    join.registry(
        join.resource(
            f"aws/iam/adot-writer/{CLUSTER}/role",
            resource_type="iam_role",
            resource_id="gpu-fault-gpu-b-adot-writer",
            arn=ADOT_ARN,
        )
    )
    commit.rollback_membership(join.request(), execution=join.execution(), joined=False)
    (delta,) = join.synced
    assert [(item.resource_key, item.status) for item in delta.resources] == [
        (f"aws/iam/adot-writer/{CLUSTER}/role", InstallationResourceStatus.DELETED)
    ]
