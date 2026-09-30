from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin import uninstall as lifecycle
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.resource_registry import (
    build_installation_snapshot,
    load_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.admin.uninstall import UninstallRequest, uninstall
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from tests._script_loader import lazy_script_module
from tests.admin._aws_cleanup_support import Aws, absent
from tests.admin.test_admin_aws_cleanup_clusters import CpuAws
from tests.admin.test_admin_site import site_file

ROOT = Path(__file__).resolve().parents[2]
STATE = lazy_script_module(ROOT / "deploy/control-plane/tools/cleanup_state.py")
LBC_ROLE = "aws/iam/lbc/role"
LBC_POLICY = "aws/iam/lbc/policy"


def registry(site: RenderedSite) -> InstallationResourceSnapshot:
    entries = [
        ("cluster/cpu-eks", "cpu_eks", "control", Policy.PRESERVE),
        ("cluster/cpu-hyperpod", "cpu_hyperpod", "control", Policy.PRESERVE),
        ("cluster/gpu-a/eks", "gpu_eks", "gpu-a", Policy.PRESERVE),
        ("cluster/gpu-a/hyperpod", "gpu_hyperpod", "hp-gpu-a", Policy.PRESERVE),
        ("aws/nlb", "nlb", "nlb-test", Policy.DELETE),
        ("aws/helm/lbc", "helm_release", "lbc-test", Policy.DELETE),
        ("aws/eks/pod-identity-agent", "eks_addon", "pod-identity", Policy.PRESERVE),
        ("aws/aurora/cluster", "aurora_cluster", "gpu-fault-aurora", Policy.DELETE),
        ("aws/aurora/writer", "aurora_instance", "writer", Policy.DELETE),
        ("aws/aurora/security-group", "security_group", "sg-db", Policy.DELETE),
    ]
    resources = [
        InstallationResource(
            site_id=site.registry_site_id,
            resource_key=key,
            resource_type=kind,
            resource_id=name,
            resource_arn=(
                f"arn:aws:eks:us-east-1:123456789012:cluster/{name}"
                if kind.endswith("_eks")
                else None
            ),
            region="us-east-1",
            account_id="123456789012",
            ownership=Ownership.EXTERNAL
            if policy is Policy.PRESERVE
            else Ownership.CREATED,
            delete_policy=policy,
        )
        for key, kind, name, policy in entries
    ]
    value = InstallationResourceSnapshot(
        site_id=site.registry_site_id, resources=resources
    )
    return value.model_copy(update={"source_sha256": value.digest()})


class Harness(CommandRunner):
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        self.site = load_site(site_file(tmp_path))
        self.snapshot = registry(self.site)
        self.existing = {item.resource_key for item in self.snapshot.resources}
        self.events: list[str] = []
        self.exports = 0
        self.syncs = 0
        self.fail_sync = False
        self.fail_cpu = False
        self.fail_aurora = False
        self.fail_cleanup: BaseException | None = None
        self.no_mutations = False
        self.database_resource_id = "cluster-resource"
        self.namespace_uid = "namespace-old"
        monkeypatch.setattr(lifecycle, "ResourceCleaner", lambda _site: self)
        monkeypatch.setattr(
            lifecycle, "fetch_installation_resource_registry", self.export
        )
        monkeypatch.setattr(lifecycle, "sync_installation_resource_snapshot", self.sync)
        monkeypatch.setattr(lifecycle, "bounded_command", self.kubectl)

    def request(self, *, delete: bool = False, reset: bool = False) -> UninstallRequest:
        return UninstallRequest(
            self.site,
            "delete" if delete else "keep",
            "DELETE_CPU_CONTROL_PLANE" if delete else "UNINSTALL_GPU_FAULT",
            reset_database=reset,
        )

    def state(self) -> dict[str, Any]:
        state = json.loads(
            (self.site.source.parent / "uninstall/state.json").read_text()
        )
        assert isinstance(state, dict), "uninstall state must be a JSON object"
        return state

    def export(
        self, site: RenderedSite, *, output: Path
    ) -> InstallationResourceSnapshot:
        self.exports += 1
        write_installation_resource_snapshot(site, self.snapshot, path=output)
        return self.snapshot

    def sync(self, _site: RenderedSite, snapshot: InstallationResourceSnapshot) -> None:
        self.syncs += 1
        assert [item.immutable_identity() for item in snapshot.resources] == [
            item.immutable_identity() for item in self.snapshot.resources
        ]
        assert all(
            item.status.value == "ACTIVE"
            for item in snapshot.resources
            if item.resource_type in {"gpu_eks", "gpu_hyperpod"}
        ), "uninstall registry sync must preserve active GPU clusters"
        if self.fail_sync:
            self.fail_sync = False
            raise BootstrapError("injected registry sync failure")

    def validate_supported(self, _resources: list[InstallationResource]) -> None:
        pass

    def exists(self, resource: InstallationResource) -> bool:
        if resource.resource_type == "helm_release":
            assert "cluster/cpu-eks" in self.existing, (
                "queried Helm after deleting CPU EKS"
            )
        return resource.resource_key in self.existing

    def delete(self, resource: InstallationResource) -> None:
        assert not self.no_mutations, (
            "completed uninstall replay must not delete resources"
        )
        assert not resource.resource_type.startswith(("cpu_", "gpu_")), (
            "generic resource deletion must not target CPU or GPU clusters"
        )
        self.events.append("delete:" + resource.resource_key)
        self.existing.discard(resource.resource_key)

    def delete_cpu_cluster(
        self, hyperpod: InstallationResource, eks: InstallationResource
    ) -> None:
        assert not self.no_mutations, (
            "completed uninstall replay must not delete CPU clusters"
        )
        self.events.append("delete:cpu")
        self.existing.difference_update(
            {hyperpod.resource_key, eks.resource_key, "aws/eks/pod-identity-agent"}
        )
        if self.fail_cpu:
            self.fail_cpu = False
            raise BootstrapError("injected failure after CPU deletion")

    def delete_aurora(
        self,
        cluster: InstallationResource,
        *,
        final_snapshot_policy: str,
        final_snapshot_identifier: str,
    ) -> str | None:
        assert not self.no_mutations, (
            "completed uninstall replay must not delete Aurora"
        )
        assert self.state()["phase"] == "AURORA_DELETE_IN_PROGRESS"
        assert "aws/nlb" not in self.existing
        assert "aws/helm/lbc" not in self.existing
        self.events.append("delete:aurora")
        self.existing.difference_update({cluster.resource_key, "aws/aurora/writer"})
        if final_snapshot_policy == "retain":
            self.existing.add("aws/aurora/final-snapshot")
        if self.fail_aurora:
            self.fail_aurora = False
            raise BootstrapError("injected failure after Aurora deletion")
        return final_snapshot_identifier if final_snapshot_policy == "retain" else None

    def prepare_aurora_delete(
        self, _resource: InstallationResource, **_kwargs: Any
    ) -> dict[str, str]:
        return {"db_cluster_resource_id": "fixture-incarnation"}

    def prepare_cpu_delete(
        self, _hyperpod: InstallationResource, _eks: InstallationResource
    ) -> dict[str, Any]:
        return {
            "cpu_eks_created_at": "2026-01-01T00:00:00Z",
            "cpu_hyperpod_arn": (
                "arn:aws:sagemaker:us-east-1:123456789012:cluster/cpu-id"
            ),
        }

    def prepare_dns_delete(self, resource: InstallationResource) -> dict[str, Any]:
        return ResourceCleaner(self.site).prepare_dns_delete(resource)

    def aws_json(self, region: str, *arguments: str, **_kwargs: Any) -> dict[str, Any]:
        assert region == "us-east-1"
        assert arguments[:2] == ("rds", "describe-db-clusters")
        cluster_id = str(self.site.release_config["health"]["aurora_cluster_id"])
        return {
            "DBClusters": [
                {
                    "DBClusterIdentifier": cluster_id,
                    "DBClusterArn": f"arn:aws:rds:us-east-1:123456789012:cluster:{cluster_id}",
                    "DbClusterResourceId": self.database_resource_id,
                    "Engine": "aurora-postgresql",
                    "Status": "available",
                    "Endpoint": "aurora.example.rds.amazonaws.com",
                    "Port": 5432,
                    "DatabaseName": "gpu_fault",
                    "MasterUsername": "administrator",
                    "MasterUserSecret": {
                        "SecretArn": "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!test"
                    },
                }
            ]
        }

    def wait_absent(self, resource: InstallationResource, **_kwargs: Any) -> None:
        assert resource.resource_key not in self.existing

    def kubectl(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout_seconds"] == 45
        if "--context" not in arguments:
            assert "cluster/cpu-eks" in self.existing, (
                "queried Kubernetes after CPU deletion"
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    def run(self, arguments: Sequence[str], **_kwargs: Any) -> str:
        if arguments[-1] == "verify-targets":
            return ""
        if arguments[-1] == "cleanup-owned":
            self.events.append("cleanup-owned")
            return ""
        config_path = Path(arguments[arguments.index("--config") + 1])
        path_flag = "--path" if "verify" in arguments else "--state-file"
        path = Path(arguments[arguments.index(path_flag) + 1])
        if "verify" not in arguments:
            assert not self.no_mutations, (
                "completed uninstall replay must not rerun cleanup"
            )
            self.events.append("cleanup")
            if not path.exists():
                inventory = path.parent / "fixture-inventory.json"
                inventory.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "cpu": {"resources": []},
                            "gpu": {"resources": []},
                        }
                    )
                )
                document = STATE.initialize(
                    path,
                    config_path=config_path,
                    inventory_path=inventory,
                    scope="all",
                    mode="reset",
                    node_mode="uninstall",
                )
                document["namespace_snapshots"] = {
                    "cpu": {"uid": self.namespace_uid, "objects": []},
                    "gpu:gpu-a": {"uid": "gpu-" + self.namespace_uid, "objects": []},
                }
            else:
                document = STATE.read_state(path)
            if self.fail_cleanup is not None:
                failure, self.fail_cleanup = self.fail_cleanup, None
                raise failure
            for phase in STATE.required_phases(document):
                STATE.transition(
                    document, phase=phase, status="COMPLETED", message="fixture"
                )
            STATE.transition(
                document,
                phase="CLEANUP_COMPLETED",
                status="COMPLETED",
                message="fixture",
            )
            STATE.atomic_write(path, document)
        document = STATE.read_state(path)
        STATE.validate_request(
            document,
            config_path=config_path,
            scope="all",
            mode="reset",
            node_mode="uninstall",
            cluster_ids=[],
        )
        return ""


def test_non_aurora_failure_never_starts_database_deletion_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    original_delete = harness.delete

    def fail_nlb(resource: InstallationResource) -> None:
        if resource.resource_key == "aws/nlb":
            raise BootstrapError("modeled NLB failure")
        original_delete(resource)

    monkeypatch.setattr(harness, "delete", fail_nlb)
    with pytest.raises(BootstrapError, match="modeled NLB failure"):
        lifecycle.uninstall(harness.request(reset=True), runner=harness)
    assert "delete:aurora" not in harness.events
    assert harness.state()["phase"] == "KUBERNETES_VERIFIED"
    assert harness.events.count("cleanup") == 1

    monkeypatch.setattr(harness, "delete", original_delete)
    lifecycle.uninstall(harness.request(reset=True), runner=harness)
    assert harness.state()["phase"] == "COMPLETED"
    assert harness.events.count("cleanup") == 1
    assert harness.events.count("delete:aurora") == 1


def test_aurora_failure_preserves_its_dependencies_and_the_pre_aurora_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_aurora = True
    with pytest.raises(BootstrapError, match="failure after Aurora"):
        lifecycle.uninstall(harness.request(reset=True), runner=harness)
    assert harness.state()["phase"] == "AURORA_DELETE_IN_PROGRESS"
    assert harness.state()["pre_aurora_sha256"]
    assert "aws/nlb" not in harness.existing
    for resource in harness.snapshot.resources:
        if lifecycle.is_aurora_resource(resource) and resource.resource_type not in {
            "aurora_cluster",
            "aurora_instance",
            "rds_managed_secret",
        }:
            assert resource.resource_key in harness.existing
    assert (
        harness.site.source.parent
        / "uninstall/installation-resources-pre-aurora-delete.json"
    ).is_file(), "an interrupted Aurora deletion must retain its pre-delete evidence"


def test_uninstall_supervision_loss_blocks_a_fresh_process_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_cleanup = ProcessSupervisionLost("completion proof unavailable")
    with pytest.raises(ProcessSupervisionLost):
        uninstall(harness.request(), runner=harness)
    assert harness.state()["supervision_lost"] is True
    before = list(harness.events), harness.exports, harness.syncs
    harness.no_mutations = True
    with pytest.raises(BootstrapError, match="supervision was lost"):
        uninstall(harness.request(), runner=harness)
    assert (harness.events, harness.exports, harness.syncs) == before, (
        "durable supervision loss must stop before any resumed external operation"
    )


def add_lbc_registry(
    harness: Harness, *, keep_role: bool = False, keep_policy: bool = False
) -> None:
    produced = build_installation_snapshot(
        harness.site,
        {
            "resources": {
                "load_balancer_controller": {
                    "reused": False,
                    "role_arn": "arn:aws:iam::123456789012:role/gpu-fault-test-site-lbc",
                    "role_ownership": "EXTERNAL" if keep_role else "CREATED",
                    "policy_arn": (
                        "arn:aws:iam::123456789012:policy/gpu-fault-test-site-lbc-policy"
                    ),
                    "policy_ownership": "EXTERNAL" if keep_policy else "CREATED",
                    "association_id": "assoc-lbc",
                    "association_ownership": "CREATED",
                    "cluster_name": "control",
                    "namespace": "kube-system",
                    "service_account": "aws-load-balancer-controller",
                }
            }
        },
    )
    resources = [
        item
        for item in harness.snapshot.resources
        if item.resource_key != "aws/helm/lbc"
    ] + [
        item
        for item in produced.resources
        if item.resource_key.startswith("aws/iam/lbc/")
        or item.resource_key == "kubernetes/lbc/helm-release"
    ]
    snapshot = InstallationResourceSnapshot(
        site_id=harness.snapshot.site_id, resources=resources
    )
    harness.snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
    harness.existing = {item.resource_key for item in resources}


@pytest.mark.parametrize(
    ("delete", "reset"), [(False, False), (False, True), (True, False)]
)
def test_uninstall_modes_and_completed_replay_are_nonmutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delete: bool, reset: bool
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    request = harness.request(delete=delete, reset=reset)
    result = uninstall(request, runner=harness)
    assert result["aurora_cluster"] == ("deleted" if delete or reset else "preserved")
    assert result["gpu_clusters"] == "preserved"
    assert harness.state()["phase"] == "COMPLETED"
    before = harness.state()
    events = list(harness.events)
    harness.no_mutations = True
    replay = uninstall(request, runner=harness)
    assert replay == result
    assert harness.events == events
    assert harness.state() == before
    assert harness.exports == harness.syncs == 1


@pytest.mark.parametrize("stage", ["cpu", "aurora"])
def test_late_retry_does_not_reenter_cpu_kubernetes_or_non_aurora_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_cpu = stage == "cpu"
    harness.fail_aurora = stage == "aurora"
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError, match="injected failure"):
        uninstall(request, runner=harness)
    assert harness.state()["phase"] == (
        "CPU_DELETE_IN_PROGRESS" if stage == "cpu" else "AURORA_DELETE_IN_PROGRESS"
    )
    result = uninstall(request, runner=harness)
    assert result["aurora_deleted_last"]
    assert harness.events.count("cleanup") == 1
    assert harness.events.count("delete:aws/nlb") == 1
    assert harness.events.count("delete:aws/helm/lbc") == 1


def test_export_ack_does_not_skip_pending_registry_sync_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_sync = True
    with pytest.raises(BootstrapError, match="registry sync failure"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []
    uninstall(harness.request(), runner=harness)
    assert harness.exports == 1
    assert harness.syncs == 2


@pytest.mark.parametrize(
    "interruption", [BootstrapError("injected failure"), KeyboardInterrupt()]
)
def test_interrupted_shell_cleans_only_owned_temporary_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: BaseException
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_cleanup = interruption
    with pytest.raises(type(interruption)):
        uninstall(harness.request(), runner=harness)
    assert harness.events == ["cleanup", "cleanup-owned"]
    assert harness.state()["phase"] == "REGISTRY_EXPORTED"
    assert not any(event.startswith("delete:") for event in harness.events), (
        "interrupted cleanup must not proceed to resource deletion"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"region": "us-west-2"},
        {"account_id": "111122223333"},
        {"resource_id": "other-control"},
        {"resource_arn": "arn:aws:eks:us-east-1:123456789012:cluster/other"},
    ],
)
def test_self_consistent_snapshot_hash_does_not_authorize_foreign_cpu_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, str]
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    resources = list(harness.snapshot.resources)
    resources[0] = resources[0].model_copy(update=change)
    value = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    harness.snapshot = value.model_copy(update={"source_sha256": value.digest()})
    with pytest.raises(BootstrapError, match="differs from site"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []
    assert harness.syncs == 0


def test_resume_cannot_rebind_the_same_state_to_changed_site_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    uninstall(harness.request(), runner=harness)
    changed = replace(harness.site, source_sha256="f" * 64)
    monkeypatch.setattr(lifecycle, "reload_site_for_mutation", lambda _site: changed)
    before = list(harness.events)
    with pytest.raises(BootstrapError, match="conflicts on site_sha256"):
        uninstall(replace(harness.request(), site=changed), runner=harness)
    assert harness.events == before


def test_retained_aurora_dependency_cannot_be_deleted_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    resources = [
        item.model_copy(update={"dependencies": ["aws/nlb"]})
        if item.resource_type == "aurora_cluster"
        else item
        for item in harness.snapshot.resources
    ]
    value = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    harness.snapshot = value.model_copy(update={"source_sha256": value.digest()})
    with pytest.raises(BootstrapError, match="retained resource depends"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []


def test_effective_legacy_policy_is_separate_from_all_registry_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    resources = [
        item.model_copy(
            update={"ownership": Ownership.REUSED, "delete_policy": Policy.PRESERVE}
        )
        if item.resource_key == "aws/nlb"
        else item
        for item in harness.snapshot.resources
    ]
    value = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    harness.snapshot = value.model_copy(update={"source_sha256": value.digest()})
    result = uninstall(harness.request(), runner=harness)
    assert harness.state()["effective_policies"]["aws/nlb"] == "DELETE"
    for name in (
        "installation-resources-delete-plan.json",
        "installation-resources-final.json",
    ):
        persisted = load_installation_resource_snapshot(
            harness.site.source.parent / "uninstall" / name
        )
        entry = next(
            item for item in persisted.resources if item.resource_key == "aws/nlb"
        )
        original = next(item for item in resources if item.resource_key == "aws/nlb")
        assert entry.immutable_identity() == original.immutable_identity()
    assert result["registry_entries_deleted"] == 2


def test_a_retained_resource_alias_cannot_be_deleted_through_another_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    aurora_group = next(
        item
        for item in harness.snapshot.resources
        if item.resource_key == "aws/aurora/security-group"
    )
    value = InstallationResourceSnapshot(
        site_id="test-site",
        resources=[
            *harness.snapshot.resources,
            aurora_group.model_copy(
                update={"resource_key": "aws/network/shared-group"}
            ),
        ],
    )
    harness.snapshot = value.model_copy(update={"source_sha256": value.digest()})
    with pytest.raises(BootstrapError, match="aliases a physical resource"):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []


@pytest.mark.parametrize(
    ("keep_role", "keep_policy"), [(False, False), (False, True), (True, True)]
)
def test_legacy_lbc_execution_order_preserves_all_stored_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_role: bool, keep_policy: bool
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    add_lbc_registry(harness, keep_role=keep_role, keep_policy=keep_policy)
    original = {
        item.resource_key: item.immutable_identity()
        for item in harness.snapshot.resources
    }
    by_key = {item.resource_key: item for item in harness.snapshot.resources}
    assert by_key[LBC_ROLE].dependencies == []
    assert by_key[LBC_POLICY].dependencies == [LBC_ROLE]
    uninstall(harness.request(), runner=harness)
    assert ("delete:" + LBC_ROLE in harness.events) is not keep_role
    assert ("delete:" + LBC_POLICY in harness.events) is not keep_policy
    if not keep_role and not keep_policy:
        assert harness.events.index("delete:" + LBC_ROLE) < harness.events.index(
            "delete:" + LBC_POLICY
        )
    for name in (
        "installation-resources-before.json",
        "installation-resources-delete-plan.json",
        "installation-resources-pre-aurora-delete.json",
        "installation-resources-final.json",
    ):
        snapshot = load_installation_resource_snapshot(
            harness.site.source.parent / "uninstall" / name
        )
        assert {
            item.resource_key: item.immutable_identity() for item in snapshot.resources
        } == original
    assert {
        item.resource_key: item.immutable_identity()
        for item in harness.snapshot.resources
    } == original


def test_retained_lbc_role_blocks_policy_deletion_before_any_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    add_lbc_registry(harness, keep_role=True)
    with pytest.raises(
        BootstrapError,
        match="retained resource depends on a deletion target: " + LBC_POLICY,
    ):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []
    assert harness.syncs == 0


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        ("missing-dependency", "missing installation dependency"),
        ("stored-cycle", "dependencies contain a cycle"),
        ("execution-cycle", "dependencies contain a cycle"),
        ("retained-consumer", "retained resource depends"),
    ],
)
def test_lbc_compatibility_does_not_drop_other_dependency_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, error: str
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    add_lbc_registry(harness)
    scenarios: dict[str, dict[str, dict[str, Any]]] = {
        "missing-dependency": {LBC_ROLE: {"dependencies": ["aws/missing"]}},
        "stored-cycle": {LBC_ROLE: {"dependencies": [LBC_POLICY]}},
        "execution-cycle": {
            LBC_POLICY: {"dependencies": [LBC_ROLE, "aws/nlb"]},
            "aws/nlb": {"dependencies": [LBC_ROLE]},
        },
        "retained-consumer": {
            "aws/nlb": {
                "dependencies": [LBC_POLICY],
                "ownership": Ownership.EXTERNAL,
                "delete_policy": Policy.PRESERVE,
            }
        },
    }
    changes = scenarios[scenario]
    resources = [
        item.model_copy(update=changes.get(item.resource_key, {}))
        for item in harness.snapshot.resources
    ]
    snapshot = InstallationResourceSnapshot(
        site_id=harness.snapshot.site_id, resources=resources
    )
    harness.snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
    with pytest.raises(BootstrapError, match=error):
        uninstall(harness.request(), runner=harness)
    assert harness.events == []
    assert harness.syncs == 0


@pytest.mark.parametrize("replacement", ["eks", "hyperpod"])
def test_cpu_recreation_between_uninstall_attempts_blocks_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    aws = CpuAws()
    aws.responses[("eks", "list-pod-identity-associations")] = absent("AccessDenied")
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    original_exists = harness.exists

    def exists(resource: InstallationResource) -> bool:
        if resource.resource_type in {"cpu_eks", "cpu_hyperpod"}:
            return ResourceCleaner(harness.site).exists(resource)
        return original_exists(resource)

    def delete_cpu(hyperpod: InstallationResource, eks: InstallationResource) -> None:
        ResourceCleaner(harness.site).delete_cpu_cluster(hyperpod, eks)

    monkeypatch.setattr(harness, "exists", exists)
    monkeypatch.setattr(harness, "delete_cpu_cluster", delete_cpu)
    monkeypatch.setattr(
        harness, "prepare_cpu_delete", ResourceCleaner(harness.site).prepare_cpu_delete
    )
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError, match="AccessDenied"):
        uninstall(request, runner=harness)
    assert harness.state()["phase"] == "CPU_DELETE_IN_PROGRESS", (
        "the first attempt must stop after recording CPU deletion intent"
    )
    previous_mutations = list(aws.mutations)
    aws.responses[("eks", "list-pod-identity-associations")] = {"associations": []}
    if replacement == "eks":
        aws.cluster["createdAt"] = "2026-02-01T00:00:00Z"
    else:
        aws.hp_gone = False
        aws.hyperpod["ClusterArn"] = aws.hyperpod["ClusterArn"] + "-replacement"
    with pytest.raises(BootstrapError, match="incarnation|binding"):
        uninstall(request, runner=harness)
    assert aws.mutations == previous_mutations, (
        "a saved uninstall must not delete a replacement CPU cluster or its children"
    )


@pytest.mark.parametrize(
    "drift",
    [
        {"DbClusterResourceId": "replacement-database"},
        {"Status": "creating"},
        {"Status": "failed"},
        {"DBClusterIdentifier": "other-database"},
    ],
)
def test_completed_uninstall_revalidates_the_original_available_final_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: dict[str, str]
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    snapshot = {
        "DBClusterIdentifier": "gpu-fault-aurora",
        "DbClusterResourceId": "fixture-incarnation",
        "SnapshotType": "manual",
        "Status": "available",
    }

    def describe(arguments: list[str]) -> dict[str, Any]:
        return {
            "DBClusterSnapshots": [
                {
                    **snapshot,
                    "DBClusterSnapshotIdentifier": arguments[
                        arguments.index("--db-cluster-snapshot-identifier") + 1
                    ],
                }
            ]
        }

    aws = Aws({("rds", "describe-db-cluster-snapshots"): describe})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    original_exists = harness.exists

    def exists(resource: InstallationResource) -> bool:
        if resource.resource_type == "rds_snapshot":
            return ResourceCleaner(harness.site).exists(resource)
        return original_exists(resource)

    monkeypatch.setattr(harness, "exists", exists)
    request = harness.request(reset=True)
    uninstall(request, runner=harness)
    before = harness.state()
    harness.no_mutations = True
    snapshot.update(drift)
    with pytest.raises(BootstrapError, match="snapshot"):
        uninstall(request, runner=harness)
    assert harness.state() == before, (
        "failed read-only final snapshot verification must not rewrite completion"
    )
    assert aws.mutations == [], "completed replay must remain read-only"


def test_preserved_association_of_a_deleted_zone_is_detached_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-09-24: a resumed cold deploy recorded its GPU VPC association as
    EXTERNAL/PRESERVE (association receipt lost), and the uninstall refused with
    "retained resource depends on a deletion target: aws/route53/zone". The zone
    is always solution owned; an association cannot outlive it, so the row is
    detached before the zone instead."""

    harness = Harness(tmp_path, monkeypatch)
    resources = list(harness.snapshot.resources)
    for key, kind, name, ownership, policy, dependencies, attributes in (
        (
            "aws/route53/zone",
            "route53_zone",
            "Z123",
            Ownership.CREATED,
            Policy.DELETE,
            [],
            {"zone_name": "test-site.gpu-fault.internal"},
        ),
        (
            "aws/route53/vpc-association/us-east-1/vpc-gpu",
            "route53_vpc_association",
            "Z123:us-east-1:vpc-gpu",
            Ownership.EXTERNAL,
            Policy.PRESERVE,
            ["aws/route53/zone"],
            {"hosted_zone_id": "Z123", "vpc_id": "vpc-gpu", "vpc_region": "us-east-1"},
        ),
    ):
        resources.append(
            InstallationResource(
                site_id="test-site",
                resource_key=key,
                resource_type=kind,
                resource_id=name,
                region="us-east-1",
                account_id="123456789012",
                ownership=ownership,
                delete_policy=policy,
                dependencies=dependencies,
                attributes=attributes,
            )
        )
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    harness.snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
    harness.existing = {item.resource_key for item in resources}

    result = uninstall(harness.request(), runner=harness)

    assert result["delete_policy_residuals"] == 0, result
    association = "delete:aws/route53/vpc-association/us-east-1/vpc-gpu"
    assert association in harness.events and "delete:aws/route53/zone" in harness.events
    assert harness.events.index(association) < harness.events.index(
        "delete:aws/route53/zone"
    ), "the association is detached before its zone is deleted"
    final = load_installation_resource_snapshot(
        harness.site.source.parent / "uninstall" / "installation-resources-final.json"
    )
    statuses = {item.resource_key: item.status.value for item in final.resources}
    assert statuses["aws/route53/vpc-association/us-east-1/vpc-gpu"] == "DETACHED"
    assert statuses["aws/route53/zone"] == "DELETED"


@pytest.mark.parametrize("retry", [False, True])
def test_dns_cleanup_survives_controller_owned_nlb_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: bool
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.site.release_config["dns"] = {
        "hosted_zone_id": "Z123",
        "hostname": "control.example",
    }
    monkeypatch.setattr(lifecycle, "reload_site_for_mutation", lambda site: site)
    nlb_name = harness.site.release_config["nlb"]["name"]
    nlb_arn = (
        "arn:aws:elasticloadbalancing:us-east-1:123456789012:"
        f"loadbalancer/net/{nlb_name}/original"
    )
    resources = [
        item.model_copy(
            update={
                "resource_id": nlb_name,
                "resource_arn": nlb_arn,
                "attributes": {"dns_name": "original.elb.example"},
            }
        )
        if item.resource_key == "aws/nlb"
        else item
        for item in harness.snapshot.resources
    ]
    for key, kind, name, policy, dependencies, attributes in (
        ("aws/dns/zone", "route53_zone", "Z123", Policy.PRESERVE, [], {}),
        (
            "aws/dns/record",
            "route53_record",
            "control.example",
            Policy.DELETE,
            ["aws/dns/zone", "aws/nlb"],
            {"hosted_zone_id": "Z123", "record_type": "CNAME"},
        ),
    ):
        resources.append(
            InstallationResource(
                site_id="test-site",
                resource_key=key,
                resource_type=kind,
                resource_id=name,
                region="us-east-1",
                account_id="123456789012",
                ownership=(
                    Ownership.EXTERNAL
                    if policy is Policy.PRESERVE
                    else Ownership.CREATED
                ),
                delete_policy=policy,
                dependencies=dependencies,
                attributes=attributes,
            )
        )
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    harness.snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
    harness.existing = {item.resource_key for item in resources}
    records = [
        {
            "Name": "control.example.",
            "Type": "CNAME",
            "TTL": 60,
            "ResourceRecords": [{"Value": "original.elb.example"}],
        }
    ]
    nlb_exists = True

    def nlb(_arguments: list[str]) -> Any:
        if not nlb_exists:
            return absent("LoadBalancerNotFound")
        return {
            "LoadBalancers": [
                {
                    "LoadBalancerName": nlb_name,
                    "LoadBalancerArn": nlb_arn,
                    "DNSName": "original.elb.example",
                }
            ]
        }

    def delete_record(_arguments: list[str]) -> dict[str, Any]:
        records.clear()
        return {}

    aws = Aws(
        {
            ("elbv2", "describe-load-balancers"): nlb,
            ("elbv2", "describe-tags"): {
                "TagDescriptions": [
                    {
                        "ResourceArn": nlb_arn,
                        "Tags": [{"Key": "gpu-fault:site-id", "Value": "test-site"}],
                    }
                ]
            },
            ("route53", "list-resource-record-sets"): lambda _args: {
                "ResourceRecordSets": records
            },
            ("route53", "change-resource-record-sets"): delete_record,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)
    original_run, original_delete = harness.run, harness.delete
    fail_dns_once = retry

    def run(arguments: Sequence[str], **kwargs: Any) -> str:
        nonlocal nlb_exists
        result = original_run(arguments, **kwargs)
        if "--execute" in arguments:
            nlb_exists = False
        return result

    def delete(resource: InstallationResource) -> None:
        nonlocal fail_dns_once
        if resource.resource_type == "route53_record":
            if fail_dns_once:
                fail_dns_once = False
                raise BootstrapError("injected interruption before DNS deletion")
            ResourceCleaner(harness.site).delete(resource)
        original_delete(resource)

    monkeypatch.setattr(harness, "run", run)
    monkeypatch.setattr(harness, "delete", delete)
    if retry:
        with pytest.raises(BootstrapError, match="interruption before DNS"):
            uninstall(harness.request(), runner=harness)
        assert (
            harness.state()["dns_bindings"]["aws/dns/record"]["nlb_arn"] == nlb_arn
        ), "DNS cleanup must persist the proven target before Service removal"
    result = uninstall(harness.request(), runner=harness)
    assert result["delete_policy_residuals"] == 0, (
        "controller deletion of the registered NLB must not strand its DNS record"
    )
    assert records == [], "the bound stale CNAME must be deleted after NLB removal"
    for name in (
        "installation-resources-before.json",
        "installation-resources-delete-plan.json",
        "installation-resources-pre-aurora-delete.json",
        "installation-resources-final.json",
    ):
        saved = load_installation_resource_snapshot(
            harness.site.source.parent / "uninstall" / name
        )
        assert {item.resource_key: item.attributes for item in saved.resources} == {
            item.resource_key: item.attributes for item in resources
        }, "ephemeral DNS evidence must not rewrite the registry attributes"


@pytest.mark.parametrize(
    "binding", [None, {}, {"cpu_eks_created_at": True, "cpu_hyperpod_arn": "invalid"}]
)
def test_late_cpu_retry_requires_the_original_well_formed_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: Any
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_cpu = True
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError, match="injected failure"):
        uninstall(request, runner=harness)
    state = harness.state()
    if binding is None:
        state.pop("cpu_binding")
    else:
        state["cpu_binding"] = binding
    (harness.site.source.parent / "uninstall/state.json").write_text(json.dumps(state))
    events = list(harness.events)
    with pytest.raises(BootstrapError, match="binding"):
        uninstall(request, runner=harness)
    assert harness.events == events, (
        "a missing or malformed late CPU binding must fail before cleanup resumes"
    )


def test_late_aurora_retry_cannot_recapture_an_empty_incarnation_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.fail_aurora = True
    request = harness.request(reset=True)
    with pytest.raises(BootstrapError, match="injected failure"):
        uninstall(request, runner=harness)
    state = harness.state()
    state["aurora_binding"] = {}
    (harness.site.source.parent / "uninstall/state.json").write_text(json.dumps(state))
    events = list(harness.events)
    with pytest.raises(BootstrapError, match="Aurora incarnation binding"):
        uninstall(request, runner=harness)
    assert harness.events == events, (
        "an empty Aurora binding must not authorize capturing a replacement database"
    )
