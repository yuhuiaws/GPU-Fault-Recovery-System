from __future__ import annotations

import copy

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.installation_resources import InstallationResourceSnapshot
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_journal as journal
from scripts.e2e.regional import boot032_native as adapter
from scripts.e2e.regional import boot032_observe as observe
from tests.regional._cov95_boot032_approval import approve, execute
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World


def replace_registry(world, resources):
    snapshot = InstallationResourceSnapshot(
        site_id=world.target.metadata_name, resources=resources
    )
    world.snapshots[world.target.metadata_name] = snapshot.model_copy(
        update={"source_sha256": snapshot.digest()}
    )


@pytest.mark.parametrize(
    "omitted", ["cpu_eks", "cpu_hyperpod", "gpu_eks", "gpu_hyperpod", "aurora_cluster"]
)
def test_plan_requires_the_complete_native_physical_inventory(
    tmp_path, monkeypatch, omitted
):
    world = World(tmp_path, monkeypatch)
    replace_registry(
        world,
        [
            row
            for row in world.snapshots[world.target.metadata_name].resources
            if row.resource_type != omitted
        ],
    )
    with pytest.raises(contract.UninstallCaseError, match="physical inventory"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "missing roots must refuse before teardown"
    )


@pytest.mark.parametrize(
    "change",
    [
        "not-an-arn",
        "arn:aws:sagemaker:us-west-2:123456789012:cluster/foreign",
        "arn:aws:sagemaker:us-east-1:111122223333:cluster/foreign",
        "arn:aws:eks:us-east-1:123456789012:cluster/foreign",
    ],
)
def test_hyperpod_incarnation_must_belong_to_the_explicit_eks_scope(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)
    cpu = contract.cluster_specs(world.target)[0]
    world.cloud[cpu["eks_name"]]["hp"]["ClusterArn"] = change
    with pytest.raises(Exception, match="ARN|scope|HyperPod"):
        observe.cluster_observation(
            world.target, cpu, fixture_id=world.settings.fixture_id
        )
    assert not world.settings.native_dir.exists(), (
        "foreign HyperPod binding must not start teardown"
    )


def test_supervision_loss_before_native_entry_is_durably_blocked(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    _settings, _plan, deadline = approve(world)

    def lost(_site):
        raise ProcessSupervisionLost("fake read process cannot prove termination")

    monkeypatch.setattr(adapter, "observe_site", lost)
    with pytest.raises(ProcessSupervisionLost):
        execute(world, deadline)
    assert (world.root / "command-supervision-lost.json").is_file(), (
        "an early failed read must retain the shared durable refusal marker"
    )
    assert not harness.events, (
        "lost supervision must prevent native entry and all mutations"
    )


def test_native_namespace_object_uid_must_match_the_approved_inventory(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    NativeHarness(world, monkeypatch)
    settings, plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "prepare real native pause evidence"
    )
    path = settings.native_dir / "kubernetes-cleanup.json"
    receipt = journal.CLEANUP.read_state(path)
    receipt["namespace_snapshots"]["cpu"]["objects"][0][3] = "replacement-object-uid"
    journal.CLEANUP.atomic_write(path, receipt)
    with pytest.raises(contract.UninstallCaseError, match="object identity"):
        journal.cleanup_receipt(settings, plan["details"]["binding"])
    assert journal.native_state(settings)["phase"] == "REGISTRY_EXPORTED", (
        "replaced cleanup objects cannot authorize entering AWS phases"
    )


@pytest.mark.parametrize("side", ["target", "protected"])
@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_plan_rejects_kubeconfig_context_server_and_ca_drift(
    tmp_path, monkeypatch, side, plane
):
    world = World(tmp_path, monkeypatch)
    site = getattr(world, side)
    spec = next(item for item in contract.cluster_specs(site) if item["plane"] == plane)
    world.cloud[spec["eks_name"]]["eks"]["cluster"]["endpoint"] = (
        "https://other.example.invalid"
    )
    with pytest.raises(contract.UninstallCaseError, match="server or TLS"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "context mismatch must block native admission"
    )


@pytest.mark.parametrize("side", ["target", "protected"])
def test_any_registered_read_failure_is_unknown_not_absence(
    tmp_path, monkeypatch, side
):
    world = World(tmp_path, monkeypatch)
    original = world.fetch

    def failing(site):
        if site == getattr(world, side):
            raise PermissionError("fake denied resource read")
        return original(site)

    monkeypatch.setattr(observe, "fetch_installation_resource_registry", failing)
    with pytest.raises(PermissionError):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "read denial must never authorize deletion"
    )


def test_shared_resource_or_cluster_uid_cannot_alias_the_accepted_site(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    resources = copy.deepcopy(world.snapshots[world.target.metadata_name].resources)
    protected_nlb = next(
        row
        for row in world.snapshots[world.protected.metadata_name].resources
        if row.resource_type == "nlb"
    )
    resources = [
        row.model_copy(update={"resource_id": protected_nlb.resource_id})
        if row.resource_type == "nlb"
        else row
        for row in resources
    ]
    replace_registry(world, resources)
    with pytest.raises(contract.UninstallCaseError, match="overlaps"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "protected shared resource must not be consumed"
    )


@pytest.mark.parametrize("changed", ["runtime", "registry", "object"])
def test_target_live_drift_after_approval_stops_before_any_cleanup(
    tmp_path, monkeypatch, changed
):
    world = World(tmp_path, monkeypatch)
    harness = NativeHarness(world, monkeypatch)
    _settings, _plan, deadline = approve(world)
    if changed == "runtime":
        world.runtime[world.target.metadata_name]["release_state"]["release_id"] = (
            "different-live-release"
        )
    elif changed == "registry":
        resources = [
            row.model_copy(update={"resource_id": "replaced-nlb"})
            if row.resource_type == "nlb"
            else row
            for row in world.snapshots[world.target.metadata_name].resources
        ]
        replace_registry(world, resources)
        harness.snapshot = world.snapshots[world.target.metadata_name]
    else:
        world.uids[
            (
                world.target.metadata_name,
                "cpu",
                "deployment",
                "gpu-fault-api",
                "gpu-fault-system",
            )
        ] = "replaced-object-uid"
    with pytest.raises(contract.UninstallCaseError, match="changed|drift"):
        execute(world, deadline)
    assert not harness.events, (
        "unapproved target identity changes must stop before native destructive cleanup"
    )
