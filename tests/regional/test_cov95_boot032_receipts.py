from __future__ import annotations

import copy
import json

import pytest

from gpu_fault.admin.resource_registry import write_installation_resource_snapshot
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_journal as journal
from scripts.e2e.regional import boot032_native as adapter
from tests.regional._cov95_boot032_approval import approve, execute, restart
from tests.regional._cov95_boot032_native import NativeHarness
from tests.regional._cov95_boot032_world import World


@pytest.mark.parametrize(
    "change",
    [
        "scope",
        "seal",
        "missing-native",
        "namespace-objects",
        "fleet-seal",
        "node-targets",
    ],
)
def test_cleanup_receipt_cannot_be_rebound_or_partially_removed(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)
    NativeHarness(world, monkeypatch)
    settings, plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "create native proof before corrupting it"
    )
    path = settings.native_dir / "kubernetes-cleanup.json"
    value = journal.CLEANUP.read_state(path)
    if change == "missing-native":
        (settings.native_dir / "state.json").unlink()
        with pytest.raises(contract.UninstallCaseError, match="missing"):
            journal.pause_proof(settings, plan["details"]["binding"])
        return
    if change == "scope":
        value["scope"] = "gpu"
    elif change == "seal":
        value["content_sha256"] = "untrusted"
    elif change == "namespace-objects":
        value["namespace_snapshots"]["cpu"]["objects"] = None
    elif change == "fleet-seal":
        value["fleet_snapshot_sha256"] = "untrusted"
    else:
        value["node_targets"] = {}
    if change == "seal":
        path.write_text(json.dumps(value))
    else:
        journal.CLEANUP.atomic_write(path, value)
    with pytest.raises(
        (contract.UninstallCaseError, journal.CLEANUP.CleanupStateError)
    ):
        journal.cleanup_complete(settings, plan["details"]["binding"])
    assert journal.native_state(settings)["phase"] == "REGISTRY_EXPORTED", (
        "a modified proof must not authorize progress into native AWS deletion"
    )


@pytest.mark.parametrize(
    "change",
    [
        "site_id",
        "phase",
        "final_snapshot_policy",
        "cpu_disposition",
        "site_sha256",
        "supervision_lost",
    ],
)
def test_native_journal_rejects_unknown_state_instead_of_claiming_absence(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)
    NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "prepare real native journal"
    )
    path = settings.native_dir / "state.json"
    value = contract.read_document(path)
    value[change] = True if change == "supervision_lost" else "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(contract.UninstallCaseError, match="journal|supervision"):
        journal.native_state(settings)
    before = len(world.calls)
    assert journal.read_case(settings.case_dir / "boot032-state.json")["pause"], (
        "unknown native state must retain the original acceptance restart evidence"
    )
    assert len(world.calls) == before, (
        "journal validation must not use live reads to guess missing state"
    )


def test_completed_retirement_preserves_external_resources_and_records_detachment(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    snapshot = world.snapshots[world.target.metadata_name]
    template = snapshot.resources[-1]
    extra = [
        template.model_copy(
            update={
                "resource_key": "aws/external/shared-security-group",
                "resource_id": "sg-preserved",
                "ownership": InstallationResourceOwnership.EXTERNAL,
                "delete_policy": InstallationResourceDeletePolicy.PRESERVE,
            }
        ),
        template.model_copy(
            update={
                "resource_key": "aws/external/subscription",
                "resource_type": "sns_subscription",
                "resource_id": "arn:aws:sns:us-east-1:123456789012:fixture:subscription",
                "ownership": InstallationResourceOwnership.EXTERNAL,
                "delete_policy": InstallationResourceDeletePolicy.DETACH,
            }
        ),
    ]
    candidate = InstallationResourceSnapshot(
        site_id=snapshot.site_id, resources=[*snapshot.resources, *extra]
    )
    world.snapshots[snapshot.site_id] = candidate.model_copy(
        update={"source_sha256": candidate.digest()}
    )
    harness = NativeHarness(world, monkeypatch)
    settings, _plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "prepare full native policy plan"
    )
    restart(world)
    assert execute(world, deadline) == 0, (
        "native full retirement must honor every registered policy"
    )
    final = journal.native_snapshot(settings, "installation-resources-final.json")
    rows = {row.resource_key: row for row in final.resources}
    assert rows["aws/external/shared-security-group"].status.value == "PRESERVED", (
        "full uninstall does not mean deleting approved external resources"
    )
    assert rows["aws/external/subscription"].status.value == "DETACHED", (
        "external subscription must retain immutable ownership and a detachment verdict"
    )
    assert "aws/external/shared-security-group" in harness.existing, (
        "preservation needs an independent existence read"
    )
    assert "aws/external/subscription" not in harness.existing, (
        "detachment needs an independent absence read"
    )


def test_final_snapshot_cannot_omit_a_resource_even_with_new_valid_seals(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    NativeHarness(world, monkeypatch)
    settings, plan, deadline = approve(world)
    assert execute(world, deadline) == contract.RESTART_EXIT, (
        "capture original native registry"
    )
    restart(world)
    assert execute(world, deadline) == 0, "prepare verified completion"
    path = settings.native_dir / "installation-resources-final.json"
    before = journal.native_snapshot(settings, path.name)
    reduced = InstallationResourceSnapshot(
        site_id=before.site_id, resources=before.resources[1:]
    )
    reduced = reduced.model_copy(update={"source_sha256": reduced.digest()})
    write_installation_resource_snapshot(world.target, reduced, path=path)
    state_path = settings.native_dir / "state.json"
    state = contract.read_document(state_path)
    state["final_registry_sha256"] = reduced.digest()
    state_path.write_text(json.dumps(state))
    saved = journal.read_case(settings.case_dir / "boot032-state.json")
    with pytest.raises(contract.UninstallCaseError, match="omitted"):
        journal.final_receipts(
            settings, plan["details"]["binding"], saved["pause"]["proof"]
        )
    assert execute(world, deadline) == 1, (
        "self-consistent but incomplete final state must invalidate PASS"
    )


def test_cluster_scoped_and_absent_namespace_objects_do_not_invent_cleanup_ownership(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    binding = adapter.NativeBackend(world.settings).initial()
    binding = copy.deepcopy(binding)
    binding["target"]["resource_uids"]["cpu"] = [
        {"identity": ["cluster", "clusterrole", "", "role"], "uid": "cluster-role-uid"},
        {
            "identity": ["namespaced", "deployment", "gpu-fault-system", "absent"],
            "uid": None,
        },
    ]
    spec = contract.cluster_specs(world.target)[0]
    journal.namespace_objects(world.settings, binding, spec, {"objects": []})
    with pytest.raises(contract.UninstallCaseError, match="object identity"):
        journal.namespace_objects(
            world.settings,
            binding,
            spec,
            {"objects": [["apps/v1", "deployment", "absent", "new-uid"]]},
        )


def test_case_journal_requires_its_original_seal_and_native_state_read_errors_propagate(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    path = world.settings.case_dir / "boot032-state.json"
    value = {"schema_version": 1, "case_id": contract.CASE_ID, "events": []}
    journal.save_case(path, value, "STARTED")
    assert journal.read_case(path)["phase"] == "STARTED", (
        "atomically written state must be readable"
    )
    value["case_id"] = "another-case"
    path.write_text(json.dumps(value))
    with pytest.raises(contract.UninstallCaseError, match="integrity"):
        journal.read_case(path)
    with pytest.raises(contract.UninstallCaseError, match="phase"):
        journal.save_case(path, value, "UNKNOWN")
    native = world.settings.native_dir / "state.json"
    native.parent.mkdir(mode=0o700)
    native.write_text("{")
    native.chmod(0o600)
    with pytest.raises(json.JSONDecodeError):
        journal.native_state(world.settings)
