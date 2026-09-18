from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from gpu_fault.admin import uninstall as lifecycle
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from gpu_fault.installation_resources import InstallationResourceSnapshot
from tests.admin.test_uninstall_lifecycle import Harness


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


def replace_snapshot(harness, resources):
    value = InstallationResourceSnapshot(
        site_id=harness.snapshot.site_id, resources=resources
    )
    harness.snapshot = value.model_copy(update={"source_sha256": value.digest()})


@pytest.mark.parametrize(
    "case",
    [
        "external-aurora",
        "missing-dependency",
        "dependency-cycle",
        "duplicate-physical",
        "non-aurora-dependency",
    ],
)
def test_uninstall_ownership_or_dependency_error_prevents_cleanup(harness, case):
    resources = list(harness.snapshot.resources)
    if case == "duplicate-physical":
        resources.append(
            resources[-1].model_copy(update={"resource_key": "aws/duplicate-physical"})
        )
    elif case == "dependency-cycle":
        resources = [
            item.model_copy(update={"dependencies": ["aws/helm/lbc"]})
            if item.resource_key == "aws/nlb"
            else item.model_copy(update={"dependencies": ["aws/nlb"]})
            if item.resource_key == "aws/helm/lbc"
            else item
            for item in resources
        ]
    else:
        resources = [
            item.model_copy(
                update={"ownership": Ownership.EXTERNAL}
                if case == "external-aurora"
                else {
                    "dependencies": ["missing"]
                    if case == "missing-dependency"
                    else ["aws/nlb"]
                }
            )
            if item.resource_type == "aurora_cluster"
            else item
            for item in resources
        ]
    replace_snapshot(harness, resources)
    with pytest.raises(
        BootstrapError, match="ownership|dependency|dependencies|aliases"
    ):
        lifecycle.uninstall(harness.request(delete=True), runner=harness)
    assert harness.events == []
    assert harness.syncs == 0


@pytest.mark.parametrize("boundary", ["export", "kubernetes", "resource"])
def test_uninstall_read_failure_never_advances_to_aurora_deletion(
    harness, monkeypatch, boundary
):
    def failed(*_args, **_kwargs):
        raise BootstrapError("example read unavailable")

    if boundary == "export":
        monkeypatch.setattr(lifecycle, "fetch_installation_resource_registry", failed)
    elif boundary == "kubernetes":
        monkeypatch.setattr(
            lifecycle,
            "bounded_command",
            lambda arguments, **_kwargs: subprocess.CompletedProcess(
                arguments, 1, "", "example permission denied"
            ),
        )
    else:
        monkeypatch.setattr(harness, "exists", failed)
    with pytest.raises(BootstrapError, match="read unavailable|verification failed"):
        lifecycle.uninstall(harness.request(delete=True), runner=harness)
    assert "delete:aurora" not in harness.events
    assert "delete:cpu" not in harness.events


@pytest.mark.parametrize("stage", ["cleanup", "registry-sync", "cpu", "aurora"])
def test_uninstall_failure_resumes_original_transaction_without_repeating_finished_work(
    harness, stage
):
    if stage == "cleanup":
        harness.fail_cleanup = BootstrapError("example cleanup failure")
    elif stage == "registry-sync":
        harness.fail_sync = True
    elif stage == "cpu":
        harness.fail_cpu = True
    else:
        harness.fail_aurora = True
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError):
        lifecycle.uninstall(request, runner=harness)
    identifier = harness.state()["final_snapshot_identifier"]
    prior = list(harness.events)
    result = lifecycle.uninstall(request, runner=harness)
    assert harness.state()["phase"] == "COMPLETED"
    assert harness.state()["final_snapshot_identifier"] == identifier
    assert result["aurora_deleted_last"] is True
    assert result["gpu_clusters"] == "preserved"
    if stage in {"cpu", "aurora"}:
        assert harness.events.count("cleanup") == prior.count("cleanup")
        assert harness.events.count("delete:aws/nlb") == prior.count("delete:aws/nlb")
    assert harness.exports == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"cpu_disposition": "invalid"},
        {"final_snapshot_policy": "invalid"},
        {"reset_database": "false"},
    ],
)
def test_uninstall_rejects_invalid_policy_before_any_transport(harness, changes):
    with pytest.raises(BootstrapError, match="invalid|boolean"):
        replace(harness.request(), **changes)
    assert harness.exports == harness.syncs == 0
    assert harness.events == []


def test_uninstall_requires_exact_confirmation_before_registry_export(harness):
    with pytest.raises(BootstrapError, match="requires --confirm"):
        lifecycle.uninstall(
            replace(harness.request(), confirmation="WRONG"), runner=harness
        )
    assert harness.exports == 0
    assert harness.events == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "UNKNOWN"),
        ("cpu_disposition", "delete"),
        ("final_snapshot_policy", "skip"),
        ("site_sha256", "a" * 64),
        ("registry_sha256", "a" * 64),
        ("effective_policies", {}),
        ("supervision_lost", False),
    ],
)
def test_uninstall_resume_rejects_journal_or_authorization_drift(harness, field, value):
    harness.fail_cleanup = BootstrapError("example interruption")
    request = harness.request()
    with pytest.raises(BootstrapError, match="interruption"):
        lifecycle.uninstall(request, runner=harness)
    state = harness.state()
    state[field] = value
    path = harness.site.source.parent / "uninstall/state.json"
    path.write_text(json.dumps(state))
    prior = list(harness.events), harness.exports, harness.syncs
    with pytest.raises(BootstrapError, match="invalid|conflicts|changed|supervision"):
        lifecycle.uninstall(request, runner=harness)
    assert (harness.events, harness.exports, harness.syncs) == prior


@pytest.mark.parametrize("kind", ["cpu", "aurora"])
def test_uninstall_resume_refuses_missing_incarnation_proof(harness, kind):
    harness.fail_cleanup = BootstrapError("example interruption")
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError, match="interruption"):
        lifecycle.uninstall(request, runner=harness)
    state = harness.state()
    state[f"{kind}_binding"] = {}
    path = harness.site.source.parent / "uninstall/state.json"
    path.write_text(json.dumps(state))
    prior = list(harness.events)
    with pytest.raises(BootstrapError, match="invalid saved"):
        lifecycle.uninstall(request, runner=harness)
    assert harness.events == prior


@pytest.mark.parametrize(
    "name",
    ["installation-resources-before.json", "installation-resources-delete-plan.json"],
)
def test_uninstall_resume_requires_exported_registry_files(harness, name):
    harness.fail_cleanup = BootstrapError("example interruption")
    request = harness.request()
    with pytest.raises(BootstrapError, match="interruption"):
        lifecycle.uninstall(request, runner=harness)
    (harness.site.source.parent / "uninstall" / name).unlink()
    before = list(harness.events), harness.exports, harness.syncs
    with pytest.raises(BootstrapError, match="registry is missing"):
        lifecycle.uninstall(request, runner=harness)
    assert (harness.events, harness.exports, harness.syncs) == before


def test_uninstall_supervision_loss_remains_a_hard_barrier_after_restart(harness):
    harness.fail_cleanup = ProcessSupervisionLost("example missing completion proof")
    request = harness.request()
    with pytest.raises(ProcessSupervisionLost):
        lifecycle.uninstall(request, runner=harness)
    assert harness.state()["supervision_lost"] is True
    before = list(harness.events), harness.exports, harness.syncs
    with pytest.raises(BootstrapError, match="supervision was lost"):
        lifecycle.uninstall(request, runner=harness)
    assert (harness.events, harness.exports, harness.syncs) == before


def test_completed_uninstall_does_not_delete_recreated_resource(harness):
    request = harness.request()
    lifecycle.uninstall(request, runner=harness)
    before = list(harness.events)
    harness.no_mutations = True
    harness.existing.add("aws/nlb")
    with pytest.raises(BootstrapError, match="still exists"):
        lifecycle.uninstall(request, runner=harness)
    assert harness.events == before
