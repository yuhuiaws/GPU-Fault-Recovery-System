from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from gpu_fault.admin import uninstall as lifecycle
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.resource_registry import (
    load_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from gpu_fault.installation_resources import InstallationResourceSnapshot
from tests.admin._aws_cleanup_support import resource
from tests.admin.test_cov95_uninstall_lifecycle import replace_snapshot
from tests.admin.test_uninstall_lifecycle import Harness


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


def directory(harness):
    return harness.site.source.parent / "uninstall"


def write_state(harness, state):
    (directory(harness) / "state.json").write_text(json.dumps(state))


@pytest.mark.parametrize(
    "error", [TimeoutError(), subprocess.TimeoutExpired("example-fake", 1)]
)
def test_uninstall_kubernetes_timeout_cannot_mean_absence(harness, monkeypatch, error):
    def unavailable(*_args, **_options):
        raise error

    monkeypatch.setattr(lifecycle, "bounded_command", unavailable)
    with pytest.raises(BootstrapError, match="verification timed out"):
        lifecycle.uninstall(harness.request(delete=True), runner=harness)
    assert "delete:aurora" not in harness.events
    assert "delete:cpu" not in harness.events


def test_cleanup_inventory_must_include_every_managed_gpu_context(harness):
    document = {
        "inventory_snapshot": {
            "cpu": {"resources": []},
            "gpu": {"resources": [], "by_context": {}},
        }
    }
    with pytest.raises(BootstrapError, match="lacks a selected GPU context"):
        lifecycle.verify_installed_registry_cleanup(harness.site, document)
    assert harness.events == []


def test_uninstall_failure_before_cleanup_journal_does_not_guess_cleanup_targets(
    harness, monkeypatch
):
    def unavailable(*_args, **_options):
        raise BootstrapError("example failed before cleanup journal")

    monkeypatch.setattr(harness, "run", unavailable)
    with pytest.raises(BootstrapError, match="before cleanup journal"):
        lifecycle.uninstall(harness.request(), runner=harness)
    assert harness.events == []
    assert not (directory(harness) / "kubernetes-cleanup.json").exists(), (
        "pre-journal failure fabricated cleanup evidence"
    )


def test_uninstall_compensation_failure_retains_original_error_and_cleanup_note(
    harness, monkeypatch
):
    run = harness.run
    harness.fail_cleanup = BootstrapError("example original cleanup failure")

    def unavailable(arguments, **options):
        if arguments[-1] == "cleanup-owned":
            raise TimeoutError("example cleanup compensation timeout")
        return run(arguments, **options)

    monkeypatch.setattr(harness, "run", unavailable)
    with pytest.raises(BootstrapError, match="original cleanup failure") as failure:
        lifecycle.uninstall(harness.request(), runner=harness)
    assert failure.value.__notes__ == [
        "temporary node cleanup remains unverified: TimeoutError"
    ]
    assert harness.state()["phase"] == "REGISTRY_EXPORTED"
    assert harness.events == ["cleanup"]


def test_uninstall_rejects_cpu_gpu_overlap_before_cleanup(harness):
    document = yaml.safe_load(harness.site.source.read_text())
    document["spec"]["clusters"][0]["hyperpodClusterName"] = document["spec"]["cpu"][
        "hyperpodClusterName"
    ]
    harness.site.source.write_text(yaml.safe_dump(document))
    harness.site = load_site(harness.site.source)
    with pytest.raises(BootstrapError, match="CPU and GPU cluster identities overlap"):
        lifecycle.uninstall(harness.request(), runner=harness)
    assert harness.events == []


def test_legacy_lbc_attachment_cycle_cannot_produce_deletion_order(harness):
    role = resource("iam_role", "example-lbc").model_copy(
        update={
            "resource_key": "aws/iam/lbc/role",
            "dependencies": ["aws/iam/lbc/policy"],
        }
    )
    policy = resource("iam_policy", "example-lbc-policy").model_copy(
        update={
            "resource_key": "aws/iam/lbc/policy",
            "dependencies": ["aws/iam/lbc/role"],
        }
    )
    replace_snapshot(harness, [*harness.snapshot.resources, role, policy])
    with pytest.raises(BootstrapError, match="dependencies contain a cycle"):
        lifecycle.uninstall(harness.request(), runner=harness)
    assert harness.events == []


def test_reused_pod_identity_association_is_detached_not_preserved(harness):
    association = resource(
        "eks_pod_identity_association",
        "association-example",
        ownership=Ownership.REUSED,
    )
    replace_snapshot(harness, [*harness.snapshot.resources, association])
    harness.existing.add(association.resource_key)
    result = lifecycle.uninstall(harness.request(), runner=harness)
    assert result["registry_entries_detached"] == 1
    assert "delete:" + association.resource_key in harness.events


def test_uninstall_resume_refuses_changed_deletion_plan_identity(harness):
    harness.fail_cleanup = BootstrapError("example stop")
    request = harness.request()
    with pytest.raises(BootstrapError, match="example stop"):
        lifecycle.uninstall(request, runner=harness)
    path = directory(harness) / "installation-resources-delete-plan.json"
    plan = load_installation_resource_snapshot(path)
    resources = [
        item.model_copy(update={"resource_id": "different-nlb"})
        if item.resource_type == "nlb"
        else item
        for item in plan.resources
    ]
    changed = InstallationResourceSnapshot(site_id=plan.site_id, resources=resources)
    changed = changed.model_copy(update={"source_sha256": changed.digest()})
    write_installation_resource_snapshot(harness.site, changed, path=path)
    prior = list(harness.events), harness.syncs
    with pytest.raises(BootstrapError, match="deletion plan differs"):
        lifecycle.uninstall(request, runner=harness)
    assert (harness.events, harness.syncs) == prior


@pytest.mark.parametrize(
    "change", ["missing-cleanup", "cleanup-digest", "missing-cpu-binding"]
)
def test_uninstall_resume_requires_original_pre_cpu_cleanup_evidence(harness, change):
    harness.fail_cpu = True
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError, match="CPU deletion"):
        lifecycle.uninstall(request, runner=harness)
    state = harness.state()
    if change == "missing-cleanup":
        (directory(harness) / "kubernetes-cleanup.json").unlink()
    elif change == "cleanup-digest":
        state["cleanup_sha256"] = "f" * 64
        write_state(harness, state)
    else:
        state.pop("cpu_binding")
        write_state(harness, state)
    prior = list(harness.events)
    with pytest.raises(
        BootstrapError,
        match="cleanup state is missing|snapshot changed|lacks an incarnation binding",
    ):
        lifecycle.uninstall(request, runner=harness)
    assert harness.events == prior


def test_uninstall_does_not_accept_cleanup_state_changed_after_remote_verification(
    harness, monkeypatch
):
    run = harness.run

    def changed(arguments, **options):
        result = run(arguments, **options)
        if arguments[-1] == "verify-targets":
            path = directory(harness) / "kubernetes-cleanup.json"
            document = json.loads(path.read_text())
            document["status"] = "FAILED"
            path.write_text(json.dumps(document))
        return result

    monkeypatch.setattr(harness, "run", changed)
    with pytest.raises(BootstrapError, match="did not reach CLEANUP_COMPLETED"):
        lifecycle.uninstall(harness.request(), runner=harness)
    assert not any(event.startswith("delete:") for event in harness.events), (
        "incomplete cleanup evidence authorized resource deletion"
    )


@pytest.mark.parametrize("change", ["binding", "pre-aurora-snapshot"])
def test_resumed_aurora_deletion_keeps_its_original_incarnation_and_snapshot(
    harness, change
):
    harness.fail_aurora = True
    request = harness.request(delete=True)
    with pytest.raises(BootstrapError, match="Aurora deletion"):
        lifecycle.uninstall(request, runner=harness)
    state = harness.state()
    if change == "binding":
        state.pop("aurora_binding")
    else:
        state["pre_aurora_sha256"] = "f" * 64
    write_state(harness, state)
    prior = list(harness.events)
    with pytest.raises(
        BootstrapError,
        match="lacks an incarnation checkpoint|pre-Aurora cleanup snapshot changed",
    ):
        lifecycle.uninstall(request, runner=harness)
    assert harness.events == prior


@pytest.mark.parametrize("change", ["digest", "missing-snapshot", "different-snapshot"])
def test_completed_uninstall_replay_requires_the_original_final_snapshot(
    harness, change
):
    request = harness.request(delete=True)
    lifecycle.uninstall(request, runner=harness)
    harness.no_mutations = True
    state = harness.state()
    path = directory(harness) / "installation-resources-final.json"
    final = load_installation_resource_snapshot(path)
    if change == "digest":
        state["final_registry_sha256"] = "f" * 64
    else:
        resources = []
        for item in final.resources:
            if item.resource_key == "aws/aurora/final-snapshot":
                if change == "missing-snapshot":
                    continue
                item = item.model_copy(update={"resource_id": "different-snapshot"})
            resources.append(item)
        changed = InstallationResourceSnapshot(
            site_id=final.site_id, resources=resources
        )
        changed = changed.model_copy(update={"source_sha256": changed.digest()})
        write_installation_resource_snapshot(harness.site, changed, path=path)
        state["final_registry_sha256"] = changed.digest()
    write_state(harness, state)
    prior = list(harness.events)
    with pytest.raises(
        BootstrapError,
        match="completed uninstall snapshot changed|snapshot is missing|original binding",
    ):
        lifecycle.uninstall(request, runner=harness)
    assert harness.events == prior
