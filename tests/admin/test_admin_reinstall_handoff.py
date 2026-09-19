"""Offline uninstall/reinstall cycles with independent namespace incarnations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import installation_lifecycle as records
from gpu_fault.admin import uninstall as lifecycle
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_command import retire_site_after_uninstall
from gpu_fault.admin.site import SiteConfigError, load_site, materialized_release_config
from gpu_fault_release import regional_release_prerequisite_repair as repair
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_retained_database import RETAINED_ORIGIN_KEY
from tests.admin.test_uninstall_lifecycle import STATE, Harness, registry
from tests.regional._prerequisite_repair_support import repair_release


def test_application_wheels_do_not_import_uninstall_lifecycle_mutations():
    from scripts.component_wheels import component_modules

    for component in ("control_plane", "executor", "node_runtime"):
        modules = component_modules(component)
        assert "gpu_fault.admin.installation_lifecycle" not in modules
        assert "gpu_fault.admin.uninstall" not in modules
    assert "gpu_fault.installation_lifecycle" in component_modules("control_plane")


def retained_release(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    lifecycle.uninstall(harness.request(), runner=harness)
    archive = retire_site_after_uninstall(tmp_path)
    assert archive is not None, "completed uninstall must be archived"
    harness.site.source.write_bytes((archive / "site.yaml").read_bytes())
    harness.site.source.chmod(0o600)
    harness.site = load_site(harness.site.source)
    instance = repair_release(monkeypatch, bootstrap=True)
    instance.config.site_name = harness.site.metadata_name
    instance.config.cpu_eks_arn = harness.site.release_config["cpu_eks_arn"]
    instance.config.health.aurora_cluster_id = harness.site.release_config["health"][
        "aurora_cluster_id"
    ]
    instance.runner.aurora_cluster_id = instance.config.health.aurora_cluster_id
    instance.runner.database_state = "initialized"
    instance.runner.namespace_uid = "namespace-new"
    with materialized_release_config(harness.site) as path:
        config = json.loads(path.read_text())
    instance.config.installation_id = config["installation_id"]
    instance.config.retained_database_handoff = Path(
        config["retained_database_handoff"]
    )
    return harness, archive, instance


def test_keep_reinstall_uses_bound_handoff_and_fresh_initialized_database_read(
    tmp_path, monkeypatch
):
    harness, archive, instance = retained_release(tmp_path, monkeypatch)
    old = records.read_record(archive / "uninstall/state.json")
    assert (
        old["retained_database"]["cluster_resource_id"] == harness.database_resource_id
    )
    repair.prepare_bootstrap_workflows(instance)
    origin = copy.deepcopy(instance.state[RETAINED_ORIGIN_KEY])
    assert repair.BOOTSTRAP_ORIGIN_KEY not in instance.state
    assert origin["database_state"] == "initialized"
    assert (
        origin["retained_database_handoff"]["previous_namespace_uid"] == "namespace-old"
    )
    assert (
        records.read_record(tmp_path / records.INSTALLATION_FILE)["retained_adoption"][
            "namespace_uid"
        ]
        == "namespace-new"
    )
    repair.prepare_bootstrap_workflows(instance)
    assert instance.state[RETAINED_ORIGIN_KEY] == origin
    assert instance.runner.events.count("store-job") == 2
    assert instance.runner.jobs == {}
    assert "delete:aurora" not in harness.events


@pytest.mark.parametrize("kind", ["workflow", "remote_command", "observation"])
def test_retained_handoff_never_caches_current_row_safety(tmp_path, monkeypatch, kind):
    _harness, _archive, instance = retained_release(tmp_path, monkeypatch)
    repair.prepare_bootstrap_workflows(instance)
    instance.runner.store_blockers[kind] = 1
    with pytest.raises(ReleaseError, match="safety proof failed"):
        repair.prepare_bootstrap_workflows(instance)
    assert instance.runner.events.count("store-job") == 2
    assert instance.runner.jobs == {}


@pytest.mark.parametrize(
    "changed", ["database", "old-namespace", "third-namespace", "schema"]
)
def test_retained_adoption_rejects_incarnation_namespace_and_schema_drift(
    tmp_path, monkeypatch, changed
):
    _harness, _archive, instance = retained_release(tmp_path, monkeypatch)
    if changed == "database":
        instance.runner.database_resource_id = "recreated-database"
    elif changed == "old-namespace":
        instance.runner.namespace_uid = "namespace-old"
    elif changed == "third-namespace":
        repair.prepare_bootstrap_workflows(instance)
        instance.runner.namespace_uid = "namespace-third"
        instance.state.clear()
        instance.runner.cloud_state.clear()
    else:
        instance.runner.proof_schema_version += 1
    with pytest.raises(ReleaseError):
        repair.prepare_bootstrap_workflows(instance)
    assert instance.runner.jobs == {}


def test_retained_adoption_refuses_copied_empty_origin_and_missing_completion(
    tmp_path, monkeypatch
):
    _harness, archive, instance = retained_release(tmp_path, monkeypatch)
    instance.state[repair.BOOTSTRAP_ORIGIN_KEY] = {
        "database_state": "uninitialized_empty"
    }
    with pytest.raises(ReleaseError, match="cannot reuse"):
        repair.bootstrap_workflow_proof(instance)
    instance.state.pop(repair.BOOTSTRAP_ORIGIN_KEY)
    state_path = archive / "uninstall/state.json"
    old = records.read_record(state_path)
    old["phase"] = "AURORA_DELETE_IN_PROGRESS"
    records.write_record(state_path, old)
    with pytest.raises(ReleaseError, match="completed uninstall"):
        repair.prepare_bootstrap_workflows(instance)
    assert "store-job" not in instance.runner.events


def test_retained_old_schema_is_not_claimed_empty_or_already_ensured(
    tmp_path, monkeypatch
):
    _harness, _archive, instance = retained_release(tmp_path, monkeypatch)
    instance.runner.proof_schema_version -= 1
    repair.prepare_bootstrap_workflows(instance)
    origin = copy.deepcopy(instance.state[RETAINED_ORIGIN_KEY])
    assert origin["schema_version"] == instance.config.database_schema_version - 1
    assert origin["schema_ensure_required"] is True
    assert origin["database_state"] == "initialized"
    assert repair.BOOTSTRAP_ORIGIN_KEY not in instance.state
    instance.runner.proof_schema_version += 1
    repair.prepare_bootstrap_workflows(instance)
    assert instance.state[RETAINED_ORIGIN_KEY] == origin
    assert instance.state["bootstrap_store_safety"]["schema_version"] == (
        instance.config.database_schema_version
    )
    assert instance.state["bootstrap_store_safety"]["schema_ensure_required"] is False


def test_retained_v17_bootstrap_runs_normal_ensure_before_candidate_cpu(
    tmp_path, monkeypatch
):
    from gpu_fault_release import rollout

    _harness, _archive, instance = retained_release(tmp_path, monkeypatch)
    instance.runner.proof_schema_version = 17
    events = instance.runner.events
    run = instance.runner.run

    def bootstrap_run(arguments, **kwargs):
        text = kwargs.get("input_text")
        if (
            "apply" in arguments
            and text
            and any(
                item.get("kind") == "Namespace" for item in yaml.safe_load_all(text)
            )
        ):
            events.append("namespace-prerequisites")
            return ""
        return run(arguments, **kwargs)

    monkeypatch.setattr(instance.runner, "run", bootstrap_run)
    monkeypatch.setattr(rollout, "ensure_runtime_profile", lambda _release: None)
    monkeypatch.setattr(rollout, "bootstrap_gpu_clusters", lambda *_args: None)
    for name in (
        "_initialize_registry",
        "_apply_release_metadata_rbac",
        "_prepare_nlb",
        "_wait_nlb",
        "_validate_release",
    ):
        monkeypatch.setattr(instance, name, lambda: None, raising=False)
    monkeypatch.setattr(instance, "_require_cpu_secrets", lambda **_kwargs: None)
    monkeypatch.setattr(instance, "_upload_release", lambda _diff=None: None)
    monkeypatch.setattr(
        instance,
        "_prepare_bootstrap_workflows",
        lambda: repair.prepare_bootstrap_workflows(instance),
        raising=False,
    )

    def ensure_schema():
        proof = instance.state["bootstrap_store_safety"]
        assert proof["schema_version"] == 17 and proof["schema_ensure_required"]
        assert proof["database_state"] == "initialized"
        events.append("candidate-schema-ensure")
        instance.runner.proof_schema_version = 18

    def cpu(**_kwargs):
        assert instance.runner.proof_schema_version == 18
        events.append("candidate-cpu")

    monkeypatch.setattr(instance, "_ensure_schema", ensure_schema)
    monkeypatch.setattr(instance, "_apply_cpu", cpu)
    rollout.RegionalRelease.bootstrap(instance)
    assert events.index("store-job") < events.index("candidate-schema-ensure")
    assert events.index("candidate-schema-ensure") < events.index("candidate-cpu")
    assert repair.BOOTSTRAP_ORIGIN_KEY not in instance.state
    assert instance.state[RETAINED_ORIGIN_KEY]["schema_version"] == 17


def test_two_complete_uninstall_cycles_export_fresh_registry_and_namespace_uids(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, monkeypatch)
    lifecycle.uninstall(harness.request(), runner=harness)
    first = harness.state()
    archive = retire_site_after_uninstall(tmp_path)
    assert archive is not None
    first_cleanup = STATE.read_state(archive / "uninstall/kubernetes-cleanup.json")
    harness.site.source.write_bytes((archive / "site.yaml").read_bytes())
    harness.site.source.chmod(0o600)
    harness.site = load_site(harness.site.source)
    fresh = registry(harness.site)
    fresh = fresh.model_copy(
        update={
            "resources": [
                item.model_copy(update={"resource_id": "new-nlb"})
                if item.resource_type == "nlb"
                else item
                for item in fresh.resources
            ]
        }
    )
    harness.snapshot = fresh.model_copy(update={"source_sha256": fresh.digest()})
    harness.existing = {item.resource_key for item in harness.snapshot.resources}
    harness.namespace_uid = "namespace-second"
    lifecycle.uninstall(harness.request(), runner=harness)
    second = harness.state()
    second_cleanup = STATE.read_state(tmp_path / "uninstall/kubernetes-cleanup.json")
    assert first["installation_id"] != second["installation_id"]
    assert first["attempt_id"] != second["attempt_id"]
    assert first["registry_sha256"] != second["registry_sha256"]
    assert first_cleanup["run_id"] != second_cleanup["run_id"]
    assert second_cleanup["namespace_snapshots"]["cpu"]["uid"] == "namespace-second"
    assert first_cleanup["namespace_snapshots"]["cpu"]["uid"] == "namespace-old"
    assert harness.events.count("cleanup") == harness.exports == 2
    harness.no_mutations = True
    lifecycle.uninstall(harness.request(), runner=harness)
    assert harness.events.count("cleanup") == harness.exports == 2
    second_archive = retire_site_after_uninstall(tmp_path)
    assert second_archive is not None and second_archive != archive
    assert records.read_record(archive / "uninstall/state.json") == first
    assert records.read_record(second_archive / "uninstall/state.json") == second


@pytest.mark.parametrize(
    "boundary", ["site.yaml", "uninstall", records.INSTALLATION_FILE]
)
def test_retirement_resumes_after_move_or_new_identity_ack_loss(
    tmp_path, monkeypatch, boundary
):
    harness = Harness(tmp_path, monkeypatch)
    lifecycle.uninstall(harness.request(), runner=harness)
    move = Path.replace
    write = records.write_record
    failed = False

    def interrupted_move(source, target):
        nonlocal failed
        result = move(source, target)
        if (
            boundary != records.INSTALLATION_FILE
            and source == tmp_path / boundary
            and not failed
        ):
            failed = True
            raise OSError("modeled move ACK loss")
        return result

    def interrupted_write(path, value):
        nonlocal failed
        write(path, value)
        if (
            boundary == records.INSTALLATION_FILE
            and path == tmp_path / records.INSTALLATION_FILE
            and "retained_uninstall" in value
            and not failed
        ):
            failed = True
            raise OSError("modeled new identity ACK loss")

    monkeypatch.setattr(Path, "replace", interrupted_move)
    monkeypatch.setattr(records, "write_record", interrupted_write)
    with pytest.raises(OSError, match="ACK loss"):
        retire_site_after_uninstall(tmp_path)
    monkeypatch.setattr(Path, "replace", move)
    monkeypatch.setattr(records, "write_record", write)
    archive = retire_site_after_uninstall(tmp_path)
    assert archive is not None
    assert not (tmp_path / "uninstall").exists(), (
        "retirement retry must not leave the old uninstall transaction active"
    )
    assert records.read_record(archive / "uninstall/state.json")["phase"] == "COMPLETED"
    assert retire_site_after_uninstall(tmp_path) is None
    assert len(list(tmp_path.glob("retired-*"))) == 1


def test_orphan_cleanup_files_and_foreign_installation_cannot_resume(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, monkeypatch)
    directory = tmp_path / "uninstall"
    directory.mkdir()
    orphan = directory / "kubernetes-cleanup.json"
    orphan.write_text("{}")
    with pytest.raises(BootstrapError, match="original journal"):
        lifecycle.uninstall(harness.request(), runner=harness)
    orphan.unlink()
    lifecycle.uninstall(harness.request(), runner=harness)
    path = tmp_path / records.INSTALLATION_FILE
    record = records.read_record(path)
    record["installation_id"] = "f" * 32
    records.write_record(path, record)
    with pytest.raises(BootstrapError, match="installation_id"):
        lifecycle.uninstall(harness.request(), runner=harness)
    with pytest.raises(SiteConfigError, match="another installation"):
        retire_site_after_uninstall(tmp_path)


@pytest.mark.parametrize("key,value", [("schema_version", 2.0), ("reset_database", 0)])
def test_uninstall_resume_rejects_changed_policy_or_schema_types(
    tmp_path, monkeypatch, key, value
):
    harness = Harness(tmp_path, monkeypatch)
    lifecycle.uninstall(harness.request(), runner=harness)
    path = tmp_path / "uninstall/state.json"
    record = records.read_record(path)
    record[key] = value
    records.write_record(path, record)
    harness.no_mutations = True
    with pytest.raises(BootstrapError, match=key):
        lifecycle.uninstall(harness.request(), runner=harness)
