from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import rollback_alignment as alignment
from gpu_fault.admin.config import AdminConfig, persist_desired_admin_config
from gpu_fault.admin.site import SiteConfigError
from tests.admin.test_admin_site import site_file


def write_snapshot(root, document, name="one"):
    path = (
        root
        / "source-snapshots"
        / name
        / "repository-example/dist/release-old/release.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document))
    # The alignment refuses to publish a management manifest whose artifacts
    # are missing, so the fixture materializes every relative artifact it names.
    repository = path.parents[2]
    if not isinstance(document, dict):
        return path
    for value in (
        document.get("wheel"),
        document.get("bundle"),
        *(
            component.get("wheel")
            for component in (document.get("components") or {}).values()
            if isinstance(component, dict)
        ),
    ):
        if isinstance(value, str) and value and not value.startswith("/"):
            target = repository / value
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"artifact")
    return path


@pytest.fixture
def baseline(tmp_path):
    site = site_file(tmp_path)
    external_executor = tmp_path / "external" / "executor.whl"
    external_executor.parent.mkdir(parents=True, exist_ok=True)
    external_executor.write_bytes(b"artifact")
    manifest = {
        "release_id": "release-old",
        "wheel": "dist/old.whl",
        "bundle": "dist/old.tar",
        "delivery": {
            "sha256": "a" * 64,
            "images": {"runtime": {"reference": "example/runtime@sha256:" + "b" * 64}},
        },
        "components": {
            "control_plane": {"wheel": "dist/control.whl"},
            "executor": {"wheel": str(external_executor)},
            "legacy": {},
        },
    }
    path = write_snapshot(tmp_path, manifest)
    profile = tmp_path / "profiles/profile-old.yaml"
    profile.parent.mkdir()
    profile.write_text("profile_version: profile-old\n")
    previous = {
        "release_id": "release-old",
        "release_delivery_sha256": "a" * 64,
        "runtime_profile_version": "profile-old",
        "metadata": {"required-agent-config-digest": "c" * 64},
    }
    state = {
        "phase": "rolled-back",
        "rollback_result": {"status": "PASSED"},
        "previous": previous,
    }
    return site, path, manifest, state


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {},
        {"release_id": ""},
        {"release_id": "other", "delivery": {"sha256": "b" * 64}},
    ],
)
def test_previous_release_search_refuses_unbound_manifests(tmp_path, manifest):
    write_snapshot(tmp_path, manifest)
    with pytest.raises(SiteConfigError, match="cannot locate"):
        alignment.find_previous_release_manifest(
            tmp_path, {"release_delivery_sha256": "a" * 64}
        )


def test_previous_release_search_ignores_invalid_json(tmp_path):
    path = write_snapshot(tmp_path, {})
    path.write_text("invalid")
    with pytest.raises(SiteConfigError, match="cannot locate"):
        alignment.find_previous_release_manifest(
            tmp_path, {"release_delivery_sha256": "a" * 64}
        )


@pytest.mark.parametrize("identity", [None, "", "short"])
def test_previous_release_requires_delivery_identity_before_search(tmp_path, identity):
    with pytest.raises(SiteConfigError, match="delivery identity"):
        alignment.find_previous_release_manifest(
            tmp_path, {"release_delivery_sha256": identity}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "complete"},
        {"rollback_result": None},
        {"rollback_result": {"status": "FAILED"}},
        {"previous": None},
    ],
)
def test_management_alignment_requires_verified_rollback(baseline, changes):
    site, _path, _manifest, state = baseline
    state.update(changes)
    with pytest.raises(SiteConfigError, match="rolled-back|PASSED|previous baseline"):
        alignment.rollback_management_document(site, state)


@pytest.mark.parametrize("document", ["invalid: [yaml", "[]", "spec: []"])
def test_management_alignment_rejects_invalid_site_document(baseline, document):
    site, _path, _manifest, state = baseline
    site.write_text(document)
    with pytest.raises(SiteConfigError, match="cannot read|site is invalid"):
        alignment.rollback_management_document(site, state)


def test_management_alignment_requires_previous_profile_snapshot(baseline):
    site, _path, _manifest, state = baseline
    state["previous"]["runtime_profile_version"] = "missing"
    with pytest.raises(SiteConfigError, match="Profile snapshot is unavailable"):
        alignment.rollback_management_document(site, state)


@pytest.mark.parametrize("field", ["wheel", "bundle"])
def test_status_manifest_requires_both_artifact_paths(baseline, field):
    site, path, manifest, state = baseline
    manifest.pop(field)
    path.write_text(json.dumps(manifest))
    destination = site.parent / "status.json"
    with pytest.raises(SiteConfigError, match="wheel or bundle path"):
        alignment.rollback_management_document(
            site, state, management_manifest=destination
        )
    assert not destination.exists(), "invalid rollback manifest was materialized"


def test_status_manifest_normalizes_only_artifact_paths_and_keeps_source_binding(
    baseline,
):
    site, path, manifest, state = baseline
    current_root = Path(yaml.safe_load(site.read_text())["spec"]["repositoryRoot"])
    destination = site.parent / "status.json"
    document, config, root = alignment.rollback_management_document(
        site,
        state,
        management_manifest=destination,
        management_repository_root=current_root,
    )
    materialized = json.loads(destination.read_text())
    assert root == current_root
    assert document["spec"]["release"]["manifest"] == str(destination)
    assert materialized["wheel"] == str(path.parents[2] / manifest["wheel"])
    assert materialized["components"]["executor"]["wheel"] == str(
        site.parent / "external" / "executor.whl"
    ), "absolute component paths are kept as written"
    assert materialized["management_baseline"]["source_manifest"] == str(path)
    assert config == AdminConfig()


def test_temporary_status_site_is_removed_and_does_not_modify_managed_site(baseline):
    site, _path, _manifest, state = baseline
    original = site.read_bytes()
    with alignment.materialized_rollback_status_site(site, state) as temporary:
        assert temporary.is_file(), "rollback status site was not materialized"
        document = yaml.safe_load(temporary.read_text())
        assert document["spec"]["runtimeProfile"]["version"] == "profile-old"
    assert not temporary.exists(), "temporary rollback status site leaked"
    assert site.read_bytes() == original


def test_conflicting_previous_manifests_cannot_be_selected_by_timestamp(baseline):
    site, _path, manifest, state = baseline
    second = copy.deepcopy(manifest)
    second["wheel"] = "dist/different.whl"
    write_snapshot(site.parent, second, "two")
    with pytest.raises(SiteConfigError, match="multiple manifest identities"):
        alignment.find_previous_release_manifest(site.parent, state["previous"])


def test_rollback_overlay_preserves_configuration_missing_from_previous_snapshot(
    baseline,
):
    site, _path, _manifest, state = baseline
    current = AdminConfig().patched({"workflow": {"dispatcherWorkers": 12}})
    persist_desired_admin_config(site.parent, config=current, source="example-current")
    state["previous"]["admin_config"] = {"capacity": {"control_worker_replicas": 5}}
    _document, restored, _root = alignment.rollback_management_document(site, state)
    assert restored.capacity.control_worker_replicas == 5
    assert restored.workflow.dispatcher_workers == 12
    assert restored.aurora == current.aurora
