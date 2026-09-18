from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.e2e.regional import boot032_contract as contract
from tests.regional._cov95_boot032_world import World


@pytest.mark.parametrize(
    "kind", ["symlink", "shared", "directory", "traversal", "missing"]
)
def test_private_inputs_reject_unsafe_paths_and_unknown_reads(tmp_path, kind):
    path = tmp_path / "input.json"
    path.write_text("{}")
    path.chmod(0o600)
    if kind == "symlink":
        link = tmp_path / "alias.json"
        link.symlink_to(path)
        path = link
    elif kind == "shared":
        path.chmod(0o644)
    elif kind == "directory":
        path = tmp_path
    elif kind == "traversal":
        path = tmp_path / ".." / tmp_path.name / "input.json"
    else:
        path = tmp_path / "missing"
    with pytest.raises((contract.UninstallCaseError, OSError)):
        contract.read_document(path)


@pytest.mark.parametrize("value", [[], None, "not-an-object", 7])
def test_json_documents_must_be_structured_mappings(tmp_path, value):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    with pytest.raises(contract.UninstallCaseError, match="JSON object"):
        contract.read_document(path)


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_missing_explicit_kubeconfig_never_uses_ambient_credentials(
    tmp_path, monkeypatch, plane
):
    world = World(tmp_path, monkeypatch)
    config = dict(world.target.release_config)
    config.pop(plane + "_kubeconfig", None)
    site = replace(world.target, release_config=config, environment={})
    monkeypatch.setenv("KUBECONFIG", str(contract.kubeconfig(world.target, "gpu")))
    with pytest.raises(contract.UninstallCaseError, match="not explicit"):
        contract.kubeconfig(site, plane)


@pytest.mark.parametrize("clusters", [None, [], {}])
def test_empty_or_unreadable_gpu_cluster_scope_is_refused(
    tmp_path, monkeypatch, clusters
):
    world = World(tmp_path, monkeypatch)
    site = replace(
        world.target,
        release_config={**world.target.release_config, "clusters": clusters},
    )
    with pytest.raises(contract.UninstallCaseError, match="GPU clusters"):
        contract.cluster_specs(site)


@pytest.mark.parametrize("change", ["context", "eks", "hyperpod", "region", "account"])
def test_cluster_specs_reject_aliases_and_foreign_bindings(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)
    config = copy.deepcopy(world.target.release_config)
    gpu = config["clusters"][0]
    if change == "context":
        gpu["context"] = "cpu"
    elif change == "eks":
        gpu["eks_cluster_arn"] = config["cpu_eks_arn"]
    elif change == "hyperpod":
        gpu["hyperpod_cluster_name"] = config["cpu_hyperpod_cluster_name"]
    elif change == "region":
        gpu["eks_cluster_arn"] = gpu["eks_cluster_arn"].replace(
            "us-east-1", "us-west-2"
        )
    else:
        gpu["eks_cluster_arn"] = gpu["eks_cluster_arn"].replace(
            "123456789012", "111122223333"
        )
    with pytest.raises(contract.UninstallCaseError):
        contract.cluster_specs(replace(world.target, release_config=config))


@pytest.mark.parametrize("changed", ["site", "phase", "cpu", "gpu"])
def test_bootstrap_provenance_cannot_be_rebound_to_a_different_site(
    tmp_path, monkeypatch, changed
):
    world = World(tmp_path, monkeypatch)
    path = world.target.source.parent / "bootstrap-state.json"
    value = contract.read_document(path)
    if changed == "site":
        value["site_id"] = world.protected.metadata_name
    elif changed == "phase":
        value["phase"] = "in-progress"
    elif changed == "cpu":
        value["resources"]["initial_deploy_target"]["cpu"]["eks_arn"] = (
            world.protected.release_config["cpu_eks_arn"]
        )
    else:
        value["resources"]["initial_deploy_target"]["gpu_clusters"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(contract.UninstallCaseError, match="provenance"):
        world.settings.inputs()
    assert not world.settings.native_dir.exists(), (
        "invalid provisioning evidence cannot create native state"
    )


@pytest.mark.parametrize(
    "change", ["ordinary-name", "wrong-root", "accepted-path", "fixture-id"]
)
def test_only_canonical_separately_named_sacrificial_sites_are_eligible(
    tmp_path, monkeypatch, change
):
    world = World(tmp_path, monkeypatch)
    settings = world.settings
    if change == "ordinary-name":
        config = copy.deepcopy(world.target.release_config)
        config["cpu_hyperpod_cluster_name"] = "ordinary-cpu"
        settings = replace(
            settings, target=replace(world.target, release_config=config)
        )
    elif change == "wrong-root":
        settings = replace(
            settings, target=replace(world.target, repository_root=tmp_path)
        )
    elif change == "accepted-path":
        settings = replace(settings, target=world.protected)
    else:
        settings = replace(settings, fixture_id="ABCDEF012345")
    with pytest.raises(contract.UninstallCaseError):
        contract.validate_sites(settings)


def test_source_change_after_configure_refuses_even_semantically_equivalent_yaml(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    with world.target.source.open("a") as handle:
        handle.write("\n# externally changed after configuration\n")
    with pytest.raises(contract.UninstallCaseError, match="changed after"):
        world.settings.inputs()
    assert not world.settings.native_dir.exists(), (
        "stale loaded site cannot pass after bytes change"
    )


def test_input_owner_is_verified_and_public_manifest_stays_read_only(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    path.chmod(0o644)
    assert contract.file_digest(path, private=False), (
        "public read-only manifest is a valid input"
    )
    monkeypatch.setattr(os, "geteuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(contract.UninstallCaseError, match="another user"):
        contract.file_digest(path, private=False)
    assert path.read_text() == "{}", (
        "input validation must not modify public manifest content"
    )


def test_private_directory_requires_a_directory_not_a_regular_file(tmp_path):
    path = tmp_path / "state-dir"
    path.write_text("{}")
    path.chmod(0o600)
    with pytest.raises(contract.UninstallCaseError, match="file type"):
        contract.checked_path(Path(path), directory=True)
