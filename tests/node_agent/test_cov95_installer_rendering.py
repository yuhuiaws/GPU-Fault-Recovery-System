from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import pytest
import yaml

from gpu_fault.node_installer_rendering import (
    InstallerNode,
    configure_node_dependencies,
    literal_environment,
    load_installer_template,
    manifest_object,
    manifest_objects,
    preflight_job,
    set_installer_environment,
    validate_installer_template,
    validate_node_dependencies,
)
from tests.deploy.test_installer_template_identity import (
    DEPENDENCY_IMAGE,
    INVENTORY_SHA,
    pinned_identity,
    template_job,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_sha256", "X" * 64, "SHA-256 pins"),
        ("bundle_sha256", "", "SHA-256 pins"),
        ("template_sha256", "short", "SHA-256 pins"),
        ("config_digest", "bad identity", "config identity"),
        ("deadline_seconds", 59, "deadline"),
        ("deadline_seconds", 86401, "deadline"),
        ("require_rollback_slot", 1, "rollback-slot"),
        ("template_content_sha256", "bad", "content identity"),
        ("node_dependency_image", None, "declare image and inventory"),
        ("node_wheelhouse_sha256", None, "declare image and inventory"),
        ("node_dependency_image", "registry.example:latest", "digest pinned"),
        ("node_wheelhouse_sha256", "", "digest pinned"),
    ],
)
def test_installer_identity_rejects_unpinned_or_ambiguous_input(
    field, value, message
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(pinned_identity("template"), **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "bad/name", "node name"),
        ("uid", "", "node UID"),
        ("uid", "uid with spaces", "node UID"),
        ("address", "not-an-ip", "IP"),
        ("instance_type", "ml.unsupported", "unsupported"),
    ],
)
def test_node_binding_rejects_invalid_target(field, value, message) -> None:
    node = InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge")
    with pytest.raises(ValueError, match=message):
        replace(node, **{field: value})


@pytest.mark.parametrize("value", [None, [], {1: "non-string-key"}])
def test_manifest_object_requires_a_typed_mapping(value) -> None:
    with pytest.raises(ValueError, match="object with string keys"):
        manifest_object(value, "fixture")


def test_manifest_collection_requires_an_array() -> None:
    with pytest.raises(ValueError, match="must be an array"):
        manifest_objects({"name": "not-an-array"}, "volumes")


@pytest.mark.parametrize(
    "environment",
    [
        [{"name": "DUP", "value": "a"}, {"name": "DUP", "value": "b"}],
        [{"name": None, "value": "a"}],
    ],
)
def test_environment_rejects_duplicate_or_missing_names(environment) -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        literal_environment({"env": copy.deepcopy(environment)})
    with pytest.raises(ValueError, match="names must be unique"):
        set_installer_environment({"env": environment}, {"MISSING": "x"})


@pytest.mark.parametrize(
    "image,inventory", [("", INVENTORY_SHA), (DEPENDENCY_IMAGE, "x")]
)
def test_dependency_identity_is_complete_at_both_public_boundaries(image, inventory):
    job = template_job(offline=False)
    before = copy.deepcopy(job)
    with pytest.raises(ValueError, match="digest pinned"):
        configure_node_dependencies(job, image, inventory)
    assert job == before
    with pytest.raises(ValueError, match="digest pinned"):
        validate_node_dependencies(job, image=image, inventory_sha256=inventory)


@pytest.mark.parametrize("field,value", [("apiVersion", "batch/v2"), ("kind", "Pod")])
def test_pinned_yaml_still_must_describe_a_job(field, value) -> None:
    job = template_job()
    job[field] = value
    text = yaml.safe_dump(job)
    identity = pinned_identity(text)
    with pytest.raises(ValueError, match="not a batch/v1 Job"):
        validate_installer_template(job, identity)
    with pytest.raises(RuntimeError, match="not a Job document"):
        load_installer_template(
            text,
            expected_sha256=identity.template_content_sha256,
            identity=identity,
            origin="unit",
        )


@pytest.mark.parametrize("text", ["{broken:", "[]", "!!python/object:untrusted {}"])
def test_pinned_content_does_not_bypass_safe_yaml_shape_checks(text) -> None:
    with pytest.raises(RuntimeError, match="not a Job document"):
        load_installer_template(
            text,
            expected_sha256=hashlib.sha256(text.encode()).hexdigest(),
            origin="unit",
        )


@pytest.mark.parametrize(
    "mutation", ["delivery-mount", "subpath", "subpath-expression"]
)
def test_pinned_offline_dependencies_cannot_change_the_delivery_path(mutation) -> None:
    job = template_job()
    pod = job["spec"]["template"]["spec"]
    if mutation == "delivery-mount":
        pod["initContainers"][0]["volumeMounts"][0]["mountPath"] = "/another-location"
    else:
        mount = pod["containers"][0]["volumeMounts"][-1]
        mount["subPath" if mutation == "subpath" else "subPathExpr"] = "elsewhere"
    with pytest.raises(ValueError, match="delivery mount differs|not read-only"):
        validate_node_dependencies(
            job, image=DEPENDENCY_IMAGE, inventory_sha256=INVENTORY_SHA
        )


@pytest.mark.parametrize("mutation", ["group", "volume", "init"])
def test_dependency_configuration_refuses_conflicts(mutation) -> None:
    job = template_job(offline=False)
    pod = job["spec"]["template"]["spec"]
    if mutation == "group":
        pod["securityContext"] = {"fsGroup": 1000}
    elif mutation == "volume":
        pod["volumes"].append({"name": "node-wheelhouse", "emptyDir": {}})
    else:
        pod["initContainers"] = [{"name": "node-dependencies", "image": "untrusted"}]
    with pytest.raises(ValueError, match="conflicts|already exists"):
        configure_node_dependencies(job, DEPENDENCY_IMAGE, INVENTORY_SHA)


def test_preflight_requires_the_expected_host_mount_and_does_not_mutate_template():
    job = template_job()
    job["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]["mountPath"] = (
        "/elsewhere"
    )
    before = copy.deepcopy(job)
    with pytest.raises(ValueError, match="host root mount differs"):
        preflight_job(
            job,
            InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge"),
            pinned_identity(yaml.safe_dump(job)),
            "local",
        )
    assert job == before
