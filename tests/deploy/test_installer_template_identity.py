from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.node_installer_reconciler import NodeInstallerReconciler
from gpu_fault.node_installer_rendering import (
    InstallerIdentity,
    InstallerNode,
    configure_node_dependencies,
    load_installer_template,
    preflight_job,
    render_installer_job,
)

DEPENDENCY_IMAGE = "registry.example/dependencies@sha256:" + "a" * 64
INVENTORY_SHA = "b" * 64


def template_job(*, offline: bool = True) -> dict[str, Any]:
    job: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "previous-installer", "namespace": "gpu-fault-system"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "installer",
                            "image": "registry.example/installer@sha256:" + "c" * 64,
                            "args": ["previous-template-program"],
                            "env": [
                                {"name": name, "value": ""}
                                for name in (
                                    "TARGET_NODE_NAME",
                                    "TARGET_NODE_IP",
                                    "TARGET_NODE_UID",
                                    "NODE_INSTANCE_TYPE",
                                    "EXPECTED_GPU_COUNT",
                                    "EXPECTED_EFA_DEVICE_COUNT",
                                    "DCGM_METRICS_URL_B64",
                                    "DCGM_EXPORTER_INTERVAL_MS",
                                    "PREFLIGHT_ONLY",
                                    "NODE_WHEELHOUSE_SHA256",
                                    "REQUIRE_ROLLBACK_SLOT",
                                )
                            ],
                            "volumeMounts": [
                                {"name": "host-root", "mountPath": "/host"}
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "host-root", "hostPath": {"path": "/"}},
                        {"name": "node-secret", "secret": {}},
                        {"name": "installer", "configMap": {"name": "old-bundle"}},
                    ],
                }
            }
        },
    }
    if offline:
        configure_node_dependencies(job, DEPENDENCY_IMAGE, INVENTORY_SHA)
    return job


def pinned_identity(text: str, *, offline: bool = True) -> InstallerIdentity:
    return InstallerIdentity(
        namespace="gpu-fault-system",
        config_digest="old-config",
        artifact_sha256="1" * 64,
        bundle_sha256="2" * 64,
        template_sha256="3" * 64,
        node_action_keys_secret="gpu-fault-node-action-keys",
        deadline_seconds=840,
        metrics_url_template="http://{node_ip}:9400/metrics",
        template_content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        node_dependency_image=DEPENDENCY_IMAGE if offline else "",
        node_wheelhouse_sha256=INVENTORY_SHA if offline else "",
    )


def test_exact_template_pin_is_distinct_from_source_identity() -> None:
    text = yaml.safe_dump(template_job()).replace("\n", "\r\n")
    identity = pinned_identity(text)
    assert identity.template_content_sha256 != identity.template_sha256
    job = load_installer_template(
        text.encode(),
        expected_sha256=identity.template_content_sha256,
        identity=identity,
        origin="test template",
    )
    node = InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge")
    for rendered in (
        render_installer_job(job, node, identity, "install-test"),
        preflight_job(job, node, identity, "test-run"),
    ):
        pod = cast(dict[str, Any], rendered["spec"])["template"]["spec"]
        assert pod["initContainers"][0]["image"] == DEPENDENCY_IMAGE
        assert pod["containers"][0]["args"] == ["previous-template-program"]
    with pytest.raises(RuntimeError, match="does not match"):
        load_installer_template(
            text.replace("\r\n", "\n"),
            expected_sha256=identity.template_content_sha256,
            identity=identity,
            origin="test template",
        )


@pytest.mark.parametrize("pin", [None, "", "invalid", "9" * 64])
def test_selected_template_never_supplies_its_own_trust(pin: str | None) -> None:
    text = yaml.safe_dump(template_job())
    with pytest.raises(
        RuntimeError, match="GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"
    ):
        load_installer_template(text, expected_sha256=pin, origin="untrusted template")


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("image", "dependency image does not match"),
        ("inventory", "dependency inventory does not match"),
        ("inventory-reference", "dependency inventory does not match"),
        ("duplicate-init", "exactly one node-dependencies"),
        ("writable-mount", "dependency mount is not read-only"),
        ("wrong-volume", "dependency volume is not an emptyDir"),
    ],
)
def test_matching_content_pin_does_not_replace_dependency_identity(
    mutation: str, diagnostic: str
) -> None:
    job = template_job()
    pod = job["spec"]["template"]["spec"]
    installer = pod["containers"][0]
    if mutation == "image":
        pod["initContainers"][0]["image"] = (
            "registry.example/candidate@sha256:" + "d" * 64
        )
    elif mutation in {"inventory", "inventory-reference"}:
        value = next(
            item
            for item in installer["env"]
            if item["name"] == "NODE_WHEELHOUSE_SHA256"
        )
        value["value"] = "e" * 64
        if mutation == "inventory-reference":
            value.pop("value")
            value["valueFrom"] = {"configMapKeyRef": {"name": "mutable", "key": "pin"}}
    elif mutation == "duplicate-init":
        pod["initContainers"].append(dict(pod["initContainers"][0]))
    elif mutation == "writable-mount":
        installer["volumeMounts"][-1]["readOnly"] = False
    else:
        pod["volumes"][-1] = {"name": "node-wheelhouse", "hostPath": {"path": "/tmp"}}
    text = yaml.safe_dump(job)
    identity = pinned_identity(text)
    with pytest.raises(ValueError, match=diagnostic):
        load_installer_template(
            text,
            expected_sha256=identity.template_content_sha256,
            identity=identity,
            origin="test template",
        )


def test_legacy_absence_is_explicit_and_cannot_hide_offline_dependencies() -> None:
    text = yaml.safe_dump(template_job(offline=False))
    identity = pinned_identity(text, offline=False)
    assert load_installer_template(
        text,
        expected_sha256=identity.template_content_sha256,
        identity=identity,
        origin="legacy template",
    ) == template_job(offline=False)
    offline = yaml.safe_dump(template_job())
    unexpected = replace(
        identity, template_content_sha256=hashlib.sha256(offline.encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="unexpected offline node dependencies"):
        load_installer_template(
            offline,
            expected_sha256=unexpected.template_content_sha256,
            identity=unexpected,
            origin="legacy template",
        )


def test_content_pin_cannot_be_omitted_from_identity_at_load() -> None:
    text = yaml.safe_dump(template_job())
    identity = pinned_identity(text)
    with pytest.raises(RuntimeError, match="content pin disagrees with identity"):
        load_installer_template(
            text,
            expected_sha256=identity.template_content_sha256,
            identity=replace(identity, template_content_sha256=None),
            origin="test template",
        )


def test_legacy_absence_cannot_be_filled_from_a_mutable_environment_reference() -> None:
    job = template_job(offline=False)
    environment = job["spec"]["template"]["spec"]["containers"][0]["env"]
    inventory = next(
        item for item in environment if item["name"] == "NODE_WHEELHOUSE_SHA256"
    )
    inventory.pop("value")
    inventory["valueFrom"] = {"configMapKeyRef": {"name": "mutable", "key": "pin"}}
    text = yaml.safe_dump(job)
    identity = pinned_identity(text, offline=False)
    with pytest.raises(ValueError, match="unexpected offline node dependencies"):
        load_installer_template(
            text,
            expected_sha256=identity.template_content_sha256,
            identity=identity,
            origin="legacy template",
        )


def test_selected_template_preflight_retains_the_required_old_slot_check() -> None:
    text = yaml.safe_dump(template_job())
    identity = replace(pinned_identity(text), require_rollback_slot=True)
    job = load_installer_template(
        text,
        expected_sha256=identity.template_content_sha256,
        identity=identity,
        origin="previous template",
    )
    node = InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge")
    checked = preflight_job(job, node, identity, "test-run")
    pod = cast(dict[str, Any], checked["spec"])["template"]["spec"]
    environment = {item["name"]: item["value"] for item in pod["containers"][0]["env"]}
    assert environment["REQUIRE_ROLLBACK_SLOT"] == "true"
    assert environment["PREFLIGHT_ONLY"] == "true"
    assert yaml.safe_dump(job) == text, "preflight mutated the pinned template"


def test_reconciler_rejects_dependency_drift_before_creating_any_job() -> None:
    job = template_job()
    job["spec"]["template"]["spec"]["initContainers"][0]["image"] = (
        "registry.example/candidate@sha256:" + "e" * 64
    )
    with pytest.raises(ValueError, match="dependency image does not match"):
        NodeInstallerReconciler(
            object(),
            object(),
            namespace="gpu-fault-system",
            cluster_name="test-cluster",
            version="0.10.0",
            config_digest="old-config",
            artifact_sha256="1" * 64,
            bundle_sha256="2" * 64,
            template_sha256="3" * 64,
            job_template=job,
            dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
            node_dependency_image=DEPENDENCY_IMAGE,
            node_wheelhouse_sha256=INVENTORY_SHA,
        )


def test_preflight_fails_if_legacy_template_cannot_check_the_old_slot() -> None:
    job = template_job()
    installer = job["spec"]["template"]["spec"]["containers"][0]
    installer["env"] = [
        item for item in installer["env"] if item["name"] != "REQUIRE_ROLLBACK_SLOT"
    ]
    identity = replace(pinned_identity(yaml.safe_dump(job)), require_rollback_slot=True)
    with pytest.raises(RuntimeError, match="no env entry for REQUIRE_ROLLBACK_SLOT"):
        preflight_job(
            job,
            InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge"),
            identity,
            "test-run",
        )
