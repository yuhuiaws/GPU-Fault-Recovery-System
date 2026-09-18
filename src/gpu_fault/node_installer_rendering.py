"""Shared, in-memory node binding for reconciler and deployment preflight Jobs."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import ipaddress
import re
from dataclasses import dataclass
from typing import cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.dcgm_exporter_cadence import DCGM_EXPORTER_COLLECT_INTERVAL_MS
from gpu_fault.gpu_instance_inventory import gpu_instance_inventory

INSTALLER_DIGEST_ANNOTATION = "gpu-fault.io/installer-config-digest"
INSTALLER_ARTIFACT_ANNOTATION = "gpu-fault.io/installer-artifact-sha256"
INSTALLER_BUNDLE_ANNOTATION = "gpu-fault.io/installer-bundle-sha256"
INSTALLER_TEMPLATE_ANNOTATION = "gpu-fault.io/installer-template-sha256"
INSTALLER_JOB_LABEL = "gpu-fault.io/node-installer"
PREFLIGHT_JOB_LABEL = "gpu-fault.io/node-preflight"
TEMPLATE_CONTENT_SHA256_ENV = "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"


def manifest_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(dict[str, object], value)


def manifest_objects(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [manifest_object(item, label) for item in value]


def named_object(value: object, name: str, label: str) -> dict[str, object]:
    matches = [
        item for item in manifest_objects(value, label) if item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one {name}")
    return matches[0]


@dataclass(frozen=True)
class InstallerNode:
    name: str
    uid: str
    address: str
    instance_type: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", self.name):
            raise ValueError("invalid installer node name")
        if not self.uid or any(character.isspace() for character in self.uid):
            raise ValueError("invalid installer node UID")
        ipaddress.ip_address(self.address)
        gpu_instance_inventory(self.instance_type)


@dataclass(frozen=True)
class InstallerIdentity:
    namespace: str
    config_digest: str
    artifact_sha256: str
    bundle_sha256: str
    template_sha256: str
    node_action_keys_secret: str
    deadline_seconds: int
    metrics_url_template: str
    template_content_sha256: str | None = None
    node_dependency_image: str = ""
    node_wheelhouse_sha256: str = ""
    require_rollback_slot: bool = False

    def __post_init__(self) -> None:
        for digest in (
            self.artifact_sha256,
            self.bundle_sha256,
            self.template_sha256,
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("installer identity requires SHA-256 pins")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.config_digest):
            raise ValueError("invalid installer config identity")
        if not 60 <= self.deadline_seconds <= 86400:
            raise ValueError("invalid installer deadline")
        if type(self.require_rollback_slot) is not bool:
            raise ValueError("invalid installer rollback-slot requirement")
        if self.template_content_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.template_content_sha256
        ):
            raise ValueError(
                "installer template content identity requires a SHA-256 pin"
            )
        if not isinstance(self.node_dependency_image, str) or not isinstance(
            self.node_wheelhouse_sha256, str
        ):
            raise ValueError(
                "node dependency identity must declare image and inventory"
            )
        if self.node_dependency_image or self.node_wheelhouse_sha256:
            if not re.fullmatch(
                r"[^\s@]+@sha256:[0-9a-f]{64}", self.node_dependency_image or ""
            ) or not re.fullmatch(r"[0-9a-f]{64}", self.node_wheelhouse_sha256 or ""):
                raise ValueError(
                    "node dependency image and inventory must be digest pinned"
                )


def literal_environment(container: dict[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    seen: set[str] = set()
    for item in manifest_objects(container.get("env", []), "container environment"):
        name = item.get("name")
        if not isinstance(name, str) or name in seen:
            raise ValueError("container environment names must be unique")
        seen.add(name)
        value = item.get("value")
        if isinstance(value, str) and "valueFrom" not in item:
            result[name] = value
    return result


def validate_node_dependencies(
    template: dict[str, object],
    *,
    image: str,
    inventory_sha256: str,
) -> None:
    """Compare the selected Job's delivery path to independently trusted pins."""
    spec = manifest_object(template.get("spec"), "Job spec")
    pod = manifest_object(spec.get("template"), "Job pod template")
    pod_spec = manifest_object(pod.get("spec"), "Job pod spec")
    installer = named_object(pod_spec.get("containers"), "installer", "Job containers")
    environment = literal_environment(installer)
    init = manifest_objects(pod_spec.get("initContainers", []), "Job init containers")
    volumes = manifest_objects(pod_spec.get("volumes", []), "Job volumes")
    if not image and not inventory_sha256:
        if (
            any(item.get("name") == "node-dependencies" for item in init)
            or any(item.get("name") == "node-wheelhouse" for item in volumes)
            or installer.get("envFrom")
            or any(
                item.get("name") == "NODE_WHEELHOUSE_SHA256"
                and (item.get("value") != "" or "valueFrom" in item)
                for item in manifest_objects(
                    installer.get("env", []), "installer environment"
                )
            )
        ):
            raise ValueError(
                "installer template has unexpected offline node dependencies"
            )
        return
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) or not re.fullmatch(
        r"[0-9a-f]{64}", inventory_sha256
    ):
        raise ValueError("node dependency image and inventory must be digest pinned")
    dependency = named_object(init, "node-dependencies", "Job init containers")
    if dependency.get("image") != image:
        raise ValueError(
            "installer template node dependency image does not match identity"
        )
    if environment.get("NODE_WHEELHOUSE_SHA256") != inventory_sha256:
        raise ValueError(
            "installer template node dependency inventory does not match identity"
        )
    volume = named_object(volumes, "node-wheelhouse", "Job volumes")
    if set(volume) != {"name", "emptyDir"}:
        raise ValueError("installer template node dependency volume is not an emptyDir")
    mount = named_object(
        installer.get("volumeMounts"), "node-wheelhouse", "installer volume mounts"
    )
    if (
        mount.get("mountPath") != "/host/run/gpu-fault-node-wheelhouse"
        or mount.get("readOnly") is not True
        or "subPath" in mount
        or "subPathExpr" in mount
    ):
        raise ValueError("installer template node dependency mount is not read-only")
    if dependency.get("volumeMounts") != [
        {"name": "node-wheelhouse", "mountPath": "/wheelhouse"}
    ]:
        raise ValueError("installer template node dependency delivery mount differs")


def validate_installer_template(
    template: dict[str, object], identity: InstallerIdentity
) -> None:
    if template.get("apiVersion") != "batch/v1" or template.get("kind") != "Job":
        raise ValueError("installer template is not a batch/v1 Job")
    validate_node_dependencies(
        template,
        image=identity.node_dependency_image,
        inventory_sha256=identity.node_wheelhouse_sha256,
    )


def load_installer_template(
    template_data: bytes | str,
    *,
    expected_sha256: str | None,
    origin: str,
    identity: InstallerIdentity | None = None,
) -> dict[str, object]:
    """Load exact Job bytes only after checking an independently supplied content pin."""
    raw = template_data if isinstance(template_data, bytes) else template_data.encode()
    if not raw.strip():
        raise RuntimeError(f"{origin} is empty")
    expected = (expected_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(
            f"{TEMPLATE_CONTENT_SHA256_ENV} is required and must be the SHA-256 "
            f"of the installer template text ({origin})"
        )
    if identity is not None and identity.template_content_sha256 != expected:
        raise RuntimeError("installer template content pin disagrees with identity")
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise RuntimeError(
            f"installer template {origin} does not match "
            f"{TEMPLATE_CONTENT_SHA256_ENV}; refusing to load"
        )
    try:
        template = manifest_object(yaml.safe_load(raw), "installer template")
    except (ValueError, yaml.YAMLError):
        raise RuntimeError(
            f"installer template {origin} is not a Job document"
        ) from None
    if template.get("apiVersion") != "batch/v1" or template.get("kind") != "Job":
        raise RuntimeError(f"installer template {origin} is not a Job document")
    if identity is not None:
        validate_installer_template(template, identity)
    return template


def set_installer_environment(
    container: dict[str, object], updates: dict[str, str]
) -> None:
    unfilled = dict(updates)
    seen: set[str] = set()
    for item in manifest_objects(container.get("env"), "installer environment"):
        name = item.get("name")
        if not isinstance(name, str) or name in seen:
            raise ValueError("installer environment names must be unique")
        seen.add(name)
        if name in unfilled:
            item.clear()
            item.update(name=name, value=unfilled.pop(name))
    if unfilled:
        raise RuntimeError(
            "installer job template has no env entry for "
            + ", ".join(sorted(unfilled))
            + "; the pinned template predates the reconciler filling it"
        )


def render_installer_job(
    template: dict[str, object],
    node: InstallerNode,
    identity: InstallerIdentity,
    job_name: str,
) -> dict[str, object]:
    validate_installer_template(template, identity)
    body = copy.deepcopy(template)
    metadata = manifest_object(body.setdefault("metadata", {}), "Job metadata")
    metadata.update(
        name=job_name,
        namespace=identity.namespace,
        annotations={
            INSTALLER_DIGEST_ANNOTATION: identity.config_digest,
            INSTALLER_ARTIFACT_ANNOTATION: identity.artifact_sha256,
            INSTALLER_BUNDLE_ANNOTATION: identity.bundle_sha256,
            INSTALLER_TEMPLATE_ANNOTATION: identity.template_sha256,
        },
        labels={
            INSTALLER_JOB_LABEL: "true",
            "gpu-fault.io/node-uid": node.uid,
            "gpu-fault.io/config-digest": hashlib.sha256(
                identity.config_digest.encode()
            ).hexdigest()[:16],
        },
    )
    spec = manifest_object(body.get("spec"), "Job spec")
    spec["activeDeadlineSeconds"] = identity.deadline_seconds
    pod = manifest_object(spec.get("template"), "Job pod template")
    pod_metadata = manifest_object(pod.setdefault("metadata", {}), "pod metadata")
    pod_metadata["labels"] = dict(manifest_object(metadata["labels"], "Job labels"))
    pod_spec = manifest_object(pod.get("spec"), "Job pod spec")
    pod_spec["nodeName"] = node.name
    node_secret = named_object(pod_spec.get("volumes"), "node-secret", "Job volumes")
    node_secret["secret"] = {
        "secretName": identity.node_action_keys_secret,
        "items": [{"key": node.name, "path": "node-action-secret"}],
    }
    container = named_object(pod_spec.get("containers"), "installer", "Job containers")
    expected_gpus, expected_efa = gpu_instance_inventory(node.instance_type)
    metrics_url = identity.metrics_url_template.replace(
        "{node_name}", node.name
    ).replace("{node_ip}", node.address)
    set_installer_environment(
        container,
        {
            "TARGET_NODE_NAME": node.name,
            "TARGET_NODE_IP": node.address,
            "TARGET_NODE_UID": node.uid,
            "NODE_INSTANCE_TYPE": node.instance_type,
            "EXPECTED_GPU_COUNT": str(expected_gpus),
            "EXPECTED_EFA_DEVICE_COUNT": str(expected_efa),
            "DCGM_METRICS_URL_B64": base64.b64encode(metrics_url.encode()).decode(),
            "DCGM_EXPORTER_INTERVAL_MS": str(DCGM_EXPORTER_COLLECT_INTERVAL_MS),
        },
    )
    return body


def preflight_job(
    template: dict[str, object],
    node: InstallerNode,
    identity: InstallerIdentity,
    run_id: str,
) -> dict[str, object]:
    digest = hashlib.sha256(
        f"{node.name}\0{node.uid}\0{identity.artifact_sha256}\0{run_id}".encode()
    ).hexdigest()[:24]
    body = render_installer_job(
        template, node, identity, f"gpu-fault-preflight-{digest}"
    )
    metadata = manifest_object(body["metadata"], "Job metadata")
    labels = manifest_object(metadata["labels"], "Job labels")
    labels.pop(INSTALLER_JOB_LABEL)
    labels[PREFLIGHT_JOB_LABEL] = "true"
    spec = manifest_object(body["spec"], "Job spec")
    pod = manifest_object(spec["template"], "Job pod template")
    manifest_object(pod["metadata"], "pod metadata")["labels"] = dict(labels)
    pod_spec = manifest_object(pod["spec"], "Job pod spec")
    container = named_object(pod_spec["containers"], "installer", "Job containers")
    set_installer_environment(container, {"PREFLIGHT_ONLY": "true"})
    if identity.require_rollback_slot:
        set_installer_environment(container, {"REQUIRE_ROLLBACK_SLOT": "true"})
    root = named_object(
        container.get("volumeMounts"), "host-root", "installer volume mounts"
    )
    if root.get("mountPath") != "/host":
        raise ValueError("preflight host root mount differs")
    root["readOnly"] = True
    return body


def configure_node_dependencies(
    job: dict[str, object], image: str, inventory_sha256: str
) -> None:
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) or not re.fullmatch(
        r"[0-9a-f]{64}", inventory_sha256
    ):
        raise ValueError("node dependency image and inventory must be digest pinned")
    spec = manifest_object(job.get("spec"), "Job spec")
    pod = manifest_object(spec.get("template"), "Job pod template")
    pod_spec = manifest_object(pod.get("spec"), "Job pod spec")
    installer = named_object(pod_spec.get("containers"), "installer", "Job containers")
    pod_spec["automountServiceAccountToken"] = False
    security = manifest_object(
        pod_spec.setdefault("securityContext", {}), "pod security context"
    )
    if security.get("fsGroup", 65534) != 65534:
        raise ValueError("node dependency volume group conflicts with the pod")
    security.update(fsGroup=65534, fsGroupChangePolicy="OnRootMismatch")
    set_installer_environment(installer, {"NODE_WHEELHOUSE_SHA256": inventory_sha256})
    volumes = manifest_objects(pod_spec.get("volumes"), "Job volumes")
    if any(item.get("name") == "node-wheelhouse" for item in volumes):
        raise ValueError("node wheelhouse volume already exists")
    pod_spec["volumes"] = [
        *volumes,
        {
            "name": "node-wheelhouse",
            "emptyDir": {"sizeLimit": "512Mi"},
        },
    ]
    mounts = manifest_objects(installer.get("volumeMounts"), "installer volume mounts")
    installer["volumeMounts"] = [
        *mounts,
        {
            "name": "node-wheelhouse",
            "mountPath": "/host/run/gpu-fault-node-wheelhouse",
            "readOnly": True,
        },
    ]
    init = manifest_objects(pod_spec.get("initContainers", []), "Job init containers")
    if any(item.get("name") == "node-dependencies" for item in init):
        raise ValueError("node dependency init container already exists")
    pod_spec["initContainers"] = [
        *init,
        {
            "name": "node-dependencies",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "command": [
                "/bin/sh",
                "-ec",
                "cp /opt/gpu-fault/wheelhouse/* /wheelhouse/",
            ],
            "securityContext": {
                "runAsUser": 65534,
                "runAsNonRoot": True,
                "readOnlyRootFilesystem": True,
                "allowPrivilegeEscalation": False,
                "capabilities": {"drop": ["ALL"]},
            },
            "resources": {
                "requests": {
                    "cpu": "50m",
                    "memory": "64Mi",
                    "ephemeral-storage": "128Mi",
                },
                "limits": {
                    "cpu": "500m",
                    "memory": "128Mi",
                    "ephemeral-storage": "1Gi",
                },
            },
            "volumeMounts": [{"name": "node-wheelhouse", "mountPath": "/wheelhouse"}],
        },
    ]
