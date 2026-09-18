"""Capture the independently trusted installer identity of one live cluster."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpu_fault.node_installer_rendering import (
    TEMPLATE_CONTENT_SHA256_ENV,
    literal_environment,
    load_installer_template,
    manifest_object,
    named_object,
    validate_node_dependencies,
)
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_images import (
    node_dependency_environment,
    previous_release_schema_version,
)

if TYPE_CHECKING:
    from gpu_fault_release.rollout import RegionalRelease


@dataclass(frozen=True)
class InstallerSnapshot:
    template: str
    template_sha256: str | None
    template_content_sha256: str | None
    bundle: str
    installer_image: str
    bundle_pin: str | None


def capture_installer_snapshot(
    release: RegionalRelease,
    target: ClusterTarget,
    live_state: Mapping[str, object],
) -> InstallerSnapshot:
    deployment = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.GPU_RECONCILER_DEPLOYMENT,
        )
    )
    try:
        spec = manifest_object(deployment.get("spec"), "Reconciler spec")
        pod = manifest_object(spec.get("template"), "Reconciler pod template")
        pod_spec = manifest_object(pod.get("spec"), "Reconciler pod spec")
        container = named_object(
            pod_spec.get("containers"), "reconciler", "Reconciler containers"
        )
        environment = literal_environment(container)
        volume = named_object(
            pod_spec.get("volumes"), "installer-template", "Reconciler volumes"
        )
        template = manifest_object(
            volume.get("configMap"), "Reconciler template ConfigMap"
        ).get("name")
        if not isinstance(template, str) or not template:
            raise ValueError("Reconciler template ConfigMap name is missing")
        if (
            environment.get("GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP", template)
            != template
        ):
            raise ValueError("Reconciler template reference disagrees with its volume")
        template_content_sha256 = environment.get(TEMPLATE_CONTENT_SHA256_ENV)
        template_value = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                template,
            )
        )
        template_text = manifest_object(
            template_value.get("data"), "installer template ConfigMap data"
        ).get("job.yaml")
        if not isinstance(template_text, str):
            raise ValueError("installer template ConfigMap has no job.yaml")
        job = load_installer_template(
            template_text,
            expected_sha256=template_content_sha256,
            origin=f"{template}/job.yaml",
        )
        dependencies = node_dependency_environment(
            identity=live_state.get("node_dependencies"),
            required=previous_release_schema_version(live_state) >= 4,
        )
        validate_node_dependencies(
            job,
            image=dependencies["GPU_FAULT_NODE_DEPENDENCY_IMAGE"],
            inventory_sha256=dependencies["GPU_FAULT_NODE_WHEELHOUSE_SHA256"],
        )
        job_spec = manifest_object(job.get("spec"), "Job spec")
        job_pod = manifest_object(job_spec.get("template"), "Job pod template")
        job_pod_spec = manifest_object(job_pod.get("spec"), "Job pod spec")
        installer = named_object(
            job_pod_spec.get("containers"), "installer", "Job containers"
        )
        installer_image = installer.get("image")
        if not isinstance(installer_image, str) or not installer_image:
            raise ValueError("installer template has no installer image")
        bundle_volume = named_object(
            job_pod_spec.get("volumes"), "installer", "Job volumes"
        )
        bundle = manifest_object(
            bundle_volume.get("configMap"), "installer bundle ConfigMap"
        ).get("name")
        if not isinstance(bundle, str) or not bundle:
            raise ValueError("installer template has no bundle ConfigMap")
    except (RuntimeError, ValueError) as exc:
        raise ReleaseError(
            f"{target.cluster_id} previous installer template identity is invalid: {exc}"
        ) from exc
    return InstallerSnapshot(
        template=template,
        template_sha256=environment.get("GPU_FAULT_INSTALLER_TEMPLATE_SHA256"),
        template_content_sha256=template_content_sha256,
        bundle=bundle,
        installer_image=installer_image,
        bundle_pin=environment.get("GPU_FAULT_INSTALLER_BUNDLE_SHA256"),
    )
