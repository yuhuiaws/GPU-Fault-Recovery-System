from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.config import AdminConfig
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import repository_root
from gpu_fault_release.regional_notifications import notification_digest
from gpu_fault_release.regional_release_config import (
    ClusterTarget,
    ReleaseError,
    render_nlb_manifest,
)

from gpu_fault.dcgm_exporter_cadence import (
    DCGM_EXPORTER_COLLECT_INTERVAL_MS,
    DCGM_EXPORTER_COLLECT_INTERVAL_PLACEHOLDER,
)
from gpu_fault.gpu_instance_inventory import GPU_INSTANCE_INVENTORY as INSTANCE_TYPES

ROOT = repository_root()
DEFAULT_RUNTIME_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"
DEFAULT_DCGM_EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04"
#: What the checked-in DaemonSet carries in place of the instance-type list.
SUPPORTED_INSTANCE_TYPES_PLACEHOLDER = "REPLACE_WITH_SUPPORTED_INSTANCE_TYPES"


def _supported_instance_types() -> str:
    """The exporter's node-affinity list, as a YAML flow sequence.

    Read from the same inventory the node installer sizes GPU/EFA counts
    from rather than copied: the two must never disagree, because a GPU type
    the installer supports but the exporter does not schedule on can never
    finish an install (its ``dcgm_ready`` check has nothing to scrape), and the
    reconciler retries the Job every 300 s forever. Each type is emitted twice,
    with and without the ``ml.`` prefix, because HyperPod labels its nodes
    ``ml.<type>`` while a self-managed node pool carries the bare EC2 type;
    ``gpu_instance_inventory()`` accepts both for the same reason.
    """

    names = sorted(INSTANCE_TYPES)
    return ", ".join(f'"ml.{name}", "{name}"' for name in names)


def render_dcgm_exporter_manifest(*, namespace: str, image: str) -> str:
    """The one exporter DaemonSet text, for the plan payload and for the apply.

    Both used to substitute for themselves and the payload renderer knew only
    about namespace and image, so the plan an approver read carried a literal
    ``REPLACE_WITH_SUPPORTED_INSTANCE_TYPES`` where the applied DaemonSet
    carried the real instance types -- the applied artifact was not the planned
    one, which is the whole promise of the plan/apply gate.

    The collect period (``-c``) is rendered from
    :data:`gpu_fault.dcgm_exporter_cadence.DCGM_EXPORTER_COLLECT_INTERVAL_MS`,
    the same constant the node installer reconciler hands every install Job so
    the host collector learns the period of an ``existing`` exporter.
    """

    text = (ROOT / "deploy/dataplane/hyperpod-dcgm-exporter.yaml").read_text(
        encoding="utf-8"
    )
    text = (
        text.replace(
            "namespace: gpu-fault-system",
            f"namespace: {namespace}",
        )
        .replace(DEFAULT_DCGM_EXPORTER_IMAGE, image)
        .replace(
            SUPPORTED_INSTANCE_TYPES_PLACEHOLDER,
            _supported_instance_types(),
        )
        .replace(
            DCGM_EXPORTER_COLLECT_INTERVAL_PLACEHOLDER,
            str(DCGM_EXPORTER_COLLECT_INTERVAL_MS),
        )
    )
    for placeholder in (
        SUPPORTED_INSTANCE_TYPES_PLACEHOLDER,
        DCGM_EXPORTER_COLLECT_INTERVAL_PLACEHOLDER,
    ):
        if placeholder in text:
            raise ReleaseError(f"DCGM exporter manifest still carries {placeholder}")
    return text


#: The per-GPU-cluster metrics collector (data-plane review F7) and the
#: placeholders it carries. Every value is substituted from the release config
#: or the cluster target; none has a default, because a collector that keeps a
#: placeholder starts, scrapes and fails every remote write in silence.
DATAPLANE_ADOT_MANIFEST = "deploy/dataplane/adot-dataplane.yaml"
DATAPLANE_ADOT_DEPLOYMENT = "gpu-fault-adot-dataplane"
ADOT_IMAGE_PLACEHOLDER = "REPLACE_WITH_ADOT_IMAGE"
ADOT_IRSA_ROLE_ARN_PLACEHOLDER = "REPLACE_WITH_ADOT_IRSA_ROLE_ARN"
AMP_WORKSPACE_ID_PLACEHOLDER = "REPLACE_WITH_AMP_WORKSPACE_ID"
GPU_CLUSTER_ID_PLACEHOLDER = "REPLACE_WITH_GPU_CLUSTER_ID"
AWS_REGION_PLACEHOLDER = "REPLACE_WITH_AWS_REGION"


def render_dataplane_adot_manifest(
    *,
    namespace: str,
    image: str,
    region: str,
    amp_workspace_id: str,
    irsa_role_arn: str,
    cluster_id: str,
) -> str:
    """The data-plane ADOT collector text for one GPU cluster.

    One renderer for the plan payload and for the apply, like the DCGM
    exporter, so an approver reads the text that is about to be applied. The
    collector scrapes the data-plane Pods of THIS cluster and remote-writes to
    the SAME AMP workspace the control-plane collector uses, with ``region``
    and ``gpu_cluster`` stamped on every series so the alerts'
    ``by (control_plane_cluster, region)`` grouping keeps working.

    Fails closed twice: on an empty input (a blank IRSA role ARN would render a
    ServiceAccount annotation that IRSA ignores, and the collector would run
    without credentials) and on any ``REPLACE_WITH`` that survives
    substitution.
    """

    inputs = {
        "namespace": namespace,
        "image": image,
        "region": region,
        "amp_workspace_id": amp_workspace_id,
        "irsa_role_arn": irsa_role_arn,
        "cluster_id": cluster_id,
    }
    empty = sorted(name for name, value in inputs.items() if not str(value).strip())
    if empty:
        raise ReleaseError(
            "data-plane ADOT collector cannot be rendered without: " + ", ".join(empty)
        )
    text = (ROOT / DATAPLANE_ADOT_MANIFEST).read_text(encoding="utf-8")
    for source, destination in (
        ("namespace: gpu-fault-system", f"namespace: {namespace}"),
        # The receiver's pod discovery is pinned to the system namespace too.
        ("names: [gpu-fault-system]", f"names: [{namespace}]"),
        (ADOT_IMAGE_PLACEHOLDER, image),
        (ADOT_IRSA_ROLE_ARN_PLACEHOLDER, irsa_role_arn),
        (AMP_WORKSPACE_ID_PLACEHOLDER, amp_workspace_id),
        (GPU_CLUSTER_ID_PLACEHOLDER, cluster_id),
        (AWS_REGION_PLACEHOLDER, region),
    ):
        text = text.replace(source, destination)
    leftover = sorted(set(re.findall(r"REPLACE_WITH_[A-Z0-9_]+", text)))
    if leftover:
        raise ReleaseError(
            "data-plane ADOT collector manifest still carries " + ", ".join(leftover)
        )
    return text


def dataplane_adot_skip_reason(release: Any, target: ClusterTarget) -> str | None:
    """Why the collector cannot be applied to ``target``, or ``None``.

    The two prerequisites live outside the repository: the site's AMP
    workspace and a per-cluster IAM role with ``aps:RemoteWrite`` on it,
    trusted by the cluster's OIDC issuer for the collector's ServiceAccount.
    A site that has not created them yet must still be able to release
    everything else, so their absence skips the component instead of failing
    the release -- but the caller prints the reason, because the alternative
    is exactly the silent, inert watcher alerts this component exists to end.
    """

    if not release.config.health.amp_workspace_id:
        return "health.amp_workspace_id is not configured"
    if not target.adot_irsa_role_arn:
        return (
            f"cluster target {target.cluster_id} has no adot_irsa_role_arn "
            "(an IAM role with aps:RemoteWrite on the AMP workspace, trusted by "
            "this cluster's OIDC issuer for gpu-fault-system/"
            f"{DATAPLANE_ADOT_DEPLOYMENT})"
        )
    return None


def render_dataplane_adot_for_target(
    release: Any,
    target: ClusterTarget,
    *,
    image: str | None = None,
) -> str | None:
    """The rendered collector for ``target``, or ``None`` when it is skipped."""

    workspace_id = release.config.health.amp_workspace_id
    irsa_role_arn = target.adot_irsa_role_arn
    if dataplane_adot_skip_reason(release, target) or not irsa_role_arn:
        return None
    return render_dataplane_adot_manifest(
        namespace=release.config.namespace,
        image=image or release.adot_image,
        region=target.region,
        amp_workspace_id=str(workspace_id),
        irsa_role_arn=irsa_role_arn,
        cluster_id=target.cluster_id,
    )


def _number_text(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def admin_config_renderer_environment(
    admin_config: AdminConfig,
) -> dict[str, str]:
    capacity = admin_config.capacity
    remediation = capacity.remediation
    spool = capacity.telemetry_spool
    processor = admin_config.processor
    workflow = admin_config.workflow
    notification = admin_config.notification_delivery
    evidence = admin_config.evidence
    return {
        "GPU_FAULT_CONTROL_WORKER_REPLICAS": str(capacity.control_worker_replicas),
        # The declared topology travels with the release so the live state can
        # be read back on rollback and the runtime can check its own reserve.
        "GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT": str(
            capacity.largest_cluster_node_count
        ),
        "GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT": str(capacity.managed_node_count),
        "GPU_FAULT_TELEMETRY_SPOOL": str(spool.enabled).lower(),
        "GPU_FAULT_TELEMETRY_SPOOL_REPLICAS": str(spool.replicas),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": str(remediation.max_active_region),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": str(
            remediation.max_active_per_cluster
        ),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": str(
            remediation.max_active_per_node
        ),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": str(
            remediation.max_active_per_failure_domain
        ),
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": str(
            remediation.max_active_per_resource_class
        ),
        "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH": str(processor.max_queue_depth),
        "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": str(
            processor.max_cluster_queue_depth
        ),
        "GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH": str(
            admin_config.fault_reserved_queue_depth()
        ),
        "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH": str(
            admin_config.fault_reserved_cluster_depth()
        ),
        "GPU_FAULT_PROCESSOR_GLOBAL_ADMISSION_GUARD": str(
            min(256, processor.max_queue_depth)
        ),
        "GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS": str(processor.retry_after_seconds),
        "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS": str(
            processor.retry_backoff_seconds
        ),
        "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS": str(
            processor.retry_backoff_max_seconds
        ),
        "GPU_FAULT_PROCESSOR_RETRYABLE_RESPONSE_MAX_AGE_SECONDS": str(
            max(300, processor.retry_backoff_max_seconds)
        ),
        "GPU_FAULT_PROCESSOR_COMPLETED_RETENTION_SECONDS": str(
            processor.completed_retention_seconds
        ),
        "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS": _number_text(
            workflow.poll_interval_seconds
        ),
        "GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS": str(workflow.dispatcher_workers),
        "GPU_FAULT_NOTIFICATION_BATCH_SIZE": str(notification.batch_size),
        "GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS": str(notification.max_attempts),
        "GPU_FAULT_EVIDENCE_RETENTION_HOURS": str(evidence.retention_hours),
        "GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE": str(evidence.max_records_per_node),
    }


def _documents(text: str) -> list[dict[str, Any]]:
    return [
        document for document in yaml.safe_load_all(text) if isinstance(document, dict)
    ]


# In execution order; deploy/hyperpod/deploy.sh and the regional schema tool
# run the same three (F-J3 three-step index method).
SCHEMA_JOB_MANIFESTS = (
    "deploy/migrations/postgres-index-build-job.yaml",
    "deploy/migrations/postgres-schema-ensure-job.yaml",
    "deploy/migrations/postgres-schema-preflight-job.yaml",
)


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def render_release_payload(release: Any) -> dict[str, Any]:
    """The manifests and digests a release delivers, as one JSON-able payload.

    When the release carries an ``approved_manifest_digest`` -- the digest the
    plan was reviewed and approved against -- the freshly rendered payload is
    verified against it. Rendering reads the working tree, so re-reading it at
    apply opens a TOCTOU window: the tree can change after the plan is approved
    and before it is applied. Failing closed on a mismatch means the applied
    artifact can only ever be the one that was planned.
    """

    payload = _render_release_payload(release)
    approved = str(getattr(release, "approved_manifest_digest", "") or "")
    if approved:
        actual = _payload_digest(payload)
        if actual != approved:
            raise ReleaseError(
                "release manifests changed between plan and apply: the working "
                f"tree renders {actual} but the approved plan pinned {approved}"
            )
    return payload


def rendered_release_manifest_sha256(release: Any) -> str:
    return _payload_digest(render_release_payload(release))


def render_cpu_manifest_text(release: Any, filename: str, text: str) -> str:
    """One generated CPU manifest with every release-time placeholder filled.

    Mirrors the ``sed`` in ``apply-control-plane-role-split.sh``; the two must
    know the same placeholders. Fails closed on anything left over: a
    ``REPLACE_WITH_*`` that reaches a ConfigMap is a literal the runtime
    rejects (the archive URI, for one) or silently misroutes.
    """

    config = release.config
    replacements = {
        "gpu-fault-control-plane-wheel-0100": release.wheel_cm,
        "namespace: gpu-fault-system": f"namespace: {config.namespace}",
        "REPLACE_WITH_AWS_REGION": config.aws_region,
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION": (config.runtime_profile_version),
        DEFAULT_RUNTIME_IMAGE: release.runtime_image,
        "GPU_FAULT_ALLOW_EMAIL: 'true'": (
            f"GPU_FAULT_ALLOW_EMAIL: '{str(config.notifications.allow_email).lower()}'"
        ),
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL: 'false'": (
            "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL: "
            f"'{str(config.notifications.acknowledge_external_alert_channel).lower()}'"
        ),
    }
    for source, destination in replacements.items():
        text = text.replace(source, destination)
    text = text.replace(
        "gpu-fault.io/artifact-sha256: "
        "209840015cc3057e191931f113d35dff733cf1b7483d68dd6b6b26a7de9a112b",
        f"gpu-fault.io/artifact-sha256: {release.wheel_sha}",
    )
    if "REPLACE_WITH" in text:
        raise ReleaseError(f"{filename} still contains a placeholder")
    return text


def _render_release_payload(release: Any) -> dict[str, Any]:
    config = release.config
    generated = ROOT / "deploy/control-plane/regional/generated"
    manifest_names = [
        item.strip()
        for item in (generated / "manifest-list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if item.strip() and not item.startswith("#")
    ]
    cpu_documents: dict[str, list[dict[str, Any]]] = {}
    for filename in manifest_names:
        text = render_cpu_manifest_text(
            release,
            filename,
            (generated / filename).read_text(encoding="utf-8"),
        )
        cpu_documents[filename] = _documents(text)

    gpu_documents = {
        target.cluster_id: {
            deployment: _documents(text)
            for deployment, text in render_gpu_rollout_manifests(
                release,
                target,
                release.executor_wheel_cm,
            )
        }
        for target in config.clusters
    }
    # The schema stage is three Jobs (index-build, ensure, preflight; F-J3),
    # run in that order by deploy/control-plane/tools/ensure-postgres-schema.sh;
    # the release digest covers all three manifests.
    schema_documents: list[dict[str, Any]] = []
    for manifest in SCHEMA_JOB_MANIFESTS:
        schema_text = (ROOT / manifest).read_text(encoding="utf-8")
        for source, destination in {
            "namespace: gpu-fault-system": f"namespace: {config.namespace}",
            "REPLACE_WITH_WHEEL_CONFIGMAP": release.wheel_cm,
            DEFAULT_RUNTIME_IMAGE: release.runtime_image,
        }.items():
            schema_text = schema_text.replace(source, destination)
        schema_documents.extend(_documents(schema_text))
    refresh_text = (
        ROOT / "deploy/control-plane/regional/aurora-credential-refresh.yaml"
    ).read_text(encoding="utf-8")
    refresh_text = refresh_text.replace(
        "namespace: gpu-fault-system",
        f"namespace: {config.namespace}",
    ).replace(
        DEFAULT_RUNTIME_IMAGE,
        release.runtime_image,
    )

    nlb_documents = (
        _documents(
            render_nlb_manifest(
                config,
                (
                    ROOT
                    / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
                ).read_text(encoding="utf-8"),
            )
        )
        if config.nlb
        else []
    )
    dcgm_text = render_dcgm_exporter_manifest(
        namespace=config.namespace,
        image=release.dcgm_exporter_image,
    )
    observability_text = (
        ROOT / "deploy/observability/adot-control-plane.yaml"
    ).read_text(encoding="utf-8")
    observability_text = observability_text.replace(
        "namespace: gpu-fault-system",
        f"namespace: {config.namespace}",
    ).replace(
        "REPLACE_WITH_ADOT_IMAGE",
        release.adot_image,
    )
    payload = {
        "cpu": cpu_documents,
        "admin_config": config.admin_config.as_dict(),
        "gpu": gpu_documents,
        "schema": schema_documents,
        "aurora_refresh": _documents(refresh_text),
        "nlb": nlb_documents,
        "dcgm": _documents(dcgm_text),
        "observability": _documents(observability_text),
        # Per GPU cluster; an empty list is a cluster the release will skip
        # (no AMP workspace or no IRSA role for the collector yet), so the
        # plan an approver reads says which clusters get a scrape path.
        "dataplane_observability": {
            target.cluster_id: (
                _documents(rendered)
                if (rendered := render_dataplane_adot_for_target(release, target))
                else []
            )
            for target in config.clusters
        },
        "reconciler": {
            target.cluster_id: {
                "wheel_config_map": release.executor_wheel_cm,
                "bundle_config_map": release.bundle_cm,
                "artifact_sha256": release.node_wheel_sha,
                "bundle_sha256": release.bundle_sha,
                "template_source_sha256": release.node_template_sha,
                "config_digest": config.agent_config_digest,
                "runtime_profile_version": config.runtime_profile_version,
                "runtime_image": release.runtime_image,
                "node_installer_image": release.node_installer_image,
            }
            for target in config.clusters
        },
    }
    return payload


def stamp_gpu_deployments(
    release: Any,
    text: str,
    *,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
    runtime_image: str | None = None,
) -> str:
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or release.config.component_digests.get("executor")
        or artifact_sha
    )
    documents = list(yaml.safe_load_all(text))
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != "Deployment":
            continue
        annotations = (
            document.setdefault("spec", {})
            .setdefault("template", {})
            .setdefault("metadata", {})
            .setdefault("annotations", {})
        )
        annotations.update(
            {
                "gpu-fault.io/artifact-sha256": artifact_sha,
                "gpu-fault.io/release-rollout": release.release_id,
                "gpu-fault.io/release-sha256": artifact_sha,
                "gpu-fault.io/release-wheel-sha256": artifact_sha,
                "gpu-fault.io/executor-wheel-sha256": artifact_sha,
                "gpu-fault.io/executor-compatibility-digest": compatibility_digest,
                "gpu-fault.io/runtime-image": (runtime_image or release.runtime_image),
            }
        )
    return yaml.safe_dump_all(
        documents,
        sort_keys=False,
        width=72,
    )


def build_cpu_apply_environment(
    release: Any,
    *,
    finalize: bool,
    runtime_profile_version: str | None = None,
) -> dict[str, str]:
    config = release.config
    return {
        **os.environ,
        **admin_config_renderer_environment(config.admin_config),
        # Present only when site.yaml spec.retention turns archive-first
        # deletion on; the renderer forwards these to the control-worker.
        **config.retention.environment(),
        # The alert channel and its SNS topic; every role runs the notifier
        # guard at startup, so the renderer places these on all three.
        **config.notification_environment(),
        "KUBECONFIG": config.cpu_kubeconfig,
        "GPU_FAULT_AWS_REGION": config.aws_region,
        "GPU_FAULT_NAMESPACE": config.namespace,
        "GPU_FAULT_WHEEL_CONFIGMAP": release.wheel_cm,
        "GPU_FAULT_WHEEL_SHA256": release.wheel_sha,
        "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": release.node_wheel_sha,
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": (
            config.component_digests.get("node_runtime") or release.node_wheel_sha
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": (
            release.executor_wheel_sha
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
            config.component_digests.get("executor") or release.executor_wheel_sha
        ),
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config.agent_config_digest,
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": (
            runtime_profile_version or config.runtime_profile_version
        ),
        "GPU_FAULT_ALLOW_EMAIL": str(config.notifications.allow_email).lower(),
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": str(
            config.notifications.acknowledge_external_alert_channel
        ).lower(),
        "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": notification_digest(
            config.notifications
        ),
        "GPU_FAULT_ADMIN_CONFIG_SHA256": release.admin_config_digest,
        "GPU_FAULT_ADMIN_CONFIG_INGRESS_SHA256": (
            release.admin_config_role_digests["ingress"]
        ),
        "GPU_FAULT_ADMIN_CONFIG_WORKER_SHA256": (
            release.admin_config_role_digests["worker"]
        ),
        "GPU_FAULT_ADMIN_CONFIG_SPOOL_SHA256": (
            release.admin_config_role_digests["spool"]
        ),
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": str(config.agent_protocol_version),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": str(
            config.executor_protocol_version
        ),
        "GPU_FAULT_FINALIZE_AGENT_PIN": str(finalize).lower(),
        "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": str(finalize).lower(),
    }


def render_gpu_rollout_manifests(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    deployment_names: frozenset[str] | None = None,
    runtime_image: str | None = None,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
) -> list[tuple[str, str]]:
    config = release.config
    profile_version = runtime_profile_version or config.runtime_profile_version
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or config.component_digests.get("executor")
        or artifact_sha
    )
    replacements = {
        "gpu-fault-executor-wheel-0100": wheel_cm,
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl": (
            # A rollback captures the wheel by its ConfigMap binaryData key,
            # which is the xz-compressed storage name (`<wheel>.whl.xz`); the
            # runtime spec and the Reconciler want the wheel filename itself.
            (
                executor_wheel_filename or release.config.executor_wheel.name
            ).removesuffix(".xz")
        ),
        "namespace: gpu-fault-system": f"namespace: {config.namespace}",
        DEFAULT_RUNTIME_IMAGE: runtime_image or release.runtime_image,
        "REPLACE_WITH_AWS_REGION": target.region,
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION": profile_version,
        "REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN": target.executor_irsa_role_arn,
        "REPLACE_WITH_EXECUTOR_ARTIFACT_SHA256": artifact_sha,
        "REPLACE_WITH_EXECUTOR_COMPATIBILITY_DIGEST": compatibility_digest,
    }
    rendered = []
    for filename, deployment in inventory.GPU_ROLLOUT_DEPLOYMENTS:
        if deployment_names is not None and deployment not in deployment_names:
            continue
        text = (ROOT / "deploy/dataplane" / filename).read_text(encoding="utf-8")
        for source, destination in replacements.items():
            text = text.replace(source, destination)
        if "REPLACE_WITH" in text:
            raise ReleaseError(f"{filename} still contains a placeholder")
        rendered.append(
            (
                deployment,
                stamp_gpu_deployments(
                    release,
                    text,
                    executor_artifact_sha=artifact_sha,
                    executor_compatibility_digest=compatibility_digest,
                    runtime_image=runtime_image,
                ),
            )
        )
    return rendered


def build_reconciler_environment(
    release: Any,
    target: ClusterTarget,
    *,
    wheel_cm: str,
    bundle_cm: str,
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    node_compatibility_digest: str | None = None,
    bundle_sha256: str | None = None,
    template_sha256: str | None = None,
    template_config_map: str | None = None,
    allowed_node_names: tuple[str, ...] | None = None,
    max_unavailable: int | None = None,
    sync_registry: bool = True,
    runtime_image: str | None = None,
    node_installer_image: str | None = None,
) -> dict[str, str]:
    config = release.config
    environment = {
        **os.environ,
        "GPU_FAULT_KUBECTL_CONTEXT": target.context,
        "GPU_FAULT_NAMESPACE": config.namespace,
        "GPU_FAULT_CLUSTER_ID": target.cluster_id,
        "GPU_FAULT_HYPERPOD_CLUSTER": target.hyperpod_cluster_name,
        "GPU_FAULT_INSTALLER_CONFIG_MAP": bundle_cm,
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": config_digest,
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": artifact_sha,
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": (bundle_sha256 or release.bundle_sha),
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": (
            template_sha256 or release.node_template_sha
        ),
        "GPU_FAULT_NODE_COMPATIBILITY_DIGEST": (
            node_compatibility_digest
            or config.component_digests.get("node_runtime")
            or artifact_sha
        ),
        "GPU_FAULT_WHEEL_CONFIG_MAP": wheel_cm,
        "GPU_FAULT_EXECUTOR_WHEEL_FILENAME": (
            # A rollback hands the wheel's ConfigMap binaryData key, which is the
            # xz-compressed storage name (`<wheel>.whl.xz`); the Reconciler wants
            # the wheel filename and re-derives the `.xz` key itself.
            (executor_wheel_filename or config.executor_wheel.name).removesuffix(".xz")
        ),
        "GPU_FAULT_RUNTIME_IMAGE": runtime_image or release.runtime_image,
        "GPU_FAULT_NODE_INSTALLER_IMAGE": (
            node_installer_image or release.node_installer_image
        ),
        "GPU_FAULT_INSTALLER_ALLOWED_NODES": (
            "*" if allowed_node_names is None else ",".join(sorted(allowed_node_names))
        ),
        "GPU_FAULT_INSTALLER_MAX_UNAVAILABLE": str(
            max_unavailable or config.upgrade_max_unavailable
        ),
        "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": str(sync_registry).lower(),
        "GPU_FAULT_RUNTIME_PROFILE": (
            runtime_profile_version or config.runtime_profile_version
        ),
    }
    if target.fleet_master_file:
        environment["GPU_FAULT_FLEET_MASTER_FILE"] = target.fleet_master_file
        environment["GPU_FAULT_CONTROL_PLANE_KUBECONFIG"] = config.cpu_kubeconfig
    if template_config_map:
        environment["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] = template_config_map
    return environment


# The reconciler deploy script's expensive, idempotent products -- the node
# action key Secret (derived from the fleet master and mirrored to the control
# plane) and the rendered template ConfigMap -- are made once per release and
# handed to its later runs as hints through these variables. The script owns
# the check that the hints are still true (live node set, live ConfigMap
# content); this side owns the scope: same release, same render inputs.
INSTALLER_PRODUCTS_STATE_KEY = "node_installer_products"
INSTALLER_REUSE_NODE_SET_ENV = "GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256"
INSTALLER_REUSE_TEMPLATE_ENV = "GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP"
INSTALLER_PRODUCTS_FILE_ENV = "GPU_FAULT_RECONCILER_PRODUCTS_FILE"
# Every input the script feeds into the two products, and nothing else: no
# credential is among them, so the digest of their values can sit in the
# release state ConfigMap. The master file appears as a path, not as content.
INSTALLER_PRODUCT_INPUT_ENV = (
    "GPU_FAULT_KUBECTL_CONTEXT",
    "GPU_FAULT_NAMESPACE",
    "GPU_FAULT_CLUSTER_ID",
    "GPU_FAULT_HYPERPOD_CLUSTER",
    "GPU_FAULT_VERSION",
    "GPU_FAULT_RUNTIME_PROFILE",
    "GPU_FAULT_INSTALLER_CONFIG_MAP",
    "GPU_FAULT_INSTALLER_CONFIG_DIGEST",
    "GPU_FAULT_INSTALLER_ARTIFACT_SHA256",
    "GPU_FAULT_INSTALLER_BUNDLE_SHA256",
    "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
    # An explicit override decides which template is used, so a run that names
    # one (a rollback's steady template) shares no products with a run that
    # renders; the rollback pays the full path, the join and upgrade do not.
    "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP",
    "GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS",
    "GPU_FAULT_NODE_COMPATIBILITY_DIGEST",
    "GPU_FAULT_NODE_INSTALLER_IMAGE",
    "GPU_FAULT_NODE_ACTION_KEYS_SECRET",
    "GPU_FAULT_DCGM_METRICS_URL",
    "GPU_FAULT_REQUIRE_ROLLBACK_SLOT",
    "GPU_FAULT_FLEET_MASTER_FILE",
)
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
CONFIG_MAP_NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$")


def installer_product_inputs_digest(environment: dict[str, str]) -> str:
    """Digest of the render inputs the reusable products are a function of."""

    inputs = {key: environment.get(key, "") for key in INSTALLER_PRODUCT_INPUT_ENV}
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recorded_installer_products(
    release: Any,
    target: ClusterTarget,
    *,
    inputs_digest: str,
) -> dict[str, Any] | None:
    """What an earlier run of *this* release produced for *these* inputs.

    A record from another release, or from the same release with different
    inputs (a rollback's steady template, a changed image), is not a hint.
    """

    state = getattr(release, "state", None)
    if not isinstance(state, dict):
        return None
    record = (state.get(INSTALLER_PRODUCTS_STATE_KEY) or {}).get(target.cluster_id)
    if not isinstance(record, dict):
        return None
    if record.get("release_id") != release.release_id:
        return None
    if record.get("inputs_sha256") != inputs_digest:
        return None
    if not SHA256_HEX.fullmatch(str(record.get("node_set_sha256") or "")):
        return None
    return record


def record_installer_products(
    release: Any,
    target: ClusterTarget,
    *,
    inputs_digest: str,
    products_path: str,
) -> dict[str, Any] | None:
    """Keep the script's product report for the next run of this release.

    The report is advisory, so a missing or malformed one (a fake runner, a
    script that stopped before writing it) records nothing and raises nothing;
    the next run simply pays the full path. Records of other releases are
    dropped: they can never be hints again, and the state is bounded.
    """

    state = getattr(release, "state", None)
    if not isinstance(state, dict):
        return None
    try:
        with open(products_path, encoding="utf-8") as handle:
            products = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(products, dict):
        return None
    node_set = str(products.get("node_set_sha256") or "")
    template = str(products.get("template_config_map") or "")
    content = str(products.get("template_content_sha256") or "")
    if not SHA256_HEX.fullmatch(node_set) or not SHA256_HEX.fullmatch(content):
        return None
    if not CONFIG_MAP_NAME.fullmatch(template):
        return None
    record = {
        "release_id": release.release_id,
        "inputs_sha256": inputs_digest,
        "node_set_sha256": node_set,
        "template_config_map": template,
        "template_content_sha256": content,
        "node_action_keys_provisioned": bool(
            products.get("node_action_keys_provisioned")
        ),
        "template_rendered": bool(products.get("template_rendered")),
    }
    existing = state.get(INSTALLER_PRODUCTS_STATE_KEY)
    kept = {
        cluster_id: item
        for cluster_id, item in (existing.items() if isinstance(existing, dict) else ())
        if isinstance(item, dict) and item.get("release_id") == release.release_id
    }
    kept[target.cluster_id] = record
    state[INSTALLER_PRODUCTS_STATE_KEY] = kept
    return record
