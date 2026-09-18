from __future__ import annotations

from gpu_fault.admin.bootstrap_task_inputs import (
    TaskInputContext,
    task_input_fingerprints,
)

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import (
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    safe_name,
)
from gpu_fault.admin.grafana import dashboard_asset_digests
from gpu_fault.admin.rds_ca_bundle import RDS_CA_BUNDLE_PATH

# Orchestration identity remains an input for tasks without read-only revalidation.
_ORCHESTRATION_SOURCE = "src/gpu_fault/admin/bootstrap.py"
BOOTSTRAP_RECONCILE_SOURCES = (
    _ORCHESTRATION_SOURCE,
    "src/gpu_fault/admin/bootstrap_aurora.py",
    "src/gpu_fault/admin/bootstrap_checkpoint.py",
    "src/gpu_fault/admin/bootstrap_common.py",
    "src/gpu_fault/admin/bootstrap_platform_probes.py",
    "src/gpu_fault/admin/bootstrap_site.py",
    "src/gpu_fault/admin/bootstrap_services.py",
    "src/gpu_fault/admin/monitoring_policy.py",
    "src/gpu_fault/admin/monitoring_subscriptions.py",
    "src/gpu_fault/admin/node_key_proof.py",
    "src/gpu_fault/admin/notification_bootstrap.py",
    "src/gpu_fault/admin/notifications.py",
    "src/gpu_fault/admin/release_artifacts.py",
    "src/gpu_fault/admin/release_repositories.py",
    "src/gpu_fault/admin/resource_registry_dns.py",
)
# Assets bootstrap tasks execute or render. They belong to the per-task
# inputs, not to the shared reconcile sources: a task must re-run when the asset
# it applies changes, and must not re-run because an unrelated asset changed.
# The Grafana dashboards (``deploy/observability/dashboards/*.json``) are a
# directory rather than a fixed file list, so ``monitoring_install`` digests them
# per file through ``dashboard_asset_digests`` instead of an entry here.
BOOTSTRAP_TASK_ASSETS = (
    "deploy/node/provision-node-action-keys.sh",
    "deploy/node/provision_node_action_keys.py",
    "deploy/observability/install-amp-monitoring.sh",
    "deploy/observability/adot-control-plane.yaml",
    "deploy/observability/amp-rules.yaml",
    "deploy/observability/amp-alertmanager.yaml",
    "deploy/observability/amp-sns-publish-policy.json",
    "deploy/control-plane/regional/aurora-credential-refresh.yaml",
)
# ``resources`` key holding the EKS ARN -> HyperPod ARN map the first discovery
# learned. An EKS ARN on the command line has no HyperPod name in it, so without
# the map every deploy lists and describes every HyperPod cluster in the region
# to find the one owning it. The map is a hint, not an identity: discovery
# re-describes the hinted cluster and falls back to the inventory when the
# orchestrator no longer matches, and ``bind_initial_deploy_target`` still
# validates the resolved identity against the site.
HYPERPOD_HINTS = "hyperpod_by_eks"


def remember_hyperpod_hints(
    state: BootstrapState,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> None:
    hints = {cluster.eks_arn: cluster.hyperpod_arn for cluster in (cpu, *gpu_clusters)}
    if state.value["resources"].get(HYPERPOD_HINTS) != hints:
        state.record(HYPERPOD_HINTS, hints)


def load_hyperpod_hints(state_dir: Path) -> dict[str, str]:
    """The persisted EKS -> HyperPod map, read before ``BootstrapState`` can be
    opened (the site id that opens it is itself a product of discovery)."""

    path = state_dir / "bootstrap-state.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    resources = document.get("resources") if isinstance(document, dict) else None
    hints = resources.get(HYPERPOD_HINTS) if isinstance(resources, dict) else None
    if not isinstance(hints, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in hints.items()
        if isinstance(key, str) and isinstance(value, str) and value
    }


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _manifest_wheel_sha256(root: Path, manifest: Path) -> str | None:
    """Digest the control-plane wheel the release manifest points at.

    `aurora_refresh` uploads exactly this wheel as a content-addressed ConfigMap,
    so the wheel bytes decide whether the task still has anything to do. The
    manifest digest and the release ID change on every rebuild even when the
    wheel is byte-identical, which is why they are not the task input.
    """

    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError:
        return None
    raw = str((value or {}).get("wheel") or "")
    if not raw:
        return None
    wheel = Path(raw).expanduser()
    if not wheel.is_absolute():
        wheel = root / wheel
    return _file_sha256(wheel)


def bind_bootstrap_inputs(
    state: BootstrapState,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    gpu_clusters: tuple[ClusterIdentity, ...],
    release: Mapping[str, Any] | None = None,
) -> str:
    """Bind before scheduling, then rebind against the actual signed candidate.

    Ordinary infrastructure fingerprints are identical across both binds.
    Custody key fingerprints include the candidate and remain provisional until
    the release dependency completes; current-process completion is no exemption.
    """

    root = request.repository_root.resolve()
    manifest = (
        Path(str(release["manifest"])).expanduser().resolve()
        if release is not None
        else None
    )
    release_value = release or {}
    from gpu_fault.admin.node_key_custody_admin_config import load_admin_custody

    custody_registration = load_admin_custody(request.state_dir)
    sources = {
        relative: _file_sha256(root / relative)
        for relative in BOOTSTRAP_RECONCILE_SOURCES
    }
    if custody_registration is not None:
        custody_paths = [
            *sorted((root / "src/gpu_fault/admin").glob("node_key_custody*.py")),
            root / "src/gpu_fault/admin/bootstrap_tasks.py",
            root / "src/gpu_fault/admin/bootstrap_task_inputs.py",
        ]
        sources.update(
            {
                path.relative_to(root).as_posix(): _file_sha256(path)
                for path in custody_paths
            }
        )
    assets = {
        relative: _file_sha256(root / relative) for relative in BOOTSTRAP_TASK_ASSETS
    }
    cpu_identity = {
        "input_arn": cpu.input_arn,
        "eks_arn": cpu.eks_arn,
        "hyperpod_arn": cpu.hyperpod_arn,
        "vpc_id": cpu.vpc_id,
        "subnet_ids": list(cpu.subnet_ids),
    }
    gpu_identity = [
        {
            "input_arn": item.input_arn,
            "eks_arn": item.eks_arn,
            "hyperpod_arn": item.hyperpod_arn,
            "hyperpod_name": item.hyperpod_name,
            "vpc_id": item.vpc_id,
            "subnet_ids": list(item.subnet_ids),
        }
        for item in gpu_clusters
    ]
    release_identity = {
        "release_id": release_value.get("release_id"),
        "manifest_sha256": _file_sha256(manifest) if manifest is not None else None,
        "agent_config_digest": release_value.get("agent_config_digest"),
        "images": release_value.get("images"),
    }
    images = release_value.get("images")
    images = images if isinstance(images, Mapping) else {}
    control_plane_wheel_sha256 = (
        _manifest_wheel_sha256(root, manifest) if manifest is not None else None
    )
    # Sender and recipient are the administrator address (see
    # ``resolve_notification_routing``), so the address is the whole identity.
    notification_identity = {"admin_email": request.alert_email}
    payload = {
        "schema_version": 1,
        "cpu": cpu_identity,
        "gpu_clusters": gpu_identity,
        "release": release_identity,
        "notifications": notification_identity,
        "admin_config_sha256": _file_sha256(
            request.state_dir.resolve() / "admin-config/desired.json"
        ),
        "reconcile_source_sha256": sources,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    common = {
        "cpu": cpu_identity,
        "site_id": state.value["site_id"],
    }

    task_digests = task_input_fingerprints(
        TaskInputContext(
            common=common,
            region=cpu.region,
            account_id=cpu.account_id,
            cpu=cpu_identity,
            gpu_clusters=gpu_identity,
            cluster_inputs={
                safe_name(str(item["hyperpod_name"])): item for item in gpu_identity
            },
            release=release_identity,
            images=images,
            assets=assets,
            sources=sources,
            notifications=notification_identity,
            admin_config_sha256=payload["admin_config_sha256"],
            control_plane_wheel_sha256=control_plane_wheel_sha256,
            dashboards=dashboard_asset_digests(root),
            grafana={
                "workspace_id": request.grafana_workspace_id,
                "viewer": request.grafana_viewer,
            },
            release_ready=release is not None,
            rds_ca_bundle_path=RDS_CA_BUNDLE_PATH,
            custody=(
                custody_registration.model_dump(mode="json")
                if custody_registration is not None
                else {}
            ),
        )
    )
    state.bind_inputs(digest, task_digests, partial=release is None)
    return digest
