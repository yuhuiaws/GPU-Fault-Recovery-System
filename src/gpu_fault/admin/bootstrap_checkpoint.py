from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import (
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    safe_name,
)

BOOTSTRAP_RECONCILE_SOURCES = (
    "src/gpu_fault/admin/bootstrap.py",
    "src/gpu_fault/admin/bootstrap_aurora.py",
    "src/gpu_fault/admin/bootstrap_checkpoint.py",
    "src/gpu_fault/admin/bootstrap_common.py",
    "src/gpu_fault/admin/bootstrap_platform_probes.py",
    "src/gpu_fault/admin/bootstrap_site.py",
    "src/gpu_fault/admin/bootstrap_services.py",
    "src/gpu_fault/admin/monitoring_subscriptions.py",
    "src/gpu_fault/admin/notification_bootstrap.py",
    "src/gpu_fault/admin/notifications.py",
    "src/gpu_fault/admin/release_artifacts.py",
    "src/gpu_fault/admin/release_repositories.py",
)
# Assets the second-phase tasks execute or render. They belong to the per-task
# inputs, not to the shared reconcile sources: a task must re-run when the asset
# it applies changes, and must not re-run because an unrelated asset changed.
BOOTSTRAP_TASK_ASSETS = (
    "deploy/node/provision-node-action-keys.sh",
    "deploy/observability/install-amp-monitoring.sh",
    "deploy/observability/adot-control-plane.yaml",
    "deploy/observability/amp-rules.yaml",
    "deploy/observability/amp-alertmanager.yaml",
    "deploy/control-plane/regional/aurora-credential-refresh.yaml",
)


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


def _platform_task_digests(
    task_digest: Callable[[object], str],
    *,
    services_source: str | None,
    assets: Mapping[str, str | None],
    images: Mapping[str, Any],
    alert_email: str | None,
    control_plane_wheel_sha256: str | None,
    gpu_identity: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Digest the probe-before-ensure tasks against their real inputs.

    These tasks install live resources whose desired state is decided by the
    assets they apply and by the images and wheel bytes they reference, not by
    the release identity. Binding `release_identity` made every release ID bump
    re-run them even when nothing they touch changed; drift that no static input
    can describe (node membership, out-of-band edits) is caught by the read-only
    probes instead.
    """

    digests = {
        "monitoring_install": task_digest(
            {
                "source": services_source,
                "assets": {
                    "installer": assets[
                        "deploy/observability/install-amp-monitoring.sh"
                    ],
                    "collector": assets["deploy/observability/adot-control-plane.yaml"],
                    "rules": assets["deploy/observability/amp-rules.yaml"],
                    "alertmanager": assets[
                        "deploy/observability/amp-alertmanager.yaml"
                    ],
                },
                "adot_image": images.get("adot"),
                "alert_email": alert_email,
            }
        ),
        "aurora_refresh": task_digest(
            {
                "source": services_source,
                "asset": assets[
                    "deploy/control-plane/regional/aurora-credential-refresh.yaml"
                ],
                "control_plane_wheel_sha256": control_plane_wheel_sha256,
                "runtime_image": images.get("runtime"),
            }
        ),
    }
    for cluster_identity in gpu_identity:
        cluster_id = safe_name(str(cluster_identity["hyperpod_name"]))
        digests[f"node_keys:{cluster_id}"] = task_digest(
            {
                "source": services_source,
                "asset": assets["deploy/node/provision-node-action-keys.sh"],
                "cluster": cluster_identity,
            }
        )
    return digests


def bind_bootstrap_inputs(
    state: BootstrapState,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    gpu_clusters: tuple[ClusterIdentity, ...],
    release: Mapping[str, Any],
) -> str:
    root = request.repository_root.resolve()
    manifest = Path(str(release["manifest"])).expanduser().resolve()
    sources = {
        relative: _file_sha256(root / relative)
        for relative in BOOTSTRAP_RECONCILE_SOURCES
    }
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
        "release_id": release.get("release_id"),
        "manifest_sha256": _file_sha256(manifest),
        "agent_config_digest": release.get("agent_config_digest"),
        "images": release.get("images"),
    }
    images = release.get("images")
    images = images if isinstance(images, Mapping) else {}
    control_plane_wheel_sha256 = _manifest_wheel_sha256(root, manifest)
    notification_identity = {
        "admin_email": request.alert_email,
        "email_sender": request.email_sender,
        "email_recipients": list(request.email_recipients),
        "email_subject_prefix": request.email_subject_prefix,
    }
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
        "common_source_sha256": {
            name: sources[name]
            for name in (
                "src/gpu_fault/admin/bootstrap_checkpoint.py",
                "src/gpu_fault/admin/bootstrap_common.py",
            )
        },
    }

    def task_digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                {"common": common, "task": value},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    bootstrap_source = sources["src/gpu_fault/admin/bootstrap.py"]
    services_source = sources["src/gpu_fault/admin/bootstrap_services.py"]
    notification_sources = {
        name: sources[name]
        for name in (
            "src/gpu_fault/admin/notification_bootstrap.py",
            "src/gpu_fault/admin/notifications.py",
        )
    }
    task_digests = {
        "release_repositories": task_digest(
            {
                "source": sources["src/gpu_fault/admin/release_repositories.py"],
                "region": cpu.region,
                "account_id": cpu.account_id,
            }
        ),
        "release": task_digest(
            {
                "sources": {
                    "artifacts": sources["src/gpu_fault/admin/release_artifacts.py"],
                    "repositories": sources[
                        "src/gpu_fault/admin/release_repositories.py"
                    ],
                },
                "release": release_identity,
            }
        ),
        "pod_identity_agent": task_digest(
            {"source": services_source, "eks_arn": cpu.eks_arn}
        ),
        "nlb_network": task_digest(
            {
                "source": bootstrap_source,
                "cpu": cpu_identity,
                "gpu_clusters": gpu_identity,
            }
        ),
        "pki": task_digest(
            {
                "source": bootstrap_source,
                "cpu": cpu_identity,
                "gpu_clusters": gpu_identity,
            }
        ),
        "aurora": task_digest(
            {
                "sources": {
                    "bootstrap": bootstrap_source,
                    "aurora": sources["src/gpu_fault/admin/bootstrap_aurora.py"],
                },
                "admin_config_sha256": payload["admin_config_sha256"],
            }
        ),
        "load_balancer_controller": task_digest(
            {"source": services_source, "eks_arn": cpu.eks_arn}
        ),
        "control_plane_role": task_digest(
            {
                "source": services_source,
                "notifications": notification_identity,
            }
        ),
        "email_notifications": task_digest(
            {
                "sources": notification_sources,
                "notifications": notification_identity,
            }
        ),
        "monitoring_resources": task_digest(
            {
                "sources": {
                    **notification_sources,
                    "subscriptions": sources[
                        "src/gpu_fault/admin/monitoring_subscriptions.py"
                    ],
                    "services": services_source,
                },
                "admin_email": request.alert_email,
            }
        ),
        **_platform_task_digests(
            task_digest,
            services_source=services_source,
            assets=assets,
            images=images,
            alert_email=request.alert_email,
            control_plane_wheel_sha256=control_plane_wheel_sha256,
            gpu_identity=gpu_identity,
        ),
    }
    for cluster_identity in gpu_identity:
        cluster_id = safe_name(str(cluster_identity["hyperpod_name"]))
        task_digests[f"executor_role:{cluster_id}"] = task_digest(
            {"source": services_source, "cluster": cluster_identity}
        )
    state.bind_inputs(digest, task_digests)
    return digest
