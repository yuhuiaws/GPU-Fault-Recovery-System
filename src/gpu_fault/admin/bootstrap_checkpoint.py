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
from gpu_fault.admin.grafana import dashboard_asset_digests

# The one module still bound into a task digest, for the two tasks below that no
# read-only probe re-proves.
_ORCHESTRATION_SOURCE = "src/gpu_fault/admin/bootstrap.py"
BOOTSTRAP_RECONCILE_SOURCES = (
    _ORCHESTRATION_SOURCE,
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
# The Grafana dashboards (``deploy/observability/dashboards/*.json``) are a
# directory rather than a fixed file list, so ``monitoring_install`` digests them
# per file through ``dashboard_asset_digests`` instead of an entry here.
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
    task_digest: Callable[[str, object], str],
    *,
    assets: Mapping[str, str | None],
    images: Mapping[str, Any],
    alert_email: str | None,
    control_plane_wheel_sha256: str | None,
    gpu_identity: Sequence[Mapping[str, Any]],
    dashboards: Mapping[str, str],
    grafana: Mapping[str, Any],
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
            "monitoring_install",
            {
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
                # The Grafana step rides on this task: a changed dashboard or a
                # changed --grafana option must re-run the import.
                "dashboards": dict(dashboards),
                "grafana": dict(grafana),
            },
        ),
        "aurora_refresh": task_digest(
            "aurora_refresh",
            {
                "asset": assets[
                    "deploy/control-plane/regional/aurora-credential-refresh.yaml"
                ],
                "control_plane_wheel_sha256": control_plane_wheel_sha256,
                "runtime_image": images.get("runtime"),
            },
        ),
    }
    for cluster_identity in gpu_identity:
        cluster_id = safe_name(str(cluster_identity["hyperpod_name"]))
        name = f"node_keys:{cluster_id}"
        digests[name] = task_digest(
            name,
            {
                "asset": assets["deploy/node/provision-node-action-keys.sh"],
                "cluster": cluster_identity,
            },
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
    }

    def task_digest(name: str, value: object) -> str:
        """Digest one task's desired state.

        The task name is part of the digest because two tasks can legitimately
        describe the same inputs -- `nlb_network` and `pki` both describe the same
        clusters -- and a shared digest would let one task's checkpoint answer for
        the other.

        What is deliberately *not* here is the bytes of the modules that do the
        work. Embedding them meant any refactor of `bootstrap.py` and its siblings
        re-ran every ensure path on the next deploy, including the unconditional
        `rds modify-db-subnet-group` and its `rds wait`, for a site where no
        desired resource had changed.

        That is only safe for a task whose completion is re-proved on every
        deploy. `bootstrap_tasks.py` revalidates `pod_identity_agent`,
        `monitoring_install`, `aurora_refresh`, `node_keys:*`,
        `load_balancer_controller`, `control_plane_role`, `email_notifications`,
        `monitoring_resources` and `executor_role:*` through a read-only probe
        that enters ensure on detected drift, so for those the probe is the
        re-convergence trigger and the module bytes are noise.

        `nlb_network` and `pki` have no probe, so the digest is their only
        trigger: `_ORCHESTRATION_SOURCE` (the `bootstrap.py` bytes) stays in
        those two, and a change to how they converge still re-runs them.
        `aurora` has no probe either but is deliberately left source-free: its
        reconcilable inputs are covered by `admin_config_sha256` and the CPU
        subnet ids, and its ensure path writes to RDS unconditionally, which is
        exactly what a refactor must not re-trigger.
        """

        return hashlib.sha256(
            json.dumps(
                {"common": common, "name": name, "task": value},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    task_digests = {
        "release_repositories": task_digest(
            "release_repositories",
            {"region": cpu.region, "account_id": cpu.account_id},
        ),
        "release": task_digest("release", {"release": release_identity}),
        "pod_identity_agent": task_digest(
            "pod_identity_agent", {"eks_arn": cpu.eks_arn}
        ),
        # These two are the tasks no read-only probe re-proves; see `task_digest`.
        "nlb_network": task_digest(
            "nlb_network",
            {
                "cpu": cpu_identity,
                "gpu_clusters": gpu_identity,
                "source": sources[_ORCHESTRATION_SOURCE],
            },
        ),
        "pki": task_digest(
            "pki",
            {
                "cpu": cpu_identity,
                "gpu_clusters": gpu_identity,
                "source": sources[_ORCHESTRATION_SOURCE],
            },
        ),
        "aurora": task_digest(
            "aurora",
            {"admin_config_sha256": payload["admin_config_sha256"]},
        ),
        "load_balancer_controller": task_digest(
            "load_balancer_controller", {"eks_arn": cpu.eks_arn}
        ),
        "control_plane_role": task_digest(
            "control_plane_role", {"notifications": notification_identity}
        ),
        "email_notifications": task_digest(
            "email_notifications", {"notifications": notification_identity}
        ),
        "monitoring_resources": task_digest(
            "monitoring_resources", {"admin_email": request.alert_email}
        ),
        **_platform_task_digests(
            task_digest,
            assets=assets,
            images=images,
            alert_email=request.alert_email,
            control_plane_wheel_sha256=control_plane_wheel_sha256,
            gpu_identity=gpu_identity,
            dashboards=dashboard_asset_digests(root),
            grafana={
                "enabled": request.grafana_enabled,
                "workspace_id": request.grafana_workspace_id,
                "create": request.grafana_create,
            },
        ),
    }
    for cluster_identity in gpu_identity:
        cluster_id = safe_name(str(cluster_identity["hyperpod_name"]))
        name = f"executor_role:{cluster_id}"
        task_digests[name] = task_digest(name, {"cluster": cluster_identity})
    state.bind_inputs(digest, task_digests)
    return digest
