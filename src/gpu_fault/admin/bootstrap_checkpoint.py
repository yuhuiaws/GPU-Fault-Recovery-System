from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
# ``resources`` key holding the EKS ARN -> HyperPod ARN map the first discovery
# learned. An EKS ARN on the command line has no HyperPod name in it, so without
# the map every deploy lists and describes every HyperPod cluster in the region
# to find the one owning it. The map is a hint, not an identity: discovery
# re-describes the hinted cluster and falls back to the inventory when the
# orchestrator no longer matches, and ``bind_initial_deploy_target`` still
# validates the resolved identity against the site.
HYPERPOD_HINTS = "hyperpod_by_eks"


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


def _release_task_digests(
    task_digest: Callable[[str, object], str],
    *,
    assets: Mapping[str, str | None],
    images: Mapping[str, Any],
    alert_email: str | None,
    control_plane_wheel_sha256: str | None,
    dashboards: Mapping[str, str],
    grafana: Mapping[str, Any],
) -> dict[str, str]:
    """Digest the two probe-before-ensure tasks that ship a release image.

    These tasks install live resources whose desired state is decided by the
    assets they apply and by the images and wheel bytes they reference, not by
    the release identity. Binding `release_identity` made every release ID bump
    re-run them even when nothing they touch changed; drift that no static input
    can describe (node membership, out-of-band edits) is caught by the read-only
    probes instead.

    Besides `release` itself they are the only tasks whose digest needs the
    built release, so they are bound after the build (`bind_bootstrap_inputs`)
    while every other task is bound before the graph starts
    (`bind_foundation_inputs`).
    """

    return {
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


@dataclass(frozen=True)
class _FoundationInputs:
    """What the task digests read that is known before the release is built,
    and the digests of the tasks that read nothing else.

    Both binds are computed from one of these, so the digest a task gets before
    the graph starts is byte-identical to the one the full bind records for it
    after the build.
    """

    root: Path
    sources: Mapping[str, str | None]
    assets: Mapping[str, str | None]
    cpu_identity: Mapping[str, Any]
    gpu_identity: Sequence[Mapping[str, Any]]
    notification_identity: Mapping[str, str | None]
    admin_config_sha256: str | None
    task_digest: Callable[[str, object], str]
    task_digests: Mapping[str, str]


def _foundation_inputs(
    state: BootstrapState,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    gpu_clusters: tuple[ClusterIdentity, ...],
) -> _FoundationInputs:
    root = request.repository_root.resolve()
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
    # Sender and recipient are the administrator address (see
    # ``resolve_notification_routing``), so the address is the whole identity.
    notification_identity = {"admin_email": request.alert_email}
    # Written by ``initialize_desired_admin_config`` before deploy enters
    # bootstrap, so both binds digest the same bytes.
    admin_config_sha256 = _file_sha256(
        request.state_dir.resolve() / "admin-config/desired.json"
    )
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
        `monitoring_install`, `aurora_ready`, `aurora_refresh`, `node_keys:*`,
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
            {"admin_config_sha256": admin_config_sha256},
        ),
        # Instances available and the control-plane Secret present: nothing
        # static decides it, so the probe is its only trigger (like the add-on).
        "aurora_ready": task_digest("aurora_ready", {"eks_arn": cpu.eks_arn}),
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
    }
    for cluster_identity in gpu_identity:
        cluster_id = safe_name(str(cluster_identity["hyperpod_name"]))
        # Node keys are decided by the provisioning script and the cluster, not
        # by the release; the per-cluster roles are re-proved by a read-only
        # probe, so their digest only has to change when the cluster itself does.
        name = f"node_keys:{cluster_id}"
        task_digests[name] = task_digest(
            name,
            {
                "asset": assets["deploy/node/provision-node-action-keys.sh"],
                "cluster": cluster_identity,
            },
        )
        for prefix in ("executor_role:", "adot_writer_role:"):
            name = f"{prefix}{cluster_id}"
            task_digests[name] = task_digest(name, {"cluster": cluster_identity})
    return _FoundationInputs(
        root=root,
        sources=sources,
        assets=assets,
        cpu_identity=cpu_identity,
        gpu_identity=gpu_identity,
        notification_identity=notification_identity,
        admin_config_sha256=admin_config_sha256,
        task_digest=task_digest,
        task_digests=task_digests,
    )


def bind_foundation_inputs(
    state: BootstrapState,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    gpu_clusters: tuple[ClusterIdentity, ...],
) -> None:
    """Bind the digests that do not need the built release, before the graph
    reads its checkpoint.

    Bootstrap builds the signed release on a thread beside the task graph, and
    `bind_bootstrap_inputs` can only run once that build has a manifest to
    digest -- minutes after `run_parallel` has read `completed_tasks`. Bound
    that late, a checkpoint whose inputs changed since the last deploy is
    trusted for one more deploy; and before `BootstrapState` protected its own
    run's completions, every task that had finished before the build was erased
    by the late bind and re-run on the next deploy (`nlb_network` and `pki`,
    which finish first, went missing from a live site's state this way).

    Everything but `release`, `monitoring_install` and `aurora_refresh` depends
    only on the request, the clusters, the admin config and the source
    snapshot, all fixed before the graph starts, so those digests are bound
    here through the same helper the full bind uses. The whole-input digest
    needs the release and is left to the full bind.
    """

    inputs = _foundation_inputs(
        state, request=request, cpu=cpu, gpu_clusters=gpu_clusters
    )
    state.bind_inputs(None, inputs.task_digests)


def bind_bootstrap_inputs(
    state: BootstrapState,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    gpu_clusters: tuple[ClusterIdentity, ...],
    release: Mapping[str, Any],
) -> str:
    """Bind every task digest and the whole-input digest once the release is
    built.

    In a deploy this follows `bind_foundation_inputs`: the release-independent
    digests it carries are byte-identical to that bind's, so it re-affirms
    those and adds `release`, `monitoring_install` and `aurora_refresh`.
    """

    inputs = _foundation_inputs(
        state, request=request, cpu=cpu, gpu_clusters=gpu_clusters
    )
    manifest = Path(str(release["manifest"])).expanduser().resolve()
    release_identity = {
        "release_id": release.get("release_id"),
        "manifest_sha256": _file_sha256(manifest),
        "agent_config_digest": release.get("agent_config_digest"),
        "images": release.get("images"),
    }
    images = release.get("images")
    images = images if isinstance(images, Mapping) else {}
    control_plane_wheel_sha256 = _manifest_wheel_sha256(inputs.root, manifest)
    payload = {
        "schema_version": 1,
        "cpu": inputs.cpu_identity,
        "gpu_clusters": inputs.gpu_identity,
        "release": release_identity,
        "notifications": inputs.notification_identity,
        "admin_config_sha256": inputs.admin_config_sha256,
        "reconcile_source_sha256": inputs.sources,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    task_digests = {
        **inputs.task_digests,
        "release": inputs.task_digest("release", {"release": release_identity}),
        **_release_task_digests(
            inputs.task_digest,
            assets=inputs.assets,
            images=images,
            alert_email=request.alert_email,
            control_plane_wheel_sha256=control_plane_wheel_sha256,
            dashboards=dashboard_asset_digests(inputs.root),
            grafana={
                "workspace_id": request.grafana_workspace_id,
                "viewer": request.grafana_viewer,
            },
        ),
    }
    state.bind_inputs(digest, task_digests)
    return digest
