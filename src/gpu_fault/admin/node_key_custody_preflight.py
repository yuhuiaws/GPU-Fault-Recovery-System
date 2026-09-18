"""Reuse the signed release's host preflight for same-release key activation."""

from __future__ import annotations

from typing import Any

from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.node_key_custody_admin_probe import AdminNodeKeyContext
from gpu_fault.admin.node_key_custody_models import Authorization, CustodyError
from gpu_fault.admin.release_engine import build_release
from gpu_fault.admin.site import load_site
from gpu_fault_release.regional_release_node_preflight import (
    NodeMutationPreflight,
    run_node_installer_preflight,
)


def preflight_activation(
    context: AdminNodeKeyContext,
    authorization: Authorization,
    runner: CommandRunner,
    gpu_command: Any,
) -> None:
    site = load_site(
        context.state_dir / "site.yaml", repository_root=context.repository_root
    )
    release = build_release(site)
    binding = authorization.binding
    target = release._target(context.cluster_id)
    if (
        release.release_id != binding.release.release_id
        or target.eks_cluster_arn != binding.site.gpu_eks_arn
        or target.context != context.cluster.context
        or authorization.rotate_node is None
    ):
        raise CustodyError("node key preflight release is not the authorized target")
    release.runner = runner
    release._gpu = gpu_command
    release._cpu = lambda *args: [*context.kubectl("cpu"), *args]
    candidate = NodeMutationPreflight(
        phase="upgrade",
        node_names=(authorization.rotate_node,),
        wheel_cm=release.executor_wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.node_wheel_sha,
        config_digest=binding.release.config_digest,
        runtime_profile_version=binding.release.runtime_profile_version,
        executor_wheel_filename=None,
        node_compatibility_digest=binding.release.node_digest,
        bundle_sha256=binding.release.bundle_sha256,
        template_sha256=release.node_template_sha,
        template_config_map=None,
        max_unavailable=1,
        runtime_image=release.executor_image,
        node_installer_image=release.node_installer_image,
    )
    run_node_installer_preflight(release, target, candidate)
