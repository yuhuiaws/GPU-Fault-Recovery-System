"""The ``config`` command's Aurora steps on a site: request, verify, await, roll back.

The RDS writer/reader functions are injected by the caller (``gpu_fault.admin.cli``)
so that one module stays the patch point for every test of the command; this
module only binds them to the site's Region and cluster and shapes their
results for the apply record.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.config import AdminConfig, AuroraCapacityConfig, config_command
from gpu_fault.admin.config_file import admin_config_file_path

if TYPE_CHECKING:
    from gpu_fault.admin.site import RenderedSite

AuroraCall = Callable[..., dict[str, Any]]


def site_aurora(site: RenderedSite) -> tuple[str, str]:
    """``(aws_region, cluster_id)`` of the site's control-plane Aurora."""

    return (
        str(site.release_config["aws_region"]),
        str(site.release_config["health"]["aurora_cluster_id"]),
    )


def request_site_aurora(
    site: RenderedSite,
    *,
    expected: AuroraCapacityConfig,
    desired: AuroraCapacityConfig,
    request: AuroraCall,
) -> dict[str, Any]:
    aws_region, cluster_id = site_aurora(site)
    return request(
        aws_region=aws_region,
        cluster_id=cluster_id,
        expected=expected,
        desired=desired,
    )


def verify_site_aurora(
    site: RenderedSite, *, expected: AuroraCapacityConfig, verify: AuroraCall
) -> dict[str, Any]:
    """An apply that leaves Aurora alone still proves the window it recorded."""

    aws_region, cluster_id = site_aurora(site)
    observed = verify(aws_region=aws_region, cluster_id=cluster_id, expected=expected)
    return {
        "before": observed,
        "after": observed,
        "modified": False,
        "scale_up": False,
        "verified": True,
    }


def await_site_aurora(
    site: RenderedSite,
    *,
    desired: AuroraCapacityConfig,
    requested: dict[str, Any],
    wait: AuroraCall,
) -> dict[str, Any]:
    """Finish a scale-up once the roles rolled: both instances at the new floor.

    ``requested`` is what a completed ``request`` returned: its settle loop
    already proved this window change with consecutive observations, so the
    floor needs one fresh observation, not that rule again (``window_settled``).
    """

    if not requested["scale_up"]:
        return requested
    aws_region, cluster_id = site_aurora(site)
    after = wait(
        aws_region=aws_region,
        cluster_id=cluster_id,
        desired=desired,
        modified=bool(requested["modified"]),
        window_settled=True,
    )
    return {**requested, "after": after}


def rollback_site_aurora(
    site: RenderedSite,
    changed: bool,
    current: AdminConfig,
    desired: AdminConfig,
    *,
    reconcile: AuroraCall,
) -> dict[str, Any] | None:
    if not changed:
        return None
    aws_region, cluster_id = site_aurora(site)
    return reconcile(
        aws_region=aws_region,
        cluster_id=cluster_id,
        expected=desired.aurora,
        desired=current.aurora,
    )


def aurora_drift_remedy(state_dir: Path) -> str:
    """What to do about a window somebody else moved; nothing was modified here."""

    return (
        "; reconcile Aurora to the reviewed configuration with gpu-fault-admin deploy "
        f"--state-dir {state_dir} (deploy moves the live window to desired.json), or "
        f"review the live window by writing it into {admin_config_file_path(state_dir)} "
        f"(spec.aurora) and rerunning {config_command(state_dir)}"
    )
