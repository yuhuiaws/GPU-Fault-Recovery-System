from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.operation_lock import SiteOperationBusy, site_operation_lock
from gpu_fault.admin.site import RenderedSite, load_site


@contextmanager
def membership_operation_lock(site: RenderedSite) -> Iterator[None]:
    with site_operation_lock(site.source.parent, wait=True):
        yield


@contextmanager
def administrator_operation_lock(state_dir: Path) -> Iterator[None]:
    try:
        with site_operation_lock(state_dir, wait=False):
            yield
    except SiteOperationBusy as exc:
        raise BootstrapError("another administrator mutation is in progress") from exc


def reload_site_for_mutation(site: RenderedSite) -> RenderedSite:
    current = load_site(site.source, repository_root=site.repository_root)
    expected = (
        site.release_config["site_name"],
        site.release_config["aws_region"],
        site.release_config["cpu_eks_arn"],
    )
    observed = (
        current.release_config["site_name"],
        current.release_config["aws_region"],
        current.release_config["cpu_eks_arn"],
    )
    if observed != expected:
        raise BootstrapError("managed site identity changed before mutation lock")
    return current
