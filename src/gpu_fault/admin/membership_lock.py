from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.operation_lock import SiteOperationBusy, site_operation_lock
from gpu_fault.admin.site import RenderedSite, load_site


@contextmanager
def administrator_operation_lock(state_dir: Path) -> Iterator[None]:
    """One lock, one policy: refuse at once and name the holder.

    A ``join-cluster`` typed during a 40-minute deploy used to wait silently for
    the deploy to finish; now it is told which pid and command hold the site.
    """

    with ExitStack() as stack:
        try:
            stack.enter_context(site_operation_lock(state_dir, wait=False))
        except SiteOperationBusy as exc:
            raise BootstrapError(str(exc)) from exc
        yield


@contextmanager
def membership_operation_lock(site: RenderedSite) -> Iterator[None]:
    with administrator_operation_lock(site.source.parent):
        yield


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
