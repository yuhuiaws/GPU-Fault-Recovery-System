"""The node-key custody check of ``preflight``, ``deploy -f`` and ``join-cluster``.

``node_key_custody_admin_entry.assert_site_custody_current`` probes the current
custody binding of every custody-registered GPU cluster of the site, and a
binding names the cluster's live nodes (``CustodyBinding.nodes`` needs at least
one). A registered cluster whose recorded selector selects no node -- parked at
zero by its provider or unavailable (``gpu_fault_release.regional_parked_cluster``)
-- has no binding at all, so the probe refused with "administrator custody node
inventory is incomplete or ambiguous": every ``preflight``, every ``deploy -f``
and every join of *any* cluster of the site, for one cluster without nodes.

``assert_site_custody_current`` here is that check, cluster by cluster, except
that a registered cluster the release engine proves to have no node
(``bootstrap_zero_nodes.zero_node_proof_for_entry``: the API lists no node for
its selector, then the provider documents) is skipped and narrated
(``zero-node-gpu-cluster ... step=custody``): there is no key it holds in custody
and no node binding to check. Every other registered cluster is probed exactly
as before, a node list that cannot be read proves nothing, and nothing about
custody itself changes: when nodes return, their binding is judged as any
changed binding is. A custody-registered cluster's own join without nodes is
refused up front (``cluster_join_zero_nodes``).

The custody modules stay as they are: ``src/gpu_fault/admin/node_key_custody*.py``
are node-template inputs (``config/release-identity.yaml``), and an edit to one
would make the release an Agent-wave release. This module is not one, and the
callers use it instead of the original entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gpu_fault.admin.bootstrap_common import (
    BootstrapMutationRequired,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.bootstrap_zero_nodes import zero_node_proof_for_entry
from gpu_fault.admin.node_key_custody_admin import (
    CustodyReconciliationRequired,
    provision_admin_custody,
    site_custody_context,
)
from gpu_fault.admin.node_key_custody_admin_config import load_admin_custody
from gpu_fault.admin.node_key_custody_admin_entry import (
    CUSTODY_PAUSE_ERRORS as CUSTODY_PAUSE_ERRORS,
)
from gpu_fault.admin.node_key_custody_admin_entry import (
    record_join_custody_pause as record_join_custody_pause,
)
from gpu_fault.admin.node_key_custody_admin_probe import CustodyReadRunner
from gpu_fault_release import regional_parked_cluster as parked_cluster

if TYPE_CHECKING:
    from gpu_fault.admin.site import RenderedSite

#: The narration of a registered cluster the check skips.
STEP = "custody"
RECORDED = "no node, no custody binding to check"


def custody_registered(site: RenderedSite, eks_arn: str) -> bool:
    """Whether the site's custody registration names the cluster ``eks_arn``."""

    registration = load_admin_custody(site.source.parent)
    return registration is not None and eks_arn in registration.selection.clusters


def skipped_without_nodes(
    site: RenderedSite, entry: dict[str, object], runner: CommandRunner
) -> bool:
    """Whether the registered cluster ``entry`` has no node, said if so."""

    config = site.release_config
    kubeconfig = config.get("gpu_kubeconfig")
    proof = zero_node_proof_for_entry(
        ReadOnlyProbeRunner(runner),
        gpu_kubeconfig=Path(str(kubeconfig)) if kubeconfig else None,
        entry=entry,
        region=str(config.get("aws_region") or ""),
    )
    if proof is None:
        return False
    parked_cluster.narrate(
        str(entry.get("cluster_id")),
        state=parked_cluster.state_of(proof),
        step=STEP,
        recorded=RECORDED,
        reason=str(proof.get("reason")),
    )
    return True


def assert_site_custody_current(site: RenderedSite, runner: CommandRunner) -> None:
    """Probe the custody of every registered cluster that has a node."""

    registration = load_admin_custody(site.source.parent)
    if registration is None:
        return
    from gpu_fault.admin.bootstrap import discover_cluster

    readonly = CustodyReadRunner(runner)
    for entry in site.release_config["clusters"]:
        if entry["eks_cluster_arn"] not in registration.selection.clusters:
            continue
        if skipped_without_nodes(site, entry, runner):
            continue
        # The site entry's recorded node scope is the runtime truth; the
        # rediscovery only confirms the cluster's identity.
        cluster = discover_cluster(
            readonly,
            cluster_arn=entry["eks_cluster_arn"],
            role="gpu",
            context=entry["context"],
            site_id=str(site.release_config["site_name"]),
            existing_clusters=tuple(site.release_config["clusters"]),
        )
        context = site_custody_context(site, cluster, str(entry["cluster_id"]))
        try:
            provision_admin_custody(
                readonly,
                context,
                fleet_master_file=Path("/unused-by-probe"),
                probe_only=True,
            )
        except BootstrapMutationRequired:
            raise CustodyReconciliationRequired(
                "configured custody is incomplete; public preflight cannot prepare or sign it"
            ) from None
