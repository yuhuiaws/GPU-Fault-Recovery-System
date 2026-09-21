"""Return declared warm spares to their recorded baseline before node cleanup.

``config spare --declare`` labels and cordons a node and keeps what the node
looked like before in ``<state-dir>/warm-spares/<node>.json``; nothing on the
node itself says what to put back. The Kubernetes node cleanup therefore cannot
release a spare, and an uninstall that left one behind would leave a cordoned
node with the solution's label on it. The uninstall releases every unreleased
declaration here, with the same refusals as ``config spare --release``, before
the cleanup touches node metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.uninstall_types import UninstallRequest
from gpu_fault.admin.warm_spare import (
    RECORDS_PATH,
    RELEASE_CONFIRMATION,
    AgentLookup,
    NodeApi,
    WarmSpareError,
    WarmSpareRequest,
    control_plane_agent_lookup,
    node_api_for_site,
    perform_warm_spare,
    read_record,
    site_cluster,
)

ApiFactory = Callable[[RenderedSite, Mapping[str, Any]], NodeApi]


def unreleased_spare_records(state_dir: Path) -> list[Path]:
    """Every declaration record that has not been released, oldest name first."""

    directory = state_dir / RECORDS_PATH
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        document = read_record(path)
        if document is not None and not document.get("released_at"):
            records.append(path)
    return records


def release_declared_spares(
    request: UninstallRequest,
    state_path: Path,
    state: dict[str, Any],
    *,
    reference: str,
    api_factory: ApiFactory = node_api_for_site,
    agent_lookup: AgentLookup | None = None,
    actor: str | None = None,
) -> list[str]:
    """Release the site's declared spares and journal them; refusals fail closed."""

    site = request.site
    state_dir = site.source.parent
    lookup = (
        agent_lookup if agent_lookup is not None else control_plane_agent_lookup(site)
    )
    released: list[dict[str, Any]] = []
    for record in unreleased_spare_records(state_dir):
        document = read_record(record) or {}
        node = str(document.get("node") or "")
        cluster_id = document.get("cluster_id")
        if not node or not isinstance(cluster_id, str) or not cluster_id:
            raise BootstrapError(
                f"warm-spare record {record.name} lacks its node or cluster"
            )
        spare_request = WarmSpareRequest(
            state_dir=state_dir,
            node=node,
            cluster_id=cluster_id,
            reference=reference,
            mode="release",
            confirmation=RELEASE_CONFIRMATION,
        )
        try:
            report = perform_warm_spare(
                api_factory(site, site_cluster(site, cluster_id)),
                spare_request,
                cluster_id=cluster_id,
                record=record,
                agent_lookup=lookup,
                actor=actor,
            )
        except WarmSpareError as exc:
            raise BootstrapError(
                f"uninstall cannot release warm spare {node}: {exc}"
            ) from exc
        released.append(
            {
                "node": node,
                "cluster_id": cluster_id,
                "released_at": (report.get("release") or {}).get("released_at"),
            }
        )
    if released:
        journal = state.get("released_warm_spares")
        if not isinstance(journal, list):
            journal = []
        journal.extend(released)
        state["released_warm_spares"] = journal
        write_json_atomic(state_path, state)
    return [str(item["node"]) for item in released]
