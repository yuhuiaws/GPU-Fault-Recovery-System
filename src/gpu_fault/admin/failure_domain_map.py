"""Render the node failure-domain map the remediation budget enforces.

The executor reads ``GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP`` (a JSON file,
``{cluster_id: {node_id: domain}}``) and takes one ``domain:`` budget slot per
failure domain a step touches. Nothing in the control plane can derive that
map on its own -- an incident carries node ids, an agent heartbeat carries
versions -- so the administrator renders it from the GPU clusters' node labels
and ships it as a ConfigMap the control-worker mounts. The mount and the
environment variable are both optional in the Deployment: a site without the
ConfigMap runs without the domain tier, a site with it must ship a valid one
or the worker refuses to start.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import cluster_nodes
from gpu_fault.execution.remediation_budget import FAILURE_DOMAIN_MAP_ENV
from gpu_fault.failure_domains import FAILURE_DOMAIN_LABELS, failure_domain_map

FAILURE_DOMAIN_CONFIGMAP = "gpu-fault-failure-domain-map"
FAILURE_DOMAIN_FILE = "failure-domains.json"
FAILURE_DOMAIN_MOUNT_DIR = "/etc/gpu-fault/failure-domains"
FAILURE_DOMAIN_MAP_PATH = f"{FAILURE_DOMAIN_MOUNT_DIR}/{FAILURE_DOMAIN_FILE}"
CONTROL_PLANE_NAMESPACE = "gpu-fault-system"

NodeLister = Callable[[RenderedSite, str], Mapping[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class FailureDomainMapResult:
    mapping: dict[str, dict[str, str]]
    unmapped: dict[str, list[str]] = field(default_factory=dict)
    label_keys: tuple[str, ...] = FAILURE_DOMAIN_LABELS

    def summary(self) -> dict[str, Any]:
        return {
            "clusters": {
                cluster_id: {
                    "mapped_nodes": len(nodes),
                    "failure_domains": sorted(set(nodes.values())),
                }
                for cluster_id, nodes in sorted(self.mapping.items())
            },
            "unmapped": self.unmapped,
            "label_keys": list(self.label_keys),
            "environment": {FAILURE_DOMAIN_MAP_ENV: FAILURE_DOMAIN_MAP_PATH},
        }


def build_failure_domain_map(
    site: RenderedSite,
    *,
    cluster_ids: Sequence[str] = (),
    label_keys: Sequence[str] = FAILURE_DOMAIN_LABELS,
    list_nodes: NodeLister | None = None,
) -> FailureDomainMapResult:
    # Resolved at call time so the kubectl-backed lister stays a module-level
    # collaborator a test can replace instead of a value frozen at import.
    lister: NodeLister = list_nodes or cluster_nodes
    managed = [str(item["cluster_id"]) for item in site.release_config["clusters"]]
    unknown = sorted(set(cluster_ids) - set(managed))
    if unknown:
        raise BootstrapError(
            "failure-domain map cluster is not in the managed site: "
            + ", ".join(unknown)
        )
    selected = [
        cluster_id
        for cluster_id in managed
        if not cluster_ids or cluster_id in cluster_ids
    ]
    mapping: dict[str, dict[str, str]] = {}
    unmapped: dict[str, list[str]] = {}
    for cluster_id in selected:
        inventory = lister(site, cluster_id)
        mapped = failure_domain_map(
            cluster_id, list(inventory.values()), label_keys=label_keys
        )[cluster_id]
        mapping[cluster_id] = mapped
        missing = sorted(set(inventory) - set(mapped))
        if missing:
            unmapped[cluster_id] = missing
    return FailureDomainMapResult(
        mapping=mapping, unmapped=unmapped, label_keys=tuple(label_keys)
    )


def failure_domain_configmap(
    mapping: Mapping[str, Mapping[str, str]],
    *,
    namespace: str = CONTROL_PLANE_NAMESPACE,
) -> dict[str, Any]:
    """The ConfigMap the control-worker mounts; ``map-path`` feeds the env var."""

    document = json.dumps(
        {
            cluster: dict(sorted(nodes.items()))
            for cluster, nodes in sorted(mapping.items())
        },
        indent=2,
        sort_keys=True,
    )
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": FAILURE_DOMAIN_CONFIGMAP, "namespace": namespace},
        "data": {
            FAILURE_DOMAIN_FILE: document + "\n",
            "map-path": FAILURE_DOMAIN_MAP_PATH,
        },
    }


def add_failure_domain_map_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin failure-domain-map``; the CLI passes its site options."""

    command = commands.add_parser(
        "failure-domain-map",
        usage=(
            "gpu-fault-admin failure-domain-map --state-dir STATE_DIR "
            "[--cluster-id CLUSTER_ID ...] [--label LABEL_KEY ...] "
            "[--output PATH]"
        ),
        help=(
            "render the node failure-domain ConfigMap the remediation budget "
            "enforces, from the GPU clusters' node labels"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--cluster-id",
        action="append",
        default=[],
        metavar="CLUSTER_ID",
        help="limit the map to these managed GPU clusters (default: all)",
    )
    command.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="LABEL_KEY",
        help=(
            "node label keys to read, finest domain first "
            f"(default: {', '.join(FAILURE_DOMAIN_LABELS)})"
        ),
    )
    command.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="write the ConfigMap manifest here and print a summary "
        "(default: manifest on stdout)",
    )


def run_failure_domain_map_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    """Render the ConfigMap for the parsed command line; raises BootstrapError."""

    label_keys = tuple(arguments.label) or FAILURE_DOMAIN_LABELS
    result = build_failure_domain_map(
        site,
        cluster_ids=tuple(arguments.cluster_id),
        label_keys=label_keys,
    )
    manifest = failure_domain_configmap(result.mapping)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output = cast(Path | None, arguments.output)
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.expanduser().write_text(rendered, encoding="utf-8")
        print(json.dumps({**result.summary(), "output": str(output)}, indent=2))
    return 0
