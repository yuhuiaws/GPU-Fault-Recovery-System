"""The node failure-domain map the remediation budget enforces, product-managed.

The executor reads ``GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP`` (a JSON file,
``{cluster_id: {node_id: domain}}``) and takes one ``domain:`` budget slot per
failure domain a step touches. The control plane cannot derive that map on its
own -- an incident carries node ids, an agent heartbeat carries versions -- so
the product renders it from the GPU clusters' node labels and ships it as the
ConfigMap ``gpu-fault-failure-domain-map`` the control-worker mounts:

* every ``deploy`` (upgrade and bootstrap) renders it inside the control-plane
  apply step and stamps its digest on the control-worker pod template, so the
  worker rolls exactly when the map changed
  (``gpu_fault_release.regional_release_failure_domains``);
* ``join-cluster`` and ``remove-cluster`` re-render it here, once the managed
  cluster set is final (``apply_failure_domain_map``).

The administrator configures nothing but, optionally, ``failureDomainLabels``
in ``site.yaml`` when the fleet carries topology labels the default priority
does not know. ``gpu-fault-admin failure-domain-map`` survives only as a
read-only debugging aid that writes what the product would render and lists
the ``unmapped`` nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.execution.remediation_budget import FAILURE_DOMAIN_MAP_ENV
from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_CONFIGMAP,
    FAILURE_DOMAIN_FILE,
    FAILURE_DOMAIN_LABELS,
    FAILURE_DOMAIN_MAP_ANNOTATION,
    FAILURE_DOMAIN_MAP_PATH,
    FAILURE_DOMAIN_MOUNT_DIR,
    failure_domain_configmap,
    failure_domain_map,
    failure_domain_map_sha256,
    validate_failure_domain_manifest,
)

__all__ = [
    "FAILURE_DOMAIN_CONFIGMAP",
    "FAILURE_DOMAIN_FILE",
    "FAILURE_DOMAIN_MAP_PATH",
    "FAILURE_DOMAIN_MOUNT_DIR",
    "FailureDomainMapResult",
    "add_failure_domain_map_command",
    "apply_failure_domain_map",
    "build_failure_domain_map",
    "cluster_nodes",
    "failure_domain_configmap",
    "run_failure_domain_map_command",
]

CONTROL_WORKER_DEPLOYMENT = "gpu-fault-control-worker"
# The HyperPod cluster label the release engine filters its node inventory on
# (``regional_release_gpu_rollout.gpu_node_command``); the admin-side render
# selects on it too so both produce one document, one digest, for one fleet.
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"

NodeLister = Callable[[RenderedSite, str], Mapping[str, Mapping[str, Any]]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


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


def site_failure_domain_labels(site: RenderedSite) -> tuple[str, ...]:
    """``spec.failureDomainLabels`` as rendered into the release config."""

    configured = site.release_config.get("failure_domain_labels")
    if not configured:
        return FAILURE_DOMAIN_LABELS
    return tuple(str(item) for item in configured)


def _gpu_kubectl(site: RenderedSite, target: Mapping[str, Any]) -> list[str]:
    kubeconfig = (
        site.release_config.get("gpu_kubeconfig")
        or site.environment.get("KUBECONFIG")
        or os.environ.get("KUBECONFIG")
        or str(Path.home() / ".kube/config")
    )
    return [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        str(target["context"]),
    ]


def cluster_nodes(
    site: RenderedSite, cluster_id: str, *, run: CommandRunner = subprocess.run
) -> dict[str, dict[str, Any]]:
    """One ``get nodes`` per cluster, selected on the HyperPod cluster label."""

    targets = {
        str(item["cluster_id"]): item for item in site.release_config["clusters"]
    }
    target = targets.get(cluster_id)
    if target is None:
        raise BootstrapError(
            f"failure-domain map cluster is not in the managed site: {cluster_id}"
        )
    command = [*_gpu_kubectl(site, target), "get", "nodes", "-o", "json"]
    hyperpod = str(target.get("hyperpod_cluster_name") or "")
    if hyperpod:
        command.extend(["-l", f"{HYPERPOD_CLUSTER_LABEL}={hyperpod}"])
    completed = run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise BootstrapError(
            f"failure-domain map cannot read GPU nodes for {cluster_id}: "
            f"{(completed.stderr or '').strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            f"failure-domain map received invalid node JSON for {cluster_id}"
        ) from exc
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise BootstrapError(
            f"failure-domain map received an invalid node list for {cluster_id}"
        )
    return {
        str(item["metadata"]["name"]): item
        for item in items
        if isinstance(item, dict) and (item.get("metadata") or {}).get("name")
    }


def build_failure_domain_map(
    site: RenderedSite,
    *,
    cluster_ids: Sequence[str] = (),
    label_keys: Sequence[str] | None = None,
    list_nodes: NodeLister | None = None,
) -> FailureDomainMapResult:
    # Resolved at call time so the kubectl-backed lister stays a module-level
    # collaborator a test can replace instead of a value frozen at import.
    lister: NodeLister = list_nodes or cluster_nodes
    keys = tuple(label_keys) if label_keys else site_failure_domain_labels(site)
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
            cluster_id, list(inventory.values()), label_keys=keys
        )[cluster_id]
        mapping[cluster_id] = mapped
        missing = sorted(set(inventory) - set(mapped))
        if missing:
            unmapped[cluster_id] = missing
    return FailureDomainMapResult(mapping=mapping, unmapped=unmapped, label_keys=keys)


def _cpu_kubectl(site: RenderedSite) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
    ]


def _run(run: CommandRunner, arguments: list[str], **kwargs: Any) -> str:
    completed = run(arguments, check=False, capture_output=True, text=True, **kwargs)
    if completed.returncode:
        raise BootstrapError(
            "failure-domain map apply failed: "
            + (completed.stderr or completed.stdout or "").strip()
        )
    return str(completed.stdout or "")


def apply_failure_domain_map(
    site: RenderedSite,
    *,
    run: CommandRunner = subprocess.run,
) -> FailureDomainMapResult:
    """Render the map for the managed site, validate it, ship it, roll on change.

    The ConfigMap is applied to the CPU control plane and the control-worker's
    pod template is stamped with the document digest; Kubernetes rolls the
    worker only when that annotation actually changed. A worker Deployment
    that does not exist yet (mid-bootstrap) is left to ``deploy``, which
    stamps the same digest through the role-split apply.
    """

    result = build_failure_domain_map(
        site,
        list_nodes=lambda current, cluster_id: cluster_nodes(
            current, cluster_id, run=run
        ),
    )
    manifest = failure_domain_configmap(
        result.mapping, namespace=str(site.release_config["namespace"])
    )
    try:
        validate_failure_domain_manifest(manifest)
    except ValueError as exc:
        raise BootstrapError(f"failure-domain map render is invalid: {exc}") from exc
    kubectl = _cpu_kubectl(site)
    _run(run, [*kubectl, "apply", "-f", "-"], input=json.dumps(manifest))
    exists = run(
        [*kubectl, "get", "deployment", CONTROL_WORKER_DEPLOYMENT],
        check=False,
        capture_output=True,
        text=True,
    )
    if exists.returncode == 0:
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            FAILURE_DOMAIN_MAP_ANNOTATION: failure_domain_map_sha256(
                                manifest
                            )
                        }
                    }
                }
            }
        }
        _run(
            run,
            [
                *kubectl,
                "patch",
                "deployment",
                CONTROL_WORKER_DEPLOYMENT,
                "--type=merge",
                "-p",
                json.dumps(patch),
            ],
        )
    return result


def add_failure_domain_map_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin failure-domain-map``; the CLI passes its site options."""

    command = commands.add_parser(
        "failure-domain-map",
        usage=(
            "gpu-fault-admin failure-domain-map --state-dir STATE_DIR [--output PATH]"
        ),
        help=(
            "read-only debugging aid: show the node failure-domain ConfigMap "
            "deploy/join-cluster/remove-cluster render and apply automatically, "
            "and the nodes it leaves unmapped"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="write the rendered ConfigMap manifest here as well "
        "(nothing is applied; the summary is printed either way)",
    )


def run_failure_domain_map_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    """Print the summary (``unmapped`` included) and optionally write the manifest.

    Reads the label keys from ``site.yaml``; never applies anything.
    Raises ``BootstrapError``.
    """

    result = build_failure_domain_map(site)
    summary: dict[str, Any] = {**result.summary(), "applied_by": "deploy"}
    output = cast(Path | None, arguments.output)
    if output is not None:
        manifest = failure_domain_configmap(
            result.mapping, namespace=str(site.release_config["namespace"])
        )
        output.expanduser().write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["output"] = str(output)
        summary["sha256"] = failure_domain_map_sha256(manifest)
    print(json.dumps(summary, indent=2))
    return 0
