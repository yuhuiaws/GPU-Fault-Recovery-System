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
* single ``join-cluster`` and ``remove-cluster`` re-render the committed set;
  batch join publishes the merged verified PENDING set before activation, then
  rechecks its convergence receipt at each commit.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import run_command
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
from gpu_fault_release.regional_resource_probe import (
    ProbeState,
    ResourceRef,
    probe_resource,
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
    "verify_failure_domain_publication",
]

CONTROL_WORKER_DEPLOYMENT = "gpu-fault-control-worker"
# Six replicas rolling one at a time with readiness at 5 s intervals settle well
# inside this; a worker that cannot become Ready is a failed command, not a hang.
CONTROL_WORKER_ROLLOUT_TIMEOUT_SECONDS = 600
# The HyperPod cluster label the release engine filters its node inventory on
# (``regional_release_gpu_rollout.gpu_node_command``); the admin-side render
# selects on it too so both produce one document, one digest, for one fleet.
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"

NodeLister = Callable[[RenderedSite, str], Mapping[str, Mapping[str, Any]]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class _ResourceReader:
    def __init__(self, run: CommandRunner) -> None:
        self.run = run

    def probe_output(
        self, arguments: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        result = self.run(
            arguments, timeout_seconds=timeout_seconds or 30, capture=True
        )
        return (
            result.returncode,
            result.stdout or "",
            diagnostic_text(result.stderr or "", sensitive=True),
        )


@dataclass(frozen=True)
class FailureDomainMapResult:
    mapping: dict[str, dict[str, str]]
    unmapped: dict[str, list[str]] = field(default_factory=dict)
    label_keys: tuple[str, ...] = FAILURE_DOMAIN_LABELS
    node_uids: dict[str, dict[str, str]] = field(default_factory=dict)
    worker_uid: str | None = None

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
    site: RenderedSite, cluster_id: str, *, run: CommandRunner | None = None
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
    completed = (run or run_command)(command, capture=True, timeout_seconds=120)
    if completed.returncode:
        raise BootstrapError(
            f"failure-domain map cannot read GPU nodes for {cluster_id}: "
            f"{diagnostic_text(completed.stderr or '', sensitive=True)}"
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
    nodes: dict[str, dict[str, Any]] = {}
    for item in items:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        labels = metadata.get("labels") if isinstance(metadata, dict) else None
        if (
            not isinstance(metadata, dict)
            or item.get("kind") != "Node"
            or not isinstance(metadata.get("name"), str)
            or not metadata["name"]
            or metadata["name"] in nodes
            or not isinstance(metadata.get("uid"), str)
            or not metadata["uid"]
            or not isinstance(labels, dict)
            or hyperpod
            and labels.get(HYPERPOD_CLUSTER_LABEL) != hyperpod
        ):
            raise BootstrapError(
                "failure-domain map received an incomplete or conflicting node identity"
            )
        nodes[metadata["name"]] = item
    return nodes


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
    node_uids: dict[str, dict[str, str]] = {}
    for cluster_id in selected:
        inventory = lister(site, cluster_id)
        node_uids[cluster_id] = {
            name: str(node["metadata"]["uid"]) for name, node in inventory.items()
        }
        mapped = failure_domain_map(
            cluster_id, list(inventory.values()), label_keys=keys
        )[cluster_id]
        mapping[cluster_id] = mapped
        missing = sorted(set(inventory) - set(mapped))
        if missing:
            unmapped[cluster_id] = missing
    return FailureDomainMapResult(
        mapping=mapping, unmapped=unmapped, label_keys=keys, node_uids=node_uids
    )


def _cpu_kubectl(site: RenderedSite) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
    ]


def _run(run: CommandRunner, arguments: list[str], **kwargs: Any) -> str:
    completed = run(arguments, capture=True, **kwargs)
    if completed.returncode:
        raise BootstrapError(
            "failure-domain map apply failed: "
            + diagnostic_text(
                completed.stderr or completed.stdout or "", sensitive=True
            )
        )
    return str(completed.stdout or "")


def apply_failure_domain_map(
    site: RenderedSite,
    *,
    run: CommandRunner | None = None,
    allow_absent_worker: bool = False,
    prepared: FailureDomainMapResult | None = None,
) -> FailureDomainMapResult:
    """Render the map for the managed site, validate it, ship it, roll on change.

    The ConfigMap is applied to the CPU control plane and the control-worker's
    pod template is stamped with the document digest; Kubernetes rolls the
    worker only when that annotation actually changed. A worker Deployment
    may be absent only for an explicitly declared bootstrap call. Managed
    membership changes require the worker and wait for the new template.
    A batch may supply its journaled, call-local node snapshot as ``prepared``.
    """

    active_run = run or run_command
    result = prepared or build_failure_domain_map(
        site,
        list_nodes=lambda current, cluster_id: cluster_nodes(
            current, cluster_id, run=active_run
        ),
    )
    if set(result.mapping) != {
        str(item["cluster_id"]) for item in site.release_config["clusters"]
    } or result.label_keys != site_failure_domain_labels(site):
        raise BootstrapError("failure-domain map preparation differs from the site")
    manifest = failure_domain_configmap(
        result.mapping, namespace=str(site.release_config["namespace"])
    )
    try:
        validate_failure_domain_manifest(manifest)
    except ValueError as exc:
        raise BootstrapError(f"failure-domain map render is invalid: {exc}") from exc
    kubectl = _cpu_kubectl(site)
    worker = probe_resource(
        _ResourceReader(active_run),
        kubectl[:-2],
        ResourceRef(
            "deployment",
            "Deployment",
            CONTROL_WORKER_DEPLOYMENT,
            str(site.release_config["namespace"]),
        ),
    )
    if worker.state is ProbeState.ERROR:
        raise BootstrapError("failure-domain map cannot verify the control-worker")
    if worker.state is ProbeState.ABSENT and not allow_absent_worker:
        raise BootstrapError("failure-domain map control-worker is missing")
    current = worker.document
    metadata = current.get("metadata") if current is not None else None
    if current is not None and (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("resourceVersion"), str)
        or not metadata["resourceVersion"]
        or metadata.get("deletionTimestamp")
    ):
        raise BootstrapError("failure-domain map worker identity is incomplete")
    current_annotations: dict[str, Any] = {}
    if current is not None:
        spec = current.get("spec")
        template = spec.get("template") if isinstance(spec, dict) else None
        template_metadata = (
            template.get("metadata") if isinstance(template, dict) else None
        )
        if not isinstance(template_metadata, dict):
            raise BootstrapError("failure-domain map worker template is invalid")
        annotations = template_metadata.get("annotations")
        annotations = {} if annotations is None else annotations
        if not isinstance(annotations, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in annotations.items()
        ):
            raise BootstrapError("failure-domain map worker annotations are invalid")
        current_annotations = annotations
    _run(
        active_run,
        [*kubectl, "apply", "-f", "-"],
        input_text=json.dumps(manifest),
        timeout_seconds=120,
    )
    if current is not None:
        assert isinstance(metadata, dict)
        digest = failure_domain_map_sha256(manifest)
        patch = {
            "metadata": {
                "uid": metadata["uid"],
                "resourceVersion": metadata["resourceVersion"],
            },
            "spec": {
                "template": {
                    "metadata": {"annotations": {FAILURE_DOMAIN_MAP_ANNOTATION: digest}}
                }
            },
        }
        if current_annotations.get(FAILURE_DOMAIN_MAP_ANNOTATION) != digest:
            _run(
                active_run,
                [
                    *kubectl,
                    "patch",
                    "deployment",
                    CONTROL_WORKER_DEPLOYMENT,
                    "--type=merge",
                    "-p",
                    json.dumps(patch),
                ],
                timeout_seconds=120,
            )
        _run(
            active_run,
            [
                *kubectl,
                "rollout",
                "status",
                f"deployment/{CONTROL_WORKER_DEPLOYMENT}",
                f"--timeout={CONTROL_WORKER_ROLLOUT_TIMEOUT_SECONDS}s",
            ],
            timeout_seconds=660,
        )
        final_worker = probe_resource(
            _ResourceReader(active_run),
            kubectl[:-2],
            ResourceRef(
                "deployment",
                "Deployment",
                CONTROL_WORKER_DEPLOYMENT,
                str(site.release_config["namespace"]),
            ),
        )
        if (
            final_worker.state is not ProbeState.PRESENT
            or final_worker.document is None
        ):
            raise BootstrapError(
                "failure-domain map worker is unavailable after rollout"
            )
        observed = final_worker.document
        observed_metadata = observed.get("metadata")
        observed_spec = observed.get("spec")
        template = (
            observed_spec.get("template") if isinstance(observed_spec, dict) else None
        )
        template_metadata = (
            template.get("metadata") if isinstance(template, dict) else None
        )
        annotations = (
            template_metadata.get("annotations")
            if isinstance(template_metadata, dict)
            else None
        )
        if (
            not isinstance(observed_metadata, dict)
            or observed_metadata.get("uid") != metadata["uid"]
            or not isinstance(annotations, dict)
            or annotations.get(FAILURE_DOMAIN_MAP_ANNOTATION) != digest
        ):
            raise BootstrapError("failure-domain map worker changed during rollout")
        result = replace(result, worker_uid=str(metadata["uid"]))
    return result


def verify_failure_domain_publication(
    site: RenderedSite,
    *,
    digest: str,
    worker_uid: str,
    configmap_uid: str | None = None,
    run: CommandRunner | None = None,
) -> dict[str, str]:
    """Recheck a converged publication without GPU LISTs or another rollout."""
    reader = _ResourceReader(run or run_command)
    prefix = _cpu_kubectl(site)[:-2]
    namespace = str(site.release_config["namespace"])

    def read(kind: str, name: str) -> dict[str, Any]:
        result = probe_resource(
            reader, prefix, ResourceRef(kind.lower(), kind, name, namespace)
        )
        if result.state is not ProbeState.PRESENT or result.document is None:
            raise BootstrapError("failure-domain publication cannot be read")
        return dict(result.document)

    manifest = read("ConfigMap", FAILURE_DOMAIN_CONFIGMAP)
    uid = manifest["metadata"]["uid"]
    if failure_domain_map_sha256(manifest) != digest or (
        configmap_uid is not None and configmap_uid != uid
    ):
        raise BootstrapError("failure-domain map changed after publication")
    worker = read("Deployment", CONTROL_WORKER_DEPLOYMENT)
    metadata = worker.get("metadata") or {}
    spec = worker.get("spec") or {}
    status = worker.get("status") or {}
    annotations = ((spec.get("template") or {}).get("metadata") or {}).get(
        "annotations"
    ) or {}
    replicas = spec.get("replicas")
    if (
        metadata.get("uid") != worker_uid
        or metadata.get("deletionTimestamp")
        or annotations.get(FAILURE_DOMAIN_MAP_ANNOTATION) != digest
        or type(replicas) is not int
        or replicas <= 0
        or type(metadata.get("generation")) is not int
        or status.get("observedGeneration") != metadata["generation"]
        or any(
            status.get(key) != replicas
            for key in (
                "replicas",
                "updatedReplicas",
                "readyReplicas",
                "availableReplicas",
            )
        )
        or status.get("unavailableReplicas", 0) != 0
    ):
        raise BootstrapError("failure-domain worker publication is not converged")
    confirmed = read("ConfigMap", FAILURE_DOMAIN_CONFIGMAP)
    if (
        confirmed["metadata"]["uid"] != uid
        or failure_domain_map_sha256(confirmed) != digest
    ):
        raise BootstrapError(
            "failure-domain map changed during convergence verification"
        )
    return {"map_sha256": digest, "worker_uid": worker_uid, "configmap_uid": str(uid)}


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
