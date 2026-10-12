"""One cluster's GPU node scope: what the site records against what it serves.

A cluster's node selector has to say the same thing in three places. The site
document (``nodeSelector`` / ``nodeSelectorSource``) is the one source of
truth, written by ``join-cluster`` and ``node-scope --converge``. The
per-cluster connection Secret (``node-selector`` / ``node-selector-source``)
is what the data plane reads into ``GPU_FAULT_NODE_SELECTOR`` at Pod start.
And the Pod templates of the three Deployments that consume the Secret -- the
cluster executor, the Completion Watcher and the node installer reconciler --
carry :data:`NODE_SCOPE_ANNOTATION`, the digest of the scope they were
rendered for, so a changed scope moves the template and Kubernetes restarts
the Pods, which otherwise keep the environment they started with.

The release diff used to compare only the cluster *membership* digest (ids,
ARNs, namespaces, CIDRs, provider), so a converged selector in ``site.yaml``
classified NOOP and the data planes kept serving the legacy selector from
Secrets rendered at join time, while ``status --full`` and the coverage checks
already selected on the label. This module closes that gap in three ways:

* :data:`RECORDED_SCOPES_INPUT` -- the digest of every cluster's recorded
  scope is a release input, persisted in the release state, so a changed
  entry is a ``DATA_PLANE_COMPATIBLE`` release that re-renders the Secret
  (``ReleaseComponent.CONNECTION``) and rolls the consumers;
* :data:`SERVED_SCOPE_DRIFT_INPUT` -- a classification-time probe of the
  Secrets, so a state recorded before the input existed, or a Secret that
  disagrees for any other reason, is rolled too;
* :func:`check_node_scope_agreement` -- the ``gpu_cluster:<id>`` verify check
  refuses a cluster whose Secret or consumers disagree with the site, naming
  the cluster and both values.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Mapping

from gpu_fault.cluster_scope import (
    CONNECTION_SECRET_NODE_SELECTOR_KEY,
    CONNECTION_SECRET_NODE_SELECTOR_SOURCE_KEY,
    node_selector,
    node_selector_source,
)
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError

__all__ = [
    "CONNECTION_SECRET",
    "NODE_SCOPE_ANNOTATION",
    "NODE_SCOPE_CONSUMERS",
    "RECORDED_SCOPES_DIGEST_STATE_KEY",
    "RECORDED_SCOPES_INPUT",
    "RECORDED_SCOPES_STATE_KEY",
    "SERVED_SCOPE_DRIFT_INPUT",
    "NodeScope",
    "check_node_scope_agreement",
    "cluster_node_scopes",
    "cluster_node_scopes_digest",
    "connection_secret_data",
    "consumer_node_scope_annotation",
    "node_scope_digest",
    "node_scope_drift",
    "recorded_node_scope",
    "scopes_disagree",
    "served_node_scope",
    "stamp_node_scope_annotation",
    "stamp_reconciler_node_scope",
]

#: The per-cluster Secret every data-plane Deployment mounts.
CONNECTION_SECRET = "gpu-fault-regional-connection"
#: Pod-template annotation: ``node_scope_digest`` of the scope a consumer
#: Deployment was rendered for. A changed scope changes the template.
NODE_SCOPE_ANNOTATION = "gpu-fault.io/node-scope-sha256"
#: The Deployments whose Pods read ``GPU_FAULT_NODE_SELECTOR`` from the
#: connection Secret (``deploy/dataplane/*.yaml``); the node-resource
#: collector reads the Secret but not the selector.
NODE_SCOPE_CONSUMERS: tuple[str, ...] = (
    inventory.GPU_EXECUTOR_DEPLOYMENT,
    inventory.GPU_WATCHER_DEPLOYMENT,
    inventory.GPU_RECONCILER_DEPLOYMENT,
)
#: Release-diff input: the digest of every cluster's recorded scope changed.
RECORDED_SCOPES_INPUT = "cluster_node_scopes"
#: Release-diff input: a cluster's connection Secret serves another scope.
SERVED_SCOPE_DRIFT_INPUT = "cluster_node_scope_drift"
#: Release-state keys: the recorded scopes and their digest, written by every
#: state checkpoint so the next release's diff reads what this one rolled.
RECORDED_SCOPES_STATE_KEY = "cluster_node_scopes"
RECORDED_SCOPES_DIGEST_STATE_KEY = "cluster_node_scopes_sha256"

NodeScope = dict[str, str]


def recorded_node_scope(target: object) -> NodeScope:
    """``{selector, source}`` the site records for ``target``.

    ``target`` is anything :func:`gpu_fault.cluster_scope.node_selector`
    reads: a ``ClusterTarget``, a site entry or a rendered release-config
    entry. A partial double without a resolvable selector (the release diff
    tests) records an empty one; a real ``ClusterTarget`` resolved its
    selector when the configuration loaded and never reaches that branch.
    """

    try:
        selector = node_selector(target)
    except ValueError:
        selector = ""
    # The source is judged on the *recorded* text (a HyperPod entry without a
    # selector is the legacy default), not on the resolved label.
    return {"selector": selector, "source": node_selector_source(target).value}


def _decoded(data: Mapping[str, Any] | None, key: str) -> str:
    value = (data or {}).get(key)
    if not isinstance(value, str) or not value:
        return ""
    try:
        return base64.b64decode(value, validate=True).decode("utf-8").strip()
    except (binascii.Error, UnicodeDecodeError):
        return ""


def _provider_default_selector(target: object) -> str:
    """The selector the data plane runs without a ``node-selector`` key."""

    try:
        return node_selector(target, selector="")
    except ValueError:
        # eks-managed: no default; the manifests' optional key is required.
        return ""


def served_node_scope(data: Mapping[str, Any] | None, target: object) -> NodeScope:
    """``{selector, source}`` the connection Secret ``data`` serves ``target``.

    ``data`` is the Secret's ``data`` as ``kubectl get -o json`` returns it
    (base64 values). Both keys are optional on the manifests, so a Secret
    rendered before they existed serves the data plane's defaults: the
    HyperPod cluster label with the ``provider-default-legacy`` source. A
    Secret with the selector but no source key (rendered between the two keys
    appearing) derives the source as the data plane does: the provider default
    is the legacy default, the ownership label is the convention, anything
    else came from the retired flags. An ``eks-managed`` cluster has no default
    selector; a Secret without the key serves an empty one, which never agrees
    with a recorded scope.
    """

    default = _provider_default_selector(target)
    selector = _decoded(data, CONNECTION_SECRET_NODE_SELECTOR_KEY) or default
    source = _decoded(data, CONNECTION_SECRET_NODE_SELECTOR_SOURCE_KEY)
    if not source:
        source = node_selector_source(
            target, selector="" if selector == default else selector, source=""
        ).value
    return {"selector": selector, "source": source}


def scopes_disagree(
    recorded: Mapping[str, str],
    served: Mapping[str, str],
    data: Mapping[str, Any] | None,
) -> bool:
    """Whether the Secret ``data`` (serving ``served``) disagrees with ``recorded``.

    The selector must agree; the source, an audit key the data plane never
    reads, must agree when the Secret records one. A Secret that predates the
    source key is judged on the selector it serves, so a legacy HyperPod site
    is not rolled for a missing audit key.
    """

    if str(recorded.get("selector") or "") != str(served.get("selector") or ""):
        return True
    return bool(_decoded(data, CONNECTION_SECRET_NODE_SELECTOR_SOURCE_KEY)) and str(
        recorded.get("source") or ""
    ) != str(served.get("source") or "")


def node_scope_digest(scope: Mapping[str, str]) -> str:
    """The digest a consumer's Pod template carries for ``scope``."""

    return hashlib.sha256(
        json.dumps(
            {
                "selector": str(scope.get("selector") or ""),
                "source": str(scope.get("source") or ""),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def cluster_node_scopes(config: object) -> dict[str, NodeScope]:
    """``{cluster_id: {selector, source}}`` recorded for every cluster."""

    clusters = getattr(config, "clusters", None) or ()
    return {
        str(target.cluster_id): recorded_node_scope(target)
        for target in sorted(clusters, key=lambda item: str(item.cluster_id))
    }


def cluster_node_scopes_digest(config: object) -> str:
    """One digest over every cluster's recorded scope (the release input)."""

    return hashlib.sha256(
        json.dumps(
            cluster_node_scopes(config), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def connection_secret_data(release: Any, target: Any) -> dict[str, Any] | None:
    """The connection Secret's ``data`` for ``target``, or ``None`` when absent.

    Read through the release's cached JSON reader, so a verify or status
    report that already opened a read snapshot pays one ``kubectl get`` per
    cluster. Only the two selector keys are ever decoded by the callers; the
    token stays encoded and is never printed.
    """

    value = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "secret",
            CONNECTION_SECRET,
            "--ignore-not-found",
        )
    )
    if not value:
        return None
    data = value.get("data")
    return dict(data) if isinstance(data, dict) else {}


def node_scope_drift(release: Any) -> dict[str, dict[str, NodeScope]]:
    """Clusters whose connection Secret serves another scope than the site records.

    A classification-time probe like the observability, Aurora-refresh and
    CPU-replica-profile probes: a dry run, a release without a delivered
    control plane, or a bare test double without a cluster reader never reads
    a cluster and reports no drift. A cluster without the Secret is not yet
    bootstrapped; the bootstrap or join renders it and it is not drift.
    """

    runner = getattr(release, "runner", None)
    config = getattr(release, "config", None)
    if runner is None or getattr(runner, "dry_run", False):
        return {}
    if not (getattr(config, "delivery_component_digests", None) or {}).get("cpu"):
        return {}
    if not callable(getattr(release, "_get_json", None)) or not callable(
        getattr(release, "_gpu", None)
    ):
        return {}
    drift: dict[str, dict[str, NodeScope]] = {}
    for target in getattr(config, "clusters", None) or ():
        data = connection_secret_data(release, target)
        if data is None:
            continue
        recorded = recorded_node_scope(target)
        served = served_node_scope(data, target)
        if scopes_disagree(recorded, served, data):
            drift[str(target.cluster_id)] = {"recorded": recorded, "served": served}
    return drift


def stamp_node_scope_annotation(
    document: dict[str, Any], scope: Mapping[str, str]
) -> None:
    """Record ``scope``'s digest on a Deployment document's Pod template."""

    annotations = (
        document.setdefault("spec", {})
        .setdefault("template", {})
        .setdefault("metadata", {})
        .setdefault("annotations", {})
    )
    annotations[NODE_SCOPE_ANNOTATION] = node_scope_digest(scope)


def consumer_node_scope_annotation(document: Mapping[str, Any]) -> str | None:
    """The scope digest a live Deployment's Pod template carries, if any.

    ``None`` for a Deployment rendered before the stamp existed: what its Pods
    run cannot be told from the template, and the next release that rolls it
    stamps it.
    """

    template = (document.get("spec") or {}).get("template") or {}
    annotations = (template.get("metadata") or {}).get("annotations") or {}
    value = annotations.get(NODE_SCOPE_ANNOTATION)
    return str(value) if value else None


def stamp_reconciler_node_scope(
    release: Any, target: Any, scope: Mapping[str, str]
) -> None:
    """Record ``scope`` on the reconciler's Pod template after the script applied it.

    The deploy script renders the reconciler Deployment; the engine owns the
    stamp, as it does for the executor and the watcher it renders itself. A
    merge patch with the value already present changes nothing; a different
    value moves the template and the ``Recreate`` strategy restarts the Pod,
    which re-reads the connection Secret.
    """

    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {NODE_SCOPE_ANNOTATION: node_scope_digest(scope)}
                }
            }
        }
    }
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "patch",
            "deployment",
            inventory.GPU_RECONCILER_DEPLOYMENT,
            "--type",
            "merge",
            "-p",
            json.dumps(patch, separators=(",", ":")),
        )
    )


def check_node_scope_agreement(
    release: Any, target: Any, *, recorded: Mapping[str, str]
) -> dict[str, Any]:
    """The served scope and the consumers' stamps; refuse any disagreement.

    Returns the ``status --full`` details: ``connection_secret`` (what the
    Secret serves, with whether the selector key is present) and
    ``consumers`` (each consumer Deployment's stamp, ``None`` when it predates
    the stamp). Raises ``ReleaseError`` naming the cluster and both values
    when the Secret serves another scope than the site records, or when a
    stamped consumer was rendered for another scope than the site records.
    """

    cluster_id = str(target.cluster_id)
    data = connection_secret_data(release, target)
    served = served_node_scope(data, target)
    expected: NodeScope = {
        "selector": str(recorded.get("selector") or ""),
        "source": str(recorded.get("source") or ""),
    }
    details: dict[str, Any] = {
        "connection_secret": {
            **served,
            "selector_recorded": bool(
                _decoded(data, CONNECTION_SECRET_NODE_SELECTOR_KEY)
            ),
        }
    }
    if scopes_disagree(expected, served, data):
        raise ReleaseError(
            f"{cluster_id} node scope disagreement: site.yaml records selector "
            f"{expected['selector']!r} (source {expected['source']}); the "
            f"connection Secret {CONNECTION_SECRET} serves selector "
            f"{served['selector']!r} (source {served['source']}); gpu-fault-admin "
            "deploy re-renders the Secret and restarts its consumers"
        )
    digest = node_scope_digest(expected)
    consumers: dict[str, str | None] = {}
    for name in NODE_SCOPE_CONSUMERS:
        document = release._get_json(
            release._gpu(
                target, "-n", release.config.namespace, "get", "deployment", name
            )
        )
        stamp = consumer_node_scope_annotation(document)
        consumers[name] = stamp
        if stamp is not None and stamp != digest:
            raise ReleaseError(
                f"{cluster_id} {name} Pods were rendered for node scope "
                f"{stamp[:12]} while site.yaml records {digest[:12]} (selector "
                f"{expected['selector']!r}, source {expected['source']}); "
                "gpu-fault-admin deploy rolls the consumers"
            )
    details["consumers"] = consumers
    return details
