"""GF-GENERAL-LIVE-014: the device plugins a variant targets, and how each one is
told apart on the cluster the variant runs against.

``general_live_device_plugin`` keeps the variants, the fixture and the verdicts;
this module carries the two :class:`PluginTarget` definitions, the provider of
the target cluster, and the DaemonSet selection with its cross-check.

The EFA plugin's DaemonSet name is provider-specific
(``node_provider.EFA_DEVICE_PLUGIN_NAMES``: ``hyperpod-dependencies-aws-efa-
k8s-device-plugin`` on HyperPod, ``aws-efa-k8s-device-plugin`` on plain EKS).
The product resolves it from its own ``GPU_FAULT_CLUSTER_PROVIDER`` -- every GPU
cluster's Deployments carry the variable -- but the harness process runs on the
coordinator host, where the variable is absent and the product helper's default
is HyperPod: ``efa_device_plugin_name()`` read bare there asked the plain-EKS
cluster for the HyperPod name, and ``efa_plugin`` ended BLOCKED "expected one
hyperpod-dependencies-aws-efa-k8s-device-plugin DaemonSet on the cluster, found
0" (live 2026-10-27 10:53Z, plain-EKS lane; the same variant reached the freeze
on HyperPod). :func:`target_cluster_provider` therefore derives the provider
from the target node's own labels -- the test the lane preflight already applies
(``general_live_fixture.node_provider``) -- and :func:`efa_target` hands it to
the product helper as an explicit environment. A product-side
``GPU_FAULT_EFA_DEVICE_PLUGIN_NAME`` override is not read: the harness has no
reader of the executor's effective environment, so the per-provider default is
what it expects, and the cross-check below names the disagreement when the
cluster's DaemonSets say otherwise.

Selecting the DaemonSet (:func:`plugin_daemonset`): scheduled, not a companion
(the upstream k8s-device-plugin chart's ``...-gpu-feature-discovery`` and
``...-mps-control-daemon`` siblings share the plugin's labels and, since v0.15,
its image; only the kubelet device-plugins mount tells them apart --
``collector_device_plugin_fixture.companion_daemonset``), naming the target's
token, and carrying the Pod label the product's restart selects by
(``name=<EFA DaemonSet name>`` / ``app.kubernetes.io/name=nvidia-device-plugin``,
``adapters/kubernetes/node_operations._restart_device_plugin``) -- exactly one.
That one must then carry the plugin's provider-independent identity: the
``app.kubernetes.io/name`` value both providers' charts stamp
(``aws-efa-k8s-device-plugin`` / ``nvidia-device-plugin``) or the container
image basename. A refusal lists the expected name and every DaemonSet actually
carrying the identity, so a wrong provider or a renamed chart reads as such.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from gpu_fault.cluster_scope import PROVIDER_ENV, ClusterProvider
from gpu_fault.node_provider import efa_device_plugin_name
from scripts.e2e.regional.collector_device_plugin_fixture import companion_daemonset
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    node_provider,
)

#: The chart label whose value is the plugin's identity on every provider.
IDENTITY_LABEL = "app.kubernetes.io/name"
#: The Pod label the product's ``RESTART_EFA_DEVICE_PLUGIN`` selects by
#: (``name=<efa_device_plugin_name()>``); the GPU restart selects by
#: ``IDENTITY_LABEL``.
EFA_SELECTOR_LABEL = "name"
EFA_IDENTITY = "aws-efa-k8s-device-plugin"
GPU_IDENTITY = "nvidia-device-plugin"
GPU_IMAGE = "k8s-device-plugin"

#: The lane the preflight reads off the node's labels -> the product's provider.
LANE_PROVIDERS: dict[str, ClusterProvider] = {
    "hp": ClusterProvider.HYPERPOD,
    "eks": ClusterProvider.EKS_MANAGED,
}


@dataclass(frozen=True)
class PluginTarget:
    """One device plugin: how it is discovered and how the product grades it."""

    kind: str  # "gpu" | "efa"
    resource: str  # the kubelet resource name
    daemonset_token: str  # the substring that identifies its DaemonSet
    binaries: tuple[str, ...]  # candidate process basenames (probe allow-list)
    restart_operation: str
    validate_operation: str
    #: The plugin's socket under ``/var/lib/kubelet/device-plugins`` (verified
    #: live on both clusters); the liveness sample's ``sockets`` must name it.
    socket_name: str
    #: The Pod label the product's restart selects by; the DaemonSet's Pod
    #: template must carry it, or the product would find nothing to restart.
    selector_label: tuple[str, str]
    #: The chart's provider-independent identity: the ``IDENTITY_LABEL`` value
    #: and the container image basename (either one confirms the match).
    identity_name: str
    image_basename: str
    #: Where ``daemonset_token`` came from (evidence and refusal reasons).
    name_source: str

    @property
    def liveness_metric(self) -> str:
        return f"{self.kind}_device_plugin_unresponsive"

    @property
    def mismatch_metric(self) -> str:
        return f"{self.kind}_kubernetes_allocatable_mismatch"

    @property
    def allocatable_metric(self) -> str:
        return f"{self.kind}_kubernetes_allocatable_count"

    @property
    def expected_metric(self) -> str:
        return f"{self.kind}_kubernetes_expected_count"

    @property
    def metrics(self) -> set[str]:
        return {
            self.liveness_metric,
            self.mismatch_metric,
            self.allocatable_metric,
            self.expected_metric,
            f"{self.kind}_kubernetes_allocatable_missing_count",
        }

    @property
    def selector(self) -> str:
        key, value = self.selector_label
        return f"{key}={value}"


GPU_TARGET = PluginTarget(
    kind="gpu",
    resource="nvidia.com/gpu",
    daemonset_token=GPU_IDENTITY,
    binaries=("nvidia-device-plugin",),
    restart_operation="RESTART_GPU_DEVICE_PLUGIN",
    validate_operation="VALIDATE_GPU",
    socket_name="nvidia-gpu.sock",
    selector_label=(IDENTITY_LABEL, GPU_IDENTITY),
    identity_name=GPU_IDENTITY,
    image_basename=GPU_IMAGE,
    name_source="the k8s-device-plugin chart's name token, the same on both providers",
)


def efa_target(provider: ClusterProvider) -> PluginTarget:
    """The EFA plugin target on a ``provider`` cluster.

    The DaemonSet name is the product's own per-provider default
    (``node_provider.efa_device_plugin_name``) resolved with ``provider`` as an
    explicit environment -- never from this process's own, which on the
    coordinator host carries no ``GPU_FAULT_CLUSTER_PROVIDER`` and so reads as
    HyperPod on every lane.
    """

    name = efa_device_plugin_name({PROVIDER_ENV: provider.value})
    return PluginTarget(
        kind="efa",
        resource="vpc.amazonaws.com/efa",
        daemonset_token=name,
        # The process basename differs by provider: the HyperPod EFA plugin's
        # container runs ``efa-k8s-device-plugin`` although its Pod command/
        # image basename is ``aws-efa-k8s-device-plugin`` (live refusal record,
        # hp lane 2026-10-27); the probe signals only the intersection of this
        # request with its own static allow-list.
        binaries=(
            "efa-k8s-device-plugin",
            "aws-efa-k8s-device-plugin",
            "k8s-device-plugin",
            "device-plugin",
        ),
        restart_operation="RESTART_EFA_DEVICE_PLUGIN",
        validate_operation="VALIDATE_FABRIC",
        socket_name="aws-efa-device-plugin.sock",
        selector_label=(EFA_SELECTOR_LABEL, name),
        identity_name=EFA_IDENTITY,
        image_basename=EFA_IDENTITY,
        name_source=(
            f"the product's {provider.value} default, {PROVIDER_ENV}="
            f"{provider.value} derived from the target node's labels"
        ),
    )


def target_cluster_provider(ctx: VariantContext) -> ClusterProvider:
    """The provider of the cluster the variant targets, from the target node.

    The lane preflight classified the node by the same labels
    (``general_live_fixture.node_provider``: the HyperPod compute-type label,
    else the EKS node-group label); a node carrying neither cannot be aimed at
    a provider-specific DaemonSet and is refused before any mutation.
    """

    labels = ctx.regional.node_metadata(ctx.settings.node).get("labels") or {}
    provider = LANE_PROVIDERS.get(node_provider(labels) or "")
    if provider is None:
        raise VariantBlocked(
            "FIXTURE_NOT_READY",
            f"{ctx.settings.node} carries neither the HyperPod compute-type label "
            "nor the EKS node-group label, so its cluster's provider -- and with "
            "it the EFA device-plugin DaemonSet name -- cannot be derived",
        )
    return provider


# --------------------------------------------------------------------------- #
# DaemonSet selection
# --------------------------------------------------------------------------- #
def image_basename(image: str) -> str:
    """``registry/repo/name:tag@digest`` -> ``name``."""

    return image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]


def daemonset_ref(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return f"{metadata.get('namespace')}/{metadata.get('name')}"


def daemonset_labels(item: dict[str, Any]) -> dict[str, str]:
    """The DaemonSet's labels under its Pod template's (the Pods carry those)."""

    metadata = item.get("metadata") or {}
    template = ((item.get("spec") or {}).get("template") or {}).get("metadata") or {}
    return {
        str(key): str(value)
        for source in (metadata.get("labels") or {}, template.get("labels") or {})
        for key, value in source.items()
    }


def daemonset_images(item: dict[str, Any]) -> list[str]:
    spec = ((item.get("spec") or {}).get("template") or {}).get("spec") or {}
    return [
        str(container["image"])
        for container in spec.get("containers") or []
        if isinstance(container, dict) and container.get("image")
    ]


def _scheduled(item: dict[str, Any]) -> int:
    return int((item.get("status") or {}).get("desiredNumberScheduled") or 0)


def carries_identity(item: dict[str, Any], target: PluginTarget) -> bool:
    """The chart's identity label or the plugin's image basename."""

    if daemonset_labels(item).get(IDENTITY_LABEL) == target.identity_name:
        return True
    return any(
        image_basename(image) == target.image_basename
        for image in daemonset_images(item)
    )


def carries_selector(item: dict[str, Any], target: PluginTarget) -> bool:
    """The Pod template carries the label the product's restart selects by."""

    key, value = target.selector_label
    return daemonset_labels(item).get(key) == value


def describe_daemonset(item: dict[str, Any], target: PluginTarget) -> str:
    """``namespace/name (images; why it was not selected)`` for a refusal."""

    notes = [", ".join(daemonset_images(item)) or "no image"]
    if _scheduled(item) < 1:
        notes.append("0 scheduled")
    if companion_daemonset(item):
        notes.append("companion: no kubelet device-plugins mount")
    if not carries_selector(item, target):
        notes.append(f"without the product's selector label {target.selector}")
    return f"{daemonset_ref(item)} ({'; '.join(notes)})"


def discovery_refusal(
    target: PluginTarget,
    matches: list[dict[str, Any]],
    carrying: list[dict[str, Any]],
) -> str:
    """Why no DaemonSet was selected, naming the expected one and the present ones."""

    if len(matches) != 1:
        found = f"found {len(matches)}"
        if matches:
            found += ": " + ", ".join(daemonset_ref(item) for item in matches)
        head = (
            f"expected one {target.daemonset_token} DaemonSet ({target.name_source}) "
            f"whose Pod template carries {target.selector} on the cluster, {found}"
        )
    else:
        item = matches[0]
        head = (
            f"{daemonset_ref(item)} is the one {target.daemonset_token} DaemonSet but "
            f"carries neither {IDENTITY_LABEL}={target.identity_name} nor an image "
            f"named {target.image_basename} (labels "
            f"{json.dumps(daemonset_labels(item), sort_keys=True)}; images "
            f"{daemonset_images(item)}), so it is not the {target.kind} device plugin"
        )
    present = ", ".join(describe_daemonset(item, target) for item in carrying)
    return (
        f"{head}; DaemonSets carrying the {target.kind} plugin's identity "
        f"({IDENTITY_LABEL}={target.identity_name} or image {target.image_basename}): "
        f"{present or 'none'}"
    )


def plugin_daemonset(
    items: list[dict[str, Any]], target: PluginTarget
) -> dict[str, Any]:
    """The one DaemonSet ``target`` names on the cluster, cross-checked.

    Scheduled, not a companion, naming ``target.daemonset_token`` and carrying
    the product's selector label -- exactly one -- and that one carrying the
    plugin's identity. Anything else is ``FIXTURE_NOT_READY`` with the expected
    name, what was found, and every DaemonSet actually carrying the identity,
    so the reason reads on its own (a wrong provider shows as the other
    provider's DaemonSet present under the expected identity).
    """

    candidates = [
        item for item in items if _scheduled(item) > 0 and not companion_daemonset(item)
    ]
    matches = [
        item
        for item in candidates
        if target.daemonset_token.lower() in json.dumps(item, sort_keys=True).lower()
        and carries_selector(item, target)
    ]
    carrying = [item for item in items if carries_identity(item, target)]
    if len(matches) == 1 and carries_identity(matches[0], target):
        return matches[0]
    raise VariantBlocked(
        "FIXTURE_NOT_READY", discovery_refusal(target, matches, carrying)
    )


# --------------------------------------------------------------------------- #
# process basenames the probe may signal
# --------------------------------------------------------------------------- #
#: Launcher/shim basenames that are never the plugin itself; a ``/bin/sh -c``
#: entrypoint must not widen the set of processes the probe may signal.
_BINARY_NOISE = frozenset(
    {"sh", "bash", "dash", "ash", "env", "tini", "dumb-init", "pause", "exec"}
)
_SAFE_BINARY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def plugin_binary_candidates(pod: dict[str, Any]) -> tuple[str, ...]:
    """Process basenames the plugin container might run, from the Pod spec.

    The probe's static allow-list is provider-blind; these are the discovered
    Pod's *own* declared command and image basenames, passed as
    ``--pod-command-binary`` so the probe matches the real plugin process (the
    HyperPod EFA plugin's basename is none of the static names) while still
    requiring the pod-UID cgroup match. Shim names are dropped.
    """

    candidates: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        base = value.rsplit("/", 1)[-1]
        if base and base not in _BINARY_NOISE and _SAFE_BINARY.fullmatch(base):
            if base not in candidates:
                candidates.append(base)

    spec = pod.get("spec") or {}
    for container in spec.get("containers") or []:
        command = container.get("command") or []
        if command:
            add(command[0])
        image = container.get("image")
        if isinstance(image, str):
            add(image_basename(image))
    return tuple(candidates)
