"""The installer identity a release puts on a GPU node, as the fleet sees it.

Two parties decide whether a node runs the release:

* the fleet deployment calls a node READY when its Agent's heartbeat reports
  the nine-field identity the release asks for (``FleetDeploymentRequest``:
  protocol, version, artifact, compatibility, bundle, template, policy,
  Runtime Profile, config digest), and only the nodes that are not READY get a
  wave -- the one-node canary, the wave safety gate and lease, the
  failure-domain cap and the per-wave convergence wait;
* the node installer reconciler re-installs every node whose installer
  identity (its ``gpu-fault.io/installer-*`` annotations) differs from its
  own, and that identity also holds the DCGM host engine mode of a plain EKS
  cluster (``INSTALLER_DCGM_HOSTENGINE_ANNOTATION``).

No heartbeat field carried the mode: the template identity was the digest of
the node template's *source* files (``node_template_inputs``) and the Agent
config digest leaves the mode out on purpose
(``node_agent.config.dcgm_hostengine_from_environment``). A release that only
changed the mode -- a node group's ``gpu-fault.io/dcgm-hostengine`` label
re-read by the same source, or the ``dcgm_hostengine_drift`` repair -- opened a
fleet deployment that was SUCCEEDED before its first wave, and the steady-state
reconciler that followed (``*``) re-installed every Agent of the cluster at
once, outside every gate, inside one convergence wait.

``installer_template_identity`` closes it at the identity: the template
identity the release hands the reconciler is the source digest folded with the
cluster's node configuration. The reconciler stamps it on the node, the
installer writes it into the Agent environment
(``GPU_FAULT_NODE_INSTALLER_TEMPLATE_SHA256``), the heartbeat reports it and
the fleet compares it, so a mode change leaves the nodes PENDING and rolls
through the Agent waves like every other node change -- in both directions,
and in a rollback. Only ``host`` folds, the reconciler's own rule for its Job
names and its mode annotation: HyperPod, the embedded engine and every cluster
rendered before the mode existed keep the source digest byte for byte, and with
it the identity their Agents already report. Nothing on the wire changes.

``node_installer_identity_drift`` is the classification half. A release that
re-deploys the reconciler without the Agent waves (an executor, image or
node-scope change) hands it ``*`` at once, so an installed node whose installer
identity differs from the release's -- every Agent of a host-mode cluster
installed before the fold, or a node a refused release left behind -- would be
re-installed outside every gate. The diff plans the Agent waves for it instead.

``pin_installer_version`` keeps the operator's shell out of the identity. The
reconciler script renders the installer Job and the reconciler's node identity
(``gpu-fault.io/installer-version``) with ``GPU_FAULT_VERSION``, falling back
to a default of its own, and the environment it is handed starts from the
deploy host's -- so a value exported there named a version the release never
shipped: the script refuses a bundle ConfigMap without
``gpu-fault-node-installer-<version>.tar.gz``, and every reconciler deploy of
the release failed with "installer ConfigMap is missing the candidate bundle".
The engine now hands it the version of the bundle it actually rolls -- the
candidate's bundle file, a rollback's captured bundle key -- which is the
version every node installed from that bundle carries, whatever the operator
exported.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from gpu_fault.cluster_scope import (
    DcgmHostEngine,
    dcgm_hostengine_environment,
    node_selector,
    selector_matches,
)
from gpu_fault.node_installer_reconciler import INSTALLER_STATE_ANNOTATION
from gpu_fault.node_installer_rendering import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DCGM_HOSTENGINE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
)
from gpu_fault_release.regional_release_images import NodeDependencyTarget
from gpu_fault_release.regional_release_profile_model import (
    target_agent_config_digest,
)

__all__ = [
    "INSTALLER_VERSION_ENV",
    "NODE_INSTALLER_IDENTITY_DRIFT_INPUT",
    "candidate_installer_template_identity",
    "identity_dcgm_hostengine",
    "installed_template_matches",
    "installer_bundle_version",
    "installer_template_identity",
    "node_installer_identity_drift",
    "pin_installer_version",
    "reconciler_template_identity",
    "released_installer_version",
]

#: The diff input of installed nodes the release's reconciler would re-install.
NODE_INSTALLER_IDENTITY_DRIFT_INPUT = "node_installer_identity_drift"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
#: The installer version ``deploy-node-installer-reconciler.sh`` renders the Job
#: and the reconciler's node identity (``installer-version``) with.
INSTALLER_VERSION_ENV = "GPU_FAULT_VERSION"
#: ``build-node-installer-bundle.sh``'s bundle file, which the installer Job
#: unpacks by the same version.
INSTALLER_BUNDLE_NAME = re.compile(
    r"gpu-fault-node-installer-(?P<version>[0-9A-Za-z][0-9A-Za-z.+_-]{0,63})\.tar\.gz"
)


def identity_dcgm_hostengine(value: object) -> str:
    """The node-identity spelling of a DCGM host engine mode.

    ``host``, or empty for the embedded engine, HyperPod and a reconciler
    rendered before the mode existed -- the reconciler's own normalisation
    (``node_installer_reconciler._desired_dcgm_hostengine``). Any other
    spelling fails closed.
    """

    text = str(value or "").strip()
    if text == DcgmHostEngine.HOST.value:
        return text
    if text in ("", DcgmHostEngine.EMBEDDED.value):
        return ""
    raise ValueError(f"DCGM host engine must be host or embedded, got {text!r}")


def installer_template_identity(
    template_source_sha256: str, dcgm_hostengine: object = None
) -> str:
    """The template identity a node installed from this source reports.

    The source digest itself unless the node configuration the template is
    rendered for differs from the one every older release rendered; today that
    is the ``host`` DCGM engine, folded the way the reconciler names its Jobs
    (``<source>:dcgm-hostengine:host``).
    """

    mode = identity_dcgm_hostengine(dcgm_hostengine)
    if not mode:
        return template_source_sha256
    if not SHA256_HEX.fullmatch(str(template_source_sha256 or "")):
        raise ValueError("installer template source digest must be lowercase SHA-256")
    return hashlib.sha256(
        f"{template_source_sha256}:dcgm-hostengine:{mode}".encode()
    ).hexdigest()


def candidate_installer_template_identity(
    release: Any, target: Any, template_sha256: str | None = None
) -> str:
    """The template identity the candidate release installs on ``target``.

    ``template_sha256`` absent or equal to the release's template source digest
    names the candidate's template, which is folded with the cluster's mode;
    any other value is an identity already resolved (the paused reconciler's)
    and passes through unchanged.
    """

    source = str(getattr(release, "node_template_sha", "") or "")
    if template_sha256 and template_sha256 != source:
        return str(template_sha256)
    return installer_template_identity(source, dcgm_hostengine_environment(target))


def reconciler_template_identity(
    release: Any,
    target: Any,
    template_sha256: str | None,
    node_dependency_target: NodeDependencyTarget,
) -> str | None:
    """``GPU_FAULT_INSTALLER_TEMPLATE_SHA256`` for a reconciler or its preflight.

    The candidate's identity for ``target``; a rollback restores the previous
    release's captured identity, which the reconciler environment reads from
    the snapshot, so its value passes through untouched.
    """

    if node_dependency_target is not NodeDependencyTarget.CANDIDATE:
        return template_sha256
    return candidate_installer_template_identity(release, target, template_sha256)


def installer_bundle_version(name: object) -> str:
    """The installer version a node bundle file was built for; empty if none."""

    match = INSTALLER_BUNDLE_NAME.fullmatch(str(name or ""))
    return match.group("version") if match else ""


def released_installer_version(
    release: Any, target: Any, node_dependency_target: NodeDependencyTarget
) -> str:
    """The installer version of the bundle a reconciler deploy hands its nodes.

    The candidate's bundle file; for a rollback, the key of the previous
    bundle the snapshot captured on ``target``. Empty when neither names a
    version, which leaves the script's own default.
    """

    if node_dependency_target is NodeDependencyTarget.CANDIDATE:
        bundle = getattr(getattr(release, "config", None), "bundle", None)
        return installer_bundle_version(getattr(bundle, "name", ""))
    state = getattr(release, "state", None)
    previous = state.get("previous") if isinstance(state, Mapping) else None
    clusters = previous.get("clusters") if isinstance(previous, Mapping) else None
    cluster = clusters.get(target.cluster_id) if isinstance(clusters, Mapping) else None
    if not isinstance(cluster, Mapping):
        return ""
    return installer_bundle_version(cluster.get("bundle_key"))


def pin_installer_version(
    environment: dict[str, str],
    release: Any,
    target: Any,
    node_dependency_target: NodeDependencyTarget,
) -> None:
    """Hand the reconciler script the released installer version, never the
    deploy host's ``GPU_FAULT_VERSION``: the version names the bundle the Job
    unpacks and is part of the reconciler's node identity."""

    environment[INSTALLER_VERSION_ENV] = released_installer_version(
        release, target, node_dependency_target
    )


def installed_template_matches(annotations: Mapping[str, Any], expected: str) -> bool:
    """Whether a node's template annotation names ``expected``.

    ``expected`` is a template identity -- what the reconciler, its nodes and
    the Agents carry -- or, for a caller that holds only the release state's
    ``node_template_sha256``, the template source digest: a node installed from
    that source in the mode its own annotation records carries the folded
    identity of the two.
    """

    installed = str(annotations.get(INSTALLER_TEMPLATE_ANNOTATION) or "")
    if installed == expected:
        return True
    mode = str(annotations.get(INSTALLER_DCGM_HOSTENGINE_ANNOTATION) or "")
    if not mode or not SHA256_HEX.fullmatch(str(expected or "")):
        return False
    try:
        return installed == installer_template_identity(expected, mode)
    except ValueError:
        return False


def _reads_clusters(release: Any) -> bool:
    """The gate of every classification-time probe (``node_scope_drift``): never
    a dry run, a release without a delivered control plane or a bare test
    double without a cluster reader."""

    runner = getattr(release, "runner", None)
    config = getattr(release, "config", None)
    return (
        runner is not None
        and not getattr(runner, "dry_run", False)
        and bool((getattr(config, "delivery_component_digests", None) or {}).get("cpu"))
        and callable(getattr(release, "_get_json", None))
        and callable(getattr(release, "_gpu", None))
    )


def _stale_installed_nodes(release: Any, target: Any) -> list[str]:
    """Installed nodes of ``target`` whose installer identity is not the release's.

    Judged by the annotations that mirror what each node's Agent reports --
    artifact, Agent config digest, bundle and template identity, which folds
    the DCGM mode -- because those are what the fleet compares: a node they
    name is PENDING in the Agent waves the diff then plans.
    """

    selector = node_selector(target)
    desired = {
        INSTALLER_ARTIFACT_ANNOTATION: str(release.node_wheel_sha),
        INSTALLER_DIGEST_ANNOTATION: target_agent_config_digest(release.config, target),
        INSTALLER_BUNDLE_ANNOTATION: str(release.bundle_sha),
        INSTALLER_TEMPLATE_ANNOTATION: candidate_installer_template_identity(
            release, target
        ),
    }
    listing = release._get_json(release._gpu(target, "get", "nodes", "-l", selector))
    stale = []
    for item in listing.get("items") or ():
        metadata = item.get("metadata") if isinstance(item, Mapping) else None
        if not isinstance(metadata, Mapping) or not selector_matches(
            metadata.get("labels") or {}, selector
        ):
            continue
        annotations = metadata.get("annotations") or {}
        if annotations.get(INSTALLER_STATE_ANNOTATION) != "Succeeded":
            # Not installed, or an install the steady-state reconciler owns
            # whatever the release (a retry, a node that never installed).
            continue
        if any(annotations.get(key) != pin for key, pin in desired.items()):
            stale.append(str(metadata.get("name") or ""))
    return sorted(stale)


def node_installer_identity_drift(release: Any) -> dict[str, list[str]]:
    """GPU clusters whose installed nodes the release's reconciler would re-install.

    ``{cluster_id: [node, ...]}``: nodes the reconciler recorded as installed
    whose artifact, Agent config digest, bundle or template identity differs
    from what this release's reconciler installs. A node it re-installs for a
    reason of its own -- a Node object re-created on a new machine, a node-key
    activation, a failed install it retries -- holds no Agent of this identity
    for a wave to protect and is not drift. A classification-time probe with
    the gate of ``node_scope_drift``; a cluster with no installed node reports
    nothing.
    """

    if not _reads_clusters(release):
        return {}
    drift: dict[str, list[str]] = {}
    for target in getattr(release.config, "clusters", None) or ():
        stale = _stale_installed_nodes(release, target)
        if stale:
            drift[str(target.cluster_id)] = stale
    return drift
