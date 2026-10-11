"""``join-cluster --reclaim-cluster-token``: re-attach a cluster the site lost.

A rollback that rebuilt the management site from a stale baseline, or an
operator editing the site, can drop a joined cluster from ``site.yaml`` while
its token file, data plane, node Agents and IAM roles all stay in place. The
ordinary join then refuses the orphaned token ("existing cluster token has no
matching join ownership") because a fresh journal records no
``retained_cluster_token``, and ``remove-cluster`` needs the cluster in the
site. This module is the one consented path through that refusal. The operator
passes ``--reclaim-cluster-token`` and the join adopts the token only on proof:

1. identity -- a committed join journal of this site (the current journal or an
   archived attempt in the same state directory) records this token file for
   this cluster id with the same EKS and HyperPod ARNs;
2. data plane -- the ``gpu-fault-regional-connection`` Secret on the target
   cluster, the one the Executor and node Agents read, carries the same token
   (compared by SHA-256, never printed) and the same cluster id;
3. control plane -- the regional registry Secret (current or previous payload)
   or the installation registry still records the cluster id; a registry row
   naming a different EKS cluster or a different token refuses.

Any missing proof refuses with the reason and leaves the journal in
``BLOCKED_IDENTITY`` as before. On success the journal gets the
``retained_cluster_token`` binding the ordinary retry path already honours, a
``reclaimed_cluster_token`` record of the proof (digests, sources, timestamps)
and the committed namespace UID, and the join continues unchanged:
``write_secret`` keeps the existing token file, the data plane is released
again for the cluster, the registry and ``site.yaml`` take it back.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    KubernetesResourceNotFound,
)
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.resource_registry import (
    LegacyInstallationRegistryMissing,
    fetch_installation_resource_registry,
)
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault.installation_resources import InstallationResourceStatus
from gpu_fault_release.regional_release_registry import (
    REGISTRY_BACKUP_KEY,
    REGISTRY_CURRENT_KEY,
    REGISTRY_SECRET,
)

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest

CONNECTION_SECRET = "gpu-fault-regional-connection"
CONNECTION_TOKEN_KEY = "cluster-token"
CONNECTION_CLUSTER_ID_KEY = "cluster-id"
MIN_TOKEN_LENGTH = 32
RECLAIM_FLAG = "--reclaim-cluster-token"
RUNBOOK_SECTION = (
    'administrator operations, "re-attaching a cluster after a rollback or '
    'operator error removed it from the site"'
)
# Journals that committed the cluster into the site: the step after which the
# token is bound to a registered, activated member.
COMMITTED_JOIN_STEPS = frozenset({"SITE_UPDATED", "RELEASE_STATE_UPDATED", "ACTIVATED"})


class ClusterTokenReclaimRefused(join.JoinTargetIdentityError):
    """A reclaim proof is missing; the journal stays blocked as before."""


def _refusal(cluster_id: str, reason: str) -> ClusterTokenReclaimRefused:
    return ClusterTokenReclaimRefused(
        f"join-cluster {RECLAIM_FLAG} refused for {cluster_id}: {reason} "
        f"(see {RUNBOOK_SECTION})"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def token_digest(token_file: Path, *, cluster_id: str) -> str:
    """SHA-256 of the token material as the data plane stores it (stripped)."""

    if token_file.is_symlink():
        raise _refusal(cluster_id, f"the token file {token_file} is a symlink")
    if not token_file.is_file():
        raise _refusal(cluster_id, f"there is no token file at {token_file}")
    material = token_file.read_text(encoding="utf-8").strip()
    if len(material) < MIN_TOKEN_LENGTH:
        raise _refusal(
            cluster_id,
            f"the token file {token_file} is too short to be a cluster token",
        )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _journal_documents(state_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    candidates = [
        state_dir / "state.json",
        *sorted(state_dir.glob("state.attempt-*.json")),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            documents.append((path, value))
    return documents


def committed_join_evidence(
    state_dir: Path,
    *,
    cluster_id: str,
    target: ClusterIdentity,
    token_file: Path,
) -> dict[str, Any]:
    """Proof 1: a committed join of this site bound the token file to this identity.

    Scans the current journal and the archived attempts of the join state
    directory (one directory per GPU cluster ARN) for an attempt that committed
    the cluster into the site and recorded this token path for this cluster id,
    EKS ARN and HyperPod ARN. The newest committed attempt wins.
    """

    matches: list[dict[str, Any]] = []
    for path, journal in _journal_documents(state_dir):
        completed = set(journal.get("completed_steps") or [])
        if not completed & COMMITTED_JOIN_STEPS and journal.get("phase") != "COMPLETED":
            continue
        evidence = journal.get("evidence") or {}
        discovered = evidence.get("DISCOVERED") or {}
        recorded_target = discovered.get("target") or {}
        local = (
            evidence.get("LOCAL_INPUTS_READY")
            or evidence.get("LOCAL_INPUTS_STARTED")
            or {}
        )
        if (
            str(discovered.get("cluster_id") or "") != cluster_id
            or str(recorded_target.get("eks_arn") or "") != target.eks_arn
            or str(recorded_target.get("hyperpod_arn") or "") != target.hyperpod_arn
            or str(local.get("token_file") or "") != str(token_file)
        ):
            continue
        final = evidence.get("FINAL_VERIFIED") or {}
        completed_at = journal.get("step_completed_at") or {}
        matches.append(
            {
                "source": "join-journal",
                "journal": str(path),
                "attempt": int(journal.get("attempt") or 0),
                "phase": str(journal.get("phase") or ""),
                "committed_at": str(
                    final.get("committed_at")
                    or completed_at.get("SITE_UPDATED")
                    or journal.get("updated_at")
                    or ""
                ),
                "registry_generation": final.get("registry_generation"),
                "live_release_state_sha256": final.get("live_release_state_sha256"),
                "eks_arn": target.eks_arn,
                "hyperpod_arn": target.hyperpod_arn,
                "namespace_uid": str(local.get("namespace_uid") or ""),
            }
        )
    if not matches:
        raise _refusal(
            cluster_id,
            f"no committed join journal under {state_dir} records the token file "
            f"{token_file} for cluster {cluster_id} with EKS ARN {target.eks_arn} "
            f"and HyperPod ARN {target.hyperpod_arn or '(none)'}; the token's "
            "identity cannot be proven",
        )
    return max(matches, key=lambda item: (item["attempt"], item["committed_at"]))


def _secret_data(output: str) -> dict[str, str] | None:
    try:
        document = json.loads(output) if output.strip() else None
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict) or document.get("kind") != "Secret":
        return None
    data = document.get("data")
    if not isinstance(data, dict):
        return None
    decoded: dict[str, str] = {}
    for key, value in data.items():
        try:
            decoded[str(key)] = base64.b64decode(str(value)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
    return decoded


def data_plane_token_proof(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    kubeconfig: Path,
    target: ClusterIdentity,
    cluster_id: str,
    digest: str,
) -> dict[str, Any]:
    """Proof 2: the installed data plane holds the same token for the same cluster id.

    Reads the connection Secret the Executor and node Agents mount, under the
    cluster's kubeconfig context, as a sensitive command: the material is
    compared by digest and never printed or journaled.
    """

    namespace = str(site.release_config["namespace"])
    environment = effective_environment(site)
    if not kubeconfig.is_file():
        raise _refusal(
            cluster_id,
            f"the GPU kubeconfig {kubeconfig} does not exist; the data plane of this "
            "cluster cannot be reached to prove it holds the token",
        )
    contexts = runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "config",
            "get-contexts",
            "-o",
            "name",
        ],
        env=environment,
    )
    if target.context not in str(contexts).splitlines():
        raise _refusal(
            cluster_id,
            f"the GPU kubeconfig {kubeconfig} has no context {target.context}; the "
            "data plane of this cluster cannot be reached to prove it holds the token",
        )
    try:
        output = runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig),
                "--context",
                target.context,
                "-n",
                namespace,
                "get",
                "secret",
                CONNECTION_SECRET,
                "-o",
                "json",
            ],
            env=environment,
            sensitive=True,
        )
    except KubernetesResourceNotFound:
        output = ""
    except BootstrapError as exc:
        raise _refusal(
            cluster_id,
            f"cannot read the data-plane Secret {namespace}/{CONNECTION_SECRET} on "
            f"context {target.context}: {exc}",
        ) from exc
    data = _secret_data(output)
    if data is None:
        raise _refusal(
            cluster_id,
            f"the GPU cluster (context {target.context}) has no readable "
            f"{namespace}/{CONNECTION_SECRET} Secret; the data plane of this site is "
            "not installed there",
        )
    installed = data.get(CONNECTION_TOKEN_KEY, "").strip()
    if not installed:
        raise _refusal(
            cluster_id,
            f"the data-plane Secret {namespace}/{CONNECTION_SECRET} carries no "
            f"{CONNECTION_TOKEN_KEY}",
        )
    installed_digest = hashlib.sha256(installed.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(installed_digest, digest):
        raise _refusal(
            cluster_id,
            f"the data plane on context {target.context} holds a different cluster "
            "token than the site's token file (SHA-256 differs); the file is not the "
            "credential this cluster uses",
        )
    recorded_id = data.get(CONNECTION_CLUSTER_ID_KEY, "").strip()
    if recorded_id != cluster_id:
        raise _refusal(
            cluster_id,
            f"the data plane on context {target.context} is installed for cluster id "
            f"{recorded_id or '(none)'}, not {cluster_id}",
        )
    return {
        "source": f"{namespace}/{CONNECTION_SECRET}",
        "context": target.context,
        "namespace": namespace,
        "token_sha256": installed_digest,
        "cluster_id": recorded_id,
        "observed_at": _utc_now(),
    }


def _registry_rows(runner: CommandRunner, site: RenderedSite) -> dict[str, list[Any]]:
    try:
        output = runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(site.release_config["cpu_kubeconfig"]),
                "-n",
                str(site.release_config["namespace"]),
                "get",
                "secret",
                REGISTRY_SECRET,
                "-o",
                "json",
            ],
            env=effective_environment(site),
            sensitive=True,
        )
    except KubernetesResourceNotFound:
        return {}
    data = _secret_data(output) or {}
    rows: dict[str, list[Any]] = {}
    for key in (REGISTRY_CURRENT_KEY, REGISTRY_BACKUP_KEY):
        raw = data.get(key)
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            rows[key] = value
    return rows


def _registry_membership(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    cluster_id: str,
    target: ClusterIdentity,
    digest: str,
) -> dict[str, Any] | None:
    found: list[dict[str, Any]] = []
    for key, rows in _registry_rows(runner, site).items():
        for row in rows:
            if not isinstance(row, Mapping) or str(row.get("cluster_id")) != cluster_id:
                continue
            if str(row.get("eks_cluster_arn") or "") != target.eks_arn:
                raise _refusal(
                    cluster_id,
                    f"the regional registry Secret ({REGISTRY_SECRET}/{key}) registers "
                    f"{cluster_id} for EKS cluster {row.get('eks_cluster_arn')}, not "
                    f"{target.eks_arn}",
                )
            token = str(row.get("token") or "").strip()
            if token and not hmac.compare_digest(
                hashlib.sha256(token.encode("utf-8")).hexdigest(), digest
            ):
                raise _refusal(
                    cluster_id,
                    f"the regional registry Secret ({REGISTRY_SECRET}/{key}) registers "
                    f"{cluster_id} with a different token than the site's token file",
                )
            found.append(
                {
                    "key": key,
                    "lifecycle_state": row.get("lifecycle_state"),
                    "token_sha256_matches": bool(token),
                }
            )
    if not found:
        return None
    return {"source": f"{REGISTRY_SECRET}", "rows": found}


def _installation_membership(
    site: RenderedSite,
    *,
    cluster_id: str,
    target: ClusterIdentity,
    output: Path,
) -> dict[str, Any] | None:
    try:
        snapshot = fetch_installation_resource_registry(
            site, output=output, allow_empty=True
        )
    except (LegacyInstallationRegistryMissing, BootstrapError) as exc:
        return {"source": "installation-registry", "unavailable": str(exc)}
    active = {
        item.resource_key: item
        for item in snapshot.resources
        if item.status is InstallationResourceStatus.ACTIVE
    }
    keys: list[str] = []
    eks_row = active.get(f"cluster/{cluster_id}/eks")
    if eks_row is not None and (eks_row.resource_arn or "") in {"", target.eks_arn}:
        keys.append(eks_row.resource_key)
    for key in (
        f"aws/iam/executor/{cluster_id}/role",
        f"aws/iam/executor/{cluster_id}/oidc-provider",
        f"aws/iam/adot-writer/{cluster_id}/role",
    ):
        if key in active:
            keys.append(key)
    if not keys:
        return None
    return {
        "source": "installation-registry",
        "snapshot_sha256": snapshot.source_sha256,
        "active_keys": keys,
        "snapshot": str(output),
    }


def control_plane_membership_proof(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    cluster_id: str,
    target: ClusterIdentity,
    digest: str,
    snapshot_output: Path,
) -> dict[str, Any]:
    """Proof 3: the control plane's durable records still know this cluster id.

    Either the regional registry Secret on the CPU cluster (current or previous
    payload) carries the cluster's registration -- for the same EKS cluster and,
    when the row holds a token, the same token -- or the installation registry
    in Aurora holds ACTIVE rows under the cluster id (its EKS row or the IAM
    rows the committed join recorded). Neither knowing the cluster refuses.
    """

    registry = _registry_membership(
        runner, site=site, cluster_id=cluster_id, target=target, digest=digest
    )
    installation = _installation_membership(
        site, cluster_id=cluster_id, target=target, output=snapshot_output
    )
    sources = [
        entry["source"]
        for entry in (registry, installation)
        if entry is not None and "unavailable" not in entry
    ]
    if not sources:
        detail = (
            f"; installation registry: {installation['unavailable']}"
            if isinstance(installation, dict) and "unavailable" in installation
            else ""
        )
        raise _refusal(
            cluster_id,
            f"neither the regional registry Secret ({REGISTRY_SECRET}, "
            f"{REGISTRY_CURRENT_KEY} or {REGISTRY_BACKUP_KEY}) nor the installation "
            f"registry records cluster {cluster_id}{detail}; the control plane has "
            "no record of this cluster, so its token cannot be reclaimed",
        )
    return {
        "sources": sources,
        "regional_registry": registry,
        "installation_registry": installation,
        "observed_at": _utc_now(),
    }


def reclaim_cluster_token(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner,
    target: ClusterIdentity,
    cluster_id: str,
    token_file: Path,
    gpu_kubeconfig: Path,
    binding: Mapping[str, Any],
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Adopt an orphaned token on the three proofs; journal the binding and proof.

    Returns the ``reclaimed_cluster_token`` record (it carries the committed
    namespace UID the caller seeds its local inputs with). Raises
    ``ClusterTokenReclaimRefused`` -- a ``JoinTargetIdentityError`` -- when any
    proof is missing, without touching the journal.
    """

    digest = token_digest(token_file, cluster_id=cluster_id)
    identity = committed_join_evidence(
        state_dir, cluster_id=cluster_id, target=target, token_file=token_file
    )
    data_plane = data_plane_token_proof(
        runner,
        site=request.site,
        kubeconfig=gpu_kubeconfig,
        target=target,
        cluster_id=cluster_id,
        digest=digest,
    )
    attempt = int(state.get("attempt") or 1)
    control_plane = control_plane_membership_proof(
        runner,
        site=request.site,
        cluster_id=cluster_id,
        target=target,
        digest=digest,
        snapshot_output=state_dir
        / f"installation-resources-reclaim-{attempt:03d}.json",
    )
    record = {
        "reclaimed_at": _utc_now(),
        "cluster_id": cluster_id,
        "token_file": str(token_file),
        "token_sha256": digest,
        "identity": identity,
        "data_plane": data_plane,
        "control_plane": control_plane,
        "namespace_uid": identity.get("namespace_uid") or "",
    }
    state["retained_cluster_token"] = dict(binding)
    state["reclaimed_cluster_token"] = record
    state["updated_at"] = _utc_now()
    write_json_atomic(state_path, state)
    return record


__all__ = [
    "CONNECTION_SECRET",
    "ClusterTokenReclaimRefused",
    "RECLAIM_FLAG",
    "committed_join_evidence",
    "control_plane_membership_proof",
    "data_plane_token_proof",
    "reclaim_cluster_token",
    "token_digest",
]
