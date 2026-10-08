"""The site's GPU kubeconfig entries a GPU cluster owns.

``aws eks update-kubeconfig --alias <context>`` writes one context named by the
alias and a cluster entry and a user entry both named by the EKS cluster ARN,
so two aliases of the same cluster -- the first deploy's
``gpu-fault-gpu-<n>-<eks name>`` and a join's ``gpu-fault-gpu-<eks name>-<sha8>``
-- share one cluster and one user entry.

``remove-cluster`` prunes every entry of the removed cluster that no remaining
context still references (:func:`prune_removed_cluster`), so a later join of
the same cluster does not meet its own stale alias. ``join-cluster`` reclaims
an alias it still finds only when the alias provably names the same EKS
cluster -- the kubeconfig cluster entry's server and CA equal the discovered
cluster's endpoint and CA -- and the site has released the cluster: a
``remove-cluster`` journal COMPLETED for it, or no registration of it in the
control-plane registry (:func:`reclaim_released_alias`). Anything else keeps
the fail-closed refusal (live 2026-10-08: a cluster joined, removed and
re-joined derives the identical alias, and the first mixed-site re-join was
refused by the alias it had left behind). Nothing here reads or writes the
CPU kubeconfig.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    safe_name,
)
from gpu_fault.admin.site import RenderedSite

PRUNE_EVIDENCE = "KUBECONFIG_PRUNED"


class KubeconfigAliasRefused(BootstrapError):
    """An existing alias the join may not take over; the reason is the message."""


@dataclass(frozen=True)
class KubeconfigEntries:
    contexts: tuple[str, ...]
    clusters: tuple[str, ...]
    users: tuple[str, ...]


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_kubeconfig(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise BootstrapError("GPU kubeconfig must not be a symlink")
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BootstrapError("cannot read the GPU kubeconfig") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise BootstrapError("GPU kubeconfig must contain one mapping")
    return document


def _named(document: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = document.get(key) or []
    if not isinstance(value, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        for item in value
    ):
        raise BootstrapError(f"GPU kubeconfig {key} are invalid")
    return value


def _context_reference(entry: Mapping[str, Any], field: str) -> str:
    body = entry.get("context")
    return str(body.get(field) or "") if isinstance(body, dict) else ""


def cluster_entries(
    document: Mapping[str, Any], *, context: str, eks_arn: str
) -> KubeconfigEntries:
    """The entries that belong to one GPU cluster and to nothing else.

    Every context named ``context`` or pointing at the cluster entry ``eks_arn``
    (both alias schemes), plus the cluster and user entries those contexts
    reference that no other context references.
    """

    contexts = _named(document, "contexts")
    owned = [
        item
        for item in contexts
        if item["name"] == context or _context_reference(item, "cluster") == eks_arn
    ]
    owned_names = {item["name"] for item in owned}
    others = [item for item in contexts if item["name"] not in owned_names]
    kept_clusters = {_context_reference(item, "cluster") for item in others}
    kept_users = {_context_reference(item, "user") for item in others}
    clusters = {
        _context_reference(item, "cluster")
        for item in owned
        if _context_reference(item, "cluster")
    } - kept_clusters
    users = {
        _context_reference(item, "user")
        for item in owned
        if _context_reference(item, "user")
    } - kept_users
    present_clusters = {item["name"] for item in _named(document, "clusters")}
    present_users = {item["name"] for item in _named(document, "users")}
    return KubeconfigEntries(
        contexts=tuple(sorted(owned_names)),
        clusters=tuple(sorted(clusters & present_clusters)),
        users=tuple(sorted(users & present_users)),
    )


def _write_kubeconfig(path: Path, document: Mapping[str, Any]) -> None:
    content = yaml.safe_dump(dict(document), sort_keys=False)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".kubeconfig.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def prune_cluster_entries(
    path: Path,
    *,
    context: str,
    eks_arn: str,
    cpu_kubeconfig: Path,
    remaining_contexts: set[str],
) -> dict[str, Any]:
    """Remove one cluster's entries from the GPU kubeconfig at ``path``.

    Refuses the CPU kubeconfig and any entry a remaining cluster's context still
    uses. ``current-context`` never dangles: it moves to a remaining cluster's
    context when it named a pruned one, or is dropped. Returns the evidence the
    removal journal records: the file digests before and after and the names
    removed (``released`` is true once the kubeconfig holds nothing of the
    cluster, whether this call removed it or it was already gone).
    """

    if path.resolve() == cpu_kubeconfig.resolve():
        raise BootstrapError("remove-cluster must not modify the CPU kubeconfig")
    before = _digest(path)
    document = load_kubeconfig(path)
    entries = cluster_entries(document, context=context, eks_arn=eks_arn)
    shared = sorted(set(entries.contexts) & remaining_contexts)
    if shared:
        raise BootstrapError(
            "remove-cluster refuses to prune a kubeconfig context a remaining "
            f"cluster still uses: {', '.join(shared)}"
        )
    evidence: dict[str, Any] = {
        "gpu_kubeconfig": str(path),
        "before_sha256": before,
        "contexts": list(entries.contexts),
        "clusters": list(entries.clusters),
        "users": list(entries.users),
        "released": True,
        "pruned_at": datetime.now(timezone.utc).isoformat(),
    }
    if not (entries.contexts or entries.clusters or entries.users):
        evidence["after_sha256"] = before
        evidence["current_context"] = document.get("current-context")
        return evidence
    updated = dict(document)
    for key, removed in (
        ("contexts", set(entries.contexts)),
        ("clusters", set(entries.clusters)),
        ("users", set(entries.users)),
    ):
        updated[key] = [
            item for item in _named(document, key) if item["name"] not in removed
        ]
    current = updated.get("current-context")
    if current in set(entries.contexts):
        present = {item["name"] for item in updated["contexts"]}
        candidates = sorted(remaining_contexts & present)
        if candidates:
            updated["current-context"] = candidates[0]
        else:
            updated.pop("current-context", None)
    _write_kubeconfig(path, updated)
    evidence["after_sha256"] = _digest(path)
    evidence["current_context"] = updated.get("current-context")
    return evidence


def kubeconfig_released(state: Mapping[str, Any]) -> bool:
    """Whether a removal journal records the cluster's kubeconfig entries gone."""

    evidence = (state.get("evidence") or {}).get(PRUNE_EVIDENCE)
    return isinstance(evidence, dict) and evidence.get("released") is True


def prune_removed_cluster(
    site: RenderedSite,
    *,
    target: Mapping[str, Any],
    remaining: list[dict[str, Any]],
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Prune a removed cluster's entries from the site's own GPU kubeconfig.

    Runs after the removal is COMPLETED (every verification read the cluster
    through its context before) and again on a rerun until the entries are
    gone. Only the kubeconfig the site configures (``spec.gpuKubeconfig``, the
    site environment's ``KUBECONFIG``) is edited -- never the operator's
    default kubeconfig -- and a site without one records that nothing was
    pruned. The evidence lands in the journal under ``KUBECONFIG_PRUNED``.
    """

    if kubeconfig_released(state):
        return dict(state["evidence"][PRUNE_EVIDENCE])
    configured = site.environment.get("KUBECONFIG")
    if not configured:
        evidence: dict[str, Any] = {
            "released": False,
            "reason": "the site configures no GPU kubeconfig; nothing to prune",
        }
    else:
        evidence = prune_cluster_entries(
            Path(configured).expanduser().resolve(),
            context=str(target["context"]),
            eks_arn=str(target["eks_cluster_arn"]),
            cpu_kubeconfig=Path(
                str(site.release_config["cpu_kubeconfig"])
            ).expanduser(),
            remaining_contexts={str(item["context"]) for item in remaining},
        )
    state.setdefault("evidence", {})[PRUNE_EVIDENCE] = evidence
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)
    return evidence


def prune_result(
    site: RenderedSite,
    target: Mapping[str, Any],
    remaining: list[dict[str, Any]],
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """The removal result fragment naming the contexts pruned from the kubeconfig."""

    evidence = prune_removed_cluster(
        site, target=target, remaining=remaining, state_path=state_path, state=state
    )
    return {"gpu_kubeconfig_entries_pruned": list(evidence.get("contexts") or [])}


def alias_target(document: Mapping[str, Any], context: str) -> tuple[str, str, str]:
    """``(cluster entry name, server, certificate-authority-data)`` of an alias."""

    entry = next(
        (item for item in _named(document, "contexts") if item["name"] == context), None
    )
    cluster_name = _context_reference(entry, "cluster") if entry else ""
    cluster = next(
        (item for item in _named(document, "clusters") if item["name"] == cluster_name),
        None,
    )
    body = cluster.get("cluster") if cluster else None
    if not isinstance(body, dict) or not body.get("server"):
        raise KubeconfigAliasRefused("the alias has no cluster entry with a server")
    return (
        cluster_name,
        str(body.get("server") or ""),
        str(body.get("certificate-authority-data") or ""),
    )


def _completed_removal(
    site: RenderedSite, *, cluster_id: str, eks_arn: str
) -> dict[str, Any] | None:
    from gpu_fault.admin.cluster_removal_state import read_removal_state

    path = site.source.parent / "remove-cluster" / safe_name(cluster_id) / "state.json"
    if not path.is_file():
        return None
    state = read_removal_state(path)
    if (
        state.get("phase") != "COMPLETED"
        or str(state["target"].get("eks_cluster_arn") or "") != eks_arn
    ):
        return None
    return {
        "remove_cluster_attempt": state["attempt_id"],
        "completed_at": (state.get("step_completed_at") or {}).get("VERIFIED"),
    }


def reclaim_released_alias(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    target: ClusterIdentity,
    cluster_id: str,
    kubeconfig: Path,
    registry_states: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove that an existing alias names the joining cluster and is released.

    Same target: the kubeconfig cluster entry behind the alias has the server
    endpoint and the CA of ``eks describe-cluster`` for the discovered cluster.
    Released: the site does not list the cluster (the caller's join path) and
    either its ``remove-cluster`` journal is COMPLETED for this EKS cluster or
    the control-plane registry holds no registration for it. Returns the
    ``kubeconfig_alias_replaced`` evidence; raises
    :class:`KubeconfigAliasRefused` otherwise.
    """

    if any(
        item.get("cluster_id") == cluster_id for item in site.release_config["clusters"]
    ):
        raise KubeconfigAliasRefused("the cluster is still managed by the site")
    cluster_name, server, authority = alias_target(
        load_kubeconfig(kubeconfig), target.context
    )
    described = runner.aws_json(
        target.region, "eks", "describe-cluster", "--name", target.eks_name
    ).get("cluster")
    if not isinstance(described, dict):
        raise KubeconfigAliasRefused("eks describe-cluster returned no cluster")
    live_authority = str(
        (described.get("certificateAuthority") or {}).get("data") or ""
    )
    if str(described.get("arn") or target.eks_arn) != target.eks_arn:
        raise KubeconfigAliasRefused("eks describe-cluster names another cluster ARN")
    if str(described.get("endpoint") or "") != server or not server:
        raise KubeconfigAliasRefused("the alias names another server endpoint")
    if not live_authority or live_authority != authority:
        raise KubeconfigAliasRefused("the alias carries another certificate authority")
    released = _completed_removal(site, cluster_id=cluster_id, eks_arn=target.eks_arn)
    if released is None:
        states = (
            registry_states() if registry_states is not None else _registry_states(site)
        )
        if cluster_id in states:
            raise KubeconfigAliasRefused(
                "the cluster is still registered with the control plane "
                f"({states[cluster_id]}) and has no COMPLETED remove-cluster journal"
            )
        released = {"registry": "absent"}
    return {
        "context": target.context,
        "cluster_entry": cluster_name,
        "endpoint": server,
        "certificate_authority_sha256": hashlib.sha256(authority.encode()).hexdigest(),
        "released_by": released,
        "replaced_at": datetime.now(timezone.utc).isoformat(),
    }


def _registry_states(site: RenderedSite) -> Mapping[str, Any]:
    from gpu_fault.admin.cluster_join_evidence import membership_runtime_snapshot

    states = membership_runtime_snapshot(site).get("registry_cluster_states")
    if not isinstance(states, Mapping):
        raise KubeconfigAliasRefused("the control-plane registry states are unreadable")
    return states


__all__ = [
    "KubeconfigAliasRefused",
    "KubeconfigEntries",
    "PRUNE_EVIDENCE",
    "alias_target",
    "cluster_entries",
    "kubeconfig_released",
    "load_kubeconfig",
    "prune_cluster_entries",
    "prune_removed_cluster",
    "prune_result",
    "reclaim_released_alias",
]
