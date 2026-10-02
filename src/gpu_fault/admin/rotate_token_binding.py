"""Bind a rotation journal to the site facts the rotation reads, not to site.yaml's bytes.

The journal used to carry only ``site_sha256``, the raw digest of ``site.yaml``,
and a resume refused on any mismatch. A release rewrites the file too --
``spec.repositoryRoot``, ``spec.release.manifest``,
``spec.runtimeProfile.source`` -- and none of those is read by the rotation,
so a deploy run between two steps left a journal that could neither finish
nor roll back. The journal now also records ``site_binding``, a canonical
projection of the facts the rotation and its acceptance probe actually read
from the site, and ``site_binding_sha256`` over it. A resume whose raw digest
differs proceeds iff the projection is unchanged; a changed projection still
fails closed and names the fact that moved.

A journal from before the projection has nothing to compare. It is rebound
only when live state proves the rotation already converged onto the recorded
new token (the journal is past ``TOKEN_FILE_WRITTEN``, the token file's digest
is ``new_token_sha256``, and the durable registry head holds that digest for
the cluster with the retiring digest either gone or still equal to the
journal's ``old_token_sha256``) -- never for a rollback, and never earlier in
the machine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.rotate_token_journal import (
    STEP_TOKEN_FILE_WRITTEN,
    last_completed_step,
    step_done,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_online_registry import durable_registrations
from gpu_fault_release.regional_release_registry import REGISTRY_SECRET

SITE_CHANGED = "rotate-token site changed since the rotation started"
# Per-cluster facts the rotation reads from ``site.release_config["clusters"]``:
# the target lookup and identity re-check, ``kubectl --context`` for every GPU
# mutation, and the token file it rewrites.
CLUSTER_BINDING_FIELDS = (
    "cluster_id",
    "context",
    "eks_cluster_arn",
    "hyperpod_cluster_name",
    "token_file",
)


def rotation_site_binding(site: RenderedSite, cluster_id: str) -> dict[str, Any]:
    """The rotation-relevant projection of ``site``; digests only, never a token."""

    config = site.release_config
    namespace = str(config["namespace"])
    clusters = sorted(
        (
            {field: item.get(field) for field in CLUSTER_BINDING_FIELDS}
            for item in config["clusters"]
        ),
        key=lambda item: str(item["cluster_id"]),
    )
    target = [item for item in clusters if item["cluster_id"] == cluster_id]
    return {
        "site_name": config["site_name"],
        "cluster_id": cluster_id,
        "token_file": target[0]["token_file"] if target else None,
        "cpu_kubeconfig": config["cpu_kubeconfig"],
        "cpu_eks_arn": config.get("cpu_eks_arn"),
        "namespace": namespace,
        "gpu_kubeconfig": site.environment.get("KUBECONFIG"),
        "registry_secret": {"namespace": namespace, "name": REGISTRY_SECRET},
        "clusters": clusters,
    }


def site_binding_sha256(binding: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(binding))


def binding_differences(recorded: object, current: Mapping[str, Any]) -> list[str]:
    if not isinstance(recorded, dict):
        return ["(recorded binding projection unavailable)"]
    return sorted(
        key
        for key in set(recorded) | set(current)
        if recorded.get(key) != current.get(key)
    )


def legacy_rebind_pending(
    existing: Mapping[str, Any],
    site: RenderedSite,
    cluster_id: str,
    *,
    rollback: bool,
) -> bool:
    """Whether a journal whose site bytes changed may resume, and how.

    ``False``: the site is the one the rotation started on (raw digest or the
    recorded binding matches) -- resume as usual. ``True``: the journal
    predates the binding and only live evidence can rebind it; the caller
    must gather that evidence before mutating anything. Raises when the
    rotation-relevant binding differs, or when a legacy journal is asked to
    roll back or has not yet written the token file.
    """

    if existing.get("site_sha256") == site.source_sha256:
        return False
    binding = rotation_site_binding(site, cluster_id)
    recorded = existing.get("site_binding_sha256")
    if recorded is not None:
        if recorded == site_binding_sha256(binding):
            return False
        differences = binding_differences(existing.get("site_binding"), binding)
        raise BootstrapError(
            f"{SITE_CHANGED}; the rotation-relevant site binding differs on: "
            + ", ".join(differences)
        )
    if rollback:
        raise BootstrapError(
            f"{SITE_CHANGED}; a journal without a site binding is never rebound "
            "for --rollback"
        )
    if not step_done(existing, STEP_TOKEN_FILE_WRITTEN):
        raise BootstrapError(
            f"{SITE_CHANGED}; a journal without a site binding is rebound only "
            f"after {STEP_TOKEN_FILE_WRITTEN}"
        )
    return True


def registry_head_evidence(
    release: Any,
    cluster_id: str,
    digest: str,
    *,
    retiring_digest: str | None = None,
) -> dict[str, Any]:
    """Whether the durable registry head holds ``digest`` for ``cluster_id``.

    Reads through the registry client the rotation already publishes with and
    restores the redacted digests from the CPU Secret; the revision's content
    digest proves the two agree. Without ``retiring_digest`` the head must
    carry no retiring digest at all (a head that still does cannot be
    reconstructed from the Secret and reports as not converged). With it --
    the journal's ``old_token_sha256`` -- a retiring digest equal to it is
    accepted too: the shape a rotation stopped between the control-plane roll
    and the final drop leaves behind.
    """

    candidates = {cluster_id: retiring_digest} if retiring_digest else None
    try:
        status, durable = durable_registrations(release, retiring_digests=candidates)
    except ReleaseError as exc:
        return {
            "converged": False,
            "reason": (
                "the durable registry head cannot be reconstructed from the CPU "
                f"Secret: {exc}"
            ),
        }
    entry = next((item for item in durable if item.cluster_id == cluster_id), None)
    if entry is None:
        return {
            "converged": False,
            "reason": f"{cluster_id} is not in the durable registry head",
        }
    if entry.token_sha256 != digest:
        return {
            "converged": False,
            "reason": (
                "the durable registry head token digest is not the recorded new digest"
            ),
        }
    retiring = entry.retiring_token_sha256
    if retiring is not None and retiring != retiring_digest:
        return {
            "converged": False,
            "reason": "the durable registry head still carries a retiring digest",
        }
    return {
        "converged": True,
        "generation": status.generation,
        "content_sha256": status.content_sha256,
        "token_sha256": digest,
        "retiring_token_present": retiring is not None,
    }


def rebind_legacy_journal(
    state: dict[str, Any],
    *,
    site: RenderedSite,
    cluster_id: str,
    token_file_sha256: str,
    release: Any,
    now: datetime,
) -> None:
    """Rebind a pre-binding journal to ``site`` on proof the rotation converged.

    Writes ``site_rebound`` (when, from and to which raw digest, and the
    evidence read) plus the binding fields a new journal would carry; the
    caller persists. Every refusal keeps the existing site-changed message.
    """

    digest = str(state.get("new_token_sha256") or "")
    evidence = [
        f"journal is past {STEP_TOKEN_FILE_WRITTEN} "
        f"(last completed step {last_completed_step(state)})"
    ]
    if not digest or token_file_sha256 != digest:
        raise BootstrapError(
            f"{SITE_CHANGED}; the journal cannot be rebound: the token file digest "
            "is not the recorded new_token_sha256"
        )
    evidence.append("token file sha256 equals new_token_sha256")
    head = registry_head_evidence(
        release,
        cluster_id,
        digest,
        retiring_digest=str(state.get("old_token_sha256") or "") or None,
    )
    if not head["converged"]:
        raise BootstrapError(
            f"{SITE_CHANGED}; the journal cannot be rebound: {head['reason']}"
        )
    evidence.append(
        f"durable registry head generation {head['generation']} holds "
        f"new_token_sha256 for {cluster_id} (content_sha256 {head['content_sha256']})"
    )
    evidence.append(
        "retiring digest still present (equals old_token_sha256); the final step "
        "drops it"
        if head["retiring_token_present"]
        else "retiring digest already absent from the durable head"
    )
    binding = rotation_site_binding(site, cluster_id)
    state["site_rebound"] = {
        "at": now.isoformat(),
        "from_site_sha256": state.get("site_sha256"),
        "to_site_sha256": site.source_sha256,
        "evidence": evidence,
    }
    state["site_binding"] = binding
    state["site_binding_sha256"] = site_binding_sha256(binding)
