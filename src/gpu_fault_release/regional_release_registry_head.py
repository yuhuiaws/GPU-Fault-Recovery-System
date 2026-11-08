"""A rollback passes only while the durable registry head is the restored Secret.

Both CPU stages of a rollback restore the cluster registry into its Secret and
publish it to the durable head (``RollbackRestartPlan.restore_secret_and_registry``);
the head, not the Secret, is what every running control-plane Pod judges a
cluster's Agents and executors by. The rollback's verification used to read
neither, so a rollback whose restore never reached the head PASSED while the
head still held a cluster to the candidate's pins (a resumed rollback whose
restore publish had failed, before such a restore kept its backup: gpu-a on
hyperpod-v1-boot020 and the candidate's Agent config digest), and the only
signal left was ``GpuFaultRegionalRegistrySecretDrift`` about 15 minutes
later.

The verify phase now compares the two first (``verify_rollback``), after the
CPU stages' restore publish converged: the same publish-and-converge the
stages run, which also publishes a restore the Secret still marks as
unpublished before anything is compared. It compares on every run of the
phase -- a resumed rollback, and one that restored nothing -- and every
cluster the registry has an entry for, one parked at zero nodes included. The
comparison is the deploy host's ``secret_drift`` (``registry_head_drift``).
On a mismatch the rollback fails closed: the reason names the clusters that
differ and the exit, and the Secret's registry is marked as a restore whose
publish has not converged (the mark the stages' restores leave), so rerunning
``gpu-fault-admin deploy`` resumes the rollback, publishes it once and
verifies again. A dry run compares nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.rotate_token_journal import (
    load_rotation_state,
    rotation_journal_path,
)
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_online_registry import registry_head_drift
from gpu_fault_release.regional_release_registry import (
    REGISTRY_BACKUP_KEY,
    keep_registry_restore_pending,
    registry_restore_pending,
)

#: The phase-detail key of the verify phase's registry evidence.
EVIDENCE_KEY = "registry_head"


def verify_rollback(
    release: Any, previous: dict[str, Any], compensation: Any
) -> dict[str, Any]:
    """The rollback's verify phase: the durable registry head, then the site.

    The head first: an Agent the restored site is judged by is judged by the
    head's pins, so a head still on the candidate's names the cause before a
    heartbeat mismatch could. Returns the phase details.
    """

    evidence = release._verify_rollback_registry_head()
    release._validate_rollback(
        previous,
        restore_cpu=compensation.restores_cpu,
        cluster_components=compensation.cluster_components,
    )
    return dict(evidence)


def verify_registry_head(release: Any) -> dict[str, Any]:
    """Prove the durable registry head is the restored registry Secret.

    A restore the Secret still marks as unpublished (``registry_restore_pending``)
    is published and completed first, as the CPU stages do. Raises with the
    clusters that differ and the operator exit when the head is not the
    Secret; returns ``{"registry_head": ...}`` evidence when it is.
    """

    if release.runner.dry_run:
        return {}
    republished = registry_restore_pending(release)
    if republished:
        release._publish_restored_registry()
        release._complete_registry_restore()
    drift = registry_head_drift(
        release, retiring_digests=rotation_retiring_digests(release)
    )
    if drift["matches"]:
        evidence = {
            "matches_secret": True,
            "generation": drift["generation"],
            "content_sha256": drift["content_sha256"],
            "republished": republished,
        }
        narrate_step(
            "rollback-registry-head",
            result="matches-secret",
            generation=str(drift["generation"]),
            republished=str(republished).lower(),
        )
        return {EVIDENCE_KEY: evidence}
    clusters = [str(item) for item in drift["clusters"]]
    try:
        keep_registry_restore_pending(release)
        marked: str | None = None
    except ReleaseError as exc:
        marked = str(exc)
    narrate_step(
        "rollback-registry-head",
        result="differs",
        clusters=",".join(clusters) or "unattributed",
    )
    raise ReleaseError(_mismatch(release, clusters, str(drift["reason"]), marked))


def _mismatch(
    release: Any, clusters: list[str], reason: str, mark_error: str | None
) -> str:
    where = (
        "on " + ", ".join(clusters)
        if clusters
        else "on a credential digest the redacted head attributes to no cluster"
    )
    state_dir = getattr(release.config, "admin_state_dir", None)
    command = f"gpu-fault-admin deploy --state-dir {state_dir or '<state-dir>'}"
    head = (
        "rollback registry head mismatch: the durable registry head the control "
        f"plane serves is not the restored registry Secret {where} ({reason}); "
    )
    if mark_error is not None:
        return (
            head + f"the Secret could not keep the restore pending ({mark_error}); "
            f"rerun `{command}` to resume the rollback and compare again"
        )
    return head + (
        f"the Secret keeps the restored registry as {REGISTRY_BACKUP_KEY} until a "
        "publish converges, and join-cluster and remove-cluster refuse meanwhile; "
        f"rerun `{command}` to resume the rollback, which publishes it once and "
        "verifies again"
    )


def rotation_retiring_digests(release: Any) -> dict[str, str]:
    """The old token digest each cluster's rotate-token journal names.

    A ``rotate-token --keep-window`` rotation completes with the head still
    carrying the old token's digest as retiring until the next publish after
    the window closes; the Secret never carries it. ``durable_registrations``
    accepts a journal's digest only where the head holds exactly that digest
    (the revision's content digest proves it), as rotate-token's own head
    evidence does. Without a managed state directory there is no journal.
    """

    state_dir = getattr(release.config, "admin_state_dir", None)
    if state_dir is None:
        return {}
    digests: dict[str, str] = {}
    for target in release.config.clusters:
        path = rotation_journal_path(Path(state_dir), target.cluster_id)
        try:
            state = load_rotation_state(path)
        except (BootstrapError, OSError, ValueError):
            continue
        digest = str((state or {}).get("old_token_sha256") or "")
        if digest:
            digests[target.cluster_id] = digest
    return digests
