"""The rollback baseline of one release attempt: ``release-deploy/<id>/site.before.yaml``.

A release attempt rewrites ``site.yaml`` (release manifest, Agent config
digest, Runtime Profile source and version, images) before the engine runs.
When the attempt fails and the engine rolls back, ``rollback_alignment``
rebuilds the management site from the attempt's baseline -- the site as it was
committed before the attempt started -- and ``sync-state`` commits what that
document declares, cluster membership included.

The baseline used to be written once, when a release id was first deployed,
and reused by every later attempt of the same release (a ``config`` re-roll
reuses the live release's directory). On a live site a cluster was joined
after the first deploy; a later ``config`` attempt failed its verify, the
automatic rollback rebuilt the site from the one-cluster baseline and
``sync-state`` committed one cluster. The joined cluster disappeared from the
management site while its data plane, token, node Agents and IAM roles kept
running against a control plane that no longer registered it.

Two rules now hold:

* the baseline is **per attempt**: it is written from the current committed
  ``site.yaml`` at the start of every attempt whose previous attempt is
  committed or fully rolled back (``record_rollback_baseline``); an attempt
  that resumes a transaction still in flight keeps the baseline that attempt
  started from, and so does an attempt after a rollback whose management
  alignment did not run through -- there the recorded baseline is still the
  last committed site while ``site.yaml`` is not;
* a rollback **refuses to change membership**: before the management site is
  rewritten, the baseline's cluster set is compared with the committed
  membership the engine recorded, and a difference fails closed naming the
  clusters that would be dropped or added (``require_baseline_membership``).

Membership itself has one source of truth -- ``site.yaml`` as committed by
``join-cluster`` / ``remove-cluster`` and mirrored into the engine state by
``sync-state`` -- so neither command writes into the release directory; the
attempt start copies the committed site, and the guard proves the copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.site import SiteConfigError

BASELINE_FILE = "site.before.yaml"
SUPERSEDED_PREFIX = "site.before.superseded-"
MEMBERSHIP_EVIDENCE_FILE = "membership-baseline.json"
# Rollback records of a previous attempt after which ``site.yaml`` is not the
# committed site: the engine rolled back but the management alignment did not
# rewrite the site (or did not finish), so the recorded baseline is still the
# last committed document and must not be replaced by the unaligned one.
UNALIGNED_ROLLBACK_STATUSES = frozenset(
    {"FAILED", "ALIGNMENT_FAILED", "CLEANUP_PENDING"}
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_private_atomic(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _previous_attempt_record(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / "state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def baseline_refresh_decision(
    live_state: Mapping[str, Any] | None,
    *,
    previous_record: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Whether the attempt starting now may replace the recorded baseline.

    ``live_state`` is the engine's live release state (``{}`` when the site has
    none yet, ``None`` when the caller could not read it); ``previous_record``
    is the ``release-deploy/<id>/state.json`` the previous attempt left. The
    answer is a ``(refresh, reason)`` pair; the reason is recorded in the plan.
    """

    if live_state is None:
        return False, (
            "the live release state could not be read; the recorded baseline is kept"
        )
    rollback = (
        previous_record.get("rollback")
        if isinstance(previous_record, Mapping)
        else None
    )
    if isinstance(rollback, Mapping) and (
        rollback.get("management_state_synced") is False
        or str(rollback.get("status") or "") in UNALIGNED_ROLLBACK_STATUSES
    ):
        return False, (
            "the previous attempt's rollback did not align the management site "
            f"(status {rollback.get('status')}); the recorded baseline is still "
            "the last committed site"
        )
    if not live_state:
        return True, "no live release state: the site has no committed release yet"
    phase = str(live_state.get("phase") or "")
    release_id = str(live_state.get("release_id") or "unknown")
    if (
        phase == "complete"
        and live_state.get("transaction_committed") is True
        and live_state.get("commit_cleanup_completed") is not False
    ):
        return True, f"live release {release_id} is committed"
    rollback_result = live_state.get("rollback_result")
    if (
        phase == "rolled-back"
        and isinstance(rollback_result, Mapping)
        and rollback_result.get("status") == "PASSED"
        and live_state.get("rollback_cleanup_completed") is not False
    ):
        return True, (
            f"live release {release_id} is rolled back to its previous release"
        )
    return False, (
        f"live release {release_id} is mid-transaction (phase {phase or 'unknown'}); "
        "the recorded baseline belongs to the attempt being resumed"
    )


def _next_superseded_path(state_dir: Path) -> Path:
    taken = {path.name for path in state_dir.glob(f"{SUPERSEDED_PREFIX}*.yaml")}
    index = 1
    while f"{SUPERSEDED_PREFIX}{index:03d}.yaml" in taken:
        index += 1
    return state_dir / f"{SUPERSEDED_PREFIX}{index:03d}.yaml"


def record_rollback_baseline(
    state_dir: Path,
    site_file: Path,
    *,
    live_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Record the committed site as this attempt's rollback baseline.

    The first attempt of a release writes it. A later attempt replaces it with
    the current ``site.yaml`` when ``baseline_refresh_decision`` allows and the
    document changed, keeping the superseded copy next to it as
    ``site.before.superseded-NNN.yaml``; otherwise the recorded baseline is
    kept. Returns the evidence recorded in the attempt's plan: path, digest,
    ``action`` (``written`` / ``refreshed`` / ``kept``) and the reason.
    """

    backup = state_dir / BASELINE_FILE
    current = site_file.read_bytes()
    digest = _sha256(current)
    if not backup.exists():
        _write_private_atomic(backup, current)
        return {
            "path": str(backup),
            "sha256": digest,
            "action": "written",
            "reason": "first attempt of this release: the committed site is recorded",
            "recorded_at": _utc_now(),
        }
    existing = backup.read_bytes()
    existing_digest = _sha256(existing)
    refresh, reason = baseline_refresh_decision(
        live_state, previous_record=_previous_attempt_record(state_dir)
    )
    if not refresh:
        return {
            "path": str(backup),
            "sha256": existing_digest,
            "action": "kept",
            "reason": reason,
            "recorded_at": _utc_now(),
        }
    if existing == current:
        return {
            "path": str(backup),
            "sha256": existing_digest,
            "action": "kept",
            "reason": f"{reason}; the recorded baseline already equals the committed site",
            "recorded_at": _utc_now(),
        }
    superseded = _next_superseded_path(state_dir)
    _write_private_atomic(superseded, existing)
    _write_private_atomic(backup, current)
    return {
        "path": str(backup),
        "sha256": digest,
        "action": "refreshed",
        "reason": reason,
        "superseded_sha256": existing_digest,
        "superseded_copy": str(superseded),
        "recorded_at": _utc_now(),
    }


def load_rollback_baseline(path: Path) -> dict[str, Any]:
    """The baseline document, or a ``SiteConfigError`` that says what is wrong.

    A missing or unreadable baseline refuses instead of falling back to the
    current ``site.yaml``: that file carries the failed attempt's release
    fields, and a silent fallback would hide which document the rollback
    restored.
    """

    if not path.is_file():
        raise SiteConfigError(
            f"rollback refused: the rollback baseline {path} does not exist; it is "
            "written when a release attempt is prepared on this host, so a release "
            "prepared elsewhere has none -- rerun the same deploy command from this "
            "host to record it, then roll back"
        )
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SiteConfigError(
            f"rollback refused: cannot read the rollback baseline {path}: {exc}"
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("spec"), dict):
        raise SiteConfigError(
            f"rollback refused: the rollback baseline {path} is not a RegionalSite "
            "document"
        )
    return document


def site_cluster_ids(document: Mapping[str, Any]) -> list[str]:
    spec = document.get("spec")
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    if not isinstance(clusters, list):
        return []
    return sorted(
        {
            str(item.get("clusterId"))
            for item in clusters
            if isinstance(item, Mapping) and item.get("clusterId")
        }
    )


def site_cluster_profile_pins(document: Mapping[str, Any]) -> dict[str, str]:
    """``{cluster_id: runtime profile version}`` for clusters pinning their own."""

    spec = document.get("spec")
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    pins: dict[str, str] = {}
    if not isinstance(clusters, list):
        return pins
    for item in clusters:
        if not isinstance(item, Mapping) or not item.get("clusterId"):
            continue
        profile = item.get("runtimeProfile")
        if isinstance(profile, Mapping) and profile.get("version"):
            pins[str(item["clusterId"])] = str(profile["version"])
    return pins


def committed_cluster_ids(
    live_state: Mapping[str, Any],
) -> tuple[list[str] | None, str]:
    """The cluster membership the engine committed, and where it was read.

    Every checkpoint records ``cluster_ids`` from the configuration that wrote
    it; for a rolled-back state that is the rollback's own configuration, the
    committed site. The ``previous`` snapshot's ``clusters`` mapping names the
    same set when the failed plan captured the GPU clusters, and is the
    fallback for a state written before ``cluster_ids`` existed.
    """

    ids = live_state.get("cluster_ids")
    if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
        return sorted(set(ids)), "cluster_ids"
    previous = live_state.get("previous")
    if isinstance(previous, Mapping):
        clusters = previous.get("clusters")
        if isinstance(clusters, Mapping) and clusters:
            return sorted(str(key) for key in clusters), "previous.clusters"
    return None, "unavailable"


def live_cluster_profile_pins(live_state: Mapping[str, Any]) -> dict[str, str] | None:
    pins = live_state.get("cluster_runtime_profiles")
    if not isinstance(pins, Mapping):
        return None
    return {
        str(cluster_id): str(entry.get("version") or "")
        for cluster_id, entry in pins.items()
        if isinstance(entry, Mapping)
    }


def _current_site_cluster_ids(site_file: Path) -> list[str] | None:
    try:
        document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return site_cluster_ids(document) if isinstance(document, Mapping) else None


def _membership_text(cluster_ids: list[str] | None) -> str:
    if cluster_ids is None:
        return "unreadable"
    return ", ".join(cluster_ids) or "none"


def require_baseline_membership(
    document: Mapping[str, Any],
    live_state: Mapping[str, Any],
    *,
    baseline_path: Path,
    site_file: Path,
) -> dict[str, Any]:
    """Refuse a baseline whose cluster set differs from the committed membership.

    Returns the membership evidence (both sets, the source the committed set
    was read from, and the per-cluster Runtime Profile pins on both sides for
    the record) when they agree; raises ``SiteConfigError`` naming the clusters
    the rewrite would drop or add otherwise. The pins are recorded, not judged:
    a deploy legitimately repairs a cluster's lost override before its release
    runs, and the engine restores the committed pins through the registry.
    """

    baseline_ids = site_cluster_ids(document)
    committed, source = committed_cluster_ids(live_state)
    if committed is None:
        raise SiteConfigError(
            "rollback refused: the live release state records no cluster membership "
            "(neither cluster_ids nor previous.clusters), so the management site "
            f"{site_file} cannot be rewritten from {baseline_path} with proof that "
            "its membership is the committed one"
        )
    dropped = sorted(set(committed) - set(baseline_ids))
    added = sorted(set(baseline_ids) - set(committed))
    if dropped or added:
        current = _current_site_cluster_ids(site_file)
        change = []
        if dropped:
            change.append("drop " + ", ".join(dropped))
        if added:
            change.append("add " + ", ".join(added))
        raise SiteConfigError(
            f"rollback refused: the rollback baseline {baseline_path} lists clusters "
            f"[{', '.join(baseline_ids) or 'none'}] but the committed release state "
            f"({source}) lists [{', '.join(committed) or 'none'}]; restoring it would "
            + " and ".join(change)
            + f". The management site {site_file} was not rewritten and keeps its "
            f"current membership [{_membership_text(current)}]; the release engine "
            "is rolled back. Rerun the same deploy command: a new attempt records "
            "the committed site as its baseline before it starts"
        )
    return {
        "baseline_cluster_ids": baseline_ids,
        "committed_cluster_ids": committed,
        "committed_source": source,
        "baseline_cluster_profile_pins": site_cluster_profile_pins(document),
        "live_cluster_profile_pins": live_cluster_profile_pins(live_state),
    }


__all__ = [
    "BASELINE_FILE",
    "MEMBERSHIP_EVIDENCE_FILE",
    "SUPERSEDED_PREFIX",
    "UNALIGNED_ROLLBACK_STATUSES",
    "baseline_refresh_decision",
    "committed_cluster_ids",
    "live_cluster_profile_pins",
    "load_rollback_baseline",
    "record_rollback_baseline",
    "require_baseline_membership",
    "site_cluster_ids",
    "site_cluster_profile_pins",
]
