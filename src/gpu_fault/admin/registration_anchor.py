"""The site Runtime Profile registration anchor once its cluster left the site.

``spec.runtimeProfile.registrationClusterId`` names the cluster whose provider
the site base profile family follows (``bootstrap_discovery.regional_profile_for``)
and under which a new site profile version is registered -- the control plane
accepts only a registered cluster. ``remove-cluster`` moves it with the
removal of that cluster while a member of its provider remains
(``cluster_removal_state.registration_reanchor``) and keeps it when the last
cluster leaves. The anchor then names a cluster that *departed*: its COMPLETED
removal journal (``remove-cluster/<cluster id>/state.json``) proves it left the
site and records its provider, the family the site base keeps.

* A join fills a departed anchor (``fill_departed_anchor``): once the joined
  clusters are members, the anchor moves to a member of the departed
  anchor's provider -- one on the site profile before one pinned to its own,
  then the first by cluster id, as ``remove-cluster`` chooses.
* The first join into an emptied site must bring a cluster of that provider
  (``require_anchor_family``): otherwise the site would have members and no
  anchor of its family, and every deploy would refuse it.
* A deploy refuses a departed anchor while members remain and names the way
  out (``require_registration_anchor``). A site an older release left so (it
  removed the anchor while members remained and kept naming it) moves the
  anchor explicitly with ``deploy --registration-cluster <cluster id>
  --reference <CHG>`` (``reanchor_registration``), proof-checked the same way.

Every move is appended to ``registration-anchor/journal.json``. The versions
registered under the departed anchor keep it as their label, as they do after
a ``remove-cluster`` moved it.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, safe_name
from gpu_fault.admin.cluster_removal_state import (
    REGISTRATION_REANCHOR,
    read_removal_state,
)
from gpu_fault.admin.profile_approval import APPROVAL_PATTERN
from gpu_fault.cluster_scope import cluster_provider

__all__ = [
    "DepartedAnchor",
    "JOURNAL",
    "departed_anchor",
    "departed_anchor_refusal",
    "fill_departed_anchor",
    "reanchor_registration",
    "require_anchor_family",
    "require_registration_anchor",
]

JOURNAL = Path("registration-anchor/journal.json")
#: ``trigger`` of a journal entry: a join filled the anchor, or the operator
#: moved it with the deploy flag.
JOIN_TRIGGER = "join-cluster"
DEPLOY_TRIGGER = "deploy --registration-cluster"


@dataclass(frozen=True)
class DepartedAnchor:
    """A registration anchor that names no member of the site."""

    cluster_id: str
    #: The departed cluster's provider from its COMPLETED removal journal;
    #: ``None`` when no such journal proves it left the site.
    provider: str | None
    journal: Path


def _document(site_file: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BootstrapError(f"cannot read the site document {site_file}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("spec"), dict):
        raise BootstrapError(f"site document {site_file} has no spec")
    return document


def _members(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    clusters = document["spec"].get("clusters") or []
    return [item for item in clusters if isinstance(item, dict)]


def _anchor(document: Mapping[str, Any]) -> str:
    profile = document["spec"].get("runtimeProfile")
    value = profile.get("registrationClusterId") if isinstance(profile, dict) else ""
    return str(value or "")


def departed_anchor(
    document: Mapping[str, Any], state_dir: Path
) -> DepartedAnchor | None:
    """The site's anchor when it names no member, else ``None``.

    The proof is the anchor's own removal journal: schema-valid, COMPLETED,
    of this site and of this cluster. Anything less (no journal, a removal
    still in progress, a journal of another site) proves nothing, and the
    provider stays unknown.
    """

    anchor = _anchor(document)
    if not anchor or any(
        item.get("clusterId") == anchor for item in _members(document)
    ):
        return None
    removals = state_dir / "remove-cluster"
    try:
        path = removals / safe_name(anchor) / "state.json"
    except BootstrapError:
        return DepartedAnchor(anchor, None, removals)
    metadata = document.get("metadata")
    site_id = metadata.get("name") if isinstance(metadata, dict) else None
    try:
        state = read_removal_state(path) if path.is_file() else {}
        target = state.get("target") or {}
        proven = (
            state.get("phase") == "COMPLETED"
            and state.get("cluster_id") == anchor
            and target.get("cluster_id") == anchor
            and state.get("site_id") == site_id
        )
        provider = cluster_provider(target.get("provider")).value if proven else None
    except (BootstrapError, ValueError):
        provider = None
    return DepartedAnchor(anchor, provider, path)


def _family(document: Mapping[str, Any], provider: str) -> list[str]:
    """Cluster ids of the members of ``provider``: those on the site profile
    first, then those pinned to their own, each by cluster id."""

    members = sorted(
        (
            item
            for item in _members(document)
            if cluster_provider(item.get("provider")).value == provider
        ),
        key=lambda item: ("runtimeProfile" in item, str(item.get("clusterId") or "")),
    )
    return [str(item["clusterId"]) for item in members]


def departed_anchor_refusal(
    document: Mapping[str, Any], departed: DepartedAnchor, state_dir: Path
) -> str:
    """Why a deploy cannot run on a departed anchor, and the way out."""

    reason = (
        f"registration cluster {departed.cluster_id} is not among the managed GPU "
        "clusters; the site Runtime Profile base cannot be re-derived"
    )
    if departed.provider is None:
        return (
            f"{reason}: no completed remove-cluster journal ({departed.journal}) "
            "proves it left the site; while it exists, join it back with "
            "gpu-fault-admin join-cluster (same cluster id) or, if the site lost "
            "its entry, re-attach it with join-cluster --reclaim-cluster-token"
        )
    family = _family(document, departed.provider)
    if not family:
        return (
            f"{reason}: it left the site (remove-cluster journal "
            f"{departed.journal}) and no {departed.provider} cluster remains to "
            "take over the anchor; the site base profile family follows its "
            f"provider: join a {departed.provider} cluster with gpu-fault-admin "
            "join-cluster (it takes over the anchor), then deploy"
        )
    return (
        f"{reason}: it left the site (remove-cluster journal {departed.journal}); "
        f"move the anchor to a remaining {departed.provider} cluster "
        f"({', '.join(family)}) and continue with gpu-fault-admin deploy "
        f"--state-dir {state_dir} --registration-cluster {family[0]} "
        "--reference <CHG>"
    )


def require_registration_anchor(state_dir: Path) -> None:
    """Stop a deploy whose site anchor departed while members remain.

    Reads ``site.yaml`` and the removal journal only, so the refusal comes
    before the source snapshot and the release gates, with the way out. A
    state dir without a site, a site without members and a member anchor
    pass.
    """

    site_file = state_dir / "site.yaml"
    try:
        document = _document(site_file)
    except BootstrapError:
        return  # no site yet, or one the deploy's own reading reports
    departed = departed_anchor(document, state_dir)
    if departed is not None and _members(document):
        raise BootstrapError(departed_anchor_refusal(document, departed, state_dir))


def require_anchor_family(site_file: Path, providers: Iterable[object]) -> None:
    """Refuse a first join into an emptied site that brings no cluster of the
    departed anchor's provider.

    The site base profile family follows that provider; a member of another
    one only runs its own profile, and with no member to take over the
    anchor every later deploy would refuse the site. Nothing is changed.
    """

    document = _document(site_file)
    if _members(document):
        return
    departed = departed_anchor(document, site_file.parent)
    if departed is None or departed.provider is None:
        return
    joining = {cluster_provider(item) for item in providers}
    if cluster_provider(departed.provider) in joining:
        return
    raise BootstrapError(
        f"the site's Runtime Profile registration cluster {departed.cluster_id} "
        f"left the site (remove-cluster journal {departed.journal}) and the site "
        f"has no GPU cluster: the site base profile family follows its provider "
        f"({departed.provider}), so the first cluster to join must be a "
        f"{departed.provider} cluster, which takes over the anchor; join one "
        "first, or in the same join-cluster command. Nothing was changed"
    )


def _move_anchor(
    site_file: Path,
    *,
    departed: DepartedAnchor,
    cluster_id: str,
    trigger: str,
    reference: str | None,
) -> dict[str, Any]:
    path = site_file.parent / JOURNAL
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        journal = {"schema_version": 1, "entries": []}
    except (OSError, ValueError) as exc:
        raise BootstrapError(f"registration anchor journal {path} is invalid") from exc
    if not isinstance(journal, dict) or not isinstance(journal.get("entries"), list):
        raise BootstrapError(f"registration anchor journal {path} is invalid")
    before = site_file.read_bytes()
    document = yaml.safe_load(before)
    document["spec"]["runtimeProfile"]["registrationClusterId"] = cluster_id
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=site_file.parent,
        prefix=f".{site_file.name}.anchor.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        yaml.safe_dump(document, temporary, sort_keys=False)
        written = Path(temporary.name)
    written.chmod(0o600)
    written.replace(site_file)
    entry = {
        "at": datetime.now(UTC).isoformat(),
        "trigger": trigger,
        "reference": reference,
        "operator": operator_identity.local_operator_identity(),
        "previous": departed.cluster_id,
        "registration_cluster_id": cluster_id,
        "provider": departed.provider,
        "proof": str(departed.journal),
        "site_sha256_before": hashlib.sha256(before).hexdigest(),
        "site_sha256_after": hashlib.sha256(site_file.read_bytes()).hexdigest(),
    }
    journal["entries"] = [*journal["entries"], entry]
    write_json_atomic(path, journal)
    return entry


def fill_departed_anchor(site_file: Path) -> dict[str, str]:
    """``{"registration_reanchor": <id>}`` when a join filled the anchor.

    Called once the joined clusters are members (the joins' journals are
    final): a departed anchor whose provider has a member moves to the first
    of them (``_family``). A member anchor, an unproven departure and a
    provider without a member leave the site as it is (``{}``).
    """

    document = _document(site_file)
    departed = departed_anchor(document, site_file.parent)
    if departed is None or departed.provider is None:
        return {}
    family = _family(document, departed.provider)
    if not family:
        return {}
    _move_anchor(
        site_file,
        departed=departed,
        cluster_id=family[0],
        trigger=JOIN_TRIGGER,
        reference=None,
    )
    return {REGISTRATION_REANCHOR: family[0]}


def reanchor_registration(
    state_dir: Path, *, cluster_id: str, reference: str
) -> dict[str, Any]:
    """Move a departed anchor to ``cluster_id``, the operator's explicit choice.

    The exit for a site an older ``remove-cluster`` left with a departed
    anchor and members: the anchor must not name a member, its removal
    journal must prove it left, and ``cluster_id`` must be a member of the
    departed anchor's provider -- the base family never changes here. The
    caller holds the administrator operation lock.
    """

    if not APPROVAL_PATTERN.fullmatch(reference.strip()):
        raise BootstrapError("--reference has an invalid format")
    site_file = state_dir / "site.yaml"
    document = _document(site_file)
    departed = departed_anchor(document, state_dir)
    if departed is None:
        raise BootstrapError(
            f"registration cluster {_anchor(document)} is a member of the site; "
            "--registration-cluster moves only an anchor whose cluster left the "
            "site (remove-cluster moves it with the removal of that cluster)"
        )
    if departed.provider is None:
        raise BootstrapError(departed_anchor_refusal(document, departed, state_dir))
    family = _family(document, departed.provider)
    if cluster_id not in family:
        raise BootstrapError(
            f"{cluster_id} cannot take over the registration anchor: the site base "
            f"profile family follows the departed {departed.cluster_id} "
            f"({departed.provider}); {departed.provider} clusters: "
            + (", ".join(family) or "none")
        )
    return _move_anchor(
        site_file,
        departed=departed,
        cluster_id=cluster_id,
        trigger=DEPLOY_TRIGGER,
        reference=reference.strip(),
    )
