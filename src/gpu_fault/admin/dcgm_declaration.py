"""A plain EKS cluster's DCGM host engine declaration against its site entry.

``gpu-fault.io/dcgm-hostengine`` on the node groups a cluster's recorded
selector covers is node-image capability (``bootstrap_ownership``), re-read by
every deploy and join: the deploy's bootstrap discovery resolves it, rewrites
the entry's ``dcgmHostEngine``/``dcgmHostEngineSource``, and the release diff
then sees the ``dcgm_hostengines`` input move (DATA_PLANE_COMPATIBLE: node
preflight, exporter re-render, node installer reconciler, Agent re-install).

Between the label flip and that deploy the declaration differs from the
record, and no release input shows it: the engine classifies the recorded
entry. Live 2026-10-17 the operator labelled the group, ``status`` said
``next deploy: NOOP`` and a deploy of the unchanged source took the NOOP
shortcut without reading a single node group.

This module reads the declaration with discovery's own functions
(``bootstrap_discovery.eks_nodegroup_documents`` and ``recorded_scope_groups``,
``bootstrap_ownership.resolve_dcgm_hostengine``) for three readers:

* ``status`` (quick and ``--full``) reports a pending change as the next
  deploy -- ``next deploy: DATA_PLANE_COMPATIBLE (dcgm_hostengine: <cluster>
  declared host, recorded embedded)`` -- and a refusal (mixed or unknown label
  values) as what that deploy will do (``refused``); no check verdict moves;
* ``node-scope`` reports it per cluster;
* the deploy's NOOP shortcut (``scripts/staging_deploy.py``) takes the
  application release for a pending change and refuses a blocked one before
  anything changes. It reuses the reading of the quick status it already ran
  and reads the node groups itself only when that status came from a deploy
  host that does not read them.

Nothing here writes. The entry is rewritten only by the deploy's bootstrap,
inside the release path, and the success record is signed only after that
release, so a recorded mode no release was started for never stands as
success; the release state is what decides what was rolled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_discovery import (
    eks_nodegroup_documents,
    recorded_scope_groups,
)
from gpu_fault.admin.bootstrap_ownership import (
    NodeGroupFacts,
    eks_nodegroup_facts,
    resolve_dcgm_hostengine,
)
from gpu_fault.cluster_scope import (
    dcgm_hostengine,
    dcgm_hostengine_source,
    is_eks_managed,
    node_selector,
)
from gpu_fault_release.regional_release_dcgm import DCGM_HOSTENGINES_INPUT
from gpu_fault_release.regional_release_diff import diff_from_changed

__all__ = [
    "BLOCKED",
    "CONVERGED",
    "PENDING",
    "READ_ERRORS",
    "STATUS_FIELD",
    "UNREADABLE",
    "DcgmDeclaration",
    "declaration_from_facts",
    "declarations_document",
    "declarations_from_report",
    "eks_managed_entries",
    "next_deploy_with_declarations",
    "read_dcgm_declarations",
    "site_cluster_entries",
    "unreadable_findings",
]

#: The ``status`` document's field: each plain EKS cluster's declaration.
STATUS_FIELD = "dcgm_hostengine_declarations"
CONVERGED = "converged"
PENDING = "pending deploy"
BLOCKED = "blocked"
UNREADABLE = "unreadable"
#: What a read reports as ``unreadable`` instead of raising: the AWS CLI's
#: failures, an unreadable answer or entry. Anything else is a defect.
READ_ERRORS = (BootstrapError, OSError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class DcgmDeclaration:
    """One cluster's recorded mode beside what its node groups declare today."""

    cluster_id: str
    recorded: str
    recorded_source: str
    declared: str | None = None
    declared_source: str | None = None
    #: The node groups whose labels were counted (the recorded selector's).
    node_groups: tuple[str, ...] = ()
    #: The refusal the next deploy's discovery raises for this declaration.
    refusal: str | None = None
    #: Why the node groups could not be read.
    error: str | None = None

    @property
    def state(self) -> str:
        if self.error is not None:
            return UNREADABLE
        if self.refusal is not None:
            return BLOCKED
        return CONVERGED if self.declared == self.recorded else PENDING

    def summary(self) -> str:
        """The reason ``status`` names beside the next deploy."""

        prefix = f"dcgm_hostengine: {self.cluster_id}"
        state = self.state
        if state == UNREADABLE:
            return f"{prefix} declaration unreadable: {self.error}"
        if state == BLOCKED:
            return f"{prefix}: {self.refusal}"
        if state == PENDING:
            return f"{prefix} declared {self.declared}, recorded {self.recorded}"
        return f"{prefix} declared and recorded {self.recorded}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "recorded_source": self.recorded_source,
            "declared": self.declared,
            "declared_source": self.declared_source,
            "node_groups": list(self.node_groups),
            "state": self.state,
            "refusal": self.refusal,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, cluster_id: str, value: object) -> DcgmDeclaration | None:
        """The declaration a ``status`` document carries; None when malformed."""

        if not isinstance(value, Mapping):
            return None

        def text(name: str) -> str | None:
            item = value.get(name)
            return item if isinstance(item, str) else None

        groups = value.get("node_groups")
        recorded, recorded_source = text("recorded"), text("recorded_source")
        if (
            recorded is None
            or recorded_source is None
            or not isinstance(groups, list)
            or not all(isinstance(item, str) for item in groups)
        ):
            return None
        item = cls(
            cluster_id=cluster_id,
            recorded=recorded,
            recorded_source=recorded_source,
            declared=text("declared"),
            declared_source=text("declared_source"),
            node_groups=tuple(groups),
            refusal=text("refusal"),
            error=text("error"),
        )
        return item if value.get("state") == item.state else None


def _entry_text(entry: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = entry.get(name)
        if value:
            return str(value)
    return ""


def _recorded(entry: Mapping[str, Any]) -> DcgmDeclaration:
    """The entry's recorded mode, with nothing declared yet."""

    return DcgmDeclaration(
        cluster_id=_entry_text(entry, "clusterId", "cluster_id"),
        recorded=dcgm_hostengine(entry).value,
        recorded_source=dcgm_hostengine_source(entry).value,
    )


def site_cluster_entries(site_file: Path) -> list[Mapping[str, Any]]:
    """The recorded ``spec.clusters`` entries (either spelling is read alike)."""

    try:
        document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BootstrapError(f"cannot read {site_file}: {exc}") from exc
    spec = document.get("spec") if isinstance(document, Mapping) else None
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    return [
        item
        for item in (clusters if isinstance(clusters, list) else [])
        if isinstance(item, Mapping)
    ]


def eks_managed_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The plain EKS entries: HyperPod is always ``host`` and declares nothing."""

    return [entry for entry in entries if is_eks_managed(entry)]


def declaration_from_facts(
    entry: Mapping[str, Any],
    facts: Sequence[NodeGroupFacts],
    *,
    cluster: str,
    selector: str | None = None,
) -> DcgmDeclaration:
    """``entry``'s recorded mode against what ``facts`` declare, as discovery reads it.

    ``facts`` are the cluster's node groups; the ones that count are those the
    entry's recorded selector covers (``recorded_scope_groups``, what a deploy
    selects for a managed cluster) and the mode is ``resolve_dcgm_hostengine``'s,
    so either refusal is the one the next deploy's discovery raises. A HyperPod
    entry is ``host`` from the provider: declared is recorded.
    """

    base = _recorded(entry)
    if not is_eks_managed(entry):
        return replace(
            base, declared=base.recorded, declared_source=base.recorded_source
        )
    try:
        covered = recorded_scope_groups(
            facts,
            selector=selector if selector is not None else node_selector(entry),
            eks_name=cluster,
        )
        mode, source = resolve_dcgm_hostengine(covered, cluster=cluster)
    except BootstrapError as exc:
        return replace(base, refusal=str(exc))
    return replace(
        base,
        declared=mode.value,
        declared_source=source.value,
        node_groups=tuple(item.name for item in covered),
    )


def _read_declaration(
    entry: Mapping[str, Any], runner: CommandRunner
) -> DcgmDeclaration:
    try:
        parsed = Arn.parse(_entry_text(entry, "eksClusterArn", "eks_cluster_arn"))
        cluster = parsed.resource_name
        documents = eks_nodegroup_documents(
            runner, region=parsed.region, eks_name=cluster
        )
        return declaration_from_facts(
            entry, [eks_nodegroup_facts(item) for item in documents], cluster=cluster
        )
    except READ_ERRORS as exc:
        cluster_id = _entry_text(entry, "clusterId", "cluster_id")
        try:
            base = _recorded(entry)
        except ValueError:
            base = DcgmDeclaration(cluster_id, "unknown", "unknown")
        return replace(
            base, cluster_id=cluster_id, error=str(exc) or type(exc).__name__
        )


def read_dcgm_declarations(
    entries: Sequence[Mapping[str, Any]], runner: CommandRunner | None = None
) -> list[DcgmDeclaration]:
    """Every plain EKS entry's declaration, read now; a failed read is reported.

    Per cluster one ``list-nodegroups`` and one ``describe-nodegroup`` per
    group (discovery's read); HyperPod entries cost nothing.
    """

    selected = eks_managed_entries(entries)
    if not selected:
        return []
    active = runner or CommandRunner()
    return [_read_declaration(entry, active) for entry in selected]


def declarations_document(
    declarations: Sequence[DcgmDeclaration],
) -> dict[str, dict[str, Any]]:
    """``status``'s ``dcgm_hostengine_declarations`` field."""

    return {item.cluster_id: item.as_dict() for item in declarations}


def declarations_from_report(
    report: Mapping[str, Any] | None, entries: Sequence[Mapping[str, Any]]
) -> list[DcgmDeclaration] | None:
    """The declarations a ``status`` report already read for ``entries``, or None.

    None when the report has no such field (a deploy host that does not read
    them), describes another record (a cluster missing, a recorded mode that
    differs from ``entries``) or could not read one: the caller reads them
    itself then.
    """

    block = (report or {}).get(STATUS_FIELD)
    if not isinstance(block, Mapping):
        return None
    declarations = []
    for entry in eks_managed_entries(entries):
        recorded = _recorded(entry)
        item = DcgmDeclaration.from_dict(
            recorded.cluster_id, block.get(recorded.cluster_id)
        )
        if (
            item is None
            or item.state == UNREADABLE
            or (item.recorded, item.recorded_source)
            != (recorded.recorded, recorded.recorded_source)
        ):
            return None
        declarations.append(item)
    return declarations


def next_deploy_with_declarations(
    next_deploy: Mapping[str, Any], declarations: Sequence[DcgmDeclaration]
) -> dict[str, Any]:
    """``next_deploy`` as the next deploy finds it once it re-reads the labels.

    The engine classifies the recorded entries; the deploy's bootstrap rewrites
    a flipped entry first, so a pending declaration adds the
    ``dcgm_hostengines`` input -- at least DATA_PLANE_COMPATIBLE -- and names
    itself in ``pending``. A refusal is what the deploy does instead
    (``refused``). Converged and unreadable declarations change nothing (an
    unreadable one is a ``status`` finding).
    """

    value = dict(next_deploy)
    pending = [item.summary() for item in declarations if item.state == PENDING]
    refused = [item.summary() for item in declarations if item.state == BLOCKED]
    if pending:
        changed = {str(item) for item in value.get("changed") or ()}
        changed.add(DCGM_HOSTENGINES_INPUT)
        value["kind"] = diff_from_changed(changed).kind.value
        value["changed"] = sorted(changed)
        value["pending"] = pending
    if refused:
        value["refused"] = refused
    return value


def unreadable_findings(declarations: Sequence[DcgmDeclaration]) -> list[str]:
    """``status`` finding lines for the declarations that could not be read."""

    return [
        f"DCGM host engine declaration of {item.cluster_id} could not be read "
        f"({item.error}); the next deploy reads it again"
        for item in declarations
        if item.state == UNREADABLE
    ]
