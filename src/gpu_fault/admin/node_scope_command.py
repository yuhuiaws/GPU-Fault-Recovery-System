"""``gpu-fault-admin node-scope``: show a cluster's node scope; converge a legacy one.

A cluster joined under the ownership convention selects its nodes through
``cluster_scope.OWNERSHIP_LABEL=<site id>``. A cluster joined before it keeps
the selector it recorded (``NodeSelectorSource.PROVIDER_DEFAULT_LEGACY`` for
HyperPod's cluster-name label, ``FLAG_LEGACY`` for the retired plain-EKS
flags): a deploy never re-derives a managed cluster's membership, because a
quietly changed node set is exactly what the convention exists to prevent.

This verb is the explicit, journaled path from the recorded selector to the
label. Read-only by default, it shows per cluster what the site records,
what the node groups declare today, the two node sets side by side and what
the cluster's connection Secret -- the selector the data plane actually runs
-- serves (``data_plane``). ``--converge`` rewrites one cluster's site entry
to the label selector only when the label-derived node set is the current
node set to the name: a group the administrator forgot to label, a current
node the provider has not labelled yet (HyperPod applies ``KubernetesConfig``
labels through its own reconciliation) or a labelled group that would bring
new nodes each refuse, naming the nodes. The recorded scope is a release
input (``regional_release_node_scope``): the next ``deploy`` classifies the
change ``DATA_PLANE_COMPATIBLE``, re-renders the connection Secret and
restarts the executor, Completion Watcher and node installer reconciler that
read it, and its verify refuses a cluster whose Secret still disagrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import operator_identity
from gpu_fault.admin.bootstrap import discover_cluster
from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_ownership import (
    NodeGroupFacts,
    NodeScope,
    eks_nodegroup_facts,
    hyperpod_instance_group_facts,
    resolve_node_scope,
)
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.ownership_release import stale_claim_lines, stale_ownership_claims
from gpu_fault.admin.site import RenderedSite
from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_INSTANCE_GROUP_LABEL,
    OWNERSHIP_LABEL,
    ClusterProvider,
    NodeSelectorSource,
    cluster_provider,
    node_selector,
    node_selector_source,
    selector_matches,
)
from gpu_fault_release.regional_release_node_scope import (
    CONNECTION_SECRET,
    served_node_scope,
)

__all__ = [
    "COMMAND",
    "add_node_scope_command",
    "assess_cluster",
    "run_node_scope_command",
]

COMMAND = "node-scope"
JOURNAL_DIR = "node-scope"
CONVERGED = "converged"
READY = "ready"
BLOCKED = "blocked"
#: ``data_plane.state``: the connection Secret serves the recorded scope, the
#: cluster has no Secret yet (not bootstrapped), or ``deploy`` must still roll.
DATA_PLANE_CONVERGED = "converged"
DATA_PLANE_ABSENT = "absent"
DATA_PLANE_PENDING = "pending deploy"


def add_node_scope_command(
    commands: Any, add_managed_site_arguments: Callable[[argparse.ArgumentParser], None]
) -> None:
    parser = commands.add_parser(
        COMMAND,
        usage=(
            f"gpu-fault-admin {COMMAND} --state-dir STATE_DIR [--gpu-cluster-arn GPU_ARN] "
            "[--converge --reference REFERENCE [--dry-run]]"
        ),
        help=(
            "show each GPU cluster's recorded node selector, its source and what "
            f"the {OWNERSHIP_LABEL} labels select today; --converge rewrites one "
            "legacy cluster's entry to the label selector when both select the "
            "same nodes"
        ),
    )
    add_managed_site_arguments(parser)
    parser.add_argument(
        "--gpu-cluster-arn",
        metavar="GPU_ARN",
        help="one managed GPU cluster (EKS or HyperPod ARN); required with --converge",
    )
    parser.add_argument(
        "--converge",
        action="store_true",
        help=(
            "record the label-derived selector in the cluster's site entry "
            "(nodeSelectorSource: label-convention) and journal the change; refused "
            "unless the label selects exactly the current nodes. Follow with deploy"
        ),
    )
    parser.add_argument(
        "--reference",
        metavar="REFERENCE",
        help="change request recorded in the journal; required with --converge",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --converge: report the decision, write nothing",
    )


# --- facts -------------------------------------------------------------------


def _group_label(provider: ClusterProvider) -> str:
    return (
        EKS_NODEGROUP_LABEL
        if provider is ClusterProvider.EKS_MANAGED
        else HYPERPOD_INSTANCE_GROUP_LABEL
    )


def _node_group_facts(
    runner: CommandRunner, *, region: str, entry: Mapping[str, Any]
) -> tuple[list[NodeGroupFacts], str]:
    """The provider's node groups for ``entry`` and the cluster's display name."""

    provider = cluster_provider(entry.get("provider"))
    if provider is ClusterProvider.EKS_MANAGED:
        eks_name = str(entry["eks_cluster_arn"]).rsplit("/", 1)[-1]
        names = [
            str(item)
            for item in runner.aws_json(
                region, "eks", "list-nodegroups", "--cluster-name", eks_name
            ).get("nodegroups", [])
            if item
        ]
        facts = [
            eks_nodegroup_facts(
                runner.aws_json(
                    region,
                    "eks",
                    "describe-nodegroup",
                    "--cluster-name",
                    eks_name,
                    "--nodegroup-name",
                    name,
                ).get("nodegroup")
                or {}
            )
            for name in sorted(names)
        ]
        return facts, eks_name
    hyperpod_name = str(entry.get("hyperpod_cluster_name") or "")
    described = runner.aws_json(
        region, "sagemaker", "describe-cluster", "--cluster-name", hyperpod_name
    )
    facts = [
        hyperpod_instance_group_facts(group, cluster_name=hyperpod_name)
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    ]
    return facts, hyperpod_name


def _kubectl(
    site: RenderedSite, entry: Mapping[str, Any], *arguments: str
) -> list[str]:
    command = ["kubectl"]
    kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if kubeconfig:
        command.extend(["--kubeconfig", str(kubeconfig)])
    command.extend(["--context", str(entry["context"]), *arguments])
    return command


def _data_plane_scope(
    runner: CommandRunner, *, site: RenderedSite, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """What the cluster's connection Secret serves, against the recorded scope.

    The Secret is read as a sensitive command (its token is never printed) and
    only the two selector keys are decoded.
    """

    raw = runner.run(
        _kubectl(
            site,
            entry,
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "secret",
            CONNECTION_SECRET,
            "--ignore-not-found",
            "-o",
            "json",
        ),
        capture=True,
        sensitive=True,
        timeout_seconds=60,
    )
    if not raw.strip():
        return {"selector": None, "source": None, "state": DATA_PLANE_ABSENT}
    try:
        data = json.loads(raw).get("data")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise BootstrapError(
            f"{entry['cluster_id']}: the connection Secret is unreadable"
        ) from exc
    served = served_node_scope(data if isinstance(data, dict) else {}, entry)
    recorded = {
        "selector": node_selector(entry),
        "source": node_selector_source(entry).value,
    }
    return {
        **served,
        "state": DATA_PLANE_CONVERGED if served == recorded else DATA_PLANE_PENDING,
    }


def _nodes(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    entry: Mapping[str, Any],
    selector: str,
) -> dict[str, dict[str, str]]:
    """``{node name: labels}`` the API server returns for ``selector``, re-checked."""

    command = _kubectl(site, entry, "get", "nodes", "-l", selector, "-o", "json")
    try:
        document = json.loads(runner.run(command, capture=True, timeout_seconds=60))
        items = document["items"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BootstrapError(
            f"{entry['cluster_id']}: the node list for selector {selector!r} is unreadable"
        ) from exc
    nodes: dict[str, dict[str, str]] = {}
    for item in items:
        metadata = item.get("metadata") or {}
        labels = {str(k): str(v) for k, v in (metadata.get("labels") or {}).items()}
        name = str(metadata.get("name") or "")
        if name and selector_matches(labels, selector):
            nodes[name] = labels
    return nodes


# --- assessment --------------------------------------------------------------


def assess_cluster(
    runner: CommandRunner, *, site: RenderedSite, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """One cluster's recorded scope against the label convention, with node sets."""

    provider = cluster_provider(entry.get("provider"))
    region = str(entry.get("region") or site.release_config["aws_region"])
    site_id = str(site.release_config["site_name"])
    current_selector = node_selector(entry)
    current_source = node_selector_source(entry)
    facts, cluster = _node_group_facts(runner, region=region, entry=entry)
    report: dict[str, Any] = {
        "cluster_id": entry["cluster_id"],
        "provider": provider.value,
        "selector": current_selector,
        "source": current_source.value,
        "node_groups": [item.describe() for item in facts],
        "label": {"selector": None, "owned_groups": [], "refusal": None},
        "nodes": {},
        "data_plane": _data_plane_scope(runner, site=site, entry=entry),
    }
    try:
        scope: NodeScope | None = resolve_node_scope(
            facts,
            provider=provider,
            cluster=cluster,
            site_id=site_id,
            hyperpod_cluster_name=str(entry.get("hyperpod_cluster_name") or ""),
        )
    except BootstrapError as exc:
        scope = None
        report["label"]["refusal"] = str(exc)
    current_nodes = _nodes(runner, site=site, entry=entry, selector=current_selector)
    report["nodes"]["current"] = sorted(current_nodes)
    if scope is None:
        report["convergence"] = f"{BLOCKED}: {report['label']['refusal']}"
        return report
    report["label"]["selector"] = scope.selector
    report["label"]["owned_groups"] = list(scope.owned_groups)
    label_nodes = _nodes(runner, site=site, entry=entry, selector=scope.selector)
    group_label = _group_label(provider)
    would_join = sorted(set(label_nodes) - set(current_nodes))
    would_leave = sorted(set(current_nodes) - set(label_nodes))
    unlabelled = sorted(
        name
        for name in would_leave
        if current_nodes[name].get(group_label) in scope.owned_groups
    )
    report["nodes"].update(
        label=sorted(label_nodes),
        would_join=would_join,
        would_leave=would_leave,
        owned_group_nodes_without_label=unlabelled,
    )
    if would_join or would_leave:
        reasons = []
        if unlabelled:
            reasons.append(
                "nodes of a labelled group do not carry the label yet: "
                + ", ".join(unlabelled)
                + " (HyperPod applies KubernetesConfig labels through its own "
                "reconciliation; label the nodes or wait, then retry)"
            )
        departing = sorted(set(would_leave) - set(unlabelled))
        if departing:
            reasons.append(
                "nodes the label does not select would leave the managed set: "
                + ", ".join(departing)
                + " (label their group, or detach the cluster and re-join it)"
            )
        if would_join:
            reasons.append(
                "nodes the label selects are not managed today: "
                + ", ".join(would_join)
                + " (remove the label from their group, or detach and re-join)"
            )
        report["convergence"] = f"{BLOCKED}: " + "; ".join(reasons)
        return report
    report["convergence"] = (
        CONVERGED
        if current_source is NodeSelectorSource.LABEL_CONVENTION
        and current_selector == scope.selector
        else READY
    )
    return report


# --- converge ----------------------------------------------------------------


def _write_site(source: Path, document: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=source.parent,
        prefix=f".{source.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        yaml.safe_dump(document, temporary, sort_keys=False)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(source)


def _journal(state_dir: Path, cluster_id: str) -> tuple[Path, list[dict[str, Any]]]:
    directory = state_dir / JOURNAL_DIR / cluster_id
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "journal.json"
    entries: list[dict[str, Any]] = []
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            entries = [item for item in loaded if isinstance(item, dict)]
    return path, entries


def converge_cluster(
    site: RenderedSite,
    report: Mapping[str, Any],
    *,
    reference: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the label selector in the site entry and append the journal."""

    if report["convergence"] not in (READY, CONVERGED):
        raise BootstrapError(
            f"{report['cluster_id']} cannot be converged: {report['convergence']}"
        )
    state_dir = site.source.parent
    cluster_id = str(report["cluster_id"])
    label_selector = str(report["label"]["selector"])
    current = site.source.read_bytes()
    document = yaml.safe_load(current)
    clusters = document["spec"]["clusters"]
    matching = [item for item in clusters if item.get("clusterId") == cluster_id]
    if len(matching) != 1:
        raise BootstrapError(f"site.yaml has no unique entry for {cluster_id}")
    entry = matching[0]
    previous = {
        "selector": str(entry.get("nodeSelector") or ""),
        "source": str(entry.get("nodeSelectorSource") or report["source"]),
    }
    journal_path, entries = _journal(state_dir, cluster_id)
    attempt = len(entries) + 1
    backup = state_dir / f"site.before-node-scope-{cluster_id}-{attempt:03d}.yaml"
    backup.write_bytes(current)
    backup.chmod(0o600)
    entry["nodeSelector"] = label_selector
    entry["nodeSelectorSource"] = NodeSelectorSource.LABEL_CONVENTION.value
    _write_site(site.source, document)
    record = {
        "at": (now or datetime.now(UTC)).isoformat(),
        "reference": reference,
        "operator": operator_identity.local_operator_identity(),
        "cluster_id": cluster_id,
        "previous": previous,
        "converged": {
            "selector": label_selector,
            "source": NodeSelectorSource.LABEL_CONVENTION.value,
        },
        "owned_groups": list(report["label"]["owned_groups"]),
        "nodes": list(report["nodes"]["current"]),
        "site_sha256_before": hashlib.sha256(current).hexdigest(),
        "site_sha256_after": hashlib.sha256(site.source.read_bytes()).hexdigest(),
        "site_backup": str(backup),
    }
    entries.append(record)
    journal_path.write_text(
        json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8"
    )
    journal_path.chmod(0o600)
    return {**record, "journal": str(journal_path)}


# --- command -----------------------------------------------------------------


def _selected_entries(
    site: RenderedSite, gpu_cluster_arn: str | None, runner: CommandRunner
) -> list[dict[str, Any]]:
    entries = [dict(item) for item in site.release_config["clusters"]]
    if not gpu_cluster_arn:
        return entries
    arn = gpu_cluster_arn.strip()
    eks_arn, hyperpod_name = arn, ""
    if Arn.parse(arn).service == "sagemaker":
        identity = discover_cluster(
            runner,
            cluster_arn=arn,
            role="gpu",
            context="gpu-fault-admin-node-scope",
            resolve_node_scope=False,
        )
        eks_arn, hyperpod_name = identity.eks_arn, identity.hyperpod_name
    selected = [
        item
        for item in entries
        if item.get("eks_cluster_arn") == eks_arn
        or (hyperpod_name and item.get("hyperpod_cluster_name") == hyperpod_name)
    ]
    if len(selected) != 1:
        raise BootstrapError(f"{arn} is not a managed GPU cluster of this site")
    return selected


def run_node_scope_command(
    arguments: argparse.Namespace,
    *,
    site: RenderedSite,
    runner: CommandRunner | None = None,
    now: datetime | None = None,
) -> int:
    converge = bool(getattr(arguments, "converge", False))
    reference = getattr(arguments, "reference", None)
    gpu_cluster_arn = getattr(arguments, "gpu_cluster_arn", None)
    dry_run = bool(getattr(arguments, "dry_run", False))
    if converge and not gpu_cluster_arn:
        raise BootstrapError("--converge needs --gpu-cluster-arn")
    if converge and not reference:
        raise BootstrapError("--converge needs --reference")
    if reference and not converge:
        raise BootstrapError("--reference is only used with --converge")
    active = runner or CommandRunner()
    entries = _selected_entries(site, gpu_cluster_arn, active)
    reports = [assess_cluster(active, site=site, entry=entry) for entry in entries]
    # A cluster the site de-registered whose node groups still carry this
    # site's id: the next deploy of this id would re-adopt them without a new
    # declaration, and another site is refused until the label is released.
    stale_claims = stale_ownership_claims(site, active)
    result: dict[str, Any] = {
        "site_id": site.release_config["site_name"],
        "ownership_label": OWNERSHIP_LABEL,
        "clusters": reports,
        "stale_ownership_claims": stale_claims,
    }
    if converge:
        (report,) = reports
        if report["convergence"] not in (READY, CONVERGED):
            print(json.dumps(result, indent=2, sort_keys=True))
            raise BootstrapError(
                f"{report['cluster_id']} cannot be converged: {report['convergence']}"
            )
        if report["convergence"] == CONVERGED:
            result["status"] = "NOOP"
        elif dry_run:
            result["status"] = "DRY_RUN"
            result["would_record"] = {
                "nodeSelector": report["label"]["selector"],
                "nodeSelectorSource": NodeSelectorSource.LABEL_CONVENTION.value,
            }
        else:
            with administrator_operation_lock(site.source.parent):
                result["status"] = "CONVERGED"
                result["journal_entry"] = converge_cluster(
                    site, report, reference=str(reference), now=now
                )
            result["next_step"] = (
                f"gpu-fault-admin deploy --state-dir {site.source.parent} classifies "
                "the change DATA_PLANE_COMPATIBLE, re-renders the connection Secret "
                "and restarts the executor, Completion Watcher and node installer "
                "reconciler that read it"
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"gpu-fault-admin {COMMAND}: "
        + "; ".join(
            f"{item['cluster_id']} {item['source']} -> {item['convergence']}; "
            f"data plane {item['data_plane']['state']}"
            for item in reports
        ),
        file=sys.stderr,
        flush=True,
    )
    for line in stale_claim_lines(stale_claims):
        print(
            f"gpu-fault-admin {COMMAND}: finding: {line}", file=sys.stderr, flush=True
        )
    return 0
