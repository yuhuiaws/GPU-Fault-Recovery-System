"""A ``verify`` scoped to the GPU cluster a membership verb changes.

``join-cluster`` and ``remove-cluster`` end with one ``verify`` of the whole
candidate (or remaining) site, and every GPU cluster's own checks used to gate
it. So a node lost on one cluster failed -- and rolled back -- the join of
another: live 2026-11-04 an EC2 Spot interruption took a plain EKS GPU node
while a HyperPod cluster joined; its Agent was still ACTIVE and its NotReady
Node still listed, ``control_api`` failed on ``gpu-eks-cluster Agent coverage
drift`` and the HyperPod join was rolled back, though the join had not touched
that cluster.

GPU data-plane clusters are independent, so the verb asks for a verify scoped
to its target (engine-internal flags, never typed by an operator)::

    verify --config <cfg> --joining-cluster-id <id> [--joining-cluster-id ...]
    verify --config <cfg> --removed-cluster-id <id>

What still gates, exactly as in every verify -- the shared control plane: the
CPU part of ``regional_contexts``, ``cpu_secrets``, ``cpu_workloads``,
``email_notifications``, ``aurora_credential_refresh``, ``adot_self_metrics``,
``runtime_profile``, the CPU role split of ``read_only_verifiers``, the CPU
Pods of ``runtime_component_identity``, the shared half of ``control_api``
(``/healthz``, regional mode, every required and compatible pin and window,
the regional registry against the site, the remote commands and executor
errors the Store cannot attribute to a cluster), ``nlb_runtime``, ``aurora``
and ``monitoring``.

What gates for the target cluster(s) only -- each cluster's own data plane:
its ``gpu_cluster:<id>`` check (Deployments, node scope, Ready nodes,
installers, executor readiness and its TLS path to the control plane), its
section of ``control_api`` (Agent coverage, Agent Runtime Profile, fleet
readiness, collector readiness, its unclaimed remote commands and its
executor's internal errors -- ``regional_admin_checks_remote``), its executor
verifier in ``read_only_verifiers``, its Pods in ``runtime_component_identity`` and its
part of ``regional_contexts`` (context, provider facts, API ``/readyz``, Agent
endpoint CIDRs, executor IAM, retired collectors).

Another cluster's finding there never fails the report: its check reads
``WARN``, names it under ``details.not_gating`` (cluster -> findings) and the
report lists every one under ``cluster_scope.not_gating_findings``, which the
verb copies into its result and into the deploy host's record ``status``
reads. A ``remove-cluster`` targets the cluster it removed, which the
remaining site no longer names, so there every remaining cluster's own checks
are findings and the shared control plane is what gates. An unscoped verify
(``deploy``, ``config``) judges every cluster, as before.

The same scope covers the verb's other engine calls, so another cluster's
node loss (or any finding of its own) neither refuses nor rolls the verb back
at an earlier step either:

* the join's candidate ``preflight`` (``--joining-cluster-id``): another
  cluster's part of ``regional_contexts`` and its blockers in
  ``workflow_safety`` are findings (``WARN``, ``cluster_scope``);
* the ``join-cluster`` mode, always scoped to its ``--cluster-id``: another
  cluster's context, its open remote commands (the mode's idle check) and its
  open commands and destructive workflows in every fleet wave gate are
  findings, narrated as ``not-gating`` lines -- the waves restart Agents of
  the joining cluster only;
* ``sync-state`` (``--joining-cluster-id`` / ``--removed-cluster-id``):
  another cluster's Agent set is a finding, not a refusal to commit;
* ``drain-cluster`` scopes its idle check to the clusters it drains.

Counts the Store or a probe attributes to clusters gate unless they are
another cluster of the site (``other_cluster``): the target's, those of a
cluster the site does not name and those nothing could attribute
(``UNATTRIBUTED``) still gate, and so does every count whose per-cluster
breakdown is missing or does not add up to the fleet's. A ``deploy``,
``config``, ``status`` and every other mode bind no scope.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_narration import narrate_step

JOIN_VERB = "join-cluster"
REMOVE_VERB = "remove-cluster"
#: The idle check of ``drain-cluster``: the drained clusters' commands gate.
DRAIN_VERB = "drain-cluster"
JOINING_FLAG = "--joining-cluster-id"
REMOVED_FLAG = "--removed-cluster-id"
#: The engine modes the two flags scope (``join-cluster`` scopes itself).
SCOPED_MODES = ("verify", "preflight", "sync-state")
#: A probe's cluster key for a record it could not place: gates every scope.
UNATTRIBUTED = ""
#: Per check details: cluster id -> the findings that did not gate this verify.
NOT_GATING_KEY = "not_gating"
#: Report key: the verb, its targets and every finding that did not gate.
SCOPE_KEY = "cluster_scope"
#: How many findings a check's one-line summary names before counting the rest.
SUMMARY_LIMIT = 3


@dataclass(frozen=True)
class VerifyScope:
    """The membership verb a verify answers for, and the clusters that gate."""

    verb: str
    targets: tuple[str, ...]
    #: The ``not-gating`` lines already printed (a wave gate polls).
    narrated: set[tuple[str, str, str]] = field(
        default_factory=set, compare=False, hash=False, repr=False
    )

    def gates(self, cluster_id: str) -> bool:
        return cluster_id in self.targets

    @property
    def description(self) -> str:
        return f"{self.verb} {','.join(self.targets)}"


def add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """The two engine-internal ``verify`` flags, hidden from ``--help``."""

    parser.add_argument(
        JOINING_FLAG,
        action="append",
        dest="joining_cluster_ids",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(REMOVED_FLAG, dest="removed_cluster_id", help=argparse.SUPPRESS)


def validate_scope_arguments(arguments: argparse.Namespace) -> None:
    """Refuse a scope outside ``SCOPED_MODES`` or of a deploy's preflight, two
    scopes, or an empty cluster id."""

    joining = getattr(arguments, "joining_cluster_ids", None) or []
    removed = getattr(arguments, "removed_cluster_id", None)
    if not joining and removed is None:
        return
    if arguments.mode not in SCOPED_MODES:
        raise ReleaseError(
            f"{JOINING_FLAG}/{REMOVED_FLAG} only scope {', '.join(SCOPED_MODES)}"
        )
    if getattr(arguments, "for_deploy", False):
        raise ReleaseError(f"{JOINING_FLAG}/{REMOVED_FLAG} never scope a deploy")
    if joining and removed is not None:
        raise ReleaseError(f"{JOINING_FLAG} and {REMOVED_FLAG} exclude each other")
    named = [*joining, *([] if removed is None else [removed])]
    if any(not str(value).strip() for value in named):
        raise ReleaseError(f"{JOINING_FLAG}/{REMOVED_FLAG} values must be non-empty")


def scope_of(arguments: argparse.Namespace, release: Any) -> VerifyScope | None:
    """The scope a verify was asked for, judged against the site it verifies.

    A joining cluster must be one the (candidate) site names; a removed one
    must be one it no longer names. Either mistake is refused rather than
    verified leniently. Without a scope flag nothing is read.
    """

    validate_scope_arguments(arguments)
    joining = sorted(set(getattr(arguments, "joining_cluster_ids", None) or []))
    removed = getattr(arguments, "removed_cluster_id", None)
    if arguments.mode == JOIN_VERB:
        # The mode joins its --cluster-id (``join_target`` validates it).
        return VerifyScope(JOIN_VERB, (str(arguments.cluster_id),))
    if not joining and removed is None:
        return None
    configured = {str(target.cluster_id) for target in release.config.clusters}
    if joining:
        unknown = [value for value in joining if value not in configured]
        if unknown:
            raise ReleaseError(
                f"{JOINING_FLAG} names clusters the site does not: {', '.join(unknown)}"
            )
        return VerifyScope(JOIN_VERB, tuple(joining))
    if removed in configured:
        raise ReleaseError(
            f"{REMOVED_FLAG} {removed} is still in the site; a removal verifies "
            "the remaining site"
        )
    return VerifyScope(REMOVE_VERB, (str(removed),))


def bind_scope(release: Any, arguments: argparse.Namespace) -> None:
    """Give ``release`` the scope its mode was asked for (or none)."""

    release.verify_scope = scope_of(arguments, release)


@contextmanager
def scoped_to(release: Any, scope: VerifyScope) -> Iterator[None]:
    """``release`` scoped to ``scope`` for the block, then as it was."""

    previous = getattr(release, "verify_scope", None)
    release.verify_scope = scope
    try:
        yield
    finally:
        release.verify_scope = previous


def verify_scope(release: Any) -> VerifyScope | None:
    scope = getattr(release, "verify_scope", None)
    return scope if isinstance(scope, VerifyScope) else None


def gates(release: Any, cluster_id: object) -> bool:
    """Whether ``cluster_id``'s own findings fail this report (always unscoped)."""

    scope = verify_scope(release)
    return scope is None or scope.gates(str(cluster_id))


def other_cluster(release: Any, cluster_id: object) -> bool:
    """Whether a count attributed to ``cluster_id`` is another site cluster's,
    reported instead of gated on: never unscoped, never the target's, never
    a cluster the site does not name, never ``UNATTRIBUTED``."""

    scope = verify_scope(release)
    cluster = str(cluster_id)
    if scope is None or scope.gates(cluster) or cluster == UNATTRIBUTED:
        return False
    return cluster in {str(target.cluster_id) for target in release.config.clusters}


def not_gating_ids(release: Any, cluster_ids: Iterable[object]) -> frozenset[str]:
    return frozenset(str(item) for item in cluster_ids if not gates(release, item))


def record_not_gating(container: dict[str, Any], cluster_id: str, finding: str) -> None:
    """File one finding of a cluster the verb does not target under ``container``."""

    findings = container.setdefault(NOT_GATING_KEY, {})
    findings.setdefault(str(cluster_id), []).append(str(finding))


def narrate_not_gating(release: Any, check: str, cluster_id: str, finding: str) -> None:
    """One ``not-gating`` log line per finding: a mode without a report (the
    ``join-cluster`` mode, ``sync-state``) records them in its log."""

    scope = verify_scope(release)
    key = (check, str(cluster_id), str(finding))
    if scope is None or key in scope.narrated:
        return
    scope.narrated.add(key)
    narrate_step(
        "not-gating",
        verb=scope.verb,
        targets=",".join(scope.targets),
        cluster=cluster_id,
        check=check,
        finding=json.dumps(str(finding)),
    )


def file_not_gating(
    release: Any, container: dict[str, Any], check: str, cluster_id: str, finding: str
) -> None:
    """``record_not_gating`` under ``container`` and ``narrate_not_gating``."""

    record_not_gating(container, cluster_id, finding)
    narrate_not_gating(release, check, cluster_id, finding)


def _counts(value: object) -> dict[str, int] | None:
    """A ``cluster -> count`` mapping of non-negative integers, else None."""

    if not isinstance(value, Mapping):
        return None
    try:
        counts = {str(key): int(count) for key, count in value.items()}
    except (TypeError, ValueError):
        return None
    return None if any(count < 0 for count in counts.values()) else counts


def gating_count(
    release: Any,
    container: dict[str, Any],
    *,
    check: str,
    total: int,
    by_cluster: object,
    finding: Callable[[str, int], str],
) -> int:
    """How much of ``total`` (a fleet-wide count of blockers) gates this call.

    All of it unscoped, or when ``by_cluster`` (cluster -> count) is not a
    breakdown that adds up to ``total``. Otherwise every count but another
    site cluster's (``other_cluster``); each of those is filed under
    ``container`` and narrated, worded by ``finding``.
    """

    counts = _counts(by_cluster) if total > 0 else None
    if verify_scope(release) is None or counts is None:
        return total
    if sum(counts.values()) != total:
        return total
    gating = 0
    for cluster_id, count in sorted(counts.items()):
        if not other_cluster(release, cluster_id):
            gating += count
        elif count:
            file_not_gating(
                release, container, check, cluster_id, finding(cluster_id, count)
            )
    return gating


def gating_open_commands(
    release: Any,
    container: dict[str, Any],
    *,
    check: str,
    open_by_status: Mapping[str, int],
    by_cluster_status: object,
) -> dict[str, int]:
    """The open remote commands (status -> count) that gate this call.

    All of them unscoped, or when ``by_cluster_status`` (cluster -> status ->
    count, the Store's breakdown) does not add up to ``open_by_status`` for
    every status. Otherwise every cluster's but another site cluster's, which
    is filed and narrated as ``open remote commands: <status>=<n>``.
    """

    statuses = {str(key): int(count or 0) for key, count in open_by_status.items()}
    if verify_scope(release) is None or not any(statuses.values()):
        return statuses
    if not isinstance(by_cluster_status, Mapping):
        return statuses
    per_cluster: dict[str, dict[str, int]] = {}
    for cluster_id, cells in by_cluster_status.items():
        counts = _counts(
            {key: value for key, value in cells.items() if key in statuses}
            if isinstance(cells, Mapping)
            else None
        )
        if counts is None:
            return statuses
        per_cluster[str(cluster_id)] = counts
    if any(
        sum(cells.get(status, 0) for cells in per_cluster.values()) != count
        for status, count in statuses.items()
    ):
        return statuses
    gating = {status: 0 for status in statuses}
    for cluster_id, cells in sorted(per_cluster.items()):
        opened = {status: count for status, count in sorted(cells.items()) if count}
        if not opened:
            continue
        if not other_cluster(release, cluster_id):
            for status, count in opened.items():
                gating[status] += count
            continue
        text = "open remote commands: " + ", ".join(
            f"{status}={count}" for status, count in opened.items()
        )
        file_not_gating(release, container, check, cluster_id, text)
    return gating


def listed(ids: object, cluster_id: str, limit: int = 5) -> str:
    """The first ``limit`` record ids a probe listed for ``cluster_id``."""

    values = ids.get(cluster_id) if isinstance(ids, Mapping) else None
    names = [str(item) for item in values or ()] if isinstance(values, list) else []
    more = f" +{len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def scoped_cluster_check(
    release: Any, cluster_id: str, function: Callable[[], CheckValue]
) -> Callable[[], CheckValue]:
    """``function`` as is for a gating cluster; otherwise a failure is a finding."""

    scope = verify_scope(release)
    if scope is None or scope.gates(cluster_id):
        return function

    def judged() -> CheckValue:
        try:
            return function()
        except Exception as exc:
            details: dict[str, Any] = {"failure_details": CheckFailure.details_of(exc)}
            record_not_gating(details, cluster_id, str(exc))
            return CheckValue(
                f"{cluster_id} is not a target of this {scope.verb}", details
            )

    return judged


def judge_clusters(
    release: Any,
    report: dict[str, Any],
    judge: Callable[[Any, str, dict[str, Any]], None],
) -> None:
    """Run ``judge`` per cluster of ``report``; file the other clusters' failures."""

    for cluster_id, cluster in (report.get("clusters") or {}).items():
        if gates(release, cluster_id):
            judge(release, cluster_id, cluster)
            continue
        try:
            judge(release, cluster_id, cluster)
        except Exception as exc:
            record_not_gating(report, cluster_id, str(exc))


def _findings(check: Mapping[str, Any]) -> list[tuple[str, str]]:
    details = check.get("details")
    findings = details.get(NOT_GATING_KEY) if isinstance(details, dict) else None
    return [
        (str(cluster_id), str(finding))
        for cluster_id, items in sorted(
            (findings if isinstance(findings, dict) else {}).items()
        )
        for finding in (items if isinstance(items, list) else [])
    ]


def mark_not_gating(release: Any, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A passing check that filed another cluster's findings reads ``WARN``."""

    scope = verify_scope(release)
    if scope is None:
        return checks
    marked = []
    for check in checks:
        findings = _findings(check)
        if findings and check.get("status") in {"PASS", "WARN"}:
            shown = "; ".join(
                f"{node}: {text}" for node, text in findings[:SUMMARY_LIMIT]
            )
            if len(findings) > SUMMARY_LIMIT:
                shown += (
                    f"; +{len(findings) - SUMMARY_LIMIT} more in details.not_gating"
                )
            check = {
                **check,
                "status": "WARN",
                "summary": f"{check.get('summary')}; not gating this "
                f"{scope.description} (other clusters' own findings): {shown}",
            }
        marked.append(check)
    return marked


def annotate_scope(release: Any, report: dict[str, Any]) -> None:
    """Name the scope and every finding that did not gate, at the report's top:
    the other clusters' own findings, and the zero-node findings of a cluster
    without nodes that is not parked (``zero_node_clusters``,
    ``regional_parked_cluster``), which gate no verify or preflight."""

    scope = verify_scope(release)
    if scope is None:
        return
    zero_nodes = report.get("zero_node_clusters")
    report[SCOPE_KEY] = {
        "verb": scope.verb,
        "target_cluster_ids": list(scope.targets),
        "not_gating_findings": [
            *(
                {"check": str(check.get("name")), "cluster_id": node, "finding": text}
                for check in report.get("checks") or []
                if isinstance(check, dict) and check.get("status") == "WARN"
                for node, text in _findings(check)
            ),
            *(
                {
                    "check": str(item.get("check")),
                    "cluster_id": str(cluster_id),
                    "finding": str(item.get("finding")),
                }
                for cluster_id, entry in sorted(
                    (zero_nodes if isinstance(zero_nodes, dict) else {}).items()
                )
                if isinstance(entry, dict)
                for item in entry.get("not_gating_findings") or ()
                if isinstance(item, dict)
            ),
        ],
    }


def not_gating_findings(report: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """The findings a scoped verify report did not gate on; empty otherwise."""

    scope = (report or {}).get(SCOPE_KEY)
    findings = scope.get("not_gating_findings") if isinstance(scope, dict) else None
    return [
        {key: str(item.get(key) or "") for key in ("check", "cluster_id", "finding")}
        for item in (findings if isinstance(findings, list) else [])
        if isinstance(item, dict)
    ]


#: The fleet wave probe's per-cluster breakdowns (``rollout_wave_safety.py``).
OPEN_REMOTE_BY_CLUSTER = "open_remote_by_cluster"
DESTRUCTIVE_BY_CLUSTER = "destructive_workflow_count_by_cluster"
DESTRUCTIVE_IDS_BY_CLUSTER = "destructive_workflows_by_cluster"
#: Its never-dispatched lists (``regional_release_departed_nodes`` and
#: ``regional_release_departure_absorption`` read them under these names).
NEVER_DISPATCHED_KEYS = (
    "never_dispatched_destructive_workflows",
    "never_dispatched_young_destructive_workflows",
)
WAVE_CHECK = "fleet_wave_safety"


def scoped_wave_safety(release: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """The fleet wave gate's evidence as a membership verb's waves judge it.

    A ``join-cluster`` restarts Agents of the joining cluster only, so another
    site cluster's open remote commands and destructive workflows -- the
    record the product opens for a node a spot reclaim took, a remediation in
    progress there -- cannot be dropped mid-way by its waves: they are
    findings, and ``open_remote``, ``destructive_workflow_count``, its ids and
    the never-dispatched lists keep the gating clusters' only. Unscoped (an
    upgrade's or a rollback's waves) and without a breakdown that adds up,
    the evidence is returned as it is.
    """

    if verify_scope(release) is None:
        return snapshot
    result = dict(snapshot)
    result["open_remote"] = gating_open_commands(
        release,
        result,
        check=WAVE_CHECK,
        open_by_status=dict(snapshot.get("open_remote") or {}),
        by_cluster_status=snapshot.get(OPEN_REMOTE_BY_CLUSTER),
    )
    total = int(snapshot.get("destructive_workflow_count") or 0)
    ids = snapshot.get(DESTRUCTIVE_IDS_BY_CLUSTER)
    gating = gating_count(
        release,
        result,
        check=WAVE_CHECK,
        total=total,
        by_cluster=snapshot.get(DESTRUCTIVE_BY_CLUSTER),
        finding=lambda cluster, count: (
            f"{count} active destructive workflow(s): {listed(ids, cluster)}"
        ),
    )
    if gating == total:
        return result
    keep = {
        str(cluster)
        for cluster in (ids if isinstance(ids, Mapping) else {})
        if not other_cluster(release, cluster)
    }
    result["destructive_workflow_count"] = gating
    result["destructive_workflows"] = [
        str(item) for cluster in sorted(keep) for item in (ids or {}).get(cluster) or []
    ][:20]
    for key in NEVER_DISPATCHED_KEYS:
        result[key] = [
            entry
            for entry in snapshot.get(key) or []
            if isinstance(entry, dict)
            and not other_cluster(release, entry.get("cluster_id") or "")
        ]
    return result
