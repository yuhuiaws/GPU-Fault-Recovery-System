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
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError

JOIN_VERB = "join-cluster"
REMOVE_VERB = "remove-cluster"
JOINING_FLAG = "--joining-cluster-id"
REMOVED_FLAG = "--removed-cluster-id"
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
    """Refuse a scope outside ``verify``, two scopes, or an empty cluster id."""

    joining = getattr(arguments, "joining_cluster_ids", None) or []
    removed = getattr(arguments, "removed_cluster_id", None)
    if not joining and removed is None:
        return
    if arguments.mode != "verify":
        raise ReleaseError(f"{JOINING_FLAG}/{REMOVED_FLAG} only scope a verify")
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
    """Give ``release`` the scope its ``verify`` was asked for (or none)."""

    release.verify_scope = scope_of(arguments, release)


def verify_scope(release: Any) -> VerifyScope | None:
    scope = getattr(release, "verify_scope", None)
    return scope if isinstance(scope, VerifyScope) else None


def gates(release: Any, cluster_id: object) -> bool:
    """Whether ``cluster_id``'s own findings fail this report (always unscoped)."""

    scope = verify_scope(release)
    return scope is None or scope.gates(str(cluster_id))


def not_gating_ids(release: Any, cluster_ids: Iterable[object]) -> frozenset[str]:
    return frozenset(str(item) for item in cluster_ids if not gates(release, item))


def record_not_gating(container: dict[str, Any], cluster_id: str, finding: str) -> None:
    """File one finding of a cluster the verb does not target under ``container``."""

    findings = container.setdefault(NOT_GATING_KEY, {})
    findings.setdefault(str(cluster_id), []).append(str(finding))


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
    """Name the scope and every finding that did not gate, at the report's top."""

    scope = verify_scope(release)
    if scope is None:
        return
    report[SCOPE_KEY] = {
        "verb": scope.verb,
        "target_cluster_ids": list(scope.targets),
        "not_gating_findings": [
            {"check": str(check.get("name")), "cluster_id": node, "finding": text}
            for check in report.get("checks") or []
            if isinstance(check, dict) and check.get("status") == "WARN"
            for node, text in _findings(check)
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
