"""GF-REGIONAL-DESTR-013 on an eks-managed registration: the EC2/EKS readings
and judgements the audit dispatches to instead of the SageMaker ones.

Judged for the executor role only: on plain EKS the node group, Auto Scaling
and Karpenter launch and terminate instances legitimately, so HyperPod's
actor-independent rule for its SageMaker verbs does not transfer; refused calls
(an ``errorCode``, Karpenter's ``RunInstances`` dry-runs) are set aside and
counted. Pure functions except ``eks_iam_decisions``, which runs the IAM
simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.destr013_audit_evidence import (  # noqa: E402
    AuditError,
    arn_parts,
    json_object,
    run,
)
from scripts.e2e.regional.guardrail_audit_evidence import (  # noqa: E402
    EC2_REPLACEMENT_MUTATIONS,
)

# The eks-managed vocabulary, in ``ec2.amazonaws.com``.
EKS_MANAGED = "eks-managed"
EKS_EVENT_SOURCE = "ec2.amazonaws.com"
EKS_POSITIVE_CONTROL_EVENT = "RebootInstances"
EKS_FORBIDDEN_EVENT_NAMES = tuple(sorted(EC2_REPLACEMENT_MUTATIONS))
EKS_EVENT_NAMES = (*EKS_FORBIDDEN_EVENT_NAMES, EKS_POSITIVE_CONTROL_EVENT)
EKS_IAM_ACTIONS = (
    "ec2:RebootInstances",
    *(f"ec2:{name}" for name in EKS_FORBIDDEN_EVENT_NAMES),
)
# The executor policy grants ec2:RebootInstances under this tag condition; the
# simulation supplies it so the condition is evaluated, not left unanswered.
EKS_CLUSTER_TAG_CONTEXT = "aws:ResourceTag/eks:cluster-name"


def eks_iam_decisions(
    region: str,
    role_arn: str,
    *,
    cluster_arn: str,
    eks_name: str,
    instance_ids: list[str],
) -> dict[str, str]:
    """``ec2:RebootInstances`` on the GPU instances (with the tag condition the
    executor policy carries supplied as context) and every EC2 replacement
    verb, simulated against the instances themselves, never ``*``."""

    eks = arn_parts(cluster_arn, "eks")
    prefix = f"arn:{eks[1]}:ec2:{region}:{eks[4]}:instance/"
    resources = [f"{prefix}{instance_id}" for instance_id in instance_ids] or [
        f"{prefix}*"
    ]
    value = json_object(
        run(
            [
                "aws",
                "iam",
                "simulate-principal-policy",
                "--policy-source-arn",
                role_arn,
                "--action-names",
                *EKS_IAM_ACTIONS,
                "--resource-arns",
                *resources,
                "--context-entries",
                f"ContextKeyName={EKS_CLUSTER_TAG_CONTEXT},ContextKeyValues={eks_name},"
                "ContextKeyType=string",
                "--region",
                region,
                "--output",
                "json",
            ]
        ).stdout,
        "IAM simulation",
    )
    decisions: dict[str, str] = {}
    for item in value.get("EvaluationResults", []):
        if not isinstance(item, dict):
            raise AuditError("IAM simulation result is malformed")
        decision = str(item.get("EvalDecision"))
        per_resource = {
            str(row.get("EvalResourceDecision"))
            for row in item.get("ResourceSpecificResults") or []
            if isinstance(row, dict)
        }
        if per_resource - {decision}:
            # One instance allowed and another denied is neither answer.
            decision = "mixed"
        decisions[str(item.get("EvalActionName"))] = decision
    return decisions


def eks_iam_errors(decisions: dict[str, str]) -> list[str]:
    errors = []
    for action in EKS_IAM_ACTIONS[1:]:
        if decisions.get(action) not in {"implicitDeny", "explicitDeny"}:
            errors.append(f"executor IAM permits {action}")
    if decisions.get("ec2:RebootInstances") != "allowed":
        errors.append("executor IAM does not allow ec2:RebootInstances")
    return errors


def eks_cloudtrail_invariant_errors(
    trail: dict[str, Any],
    *,
    instance_cluster_tags: dict[str, str],
    eks_name: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """The executor role made no EC2 replacement call, and its reboots -- the
    positive control -- hit the registered cluster's instances only.

    ``instance_cluster_tags`` maps every instance EC2 still describes to its
    ``eks:cluster-name`` tag (``""`` when untagged). An instance EC2 no longer
    knows (a replaced Spot node past its grace) cannot be bound and is
    recorded, not judged: it is not a replacement by the product, which
    RunInstances/TerminateInstances by the role would be.
    """

    events = trail["events"]
    errors = []
    for name in EKS_FORBIDDEN_EVENT_NAMES:
        if events.get(name):
            errors.append(
                f"CloudTrail contains an accepted EC2 {name} by the executor role"
            )
    bound: set[str] = set()
    unbound: set[str] = set()
    outside: set[str] = set()
    for item in events.get(EKS_POSITIVE_CONTROL_EVENT, []):
        instance_ids = item.get("instance_ids") or []
        if not instance_ids:
            errors.append(
                "CloudTrail RebootInstances by the executor role names no instance"
            )
        for instance_id in instance_ids:
            tag = instance_cluster_tags.get(instance_id)
            if tag is None:
                unbound.add(instance_id)
            elif tag != eks_name:
                outside.add(instance_id)
            else:
                bound.add(instance_id)
    if outside:
        errors.append(
            f"CloudTrail contains {len(outside)} RebootInstances target(s) by the "
            "executor role outside the registered EKS cluster"
        )
    return errors, {
        "bound": sorted(bound),
        "unbound": sorted(unbound),
        "outside_cluster": sorted(outside),
    }


def eks_inventory_errors(inventory: dict[str, Any]) -> list[str]:
    """Every GPU node is an EC2 instance tagged to the registered cluster, and
    every managed node group they belong to is ACTIVE on that cluster."""

    errors = []
    name = inventory.get("cluster_name")
    rows = inventory.get("nodes") or []
    if not rows:
        errors.append("EKS GPU node inventory is empty")
    if any(not row.get("instance_id") for row in rows):
        errors.append("an EKS GPU node has no EC2 providerID")
    tags = inventory.get("instance_cluster_tags") or {}
    untagged = [
        row["instance_id"]
        for row in rows
        if row.get("instance_id") and tags.get(str(row["instance_id"])) != name
    ]
    if untagged:
        errors.append(
            f"{len(untagged)} GPU instance(s) are not tagged eks:cluster-name={name}"
        )
    for group, facts in sorted((inventory.get("nodegroups") or {}).items()):
        if facts.get("cluster_name") != name or facts.get("status") != "ACTIVE":
            errors.append(
                f"managed node group {group} is not ACTIVE on the registered EKS cluster"
            )
        if facts.get("health_issue_codes"):
            errors.append(
                f"managed node group {group} reports health issues: "
                f"{facts['health_issue_codes']}"
            )
    return errors
