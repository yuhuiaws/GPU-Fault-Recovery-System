"""GF-REGIONAL-COLLECT-015's provider-side readings, for HyperPod and plain EKS.

The case audits the one provider mutation a whole-node reboot leaves behind
and the provider's own recovery authority that must not compete with it. On
HyperPod that is ``BatchRebootClusterNodes`` in CloudTrail's
``sagemaker.amazonaws.com`` source and the cluster's ``NodeRecovery=None``.
On an ``eks-managed`` cluster it is EC2 ``RebootInstances`` in
``ec2.amazonaws.com`` -- read beside Karpenter's refused ``RunInstances``
dry-runs, which are not mutations -- and, since plain EKS has no HyperPod
NodeRecovery to compete with (``not-applicable``), the instance's own
replacement authority: its managed node group must be ``ACTIVE`` without
health issues and its Auto Scaling instance ``InService``/``HEALTHY``, so no
health replacement is in flight for the instance the product rebooted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional.collector_reboot_evidence import (
    EKS_NODEGROUP_LABEL,
    partition_provider_events,
    provider_event_source,
    reboot_event_names,
)
from scripts.e2e.regional.guardrail_audit_evidence import EC2_REPLACEMENT_MUTATIONS
from scripts.e2e.regional.regional_live_fixture import (
    provider_event_actor_matches_role,
)
from scripts.e2e.regional.warm_spare_provider import (
    EKS_MANAGED,
    HYPERPOD,
    PROVIDER_REPLACE_EVENTS,
    normalized,
)

#: The provider's own recovery authority a reboot must not compete with: HyperPod
#: NodeRecovery, or nothing of the kind on plain EKS.
NODE_RECOVERY_EXPECTED = {HYPERPOD: "None", EKS_MANAGED: "not-applicable"}


def replace_event_names(provider: str | None) -> frozenset[str]:
    """The provider mutations that would mean a replacement instead of a reboot."""

    if normalized(provider) == EKS_MANAGED:
        return EC2_REPLACEMENT_MUTATIONS
    return PROVIDER_REPLACE_EVENTS


def node_recovery_guard_errors(
    node_recovery: Any, *, provider: str | None
) -> list[str]:
    """``NodeRecovery`` must be None on HyperPod; plain EKS has none to read."""

    name = normalized(provider)
    expected = NODE_RECOVERY_EXPECTED[name]
    if name == HYPERPOD:
        if node_recovery in (None, "None"):
            return []
        return [
            f"HyperPod cluster NodeRecovery is {node_recovery!r}; the reboot must "
            "not compete with automatic recovery"
        ]
    if node_recovery == expected:
        return []
    return [
        f"EKS cluster recovery reads {node_recovery!r}; plain EKS has no HyperPod "
        f"NodeRecovery to compete with (expected {expected!r})"
    ]


def reboot_count_errors(
    events: list[dict[str, Any]],
    *,
    provider: str | None,
    executor_role_arn: str,
    waited: list[dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Exactly one accepted reboot call, by the executor role, no replacement.

    The window is split by ``partition_provider_events``: calls AWS refused
    (an ``errorCode``, nothing ran) are kept as ``refused_provider_calls`` and
    never counted -- on plain EKS Karpenter dry-runs ``RunInstances`` about
    once a minute (``Client.DryRunOperation``); a refused reboot attempt is
    reported in the reboot count's own words rather than mistaken for the one.
    ``waited`` is the visibility wait's own (name-filtered) inventory when the
    caller polled for the reboot; the reboot is counted from it, the whole
    window from ``events`` carries the replacement claim and the refused calls.
    """

    mutations, refused = partition_provider_events(events)
    accepted = [dict(item) for item in mutations if isinstance(item, dict)]
    names = reboot_event_names(normalized(provider))
    counted = accepted if waited is None else partition_provider_events(waited)[0]
    reboot = [
        dict(item)
        for item in counted
        if isinstance(item, dict) and item.get("event_name") in names
    ]
    refused_reboots = [item for item in refused if item.get("event_name") in names]
    replace = [
        item
        for item in accepted
        if item.get("event_name") in replace_event_names(provider)
    ]
    errors: list[str] = []
    if len(reboot) != 1:
        errors.append(
            f"CloudTrail reboot count is not one ({len(reboot)} accepted "
            f"{sorted(names)} calls; {len(refused_reboots)} refused reboot calls, "
            f"{len(refused)} refused provider calls in the window)"
        )
    elif not provider_event_actor_matches_role(reboot[0], executor_role_arn):
        errors.append("reboot actor is not the executor role")
    if replace:
        errors.append(
            "CloudTrail contains provider replacement: "
            f"{sorted({str(item.get('event_name')) for item in replace})}"
        )
    return errors, {
        "reboot_events": reboot,
        "replace_events": replace,
        "refused_provider_calls": refused,
    }


def provider_calls(
    regional: Any,
    started_at: datetime,
    *,
    provider: str | None,
    executor_role_arn: str,
) -> dict[str, Any]:
    """Wait for the reboot's provider mutation, then judge the whole window.

    The positive claim (one reboot) polls CloudTrail in the provider's own
    event source until it is visible or the delivery window closes; the
    negative claim (no replacement) cannot be proven inside that window and is
    recorded as provisional, for DESTR-013's later full-window audit.
    """

    source = provider_event_source(normalized(provider))
    waited = regional.wait_provider_events(
        started_at,
        event_names=set(reboot_event_names(normalized(provider))),
        expected_count=1,
        event_source=source,
    )
    ended_at = datetime.now(timezone.utc)
    events = regional.provider_events(started_at, ended_at, event_source=source)
    errors, evidence = reboot_count_errors(
        events, provider=provider, executor_role_arn=executor_role_arn, waited=waited
    )
    return {
        "errors": errors,
        "evidence": {
            "provider": normalized(provider),
            "provider_event_source": source,
            "provider_events": events,
            **evidence,
            "provider_events_provisional": regional.provider_events_provisional(
                ended_at
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Plain EKS: the instance's replacement authority
# --------------------------------------------------------------------------- #
def eks_replacement_guard(
    *,
    node: str,
    instance_id: str,
    nodegroup_name: str,
    nodegroup: dict[str, Any],
    autoscaling: dict[str, Any],
) -> dict[str, Any]:
    """``describe-nodegroup`` and ``describe-auto-scaling-instances`` reduced
    to the facts the verdict reads: group status and health, and the one
    instance's lifecycle and health as its Auto Scaling group sees them."""

    group = nodegroup.get("nodegroup") or {}
    issues = (group.get("health") or {}).get("issues") or []
    repair = (group.get("nodeRepairConfig") or {}).get("enabled")
    instances = [
        item
        for item in autoscaling.get("AutoScalingInstances") or []
        if isinstance(item, dict) and item.get("InstanceId") == instance_id
    ]
    instance = instances[0] if len(instances) == 1 else None
    return {
        "provider": EKS_MANAGED,
        "node": node,
        "instance_id": instance_id or None,
        "nodegroup": {
            "name": nodegroup_name or None,
            "status": group.get("status"),
            "health_issue_codes": sorted(
                {str(item.get("code")) for item in issues if isinstance(item, dict)}
            ),
            # EKS node auto repair, plain EKS's NodeRecovery: opt-in, so a
            # missing field is recorded as None (off), never read as True.
            "node_repair_config": {
                "enabled": repair if isinstance(repair, bool) else None
            },
        },
        "autoscaling_instance": (
            None
            if instance is None
            else {
                "group": instance.get("AutoScalingGroupName"),
                "lifecycle_state": instance.get("LifecycleState"),
                "health_status": instance.get("HealthStatus"),
            }
        ),
    }


def replacement_guard_errors(guard: dict[str, Any]) -> list[str]:
    """No provider-side replacement may be in flight for the rebooted instance.

    Fail-closed: an instance whose node group or Auto Scaling group cannot be
    read has no readable replacement authority, and the case says so instead
    of passing on silence. HyperPod's authority is ``NodeRecovery``, judged by
    ``node_recovery_guard_errors``; this guard carries nothing else for it.
    """

    if guard.get("provider") != EKS_MANAGED:
        return []
    errors: list[str] = []
    if not guard.get("instance_id"):
        errors.append(
            "the node has no EC2 providerID; its replacement authority is unreadable"
        )
    group = guard.get("nodegroup") or {}
    if not group.get("name"):
        errors.append(f"the node carries no {EKS_NODEGROUP_LABEL} label")
    elif group.get("status") != "ACTIVE":
        errors.append(
            f"managed node group {group.get('name')} is {group.get('status')!r}, "
            "not ACTIVE"
        )
    if group.get("health_issue_codes"):
        errors.append(
            f"managed node group {group.get('name')} reports health issues: "
            f"{group.get('health_issue_codes')}"
        )
    if (group.get("node_repair_config") or {}).get("enabled") is True:
        errors.append(
            f"EKS node auto repair is enabled on managed node group {group.get('name')}; "
            "it competes with the product's recovery (the eks-managed counterpart "
            "of HyperPod NodeRecovery=Automatic)"
        )
    instance = guard.get("autoscaling_instance")
    if instance is None:
        errors.append(
            "the instance is in no Auto Scaling group; no replacement authority "
            "is readable for it"
        )
    elif (
        instance.get("lifecycle_state") != "InService"
        or instance.get("health_status") != "HEALTHY"
    ):
        errors.append(
            "an Auto Scaling health replacement is in flight for "
            f"{guard.get('instance_id')}: {instance.get('lifecycle_state')}/"
            f"{instance.get('health_status')}"
        )
    return errors


def _aws(regional: Any, *arguments: str) -> dict[str, Any]:
    value = json.loads(
        regional.run(
            [
                "aws",
                *arguments,
                "--region",
                regional.settings.region,
                "--output",
                "json",
            ],
            timeout=180,
        ).stdout
    )
    return value if isinstance(value, dict) else {}


def read_eks_replacement_guard(
    regional: Any, *, node: str, cluster_name: str
) -> dict[str, Any]:
    """The live guard: the Node's group label and providerID, then the two
    read-only AWS descriptions (``eks:DescribeNodegroup``,
    ``autoscaling:DescribeAutoScalingInstances``)."""

    document = json.loads(regional.kubectl("gpu", "get", "node", node, "-o", "json"))
    labels = (document.get("metadata") or {}).get("labels") or {}
    nodegroup_name = str(labels.get(EKS_NODEGROUP_LABEL) or "")
    provider_id = str((document.get("spec") or {}).get("providerID") or "")
    instance_id = provider_id.rsplit("/", 1)[-1] if provider_id else ""
    nodegroup = (
        _aws(
            regional,
            "eks",
            "describe-nodegroup",
            "--cluster-name",
            cluster_name,
            "--nodegroup-name",
            nodegroup_name,
        )
        if nodegroup_name
        else {}
    )
    autoscaling = (
        _aws(
            regional,
            "autoscaling",
            "describe-auto-scaling-instances",
            "--instance-ids",
            instance_id,
        )
        if instance_id
        else {}
    )
    return eks_replacement_guard(
        node=node,
        instance_id=instance_id,
        nodegroup_name=nodegroup_name,
        nodegroup=nodegroup,
        autoscaling=autoscaling,
    )


def replacement_guard(
    regional: Any,
    *,
    node: str,
    cluster_name: str,
    provider: str | None,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """The provider's recovery authority and its errors, per provider."""

    node_recovery = recovery.get("node_recovery")
    errors = node_recovery_guard_errors(node_recovery, provider=provider)
    if normalized(provider) != EKS_MANAGED:
        guard: dict[str, Any] = {"provider": HYPERPOD, "node_recovery": node_recovery}
        return {"guard": guard, "errors": errors}
    guard = read_eks_replacement_guard(regional, node=node, cluster_name=cluster_name)
    guard["node_recovery"] = node_recovery
    errors.extend(replacement_guard_errors(guard))
    return {"guard": guard, "errors": errors}
