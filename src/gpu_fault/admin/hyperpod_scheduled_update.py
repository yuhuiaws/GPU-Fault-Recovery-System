"""``gpu-fault-admin hyperpod-scheduled-update``: print the command that turns a
HyperPod instance group's automatic updates off; the operator runs it.

GPU nodes the site manages are not re-imaged by automatic provider schedules
(管理员日常运维 / administrator-operations §5.0.5). An instance group's
``ScheduledUpdateConfig`` re-images every instance of the group on its cron
schedule (in place: same instance id, root volume replaced, the Node Agent
reinstalled by the node installer reconciler); ``AutoPatchConfig`` does the
same on its patch schedule. This read-only verb describes the cluster and, for
the selected groups that carry either, prints the exact ``aws sagemaker
update-cluster --instance-groups`` definition without them. The definition
comes from the ownership release's full-definition builder
(``ownership_release.hyperpod_instance_group_specification``: every
``ClusterInstanceGroupSpecification`` field copied from the current details,
``ImageId``/``ImageReleaseVersion`` the running image, ``InstanceCount`` the
``TargetCount``, ``MinInstanceCount`` the ``MinCount``, the ``CurrentLabels`` --
the ownership label stays -- and the current taints), and the release's
refusals apply (``ownership_release.hyperpod_write_refusal``): no command while
the cluster or the group is not ``InService``, an operation or a software
update is in flight, a ``Desired*`` change is pending, the running image
release is ``EndOfLife`` or the group carries a field the builder does not
know. Nothing is written. Run the command outside any ``config``/``deploy``
and re-run this verb: the group then reports ``none``.

Removing the schedule: ``CreateCluster`` and ``UpdateCluster`` are the only
operations whose input carries ``ScheduledUpdateConfig`` or ``AutoPatchConfig``,
and the service model makes ``ScheduleExpression`` required with a minimum
length of 1 (and ``PatchingStrategy`` required), so neither an empty structure
nor an empty expression is a valid request: a definition without the member is
the only request that can say "no schedule" -- and the service read it as
"remove" on the live CPU group (2026-11-02). The output carries the
``describe-cluster`` read that shows the outcome.

The site's CPU control-plane cluster is covered too: with ``--state-dir`` its
ARN (``spec.cpu``) selects every instance group -- the control-plane Pods carry
no node selector and its groups no ownership label. ``--rolling`` keeps each
selected group's ``ScheduleExpression`` and makes the update roll instead
(``hyperpod_rollback.deployment_config``: ``MaximumBatchSize`` and
``RollbackMaximumBatchSize`` ``INSTANCE_COUNT`` 1, ``WaitIntervalInSeconds``
from ``--wait-interval-seconds``, default 600 s, and an
``AutoRollbackConfiguration`` of the ``--rollback-alarm`` alarms, which it
requires): without a rolling policy the CPU group's weekly schedule was
observed (2026-11-01) to re-image all three control-plane nodes at once. The
service refuses a rolling schedule without the rollback batch or the alarms,
and refuses to modify an existing ``ScheduledUpdateConfig`` in place
(``hyperpod_rollback.SERVICE_REFUSALS``), so a group that has one gets four
ordered steps: delete it (the definition without the member), verify it is
gone, create the rolling one, verify it. The named alarms are read
(``cloudwatch describe-alarms``): a missing one refuses the plan; one that
cannot trip -- no data, or a load balancer other than the current API NLB --
is a warning, with the recommended pair's ``put-metric-alarm`` commands. Same
builder, same refusals; the operator runs the printed commands.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin.automatic_node_updates import HELPER_VERB, hyperpod_group_updates
from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_discovery import (
    find_hyperpod_for_eks,
    orchestrator_eks_arn,
)
from gpu_fault.admin.bootstrap_ownership import (
    hyperpod_groups_for_selector,
    recorded_node_scope,
)
from gpu_fault.admin.bootstrap_site import load_existing_site
from gpu_fault.admin.capacity_defaults import INGRESS_REPLICAS
from gpu_fault.admin.ownership_release import (
    hyperpod_current_labels,
    hyperpod_instance_group_specification,
    hyperpod_write_refusal,
)
from gpu_fault.cluster_scope import OWNERSHIP_LABEL

__all__ = ["VERB", "add_command", "plan_document", "run_command"]

VERB = HELPER_VERB
#: The members the printed definition leaves out.
REMOVED_FIELDS = ("ScheduledUpdateConfig", "AutoPatchConfig")
DISABLE = "disable"
#: ``--rolling``: the schedule stays and rolls one instance at a time.
ROLLING = "rolling"
NONE = "none"
REFUSED = "refused"
#: The batch ``--rolling`` writes, for the update and for its rollback.
ROLLING_BATCH = rollback.ROLLING_BATCH
DEFAULT_WAIT_SECONDS = 600
#: ``DeploymentConfiguration.WaitIntervalInSeconds`` bounds (service model and
#: ``aws sagemaker update-cluster help``: min 0, max 3600).
MAX_WAIT_SECONDS = 3600
#: Exit status when a selected group's automatic update cannot be removed yet.
EXIT_REFUSED = 2
SECTION = "管理员日常运维 / administrator-operations §5.0.5"
CPU_SELECTION = "every instance group of the site's CPU control-plane cluster"
#: What the verify reads project per group (``describe-cluster --query``).
_DISABLED_FIELDS = ("Status", "ScheduledUpdateConfig", "AutoPatchConfig")
_DELETED_FIELDS = ("Status", "ScheduledUpdateConfig")
_DEPLOYMENT = "ScheduledUpdateConfig.DeploymentConfig."
_ROLLING_FIELDS = {
    "Status": "Status",
    "ScheduleExpression": "ScheduledUpdateConfig.ScheduleExpression",
    "MaximumBatchSize": _DEPLOYMENT + "RollingUpdatePolicy.MaximumBatchSize",
    "RollbackMaximumBatchSize": _DEPLOYMENT
    + "RollingUpdatePolicy.RollbackMaximumBatchSize",
    "WaitIntervalInSeconds": _DEPLOYMENT + "WaitIntervalInSeconds",
    "RollbackAlarms": _DEPLOYMENT + "AutoRollbackConfiguration[].AlarmName",
}


def add_command(commands: Any) -> None:
    parser = commands.add_parser(
        VERB,
        usage=(
            f"gpu-fault-admin {VERB} --gpu-cluster-arn GPU_ARN "
            "[--instance-group NAME ...] [--state-dir STATE_DIR] "
            "[--rolling --rollback-alarm NAME [--rollback-alarm NAME ...] "
            "[--wait-interval-seconds SECONDS]]"
        ),
        help=(
            "print the aws sagemaker update-cluster command that removes the "
            "ScheduledUpdateConfig/AutoPatchConfig of the HyperPod instance groups "
            "the site manages, or with --rolling the commands that make the "
            "schedule roll one instance at a time with rollback alarms "
            "(read-only; the operator runs them)"
        ),
        description=(
            "Read-only: describes the HyperPod cluster and prints, as JSON, each "
            "selected instance group's automatic updates and the update-cluster "
            "command whose full definitions (built like the ownership release's) "
            "leave ScheduledUpdateConfig and AutoPatchConfig out -- or, with "
            "--rolling, keep the ScheduleExpression and roll one instance at a "
            "time, rolled back one at a time on the --rollback-alarm alarms "
            "(read with cloudwatch describe-alarms); an existing schedule is "
            "deleted first and created again, in four ordered steps, because the "
            "service does not modify one in place. Refused while an update is in "
            "flight or pending, or when a named alarm does not exist. Selected: "
            "--instance-group, else the groups the managed site selects "
            "(--state-dir; every group of its CPU control-plane cluster), else "
            f"every group carrying {OWNERSHIP_LABEL}."
        ),
    )
    parser.add_argument(
        "--gpu-cluster-arn",
        required=True,
        metavar="GPU_ARN",
        help="the HyperPod cluster ARN, or the ARN of the EKS cluster it orchestrates",
    )
    parser.add_argument(
        "--instance-group",
        action="append",
        default=[],
        metavar="NAME",
        help="only this instance group (repeatable)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        metavar="STATE_DIR",
        help=(
            "a managed site's private state directory: select the groups its "
            "recorded node selector covers -- every group of its CPU control-plane "
            "cluster (also what a bound deploy host requires)"
        ),
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help=(
            "keep each selected group's ScheduledUpdateConfig and make it roll one "
            "instance at a time (DeploymentConfig RollingUpdatePolicy "
            "MaximumBatchSize and RollbackMaximumBatchSize INSTANCE_COUNT 1, "
            "AutoRollbackConfiguration the --rollback-alarm alarms) instead of "
            "removing it"
        ),
    )
    parser.add_argument(
        "--rollback-alarm",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "with --rolling, required: an existing CloudWatch alarm that rolls the "
            "update back when it goes to ALARM (repeatable, 1-"
            f"{rollback.MAX_ROLLBACK_ALARMS}); for the CPU control plane, the API "
            "NLB HealthyHostCount pair (" + SECTION + ")"
        ),
    )
    parser.add_argument(
        "--wait-interval-seconds",
        type=int,
        default=DEFAULT_WAIT_SECONDS,
        metavar="SECONDS",
        help=(
            "with --rolling: the WaitIntervalInSeconds between batches (0-"
            f"{MAX_WAIT_SECONDS}, default {DEFAULT_WAIT_SECONDS})"
        ),
    )


# --- reads -----------------------------------------------------------------------


def _site_clusters(site: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    spec = (site or {}).get("spec")
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    return [item for item in clusters or () if isinstance(item, Mapping)]


def _site_cpu(site: Mapping[str, Any] | None) -> tuple[str, str]:
    """``(eksArn, hyperpodClusterName)`` of the site's CPU control-plane cluster."""

    spec = (site or {}).get("spec")
    cpu = spec.get("cpu") if isinstance(spec, Mapping) else None
    if not isinstance(cpu, Mapping):
        return "", ""
    return str(cpu.get("eksArn") or ""), str(cpu.get("hyperpodClusterName") or "")


def _site_nlb_name(site: Mapping[str, Any] | None) -> str:
    """``spec.nlb.name``: the API NLB the Service's load-balancer-name pins."""

    spec = (site or {}).get("spec")
    nlb = spec.get("nlb") if isinstance(spec, Mapping) else None
    return str(nlb.get("name") or "") if isinstance(nlb, Mapping) else ""


def _describe(
    runner: CommandRunner, *, arn: str, site: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The HyperPod ``describe-cluster`` document behind ``arn``."""

    parsed = Arn.parse(arn)
    if parsed.service == "sagemaker":
        described = runner.aws_json(
            parsed.region, "sagemaker", "describe-cluster", "--cluster-name", arn
        )
    elif parsed.service == "eks":
        cpu_eks_arn, cpu_hyperpod = _site_cpu(site)
        hint = next(
            (
                str(item.get("hyperpodClusterName") or "")
                for item in _site_clusters(site)
                if item.get("eksClusterArn") == arn
            ),
            cpu_hyperpod if cpu_eks_arn == arn else "",
        )
        found = find_hyperpod_for_eks(
            runner, eks_arn=arn, region=parsed.region, hint=hint or None
        )
        if found is None:
            raise BootstrapError(
                f"{arn} has no HyperPod orchestrator: a plain EKS (eks-managed) "
                "cluster has no ScheduledUpdateConfig -- its managed node groups "
                "change version only on UpdateNodegroupVersion; what else "
                f"replaces their nodes is in {SECTION}"
            )
        described = dict(found)
    else:
        raise BootstrapError("--gpu-cluster-arn must be a sagemaker or eks ARN")
    if not orchestrator_eks_arn(described):
        raise BootstrapError(
            f"HyperPod cluster {arn} does not expose an EKS orchestrator"
        )
    return described


def _selected(
    described: Mapping[str, Any],
    *,
    requested: Sequence[str],
    site: Mapping[str, Any] | None,
) -> tuple[list[str], str]:
    """``(group names, how they were selected)``."""

    name = str(described.get("ClusterName") or "")
    groups = [
        group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    ]
    names = [str(group.get("InstanceGroupName") or "") for group in groups]
    if requested:
        unknown = sorted(set(requested) - set(names))
        if unknown:
            raise BootstrapError(
                f"HyperPod cluster {name} has no instance group "
                f"{', '.join(unknown)}; its groups are {', '.join(names) or 'none'}"
            )
        return [item for item in names if item in set(requested)], "--instance-group"
    cpu_eks_arn, cpu_hyperpod = _site_cpu(site)
    if site is not None and (
        (cpu_hyperpod and name == cpu_hyperpod)
        or (cpu_eks_arn and orchestrator_eks_arn(described) == cpu_eks_arn)
    ):
        # Any CPU node hosts control-plane Pods, and no CPU group carries the
        # ownership label: every group is the control plane's.
        return names, CPU_SELECTION
    if site is not None:
        recorded = recorded_node_scope(
            _site_clusters(site),
            eks_arn=orchestrator_eks_arn(described),
            hyperpod_name=name,
        )
        if recorded is not None:
            return (
                hyperpod_groups_for_selector(
                    groups, cluster_name=name, selector=recorded.selector
                ),
                "the site's recorded node selector",
            )
        site_id = str((site.get("metadata") or {}).get("name") or "")
        return (
            [
                str(group.get("InstanceGroupName") or "")
                for group in groups
                if hyperpod_current_labels(group).get(OWNERSHIP_LABEL) == site_id
            ],
            f"the {OWNERSHIP_LABEL}={site_id} label",
        )
    return (
        [
            str(group.get("InstanceGroupName") or "")
            for group in groups
            if OWNERSHIP_LABEL in hyperpod_current_labels(group)
        ],
        f"the {OWNERSHIP_LABEL} label",
    )


def _alarm_facts(
    runner: CommandRunner, *, region: str, names: Sequence[str], nlb_name: str
) -> dict[str, Any]:
    """The named alarms (``hyperpod_rollback.read_rollback_alarms``) and, given
    the site's API NLB name, the NLB they should watch; a failed alarm read is
    ``state: unknown`` -- the plan then refuses."""

    def read_alarms(wanted: list[str]) -> Mapping[str, Any]:
        return runner.aws_json(region, *rollback.describe_alarms_arguments(wanted))

    def read_nlb() -> Mapping[str, Any]:
        balancers = runner.aws_json(
            region, "elbv2", "describe-load-balancers", "--names", nlb_name
        )
        groups = runner.aws_json(
            region,
            "elbv2",
            "describe-target-groups",
            "--load-balancer-arn",
            rollback.load_balancer_arn(balancers),
        )
        return rollback.api_nlb(nlb_name, balancers, groups)

    try:
        return rollback.read_rollback_alarms(
            names, read_alarms=read_alarms, read_nlb=read_nlb if nlb_name else None
        )
    except Exception as exc:  # noqa: BLE001 - recorded; the plan refuses
        return {"state": rollback.UNKNOWN, "error": str(exc).strip()[:300]}


# --- the plan ---------------------------------------------------------------------


def rolling_schedule(
    schedule: Mapping[str, Any], wait_seconds: int, *, alarms: Sequence[str]
) -> dict[str, Any]:
    """``schedule`` with its expression kept and a one-instance rolling policy,
    rolled back one instance at a time on ``alarms``
    (``hyperpod_rollback.deployment_config``); any other member stays."""

    return {
        **schedule,
        "DeploymentConfig": rollback.deployment_config(
            schedule.get("DeploymentConfig"), wait_seconds=wait_seconds, alarms=alarms
        ),
    }


def _group_plan(
    details: Mapping[str, Any],
    *,
    cluster_status: str,
    rolling: int | None = None,
    alarms: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """One group's record and, when it needs and allows one, its definition:
    without ``rolling`` the definition minus ``REMOVED_FIELDS``; with it (the
    wait in seconds) the schedule kept and made to roll (``rolling_schedule``),
    the current one to be deleted first (``replaces``)."""

    facts = hyperpod_group_updates(details)
    record: dict[str, Any] = {
        "name": facts["name"],
        "ownership_label": hyperpod_current_labels(details).get(OWNERSHIP_LABEL),
        **{field: details.get(field) for field in REMOVED_FIELDS},
    }
    schedule = details.get("ScheduledUpdateConfig")
    target: dict[str, Any] | None = None
    if rolling is not None:
        if isinstance(schedule, Mapping) and schedule.get("ScheduleExpression"):
            target = rolling_schedule(schedule, rolling, alarms=alarms)
        if target is None or rollback.same_schedule(target, schedule):
            record.update(
                outcome=NONE,
                reason="no-schedule" if target is None else "rolls-this-way",
            )
            return record, None
    removes = [field for field in REMOVED_FIELDS if details.get(field) is not None]
    if not removes:
        record["outcome"] = NONE
        return record, None
    reason = hyperpod_write_refusal(cluster_status, details, allow_label_release=False)
    specification = None
    if reason is None:
        try:
            specification = hyperpod_instance_group_specification(
                details, labels=hyperpod_current_labels(details)
            )
        except BootstrapError as exc:
            reason = str(exc)
    if specification is None:
        record.update(outcome=REFUSED, refusal=reason)
        return record, None
    if target is not None:
        specification["ScheduledUpdateConfig"] = target
        record.update(outcome=ROLLING, rolling_schedule=target, replaces=schedule)
        return record, specification
    for field in REMOVED_FIELDS:
        specification.pop(field, None)
    record.update(outcome=DISABLE, removes=removes)
    return record, specification


def _update_command(region: str, cluster: str, definitions: Sequence[Any]) -> str:
    text = json.dumps(list(definitions), separators=(",", ":"))
    return (
        f"aws sagemaker update-cluster --region {region} --cluster-name {cluster} "
        f"--instance-groups {shlex.quote(text)}"
    )


def _verify(
    region: str, cluster: str, groups: Sequence[str], fields: Mapping[str, str]
) -> str:
    """The ``describe-cluster`` read of ``groups``: the cluster status and,
    per group, ``fields`` (projection name: JMESPath)."""

    match = " || ".join(f'InstanceGroupName==`"{name}"`' for name in groups)
    projection = ",".join(
        ["InstanceGroupName:InstanceGroupName"]
        + [f"{key}:{path}" for key, path in fields.items()]
    )
    query = (
        "{ClusterStatus:ClusterStatus,InstanceGroups:InstanceGroups[?"
        + match
        + "].{"
        + projection
        + "}}"
    )
    return (
        f"aws sagemaker describe-cluster --region {region} --cluster-name {cluster} "
        f"--query {shlex.quote(query)}"
    )


def _expect(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a verify read answers once the step applied."""

    return {
        "ClusterStatus": "InService",
        "InstanceGroups": [
            {"InstanceGroupName": group["name"], "Status": "InService", **group["is"]}
            for group in groups
        ],
    }


def _rolling_expectation(target: Mapping[str, Any]) -> dict[str, Any]:
    deployment = target["DeploymentConfig"]
    policy = deployment["RollingUpdatePolicy"]
    return {
        "ScheduleExpression": target["ScheduleExpression"],
        "MaximumBatchSize": policy["MaximumBatchSize"],
        "RollbackMaximumBatchSize": policy["RollbackMaximumBatchSize"],
        "WaitIntervalInSeconds": deployment["WaitIntervalInSeconds"],
        "RollbackAlarms": rollback.rollback_alarm_names(target),
    }


def _manual_update(
    region: str, cluster: str, group: str, *, wait: int | None, alarms: Sequence[str]
) -> str:
    """``update-cluster-software`` of one group, one instance at a time and
    rolled back one at a time: the same ``DeploymentConfiguration``. Without
    ``--rolling`` the wait and the alarms are placeholders that keep the JSON
    invalid -- the CLI refuses it until they are filled in."""

    config = rollback.deployment_config(
        None, wait_seconds=wait or 0, alarms=list(alarms) or ["-"]
    )
    members = {
        key: json.dumps(value, separators=(",", ":")) for key, value in config.items()
    }
    if wait is None:
        members["WaitIntervalInSeconds"] = "<seconds>"
    if not alarms:
        members["AutoRollbackConfiguration"] = '[{"AlarmName":<rollback alarm>}]'
    text = "{" + ",".join(f'"{key}":{value}' for key, value in members.items()) + "}"
    return (
        f"aws sagemaker update-cluster-software --region {region} "
        f"--cluster-name {cluster} --instance-groups InstanceGroupName={group} "
        f"--deployment-config {shlex.quote(text)}"
    )


def _alarm_block(
    facts: Mapping[str, Any] | None,
    *,
    names: Sequence[str],
    region: str,
    wait: int,
    cpu: bool,
    writes: bool,
) -> dict[str, Any]:
    """``rollback_alarms``: what the named alarms are, why they may not trip,
    the recommended pair's commands, and a ``refusal`` when the plan writes
    and an alarm is missing or could not be read."""

    block: dict[str, Any] = {"names": list(names), **dict(facts or {})}
    warnings: list[str] = []
    missing: list[str] = []
    for record in block.get("alarms") or ():
        if not record.get("found"):
            missing.append(str(record["name"]))
        warnings.extend(
            f"rollback alarm {record['name']} {problem}"
            for problem in record.get("problems") or ()
        )
    nlb = block.get("api_nlb")
    if block.get("state") == rollback.UNKNOWN:
        warnings.append(
            f"could not read the rollback alarms ({block.get('error')}): whether "
            "they exist is unknown; the operator identity needs "
            "cloudwatch:DescribeAlarms"
        )
    elif cpu and not isinstance(nlb, Mapping):
        warnings.append(
            "the site names no API NLB (spec.nlb.name; pass --state-dir): the "
            "alarms' dimensions were not checked"
        )
    elif isinstance(nlb, Mapping) and nlb.get("state") == rollback.UNKNOWN:
        warnings.append(
            f"could not read the API NLB ({nlb.get('error')}): the alarms' "
            "dimensions were not checked"
        )
    elif isinstance(nlb, Mapping):
        block["suggested"] = rollback.suggested_alarms(
            region=region, nlb=nlb, replicas=INGRESS_REPLICAS, wait_seconds=wait
        )
    block["warnings"] = warnings
    if writes and block.get("state") == rollback.UNKNOWN:
        block["refusal"] = (
            "cannot read the rollback alarms, so cannot tell that they exist: "
            f"{block.get('error')}"
        )
    elif writes and missing:
        block["refusal"] = (
            f"rollback alarm {', '.join(missing)} does not exist: an "
            "AutoRollbackConfiguration on a missing alarm protects nothing; create "
            "it first (rollback_alarms.suggested for the CPU control plane's API "
            f"NLB pair, {SECTION}) and re-run"
        )
    return block


def plan_document(
    described: Mapping[str, Any],
    *,
    region: str,
    selected: Sequence[str],
    selection: str,
    rolling: int | None = None,
    alarms: Sequence[str] = (),
    alarm_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON the verb prints for ``described`` and the ``selected`` groups;
    ``rolling`` (the wait in seconds) is ``--rolling``, ``alarms`` its
    ``--rollback-alarm`` names and ``alarm_facts`` what ``_alarm_facts`` read."""

    name = str(described.get("ClusterName") or "")
    status = str(described.get("ClusterStatus") or "")
    by_name = {
        str(group.get("InstanceGroupName") or ""): group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    }
    records, specifications = [], []
    for group in selected:
        record, specification = _group_plan(
            by_name[group], cluster_status=status, rolling=rolling, alarms=alarms
        )
        records.append(record)
        if specification is not None:
            specifications.append(specification)
    alarm_block = (
        None
        if rolling is None
        else _alarm_block(
            alarm_facts,
            names=alarms,
            region=region,
            wait=rolling,
            cpu=selection == CPU_SELECTION,
            writes=bool(specifications),
        )
    )
    refused = [item for item in records if item["outcome"] == REFUSED]
    blocked = refused or (alarm_block or {}).get("refusal")
    document: dict[str, Any] = {
        "cluster": {
            "hyperpod_name": name,
            "hyperpod_arn": described.get("ClusterArn"),
            "eks_arn": orchestrator_eks_arn(described),
            "region": region,
            "status": status,
        },
        "selection": selection,
        "mode": DISABLE if rolling is None else ROLLING,
        "instance_groups": records,
        "status": ("REFUSED" if blocked else "PLAN" if specifications else "NOOP"),
        "removal": (
            "UpdateCluster is the only operation that can drop the schedule and "
            "the model rejects an empty ScheduledUpdateConfig/ScheduleExpression, "
            "so the definition leaves the member out (the service read that as "
            "remove on the live CPU group, 2026-11-02) -- run verify afterwards"
        )
        if rolling is None
        else (
            "the service does not modify an existing ScheduledUpdateConfig in place "
            f"(ValidationException, {rollback.SERVICE_REFUSALS[2]['at']}), so step 1 "
            "deletes it (the definition without ScheduledUpdateConfig) and step 3 "
            "creates the rolling one: the ScheduleExpression kept, "
            "RollingUpdatePolicy MaximumBatchSize and RollbackMaximumBatchSize "
            f"INSTANCE_COUNT 1, WaitIntervalInSeconds {rolling} and an "
            f"AutoRollbackConfiguration of {', '.join(alarms)}: the service then "
            "re-images one instance at a time on that schedule and rolls back one "
            "at a time when an alarm goes to ALARM -- run the steps in order"
        ),
        "manual_update": {
            item["name"]: _manual_update(
                region, name, item["name"], wait=rolling, alarms=alarms
            )
            for item in records
        },
        "section": SECTION,
    }
    if alarm_block is not None:
        document["rollback_alarms"] = alarm_block
    if specifications and not blocked:
        document["update_cluster"] = _update_cluster(
            region, name, records, specifications, rolling=rolling is not None
        )
    return document


def _update_cluster(
    region: str,
    cluster: str,
    records: Sequence[Mapping[str, Any]],
    specifications: Sequence[dict[str, Any]],
    *,
    rolling: bool,
) -> dict[str, Any]:
    """``update_cluster``: the final definitions, their command, the verify
    read with what it answers, and the ordered ``steps`` -- two for a removal,
    four (delete, verify, create, verify) for a rolling schedule."""

    command = _update_command(region, cluster, specifications)
    names = [str(spec["InstanceGroupName"]) for spec in specifications]
    run_when = (
        "in a maintenance window, never while gpu-fault-admin config or deploy "
        "runs; the cluster is Updating until the definition applies"
    )
    if not rolling:
        verify = _verify(region, cluster, names, {key: key for key in _DISABLED_FIELDS})
        expect = _expect(
            [{"name": n, "is": dict.fromkeys(_DISABLED_FIELDS[1:])} for n in names]
        )
        return {
            "instance_groups": list(specifications),
            "command": command,
            "verify": verify,
            "expect": expect,
            "steps": [
                {"step": 1, "action": "remove", "command": command},
                {"step": 2, "action": "verify", "command": verify, "expect": expect},
            ],
            "run_when": run_when,
        }
    deletes = [
        {key: value for key, value in spec.items() if key != "ScheduledUpdateConfig"}
        for spec in specifications
    ]
    targets = {
        str(item["name"]): item["rolling_schedule"]
        for item in records
        if item["outcome"] == ROLLING
    }
    deleted = _verify(region, cluster, names, {key: key for key in _DELETED_FIELDS})
    verify = _verify(region, cluster, names, _ROLLING_FIELDS)
    expect = _expect(
        [{"name": n, "is": _rolling_expectation(targets[n])} for n in names]
    )
    roles = sorted({str(spec.get("ExecutionRole") or "") for spec in specifications})
    return {
        "instance_groups": list(specifications),
        "command": command,
        "verify": verify,
        "expect": expect,
        "steps": [
            {
                "step": 1,
                "action": "delete",
                "instance_groups": deletes,
                "command": _update_command(region, cluster, deletes),
            },
            {
                "step": 2,
                "action": "verify-deleted",
                "command": deleted,
                "expect": _expect(
                    [{"name": n, "is": {"ScheduledUpdateConfig": None}} for n in names]
                ),
            },
            {"step": 3, "action": "create", "command": command},
            {"step": 4, "action": "verify", "command": verify, "expect": expect},
        ],
        "between_steps": (
            "between step 1 and step 3 the group has no ScheduledUpdateConfig -- no "
            "automatic update at all, which is safe; if step 3 fails the group "
            "stays without a schedule, still safe, but run step 3 again (a re-run "
            "of this verb then reports none: there is no ScheduleExpression left "
            "to keep). Re-run step 2 until it answers its expect (about 30 s live) "
            "before step 3"
        ),
        "execution_role": {
            "requires": "cloudwatch:DescribeAlarms",
            "why": rollback.AMI_UPDATE_EVIDENCE["quotes"][1],
            "check": [
                "aws iam simulate-principal-policy --policy-source-arn "
                f"{shlex.quote(role)} --action-names cloudwatch:DescribeAlarms "
                "--query 'EvaluationResults[].EvalDecision' --output text"
                for role in roles
                if role
            ],
            "expect": "allowed",
        },
        "run_when": run_when,
    }


def _detail(item: Mapping[str, Any], *, rolling: int | None) -> str:
    """What the per-group stderr line says after the outcome."""

    outcome = item["outcome"]
    if outcome == DISABLE:
        return f"removes {', '.join(item['removes'])}"
    if outcome == REFUSED:
        return str(item.get("refusal"))
    if rolling is None:
        return "no automatic update"
    if outcome == ROLLING:
        target = item["rolling_schedule"]
        return (
            f"keeps ScheduledUpdateConfig {target.get('ScheduleExpression')}, "
            f"re-imaging one instance at a time {rolling} s apart, rolled back one "
            "instance at a time on alarm "
            + " or ".join(rollback.rollback_alarm_names(target))
            + "; the current ScheduledUpdateConfig is deleted first: the service "
            "does not modify one in place"
        )
    if item.get("reason") == "rolls-this-way":
        schedule = item.get("ScheduledUpdateConfig")
        return (
            "rolls this way already: one instance at a time "
            f"{rolling} s apart, rolled back one instance at a time on alarm "
            + " or ".join(rollback.rollback_alarm_names(schedule))
        )
    return "no ScheduledUpdateConfig to roll"


def _say(text: str) -> None:
    print(f"gpu-fault-admin {VERB}: {text}", file=sys.stderr, flush=True)


def _announce(document: Mapping[str, Any], *, rolling: int | None) -> None:
    cluster = document["cluster"]["hyperpod_name"]
    for item in document["instance_groups"]:
        _say(
            f"{cluster}/{item['name']}: {item['outcome']} "
            f"({_detail(item, rolling=rolling)})"
        )
    alarms = document.get("rollback_alarms") or {}
    for record in alarms.get("alarms") or ():
        _say(f"rollback alarm {record['name']}: {rollback.alarm_summary(record)}")
    for warning in alarms.get("warnings") or ():
        _say(f"warning: {warning}")
    suggested = alarms.get("suggested") or {}
    if alarms.get("warnings") and suggested.get("commands"):
        for item in suggested["commands"]:
            _say(
                f"suggested rollback alarm {item['name']} on the current API NLB "
                f"(never run by this verb): {item['command']}"
            )
    if alarms.get("refusal"):
        _say(f"refused: {alarms['refusal']}")
    update = document.get("update_cluster")
    if not update:
        return
    if rolling is None:
        _say(f"run in a maintenance window: {update['command']}")
        _say(f"then verify: {update['verify']}")
        return
    _say(
        "run in a maintenance window, in this order -- " + update["between_steps"] + ":"
    )
    for check in update["execution_role"]["check"]:
        _say(
            "before step 1, the execution role must allow cloudwatch:DescribeAlarms "
            f"(expect allowed): {check}"
        )
    for step in update["steps"]:
        expect = step.get("expect")
        _say(
            f"step {step['step']} {step['action']}"
            + (
                f" (expect {json.dumps(expect, separators=(',', ':'))})"
                if expect
                else ""
            )
            + f": {step['command']}"
        )


def run_command(
    arguments: argparse.Namespace, *, runner: CommandRunner | None = None
) -> int:
    """Print the plan; exit 0, or ``EXIT_REFUSED`` when a group is refused."""

    active = runner or CommandRunner()
    arn = str(arguments.gpu_cluster_arn).strip()
    wait = getattr(arguments, "wait_interval_seconds", DEFAULT_WAIT_SECONDS)
    rolling = wait if getattr(arguments, "rolling", False) else None
    alarms = [str(item) for item in getattr(arguments, "rollback_alarm", None) or ()]
    if not 0 <= wait <= MAX_WAIT_SECONDS:
        raise BootstrapError(
            f"--wait-interval-seconds must be 0-{MAX_WAIT_SECONDS} (the service's "
            "WaitIntervalInSeconds range)"
        )
    if rolling is None and alarms:
        raise BootstrapError(
            "--rollback-alarm needs --rolling: a removed schedule has no rollback"
        )
    refusal = rollback.alarm_name_refusal(alarms) if rolling is not None else None
    if refusal:
        raise BootstrapError(refusal)
    state_dir = getattr(arguments, "state_dir", None)
    site = (
        load_existing_site(Path(state_dir).expanduser().resolve())
        if state_dir is not None
        else None
    )
    described = _describe(active, arn=arn, site=site)
    selected, selection = _selected(
        described, requested=tuple(arguments.instance_group or ()), site=site
    )
    region = Arn.parse(arn).region
    document = plan_document(
        described,
        region=region,
        selected=selected,
        selection=selection,
        rolling=rolling,
        alarms=alarms,
        alarm_facts=_alarm_facts(
            active,
            region=region,
            names=alarms,
            nlb_name=_site_nlb_name(site) if selection == CPU_SELECTION else "",
        )
        if rolling is not None
        else None,
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    _announce(document, rolling=rolling)
    if not selected:
        _say(
            f"{document['cluster']['hyperpod_name']}: no instance group selected by "
            f"{selection}; name one with --instance-group"
        )
    return EXIT_REFUSED if document["status"] == "REFUSED" else 0
