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
the only request that can say "no schedule". The model does not state whether
the service reads an omitted member of an instance group definition as
"remove" or as "unchanged" (elsewhere the definition is declarative: the
``KubernetesConfig`` labels are the declared state and an omitted
``ImageReleaseVersion`` means the latest release, not the running one), so the
output also carries the ``describe-cluster`` read that shows the outcome.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
NONE = "none"
REFUSED = "refused"
#: Exit status when a selected group's automatic update cannot be removed yet.
EXIT_REFUSED = 2
SECTION = "管理员日常运维 / administrator-operations §5.0.5"


def add_command(commands: Any) -> None:
    parser = commands.add_parser(
        VERB,
        usage=(
            f"gpu-fault-admin {VERB} --gpu-cluster-arn GPU_ARN "
            "[--instance-group NAME ...] [--state-dir STATE_DIR]"
        ),
        help=(
            "print the aws sagemaker update-cluster command that removes the "
            "ScheduledUpdateConfig/AutoPatchConfig of the HyperPod instance groups "
            "the site manages (read-only; the operator runs it)"
        ),
        description=(
            "Read-only: describes the HyperPod cluster and prints, as JSON, each "
            "selected instance group's automatic updates and one update-cluster "
            "command whose full definitions (built like the ownership release's) "
            "leave ScheduledUpdateConfig and AutoPatchConfig out; refused while an "
            "update is in flight or pending. Selected: --instance-group, else the "
            "groups the managed site selects (--state-dir), else every group "
            f"carrying {OWNERSHIP_LABEL}."
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
            "recorded node selector covers (also what a bound deploy host requires)"
        ),
    )


# --- reads -----------------------------------------------------------------------


def _site_clusters(site: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    spec = (site or {}).get("spec")
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    return [item for item in clusters or () if isinstance(item, Mapping)]


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
        hint = next(
            (
                str(item.get("hyperpodClusterName") or "")
                for item in _site_clusters(site)
                if item.get("eksClusterArn") == arn
            ),
            "",
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


# --- the plan ---------------------------------------------------------------------


def _group_plan(
    details: Mapping[str, Any], *, cluster_status: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """One group's record and, when it needs and allows one, its definition."""

    facts = hyperpod_group_updates(details)
    record: dict[str, Any] = {
        "name": facts["name"],
        "ownership_label": hyperpod_current_labels(details).get(OWNERSHIP_LABEL),
        **{field: details.get(field) for field in REMOVED_FIELDS},
    }
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
    for field in REMOVED_FIELDS:
        specification.pop(field, None)
    record.update(outcome=DISABLE, removes=removes)
    return record, specification


def plan_document(
    described: Mapping[str, Any],
    *,
    region: str,
    selected: Sequence[str],
    selection: str,
) -> dict[str, Any]:
    """The JSON the verb prints for ``described`` and the ``selected`` groups."""

    name = str(described.get("ClusterName") or "")
    status = str(described.get("ClusterStatus") or "")
    by_name = {
        str(group.get("InstanceGroupName") or ""): group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    }
    records, specifications = [], []
    for group in selected:
        record, specification = _group_plan(by_name[group], cluster_status=status)
        records.append(record)
        if specification is not None:
            specifications.append(specification)
    refused = [item for item in records if item["outcome"] == REFUSED]
    document: dict[str, Any] = {
        "cluster": {
            "hyperpod_name": name,
            "hyperpod_arn": described.get("ClusterArn"),
            "eks_arn": orchestrator_eks_arn(described),
            "region": region,
            "status": status,
        },
        "selection": selection,
        "instance_groups": records,
        "status": ("REFUSED" if refused else "PLAN" if specifications else "NOOP"),
        "removal": (
            "UpdateCluster is the only operation that can drop the schedule and "
            "the model rejects an empty ScheduledUpdateConfig/ScheduleExpression, "
            "so the definition leaves the member out; the model does not say "
            "whether the service reads the omission as remove or unchanged -- "
            "run verify afterwards"
        ),
        "manual_update": {
            item["name"]: (
                f"aws sagemaker update-cluster-software --region {region} "
                f"--cluster-name {name} --instance-groups "
                f"InstanceGroupName={item['name']} --deployment-config "
                "'RollingUpdatePolicy={MaximumBatchSize={Type=INSTANCE_COUNT,"
                "Value=1}},WaitIntervalInSeconds=<seconds>'"
            )
            for item in records
        },
        "section": SECTION,
    }
    if specifications and not refused:
        definitions = json.dumps(specifications, separators=(",", ":"))
        document["update_cluster"] = {
            "instance_groups": specifications,
            "command": (
                f"aws sagemaker update-cluster --region {region} --cluster-name "
                f"{name} --instance-groups {shlex.quote(definitions)}"
            ),
            "verify": (
                f"aws sagemaker describe-cluster --region {region} --cluster-name "
                f"{name} --query "
                + shlex.quote(
                    "InstanceGroups[].{InstanceGroupName:InstanceGroupName,"
                    "ScheduledUpdateConfig:ScheduledUpdateConfig,"
                    "AutoPatchConfig:AutoPatchConfig}"
                )
            ),
            "run_when": (
                "in a maintenance window, never while gpu-fault-admin config or "
                "deploy runs; the cluster is Updating until the definition applies"
            ),
        }
    return document


def run_command(
    arguments: argparse.Namespace, *, runner: CommandRunner | None = None
) -> int:
    """Print the plan; exit 0, or ``EXIT_REFUSED`` when a group is refused."""

    active = runner or CommandRunner()
    arn = str(arguments.gpu_cluster_arn).strip()
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
    document = plan_document(
        described, region=Arn.parse(arn).region, selected=selected, selection=selection
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    cluster = document["cluster"]["hyperpod_name"]
    for item in document["instance_groups"]:
        detail = (
            f"removes {', '.join(item['removes'])}"
            if item["outcome"] == DISABLE
            else item.get("refusal")
            if item["outcome"] == REFUSED
            else "no automatic update"
        )
        print(
            f"gpu-fault-admin {VERB}: {cluster}/{item['name']}: {item['outcome']} "
            f"({detail})",
            file=sys.stderr,
            flush=True,
        )
    if not selected:
        print(
            f"gpu-fault-admin {VERB}: {cluster}: no instance group selected by "
            f"{selection}; name one with --instance-group",
            file=sys.stderr,
            flush=True,
        )
    update = document.get("update_cluster")
    if update:
        print(
            f"gpu-fault-admin {VERB}: run in a maintenance window: {update['command']}",
            file=sys.stderr,
            flush=True,
        )
    return EXIT_REFUSED if document["status"] == "REFUSED" else 0
