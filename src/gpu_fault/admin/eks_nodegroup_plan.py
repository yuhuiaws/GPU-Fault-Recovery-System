"""The plan of ``gpu-fault-admin prepare-eks-nodegroup``: facts, refusals, steps.

Everything here is a read: ``eks describe-cluster``/``list-nodegroups``/
``describe-nodegroup``, ``ec2 describe-launch-template-versions`` and the
HyperPod inventory that proves the EKS cluster is a plain one. The result, per
managed node group, is what is already compliant and the exact provider
commands that would change the rest, in the order apply runs them:

1. ``ec2 create-launch-template-version --source-version <current>`` whose
   only change is the ``UserData`` carrying the managed part
   (``eks_nodegroup_user_data``) -- or the earlier version a previous run
   created with exactly that data, which is reused;
2. ``eks update-nodegroup-config --node-repair-config enabled=false`` when
   EKS node auto repair is on or unset (the product records it explicitly);
3. ``eks update-nodegroup-config --labels``: the ownership label
   ``gpu-fault.io/control-plane=<site id>`` and, in DCGM host mode, the
   ``gpu-fault.io/dcgm-hostengine=host`` declaration;
4. ``eks update-nodegroup-version`` onto that launch template version with the
   AMI release version pinned to the running one, so the roll changes the user
   data and nothing else.

Each group's current nodes are read too (``eks_nodegroup_nodes``, a read-only
SSM inventory): what they carry today, the DCGM build drift across them and
the driver CUDA major the hand-written user data parts are judged against.

A node group the verb cannot prepare safely is refused with the reason and the
manual steps of administrator operations §5.0.1: an AMI type other than
``AL2023_x86_64_NVIDIA``, no launch template (a managed node group created
without one cannot start using one), a launch template that names its own
``ImageId`` (a custom AMI), a status other than ``ACTIVE``, instance types
outside the GPU inventory, or user data the merge cannot take apart. A
HyperPod cluster is refused before anything else is read.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError, CommandRunner
from gpu_fault.admin.bootstrap_ownership import eks_nodegroup_facts
from gpu_fault.admin.eks_nodegroup_nodes import inventory_commands
from gpu_fault.admin.eks_nodegroup_user_data import (
    DCGM_HOSTENGINE_EMBEDDED,
    DCGM_HOSTENGINE_HOST,
    DCGM_HOSTENGINE_LABEL,
    DCGM_PACKAGE_VERSION,
    NODE_PACKAGES,
    DcgmPin,
    UserDataError,
    UserDataMerge,
    adoption_drops,
    dcgm_pin,
    decode_user_data,
    encode_user_data,
    existing_managed_script,
    managed_script,
    merge_managed_part,
    part_findings,
    pinned_dcgm_version,
)
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from gpu_fault.eks_node_repair import NODE_REPAIR_DISABLED, node_repair_state
from gpu_fault.gpu_instance_inventory import GPU_INSTANCE_INVENTORY

__all__ = [
    "ClusterFacts",
    "GroupPlan",
    "Step",
    "aws_command",
    "plan_group",
    "read_cluster",
    "select_nodegroups",
    "with_version",
]

SUPPORTED_AMI_TYPE = "AL2023_x86_64_NVIDIA"
MANUAL_STEPS = (
    "the manual steps of administrator operations §5.0.1 "
    "(docs/管理员日常运维.md, docs/en/administrator-operations.md)"
)
#: The section documenting this verb (the automated counterpart of §5.0.1).
AUTOMATION_SECTION = "administrator operations §5.0.1a"
VERSION_DESCRIPTION_PREFIX = "gpu-fault node prerequisites"
STEP_LAUNCH_TEMPLATE = "launch-template-version"
STEP_NODE_REPAIR = "node-repair"
STEP_LABELS = "labels"
STEP_ROLL = "roll"
#: Stands for the version number create-launch-template-version returns.
NEW_VERSION = "<new version>"

COMPLIANT = "COMPLIANT"
CHANGES = "CHANGES"
REFUSED = "REFUSED"
IN_PROGRESS = "IN_PROGRESS"


def aws_command(region: str, arguments: Sequence[str]) -> str:
    """The shell form of what ``CommandRunner.aws_json`` runs."""

    return shlex.join(["aws", *arguments, "--region", region, "--output", "json"])


@dataclass(frozen=True)
class ClusterFacts:
    arn: str
    region: str
    name: str
    #: ``describe-nodegroup`` ``nodegroup`` documents by name.
    nodegroups: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Step:
    name: str
    summary: str
    #: The ``aws`` arguments (without ``--region``/``--output``); the roll's
    #: version is :data:`NEW_VERSION` until the launch template step ran.
    arguments: tuple[str, ...]

    def describe(self, region: str) -> dict[str, Any]:
        return {
            "step": self.name,
            "summary": self.summary,
            "command": aws_command(region, self.arguments),
        }


def with_version(step: Step, version: int) -> Step:
    """The roll step pointed at the launch template version apply created."""

    arguments = list(step.arguments)
    index = arguments.index("--launch-template") + 1
    template = json.loads(arguments[index])
    template["version"] = str(version)
    arguments[index] = json.dumps(template, separators=(",", ":"))
    return Step(
        step.name,
        step.summary.replace(NEW_VERSION, str(version)),
        tuple(arguments),
    )


@dataclass(frozen=True)
class GroupPlan:
    name: str
    document: Mapping[str, Any]
    verdict: str
    refusal: str | None = None
    template: Mapping[str, Any] = field(default_factory=dict)
    current_version: int | None = None
    template_data: Mapping[str, Any] = field(default_factory=dict)
    instance_types: tuple[str, ...] = ()
    dcgm: Mapping[str, Any] = field(default_factory=dict)
    script: str = ""
    merge: UserDataMerge | None = None
    reuse_version: int | None = None
    steps: tuple[Step, ...] = ()
    #: A journaled update of an earlier run still in progress (resumed).
    pending_update: Mapping[str, Any] | None = None
    pin: DcgmPin | None = None
    #: The read-only inventory of the group's current nodes.
    nodes: Mapping[str, Any] | None = None
    #: Per hand-written shell part: overlaps, contradictions, adoption drops.
    part_reports: tuple[Mapping[str, Any], ...] = ()

    @property
    def mode(self) -> str:
        return str(self.dcgm.get("mode") or DCGM_HOSTENGINE_EMBEDDED)

    def step(self, name: str) -> Step | None:
        return next((item for item in self.steps if item.name == name), None)

    def describe(self, region: str) -> dict[str, Any]:
        document = self.document
        labels = document.get("labels") or {}
        report: dict[str, Any] = {
            "nodegroup": self.name,
            "verdict": self.verdict,
            "refusal": self.refusal,
            "status": document.get("status"),
            "ami_type": document.get("amiType"),
            "release_version": document.get("releaseVersion"),
            "capacity_type": document.get("capacityType"),
            "instance_types": list(self.instance_types),
            "scaling": document.get("scalingConfig"),
            "update_config": document.get("updateConfig"),
            "launch_template": {**self.template, "version": self.current_version},
            "node_repair": node_repair_state(document),
            "labels": {
                OWNERSHIP_LABEL: labels.get(OWNERSHIP_LABEL),
                DCGM_HOSTENGINE_LABEL: labels.get(DCGM_HOSTENGINE_LABEL),
            },
            "dcgm_hostengine": dict(self.dcgm),
            "steps": [item.describe(region) for item in self.steps],
        }
        if self.pending_update is not None:
            report["pending_update"] = dict(self.pending_update)
        if self.merge is not None:
            summary = self.merge.summary()
            reports = {item["index"]: item for item in self.part_reports}
            for part in summary["parts"]:
                part.update(reports.get(part["index"], {}))
            report["user_data"] = {**summary, "reuse_version": self.reuse_version}
            report["managed_part_script"] = self.script
            if self.merge.changed:
                report["merged_user_data"] = self.merge.text
        if self.nodes is not None:
            report["current_nodes"] = dict(self.nodes)
        if self.verdict in (COMPLIANT, CHANGES):
            report["roll"] = _roll_note(document, self.step(STEP_ROLL) is not None)
            report["verification"] = {
                "method": (
                    "SSM AWS-RunShellScript (read-only) on the group's running "
                    "instances after the steps"
                ),
                "commands": inventory_commands(wait_for_cloud_init=True),
            }
        return report

    def finding_lines(self) -> list[str]:
        """The plan's findings for the operator: part contradictions, drift."""

        names = {
            part.index: part.filename or f"part {part.index}"
            for part in (self.merge.parts if self.merge is not None else ())
        }
        lines = [
            f"{self.name}: user data part {item['index']} ({names.get(item['index'])}) "
            "is hand-written and kept beside the managed part; it also installs what "
            "the managed part installs ("
            + "; ".join(item["overlaps"])
            + f"); --adopt-user-data-part {item['index']} replaces it"
            for item in self.part_reports
            if item.get("disposition") == "kept" and item.get("overlaps")
        ]
        lines += [
            f"{self.name}: user data {line}"
            for item in self.part_reports
            for line in item.get("contradictions", ())
        ]
        lines += [
            f"{self.name}: adopting part {item['index']} drops "
            + ", ".join(item["adoption_drops"])
            for item in self.part_reports
            if item.get("adoption_drops")
        ]
        nodes = self.nodes or {}
        if nodes.get("outcome") == "unavailable":
            lines.append(
                f"{self.name}: current nodes unreadable: {nodes.get('reason')}"
            )
        lines += [f"{self.name}: nodes: {line}" for line in nodes.get("findings", ())]
        return lines


def _roll_note(document: Mapping[str, Any], needed: bool) -> dict[str, Any]:
    note: dict[str, Any] = {"needed": needed}
    if needed:
        note["caveat"] = (
            "the roll launches a replacement instance before it drains one "
            "(updateConfig); without capacity for it EKS fails the update with "
            "NodeCreationFailure after about 17 minutes and keeps the old nodes "
            "and launch template version -- harmless, rerun later"
            + (
                " (Spot: one more instance of the type must be available in the "
                "group's zones)"
                if str(document.get("capacityType") or "") == "SPOT"
                else ""
            )
        )
    return note


# --- reads -------------------------------------------------------------------


def read_cluster(
    runner: CommandRunner,
    *,
    arn: str,
    cpu_eks_arn: str,
    find_hyperpod: Callable[..., Mapping[str, Any] | None],
) -> ClusterFacts:
    """The plain EKS cluster's identity and node groups; HyperPod is refused."""

    parsed = Arn.parse(arn)
    if parsed.service == "sagemaker":
        raise BootstrapError(
            f"{arn} is a HyperPod cluster; prepare-eks-nodegroup never touches "
            "HyperPod instance groups (they are labelled with update-cluster, "
            "administrator operations §5.0.0)"
        )
    if parsed.service != "eks":
        raise BootstrapError(f"{arn} is not an EKS cluster ARN")
    name = parsed.resource_name
    if arn == cpu_eks_arn:
        raise BootstrapError(f"{arn} is the CPU control plane, not a GPU cluster")
    hyperpod = find_hyperpod(runner, eks_arn=arn, region=parsed.region)
    if hyperpod is not None:
        raise BootstrapError(
            f"EKS cluster {name} is orchestrated by HyperPod cluster "
            f"{hyperpod.get('ClusterName')}; prepare-eks-nodegroup never touches "
            "HyperPod clusters"
        )
    cluster = (
        runner.aws_json(parsed.region, "eks", "describe-cluster", "--name", name).get(
            "cluster"
        )
        or {}
    )
    if cluster.get("status") != "ACTIVE":
        raise BootstrapError(
            f"EKS cluster {name} is {cluster.get('status') or 'unknown'}, not ACTIVE"
        )
    if str(cluster.get("arn") or arn) != arn:
        raise BootstrapError(f"EKS cluster {name} describes a different ARN")
    names = sorted(
        str(item)
        for item in runner.aws_json(
            parsed.region, "eks", "list-nodegroups", "--cluster-name", name
        ).get("nodegroups")
        or ()
        if item
    )
    documents = {
        item: runner.aws_json(
            parsed.region,
            "eks",
            "describe-nodegroup",
            "--cluster-name",
            name,
            "--nodegroup-name",
            item,
        ).get("nodegroup")
        or {}
        for item in names
    }
    return ClusterFacts(arn=arn, region=parsed.region, name=name, nodegroups=documents)


def template_version(
    runner: CommandRunner, *, region: str, template: Mapping[str, Any]
) -> tuple[int, Mapping[str, Any]]:
    """``(version number, LaunchTemplateData)`` of the group's launch template."""

    listed = runner.aws_json(
        region,
        "ec2",
        "describe-launch-template-versions",
        "--launch-template-id",
        str(template.get("id")),
        "--versions",
        str(template.get("version") or "$Default"),
    ).get("LaunchTemplateVersions") or [{}]
    item = listed[0] if isinstance(listed[0], Mapping) else {}
    number = item.get("VersionNumber")
    if not isinstance(number, int) or isinstance(number, bool):
        raise BootstrapError(
            f"launch template {template.get('id')} has no version "
            f"{template.get('version')}"
        )
    data = item.get("LaunchTemplateData")
    return number, data if isinstance(data, Mapping) else {}


def _types(document: Mapping[str, Any], data: Mapping[str, Any]) -> tuple[str, ...]:
    facts = eks_nodegroup_facts(
        {**document, "instanceType": str(data.get("InstanceType") or "")}
        if not document.get("instanceTypes")
        else document
    )
    return facts.instance_types


def _gpu_types(types: Sequence[str]) -> bool:
    return bool(types) and all(
        item.removeprefix("ml.") in GPU_INSTANCE_INVENTORY for item in types
    )


def select_nodegroups(
    runner: CommandRunner,
    cluster: ClusterFacts,
    *,
    requested: Sequence[str],
    site_id: str,
) -> dict[str, tuple[int | None, Mapping[str, Any]]]:
    """The groups to prepare with their launch template versions.

    Named groups must exist; without names, the groups whose instance types
    are all GPU inventory types. A group of the cluster labelled for another
    site refuses the whole cluster, as the join would.
    """

    foreign = sorted(
        f"{name} carries {OWNERSHIP_LABEL}={value}"
        for name, document in cluster.nodegroups.items()
        if (value := (document.get("labels") or {}).get(OWNERSHIP_LABEL))
        and value != site_id
    )
    if foreign:
        raise BootstrapError(
            f"managed node groups of EKS cluster {cluster.name} name another site: "
            + "; ".join(foreign)
            + f"; this site is {OWNERSHIP_LABEL}={site_id}. One GPU cluster belongs "
            "to one site"
        )
    missing = sorted(set(requested) - set(cluster.nodegroups))
    if missing:
        raise BootstrapError(
            f"EKS cluster {cluster.name} has no managed node group "
            + ", ".join(missing)
            + "; its node groups: "
            + (", ".join(sorted(cluster.nodegroups)) or "none")
        )
    selected: dict[str, tuple[int | None, Mapping[str, Any]]] = {}
    for name, document in sorted(cluster.nodegroups.items()):
        if requested and name not in requested:
            continue
        template = document.get("launchTemplate") or {}
        version: int | None = None
        data: Mapping[str, Any] = {}
        if template.get("id"):
            version, data = template_version(
                runner, region=cluster.region, template=template
            )
        if requested or _gpu_types(_types(document, data)):
            selected[name] = (version, data)
    if not selected:
        raise BootstrapError(
            f"EKS cluster {cluster.name} has no managed node group running GPU "
            "inventory instance types ("
            + ", ".join(sorted(GPU_INSTANCE_INVENTORY))
            + "); name the group with --nodegroup"
        )
    return selected


# --- the plan ----------------------------------------------------------------


def _refused(name: str, document: Mapping[str, Any], reason: str) -> GroupPlan:
    return GroupPlan(
        name=name,
        document=document,
        verdict=REFUSED,
        refusal=f"{reason}; prepare it by hand with {MANUAL_STEPS}",
    )


def _dcgm(labels: Mapping[str, Any], requested: str | None) -> tuple[str, str] | str:
    """``(mode, source)``, or the refusal for an unknown declared value."""

    declared = labels.get(DCGM_HOSTENGINE_LABEL)
    if requested:
        return requested, "flag"
    if declared is None:
        return DCGM_HOSTENGINE_EMBEDDED, "default"
    if declared in (DCGM_HOSTENGINE_HOST, DCGM_HOSTENGINE_EMBEDDED):
        return str(declared), "label"
    return (
        f"the node group declares {DCGM_HOSTENGINE_LABEL}={declared}, which this "
        f"verb does not know; pass --dcgm-hostengine {DCGM_HOSTENGINE_HOST} or "
        f"{DCGM_HOSTENGINE_EMBEDDED}"
    )


def _label_payload(
    labels: Mapping[str, Any], *, site_id: str, mode: str, mode_source: str
) -> dict[str, Any]:
    add: dict[str, str] = {}
    remove: list[str] = []
    if labels.get(OWNERSHIP_LABEL) != site_id:
        add[OWNERSHIP_LABEL] = site_id
    if mode == DCGM_HOSTENGINE_HOST:
        if labels.get(DCGM_HOSTENGINE_LABEL) != DCGM_HOSTENGINE_HOST:
            add[DCGM_HOSTENGINE_LABEL] = DCGM_HOSTENGINE_HOST
    elif mode_source == "flag" and labels.get(DCGM_HOSTENGINE_LABEL) not in (
        None,
        DCGM_HOSTENGINE_EMBEDDED,
    ):
        remove.append(DCGM_HOSTENGINE_LABEL)
    payload: dict[str, Any] = {}
    if add:
        payload["addOrUpdateLabels"] = add
    if remove:
        payload["removeLabels"] = remove
    return payload


def version_description(source_version: int, user_data: str) -> str:
    digest = hashlib.sha256(user_data.encode("utf-8")).hexdigest()[:12]
    return (
        f"{VERSION_DESCRIPTION_PREFIX}: {' '.join(NODE_PACKAGES)} "
        f"(gpu-fault-admin prepare-eks-nodegroup; from version {source_version}; "
        f"user data sha256 {digest})"
    )


def reusable_version(
    runner: CommandRunner,
    *,
    region: str,
    template_id: str,
    current_version: int,
    current_data: Mapping[str, Any],
    user_data: str,
) -> int | None:
    """A later version an earlier run created with exactly the merged data.

    Its launch template data must equal the current version's with only
    ``UserData`` replaced -- anything else is someone else's change and is
    never rolled. A read failure means "none": the create then runs.
    """

    try:
        listed = runner.aws_json(
            region,
            "ec2",
            "describe-launch-template-versions",
            "--launch-template-id",
            template_id,
            "--min-version",
            str(current_version),
        ).get("LaunchTemplateVersions")
    except BootstrapError:
        return None
    expected = {**current_data, "UserData": encode_user_data(user_data)}
    candidates = sorted(
        (
            item
            for item in listed or ()
            if isinstance(item, Mapping)
            and int(item.get("VersionNumber") or 0) > current_version
            and str(item.get("VersionDescription") or "").startswith(
                VERSION_DESCRIPTION_PREFIX
            )
            and item.get("LaunchTemplateData") == expected
        ),
        key=lambda item: int(item.get("VersionNumber") or 0),
    )
    return int(candidates[-1]["VersionNumber"]) if candidates else None


def _steps(
    *,
    cluster: ClusterFacts,
    name: str,
    document: Mapping[str, Any],
    template_id: str,
    current_version: int,
    merge: UserDataMerge,
    reuse: int | None,
    labels: Mapping[str, Any],
) -> tuple[Step, ...]:
    base = ("--cluster-name", cluster.name, "--nodegroup-name", name)
    steps: list[Step] = []
    target = str(reuse) if reuse is not None else NEW_VERSION
    if merge.changed and reuse is None:
        steps.append(
            Step(
                STEP_LAUNCH_TEMPLATE,
                f"create launch template {template_id} version from version "
                f"{current_version}, changing only UserData ({merge.action} the "
                "managed part)",
                (
                    "ec2",
                    "create-launch-template-version",
                    "--launch-template-id",
                    template_id,
                    "--source-version",
                    str(current_version),
                    "--version-description",
                    version_description(current_version, merge.text),
                    "--launch-template-data",
                    json.dumps(
                        {"UserData": encode_user_data(merge.text)},
                        separators=(",", ":"),
                    ),
                ),
            )
        )
    if node_repair_state(document) != NODE_REPAIR_DISABLED:
        steps.append(
            Step(
                STEP_NODE_REPAIR,
                f"EKS node auto repair {node_repair_state(document)} -> disabled",
                (
                    "eks",
                    "update-nodegroup-config",
                    *base,
                    "--node-repair-config",
                    '{"enabled":false}',
                ),
            )
        )
    if labels:
        steps.append(
            Step(
                STEP_LABELS,
                "labels: " + json.dumps(labels, sort_keys=True),
                (
                    "eks",
                    "update-nodegroup-config",
                    *base,
                    "--labels",
                    json.dumps(labels, sort_keys=True, separators=(",", ":")),
                ),
            )
        )
    if merge.changed:
        release = str(document.get("releaseVersion") or "")
        steps.append(
            Step(
                STEP_ROLL,
                f"roll the node group onto launch template version {target}"
                + (f", AMI release {release} unchanged" if release else ""),
                (
                    "eks",
                    "update-nodegroup-version",
                    *base,
                    "--launch-template",
                    json.dumps(
                        {"id": template_id, "version": target}, separators=(",", ":")
                    ),
                    *(("--release-version", release) if release else ()),
                ),
            )
        )
    return tuple(steps)


def plan_group(
    runner: CommandRunner,
    cluster: ClusterFacts,
    *,
    name: str,
    version: int | None,
    data: Mapping[str, Any],
    site_id: str,
    dcgm_hostengine: str | None,
    dcgm_version: str | None,
    pending_update: Mapping[str, Any] | None = None,
    adopt: Sequence[str] = (),
    read_nodes: Callable[[str, DcgmPin | None], Mapping[str, Any]] | None = None,
) -> GroupPlan:
    """What ``name`` needs; a refusal names the manual steps.

    ``read_nodes(mode, pin)`` reads the group's current nodes (the plan's
    read-only SSM inventory); ``adopt`` names hand-written parts the managed
    part replaces (``--adopt-user-data-part``).
    """

    document = cluster.nodegroups[name]
    status = str(document.get("status") or "")
    if status != "ACTIVE":
        if status == "UPDATING" and pending_update is not None:
            return GroupPlan(
                name=name,
                document=document,
                verdict=IN_PROGRESS,
                pending_update=pending_update,
            )
        return GroupPlan(
            name=name,
            document=document,
            verdict=REFUSED,
            refusal=(
                f"node group {name} is {status or 'unknown'}, not ACTIVE, and no "
                "update of this verb is pending: rerun once it is ACTIVE, or "
                f"follow {MANUAL_STEPS}"
            ),
        )
    ami = str(document.get("amiType") or "")
    if ami != SUPPORTED_AMI_TYPE:
        return _refused(
            name,
            document,
            f"node group {name} runs amiType {ami or 'unknown'}; the automated "
            f"preparation supports {SUPPORTED_AMI_TYPE} only",
        )
    template = document.get("launchTemplate") or {}
    if not template.get("id") or version is None:
        return _refused(
            name,
            document,
            f"node group {name} was created without a launch template and cannot "
            "start using one: create a new GPU node group with a launch template "
            "(ownership label, node auto repair off)",
        )
    if data.get("ImageId"):
        return _refused(
            name,
            document,
            f"launch template {template.get('id')} version {version} names ImageId "
            f"{data.get('ImageId')} (a custom AMI: the user data needs a NodeConfig "
            "part of its own)",
        )
    types = _types(document, data)
    if not _gpu_types(types):
        return _refused(
            name,
            document,
            f"node group {name} runs {', '.join(types) or 'unknown'} instance types; "
            f"the ownership label marks GPU node groups only (inventory: "
            + ", ".join(sorted(GPU_INSTANCE_INVENTORY))
            + ")",
        )
    labels = document.get("labels") or {}
    resolved = _dcgm(labels, dcgm_hostengine)
    if isinstance(resolved, str):
        return _refused(name, document, resolved)
    mode, source = resolved
    try:
        current = decode_user_data(data.get("UserData"))
        pinned = pinned_dcgm_version(existing_managed_script(current))
        pin = (
            dcgm_pin(dcgm_version or pinned or DCGM_PACKAGE_VERSION)
            if mode == DCGM_HOSTENGINE_HOST
            else None
        )
        script = managed_script(dcgm=pin)
        merge = merge_managed_part(current, script, adopt=adopt)
    except UserDataError as exc:
        return _refused(name, document, str(exc))
    nodes = read_nodes(mode, pin) if read_nodes is not None else None
    majors = [str(item) for item in (nodes or {}).get("driver_cuda_majors") or ()]
    part_reports = tuple(
        {
            "index": part.index,
            **part_findings(part, mode=mode, pin=pin, driver_majors=majors),
            "disposition": "adopted" if part.index in merge.adopted else "kept",
            **(
                {"adoption_drops": adoption_drops(part, mode=mode)}
                if part.index in merge.adopted
                else {}
            ),
        }
        for part in merge.parts
        if part.shell and not part.managed
    )
    reuse = (
        reusable_version(
            runner,
            region=cluster.region,
            template_id=str(template["id"]),
            current_version=version,
            current_data=data,
            user_data=merge.text,
        )
        if merge.changed
        else None
    )
    steps = _steps(
        cluster=cluster,
        name=name,
        document=document,
        template_id=str(template["id"]),
        current_version=version,
        merge=merge,
        reuse=reuse,
        labels=_label_payload(labels, site_id=site_id, mode=mode, mode_source=source),
    )
    return GroupPlan(
        name=name,
        document=document,
        verdict=CHANGES if steps else COMPLIANT,
        template={
            "id": template.get("id"),
            "name": template.get("name"),
        },
        current_version=version,
        template_data=data,
        instance_types=types,
        dcgm={
            "mode": mode,
            "source": source,
            "pin": pin.value if pin else None,
            "installs": pin.describe() if pin else "no DCGM host packages",
        },
        script=script,
        merge=merge,
        reuse_version=reuse,
        steps=steps,
        pin=pin,
        nodes=nodes,
        part_reports=part_reports,
    )
