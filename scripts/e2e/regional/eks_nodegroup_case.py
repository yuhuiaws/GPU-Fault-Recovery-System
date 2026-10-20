"""What the plain EKS node-group preparation cases read (BOOT-036, BOOT-037).

Both cases drive ``gpu-fault-admin prepare-eks-nodegroup`` (administrator
operations §5.0.1a) through the site's own deploy-host CLI
(``admin_cli.admin_command``), exactly as an administrator would, and judge its
JSON document (``eks_nodegroup_verdicts``). Everything else they need is read
here, independently of the tool, so the tool's report is checked against the
provider and the cluster rather than against itself:

* the site record (``<state-dir>/site.yaml``): the site id, the CPU control
  plane, the target ``eks-managed`` member and the HyperPod members a refusal
  probe points the tool at (``site_target``);
* the provider: ``eks describe-nodegroup``/``list-nodegroups``, the launch
  template's latest version, the group's running EC2 instances, a HyperPod
  cluster's SageMaker ARN -- reads only;
* Kubernetes: the group's Nodes (``group_nodes``) with the isolating taints the
  other acceptance preflights refuse (``blocking_node_taints``);
* the control plane's store, read in an API Pod (``store_state``): the cluster's
  incidents, executable workflows and fleet Agents;
* ``snapshot``: the state a read-only run must leave exactly as it found it,
  the tool's journals under ``<state-dir>/eks-nodegroup-prepare/`` included.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.eks_nodegroup_apply import JOURNAL_DIR
from gpu_fault.admin.eks_nodegroup_plan import SUPPORTED_AMI_TYPE
from gpu_fault.admin.eks_nodegroup_prepare import CONFIRMATION as TOOL_CONFIRMATION
from gpu_fault.admin.eks_nodegroup_prepare import VERB as TOOL_VERB
from gpu_fault.cluster_scope import DCGM_HOSTENGINE_LABEL, OWNERSHIP_LABEL
from gpu_fault.gpu_instance_inventory import GPU_INSTANCE_INVENTORY
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.admin_cli import admin_command
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence
from scripts.e2e.regional.regional_workload_inventory import blocking_node_taints
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV, load_site_profile

__all__ = [
    "Clock",
    "DCGM_HOSTENGINE_LABEL",
    "EKS_MANAGED",
    "NODEGROUP_LABEL",
    "OWNERSHIP_LABEL",
    "PLAN_TIMEOUT_SECONDS",
    "SiteTarget",
    "TOOL_CONFIRMATION",
    "TOOL_VERB",
    "ToolInvoker",
    "ToolRun",
    "admin_tool",
    "aws_json",
    "cli_errors",
    "cluster_nodegroups",
    "group_nodes",
    "hyperpod_cluster_arn",
    "nodegroup_document",
    "predecessor_gate",
    "profile_arguments",
    "record_run",
    "redacted",
    "running_instances",
    "site_target",
    "snapshot",
    "store_state",
    "tool_document",
    "unsupported_nodegroup",
    "verify_plan_identity",
    "wait_until",
]

EKS_MANAGED = "eks-managed"
HYPERPOD = "hyperpod"
EMBEDDED = "embedded"
#: The label EKS puts on every Node of a managed node group.
NODEGROUP_LABEL = "eks.amazonaws.com/nodegroup"
#: A read-only plan runs inside the CLI's 900 s read-only budget.
PLAN_TIMEOUT_SECONDS = 960.0
AWS_TIMEOUT_SECONDS = 120
#: The CLI's echo of every provider command it runs (``CommandRunner.run``).
_CALL_ECHO = re.compile(r"^\+ (aws|kubectl) (\S+)(?: (\S+))?")
_TOOL_PREFIX = "gpu-fault-admin: "
#: Bookkeeping lines of the CLI's command log, not the command's own result.
_LOG_LINES = ("gpu-fault-admin: logging to ", "gpu-fault-admin: full output in ")


@dataclass(frozen=True)
class Clock:
    """Wall clock, monotonic clock and sleep; tests hand in a fake one."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def wait_until(
    probe: Callable[[], tuple[bool, Any]],
    *,
    timeout_seconds: float,
    interval_seconds: float,
    clock: Clock,
) -> tuple[bool, Any, float]:
    """Poll ``probe`` until it answers ``(True, value)`` or the bound passes.

    Returns ``(reached, last value, seconds waited)``; the probe always runs
    at least once, and the last value is reported whatever the outcome.
    """

    started = clock.monotonic()
    while True:
        reached, value = probe()
        waited = clock.monotonic() - started
        if reached or waited >= timeout_seconds:
            return reached, value, round(waited, 3)
        clock.sleep(min(interval_seconds, max(timeout_seconds - waited, 0.0)))


# --- the site -----------------------------------------------------------------


@dataclass(frozen=True)
class HyperPodMember:
    cluster_id: str
    hyperpod_name: str
    eks_arn: str


@dataclass(frozen=True)
class SiteTarget:
    """The site and its ``eks-managed`` member a case prepares, from site.yaml."""

    state_dir: Path
    site_id: str
    region: str
    cpu_cluster_arn: str
    cpu_hyperpod_name: str
    cluster_id: str
    gpu_cluster_arn: str
    eks_name: str
    #: ``dcgmHostEngine`` the site recorded for the member (``embedded`` absent).
    dcgm_mode: str
    hyperpod_members: tuple[HyperPodMember, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "state_dir": str(self.state_dir),
            "site_id": self.site_id,
            "region": self.region,
            "cpu_cluster_arn": self.cpu_cluster_arn,
            "cpu_hyperpod_name": self.cpu_hyperpod_name,
            "cluster_id": self.cluster_id,
            "gpu_cluster_arn": self.gpu_cluster_arn,
            "eks_name": self.eks_name,
            "dcgm_mode": self.dcgm_mode,
            "hyperpod_members": [
                {
                    "cluster_id": item.cluster_id,
                    "hyperpod_name": item.hyperpod_name,
                    "eks_arn": item.eks_arn,
                }
                for item in self.hyperpod_members
            ],
        }

    def tool_arguments(self, *extra: str, gpu_cluster_arn: str = "") -> list[str]:
        """The tool's target flags; ``gpu_cluster_arn`` overrides the member."""

        return [
            "--cpu-cluster-arn",
            self.cpu_cluster_arn,
            "--gpu-cluster-arn",
            gpu_cluster_arn or self.gpu_cluster_arn,
            "--state-dir",
            str(self.state_dir),
            *extra,
        ]


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int)) else ""


def site_target(state_dir: Path, cluster_id: str) -> SiteTarget:
    """The target member of the site in ``state_dir``; refuses anything else.

    Only an ``eks-managed`` member is a target: the verb never touches HyperPod
    (it refuses one before anything is read), so a case pointed at a HyperPod
    member has nothing to prove and fails its preflight instead.
    """

    path = state_dir / "site.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RegionalFixtureError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("kind") != "RegionalSite":
        raise RegionalFixtureError(f"{path} is not a RegionalSite")
    spec = document.get("spec") if isinstance(document.get("spec"), dict) else {}
    cpu = spec.get("cpu") if isinstance(spec.get("cpu"), dict) else {}
    site_id = _text((document.get("metadata") or {}).get("name"))
    region = _text(spec.get("awsRegion"))
    cpu_arn = _text(cpu.get("eksArn"))
    if not (site_id and region and cpu_arn):
        raise RegionalFixtureError(
            f"{path} names no site id, Region or CPU EKS ARN (metadata.name, "
            "spec.awsRegion, spec.cpu.eksArn)"
        )
    clusters = [item for item in spec.get("clusters") or () if isinstance(item, dict)]
    entry = next(
        (item for item in clusters if _text(item.get("clusterId")) == cluster_id),
        None,
    )
    if entry is None:
        raise RegionalFixtureError(f"{path} has no member cluster {cluster_id}")
    provider = _text(entry.get("provider")) or HYPERPOD
    if provider != EKS_MANAGED:
        raise RegionalFixtureError(
            f"{cluster_id} is a {provider} member; prepare-eks-nodegroup prepares "
            f"{EKS_MANAGED} node groups only, so these cases do not apply to it"
        )
    gpu_arn = _text(entry.get("eksClusterArn"))
    if not gpu_arn.startswith("arn:") or "/" not in gpu_arn:
        raise RegionalFixtureError(f"{cluster_id} records no EKS cluster ARN")
    members = tuple(
        HyperPodMember(
            cluster_id=_text(item.get("clusterId")),
            hyperpod_name=_text(item.get("hyperpodClusterName")),
            eks_arn=_text(item.get("eksClusterArn")),
        )
        for item in clusters
        if (_text(item.get("provider")) or HYPERPOD) == HYPERPOD
        and _text(item.get("hyperpodClusterName"))
        and _text(item.get("eksClusterArn"))
    )
    return SiteTarget(
        state_dir=state_dir,
        site_id=site_id,
        region=region,
        cpu_cluster_arn=cpu_arn,
        cpu_hyperpod_name=_text(cpu.get("hyperpodClusterName")),
        cluster_id=cluster_id,
        gpu_cluster_arn=gpu_arn,
        eks_name=gpu_arn.rsplit("/", 1)[-1],
        dcgm_mode=_text(entry.get("dcgmHostEngine")) or EMBEDDED,
        hyperpod_members=members,
    )


# --- the tool -----------------------------------------------------------------


@dataclass(frozen=True)
class ToolRun:
    """One ``gpu-fault-admin prepare-eks-nodegroup`` run, as it ended."""

    arguments: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def document(self) -> dict[str, Any] | None:
        return tool_document(self.stdout)

    @property
    def error_lines(self) -> list[str]:
        """The CLI's own ``gpu-fault-admin: ...`` lines: why it refused."""

        return [
            line[len(_TOOL_PREFIX) :]
            for line in self.stderr.splitlines()
            if line.startswith(_TOOL_PREFIX) and not line.startswith(_LOG_LINES)
        ]

    @property
    def provider_calls(self) -> list[str]:
        """``<program> <service> <operation>`` of every provider call it echoed."""

        calls = []
        for line in self.stderr.splitlines():
            match = _CALL_ECHO.match(line)
            if match:
                calls.append(" ".join(item for item in match.groups() if item))
        return calls

    def describe(self) -> dict[str, Any]:
        document = self.document
        return {
            "arguments": list(self.arguments),
            "returncode": self.returncode,
            "document_status": (document or {}).get("status"),
            "error_lines": self.error_lines,
            "provider_calls": self.provider_calls,
        }


def tool_document(stdout: str) -> dict[str, Any] | None:
    """The JSON document the verb printed, ``None`` when it printed none.

    The verb prints one ``json.dumps(indent=2)`` object; anything the command
    log adds is on other lines, so the document starts at the first line that
    is exactly ``{``.
    """

    lines = stdout.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() != "{":
            continue
        try:
            value, _end = json.JSONDecoder().raw_decode("".join(lines[index:]))
        except ValueError:
            return None
        if isinstance(value, dict) and value.get("verb") == TOOL_VERB:
            return value
        return None
    return None


class ToolInvoker(Protocol):
    def __call__(self, arguments: Sequence[str], *, timeout: float) -> ToolRun: ...


def admin_tool(fixture: Any, state_dir: Path) -> ToolInvoker:
    """Run the verb with the deploy-host CLI that owns ``state_dir``."""

    command = admin_command(state_dir)

    def invoke(arguments: Sequence[str], *, timeout: float) -> ToolRun:
        completed = fixture.run(
            [*command, TOOL_VERB, *arguments], check=False, timeout=timeout
        )
        return ToolRun(
            tuple(arguments),
            int(completed.returncode),
            str(completed.stdout or ""),
            str(completed.stderr or ""),
        )

    return invoke


# --- the provider ---------------------------------------------------------------


def aws_json(fixture: Any, region: str, *arguments: str) -> dict[str, Any]:
    completed = fixture.run(
        ["aws", *arguments, "--region", region, "--output", "json"],
        timeout=AWS_TIMEOUT_SECONDS,
    )
    try:
        value = json.loads(completed.stdout or "{}")
    except ValueError as exc:
        raise RegionalFixtureError(
            f"aws {' '.join(arguments[:2])} returned no JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RegionalFixtureError(f"aws {' '.join(arguments[:2])} returned no object")
    return value


def cluster_nodegroups(fixture: Any, target: SiteTarget) -> dict[str, dict[str, Any]]:
    """Every managed node group of the target cluster, ``describe-nodegroup``."""

    names = aws_json(
        fixture,
        target.region,
        "eks",
        "list-nodegroups",
        "--cluster-name",
        target.eks_name,
    ).get("nodegroups")
    return {
        name: nodegroup_document(fixture, target, name)
        for name in sorted(str(item) for item in names or () if item)
    }


def latest_template_version(
    fixture: Any, target: SiteTarget, template_id: str
) -> int | None:
    listed = aws_json(
        fixture,
        target.region,
        "ec2",
        "describe-launch-template-versions",
        "--launch-template-id",
        template_id,
        "--versions",
        "$Latest",
    ).get("LaunchTemplateVersions")
    number = (listed or [{}])[0].get("VersionNumber")
    return number if isinstance(number, int) else None


def running_instances(fixture: Any, target: SiteTarget, nodegroup: str) -> list[str]:
    """The group's running instances, found by the tags the tool uses too."""

    reservations = aws_json(
        fixture,
        target.region,
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:eks:cluster-name,Values={target.eks_name}",
        f"Name=tag:eks:nodegroup-name,Values={nodegroup}",
        "Name=instance-state-name,Values=running",
    ).get("Reservations")
    return sorted(
        str(instance.get("InstanceId"))
        for reservation in reservations or ()
        for instance in (reservation or {}).get("Instances") or ()
        if instance.get("InstanceId")
    )


def hyperpod_cluster_arn(fixture: Any, region: str, name: str) -> str:
    return _text(
        aws_json(
            fixture, region, "sagemaker", "describe-cluster", "--cluster-name", name
        ).get("ClusterArn")
    )


def _gpu_types(document: Mapping[str, Any]) -> bool | None:
    """Whether the group's declared instance types are all GPU types; ``None``
    when the group declares none (its launch template does)."""

    types = [str(item) for item in document.get("instanceTypes") or () if item]
    if not types:
        return None
    return all(item.removeprefix("ml.") in GPU_INSTANCE_INVENTORY for item in types)


def unsupported_nodegroup(
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str] | None:
    """A node group of the cluster the tool must refuse, with the words of its
    refusal (``eks_nodegroup_plan.plan_group``), or ``None``.

    Only a group nobody labelled: one carrying any ownership label is either
    the site's own or makes the tool refuse the whole cluster. Refused for
    certain, whatever its launch template holds: an AMI type other than
    ``AL2023_x86_64_NVIDIA``, no launch template, or declared non-GPU
    instance types.
    """

    for name, document in sorted(documents.items()):
        if (document.get("labels") or {}).get(OWNERSHIP_LABEL):
            continue
        if document.get("status") != "ACTIVE":
            continue
        ami = str(document.get("amiType") or "")
        if ami != SUPPORTED_AMI_TYPE:
            return name, f"runs amiType {ami or 'unknown'}"
        if not (document.get("launchTemplate") or {}).get("id"):
            return name, "was created without a launch template"
        if _gpu_types(document) is False:
            return name, "the ownership label marks GPU node groups only"
    return None


# --- Kubernetes -----------------------------------------------------------------


def _node(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    labels = metadata.get("labels") or {}
    status = item.get("status") or {}
    ready = next(
        (
            condition
            for condition in status.get("conditions") or ()
            if condition.get("type") == "Ready"
        ),
        {},
    )
    spec = item.get("spec") or {}
    return {
        "name": str(metadata.get("name") or ""),
        "uid": str(metadata.get("uid") or ""),
        "instance_id": str(spec.get("providerID") or "").rsplit("/", 1)[-1],
        "ready": str(ready.get("status") or "Unknown"),
        "ready_since": ready.get("lastTransitionTime"),
        "unschedulable": bool(spec.get("unschedulable")),
        "blocking_taints": sorted(
            f"{taint.get('key')}:{taint.get('effect')}"
            for taint in blocking_node_taints(spec.get("taints") or [], labels)
        ),
        "labels": {
            key: labels.get(key)
            for key in (NODEGROUP_LABEL, OWNERSHIP_LABEL, DCGM_HOSTENGINE_LABEL)
        },
    }


def group_nodes(fixture: Any, nodegroup: str) -> list[dict[str, Any]]:
    """The group's Kubernetes Nodes, by name."""

    value = json.loads(
        fixture.kubectl(
            "gpu", "get", "nodes", "-l", f"{NODEGROUP_LABEL}={nodegroup}", "-o", "json"
        )
    )
    items = value.get("items") if isinstance(value, dict) else None
    return sorted(
        (_node(item) for item in items or () if isinstance(item, dict)),
        key=lambda item: str(item["name"]),
    )


# --- the control plane's store ----------------------------------------------------

STORE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowStatus

cluster_id, since_text = sys.argv[1], sys.argv[2]
since = datetime.fromisoformat(since_text) if since_text else None
store = ApplicationContext.from_environment().store
ACTIVE = {
    IncidentState.DETECTED,
    IncidentState.ACTION_PENDING,
    IncidentState.SAFETY_PENDING,
    IncidentState.QUARANTINED,
}
EXECUTABLE = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.RUNNING,
}


def value(item):
    return str(getattr(item, "value", item))


incidents = []
for item in store.list_incidents_by_state(cluster_id, list(IncidentState), limit=500):
    created = item.created_at
    if item.state in ACTIVE or item.state is IncidentState.ESCALATED or (
        since is not None and created >= since
    ):
        incidents.append({
            "incident_id": item.incident_id,
            "state": value(item.state),
            "node_ids": list(item.node_ids),
            "created_at": created.isoformat(),
            "active": item.state in ACTIVE,
            "reason": (list(item.reasons) or [""])[0][:200],
        })
workflows = []
for item in store.list_workflows(EXECUTABLE, limit=500):
    try:
        owner = store.get_incident(item.incident_id)
    except Exception:
        owner = None
    if owner is None or owner.cluster_id != cluster_id:
        continue
    workflows.append({
        "request_id": item.request_id,
        "status": value(item.status),
        "incident_id": item.incident_id,
        "node_ids": list(owner.node_ids),
    })
agents = [
    {
        "node_id": item.node_id,
        "lifecycle_state": value(item.lifecycle_state),
        "node_instance_id": item.node_instance_id,
        "last_seen_at": item.last_seen_at.isoformat(),
    }
    for item in store.list_agents(cluster_id)
]
print(json.dumps({
    "cluster_id": cluster_id,
    "observed_at": datetime.now(timezone.utc).isoformat(),
    "incidents": incidents,
    "workflows": workflows,
    "agents": agents,
}, sort_keys=True))
"""


def store_state(
    fixture: Any, cluster_id: str, *, since: datetime | None = None
) -> dict[str, Any]:
    """The cluster's incidents (active, escalated, or created since ``since``),
    executable workflows and Agents, read in an API Pod."""

    return dict(
        fixture.cpu_python(
            STORE_PROBE, cluster_id, since.isoformat() if since is not None else ""
        )
    )


# --- what a read-only run must not change -----------------------------------------


def _journals(state_dir: Path) -> dict[str, str]:
    root = state_dir / JOURNAL_DIR
    if not root.is_dir():
        return {}
    return {
        path.relative_to(state_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def nodegroup_document(fixture: Any, target: SiteTarget, name: str) -> dict[str, Any]:
    return dict(
        aws_json(
            fixture,
            target.region,
            "eks",
            "describe-nodegroup",
            "--cluster-name",
            target.eks_name,
            "--nodegroup-name",
            name,
        ).get("nodegroup")
        or {}
    )


def snapshot(
    fixture: Any,
    target: SiteTarget,
    nodegroups: Sequence[str],
    *,
    configuration_only: Sequence[str] = (),
) -> dict[str, Any]:
    """The named groups as the provider and Kubernetes show them, and the
    tool's journals.

    Every named group's configuration -- status, launch template and its
    latest version, labels, update config, node repair, AMI release -- which
    is all the verb ever writes; for the groups not in ``configuration_only``
    also the scaling, running instances and Nodes. Only named groups: the
    cluster's others (a refusal probe's CPU group, Karpenter's) may scale on
    their own while a case runs.
    """

    groups: dict[str, Any] = {}
    for name in sorted(set(nodegroups) | set(configuration_only)):
        document = nodegroup_document(fixture, target, name)
        template = document.get("launchTemplate") or {}
        entry: dict[str, Any] = {
            "status": document.get("status"),
            "launch_template": {
                "id": template.get("id"),
                "version": template.get("version"),
            },
            "latest_template_version": (
                latest_template_version(fixture, target, str(template["id"]))
                if template.get("id")
                else None
            ),
            "labels": dict(document.get("labels") or {}),
            "update_config": document.get("updateConfig"),
            "node_repair_config": document.get("nodeRepairConfig"),
            "release_version": document.get("releaseVersion"),
        }
        if name not in configuration_only:
            entry["scaling"] = document.get("scalingConfig")
            entry["instances"] = running_instances(fixture, target, name)
            entry["nodes"] = [
                {key: node[key] for key in ("name", "uid", "instance_id")}
                for node in group_nodes(fixture, name)
            ]
        groups[name] = entry
    return {"nodegroups": groups, "journals": _journals(target.state_dir)}


#: Lines a kept log never carries: the CLI masks sensitive commands, this is
#: the second net for anything that names a credential.
_SECRET_LINE = re.compile(r"token|secret|password|kubeconfig", re.IGNORECASE)


def redacted(text: str) -> str:
    return "".join(
        line for line in text.splitlines(keepends=True) if not _SECRET_LINE.search(line)
    )


# --- runner plumbing ---------------------------------------------------------------

HELP_TIMEOUT_SECONDS = 120.0


def cli_errors(tool: ToolInvoker) -> tuple[dict[str, Any], list[str]]:
    """Whether the site's own CLI has the verb (``--help``, read-only)."""

    run = tool(["--help"], timeout=HELP_TIMEOUT_SECONDS)
    errors = (
        []
        if run.returncode == 0 and "--apply" in run.stdout
        else [f"the site's CLI has no {TOOL_VERB} verb (exit {run.returncode})"]
    )
    return run.describe(), errors


def predecessor_gate(
    case_id: str | None, path: Path | None, identity: Mapping[str, str]
) -> dict[str, Any]:
    """The formal predecessor's evidence, bound to this release and cluster."""

    if case_id is None or path is None:
        return {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    return predecessor_evidence(path, case_id, **dict(identity))


def record_run(case_dir: Path, name: str, run: ToolRun) -> dict[str, Any]:
    """Keep one tool run beside the verdict: the plan document as JSON, the
    console output redacted; returns its summary. ``name`` must not be
    ``plan``: ``plan.json`` is the approved live-driver plan."""

    if name == "plan":
        raise ValueError("plan.json is the approved live-driver plan")

    (case_dir / f"{name}.stderr.log").write_text(redacted(run.stderr), encoding="utf-8")
    document = run.document
    if document is not None:
        write_json_atomic(case_dir / f"{name}.json", document)
    elif run.stdout.strip():
        (case_dir / f"{name}.stdout.log").write_text(
            redacted(run.stdout), encoding="utf-8"
        )
    return run.describe()


def verify_plan_identity(case_dir: Path, case_id: str, current: Any) -> None:
    """The approved plan's identity is still what the site shows."""

    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = (plan.get("details") or {}).get("preflight_identity")
    if planned != current:
        raise RegionalFixtureError(f"{case_id} plan drifted: {planned} != {current}")


def profile_arguments() -> dict[str, Any]:
    """The installed site profile's ``arguments``, empty without a profile."""

    raw = os.environ.get(SITE_PROFILE_ENV, "").strip()
    if not raw:
        return {}
    return dict(load_site_profile(Path(raw)).get("arguments") or {})
