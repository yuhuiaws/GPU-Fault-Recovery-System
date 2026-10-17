"""``gpu-fault-admin prepare-eks-nodegroup``: prepare a plain EKS GPU node group.

Before a plain EKS cluster (provider ``eks-managed``) can join a site, its GPU
managed node groups need what administrator operations §5.0.0/§5.0.1 list:
``python3.12`` and ``strace`` installed by the launch template's user data,
EKS node auto repair off and the ownership label
``gpu-fault.io/control-plane=<site id>``; their rolls use ``updateStrategy``
MINIMAL (terminate before launch: no spare GPU instance needed). This verb does
those four for the recommended AMI type ``AL2023_x86_64_NVIDIA`` with the
operator's own credentials, before any site exists -- like ``site-id`` it needs
only ``--cpu-cluster-arn`` (the site id) and the GPU cluster's EKS ARN.

Default is a read-only plan (``eks_nodegroup_plan``): per node group what is
compliant, what would change and the exact commands, plus Kubernetes findings
that are reported and never changed (``eks_nodegroup_checks``). ``--apply
--confirm PREPARE_EKS_NODEGROUP --reference <change>`` re-plans and runs
exactly those steps (``eks_nodegroup_apply``), ending with the roll and an
SSM check on the new nodes; every step is skipped when already done, so a
rerun after a stop converges. With ``--state-dir`` of a site the recorded
site name must be the derived id, the apply holds the site's administrator
lock (a config/deploy running refuses it, and none can start during the
roll) and the journal and log land in the state directory; before a site
exists ``--work-dir`` takes them. Other AMI types, node groups without a
launch template and custom AMIs follow the manual steps of §5.0.1.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import re
import shlex
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin import operator_identity
from gpu_fault.admin.bootstrap import discover_cluster
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.bootstrap_discovery import find_hyperpod_for_eks
from gpu_fault.admin.bootstrap_site import load_existing_site, site_identifier
from gpu_fault.admin.cluster_join_types import DEFAULT_ALLOWED_NAMESPACES
from gpu_fault.admin.command_log import (
    ADMIN_LOG_KIND_MUTATING,
    ADMIN_LOG_KIND_READONLY,
    announce,
    command_log,
    report_failure,
)
from gpu_fault.admin.eks_nodegroup_apply import (
    COMPLETED,
    ApplyContext,
    Journal,
    apply_group,
    settled_nodegroup,
)
from gpu_fault.admin.eks_nodegroup_checks import finding_lines, kubernetes_findings
from gpu_fault.admin.eks_nodegroup_nodes import read_nodes
from gpu_fault.admin.eks_nodegroup_plan import (
    AUTOMATION_SECTION,
    CHANGES,
    COMPLIANT,
    IN_PROGRESS,
    MANUAL_STEPS,
    REFUSED,
    STEP_ROLL,
    ClusterFacts,
    GroupPlan,
    plan_group,
    read_cluster,
    select_nodegroups,
    template_version,
)
from gpu_fault.admin.eks_nodegroup_user_data import (
    DCGM_HOSTENGINE_EMBEDDED,
    DCGM_HOSTENGINE_HOST,
    DCGM_HOSTENGINE_LABEL,
    DcgmPin,
    dcgm_pin,
)
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.workflow_reconcile import REFERENCE_PATTERN
from gpu_fault.cluster_scope import OWNERSHIP_LABEL, ClusterProvider, cluster_provider

__all__ = ["CONFIRMATION", "VERB", "add_command", "run_command"]

VERB = "prepare-eks-nodegroup"
CONFIRMATION = "PREPARE_EKS_NODEGROUP"
DEFAULT_ROLL_TIMEOUT_MINUTES = 90
MAX_ROLL_TIMEOUT_MINUTES = 360
DISCOVERY_CONTEXT = "gpu-fault-admin-prepare-eks-nodegroup"
#: The plan's read of the current nodes stays well inside its 900 s budget.
PLAN_INVENTORY_TIMEOUT_SECONDS = 240.0
_NODEGROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")


def add_command(commands: Any) -> None:
    parser = commands.add_parser(
        VERB,
        usage=(
            f"gpu-fault-admin {VERB} --cpu-cluster-arn CPU_ARN --gpu-cluster-arn "
            "EKS_ARN [--nodegroup NAME ...] [--state-dir STATE_DIR | --work-dir DIR] "
            "[--dcgm-hostengine {embedded,host} [--dcgm-version V]] "
            "[--adopt-user-data-part PART ...] [--keep-update-strategy] "
            f"[--apply --confirm {CONFIRMATION} --reference CHANGE_ID]"
        ),
        help=(
            "plan (default) or apply the preparation of a plain EKS GPU managed "
            "node group (AL2023_x86_64_NVIDIA): launch template user data for "
            "python3.12/strace, node auto repair off, the ownership label, "
            "updateStrategy MINIMAL, the roll and an SSM check of the new nodes"
        ),
        description=(
            "Read-only by default: per node group what is compliant, what would "
            "change and the exact commands, plus Kubernetes findings (namespaces, "
            "device plugins, quiesce tolerations) that are never changed. "
            f"--apply --confirm {CONFIRMATION} runs the plan. Works before any "
            "site exists; with --state-dir of a site the recorded site name must "
            "match and the apply holds the site's administrator lock."
        ),
    )
    parser.add_argument(
        "--cpu-cluster-arn",
        required=True,
        metavar="CPU_ARN",
        help="the CPU control plane's EKS or HyperPod ARN (gives the site id)",
    )
    parser.add_argument(
        "--gpu-cluster-arn",
        required=True,
        metavar="EKS_ARN",
        help="the plain EKS GPU cluster (no HyperPod orchestrator)",
    )
    parser.add_argument(
        "--nodegroup",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "managed node group to prepare; repeatable. Default: every group "
            "whose instance types are GPU inventory types"
        ),
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--state-dir",
        type=Path,
        metavar="STATE_DIR",
        help=(
            "an existing site's state directory: cross-checks the site id, takes "
            "the administrator lock for --apply and keeps the journal and log"
        ),
    )
    location.add_argument(
        "--work-dir",
        type=Path,
        metavar="DIR",
        help="where --apply keeps its journal and log before a site exists",
    )
    parser.add_argument(
        "--dcgm-hostengine",
        choices=(DCGM_HOSTENGINE_EMBEDDED, DCGM_HOSTENGINE_HOST),
        default=None,
        help=(
            f"{DCGM_HOSTENGINE_HOST}: also install DCGM 4 (pinned) in the user data, "
            f"enable nvidia-dcgm and label the group {DCGM_HOSTENGINE_LABEL}="
            f"{DCGM_HOSTENGINE_HOST}. Default: the mode the group's label declares, "
            f"else {DCGM_HOSTENGINE_EMBEDDED} (the exporter's own engine)"
        ),
    )
    parser.add_argument(
        "--dcgm-version",
        metavar="VERSION-RELEASE",
        help="the DCGM 4 package version-release host mode pins",
    )
    parser.add_argument(
        "--adopt-user-data-part",
        action="append",
        default=[],
        metavar="PART",
        help=(
            "replace the hand-written user data shell part PART (its 1-based index "
            "or file name in the plan) with the managed part; repeatable. Default: "
            "hand-written parts are kept and their overlaps reported"
        ),
    )
    parser.add_argument(
        "--keep-update-strategy",
        action="store_true",
        help=(
            "keep a node group's updateStrategy DEFAULT (launch a replacement "
            "before terminating a node: needs one more GPU instance) instead of "
            "setting MINIMAL before the roll; recorded as a finding"
        ),
    )
    parser.add_argument(
        "--allowed-namespace",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "workload namespace that must exist (checked, never created); "
            "repeatable. Default: the site's list, else training"
        ),
    )
    parser.add_argument(
        "--roll-timeout-minutes",
        type=int,
        default=DEFAULT_ROLL_TIMEOUT_MINUTES,
        metavar="MINUTES",
        help=(
            f"bound of the wait for the roll (default {DEFAULT_ROLL_TIMEOUT_MINUTES}, "
            f"at most {MAX_ROLL_TIMEOUT_MINUTES}); a rerun keeps waiting"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=f"run the plan; requires --confirm {CONFIRMATION} and --reference",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        metavar="TOKEN",
        help=f"with --apply: exactly {CONFIRMATION}",
    )
    parser.add_argument(
        "--reference",
        metavar="CHANGE_ID",
        help="approved change reference recorded in the journal (--apply)",
    )


# --- request -----------------------------------------------------------------


@dataclass(frozen=True)
class PrepareRequest:
    cpu_cluster_arn: str
    gpu_cluster_arn: str
    nodegroups: tuple[str, ...]
    state_dir: Path | None
    work_dir: Path | None
    dcgm_hostengine: str | None
    dcgm_version: str | None
    allowed_namespaces: tuple[str, ...]
    roll_timeout_seconds: float
    apply: bool
    reference: str | None
    adopt_parts: tuple[str, ...] = ()
    keep_update_strategy: bool = False

    @property
    def root(self) -> Path | None:
        """Where the journal (and, without a state dir, the log) is kept."""

        return self.state_dir or self.work_dir


def _directory(value: object) -> Path | None:
    return value.expanduser().resolve() if isinstance(value, Path) else None


def prepare_request(arguments: argparse.Namespace) -> PrepareRequest:
    apply = bool(getattr(arguments, "apply", False))
    confirm = getattr(arguments, "confirm", None)
    if confirm is not None and not apply:
        raise BootstrapError("--confirm is only used with --apply")
    if apply and confirm != CONFIRMATION:
        raise BootstrapError(f"--apply requires --confirm {CONFIRMATION}")
    reference = str(getattr(arguments, "reference", None) or "").strip() or None
    if reference is not None and not REFERENCE_PATTERN.fullmatch(reference):
        raise BootstrapError("the --reference value is invalid")
    if apply and reference is None:
        raise BootstrapError("--apply requires --reference CHANGE_ID")
    state_dir = _directory(getattr(arguments, "state_dir", None))
    work_dir = _directory(getattr(arguments, "work_dir", None))
    if apply and state_dir is None and work_dir is None:
        raise BootstrapError(
            "--apply keeps its journal under --state-dir (a site) or --work-dir"
        )
    mode = getattr(arguments, "dcgm_hostengine", None)
    version = getattr(arguments, "dcgm_version", None)
    if version is not None and mode != DCGM_HOSTENGINE_HOST:
        raise BootstrapError(
            f"--dcgm-version is used only with --dcgm-hostengine {DCGM_HOSTENGINE_HOST}"
        )
    names = tuple(dict.fromkeys(getattr(arguments, "nodegroup", None) or ()))
    invalid = [name for name in names if not _NODEGROUP_NAME.fullmatch(name)]
    if invalid:
        raise BootstrapError("invalid --nodegroup name: " + ", ".join(invalid))
    minutes = int(getattr(arguments, "roll_timeout_minutes", None) or 0)
    if not 1 <= minutes <= MAX_ROLL_TIMEOUT_MINUTES:
        raise BootstrapError(
            f"--roll-timeout-minutes must be 1..{MAX_ROLL_TIMEOUT_MINUTES}"
        )
    return PrepareRequest(
        cpu_cluster_arn=str(arguments.cpu_cluster_arn).strip(),
        gpu_cluster_arn=str(arguments.gpu_cluster_arn).strip(),
        nodegroups=names,
        state_dir=state_dir,
        work_dir=work_dir,
        dcgm_hostengine=mode,
        dcgm_version=dcgm_pin(version).value if version else None,
        allowed_namespaces=tuple(
            dict.fromkeys(getattr(arguments, "allowed_namespace", None) or ())
        ),
        roll_timeout_seconds=float(minutes * 60),
        apply=apply,
        reference=reference,
        adopt_parts=tuple(
            dict.fromkeys(getattr(arguments, "adopt_user_data_part", None) or ())
        ),
        keep_update_strategy=bool(getattr(arguments, "keep_update_strategy", False)),
    )


# --- site --------------------------------------------------------------------


@dataclass(frozen=True)
class SiteContext:
    site_id: str
    cpu: ClusterIdentity
    #: ``metadata.name`` of ``<state-dir>/site.yaml``; ``None`` without one.
    recorded_site: str | None
    #: The GPU cluster is already a member of that site.
    managed: bool
    namespaces: tuple[str, ...]
    control_plane_namespace: str


def site_context(request: PrepareRequest, cpu: ClusterIdentity) -> SiteContext:
    """The site id and, with ``--state-dir``, the recorded site it must match."""

    site_id = site_identifier(cpu, [])
    document = (
        load_existing_site(request.state_dir) if request.state_dir is not None else None
    ) or {}
    recorded = str((document.get("metadata") or {}).get("name") or "") or None
    if recorded is not None and recorded != site_id:
        raise BootstrapError(
            f"{request.state_dir} records site {recorded}, but "
            f"{request.cpu_cluster_arn} derives site id {site_id}: the CPU ARN is "
            "not this site's control plane"
        )
    spec = document.get("spec") or {}
    clusters = [item for item in spec.get("clusters") or () if isinstance(item, dict)]
    shared = {
        tuple(sorted(str(name) for name in item.get("allowedNamespaces") or ()))
        for item in clusters
    }
    shared.discard(())
    namespaces = request.allowed_namespaces or (
        next(iter(shared)) if len(shared) == 1 else DEFAULT_ALLOWED_NAMESPACES
    )
    return SiteContext(
        site_id=site_id,
        cpu=cpu,
        recorded_site=recorded,
        managed=any(
            item.get("eksClusterArn") == request.gpu_cluster_arn for item in clusters
        ),
        namespaces=tuple(namespaces),
        control_plane_namespace=str(
            spec.get("namespace") or DEFAULT_ALLOWED_NAMESPACES[0]
        ),
    )


# --- plan --------------------------------------------------------------------


@dataclass(frozen=True)
class PlanResult:
    site: SiteContext
    cluster: ClusterFacts
    plans: tuple[GroupPlan, ...]
    findings: Mapping[str, Any]

    @property
    def status(self) -> str:
        verdicts = {item.verdict for item in self.plans}
        for verdict in (REFUSED, IN_PROGRESS, CHANGES):
            if verdict in verdicts:
                return verdict
        return COMPLIANT


def _group_plan(
    runner: CommandRunner,
    request: PrepareRequest,
    site: SiteContext,
    cluster: ClusterFacts,
    *,
    name: str,
    version: int | None,
    data: Mapping[str, Any],
    clocks: tuple[Callable[[float], None], Callable[[], float]] | None = None,
) -> GroupPlan:
    """One group's plan; ``clocks`` (sleep, monotonic) read its current nodes."""

    reader: Callable[[str, DcgmPin | None], Mapping[str, Any]] | None = None
    if clocks is not None:
        sleep, clock = clocks

        def reader(mode: str, pin: DcgmPin | None) -> Mapping[str, Any]:
            return read_nodes(
                runner,
                region=cluster.region,
                cluster=cluster.name,
                nodegroup=name,
                mode=mode,
                pin=pin,
                after_roll=False,
                timeout_seconds=PLAN_INVENTORY_TIMEOUT_SECONDS,
                sleep=sleep,
                clock=clock,
            )

    return plan_group(
        runner,
        cluster,
        name=name,
        version=version,
        data=data,
        site_id=site.site_id,
        dcgm_hostengine=request.dcgm_hostengine,
        dcgm_version=request.dcgm_version,
        pending_update=Journal(
            request.root, cluster=cluster, nodegroup=name
        ).pending_update(),
        adopt=request.adopt_parts,
        read_nodes=reader,
        keep_update_strategy=request.keep_update_strategy,
    )


def build_plan(
    runner: CommandRunner,
    request: PrepareRequest,
    *,
    discover: Callable[..., ClusterIdentity],
    find_hyperpod: Callable[..., Mapping[str, Any] | None],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> PlanResult:
    cpu = discover(
        runner,
        cluster_arn=request.cpu_cluster_arn,
        role="cpu",
        context=DISCOVERY_CONTEXT,
    )
    if cluster_provider(cpu.provider) is not ClusterProvider.HYPERPOD:
        raise BootstrapError(
            "the CPU control plane must be a HyperPod EKS cluster; "
            f"{request.cpu_cluster_arn} has no HyperPod orchestrator"
        )
    site = site_context(request, cpu)
    cluster = read_cluster(
        runner,
        arn=request.gpu_cluster_arn,
        cpu_eks_arn=cpu.eks_arn,
        find_hyperpod=find_hyperpod,
    )
    selected = select_nodegroups(
        runner, cluster, requested=request.nodegroups, site_id=site.site_id
    )
    plans = tuple(
        _group_plan(
            runner,
            request,
            site,
            cluster,
            name=name,
            version=version,
            data=data,
            clocks=(sleep, clock),
        )
        for name, (version, data) in selected.items()
    )
    findings = kubernetes_findings(
        runner,
        region=cluster.region,
        eks_name=cluster.name,
        namespaces=site.namespaces,
        control_plane_namespace=site.control_plane_namespace,
    )
    return PlanResult(site=site, cluster=cluster, plans=plans, findings=findings)


def _shown(value: str) -> str:
    return value if value.startswith("<") else shlex.quote(value)


def apply_command(request: PrepareRequest) -> str:
    """The command that applies this plan, for the operator to review and run."""

    words = [
        "gpu-fault-admin",
        VERB,
        "--cpu-cluster-arn",
        request.cpu_cluster_arn,
        "--gpu-cluster-arn",
        request.gpu_cluster_arn,
    ]
    for name in request.nodegroups:
        words += ["--nodegroup", name]
    if request.state_dir is not None:
        words += ["--state-dir", str(request.state_dir)]
    else:
        words += ["--work-dir", str(request.work_dir or "<WORK_DIR>")]
    if request.dcgm_hostengine:
        words += ["--dcgm-hostengine", request.dcgm_hostengine]
    if request.dcgm_version:
        words += ["--dcgm-version", request.dcgm_version]
    for name in request.allowed_namespaces:
        words += ["--allowed-namespace", name]
    for part in request.adopt_parts:
        words += ["--adopt-user-data-part", part]
    if request.keep_update_strategy:
        words.append("--keep-update-strategy")
    minutes = int(request.roll_timeout_seconds // 60)
    if minutes != DEFAULT_ROLL_TIMEOUT_MINUTES:
        words += ["--roll-timeout-minutes", str(minutes)]
    words += [
        "--apply",
        "--confirm",
        CONFIRMATION,
        "--reference",
        request.reference or "<CHANGE_ID>",
    ]
    return " ".join(_shown(word) for word in words)


def _next_steps(request: PrepareRequest, result: PlanResult) -> list[str]:
    rolls = any(item.step(STEP_ROLL) is not None for item in result.plans)
    arn = request.gpu_cluster_arn
    if result.site.managed:
        status = f"gpu-fault-admin status --state-dir {request.state_dir} --full"
        if not rolls:
            return [status]
        return [
            "a roll replaces the group's nodes: the node installer reconciler "
            "installs the new nodes by itself (about 7 minutes) and the departed "
            "nodes' Agents retire automatically (by hand: workflow-reconcile "
            "--retire-departed-agents); do not run deploy to onboard them",
            f"{status} once the replacement nodes are installed",
        ]
    target = (
        f"--state-dir {request.state_dir}" if request.state_dir else "--state-dir <dir>"
    )
    return [
        f"join the cluster: gpu-fault-admin join-cluster {target} --gpu-cluster-arn "
        f"{arn}, or name it in the first deploy (--gpu-cluster-arn {arn}); discovery "
        f"selects the node groups labelled {OWNERSHIP_LABEL}={result.site.site_id}"
    ]


def plan_document(request: PrepareRequest, result: PlanResult) -> dict[str, Any]:
    site = result.site
    document: dict[str, Any] = {
        "verb": VERB,
        "mode": "apply" if request.apply else "plan",
        "status": result.status,
        "site_id": site.site_id,
        "ownership_label": {"key": OWNERSHIP_LABEL, "value": site.site_id},
        "cpu_cluster": {
            "input_arn": site.cpu.input_arn,
            "hyperpod_name": site.cpu.hyperpod_name,
            "eks_arn": site.cpu.eks_arn,
        },
        "gpu_cluster": {
            "arn": result.cluster.arn,
            "name": result.cluster.name,
            "region": result.cluster.region,
            "provider": ClusterProvider.EKS_MANAGED.value,
            "site_managed": site.managed,
        },
        "nodegroups": [item.describe(result.cluster.region) for item in result.plans],
        "findings": dict(result.findings),
        "next_steps": _next_steps(request, result),
    }
    if request.state_dir is not None:
        document["recorded_site"] = {
            "state_dir": str(request.state_dir),
            "site_name": site.recorded_site,
            "matches": site.recorded_site == site.site_id,
        }
    if result.status == REFUSED:
        document["manual_steps"] = MANUAL_STEPS
    elif not request.apply:
        document["apply_command"] = apply_command(request)
    return document


def _say(message: str) -> None:
    print(f"gpu-fault-admin {VERB}: {message}", file=sys.stderr, flush=True)


def _summary(result: PlanResult) -> None:
    for item in result.plans:
        if item.verdict == REFUSED:
            _say(f"{item.name}: REFUSED: {item.refusal}")
        elif item.verdict == IN_PROGRESS:
            _say(
                f"{item.name}: update {(item.pending_update or {}).get('update_id')} "
                "of an earlier run in progress; apply resumes the wait"
            )
        elif item.steps:
            _say(
                f"{item.name}: would " + "; ".join(step.summary for step in item.steps)
            )
        else:
            _say(f"{item.name}: compliant")
        dropin = item.dcgm.get("restart_dropin") if item.verdict != REFUSED else None
        if isinstance(dropin, Mapping):
            _say(
                f"{item.name}: host DCGM: the managed part writes the nvidia-dcgm "
                f"drop-in {dropin.get('path')} ({dropin.get('settings')}) before "
                "enabling it"
            )
        for line in item.finding_lines():
            _say(f"finding: {line}")
    for line in finding_lines(result.findings):
        _say(f"finding: {line}")


# --- apply -------------------------------------------------------------------


@contextlib.contextmanager
def _site_lock(state_dir: Path | None) -> Iterator[None]:
    if state_dir is None:
        yield
        return
    try:
        with administrator_operation_lock(state_dir):
            yield
    except BootstrapError as exc:
        if "another administrator mutation" not in str(exc):
            raise
        raise BootstrapError(
            f"{exc}; {VERB} never rolls a node group while gpu-fault-admin "
            f"config/deploy may run -- rerun once it finished ({AUTOMATION_SECTION})"
        ) from exc


def _replan(
    runner: CommandRunner,
    request: PrepareRequest,
    result: PlanResult,
    context: ApplyContext,
    name: str,
) -> GroupPlan:
    document = settled_nodegroup(context, name)
    cluster = replace(
        result.cluster, nodegroups={**result.cluster.nodegroups, name: document}
    )
    template = document.get("launchTemplate") or {}
    version, data = (
        template_version(runner, region=cluster.region, template=template)
        if template.get("id")
        else (None, {})
    )
    return _group_plan(
        runner, request, result.site, cluster, name=name, version=version, data=data
    )


def apply_plan(
    runner: CommandRunner,
    request: PrepareRequest,
    result: PlanResult,
    *,
    operator: str,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> list[dict[str, Any]]:
    """Run every group's plan in turn; stop at the first group that stops."""

    root = request.root
    if root is None or request.reference is None:
        raise BootstrapError("--apply needs its journal directory and --reference")
    context = ApplyContext(
        runner=runner,
        cluster=result.cluster,
        root=root,
        reference=request.reference,
        operator=operator,
        site_id=result.site.site_id,
        roll_timeout_seconds=request.roll_timeout_seconds,
        sleep=sleep,
        clock=clock,
        now=now,
    )
    outcomes: list[dict[str, Any]] = []
    for plan in result.plans:
        outcome = apply_group(
            context,
            plan,
            replan=functools.partial(
                _replan, runner, request, result, context, plan.name
            ),
        )
        outcomes.append(outcome)
        if outcome.get("outcome") != COMPLETED:
            break
    return outcomes


# --- command -----------------------------------------------------------------


def _execute(
    request: PrepareRequest,
    *,
    runner: CommandRunner,
    discover: Callable[..., ClusterIdentity],
    find_hyperpod: Callable[..., Mapping[str, Any] | None],
    identity: Callable[[], str],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    now: Callable[[], datetime],
) -> int:
    reader = runner if request.apply else ReadOnlyProbeRunner(runner)
    with _site_lock(request.state_dir if request.apply else None):
        result = build_plan(
            reader,
            request,
            discover=discover,
            find_hyperpod=find_hyperpod,
            sleep=sleep,
            clock=clock,
        )
        document = plan_document(request, result)
        _summary(result)
        if result.status == REFUSED or not request.apply:
            print(json.dumps(document, indent=2, sort_keys=True))
            if result.status == REFUSED:
                raise BootstrapError(
                    "node group(s) "
                    + ", ".join(
                        item.name for item in result.plans if item.verdict == REFUSED
                    )
                    + f" cannot be prepared automatically; nothing was changed. Use {MANUAL_STEPS}"
                )
            return 0
        operator = identity()
        outcomes = apply_plan(
            runner,
            request,
            result,
            operator=operator,
            sleep=sleep,
            clock=clock,
            now=now,
        )
    stopped = [item for item in outcomes if item.get("outcome") != COMPLETED]
    document["apply"] = {
        "status": "STOPPED" if stopped else COMPLETED,
        "reference": request.reference,
        "operator": operator,
        "results": outcomes,
        "not_started": [item.name for item in result.plans[len(outcomes) :]],
    }
    print(json.dumps(document, indent=2, sort_keys=True))
    for item in outcomes:
        _say(f"{item['nodegroup']}: {item.get('outcome')}: {item.get('reason')}")
    if stopped:
        first = stopped[0]
        raise BootstrapError(
            f"{first['nodegroup']} stopped ({first.get('outcome')}): "
            f"{first.get('reason')}; journal {first.get('journal')}"
        )
    return 0


def _resolve_operator() -> str:
    return operator_identity.resolve_operator_identity(
        fallback=operator_identity.local_operator_identity()
    )


def run_command(
    arguments: argparse.Namespace,
    *,
    runner: CommandRunner | None = None,
    discover: Callable[..., ClusterIdentity] = discover_cluster,
    find_hyperpod: Callable[..., Mapping[str, Any] | None] = find_hyperpod_for_eks,
    identity: Callable[[], str] = _resolve_operator,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Plan or apply; the exit status is 0 only for a plan or a verified apply.

    With ``--work-dir`` and no state directory the command keeps its own log
    under ``<work-dir>/logs/`` (with ``--state-dir`` the CLI already does).
    """

    request = prepare_request(arguments)
    own_log = request.work_dir if request.state_dir is None else None
    kind = ADMIN_LOG_KIND_MUTATING if request.apply else ADMIN_LOG_KIND_READONLY
    with command_log(own_log, command=VERB, kind=kind) as log_path:
        try:
            return _execute(
                request,
                runner=runner or CommandRunner(),
                discover=discover,
                find_hyperpod=find_hyperpod,
                identity=identity,
                sleep=sleep,
                clock=clock,
                now=now,
            )
        except BootstrapError as exc:
            if log_path is None:
                raise
            status = report_failure("gpu-fault-admin", exc)
            announce(f"gpu-fault-admin: full output in {log_path}")
            return status
