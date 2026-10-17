"""Apply of ``gpu-fault-admin prepare-eks-nodegroup``: steps, bounded waits, proof.

Apply runs the steps of a fresh plan (``eks_nodegroup_plan``) in order and
nothing else: before every EKS write the node group is read again and must
still be ``ACTIVE`` on the launch template version the plan saw, otherwise the
run stops without writing; before the roll it must also update with the
strategy the plan rolls with (``MINIMAL`` unless kept). Each EKS update is
waited for, bounded, on ``eks describe-update``: a configuration update for at
most :data:`CONFIG_UPDATE_TIMEOUT_SECONDS`, the roll for
``--roll-timeout-minutes``. A failed roll is a stop, not a retry loop:
``NodeCreationFailure`` (the replacement instance did not join, most often for
want of capacity) is reported as retryable, and the next run reuses the launch
template version this one created.

Every attempt is journaled under ``<state-dir|work-dir>/eks-nodegroup-prepare/
<cluster>-<sha8 of the ARN>/<node group>.json`` (operator identity, change
reference, the plan, each step's command and outcome); an update still in
flight when a run stops (``pending_update``) is resumed by the next run before
it plans again, so a rerun converges instead of being refused for UPDATING.

The proof is the read-only node inventory of ``eks_nodegroup_nodes`` on the
group's running instances after the steps, judged for the group's DCGM mode:
any node missing a check stops the run as ``VERIFICATION_FAILED``, SSM
unavailable as ``UNVERIFIED``; findings (DCGM build drift, a host engine in
embedded mode) are recorded and reported without stopping it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.deadlines import remaining_timeout
from gpu_fault.admin.eks_nodegroup_plan import (
    IN_PROGRESS,
    REFUSED,
    STEP_LAUNCH_TEMPLATE,
    STEP_ROLL,
    ClusterFacts,
    GroupPlan,
    Step,
    aws_command,
    with_version,
)
from gpu_fault.admin.eks_nodegroup_nodes import read_nodes
from gpu_fault.admin.eks_nodegroup_update_strategy import update_strategy
from gpu_fault.admin.eks_nodegroup_user_data import encode_user_data
from gpu_fault.admin.process_supervisor import ensure_supervision_safe

__all__ = [
    "ApplyContext",
    "Journal",
    "apply_group",
    "settled_nodegroup",
    "wait_for_update",
]

JOURNAL_DIR = "eks-nodegroup-prepare"
JOURNAL_SCHEMA_VERSION = 1
CONFIG_UPDATE_TIMEOUT_SECONDS = 900.0
CONFIG_UPDATE_INTERVAL_SECONDS = 10.0
ROLL_INTERVAL_SECONDS = 30.0
SETTLE_SECONDS = 180.0
SETTLE_INTERVAL_SECONDS = 5.0
SSM_TIMEOUT_SECONDS = 1500.0

COMPLETED = "COMPLETED"
ROLL_FAILED = "ROLL_FAILED"
ROLL_IN_PROGRESS = "ROLL_IN_PROGRESS"
UPDATE_FAILED = "UPDATE_FAILED"
CHANGED = "CHANGED_SINCE_PLAN"
VERIFICATION_FAILED = "VERIFICATION_FAILED"
UNVERIFIED = "UNVERIFIED"
FAILED = "FAILED"
#: A journaled update EKS answers this way for is gone, not still running.
UNKNOWN_UPDATE_ERRORS = ("ResourceNotFoundException", "InvalidParameterException")

#: What a failed roll's error code means for the operator.
ROLL_ERROR_HINTS = {
    "NodeCreationFailure": (
        "the replacement instance did not join (most often no capacity for the "
        "instance type, Spot in particular; updateStrategy DEFAULT keeps the old "
        "nodes, MINIMAL already terminated the node it replaces and the group "
        "stays short until a replacement joins): retryable, rerun later"
    ),
    "AsgInstanceLaunchFailures": (
        "the Auto Scaling group could not launch the replacement instance "
        "(capacity or launch template); the old nodes stay: retryable"
    ),
    "InsufficientFreeAddresses": (
        "the node group's subnets have no free IP address for the replacement"
    ),
    "PodEvictionFailure": (
        "a PodDisruptionBudget kept a Pod from being evicted; nothing was forced: "
        "resolve the budget and rerun"
    ),
}


@dataclass(frozen=True)
class ApplyContext:
    runner: CommandRunner
    cluster: ClusterFacts
    root: Path
    reference: str
    operator: str
    site_id: str
    roll_timeout_seconds: float
    sleep: Callable[[float], None]
    clock: Callable[[], float]
    now: Callable[[], datetime]


class Journal:
    """One node group's attempts, written after every step."""

    def __init__(self, root: Path | None, *, cluster: ClusterFacts, nodegroup: str):
        digest = hashlib.sha256(cluster.arn.encode()).hexdigest()[:8]
        self.path = (
            root / JOURNAL_DIR / f"{cluster.name}-{digest}" / f"{nodegroup}.json"
            if root is not None
            else None
        )
        self.document: dict[str, Any] = self._load() or {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "cluster_arn": cluster.arn,
            "nodegroup": nodegroup,
            "pending_update": None,
            "attempts": [],
        }
        if self.document.get("cluster_arn") != cluster.arn:
            raise BootstrapError(f"{self.path} journals another cluster")

    def _load(self) -> dict[str, Any] | None:
        if self.path is None or not self.path.is_file():
            return None
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BootstrapError(f"cannot read the journal {self.path}") from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("attempts"), list):
            raise BootstrapError(f"the journal {self.path} has an unexpected shape")
        return loaded

    def pending_update(self) -> dict[str, Any] | None:
        pending = self.document.get("pending_update")
        return dict(pending) if isinstance(pending, Mapping) else None

    def begin(self, fields: Mapping[str, Any]) -> None:
        attempts = self.document["attempts"]
        attempts.append(
            {"attempt": len(attempts) + 1, **fields, "steps": [], "outcome": None}
        )
        self._write()

    def step(self, record: Mapping[str, Any]) -> None:
        self.document["attempts"][-1]["steps"].append(dict(record))
        self._write()

    def pending(self, update: Mapping[str, Any] | None) -> None:
        self.document["pending_update"] = dict(update) if update else None
        self._write()

    def finish(self, outcome: str, *, at: str) -> None:
        self.document["attempts"][-1].update(outcome=outcome, finished_at=at)
        self._write()

    def _write(self) -> None:
        if self.path is not None:
            write_json_atomic(self.path, self.document)


# --- waits -------------------------------------------------------------------


def wait_for_update(
    runner: CommandRunner,
    *,
    cluster: ClusterFacts,
    nodegroup: str,
    update_id: str,
    timeout_seconds: float,
    interval_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Poll ``eks describe-update`` until it leaves ``InProgress``, bounded.

    Returns ``status`` (``Successful``/``Failed``/``Cancelled``, or
    ``InProgress`` with ``timed_out``), the update's ``errors`` and the
    seconds waited; it never raises for a failed update.
    """

    started = clock()
    while True:
        ensure_supervision_safe()
        update = (
            runner.aws_json(
                cluster.region,
                "eks",
                "describe-update",
                "--name",
                cluster.name,
                "--nodegroup-name",
                nodegroup,
                "--update-id",
                update_id,
            ).get("update")
            or {}
        )
        status = str(update.get("status") or "")
        elapsed = clock() - started
        if status and status != "InProgress":
            return {
                "update_id": update_id,
                "status": status,
                "errors": list(update.get("errors") or ()),
                "waited_seconds": round(elapsed, 3),
            }
        if elapsed >= timeout_seconds:
            return {
                "update_id": update_id,
                "status": "InProgress",
                "timed_out": True,
                "errors": [],
                "waited_seconds": round(elapsed, 3),
            }
        sleep(min(interval_seconds, max(timeout_seconds - elapsed, 0.0)))


def _error_codes(update: Mapping[str, Any]) -> list[str]:
    return [
        str(item.get("errorCode") or "")
        for item in update.get("errors") or ()
        if isinstance(item, Mapping)
    ]


def failure_text(update: Mapping[str, Any]) -> str:
    """One line naming the update's errors and what each one means."""

    parts = []
    for item in update.get("errors") or ():
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("errorCode") or "unknown")
        text = f"{code}: {item.get('errorMessage') or ''}".strip()
        if code in ROLL_ERROR_HINTS:
            text += f" -- {ROLL_ERROR_HINTS[code]}"
        parts.append(text)
    return "; ".join(parts) or "no error detail"


# --- steps -------------------------------------------------------------------


def _stamp(context: ApplyContext) -> str:
    return context.now().isoformat()


def _write(
    context: ApplyContext, step: Step, *, action: str, nodegroup: str
) -> dict[str, Any]:
    try:
        return context.runner.aws_json(
            context.cluster.region, *step.arguments, mutate=True
        )
    except BootstrapError as exc:
        raise BootstrapError(
            f"{step.name} of node group {nodegroup} failed: {exc}. The operator's "
            f"AWS identity needs {action}; the step can also be run by hand: "
            f"{aws_command(context.cluster.region, step.arguments)}"
        ) from exc


def settled_nodegroup(context: ApplyContext, name: str) -> Mapping[str, Any]:
    """``describe-nodegroup`` once the group left a finishing ``UPDATING``.

    ``describe-update`` can report ``Successful`` a moment before the group
    itself is ``ACTIVE`` again; a bounded settle keeps that from reading as
    somebody else's change.
    """

    started = context.clock()
    while True:
        ensure_supervision_safe()
        document = (
            context.runner.aws_json(
                context.cluster.region,
                "eks",
                "describe-nodegroup",
                "--cluster-name",
                context.cluster.name,
                "--nodegroup-name",
                name,
            ).get("nodegroup")
            or {}
        )
        if (
            document.get("status") != "UPDATING"
            or context.clock() - started >= SETTLE_SECONDS
        ):
            return document
        context.sleep(SETTLE_INTERVAL_SECONDS)


def _guard(context: ApplyContext, plan: GroupPlan, step: Step) -> str | None:
    """Why the node group no longer matches the plan, ``None`` when it does;
    the roll also needs the strategy the plan rolls with."""

    document = settled_nodegroup(context, plan.name)
    status = str(document.get("status") or "")
    template = document.get("launchTemplate") or {}
    if status != "ACTIVE":
        return f"node group {plan.name} is {status or 'unknown'}"
    planned = str(plan.strategy.get("planned") or "")
    if step.name == STEP_ROLL and planned and update_strategy(document) != planned:
        return (
            f"node group {plan.name} updates with {update_strategy(document)}, the "
            f"plan rolls it with updateStrategy {planned}"
        )
    if str(template.get("id") or "") != str(plan.template.get("id") or "") or str(
        template.get("version") or ""
    ) != str(plan.current_version):
        return (
            f"node group {plan.name} now runs launch template "
            f"{template.get('id')} version {template.get('version')}, the plan saw "
            f"{plan.template.get('id')} version {plan.current_version}"
        )
    return None


def _create_version(context: ApplyContext, plan: GroupPlan, step: Step) -> int:
    response = _write(
        context, step, action="ec2:CreateLaunchTemplateVersion", nodegroup=plan.name
    )
    created = response.get("LaunchTemplateVersion") or {}
    version = created.get("VersionNumber")
    data = created.get("LaunchTemplateData") or {}
    if not isinstance(version, int) or (
        plan.merge is not None
        and "UserData" in data
        and data.get("UserData") != encode_user_data(plan.merge.text)
    ):
        raise BootstrapError(
            "create-launch-template-version returned an unexpected version: "
            + json.dumps(created, sort_keys=True)[:300]
        )
    return version


def _eks_update(
    context: ApplyContext,
    journal: Journal,
    plan: GroupPlan,
    step: Step,
    *,
    action: str,
    timeout_seconds: float,
    interval_seconds: float,
) -> dict[str, Any]:
    response = _write(context, step, action=action, nodegroup=plan.name)
    update_id = str((response.get("update") or {}).get("id") or "")
    if not update_id:
        raise BootstrapError(f"{step.name} of {plan.name} returned no update id")
    journal.pending(
        {"update_id": update_id, "step": step.name, "started_at": _stamp(context)}
    )
    outcome = wait_for_update(
        context.runner,
        cluster=context.cluster,
        nodegroup=plan.name,
        update_id=update_id,
        timeout_seconds=remaining_timeout(timeout_seconds),
        interval_seconds=interval_seconds,
        sleep=context.sleep,
        clock=context.clock,
    )
    if not outcome.get("timed_out"):
        journal.pending(None)
    return outcome


def _record(
    context: ApplyContext,
    journal: Journal,
    result: dict[str, Any],
    step: str,
    **fields: Any,
) -> None:
    record = {"step": step, "at": _stamp(context), **fields}
    result["steps"].append(record)
    journal.step(record)


def _stop(
    context: ApplyContext,
    journal: Journal,
    result: dict[str, Any],
    outcome: str,
    reason: str,
) -> dict[str, Any]:
    result.update(outcome=outcome, reason=reason)
    journal.finish(outcome, at=_stamp(context))
    return result


def _update_stop(
    context: ApplyContext,
    journal: Journal,
    result: dict[str, Any],
    step: str,
    update: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The stop for an update that did not succeed, ``None`` when it did."""

    if update.get("status") == "Successful":
        return None
    if update.get("timed_out"):
        return _stop(
            context,
            journal,
            result,
            ROLL_IN_PROGRESS if step == STEP_ROLL else UPDATE_FAILED,
            f"update {update.get('update_id')} ({step}) is still in progress after "
            f"{update.get('waited_seconds')} s; rerun the same command to keep "
            "waiting (the journal resumes this update)",
        )
    retryable = any(code in ROLL_ERROR_HINTS for code in _error_codes(update))
    return _stop(
        context,
        journal,
        result,
        ROLL_FAILED if step == STEP_ROLL else UPDATE_FAILED,
        f"update {update.get('update_id')} ({step}) ended {update.get('status')}: "
        + failure_text(update)
        + (
            "; rerun the same command later (the prepared launch template version "
            "is reused)"
            if retryable and step == STEP_ROLL
            else ""
        ),
    )


def _resume(
    context: ApplyContext, journal: Journal, plan: GroupPlan, result: dict[str, Any]
) -> dict[str, Any] | None:
    """Wait for the update an earlier run journaled; ``None`` when it succeeded
    (or EKS no longer knows it), else the stop."""

    pending = dict(plan.pending_update or {})
    step = str(pending.get("step") or STEP_ROLL)
    try:
        update = wait_for_update(
            context.runner,
            cluster=context.cluster,
            nodegroup=plan.name,
            update_id=str(pending.get("update_id")),
            timeout_seconds=remaining_timeout(context.roll_timeout_seconds),
            interval_seconds=ROLL_INTERVAL_SECONDS,
            sleep=context.sleep,
            clock=context.clock,
        )
    except BootstrapError as exc:
        if not any(code in str(exc) for code in UNKNOWN_UPDATE_ERRORS):
            raise
        journal.pending(None)
        _record(
            context,
            journal,
            result,
            f"resume-{step}",
            update={"update_id": pending.get("update_id"), "status": "unknown"},
            note="EKS no longer knows the journaled update; the group is planned again",
        )
        return None
    if not update.get("timed_out"):
        journal.pending(None)
    _record(context, journal, result, f"resume-{step}", update=update)
    return _update_stop(context, journal, result, step, update)


def apply_group(
    context: ApplyContext,
    plan: GroupPlan,
    *,
    replan: Callable[[], GroupPlan],
) -> dict[str, Any]:
    """Run ``plan`` for one node group; the result names the outcome.

    ``replan`` reads the group again after a resumed update finished.
    """

    journal = Journal(context.root, cluster=context.cluster, nodegroup=plan.name)
    journal.begin(
        {
            "started_at": _stamp(context),
            "reference": context.reference,
            "operator": context.operator,
            "site_id": context.site_id,
            "plan": plan.describe(context.cluster.region),
        }
    )
    result: dict[str, Any] = {
        "nodegroup": plan.name,
        "journal": str(journal.path),
        "steps": [],
    }
    try:
        return _run_group(context, journal, plan, result, replan=replan)
    except BootstrapError as exc:
        _stop(context, journal, result, FAILED, str(exc))
        raise


def _run_group(
    context: ApplyContext,
    journal: Journal,
    plan: GroupPlan,
    result: dict[str, Any],
    *,
    replan: Callable[[], GroupPlan],
) -> dict[str, Any]:
    if plan.verdict == IN_PROGRESS:
        stopped = _resume(context, journal, plan, result)
        if stopped is not None:
            return stopped
        plan = replan()
        _record(context, journal, result, "replan", verdict=plan.verdict)
    if plan.verdict == REFUSED:
        return _stop(context, journal, result, REFUSED, str(plan.refusal))
    stale = journal.pending_update()
    if stale is not None:
        # The group is ACTIVE: whatever the journal still names is not in flight.
        journal.pending(None)
        _record(context, journal, result, "pending-cleared", update=stale)
    version = plan.reuse_version or plan.current_version
    if plan.reuse_version is not None:
        _record(
            context,
            journal,
            result,
            STEP_LAUNCH_TEMPLATE,
            outcome="reused",
            version=plan.reuse_version,
        )
    for step in plan.steps:
        if step.name == STEP_LAUNCH_TEMPLATE:
            version = _create_version(context, plan, step)
            _record(
                context,
                journal,
                result,
                step.name,
                outcome="created",
                version=version,
                command=aws_command(context.cluster.region, step.arguments),
            )
            continue
        changed = _guard(context, plan, step)
        if changed is not None:
            return _stop(
                context,
                journal,
                result,
                CHANGED,
                f"{changed}: nothing more is written; rerun to plan again",
            )
        if step.name == STEP_ROLL:
            if version is None:
                raise BootstrapError(f"no launch template version to roll {plan.name}")
            step = with_version(step, version)
        update = _eks_update(
            context,
            journal,
            plan,
            step,
            action=(
                "eks:UpdateNodegroupVersion"
                if step.name == STEP_ROLL
                else "eks:UpdateNodegroupConfig"
            ),
            timeout_seconds=(
                context.roll_timeout_seconds
                if step.name == STEP_ROLL
                else CONFIG_UPDATE_TIMEOUT_SECONDS
            ),
            interval_seconds=(
                ROLL_INTERVAL_SECONDS
                if step.name == STEP_ROLL
                else CONFIG_UPDATE_INTERVAL_SECONDS
            ),
        )
        _record(
            context,
            journal,
            result,
            step.name,
            command=aws_command(context.cluster.region, step.arguments),
            update=update,
        )
        stopped = _update_stop(context, journal, result, step.name, update)
        if stopped is not None:
            return stopped
    verification = read_nodes(
        context.runner,
        region=context.cluster.region,
        cluster=context.cluster.name,
        nodegroup=plan.name,
        mode=plan.mode,
        pin=plan.pin,
        after_roll=True,
        timeout_seconds=SSM_TIMEOUT_SECONDS,
        sleep=context.sleep,
        clock=context.clock,
    )
    _record(context, journal, result, "verification", **verification)
    result["verification"] = verification
    if verification["outcome"] == "unavailable":
        return _stop(
            context,
            journal,
            result,
            UNVERIFIED,
            f"the nodes could not be checked: {verification['reason']}; "
            f"{verification['remedy']}",
        )
    if verification["outcome"] == "failed":
        return _stop(
            context,
            journal,
            result,
            VERIFICATION_FAILED,
            "prerequisites missing on "
            + "; ".join(
                f"{item['instance_id']} ({item['ssm_status']}: "
                + ", ".join(
                    f"{name} {check['observed']} (expected {check['expected']})"
                    for name, check in item["checks"].items()
                    if not check["ok"]
                )
                + ")"
                for item in verification["nodes"]
                if not item["passed"]
            ),
        )
    return _stop(context, journal, result, COMPLETED, "prepared and verified")
