"""The CPU rolling update's status check and the deploy's update-window
preflight (batch 240).

``cpu_rolling_update`` -- a ``status --full`` check -- judges what the deploy's
last step (``gpu_fault.admin.cpu_rolling_update``) keeps: with
``spec.cpuControlPlane.rollingUpdate.mode`` ``rolling`` it passes only when
every scheduled CPU instance group rolls (a batch smaller than the group, the
b231 grading), its ``AutoRollbackConfiguration`` names the product's two
alarms, both alarms exist and are OK, and they watch the current API NLB and
target group; with ``unchanged`` the schedule is the customer's, and only
alarms a schedule still names are judged. A deferred or failed step in the
journal fails it with its reason, and so does a step that stopped between
removing a group's schedule and creating it again. An alarm in ALARM or INSUFFICIENT_DATA whose
configuration changed within ``GRACE`` is PENDING (a WARN): a new alarm starts
INSUFFICIENT_DATA. The deploy's own ``verify`` gate runs before the step, so
the check is not registered there (``health_checks``); asked in ``verify``
anyway, every finding is a WARN: the step that fixes it has not run yet. Each
failure names the remedy: re-run the deploy, or ``gpu-fault-admin
cpu-rolling-update`` (the same step); ``hyperpod-scheduled-update`` prints the
same requests for a manual change.

``cpu_update_window`` -- appended to every deploy preflight -- refuses to start
while the CPU group has an AMI update in flight (``SoftwareUpdateStatus``
Pending, InProgress, RollbackInProgress; ``Status`` Updating, SystemUpdating)
and warns, with the times, when the deploy may overlap the group's next
scheduled update window.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin.automatic_node_updates import rolling_batch
from gpu_fault.admin.config import default_admin_config, load_desired_admin_config
from gpu_fault.admin.cpu_rolling_update import (
    AWAITING_CREATE,
    ROLLING,
    finish_deploy,
    load_journal,
    record_last_run,
    update_in_progress,
)
from gpu_fault.admin.hyperpod_schedule_window import (
    UnsupportedSchedule,
    current_or_next_window,
)
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_errors import CheckFailure
from gpu_fault_release.regional_release_state import aws_json

__all__ = [
    "CHECK",
    "DEPLOY_MINUTES",
    "GRACE",
    "WINDOW_CHECK",
    "check_cpu_rolling_update",
    "cpu_update_window",
    "evaluate",
    "health_checks",
    "run_step",
]

CHECK = "cpu_rolling_update"
WINDOW_CHECK = "cpu_update_window"
#: An alarm put this recently may still be evaluating (PENDING, not FAIL).
GRACE = timedelta(minutes=10)
#: A conservative deploy duration for the window warning (live: 50-90 min).
DEPLOY_MINUTES = 120
REMEDY = (
    "re-run gpu-fault-admin deploy, or gpu-fault-admin cpu-rolling-update "
    "--state-dir <state dir> (the deploy's same last step); gpu-fault-admin "
    "hyperpod-scheduled-update --rolling --rollback-alarm ... prints the same "
    "requests for a manual change (administrator-operations §5.0.5)"
)


def _stamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _changed_at(alarm: Mapping[str, Any]) -> datetime | None:
    """When an alarm's configuration or state last changed."""

    stamps = [
        _stamp(alarm.get(key))
        for key in ("AlarmConfigurationUpdatedTimestamp", "StateUpdatedTimestamp")
    ]
    known = [stamp for stamp in stamps if stamp is not None]
    return max(known) if known else None


def _rolls(group: Mapping[str, Any]) -> bool:
    batch = rolling_batch(group.get("ScheduledUpdateConfig"))
    count = group.get("CurrentCount")
    if batch is None:
        return False
    if batch["type"] == "CAPACITY_PERCENTAGE":
        return int(batch["value"]) < 100
    return type(count) is int and int(batch["value"]) < count


def evaluate(
    described: Mapping[str, Any],
    alarms: Mapping[str, Any],
    nlb: Mapping[str, Any] | None,
    *,
    mode: str,
    names: Sequence[str],
    journal: Mapping[str, Any],
    now: datetime,
) -> tuple[str, str, dict[str, Any]]:
    """``(PASS | PENDING | FAIL, summary, details)`` from the reads."""

    groups = [
        group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    ]
    scheduled = [group for group in groups if group.get("ScheduledUpdateConfig")]
    problems: list[str] = []
    pending: list[str] = []
    attached = set()
    for group in scheduled:
        name = group.get("InstanceGroupName")
        named = set(rollback.rollback_alarm_names(group.get("ScheduledUpdateConfig")))
        if set(names) <= named:
            attached.add(str(name))
        if mode != ROLLING:
            continue
        if not _rolls(group):
            problems.append(
                f"instance group {name} re-images every instance at once (no rolling "
                "batch smaller than the group)"
            )
        if not set(names) <= named:
            problems.append(
                f"instance group {name}'s AutoRollbackConfiguration does not name "
                f"{' and '.join(names)}"
            )
    for name, entry in (journal.get("groups") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("phase") in {"deferred", "failed"}:
            problems.append(
                f"the deploy's CPU rolling schedule step for {name} was "
                f"{entry['phase']}: {entry.get('reason')}"
            )
        elif entry.get(AWAITING_CREATE):
            problems.append(
                f"the deploy's CPU rolling schedule step for {name} stopped at "
                f"{entry.get('phase')} after removing the group's "
                "ScheduledUpdateConfig, before creating it again: the group may "
                "have no automatic patching until the step runs again"
            )
    judge_alarms = (mode == ROLLING and bool(scheduled)) or bool(attached)
    records = {
        str(item["name"]): item for item in rollback.alarm_records(alarms, list(names))
    }
    raw = {
        str(item.get("AlarmName")): item
        for item in alarms.get("MetricAlarms") or ()
        if isinstance(item, Mapping)
    }
    for name in names if judge_alarms else ():
        record = records[name]
        changed = _changed_at(raw.get(name) or {})
        for problem in rollback.alarm_problems(record, nlb):
            # A state, not a definition: an alarm put moments ago may still be
            # evaluating its first data points.
            if (
                changed is not None
                and now - changed <= GRACE
                and problem.startswith(("is INSUFFICIENT_DATA", "is in ALARM"))
            ):
                pending.append(
                    f"rollback alarm {name} {problem} (changed {changed.isoformat()})"
                )
            else:
                problems.append(f"rollback alarm {name} {problem}")
    details = {
        "mode": mode,
        "alarm_names": list(names),
        "instance_groups": [
            {
                "name": group.get("InstanceGroupName"),
                "scheduled_update": (group.get("ScheduledUpdateConfig") or {}).get(
                    "ScheduleExpression"
                ),
                "rolling_batch": rolling_batch(group.get("ScheduledUpdateConfig")),
            }
            for group in groups
        ],
        "alarms": [records[name] for name in names],
        "api_nlb": nlb,
        "problems": problems,
        "pending": pending,
    }
    if problems:
        return "FAIL", "; ".join(problems) + f" -- {REMEDY}", details
    if pending:
        return "PENDING", "PENDING: " + "; ".join(pending), details
    if mode != ROLLING and not attached:
        return (
            "PASS",
            "unchanged: the CPU group's schedule is left as found "
            "(spec.cpuControlPlane.rollingUpdate.mode)",
            details,
        )
    if not scheduled:
        return "PASS", "no CPU instance group has a ScheduledUpdateConfig", details
    return (
        "PASS",
        "the CPU control plane rolls one batch at a time, rolled back on "
        + " and ".join(names)
        + " (both OK, on the current API NLB)",
        details,
    )


def _settings(release: Any) -> tuple[str, tuple[str, str], dict[str, Any]]:
    state_dir = getattr(release.config, "admin_state_dir", None)
    config = (
        load_desired_admin_config(state_dir)
        if state_dir is not None
        else default_admin_config()
    ).cpu_control_plane
    journal = (
        load_journal(state_dir)
        if state_dir is not None
        else {"groups": {}, "alarms": {}}
    )
    names = (config.rollback_alarms.capacity, config.rollback_alarms.not_restored)
    return config.rolling_update.mode, names, journal


def _api_nlb(release: Any) -> dict[str, Any] | None:
    nlb = dict(getattr(release.config, "nlb", None) or {})
    if not nlb:
        return None
    name = str(nlb.get("name") or f"gpu-fault-regional-{release.config.aws_region}")
    balancers = aws_json(release, ["elbv2", "describe-load-balancers", "--names", name])
    groups = aws_json(
        release,
        [
            "elbv2",
            "describe-target-groups",
            "--load-balancer-arn",
            rollback.load_balancer_arn(balancers),
        ],
    )
    return rollback.api_nlb(name, balancers, groups)


def check_cpu_rolling_update(
    release: Any,
    *,
    mode: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CheckValue:
    """The merged check: FAIL (``status``) or WARN (``verify``) per ``evaluate``."""

    try:
        rolling_mode, names, journal = _settings(release)
        described = aws_json(
            release,
            [
                "sagemaker",
                "describe-cluster",
                "--cluster-name",
                str(release.config.cpu_hyperpod_cluster_name),
            ],
        )
        alarms = aws_json(release, rollback.describe_alarms_arguments(list(names)))
        verdict, summary, details = evaluate(
            described,
            alarms,
            _api_nlb(release),
            mode=rolling_mode,
            names=names,
            journal=journal,
            now=now(),
        )
    except Exception as exc:
        if mode != "verify":
            raise
        # The deploy's own gate never fails on what its last step repairs.
        return CheckValue(
            f"could not judge the CPU rolling update ({exc}); {REMEDY}",
            status="WARN",
        )
    if verdict == "FAIL" and mode != "verify":
        raise CheckFailure(summary, details)
    if verdict == "PASS":
        return CheckValue(summary, details)
    if verdict == "FAIL":
        summary = (
            "pending the deploy's last step (it creates or re-points the alarms and "
            f"applies the schedule): {summary}"
        )
    return CheckValue(summary, details, status="WARN")


def run_step(release: Any) -> dict[str, Any]:
    """The ``cpu-rolling-update`` mode: the deploy's last step
    (``cpu_rolling_update.finish_deploy``) for this release's site, recorded
    as its ``last-run.json``; never raises for what it finds."""

    config = release.config
    state_dir = getattr(config, "admin_state_dir", None)
    if state_dir is None:
        return {
            "status": "SKIPPED",
            "warnings": ["no managed state directory: nothing to record the step in"],
        }
    result = finish_deploy(
        {
            "aws_region": config.aws_region,
            "cpu_eks_arn": config.cpu_eks_arn,
            "cpu_hyperpod_cluster_name": config.cpu_hyperpod_cluster_name,
            "site_name": config.site_name,
            "nlb": dict(getattr(config, "nlb", None) or {}),
        },
        load_desired_admin_config(state_dir).cpu_control_plane,
        state_dir,
    )
    record_last_run(state_dir, result)
    return result


def health_checks(
    release: Any, mode: str
) -> list[tuple[str, Callable[[], CheckValue]]]:
    """What ``build_health_report`` adds: the check in ``status`` only -- the
    deploy's ``verify`` gate runs before the step this check judges."""

    if mode != "status":
        return []
    return [(CHECK, lambda: check_cpu_rolling_update(release, mode=mode))]


def cpu_update_window(
    release: Any,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    deploy_minutes: int = DEPLOY_MINUTES,
) -> CheckValue:
    """Refuse a deploy during an AMI update of the CPU group; warn when the
    deploy may overlap the group's next scheduled update window."""

    try:
        described = aws_json(
            release,
            [
                "sagemaker",
                "describe-cluster",
                "--cluster-name",
                str(release.config.cpu_hyperpod_cluster_name),
            ],
        )
    except Exception as exc:
        # Only a positive finding refuses a deploy; regional_contexts already
        # fails one that cannot describe the CPU cluster at all.
        return CheckValue(
            f"could not read the CPU control-plane cluster's update state ({exc})",
            status="WARN",
        )
    groups = [
        group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    ]
    for group in groups:
        reason = update_in_progress(described, group)
        if reason:
            raise CheckFailure(
                f"the CPU control-plane instance group {group.get('InstanceGroupName')} "
                f"has an AMI update in progress: {reason}; the deploy refuses to start "
                "while the control plane is being re-imaged -- rerun once "
                "SoftwareUpdateStatus is Succeeded or RollbackComplete and the group "
                "is InService",
                {"instance_group": group.get("InstanceGroupName"), "reason": reason},
            )
    start = now()
    end = start + timedelta(minutes=deploy_minutes)
    overlaps = []
    for group in groups:
        schedule = group.get("ScheduledUpdateConfig") or {}
        expression = str(schedule.get("ScheduleExpression") or "")
        if not expression:
            continue
        wait = (schedule.get("DeploymentConfig") or {}).get("WaitIntervalInSeconds")
        count = group.get("CurrentCount")
        try:
            window = current_or_next_window(
                expression,
                now=start,
                instances=count if type(count) is int else 1,
                wait_seconds=wait if type(wait) is int else 0,
            )
        except UnsupportedSchedule as exc:
            overlaps.append(f"{group.get('InstanceGroupName')}: {exc}")
            continue
        if window is not None and window.overlaps(start, end):
            overlaps.append(
                f"{group.get('InstanceGroupName')} {expression}: the window "
                f"{window.start.isoformat()} - about {window.end.isoformat()}"
            )
    if overlaps:
        return CheckValue(
            f"the deploy (up to about {deploy_minutes} min from {start.isoformat()}) "
            "may overlap the CPU group's scheduled update window -- "
            + "; ".join(overlaps)
            + "; consider deploying after it",
            {"overlaps": overlaps},
            status="WARN",
        )
    return CheckValue(
        "no AMI update in progress on the CPU control-plane cluster, and no "
        "scheduled update window within the deploy's expected duration"
    )
