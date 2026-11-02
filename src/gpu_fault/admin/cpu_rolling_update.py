"""The CPU control plane's rolling update, owned by the deploy (batch 240).

Part A, the rollback alarms. The CPU HyperPod group's rolling schedule rolls
back on two CloudWatch alarms over the API NLB target group's
``HealthyHostCount`` (``hyperpod_rollback.alarm_definitions``: Maximum below
N-1 for 5 minutes, below N for 25). They watch the product's own NLB, so only
the deploy can create them, and a fresh install's new NLB must re-point them.
``ensure_alarms`` runs at the very end of a deploy, after every api-ha replica
is a healthy target and ``HealthyHostCount`` Maximum has been N for one whole
minute (``wait_for_full_capacity``): a new alarm starts ``INSUFFICIENT_DATA``
and is then evaluated on data that is already OK, and an update keeps its
state (``PUT_METRIC_ALARM_EVIDENCE``). An alarm that already matches is not
put again, and when neither needs a put the step does not wait for that
minute (``alarms_to_put``). A missing alarm is created with the site tag; an
existing one is
the site's when it carries the site tag, is adopted (tagged) when untagged and
on this site's NLB, and refused when another site's tag names it -- two sites
in one account and Region set ``spec.cpuControlPlane.rollbackAlarms``.

Part B, the schedule (``spec.cpuControlPlane.rollingUpdate``, default
``rolling``). Only after both alarms are OK: the group's own
``ScheduleExpression`` is kept (the product changes how nodes update, never
when), a group without a schedule is left alone, and a group that already
rolls this way is a NOOP. Otherwise the step records the group's current
``ScheduledUpdateConfig`` (the pre-install state uninstall restores), then
runs exactly the sequence proven live on 2026-11-02: delete (the definition
without the member), verify it is gone, create (the rolling definition), verify
the readback. An update in flight, a write refusal or less than
``WINDOW_MARGIN`` before the next window defers the step; a create that fails
after the delete is retried once, then fails loudly with the exact command:
the group is then without a schedule -- no automatic patching, which is safe
but must be fixed.

The journal is ``<state dir>/cpu-rolling-update/state.json``; it carries the
pre-install record across runs, so a group a crash left without a schedule is
created again, not mistaken for one that never had a schedule, and it lists
every alarm the product put (the installation ledger records them from it).
Writes go through the operator's ``CommandRunner`` with ``mutate=True``. Nothing
raises out of ``finish_deploy``: the alarms and the schedule protect the
control plane, they do not make it run.
"""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    Arn,
    BootstrapError,
    CommandRunner,
    assert_site_tag,
)
from gpu_fault.admin.capacity_defaults import INGRESS_REPLICAS
from gpu_fault.admin.hyperpod_schedule_window import (
    SCHEDULE_EVIDENCE,
    UnsupportedSchedule,
    current_or_next_window,
)
from gpu_fault.admin.ownership_release import (
    hyperpod_current_labels,
    hyperpod_instance_group_specification,
    hyperpod_write_refusal,
)

__all__ = [
    "AWAITING_CREATE",
    "DELETE_ALARMS_EVIDENCE",
    "DESCRIBE_ALARMS_SCOPE_EVIDENCE",
    "EVIDENCE",
    "PUT_METRIC_ALARM_EVIDENCE",
    "ROLLING",
    "SOFTWARE_UPDATE_EVIDENCE",
    "STATE_PATH",
    "UNCHANGED",
    "Clock",
    "Target",
    "alarms_to_put",
    "apply_schedule",
    "ensure_alarms",
    "finish_deploy",
    "journal_path",
    "load_journal",
    "load_last_run",
    "record_last_run",
    "restore_schedule",
    "update_in_progress",
]

STATE_PATH = Path("cpu-rolling-update") / "state.json"
#: Journaled before a delete, cleared once the create is verified: this group's
#: missing schedule is the step's doing, so the next run creates it again.
AWAITING_CREATE = "awaiting_create"
ROLLING = "rolling"
UNCHANGED = "unchanged"
#: ``ClusterInstanceGroupDetails``: an AMI update is in flight.
UPDATING_SOFTWARE = frozenset({"Pending", "InProgress", "RollbackInProgress"})
UPDATING_GROUP = frozenset({"Updating", "SystemUpdating"})
#: The step never writes this close before the next scheduled window.
WINDOW_MARGIN = timedelta(hours=3)
CAPACITY_TIMEOUT_SECONDS = 600
ALARM_OK_TIMEOUT_SECONDS = 600
CONVERGE_TIMEOUT_SECONDS = 900
INTERVAL_SECONDS = 15

PUT_METRIC_ALARM_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/"
        "API_PutMetricAlarm.html"
    ),
    "page": "PutMetricAlarm",
    "section": "the operation description; Request Parameters, Tags",
    "docs_date": "Last-Modified 2026-10-30",
    "quotes": (
        "When this operation creates an alarm, the alarm state is immediately set "
        "to `INSUFFICIENT_DATA`. For PromQL alarms, the alarm state is instead "
        "immediately set to `OK`. The alarm is then evaluated and its state is set "
        "appropriately. Any actions associated with the new state are then "
        "executed.",
        "When you update an existing alarm, its state is left unchanged, but the "
        "update completely overwrites the previous configuration of the alarm.",
        "If you are using this operation to update an existing alarm, any tags you "
        "specify in this parameter are ignored. To change the tags of an existing "
        "alarm, use TagResource or UntagResource.",
        "To use this field to set tags for an alarm when you create it, you must be "
        "signed on with both the cloudwatch:PutMetricAlarm and cloudwatch:TagResource "
        "permissions.",
    ),
}
#: Why the alarm read needs ``cloudwatch:DescribeAlarms`` on ``*``: it asks for
#: composite alarms too (``hyperpod_rollback.describe_alarms_arguments``).
DESCRIBE_ALARMS_SCOPE_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/"
        "API_DescribeAlarms.html"
    ),
    "page": "DescribeAlarms",
    "section": "the operation description",
    "docs_date": "Last-Modified 2026-10-30",
    "quotes": (
        "To use this operation and return information about composite alarms, you "
        "must be signed on with the cloudwatch:DescribeAlarms permission that is "
        "scoped to *. You can't return information about composite alarms if your "
        "cloudwatch:DescribeAlarms permission has a narrower scope.",
    ),
}
#: Why uninstall reads an alarm back after deleting it (the residue gate).
DELETE_ALARMS_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/"
        "API_DeleteAlarms.html"
    ),
    "page": "DeleteAlarms",
    "section": "the operation description",
    "docs_date": "Last-Modified 2026-10-30",
    "quotes": (
        "If you specify any incorrect alarm names, the alarms you specify with "
        "correct names are still deleted. Other syntax errors might result in no "
        "alarms being deleted. To confirm that alarms were deleted successfully, you "
        "can use the DescribeAlarms operation after using DeleteAlarms.",
    ),
}
#: What an AMI update in flight looks like (AWS's own spelling, typo kept).
SOFTWARE_UPDATE_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/sagemaker/latest/APIReference/"
        "API_ClusterInstanceGroupDetails.html"
    ),
    "page": "ClusterInstanceGroupDetails",
    "section": "Contents, SoftwareUpdateStatus and Status",
    "version": "API version 2017-07-24",
    "docs_date": "Last-Modified 2026-10-30",
    "quotes": (
        "Status of the last software udpate request. Status transitions follow "
        "these possible sequences: Pending -> InProgress -> Succeeded Pending -> "
        "InProgress -> RollbackInProgress -> RollbackComplete Pending -> InProgress "
        "-> RollbackInProgress -> Failed",
        "Valid Values: `InService | Creating | Updating | Failed | Degraded | "
        "SystemUpdating | Deleting`",
    ),
}
#: Batch 240's vendor evidence, pinned in the manuals.
EVIDENCE = (
    PUT_METRIC_ALARM_EVIDENCE,
    SOFTWARE_UPDATE_EVIDENCE,
    SCHEDULE_EVIDENCE,
    DESCRIBE_ALARMS_SCOPE_EVIDENCE,
    DELETE_ALARMS_EVIDENCE,
)


@dataclass(frozen=True)
class Target:
    """What the step works on: the rendered site plus the admin config."""

    region: str
    partition: str
    account: str
    cluster: str
    site_id: str
    nlb_name: str
    mode: str
    wait_seconds: int
    alarm_names: tuple[str, str]
    replicas: int = INGRESS_REPLICAS

    @classmethod
    def of(cls, release_config: Mapping[str, Any], cpu_control_plane: Any) -> Target:
        """From ``site.release_config`` and ``AdminConfig.cpu_control_plane``."""

        nlb = release_config.get("nlb") or {}
        region = str(release_config["aws_region"])
        cpu = Arn.parse(str(release_config["cpu_eks_arn"]))
        return cls(
            region=region,
            partition=cpu.partition,
            account=cpu.account,
            cluster=str(release_config["cpu_hyperpod_cluster_name"]),
            site_id=str(release_config["site_name"]),
            nlb_name=str(nlb.get("name") or f"gpu-fault-regional-{region}"),
            mode=str(cpu_control_plane.rolling_update.mode),
            wait_seconds=int(cpu_control_plane.rolling_update.wait_interval_seconds),
            alarm_names=(
                str(cpu_control_plane.rollback_alarms.capacity),
                str(cpu_control_plane.rollback_alarms.not_restored),
            ),
        )

    def alarm_arn(self, name: str) -> str:
        return (
            f"arn:{self.partition}:cloudwatch:{self.region}:{self.account}:alarm:{name}"
        )


@dataclass
class Clock:
    """Waits and time, injectable for tests."""

    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def _stamp(clock: Clock) -> str:
    return clock.now().astimezone(UTC).isoformat().replace("+00:00", "Z")


def _wait(
    clock: Clock,
    condition: Callable[[], Any],
    *,
    timeout_seconds: float,
    description: str,
) -> Any:
    """``condition()`` until it answers something truthy; raise on timeout."""

    started = clock.monotonic()
    while True:
        answer = condition()
        if answer:
            return answer
        elapsed = clock.monotonic() - started
        if elapsed >= timeout_seconds:
            raise BootstrapError(
                f"{description}: not reached within {timeout_seconds:.0f} s"
            )
        clock.sleep(min(INTERVAL_SECONDS, max(timeout_seconds - elapsed, 1.0)))


# --- the journal --------------------------------------------------------------------


def journal_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / STATE_PATH


def load_journal(state_dir: Path) -> dict[str, Any]:
    """The journal, or an empty one; a damaged file is never silently dropped."""

    path = journal_path(state_dir)
    if not path.is_file():
        return {"schema_version": 1, "groups": {}, "alarms": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BootstrapError(f"{path} is not a CPU rolling-update journal")
    value.setdefault("groups", {})
    value.setdefault("alarms", {})
    return value


LAST_RUN_PATH = Path("cpu-rolling-update") / "last-run.json"


def record_last_run(state_dir: Path, result: Mapping[str, Any]) -> None:
    """What the deploy's step did last, for the driver's summary."""

    path = state_dir.expanduser().resolve() / LAST_RUN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    write_json_atomic(path, dict(result))


def load_last_run(state_dir: Path, *, since: str) -> dict[str, Any]:
    """The step's record written at or after ``since``, else ``UNKNOWN``."""

    path = state_dir.expanduser().resolve() / LAST_RUN_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "UNKNOWN", "warnings": []}

    def moment(text: object) -> datetime:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))

    try:
        current = isinstance(value, dict) and moment(value.get("at")) >= moment(since)
    except ValueError:
        current = False
    return value if current else {"status": "UNKNOWN", "warnings": []}


def _save(state_dir: Path, journal: dict[str, Any]) -> None:
    path = journal_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    write_json_atomic(path, journal)


# --- reads -------------------------------------------------------------------------


def _describe(runner: CommandRunner, target: Target) -> dict[str, Any]:
    return runner.aws_json(
        target.region, "sagemaker", "describe-cluster", "--cluster-name", target.cluster
    )


def _groups(described: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(group.get("InstanceGroupName") or ""): group
        for group in described.get("InstanceGroups") or ()
        if isinstance(group, Mapping)
    }


def update_in_progress(
    described: Mapping[str, Any], group: Mapping[str, Any]
) -> str | None:
    """Why an AMI update is in flight on ``group``, or ``None``."""

    software = str(group.get("SoftwareUpdateStatus") or "")
    status = str(group.get("Status") or "")
    cluster = str(described.get("ClusterStatus") or "")
    if software in UPDATING_SOFTWARE:
        return f"SoftwareUpdateStatus is {software} (an AMI update is in flight)"
    if status in UPDATING_GROUP:
        return f"the instance group is {status}"
    if cluster in UPDATING_GROUP:
        return f"the cluster is {cluster}"
    return None


def read_api_nlb(runner: CommandRunner, target: Target) -> dict[str, Any]:
    """The API NLB as CloudWatch names it, plus its one target group's ARN."""

    balancers = runner.aws_json(
        target.region, "elbv2", "describe-load-balancers", "--names", target.nlb_name
    )
    arn = rollback.load_balancer_arn(balancers)
    groups = runner.aws_json(
        target.region, "elbv2", "describe-target-groups", "--load-balancer-arn", arn
    )
    arns = [
        str(item.get("TargetGroupArn") or "")
        for item in groups.get("TargetGroups") or ()
        if isinstance(item, Mapping)
    ]
    if len(arns) != 1:
        raise BootstrapError(
            f"the API NLB {target.nlb_name} has {len(arns)} target groups; the "
            "rollback alarms need exactly one"
        )
    return {
        **rollback.api_nlb(target.nlb_name, balancers, groups),
        "target_group_arn": arns[0],
    }


def _minute(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def wait_for_full_capacity(
    runner: CommandRunner, target: Target, nlb: Mapping[str, Any], *, clock: Clock
) -> dict[str, Any]:
    """Every api-ha replica healthy in the target group, then one whole minute
    after that whose ``HealthyHostCount`` Maximum is N: an alarm put now is
    evaluated on data that is already OK, so the installing control plane
    never trips it."""

    def healthy() -> datetime | None:
        health = runner.aws_json(
            target.region,
            "elbv2",
            "describe-target-health",
            "--target-group-arn",
            str(nlb["target_group_arn"]),
        )
        states = [
            str((item.get("TargetHealth") or {}).get("State") or "")
            for item in health.get("TargetHealthDescriptions") or ()
            if isinstance(item, Mapping)
        ]
        ok = len(states) >= target.replicas and set(states) == {"healthy"}
        return clock.now() if ok else None

    since: datetime = _wait(
        clock,
        healthy,
        timeout_seconds=CAPACITY_TIMEOUT_SECONDS,
        description=f"{target.replicas} healthy API targets",
    )

    def full_minute() -> datetime | None:
        now = clock.now()
        if now < _minute(since) + timedelta(minutes=1):
            return None  # no whole minute since the targets turned healthy yet
        statistics = runner.aws_json(
            target.region,
            "cloudwatch",
            "get-metric-statistics",
            "--namespace",
            "AWS/NetworkELB",
            "--metric-name",
            "HealthyHostCount",
            "--dimensions",
            f"Name=TargetGroup,Value={nlb['target_groups'][0]}",
            f"Name=LoadBalancer,Value={nlb['load_balancer']}",
            "--statistics",
            rollback.RECOMMENDED_STATISTIC,
            "--period",
            "60",
            "--start-time",
            _minute(since).isoformat(),
            "--end-time",
            now.isoformat(),
        )
        for point in statistics.get("Datapoints") or ():
            if not isinstance(point, Mapping):
                continue
            stamp = datetime.fromisoformat(
                str(point.get("Timestamp") or "").replace("Z", "+00:00")
            )
            value = point.get(rollback.RECOMMENDED_STATISTIC)
            if (
                stamp >= _minute(since)
                and stamp + timedelta(minutes=1) <= now
                and isinstance(value, (int, float))
                and value >= target.replicas
            ):
                return stamp
        return None

    minute: datetime = _wait(
        clock,
        full_minute,
        timeout_seconds=CAPACITY_TIMEOUT_SECONDS,
        description=(
            f"one whole minute of HealthyHostCount {rollback.RECOMMENDED_STATISTIC} "
            f"{target.replicas}"
        ),
    )
    return {"healthy_since": since.isoformat(), "full_minute": minute.isoformat()}


# --- part A: the alarms ---------------------------------------------------------------


def _on_this_nlb(alarm: Mapping[str, Any], target: Target) -> bool:
    dimensions = {
        str(item.get("Name")): str(item.get("Value"))
        for item in alarm.get("Dimensions") or ()
        if isinstance(item, Mapping)
    }
    return dimensions.get("LoadBalancer", "").startswith(f"net/{target.nlb_name}/")


def _owned(
    runner: CommandRunner, target: Target, name: str, existing: Mapping[str, Any]
) -> tuple[str, bool]:
    """``(arn, carries the site tag)``; raises for another site's alarm or a
    foreign untagged one."""

    arn = str(existing.get("AlarmArn") or target.alarm_arn(name))
    tags = runner.aws_json(
        target.region, "cloudwatch", "list-tags-for-resource", "--resource-arn", arn
    ).get("Tags")
    try:
        ours = assert_site_tag(
            tags,
            site_id=target.site_id,
            description=f"CloudWatch alarm {name}",
            allow_missing=True,
        )
    except BootstrapError as exc:
        raise BootstrapError(
            f"{exc}: two sites in one account and Region need their own alarm names "
            "(spec.cpuControlPlane.rollbackAlarms in admin-config.yaml)"
        ) from exc
    if not ours and not _on_this_nlb(existing, target):
        raise BootstrapError(
            f"CloudWatch alarm {name} exists without {SITE_TAG_KEY} and does not "
            f"watch this site's API NLB {target.nlb_name}; refusing to take it over "
            "(name the pair in spec.cpuControlPlane.rollbackAlarms)"
        )
    return arn, ours


def alarms_to_put(
    runner: CommandRunner,
    target: Target,
    nlb: Mapping[str, Any],
    *,
    wait_seconds: int | None,
) -> list[str]:
    """The alarms of the pair that are missing or differ from their definition
    (what ``ensure_alarms`` would put); a composite alarm of that name counts,
    so ``ensure_alarms`` gets to refuse it."""

    pair = rollback.alarm_definitions(
        nlb=nlb,
        replicas=target.replicas,
        wait_seconds=wait_seconds,
        names=target.alarm_names,
    )
    document = runner.aws_json(
        target.region, *rollback.describe_alarms_arguments(list(target.alarm_names))
    )
    metric = {
        str(item["AlarmName"]): item
        for item in document.get("MetricAlarms") or ()
        if isinstance(item, Mapping) and item.get("AlarmName")
    }
    return [
        str(definition["AlarmName"])
        for definition in pair["definitions"]
        if str(definition["AlarmName"]) not in metric
        or not rollback.definition_matches(
            metric[str(definition["AlarmName"])], definition
        )
    ]


def ensure_alarms(
    runner: CommandRunner,
    target: Target,
    nlb: Mapping[str, Any],
    *,
    wait_seconds: int | None,
) -> dict[str, Any]:
    """Create, re-point or keep the pair: ``{"alarms": [...], "errors": [...]}``.

    ``action`` per alarm: ``created``, ``updated`` (re-pointed or redefined),
    ``unchanged`` (identical: not put), ``refused`` (another site's, or a
    foreign untagged one), ``failed`` (a write AWS refused; the error is in
    ``errors``); ``tagged`` when the site tag was added. An alarm carries its
    ``arn`` only when it is the site's, which is what the journal records."""

    pair = rollback.alarm_definitions(
        nlb=nlb,
        replicas=target.replicas,
        wait_seconds=wait_seconds,
        names=target.alarm_names,
    )
    errors = [pair["note"]] if pair["note"] else []
    document = runner.aws_json(
        target.region, *rollback.describe_alarms_arguments(list(target.alarm_names))
    )
    metric = {
        str(item["AlarmName"]): item
        for item in document.get("MetricAlarms") or ()
        if isinstance(item, Mapping) and item.get("AlarmName")
    }
    composite = {
        str(item.get("AlarmName"))
        for item in document.get("CompositeAlarms") or ()
        if isinstance(item, Mapping)
    }
    site_tag = f"Key={SITE_TAG_KEY},Value={target.site_id}"
    results = []
    for definition in pair["definitions"]:
        name = str(definition["AlarmName"])
        put = [
            "aws",
            "cloudwatch",
            "put-metric-alarm",
            "--region",
            target.region,
            *rollback.put_metric_alarm_arguments(definition),
        ]
        existing = metric.get(name)
        if name in composite:
            errors.append(f"{name} is a composite alarm; refusing to redefine it")
            results.append({"name": name, "action": "refused"})
            continue
        if existing is None:
            # Captured, so a refusal carries the AWS message into the warning.
            try:
                runner.run([*put, "--tags", site_tag], mutate=True)
            except BootstrapError as exc:
                errors.append(f"{name}: {exc}")
                results.append({"name": name, "action": "failed"})
                continue
            results.append(
                {"name": name, "arn": target.alarm_arn(name), "action": "created"}
            )
            continue
        try:
            arn, ours = _owned(runner, target, name, existing)
        except BootstrapError as exc:
            errors.append(str(exc))
            results.append({"name": name, "action": "refused"})
            continue
        result: dict[str, Any] = {"name": name, "arn": arn, "action": "unchanged"}
        try:
            if not rollback.definition_matches(existing, definition):
                runner.run(put, mutate=True)
                result["action"] = "updated"
            if not ours:
                # PutMetricAlarm ignores --tags on an existing alarm.
                runner.run(
                    [
                        "aws",
                        "cloudwatch",
                        "tag-resource",
                        "--region",
                        target.region,
                        "--resource-arn",
                        arn,
                        "--tags",
                        site_tag,
                    ],
                    mutate=True,
                )
                result["tagged"] = True
        except BootstrapError as exc:
            errors.append(f"{name}: {exc}")
            # Still the site's when it carried the tag: the ledger keeps it.
            results.append(
                {**result, "action": "failed"}
                if ours
                else {"name": name, "action": "failed"}
            )
            continue
        results.append(result)
    return {"alarms": results, "errors": errors}


def alarm_states(runner: CommandRunner, target: Target) -> dict[str, str]:
    document = runner.aws_json(
        target.region, *rollback.describe_alarms_arguments(list(target.alarm_names))
    )
    return {
        str(item["AlarmName"]): str(item.get("StateValue") or "")
        for item in document.get("MetricAlarms") or ()
        if isinstance(item, Mapping) and item.get("AlarmName")
    }


def _record_alarms(
    state_dir: Path, results: Sequence[Mapping[str, Any]], *, clock: Clock
) -> None:
    """Every alarm the product put stays recorded -- a renamed pair keeps the
    old names, so uninstall still deletes them."""

    journal = load_journal(state_dir)
    configured = {str(item["name"]) for item in results}
    for name, entry in journal["alarms"].items():
        entry["configured"] = name in configured
    for item in results:
        if item.get("arn"):
            journal["alarms"][str(item["name"])] = {
                "name": str(item["name"]),
                "arn": str(item["arn"]),
                "action": str(item["action"]),
                "configured": True,
                "updated_at": _stamp(clock),
            }
    _save(state_dir, journal)


# --- part B: the schedule -------------------------------------------------------------


def _definition(
    group: Mapping[str, Any], schedule: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The full definition with ``schedule`` (``None``: the member left out)."""

    specification = hyperpod_instance_group_specification(
        group, labels=hyperpod_current_labels(group)
    )
    specification.pop("ScheduledUpdateConfig", None)
    if schedule is not None:
        specification["ScheduledUpdateConfig"] = dict(schedule)
    return specification


def _update_arguments(target: Target, definition: Mapping[str, Any]) -> list[str]:
    return [
        "sagemaker",
        "update-cluster",
        "--cluster-name",
        target.cluster,
        "--instance-groups",
        json.dumps([definition], separators=(",", ":")),
    ]


def _write(
    runner: CommandRunner,
    target: Target,
    group: Mapping[str, Any],
    schedule: Mapping[str, Any] | None,
) -> None:
    runner.aws_json(
        target.region,
        *_update_arguments(target, _definition(group, schedule)),
        mutate=True,
    )


def _schedule_of(
    runner: CommandRunner, target: Target, name: str
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, bool]:
    """``(group, its schedule, cluster and group InService)``."""

    described = _describe(runner, target)
    group = _groups(described).get(name) or {}
    settled = (
        described.get("ClusterStatus") == "InService"
        and group.get("Status") == "InService"
    )
    return group, group.get("ScheduledUpdateConfig"), settled


def _matches(
    schedule: Mapping[str, Any] | None, wanted: Mapping[str, Any] | None
) -> bool:
    if schedule is None or wanted is None:
        return schedule is None and wanted is None
    return rollback.same_schedule(schedule, wanted)


def _converge(
    runner: CommandRunner,
    target: Target,
    name: str,
    wanted: Mapping[str, Any] | None,
    *,
    clock: Clock,
) -> Mapping[str, Any]:
    """Re-describe until the cluster and ``name`` are InService with
    ``wanted`` as the group's schedule (``None``: no schedule)."""

    def settled() -> Mapping[str, Any] | None:
        group, schedule, steady = _schedule_of(runner, target, name)
        return group if steady and _matches(schedule, wanted) else None

    converged: Mapping[str, Any] = _wait(
        clock,
        settled,
        timeout_seconds=CONVERGE_TIMEOUT_SECONDS,
        description=(
            f"instance group {name} "
            + ("without a schedule" if wanted is None else "with its new schedule")
        ),
    )
    return converged


def _refusal(
    described: Mapping[str, Any],
    group: Mapping[str, Any],
    schedule: Mapping[str, Any],
    target: Target,
    clock: Clock,
) -> str | None:
    """Why the schedule step must wait: an update in flight, a write
    refusal, or ``schedule``'s next window too close (or not placeable in
    time) -- the group's own schedule, or the recorded one it lost."""

    reason = update_in_progress(described, group) or hyperpod_write_refusal(
        str(described.get("ClusterStatus") or ""), group, allow_label_release=False
    )
    if reason:
        return reason
    wait = (schedule.get("DeploymentConfig") or {}).get("WaitIntervalInSeconds")
    count = group.get("CurrentCount")
    now = clock.now()
    try:
        window = current_or_next_window(
            str(schedule.get("ScheduleExpression") or ""),
            now=now,
            instances=count if type(count) is int else 1,
            wait_seconds=max(target.wait_seconds, wait if type(wait) is int else 0),
        )
    except UnsupportedSchedule as exc:
        return f"the schedule cannot be placed in time ({exc})"
    if window is not None and window.start - WINDOW_MARGIN <= now:
        return (
            f"the next scheduled update window starts {window.start.isoformat()} "
            f"(until about {window.end.isoformat()}), within {WINDOW_MARGIN} of now"
        )
    return None


def _target_schedule(schedule: Mapping[str, Any], target: Target) -> dict[str, Any]:
    return {
        **schedule,
        "DeploymentConfig": rollback.deployment_config(
            schedule.get("DeploymentConfig"),
            wait_seconds=target.wait_seconds,
            alarms=target.alarm_names,
        ),
    }


def _create(
    runner: CommandRunner,
    target: Target,
    name: str,
    group: Mapping[str, Any],
    wanted: Mapping[str, Any],
    *,
    clock: Clock,
) -> str:
    """Create ``wanted`` on a group without a schedule, retried once; ``""``
    when it applied, else what the group was left with, loudly."""

    error = ""
    for _attempt in range(2):
        try:
            _write(runner, target, group, wanted)
            _converge(runner, target, name, wanted, clock=clock)
            return ""
        except BootstrapError as exc:
            error = str(exc)
        group, schedule, _steady = _schedule_of(runner, target, name)
        if _matches(schedule, wanted):
            return ""
        if schedule is not None:
            return (
                f"the CPU instance group {name} carries ScheduledUpdateConfig "
                f"{json.dumps(schedule, sort_keys=True)}, not the rolling one "
                f"({error}); nothing more was written"
            )
        group = _converge(runner, target, name, None, clock=clock)
    command = shlex.join(
        [
            "aws",
            *_update_arguments(target, _definition(group, wanted)),
            "--region",
            target.region,
        ]
    )
    return (
        f"the CPU instance group {name} of {target.cluster} now has NO "
        "ScheduledUpdateConfig -- no automatic patching. That is safe, but it must "
        f"be fixed: the create failed twice ({error}). Run, in a maintenance "
        f"window: {command} -- or re-run the deploy, which creates it from the "
        "journal"
    )


def apply_schedule(
    runner: CommandRunner,
    target: Target,
    state_dir: Path,
    *,
    clock: Clock,
) -> list[dict[str, Any]]:
    """Part B for every instance group of the CPU cluster; one record each
    (``outcome``: noop, no-schedule, applied, deferred, failed)."""

    journal = load_journal(state_dir)
    described = _describe(runner, target)
    results: list[dict[str, Any]] = []
    for name, group in _groups(described).items():
        entry = journal["groups"].setdefault(name, {})
        current = group.get("ScheduledUpdateConfig")
        recorded = entry.get("pre_install")
        recorded_schedule = (
            recorded.get("scheduled_update_config")
            if isinstance(recorded, Mapping)
            else None
        )
        # A group this step (or a restore) left without a schedule -- deleted,
        # then the create failed or the run stopped -- is created again from
        # the record; one that lost it otherwise is the customer's choice.
        resumable = recorded_schedule is not None and bool(entry.get(AWAITING_CREATE))
        if current is None and not resumable:
            results.append({"group": name, "outcome": "no-schedule"})
            continue
        base = current or recorded_schedule or {}
        wanted = _target_schedule(base, target)
        if current is not None and rollback.same_schedule(current, wanted):
            entry.update(
                phase="verified",
                reason="",
                updated_at=_stamp(clock),
                **{AWAITING_CREATE: False},
            )
            results.append({"group": name, "outcome": "noop"})
            continue
        reason = _refusal(described, group, base, target, clock)
        if reason:
            entry.update(phase="deferred", reason=reason, updated_at=_stamp(clock))
            _save(state_dir, journal)
            results.append({"group": name, "outcome": "deferred", "reason": reason})
            continue
        if recorded is None:
            entry["pre_install"] = {
                "recorded_at": _stamp(clock),
                "scheduled_update_config": current,
            }
        entry.update(
            target=wanted,
            phase="recorded",
            updated_at=_stamp(clock),
            **{AWAITING_CREATE: True},
        )
        _save(state_dir, journal)
        if current is not None:
            _write(runner, target, group, None)
            group = _converge(runner, target, name, None, clock=clock)
            entry.update(phase="deleted", updated_at=_stamp(clock))
            _save(state_dir, journal)
        failure = _create(runner, target, name, group, wanted, clock=clock)
        if failure:
            entry.update(phase="failed", reason=failure, updated_at=_stamp(clock))
            _save(state_dir, journal)
            results.append({"group": name, "outcome": "failed", "reason": failure})
            continue
        entry.update(
            phase="verified",
            reason="",
            updated_at=_stamp(clock),
            **{AWAITING_CREATE: False},
        )
        _save(state_dir, journal)
        results.append({"group": name, "outcome": "applied", "schedule": wanted})
    _save(state_dir, journal)
    return results


def restore_schedule(
    runner: CommandRunner,
    target: Target,
    state_dir: Path,
    *,
    recorded: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
) -> list[dict[str, Any]]:
    """Put every group back to its recorded pre-install schedule: the
    journal's, else ``recorded`` (group -> ``ScheduledUpdateConfig``, the
    installation ledger's copy). Refused -- raised, nothing written -- while an
    update is in flight or a write would be refused; delete-then-create like
    the step."""

    clock = clock or Clock()
    journal = load_journal(state_dir)
    wanted_by_group: dict[str, Any] = dict(recorded or {})
    for name, entry in journal["groups"].items():
        if isinstance(entry.get("pre_install"), Mapping):
            wanted_by_group[name] = entry["pre_install"].get("scheduled_update_config")
    described = _describe(runner, target)
    groups = _groups(described)
    results: list[dict[str, Any]] = []
    for name, wanted in wanted_by_group.items():
        group = groups.get(name)
        if group is None:
            results.append({"group": name, "outcome": "group-gone"})
            continue
        current = group.get("ScheduledUpdateConfig")
        entry = journal["groups"].setdefault(name, {})
        if _matches(current, wanted):
            entry.update(
                phase="restored",
                reason="",
                updated_at=_stamp(clock),
                **{AWAITING_CREATE: False},
            )
            results.append({"group": name, "outcome": "unchanged"})
            continue
        reason = update_in_progress(described, group) or hyperpod_write_refusal(
            str(described.get("ClusterStatus") or ""), group, allow_label_release=False
        )
        if reason:
            raise BootstrapError(
                "cannot restore the pre-install ScheduledUpdateConfig of CPU instance "
                f"group {name}: {reason}; nothing was written -- rerun once the "
                "update has finished"
            )
        entry.update(
            phase="restoring",
            updated_at=_stamp(clock),
            **{AWAITING_CREATE: wanted is not None},
        )
        _save(state_dir, journal)
        if current is not None:
            _write(runner, target, group, None)
            group = _converge(runner, target, name, None, clock=clock)
        if wanted is not None:
            _write(runner, target, group, wanted)
            _converge(runner, target, name, wanted, clock=clock)
        entry.update(
            phase="restored",
            reason="",
            updated_at=_stamp(clock),
            **{AWAITING_CREATE: False},
        )
        _save(state_dir, journal)
        results.append({"group": name, "outcome": "restored", "schedule": wanted})
    _save(state_dir, journal)
    return results


# --- the deploy's last step ------------------------------------------------------------


def _summary(target: Target, outcome: str) -> str:
    return (
        f"CPU rolling update ({target.mode}): {outcome}. Change it with "
        "spec.cpuControlPlane.rollingUpdate (mode rolling | unchanged, "
        "waitIntervalSeconds) in admin-config.yaml and gpu-fault-admin config "
        "--state-dir <state dir> (administrator-operations §5.0.5)"
    )


def _current_wait(described: Mapping[str, Any]) -> int | None:
    for group in _groups(described).values():
        deployment = (group.get("ScheduledUpdateConfig") or {}).get("DeploymentConfig")
        wait = (deployment or {}).get("WaitIntervalInSeconds")
        if type(wait) is int:
            return wait
    return None


def _alarms_ok(
    runner: CommandRunner, target: Target, clock: Clock, names: Sequence[str]
) -> dict[str, str]:
    """Wait until every alarm in ``names`` (what the step put or kept) is OK."""

    def all_ok() -> dict[str, str] | None:
        states = alarm_states(runner, target)
        ok = set(states) >= set(names) and {states[name] for name in names} <= {"OK"}
        return states if ok else None

    states: dict[str, str] = _wait(
        clock,
        all_ok,
        timeout_seconds=ALARM_OK_TIMEOUT_SECONDS,
        description=f"rollback alarms {', '.join(names)} OK",
    )
    return states


def _keep_or_restore(
    runner: CommandRunner,
    target: Target,
    state_dir: Path,
    record: dict[str, Any],
    warnings: list[str],
    clock: Clock,
) -> dict[str, Any]:
    """``unchanged``: a group this step changed goes back to its recorded
    pre-install schedule (rolling -> unchanged); any other is left as found."""

    journal = load_journal(state_dir)
    changed = [
        name
        for name, entry in journal["groups"].items()
        if isinstance(entry.get("pre_install"), Mapping)
        and entry.get("phase") != "restored"
    ]
    if not changed:
        record.update(status="DONE", schedule=[], warnings=warnings)
        record["summary"] = _summary(target, "the group's schedule is left as found")
        return record
    try:
        steps = restore_schedule(runner, target, state_dir, clock=clock)
        status = "DONE"
    except BootstrapError as exc:
        # Journaled, so status names it: a restore stopped after its delete is
        # failed (and resumed by the next run), one refused before it deferred.
        journal = load_journal(state_dir)
        steps = []
        for name in changed:
            entry = journal["groups"].setdefault(name, {})
            outcome = "failed" if entry.get(AWAITING_CREATE) else "deferred"
            entry.update(phase=outcome, reason=str(exc), updated_at=_stamp(clock))
            steps.append({"group": name, "outcome": outcome, "reason": str(exc)})
        _save(state_dir, journal)
        warnings.append(f"CPU schedule restore stopped: {exc}")
        status = (
            "FAILED"
            if any(item["outcome"] == "failed" for item in steps)
            else "DEFERRED"
        )
    record.update(status=status, schedule=steps, warnings=warnings)
    record["summary"] = _summary(
        target,
        "restored the recorded pre-install schedule"
        if status == "DONE"
        else f"the restore of the pre-install schedule is {status.lower()}",
    )
    return record


def finish_deploy(
    release_config: Mapping[str, Any],
    cpu_control_plane: Any,
    state_dir: Path,
    *,
    runner: CommandRunner | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """The deploy's last step: the alarms, then the schedule. Never raises:
    ``status`` is ``DONE``, ``DEFERRED`` or ``FAILED``, with ``warnings`` and a
    one-line ``summary`` that names how to change it."""

    active = runner or CommandRunner()
    clock = clock or Clock()
    record: dict[str, Any] = {"at": _stamp(clock)}
    warnings: list[str] = []
    try:
        target = Target.of(release_config, cpu_control_plane)
        record["mode"] = target.mode
        return _finish(active, target, state_dir, record, warnings, clock)
    except Exception as exc:  # noqa: BLE001 - the step never fails the deploy
        warnings.append(f"stopped: {type(exc).__name__}: {exc}")
        record.setdefault("schedule", [])
        record.update(
            status="FAILED",
            warnings=warnings,
            summary=(
                "CPU rolling update: stopped on an unexpected error; the "
                "cpu_rolling_update check of status --full shows what is left "
                "(administrator-operations §5.0.5)"
            ),
        )
        return record


def _finish(
    active: CommandRunner,
    target: Target,
    state_dir: Path,
    record: dict[str, Any],
    warnings: list[str],
    clock: Clock,
) -> dict[str, Any]:
    try:
        nlb = read_api_nlb(active, target)
        wait = (
            target.wait_seconds
            if target.mode == ROLLING
            else _current_wait(_describe(active, target))
        )
        # The wait guards a put; a pair that already matches is not put, so a
        # NOOP deploy does not sit out a minute for nothing.
        if alarms_to_put(active, target, nlb, wait_seconds=wait):
            record["capacity"] = wait_for_full_capacity(
                active, target, nlb, clock=clock
            )
        else:
            record["capacity"] = {"skipped": "both alarms match: nothing is put"}
        alarms = ensure_alarms(active, target, nlb, wait_seconds=wait)
        record["alarms"] = alarms
        warnings.extend(alarms["errors"])
        _record_alarms(state_dir, alarms["alarms"], clock=clock)
        if any(item["action"] == "refused" for item in alarms["alarms"]):
            raise BootstrapError("an alarm of the pair is not this site's")
        if any(item["action"] == "failed" for item in alarms["alarms"]):
            raise BootstrapError("AWS refused a write of the rollback alarms")
        if not alarms["alarms"]:
            raise BootstrapError("no rollback alarm could be defined")
        record["alarm_states"] = _alarms_ok(
            active, target, clock, [str(item["name"]) for item in alarms["alarms"]]
        )
    except BootstrapError as exc:
        warnings.append(f"rollback alarms: {exc}")
        record.update(status="FAILED", schedule=[], warnings=warnings)
        record["summary"] = _summary(
            target, "the rollback alarms are not ready, so the schedule was not touched"
        )
        return record
    if target.mode != ROLLING:
        return _keep_or_restore(active, target, state_dir, record, warnings, clock)
    try:
        steps = apply_schedule(active, target, state_dir, clock=clock)
    except BootstrapError as exc:
        steps = [{"group": "*", "outcome": "failed", "reason": str(exc)}]
    record["schedule"] = steps
    outcomes = {str(item["outcome"]) for item in steps}
    warnings.extend(
        f"CPU rolling schedule of {item['group']}: {item['outcome']}: "
        f"{item.get('reason')}"
        for item in steps
        if item["outcome"] in {"deferred", "failed"}
    )
    status = (
        "FAILED"
        if "failed" in outcomes
        else "DEFERRED"
        if "deferred" in outcomes
        else "DONE"
    )
    record.update(status=status, warnings=warnings)
    record["summary"] = _summary(
        target,
        ", ".join(f"{item['group']} {item['outcome']}" for item in steps)
        or "no instance group",
    )
    return record
