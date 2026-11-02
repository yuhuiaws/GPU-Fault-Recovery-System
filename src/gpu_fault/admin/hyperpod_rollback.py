"""Rolling HyperPod schedules the service accepts, and whether their rollback
alarms can still trip.

A rolling ``ScheduledUpdateConfig.DeploymentConfig`` written through
``UpdateCluster`` needs more than the API reference says. The reference marks
``RollingDeploymentPolicy.RollbackMaximumBatchSize`` and
``DeploymentConfiguration.AutoRollbackConfiguration`` "Required: No" (so does
the installed botocore model, which is why ``ParamValidator`` passed), yet the
service refused a rolling schedule three times on 2026-11-02
(``SERVICE_REFUSALS``): without the rollback batch, without a rollback alarm,
and as a change to an existing ``ScheduledUpdateConfig`` -- which it does not
modify in place: the existing one is deleted first (the definition without the
member) and the new one created by a second ``UpdateCluster``. The delete and
then the create were accepted the same day.

So ``hyperpod-scheduled-update --rolling`` writes ``deployment_config``:
``MaximumBatchSize`` and ``RollbackMaximumBatchSize`` both ``ROLLING_BATCH``
(one instance at a time), the wait, and ``AutoRollbackConfiguration`` naming
the operator's ``--rollback-alarm`` alarms -- never one this module invents.
An alarm only protects the roll while it can go to ALARM: the helper and the
CPU control plane's advisory read each named alarm (``alarm_records``) and
report one that is missing, has no data, or watches a load balancer other than
the current API NLB (``alarm_problems``). The recommended pair watches the
Maximum of the API NLB's ``HealthyHostCount`` (``suggested_alarms``,
``RECOMMENDED_STATISTIC``): on 2026-11-01 it was the one control-plane metric
that saw the all-at-once re-image (3 healthy targets to 0), while Container
Insights' node counts stayed 3 with no data points for the minutes in
question.

Nothing here reads or writes by itself: callers hand in the documents.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "ALARM_DETAILS_EVIDENCE",
    "AMI_UPDATE_EVIDENCE",
    "DEPLOYMENT_CONFIGURATION_EVIDENCE",
    "DESCRIBE_ALARMS_EVIDENCE",
    "MAX_ROLLBACK_ALARMS",
    "NLB_HEALTHY_HOSTS_EVIDENCE",
    "NLB_HEALTH_CHECKS_EVIDENCE",
    "RECOMMENDED_STATISTIC",
    "ROLLBACK_EVIDENCE",
    "ROLLING_BATCH",
    "ROLLING_POLICY_EVIDENCE",
    "SERVICE_REFUSALS",
    "SUGGESTED_ALARM_NAMES",
    "alarm_name_refusal",
    "alarm_problems",
    "alarm_records",
    "alarm_summary",
    "api_nlb",
    "deployment_config",
    "describe_alarms_arguments",
    "load_balancer_arn",
    "read_rollback_alarms",
    "rollback_alarm_names",
    "same_schedule",
    "suggested_alarms",
]

#: The batch ``--rolling`` writes for the update and for its rollback.
ROLLING_BATCH = {"Type": "INSTANCE_COUNT", "Value": 1}
#: ``AutoRollbackConfiguration``: 1 to 10 alarms; ``AlarmName`` 1 to 255.
MAX_ROLLBACK_ALARMS = 10
MAX_ALARM_NAME = 255
UNKNOWN = "unknown"
_MAX_REASON = 300

_API = "https://docs.aws.amazon.com/sagemaker/latest/APIReference/"
_API_VERSION = "API version 2017-07-24"
#: The API reference pages print no date; this is their HTTP Last-Modified,
#: read 2026-11-02.
_API_DATE = "Last-Modified 2026-10-30"

ROLLING_POLICY_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": _API + "API_RollingDeploymentPolicy.html",
    "page": "RollingDeploymentPolicy",
    "section": "Contents",
    "version": _API_VERSION,
    "docs_date": _API_DATE,
    "quotes": (
        "The maximum amount of instances in the cluster that SageMaker can update "
        "at a time.",
        "The maximum amount of instances in the cluster that SageMaker can roll "
        "back at a time. Type: CapacitySizeConfig object Required: No",
    ),
}
DEPLOYMENT_CONFIGURATION_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": _API + "API_DeploymentConfiguration.html",
    "page": "DeploymentConfiguration",
    "section": "Contents",
    "version": _API_VERSION,
    "docs_date": _API_DATE,
    "quotes": (
        "An array that contains the alarms that SageMaker monitors to know whether "
        "to roll back the AMI update. Type: Array of AlarmDetails objects Array "
        "Members: Minimum number of 1 item. Maximum number of 10 items. Required: No",
        "The duration in seconds that SageMaker waits before updating more "
        "instances in the cluster.",
    ),
}
ALARM_DETAILS_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": _API + "API_AlarmDetails.html",
    "page": "AlarmDetails",
    "section": "Contents",
    "version": _API_VERSION,
    "docs_date": _API_DATE,
    "quotes": (
        "The details of the alarm to monitor during the AMI update.",
        "The name of the alarm. Type: String Length Constraints: Minimum length of "
        "1. Maximum length of 255.",
    ),
}
#: What the rollback alarms do, and what they need from the cluster's role.
AMI_UPDATE_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/sagemaker/latest/dg/"
        "sagemaker-hyperpod-release-ami-update.html"
    ),
    "page": "Update your AMI version in your SageMaker HyperPod cluster",
    "section": "introduction",
    "docs_date": "Last-Modified 2026-10-31",
    "quotes": (
        "If you choose to update in batches, you can also include a list of alarms "
        "and metrics. During the wait interval, SageMaker AI observes these metrics "
        "and if any exceed their threshold, the corresponding alarm goes into the "
        "ALARM state, and SageMaker AI rolls back the AMI update.",
        "To utilize automatic rollbacks, your IAM execution role must have the "
        "permission `cloudwatch:DescribeAlarms`.",
    ),
}
#: The metric the recommended alarms watch, and its two dimensions.
NLB_HEALTHY_HOSTS_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/elasticloadbalancing/latest/network/"
        "load-balancer-cloudwatch-metrics.html"
    ),
    "page": "CloudWatch metrics for your Network Load Balancer",
    "sections": (
        "Network Load Balancer metrics",
        "Metric dimensions for Network Load Balancers",
        "Statistics for Network Load Balancer metrics",
    ),
    "docs_date": "Last-Modified 2026-10-31",
    "quotes": (
        "If there are requests flowing through the load balancer, Elastic Load "
        "Balancing measures and sends its metrics in 60-second intervals.",
        "The number of targets that are considered healthy.",
        "Reporting criteria: Reported if there are registered targets.",
        "Statistics: The most useful statistics are `Maximum` and `Minimum`.",
        "net/load-balancer-name/1234567890123456 (the final portion of the load "
        "balancer ARN).",
        "targetgroup/target-group-name/1234567890123456 (the final portion of the "
        "target group ARN).",
        "The `Minimum` and `Maximum` statistics reflect the minimum and maximum "
        "values of the data points reported by the individual load balancer nodes "
        "in each sampling window. Increases in the maximum of `HealthyHostCount` "
        "correspond to decreases in the minimum of `UnHealthyHostCount`. It's "
        "recommended to monitor maximum `HealthyHostCount`, invoking the alarm when "
        "the maximum `HealthyHostCount` falls below your required minimum, or being "
        "`0`.",
    ),
}
#: Why one node's reading can dip below the others' for part of a minute.
NLB_HEALTH_CHECKS_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/elasticloadbalancing/latest/network/"
        "target-group-health-checks.html"
    ),
    "page": "Health checks for Network Load Balancer target groups",
    "section": "Health check settings",
    "docs_date": "Last-Modified 2026-10-31",
    "quotes": (
        "Health checks for a Network Load Balancer are distributed and use a "
        "consensus mechanism to determine target health. Therefore, targets receive "
        "more than the configured number of health checks.",
    ),
}
#: Why the read names both alarm types.
DESCRIBE_ALARMS_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/"
        "API_DescribeAlarms.html"
    ),
    "page": "DescribeAlarms",
    "section": "Request Parameters, AlarmTypes",
    "docs_date": "Last-Modified 2026-10-30",
    "quotes": (
        "If you omit this parameter, only metric alarms are returned, even if "
        "composite alarms or log alarms exist in the account.",
    ),
}
ROLLBACK_EVIDENCE = (
    ROLLING_POLICY_EVIDENCE,
    DEPLOYMENT_CONFIGURATION_EVIDENCE,
    ALARM_DETAILS_EVIDENCE,
    AMI_UPDATE_EVIDENCE,
    NLB_HEALTHY_HOSTS_EVIDENCE,
    NLB_HEALTH_CHECKS_EVIDENCE,
    DESCRIBE_ALARMS_EVIDENCE,
)
#: ``UpdateCluster`` ValidationException texts on the live CPU group, in order
#: (the third names the group; ``<instance group>`` here).
SERVICE_REFUSALS: tuple[dict[str, str], ...] = (
    {
        "at": "2026-11-02T00:16:46Z",
        "member": "RollbackMaximumBatchSize",
        "text": "RollingUpdatePolicy is missing the RollbackMaximumBatchSize "
        "parameter.",
    },
    {
        "at": "2026-11-02T00:19:01Z",
        "member": "AutoRollbackConfiguration",
        "text": "AutoRollbackConfiguration is required. Add a list of alarms and "
        "try again.",
    },
    {
        "at": "2026-11-02T00:30:16Z",
        "member": "ScheduledUpdateConfig",
        "text": "You can't modify the existing ScheduledUpdateConfig for an instance "
        "group. To update the configuration, first delete the existing "
        "configuration for the <instance group> instance group, and then create a "
        "new configuration.",
    },
)

# --- the DeploymentConfig ---------------------------------------------------------


def deployment_config(
    base: Mapping[str, Any] | None, *, wait_seconds: int, alarms: Sequence[str]
) -> dict[str, Any]:
    """``base`` (a current ``DeploymentConfig``, or none) made to roll one
    instance at a time: ``MaximumBatchSize`` and ``RollbackMaximumBatchSize``
    ``ROLLING_BATCH``, ``WaitIntervalInSeconds`` and an
    ``AutoRollbackConfiguration`` of ``alarms``, in their order. A member this
    module does not set stays as it is."""

    if not alarms:
        raise ValueError("a rolling DeploymentConfig needs at least one alarm")
    deployment = dict(base or {})
    policy = dict(deployment.get("RollingUpdatePolicy") or {})
    policy["MaximumBatchSize"] = dict(ROLLING_BATCH)
    policy["RollbackMaximumBatchSize"] = dict(ROLLING_BATCH)
    deployment["RollingUpdatePolicy"] = policy
    deployment["WaitIntervalInSeconds"] = wait_seconds
    deployment["AutoRollbackConfiguration"] = [{"AlarmName": name} for name in alarms]
    return deployment


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def rollback_alarm_names(schedule: object) -> list[str]:
    """The ``AlarmName`` of every ``AutoRollbackConfiguration`` entry of a
    ``ScheduledUpdateConfig``, in order."""

    deployment = _mapping(_mapping(schedule).get("DeploymentConfig"))
    entries = deployment.get("AutoRollbackConfiguration")
    return [
        str(entry["AlarmName"])
        for entry in (entries if isinstance(entries, list) else ())
        if isinstance(entry, Mapping) and entry.get("AlarmName")
    ]


def _comparable(schedule: object) -> object:
    """``schedule`` with its rollback alarms as a sorted name list: the
    service keeps the order written, but the alarms are a set."""

    if not isinstance(schedule, Mapping):
        return schedule
    deployment = schedule.get("DeploymentConfig")
    if not isinstance(deployment, Mapping):
        return dict(schedule)
    alarms = deployment.get("AutoRollbackConfiguration")
    deployment = dict(deployment)
    if isinstance(alarms, list):
        deployment["AutoRollbackConfiguration"] = sorted(
            json.dumps(item, sort_keys=True) for item in alarms
        )
    return {**schedule, "DeploymentConfig": deployment}


def same_schedule(left: object, right: object) -> bool:
    """Whether two ``ScheduledUpdateConfig`` documents say the same thing."""

    return _comparable(left) == _comparable(right)


def alarm_name_refusal(names: Sequence[str]) -> str | None:
    """Why ``names`` cannot be an ``AutoRollbackConfiguration``, or ``None``."""

    if not names:
        refusal = SERVICE_REFUSALS[1]
        return (
            "--rolling needs at least one --rollback-alarm NAME: UpdateCluster "
            "requires an AutoRollbackConfiguration (a list of CloudWatch alarms) "
            "with a rolling DeploymentConfig, although the API reference marks it "
            f'optional (ValidationException, {refusal["at"]}: "{refusal["text"]}"). '
            "Name the existing alarm(s) that go to ALARM when the cluster loses "
            "capacity during the roll; for the CPU control plane, the API NLB "
            "HealthyHostCount pair (administrator-operations §5.0.5). No alarm "
            "is invented."
        )
    if len(names) > MAX_ROLLBACK_ALARMS:
        return (
            f"--rollback-alarm is given {len(names)} times: "
            f"AutoRollbackConfiguration takes 1 to {MAX_ROLLBACK_ALARMS} alarms"
        )
    for name in names:
        if not name.strip() or len(name) > MAX_ALARM_NAME:
            return (
                f"--rollback-alarm {name!r} is not an alarm name (1 to "
                f"{MAX_ALARM_NAME} characters, not only blanks)"
            )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        return f"--rollback-alarm names {', '.join(repeated)} more than once"
    return None


# --- the alarm reads --------------------------------------------------------------


def describe_alarms_arguments(names: Sequence[str]) -> list[str]:
    """The ``aws cloudwatch describe-alarms`` arguments for ``names``: metric
    and composite alarms both (the read returns metric alarms only otherwise),
    so "missing" means missing."""

    return [
        "cloudwatch",
        "describe-alarms",
        "--alarm-names",
        *names,
        "--alarm-types",
        "MetricAlarm",
        "CompositeAlarm",
    ]


def _metric_record(alarm: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = {
        str(item.get("Name")): str(item.get("Value"))
        for item in alarm.get("Dimensions") or ()
        if isinstance(item, Mapping)
    }
    return {
        "kind": "metric",
        "state": alarm.get("StateValue"),
        "namespace": alarm.get("Namespace"),
        "metric": alarm.get("MetricName"),
        "metric_math": bool(alarm.get("Metrics")),
        "statistic": alarm.get("Statistic") or alarm.get("ExtendedStatistic"),
        "dimensions": dimensions,
        "period": alarm.get("Period"),
        "evaluation_periods": alarm.get("EvaluationPeriods"),
        "datapoints_to_alarm": alarm.get("DatapointsToAlarm"),
        "comparison": alarm.get("ComparisonOperator"),
        "threshold": alarm.get("Threshold"),
        "treat_missing_data": alarm.get("TreatMissingData"),
    }


def alarm_records(
    document: Mapping[str, Any], names: Sequence[str]
) -> list[dict[str, Any]]:
    """One record per name from a ``describe-alarms`` answer: ``found``,
    ``kind`` (``metric``/``composite``), ``state`` and, for a metric alarm,
    what it watches (``_metric_record``)."""

    found: dict[str, dict[str, Any]] = {}
    for item in document.get("MetricAlarms") or ():
        if isinstance(item, Mapping) and item.get("AlarmName"):
            found[str(item["AlarmName"])] = _metric_record(item)
    for item in document.get("CompositeAlarms") or ():
        if isinstance(item, Mapping) and item.get("AlarmName"):
            found[str(item["AlarmName"])] = {
                "kind": "composite",
                "state": item.get("StateValue"),
                "rule": item.get("AlarmRule"),
            }
    return [
        {"name": name, "found": name in found, **found.get(name, {})} for name in names
    ]


def load_balancer_arn(load_balancers: Mapping[str, Any]) -> str:
    """The one ``LoadBalancerArn`` of a ``describe-load-balancers --names``
    answer; anything else raises."""

    arns = [
        str(item.get("LoadBalancerArn") or "")
        for item in load_balancers.get("LoadBalancers") or ()
        if isinstance(item, Mapping)
    ]
    if len(arns) != 1 or ":loadbalancer/" not in arns[0]:
        raise ValueError(f"expected one load balancer, got {len(arns)}")
    return arns[0]


def api_nlb(
    name: str, load_balancers: Mapping[str, Any], target_groups: Mapping[str, Any]
) -> dict[str, Any]:
    """The API NLB as CloudWatch names it: ``load_balancer``
    (``net/<name>/<id>``, the final portion of its ARN) and ``target_groups``
    (``targetgroup/<name>/<id>``), from ``describe-load-balancers --names`` and
    ``describe-target-groups --load-balancer-arn``."""

    arn = load_balancer_arn(load_balancers)
    return {
        "name": name,
        "load_balancer": arn.split(":loadbalancer/", 1)[-1],
        "target_groups": sorted(
            str(item.get("TargetGroupArn") or "").rsplit(":", 1)[-1]
            for item in target_groups.get("TargetGroups") or ()
            if isinstance(item, Mapping)
        ),
    }


def _reason(exc: BaseException) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def read_rollback_alarms(
    names: Sequence[str],
    *,
    read_alarms: Callable[[list[str]], Mapping[str, Any]],
    read_nlb: Callable[[], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """``{"state": "read", "alarms", "api_nlb"}``: the records of ``names``,
    each with its ``problems``, against the API NLB ``read_nlb`` returns
    (``api_nlb``; ``None`` when not asked, ``{"state": "unknown", "error"}``
    when its read fails -- then no dimension is checked). A failed alarm read
    raises: the caller decides whether that refuses or is recorded."""

    alarms = alarm_records(read_alarms(list(names)), names)
    nlb: Mapping[str, Any] | None = None
    if read_nlb is not None:
        try:
            nlb = dict(read_nlb())
        except Exception as exc:  # noqa: BLE001 - recorded: no dimension check
            nlb = {"state": UNKNOWN, "error": _reason(exc)}
    for record in alarms:
        record["problems"] = alarm_problems(record, nlb)
    return {"state": "read", "alarms": alarms, "api_nlb": nlb}


# --- what the alarms can still do --------------------------------------------------


def _dimension_text(dimensions: object) -> str:
    return ", ".join(f"{key}={value}" for key, value in _mapping(dimensions).items())


def alarm_summary(record: Mapping[str, Any]) -> str:
    """One line naming what an alarm watches and its state."""

    if not record.get("found"):
        return "not found"
    if record.get("kind") == "composite":
        return f"composite alarm ({record.get('rule')}), {record.get('state')}"
    if record.get("metric_math"):
        return f"metric math, {record.get('state')}"
    return (
        f"{record.get('namespace')} {record.get('metric')} {record.get('statistic')} "
        f"{record.get('comparison')} {record.get('threshold')} for "
        f"{record.get('datapoints_to_alarm')} of {record.get('evaluation_periods')} "
        f"x {record.get('period')} s ({_dimension_text(record.get('dimensions'))}), "
        f"{record.get('state')}"
    )


def alarm_problems(
    record: Mapping[str, Any], nlb: Mapping[str, Any] | None
) -> list[str]:
    """Why ``record`` may not roll an update back, one phrase each: missing,
    no data, already in ALARM, or -- with the API NLB known -- watching
    something other than its healthy targets."""

    if not record.get("found"):
        return [
            "does not exist (describe-alarms returns no metric or composite alarm "
            "of that name): the update has nothing to roll back on"
        ]
    problems = []
    state = record.get("state")
    if state == "INSUFFICIENT_DATA":
        problems.append(
            "is INSUFFICIENT_DATA: with no data it cannot go to ALARM, so it rolls "
            "nothing back (a load balancer that is gone, a wrong dimension, or a "
            "metric nobody publishes)"
        )
    elif state == "ALARM":
        problems.append(
            "is in ALARM now: SageMaker rolls an update back when one of its alarms "
            "is in ALARM during a wait interval"
        )
    if record.get("kind") == "composite":
        problems.append("is a composite alarm: its member alarms are not checked")
        return problems
    if record.get("metric_math"):
        problems.append("uses metric math: what it watches is not checked")
        return problems
    lb = _mapping(nlb).get("load_balancer")
    groups = _mapping(nlb).get("target_groups")
    if not lb:
        return problems
    dimensions = _mapping(record.get("dimensions"))
    watched = (
        record.get("namespace") == "AWS/NetworkELB"
        and dimensions.get("LoadBalancer") == lb
        and dimensions.get("TargetGroup") in (groups or ())
    )
    if not watched:
        problems.append(
            f"watches {record.get('namespace')} {record.get('metric')} "
            f"({_dimension_text(dimensions) or 'no dimensions'}), not the current "
            f"API NLB {lb} (target group {', '.join(groups or ()) or 'none'}): after "
            "a fresh install the NLB is new and an alarm on the old one has no "
            "data -- rollback protection is off"
        )
    return problems


# --- the recommended pair ------------------------------------------------------------

#: The capacity alarm and the not-restored alarm of the CPU control plane.
SUGGESTED_ALARM_NAMES = (
    "gpu-fault-cpu-roll-api-capacity",
    "gpu-fault-cpu-roll-api-not-restored",
)
#: The statistic the pair reads (AWS recommends the maximum of
#: ``HealthyHostCount``, ``NLB_HEALTHY_HOSTS_EVIDENCE``): "Maximum < X for k of
#: k minutes" is fewer than X healthy targets for the whole of every one of
#: those minutes, as every load balancer node saw it; Minimum also counts a
#: minute in which one node's distributed health checks dipped for a moment
#: (``NLB_HEALTH_CHECKS_EVIDENCE``). Live, 14 days to 2026-11-02 01:06Z: Minimum
#: < 3 in 19 minutes, Maximum < 3 in 5; the 2026-11-01 dips at 08:12-08:15Z and
#: 08:20-08:21Z were sub-minute (Maximum stayed 3 except at 08:14).
RECOMMENDED_STATISTIC = "Maximum"
#: A one-instance batch keeps one api-ha replica down while its node is
#: patched: about 5 minutes on the live site (2026-11-02, read-only).
NORMAL_BATCH_MINUTES = 5
#: The not-restored alarm trips this many minutes before the next batch.
RESTORE_MARGIN_MINUTES = 5
CAPACITY_MINUTES = 5


def _put_metric_alarm(
    region: str,
    name: str,
    *,
    description: str,
    nlb: Mapping[str, Any],
    minutes: int,
    threshold: int,
) -> dict[str, Any]:
    arguments = [
        "aws",
        "cloudwatch",
        "put-metric-alarm",
        "--region",
        region,
        "--alarm-name",
        name,
        "--alarm-description",
        description,
        "--namespace",
        "AWS/NetworkELB",
        "--metric-name",
        "HealthyHostCount",
        "--dimensions",
        f"Name=TargetGroup,Value={nlb['target_groups'][0]}",
        f"Name=LoadBalancer,Value={nlb['load_balancer']}",
        "--statistic",
        RECOMMENDED_STATISTIC,
        "--period",
        "60",
        "--evaluation-periods",
        str(minutes),
        "--datapoints-to-alarm",
        str(minutes),
        "--threshold",
        str(threshold),
        "--comparison-operator",
        "LessThanThreshold",
        "--treat-missing-data",
        "missing",
    ]
    return {"name": name, "command": shlex.join(arguments)}


def suggested_alarms(
    *, region: str, nlb: Mapping[str, Any], replicas: int, wait_seconds: int
) -> dict[str, Any]:
    """The ``aws cloudwatch put-metric-alarm`` commands of the recommended pair
    on the current API NLB, never run, both on the ``RECOMMENDED_STATISTIC`` of
    ``HealthyHostCount`` per minute: ``capacity`` -- fewer than ``replicas - 1``
    healthy targets for 5 whole minutes, more than the one-instance batch lost
    -- and ``not-restored`` -- fewer than ``replicas`` for the wait minus
    ``RESTORE_MARGIN_MINUTES``: the patched node's replica is not back before
    the next batch. Re-running them after a fresh install re-points the
    same names at the new NLB, so the schedule itself needs no change."""

    groups = list(nlb.get("target_groups") or ())
    if not nlb.get("load_balancer") or len(groups) != 1 or replicas < 2:
        return {
            "commands": [],
            "note": (
                f"no suggestion: the API NLB {nlb.get('name')} has "
                f"{len(groups)} target group(s) (one expected) or api-ha runs "
                f"{replicas} replica(s) (at least 2 expected)"
            ),
        }
    head = (
        "GPU fault product, CPU control-plane HyperPod rolling update "
        "(ScheduledUpdateConfig AutoRollbackConfiguration): "
    )
    commands = [
        _put_metric_alarm(
            region,
            SUGGESTED_ALARM_NAMES[0],
            description=head
            + f"fewer than {replicas - 1} of {replicas} API replicas healthy for "
            f"{CAPACITY_MINUTES} min = more capacity lost than the one-instance "
            "batch; SageMaker rolls the update back. No alarm actions.",
            nlb=nlb,
            minutes=CAPACITY_MINUTES,
            threshold=replicas - 1,
        )
    ]
    restore = wait_seconds // 60 - RESTORE_MARGIN_MINUTES
    note = ""
    if restore >= 2 * NORMAL_BATCH_MINUTES:
        commands.append(
            _put_metric_alarm(
                region,
                SUGGESTED_ALARM_NAMES[1],
                description=head + f"fewer than {replicas} of {replicas} API replicas "
                f"healthy for {restore} min = the patched node's replica did not "
                f"return before the next batch ({wait_seconds} s); SageMaker rolls "
                "the update back. No alarm actions.",
                nlb=nlb,
                minutes=restore,
                threshold=replicas,
            )
        )
    else:
        note = (
            f"no {SUGGESTED_ALARM_NAMES[1]} suggestion: a wait of {wait_seconds} s "
            f"leaves less than {2 * NORMAL_BATCH_MINUTES} min for it, and a normal "
            f"batch is one replica short for about {NORMAL_BATCH_MINUTES} min"
        )
    return {"commands": commands, "note": note}
