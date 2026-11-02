"""A fake AWS for the deploy's CPU rolling-update step (batch 240).

``World`` answers every read and write ``gpu_fault.admin.cpu_rolling_update``
makes through a ``CommandRunner``: the CPU HyperPod cluster (``FakeHyperPod``
from the batch 239 tests, refusing exactly the three requests the service
refused on 2026-11-02), the API NLB and its target health, CloudWatch alarms
(a new alarm starts INSUFFICIENT_DATA and is evaluated on the next read; an
update keeps the state and ignores ``--tags``, as PutMetricAlarm does), their
tags and the ``HealthyHostCount`` statistics. ``FakeClock`` advances on sleep.
Nothing here runs a command.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.config import default_admin_config
from gpu_fault.admin.cpu_rolling_update import Clock
from tests.admin.test_admin_hyperpod_rollback import (
    ACCOUNT,
    CAPACITY,
    CPU_EKS_ARN,
    CPU_NAME,
    LB,
    LB_ARN,
    NLB_NAME,
    NOT_RESTORED,
    OLD_LB,
    OLD_TG,
    REGION,
    TG,
    TG_ARN,
    FakeHyperPod,
    Refused,
    _described,
    _metric_alarm,
)

SITE_ID = "us-west-2-cpu-site-1a2b3c4d"
OTHER_SITE = "us-west-2-other-site-5e6f7a8b"
NOW = datetime(2026, 11, 2, 2, 0, tzinfo=UTC)
RELEASE_CONFIG: dict[str, Any] = {
    "aws_region": REGION,
    "cpu_eks_arn": CPU_EKS_ARN,
    "cpu_hyperpod_cluster_name": CPU_NAME,
    "site_name": SITE_ID,
    "nlb": {"name": NLB_NAME},
}
SITE_TAG = {"Key": "gpu-fault:site-id", "Value": SITE_ID}


def cpu_section(**spec: Any) -> Any:
    """``AdminConfig.cpu_control_plane`` with a camelCase ``spec`` laid over."""

    config = default_admin_config().patched({"cpuControlPlane": spec} if spec else {})
    config.validate()
    return config.cpu_control_plane


@dataclass
class FakeClock:
    now: datetime = NOW
    slept: list[float] = field(default_factory=list)

    def clock(self) -> Clock:
        return Clock(sleep=self.sleep, monotonic=self.monotonic, now=lambda: self.now)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)

    def monotonic(self) -> float:
        return (self.now - NOW).total_seconds()


def _flags(words: list[str]) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {}
    current = ""
    for word in words:
        if word.startswith("--"):
            current = word
            flags[current] = []
        else:
            flags[current].append(word)
    return flags


def _alarm_from_put(words: list[str]) -> dict[str, Any]:
    """A ``describe-alarms`` entry from a printed ``put-metric-alarm``."""

    flags = _flags(words)
    dimensions = [
        dict(part.split("=", 1) for part in value.split(","))
        for value in flags["--dimensions"]
    ]
    return {
        "AlarmName": flags["--alarm-name"][0],
        "AlarmDescription": flags["--alarm-description"][0],
        "Namespace": flags["--namespace"][0],
        "MetricName": flags["--metric-name"][0],
        "Dimensions": dimensions,
        "Statistic": flags["--statistic"][0],
        "Period": int(flags["--period"][0]),
        "EvaluationPeriods": int(flags["--evaluation-periods"][0]),
        "DatapointsToAlarm": int(flags["--datapoints-to-alarm"][0]),
        "Threshold": float(flags["--threshold"][0]),
        "ComparisonOperator": flags["--comparison-operator"][0],
        "TreatMissingData": flags["--treat-missing-data"][0],
        "AlarmActions": [],
        "OKActions": [],
        "InsufficientDataActions": [],
    }


class World:
    """The CPU cluster, the API NLB and CloudWatch, as a ``CommandRunner``."""

    def __init__(
        self,
        described: dict[str, Any] | None = None,
        *,
        alarms: dict[str, dict[str, Any]] | None = None,
        tags: dict[str, list[dict[str, str]]] | None = None,
        healthy: int = 3,
        nlb: bool = True,
        clock: FakeClock | None = None,
    ) -> None:
        self.hyperpod = FakeHyperPod(
            described if described is not None else _described()
        )
        self.alarms = copy.deepcopy(alarms or {})
        self.tags = copy.deepcopy(tags or {})
        self.healthy = healthy
        self.nlb = nlb
        self.clock = clock or FakeClock()
        self.calls: list[tuple[str, ...]] = []
        self.writes: list[tuple[str, ...]] = []
        #: How many creates (a request carrying a schedule) fail, throttled.
        self.fail_creates = 0
        #: How many describes after a write still answer ``Updating``.
        self.updating_reads = 1
        self._updating = 0

    # --- CommandRunner ---------------------------------------------------------

    def aws_json(self, region, service, operation, *arguments, **options):
        assert region == REGION, region
        self.calls.append((service, operation, *arguments))
        if options.get("mutate"):
            self.writes.append((service, operation, *arguments))
        handler = getattr(self, f"_{service}_{operation.replace('-', '_')}", None)
        if handler is None:
            raise AssertionError(f"unexpected aws {service} {operation}")
        return handler(list(arguments), mutate=bool(options.get("mutate")))

    def run(self, arguments, **options):
        words = list(arguments)
        assert words[:2] == ["aws", "cloudwatch"], words
        flags = _flags(words[3:])
        assert flags.pop("--region") == [REGION]
        self.calls.append(tuple(words[1:3]))
        assert options.get("mutate"), f"a write without mutate=True: {words[:3]}"
        self.writes.append(tuple(words[1:3]))
        if words[2] == "put-metric-alarm":
            alarm = _alarm_from_put(words[3:])
            name = alarm["AlarmName"]
            arn = f"arn:aws:cloudwatch:{REGION}:{ACCOUNT}:alarm:{name}"
            existing = self.alarms.get(name)
            if existing is None:
                self.alarms[name] = {
                    **alarm,
                    "AlarmArn": arn,
                    "StateValue": "INSUFFICIENT_DATA",
                    "AlarmConfigurationUpdatedTimestamp": self.clock.now.isoformat(),
                }
                if "--tags" in flags:
                    self.tags[arn] = [
                        dict(part.split("=", 1) for part in value.split(","))
                        for value in flags["--tags"]
                    ]
            else:
                # An update keeps the state and ignores --tags.
                self.alarms[name] = {
                    **alarm,
                    "AlarmArn": existing.get("AlarmArn", arn),
                    "StateValue": existing.get("StateValue", "OK"),
                    "AlarmConfigurationUpdatedTimestamp": self.clock.now.isoformat(),
                }
            return ""
        if words[2] == "tag-resource":
            arn = flags["--resource-arn"][0]
            pairs = [
                dict(part.split("=", 1) for part in v.split(","))
                for v in flags["--tags"]
            ]
            current = {item["Key"]: item["Value"] for item in self.tags.get(arn, [])}
            current.update({item["Key"]: item["Value"] for item in pairs})
            self.tags[arn] = [{"Key": k, "Value": v} for k, v in current.items()]
            return ""
        raise AssertionError(f"unexpected run {words[:3]}")

    # --- sagemaker -------------------------------------------------------------

    def _sagemaker_describe_cluster(self, arguments, *, mutate):
        assert arguments == ["--cluster-name", CPU_NAME], arguments
        described = self.hyperpod.describe()
        if self._updating:
            self._updating -= 1
            described["ClusterStatus"] = "Updating"
        return described

    def _sagemaker_update_cluster(self, arguments, *, mutate):
        assert mutate, "update-cluster without mutate=True"
        flags = _flags(arguments)
        definitions = json.loads(flags["--instance-groups"][0])
        if self.fail_creates and definitions[0].get("ScheduledUpdateConfig"):
            self.fail_creates -= 1
            raise BootstrapError(
                "command failed (254): aws sagemaker update-cluster: An error occurred "
                "(ThrottlingException) when calling the UpdateCluster operation: Rate "
                "exceeded"
            )
        try:
            self.hyperpod.update_cluster(
                flags["--cluster-name"][0], json.loads(flags["--instance-groups"][0])
            )
        except Refused as exc:
            raise BootstrapError(
                "command failed (254): aws sagemaker update-cluster: An error occurred "
                f"(ValidationException) when calling the UpdateCluster operation: {exc}"
            ) from exc
        self._updating = self.updating_reads
        return {"ClusterArn": self.hyperpod.described["ClusterArn"]}

    # --- elbv2 -----------------------------------------------------------------

    def _elbv2_describe_load_balancers(self, arguments, *, mutate):
        assert arguments == ["--names", NLB_NAME], arguments
        if not self.nlb:
            raise BootstrapError(
                "command failed (254): aws elbv2 describe-load-balancers: "
                "LoadBalancerNotFound"
            )
        return {
            "LoadBalancers": [{"LoadBalancerArn": LB_ARN, "LoadBalancerName": NLB_NAME}]
        }

    def _elbv2_describe_target_groups(self, arguments, *, mutate):
        assert arguments == ["--load-balancer-arn", LB_ARN], arguments
        return {"TargetGroups": [{"TargetGroupArn": TG_ARN, "TargetType": "ip"}]}

    def _elbv2_describe_target_health(self, arguments, *, mutate):
        assert arguments == ["--target-group-arn", TG_ARN], arguments
        states = ["healthy"] * self.healthy + ["initial"] * (3 - self.healthy)
        return {
            "TargetHealthDescriptions": [
                {"Target": {"Id": f"10.0.0.{n}"}, "TargetHealth": {"State": state}}
                for n, state in enumerate(states)
            ]
        }

    # --- cloudwatch ------------------------------------------------------------

    def _cloudwatch_describe_alarms(self, arguments, *, mutate):
        split = arguments.index("--alarm-types")
        names = arguments[1:split]
        found = []
        for name in names:
            alarm = self.alarms.get(name)
            if alarm is None:
                continue
            if alarm.get("StateValue") == "INSUFFICIENT_DATA" and self._on_current_nlb(
                alarm
            ):
                alarm["StateValue"] = "OK"  # evaluated on data that is already OK
                found.append({**alarm, "StateValue": "INSUFFICIENT_DATA"})
                continue
            found.append(copy.deepcopy(alarm))
        return {"MetricAlarms": found, "CompositeAlarms": []}

    def _on_current_nlb(self, alarm: dict[str, Any]) -> bool:
        dimensions = {
            item["Name"]: item["Value"] for item in alarm.get("Dimensions") or ()
        }
        return (
            dimensions == {"TargetGroup": TG, "LoadBalancer": LB} and self.healthy >= 3
        )

    def _cloudwatch_list_tags_for_resource(self, arguments, *, mutate):
        return {"Tags": copy.deepcopy(self.tags.get(arguments[1], []))}

    def _cloudwatch_get_metric_statistics(self, arguments, *, mutate):
        flags = _flags(arguments)
        assert flags["--statistics"] == ["Maximum"], "the minute is judged on Maximum"
        start = datetime.fromisoformat(flags["--start-time"][0])
        end = datetime.fromisoformat(flags["--end-time"][0])
        points = []
        minute = start
        while minute + timedelta(minutes=1) <= end:
            points.append(
                {"Timestamp": minute.isoformat(), "Maximum": float(self.healthy)}
            )
            minute += timedelta(minutes=1)
        return {"Datapoints": points, "Label": "HealthyHostCount"}


def live_alarms(**options: Any) -> dict[str, dict[str, Any]]:
    """The live pair (Maximum since 01:08Z), on the current NLB unless told."""

    return {
        CAPACITY: {
            **_metric_alarm(CAPACITY, threshold=2, minutes=5, **options),
            "Statistic": "Maximum",
        },
        NOT_RESTORED: {
            **_metric_alarm(NOT_RESTORED, threshold=3, minutes=25, **options),
            "Statistic": "Maximum",
        },
    }


def old_nlb_alarms() -> dict[str, dict[str, Any]]:
    return live_alarms(state="INSUFFICIENT_DATA", lb=OLD_LB, tg=OLD_TG)


__all__ = [
    "CAPACITY",
    "NOT_RESTORED",
    "NOW",
    "OTHER_SITE",
    "RELEASE_CONFIG",
    "SITE_ID",
    "SITE_TAG",
    "FakeClock",
    "World",
    "cpu_section",
    "live_alarms",
    "old_nlb_alarms",
]
