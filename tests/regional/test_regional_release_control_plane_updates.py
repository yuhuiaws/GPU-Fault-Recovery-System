"""The ``regional_contexts`` check records the CPU control-plane cluster's own
automatic node updates; the verdict stays.

``status --full`` and deploy preflight run the check. It already describes the
CPU HyperPod cluster to bind it to ``cpu_eks_arn``; that describe now also
carries every instance group's ``ScheduledUpdateConfig``/``AutoPatchConfig``
and ``CurrentCount`` -- no new read -- and ``details.control_plane`` records
them, with the managed node groups the CPU EKS cluster may also have (read the
way the GPU path reads them: ``list-nodegroups``, a describe per group, one
Auto Scaling read; ``unknown`` when a read fails). The ``status`` header names
the control plane's advisory first: on the live site a weekly
``cron(0 8 ? * SUN *)`` re-imaged all three CPU nodes at once.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault.admin.automatic_node_updates import CONTROL_PLANE, FIELD
from gpu_fault.admin.hyperpod_rollback import SUGGESTED_ALARM_NAMES
from gpu_fault_release import regional_release_preflight as preflight
from gpu_fault_release.regional_admin_commands import status_header_lines
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional.test_regional_release_automatic_updates import (
    CPU_EKS_ARN,
    EKS,
    HYPERPOD,
    SPOT_ASG,
    ContextsRelease,
    ContextsRunner,
    _nodepool,
    _report,
)

CPU_GROUP = "cpu-control-plane-c5"
CPU_SCHEDULE = "cron(0 8 ? * SUN *)"
CPU_ASG = "eks-cpu-system-00ff"


def _cpu_group(**extra: Any) -> dict[str, Any]:
    """The live CPU instance group's ``describe-cluster`` entry (full details;
    the context query keeps the advisory fields)."""

    return {
        "InstanceGroupName": CPU_GROUP,
        "InstanceType": "ml.c5.4xlarge",
        "CurrentCount": 3,
        "TargetCount": 3,
        "ExecutionRole": "arn:aws:iam::123456789012:role/hyperpod-cpu-execution",
        "LifeCycleConfig": {"OnCreate": "on_create.sh", "SourceS3Uri": "s3://lc/"},
        "Status": "InService",
        "ScheduledUpdateConfig": {"ScheduleExpression": CPU_SCHEDULE},
        **extra,
    }


def _facts(runner: ContextsRunner, clusters: tuple[Any, ...] = (HYPERPOD, EKS)):
    release = ContextsRelease(runner, clusters)
    facts = preflight.ensure_region_contexts(release)  # type: ignore[arg-type]
    assert sorted(release.iam) == sorted(item.cluster_id for item in clusters), (
        "every gate ran (in parallel: in no particular order)"
    )
    return facts


def _calls(runner: ContextsRunner, *prefix: str) -> list[list[str]]:
    return [call for call in runner.calls if call[: len(prefix)] == list(prefix)]


# --- the check ---------------------------------------------------------------------


def test_contexts_record_the_control_plane_block_from_the_describe_already_made() -> (
    None
):
    runner = ContextsRunner(cpu_groups=[_cpu_group()])

    facts = _facts(runner)

    assert facts[CONTROL_PLANE] == {
        "provider": "hyperpod",
        "hyperpod_cluster_name": "hp-cpu",
        FIELD: {
            "hyperpod_cluster_name": "hp-cpu",
            "cpu_eks_arn": CPU_EKS_ARN,
            "instance_groups": [
                {
                    "name": CPU_GROUP,
                    "scheduled_update": CPU_SCHEDULE,
                    "auto_patch": None,
                    "instances": 3,
                    "rolling_batch": None,
                }
            ],
            "eks_node_groups": {"state": "read", "node_groups": []},
        },
    }, "every CPU group, its count and no rolling batch: all instances at once"
    describes = _calls(runner, "aws", "sagemaker", "describe-cluster")
    assert len(describes) == 2, "one describe per HyperPod cluster: no new read"
    cpu_lists = [
        call
        for call in _calls(runner, "aws", "eks", "list-nodegroups")
        if call[call.index("--cluster-name") + 1] == "cpu"
    ]
    assert len(cpu_lists) == 1, "one list of the CPU cluster's managed node groups"
    assert sorted(facts["providers"]) == ["gpu-eks", "gpu-hp"], "GPU facts unchanged"


def test_cpu_managed_node_groups_are_read_the_way_the_gpu_path_reads_them() -> None:
    runner = ContextsRunner(
        cpu_groups=[_cpu_group()],
        cpu_nodegroups=("cpu-system",),
        asgs=[
            {
                "AutoScalingGroupName": CPU_ASG,
                "CapacityRebalance": False,
                "MaxInstanceLifetime": 604800,
            },
            {"AutoScalingGroupName": SPOT_ASG, "CapacityRebalance": True},
        ],
    )

    facts = _facts(runner)

    block = facts[CONTROL_PLANE][FIELD]
    assert block["eks_node_groups"] == {
        "state": "read",
        "node_groups": [
            {
                "name": "cpu-system",
                "auto_scaling_groups": [
                    {
                        "name": CPU_ASG,
                        "max_instance_lifetime": 604800,
                        "capacity_rebalance": False,
                    }
                ],
            }
        ],
    }
    cpu_asg_reads = [
        call
        for call in _calls(runner, "aws", "autoscaling", "describe-auto-scaling-groups")
        if CPU_ASG in call
    ]
    assert len(cpu_asg_reads) == 1 and SPOT_ASG not in cpu_asg_reads[0], (
        "one Auto Scaling read for the CPU node groups, beside the GPU one"
    )
    lines = status_header_lines(_report(facts), json_destination="stdout")
    assert any(
        line.startswith(f"advisory: Auto Scaling group {CPU_ASG} of EKS node group ")
        and "(the CPU control-plane cluster hp-cpu)" in line
        for line in lines
    ), lines


class DeniedCpuRunner(ContextsRunner):
    """The operator identity may not list the CPU cluster's node groups."""

    def run(self, args: list[str], **options: Any) -> str:
        if args[:3] == ["aws", "eks", "list-nodegroups"] and "cpu" in args:
            self.calls.append(list(args))
            raise ReleaseError(
                "aws eks list-nodegroups exited 254: AccessDeniedException: not "
                "authorized to perform: eks:ListNodegroups"
            )
        return super().run(args, **options)


def test_a_denied_cpu_node_group_read_is_unknown_and_the_check_still_passes() -> None:
    facts = _facts(DeniedCpuRunner(cpu_groups=[_cpu_group()]))  # PASS: it returned

    block = facts[CONTROL_PLANE][FIELD]
    assert block["eks_node_groups"]["state"] == "unknown"
    assert "eks:ListNodegroups" in block["eks_node_groups"]["error"]
    assert block["instance_groups"][0]["scheduled_update"] == CPU_SCHEDULE, (
        "the HyperPod half is recorded whatever the node group read did"
    )
    assert "gpu-eks" in facts["providers"]


def test_a_broken_control_plane_advisory_is_unknown_and_never_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_arguments: Any) -> dict[str, Any]:
        raise RuntimeError("advisory defect")

    monkeypatch.setattr(preflight, "control_plane_automatic_updates", broken)

    facts = _facts(ContextsRunner(cpu_groups=[_cpu_group()]))

    assert facts[CONTROL_PLANE][FIELD] == {
        "state": "unknown",
        "error": "advisory defect",
    }
    assert facts["providers"]["gpu-hp"]["provider"] == "hyperpod"


def test_a_cpu_gate_failure_still_fails_before_any_advisory() -> None:
    class ForeignCpu(ContextsRunner):
        def run(self, args: list[str], **options: Any) -> str:
            if (
                args[:3] == ["aws", "sagemaker", "describe-cluster"]
                and "hp-cpu" in args
            ):
                return json.dumps({"EksClusterArn": "arn:aws:eks:x:1:cluster/other"})
            return super().run(args, **options)

    runner = ForeignCpu(cpu_groups=[_cpu_group()])

    with pytest.raises(ReleaseError, match="does not target"):
        _facts(runner)
    assert not _calls(runner, "aws", "eks", "list-nodegroups"), (
        "the advisory runs only after the CPU gates passed"
    )


def test_a_site_without_gpu_clusters_still_records_the_control_plane() -> None:
    runner = ContextsRunner(cpu_groups=[_cpu_group()])

    facts = _facts(runner, clusters=())

    assert sorted(facts) == ["clusters", CONTROL_PLANE, "eks_support"]
    assert facts[CONTROL_PLANE][FIELD]["instance_groups"][0]["name"] == CPU_GROUP


# --- the status header ----------------------------------------------------------------


def test_status_full_header_names_the_control_plane_schedule_first() -> None:
    facts = _facts(
        ContextsRunner(
            cpu_groups=[_cpu_group()],
            asgs=[
                {
                    "AutoScalingGroupName": SPOT_ASG,
                    "CapacityRebalance": True,
                    "MaxInstanceLifetime": 604800,
                }
            ],
            nodepools=[_nodepool("gpu", expire="720h", budgets=[{"nodes": "10%"}])],
        )
    )

    lines = status_header_lines(_report(facts), json_destination="stdout")

    assert lines[0] == "healthy: yes (full health, 1 checks)", "the verdict stays"
    advisories = [line for line in lines if line.startswith("advisory: ")]
    assert advisories[0] == (
        f"advisory: HyperPod instance group {CPU_GROUP} of the CPU control-plane "
        f"cluster hp-cpu has ScheduledUpdateConfig {CPU_SCHEDULE} with no rolling "
        "DeploymentConfig: observed (2026-11-01) to re-image all 3 instances of the "
        "group at once -- every node the control plane runs on, so the whole "
        "control plane (api-ha, control-worker) is down for minutes each time; "
        "gpu-fault-admin hyperpod-scheduled-update --state-dir <state dir> "
        f"--gpu-cluster-arn {CPU_EKS_ARN} --instance-group {CPU_GROUP} --rolling "
        "--rollback-alarm <alarm> prints the update-cluster commands (delete the "
        "schedule, then create it again) that make it roll one instance at a time, "
        "rolled back on that alarm (without --rolling, the one that removes it: "
        "then update the group by hand in a maintenance window, one instance at a "
        "time) (administrator-operations §5.0.5)"
    ), "the control plane's own schedule leads"
    assert [line.split(":", 2)[1].strip()[:40] for line in advisories[1:]] == [
        f"Auto Scaling group {SPOT_ASG}"[:40],
        "Karpenter NodePool gpu of gpu-eks provis",
        "HyperPod instance group spot-p5en-az3 of",
    ], "then the GPU clusters, sorted by cluster id, as before"


def test_a_malformed_control_plane_entry_drops_only_its_own_lines() -> None:
    facts = _facts(ContextsRunner(cpu_groups=[_cpu_group()]))
    report = _report(facts)
    details = report["health"]["checks"][0]["details"]
    expected = [
        line
        for line in status_header_lines(copy.deepcopy(report), json_destination="x")
        if line.startswith("advisory: HyperPod instance group spot-p5en-az3")
    ]

    for broken in ("not-a-mapping", {FIELD: ["x"]}, {FIELD: {"state": "unknown"}}):
        details[CONTROL_PLANE] = broken
        lines = status_header_lines(report, json_destination="x")
        advisories = [line for line in lines if line.startswith("advisory: ")]
        assert advisories == expected, broken


# --- the rollback alarms of a rolling CPU schedule (batch 239) -----------------------
#
# The live group since 2026-11-02: one instance at a time, 1800 s apart, rolled
# back one at a time on the API NLB HealthyHostCount pair. The check reads the
# alarms (describe-alarms) and the NLB they should watch (the configured name,
# as the NLB check reads it); an alarm that cannot trip is announced.

NLB_NAME = "gpu-fault-api-site"
LB = f"net/{NLB_NAME}/0123456789abcdef"
TG = "targetgroup/k8s-gpufault-gpufaulta-0a1b2c3d4e/fedcba9876543210"
OLD_LB = f"net/{NLB_NAME}/00112233aabbccdd"
OLD_TG = "targetgroup/k8s-gpufault-gpufaulta-99aa88bb77/0011223344556677"
LB_ARN = f"arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/{LB}"
TG_ARN = f"arn:aws:elasticloadbalancing:us-west-2:123456789012:{TG}"
CAPACITY, NOT_RESTORED = SUGGESTED_ALARM_NAMES
ONE = {"Type": "INSTANCE_COUNT", "Value": 1}
ROLLING = {
    "ScheduleExpression": CPU_SCHEDULE,
    "DeploymentConfig": {
        "RollingUpdatePolicy": {
            "MaximumBatchSize": dict(ONE),
            "RollbackMaximumBatchSize": dict(ONE),
        },
        "WaitIntervalInSeconds": 1800,
        "AutoRollbackConfiguration": [
            {"AlarmName": CAPACITY},
            {"AlarmName": NOT_RESTORED},
        ],
    },
}


def _alarms(**options: Any) -> dict[str, dict[str, Any]]:
    def alarm(name: str, threshold: int, minutes: int) -> dict[str, Any]:
        return {
            "AlarmName": name,
            "StateValue": options.get("state", "OK"),
            "Namespace": "AWS/NetworkELB",
            "MetricName": "HealthyHostCount",
            "Statistic": "Maximum",
            "Dimensions": [
                {"Name": "TargetGroup", "Value": options.get("tg", TG)},
                {"Name": "LoadBalancer", "Value": options.get("lb", LB)},
            ],
            "Period": 60,
            "EvaluationPeriods": minutes,
            "DatapointsToAlarm": minutes,
            "Threshold": float(threshold),
            "ComparisonOperator": "LessThanThreshold",
            "TreatMissingData": "missing",
        }

    return {CAPACITY: alarm(CAPACITY, 2, 5), NOT_RESTORED: alarm(NOT_RESTORED, 3, 25)}


class AlarmRunner(ContextsRunner):
    """The CPU group's rollback alarms and the API NLB, as the AWS CLI answers."""

    def __init__(
        self, *, alarms: dict[str, dict[str, Any]], deny: bool = False, **options: Any
    ) -> None:
        super().__init__(**options)
        self.alarms = alarms
        self.deny = deny

    def run(self, args: list[str], **options: Any) -> str:
        if args[:3] == ["aws", "cloudwatch", "describe-alarms"]:
            self.calls.append(list(args))
            if self.deny:
                raise ReleaseError(
                    "aws cloudwatch describe-alarms exited 254: AccessDenied: not "
                    "authorized to perform: cloudwatch:DescribeAlarms"
                )
            names = args[args.index("--alarm-names") + 1 : args.index("--alarm-types")]
            found = [self.alarms[name] for name in names if name in self.alarms]
            return json.dumps({"MetricAlarms": found, "CompositeAlarms": []})
        if args[:3] == ["aws", "elbv2", "describe-load-balancers"]:
            self.calls.append(list(args))
            assert args[args.index("--names") + 1] == NLB_NAME, args
            return json.dumps({"LoadBalancers": [{"LoadBalancerArn": LB_ARN}]})
        if args[:3] == ["aws", "elbv2", "describe-target-groups"]:
            self.calls.append(list(args))
            assert args[args.index("--load-balancer-arn") + 1] == LB_ARN, args
            return json.dumps({"TargetGroups": [{"TargetGroupArn": TG_ARN}]})
        return super().run(args, **options)


def _rolling_facts(runner: AlarmRunner) -> dict[str, Any]:
    release = ContextsRelease(runner, (HYPERPOD, EKS))
    release.config.nlb = {"name": NLB_NAME}
    return preflight.ensure_region_contexts(release)  # type: ignore[arg-type]


def _advisories(facts: dict[str, Any]) -> list[str]:
    lines = status_header_lines(_report(facts), json_destination="stdout")
    assert lines[0] == "healthy: yes (full health, 1 checks)", "the verdict stays"
    return [line for line in lines if line.startswith("advisory: ")]


def test_contexts_read_the_rolling_groups_alarms_against_the_api_nlb() -> None:
    runner = AlarmRunner(
        cpu_groups=[_cpu_group(ScheduledUpdateConfig=ROLLING)], alarms=_alarms()
    )

    facts = _rolling_facts(runner)

    block = facts[CONTROL_PLANE][FIELD]
    assert block["instance_groups"][0]["rolling_batch"] == {
        "type": "INSTANCE_COUNT",
        "value": 1,
        "wait_seconds": 1800,
        "rollback_batch": {"type": "INSTANCE_COUNT", "value": 1},
        "rollback_alarms": [CAPACITY, NOT_RESTORED],
    }
    read = block["rollback_alarms"]
    assert read["state"] == "read"
    assert read["api_nlb"] == {
        "name": NLB_NAME,
        "load_balancer": LB,
        "target_groups": [TG],
    }
    assert [item["problems"] for item in read["alarms"]] == [[], []]
    (alarm_read,) = _calls(runner, "aws", "cloudwatch", "describe-alarms")
    assert alarm_read[3:9] == [
        "--alarm-names",
        CAPACITY,
        NOT_RESTORED,
        "--alarm-types",
        "MetricAlarm",
        "CompositeAlarm",
    ], "one read for every named alarm, metric and composite"
    advisories = _advisories(facts)
    assert advisories[0].startswith(
        f"advisory: HyperPod instance group {CPU_GROUP} of the CPU control-plane "
        f"cluster hp-cpu has ScheduledUpdateConfig {CPU_SCHEDULE} rolling 1 "
        "instance(s) at a time 1800 s apart, rolled back 1 instance(s) at a time on "
        f"alarm {CAPACITY} or {NOT_RESTORED}: the safe form -- "
    ), advisories[0]
    assert not any("rollback alarm" in line for line in advisories), "alarms that trip"


def test_an_alarm_on_the_previous_nlb_is_announced_and_the_check_passes() -> None:
    stale = _alarms(state="INSUFFICIENT_DATA", lb=OLD_LB, tg=OLD_TG)
    runner = AlarmRunner(
        cpu_groups=[_cpu_group(ScheduledUpdateConfig=ROLLING)], alarms=stale
    )

    advisories = _advisories(_rolling_facts(runner))

    (capacity,) = [line for line in advisories if f"rollback alarm {CAPACITY} " in line]
    assert capacity.startswith(
        f"advisory: rollback alarm {CAPACITY} of HyperPod instance group {CPU_GROUP} "
        "of the CPU control-plane cluster hp-cpu is INSUFFICIENT_DATA: "
    ), capacity
    assert f"not the current API NLB {LB} (target group {TG})" in capacity
    assert "--rolling --wait-interval-seconds 1800 --rollback-alarm " in capacity


def test_a_denied_alarm_read_is_unknown_announced_and_the_check_passes() -> None:
    runner = AlarmRunner(
        cpu_groups=[_cpu_group(ScheduledUpdateConfig=ROLLING)],
        alarms=_alarms(),
        deny=True,
    )

    facts = _rolling_facts(runner)

    read = facts[CONTROL_PLANE][FIELD]["rollback_alarms"]
    assert read["state"] == "unknown" and "cloudwatch:DescribeAlarms" in read["error"]
    assert any(
        line.startswith(
            f"advisory: could not read rollback alarm(s) {CAPACITY}, {NOT_RESTORED} "
        )
        for line in _advisories(facts)
    ), "a denied read is announced: protection may be off"


def test_a_schedule_without_rollback_alarms_reads_none() -> None:
    runner = AlarmRunner(cpu_groups=[_cpu_group()], alarms=_alarms())

    facts = _rolling_facts(runner)

    assert "rollback_alarms" not in facts[CONTROL_PLANE][FIELD]
    assert not _calls(runner, "aws", "cloudwatch") and not _calls(
        runner, "aws", "elbv2"
    )
