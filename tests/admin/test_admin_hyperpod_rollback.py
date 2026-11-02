"""``hyperpod-scheduled-update --rolling`` plans what the service accepts (batch 239).

On 2026-11-02 the service refused batch 231's rolling plan for the live CPU
control-plane group three times (``hyperpod_rollback.SERVICE_REFUSALS``): no
``RollbackMaximumBatchSize``, no ``AutoRollbackConfiguration``, and an in-place
change of the existing ``ScheduledUpdateConfig``. The API reference and the
installed service model mark both members optional, so ``ParamValidator``
passed. ``FakeHyperPod`` answers ``UpdateCluster`` the way the service did that
day -- refusing those three, accepting a delete (the definition without the
member) and then a create -- and the plan's ordered steps are run against it.

The world is the live one, sanitised: the CPU control-plane HyperPod cluster
(one group, 3 x ml.c5.4xlarge, image 2.0.4, ``cron(0 8 ? * SUN *)``), the API
NLB with its one IP target group, and the two ``HealthyHostCount`` alarms the
coordinating session created (no actions, ``TreatMissingData`` missing; the
statistic Maximum since 01:08Z); ``LIVE_ROLLING`` is what ``describe-cluster``
showed at 00:34Z.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from pathlib import Path
from typing import Any

import botocore.session
import jmespath
import pytest
import yaml
from botocore.validate import ParamValidator

from gpu_fault.admin import automatic_node_updates as updates
from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin import hyperpod_scheduled_update as helper
from gpu_fault.admin.bootstrap_common import BootstrapError

REGION = "us-west-2"
ACCOUNT = "123456789012"
CPU_NAME = "cpu-control-plane"
CPU_GROUP = "cpu-control-plane-c5"
CPU_HP_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/cpu0live01"
CPU_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/cpu-eks"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/hyperpod-cpu-execution"
SITE_ID = "us-west-2-cpu-site-1a2b3c4d"
SCHEDULE = "cron(0 8 ? * SUN *)"
WAIT = 1800
ONE = {"Type": "INSTANCE_COUNT", "Value": 1}
CAPACITY, NOT_RESTORED = rollback.SUGGESTED_ALARM_NAMES
NLB_NAME = "gpu-fault-api-site"
LB = f"net/{NLB_NAME}/0123456789abcdef"
TG = "targetgroup/k8s-gpufault-gpufaulta-0a1b2c3d4e/fedcba9876543210"
OLD_LB = f"net/{NLB_NAME}/00112233aabbccdd"
OLD_TG = "targetgroup/k8s-gpufault-gpufaulta-99aa88bb77/0011223344556677"
LB_ARN = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:loadbalancer/{LB}"
TG_ARN = f"arn:aws:elasticloadbalancing:{REGION}:{ACCOUNT}:{TG}"
#: Before 2026-11-02: the weekly schedule, no DeploymentConfig.
LIVE_BEFORE: dict[str, Any] = {"ScheduleExpression": SCHEDULE}
#: ``describe-cluster`` at 2026-11-02T00:34Z, after the delete and the create.
LIVE_ROLLING: dict[str, Any] = {
    "ScheduleExpression": SCHEDULE,
    "DeploymentConfig": {
        "RollingUpdatePolicy": {
            "MaximumBatchSize": dict(ONE),
            "RollbackMaximumBatchSize": dict(ONE),
        },
        "WaitIntervalInSeconds": WAIT,
        "AutoRollbackConfiguration": [
            {"AlarmName": CAPACITY},
            {"AlarmName": NOT_RESTORED},
        ],
    },
}


def _cpu_group(schedule: dict[str, Any] | None) -> dict[str, Any]:
    group: dict[str, Any] = {
        "ActiveOperations": {},
        "CurrentCount": 3,
        "CurrentImageId": "default",
        "CurrentImageReleaseVersion": "2.0.4",
        "DesiredImageId": "default",
        "DesiredImageReleaseVersion": "2.0.4",
        "ExecutionRole": ROLE,
        "ImageVersionStatus": "UpToDate",
        "InstanceGroupName": CPU_GROUP,
        "InstanceStorageConfigs": [
            {"EbsVolumeConfig": {"RootVolume": False, "VolumeSizeInGB": 500}}
        ],
        "InstanceType": "ml.c5.4xlarge",
        "LifeCycleConfig": {"OnCreate": "on_create.sh", "SourceS3Uri": "s3://lc/"},
        "OverrideVpcConfig": {
            "SecurityGroupIds": ["sg-example"],
            "Subnets": ["subnet-example"],
        },
        "SoftwareUpdateStatus": "Succeeded",
        "Status": "InService",
        "TargetCount": 3,
        "TargetStateCount": 3,
        "ThreadsPerCore": 2,
    }
    if schedule is not None:
        group["ScheduledUpdateConfig"] = copy.deepcopy(schedule)
    return group


def _described(schedule: dict[str, Any] | None = LIVE_BEFORE) -> dict[str, Any]:
    return {
        "AutoScaling": {"AutoScalerType": "Karpenter", "Mode": "Enable"},
        "ClusterArn": CPU_HP_ARN,
        "ClusterName": CPU_NAME,
        "ClusterStatus": "InService",
        "CreationTime": "2025-01-01T00:00:00.000000+00:00",
        "InstanceGroups": [_cpu_group(schedule)],
        "NodeProvisioningMode": "Continuous",
        "NodeRecovery": "Automatic",
        "Orchestrator": {"Eks": {"ClusterArn": CPU_EKS_ARN}},
        "RestrictedInstanceGroups": [],
        "TieredStorageConfig": {"Mode": "Disable"},
        "VpcConfig": {
            "SecurityGroupIds": ["sg-example"],
            "Subnets": ["subnet-example"],
        },
    }


def _metric_alarm(
    name: str,
    *,
    threshold: int,
    minutes: int,
    state: str = "OK",
    lb: str = LB,
    tg: str = TG,
) -> dict[str, Any]:
    """One ``describe-alarms`` ``MetricAlarms`` entry shaped like the live ones."""

    return {
        "ActionsEnabled": True,
        "AlarmActions": [],
        "AlarmArn": f"arn:aws:cloudwatch:{REGION}:{ACCOUNT}:alarm:{name}",
        "AlarmName": name,
        "ComparisonOperator": "LessThanThreshold",
        "DatapointsToAlarm": minutes,
        "Dimensions": [
            {"Name": "TargetGroup", "Value": tg},
            {"Name": "LoadBalancer", "Value": lb},
        ],
        "EvaluationPeriods": minutes,
        "InsufficientDataActions": [],
        "MetricName": "HealthyHostCount",
        "Namespace": "AWS/NetworkELB",
        "OKActions": [],
        "Period": 60,
        "StateValue": state,
        "Statistic": "Maximum",
        "Threshold": float(threshold),
        "TreatMissingData": "missing",
    }


def _live_alarms(**options: Any) -> dict[str, dict[str, Any]]:
    """The live pair: Maximum below N-1 (5 of 5) and below N (25 of 25), N=3."""

    return {
        CAPACITY: _metric_alarm(CAPACITY, threshold=2, minutes=5, **options),
        NOT_RESTORED: _metric_alarm(NOT_RESTORED, threshold=3, minutes=25, **options),
    }


class Refused(Exception):
    """An ``UpdateCluster`` ValidationException."""


class FakeHyperPod:
    """``UpdateCluster``/``DescribeCluster`` as the service answered them on
    2026-11-02: a rolling policy without ``RollbackMaximumBatchSize`` is
    refused, then one without rollback alarms, then any change to an existing
    ``ScheduledUpdateConfig``; a definition without the member deletes it and
    one with it creates it. Every request must keep the image and the count --
    anything else would re-image or scale the control plane."""

    def __init__(self, described: dict[str, Any]) -> None:
        self.described = copy.deepcopy(described)
        self.accepted: list[list[dict[str, Any]]] = []

    def describe(self) -> dict[str, Any]:
        return copy.deepcopy(self.described)

    def update_cluster(self, cluster: str, definitions: list[dict[str, Any]]) -> None:
        assert cluster == CPU_NAME, cluster
        groups = {
            group["InstanceGroupName"]: group
            for group in self.described["InstanceGroups"]
        }
        for spec in definitions:
            group = groups[spec["InstanceGroupName"]]
            assert spec["ImageReleaseVersion"] == group["CurrentImageReleaseVersion"]
            assert spec["InstanceCount"] == group["TargetCount"]
            new = spec.get("ScheduledUpdateConfig")
            old = group.get("ScheduledUpdateConfig")
            if new is None:
                continue
            deployment = new.get("DeploymentConfig") or {}
            if "RollingUpdatePolicy" in deployment:
                if "RollbackMaximumBatchSize" not in deployment["RollingUpdatePolicy"]:
                    raise Refused(rollback.SERVICE_REFUSALS[0]["text"])
                if not deployment.get("AutoRollbackConfiguration"):
                    raise Refused(rollback.SERVICE_REFUSALS[1]["text"])
            if old is not None and new != old:
                raise Refused(
                    rollback.SERVICE_REFUSALS[2]["text"].replace(
                        "<instance group>", group["InstanceGroupName"]
                    )
                )
        for spec in definitions:
            group = groups[spec["InstanceGroupName"]]
            if "ScheduledUpdateConfig" in spec:
                group["ScheduledUpdateConfig"] = copy.deepcopy(
                    spec["ScheduledUpdateConfig"]
                )
            else:
                group.pop("ScheduledUpdateConfig", None)
        self.accepted.append(copy.deepcopy(definitions))


def _execute(fake: FakeHyperPod, command: str) -> Any:
    """Run one printed ``aws sagemaker`` command against ``fake``; a
    ``describe-cluster`` answers with its ``--query`` applied, as the CLI does."""

    words = shlex.split(command)
    assert words[:2] == ["aws", "sagemaker"], words
    options = dict(zip(words[3::2], words[4::2], strict=True))
    assert (options["--region"], options["--cluster-name"]) == (REGION, CPU_NAME)
    if words[2] == "update-cluster":
        fake.update_cluster(CPU_NAME, json.loads(options["--instance-groups"]))
        return None
    assert words[2] == "describe-cluster", words
    return jmespath.search(options["--query"], fake.describe())


class ServiceRunner:
    """The helper's reads against ``FakeHyperPod``, the alarms and the API
    NLB; any write fails."""

    def __init__(
        self,
        fake: FakeHyperPod,
        *,
        alarms: dict[str, dict[str, Any]] | None = None,
        deny_alarms: bool = False,
    ) -> None:
        self.fake = fake
        self.alarms = alarms if alarms is not None else _live_alarms()
        self.deny_alarms = deny_alarms
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def aws_json(self, region, service, operation, *arguments, **options):
        self.calls.append((service, operation, tuple(arguments)))
        assert not options.get("mutate"), "the verb writes nothing"
        assert region == REGION
        if (service, operation) == ("sagemaker", "describe-cluster"):
            selected = arguments[arguments.index("--cluster-name") + 1]
            assert selected in {CPU_HP_ARN, CPU_NAME}, selected
            return self.fake.describe()
        if (service, operation) == ("cloudwatch", "describe-alarms"):
            if self.deny_alarms:
                raise BootstrapError(
                    "aws cloudwatch describe-alarms exited 254: AccessDenied: not "
                    "authorized to perform: cloudwatch:DescribeAlarms"
                )
            split = arguments.index("--alarm-types")
            assert arguments[split + 1 :] == ("MetricAlarm", "CompositeAlarm")
            names = arguments[1:split]
            return {
                "CompositeAlarms": [],
                "LogAlarms": [],
                "MetricAlarms": [self.alarms[n] for n in names if n in self.alarms],
            }
        if (service, operation) == ("elbv2", "describe-load-balancers"):
            assert arguments == ("--names", NLB_NAME), arguments
            return {
                "LoadBalancers": [
                    {
                        "LoadBalancerArn": LB_ARN,
                        "LoadBalancerName": NLB_NAME,
                        "Type": "network",
                        "Scheme": "internet-facing",
                        "State": {"Code": "active"},
                    }
                ]
            }
        if (service, operation) == ("elbv2", "describe-target-groups"):
            assert arguments == ("--load-balancer-arn", LB_ARN), arguments
            return {
                "TargetGroups": [
                    {
                        "TargetGroupArn": TG_ARN,
                        "TargetType": "ip",
                        "Port": 8080,
                        "HealthCheckPath": "/healthz",
                    }
                ]
            }
        raise AssertionError(f"unexpected call {service} {operation} {arguments}")


def _state_dir(tmp_path: Path) -> Path:
    (tmp_path / "site.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.io/v1",
                "kind": "RegionalSite",
                "metadata": {"name": SITE_ID},
                "spec": {
                    "cpu": {
                        "kubeconfig": "cpu.kubeconfig",
                        "eksArn": CPU_EKS_ARN,
                        "hyperpodClusterName": CPU_NAME,
                    },
                    "nlb": {"name": NLB_NAME},
                    "clusters": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _arguments(*extra: str) -> Any:
    return admin_cli.parser().parse_args(
        ["hyperpod-scheduled-update", "--gpu-cluster-arn", CPU_HP_ARN, *extra]
    )


def _plan(
    runner: ServiceRunner,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *extra: str,
    alarms: tuple[str, ...] = (CAPACITY, NOT_RESTORED),
    wait: int = WAIT,
) -> tuple[int, dict[str, Any], str]:
    """``--state-dir <site> --rolling --wait-interval-seconds <wait>`` with one
    ``--rollback-alarm`` per alarm, the way the approved schedule was planned."""

    named = [word for name in alarms for word in ("--rollback-alarm", name)]
    arguments = _arguments(
        "--state-dir",
        str(_state_dir(tmp_path)),
        "--rolling",
        "--wait-interval-seconds",
        str(wait),
        *named,
        *extra,
    )
    code = helper.run_command(arguments, runner=runner)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def _model() -> Any:
    return botocore.session.get_session().get_service_model("sagemaker")


def _valid(operation: str, request: dict[str, Any], service: str = "sagemaker") -> None:
    shape = (
        botocore.session.get_session()
        .get_service_model(service)
        .operation_model(operation)
        .input_shape
    )
    report = ParamValidator().validate(request, shape)
    assert not report.has_errors(), report.generate_report()


# --- the request ------------------------------------------------------------------


def test_the_rolling_plan_carries_the_rollback_batch_in_both_places(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, document, err = _plan(
        ServiceRunner(FakeHyperPod(_described())), capsys, tmp_path
    )

    assert code == 0, err
    assert rollback.ROLLING_BATCH == ONE == helper.ROLLING_BATCH
    (spec,) = document["update_cluster"]["instance_groups"]
    assert spec["ScheduledUpdateConfig"] == LIVE_ROLLING, (
        "the request is exactly the schedule the service accepted on 2026-11-02"
    )
    words = shlex.split(document["update_cluster"]["command"])
    assert json.loads(words[words.index("--instance-groups") + 1]) == [spec], (
        "the printed update-cluster carries RollbackMaximumBatchSize too"
    )
    _valid("UpdateCluster", {"ClusterName": CPU_NAME, "InstanceGroups": [spec]})
    manual = shlex.split(document["manual_update"][CPU_GROUP])
    assert manual[:3] == ["aws", "sagemaker", "update-cluster-software"]
    deployment = json.loads(manual[manual.index("--deployment-config") + 1])
    assert deployment == LIVE_ROLLING["DeploymentConfig"], (
        "update-cluster-software takes the same DeploymentConfiguration"
    )
    _valid(
        "UpdateClusterSoftware",
        {
            "ClusterName": CPU_NAME,
            "InstanceGroups": [{"InstanceGroupName": CPU_GROUP}],
            "DeploymentConfig": deployment,
        },
    )


def test_the_api_reference_and_the_model_say_optional_the_service_did_not() -> None:
    """Why the lane's own ParamValidator passed: the installed model (like the
    API reference) requires neither member -- the service does."""

    model = _model()
    assert model.shape_for("RollingDeploymentPolicy").required_members == [
        "MaximumBatchSize"
    ]
    configuration = model.shape_for("DeploymentConfiguration")
    assert configuration.required_members == []
    assert dict(configuration.members["AutoRollbackConfiguration"].metadata) == {
        "min": 1,
        "max": rollback.MAX_ROLLBACK_ALARMS,
    }
    alarm = model.shape_for("AlarmDetails").members["AlarmName"]
    assert alarm.metadata["max"] == rollback.MAX_ALARM_NAME
    b231 = {
        "ScheduleExpression": SCHEDULE,
        "DeploymentConfig": {
            "RollingUpdatePolicy": {"MaximumBatchSize": dict(ONE)},
            "WaitIntervalInSeconds": WAIT,
        },
    }
    _valid(
        "UpdateCluster",
        {
            "ClusterName": CPU_NAME,
            "InstanceGroups": [
                {
                    "InstanceGroupName": CPU_GROUP,
                    "InstanceType": "ml.c5.4xlarge",
                    "InstanceCount": 3,
                    "LifeCycleConfig": {"OnCreate": "x.sh", "SourceS3Uri": "s3://lc/"},
                    "ExecutionRole": ROLE,
                    "ScheduledUpdateConfig": b231,
                }
            ],
        },
    )
    assert [item["at"] for item in rollback.SERVICE_REFUSALS] == [
        "2026-11-02T00:16:46Z",
        "2026-11-02T00:19:01Z",
        "2026-11-02T00:30:16Z",
    ]


def test_a_fake_service_refusing_the_live_requests_accepts_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeHyperPod(_described())
    code, document, err = _plan(ServiceRunner(fake), capsys, tmp_path)
    assert code == 0, err
    (create,) = document["update_cluster"]["instance_groups"]
    first = copy.deepcopy(create)
    deployment = first["ScheduledUpdateConfig"]["DeploymentConfig"]
    del deployment["RollingUpdatePolicy"]["RollbackMaximumBatchSize"]
    del deployment["AutoRollbackConfiguration"]
    second = copy.deepcopy(create)
    del second["ScheduledUpdateConfig"]["DeploymentConfig"]["AutoRollbackConfiguration"]

    for request, refusal in ((first, 0), (second, 1), (create, 2)):
        text = rollback.SERVICE_REFUSALS[refusal]["text"].replace(
            "<instance group>", CPU_GROUP
        )
        with pytest.raises(Refused, match=re.escape(text)):
            fake.update_cluster(CPU_NAME, [request])
    assert fake.accepted == [] and fake.describe() == _described(), (
        "the three requests of 2026-11-02 change nothing"
    )

    for step in document["update_cluster"]["steps"]:
        answer = _execute(fake, step["command"])
        if "expect" in step:
            assert answer == step["expect"], step["action"]

    assert fake.describe() == _described(LIVE_ROLLING), "the live group at 00:34Z"
    assert len(fake.accepted) == 2, "two UpdateCluster calls: delete, then create"


# --- the ordered steps --------------------------------------------------------------


def test_an_existing_schedule_is_deleted_then_created_in_that_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeHyperPod(_described())
    code, document, err = _plan(ServiceRunner(fake), capsys, tmp_path)

    assert code == 0, err
    (record,) = document["instance_groups"]
    assert record["replaces"] == LIVE_BEFORE, "the schedule step 1 deletes"
    update = document["update_cluster"]
    steps = update["steps"]
    assert [(step["step"], step["action"]) for step in steps] == [
        (1, "delete"),
        (2, "verify-deleted"),
        (3, "create"),
        (4, "verify"),
    ]
    (create,) = update["instance_groups"]
    (delete,) = steps[0]["instance_groups"]
    assert delete == {k: v for k, v in create.items() if k != "ScheduledUpdateConfig"}
    _code, removal, _err = _plan_disable(fake, capsys, tmp_path)
    assert removal["update_cluster"]["instance_groups"] == [delete], (
        "step 1 is the disable definition"
    )
    assert steps[2]["command"] == update["command"], "the final definition's command"
    assert steps[3]["command"] == update["verify"]
    assert _execute(fake, steps[1]["command"]) != steps[1]["expect"], (
        "before step 1 the schedule is still there"
    )
    with pytest.raises(Refused, match="You can't modify the existing"):
        _execute(fake, steps[2]["command"])
    for step in steps:
        answer = _execute(fake, step["command"])
        if "expect" in step:
            assert answer == step["expect"], step["action"]
    marks = [err.index(f"step {step['step']} {step['action']}") for step in steps]
    assert marks == sorted(marks), "stderr lists the steps in order"
    assert (
        "between step 1 and step 3 the group has no ScheduledUpdateConfig -- no "
        "automatic update at all, which is safe; if step 3 fails the group stays "
        "without a schedule, still safe, but run step 3 again"
    ) in update["between_steps"]
    assert update["between_steps"] in err


def _plan_disable(
    fake: FakeHyperPod, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> tuple[int, dict[str, Any], str]:
    arguments = _arguments("--state-dir", str(_state_dir(tmp_path)))
    code = helper.run_command(arguments, runner=ServiceRunner(fake))  # type: ignore[arg-type]
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


@pytest.mark.parametrize(
    "change,wait,alarms",
    [
        ("another wait", 900, (CAPACITY, NOT_RESTORED)),
        ("other alarms", WAIT, (CAPACITY,)),
    ],
)
def test_any_other_deployment_change_is_deleted_and_created_too(
    change: str,
    wait: int,
    alarms: tuple[str, ...],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeHyperPod(_described(LIVE_ROLLING))

    code, document, err = _plan(
        ServiceRunner(fake), capsys, tmp_path, alarms=alarms, wait=wait
    )

    assert code == 0, err
    assert document["instance_groups"][0]["replaces"] == LIVE_ROLLING, change
    for step in document["update_cluster"]["steps"]:
        answer = _execute(fake, step["command"])
        if "expect" in step:
            assert answer == step["expect"], (change, step["action"])
    schedule = fake.describe()["InstanceGroups"][0]["ScheduledUpdateConfig"]
    assert schedule["DeploymentConfig"]["WaitIntervalInSeconds"] == wait
    assert rollback.rollback_alarm_names(schedule) == list(alarms)


def test_the_live_group_that_rolls_this_way_gets_no_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for alarms in ((CAPACITY, NOT_RESTORED), (NOT_RESTORED, CAPACITY)):
        runner = ServiceRunner(FakeHyperPod(_described(LIVE_ROLLING)))

        code, document, err = _plan(runner, capsys, tmp_path, alarms=alarms)

        assert code == 0, err
        assert document["status"] == "NOOP" and "update_cluster" not in document
        (record,) = document["instance_groups"]
        assert (record["outcome"], record["reason"]) == ("none", "rolls-this-way"), (
            f"{alarms}: the alarms are a set"
        )
        assert (
            f"{CPU_NAME}/{CPU_GROUP}: none (rolls this way already: one instance at "
            f"a time {WAIT} s apart, rolled back one instance at a time on alarm "
            f"{CAPACITY} or {NOT_RESTORED})"
        ) in err
        block = document["rollback_alarms"]
        assert block["warnings"] == [] and "refusal" not in block
        assert [item["problems"] for item in block["alarms"]] == [[], []]
        assert block["api_nlb"] == {
            "name": NLB_NAME,
            "load_balancer": LB,
            "target_groups": [TG],
        }
        assert "suggested rollback alarm" not in err, "nothing to fix"
        assert {call[1] for call in runner.calls} <= {
            "describe-cluster",
            "describe-alarms",
            "describe-load-balancers",
            "describe-target-groups",
        }, "reads only"


def test_the_verify_step_reads_and_shows_the_rollback_batch_and_alarms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, document, err = _plan(
        ServiceRunner(FakeHyperPod(_described())), capsys, tmp_path
    )

    assert code == 0, err
    update = document["update_cluster"]
    query = shlex.split(update["verify"])[-1]
    for projection in (
        "RollbackMaximumBatchSize:ScheduledUpdateConfig.DeploymentConfig."
        "RollingUpdatePolicy.RollbackMaximumBatchSize",
        "RollbackAlarms:ScheduledUpdateConfig.DeploymentConfig."
        "AutoRollbackConfiguration[].AlarmName",
        "WaitIntervalInSeconds:ScheduledUpdateConfig.DeploymentConfig."
        "WaitIntervalInSeconds",
    ):
        assert projection in query, projection
    expect = {
        "ClusterStatus": "InService",
        "InstanceGroups": [
            {
                "InstanceGroupName": CPU_GROUP,
                "Status": "InService",
                "ScheduleExpression": SCHEDULE,
                "MaximumBatchSize": ONE,
                "RollbackMaximumBatchSize": ONE,
                "WaitIntervalInSeconds": WAIT,
                "RollbackAlarms": [CAPACITY, NOT_RESTORED],
            }
        ],
    }
    assert update["expect"] == update["steps"][3]["expect"] == expect
    assert jmespath.search(query, _described(LIVE_ROLLING)) == expect, (
        "the live describe at 00:34Z answers it"
    )
    assert jmespath.search(query, _described()) != expect
    assert (
        "step 4 verify (expect "
        + json.dumps(expect, separators=(",", ":"))
        + f"): {update['verify']}"
    ) in err
    assert '"RollbackMaximumBatchSize":{"Type":"INSTANCE_COUNT","Value":1}' in err


def test_the_execution_role_check_is_printed_and_never_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = ServiceRunner(FakeHyperPod(_described()))

    _code, document, err = _plan(runner, capsys, tmp_path)

    role = document["update_cluster"]["execution_role"]
    assert role["requires"] == "cloudwatch:DescribeAlarms"
    assert role["why"] == rollback.AMI_UPDATE_EVIDENCE["quotes"][1]
    assert role["check"] == [
        f"aws iam simulate-principal-policy --policy-source-arn {ROLE} "
        "--action-names cloudwatch:DescribeAlarms --query "
        "'EvaluationResults[].EvalDecision' --output text"
    ]
    assert f"(expect allowed): {role['check'][0]}" in err
    assert all(call[0] != "iam" for call in runner.calls), "printed, never run"


# --- the rollback alarms ------------------------------------------------------------


def test_rolling_without_a_rollback_alarm_is_refused_before_any_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = ServiceRunner(FakeHyperPod(_described()))

    with pytest.raises(BootstrapError) as raised:
        _plan(runner, capsys, tmp_path, alarms=())

    message = str(raised.value)
    assert message.startswith("--rolling needs at least one --rollback-alarm NAME"), (
        message
    )
    assert (
        "UpdateCluster requires an AutoRollbackConfiguration (a list of CloudWatch "
        "alarms) with a rolling DeploymentConfig"
    ) in message
    assert rollback.SERVICE_REFUSALS[1]["text"] in message, "the live refusal"
    assert "No alarm is invented." in message
    assert runner.calls == [], "refused before the describe"


@pytest.mark.parametrize(
    "arguments,refusal",
    [
        (["--rollback-alarm", CAPACITY], "--rollback-alarm needs --rolling"),
        (
            ["--rolling", *[f"--rollback-alarm=a{n}" for n in range(11)]],
            "takes 1 to 10 alarms",
        ),
        (
            ["--rolling", "--rollback-alarm", CAPACITY, "--rollback-alarm", CAPACITY],
            f"names {CAPACITY} more than once",
        ),
        (["--rolling", "--rollback-alarm", "  "], "is not an alarm name"),
        (["--rolling", "--rollback-alarm", "x" * 256], "is not an alarm name"),
    ],
    ids=["without-rolling", "eleven", "twice", "blank", "too-long"],
)
def test_the_alarm_names_are_checked_before_any_read(
    arguments: list[str], refusal: str
) -> None:
    runner = ServiceRunner(FakeHyperPod(_described()))

    with pytest.raises(BootstrapError, match=re.escape(refusal)):
        helper.run_command(_arguments(*arguments), runner=runner)  # type: ignore[arg-type]
    assert runner.calls == []


def test_a_missing_rollback_alarm_refuses_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alarms = _live_alarms()
    del alarms[NOT_RESTORED]

    code, document, err = _plan(
        ServiceRunner(FakeHyperPod(_described()), alarms=alarms), capsys, tmp_path
    )

    assert code == helper.EXIT_REFUSED
    assert document["status"] == "REFUSED" and "update_cluster" not in document
    block = document["rollback_alarms"]
    assert block["refusal"].startswith(
        f"rollback alarm {NOT_RESTORED} does not exist: an AutoRollbackConfiguration "
        "on a missing alarm protects nothing"
    ), block["refusal"]
    assert [item["found"] for item in block["alarms"]] == [True, False]
    assert f"refused: {block['refusal']}" in err
    suggested = block["suggested"]["commands"]
    assert [item["name"] for item in suggested] == [CAPACITY, NOT_RESTORED]
    assert (
        f"suggested rollback alarm {NOT_RESTORED} on the current API NLB (never run "
        f"by this verb): {suggested[1]['command']}"
    ) in err


def test_an_unreadable_alarm_refuses_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, document, err = _plan(
        ServiceRunner(FakeHyperPod(_described()), deny_alarms=True), capsys, tmp_path
    )

    assert code == helper.EXIT_REFUSED and "update_cluster" not in document
    block = document["rollback_alarms"]
    assert block["state"] == "unknown"
    assert block["refusal"].startswith(
        "cannot read the rollback alarms, so cannot tell that they exist"
    ), block["refusal"]
    assert "operator identity needs cloudwatch:DescribeAlarms" in block["warnings"][0]
    assert "refused: cannot read the rollback alarms" in err


def _cli_request(command: str) -> dict[str, Any]:
    """A printed ``aws cloudwatch put-metric-alarm`` as its API request."""

    words = shlex.split(command)
    assert words[:3] == ["aws", "cloudwatch", "put-metric-alarm"], words
    flags: dict[str, list[str]] = {}
    for word in words[3:]:
        if word.startswith("--"):
            flag = "".join(part.capitalize() for part in word[2:].split("-"))
            flags[flag] = []
        else:
            flags[flag].append(word)
    assert flags.pop("Region") == [REGION]
    request: dict[str, Any] = {key: value[0] for key, value in flags.items()}
    request["Dimensions"] = [
        dict(item.split("=", 1) for item in value.split(","))
        for value in flags["Dimensions"]
    ]
    for key in ("Period", "EvaluationPeriods", "DatapointsToAlarm"):
        request[key] = int(request[key])
    request["Threshold"] = float(request["Threshold"])
    return request


def test_an_alarm_on_the_previous_install_nlb_is_a_warning_with_the_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = _live_alarms(state="INSUFFICIENT_DATA", lb=OLD_LB, tg=OLD_TG)

    code, document, err = _plan(
        ServiceRunner(FakeHyperPod(_described(LIVE_ROLLING)), alarms=stale),
        capsys,
        tmp_path,
    )

    assert code == 0 and document["status"] == "NOOP", (
        "the schedule is right; the alarms are what stopped working"
    )
    block = document["rollback_alarms"]
    (capacity, _restored) = block["alarms"]
    assert capacity["problems"] == [
        "is INSUFFICIENT_DATA: with no data it cannot go to ALARM, so it rolls "
        "nothing back (a load balancer that is gone, a wrong dimension, or a metric "
        "nobody publishes)",
        f"watches AWS/NetworkELB HealthyHostCount (TargetGroup={OLD_TG}, "
        f"LoadBalancer={OLD_LB}), not the current API NLB {LB} (target group {TG}): "
        "after a fresh install the NLB is new and an alarm on the old one has no "
        "data -- rollback protection is off",
    ]
    assert f"warning: rollback alarm {CAPACITY} is INSUFFICIENT_DATA" in err
    commands = block["suggested"]["commands"]
    assert block["suggested"]["note"] == ""
    for item, live in zip(commands, _live_alarms().values(), strict=True):
        request = _cli_request(item["command"])
        _valid("PutMetricAlarm", request, service="cloudwatch")
        same = (
            "AlarmName",
            "Namespace",
            "MetricName",
            "Dimensions",
            "Statistic",
            "Period",
            "EvaluationPeriods",
            "DatapointsToAlarm",
            "Threshold",
            "ComparisonOperator",
            "TreatMissingData",
        )
        assert {key: request[key] for key in same} == {
            key: live[key] for key in same
        }, "the suggestion is the live pair, on the current NLB"
        assert "No alarm actions." in request["AlarmDescription"]
        assert f"suggested rollback alarm {item['name']}" in err


def test_the_recommended_pair_reads_the_maximum_of_healthy_targets() -> None:
    """AWS recommends the maximum of HealthyHostCount; Maximum below X for k of k
    minutes is fewer than X healthy targets for the whole of each minute, where
    Minimum also counts one node's momentary dip (live 2026-11-02: 19 such
    minutes in 14 days against 5)."""

    nlb = {"name": NLB_NAME, "load_balancer": LB, "target_groups": [TG]}
    suggested = rollback.suggested_alarms(
        region=REGION, nlb=nlb, replicas=3, wait_seconds=WAIT
    )

    assert rollback.RECOMMENDED_STATISTIC == "Maximum"
    assert [
        _cli_request(item["command"])["Statistic"] for item in suggested["commands"]
    ] == ["Maximum", "Maximum"]
    assert all(
        "--statistic Maximum" in item["command"] for item in suggested["commands"]
    ), "the printed commands say Maximum"
    assert (
        "It's recommended to monitor maximum `HealthyHostCount`"
        in rollback.NLB_HEALTHY_HOSTS_EVIDENCE["quotes"][-1]
    )
    assert "consensus mechanism" in rollback.NLB_HEALTH_CHECKS_EVIDENCE["quotes"][0]


def test_the_suggestions_follow_the_wait_and_skip_a_wait_too_short() -> None:
    nlb = {"name": NLB_NAME, "load_balancer": LB, "target_groups": [TG]}

    long = rollback.suggested_alarms(
        region=REGION, nlb=nlb, replicas=3, wait_seconds=900
    )
    short = rollback.suggested_alarms(
        region=REGION, nlb=nlb, replicas=3, wait_seconds=600
    )
    none = rollback.suggested_alarms(
        region=REGION,
        nlb={**nlb, "target_groups": [TG, OLD_TG]},
        replicas=3,
        wait_seconds=WAIT,
    )

    assert _cli_request(long["commands"][1]["command"])["EvaluationPeriods"] == 10
    assert [item["name"] for item in short["commands"]] == [CAPACITY]
    assert short["note"].startswith(
        f"no {NOT_RESTORED} suggestion: a wait of 600 s leaves less than 10 min for it"
    ), short["note"]
    assert none["commands"] == [] and "2 target group(s)" in none["note"]


def test_a_cluster_other_than_the_cpu_one_checks_no_nlb(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = ServiceRunner(FakeHyperPod(_described()))
    arguments = _arguments(
        "--instance-group", CPU_GROUP, "--rolling", "--rollback-alarm", CAPACITY
    )

    code = helper.run_command(arguments, runner=runner)  # type: ignore[arg-type]

    document = json.loads(capsys.readouterr().out)
    assert code == 0 and document["selection"] == "--instance-group"
    block = document["rollback_alarms"]
    assert block["api_nlb"] is None and "suggested" not in block
    assert block["warnings"] == [], "no NLB to hold the alarms against"
    assert {call[0] for call in runner.calls} == {"sagemaker", "cloudwatch"}


def test_alarm_records_cover_composite_math_and_missing_alarms() -> None:
    records = rollback.alarm_records(
        {
            "MetricAlarms": [{"AlarmName": "math", "Metrics": [{"Id": "m1"}]}],
            "CompositeAlarms": [
                {"AlarmName": "both", "StateValue": "ALARM", "AlarmRule": "ALARM(a)"}
            ],
        },
        ["math", "both", "gone"],
    )
    nlb = {"load_balancer": LB, "target_groups": [TG]}

    assert [rollback.alarm_problems(item, nlb) for item in records] == [
        ["uses metric math: what it watches is not checked"],
        [
            "is in ALARM now: SageMaker rolls an update back when one of its alarms "
            "is in ALARM during a wait interval",
            "is a composite alarm: its member alarms are not checked",
        ],
        [
            "does not exist (describe-alarms returns no metric or composite alarm of "
            "that name): the update has nothing to roll back on"
        ],
    ]
    assert rollback.alarm_summary(records[1]) == "composite alarm (ALARM(a)), ALARM"


# --- the advisory -------------------------------------------------------------------

CPU_WHERE = f"the CPU control-plane cluster {CPU_NAME}"
SECTION = "(administrator-operations §5.0.5)"


def _block(schedule: dict[str, Any] | None = LIVE_ROLLING) -> dict[str, Any]:
    return updates.control_plane_updates(
        _described(schedule)["InstanceGroups"],
        hyperpod_cluster_name=CPU_NAME,
        cpu_eks_arn=CPU_EKS_ARN,
    )


def test_the_live_rolling_group_is_the_safe_form_with_its_rollback() -> None:
    block = _block()

    assert block["instance_groups"][0]["rolling_batch"] == {
        "type": "INSTANCE_COUNT",
        "value": 1,
        "wait_seconds": WAIT,
        "rollback_batch": {"type": "INSTANCE_COUNT", "value": 1},
        "rollback_alarms": [CAPACITY, NOT_RESTORED],
    }
    (line,) = updates.control_plane_lines(block)
    assert (
        f"advisory: HyperPod instance group {CPU_GROUP} of {CPU_WHERE} has "
        f"ScheduledUpdateConfig {SCHEDULE} rolling 1 instance(s) at a time {WAIT} s "
        f"apart, rolled back 1 instance(s) at a time on alarm {CAPACITY} or "
        f"{NOT_RESTORED}: the safe form -- "
    ) in line, "discovery reads no alarm: the names, no state"


def test_the_advisory_names_the_alarms_and_leaves_judging_them_to_the_check() -> None:
    """The header line names the rollback batch and alarms; whether the alarms
    can trip is the ``cpu_rolling_update`` check's (batch 240), so a block
    carrying an alarm read adds no line of its own."""

    with_read = {
        **_block(),
        "rollback_alarms": {"state": "read", "alarms": [{"name": CAPACITY}]},
    }

    assert updates.control_plane_lines(with_read) == updates.control_plane_lines(
        _block()
    ), "no alarm judgement in the advisory"
