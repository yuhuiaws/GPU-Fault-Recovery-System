"""``gpu-fault-admin hyperpod-scheduled-update``: the command that removes a
HyperPod instance group's automatic updates, printed, never run.

The cluster modelled is the live one, sanitised: ``spot-p5en-az3`` (4 x
``ml.p5en.48xlarge`` on Spot, labelled for the site, image release 2.0.4,
``cron(0 12 ? * SUN *)``) and the unmanaged g6 group
``accelerated-g6-4`` (``cron(0 16 ? * SUN *)``). The definition the verb
prints is the ownership release's full definition minus
``ScheduledUpdateConfig``/``AutoPatchConfig`` -- the ownership label stays --
and the release's refusals hold.
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path
from typing import Any

import botocore.session
import pytest
import yaml
from botocore.validate import ParamValidator

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import hyperpod_scheduled_update as helper
from gpu_fault.admin import ownership_release as release
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_host_binding import is_readonly_command
from gpu_fault.cluster_scope import OWNERSHIP_LABEL

SITE_ID = "us-west-2-hp-site-1a2b3c4d"
REGION = "us-west-2"
ACCOUNT = "123456789012"
HP_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hplive01"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/hp-gpu-eks"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/hyperpod-execution"
VPC = {"SecurityGroupIds": ["sg-example"], "Subnets": ["subnet-example"]}
SITE_LABELS = {OWNERSHIP_LABEL: SITE_ID, "team": "training"}
SITE_GROUP = "spot-p5en-az3"
G6_GROUP = "accelerated-g6-4"


def _group(name: str, instance_type: str, count: int, schedule: str, **extra: Any):
    return {
        "ActiveOperations": {},
        "CurrentCount": count,
        "CurrentImageId": "default",
        "CurrentImageReleaseVersion": "2.0.4",
        "DesiredImageId": "default",
        "DesiredImageReleaseVersion": "2.0.4",
        "ExecutionRole": ROLE,
        "ImageVersionStatus": "UpToDate",
        "InstanceGroupName": name,
        "InstanceStorageConfigs": [
            {"EbsVolumeConfig": {"RootVolume": False, "VolumeSizeInGB": 500}}
        ],
        "InstanceType": instance_type,
        "LifeCycleConfig": {"OnCreate": "on_create.sh", "SourceS3Uri": "s3://lc/"},
        "ScheduledUpdateConfig": {"ScheduleExpression": schedule},
        "SoftwareUpdateStatus": "Succeeded",
        "Status": "InService",
        "TargetCount": count,
        "TargetStateCount": count,
        "ThreadsPerCore": 1,
        **extra,
    }


def _described() -> dict[str, Any]:
    return {
        "ClusterArn": HP_ARN,
        "ClusterName": "hp-gpu",
        "ClusterStatus": "InService",
        "CreationTime": "2025-01-01T00:00:00.000000+00:00",
        "InstanceGroups": [
            _group(G6_GROUP, "ml.g6.12xlarge", 0, "cron(0 16 ? * SUN *)"),
            _group(
                SITE_GROUP,
                "ml.p5en.48xlarge",
                4,
                "cron(0 12 ? * SUN *)",
                CapacityRequirements={"Spot": {}},
                KubernetesConfig={
                    "CurrentLabels": dict(SITE_LABELS),
                    "CurrentTaints": [],
                    "DesiredLabels": dict(SITE_LABELS),
                    "DesiredTaints": [],
                },
                OverrideVpcConfig=copy.deepcopy(VPC),
            ),
        ],
        "NodeProvisioningMode": "Continuous",
        "NodeRecovery": "None",
        "Orchestrator": {"Eks": {"ClusterArn": EKS_ARN}},
        "RestrictedInstanceGroups": [],
        "VpcConfig": copy.deepcopy(VPC),
    }


#: The alarm the --rolling tests name; ``describe-alarms`` finds it, OK.
ALARM = "hp-gpu-reimage"


def _alarm(name: str, state: str = "OK") -> dict[str, Any]:
    return {
        "AlarmName": name,
        "StateValue": state,
        "Namespace": "AWS/NetworkELB",
        "MetricName": "HealthyHostCount",
        "Statistic": "Maximum",
        "Dimensions": [],
        "Period": 60,
        "EvaluationPeriods": 5,
        "DatapointsToAlarm": 5,
        "Threshold": 2.0,
        "ComparisonOperator": "LessThanThreshold",
        "TreatMissingData": "missing",
    }


class DescribeRunner:
    """``describe-cluster`` (and the region inventory) and ``describe-alarms``
    only; any write fails."""

    def __init__(
        self,
        described: dict[str, Any] | None = None,
        *,
        hyperpods=None,
        alarms: dict[str, dict[str, Any]] | None = None,
    ):
        self.described = described if described is not None else _described()
        self.hyperpods = hyperpods
        self.alarms = alarms if alarms is not None else {ALARM: _alarm(ALARM)}
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def aws_json(self, region, service, operation, *arguments, **options):
        self.calls.append((service, operation, tuple(arguments)))
        assert not options.get("mutate"), "the verb writes nothing"
        assert region == REGION
        if (service, operation) == ("cloudwatch", "describe-alarms"):
            names = arguments[1 : arguments.index("--alarm-types")]
            return {
                "MetricAlarms": [self.alarms[n] for n in names if n in self.alarms],
                "CompositeAlarms": [],
            }
        if (service, operation) == ("sagemaker", "describe-cluster"):
            selected = arguments[arguments.index("--cluster-name") + 1]
            assert selected in {HP_ARN, "hp-gpu"}, selected
            return copy.deepcopy(self.described)
        if (service, operation) == ("sagemaker", "list-clusters"):
            return {
                "ClusterSummaries": [
                    {
                        "ClusterName": item["ClusterName"],
                        "ClusterArn": item["ClusterArn"],
                    }
                    for item in (
                        self.hyperpods
                        if self.hyperpods is not None
                        else [self.described]
                    )
                ]
            }
        raise AssertionError(f"unexpected call {service} {operation}")


def _arguments(*extra: str, arn: str = HP_ARN):
    return admin_cli.parser().parse_args(
        ["hyperpod-scheduled-update", "--gpu-cluster-arn", arn, *extra]
    )


def _run(runner: DescribeRunner, capsys, *extra: str, arn: str = HP_ARN):
    code = helper.run_command(_arguments(*extra, arn=arn), runner=runner)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def _shape(name: str) -> Any:
    model = botocore.session.get_session().get_service_model("sagemaker")
    return model.shape_for(name)


# --- the definition -------------------------------------------------------------


def test_the_definition_is_the_ownership_release_one_minus_the_schedule(capsys) -> None:
    code, document, err = _run(DescribeRunner(), capsys)

    assert code == 0, err
    assert document["status"] == "PLAN"
    (record,) = document["instance_groups"]
    assert record["name"] == SITE_GROUP, "the labelled group; the g6 group is not ours"
    assert record["outcome"] == "disable"
    assert record["removes"] == ["ScheduledUpdateConfig"]
    assert record["ownership_label"] == SITE_ID
    details = _described()["InstanceGroups"][1]
    (spec,) = document["update_cluster"]["instance_groups"]
    expected = release.hyperpod_instance_group_specification(
        details, labels=dict(SITE_LABELS)
    )
    expected.pop("ScheduledUpdateConfig")
    assert spec == expected, "the ownership release's builder, minus the schedule"
    for key in release.HYPERPOD_SPEC_FIELDS:
        if key in details and key != "ScheduledUpdateConfig":
            assert spec[key] == details[key], f"{key} is carried verbatim"
    assert "ScheduledUpdateConfig" not in spec and "AutoPatchConfig" not in spec
    assert spec["InstanceCount"] == 4, "InstanceCount is TargetCount"
    assert (spec["ImageId"], spec["ImageReleaseVersion"]) == ("default", "2.0.4"), (
        "the running image is named: an omitted release version is the latest"
    )
    assert spec["CapacityRequirements"] == {"Spot": {}}
    assert spec["KubernetesConfig"] == {"Labels": SITE_LABELS, "Taints": []}, (
        "every label stays, the ownership label included"
    )
    assert set(spec) <= set(_shape("ClusterInstanceGroupSpecification").members)
    request = {"ClusterName": "hp-gpu", "InstanceGroups": [spec]}
    report = ParamValidator().validate(
        request,
        botocore.session.get_session()
        .get_service_model("sagemaker")
        .operation_model("UpdateCluster")
        .input_shape,
    )
    assert not report.has_errors(), report.generate_report()
    words = shlex.split(document["update_cluster"]["command"])
    assert words[:7] == [
        "aws",
        "sagemaker",
        "update-cluster",
        "--region",
        REGION,
        "--cluster-name",
        "hp-gpu",
    ]
    assert words[7] == "--instance-groups" and json.loads(words[8]) == [spec], (
        "the printed command carries exactly the definition in the JSON"
    )
    assert (
        "ScheduledUpdateConfig:ScheduledUpdateConfig"
        in document["update_cluster"]["verify"]
    )
    assert document["update_cluster"]["expect"] == {
        "ClusterStatus": "InService",
        "InstanceGroups": [
            {
                "InstanceGroupName": SITE_GROUP,
                "Status": "InService",
                "ScheduledUpdateConfig": None,
                "AutoPatchConfig": None,
            }
        ],
    }, "the verify read answers exactly this once the removal applied"
    assert [step["action"] for step in document["update_cluster"]["steps"]] == [
        "remove",
        "verify",
    ], "one UpdateCluster removes a schedule (accepted live 2026-11-02)"
    assert (
        "never while gpu-fault-admin config or deploy runs"
        in (document["update_cluster"]["run_when"])
    )
    assert document["manual_update"] == {
        SITE_GROUP: (
            f"aws sagemaker update-cluster-software --region {REGION} --cluster-name "
            f"hp-gpu --instance-groups InstanceGroupName={SITE_GROUP} "
            '--deployment-config \'{"RollingUpdatePolicy":{"MaximumBatchSize":'
            '{"Type":"INSTANCE_COUNT","Value":1},"RollbackMaximumBatchSize":'
            '{"Type":"INSTANCE_COUNT","Value":1}},"WaitIntervalInSeconds":<seconds>,'
            '"AutoRollbackConfiguration":[{"AlarmName":<rollback alarm>}]}\''
        )
    }, "one instance at a time and rolled back so; the placeholders keep it invalid"
    assert "rollback_alarms" not in document, "removal names no alarm"
    assert (
        f"run in a maintenance window: {document['update_cluster']['command']}" in err
    )


def test_omitting_the_member_is_the_only_request_that_can_mean_no_schedule() -> None:
    """What the installed service model settles offline: only CreateCluster and
    UpdateCluster carry the member, and an empty value is not a valid request.
    How the service reads the omission the model does not say; on the live CPU
    group (2026-11-02) it removed the schedule -- the verify read the verb
    prints shows it."""

    model = botocore.session.get_session().get_service_model("sagemaker")

    def carries(shape: Any, seen: frozenset[str] = frozenset()) -> bool:
        if shape is None or shape.name in seen:
            return False
        if shape.name in {"ScheduledUpdateConfig", "ClusterAutoPatchConfig"}:
            return True
        seen = seen | {shape.name}
        children = (
            list(shape.members.values())
            if shape.type_name == "structure"
            else [shape.member]
            if shape.type_name == "list"
            else [shape.key, shape.value]
            if shape.type_name == "map"
            else []
        )
        return any(carries(child, seen) for child in children)

    carriers = sorted(
        name
        for name in model.operation_names
        if carries(model.operation_model(name).input_shape)
    )
    assert carriers == ["CreateCluster", "UpdateCluster"], (
        "no other operation can set or clear a schedule"
    )
    base = release.hyperpod_instance_group_specification(
        _described()["InstanceGroups"][1], labels=dict(SITE_LABELS)
    )
    base.pop("ScheduledUpdateConfig")
    shape = model.operation_model("UpdateCluster").input_shape

    def errors(**extra: Any) -> str:
        report = ParamValidator().validate(
            {"ClusterName": "hp-gpu", "InstanceGroups": [{**base, **extra}]}, shape
        )
        return report.generate_report() if report.has_errors() else ""

    assert errors() == "", "the definition without the member is valid"
    assert "Missing required parameter" in errors(ScheduledUpdateConfig={})
    assert "valid min length: 1" in errors(
        ScheduledUpdateConfig={"ScheduleExpression": ""}
    )
    assert "Missing required parameter" in errors(AutoPatchConfig={})


def test_automatic_patching_is_removed_with_the_schedule(capsys) -> None:
    described = _described()
    described["InstanceGroups"][1]["AutoPatchConfig"] = {
        "PatchingStrategy": "WhenIdle",
        "CurrentPatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
        "DesiredPatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
    }

    code, document, _err = _run(DescribeRunner(described), capsys)

    assert code == 0
    (record,) = document["instance_groups"]
    assert record["removes"] == ["ScheduledUpdateConfig", "AutoPatchConfig"]
    (spec,) = document["update_cluster"]["instance_groups"]
    assert "AutoPatchConfig" not in spec and "ScheduledUpdateConfig" not in spec


def test_a_group_without_automatic_updates_needs_no_command(capsys) -> None:
    described = _described()
    del described["InstanceGroups"][1]["ScheduledUpdateConfig"]

    code, document, err = _run(DescribeRunner(described), capsys)

    assert code == 0
    assert document["status"] == "NOOP" and "update_cluster" not in document
    assert document["instance_groups"][0]["outcome"] == "none"
    assert f"hp-gpu/{SITE_GROUP}: none (no automatic update)" in err


# --- refusals ---------------------------------------------------------------------


def _with(**changes: Any) -> dict[str, Any]:
    described = _described()
    group = described["InstanceGroups"][1]
    for key, value in changes.items():
        if key == "ClusterStatus":
            described[key] = value
        elif key == "DesiredLabels":
            group["KubernetesConfig"]["DesiredLabels"] = value
        else:
            group[key] = value
    return described


@pytest.mark.parametrize(
    "described,reason",
    [
        (_with(ClusterStatus="Updating"), "the cluster is Updating"),
        (_with(Status="Updating"), "the instance group is Updating"),
        (_with(ActiveOperations={"Scaling": 1}), "ActiveOperations"),
        (_with(SoftwareUpdateStatus="InProgress"), "software update is InProgress"),
        (_with(CurrentCount=3), "CurrentCount 3 differs from TargetCount 4"),
        (_with(DesiredImageReleaseVersion="2.0.5"), "DesiredImageReleaseVersion 2.0.5"),
        (_with(DesiredLabels={"team": "training"}), "DesiredLabels"),
        (_with(ImageVersionStatus="EndOfLife"), "EndOfLife"),
        (_with(NextYearsField={"x": 1}), "NextYearsField"),
    ],
    ids=[
        "cluster-updating",
        "group-updating",
        "active-operations",
        "software-update",
        "scaling",
        "pending-image-release",
        "ownership-release-converging",
        "end-of-life",
        "unknown-field",
    ],
)
def test_the_ownership_release_refusals_hold(
    described: dict[str, Any], reason: str, capsys
) -> None:
    code, document, err = _run(DescribeRunner(described), capsys)

    assert code == helper.EXIT_REFUSED
    assert document["status"] == "REFUSED"
    assert "update_cluster" not in document, "no command is printed at all"
    (record,) = document["instance_groups"]
    assert record["outcome"] == "refused" and reason in record["refusal"], record
    assert "run in a maintenance window" not in err


def test_the_ownership_release_still_tolerates_its_own_converging_release() -> None:
    details = _with(DesiredLabels={"team": "training"})["InstanceGroups"][1]

    assert release.hyperpod_write_refusal("InService", details) is None, (
        "the release's own pending removal of the label is not a refusal there"
    )
    assert "DesiredLabels" in str(
        release.hyperpod_write_refusal("InService", details, allow_label_release=False)
    )


# --- selection --------------------------------------------------------------------


def test_an_instance_group_can_be_named_and_an_unknown_one_is_refused(capsys) -> None:
    code, document, _err = _run(DescribeRunner(), capsys, "--instance-group", G6_GROUP)

    assert code == 0
    (record,) = document["instance_groups"]
    assert record["name"] == G6_GROUP and record["ownership_label"] is None
    assert document["selection"] == "--instance-group"
    (spec,) = document["update_cluster"]["instance_groups"]
    assert spec["InstanceCount"] == 0 and "KubernetesConfig" in spec
    with pytest.raises(BootstrapError, match="has no instance group nope"):
        _run(DescribeRunner(), capsys, "--instance-group", "nope")


def _state_dir(tmp_path: Path, clusters: list[dict[str, Any]]) -> Path:
    (tmp_path / "site.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.io/v1",
                "kind": "RegionalSite",
                "metadata": {"name": SITE_ID},
                "spec": {"clusters": clusters},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_a_managed_site_selects_what_its_recorded_selector_covers(
    tmp_path: Path, capsys
) -> None:
    state = _state_dir(
        tmp_path,
        [
            {
                "clusterId": "hp-gpu",
                "eksClusterArn": EKS_ARN,
                "hyperpodClusterName": "hp-gpu",
            }
        ],
    )
    runner = DescribeRunner()

    code, document, _err = _run(runner, capsys, "--state-dir", str(state), arn=EKS_ARN)

    assert code == 0
    assert [item["name"] for item in document["instance_groups"]] == [
        G6_GROUP,
        SITE_GROUP,
    ], "a legacy (unrecorded) HyperPod selector manages every group"
    assert document["selection"] == "the site's recorded node selector"
    assert [call[:2] for call in runner.calls] == [("sagemaker", "describe-cluster")], (
        "the site entry names the HyperPod cluster: one describe, no inventory"
    )


def test_a_site_that_does_not_manage_the_cluster_selects_its_labelled_groups(
    tmp_path: Path, capsys
) -> None:
    state = _state_dir(tmp_path, [])

    code, document, _err = _run(DescribeRunner(), capsys, "--state-dir", str(state))

    assert code == 0
    assert [item["name"] for item in document["instance_groups"]] == [SITE_GROUP]
    assert document["selection"] == f"the {OWNERSHIP_LABEL}={SITE_ID} label"


def test_an_eks_arn_without_a_hyperpod_orchestrator_is_refused(capsys) -> None:
    runner = DescribeRunner(hyperpods=[])

    with pytest.raises(BootstrapError, match="UpdateNodegroupVersion"):
        _run(runner, capsys, arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-eks")


# --- the CLI ----------------------------------------------------------------------


def test_the_verb_is_read_only_and_dispatched_by_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    arguments = _arguments()
    assert is_readonly_command(
        arguments, readonly_commands=admin_cli.READONLY_COMMANDS
    ), "the verb writes nothing: its log rotates under logs/readonly"
    assert "hyperpod-scheduled-update" in admin_cli.READONLY_COMMANDS
    runner = DescribeRunner(_with(ActiveOperations={"Scaling": 1}))
    monkeypatch.setattr(helper, "CommandRunner", lambda: runner)

    code = admin_cli.run(arguments)

    assert code == helper.EXIT_REFUSED, "a refused plan exits 2 through the CLI"
    assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"
    assert all(call[1] == "describe-cluster" for call in runner.calls), (
        "one describe is the whole conversation with the provider"
    )
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(["hyperpod-scheduled-update"])


# --- --rolling: keep the schedule, re-image one instance at a time --------------------
#
# The live CPU control-plane group, sanitised: 3 x ml.c5.4xlarge with
# cron(0 8 ? * SUN *) and no DeploymentConfig -- observed on 2026-11-01 to
# re-image all three control-plane nodes at once. The service then refused a
# rolling schedule without RollbackMaximumBatchSize, without rollback alarms,
# and as an in-place change (2026-11-02): the plan rolls back one instance at a
# time on the named alarms, and deletes the old schedule before creating the
# new one (tests/admin/test_admin_hyperpod_rollback.py runs it against a fake
# service that refuses exactly those).

CPU_NAME = "cpu-control-plane"
CPU_GROUP = "cpu-control-plane-c5"
CPU_HP_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/cpu0live01"
CPU_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/cpu-eks"
CPU_SCHEDULE = "cron(0 8 ? * SUN *)"


def _rolling(
    expression: str, wait: int = 600, alarms: tuple[str, ...] = (ALARM,)
) -> dict[str, Any]:
    return {
        "ScheduleExpression": expression,
        "DeploymentConfig": {
            "RollingUpdatePolicy": {
                "MaximumBatchSize": {"Type": "INSTANCE_COUNT", "Value": 1},
                "RollbackMaximumBatchSize": {"Type": "INSTANCE_COUNT", "Value": 1},
            },
            "WaitIntervalInSeconds": wait,
            "AutoRollbackConfiguration": [{"AlarmName": name} for name in alarms],
        },
    }


def _validated(spec: dict[str, Any], cluster: str) -> None:
    report = ParamValidator().validate(
        {"ClusterName": cluster, "InstanceGroups": [spec]},
        botocore.session.get_session()
        .get_service_model("sagemaker")
        .operation_model("UpdateCluster")
        .input_shape,
    )
    assert not report.has_errors(), report.generate_report()


def test_rolling_keeps_the_schedule_and_rolls_one_instance_at_a_time(capsys) -> None:
    code, document, err = _run(
        DescribeRunner(), capsys, "--rolling", "--rollback-alarm", ALARM
    )

    assert code == 0, err
    assert (document["mode"], document["status"]) == ("rolling", "PLAN")
    (record,) = document["instance_groups"]
    expected = _rolling("cron(0 12 ? * SUN *)")
    assert (record["outcome"], record["rolling_schedule"]) == ("rolling", expected)
    assert record["replaces"] == {"ScheduleExpression": "cron(0 12 ? * SUN *)"}
    assert "removes" not in record
    (spec,) = document["update_cluster"]["instance_groups"]
    details = _described()["InstanceGroups"][1]
    built = release.hyperpod_instance_group_specification(
        details, labels=dict(SITE_LABELS)
    )
    assert spec == {**built, "ScheduledUpdateConfig": expected}, (
        "the ownership release's full definition, the schedule made to roll"
    )
    _validated(spec, "hp-gpu")
    assert (
        "creates the rolling one: the ScheduleExpression kept" in (document["removal"])
    )
    assert (
        f"hp-gpu/{SITE_GROUP}: rolling (keeps ScheduledUpdateConfig cron(0 12 ? * SUN "
        "*), re-imaging one instance at a time 600 s apart, rolled back one instance "
        f"at a time on alarm {ALARM}; the current ScheduledUpdateConfig is deleted "
        "first: the service does not modify one in place)"
    ) in err
    steps = document["update_cluster"]["steps"]
    assert f"step 3 create: {steps[2]['command']}" in err
    assert steps[2]["command"] == document["update_cluster"]["command"], (
        "update_cluster.command stays the final definition's command"
    )


def test_rolling_keeps_auto_patching_and_sets_the_rollback_members(capsys) -> None:
    described = _described()
    group = described["InstanceGroups"][1]
    group["ScheduledUpdateConfig"]["DeploymentConfig"] = {
        "RollingUpdatePolicy": {
            "MaximumBatchSize": {"Type": "CAPACITY_PERCENTAGE", "Value": 50},
            "RollbackMaximumBatchSize": {"Type": "INSTANCE_COUNT", "Value": 2},
        },
        "WaitIntervalInSeconds": 60,
        "AutoRollbackConfiguration": [{"AlarmName": "an-older-alarm"}],
    }
    group["AutoPatchConfig"] = {
        "PatchingStrategy": "WhenIdle",
        "CurrentPatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
        "DesiredPatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
    }

    code, document, err = _run(
        DescribeRunner(described),
        capsys,
        "--rolling",
        "--wait-interval-seconds",
        "900",
        "--rollback-alarm",
        ALARM,
    )

    assert code == 0, err
    (spec,) = document["update_cluster"]["instance_groups"]
    assert spec["ScheduledUpdateConfig"] == _rolling("cron(0 12 ? * SUN *)", 900), (
        "batch, rollback batch, wait and alarms are what --rolling says"
    )
    assert "AutoPatchConfig" in spec, "--rolling touches the schedule only"
    (delete,) = document["update_cluster"]["steps"][0]["instance_groups"]
    assert delete == {k: v for k, v in spec.items() if k != "ScheduledUpdateConfig"}
    assert "AutoPatchConfig" in delete, "the delete step keeps automatic patching"
    _validated(spec, "hp-gpu")
    _validated(delete, "hp-gpu")


def test_a_group_that_rolls_this_way_already_or_has_no_schedule_needs_nothing(
    capsys,
) -> None:
    rolling = _described()
    rolling["InstanceGroups"][1]["ScheduledUpdateConfig"] = _rolling(
        "cron(0 12 ? * SUN *)"
    )
    unscheduled = _described()
    del unscheduled["InstanceGroups"][1]["ScheduledUpdateConfig"]
    unscheduled["InstanceGroups"][1]["AutoPatchConfig"] = {
        "PatchingStrategy": "WhenIdle"
    }

    for described, why in (
        (rolling, "rolls this way already"),
        (unscheduled, "no ScheduledUpdateConfig to roll"),
    ):
        code, document, err = _run(
            DescribeRunner(described), capsys, "--rolling", "--rollback-alarm", ALARM
        )

        assert code == 0
        assert document["status"] == "NOOP" and "update_cluster" not in document
        assert document["instance_groups"][0]["outcome"] == "none"
        assert f"{SITE_GROUP}: none ({why}" in err
    for change in (["--wait-interval-seconds", "300"], ["--rollback-alarm", "other"]):
        runner = DescribeRunner(
            rolling, alarms={n: _alarm(n) for n in (ALARM, "other")}
        )
        _code, document, _err = _run(
            runner, capsys, "--rolling", "--rollback-alarm", ALARM, *change
        )
        assert document["instance_groups"][0]["outcome"] == "rolling", (
            f"{change} is a change: deleted and created again"
        )


def test_the_refusals_hold_for_rolling_too(capsys) -> None:
    code, document, err = _run(
        DescribeRunner(_with(ActiveOperations={"Scaling": 1})),
        capsys,
        "--rolling",
        "--rollback-alarm",
        ALARM,
    )

    assert code == helper.EXIT_REFUSED
    assert document["status"] == "REFUSED" and "update_cluster" not in document
    assert "run in a maintenance window" not in err


@pytest.mark.parametrize("wait", ["-1", "3601"])
def test_the_wait_is_bounded_by_the_service_model(wait: str, capsys) -> None:
    member = _shape("DeploymentConfiguration").members["WaitIntervalInSeconds"]

    assert (member.metadata["min"], member.metadata["max"]) == (
        0,
        helper.MAX_WAIT_SECONDS,
    ), "the installed service model's range"
    with pytest.raises(BootstrapError, match="--wait-interval-seconds must be 0-3600"):
        _run(DescribeRunner(), capsys, "--rolling", "--wait-interval-seconds", wait)


def test_rolling_schedule_keeps_the_expression_and_accepts_no_wait() -> None:
    assert helper.rolling_schedule(
        {"ScheduleExpression": CPU_SCHEDULE}, 0, alarms=[ALARM]
    ) == _rolling(CPU_SCHEDULE, 0)
    with pytest.raises(ValueError, match="at least one alarm"):
        helper.rolling_schedule({"ScheduleExpression": CPU_SCHEDULE}, 0, alarms=[])


def _cpu_described() -> dict[str, Any]:
    return {
        "ClusterArn": CPU_HP_ARN,
        "ClusterName": CPU_NAME,
        "ClusterStatus": "InService",
        "CreationTime": "2025-01-01T00:00:00.000000+00:00",
        "InstanceGroups": [_group(CPU_GROUP, "ml.c5.4xlarge", 3, CPU_SCHEDULE)],
        "NodeProvisioningMode": "Continuous",
        "NodeRecovery": "Automatic",
        "Orchestrator": {"Eks": {"ClusterArn": CPU_EKS_ARN}},
        "VpcConfig": copy.deepcopy(VPC),
    }


class CpuDescribeRunner(DescribeRunner):
    """The CPU control-plane cluster's ``describe-cluster``."""

    def aws_json(self, region, service, operation, *arguments, **options):
        if (service, operation) == ("sagemaker", "describe-cluster"):
            self.calls.append((service, operation, tuple(arguments)))
            assert not options.get("mutate"), "the verb writes nothing"
            selected = arguments[arguments.index("--cluster-name") + 1]
            assert selected in {CPU_HP_ARN, CPU_NAME}, selected
            return copy.deepcopy(self.described)
        return super().aws_json(region, service, operation, *arguments, **options)


def _cpu_state_dir(tmp_path: Path) -> Path:
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
                    "clusters": [
                        {
                            "clusterId": "hp-gpu",
                            "eksClusterArn": EKS_ARN,
                            "hyperpodClusterName": "hp-gpu",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("arn", [CPU_EKS_ARN, CPU_HP_ARN], ids=["eks-arn", "hp-arn"])
def test_the_sites_cpu_cluster_rolls_every_group_from_one_describe(
    arn: str, tmp_path: Path, capsys
) -> None:
    runner = CpuDescribeRunner(_cpu_described())

    code, document, err = _run(
        runner,
        capsys,
        "--state-dir",
        str(_cpu_state_dir(tmp_path)),
        "--rolling",
        "--rollback-alarm",
        ALARM,
        arn=arn,
    )

    assert code == 0, err
    assert document["selection"] == (
        "every instance group of the site's CPU control-plane cluster"
    ), "no ownership label on a CPU group: every group is the control plane's"
    assert [call[:2] for call in runner.calls] == [
        ("sagemaker", "describe-cluster"),
        ("cloudwatch", "describe-alarms"),
    ], "site.yaml names the CPU HyperPod cluster: one describe, no inventory"
    (record,) = document["instance_groups"]
    assert (record["name"], record["outcome"]) == (CPU_GROUP, "rolling")
    (spec,) = document["update_cluster"]["instance_groups"]
    assert spec["ScheduledUpdateConfig"] == _rolling(CPU_SCHEDULE)
    assert spec["InstanceCount"] == 3 and spec["InstanceType"] == "ml.c5.4xlarge"
    _validated(spec, CPU_NAME)
    words = shlex.split(document["update_cluster"]["command"])
    assert words[words.index("--cluster-name") + 1] == CPU_NAME
    assert document["rollback_alarms"]["warnings"] == [
        "the site names no API NLB (spec.nlb.name; pass --state-dir): the alarms' "
        "dimensions were not checked"
    ], "this site.yaml has no spec.nlb: nothing to check the dimensions against"


def test_the_sites_cpu_cluster_can_drop_its_schedule_as_well(
    tmp_path: Path, capsys
) -> None:
    code, document, _err = _run(
        CpuDescribeRunner(_cpu_described()),
        capsys,
        "--state-dir",
        str(_cpu_state_dir(tmp_path)),
        arn=CPU_HP_ARN,
    )

    assert code == 0
    assert document["mode"] == "disable"
    (record,) = document["instance_groups"]
    assert record["removes"] == ["ScheduledUpdateConfig"]
