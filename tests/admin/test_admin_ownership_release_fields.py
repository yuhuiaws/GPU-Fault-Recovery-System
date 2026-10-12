"""The HyperPod instance group fields the ownership release carries or reads.

The release re-sends an owned instance group's full definition through
``update-cluster`` and refuses to write when the description carries a field
it cannot classify. Its first tables were written from memory, and the live
cluster's groups carried ``CapacityRequirements``, ``ActiveOperations``,
``TargetStateCount``, ``ImageVersionStatus``, ``SoftwareUpdateStatus`` and the
image release versions -- so the first live release would have refused. This
module pins the tables to the installed botocore service model (the authority
for ``ClusterInstanceGroupDetails`` and ``ClusterInstanceGroupSpecification``)
and to a ``describe-cluster`` document shaped exactly like the live one: the
account, ARNs, names and ids are placeholders, the field set and the value
shapes are the live ones. The live key set is accepted, and the generated
specification round-trips every specification field.
"""

from __future__ import annotations

import copy
from typing import Any

import botocore.session
import pytest

from gpu_fault.admin import ownership_release as release
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin.test_admin_ownership_release import HP_TARGET, REGION, SITE_ID, FakeAws

#: The union of instance group keys the live cluster's ``describe-cluster``
#: shows, the keys of its ``KubernetesConfig`` and the document's top level.
LIVE_INSTANCE_GROUP_KEYS = frozenset(
    {
        "ActiveOperations",
        "CapacityRequirements",
        "CurrentCount",
        "CurrentImageId",
        "CurrentImageReleaseVersion",
        "DesiredImageId",
        "DesiredImageReleaseVersion",
        "ExecutionRole",
        "ImageVersionStatus",
        "InstanceGroupName",
        "InstanceStorageConfigs",
        "InstanceType",
        "KubernetesConfig",
        "LifeCycleConfig",
        "OverrideVpcConfig",
        "ScheduledUpdateConfig",
        "SoftwareUpdateStatus",
        "Status",
        "TargetCount",
        "TargetStateCount",
        "ThreadsPerCore",
    }
)
LIVE_KUBERNETES_CONFIG_KEYS = frozenset(
    {"CurrentLabels", "CurrentTaints", "DesiredLabels", "DesiredTaints"}
)
LIVE_CLUSTER_KEYS = frozenset(
    {
        "ClusterArn",
        "ClusterName",
        "ClusterStatus",
        "CreationTime",
        "InstanceGroups",
        "NodeProvisioningMode",
        "NodeRecovery",
        "Orchestrator",
        "RestrictedInstanceGroups",
        "VpcConfig",
    }
)
#: What the operator's by-hand ``update-cluster`` carried and the service
#: accepted, plus the two image fields that name the running image.
LIVE_SPECIFICATION_KEYS = frozenset(
    {
        "InstanceGroupName",
        "InstanceType",
        "LifeCycleConfig",
        "ExecutionRole",
        "ThreadsPerCore",
        "InstanceStorageConfigs",
        "OverrideVpcConfig",
        "ScheduledUpdateConfig",
        "CapacityRequirements",
        "InstanceCount",
        "KubernetesConfig",
        "ImageId",
        "ImageReleaseVersion",
    }
)
ACCOUNT = "123456789012"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/hyperpod-execution"
LIFECYCLE = {"OnCreate": "on_create.sh", "SourceS3Uri": "s3://lifecycle/"}
STORAGE = [{"EbsVolumeConfig": {"RootVolume": False, "VolumeSizeInGB": 500}}]
VPC = {"SecurityGroupIds": ["sg-example"], "Subnets": ["subnet-example"]}
TAINTS = [{"Key": "nvidia.com/gpu", "Value": "true", "Effect": "NoSchedule"}]


def _live_group(
    name: str, instance_type: str, count: int, schedule: str, **extra: Any
) -> dict[str, Any]:
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
        "InstanceStorageConfigs": copy.deepcopy(STORAGE),
        "InstanceType": instance_type,
        "LifeCycleConfig": dict(LIFECYCLE),
        "ScheduledUpdateConfig": {"ScheduleExpression": schedule},
        "SoftwareUpdateStatus": "Succeeded",
        "Status": "InService",
        "TargetCount": count,
        "TargetStateCount": count,
        "ThreadsPerCore": 1,
        **extra,
    }


#: The live ``describe-cluster`` document, sanitised: an unlabelled general
#: purpose group at zero instances and the site's spot GPU group.
LIVE_DESCRIBE_CLUSTER: dict[str, Any] = {
    "ClusterArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hplive01",
    "ClusterName": HP_TARGET["hyperpod_cluster_name"],
    "ClusterStatus": "InService",
    "CreationTime": "2025-01-01T00:00:00.000000+00:00",
    "InstanceGroups": [
        _live_group("accelerated-g6-4", "ml.g6.12xlarge", 0, "cron(0 16 ? * SUN *)"),
        _live_group(
            "spot-p5en-az3",
            "ml.p5en.48xlarge",
            4,
            "cron(0 12 ? * SUN *)",
            CapacityRequirements={"Spot": {}},
            KubernetesConfig={
                "CurrentLabels": {OWNERSHIP_LABEL: SITE_ID},
                "CurrentTaints": [],
                "DesiredLabels": {OWNERSHIP_LABEL: SITE_ID},
                "DesiredTaints": [],
            },
            OverrideVpcConfig=copy.deepcopy(VPC),
        ),
    ],
    "NodeProvisioningMode": "Continuous",
    "NodeRecovery": "None",
    "Orchestrator": {"Eks": {"ClusterArn": HP_TARGET["eks_cluster_arn"]}},
    "RestrictedInstanceGroups": [],
    "VpcConfig": copy.deepcopy(VPC),
}


def _shape(name: str) -> Any:
    model = botocore.session.get_session().get_service_model("sagemaker")
    return model.shape_for(name)


def _structure(shape: Any) -> Any:
    """What a value of ``shape`` looks like on the wire, shape names aside.

    ``ClusterSlurmConfigDetails`` and ``ClusterSlurmConfig`` are two names for
    one structure; a field is verbatim when the two structures agree member
    by member, not when the model happens to reuse one name.
    """

    if shape.type_name == "structure":
        return {key: _structure(member) for key, member in shape.members.items()}
    if shape.type_name == "list":
        return [_structure(shape.member)]
    if shape.type_name == "map":
        return {_structure(shape.key): _structure(shape.value)}
    return shape.type_name


# --- the live document ----------------------------------------------------------


def test_the_live_fixture_carries_exactly_the_documented_key_sets() -> None:
    groups = LIVE_DESCRIBE_CLUSTER["InstanceGroups"]

    assert set().union(*(set(group) for group in groups)) == LIVE_INSTANCE_GROUP_KEYS
    assert set(groups[1]["KubernetesConfig"]) == LIVE_KUBERNETES_CONFIG_KEYS
    assert set(LIVE_DESCRIBE_CLUSTER) == LIVE_CLUSTER_KEYS
    assert LIVE_INSTANCE_GROUP_KEYS <= set(
        _shape("ClusterInstanceGroupDetails").members
    ), (
        "the installed service model knows every live key; were it older than the "
        "live API, the missing keys would have to be classified by hand"
    )


def test_live_instance_groups_release_without_an_unknown_field_refusal() -> None:
    world = FakeAws(hyperpod=LIVE_DESCRIBE_CLUSTER)

    evidence = release.release_cluster_ownership(
        world,  # type: ignore[arg-type]
        region=REGION,
        site_id=SITE_ID,
        target=HP_TARGET,
        sleep=world.sleep,
    )

    ((spec,),) = world.update_specs
    details = LIVE_DESCRIBE_CLUSTER["InstanceGroups"][1]
    assert set(spec) == LIVE_SPECIFICATION_KEYS, (
        "the by-hand definition the service accepted, plus the running image"
    )
    for key in release.HYPERPOD_SPEC_FIELDS:
        if key in details:
            assert spec[key] == details[key], f"{key} is carried verbatim"
    assert spec["InstanceCount"] == details["TargetCount"]
    assert spec["ImageId"] == "default", "default stays default: no custom AMI pin"
    assert spec["ImageReleaseVersion"] == "2.0.4", (
        "the running release is named; omitted, the service would use the latest"
    )
    assert spec["KubernetesConfig"] == {"Labels": {}, "Taints": []}, (
        "the only label was the ownership label; the empty taints are kept"
    )
    assert set(spec) <= set(_shape("ClusterInstanceGroupSpecification").members), (
        "nothing outside ClusterInstanceGroupSpecification is sent"
    )
    by_name = {item["name"]: item for item in evidence["groups"]}
    assert by_name["spot-p5en-az3"]["outcome"] == release.RELEASED
    assert by_name["spot-p5en-az3"]["target_state_count"] == 4
    assert by_name["accelerated-g6-4"]["outcome"] == release.UNLABELLED
    assert by_name["accelerated-g6-4"]["target_state_count"] == 0
    assert evidence["outcome"] == release.RELEASED
    hyperpod = world.hyperpod
    assert hyperpod is not None
    assert hyperpod["InstanceGroups"][1]["KubernetesConfig"]["CurrentLabels"] == {}


def test_a_live_group_with_a_genuinely_unknown_field_still_refuses() -> None:
    world = FakeAws(hyperpod=LIVE_DESCRIBE_CLUSTER)
    hyperpod = world.hyperpod
    assert hyperpod is not None
    hyperpod["InstanceGroups"][1]["NextYearsField"] = {"x": 1}

    with pytest.raises(BootstrapError, match="NextYearsField"):
        release.release_cluster_ownership(
            world,  # type: ignore[arg-type]
            region=REGION,
            site_id=SITE_ID,
            target=HP_TARGET,
            sleep=world.sleep,
        )

    assert world.update_specs == [], "fail closed on a field the model does not know"


# --- the service model ----------------------------------------------------------


def test_field_tables_classify_every_details_member_of_the_service_model() -> None:
    details = _shape("ClusterInstanceGroupDetails")
    specification = _shape("ClusterInstanceGroupSpecification")
    verbatim = set(release.HYPERPOD_SPEC_FIELDS)
    derived = dict(release.HYPERPOD_DERIVED_FIELDS)
    status = set(release.HYPERPOD_STATUS_FIELDS)

    assert verbatim | set(derived) | status == set(details.members), (
        "every ClusterInstanceGroupDetails member is carried verbatim, derived or "
        "read; a member the installed model adds must be classified here before "
        "the release can carry a group that reports it"
    )
    assert len(verbatim) + len(derived) + len(status) == len(details.members), (
        "no member is in two tables"
    )
    for key in verbatim:
        assert _structure(details.members[key]) == _structure(
            specification.members[key]
        ), f"{key} has one structure on both sides, which is what verbatim means"
    for key in ("KubernetesConfig", "InstanceRequirements", "AutoPatchConfig"):
        assert _structure(details.members[key]) != _structure(
            specification.members[key]
        ), f"{key} describes Current*/Desired* on the details side: it is derived"
    assert verbatim | set(derived.values()) == set(specification.members), (
        "every ClusterInstanceGroupSpecification member can be produced, so the "
        "full definition really is full"
    )
    assert set(specification.required_members) == {
        "InstanceCount",
        "InstanceGroupName",
        "ExecutionRole",
    }
    assert {"InstanceGroupName", "ExecutionRole"} <= set(
        release.HYPERPOD_REQUIRED_FIELDS
    )
    assert set(details.members["SoftwareUpdateStatus"].enum) - (
        release.HYPERPOD_SETTLED_SOFTWARE_UPDATES
    ) == {"Pending", "InProgress", "RollbackInProgress", "Failed"}, (
        "exactly these SoftwareUpdateStatus values refuse the write"
    )
    assert set(details.members["ImageVersionStatus"].enum) == {
        "UpToDate",
        "UpdateAvailable",
        "SecurityUpdateRequired",
        release.HYPERPOD_IMAGE_END_OF_LIFE,
    }, "only EndOfLife refuses; the other three are not an update in flight"
    assert list(details.members["ActiveOperations"].key.enum) == ["Scaling"]


def _every_details_member() -> dict[str, Any]:
    """A description carrying every ``ClusterInstanceGroupDetails`` member, settled."""

    schedule = {"NextPatchDate": "2025-01-05T00:00:00Z"}
    deployment = {
        "RollingUpdatePolicy": {
            "MaximumBatchSize": {"Type": "INSTANCE_COUNT", "Value": 1}
        }
    }
    return {
        **_live_group("flex", "ml.p5en.48xlarge", 2, "cron(0 12 ? * SUN *)"),
        "ActiveSoftwareUpdateConfig": copy.deepcopy(deployment),
        "AutoPatchConfig": {
            "PatchingStrategy": "WhenIdle",
            "CurrentPatchSchedule": dict(schedule),
            "DesiredPatchSchedule": dict(schedule),
            "DeploymentConfig": copy.deepcopy(deployment),
        },
        "CapacityRequirements": {"OnDemand": {}},
        "InstanceRequirements": {
            "CurrentInstanceTypes": ["ml.p5en.48xlarge", "ml.p5.48xlarge"],
            "DesiredInstanceTypes": ["ml.p5en.48xlarge", "ml.p5.48xlarge"],
        },
        "InstanceTypeDetails": [
            {"InstanceType": "ml.p5en.48xlarge", "CurrentCount": 2, "ThreadsPerCore": 1}
        ],
        "KubernetesConfig": {
            "CurrentLabels": {OWNERSHIP_LABEL: SITE_ID, "team": "research"},
            "DesiredLabels": {OWNERSHIP_LABEL: SITE_ID, "team": "research"},
            "CurrentTaints": copy.deepcopy(TAINTS),
            "DesiredTaints": copy.deepcopy(TAINTS),
        },
        "MinCount": 1,
        "NetworkInterface": {"InterfaceType": "efa"},
        "OnStartDeepHealthChecks": ["InstanceStress", "InstanceConnectivity"],
        "OverrideVpcConfig": copy.deepcopy(VPC),
        "SlurmConfig": {"NodeType": "Compute", "PartitionNames": ["gpu"]},
        "TrainingPlanArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-plan/plan",
        "TrainingPlanStatus": "Active",
    }


def test_every_details_member_is_carried_or_derived_into_the_specification() -> None:
    details = _every_details_member()
    assert set(details) == set(_shape("ClusterInstanceGroupDetails").members), (
        "the fixture carries every member the installed model knows"
    )

    spec = release.hyperpod_instance_group_specification(
        details, labels={"team": "research"}
    )

    assert set(spec) == set(_shape("ClusterInstanceGroupSpecification").members), (
        "every specification member is produced and nothing else is sent"
    )
    for key in release.HYPERPOD_SPEC_FIELDS:
        assert spec[key] == details[key], f"{key} is carried verbatim"
    assert spec["InstanceCount"] == 2 and spec["MinInstanceCount"] == 1
    assert spec["ImageId"] == "default" and spec["ImageReleaseVersion"] == "2.0.4"
    assert spec["InstanceRequirements"] == {
        "InstanceTypes": ["ml.p5en.48xlarge", "ml.p5.48xlarge"]
    }, "a flexible group keeps its current instance types, in priority order"
    assert spec["AutoPatchConfig"] == {
        "PatchingStrategy": "WhenIdle",
        "PatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
        "DeploymentConfig": details["AutoPatchConfig"]["DeploymentConfig"],
    }, "the current patch schedule becomes the specification's schedule"
    assert spec["KubernetesConfig"] == {
        "Labels": {"team": "research"},
        "Taints": TAINTS,
    }


def test_a_flexible_group_needs_no_instance_type() -> None:
    details = _every_details_member()
    del details["InstanceType"]

    spec = release.hyperpod_instance_group_specification(details, labels={})

    assert "InstanceType" not in spec
    assert spec["InstanceRequirements"]["InstanceTypes"] == [
        "ml.p5en.48xlarge",
        "ml.p5.48xlarge",
    ]
    del details["InstanceRequirements"]
    with pytest.raises(
        BootstrapError, match="InstanceType \\(or InstanceRequirements\\)"
    ):
        release.hyperpod_instance_group_specification(details, labels={})


def test_a_group_without_a_release_version_or_image_sends_neither() -> None:
    """A group the service describes without image fields (older clusters)."""

    details = _every_details_member()
    for key in (
        "CurrentImageId",
        "DesiredImageId",
        "CurrentImageReleaseVersion",
        "DesiredImageReleaseVersion",
        "ImageVersionStatus",
    ):
        del details[key]

    spec = release.hyperpod_instance_group_specification(details, labels={})

    assert "ImageId" not in spec and "ImageReleaseVersion" not in spec, (
        "nothing is invented: only what the description reports is reproduced"
    )
