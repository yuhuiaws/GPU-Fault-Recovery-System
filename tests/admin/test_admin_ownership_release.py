"""Releasing ``gpu-fault.io/control-plane`` when a GPU cluster is de-registered.

The world modelled: site ``test-site`` owns the GPU node group ``gpu-ng`` of
the plain EKS cluster ``gpu-eks`` (the CPU group ``cpu-ng`` was never
labelled) and the instance group ``spot-p5en`` of the HyperPod cluster
``hp-gpu`` (the ``controller`` group was never labelled). ``remove-cluster``
and ``uninstall`` end with the release step this module pins: exactly the
ownership label leaves exactly the groups the site selected, nothing is
written when the provider is mid-update, and a refused write names the manual
command.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import deploy_command
from gpu_fault.admin import ownership_release as release
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_state import SCHEMA_VERSION
from gpu_fault.admin.site import load_site
from gpu_fault.cluster_scope import OWNERSHIP_LABEL, is_eks_managed
from tests.admin.test_admin_site import site_file

SITE_ID = "test-site"
REGION = "us-east-1"
EKS_TARGET = {
    "cluster_id": "gpu-e",
    "provider": "eks-managed",
    "eks_cluster_arn": f"arn:aws:eks:{REGION}:123456789012:cluster/gpu-eks",
    "hyperpod_cluster_name": "",
    "context": "gpu-eks-context",
}
HP_TARGET = {
    "cluster_id": "gpu-a",
    "provider": "hyperpod",
    "eks_cluster_arn": f"arn:aws:eks:{REGION}:123456789012:cluster/gpu-a",
    "hyperpod_cluster_name": "hp-gpu",
    "context": "gpu-a-context",
}
KUBECTL = [
    "kubectl",
    "--kubeconfig",
    "/secure/gpu.kubeconfig",
    "--context",
    "gpu-eks-context",
]
LIFECYCLE = {"SourceS3Uri": "s3://lifecycle/", "OnCreate": "on_create.sh"}
ROLE = "arn:aws:iam::123456789012:role/hyperpod-execution"
TAINTS = [{"Key": "nvidia.com/gpu", "Value": "true", "Effect": "NoSchedule"}]
NODEPOOL_PATCH = '[{"op":"remove","path":"/spec/template/metadata/labels/gpu-fault.io~1control-plane"}]'


def _nodegroup(labels: dict[str, str], *, status: str = "ACTIVE") -> dict[str, Any]:
    return {
        "status": status,
        "instanceTypes": ["p5en.48xlarge"],
        "labels": dict(labels),
    }


def _instance_group(
    name: str,
    labels: dict[str, str],
    *,
    desired: dict[str, str] | None = None,
    instance_type: str = "ml.p5en.48xlarge",
    **overrides: Any,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "InstanceGroupName": name,
        "InstanceType": instance_type,
        "CurrentCount": 2,
        "TargetCount": 2,
        "Status": "InService",
        "LifeCycleConfig": dict(LIFECYCLE),
        "ExecutionRole": ROLE,
        "ThreadsPerCore": 1,
        "InstanceStorageConfigs": [{"EbsVolumeConfig": {"VolumeSizeInGB": 500}}],
        "OnStartDeepHealthChecks": ["InstanceStress"],
        "CurrentImageId": "default",
        "DesiredImageId": "default",
        "CurrentImageReleaseVersion": "2.0.4",
        "DesiredImageReleaseVersion": "2.0.4",
        "ImageVersionStatus": "UpToDate",
        "SoftwareUpdateStatus": "Succeeded",
        "ActiveOperations": {},
        "TargetStateCount": 2,
        "KubernetesConfig": {
            "CurrentLabels": dict(labels),
            "DesiredLabels": dict(labels if desired is None else desired),
            "CurrentTaints": copy.deepcopy(TAINTS),
            "DesiredTaints": copy.deepcopy(TAINTS),
        },
    }
    group.update(overrides)
    return group


class FakeAws:
    """``aws`` and ``kubectl`` answers whose label state the update calls change.

    An EKS release converges after ``converge_after`` describes (the node group
    reports ``UPDATING`` in between); a HyperPod release first shows up in
    ``DesiredLabels`` and reaches ``CurrentLabels`` after ``converge_after``
    describes, as the service applies it asynchronously.
    """

    def __init__(
        self,
        *,
        nodegroups: dict[str, dict[str, Any]] | None = None,
        hyperpod: dict[str, Any] | None = None,
        nodepools: list[dict[str, Any]] | None = None,
        no_karpenter: bool = False,
        converge_after: int = 2,
        deny: frozenset[str] = frozenset(),
        never_converge: bool = False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.nodegroups = copy.deepcopy(nodegroups or {})
        self.hyperpod = copy.deepcopy(hyperpod)
        self.nodepools = nodepools or []
        self.no_karpenter = no_karpenter
        self.converge_after = converge_after
        self.deny = deny
        self.never_converge = never_converge
        self.pending: dict[str, int] = {}
        self.sleeps: list[float] = []
        self.update_specs: list[list[dict[str, Any]]] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    # --- aws ---------------------------------------------------------------

    def aws_json(
        self, region: str, *arguments: str, mutate: bool = False, **_options: Any
    ) -> dict[str, Any]:
        self.calls.append(["aws", *arguments, "--region", region])
        service, operation = arguments[0], arguments[1]
        if operation in self.deny:
            raise BootstrapError(
                f"aws {service} {operation} failed (exit 254): An error occurred "
                f"(AccessDeniedException) when calling the operation: not authorized"
            )
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": sorted(self.nodegroups)}
        if (service, operation) == ("eks", "describe-nodegroup"):
            name = arguments[arguments.index("--nodegroup-name") + 1]
            document = copy.deepcopy(self.nodegroups[name])
            self._tick(name, document["labels"])
            if name in self.pending:
                document["status"] = "UPDATING"
            return {"nodegroup": {"nodegroupName": name, **document}}
        if (service, operation) == ("eks", "update-nodegroup-config"):
            assert mutate, "a label write is a mutation"
            name = arguments[arguments.index("--nodegroup-name") + 1]
            labels = arguments[arguments.index("--labels") + 1]
            assert labels == f"removeLabels={OWNERSHIP_LABEL}", labels
            self.pending[name] = self.converge_after
            return {"update": {"id": "u-1", "status": "InProgress"}}
        if (service, operation) == ("sagemaker", "describe-cluster"):
            assert self.hyperpod is not None, "no HyperPod cluster in this world"
            document = copy.deepcopy(self.hyperpod)
            for group in document["InstanceGroups"]:
                kubernetes = group.get("KubernetesConfig") or {}
                self._tick(group["InstanceGroupName"], kubernetes.get("CurrentLabels"))
            return document
        if (service, operation) == ("sagemaker", "update-cluster"):
            assert mutate, "a label write is a mutation"
            assert self.hyperpod is not None, "no HyperPod cluster in this world"
            specs = json.loads(arguments[arguments.index("--instance-groups") + 1])
            self.update_specs.append(specs)
            for spec in specs:
                group = next(
                    item
                    for item in self.hyperpod["InstanceGroups"]
                    if item["InstanceGroupName"] == spec["InstanceGroupName"]
                )
                group["KubernetesConfig"]["DesiredLabels"] = dict(
                    spec["KubernetesConfig"]["Labels"]
                )
                self.pending[spec["InstanceGroupName"]] = self.converge_after
            return {"ClusterArn": self.hyperpod["ClusterArn"]}
        raise AssertionError(f"unexpected aws call {service} {operation}")

    def _tick(self, name: str, labels: dict[str, str] | None) -> None:
        """Advance a pending release of ``name`` by one describe."""

        if name not in self.pending or self.never_converge:
            return
        self.pending[name] -= 1
        if self.pending[name] <= 0:
            del self.pending[name]
            if name in self.nodegroups:
                self.nodegroups[name]["labels"].pop(OWNERSHIP_LABEL, None)
            if self.hyperpod is not None:
                for group in self.hyperpod["InstanceGroups"]:
                    if group["InstanceGroupName"] == name:
                        group["KubernetesConfig"]["CurrentLabels"].pop(
                            OWNERSHIP_LABEL, None
                        )
        if labels is not None and name not in self.pending:
            labels.pop(OWNERSHIP_LABEL, None)

    # --- kubectl -----------------------------------------------------------

    def run(self, arguments: list[str], **_options: Any) -> str:
        self.calls.append(list(arguments))
        assert arguments[0] == "kubectl" and "get" in arguments, arguments
        assert release.KARPENTER_NODEPOOL_RESOURCE in arguments, arguments
        if self.no_karpenter:
            raise BootstrapError(
                "kubectl failed (exit 1): error: the server doesn't have a resource "
                'type "nodepools"'
            )
        return json.dumps({"items": self.nodepools})


def _eks_world(**kwargs: Any) -> FakeAws:
    return FakeAws(
        nodegroups={
            "gpu-ng": _nodegroup({OWNERSHIP_LABEL: SITE_ID, "node-type": "gpu"}),
            "cpu-ng": _nodegroup({"node-type": "cpu"}),
        },
        no_karpenter=True,
        **kwargs,
    )


def _hyperpod_world(**kwargs: Any) -> FakeAws:
    return FakeAws(
        hyperpod={
            "ClusterName": "hp-gpu",
            "ClusterArn": f"arn:aws:sagemaker:{REGION}:123456789012:cluster/hpgpu01",
            "ClusterStatus": "InService",
            "InstanceGroups": [
                _instance_group(
                    "spot-p5en", {OWNERSHIP_LABEL: SITE_ID, "team": "research"}
                ),
                _instance_group(
                    "controller", {}, instance_type="ml.m5.2xlarge", TargetCount=1
                ),
            ],
        },
        **kwargs,
    )


def _release(world: FakeAws, target: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    kubectl = kwargs.pop("kubectl", KUBECTL if target is EKS_TARGET else None)
    return release.release_cluster_ownership(
        world,  # type: ignore[arg-type]
        region=REGION,
        site_id=SITE_ID,
        target=target,
        kubectl=kubectl,
        sleep=world.sleep,
        **kwargs,
    )


def _updates(world: FakeAws) -> list[list[str]]:
    return [
        call
        for call in world.calls
        if call[0] == "aws" and call[2] in {"update-nodegroup-config", "update-cluster"}
    ]


# --- EKS managed node groups ---------------------------------------------------


def test_eks_release_removes_only_the_ownership_label_and_waits_for_it() -> None:
    world = _eks_world()

    evidence = _release(world, EKS_TARGET)

    assert _updates(world) == [
        [
            "aws",
            "eks",
            "update-nodegroup-config",
            "--cluster-name",
            "gpu-eks",
            "--nodegroup-name",
            "gpu-ng",
            "--labels",
            f"removeLabels={OWNERSHIP_LABEL}",
            "--region",
            REGION,
        ]
    ], "exactly the labelled GPU group is written, with the exact removeLabels call"
    assert world.nodegroups["gpu-ng"]["labels"] == {"node-type": "gpu"}, (
        "every other label of the group stays"
    )
    assert world.nodegroups["cpu-ng"]["labels"] == {"node-type": "cpu"}
    assert evidence["outcome"] == release.RELEASED
    assert evidence["provider"] == "eks-managed" and evidence["cluster"] == "gpu-eks"
    by_name = {item["name"]: item for item in evidence["groups"]}
    assert by_name["gpu-ng"]["before"] == SITE_ID
    assert by_name["gpu-ng"]["after"] is None
    assert by_name["gpu-ng"]["outcome"] == release.RELEASED
    assert by_name["gpu-ng"]["command"] == release.release_command(
        "eks-managed", cluster="gpu-eks", group="gpu-ng"
    )
    assert by_name["cpu-ng"]["outcome"] == release.UNLABELLED
    assert evidence["nodepools"] == [], "no Karpenter CRD on this cluster"
    assert world.sleeps, "the step waited for describe-nodegroup to drop the key"
    assert evidence["converged_after_seconds"] >= 0
    assert evidence["operator_action"] == "eks:UpdateNodegroupConfig"


def test_eks_release_is_idempotent_once_the_label_is_gone() -> None:
    world = FakeAws(
        nodegroups={"gpu-ng": _nodegroup({"node-type": "gpu"})}, no_karpenter=True
    )

    evidence = _release(world, EKS_TARGET)

    assert _updates(world) == [], "nothing to release, nothing written"
    assert evidence["outcome"] == release.RELEASED
    assert evidence["groups"][0]["outcome"] == release.UNLABELLED
    assert world.sleeps == [], "no wait without a write"


def test_eks_release_leaves_another_sites_label_alone() -> None:
    world = FakeAws(
        nodegroups={"gpu-ng": _nodegroup({OWNERSHIP_LABEL: "other-site"})},
        no_karpenter=True,
    )

    evidence = _release(world, EKS_TARGET)

    assert _updates(world) == [], "a label naming another site is not ours to remove"
    assert evidence["groups"][0]["outcome"] == release.FOREIGN
    assert evidence["groups"][0]["before"] == "other-site"
    assert evidence["outcome"] == release.RELEASED


def test_eks_release_refuses_a_node_group_that_is_mid_update() -> None:
    world = FakeAws(
        nodegroups={
            "gpu-ng": _nodegroup({OWNERSHIP_LABEL: SITE_ID}, status="UPDATING")
        },
        no_karpenter=True,
    )

    with pytest.raises(BootstrapError, match="UPDATING"):
        _release(world, EKS_TARGET)

    assert _updates(world) == [], "fail closed: nothing is written mid-update"


def test_eks_release_denied_fails_closed_with_the_manual_command() -> None:
    world = _eks_world(deny=frozenset({"update-nodegroup-config"}))

    with pytest.raises(BootstrapError) as failure:
        _release(world, EKS_TARGET)

    message = str(failure.value)
    assert "AccessDeniedException" in message
    assert "eks:UpdateNodegroupConfig" in message, "names the missing operator action"
    assert release.release_command(
        "eks-managed", cluster="gpu-eks", group="gpu-ng"
    ) in (message), "the exact manual command is in the refusal"
    assert release.KEEP_FLAG in message, "the opt-out is the other way out"
    assert world.nodegroups["gpu-ng"]["labels"][OWNERSHIP_LABEL] == SITE_ID


def test_keep_ownership_label_records_retained_and_writes_nothing() -> None:
    world = _eks_world()

    evidence = _release(world, EKS_TARGET, keep=True)

    assert _updates(world) == []
    assert evidence["outcome"] == release.RETAINED
    assert release.KEEP_FLAG in evidence["reason"]
    by_name = {item["name"]: item for item in evidence["groups"]}
    assert by_name["gpu-ng"]["outcome"] == release.RETAINED
    assert by_name["gpu-ng"]["after"] == SITE_ID
    assert by_name["cpu-ng"]["outcome"] == release.UNLABELLED
    assert world.sleeps == []


def test_release_convergence_is_bounded() -> None:
    world = _eks_world(never_converge=True)
    clock = iter(float(value) for value in range(0, 10_000, 100))

    with pytest.raises(BootstrapError, match="did not converge"):
        _release(
            world,
            EKS_TARGET,
            timeout_seconds=300,
            interval_seconds=100,
            clock=lambda: next(clock),
        )

    assert len(_updates(world)) == 1, "the write happened once; the wait gave up"


# --- HyperPod instance groups --------------------------------------------------


def test_hyperpod_release_sends_the_full_definition_minus_the_label() -> None:
    world = _hyperpod_world()

    evidence = _release(world, HP_TARGET)

    (update,) = _updates(world)
    assert update[:5] == [
        "aws",
        "sagemaker",
        "update-cluster",
        "--cluster-name",
        "hp-gpu",
    ]
    assert update[5] == "--instance-groups"
    (specs,) = world.update_specs
    assert specs == [
        {
            "InstanceGroupName": "spot-p5en",
            "InstanceType": "ml.p5en.48xlarge",
            "InstanceCount": 2,
            "LifeCycleConfig": LIFECYCLE,
            "ExecutionRole": ROLE,
            "ThreadsPerCore": 1,
            "InstanceStorageConfigs": [{"EbsVolumeConfig": {"VolumeSizeInGB": 500}}],
            "OnStartDeepHealthChecks": ["InstanceStress"],
            "ImageId": "default",
            "ImageReleaseVersion": "2.0.4",
            "KubernetesConfig": {"Labels": {"team": "research"}, "Taints": TAINTS},
        }
    ], (
        "the owned group alone, with its full current definition, InstanceCount "
        "equal to TargetCount, the running image named (ImageId and "
        "ImageReleaseVersion) and only the ownership key removed from the labels"
    )
    group = world.hyperpod["InstanceGroups"][0]
    assert group["KubernetesConfig"]["CurrentLabels"] == {"team": "research"}
    assert group["KubernetesConfig"]["CurrentTaints"] == TAINTS
    assert evidence["outcome"] == release.RELEASED
    by_name = {item["name"]: item for item in evidence["groups"]}
    assert by_name["spot-p5en"]["outcome"] == release.RELEASED
    assert by_name["spot-p5en"]["target_state_count"] == 2, (
        "TargetStateCount is informational and recorded, never a refusal"
    )
    assert by_name["controller"]["outcome"] == release.UNLABELLED
    assert evidence["operator_action"] == "sagemaker:UpdateCluster"
    assert world.sleeps, "HyperPod applies the labels asynchronously; the step waited"
    assert not any(call[0] == "kubectl" for call in world.calls), (
        "HyperPod instance groups are the node groups; no Karpenter probe"
    )


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda doc: doc.update(ClusterStatus="Updating"), "Updating"),
        (lambda doc: doc["InstanceGroups"][0].update(Status="Updating"), "Updating"),
        (lambda doc: doc["InstanceGroups"][0].update(CurrentCount=1), "CurrentCount"),
        (
            lambda doc: doc["InstanceGroups"][0]["KubernetesConfig"][
                "DesiredLabels"
            ].update(team="platform"),
            "DesiredLabels",
        ),
        (
            lambda doc: doc["InstanceGroups"][0]["KubernetesConfig"].update(
                DesiredTaints=[]
            ),
            "DesiredTaints",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(DesiredImageId="ami-new"),
            "image",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(FutureField={"x": 1}),
            "FutureField",
        ),
        (lambda doc: doc["InstanceGroups"][0].pop("ExecutionRole"), "ExecutionRole"),
        (
            lambda doc: doc["InstanceGroups"][0].update(
                ActiveOperations={"Scaling": 1}
            ),
            'ActiveOperations {"Scaling": 1}',
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(SoftwareUpdateStatus="Pending"),
            "software update is Pending",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(
                SoftwareUpdateStatus="InProgress"
            ),
            "software update is InProgress",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(
                SoftwareUpdateStatus="RollbackInProgress"
            ),
            "software update is RollbackInProgress",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(SoftwareUpdateStatus="Failed"),
            "software update is Failed (its rollback did not complete",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(
                DesiredImageReleaseVersion="2.1.0"
            ),
            "DesiredImageReleaseVersion 2.1.0 differs from CurrentImageReleaseVersion",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(ImageVersionStatus="EndOfLife"),
            "image release 2.0.4 is EndOfLife",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(
                InstanceRequirements={
                    "CurrentInstanceTypes": ["ml.p5en.48xlarge"],
                    "DesiredInstanceTypes": ["ml.p5.48xlarge"],
                }
            ),
            "InstanceRequirements.DesiredInstanceTypes differs",
        ),
        (
            lambda doc: doc["InstanceGroups"][0].update(
                AutoPatchConfig={
                    "PatchingStrategy": "WhenIdle",
                    "CurrentPatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
                    "DesiredPatchSchedule": {"NextPatchDate": "2025-01-12T00:00:00Z"},
                }
            ),
            "AutoPatchConfig.DesiredPatchSchedule differs",
        ),
    ],
    ids=[
        "cluster-updating",
        "group-updating",
        "counts-differ",
        "other-label-pending",
        "taints-pending",
        "image-pending",
        "unknown-field",
        "missing-required-field",
        "operation-in-flight",
        "software-update-pending",
        "software-update-in-progress",
        "software-rollback-in-progress",
        "software-update-failed",
        "release-version-pending",
        "image-end-of-life",
        "instance-types-pending",
        "patch-schedule-pending",
    ],
)
def test_hyperpod_release_refuses_to_write_when_update_cluster_could_do_more(
    mutate: Any, reason: str
) -> None:
    world = _hyperpod_world()
    mutate(world.hyperpod)

    with pytest.raises(BootstrapError) as failure:
        _release(world, HP_TARGET)

    assert reason in str(failure.value), str(failure.value)
    assert release.release_command("hyperpod", cluster="hp-gpu", group="spot-p5en") in (
        str(failure.value)
    ), "the refusal names the manual command"
    assert _updates(world) == [], "fail closed: UpdateCluster can scale or re-image"


@pytest.mark.parametrize(
    "field, value",
    [
        ("ImageVersionStatus", "UpdateAvailable"),
        ("ImageVersionStatus", "SecurityUpdateRequired"),
        ("SoftwareUpdateStatus", "RollbackComplete"),
    ],
    ids=["update-available", "security-update-required", "rollback-complete"],
)
def test_hyperpod_release_names_the_running_image_when_a_newer_release_exists(
    field: str, value: str
) -> None:
    """A newer release on offer is not an update in flight.

    ``UpdateAvailable`` and ``SecurityUpdateRequired`` grade the running
    release against the catalogue, and ``RollbackComplete`` is a settled
    software update; the write goes through and names the running
    ``ImageId``/``ImageReleaseVersion``, so ``UpdateCluster`` cannot move the
    group to the newer release (an omitted release version would mean "the
    latest available").
    """

    world = _hyperpod_world()
    hyperpod = world.hyperpod
    assert hyperpod is not None
    hyperpod["InstanceGroups"][0][field] = value

    evidence = _release(world, HP_TARGET)

    (specs,) = world.update_specs
    assert specs[0]["ImageId"] == "default"
    assert specs[0]["ImageReleaseVersion"] == "2.0.4"
    assert evidence["groups"][0]["outcome"] == release.RELEASED


def test_hyperpod_in_flight_release_only_waits() -> None:
    world = _hyperpod_world(converge_after=1)
    group = world.hyperpod["InstanceGroups"][0]
    group["KubernetesConfig"]["DesiredLabels"] = {"team": "research"}
    world.pending["spot-p5en"] = 2

    evidence = _release(world, HP_TARGET)

    assert _updates(world) == [], (
        "DesiredLabels already lack the key: the previous attempt's release is "
        "converging, so the step waits instead of writing again"
    )
    assert evidence["groups"][0]["outcome"] == release.RELEASED
    assert group["KubernetesConfig"]["CurrentLabels"] == {"team": "research"}


def test_hyperpod_release_denied_fails_closed_with_the_manual_command() -> None:
    world = _hyperpod_world(deny=frozenset({"update-cluster"}))

    with pytest.raises(BootstrapError) as failure:
        _release(world, HP_TARGET)

    message = str(failure.value)
    assert "sagemaker:UpdateCluster" in message
    assert release.release_command("hyperpod", cluster="hp-gpu", group="spot-p5en") in (
        message
    )
    assert release.KEEP_FLAG in message


# --- Karpenter NodePools -------------------------------------------------------


def _nodepool(name: str, labels: dict[str, str]) -> dict[str, Any]:
    return {
        "metadata": {"name": name},
        "spec": {"template": {"metadata": {"labels": labels}}},
    }


def test_karpenter_nodepool_labels_are_reported_manual_never_edited() -> None:
    world = FakeAws(
        nodegroups={"gpu-ng": _nodegroup({OWNERSHIP_LABEL: SITE_ID})},
        nodepools=[
            _nodepool("gpu-pool", {OWNERSHIP_LABEL: SITE_ID, "tier": "gpu"}),
            _nodepool("cpu-pool", {"tier": "cpu"}),
            _nodepool("other-pool", {OWNERSHIP_LABEL: "another-site"}),
        ],
    )

    evidence = _release(world, EKS_TARGET)

    assert len(_updates(world)) == 1, "the managed node group is still released"
    assert evidence["outcome"] == release.MANUAL
    assert evidence["nodepools"] == [
        {
            "name": "gpu-pool",
            "before": SITE_ID,
            "outcome": release.MANUAL,
            "edit": (
                "kubectl --kubeconfig /secure/gpu.kubeconfig --context gpu-eks-context "
                f"patch nodepool gpu-pool --type=json -p '{NODEPOOL_PATCH}'"
            ),
        }
    ], "only the NodePool carrying this site's label, with the exact edit"
    assert not any("patch" in call for call in world.calls), (
        "customer provisioning CRDs are never edited"
    )


def test_karpenter_nodepools_are_retained_too_under_the_keep_flag() -> None:
    world = FakeAws(
        nodegroups={"gpu-ng": _nodegroup({OWNERSHIP_LABEL: SITE_ID})},
        nodepools=[_nodepool("gpu-pool", {OWNERSHIP_LABEL: SITE_ID})],
    )

    evidence = _release(world, EKS_TARGET, keep=True)

    assert evidence["outcome"] == release.RETAINED
    assert evidence["nodepools"][0]["outcome"] == release.RETAINED
    assert _updates(world) == []


def test_a_kubectl_failure_other_than_a_missing_crd_fails_closed() -> None:
    class Broken(FakeAws):
        def run(self, arguments: list[str], **_options: Any) -> str:
            raise BootstrapError(
                "kubectl failed (exit 1): Unable to connect to the server"
            )

    world = Broken(nodegroups={"gpu-ng": _nodegroup({OWNERSHIP_LABEL: SITE_ID})})

    with pytest.raises(BootstrapError, match="Unable to connect"):
        _release(world, EKS_TARGET)

    assert _updates(world) == [], "the NodePool read comes before any write"


# --- manual commands -----------------------------------------------------------


def test_release_commands_are_the_exact_provider_calls() -> None:
    assert release.release_command(
        "eks-managed", cluster="gpu-eks", group="gpu-ng"
    ) == (
        "aws eks update-nodegroup-config --cluster-name gpu-eks --nodegroup-name "
        f"gpu-ng --labels removeLabels={OWNERSHIP_LABEL}"
    )
    assert release.release_command("hyperpod", cluster="hp-gpu", group="spot-p5en") == (
        "aws sagemaker update-cluster --cluster-name hp-gpu --instance-groups "
        "'<the full current definition of spot-p5en with "
        f"KubernetesConfig.Labels minus {OWNERSHIP_LABEL}>'"
    )
    assert release.nodepool_release_edit(["kubectl", "--context", "c"], "pool") == (
        f"kubectl --context c patch nodepool pool --type=json -p '{NODEPOOL_PATCH}'"
    )
    assert release.OPERATOR_ACTIONS == {
        "eks-managed": "eks:UpdateNodegroupConfig",
        "hyperpod": "sagemaker:UpdateCluster",
    }


# --- stale claims --------------------------------------------------------------

#: The plain EKS cluster of a mixed site as ``site.yaml`` lists it: an explicit
#: provider, no HyperPod name, the selector join-cluster recorded from the
#: labelled node groups. ``EKS_TARGET`` is the same cluster's rendered entry.
EKS_MEMBER = {
    "clusterId": "gpu-e",
    "context": "gpu-eks-context",
    "provider": "eks-managed",
    "eksClusterArn": EKS_TARGET["eks_cluster_arn"],
    "nodeSelector": f"{OWNERSHIP_LABEL}={SITE_ID}",
    "nodeSelectorSource": "label-convention",
}
OTHER_EKS_ARN = f"arn:aws:eks:{REGION}:123456789012:cluster/gpu-eks-old"


def _site_document(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), path
    return document


def _write_site_document(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)


def _mixed_site(tmp_path: Path) -> Path:
    """``site_file``'s HyperPod cluster ``gpu-a`` (no ``provider`` key, as every
    HyperPod entry is written) plus the plain EKS cluster ``gpu-e``."""

    path = site_file(tmp_path)
    document = _site_document(path)
    (hyperpod,) = document["spec"]["clusters"]
    shared = {
        key: value for key, value in hyperpod.items() if key != "hyperpodClusterName"
    }
    document["spec"]["clusters"].append({**shared, **EKS_MEMBER})
    _write_site_document(path, document)
    return path


def _claim_world() -> FakeAws:
    """The mixed site's GPU clusters, each GPU group still carrying the site id."""

    return FakeAws(
        nodegroups={
            "gpu-ng": _nodegroup({OWNERSHIP_LABEL: SITE_ID, "node-type": "gpu"}),
            "cpu-ng": _nodegroup({"node-type": "cpu"}),
        },
        hyperpod={
            "ClusterName": "hp-gpu-a",
            "ClusterArn": f"arn:aws:sagemaker:{REGION}:123456789012:cluster/hp",
            "ClusterStatus": "InService",
            "InstanceGroups": [
                _instance_group("spot-p5en", {OWNERSHIP_LABEL: SITE_ID}),
                _instance_group("controller", {}, instance_type="ml.m5.2xlarge"),
            ],
        },
        no_karpenter=True,
    )


def _tombstone(
    state_dir: Path,
    *,
    released: bool = True,
    legacy: bool = False,
    cluster_id: str = "gpu-a",
    rejoined: bool = False,
    **recorded: Any,
) -> None:
    """The COMPLETED ``remove-cluster`` journal of the site's ``cluster_id``.

    The removal's site commit drops the cluster from ``site.yaml``;
    ``rejoined`` lists it again, as the site reads once the same cluster
    joined again. ``recorded`` overrides fields of the journal's target (what
    the removal recorded differs from today's member: another EKS ARN or
    HyperPod name, another spelling of the provider).
    """

    from gpu_fault.admin.cluster_removal_state import STEP_DEPENDENCIES

    site = load_site(state_dir / "site.yaml")
    (target,) = [
        {**item, **recorded}
        for item in site.release_config["clusters"]
        if item["cluster_id"] == cluster_id
    ]
    steps = sorted(STEP_DEPENDENCIES)
    if legacy:
        steps = [step for step in steps if step != release.STEP]
    provider_identity = {"eks_arn": target["eks_cluster_arn"]}
    if not is_eks_managed(target):
        provider_identity["hyperpod_arn"] = (
            f"arn:aws:sagemaker:{REGION}:123456789012:cluster/hp"
        )
    evidence: dict[str, Any] = {
        "DISCOVERED": {
            "target": target,
            "provider_identity": provider_identity,
            "namespace_uid": "ns-uid",
            "registry_digest": "d" * 64,
            "token_sha256": "t" * 64,
        }
    }
    if not legacy:
        evidence[release.STEP] = {
            "outcome": release.RELEASED if released else release.RETAINED,
            "groups": [],
        }
    path = state_dir / "remove-cluster" / cluster_id / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "site_id": SITE_ID,
                "cluster_id": cluster_id,
                "site_identity": {
                    key: str(site.release_config[key])
                    for key in ("site_name", "aws_region", "cpu_eks_arn", "namespace")
                },
                "target": target,
                "attempt_id": "a" * 32,
                "phase": "COMPLETED",
                "completed_steps": steps,
                "evidence": evidence,
                "updated_at": "now",
            }
        ),
        encoding="utf-8",
    )
    if not rejoined:
        document = _site_document(state_dir / "site.yaml")
        document["spec"]["clusters"] = [
            item
            for item in document["spec"]["clusters"]
            if item["clusterId"] != cluster_id
        ]
        _write_site_document(state_dir / "site.yaml", document)


def _listed(path: Path, cluster_id: str) -> dict[str, Any]:
    """The ``site.yaml`` entry of ``cluster_id`` as the site lists it now."""

    (entry,) = [
        item
        for item in _site_document(path)["spec"]["clusters"]
        if item["clusterId"] == cluster_id
    ]
    return dict(entry)


def _join(path: Path, entry: dict[str, Any]) -> None:
    """``site.yaml`` lists ``entry`` again, as a join-cluster commit leaves it."""

    document = _site_document(path)
    document["spec"]["clusters"].append(entry)
    _write_site_document(path, document)


def _completed_uninstall(state_dir: Path, *, outcome: str) -> None:
    """``uninstall/state.json`` COMPLETED with its per-cluster release outcome."""

    site = load_site(state_dir / "site.yaml")
    path = state_dir / "uninstall" / "state.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "phase": "COMPLETED",
                "cpu_disposition": "delete",
                "ownership_release": {
                    "label": OWNERSHIP_LABEL,
                    "outcome": outcome,
                    "clusters": [
                        {"cluster_id": item["cluster_id"], "outcome": outcome}
                        for item in site.release_config["clusters"]
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["released", "legacy-journal"])
def test_stale_claims_name_the_removed_clusters_groups_still_carrying_the_label(
    tmp_path: Path, legacy: bool
) -> None:
    path = site_file(tmp_path)
    _tombstone(path.parent, released=True, legacy=legacy)
    world = FakeAws(
        hyperpod={
            "ClusterName": "hp-gpu-a",
            "ClusterArn": f"arn:aws:sagemaker:{REGION}:123456789012:cluster/hp",
            "ClusterStatus": "InService",
            "InstanceGroups": [
                _instance_group("spot-p5en", {OWNERSHIP_LABEL: SITE_ID}),
                _instance_group("controller", {}, instance_type="ml.m5.2xlarge"),
            ],
        }
    )

    findings = release.stale_ownership_claims(load_site(path), world)  # type: ignore[arg-type]

    assert len(findings) == 1
    finding = findings[0]
    assert finding["cluster_id"] == "gpu-a"
    assert finding["deregistered_by"] == "remove-cluster"
    assert finding["groups"] == ["spot-p5en"], "only the group still claiming this site"
    assert finding["release_commands"] == [
        release.release_command("hyperpod", cluster="hp-gpu-a", group="spot-p5en")
    ]
    assert finding["recorded_outcome"] == (None if legacy else release.RELEASED)
    lines = release.stale_claim_lines(findings)
    assert len(lines) == 1 and "stale ownership claim" in lines[0]
    assert "gpu-a" in lines[0] and "spot-p5en" in lines[0]
    assert release.stale_claim_lines([]) == []


def test_no_stale_claim_when_the_removed_clusters_groups_are_clean(
    tmp_path: Path,
) -> None:
    path = site_file(tmp_path)
    _tombstone(path.parent, released=True)
    world = FakeAws(
        hyperpod={
            "ClusterName": "hp-gpu-a",
            "ClusterArn": f"arn:aws:sagemaker:{REGION}:123456789012:cluster/hp",
            "ClusterStatus": "InService",
            "InstanceGroups": [_instance_group("spot-p5en", {"team": "research"})],
        }
    )

    assert release.stale_ownership_claims(load_site(path), world) == []  # type: ignore[arg-type]


def test_no_tombstone_means_no_aws_read(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    world = FakeAws()

    assert release.stale_ownership_claims(load_site(path), world) == []  # type: ignore[arg-type]
    assert world.calls == [], "a site that de-registered nothing reads nothing"


# --- stale claims: a cluster that joined the site again ------------------------


@pytest.mark.parametrize(
    "cluster_id", ["gpu-a", "gpu-e"], ids=["hyperpod", "eks-managed"]
)
def test_a_cluster_that_joined_again_is_no_stale_claim(
    tmp_path: Path, cluster_id: str
) -> None:
    """Live S5: remove-cluster COMPLETED, then the same cluster joined again.

    Its groups carry the label because the site manages them again; the
    finding told the operator to strip it from a registered cluster, after
    which discovery no longer selects those nodes.
    """

    path = _mixed_site(tmp_path)
    _tombstone(path.parent, cluster_id=cluster_id, rejoined=True)
    site = load_site(path)
    world = _claim_world()

    assert release.deregistered_clusters(site) == [], (
        "the re-join supersedes the tombstone of the same cluster"
    )
    assert release.stale_ownership_claims(site, world) == []  # type: ignore[arg-type]
    assert release.status_findings(site, world) == []  # type: ignore[arg-type]
    assert world.calls == [], "a member's node groups are never read as a claim"
    assert (path.parent / "remove-cluster" / cluster_id / "state.json").is_file(), (
        "the tombstone itself stays for the readers that need it"
    )


def test_a_removed_plain_eks_clusters_labelled_group_is_a_stale_claim(
    tmp_path: Path,
) -> None:
    path = _mixed_site(tmp_path)
    _tombstone(path.parent, cluster_id="gpu-e")
    world = _claim_world()

    findings = release.stale_ownership_claims(load_site(path), world)  # type: ignore[arg-type]

    (finding,) = findings
    assert finding["cluster_id"] == "gpu-e"
    assert finding["deregistered_by"] == "remove-cluster"
    assert finding["provider"] == "eks-managed" and finding["cluster"] == "gpu-eks"
    assert finding["group_kind"] == "managed node group"
    assert finding["groups"] == ["gpu-ng"], "the unlabelled cpu-ng claims nothing"
    command = (
        "aws eks update-nodegroup-config --cluster-name gpu-eks --nodegroup-name "
        f"gpu-ng --labels removeLabels={OWNERSHIP_LABEL}"
    )
    assert finding["release_commands"] == [command]
    assert release.stale_claim_lines(findings) == [
        "stale ownership claim: remove-cluster de-registered gpu-e (eks-managed "
        "gpu-eks) but managed node group(s) gpu-ng still carry "
        f"{OWNERSHIP_LABEL}={SITE_ID}; release with {command}"
    ]
    assert not any("describe-cluster" in call for call in world.calls), (
        "the HyperPod member gpu-a is not read as a claim"
    )


@pytest.mark.parametrize(
    ("cluster_id", "recorded", "described"),
    [
        ("gpu-e", {"eks_cluster_arn": OTHER_EKS_ARN}, "gpu-eks-old"),
        ("gpu-a", {"hyperpod_cluster_name": "hp-gpu-old"}, "hp-gpu-old"),
    ],
    ids=["eks-managed-arn", "hyperpod-name"],
)
def test_a_member_reusing_the_cluster_id_for_another_cluster_keeps_the_claim(
    tmp_path: Path, cluster_id: str, recorded: dict[str, str], described: str
) -> None:
    """The id alone never supersedes a tombstone: the removed cluster is another
    provider cluster than today's member, its groups are still a stale claim."""

    path = _mixed_site(tmp_path)
    _tombstone(path.parent, cluster_id=cluster_id, rejoined=True, **recorded)
    world = _claim_world()

    (finding,) = release.stale_ownership_claims(load_site(path), world)  # type: ignore[arg-type]

    assert finding["cluster_id"] == cluster_id
    assert finding["deregistered_by"] == "remove-cluster"
    assert finding["cluster"] == described, "the removed cluster is the one read"
    assert finding["groups"] in (["gpu-ng"], ["spot-p5en"])


@pytest.mark.parametrize(
    "cluster_id", ["gpu-a", "gpu-e"], ids=["hyperpod", "eks-managed"]
)
def test_a_cluster_that_joined_again_under_another_cluster_id_is_no_stale_claim(
    tmp_path: Path, cluster_id: str
) -> None:
    """``join-cluster --cluster-id`` named it differently; the claim check would
    read the same provider cluster, whose groups carry the member's label."""

    path = _mixed_site(tmp_path)
    entry = _listed(path, cluster_id)
    _tombstone(path.parent, cluster_id=cluster_id)
    _join(path, {**entry, "clusterId": f"{cluster_id}-again"})
    site = load_site(path)
    world = _claim_world()

    assert f"{cluster_id}-again" in {
        item["cluster_id"] for item in site.release_config["clusters"]
    }
    assert release.deregistered_clusters(site) == [], (
        "the member reads the tombstone's node groups: superseded"
    )
    assert release.stale_ownership_claims(site, world) == []  # type: ignore[arg-type]
    assert world.calls == [], "a member's node groups are never read as a claim"


@pytest.mark.parametrize(
    "joined_as", ["gpu-a", "gpu-a-again"], ids=["same-cluster-id", "other-cluster-id"]
)
def test_a_hyperpod_cluster_recreated_under_its_name_on_another_eks_is_no_claim(
    tmp_path: Path, joined_as: str
) -> None:
    """``describe-cluster`` reads a HyperPod cluster by its name: the removed
    cluster (orchestrated by another EKS cluster) and the member are one
    describe target, and its instance groups carry the member's label."""

    path = _mixed_site(tmp_path)
    entry = _listed(path, "gpu-a")
    _tombstone(path.parent, cluster_id="gpu-a", eks_cluster_arn=OTHER_EKS_ARN)
    _join(path, {**entry, "clusterId": joined_as})
    site = load_site(path)
    world = _claim_world()

    (member,) = [
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] == joined_as
    ]
    assert member["eks_cluster_arn"] != OTHER_EKS_ARN
    assert member["hyperpod_cluster_name"] == "hp-gpu-a"
    assert release.deregistered_clusters(site) == []
    assert release.stale_ownership_claims(site, world) == []  # type: ignore[arg-type]
    assert world.calls == [], "the member's instance groups are not read as a claim"


@pytest.mark.parametrize(
    "recorded",
    [{}, {"provider": None}, {"provider": ""}, {"provider": "hyperpod"}],
    ids=["absent", "none", "empty", "explicit"],
)
def test_a_hyperpod_journal_matches_its_member_whatever_the_provider_spelling(
    tmp_path: Path, recorded: dict[str, Any]
) -> None:
    """``site.yaml`` writes no ``provider`` for HyperPod (absent means hyperpod)
    and the live tombstone recorded ``provider: None``: both sides compare
    normalized, never as raw fields."""

    path = _mixed_site(tmp_path)
    _tombstone(path.parent, rejoined=True, **recorded)
    site = load_site(path)
    (member,) = [
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] == "gpu-a"
    ]
    journal = json.loads(
        (path.parent / "remove-cluster/gpu-a/state.json").read_text(encoding="utf-8")
    )
    world = _claim_world()

    assert "provider" not in member, "the HyperPod member carries no provider key"
    assert journal["target"].get("provider", "absent") == (
        recorded.get("provider", "absent")
    )
    assert release.stale_ownership_claims(site, world) == []  # type: ignore[arg-type]
    assert world.calls == [], "the HyperPod member is not read as a claim"


def test_an_unreadable_journal_is_reported_even_when_its_cluster_is_a_member(
    tmp_path: Path,
) -> None:
    path = _mixed_site(tmp_path)
    journal = path.parent / "remove-cluster" / "gpu-a" / "state.json"
    journal.parent.mkdir(parents=True)
    journal.write_text("{not json", encoding="utf-8")
    world = _claim_world()

    findings = release.stale_ownership_claims(load_site(path), world)  # type: ignore[arg-type]

    assert release.stale_claim_lines(findings) == [
        "stale ownership claim check: gpu-a could not be read: cannot read "
        "remove-cluster state"
    ], "a journal nobody can read is not vouched for by the member list"
    assert world.calls == []


def test_a_completed_uninstall_reports_every_cluster_the_site_listed(
    tmp_path: Path,
) -> None:
    path = _mixed_site(tmp_path)
    _completed_uninstall(path.parent, outcome=release.RETAINED)
    world = _claim_world()

    findings = release.stale_ownership_claims(load_site(path), world)  # type: ignore[arg-type]

    assert [
        (item["cluster_id"], item["deregistered_by"], item["recorded_outcome"])
        for item in findings
    ] == [("gpu-a", "uninstall", "retained"), ("gpu-e", "uninstall", "retained")]
    assert [item["groups"] for item in findings] == [["spot-p5en"], ["gpu-ng"]]


def test_the_deploy_after_an_uninstall_retires_it_so_a_redeployed_site_has_no_claim(
    tmp_path: Path,
) -> None:
    """The next deploy's first step archives ``uninstall/`` together with
    ``site.yaml`` and the ``remove-cluster`` journals, so a site deployed again
    into the same state directory reads neither as a de-registration."""

    path = _mixed_site(tmp_path)
    redeployed = _site_document(path)
    _tombstone(path.parent, cluster_id="gpu-e")
    _completed_uninstall(path.parent, outcome=release.RETAINED)
    before = release.deregistered_clusters(load_site(path))
    assert [(item["cluster_id"], item["deregistered_by"]) for item in before] == [
        ("gpu-e", "remove-cluster"),
        ("gpu-a", "uninstall"),
    ]

    archive = deploy_command.retire_site_after_uninstall(path.parent)
    _write_site_document(path, redeployed)
    world = _claim_world()

    assert archive is not None and (archive / "uninstall/state.json").is_file(), (
        "the completed uninstall is archived, not lost"
    )
    assert (archive / "remove-cluster/gpu-e/state.json").is_file(), (
        "the earlier tombstone travels with it"
    )
    site = load_site(path)
    assert release.deregistered_clusters(site) == []
    assert release.stale_ownership_claims(site, world) == []  # type: ignore[arg-type]
    assert world.calls == [], "the re-deployed members are not read as claims"
