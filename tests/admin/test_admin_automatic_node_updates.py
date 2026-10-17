"""Automatic node updates on the site's GPU nodes: recorded and announced, never refused.

The world modelled is the live one, sanitised: the HyperPod cluster ``hp-gpu``
whose site GPU group ``spot-p5en-az3`` (labelled for the site) and whose
unmanaged g6 group ``accelerated-g6-4`` both carry a weekly
``ScheduledUpdateConfig``, and a plain EKS cluster whose Spot GPU node group's
Auto Scaling group has Capacity Rebalance on (EKS sets it) and, in some tests,
a ``MaxInstanceLifetime``, next to Karpenter NodePools with the default 720h
expiry. Every advisory is a recorded fact and a stderr/header line: discovery
returns the same scope, ``node-scope`` the same convergence and exit code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import automatic_node_updates as updates
from gpu_fault.admin import bootstrap_discovery, bootstrap_ownership
from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import node_scope_command as node_scope
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.bootstrap_ownership import RecordedNodeScope
from gpu_fault.admin.site import load_site
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin.test_admin_node_scope_verb import ScopeRunner
from tests.admin.test_admin_site import site_file

SITE_ID = "test-site"
REGION = "us-east-1"
HP_ARN = f"arn:aws:sagemaker:{REGION}:123456789012:cluster/hplive01"
EKS_ARN = f"arn:aws:eks:{REGION}:123456789012:cluster/gpu-eks-cluster"
SPOT_ASG = "eks-gpu-ng-0011aabb-2233-4455-6677-8899aabbccdd"
SITE_SCHEDULE = "cron(0 12 ? * SUN *)"
G6_SCHEDULE = "cron(0 16 ? * SUN *)"
DENIED = (
    "aws autoscaling describe-auto-scaling-groups exited 254: An error occurred "
    "(AccessDenied) when calling the DescribeAutoScalingGroups operation: User: "
    "arn:aws:sts::123456789012:assumed-role/deploy-host is not authorized to "
    "perform: autoscaling:DescribeAutoScalingGroups"
)


def _instance_group(
    name: str, instance_type: str, *, labels: dict[str, str], schedule: str | None
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "InstanceGroupName": name,
        "InstanceType": instance_type,
        "KubernetesConfig": {"CurrentLabels": dict(labels)},
    }
    if schedule:
        group["ScheduledUpdateConfig"] = {"ScheduleExpression": schedule}
    return group


def _hyperpod(**extra_groups: dict[str, Any]) -> dict[str, Any]:
    return {
        "ClusterArn": HP_ARN,
        "ClusterName": "hp-gpu",
        "InstanceGroups": [
            _instance_group(
                "accelerated-g6-4", "ml.g6.12xlarge", labels={}, schedule=G6_SCHEDULE
            ),
            _instance_group(
                "spot-p5en-az3",
                "ml.p5en.48xlarge",
                labels={OWNERSHIP_LABEL: SITE_ID},
                schedule=SITE_SCHEDULE,
            ),
            *extra_groups.values(),
        ],
        "Orchestrator": {"Eks": {"ClusterArn": EKS_ARN}},
    }


def _nodegroup(name: str, *, labels: dict[str, str], asg: str | None) -> dict[str, Any]:
    group: dict[str, Any] = {
        "nodegroupName": name,
        "status": "ACTIVE",
        "capacityType": "SPOT",
        "instanceTypes": ["p5en.48xlarge"],
        "labels": dict(labels),
        "nodeRepairConfig": {"enabled": False},
        "subnets": ["subnet-a"],
    }
    if asg is not None:
        group["resources"] = {"autoScalingGroups": [{"name": asg}]}
    return group


def _asg(name: str, *, lifetime: int | None = None, rebalance: bool = True) -> dict:
    item: dict[str, Any] = {
        "AutoScalingGroupName": name,
        "CapacityRebalance": rebalance,
    }
    if lifetime is not None:
        item["MaxInstanceLifetime"] = lifetime
    return item


def _nodepool(
    name: str,
    *,
    site: str | None = SITE_ID,
    expire: str | None = "720h",
    budgets: list[dict[str, Any]] | None = None,
    v1beta1: bool = False,
) -> dict[str, Any]:
    labels = {OWNERSHIP_LABEL: site} if site else {}
    disruption: dict[str, Any] = {"consolidationPolicy": "WhenEmptyOrUnderutilized"}
    if budgets is not None:
        disruption["budgets"] = budgets
    template: dict[str, Any] = {"metadata": {"labels": labels}, "spec": {}}
    if expire is not None and not v1beta1:
        template["spec"]["expireAfter"] = expire
    if expire is not None and v1beta1:
        disruption["expireAfter"] = expire
    return {
        "metadata": {"name": name},
        "spec": {"template": template, "disruption": disruption},
    }


# --- the facts -------------------------------------------------------------------


def test_a_hyperpod_block_records_the_selected_groups_schedule_and_patching() -> None:
    patched = _instance_group(
        "spot-p5-patched", "ml.p5.48xlarge", labels={}, schedule=None
    )
    patched["AutoPatchConfig"] = {
        "PatchingStrategy": "WhenIdle",
        "CurrentPatchSchedule": {"NextPatchDate": "2025-01-05T00:00:00Z"},
    }
    document = _hyperpod(patched=patched)

    block = updates.hyperpod_updates(
        document["InstanceGroups"],
        selected={"spot-p5en-az3", "spot-p5-patched"},
        gpu_cluster_arn=EKS_ARN,
    )

    assert block == {
        "gpu_cluster_arn": EKS_ARN,
        "instance_groups": [
            {
                "name": "spot-p5en-az3",
                "scheduled_update": SITE_SCHEDULE,
                "auto_patch": None,
            },
            {
                "name": "spot-p5-patched",
                "scheduled_update": None,
                "auto_patch": "WhenIdle",
            },
        ],
    }, "only the selected groups; the unmanaged g6 group is not the site's"


def test_hyperpod_coverage_follows_the_recorded_selector() -> None:
    groups = _hyperpod()["InstanceGroups"]

    labelled = bootstrap_ownership.hyperpod_groups_for_selector(
        groups,
        cluster_name="hp-gpu",
        selector=f"{OWNERSHIP_LABEL}={SITE_ID},sagemaker.amazonaws.com/cluster-name=hp-gpu",
    )
    legacy = bootstrap_ownership.hyperpod_groups_for_selector(
        groups, cluster_name="hp-gpu", selector=""
    )

    assert labelled == ["spot-p5en-az3"], "the ownership label covers its group"
    assert legacy == ["accelerated-g6-4", "spot-p5en-az3"], (
        "a legacy cluster-name selector covers every group of the cluster"
    )


def test_asg_facts_record_lifetime_and_rebalance_and_unknown_when_denied() -> None:
    nodegroups = {
        "gpu-ng": _nodegroup("gpu-ng", labels={}, asg=SPOT_ASG),
        "gpu-ng-2": _nodegroup("gpu-ng-2", labels={}, asg="eks-gone"),
        "no-asg": _nodegroup("no-asg", labels={}, asg=None),
        "cpu-ng": _nodegroup("cpu-ng", labels={}, asg="eks-cpu"),
    }
    reads: list[list[str]] = []

    def read(names: Any) -> dict[str, Any]:
        reads.append(list(names))
        return {
            "AutoScalingGroups": [
                _asg(SPOT_ASG, lifetime=604800),
                _asg("eks-cpu", rebalance=False),
            ]
        }

    facts = updates.eks_node_group_updates(
        nodegroups, selected=["gpu-ng", "gpu-ng-2", "no-asg"], read_asgs=read
    )

    assert reads == [["eks-gone", SPOT_ASG]], "one read for every selected group"
    assert facts == [
        {
            "name": "gpu-ng",
            "auto_scaling_groups": [
                {
                    "name": SPOT_ASG,
                    "max_instance_lifetime": 604800,
                    "capacity_rebalance": True,
                }
            ],
        },
        {
            "name": "gpu-ng-2",
            "auto_scaling_groups": [
                {
                    "name": "eks-gone",
                    "max_instance_lifetime": "unknown",
                    "capacity_rebalance": "unknown",
                    "error": "not returned by DescribeAutoScalingGroups",
                }
            ],
        },
        {"name": "no-asg", "auto_scaling_groups": []},
    ], "the CPU group is not selected and is not reported"

    def denied(_names: Any) -> dict[str, Any]:
        raise BootstrapError(DENIED)

    (group,) = updates.eks_node_group_updates(
        nodegroups, selected=["gpu-ng"], read_asgs=denied
    )
    (asg,) = group["auto_scaling_groups"]
    assert asg["max_instance_lifetime"] == asg["capacity_rebalance"] == "unknown"
    assert "autoscaling:DescribeAutoScalingGroups" in asg["error"], (
        "a denied read is recorded with its reason, never raised"
    )
    cleared = updates.eks_node_group_updates(
        {"gpu-ng": nodegroups["gpu-ng"]},
        selected=["gpu-ng"],
        read_asgs=lambda _names: {"AutoScalingGroups": [_asg(SPOT_ASG, lifetime=0)]},
    )
    assert cleared[0]["auto_scaling_groups"][0]["max_instance_lifetime"] is None, (
        "0 is how a lifetime is cleared: unset"
    )


@pytest.mark.parametrize(
    "budgets,expected",
    [
        (None, False),
        ([{"nodes": "10%"}], False),
        ([{"nodes": "0"}], True),
        ([{"nodes": "0%"}], True),
        (
            [
                {"nodes": "10%"},
                {"nodes": "0", "schedule": "0 4 * * 0", "duration": "166h"},
            ],
            True,
        ),
        ([{"nodes": "0", "reasons": ["Underutilized", "Empty"]}], False),
        ([{"nodes": "0", "reasons": ["Drifted", "Underutilized", "Empty"]}], True),
        ([{"nodes": "1"}], False),
    ],
)
def test_a_zero_budget_allows_no_node_for_any_reason(budgets, expected) -> None:
    assert updates.zero_budget(budgets) is expected, budgets


def test_only_the_sites_nodepools_are_reported() -> None:
    document = {
        "items": [
            _nodepool("gpu-default", budgets=[{"nodes": "10%"}]),
            _nodepool("gpu-pinned", expire="Never", budgets=[{"nodes": "0"}]),
            _nodepool("gpu-beta", expire="336h", v1beta1=True),
            _nodepool("gpu-other-site", site="another-site"),
            _nodepool("general", site=None),
        ]
    }

    pools = updates.nodepool_updates(document, site_id=SITE_ID)

    assert pools == [
        {
            "name": "gpu-beta",
            "expire_after": "336h",
            "budgets": None,
            "zero_budget": False,
        },
        {
            "name": "gpu-default",
            "expire_after": "720h",
            "budgets": [{"nodes": "10%"}],
            "zero_budget": False,
        },
        {
            "name": "gpu-pinned",
            "expire_after": "Never",
            "budgets": [{"nodes": "0"}],
            "zero_budget": True,
        },
    ], "the v1beta1 spec.disruption.expireAfter is read when the v1 field is absent"
    assert updates.karpenter_updates(lambda: None, site_id=SITE_ID) == {
        "state": "absent",
        "nodepools": [],
    }, "a cluster without the Karpenter CRD"

    def broken() -> dict[str, Any]:
        raise BootstrapError("nodepools.karpenter.sh is forbidden")

    assert updates.karpenter_updates(broken, site_id=SITE_ID) == {
        "state": "unknown",
        "nodepools": [],
        "error": "nodepools.karpenter.sh is forbidden",
    }


def test_advisory_lines_name_the_setting_the_remedy_and_the_section() -> None:
    block = {
        "gpu_cluster_arn": EKS_ARN,
        "instance_groups": [
            {
                "name": "spot-p5en-az3",
                "scheduled_update": SITE_SCHEDULE,
                "auto_patch": None,
            },
            {"name": "quiet", "scheduled_update": None, "auto_patch": None},
            {"name": "patched", "scheduled_update": None, "auto_patch": "WhenIdle"},
        ],
        "node_groups": [
            {
                "name": "gpu-ng",
                "auto_scaling_groups": [
                    {
                        "name": SPOT_ASG,
                        "max_instance_lifetime": 604800,
                        "capacity_rebalance": True,
                    },
                    {
                        "name": "eks-b",
                        "max_instance_lifetime": None,
                        "capacity_rebalance": True,
                    },
                    {
                        "name": "eks-c",
                        "max_instance_lifetime": "unknown",
                        "capacity_rebalance": "unknown",
                    },
                ],
            }
        ],
        "karpenter": {
            "state": "read",
            "nodepools": [
                {
                    "name": "gpu",
                    "expire_after": "720h",
                    "budgets": None,
                    "zero_budget": False,
                },
                {
                    "name": "gpu-budget",
                    "expire_after": "Never",
                    "budgets": [{"nodes": "10%"}],
                    "zero_budget": False,
                },
                {
                    "name": "gpu-expires",
                    "expire_after": None,
                    "budgets": [{"nodes": "0"}],
                    "zero_budget": True,
                },
                {
                    "name": "gpu-pinned",
                    "expire_after": "Never",
                    "budgets": [{"nodes": "0"}],
                    "zero_budget": True,
                },
            ],
        },
    }

    lines = updates.advisory_lines("gpu-b", block)

    assert lines == [
        "advisory: HyperPod instance group spot-p5en-az3 of gpu-b has "
        f"ScheduledUpdateConfig {SITE_SCHEDULE}: the service re-images every instance "
        "of the group on that schedule; gpu-fault-admin hyperpod-scheduled-update "
        f"--state-dir <state dir> --gpu-cluster-arn {EKS_ARN} prints the "
        "update-cluster command that removes it; update the group by hand in a "
        "maintenance window (administrator-operations §5.0.5)",
        "advisory: HyperPod instance group patched of gpu-b has AutoPatchConfig "
        "(PatchingStrategy WhenIdle): the service cordons and re-images its "
        "instances on the patch schedule; gpu-fault-admin hyperpod-scheduled-update "
        f"--state-dir <state dir> --gpu-cluster-arn {EKS_ARN} prints the "
        "update-cluster command that removes it (administrator-operations §5.0.5)",
        f"advisory: Auto Scaling group {SPOT_ASG} of EKS node group gpu-ng (gpu-b) "
        "sets MaxInstanceLifetime 604800 s: EC2 Auto Scaling replaces every instance "
        "at that age; clear it with aws autoscaling update-auto-scaling-group "
        f"--auto-scaling-group-name {SPOT_ASG} --max-instance-lifetime 0 "
        "(administrator-operations §5.0.5)",
        "advisory: Karpenter NodePool gpu of gpu-b provisions site GPU nodes with "
        "expireAfter 720h and no zero disruption budget (budgets null): Karpenter "
        "replaces them on expiry, drift and consolidation; set "
        'spec.template.spec.expireAfter: Never and a disruption budget nodes: "0" '
        "(or a maintenance-window schedule budget) (administrator-operations §5.0.5)",
        "advisory: Karpenter NodePool gpu-budget of gpu-b provisions site GPU nodes "
        'with no zero disruption budget (budgets [{"nodes":"10%"}]): Karpenter '
        "replaces them on expiry, drift and consolidation; set "
        'spec.template.spec.expireAfter: Never and a disruption budget nodes: "0" '
        "(or a maintenance-window schedule budget) (administrator-operations §5.0.5)",
        "advisory: Karpenter NodePool gpu-expires of gpu-b provisions site GPU nodes "
        "with no expireAfter (Karpenter's default 720h): Karpenter replaces them on "
        "expiry, drift and consolidation; set spec.template.spec.expireAfter: Never "
        'and a disruption budget nodes: "0" (or a maintenance-window schedule '
        "budget) (administrator-operations §5.0.5)",
    ], (
        "Capacity Rebalance, an unset lifetime, an unknown read and a pinned "
        "NodePool are recorded, never announced"
    )
    assert updates.advisory_lines("gpu-b", None) == []


# --- discovery -------------------------------------------------------------------


def test_hyperpod_discovery_announces_the_selected_groups_and_keeps_its_scope(
    capsys,
) -> None:
    document = _hyperpod()

    joined = bootstrap_ownership.hyperpod_node_scope(
        document, site_id=SITE_ID, recorded=None
    )
    joined_err = capsys.readouterr().err
    legacy = bootstrap_ownership.hyperpod_node_scope(
        document,
        site_id=SITE_ID,
        recorded=RecordedNodeScope("", "provider-default-legacy"),
    )
    legacy_err = capsys.readouterr().err

    assert joined == (
        f"{OWNERSHIP_LABEL}={SITE_ID},sagemaker.amazonaws.com/cluster-name=hp-gpu",
        "label-convention",
    ), "the scope a join resolves is unchanged"
    assert legacy == ("", "provider-default-legacy"), "a recorded scope is kept"
    assert joined_err.splitlines() == [
        "gpu-fault-admin: advisory: HyperPod instance group spot-p5en-az3 of "
        f"HyperPod cluster hp-gpu has ScheduledUpdateConfig {SITE_SCHEDULE}: the "
        "service re-images every instance of the group on that schedule; "
        "gpu-fault-admin hyperpod-scheduled-update --state-dir <state dir> "
        f"--gpu-cluster-arn {HP_ARN} prints the update-cluster command that removes "
        "it; update the group by hand in a maintenance window "
        "(administrator-operations §5.0.5)"
    ], "the unlabelled g6 group is not the site's and is not announced"
    assert "accelerated-g6-4" in legacy_err and "spot-p5en-az3" in (legacy_err), (
        "a legacy cluster-name selector manages every group"
    )


class EksDiscoveryRunner:
    """``list-nodegroups``/``describe-nodegroup`` and the Auto Scaling read."""

    def __init__(self, *, asgs: list[dict[str, Any]] | None, deny: bool = False):
        self.asgs = asgs
        self.deny = deny
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def aws_json(self, region, service, operation, *arguments, **options):
        self.calls.append((service, operation, tuple(arguments)))
        assert region == REGION and not options.get("mutate"), "a read-only read"
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": ["cpu-ng", "gpu-ng"]}
        if (service, operation) == ("eks", "describe-nodegroup"):
            name = arguments[arguments.index("--nodegroup-name") + 1]
            if name == "cpu-ng":
                group = _nodegroup("cpu-ng", labels={}, asg="eks-cpu")
                group["instanceTypes"] = ["m7i-flex.xlarge"]
                return {"nodegroup": group}
            return {
                "nodegroup": _nodegroup(
                    "gpu-ng", labels={OWNERSHIP_LABEL: SITE_ID}, asg=SPOT_ASG
                )
            }
        if (service, operation) == ("autoscaling", "describe-auto-scaling-groups"):
            assert arguments[arguments.index("--query") + 1] == updates.ASG_QUERY
            if self.deny:
                raise BootstrapError(DENIED)
            return {"AutoScalingGroups": copy.deepcopy(self.asgs or [])}
        raise AssertionError(f"unexpected call {service} {operation}")


@pytest.mark.parametrize("deny", [False, True])
def test_eks_discovery_announces_a_set_lifetime_and_a_denied_read_changes_nothing(
    deny: bool, capsys
) -> None:
    runner = EksDiscoveryRunner(asgs=[_asg(SPOT_ASG, lifetime=1209600)], deny=deny)

    selected, scope = bootstrap_discovery.gpu_nodegroups(
        runner,  # type: ignore[arg-type]
        region=REGION,
        eks_name="gpu-eks-cluster",
        site_id=SITE_ID,
    )

    assert [item["nodegroupName"] for item in selected] == ["gpu-ng"]
    assert scope.selector == f"{OWNERSHIP_LABEL}={SITE_ID}"
    reads = [call for call in runner.calls if call[0] == "autoscaling"]
    assert len(reads) == 1 and SPOT_ASG in reads[0][2], (
        "one read, for the selected group's Auto Scaling group only"
    )
    assert "eks-cpu" not in reads[0][2]
    err = capsys.readouterr().err
    if deny:
        assert "advisory" not in err, "an unknown lifetime is recorded, never announced"
    else:
        assert err.splitlines() == [
            f"gpu-fault-admin: advisory: Auto Scaling group {SPOT_ASG} of EKS node "
            "group gpu-ng (EKS cluster gpu-eks-cluster) sets MaxInstanceLifetime "
            "1209600 s: EC2 Auto Scaling replaces every instance at that age; clear "
            "it with aws autoscaling update-auto-scaling-group "
            f"--auto-scaling-group-name {SPOT_ASG} --max-instance-lifetime 0 "
            "(administrator-operations §5.0.5)"
        ]


def test_an_identity_only_eks_read_does_not_read_auto_scaling_groups() -> None:
    runner = EksDiscoveryRunner(asgs=[_asg(SPOT_ASG, lifetime=1209600)])

    bootstrap_discovery.gpu_nodegroups(
        runner,  # type: ignore[arg-type]
        region=REGION,
        eks_name="gpu-eks-cluster",
        site_id=None,
        recorded=RecordedNodeScope("", ""),
    )

    assert not [call for call in runner.calls if call[0] == "autoscaling"], (
        "remove-cluster and rotate-token resolve no scope and read nothing more"
    )


# --- node-scope --------------------------------------------------------------------


class UpdatesScopeRunner(ScopeRunner):
    """``ScopeRunner`` plus the Auto Scaling read and the Karpenter NodePool list."""

    def __init__(self, *, asgs=None, deny=False, nodepools=None, **kwargs):
        super().__init__(**kwargs)
        self.asgs = asgs
        self.deny = deny
        self.nodepools = nodepools

    def aws_json(self, region, service, operation, *arguments, **options):
        if (service, operation) == ("autoscaling", "describe-auto-scaling-groups"):
            self.commands.append(["aws", service, operation, *arguments])
            if self.deny:
                raise BootstrapError(DENIED)
            return {"AutoScalingGroups": copy.deepcopy(self.asgs or [])}
        return super().aws_json(region, service, operation, *arguments, **options)

    def run(self, arguments, **options):
        if updates.KARPENTER_NODEPOOLS in arguments:
            self.commands.append(list(arguments))
            if self.nodepools is None:
                raise BootstrapError(
                    "kubectl get exited 1: error: the server doesn't have a "
                    'resource type "nodepools"'
                )
            return json.dumps({"items": self.nodepools})
        return super().run(arguments, **options)


def _eks_site(tmp_path: Path) -> Path:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    entry = {
        **document["spec"]["clusters"][0],
        "clusterId": "gpu-eks-cluster",
        "context": "gpu-eks",
        "provider": "eks-managed",
        "nodeSelector": f"{OWNERSHIP_LABEL}={SITE_ID}",
        "nodeSelectorSource": "label-convention",
        "eksClusterArn": EKS_ARN,
    }
    entry.pop("hyperpodClusterName")
    document["spec"]["clusters"] = [entry]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def _node_scope_arguments(state_dir: Path):
    return admin_cli.parser().parse_args(["node-scope", "--state-dir", str(state_dir)])


@pytest.mark.parametrize("deny", [False, True])
def test_node_scope_reports_the_eks_block_and_keeps_convergence_and_exit_code(
    tmp_path: Path, capsys, deny: bool
) -> None:
    labels = {"eks.amazonaws.com/nodegroup": "gpu-ng", OWNERSHIP_LABEL: SITE_ID}
    runner = UpdatesScopeRunner(
        asgs=[_asg(SPOT_ASG, lifetime=1209600)],
        deny=deny,
        nodepools=[_nodepool("gpu", budgets=None)],
        nodegroups={
            "cpu-ng": {"instanceTypes": ["m7i-flex.xlarge"], "labels": {}},
            "gpu-ng": {
                "instanceTypes": ["p5en.48xlarge"],
                "labels": {OWNERSHIP_LABEL: SITE_ID},
                "resources": {"autoScalingGroups": [{"name": SPOT_ASG}]},
            },
        },
        nodes={"gpu-eks": [{"metadata": {"name": "ip-10-0-38-180", "labels": labels}}]},
    )
    site = load_site(_eks_site(tmp_path))

    code = node_scope.run_node_scope_command(
        _node_scope_arguments(tmp_path), site=site, runner=runner
    )

    assert code == 0, "advisories never change the exit code"
    captured = capsys.readouterr()
    (cluster,) = json.loads(captured.out)["clusters"]
    assert cluster["convergence"] == "converged", "nor the convergence verdict"
    block = cluster["automatic_node_updates"]
    (group,) = block["node_groups"]
    assert group["name"] == "gpu-ng", "the selected node group only"
    (asg,) = group["auto_scaling_groups"]
    assert block["karpenter"]["state"] == "read"
    assert [pool["name"] for pool in block["karpenter"]["nodepools"]] == ["gpu"]
    nodepool_line = (
        "gpu-fault-admin node-scope: advisory: Karpenter NodePool gpu of "
        "gpu-eks-cluster provisions site GPU nodes with expireAfter 720h and no zero "
        "disruption budget"
    )
    assert nodepool_line in captured.err
    lifetime_line = f"advisory: Auto Scaling group {SPOT_ASG} of EKS node group gpu-ng"
    if deny:
        assert asg["max_instance_lifetime"] == "unknown"
        assert asg["capacity_rebalance"] == "unknown"
        assert "AccessDenied" in asg["error"]
        assert lifetime_line not in captured.err
    else:
        assert asg == {
            "name": SPOT_ASG,
            "max_instance_lifetime": 1209600,
            "capacity_rebalance": True,
        }
        assert lifetime_line in captured.err


def test_node_scope_records_a_cluster_without_karpenter_as_absent(
    tmp_path: Path, capsys
) -> None:
    labels = {"eks.amazonaws.com/nodegroup": "gpu-ng", OWNERSHIP_LABEL: SITE_ID}
    runner = UpdatesScopeRunner(
        asgs=[_asg(SPOT_ASG)],
        nodepools=None,
        nodegroups={
            "gpu-ng": {
                "instanceTypes": ["p5en.48xlarge"],
                "labels": {OWNERSHIP_LABEL: SITE_ID},
                "resources": {"autoScalingGroups": [{"name": SPOT_ASG}]},
            }
        },
        nodes={"gpu-eks": [{"metadata": {"name": "ip-10-0-38-180", "labels": labels}}]},
    )

    code = node_scope.run_node_scope_command(
        _node_scope_arguments(tmp_path),
        site=load_site(_eks_site(tmp_path)),
        runner=runner,
    )

    assert code == 0
    captured = capsys.readouterr()
    (cluster,) = json.loads(captured.out)["clusters"]
    assert cluster["automatic_node_updates"]["karpenter"] == {
        "state": "absent",
        "nodepools": [],
    }
    assert cluster["automatic_node_updates"]["node_groups"][0][
        "auto_scaling_groups"
    ] == [
        {"name": SPOT_ASG, "max_instance_lifetime": None, "capacity_rebalance": True}
    ], "Capacity Rebalance (EKS sets it on Spot groups) is recorded"
    assert "advisory" not in captured.err, "nothing to announce"


def test_node_scope_reports_the_hyperpod_schedule_from_the_describe_it_reads(
    tmp_path: Path, capsys
) -> None:
    runner = UpdatesScopeRunner(
        hyperpod_groups=[
            _instance_group(
                "spot-p5en",
                "ml.p5en.48xlarge",
                labels={OWNERSHIP_LABEL: SITE_ID},
                schedule=SITE_SCHEDULE,
            ),
            _instance_group(
                "accelerated-g6", "ml.g6.12xlarge", labels={}, schedule=G6_SCHEDULE
            ),
        ],
        nodes={"gpu-a": []},
        secrets={
            "gpu-a": {
                "node-selector": "sagemaker.amazonaws.com/cluster-name=hp-gpu-a",
                "node-selector-source": "provider-default-legacy",
            }
        },
    )

    code = node_scope.run_node_scope_command(
        _node_scope_arguments(tmp_path),
        site=load_site(site_file(tmp_path)),
        runner=runner,
    )

    assert code == 0
    captured = capsys.readouterr()
    (cluster,) = json.loads(captured.out)["clusters"]
    block = cluster["automatic_node_updates"]
    assert block["instance_groups"] == [
        {"name": "spot-p5en", "scheduled_update": SITE_SCHEDULE, "auto_patch": None},
        {"name": "accelerated-g6", "scheduled_update": G6_SCHEDULE, "auto_patch": None},
    ], "the legacy cluster-name selector manages both groups"
    assert not [command for command in runner.commands if "autoscaling" in command], (
        "HyperPod reads nothing new"
    )
    assert (
        "gpu-fault-admin node-scope: advisory: HyperPod instance group spot-p5en of "
        f"gpu-a has ScheduledUpdateConfig {SITE_SCHEDULE}"
    ) in captured.err


def test_a_broken_advisory_is_unknown_and_node_scope_still_succeeds(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_arguments: Any, **_options: Any) -> dict[str, Any]:
        raise RuntimeError("advisory defect")

    monkeypatch.setattr(node_scope, "automatic_updates_report", broken)
    runner = UpdatesScopeRunner(
        hyperpod_groups=[
            _instance_group(
                "spot-p5en",
                "ml.p5en.48xlarge",
                labels={OWNERSHIP_LABEL: SITE_ID},
                schedule=SITE_SCHEDULE,
            )
        ],
        nodes={"gpu-a": []},
    )

    code = node_scope.run_node_scope_command(
        _node_scope_arguments(tmp_path),
        site=load_site(site_file(tmp_path)),
        runner=runner,
    )

    assert code == 0
    (cluster,) = json.loads(capsys.readouterr().out)["clusters"]
    assert cluster["automatic_node_updates"] == {
        "state": "unknown",
        "error": "advisory defect",
    }


def test_guarded_returns_the_block_or_unknown() -> None:
    assert updates.guarded(lambda: {"instance_groups": []}) == {"instance_groups": []}

    def broken() -> dict[str, Any]:
        raise KeyError("InstanceGroups")

    assert updates.guarded(broken) == {"state": "unknown", "error": "'InstanceGroups'"}


def test_advisory_lines_skip_a_malformed_block_instead_of_failing() -> None:
    block = {
        "instance_groups": ["not-a-group", None],
        "node_groups": [{"name": "gpu-ng", "auto_scaling_groups": "bad"}],
        "karpenter": {"nodepools": [7]},
    }

    assert updates.advisory_lines("gpu-b", block) == [], "a status header never fails"
    assert updates.advisory_lines("gpu-b", {"state": "unknown", "error": "x"}) == []
