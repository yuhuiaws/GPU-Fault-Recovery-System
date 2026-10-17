"""The ``regional_contexts`` check records automatic node updates; the verdict stays.

``status --full`` and deploy preflight run the check. Per HyperPod GPU cluster
it now records the ``ScheduledUpdateConfig``/``AutoPatchConfig`` of the
instance groups the recorded selector covers -- read by the describe the check
already makes, its query widened -- and per eks-managed cluster the selected
node groups' Auto Scaling ``MaxInstanceLifetime``/``CapacityRebalance`` and
the site's Karpenter NodePools. A denied or failed read is ``unknown``; the
check passes exactly when it passed before, and the ``status`` header gains
one advisory line per finding.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import jmespath
import pytest

from gpu_fault.admin.automatic_node_updates import ASG_QUERY, FIELD
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from gpu_fault_release import regional_release_preflight as preflight
from gpu_fault_release.regional_admin_commands import status_header_lines
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError

SITE_ID = "us-west-2-site-1a2b3c4d"
REGION = "us-west-2"
ACCOUNT = "123456789012"
CPU_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/cpu"
HP_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/hp-gpu-eks"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-eks-cluster"
SPOT_ASG = "eks-gpu-p5en-efa-spot-node-group-0011aabb"
SCHEDULE = "cron(0 12 ? * SUN *)"
NOT_SERVED = 'error: the server doesn\'t have a resource type "nodepools"'


def _target(cluster_id: str, **fields: Any) -> ClusterTarget:
    return ClusterTarget(
        cluster_id=cluster_id,
        context=f"{cluster_id}-context",
        executor_irsa_role_arn=f"arn:aws:iam::{ACCOUNT}:role/{cluster_id}-executor",
        region=REGION,
        agent_endpoint_allowed_cidrs=("10.0.0.0/16",),
        **fields,
    )


HYPERPOD = _target(
    "gpu-hp",
    hyperpod_cluster_name="hp-gpu",
    eks_cluster_arn=HP_EKS_ARN,
    node_selector=(
        f"{OWNERSHIP_LABEL}={SITE_ID},sagemaker.amazonaws.com/cluster-name=hp-gpu"
    ),
    node_selector_source="label-convention",
)
EKS = _target(
    "gpu-eks",
    hyperpod_cluster_name="",
    eks_cluster_arn=EKS_ARN,
    provider="eks-managed",
    node_selector=f"{OWNERSHIP_LABEL}={SITE_ID}",
    node_selector_source="label-convention",
)


def _live_instance_groups() -> list[dict[str, Any]]:
    """Two ``describe-cluster`` groups shaped like the live ones (full details)."""

    common = {
        "ActiveOperations": {},
        "CurrentImageId": "default",
        "CurrentImageReleaseVersion": "2.0.4",
        "ExecutionRole": f"arn:aws:iam::{ACCOUNT}:role/hyperpod-execution",
        "LifeCycleConfig": {"OnCreate": "on_create.sh", "SourceS3Uri": "s3://lc/"},
        "Status": "InService",
        "ThreadsPerCore": 1,
    }
    return [
        {
            **common,
            "InstanceGroupName": "accelerated-g6-4",
            "InstanceType": "ml.g6.12xlarge",
            "CurrentCount": 0,
            "TargetCount": 0,
            "ScheduledUpdateConfig": {"ScheduleExpression": "cron(0 16 ? * SUN *)"},
        },
        {
            **common,
            "InstanceGroupName": "spot-p5en-az3",
            "InstanceType": "ml.p5en.48xlarge",
            "CurrentCount": 4,
            "TargetCount": 4,
            "CapacityRequirements": {"Spot": {}},
            "KubernetesConfig": {
                "CurrentLabels": {OWNERSHIP_LABEL: SITE_ID},
                "DesiredLabels": {OWNERSHIP_LABEL: SITE_ID},
                "CurrentTaints": [],
                "DesiredTaints": [],
            },
            "ScheduledUpdateConfig": {"ScheduleExpression": SCHEDULE},
        },
    ]


def _describe_cluster(name: str) -> dict[str, Any]:
    """The full ``describe-cluster`` answer; the runner applies the query."""

    return {
        "ClusterName": name,
        "ClusterStatus": "InService",
        "NodeRecovery": "None" if name == "hp-gpu" else "Automatic",
        "Orchestrator": {
            "Eks": {"ClusterArn": HP_EKS_ARN if name == "hp-gpu" else CPU_EKS_ARN}
        },
        "InstanceGroups": _live_instance_groups() if name == "hp-gpu" else [],
        "VpcConfig": {"Subnets": ["subnet-a"]},
    }


class ContextsRunner:
    """Every read ``ensure_region_contexts`` makes, with the AWS CLI ``--query``
    applied (``jmespath``, as the CLI does) so the widened query is exercised."""

    def __init__(
        self,
        *,
        asgs: list[dict[str, Any]] | None = None,
        deny_asg: bool = False,
        nodepools: list[dict[str, Any]] | None = None,
        nodepool_error: str | None = None,
    ) -> None:
        self.asgs = asgs if asgs is not None else []
        self.deny_asg = deny_asg
        self.nodepools = nodepools
        self.nodepool_error = nodepool_error
        self.calls: list[list[str]] = []

    @staticmethod
    def _answer(args: list[str], document: dict[str, Any]) -> str:
        if "--query" in args:
            document = jmespath.search(args[args.index("--query") + 1], document)
        return json.dumps(document)

    def run(self, args: list[str], **_options: Any) -> str:
        self.calls.append(list(args))
        if args[0] == "kubectl":
            context = args[args.index("--context") + 1] if "--context" in args else ""
            if "config" in args and "view" in args:
                return {
                    "": CPU_EKS_ARN,
                    HYPERPOD.context: HP_EKS_ARN,
                    EKS.context: EKS_ARN,
                }[context]
            if "--raw=/readyz" in args:
                return "ok"
            if "nodes" in args:
                return json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "gpu-node-a"},
                                "status": {
                                    "addresses": [
                                        {"type": "InternalIP", "address": "10.0.1.10"}
                                    ]
                                },
                            }
                        ]
                    }
                )
        if args[:3] == ["aws", "sagemaker", "describe-cluster"]:
            name = args[args.index("--cluster-name") + 1]
            return self._answer(args, _describe_cluster(name))
        if args[:3] == ["aws", "eks", "describe-cluster"]:
            return self._answer(args, {"cluster": {"arn": EKS_ARN, "status": "ACTIVE"}})
        if args[:3] == ["aws", "eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": ["cpu-ng", "gpu-ng"]})
        if args[:3] == ["aws", "eks", "describe-nodegroup"]:
            name = args[args.index("--nodegroup-name") + 1]
            gpu = name == "gpu-ng"
            return self._answer(
                args,
                {
                    "nodegroup": {
                        "nodegroupName": name,
                        "status": "ACTIVE",
                        "instanceTypes": ["p5en.48xlarge"] if gpu else ["m7i.xlarge"],
                        "labels": {OWNERSHIP_LABEL: SITE_ID} if gpu else {},
                        "nodeRepairConfig": {"enabled": False},
                        "resources": {
                            "autoScalingGroups": [
                                {"name": SPOT_ASG if gpu else "eks-cpu-ng-00ff"}
                            ]
                        },
                    }
                },
            )
        if args[:3] == ["aws", "autoscaling", "describe-auto-scaling-groups"]:
            if self.deny_asg:
                raise ReleaseError(
                    "aws autoscaling describe-auto-scaling-groups exited 254: "
                    "AccessDenied: not authorized to perform: "
                    "autoscaling:DescribeAutoScalingGroups"
                )
            names = args[args.index("--auto-scaling-group-names") + 1 :]
            names = names[: names.index("--query")]
            response = {
                "AutoScalingGroups": [
                    {**item, "Instances": [{"InstanceId": "i-0abc"}], "MinSize": 2}
                    for item in self.asgs
                    if item["AutoScalingGroupName"] in names
                ]
            }
            return self._answer(args, response)
        raise AssertionError(f"unexpected context read: {args}")

    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        self.calls.append(list(args))
        assert args[:4] == ["kubectl", "--context", EKS.context, "get"], args
        assert timeout_seconds == 60
        if self.nodepool_error is not None:
            return 1, "", self.nodepool_error
        if self.nodepools is None:
            return 1, "", NOT_SERVED
        return 0, json.dumps({"items": self.nodepools}), ""


class ContextsRelease:
    """What ``ensure_region_contexts`` reads off a release: config, runner, the
    kubectl prefixes and the IAM/collector checks (trivial here)."""

    def __init__(self, runner: Any, clusters: tuple[ClusterTarget, ...]) -> None:
        self.runner = runner
        self.config = SimpleNamespace(
            cpu_eks_arn=CPU_EKS_ARN,
            cpu_hyperpod_cluster_name="hp-cpu",
            clusters=clusters,
            aws_region=REGION,
            site_name=SITE_ID,
            namespace="gpu-fault-system",
        )
        self.iam: list[str] = []

    def _cpu(self, *args: str) -> list[str]:
        return ["kubectl", "--kubeconfig", "/secure/cpu.kubeconfig", *args]

    def _gpu(self, target: ClusterTarget, *args: str) -> list[str]:
        return ["kubectl", "--context", target.context, *args]

    def _validate_executor_iam_role(self, target: ClusterTarget) -> None:
        self.iam.append(target.cluster_id)

    def _get_json(self, args: list[str]) -> dict[str, Any]:
        return {"items": []}


def _nodepool(name: str, *, expire: str, budgets: list[dict[str, Any]] | None) -> dict:
    disruption: dict[str, Any] = {}
    if budgets is not None:
        disruption["budgets"] = budgets
    return {
        "metadata": {"name": name},
        "spec": {
            "template": {
                "metadata": {"labels": {OWNERSHIP_LABEL: SITE_ID}},
                "spec": {"expireAfter": expire},
            },
            "disruption": disruption,
        },
    }


def _contexts(runner: ContextsRunner) -> dict[str, Any]:
    release = ContextsRelease(runner, (HYPERPOD, EKS))
    facts = preflight.ensure_region_contexts(release)  # type: ignore[arg-type]
    assert release.iam == ["gpu-hp", "gpu-eks"], "every gate still ran"
    return facts


# --- the queries -------------------------------------------------------------------


def test_the_widened_hyperpod_query_still_binds_and_carries_the_group_fields() -> None:
    projected = jmespath.search(
        preflight.HYPERPOD_CONTEXT_QUERY, _describe_cluster("hp-gpu")
    )

    assert projected["EksClusterArn"] == HP_EKS_ARN
    assert projected["NodeRecovery"] == "None"
    assert [set(item) for item in projected["InstanceGroups"]] == [
        {
            "InstanceGroupName",
            "InstanceType",
            "KubernetesConfig",
            "ScheduledUpdateConfig",
            "AutoPatchConfig",
        }
    ] * 2
    assert projected["InstanceGroups"][1]["ScheduledUpdateConfig"] == {
        "ScheduleExpression": SCHEDULE
    }


def test_the_auto_scaling_query_keeps_one_json_object_without_the_instances() -> None:
    projected = jmespath.search(
        ASG_QUERY,
        {
            "AutoScalingGroups": [
                {
                    "AutoScalingGroupName": SPOT_ASG,
                    "CapacityRebalance": True,
                    "Instances": [{"InstanceId": "i-0abc"}],
                    "MinSize": 2,
                }
            ]
        },
    )

    assert projected == {
        "AutoScalingGroups": [
            {
                "AutoScalingGroupName": SPOT_ASG,
                "MaxInstanceLifetime": None,
                "CapacityRebalance": True,
            }
        ]
    }


# --- the check ---------------------------------------------------------------------


def test_contexts_record_each_clusters_automatic_updates_beside_its_facts() -> None:
    runner = ContextsRunner(
        asgs=[
            {
                "AutoScalingGroupName": SPOT_ASG,
                "CapacityRebalance": True,
                "MaxInstanceLifetime": 604800,
            }
        ],
        nodepools=[
            _nodepool("gpu", expire="720h", budgets=None),
            _nodepool("gpu-pinned", expire="Never", budgets=[{"nodes": "0"}]),
        ],
    )

    facts = _contexts(runner)

    hyperpod = facts["providers"]["gpu-hp"]
    assert hyperpod == {
        "provider": "hyperpod",
        FIELD: {
            "gpu_cluster_arn": HP_EKS_ARN,
            "instance_groups": [
                {
                    "name": "spot-p5en-az3",
                    "scheduled_update": SCHEDULE,
                    "auto_patch": None,
                }
            ],
        },
    }, "the group the ownership selector covers; the unmanaged g6 group is not"
    eks = facts["providers"]["gpu-eks"]
    assert eks["node_groups"] == {
        "cpu-ng": {
            "status": "ACTIVE",
            "node_repair": "disabled",
            "update_strategy": "DEFAULT",
            "gpu": False,
        },
        "gpu-ng": {
            "status": "ACTIVE",
            "node_repair": "disabled",
            "update_strategy": "DEFAULT",
            "gpu": True,
        },
    }, "the node group facts are what they were"
    assert eks[FIELD]["node_groups"] == [
        {
            "name": "gpu-ng",
            "auto_scaling_groups": [
                {
                    "name": SPOT_ASG,
                    "max_instance_lifetime": 604800,
                    "capacity_rebalance": True,
                }
            ],
        }
    ]
    assert eks[FIELD]["karpenter"]["state"] == "read"
    assert [pool["name"] for pool in eks[FIELD]["karpenter"]["nodepools"]] == [
        "gpu",
        "gpu-pinned",
    ]
    describes = [
        call
        for call in runner.calls
        if call[:3] == ["aws", "sagemaker", "describe-cluster"]
    ]
    assert len(describes) == 2, "one describe per HyperPod cluster: no new read"
    assert all(
        call[call.index("--query") + 1] == preflight.HYPERPOD_CONTEXT_QUERY
        for call in describes
    ), "the context describe carries the widened query"
    asg_reads = [call for call in runner.calls if call[1] == "autoscaling"]
    assert len(asg_reads) == 1 and "eks-cpu-ng-00ff" not in asg_reads[0], (
        "one Auto Scaling read, for the node groups hosting GPU nodes"
    )


@pytest.mark.parametrize(
    "runner",
    [
        ContextsRunner(deny_asg=True, nodepools=None),
        ContextsRunner(nodepool_error="Error from server (Forbidden): nodepools"),
    ],
    ids=["asg-denied-no-karpenter", "nodepools-forbidden"],
)
def test_a_failed_advisory_read_is_unknown_and_the_check_still_passes(
    runner: ContextsRunner,
) -> None:
    facts = _contexts(runner)  # returning at all is the check's PASS

    block = facts["providers"]["gpu-eks"][FIELD]
    (asg,) = block["node_groups"][0]["auto_scaling_groups"]
    if runner.deny_asg:
        assert asg["max_instance_lifetime"] == asg["capacity_rebalance"] == "unknown"
        assert "autoscaling:DescribeAutoScalingGroups" in asg["error"]
        assert block["karpenter"] == {"state": "absent", "nodepools": []}
    else:
        assert block["karpenter"]["state"] == "unknown"
        assert "Forbidden" in block["karpenter"]["error"]


def test_a_release_runner_without_a_probe_still_passes_the_check() -> None:
    class PlainRunner(ContextsRunner):
        probe_output = None  # type: ignore[assignment]

    facts = _contexts(PlainRunner())

    assert facts["providers"]["gpu-eks"][FIELD]["karpenter"]["state"] == "unknown"


def test_validate_eks_managed_cluster_hands_back_the_documents_it_read() -> None:
    documents: dict[str, dict[str, Any]] = {}

    facts = preflight.validate_eks_managed_cluster(
        ContextsRunner(),  # type: ignore[arg-type]
        region=REGION,
        expected_eks_arn=EKS_ARN,
        node_selector=f"{OWNERSHIP_LABEL}={SITE_ID}",
        described=documents,
    )

    assert sorted(documents) == ["cpu-ng", "gpu-ng"]
    assert documents["gpu-ng"]["resources"] == {
        "autoScalingGroups": [{"name": SPOT_ASG}]
    }
    assert facts["gpu-ng"] == {
        "status": "ACTIVE",
        "node_repair": "disabled",
        "update_strategy": "DEFAULT",
        "gpu": True,
    }


# --- the status header --------------------------------------------------------------


def _report(facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "healthy": True,
        "health_scope": "full",
        "live_release": {
            "release_id": "r-1",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "NOOP"},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [
                {
                    "name": "regional_contexts",
                    "status": "PASS",
                    "summary": "CPU/GPU contexts, HyperPod ownership and executor IAM are valid",
                    "details": copy.deepcopy(facts),
                }
            ],
        },
    }


def test_status_full_header_carries_one_advisory_line_per_finding() -> None:
    facts = _contexts(
        ContextsRunner(
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
    assert "failing checks: none" in lines
    advisories = [line for line in lines if line.startswith("advisory: ")]
    assert [line.split(":", 2)[1].strip() for line in advisories] == [
        f"Auto Scaling group {SPOT_ASG} of EKS node group gpu-ng (gpu-eks) sets "
        "MaxInstanceLifetime 604800 s",
        "Karpenter NodePool gpu of gpu-eks provisions site GPU nodes with "
        'expireAfter 720h and no zero disruption budget (budgets [{"nodes"',
        "HyperPod instance group spot-p5en-az3 of gpu-hp has "
        "ScheduledUpdateConfig cron(0 12 ? * SUN *)",
    ], "sorted by cluster id; the unmanaged g6 group never appears"
    assert lines[-1] == "full JSON report: stdout"


def test_a_quick_status_without_the_contexts_check_has_no_advisory() -> None:
    report = _report({"clusters": []})
    report["health"]["checks"] = [
        {"name": "control_api", "status": "PASS", "summary": "ok", "details": {}}
    ]

    lines = status_header_lines(report, json_destination="stdout")

    assert not [line for line in lines if line.startswith("advisory: ")]


def test_an_advisory_that_breaks_is_unknown_and_never_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_arguments: Any) -> dict[str, Any]:
        raise RuntimeError("advisory defect")

    monkeypatch.setattr(preflight, "hyperpod_automatic_updates", broken)
    monkeypatch.setattr(preflight, "eks_automatic_updates", broken)

    facts = _contexts(ContextsRunner())

    for cluster_id in ("gpu-hp", "gpu-eks"):
        assert facts["providers"][cluster_id][FIELD] == {
            "state": "unknown",
            "error": "advisory defect",
        }, cluster_id
    assert facts["providers"]["gpu-eks"]["node_groups"]["gpu-ng"]["gpu"] is True
