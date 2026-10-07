"""COLLECT-015 on both providers: the recovery authority and the reboot count.

Live on the two-node ``eks-managed`` cluster (attempt 7) the case really
rebooted the node through the eks adapter and still failed with
``HyperPod cluster NodeRecovery is 'not-applicable'`` -- the HyperPod-only
guard read against a cluster that has no NodeRecovery -- and
``CloudTrail reboot count is not one``: the runner waited for SageMaker reboot
names in the SageMaker event source, where an EC2 ``RebootInstances`` never
appears. The provider-aware path reads the provider's own source, keeps the
refused calls (Karpenter's ``RunInstances`` dry-runs) apart from the
mutations, and replaces the NodeRecovery guard on plain EKS with the
instance's replacement authority (node group, Auto Scaling instance).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from scripts.e2e.regional import collect015_provider as collect015
from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional.collector_reboot_evidence import normalize_provider_event
from tests.regional.test_collector_reboot_scope_eks import (
    ROLE,
    ec2_cloudtrail_item,
    karpenter_run_instances_item,
)

INSTANCE = "i-00000000000000001"
NODE = "ip-10-0-41-164.us-west-2.compute.internal"


def _events(*items: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(normalize_provider_event(item)) for item in items]


def _live_eks_window() -> list[dict[str, Any]]:
    """The live shape: one accepted RebootInstances by the executor role and
    three Karpenter RunInstances dry-runs AWS refused."""

    return _events(
        karpenter_run_instances_item(1),
        ec2_cloudtrail_item(),
        karpenter_run_instances_item(2),
        karpenter_run_instances_item(3),
    )


def test_node_recovery_guard_reads_each_providers_own_authority() -> None:
    assert collect015.node_recovery_guard_errors("None", provider="hyperpod") == []
    assert collect015.node_recovery_guard_errors(None, provider=None) == []
    assert collect015.node_recovery_guard_errors("Automatic", provider="hyperpod") == [
        "HyperPod cluster NodeRecovery is 'Automatic'; the reboot must not "
        "compete with automatic recovery"
    ]
    assert (
        collect015.node_recovery_guard_errors("not-applicable", provider="eks-managed")
        == []
    )
    errors = collect015.node_recovery_guard_errors("None", provider="eks-managed")
    assert errors == [
        "EKS cluster recovery reads 'None'; plain EKS has no HyperPod NodeRecovery "
        "to compete with (expected 'not-applicable')"
    ]


def test_collect015_workflow_errors_accept_not_applicable_on_eks() -> None:
    workflow = {
        "status": "SUCCEEDED",
        "step_executions": [{"operation": "RESTART_NODE", "status": "SUCCEEDED"}],
    }
    assert (
        destructive.collect015_workflow_errors(
            workflow,
            submission={"state": "SUBMITTED"},
            node_after={"unschedulable": False},
            node_recovery="not-applicable",
            allow_replace="false",
            provider="eks-managed",
        )
        == []
    )
    hyperpod = destructive.collect015_workflow_errors(
        workflow,
        submission={"state": "SUBMITTED"},
        node_after={"unschedulable": False},
        node_recovery="not-applicable",
        allow_replace="false",
    )
    assert hyperpod == [
        "HyperPod cluster NodeRecovery is 'not-applicable'; the reboot must not "
        "compete with automatic recovery"
    ], "the HyperPod guard still refuses anything but None"


def test_eks_reboot_count_ignores_karpenter_dry_runs() -> None:
    errors, evidence = collect015.reboot_count_errors(
        _live_eks_window(), provider="eks-managed", executor_role_arn=ROLE
    )
    assert errors == [], errors
    assert [item["event_name"] for item in evidence["reboot_events"]] == [
        "RebootInstances"
    ]
    assert evidence["replace_events"] == []
    assert [item["event_name"] for item in evidence["refused_provider_calls"]] == [
        "RunInstances"
    ] * 3
    assert {item["api_error_code"] for item in evidence["refused_provider_calls"]} == {
        "Client.DryRunOperation"
    }


def test_eks_reboot_count_names_the_empty_and_refused_windows() -> None:
    errors, _ = collect015.reboot_count_errors(
        [], provider="eks-managed", executor_role_arn=ROLE
    )
    assert errors == [
        "CloudTrail reboot count is not one (0 accepted ['RebootInstances'] calls; "
        "0 refused reboot calls, 0 refused provider calls in the window)"
    ]
    refused_reboot = ec2_cloudtrail_item()
    detail = refused_reboot["CloudTrailEvent"].replace(
        '"responseElements"', '"errorCode": "Client.UnauthorizedOperation", "x"'
    )
    refused_reboot["CloudTrailEvent"] = detail
    errors, evidence = collect015.reboot_count_errors(
        _events(refused_reboot, karpenter_run_instances_item(1)),
        provider="eks-managed",
        executor_role_arn=ROLE,
    )
    assert errors == [
        "CloudTrail reboot count is not one (0 accepted ['RebootInstances'] calls; "
        "1 refused reboot calls, 2 refused provider calls in the window)"
    ], "a refused reboot attempt is never mistaken for the one accepted reboot"
    assert evidence["reboot_events"] == []


def test_an_accepted_launch_or_termination_is_a_replacement_on_eks() -> None:
    errors, evidence = collect015.reboot_count_errors(
        _events(ec2_cloudtrail_item(), karpenter_run_instances_item(1, dry_run=False)),
        provider="eks-managed",
        executor_role_arn=ROLE,
    )
    assert errors == ["CloudTrail contains provider replacement: ['RunInstances']"]
    assert [item["event_name"] for item in evidence["replace_events"]] == [
        "RunInstances"
    ]


def test_a_reboot_by_another_actor_fails_on_eks() -> None:
    errors, _ = collect015.reboot_count_errors(
        _live_eks_window(),
        provider="eks-managed",
        executor_role_arn="arn:aws:iam::123456789012:role/someone-else",
    )
    assert errors == ["reboot actor is not the executor role"]


def test_hyperpod_reboot_count_keeps_its_sagemaker_vocabulary() -> None:
    events = [
        {"event_name": "BatchRebootClusterNodes", "session_issuer_role_name": "exec"}
    ]
    errors, evidence = collect015.reboot_count_errors(
        events, provider="hyperpod", executor_role_arn="arn:aws:iam::1:role/exec"
    )
    assert errors == [], errors
    assert evidence["reboot_events"] == events
    replaced = [*events, {"event_name": "BatchReplaceClusterNodes"}]
    errors, _ = collect015.reboot_count_errors(
        replaced, provider=None, executor_role_arn="arn:aws:iam::1:role/exec"
    )
    assert errors == [
        "CloudTrail contains provider replacement: ['BatchReplaceClusterNodes']"
    ]


class _Regional:
    """Records the CloudTrail reads the provider-calls helper makes."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    def wait_provider_events(self, started_at: datetime, **kwargs: Any) -> list:
        self.calls.append({"wait": kwargs})
        return [
            item for item in self.events if item["event_name"] in kwargs["event_names"]
        ]

    def provider_events(
        self, started_at: datetime, ended_at: datetime, *, event_source: str
    ) -> list[dict[str, Any]]:
        self.calls.append({"read": event_source})
        return self.events

    @staticmethod
    def provider_events_provisional(_ended_at: datetime) -> bool:
        return True


def test_provider_calls_read_the_ec2_source_on_eks() -> None:
    regional = _Regional(_live_eks_window())
    result = collect015.provider_calls(
        regional,
        datetime.now(timezone.utc),
        provider="eks-managed",
        executor_role_arn=ROLE,
    )
    assert result["errors"] == [], result
    assert regional.calls[0]["wait"] == {
        "event_names": {"RebootInstances"},
        "expected_count": 1,
        "event_source": "ec2.amazonaws.com",
    }
    assert regional.calls[1] == {"read": "ec2.amazonaws.com"}
    evidence = result["evidence"]
    assert evidence["provider"] == "eks-managed"
    assert evidence["provider_event_source"] == "ec2.amazonaws.com"
    assert len(evidence["provider_events"]) == 4
    assert len(evidence["refused_provider_calls"]) == 3
    assert evidence["provider_events_provisional"] is True


def _nodegroup(
    status: str = "ACTIVE",
    issues: list[dict[str, str]] | None = None,
    repair: bool | None = None,
):
    group: dict[str, Any] = {
        "nodegroupName": "gpu-group",
        "clusterName": "gpu-eks-cluster",
        "status": status,
        "health": {"issues": issues or []},
    }
    if repair is not None:
        group["nodeRepairConfig"] = {"enabled": repair}
    return {"nodegroup": group}


def _autoscaling(
    lifecycle: str = "InService", health: str = "HEALTHY", instance_id: str = INSTANCE
):
    return {
        "AutoScalingInstances": [
            {
                "InstanceId": instance_id,
                "AutoScalingGroupName": "eks-gpu-group-synthetic",
                "LifecycleState": lifecycle,
                "HealthStatus": health,
            }
        ]
    }


def _guard(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "node": NODE,
        "instance_id": INSTANCE,
        "nodegroup_name": "gpu-group",
        "nodegroup": _nodegroup(),
        "autoscaling": _autoscaling(),
    }
    arguments.update(overrides)
    return collect015.eks_replacement_guard(**arguments)


def test_an_active_group_and_an_in_service_instance_pass_the_eks_guard() -> None:
    guard = _guard()
    assert guard == {
        "provider": "eks-managed",
        "node": NODE,
        "instance_id": INSTANCE,
        "nodegroup": {
            "name": "gpu-group",
            "status": "ACTIVE",
            "health_issue_codes": [],
            "node_repair_config": {"enabled": None},
        },
        "autoscaling_instance": {
            "group": "eks-gpu-group-synthetic",
            "lifecycle_state": "InService",
            "health_status": "HEALTHY",
        },
    }
    assert collect015.replacement_guard_errors(guard) == []
    disabled = _guard(nodegroup=_nodegroup(repair=False))
    assert disabled["nodegroup"]["node_repair_config"] == {"enabled": False}
    assert collect015.replacement_guard_errors(disabled) == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"instance_id": ""}, "no EC2 providerID"),
        (
            {"nodegroup": _nodegroup(repair=True)},
            "EKS node auto repair is enabled on managed node group gpu-group",
        ),
        (
            {"nodegroup_name": "", "nodegroup": {}},
            "carries no eks.amazonaws.com/nodegroup",
        ),
        ({"nodegroup": _nodegroup("DEGRADED")}, "is 'DEGRADED', not ACTIVE"),
        (
            {"nodegroup": _nodegroup(issues=[{"code": "Ec2InstanceHealthIssue"}])},
            "health issues: ['Ec2InstanceHealthIssue']",
        ),
        ({"autoscaling": {"AutoScalingInstances": []}}, "in no Auto Scaling group"),
        (
            {"autoscaling": _autoscaling(lifecycle="Terminating")},
            f"in flight for {INSTANCE}: Terminating/HEALTHY",
        ),
        (
            {"autoscaling": _autoscaling(health="Unhealthy")},
            f"in flight for {INSTANCE}: InService/Unhealthy",
        ),
    ],
)
def test_the_eks_guard_fails_closed_on_every_replacement_signal(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = collect015.replacement_guard_errors(_guard(**overrides))
    assert any(fragment in item for item in errors), (fragment, errors)


def test_the_hyperpod_guard_is_the_node_recovery_reading() -> None:
    assert collect015.replacement_guard_errors({"provider": "hyperpod"}) == []
    result = collect015.replacement_guard(
        object(),
        node=NODE,
        cluster_name="hp",
        provider="hyperpod",
        recovery={"node_recovery": "None"},
    )
    assert result == {
        "guard": {"provider": "hyperpod", "node_recovery": "None"},
        "errors": [],
    }


class _EksRegional:
    """A node in an ACTIVE managed node group backed by a healthy ASG instance."""

    class settings:
        region = "us-west-2"

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def kubectl(self, _plane: str, *arguments: str, **_kwargs: Any) -> str:
        self.commands.append(["kubectl", *arguments])
        import json

        return json.dumps(
            {
                "metadata": {
                    "name": NODE,
                    "labels": {"eks.amazonaws.com/nodegroup": "gpu-group"},
                },
                "spec": {"providerID": f"aws:///us-west-2c/{INSTANCE}"},
            }
        )

    def run(self, command: list[str], **_kwargs: Any) -> Any:
        import json
        from types import SimpleNamespace

        self.commands.append(list(command))
        if command[1] == "eks":
            return SimpleNamespace(stdout=json.dumps(_nodegroup()))
        return SimpleNamespace(stdout=json.dumps(_autoscaling()))


def test_the_live_eks_guard_reads_the_group_and_the_asg_instance() -> None:
    regional = _EksRegional()
    result = collect015.replacement_guard(
        regional,
        node=NODE,
        cluster_name="gpu-eks-cluster",
        provider="eks-managed",
        recovery={"node_recovery": "not-applicable", "status": "ACTIVE"},
    )
    assert result["errors"] == [], result
    assert result["guard"]["node_recovery"] == "not-applicable"
    assert result["guard"]["autoscaling_instance"]["lifecycle_state"] == "InService"
    assert regional.commands[1][:6] == [
        "aws",
        "eks",
        "describe-nodegroup",
        "--cluster-name",
        "gpu-eks-cluster",
        "--nodegroup-name",
    ]
    assert regional.commands[2][:5] == [
        "aws",
        "autoscaling",
        "describe-auto-scaling-instances",
        "--instance-ids",
        INSTANCE,
    ]
    assert all("--region" in command for command in regional.commands[1:]), (
        regional.commands
    )
