"""The zero-node check is ``ClaimLoop`` housekeeping, and the executor owns its gauges.

The check runs beside the spare sweep, the node-key sync and the node
inventory on every turn of the claim loop, whether or not the turn claimed:
five long restarts holding every slot must not keep the gauges an alert pages
on from moving. The executor hands the check its own metric family, and the
bootstrap builds the check from the variables the executor already has.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_executor.bootstrap import _zero_node_probe_from_environment
from gpu_fault.cluster_executor.executor import ClusterActionExecutor
from gpu_fault.cluster_executor.zero_node_probe import ZeroNodeCapacityProbe
from gpu_fault.cluster_scope import ClusterProvider
from gpu_fault.models import WorkflowOperation
from tests.execution.test_cluster_executor_claim_slots import (
    FastRenewalTick,
    GatedAdapter,
    LeasingControlPlane,
    eventually,
    running,
    slot_executor,
)
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    remote_command,
)
from tests.execution.test_cluster_executor_saturated_slots import ScheduledJob, Ticker
from tests.execution.test_cluster_executor_zero_node_probe import eks_probe, gauges
from tests.regional._parked_cluster_support import GROUP, OWNERSHIP, nodegroup


class Check:
    """``ZeroNodeCapacityProbe.observe``: due-gated, never raises into the loop."""

    def __init__(self, job: ScheduledJob) -> None:
        self.job = job
        self.gauges: list[Any] = []

    def observe(self, gauges: Any) -> None:
        if self.job.due():
            self.gauges.append(gauges)
            self.job.run()


def test_the_zero_node_check_runs_on_schedule_while_every_slot_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalTick)
    restarts = [
        remote_command(f"restart-{index}", operation=WorkflowOperation.RESTART_NODE)
        for index in range(5)
    ]
    client = LeasingControlPlane(restarts)
    adapter = GatedAdapter(*(command.command_id for command in restarts))
    ticker = Ticker()
    check = Check(ScheduledJob(ticker, 60))
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=5,
        max_concurrent_commands=5,
        lease_seconds=120,
        zero_node_probe=check,
    )

    with running(executor, adapter):
        eventually(
            lambda: len(adapter.started_ids()) == 5, "the five restarts never ran"
        )
        for minute in range(1, 4):
            ticker.seconds += 60
            eventually(
                lambda: len(check.job.runs) == 1 + minute,
                f"the zero-node check missed minute {minute}: {check.job.runs}",
            )

    assert check.job.runs == [0.0, 60.0, 120.0, 180.0], check.job.runs
    assert all(item is executor.metrics for item in check.gauges), (
        "the check sets the executor's own family"
    )


def test_the_executor_exports_the_verdict_of_its_check() -> None:
    probe = eks_probe(nodegroups={GROUP: nodegroup(desired=2)})
    executor = ClusterActionExecutor(
        FakeExecutorClient(),
        [],
        executor_id="cluster-a/executor-0",
        allowed_namespaces={"training"},
        zero_node_probe=probe,
    )

    executor.observe_zero_node_capacity()

    values = gauges(executor.metrics)
    assert (values["zero_node_parked"], values["zero_node_unavailable"]) == (0, 1)


def test_an_executor_without_a_check_is_unchanged() -> None:
    executor = ClusterActionExecutor(
        FakeExecutorClient(),
        [],
        executor_id="cluster-a/executor-0",
        allowed_namespaces={"training"},
    )

    executor.observe_zero_node_capacity()

    assert executor.zero_node_probe is None
    assert gauges(executor.metrics)["zero_node_unavailable"] == 0


class Adapter:
    def __init__(self) -> None:
        self.core = object()
        self.custom = object()


@pytest.mark.parametrize(
    ("environment", "provider", "names"),
    [
        (
            {
                "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
                "GPU_FAULT_NODE_SELECTOR": OWNERSHIP,
                "GPU_FAULT_EKS_CLUSTER_NAME": "gpu-eks-cluster",
            },
            ClusterProvider.EKS_MANAGED,
            ("", "gpu-eks-cluster"),
        ),
        (
            {"GPU_FAULT_HYPERPOD_CLUSTER": "hp-a"},
            ClusterProvider.HYPERPOD,
            ("hp-a", ""),
        ),
    ],
    ids=["eks-managed", "hyperpod-legacy-selector"],
)
def test_the_bootstrap_builds_the_check_from_the_executors_own_variables(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    provider: ClusterProvider,
    names: tuple[str, str],
) -> None:
    for name in (
        "GPU_FAULT_CLUSTER_PROVIDER",
        "GPU_FAULT_NODE_SELECTOR",
        "GPU_FAULT_EKS_CLUSTER_NAME",
        "GPU_FAULT_HYPERPOD_CLUSTER",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    adapter = Adapter()

    probe = _zero_node_probe_from_environment(adapter)  # type: ignore[arg-type]

    assert isinstance(probe, ZeroNodeCapacityProbe), probe
    assert probe.provider is provider
    assert (probe.hyperpod_cluster, probe.eks_cluster) == names
    assert probe.region_name == "us-west-2"
    assert probe.core is adapter.core and probe.custom is adapter.custom
    if provider is ClusterProvider.HYPERPOD:
        assert probe.node_selector == "sagemaker.amazonaws.com/cluster-name=hp-a"


@pytest.mark.parametrize(
    ("environment", "reason"),
    [
        ({}, "GPU_FAULT_NODE_SELECTOR and GPU_FAULT_HYPERPOD_CLUSTER are unset"),
        (
            {
                "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
                "GPU_FAULT_NODE_SELECTOR": OWNERSHIP,
            },
            "EKS cluster name",
        ),
    ],
    ids=["no-scope", "eks-without-its-name"],
)
def test_a_check_that_cannot_be_built_is_said_and_left_off(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    environment: dict[str, str],
    reason: str,
) -> None:
    for name in (
        "GPU_FAULT_CLUSTER_PROVIDER",
        "GPU_FAULT_NODE_SELECTOR",
        "GPU_FAULT_EKS_CLUSTER_NAME",
        "GPU_FAULT_HYPERPOD_CLUSTER",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        probe = _zero_node_probe_from_environment(Adapter())  # type: ignore[arg-type]

    assert probe is None
    assert "the zero-node check is off" in caplog.text and reason in caplog.text
