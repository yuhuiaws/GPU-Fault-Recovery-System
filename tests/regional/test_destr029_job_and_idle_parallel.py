"""GF-REGIONAL-DESTR-029: a two-node training job and an idle GPU node of one
node group recover in parallel from a reboot-class and a reset-class fault.

The runner's own logic: which three nodes it may take (live, from the node
group), how it splits them into a job and an idle node, the
NOT_RUN(CAPABILITY_ABSENT) path a short cluster takes, the plan identity it
binds, the two differently-typed domain expectations it hands the oracle, and
its place in the catalog. The verdicts on the store records are the shared
oracle's (``tests/regional/test_workload_domain_oracle.py`` -- the
job-and-idle layout, PASS / tripped / missing); the reboot proof is DESTR-002's
and the reset contract is DESTR-001's.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL
from scripts.e2e.regional import run_destr029_job_and_idle_parallel as destr029
from scripts.e2e.regional import workload_domain_oracle as oracle
from scripts.e2e.regional.live_driver_guard import CaseRunner
from scripts.e2e.regional.managed_workload_fixture import managed_workload_name
from scripts.e2e.regional.regional_case_contract import (
    case_metadata,
    formal_predecessor,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveSettings,
)

CLUSTER = "gpu-eks-cluster"


# --------------------------------------------------------------------------- #
# Target selection and layout (pure)
# --------------------------------------------------------------------------- #
def test_the_first_three_eligible_nodes_in_name_order_are_the_targets() -> None:
    candidates = {
        "node-d": [],
        "node-b": ["node-b is cordoned"],
        "node-c": [],
        "node-a": [],
    }

    assert destr029.choose_targets(candidates, count=3, dns_nodes=[]) == (
        "node-a",
        "node-c",
        "node-d",
    )


def test_a_short_eligible_set_has_no_targets() -> None:
    assert (
        destr029.choose_targets({"node-a": [], "node-b": []}, count=3, dns_nodes=[])
        is None
    )


def test_the_job_master_worker_and_idle_node_are_assigned_in_order() -> None:
    plan = destr029.layout(("n0", "n1", "n2"), "destr029-run")

    assert (plan.master, plan.worker, plan.idle) == ("n0", "n1", "n2")
    assert plan.job_id == "destr029-run-job"
    assert plan.job_nodes == ("n0", "n1")
    with pytest.raises(ValueError, match="need 3 targets"):
        destr029.layout(("n0", "n1"), "destr029-run")


def test_the_case_needs_workload_reboot_and_reset_ownership() -> None:
    owned = {
        "capabilities": [
            {
                "capability": name,
                "mode": "OWN",
                "owner": "gpu-fault-kubernetes-adapter",
                "adapter": "regional-cluster-executor",
            }
            for name in ("workloadStop", "workloadRestart")
        ]
        + [
            {
                "capability": "nodeReboot",
                "mode": "OWN",
                "owner": "gpu-fault-node-agent",
                "adapter": "hyperpod",
            },
            {
                "capability": "gpuReset",
                "mode": "OWN",
                "owner": "gpu-fault-node-agent",
                "adapter": "node-action",
            },
        ],
        "warnings": [],
    }

    assert destr029.profile_errors(owned) == []
    missing = destr029.profile_errors(None)
    assert "runtime profile has no nodeReboot capability" in missing
    assert "runtime profile has no gpuReset capability" in missing


# --------------------------------------------------------------------------- #
# A cluster that cannot carry the case: NOT_RUN(CAPABILITY_ABSENT)
# --------------------------------------------------------------------------- #
class TwoNodeCluster:
    """A fixture whose only GPU node group has two nodes (one short)."""

    def __init__(self, settings: RegionalLiveSettings) -> None:
        self.settings = settings

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-1", "cluster_id": self.settings.cluster_id}

    def gpu_nodes(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f"eks-{index}",
                "uid": f"uid-{index}",
                "gpu_allocatable": 8,
                "labels": {EKS_NODEGROUP_LABEL: "gpu-group"},
            }
            for index in range(2)
        ]


def settings(tmp_path: Path) -> destr029.Settings:
    for name in ("cpu.kubeconfig", "gpu.kubeconfig", "site.yaml"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return destr029.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            gpu_context="gpu",
            namespace="gpu-fault-system",
            cluster_id=CLUSTER,
            region="us-west-2",
        ),
        host_probe_image="registry/probe@sha256:abc",
        site_file=tmp_path / "site.yaml",
        manifest=tmp_path / "job.yaml",
        predecessor_path=tmp_path / "predecessor.json",
        node_group="",
    )


def test_a_short_node_group_plans_a_not_run_naming_the_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destr029, "RegionalLiveFixture", TwoNodeCluster)
    case_dir = tmp_path / "cases" / destr029.CASE_ID
    case_dir.mkdir(parents=True)

    preflight = destr029.read_only_preflight(settings(tmp_path), case_dir)
    details = destr029.plan_details(settings(tmp_path), preflight)

    assert preflight["errors"] == []
    assert preflight["not_run"] == {
        "verdict": "NOT_RUN",
        "reason": "CAPABILITY_ABSENT",
        "detail": "needs >=3 GPU nodes in one node group; cluster gpu-eks-cluster "
        f"group gpu-group ({EKS_NODEGROUP_LABEL}) has 2",
    }
    assert details["not_run"] == preflight["not_run"]
    assert "layout" not in details


def test_executing_a_not_run_plan_records_the_verdict_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destr029, "RegionalLiveFixture", TwoNodeCluster)
    case_dir = tmp_path / "cases" / destr029.CASE_ID
    case_dir.mkdir(parents=True)
    record = oracle.capability_absent("needs >=3 GPU nodes in one node group; x")
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"not_run": record, "node_group_gate": {"minimum": 3}}}),
        encoding="utf-8",
    )

    code = destr029.execute_case(
        settings(tmp_path), tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )

    evidence = json.loads((case_dir / f"{destr029.CASE_ID}.json").read_text())
    assert code == 0
    assert {key: evidence[key] for key in ("verdict", "reason", "detail")} == record
    assert evidence["mutation_performed"] is False
    assert not (case_dir / "host-probes").exists(), "a NOT_RUN made a probe"


# --------------------------------------------------------------------------- #
# The plan identity the execution must find unchanged
# --------------------------------------------------------------------------- #
def planned_preflight() -> dict[str, Any]:
    targets = ["a", "b", "c"]
    return {
        "release_id": "release-1",
        "node_group": {
            "label": HYPERPOD_INSTANCE_GROUP_LABEL,
            "name": "p5en",
            "nodes": targets,
        },
        "targets": targets,
        "layout": {"job_id": "job", "master": "a", "worker": "b", "idle": "c"},
        "nodes": {
            node: {"uid": f"uid-{node}", "boot_id": f"boot-{node}"} for node in targets
        },
        "stores": {node: {"agent": {"generation": 7}} for node in targets},
        "runtime_identity": {"release_id": "release-1"},
        "storm": {"cluster_gpu_agents": 6, "breaker": None},
        "node_group_gate": {"minimum_gpu_nodes_in_one_node_group": 3},
        "focused_tests": {"passed": True},
        "predecessor": {"valid": True},
    }


def test_the_plan_binds_the_targets_and_refuses_a_rebooted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        destr029,
        "record_focused_tests",
        lambda details, result: details.update(focused_tests=result),
    )
    preflight = planned_preflight()
    details = destr029.plan_details(settings(tmp_path), preflight)
    (tmp_path / "plan.json").write_text(
        json.dumps({"details": details}), encoding="utf-8"
    )

    destr029.verify_plan_identity(tmp_path, preflight)
    rebooted = planned_preflight()
    rebooted["nodes"]["a"]["boot_id"] = "boot-a2"

    assert details["storm_premise"] == {
        "threshold": 3,
        "cluster_gpu_agents": 6,
        "breaker_state": "CLOSED",
        "case_counted_nodes": 2,
    }
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        destr029.verify_plan_identity(tmp_path, rebooted)


# --------------------------------------------------------------------------- #
# The two differently-typed domain expectations handed to the oracle
# --------------------------------------------------------------------------- #
def test_the_oracle_sees_a_job_domain_and_an_idle_domain() -> None:
    plan = destr029.layout(("n0", "n1", "n2"), "destr029-run")

    domains = destr029.oracle_domains(plan, job_workflow_id="wj", idle_workflow_id="wi")

    assert [domain.label for domain in domains] == ["job", "idle"]
    assert [domain.is_job for domain in domains] == [True, False]
    job, idle = domains
    assert (job.faulted_node, job.allocation) == ("n0", ("n0", "n1"))
    # The product's job token names the submitted PyTorchJob object: the
    # manifest's fixed name renamed after the job id, not the bare job id.
    assert job.workload_ids == (
        "gpu-fault-system/pytorchjob/"
        + managed_workload_name("gpu-fault-two-node-replace", "destr029-run-job"),
    )
    assert (idle.faulted_node, idle.allocation, idle.job_id) == ("n2", (), "")
    assert destr029.CASE_COUNTED_NODES == 2


def test_run_ids_keep_rounds_apart(tmp_path: Path) -> None:
    first = destr029.run_identity(tmp_path / "round-13-hp", 1)
    second = destr029.run_identity(tmp_path / "other" / "round-14-hp", 1)

    assert first != second
    assert first.startswith("destr029-hp-"), first


# --------------------------------------------------------------------------- #
# The case in the catalog and on the command line
# --------------------------------------------------------------------------- #
def test_the_case_is_catalogued_behind_destr002() -> None:
    metadata = case_metadata(destr029.CASE_ID)

    assert metadata.confirmation == destr029.CONFIRMATION
    assert metadata.risk == "destructive"
    assert formal_predecessor(destr029.CASE_ID) == destr029.PREDECESSOR_CASE_ID
    assert isinstance(destr029.CASE, CaseRunner), destr029.CASE
    assert destr029.MIN_GROUP_NODES == 3


@pytest.mark.parametrize("flag", ["--node", "--fault-node", "--master", "--idle-node"])
def test_no_profile_node_can_choose_the_targets(flag: str) -> None:
    with pytest.raises(SystemExit):
        destr029.parser().parse_args(["--run-dir", "run", flag, "node-x"])


def test_the_node_group_override_is_a_selection_flag(tmp_path: Path) -> None:
    arguments = destr029.parser().parse_args(
        ["--run-dir", str(tmp_path), "--plan", "--node-group", "p5en"]
    )

    assert isinstance(arguments, argparse.Namespace), arguments
    assert arguments.node_group == "p5en"


# --------------------------------------------------------------------------- #
# The live contracts (attempt 39 on HyperPod: the product recovered both nodes,
# the old contracts failed it anyway)
# --------------------------------------------------------------------------- #
LIVE_JOB_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
)


def _node(boot: str) -> dict[str, Any]:
    return {
        "boot_id": boot,
        "uid": "uid-1",
        "gpu_allocatable": "8",
        "ready": "True",
        "unschedulable": False,
        "ownership_annotations": {},
    }


class _Nodes:
    def __init__(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.nodes = nodes

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return dict(self.nodes[node])


def _target(tmp_path: Path, label: str, node: str, state: dict[str, Any]) -> Any:
    directory = tmp_path / label
    directory.mkdir()
    return destr029.FaultTarget(
        label=label,
        node=node,
        kind="reboot" if label == "job" else "reset",
        probe=None,  # type: ignore[arg-type]
        run_id=f"run-{label}",
        directory=directory,
        state=state,
    )


def test_the_job_node_reboot_the_product_ran_live_passes(tmp_path: Path) -> None:
    """Attempt 39: the job workflow stopped the job, rebooted its master,
    validated, restored scheduling and restarted the job; the old contract
    judged it with DESTR-009's XID 11 checks and a lost-reset-proof line."""

    state = {
        "event": {"xid": 79},
        "decision": {"action": "REBOOT_NODE"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [{"operation": step} for step in LIVE_JOB_STEPS],
            "step_executions": [
                {"operation": step, "status": "SUCCEEDED"} for step in LIVE_JOB_STEPS
            ],
            "completed_operations": ["RESTART_NODE"],
        },
        "agent": {
            "lifecycle_state": "ACTIVE",
            "boot_id": "agent-new",
            "artifact_sha256": "a1",
        },
    }
    run = argparse.Namespace(
        preflight={
            "nodes": {"n0": _node("boot-old"), "n1": _node("peer-boot")},
            "stores": {
                "n0": {"agent": {"boot_id": "agent-old", "artifact_sha256": "a1"}}
            },
        },
        regional=_Nodes({"n0": _node("boot-new"), "n1": _node("peer-boot")}),
        plan=argparse.Namespace(worker="n1"),
    )
    item = _target(tmp_path, "job", "n0", state)

    errors = destr029.job_node_contract(run, item)

    assert errors == []
    assert item.sampler_stopped, "the boot ended the sampler's transient unit"
    assert (
        json.loads((item.directory / "node-after.json").read_text())["boot_id"]
        == "boot-new"
    )


def test_a_rebooted_healthy_worker_fails_the_job_contract(tmp_path: Path) -> None:
    state = {
        "event": {"xid": 79},
        "decision": {"action": "REBOOT_NODE"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [{"operation": step} for step in LIVE_JOB_STEPS],
            "step_executions": [
                {"operation": step, "status": "SUCCEEDED"} for step in LIVE_JOB_STEPS
            ],
        },
        "agent": {"lifecycle_state": "ACTIVE", "boot_id": "agent-new"},
    }
    run = argparse.Namespace(
        preflight={
            "nodes": {"n0": _node("boot-old"), "n1": _node("peer-boot")},
            "stores": {"n0": {"agent": {"boot_id": "agent-old"}}},
        },
        regional=_Nodes({"n0": _node("boot-new"), "n1": _node("peer-boot-2")}),
        plan=argparse.Namespace(worker="n1"),
    )

    errors = destr029.job_node_contract(run, _target(tmp_path, "job", "n0", state))

    assert len(errors) == 1 and "n1: the job's healthy peer rebooted" in errors[0]


def test_the_idle_reset_is_judged_on_the_post_reset_host_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attempt 39 compared the idle node's baseline with an ``after`` nobody
    filled, so every service, GPU and ledger check failed on an empty dict."""

    calls: list[str] = []
    seen: dict[str, Any] = {}

    def post_reset_host(host, case_dir, result, after, **kwargs):  # noqa: ANN001, ANN003
        calls.append("post_reset_host")
        after.update({"read": "after the reset"})
        assert kwargs["reboot"] is None
        return None

    def host_errors(baseline, after, **kwargs):  # noqa: ANN001, ANN003
        calls.append("host_errors")
        seen["after"] = dict(after)
        return []

    monkeypatch.setattr(destr029.reset_case, "workflow_errors", lambda state: [])
    monkeypatch.setattr(
        destr029.reset_case,
        "node_restored_errors",
        lambda regional, node, directory: ([], _node("idle-boot")),
    )
    monkeypatch.setattr(destr029.reset_case, "post_reset_host", post_reset_host)
    monkeypatch.setattr(destr029.reset_case, "host_errors", host_errors)
    run = argparse.Namespace(
        preflight={"nodes": {"n2": _node("idle-boot")}},
        regional=_Nodes({"n2": _node("idle-boot")}),
        started_at=datetime(2026, 10, 30, 9, 13, tzinfo=timezone.utc),
    )
    item = _target(tmp_path, "idle", "n2", {"workflow": {"request_id": "wi"}})

    assert destr029.idle_node_contract(run, item) == []
    assert calls == ["post_reset_host", "host_errors"]
    assert seen["after"] == {"read": "after the reset"}
