"""GF-REGIONAL-DESTR-028: two two-node training jobs of one node group recover
in parallel from a reboot-class fault on one node of each.

The runner's own logic: which four nodes it may take (live, from the node
group, never from a profile's node list), how it pairs them into two jobs, the
NOT_RUN(CAPABILITY_ABSENT) path a short cluster takes, the plan identity it
binds, the job-domain expectations it hands the oracle, and its place in the
catalog. The verdicts on the store records are the shared oracle's
(``tests/regional/test_workload_domain_oracle.py`` -- the two-two-node-jobs
layout, PASS / tripped / missing); the per-node reboot proof is DESTR-002's.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL
from scripts.e2e.regional import run_destr028_two_job_parallel_reboot as destr028
from scripts.e2e.regional import workload_domain_oracle as oracle
from scripts.e2e.regional.live_driver_guard import CaseRunner
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
# Target selection and job layout (pure)
# --------------------------------------------------------------------------- #
def test_the_first_four_eligible_nodes_in_name_order_are_the_targets() -> None:
    candidates = {
        "node-e": [],
        "node-b": ["node-b is cordoned"],
        "node-d": [],
        "node-a": [],
        "node-c": [],
    }

    assert destr028.choose_targets(candidates, count=4, dns_nodes=[]) == (
        "node-a",
        "node-c",
        "node-d",
        "node-e",
    )


def test_a_short_eligible_set_has_no_targets() -> None:
    candidates = {"node-a": [], "node-b": [], "node-c": ["busy"]}

    assert destr028.choose_targets(candidates, count=4, dns_nodes=[]) is None


def test_targets_never_hold_every_kube_dns_endpoint() -> None:
    candidates = {f"node-{name}": [] for name in "abcde"}

    chosen = destr028.choose_targets(
        candidates, count=4, dns_nodes=["node-a", "node-b", "node-c", "node-d"]
    )

    assert chosen is not None and not set(
        ["node-a", "node-b", "node-c", "node-d"]
    ) <= set(chosen)
    assert (
        destr028.choose_targets(
            {f"node-{name}": [] for name in "abcd"},
            count=4,
            dns_nodes=["node-a", "node-b", "node-c", "node-d"],
        )
        is None
    )


def test_two_jobs_pair_the_targets_master_first() -> None:
    jobs = destr028.job_layout(("n0", "n1", "n2", "n3"), "destr028-run")

    assert [(job.label, job.master, job.worker, job.faulted_node) for job in jobs] == [
        ("a", "n0", "n1", "n0"),
        ("b", "n2", "n3", "n2"),
    ]
    assert [job.job_id for job in jobs] == ["destr028-run-a", "destr028-run-b"]
    with pytest.raises(ValueError, match="need 4 targets"):
        destr028.job_layout(("n0", "n1"), "destr028-run")


def test_the_job_recovery_ownership_is_required() -> None:
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
            }
        ],
        "warnings": [],
    }

    assert destr028.profile_errors(owned) == []
    missing = destr028.profile_errors(None)
    assert "runtime profile has no workloadStop capability" in missing
    assert "runtime profile has no nodeReboot capability" in missing


# --------------------------------------------------------------------------- #
# A cluster that cannot carry the case: NOT_RUN(CAPABILITY_ABSENT)
# --------------------------------------------------------------------------- #
class ThreeNodeCluster:
    """A fixture whose only GPU node group has three nodes (one short)."""

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
            for index in range(3)
        ]


def settings(tmp_path: Path) -> destr028.Settings:
    for name in ("cpu.kubeconfig", "gpu.kubeconfig", "site.yaml"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return destr028.Settings(
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
    monkeypatch.setattr(destr028, "RegionalLiveFixture", ThreeNodeCluster)
    case_dir = tmp_path / "cases" / destr028.CASE_ID
    case_dir.mkdir(parents=True)

    preflight = destr028.read_only_preflight(settings(tmp_path), case_dir)
    details = destr028.plan_details(settings(tmp_path), preflight)

    assert preflight["errors"] == []
    assert preflight["not_run"] == {
        "verdict": "NOT_RUN",
        "reason": "CAPABILITY_ABSENT",
        "detail": "needs >=4 GPU nodes in one node group; cluster gpu-eks-cluster "
        f"group gpu-group ({EKS_NODEGROUP_LABEL}) has 3",
    }
    assert details["not_run"] == preflight["not_run"]
    assert "jobs" not in details


def test_executing_a_not_run_plan_records_the_verdict_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destr028, "RegionalLiveFixture", ThreeNodeCluster)
    case_dir = tmp_path / "cases" / destr028.CASE_ID
    case_dir.mkdir(parents=True)
    record = oracle.capability_absent("needs >=4 GPU nodes in one node group; x")
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"not_run": record, "node_group_gate": {"minimum": 4}}}),
        encoding="utf-8",
    )

    code = destr028.execute_case(
        settings(tmp_path), tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )

    evidence = json.loads((case_dir / f"{destr028.CASE_ID}.json").read_text())
    assert code == 0
    assert {key: evidence[key] for key in ("verdict", "reason", "detail")} == record
    assert evidence["mutation_performed"] is False
    assert not (case_dir / "host-probes").exists(), "a NOT_RUN made a probe"


def test_the_not_run_result_is_never_a_pass() -> None:
    result = destr028.not_run_result(
        2, {"release_id": "r", "cluster_id": "c"}, oracle.capability_absent("short")
    )

    assert (result["verdict"], result["reason"], result["detail"]) == (
        "NOT_RUN",
        "CAPABILITY_ABSENT",
        "short",
    )
    assert result["case_id"] == destr028.CASE_ID


# --------------------------------------------------------------------------- #
# The plan identity the execution must find unchanged
# --------------------------------------------------------------------------- #
def planned_preflight() -> dict[str, Any]:
    targets = ["a", "b", "c", "d"]
    return {
        "release_id": "release-1",
        "node_group": {
            "label": HYPERPOD_INSTANCE_GROUP_LABEL,
            "name": "p5en",
            "nodes": targets,
        },
        "targets": targets,
        "jobs": [
            {"label": "a", "master": "a", "worker": "b", "faulted_node": "a"},
            {"label": "b", "master": "c", "worker": "d", "faulted_node": "c"},
        ],
        "nodes": {
            node: {"uid": f"uid-{node}", "boot_id": f"boot-{node}"} for node in targets
        },
        "stores": {node: {"agent": {"generation": 7}} for node in targets},
        "runtime_identity": {"release_id": "release-1"},
        "storm": {"cluster_gpu_agents": 8, "breaker": None},
        "node_group_gate": {"minimum_gpu_nodes_in_one_node_group": 4},
        "focused_tests": {"passed": True},
        "predecessor": {"valid": True},
    }


def test_the_plan_binds_the_targets_and_refuses_a_rebooted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        destr028,
        "record_focused_tests",
        lambda details, result: details.update(focused_tests=result),
    )
    preflight = planned_preflight()
    identity = destr028.plan_identity(preflight)
    details = destr028.plan_details(settings(tmp_path), preflight)
    (tmp_path / "plan.json").write_text(
        json.dumps({"details": details}), encoding="utf-8"
    )

    destr028.verify_plan_identity(tmp_path, preflight)
    rebooted = planned_preflight()
    rebooted["nodes"]["c"]["boot_id"] = "boot-c2"

    assert identity["targets"]["a"] == {
        "uid": "uid-a",
        "boot_id": "boot-a",
        "agent_generation": 7,
    }
    assert details["storm_premise"] == {
        "threshold": 3,
        "cluster_gpu_agents": 8,
        "breaker_state": "CLOSED",
        "case_counted_nodes": 2,
    }
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        destr028.verify_plan_identity(tmp_path, rebooted)


# --------------------------------------------------------------------------- #
# The job-domain expectations handed to the oracle
# --------------------------------------------------------------------------- #
def test_the_oracle_sees_two_job_domains_each_faulted_on_one_node() -> None:
    jobs = destr028.job_layout(("n0", "n1", "n2", "n3"), "destr028-run")

    domains = destr028.oracle_domains(jobs, {"a": "w-a", "b": "w-b"})

    assert [domain.label for domain in domains] == ["job-a", "job-b"]
    assert [domain.is_job for domain in domains] == [True, True]
    assert domains[0].faulted_node == "n0"
    assert domains[0].allocation == ("n0", "n1")
    assert domains[0].workflow_id == "w-a"
    assert domains[0].workload_ids == ("gpu-fault-system/pytorchjob/destr028-run-a",)
    # The breaker must count exactly the two faulted masters, below the floor.
    assert destr028.CASE_COUNTED_NODES == 2


def test_run_ids_keep_rounds_apart(tmp_path: Path) -> None:
    first = destr028.run_identity(tmp_path / "round-13-hp", 1)
    second = destr028.run_identity(tmp_path / "other" / "round-14-hp", 1)

    assert first != second
    assert first.startswith("destr028-hp-"), first


# --------------------------------------------------------------------------- #
# The case in the catalog and on the command line
# --------------------------------------------------------------------------- #
def test_the_case_is_catalogued_behind_destr002() -> None:
    metadata = case_metadata(destr028.CASE_ID)

    assert metadata.confirmation == destr028.CONFIRMATION
    assert metadata.risk == "destructive"
    assert formal_predecessor(destr028.CASE_ID) == destr028.PREDECESSOR_CASE_ID
    assert isinstance(destr028.CASE, CaseRunner), destr028.CASE
    assert destr028.MIN_GROUP_NODES == 4


@pytest.mark.parametrize("flag", ["--node", "--fault-node", "--node-a", "--master"])
def test_no_profile_node_can_choose_the_targets(flag: str) -> None:
    with pytest.raises(SystemExit):
        destr028.parser().parse_args(["--run-dir", "run", flag, "node-x"])


def test_the_node_group_override_is_a_selection_flag(tmp_path: Path) -> None:
    arguments = destr028.parser().parse_args(
        ["--run-dir", str(tmp_path), "--plan", "--node-group", "p5en"]
    )

    assert isinstance(arguments, argparse.Namespace), arguments
    assert arguments.node_group == "p5en"
