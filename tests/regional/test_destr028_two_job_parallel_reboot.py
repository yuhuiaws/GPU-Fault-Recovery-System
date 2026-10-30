"""GF-REGIONAL-DESTR-028: two training jobs of one node group recover in
parallel from a reboot-class fault on one node of each.

The runner's own logic: which nodes it may take (live, from the node group,
never from a profile's node list), the layout they carry -- two two-node jobs
on four eligible nodes ("2+2"), a two-node and a single-node job on three
("2+1") --, how it pairs them into jobs and names each job's workload, the
NOT_RUN(CAPABILITY_ABSENT) path a short cluster takes, the plan identity it
binds, the job-domain expectations it hands the oracle, the faulted-node
contract it judges each job with, and its place in the catalog. The verdicts on
the store records are the shared oracle's
(``tests/regional/test_workload_domain_oracle.py`` -- the two-job layouts,
PASS / tripped / missing); the per-node reboot proof is
``tests/regional/test_job_reboot_contract.py``'s.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL
from scripts.e2e.regional import job_reboot_contract
from scripts.e2e.regional import run_destr028_two_job_parallel_reboot as destr028
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
TWO_NODE = "gpu-fault-two-node-replace"
SINGLE_NODE = "gpu-fault-single-node-warm-spare"


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


def test_four_eligible_nodes_carry_2_plus_2_and_three_carry_2_plus_1() -> None:
    """HyperPod today: four GPU nodes, the arbiters packed onto one of them,
    so three are eligible and the case runs as 2+1."""

    four = {f"node-{name}": [] for name in "abcd"}
    three = {**four, "node-d": ["node-d hosts arbiter Pods"]}
    two = {**three, "node-c": ["node-c is cordoned"]}

    assert destr028.choose_layout_targets(four, dns_nodes=[]) == (
        "node-a",
        "node-b",
        "node-c",
        "node-d",
    )
    assert destr028.choose_layout_targets(three, dns_nodes=[]) == (
        "node-a",
        "node-b",
        "node-c",
    )
    assert destr028.choose_layout_targets(two, dns_nodes=[]) is None
    assert destr028.layout_name(("n0", "n1", "n2", "n3")) == "2+2"
    assert destr028.layout_name(("n0", "n1", "n2")) == "2+1"


def test_four_nodes_holding_all_of_kube_dns_fall_back_to_2_plus_1() -> None:
    four = {f"node-{name}": [] for name in "abcd"}

    assert destr028.choose_layout_targets(
        four, dns_nodes=["node-a", "node-b", "node-c", "node-d"]
    ) == ("node-a", "node-b", "node-c")


def test_2_plus_2_pairs_the_targets_master_first() -> None:
    jobs = destr028.job_layout(("n0", "n1", "n2", "n3"), "destr028-run")

    assert [
        (job.label, job.nodes, job.master, job.workers, job.faulted_node)
        for job in jobs
    ] == [
        ("a", ("n0", "n1"), "n0", ("n1",), "n0"),
        ("b", ("n2", "n3"), "n2", ("n3",), "n2"),
    ]
    assert [job.gpu_count for job in jobs] == [16, 16]
    assert [job.job_id for job in jobs] == ["destr028-run-a", "destr028-run-b"]
    # One manifest, two jobs: each is renamed after its own job id, so the
    # second submission never collides with the first's fixed name.
    assert [job.workload_name for job in jobs] == [
        managed_workload_name(TWO_NODE, "destr028-run-a"),
        managed_workload_name(TWO_NODE, "destr028-run-b"),
    ]
    assert jobs[0].workload_name != jobs[1].workload_name


def test_2_plus_1_gives_job_b_its_single_node() -> None:
    a, b = destr028.job_layout(("n0", "n1", "n2"), "destr028-run")

    assert (a.nodes, a.workers, a.gpu_count) == (("n0", "n1"), ("n1",), 16)
    assert (b.nodes, b.master, b.workers, b.faulted_node, b.gpu_count) == (
        ("n2",),
        "n2",
        (),
        "n2",
        8,
    )
    assert b.workload_name == managed_workload_name(SINGLE_NODE, "destr028-run-b")
    assert b.workload_id == f"gpu-fault-system/pytorchjob/{b.workload_name}"
    with pytest.raises(ValueError, match="need 3 or 4 targets"):
        destr028.job_layout(("n0", "n1"), "destr028-run")


def test_the_shipped_manifests_carry_the_names_the_layout_renames() -> None:
    assert (
        job_reboot_contract.manifest_workload_name(destr028.TRAINING_MANIFEST)
        == destr028.WORKLOAD_BASES[2]
        == TWO_NODE
    )
    assert (
        job_reboot_contract.manifest_workload_name(
            destr028.SINGLE_NODE_TRAINING_MANIFEST
        )
        == destr028.WORKLOAD_BASES[1]
        == SINGLE_NODE
    )


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
    monkeypatch.setattr(destr028, "RegionalLiveFixture", TwoNodeCluster)
    case_dir = tmp_path / "cases" / destr028.CASE_ID
    case_dir.mkdir(parents=True)

    preflight = destr028.read_only_preflight(settings(tmp_path), case_dir)
    details = destr028.plan_details(settings(tmp_path), preflight)

    assert preflight["errors"] == []
    assert preflight["not_run"] == {
        "verdict": "NOT_RUN",
        "reason": "CAPABILITY_ABSENT",
        "detail": "needs >=3 GPU nodes in one node group; cluster gpu-eks-cluster "
        f"group gpu-group ({EKS_NODEGROUP_LABEL}) has 2",
    }
    assert details["not_run"] == preflight["not_run"]
    assert "jobs" not in details


def test_executing_a_not_run_plan_records_the_verdict_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destr028, "RegionalLiveFixture", TwoNodeCluster)
    case_dir = tmp_path / "cases" / destr028.CASE_ID
    case_dir.mkdir(parents=True)
    record = oracle.capability_absent("needs >=3 GPU nodes in one node group; x")
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"not_run": record, "node_group_gate": {"minimum": 3}}}),
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
# Each job's manifest: pinned to its nodes and renamed after its job id
# --------------------------------------------------------------------------- #
def test_each_job_is_rendered_pinned_and_renamed(tmp_path: Path) -> None:
    value = dataclasses.replace(settings(tmp_path), manifest=destr028.TRAINING_MANIFEST)
    jobs = destr028.job_layout(
        ("n0", "n1", "n2"), "destr028-run", workload_bases=value.workload_bases()
    )
    rendered = {}
    for spec in jobs:
        directory = tmp_path / spec.label
        directory.mkdir()
        path = destr028.render_job_manifest(value, spec, directory)
        assert path.stat().st_mode & 0o777 == 0o600, oct(path.stat().st_mode)
        rendered[spec.label] = yaml.safe_load(path.read_text(encoding="utf-8"))

    two_node = rendered["a"]["spec"]["pytorchReplicaSpecs"]
    assert rendered["a"]["metadata"]["name"] == jobs[0].workload_name
    assert two_node["Master"]["template"]["spec"]["nodeName"] == "n0"
    assert two_node["Worker"]["template"]["spec"]["nodeName"] == "n1"
    single = rendered["b"]["spec"]["pytorchReplicaSpecs"]
    assert rendered["b"]["metadata"]["name"] == jobs[1].workload_name
    assert set(single) == {"Master"}
    assert single["Master"]["template"]["spec"]["nodeName"] == "n2"


def test_a_job_the_plan_named_otherwise_is_refused(tmp_path: Path) -> None:
    value = dataclasses.replace(settings(tmp_path), manifest=destr028.TRAINING_MANIFEST)
    spec = dataclasses.replace(
        destr028.job_layout(("n0", "n1", "n2"), "destr028-run")[0],
        workload_name="someone-elses-job",
    )

    with pytest.raises(RegionalFixtureError, match="the plan expects"):
        destr028.render_job_manifest(value, spec, tmp_path)


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
        "layout": "2+2",
        "jobs": [
            job.as_dict()
            for job in destr028.job_layout(tuple(targets), "destr028-plan")
        ],
        "nodes": {
            node: {"uid": f"uid-{node}", "boot_id": f"boot-{node}"} for node in targets
        },
        "stores": {node: {"agent": {"generation": 7}} for node in targets},
        "runtime_identity": {"release_id": "release-1"},
        "storm": {"cluster_gpu_agents": 8, "breaker": None},
        "node_group_gate": {"minimum_gpu_nodes_in_one_node_group": 3},
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
    shrunk = planned_preflight()
    shrunk["targets"] = ["a", "b", "c"]

    assert identity["targets"]["a"] == {
        "uid": "uid-a",
        "boot_id": "boot-a",
        "agent_generation": 7,
    }
    assert details["layout"] == "2+2"
    assert details["storm_premise"] == {
        "threshold": 3,
        "cluster_gpu_agents": 8,
        "breaker_state": "CLOSED",
        "case_counted_nodes": 2,
    }
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        destr028.verify_plan_identity(tmp_path, rebooted)
    # A fourth node lost between the plan and the execution changes the layout.
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        destr028.verify_plan_identity(tmp_path, shrunk)


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
    # The product's job token names the submitted (renamed) PyTorchJob.
    assert domains[0].workload_ids == (
        "gpu-fault-system/pytorchjob/"
        + managed_workload_name(TWO_NODE, "destr028-run-a"),
    )
    # The breaker must count exactly the two faulted masters, below the floor.
    assert destr028.CASE_COUNTED_NODES == 2


def test_the_single_node_job_is_a_job_domain_of_its_one_node() -> None:
    jobs = destr028.job_layout(("n0", "n1", "n2"), "destr028-run")

    domains = destr028.oracle_domains(jobs, {"a": "w-a", "b": "w-b"})

    assert [(item.label, item.faulted_node, item.allocation) for item in domains] == [
        ("job-a", "n0", ("n0", "n1")),
        ("job-b", "n2", ("n2",)),
    ]
    assert domains[1].is_job and domains[1].job_id == "destr028-run-b"
    assert domains[1].workload_ids == (
        "gpu-fault-system/pytorchjob/"
        + managed_workload_name(SINGLE_NODE, "destr028-run-b"),
    )


def test_run_ids_keep_rounds_apart(tmp_path: Path) -> None:
    first = destr028.run_identity(tmp_path / "round-13-hp", 1)
    second = destr028.run_identity(tmp_path / "other" / "round-14-hp", 1)

    assert first != second
    assert first.startswith("destr028-hp-"), first


# --------------------------------------------------------------------------- #
# Each job's faulted master: the job reboot contract (the shapes DESTR-029
# recorded live on HyperPod for its job node)
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


def _job_state() -> dict[str, Any]:
    return {
        "event": {"xid": 79},
        "decision": {"action": "REBOOT_NODE"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [{"operation": step} for step in LIVE_JOB_STEPS],
            "step_executions": [
                {"operation": step, "status": "SUCCEEDED"} for step in LIVE_JOB_STEPS
            ],
        },
        "agent": {
            "lifecycle_state": "ACTIVE",
            "boot_id": "agent-new",
            "artifact_sha256": "a1",
        },
    }


def _job_run(tmp_path: Path, spec: destr028.JobSpec) -> destr028.JobRun:
    directory = tmp_path / spec.label
    directory.mkdir()
    return destr028.JobRun(
        spec=spec,
        fixture=None,  # type: ignore[arg-type]
        probe=None,  # type: ignore[arg-type]
        run_id=f"run-{spec.label}",
        directory=directory,
        state=_job_state(),
    )


def _run(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> argparse.Namespace:
    return argparse.Namespace(
        preflight={
            "nodes": before,
            "stores": {
                node: {"agent": {"boot_id": "agent-old", "artifact_sha256": "a1"}}
                for node in before
            },
        },
        regional=_Nodes(after),
    )


def test_both_layouts_pass_the_job_reboot_contract(tmp_path: Path) -> None:
    a, b = destr028.job_layout(("n0", "n1", "n2"), "destr028-run")
    run = _run(
        {"n0": _node("n0-old"), "n1": _node("n1-boot"), "n2": _node("n2-old")},
        {"n0": _node("n0-new"), "n1": _node("n1-boot"), "n2": _node("n2-new")},
    )
    two_node, single_node = _job_run(tmp_path, a), _job_run(tmp_path, b)

    assert destr028.job_node_contract(run, two_node) == []
    assert destr028.job_node_contract(run, single_node) == []
    assert two_node.sampler_stopped and single_node.sampler_stopped, (
        "the boot ended both samplers' transient units"
    )
    assert (
        json.loads((single_node.directory / "node-after.json").read_text())["boot_id"]
        == "n2-new"
    )


def test_a_rebooted_healthy_worker_fails_its_job(tmp_path: Path) -> None:
    a, _b = destr028.job_layout(("n0", "n1", "n2"), "destr028-run")
    run = _run(
        {"n0": _node("n0-old"), "n1": _node("n1-boot")},
        {"n0": _node("n0-new"), "n1": _node("n1-boot-2")},
    )

    errors = destr028.job_node_contract(run, _job_run(tmp_path, a))

    assert len(errors) == 1, errors
    assert errors[0].startswith(
        "destr028-run-a: n1: the job's healthy peer rebooted"
    ), errors


def test_a_master_that_did_not_reboot_fails_its_job(tmp_path: Path) -> None:
    _a, b = destr028.job_layout(("n0", "n1", "n2"), "destr028-run")
    run = _run({"n2": _node("n2-old")}, {"n2": _node("n2-old")})

    errors = destr028.job_node_contract(run, _job_run(tmp_path, b))

    assert "destr028-run-b: n2: Kubernetes Node boot ID did not change" in errors


# --------------------------------------------------------------------------- #
# The case in the catalog and on the command line
# --------------------------------------------------------------------------- #
def test_the_case_is_catalogued_behind_destr002() -> None:
    metadata = case_metadata(destr028.CASE_ID)

    assert metadata.confirmation == destr028.CONFIRMATION
    assert metadata.risk == "destructive"
    assert formal_predecessor(destr028.CASE_ID) == destr028.PREDECESSOR_CASE_ID
    assert isinstance(destr028.CASE, CaseRunner), destr028.CASE
    assert destr028.MIN_GROUP_NODES == 3
    assert destr028.LAYOUTS == {4: (2, 2), 3: (2, 1)}


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
    assert arguments.single_node_manifest == ""
