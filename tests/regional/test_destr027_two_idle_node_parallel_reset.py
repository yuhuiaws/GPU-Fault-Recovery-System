"""GF-REGIONAL-DESTR-027: two idle GPU nodes of one node group fault together
and recover as two parallel b220 workflows.

The runner's own logic: which nodes it may take (live, from the node group,
never from a profile's node list), the NOT_RUN(CAPABILITY_ABSENT) path a short
cluster takes through plan and execute, the plan identity it binds, and its
place in the catalog. The verdicts on the store records are the shared
oracle's (``tests/regional/test_workload_domain_oracle.py``); the per-node
reset contract is DESTR-001's.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL
from scripts.e2e.regional import run_destr027_two_idle_node_parallel_reset as destr027
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
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,62}")


def clean_snapshot(**changes: Any) -> dict[str, Any]:
    return {
        "name": "node-a",
        "uid": "uid-a",
        "boot_id": "boot-a",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "gpu_allocatable": "8",
        "ownership_annotations": {},
        "spare_pool_annotations": {},
        **changes,
    }


def clean_store(**changes: Any) -> dict[str, Any]:
    return {
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": sorted(destr027.AGENT_OPERATIONS),
            "generation": 7,
        },
        "event": None,
        "gpu_identity": {
            "product": "NVIDIA H200",
            "products": ["NVIDIA H200"],
            "device_count": 8,
            "source_boot_id": "boot-a",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
        **changes,
    }


def reasons(
    *,
    snapshot: dict[str, Any] | None = None,
    store: dict[str, Any] | None = None,
    business: list[dict[str, Any]] | None = None,
    arbiters: list[str] | None = None,
    gpu: list[dict[str, Any]] | None = None,
    state: str = "IDLE",
) -> list[str]:
    return destr027.node_ineligibility(
        "node-a",
        snapshot=snapshot or clean_snapshot(),
        store=store or clean_store(),
        business_workloads=business or [],
        arbiter_pods=arbiters or [],
        gpu_workloads=gpu or [],
        workload_view={"workload_state": state},
    )


# --------------------------------------------------------------------------- #
# Which nodes the case may take
# --------------------------------------------------------------------------- #
def test_a_clean_idle_node_is_eligible() -> None:
    assert reasons() == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"snapshot": clean_snapshot(ready="False")}, "node-a is not Ready"),
        ({"snapshot": clean_snapshot(unschedulable=True)}, "node-a is cordoned"),
        (
            {
                "snapshot": clean_snapshot(
                    spare_pool_annotations={
                        "gpu-fault.io/spare-pool-state": "AVAILABLE"
                    }
                )
            },
            "node-a is a declared warm spare",
        ),
        (
            {
                "snapshot": clean_snapshot(
                    ownership_annotations={"gpu-fault.io/incident-id": "i"}
                )
            },
            "node-a is owned by an incident",
        ),
        (
            {"business": [{"namespace": "team", "name": "trainer"}]},
            "node-a runs workloads",
        ),
        ({"gpu": [{"node": "node-a", "name": "holder"}]}, "node-a runs GPU Pods"),
        (
            {"arbiters": ["gpu-fault-system/executor-0"]},
            "node-a hosts controller/executor arbiter Pods",
        ),
        (
            {"store": clean_store(event={"xid": 46})},
            "node-a had an XID in the last 600s",
        ),
        (
            {
                "store": clean_store(
                    agent={"lifecycle_state": "DRAINING", "allowed_operations": []}
                )
            },
            "node-a Node Agent is not ACTIVE",
        ),
        ({"state": "UNKNOWN"}, "node-a workload state is UNKNOWN, not IDLE"),
        ({"state": "ACTIVE"}, "node-a workload state is ACTIVE, not IDLE"),
        (
            {"store": clean_store(gpu_identity=None)},
            "node-a has no GPU inventory identity",
        ),
    ],
    ids=[
        "not-ready",
        "cordoned",
        "warm-spare",
        "owned",
        "workload",
        "gpu-pod",
        "arbiter",
        "recent-xid",
        "agent",
        "unknown-workload-state",
        "active-workload-state",
        "no-gpu-identity",
    ],
)
def test_a_node_the_case_must_not_take_names_why(
    change: dict[str, Any], expected: str
) -> None:
    found = reasons(**change)

    assert any(item.startswith(expected) for item in found), found


def test_the_first_two_eligible_nodes_in_name_order_are_the_targets() -> None:
    candidates = {
        "node-d": [],
        "node-b": ["node-b is cordoned"],
        "node-c": [],
        "node-a": [],
    }

    assert destr027.choose_targets(candidates, dns_nodes=[]) == ("node-a", "node-c")


def test_a_pair_that_holds_every_kube_dns_endpoint_is_skipped() -> None:
    candidates = {"node-a": [], "node-b": [], "node-c": []}

    assert destr027.choose_targets(candidates, dns_nodes=["node-a", "node-b"]) == (
        "node-a",
        "node-c",
    )
    assert (
        destr027.choose_targets({"node-a": [], "node-b": []}, dns_nodes=["node-a"])
        is None
    )
    assert (
        destr027.choose_targets({"node-a": [], "node-b": ["x"]}, dns_nodes=[]) is None
    )


def test_the_reset_must_be_owned_by_the_node_agent() -> None:
    owned = {
        "capabilities": [
            {
                "capability": "gpuReset",
                "mode": "OWN",
                "owner": "gpu-fault-node-agent",
                "adapter": "node-action",
            }
        ],
        "warnings": [],
    }
    delegated = {
        "capabilities": [{**owned["capabilities"][0], "mode": "DELEGATE"}],
        "warnings": ["drift"],
    }

    assert destr027.profile_errors(owned) == []
    assert destr027.profile_errors(delegated) == [
        "gpuReset is not OWN by the Node Agent",
        "runtime profile has warnings",
    ]
    assert destr027.profile_errors(None) == [
        "runtime profile has no gpuReset capability"
    ]


def test_only_open_records_on_the_targets_refuse_the_case() -> None:
    store = {
        "active_workflow_incidents": [
            {
                "node_ids": ["node-c"],
                "workflow_request_id": "w-other",
                "status": "RUNNING",
            },
            {"node_ids": ["node-b"], "workflow_request_id": "w-b", "status": "PENDING"},
        ],
        "open_incidents": [
            {"node_ids": ["node-a"], "incident_id": "i-a", "state": "ESCALATED"},
            {"node_ids": ["node-d"], "incident_id": "i-d", "state": "ESCALATED"},
        ],
    }

    assert destr027.residue_errors(store, ("node-a", "node-b")) == [
        "['node-b'] already have an open workflow w-b (PENDING)",
        "['node-a'] already have incident i-a in ESCALATED",
    ]


# --------------------------------------------------------------------------- #
# A cluster that cannot carry the case: NOT_RUN(CAPABILITY_ABSENT)
# --------------------------------------------------------------------------- #
class OneNodeCluster:
    """A fixture whose only GPU node group has one node."""

    def __init__(self, settings: RegionalLiveSettings) -> None:
        self.settings = settings

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-1", "cluster_id": self.settings.cluster_id}

    def gpu_nodes(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "eks-0",
                "uid": "uid-0",
                "gpu_allocatable": 8,
                "labels": {EKS_NODEGROUP_LABEL: "gpu-group"},
            }
        ]


def settings(tmp_path: Path) -> destr027.Settings:
    for name in ("cpu.kubeconfig", "gpu.kubeconfig"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return destr027.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            gpu_context="gpu",
            namespace="gpu-fault-system",
            cluster_id=CLUSTER,
            region="us-west-2",
        ),
        host_probe_image="registry/probe@sha256:abc",
        predecessor_path=tmp_path / "predecessor.json",
        node_group="",
    )


def test_a_short_node_group_plans_a_not_run_naming_the_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destr027, "RegionalLiveFixture", OneNodeCluster)
    case_dir = tmp_path / "cases" / destr027.CASE_ID
    case_dir.mkdir(parents=True)

    preflight = destr027.read_only_preflight(settings(tmp_path), case_dir)
    details = destr027.plan_details(settings(tmp_path), preflight)

    assert preflight["errors"] == []
    assert preflight["not_run"] == {
        "verdict": "NOT_RUN",
        "reason": "CAPABILITY_ABSENT",
        "detail": "needs >=2 GPU nodes in one node group; cluster gpu-eks-cluster "
        f"group gpu-group ({EKS_NODEGROUP_LABEL}) has 1",
    }
    assert details["not_run"] == preflight["not_run"]
    assert "targets" not in details
    assert (
        json.loads((case_dir / "preflight.json").read_text())["not_run"]
        == (preflight["not_run"])
    )


def test_executing_a_not_run_plan_records_the_verdict_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(destr027, "RegionalLiveFixture", OneNodeCluster)
    case_dir = tmp_path / "cases" / destr027.CASE_ID
    case_dir.mkdir(parents=True)
    record = oracle.capability_absent("needs >=2 GPU nodes in one node group; x")
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"not_run": record, "node_group_gate": {"minimum": 2}}}),
        encoding="utf-8",
    )

    code = destr027.execute_case(
        settings(tmp_path), tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )

    evidence = json.loads((case_dir / f"{destr027.CASE_ID}.json").read_text())
    assert code == 0
    assert {key: evidence[key] for key in ("verdict", "reason", "detail")} == record
    assert evidence["mutation_performed"] is False
    assert evidence["release_id"] == "release-1"
    assert not (case_dir / "host-probes").exists(), "a NOT_RUN made a probe"


def test_the_not_run_result_is_never_a_pass() -> None:
    result = destr027.not_run_result(
        2, {"release_id": "r", "cluster_id": "c"}, oracle.capability_absent("short")
    )

    assert (result["verdict"], result["reason"], result["detail"]) == (
        "NOT_RUN",
        "CAPABILITY_ABSENT",
        "short",
    )
    assert result["case_id"] == destr027.CASE_ID


# --------------------------------------------------------------------------- #
# The plan identity the execution must find unchanged
# --------------------------------------------------------------------------- #
def planned_preflight() -> dict[str, Any]:
    return {
        "release_id": "release-1",
        "node_group": {
            "label": HYPERPOD_INSTANCE_GROUP_LABEL,
            "name": "p5en",
            "nodes": ["a", "b", "c"],
        },
        "targets": ["a", "b"],
        "nodes": {
            "a": clean_snapshot(name="a", uid="uid-a", boot_id="boot-a"),
            "b": clean_snapshot(name="b", uid="uid-b", boot_id="boot-b"),
        },
        "stores": {
            "a": {**clean_store(), "profile": {"profile_version": "p-1"}},
            "b": clean_store(),
        },
        "runtime_identity": {"release_id": "release-1"},
        "storm": {"cluster_gpu_agents": 4, "breaker": None},
        "node_group_gate": {"minimum_gpu_nodes_in_one_node_group": 2},
        "focused_tests": {"passed": True},
        "predecessor": {"valid": True},
    }


def test_the_plan_binds_the_targets_and_refuses_a_rebooted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The source digest of the focused tests is the live driver guard's
    # concern; recording the result is enough here.
    monkeypatch.setattr(
        destr027,
        "record_focused_tests",
        lambda details, result: details.update(focused_tests=result),
    )
    preflight = planned_preflight()
    identity = destr027.plan_identity(preflight)
    case_dir = tmp_path
    details = destr027.plan_details(settings(tmp_path), preflight)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": details}), encoding="utf-8"
    )

    destr027.verify_plan_identity(case_dir, preflight)
    rebooted = planned_preflight()
    rebooted["nodes"]["b"]["boot_id"] = "boot-b2"

    assert identity["targets"] == {
        "a": {
            "uid": "uid-a",
            "boot_id": "boot-a",
            "agent_generation": 7,
            "runtime_profile_version": "p-1",
        },
        "b": {
            "uid": "uid-b",
            "boot_id": "boot-b",
            "agent_generation": 7,
            "runtime_profile_version": None,
        },
    }
    assert identity["node_group"] == {
        "label": HYPERPOD_INSTANCE_GROUP_LABEL,
        "name": "p5en",
    }
    assert details["storm_premise"] == {
        "threshold": 3,
        "cluster_gpu_agents": 4,
        "breaker_state": "CLOSED",
        "case_counted_nodes": 2,
    }
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        destr027.verify_plan_identity(case_dir, rebooted)


def test_run_ids_keep_rounds_apart_and_fit_the_probe_names(tmp_path: Path) -> None:
    first = destr027.run_identity(tmp_path / "round-13-hp", 1)
    second = destr027.run_identity(tmp_path / "other" / "round-14-hp", 1)

    assert first != second
    assert first.startswith("destr027-hp-"), first
    assert RUN_ID.fullmatch(f"{first}-nb") is not None


# --------------------------------------------------------------------------- #
# The case in the catalog and on the command line
# --------------------------------------------------------------------------- #
def test_the_case_is_catalogued_behind_destr001() -> None:
    metadata = case_metadata(destr027.CASE_ID)

    assert metadata.confirmation == destr027.CONFIRMATION
    assert metadata.risk == "destructive"
    assert formal_predecessor(destr027.CASE_ID) == destr027.PREDECESSOR_CASE_ID
    assert isinstance(destr027.CASE, CaseRunner), destr027.CASE
    assert destr027.MIN_GROUP_NODES == 2


@pytest.mark.parametrize("flag", ["--node", "--fault-node", "--node-a", "--node-b"])
def test_no_profile_node_can_choose_the_targets(flag: str) -> None:
    """A site profile binds any flag named like one of its keys; the runner
    takes no node flag, so its targets always come from the live group."""

    with pytest.raises(SystemExit):
        destr027.parser().parse_args(["--run-dir", "run", flag, "node-x"])


def test_the_node_group_override_is_the_only_selection_flag(tmp_path: Path) -> None:
    arguments = destr027.parser().parse_args(
        ["--run-dir", str(tmp_path), "--plan", "--node-group", "p5en"]
    )

    assert isinstance(arguments, argparse.Namespace), arguments
    assert arguments.node_group == "p5en"
