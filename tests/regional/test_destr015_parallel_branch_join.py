from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr015_verdicts as verdicts
from scripts.e2e.regional import run_destr015_parallel_branch_join as destr015
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

yaml = importlib.import_module("yaml")

ROOT = Path(__file__).resolve().parents[2]
NODE_A = "node-b"
NODE_B = "node-c"
NODES = (NODE_A, NODE_B)
INCIDENT = "inc-destr015-test"
REQUEST = "workflow-destr015-test"
T0 = "2026-09-06T10:00:00+00:00"


def _at(minute: int, second: int = 0) -> str:
    return f"2026-09-06T10:{minute:02d}:{second:02d}+00:00"


def _step(
    operation: str,
    nodes: list[str],
    branch_id: str,
    *,
    depends: list[int] | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "execution_owner": "test",
        "node_ids": list(nodes),
        "gpu_uuids": [],
        "workload_ids": ["training/pytorchjob/job"],
        "parameters": dict(parameters or {}),
        "depends_on_step_indexes": list(depends or []),
        "branch_id": branch_id,
        "branch_node_ids": list(nodes) if branch_id.startswith("branch:") else [],
    }


def _execution(
    index: int,
    operation: str,
    status: str,
    *,
    started_at: str,
    updated_at: str,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step_index": index,
        "operation": operation,
        "status": status,
        "phase": "official",
        "adapter_operation_id": f"remote/{index}",
        "error": error,
        "details": dict(details or {}),
        "started_at": started_at,
        "updated_at": updated_at,
    }


def _branch(node: str, base: int, minute: int, second: int) -> list[dict[str, Any]]:
    """Seven hardware steps of one node branch, each taking one minute."""

    return [
        _execution(
            base + offset,
            operation,
            "SUCCEEDED",
            started_at=_at(minute + offset, second),
            updated_at=_at(minute + offset, second + 30),
        )
        for offset, operation in enumerate(verdicts.BRANCH_OPERATIONS)
    ]


def happy_workflow() -> dict[str, Any]:
    a = f"branch:{NODE_A}"
    b = f"branch:{NODE_B}"
    steps = [
        _step("FREEZE_EVIDENCE", [NODE_A, NODE_B], "shared"),  # 0
        _step("STOP_WORKLOADS", [NODE_A, NODE_B], "shared", depends=[0]),  # 1
    ]
    for node, branch in ((NODE_A, a), (NODE_B, b)):
        previous = 1
        for operation in verdicts.BRANCH_OPERATIONS:
            steps.append(_step(operation, [node], branch, depends=[previous]))
            previous = len(steps) - 1
    steps.append(
        _step(
            "RESTART_WORKLOAD",
            [NODE_A, NODE_B],
            "join",
            depends=[8, 15],
            parameters={"source_gpu_count": 16, "restart_budget": 1},
        )
    )  # 16
    executions = [
        _execution(0, "FREEZE_EVIDENCE", "SUCCEEDED", started_at=T0, updated_at=T0),
        _execution(
            1,
            "STOP_WORKLOADS",
            "SUCCEEDED",
            started_at=_at(0, 5),
            updated_at=_at(0, 50),
        ),
        # The two branches run side by side: node-c starts ten seconds after
        # node-b and each step takes about half a minute.
        *_branch(NODE_A, 2, 1, 0),
        *_branch(NODE_B, 9, 1, 10),
        _execution(
            16,
            "RESTART_WORKLOAD",
            "SUCCEEDED",
            started_at=_at(8, 0),
            updated_at=_at(9, 0),
            details={
                "source_gpu_count": 16,
                "target_gpu_count": 16,
                "restart_count": 1,
            },
        ),
    ]
    return {
        "request_id": REQUEST,
        "incident_id": INCIDENT,
        "status": "SUCCEEDED",
        "dag_enabled": True,
        "dag_revision": 1,
        "merge_revision": 1,
        "official_steps": steps,
        "step_executions": executions,
        "completed_step_indexes": list(range(len(steps))),
        "superseded_step_indexes": [],
        "branch_escalation_counts": {},
        "exhausted_branch_ids": [],
        "workload_withdrawn_at": None,
        "terminal_failure_reason": None,
        "lifetime_deadline_at": "2026-09-06T11:00:00+00:00",
    }


def happy_incident() -> dict[str, Any]:
    return {
        "incident_id": INCIDENT,
        "state": "RECOVERED",
        "node_ids": [NODE_A, NODE_B],
        "workflow_request_id": REQUEST,
        "job_id": "destr015-job",
        "attempt_id": "destr015-job-a001",
    }


def happy_budget() -> dict[str, Any]:
    return {"job_id": "destr015-job", "budget": 1, "restart_count": 1}


def _errors(workflow: dict[str, Any], incident: dict[str, Any] | None = None):
    return verdicts.workflow_errors(
        workflow,
        incident or happy_incident(),
        nodes=NODES,
        expected_gpu_count=16,
        restart_budget=happy_budget(),
    )


def _branch_exec(workflow: dict[str, Any], node: str, operation: str) -> dict[str, Any]:
    steps = workflow["official_steps"]
    for item in workflow["step_executions"]:
        step = steps[item["step_index"]]
        if step["node_ids"] == [node] and item["operation"] == operation:
            return item
    raise AssertionError(f"no {operation} execution for {node}")


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_confirmation_token_matches_the_case_contract_derivation() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr015.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="destructive",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-015",
        predecessor=destr015.PREDECESSOR_CASE_ID,
    )
    assert destr015.CONFIRMATION == metadata.confirmation == "DESTR015_EXECUTE"
    assert destr015.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-012"


# --------------------------------------------------------------------------- #
# Workflow verdicts
# --------------------------------------------------------------------------- #
def test_destr015_workflow_contract_requires_one_join_after_parallel_branches() -> None:
    assert _errors(happy_workflow()) == []


def test_a_workflow_that_did_not_succeed_fails() -> None:
    workflow = happy_workflow()
    workflow["status"] = "FAILED"
    assert any("not SUCCEEDED" in item for item in _errors(workflow)), (
        "expected any('not SUCCEEDED' in item for item in _errors(workflow)) to be true"
    )

    incident = happy_incident()
    incident["state"] = "QUARANTINED"
    assert any("RECOVERED" in item for item in _errors(happy_workflow(), incident)), (
        "expected any('RECOVERED' in item for item in _errors(happy_workflow(), incid... to be true"
    )


def test_the_join_must_run_exactly_once() -> None:
    workflow = happy_workflow()
    workflow["step_executions"] = [
        item
        for item in workflow["step_executions"]
        if item["operation"] != "RESTART_WORKLOAD"
    ]
    assert any("RESTART_WORKLOAD" in item for item in _errors(workflow)), (
        "expected any('RESTART_WORKLOAD' in item for item in _errors(workflow)) to be true"
    )

    doubled = happy_workflow()
    doubled["official_steps"].append(
        _step("RESTART_WORKLOAD", [NODE_A, NODE_B], "join", depends=[16])
    )
    doubled["completed_step_indexes"].append(17)
    assert any("exactly one RESTART_WORKLOAD" in item for item in _errors(doubled)), (
        "expected any('exactly one RESTART_WORKLOAD' in item for item in _errors(doub... to be true"
    )


def test_each_node_branch_needs_every_hardware_step_once() -> None:
    workflow = happy_workflow()
    reset = _branch_exec(workflow, NODE_B, "RESET_GPU")
    workflow["step_executions"].remove(reset)
    errors = _errors(workflow)
    assert any(NODE_B in item and "RESET_GPU" in item for item in errors), errors

    twice = happy_workflow()
    twice["step_executions"].append(
        dict(_branch_exec(twice, NODE_A, "RESET_GPU"), adapter_operation_id="remote/x")
    )
    assert any(NODE_A in item and "RESET_GPU" in item for item in _errors(twice)), (
        "expected any(NODE_A in item and 'RESET_GPU' in item for item in _errors(twice)) to be true"
    )


def test_a_branch_step_that_did_not_succeed_fails() -> None:
    workflow = happy_workflow()
    _branch_exec(workflow, NODE_A, "VALIDATE_GPU")["status"] = "FAILED"
    assert any(
        "VALIDATE_GPU" in item and "SUCCEEDED" in item for item in _errors(workflow)
    ), (
        "expected any( 'VALIDATE_GPU' in item and 'SUCCEEDED' in item for item in _er... to be true"
    )


def test_a_node_mutation_beyond_reset_fails() -> None:
    workflow = happy_workflow()
    workflow["official_steps"].append(
        _step("RESTART_NODE", [NODE_B], f"branch:{NODE_B}", depends=[15])
    )
    workflow["completed_step_indexes"].append(17)
    assert any("RESTART_NODE" in item for item in _errors(workflow)), (
        "expected any('RESTART_NODE' in item for item in _errors(workflow)) to be true"
    )


def test_the_join_must_depend_on_both_branch_tails() -> None:
    workflow = happy_workflow()
    workflow["official_steps"][16]["depends_on_step_indexes"] = [8]
    assert any("depend" in item and NODE_B in item for item in _errors(workflow)), (
        "expected any('depend' in item and NODE_B in item for item in _errors(workflow)) to be true"
    )


def test_the_join_must_start_after_both_branches_released_their_nodes() -> None:
    workflow = happy_workflow()
    restart = _branch_exec(workflow, NODE_A, "RESTORE_SCHEDULING")
    # node-b's release lands after the restart had already begun.
    restart["updated_at"] = _at(8, 30)
    assert any(
        "before" in item and "RESTORE_SCHEDULING" in item for item in _errors(workflow)
    ), (
        "expected any( 'before' in item and 'RESTORE_SCHEDULING' in item for item in ... to be true"
    )


def test_serialized_branches_are_reported_as_not_parallel() -> None:
    workflow = happy_workflow()
    steps = workflow["official_steps"]
    for item in workflow["step_executions"]:
        if steps[item["step_index"]]["node_ids"] == [NODE_B]:
            # Shift node-c's whole branch to after node-b finished.
            offset = int(item["started_at"][14:16]) + 7
            item["started_at"] = _at(offset, 0)
            item["updated_at"] = _at(offset, 30)
    workflow["step_executions"][-1]["started_at"] = _at(16, 0)
    workflow["step_executions"][-1]["updated_at"] = _at(17, 0)
    assert any("parallel" in item for item in _errors(workflow)), (
        "expected any('parallel' in item for item in _errors(workflow)) to be true"
    )


def test_restart_details_must_carry_the_full_allocation_and_one_restart() -> None:
    workflow = happy_workflow()
    workflow["step_executions"][-1]["details"]["target_gpu_count"] = 8
    assert any("target GPU count" in item for item in _errors(workflow)), (
        "expected any('target GPU count' in item for item in _errors(workflow)) to be true"
    )

    budget = dict(happy_budget(), restart_count=2)
    errors = verdicts.workflow_errors(
        happy_workflow(),
        happy_incident(),
        nodes=NODES,
        expected_gpu_count=16,
        restart_budget=budget,
    )
    assert any("restart budget" in item for item in errors), (
        "expected any('restart budget' in item for item in errors) to be true"
    )


def test_the_dag_must_be_a_clean_in_window_merge() -> None:
    plain = happy_workflow()
    plain["dag_enabled"] = False
    assert any("dag_enabled" in item for item in _errors(plain)), (
        "expected any('dag_enabled' in item for item in _errors(plain)) to be true"
    )

    superseded = happy_workflow()
    superseded["superseded_step_indexes"] = [9]
    assert any("superseded" in item for item in _errors(superseded)), (
        "expected any('superseded' in item for item in _errors(superseded)) to be true"
    )

    escalated = happy_workflow()
    escalated["branch_escalation_counts"] = {NODE_A: 1}
    assert any("escalat" in item for item in _errors(escalated)), (
        "expected any('escalat' in item for item in _errors(escalated)) to be true"
    )

    withdrawn = happy_workflow()
    withdrawn["workload_withdrawn_at"] = T0
    assert any("withdrawn" in item for item in _errors(withdrawn)), (
        "expected any('withdrawn' in item for item in _errors(withdrawn)) to be true"
    )

    shared_twice = happy_workflow()
    shared_twice["official_steps"].append(
        _step("STOP_WORKLOADS", [NODE_B], f"branch:{NODE_B}", depends=[9])
    )
    shared_twice["completed_step_indexes"].append(17)
    assert any(
        "exactly one STOP_WORKLOADS" in item for item in _errors(shared_twice)
    ), (
        "expected any('exactly one STOP_WORKLOADS' in item for item in _errors(shared... to be true"
    )


def test_the_shared_stop_must_cover_both_nodes() -> None:
    workflow = happy_workflow()
    workflow["official_steps"][1]["node_ids"] = [NODE_A]
    assert any(
        "STOP_WORKLOADS" in item and NODE_B in item for item in _errors(workflow)
    ), (
        "expected any( 'STOP_WORKLOADS' in item and NODE_B in item for item in _error... to be true"
    )


# --------------------------------------------------------------------------- #
# Injection
# --------------------------------------------------------------------------- #
def _state(node: str, request_id: str = REQUEST, xid: int = 46) -> dict[str, Any]:
    return {
        "event": {
            "event_id": f"evt-{node}",
            "xid": xid,
            "node_id": node,
            "evidence_ref": f"kmsg://{node}/1",
        },
        "decision": {"official_action": "RESET_GPU", "disposition": "EXECUTABLE"},
        "incident": {"incident_id": INCIDENT, "workflow_request_id": request_id},
        "workflow": {"request_id": request_id, "status": "RUNNING"},
    }


def test_injection_must_land_both_events_in_one_workflow() -> None:
    assert verdicts.injection_errors(_state(NODE_A), _state(NODE_B), nodes=NODES) == []
    split = verdicts.injection_errors(
        _state(NODE_A), _state(NODE_B, "workflow-other"), nodes=NODES
    )
    assert any("one workflow" in item for item in split), (
        "expected any('one workflow' in item for item in split) to be true"
    )
    wrong_xid = verdicts.injection_errors(
        _state(NODE_A), _state(NODE_B, xid=79), nodes=NODES
    )
    assert any("XID 46" in item for item in wrong_xid), (
        "expected any('XID 46' in item for item in wrong_xid) to be true"
    )
    replayed = _state(NODE_B)
    replayed["event"]["evidence_ref"] = "api-replay://x"
    assert any(
        "kmsg://" in item
        for item in verdicts.injection_errors(_state(NODE_A), replayed, nodes=NODES)
    ), (
        "expected any( 'kmsg://' in item for item in verdicts.injection_errors(_state... to be true"
    )


def test_injection_spread_must_fit_inside_the_aggregation_window() -> None:
    assert (
        verdicts.spread_errors(
            {NODE_A: T0, NODE_B: _at(0, 2)}, aggregation_window_seconds=5
        )
        == []
    )
    late = verdicts.spread_errors(
        {NODE_A: T0, NODE_B: _at(0, 9)}, aggregation_window_seconds=5
    )
    assert any("aggregation window" in item for item in late), (
        "expected any('aggregation window' in item for item in late) to be true"
    )


# --------------------------------------------------------------------------- #
# Host, nodes, workload, provider
# --------------------------------------------------------------------------- #
def _ledger(*rows: tuple[str, str, str]) -> list[dict[str, Any]]:
    return [
        {
            "command_id": command_id,
            "operation": operation,
            "state": state,
            "attempt": 1,
            "completed_at": _at(5),
            "started_at": _at(4),
        }
        for command_id, operation, state in rows
    ]


def _host(node: str, *, boot_id: str = "boot-1", after: bool = False) -> dict[str, Any]:
    rows = [("old-quiesce", "QUIESCE_GPU_SERVICES", "SUCCEEDED")]
    if after:
        rows.extend(
            [
                (f"{node}-quiesce", "QUIESCE_GPU_SERVICES", "SUCCEEDED"),
                (f"{node}-verify", "VERIFY_NO_GPU_CLIENTS", "SUCCEEDED"),
                (f"{node}-reset", "RESET_GPU", "SUCCEEDED"),
                (f"{node}-restore", "RESTORE_GPU_SERVICES", "SUCCEEDED"),
            ]
        )
    return {
        "boot_id": boot_id,
        "ledger": _ledger(*rows),
        "quiesce_states": [],
        "gpu_fault_timers": [],
        "gpu_inventory": [{"pci_bdf": f"0000:0{index}:00.0"} for index in range(8)],
        "services": {"kubelet.service": {"ActiveState": "active"}},
    }


def _hosts(**overrides: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    value = {
        node: {"before": _host(node), "after": _host(node, after=True)}
        for node in NODES
    }
    for key, item in overrides.items():
        node, phase = key.rsplit("_", 1)
        value[node.replace("_", "-")][phase] = item
    return value


def test_host_happy_shape_passes() -> None:
    assert verdicts.host_errors(_hosts(), nodes=NODES, expected_gpu_count=8) == []


def test_host_rejects_a_reboot_a_missing_reset_and_a_double_reset() -> None:
    rebooted = _hosts(node_c_after=_host(NODE_B, boot_id="boot-2", after=True))
    assert any(
        "boot" in item
        for item in verdicts.host_errors(rebooted, nodes=NODES, expected_gpu_count=8)
    ), (
        "expected any( 'boot' in item for item in verdicts.host_errors(rebooted, node... to be true"
    )

    no_reset = _host(NODE_A, after=True)
    no_reset["ledger"] = [
        row for row in no_reset["ledger"] if row["operation"] != "RESET_GPU"
    ]
    errors = verdicts.host_errors(
        _hosts(node_b_after=no_reset), nodes=NODES, expected_gpu_count=8
    )
    assert any(NODE_A in item and "RESET_GPU" in item for item in errors), (
        "expected any(NODE_A in item and 'RESET_GPU' in item for item in errors) to be true"
    )

    twice = _host(NODE_A, after=True)
    twice["ledger"].append(twice["ledger"][-2] | {"command_id": "second-reset"})
    errors = verdicts.host_errors(
        _hosts(node_b_after=twice), nodes=NODES, expected_gpu_count=8
    )
    assert any(NODE_A in item and "RESET_GPU" in item for item in errors), (
        "expected any(NODE_A in item and 'RESET_GPU' in item for item in errors) to be true"
    )


def test_host_rejects_quiesce_residue_and_a_stopped_service() -> None:
    residue = _host(NODE_B, after=True)
    residue["quiesce_states"] = [{"incident_id": INCIDENT}]
    residue["services"]["kubelet.service"] = {"ActiveState": "inactive"}
    errors = verdicts.host_errors(
        _hosts(node_c_after=residue), nodes=NODES, expected_gpu_count=8
    )
    assert any("quiesce" in item for item in errors), (
        "expected any('quiesce' in item for item in errors) to be true"
    )
    assert any("kubelet.service" in item for item in errors), (
        "expected any('kubelet.service' in item for item in errors) to be true"
    )


def _node(node: str, **overrides: Any) -> dict[str, Any]:
    return {
        "name": node,
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": "8",
        **overrides,
    }


def test_schedulability_requires_both_nodes_released() -> None:
    nodes = {NODE_A: _node(NODE_A), NODE_B: _node(NODE_B)}
    assert verdicts.schedulability_errors(nodes, nodes=NODES) == []
    cordoned = {
        NODE_A: _node(NODE_A, unschedulable=True),
        NODE_B: _node(
            NODE_B,
            taints=[
                {
                    "key": "gpu-fault.io/quarantined",
                    "value": "x",
                    "effect": "NoSchedule",
                }
            ],
            ownership_annotations={"gpu-fault.io/owner": INCIDENT},
        ),
    }
    errors = verdicts.schedulability_errors(cordoned, nodes=NODES)
    assert any(NODE_A in item and "schedulable" in item for item in errors), (
        "expected any(NODE_A in item and 'schedulable' in item for item in errors) to be true"
    )
    assert any(NODE_B in item and "taint" in item for item in errors), (
        "expected any(NODE_B in item and 'taint' in item for item in errors) to be true"
    )
    assert any(NODE_B in item and "annotation" in item for item in errors), (
        "expected any(NODE_B in item and 'annotation' in item for item in errors) to be true"
    )


def _pods(*nodes: str, prefix: str = "new") -> list[dict[str, Any]]:
    return [
        {
            "name": f"pod-{index}",
            "uid": f"{prefix}-{index}",
            "node": node,
            "phase": "Running",
            "ready": True,
        }
        for index, node in enumerate(nodes)
    ]


def test_workload_must_be_replaced_once_on_the_same_two_nodes() -> None:
    source = {"old-0", "old-1"}
    assert (
        verdicts.workload_errors(
            pods=_pods(NODE_A, NODE_B), source_uids=source, nodes=NODES
        )
        == []
    )
    assert any(
        "UID" in item
        for item in verdicts.workload_errors(
            pods=_pods(NODE_A, NODE_B, prefix="old"), source_uids=source, nodes=NODES
        )
    ), (
        "expected any( 'UID' in item for item in verdicts.workload_errors( pods=_pods... to be true"
    )
    assert any(
        "nodes" in item
        for item in verdicts.workload_errors(
            pods=_pods(NODE_A, "node-d"), source_uids=source, nodes=NODES
        )
    ), (
        "expected any( 'nodes' in item for item in verdicts.workload_errors( pods=_po... to be true"
    )
    assert any(
        "two" in item
        for item in verdicts.workload_errors(
            pods=_pods(NODE_A), source_uids=source, nodes=NODES
        )
    ), (
        "expected any( 'two' in item for item in verdicts.workload_errors( pods=_pods... to be true"
    )


def test_cloudtrail_must_show_no_provider_mutation() -> None:
    assert verdicts.cloudtrail_errors([]) == []
    assert verdicts.cloudtrail_errors(
        [{"event_name": "BatchRebootClusterNodes", "username": "x"}]
    ), (
        "expected verdicts.cloudtrail_errors( [{'event_name': 'BatchRebootClusterNode... to be true"
    )


# --------------------------------------------------------------------------- #
# Timeline and lifetime arithmetic
# --------------------------------------------------------------------------- #
def test_step_transitions_record_only_changes() -> None:
    first = [_execution(5, "RESET_GPU", "WAITING", started_at=T0, updated_at=T0)]
    state, changes = verdicts.step_transitions({}, first)
    assert len(changes) == 1
    state, changes = verdicts.step_transitions(state, first)
    assert changes == []
    second = [_execution(5, "RESET_GPU", "SUCCEEDED", started_at=T0, updated_at=T0)]
    state, changes = verdicts.step_transitions(state, second)
    assert [item["status"] for item in changes] == ["SUCCEEDED"]


def test_lifetime_arithmetic_refuses_an_estimate_over_the_lifetime() -> None:
    estimate = verdicts.estimated_duration_seconds()
    assert 0 < estimate < 3600
    assert (
        verdicts.lifetime_errors(estimated_seconds=estimate, lifetime_seconds=3600)
        == []
    )
    assert verdicts.lifetime_errors(estimated_seconds=estimate, lifetime_seconds=600), (
        "expected verdicts.lifetime_errors(estimated_seconds=estimate, lifetime_secon... to be true"
    )
    assert verdicts.lifetime_errors(
        estimated_seconds=estimate, lifetime_seconds=None
    ), (
        "expected verdicts.lifetime_errors(estimated_seconds=estimate, lifetime_secon... to be true"
    )


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def _preflight_arguments(**overrides: Any) -> dict[str, Any]:
    profile = {
        "capabilities": [
            {
                "capability": name,
                "mode": "OWN",
                "owner": "gpu-fault-kubernetes-adapter",
                "adapter": "regional-cluster-executor",
            }
            for name in ("workloadStop", "workloadRestart")
        ]
        + [{"capability": "gpuReset", "mode": "OWN", "owner": "gpu-fault-node-agent"}]
    }
    value: dict[str, Any] = {
        "node_snapshots": {NODE_A: _node(NODE_A), NODE_B: _node(NODE_B)},
        "agents": {
            node: {
                "lifecycle_state": "ACTIVE",
                "allowed_operations": list(verdicts.AGENT_OPERATIONS),
            }
            for node in NODES
        },
        "profile": profile,
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "gpu_workloads": [],
        "business_workloads": {NODE_A: [], NODE_B: []},
        "predecessor": {"valid": True},
        "tests": {"passed": True},
        "control_env": {"aggregation_window_seconds": 5, "job_lifetime_seconds": 3600},
        "runtime_identity_errors": [],
    }
    value.update(overrides)
    return value


def test_preflight_happy_shape_passes() -> None:
    assert destr015.preflight_errors(nodes=NODES, **_preflight_arguments()) == []


def test_preflight_refuses_identical_or_unready_nodes_and_inactive_agents() -> None:
    assert destr015.preflight_errors(
        nodes=(NODE_A, NODE_A), **_preflight_arguments()
    ), (
        "expected destr015.preflight_errors(nodes=(NODE_A, NODE_A), **_preflight_argu... to be true"
    )
    unready = _preflight_arguments()
    unready["node_snapshots"][NODE_B] = _node(NODE_B, unschedulable=True)
    assert any(
        NODE_B in item for item in destr015.preflight_errors(nodes=NODES, **unready)
    ), (
        "expected any( NODE_B in item for item in destr015.preflight_errors(nodes=NOD... to be true"
    )
    inactive = _preflight_arguments()
    inactive["agents"][NODE_A] = {
        "lifecycle_state": "DRAINING",
        "allowed_operations": list(verdicts.AGENT_OPERATIONS),
    }
    assert any(
        "Node Agent" in item
        for item in destr015.preflight_errors(nodes=NODES, **inactive)
    ), (
        "expected any( 'Node Agent' in item for item in destr015.preflight_errors(nod... to be true"
    )
    partial = _preflight_arguments()
    partial["agents"][NODE_A] = {
        "lifecycle_state": "ACTIVE",
        "allowed_operations": ["QUIESCE_GPU_SERVICES"],
    }
    assert any(
        "RESET_GPU" in item
        for item in destr015.preflight_errors(nodes=NODES, **partial)
    ), (
        "expected any( 'RESET_GPU' in item for item in destr015.preflight_errors(node... to be true"
    )


def test_preflight_refuses_busy_cluster_and_bad_ownership() -> None:
    busy = _preflight_arguments(
        gpu_workloads=[{"node": NODE_A}],
        queue={"depth": 3},
        remote_commands={"open_by_cluster": {"c": 1}},
    )
    errors = destr015.preflight_errors(nodes=NODES, **busy)
    assert any("GPU workload" in item for item in errors), (
        "expected any('GPU workload' in item for item in errors) to be true"
    )
    assert any("queue" in item for item in errors), (
        "expected any('queue' in item for item in errors) to be true"
    )
    assert any("remote command" in item for item in errors), (
        "expected any('remote command' in item for item in errors) to be true"
    )
    delegated = _preflight_arguments()
    delegated["profile"]["capabilities"][0]["mode"] = "DELEGATE"
    assert any(
        "OWN" in item for item in destr015.preflight_errors(nodes=NODES, **delegated)
    ), (
        "expected any( 'OWN' in item for item in destr015.preflight_errors(nodes=NODE... to be true"
    )
    observe = _preflight_arguments()
    observe["profile"]["capabilities"][2]["mode"] = "OBSERVE"
    assert any(
        "gpuReset" in item for item in destr015.preflight_errors(nodes=NODES, **observe)
    ), (
        "expected any( 'gpuReset' in item for item in destr015.preflight_errors(nodes... to be true"
    )


def test_preflight_refuses_a_narrow_window_a_short_lifetime_and_bad_predecessor() -> (
    None
):
    narrow = _preflight_arguments(
        control_env={"aggregation_window_seconds": 1, "job_lifetime_seconds": 3600}
    )
    assert any(
        "aggregation window" in item
        for item in destr015.preflight_errors(nodes=NODES, **narrow)
    ), (
        "expected any( 'aggregation window' in item for item in destr015.preflight_er... to be true"
    )
    short = _preflight_arguments(
        control_env={"aggregation_window_seconds": 5, "job_lifetime_seconds": 600}
    )
    assert any(
        "lifetime" in item for item in destr015.preflight_errors(nodes=NODES, **short)
    ), (
        "expected any( 'lifetime' in item for item in destr015.preflight_errors(nodes... to be true"
    )
    failed = _preflight_arguments(
        predecessor={"valid": False},
        tests={"passed": False},
        runtime_identity_errors=["drift"],
    )
    errors = destr015.preflight_errors(nodes=NODES, **failed)
    assert any("DESTR-012" in item for item in errors), (
        "expected any('DESTR-012' in item for item in errors) to be true"
    )
    assert any("regression" in item for item in errors), (
        "expected any('regression' in item for item in errors) to be true"
    )
    assert "drift" in errors


# --------------------------------------------------------------------------- #
# Runner helpers
# --------------------------------------------------------------------------- #
def test_plan_identity_digest_is_stable_across_key_order() -> None:
    preflight = {
        "release_id": "rel",
        "nodes": {
            NODE_A: {"uid": "a", "boot_id": "ba"},
            NODE_B: {"uid": "b", "boot_id": "bb"},
        },
        "store": {"profile": {"profile_version": "p1"}},
        "runtime_identity": {"x": 1},
    }
    identity = destr015.plan_identity(preflight, nodes=NODES)
    shuffled = json.loads(json.dumps(identity, sort_keys=True))
    shuffled = dict(reversed(list(shuffled.items())))
    assert destr015.identity_digest(identity) == destr015.identity_digest(shuffled)
    assert identity["node_uids"] == {NODE_A: "a", NODE_B: "b"}
    assert identity["node_boot_ids"] == {NODE_A: "ba", NODE_B: "bb"}


def test_two_node_manifest_pins_master_and_worker_to_the_two_nodes(
    tmp_path: Path,
) -> None:
    rendered = destr015.render_two_node_manifest(
        destr015.DEFAULT_MANIFEST,
        tmp_path / "w.yaml",
        master_node=NODE_A,
        worker_node=NODE_B,
    )
    document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    assert replicas["Master"]["template"]["spec"]["nodeName"] == NODE_A
    assert replicas["Worker"]["template"]["spec"]["nodeName"] == NODE_B
    assert rendered.stat().st_mode & 0o777 == 0o600


def test_two_node_manifest_refuses_the_same_node_twice(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        destr015.render_two_node_manifest(
            destr015.DEFAULT_MANIFEST,
            tmp_path / "w.yaml",
            master_node=NODE_A,
            worker_node=NODE_A,
        )


def test_derived_identity_is_deterministic_per_run_and_attempt(tmp_path: Path) -> None:
    first = destr015.derived_identity(tmp_path, 1)
    assert first == destr015.derived_identity(tmp_path, 1)
    assert first != destr015.derived_identity(tmp_path, 2)
    assert first[0].startswith("destr015-") and first[1] == f"{first[0]}-a001"


def test_evidence_components_carry_digests_only() -> None:
    details = {
        "preflight_identity": {"release_id": "rel"},
        "workflow": happy_workflow(),
        "incident": happy_incident(),
        "hosts": {"x": 1},
    }
    components = destr015.evidence_components(details)
    assert set(components) == {"preflight_identity", "workflow", "incident", "hosts"}
    for value in components.values():
        assert len(value) == 64 and int(value, 16) >= 0
    expected = hashlib.sha256(
        json.dumps({"release_id": "rel"}, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert components["preflight_identity"] == expected
    assert len(destr015.case_digest(components)) == 64


def test_parser_accepts_the_documented_arguments() -> None:
    parser = destr015.parser()
    arguments = parser.parse_args(["--run-dir", "/tmp/run"])
    assert arguments.execute is False and arguments.plan is False
    assert arguments.manifest == destr015.DEFAULT_MANIFEST
    arguments = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            "DESTR015_EXECUTE",
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node-a",
            NODE_A,
            "--node-b",
            NODE_B,
            "--pci-bdf-a",
            "0000:0a:00.0",
            "--pci-bdf-b",
            "0000:0b:00.0",
            "--host-probe-image",
            "img",
            "--site-file",
            "/tmp/site.yaml",
        ]
    )
    assert arguments.node_a == NODE_A and arguments.node_b == NODE_B
    assert arguments.pci_bdf_a == "0000:0a:00.0"
    help_text = parser.format_help()
    for option in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert option in help_text


def test_preflight_rejects_nodes_hosting_arbiters_or_all_dns_endpoints() -> None:
    """Live 2026-09-07 attempt 3: both kube-dns replicas sat on node-a/node-b, the
    parallel quiesce dropped both endpoints, and the executor's RESTORE step
    died with gaierror. Spec precondition 2 also bars arbiter replicas."""
    errors = destr015.preflight_errors(
        nodes=NODES,
        arbiter_pods={NODE_A: ["gpu-fault-system/gpu-fault-cluster-executor-x"]},
        dns_nodes=[NODE_A, NODE_B],
        **_preflight_arguments(),
    )

    assert any("arbiter Pods" in error and NODE_A in error for error in errors), errors
    assert any("every kube-dns endpoint" in error for error in errors), errors
    assert (
        destr015.preflight_errors(
            nodes=NODES,
            arbiter_pods={"other-node": ["gpu-fault-system/executor"]},
            dns_nodes=[NODE_A, "other-node"],
            **_preflight_arguments(),
        )
        == []
    )
