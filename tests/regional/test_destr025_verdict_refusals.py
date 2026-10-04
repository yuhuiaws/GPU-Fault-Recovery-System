"""GF-REGIONAL-DESTR-025 verdict refusals the pass-shape tests never reach.

Every function here feeds a synthetic snapshot one defect away from the pass
shape into the pure verdict functions of ``destr025_verdicts`` and pins the
exact refusal the live runner would record. Nothing touches a cluster.
"""

from __future__ import annotations

import copy
from typing import Any

from scripts.e2e.regional import destr025_verdicts as verdicts
from tests.regional.test_destr025_verdicts import (
    NODE,
    REQUEST,
    pass_commands,
    pass_incident,
    pass_workflow,
)


def _reset_execution(workflow: dict[str, Any]) -> dict[str, Any]:
    return next(
        item
        for item in workflow["step_executions"]
        if item["operation"] == "RESET_GPU" and item["status"] == "FAILED"
    )


def _restart_executions(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in workflow["step_executions"]
        if item["operation"] == "RESTART_NODE"
    ]


def _joined(errors: list[str]) -> str:
    return "\n".join(errors)


# ---------------------------------------------------------------- workflow


def test_a_failure_reason_and_a_wrong_rung_target_are_named() -> None:
    workflow = pass_workflow()
    workflow["terminal_failure_reason"] = "step timeout"
    workflow["events"][0]["details"]["to_operation"] = "VALIDATE_GPU"
    workflow["events"].append(
        {"kind": "BRANCH_ESCALATION", "code": "BRANCH_EXHAUSTED", "details": {}}
    )
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    joined = _joined(errors)
    assert "failure reason: step timeout" in joined, errors
    assert "did not escalate to RESTART_NODE: VALIDATE_GPU" in joined, errors
    assert "REPLACE_NODE" not in joined, "a non-replace rung is not a replace rung"
    assert "exhausted-branch event" in joined, errors


def test_a_workflow_without_a_failed_reset_skips_the_ladder_judgement() -> None:
    workflow = pass_workflow()
    _reset_execution(workflow)["status"] = "SUCCEEDED"
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    assert errors == ["the record has no single FAILED RESET_GPU execution"], errors


def test_the_reboot_rung_must_succeed_once_behind_the_reset() -> None:
    missing = pass_workflow()
    missing["step_executions"] = [
        item
        for item in missing["step_executions"]
        if item["operation"] != "RESTART_NODE"
    ]
    errors = verdicts.workflow_errors(missing, pass_incident(), node=NODE)
    assert "exactly one SUCCEEDED RESTART_NODE: 0" in _joined(errors), errors

    ahead = pass_workflow()
    for item in _restart_executions(ahead):
        item["step_index"] = 2
    ahead["step_executions"].append(
        {
            "step_index": 8,
            "operation": "RESTART_NODE",
            "status": "FAILED",
            "error": "provider refused",
            "details": {},
        }
    )
    errors = verdicts.workflow_errors(ahead, pass_incident(), node=NODE)
    joined = _joined(errors)
    assert "RESTART_NODE at step 2 is not behind the failed RESET_GPU at step 4" in (
        joined
    ), errors
    assert "RESTART_NODE has a FAILED execution" in joined, errors


def test_a_reset_outside_the_plan_is_not_a_planned_step() -> None:
    workflow = pass_workflow()
    _reset_execution(workflow)["step_index"] = 40
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    assert "the failed reset index 40 is not a planned step" in errors, errors


def test_unrelated_records_are_not_successors() -> None:
    unrelated = [
        {"request_id": "workflow-other", "predecessor_workflow_id": "workflow-z"},
        {"request_id": "", "predecessor_workflow_id": None},
    ]
    assert (
        verdicts.successor_errors(
            unrelated, request_id=REQUEST, workflow=pass_workflow()
        )
        == []
    ), "records that neither carry our suffix nor name us are not successors"


def test_a_command_of_another_workflow_is_named() -> None:
    commands = pass_commands()
    commands[1]["workflow_request_id"] = "workflow-foreign"
    errors = verdicts.command_errors(commands, pass_workflow())
    assert errors == ["command cmd-3 belongs to another workflow"], errors


# -------------------------------------------------------------------- hosts


def _host(boot_id: str | None, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "ledger": ledger,
        "gpu_inventory": [{"pci_bdf": "0000:59:00.0", "index": 0}],
        "compute_clients": [],
        "quiesce_states": [],
        "services": {
            "nvidia-fabricmanager.service": {"ActiveState": "active"},
            "nvidia-persistenced.service": {"ActiveState": "inactive"},
        },
    }
    if boot_id is not None:
        value["boot_id"] = boot_id
    return value


def _holder() -> dict[str, Any]:
    return {
        "matched_row": {"command_id": "cmd-3"},
        "hold_started_at": "2026-01-01T00:03:10+00:00",
        "holder_error": None,
        "unit_state": {"ActiveState": "inactive"},
    }


def test_host_evidence_names_every_residual_of_a_bad_boot() -> None:
    after = _host("boot-2", [{"command_id": "c", "operation": "QUIESCE_GPU_SERVICES"}])
    after["gpu_inventory"] = []
    after["quiesce_states"] = ["0000:59:00.0"]
    after["compute_clients"] = [{"pid": 4242}]
    after["services"]["nvidia-fabricmanager.service"] = {"ActiveState": "failed"}
    errors = verdicts.host_errors(
        _host(None, []), after, holder_status=_holder(), expected_gpu_count=1
    )
    joined = _joined(errors)
    assert "a host boot id snapshot is missing" in joined, errors
    assert "no VERIFY_NO_GPU_CLIENTS row" in joined, errors
    assert "GPU inventory is not 1 after the reboot: 0" in joined, errors
    assert "quiesce state survives" in joined, errors
    assert "still has NVIDIA compute clients" in joined, errors
    assert (
        "did not return active after the reboot: nvidia-fabricmanager.service"
    ) in joined, errors
    assert "nvidia-persistenced.service" not in joined, (
        "a unit that was not active before the case is not judged after it"
    )


# ---------------------------------------------------------------- preflight


def _clean_arguments() -> dict[str, Any]:
    return {
        "node": NODE,
        "node_snapshot": {
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "gpu_allocatable": "8",
        },
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(verdicts.AGENT_OPERATIONS),
        },
        "profile": {
            "capabilities": [
                {
                    "capability": "gpuReset",
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                },
                {
                    "capability": "nodeReboot",
                    "mode": "OWN",
                    "owner": "gpu-fault-hyperpod-adapter",
                    "adapter": "regional-cluster-executor",
                },
            ],
            "warnings": [],
        },
        "queue": {"depth": 0, "fault_backlog_depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "gpu_workloads": [],
        "business_workloads": [],
        "event": None,
        "open_incidents": [],
        "predecessor": {"valid": True},
        "tests": {"passed": True},
        "control_env": {
            "node_lifetime_seconds": 3600,
            "max_rungs": 2,
            "poll_interval_seconds": 5.0,
        },
        "executor_env": [
            {"pod": "executor-a", "allow_reboot": "true", "allow_replace": "false"}
        ],
        "budget": {"readable": True, "scopes": {"node:x": {"active": 0, "limit": 1}}},
        "identity_errors": [],
        "verify_max_attempts": 6,
    }


def test_verify_attempts_are_bounded() -> None:
    assert verdicts.verify_attempt_errors(0) == [
        "verify attempts 0 is outside 1..20"
    ], "zero attempts never verifies"
    assert verdicts.verify_attempt_errors(21) == [
        "verify attempts 21 is outside 1..20"
    ], "the ceiling keeps the refused reset within the lifetime"
    assert verdicts.verify_attempt_errors(1) == [], "the floor is inclusive"


def test_preflight_names_a_node_and_an_agent_that_cannot_take_the_case() -> None:
    arguments = _clean_arguments()
    arguments["node_snapshot"]["gpu_allocatable"] = "0"
    arguments["agent"] = {"lifecycle_state": "DRAINING", "allowed_operations": []}
    errors = verdicts.preflight_errors(**arguments)
    assert errors == [
        f"{NODE} allocates no GPU",
        f"{NODE} Node Agent is not ACTIVE",
        f"{NODE} Node Agent does not allow {sorted(verdicts.AGENT_OPERATIONS)}",
    ], errors


def test_preflight_refuses_a_profile_without_the_owned_capabilities() -> None:
    arguments = _clean_arguments()
    arguments["profile"] = None
    errors = verdicts.preflight_errors(**arguments)
    assert errors == [
        "the runtime profile has no gpuReset capability",
        "the runtime profile has no nodeReboot capability",
    ], errors

    arguments = _clean_arguments()
    reset, reboot = arguments["profile"]["capabilities"]
    reset["mode"] = "SHARED"
    reboot["adapter"] = "something-else"
    arguments["profile"]["warnings"] = ["capability drift"]
    errors = verdicts.preflight_errors(**arguments)
    assert errors == [
        "gpuReset is not OWN by the Node Agent",
        "nodeReboot is not OWN by the HyperPod adapter",
        "the runtime profile has warnings",
    ], errors


def test_preflight_refuses_a_busy_cluster_and_a_missing_executor() -> None:
    arguments = _clean_arguments()
    arguments["tests"] = {"passed": False}
    arguments["business_workloads"] = [{"name": "trainer"}]
    arguments["gpu_workloads"] = [{"name": "other-gpu-job"}]
    arguments["event"] = {"xid": 46}
    arguments["queue"] = {"depth": 3, "fault_backlog_depth": 3}
    arguments["remote_commands"] = {"open_by_cluster": {"cluster-a": 1}}
    arguments["executor_env"] = []
    arguments["control_env"]["max_rungs"] = 0
    errors = verdicts.preflight_errors(**arguments)
    assert errors == [
        "focused regression tests failed",
        f"{NODE} carries a non-system workload: [{{'name': 'trainer'}}]",
        "the GPU cluster already has a GPU workload: [{'name': 'other-gpu-job'}]",
        f"{NODE} has a recent XID event",
        "the processor queue is not empty",
        "the remote command queue is not empty",
        "no ready executor replica reported its environment",
        "GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS is below 1",
    ], errors


# ----------------------------------------------------------------- identity


def _identity() -> dict[str, Any]:
    return {
        "release_state": {"release_id": "r1"},
        "deployments": {
            "gpu": {
                "gpu-fault-cluster-executor": {
                    "generation": 4,
                    "observed_generation": 4,
                    "template_sha256": "t-base",
                },
                "gpu-fault-node-agent-installer": {
                    "generation": 1,
                    "observed_generation": 1,
                    "template_sha256": "inst",
                },
            }
        },
    }


def test_a_deployment_that_disappeared_is_drift() -> None:
    after = copy.deepcopy(_identity())
    del after["deployments"]["gpu"]["gpu-fault-node-agent-installer"]
    errors = verdicts.window_identity_errors(
        _identity(), after, deployment="gpu-fault-cluster-executor", generation_delta=0
    )
    assert errors == ["gpu deployment gpu-fault-node-agent-installer disappeared"], (
        errors
    )
