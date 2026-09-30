"""GF-REGIONAL-DESTR-025: a single-node reset that fails with a known outcome
escalates to a real reboot *inside its own record* (one workflow, one incident,
no ``workflow-reboot-after-<id>`` successor). Pure verdict contracts against
synthetic snapshots plus the runner's command-line surface."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr025_verdicts as verdicts
from scripts.e2e.regional import run_destr025_single_node_reset_escalation as destr025
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

ROOT = Path(__file__).resolve().parents[2]
NODE = "node-a"
REQUEST = "workflow-destr025-test"
INCIDENT = "inc-destr025-test"
INITIAL = "branch:initial"
SUCCESSOR = f"branch:{NODE}:successor:1"
FLAT_PLAN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)
RUNG_PLAN = (
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)


def _step(operation: str, branch_id: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "execution_owner": "test",
        "node_ids": [NODE],
        "gpu_uuids": [],
        "workload_ids": [],
        "parameters": {},
        "depends_on_step_indexes": [],
        "branch_id": branch_id,
        "branch_node_ids": [NODE] if branch_id != INITIAL else [],
    }


def _execution(
    index: int,
    operation: str,
    status: str,
    *,
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
        "started_at": f"2026-01-01T00:{index:02d}:00+00:00",
        "updated_at": f"2026-01-01T00:{index:02d}:30+00:00",
    }


def pass_workflow() -> dict[str, Any]:
    steps = [_step(op, INITIAL) for op in FLAT_PLAN] + [
        _step(op, SUCCESSOR) for op in RUNG_PLAN
    ]
    executions = [
        _execution(0, "FREEZE_EVIDENCE", "SUCCEEDED"),
        _execution(1, "MARK_UNSCHEDULABLE", "SUCCEEDED"),
        _execution(2, "QUIESCE_GPU_SERVICES", "SUCCEEDED"),
        _execution(3, "VERIFY_NO_GPU_CLIENTS", "SUCCEEDED"),
        _execution(
            4,
            "RESET_GPU",
            "FAILED",
            error="GPU reset refused: clients are still active",
            details={"gpu_client_quiesce_attempt": 1},
        ),
        _execution(8, "RESTART_NODE", "WAITING"),
        _execution(8, "RESTART_NODE", "SUCCEEDED"),
        _execution(9, "VALIDATE_GPU", "SUCCEEDED"),
        _execution(10, "VALIDATE_HOST", "SUCCEEDED"),
        _execution(11, "VALIDATE_FABRIC", "SUCCEEDED"),
        _execution(12, "RESTORE_SCHEDULING", "SUCCEEDED"),
    ]
    return {
        "request_id": REQUEST,
        "incident_id": INCIDENT,
        "status": "SUCCEEDED",
        "fencing_token": 3,
        "dag_enabled": True,
        "dag_revision": 2,
        "official_steps": steps,
        "step_executions": executions,
        "completed_step_indexes": [0, 1, 2, 3, 8, 9, 10, 11, 12],
        "superseded_step_indexes": [4, 5, 6, 7],
        "branch_escalation_counts": {NODE: 1},
        "exhausted_branch_ids": [],
        "terminal_failure_reason": None,
        "preempted_by_workflow_id": None,
        "predecessor_workflow_id": None,
        "events": [
            {
                "kind": "BRANCH_ESCALATION",
                "code": "BRANCH_ESCALATED",
                "step_index": 4,
                "operation": "RESET_GPU",
                "details": {
                    "node_id": NODE,
                    "from_operation": "RESET_GPU",
                    "to_operation": "RESTART_NODE",
                    "rung_count": 1,
                },
            }
        ],
    }


def pass_incident() -> dict[str, Any]:
    return {"incident_id": INCIDENT, "state": "RECOVERED", "fencing_token": 3}


def pass_commands() -> list[dict[str, Any]]:
    return [
        {
            "command_id": f"cmd-{index}",
            "workflow_request_id": REQUEST,
            "incident_id": INCIDENT,
            "step_index": index,
            "fencing_token": 3,
            "status": "SUCCEEDED" if index != 4 else "FAILED",
        }
        for index in (2, 3, 4, 8, 12)
    ]


def _with_execution(workflow: dict[str, Any], **changes: Any) -> dict[str, Any]:
    value = copy.deepcopy(workflow)
    value.update(changes)
    return value


# ---------------------------------------------------------------- workflow


def test_in_record_reboot_without_successor_is_the_pass_shape() -> None:
    workflow = pass_workflow()
    assert verdicts.workflow_errors(workflow, pass_incident(), node=NODE) == [], (
        "the in-record reboot shape must judge clean"
    )
    assert verdicts.successor_errors([], request_id=REQUEST, workflow=workflow) == [], (
        "no successor record means no successor error"
    )
    assert verdicts.command_errors(pass_commands(), workflow) == [], (
        "commands under the record's fencing token must judge clean"
    )


def test_a_reboot_after_successor_fails_the_case() -> None:
    successor = {
        "request_id": f"workflow-reboot-after-{REQUEST}",
        "status": "SUCCEEDED",
        "predecessor_workflow_id": None,
    }
    errors = verdicts.successor_errors(
        [successor], request_id=REQUEST, workflow=pass_workflow()
    )
    assert any("workflow-reboot-after-" in item for item in errors), (
        f"a reboot-after successor must be named in the errors: {errors}"
    )
    linked = {"request_id": "workflow-x", "predecessor_workflow_id": REQUEST}
    errors = verdicts.successor_errors(
        [linked], request_id=REQUEST, workflow=pass_workflow()
    )
    assert errors, "a record that names ours as predecessor is a successor"
    preempted = _with_execution(pass_workflow(), preempted_by_workflow_id="workflow-y")
    errors = verdicts.successor_errors([], request_id=REQUEST, workflow=preempted)
    assert errors, "a preempted record did not finish its own ladder"


def test_a_replace_rung_or_an_exhausted_branch_fails_the_case() -> None:
    workflow = pass_workflow()
    workflow["events"].append(
        {
            "kind": "BRANCH_ESCALATION",
            "code": "BRANCH_ESCALATED",
            "step_index": 9,
            "operation": "VALIDATE_GPU",
            "details": {
                "from_operation": "VALIDATE_GPU",
                "to_operation": "REPLACE_NODE",
                "rung_count": 2,
            },
        }
    )
    workflow["branch_escalation_counts"] = {NODE: 2}
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    assert any("REPLACE_NODE" in item for item in errors), (
        f"a second rung to REPLACE_NODE must fail: {errors}"
    )
    exhausted = _with_execution(
        pass_workflow(), exhausted_branch_ids=[SUCCESSOR], status="FAILED"
    )
    errors = verdicts.workflow_errors(exhausted, pass_incident(), node=NODE)
    assert any("exhausted" in item for item in errors), (
        f"an exhausted branch must fail: {errors}"
    )


def test_an_unknown_reset_outcome_is_not_this_case() -> None:
    workflow = pass_workflow()
    reset = next(
        item for item in workflow["step_executions"] if item["operation"] == "RESET_GPU"
    )
    reset["details"] = {
        "manual_confirmation_required": True,
        "node_action_interrupted": True,
    }
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    assert any("known" in item for item in errors), (
        f"an unknown-outcome reset failure must be refused: {errors}"
    )
    assert verdicts.known_failure(reset) is False, (
        "unresolved receipts are not a known failure"
    )
    assert verdicts.known_failure(pass_workflow()["step_executions"][4]) is True, (
        "the client-barrier refusal is a known failure"
    )


def test_a_validation_that_ran_before_the_reboot_fails_the_case() -> None:
    workflow = pass_workflow()
    executions = [
        item for item in workflow["step_executions"] if item["step_index"] != 9
    ]
    executions.append(_execution(6, "VALIDATE_GPU", "SUCCEEDED"))
    workflow["step_executions"] = executions
    workflow["completed_step_indexes"] = [0, 1, 2, 3, 6, 8, 10, 11, 12]
    workflow["superseded_step_indexes"] = [4, 5, 7, 9]
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    assert any("VALIDATE_GPU" in item and "after" in item for item in errors), (
        f"a validation not placed behind the reboot must fail: {errors}"
    )


def test_the_record_must_end_succeeded_recovered_and_grown_into_a_dag() -> None:
    failed = _with_execution(pass_workflow(), status="FAILED", dag_enabled=False)
    errors = verdicts.workflow_errors(
        failed, {**pass_incident(), "state": "QUARANTINED"}, node=NODE
    )
    assert any("SUCCEEDED" in item for item in errors), (
        f"status must be judged: {errors}"
    )
    assert any("dag_enabled" in item for item in errors), (
        f"the grown DAG must be judged: {errors}"
    )
    assert any("RECOVERED" in item for item in errors), (
        f"the incident state must be judged: {errors}"
    )
    counted = _with_execution(pass_workflow(), branch_escalation_counts={NODE: 0})
    errors = verdicts.workflow_errors(counted, pass_incident(), node=NODE)
    assert any("branch_escalation_counts" in item for item in errors), (
        f"the rung count must be exactly one: {errors}"
    )
    unretired = _with_execution(pass_workflow(), superseded_step_indexes=[4])
    errors = verdicts.workflow_errors(unretired, pass_incident(), node=NODE)
    assert any("retired" in item for item in errors), (
        f"the flat tail behind the failed reset must be retired: {errors}"
    )


def test_a_reset_that_did_not_fail_on_active_clients_is_not_this_case() -> None:
    workflow = pass_workflow()
    reset = workflow["step_executions"][4]
    reset["error"] = "nvidia-smi exited 1"
    reset["details"] = {}
    errors = verdicts.workflow_errors(workflow, pass_incident(), node=NODE)
    assert any("clients are still active" in item for item in errors), (
        f"the armed failure mode must be the one judged: {errors}"
    )


def test_commands_under_another_fencing_token_fail() -> None:
    commands = pass_commands()
    commands[0]["fencing_token"] = 2
    errors = verdicts.command_errors(commands, pass_workflow())
    assert errors, "a command minted under another fencing token must fail"
    assert verdicts.command_errors([], pass_workflow()), (
        "a record with no remote commands never ran on the node"
    )


# -------------------------------------------------------------------- hosts


def _host(boot_id: str, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "boot_id": boot_id,
        "ledger": ledger,
        "gpu_inventory": [{"pci_bdf": "0000:59:00.0", "index": 0, "uuid": "GPU-a"}],
        "compute_clients": [],
        "quiesce_states": [],
        "services": {"nvidia-fabricmanager.service": {"ActiveState": "active"}},
    }


def _holder() -> dict[str, Any]:
    return {
        "matched_row": {"command_id": "cmd-3"},
        "hold_started_at": "2026-01-01T00:03:10+00:00",
        "holder_error": None,
        "unit_state": {"ActiveState": "inactive"},
    }


def test_host_evidence_pass_shape_rebooted_and_never_reset() -> None:
    baseline = _host("boot-1", [])
    after = _host(
        "boot-2",
        [
            {
                "command_id": "cmd-3",
                "operation": "VERIFY_NO_GPU_CLIENTS",
                "state": "SUCCEEDED",
            },
            {
                "command_id": "cmd-2",
                "operation": "QUIESCE_GPU_SERVICES",
                "state": "SUCCEEDED",
            },
        ],
    )
    assert (
        verdicts.host_errors(
            baseline, after, holder_status=_holder(), expected_gpu_count=1
        )
        == []
    ), "a rebooted node with no reset success is the pass shape"


def test_host_evidence_refuses_an_unchanged_boot_or_a_successful_reset() -> None:
    baseline = _host("boot-1", [])
    same_boot = _host(
        "boot-1",
        [
            {
                "command_id": "cmd-3",
                "operation": "VERIFY_NO_GPU_CLIENTS",
                "state": "SUCCEEDED",
            }
        ],
    )
    errors = verdicts.host_errors(
        baseline, same_boot, holder_status=_holder(), expected_gpu_count=1
    )
    assert any("boot id" in item for item in errors), (
        f"an unchanged boot must fail: {errors}"
    )
    reset_ok = _host(
        "boot-2",
        [
            {
                "command_id": "cmd-3",
                "operation": "VERIFY_NO_GPU_CLIENTS",
                "state": "SUCCEEDED",
            },
            {"command_id": "cmd-4", "operation": "RESET_GPU", "state": "SUCCEEDED"},
        ],
    )
    errors = verdicts.host_errors(
        baseline, reset_ok, holder_status=_holder(), expected_gpu_count=1
    )
    assert any("RESET_GPU" in item for item in errors), (
        f"a reset that succeeded means the holder never broke it: {errors}"
    )
    gone = {**_holder(), "matched_row": None, "hold_started_at": None}
    errors = verdicts.host_errors(
        baseline,
        _host(
            "boot-2",
            [
                {
                    "command_id": "cmd-3",
                    "operation": "VERIFY_NO_GPU_CLIENTS",
                    "state": "SUCCEEDED",
                }
            ],
        ),
        holder_status=gone,
        expected_gpu_count=1,
    )
    assert any("holder" in item for item in errors), (
        f"an unarmed holder must fail: {errors}"
    )


# ----------------------------------------------------------------- identity


def test_plan_identity_is_order_independent_and_evidence_is_digests_only() -> None:
    preflight = {
        "release_id": "rel-1",
        "node": {"uid": "node-uid", "boot_id": "boot-1"},
        "store": {"profile": {"profile_version": "v7"}},
        "runtime_identity": {"release_state": {"phase": "deployed"}},
    }
    identity = verdicts.plan_identity(preflight, node=NODE)
    shuffled = dict(reversed(list(identity.items())))
    assert verdicts.identity_digest(identity) == verdicts.identity_digest(shuffled), (
        "the identity digest must not depend on key order"
    )
    components = verdicts.evidence_components(
        {"preflight_identity": {"release_id": "rel-1"}, "workflow": pass_workflow()}
    )
    expected = hashlib.sha256(
        json.dumps({"release_id": "rel-1"}, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert components["preflight_identity"] == expected, "components are sha256 digests"
    assert len(verdicts.case_digest(components)) == 64, "the case digest is one sha256"


def test_the_estimate_fits_only_a_long_enough_node_lifetime() -> None:
    estimate = verdicts.estimated_duration_seconds(
        verify_max_attempts=6, poll_interval_seconds=5.0
    )
    assert estimate > verdicts.REBOOT_ALLOWANCE_SECONDS, (
        "the estimate covers more than the reboot alone"
    )
    assert (
        verdicts.lifetime_errors(
            estimated_seconds=estimate, lifetime_seconds=estimate + 1
        )
        == []
    ), "a lifetime above the estimate fits"
    assert verdicts.lifetime_errors(
        estimated_seconds=estimate, lifetime_seconds=estimate
    ), "a lifetime equal to the estimate does not fit"


# ---------------------------------------------------------------- preflight


def _clean_preflight_arguments() -> dict[str, Any]:
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


def test_a_clean_idle_node_passes_preflight() -> None:
    assert verdicts.preflight_errors(**_clean_preflight_arguments()) == [], (
        "a Ready, idle, unowned node with an ACTIVE agent passes"
    )


def test_preflight_refuses_owned_nodes_open_incidents_and_missing_predecessor() -> None:
    arguments = _clean_preflight_arguments()
    arguments["node_snapshot"]["taints"] = [{"key": "gpu-fault.io/quarantined"}]
    arguments["open_incidents"] = [
        {"incident_id": "inc-old", "node_ids": [NODE], "state": "QUARANTINED"}
    ]
    arguments["predecessor"] = {"valid": False}
    arguments["executor_env"] = [
        {"pod": "executor-a", "allow_reboot": "false", "allow_replace": "false"}
    ]
    arguments["budget"] = {
        "readable": True,
        "scopes": {"node:x": {"active": 1, "limit": 1}},
    }
    errors = verdicts.preflight_errors(**arguments)
    joined = "\n".join(errors)
    for needle in ("schedulable", "GF-REGIONAL-DESTR-002", "allow", "budget"):
        assert needle in joined, f"{needle!r} missing from preflight errors: {errors}"
    assert len(errors) >= 5, f"every refusal must be listed: {errors}"


# ------------------------------------------------------------------- runner


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr025.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "plan is the default mode"
    assert plan.verify_max_attempts == 6, "the executor verify attempts default to 6"
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr025.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
            "--pci-bdf",
            "0000:59:00",
            "--holder-device",
            "/dev/nvidia0",
        ]
    )
    assert execute.execute is True, "execute mode parses"
    assert execute.confirm == destr025.CONFIRMATION, "the confirmation is carried"
    assert execute.node == NODE, "the single target node binds to --node"
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser() builds argparse"


def test_the_help_text_offers_the_documented_live_flags() -> None:
    help_text = destr025.parser().format_help()
    for flag in (
        "--plan",
        "--execute",
        "--confirm",
        "--maintenance-window-end",
        "--site-profile",
        "--node",
        "--pci-bdf",
        "--holder-device",
        "--verify-max-attempts",
        "--predecessor-evidence",
        "--executor-role-arn",
        "--hyperpod-cluster",
        "--host-probe-image",
    ):
        assert flag in help_text, f"{flag} is missing from the runner help"


def test_the_case_constants_follow_the_contract() -> None:
    assert destr025.CASE_ID == "GF-REGIONAL-DESTR-025", "the case id is fixed"
    assert destr025.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-002", (
        "the reboot happy path is the predecessor"
    )
    metadata = RegionalCaseMetadata(
        case_id=destr025.CASE_ID,
        title="",
        category="DESTR",
        level="",
        risk="",
        automation="",
        procedure="",
        predecessor=destr025.PREDECESSOR_CASE_ID,
    )
    assert destr025.CONFIRMATION == metadata.confirmation == "DESTR025_EXECUTE", (
        "the confirmation token derives from the case id"
    )
    assert destr025.CASE.case_id == destr025.CASE_ID, "the CaseRunner names the case"
    path = ROOT / "scripts/e2e/regional/run_destr025_single_node_reset_escalation.py"
    assert path.stat().st_mode & 0o777 == 0o775, "the runner is executable"
    assert (
        path.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env python3"
    ), "the runner carries the python3 shebang"


def test_plan_details_name_the_mutation_and_the_stop_conditions() -> None:
    settings = destr025.Settings(
        regional=None,  # type: ignore[arg-type]
        node=NODE,
        host_probe_image="image",
        hyperpod_cluster="cluster",
        executor_role_arn="arn:aws:iam::1:role/x",
        pci_bdf="",
        device="",
        verify_max_attempts=6,
        predecessor_path=Path("/tmp/none.json"),
    )
    details = destr025.plan_details(settings, {"release_id": "rel-1", "node": {}})
    assert details["risk"] == "destructive-provider-reboot", "a real reboot is planned"
    assert details["node"] == NODE, "the plan names the node"
    assert any(
        "workflow-reboot-after" in item for item in details["stop_conditions"]
    ), "a successor record is a stop condition"
    assert details["preflight_identity_digest"] == verdicts.identity_digest(
        details["preflight_identity"]
    ), "the plan pins the identity digest it will re-check"


def test_env_window_close_action_skips_a_refused_open(tmp_path) -> None:
    from scripts.e2e.regional.run_destr025_single_node_reset_escalation import (
        env_window_close_action,
    )

    record = tmp_path / "executor-env-window.json"
    assert env_window_close_action(False, record) == "none", (
        "a window this attempt never touched owes no close"
    )
    assert env_window_close_action(True, record) == "never-recorded", (
        "an open refused before writing its record has nothing to restore"
    )
    record.write_text("{}")
    assert env_window_close_action(True, record) == "close", (
        "a recorded open is closed through its own record"
    )


def _identity(generation: int, template: str) -> dict:
    return {
        "release_state": {"release_id": "r1"},
        "deployments": {
            "gpu": {
                "gpu-fault-cluster-executor": {
                    "generation": generation,
                    "observed_generation": generation,
                    "template_sha256": template,
                    "images": ["img@sha256:aa"],
                },
                "gpu-fault-node-agent-installer": {
                    "generation": 1,
                    "observed_generation": 1,
                    "template_sha256": "inst",
                    "images": ["img@sha256:bb"],
                },
            }
        },
    }


def test_window_identity_tolerates_only_the_executor_window_rollouts() -> None:
    from scripts.e2e.regional.destr025_verdicts import window_identity_errors

    before = _identity(4, "t-base")
    # While the window is open the executor template differs and its
    # generation moved by exactly one write.
    assert (
        window_identity_errors(
            before,
            _identity(5, "t-open"),
            deployment="gpu-fault-cluster-executor",
            generation_delta=1,
            allow_template_change=True,
        )
        == []
    ), "the open window's own rollout is not drift"
    # After the close the template must be back and the generation moved by two.
    assert (
        window_identity_errors(
            before,
            _identity(6, "t-base"),
            deployment="gpu-fault-cluster-executor",
            generation_delta=2,
        )
        == []
    ), "a closed window restores the template and leaves two generation writes"
    errors = window_identity_errors(
        before,
        _identity(6, "t-open"),
        deployment="gpu-fault-cluster-executor",
        generation_delta=2,
    )
    assert any("other than its generation" in e for e in errors), (
        "a template that did not return to baseline is drift"
    )
    errors = window_identity_errors(
        before,
        _identity(7, "t-base"),
        deployment="gpu-fault-cluster-executor",
        generation_delta=2,
    )
    assert any("generation" in e for e in errors), "an extra rollout is drift"
    other = _identity(6, "t-base")
    other["deployments"]["gpu"]["gpu-fault-node-agent-installer"]["generation"] = 2
    errors = window_identity_errors(
        before, other, deployment="gpu-fault-cluster-executor", generation_delta=2
    )
    assert errors == [
        "gpu deployment gpu-fault-node-agent-installer identity drifted"
    ], "any other Deployment must be identical"
    other = _identity(6, "t-base")
    other["release_state"] = {"release_id": "r2"}
    assert "regional release state identity drifted" in window_identity_errors(
        before, other, deployment="gpu-fault-cluster-executor", generation_delta=2
    ), "release state must not move"
