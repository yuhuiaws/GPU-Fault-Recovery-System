"""GF-REGIONAL-DESTR-026: two GPUs of ONE node fault inside the aggregation
window, one quiesce window resets both, the training job restarts once.

Pure verdict contracts against synthetic store, ledger and host snapshots;
cross-checks that the verdicts accept the plan the control plane really
compiles and the result the Node Agent really records (through the host
probe's ledger read); and the runner's command-line surface."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from scripts.e2e.regional import destr026_verdicts as verdicts
from scripts.e2e.regional import run_destr026_same_node_two_gpu_reset as destr026
from scripts.e2e.regional.destr015_verdicts import AGENT_OPERATIONS
from scripts.e2e.regional.probes import destructive_node_probe as probe
from scripts.e2e.regional.regional_case_contract import (
    RegionalCaseMetadata,
    formal_predecessor,
)
from tests.node_agent.test_reset_two_gpus_one_window import reset_both, two_gpu_agent
from tests.orchestration.test_same_node_job_reset_merge import (
    GPU_A,
    GPU_B,
    LINEAR_PLAN,
    NODE,
    PARTNER,
    job_context,
    job_fault,
    start_steps,
    through,
)

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [GPU_A, GPU_B]
OTHERS = [f"GPU-a{index}" for index in range(2, 8)]
REQUEST = "workflow-destr026"
INCIDENT = "inc-destr026"
SOURCE_ATTEMPT = "destr026-job-a001"
T0 = datetime.now(timezone.utc).replace(microsecond=0)


def at(seconds: int) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


# --------------------------------------------------------------------------- #
# Synthetic pass shape
# --------------------------------------------------------------------------- #
def reset_receipt() -> dict[str, Any]:
    """The live p5en shape: GPU a busy until the display stack is released."""

    reset = verdicts.RESET_COMMAND
    return {
        "reset_gpu_uuids": list(TARGETS),
        "reset_attempts": 3,
        "reset_successes": 2,
        "reset_busy_refusals": 1,
        "physical_calls": [
            {
                "command": reset + GPU_A,
                "attempt": 1,
                "outcome": "busy",
                "target_persistence": [
                    {"gpu_uuid": GPU_A, "persistence_mode": "Disabled"}
                ],
            },
            {"command": "rmmod nvidia_drm", "outcome": "released"},
            {"command": "rmmod nvidia_modeset", "outcome": "released"},
            {"command": reset + GPU_A, "attempt": 2, "outcome": "reset"},
            {"command": reset + GPU_B, "attempt": 1, "outcome": "reset"},
        ],
        "released_kernel_modules": ["nvidia_drm", "nvidia_modeset"],
        "reloaded_kernel_modules": ["nvidia_modeset", "nvidia_drm"],
    }


KEPT = [
    {"service": "nvidia-fabricmanager", "reason": "per-GPU reset window"},
    {"service": "nvidia-persistenced", "reason": "released through the daemon"},
]
STOPPED = ["nvidia-dcgm", "nvidia-persistenced", "gpu-fault-metrics-collector"]


def step_details(operation: str) -> dict[str, Any]:
    if operation == "QUIESCE_GPU_SERVICES":
        receipt = {"active_services": STOPPED, "kept_services": KEPT}
        return {
            "node_results": {NODE: receipt},
            "stopped_services": {NODE: STOPPED},
            "kept_services": {NODE: KEPT},
        }
    if operation == "RESET_GPU":
        return {"node_results": {NODE: reset_receipt()}}
    if operation == "RESTORE_GPU_SERVICES":
        receipt = {"restored_services": STOPPED, "kept_services": KEPT}
        return {"node_results": {NODE: receipt}, "restored_services": {NODE: STOPPED}}
    if operation == "RESTART_WORKLOAD":
        return {"source_gpu_count": 16, "target_gpu_count": 16}
    return {}


def plan_step(operation: str) -> dict[str, Any]:
    scoped = operation in verdicts.GPU_SCOPED_STEPS
    mapped = operation in verdicts.MAPPED_NODE_ACTIONS
    return {
        "operation": operation,
        "node_ids": [NODE, PARTNER] if operation in verdicts.JOB_STEPS else [NODE],
        "gpu_uuids": list(TARGETS) if scoped else [GPU_A],
        "parameters": {"gpu_uuids_by_node": {NODE: list(TARGETS)}} if mapped else {},
        "depends_on_step_indexes": [],
    }


def complete(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """A terminal SUCCEEDED record: every step executed once, in order."""

    executions = [
        {
            "step_index": index,
            "operation": step["operation"],
            "status": "SUCCEEDED",
            "phase": "official",
            "adapter_operation_id": f"remote/{index}",
            "details": step_details(step["operation"]),
            "started_at": at(60 * index),
            "updated_at": at(60 * index + 30),
        }
        for index, step in enumerate(steps)
    ]
    return {
        "request_id": REQUEST,
        "incident_id": INCIDENT,
        "status": "SUCCEEDED",
        "dag_enabled": False,
        "official_steps": steps,
        "step_executions": executions,
        "completed_step_indexes": list(range(len(steps))),
        "completed_operations": [step["operation"] for step in steps],
        "superseded_step_indexes": [],
        "branch_escalation_counts": {},
        "exhausted_branch_ids": [],
        "workload_withdrawn_at": None,
        "terminal_failure_reason": None,
    }


def pass_workflow() -> dict[str, Any]:
    return complete([plan_step(operation) for operation in LINEAR_PLAN])


def pass_incident() -> dict[str, Any]:
    return {
        "incident_id": INCIDENT,
        "state": "RECOVERED",
        "gpu_uuids": list(TARGETS),
        "node_ids": [NODE],
        "workflow_request_id": REQUEST,
    }


def workflow_errors(workflow: dict[str, Any], **overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "node": NODE,
        "partner": PARTNER,
        "targets": TARGETS,
        "expected_gpu_count": 16,
        "restart_budget": {"restart_count": 1, "budget": 1},
        **overrides,
    }
    incident = arguments.pop("incident", pass_incident())
    return verdicts.workflow_errors(workflow, incident, **arguments)


def event_state(label: str, gpu: str, **workflow: Any) -> dict[str, Any]:
    return {
        "event": {
            "event_id": f"kernel-xid-46-{label}",
            "xid": 46,
            "node_id": NODE,
            "gpu_uuid": gpu,
            "evidence_ref": f"kmsg://{NODE}/boot/{label}",
            "observed_at": at(1),
        },
        "decision": {"official_action": "RESET_GPU", "disposition": "EXECUTABLE"},
        "incident": {"incident_id": INCIDENT, "workflow_request_id": REQUEST},
        "workflow": {"request_id": REQUEST, **workflow},
    }


def ledger_row(operation: str, index: int, **values: Any) -> dict[str, Any]:
    return {
        "command_id": f"{REQUEST}/{index}/{operation}/{NODE}",
        "attempt": 1,
        "state": "SUCCEEDED",
        "operation": operation,
        "started_at": at(60 * index),
        "completed_at": at(60 * index + 30),
        "incident_id": INCIDENT,
        "workflow_request_id": REQUEST,
        "gpu_uuids": list(TARGETS),
        **values,
    }


def inventory() -> list[dict[str, str]]:
    return [
        {"index": str(index), "uuid": uuid, "pci_bdf": f"0000:{0x59 + index:02x}:00"}
        for index, uuid in enumerate([*TARGETS, *OTHERS])
    ]


SERVICES = {
    "nvidia-fabricmanager.service": {
        "ActiveState": "active",
        "MainPID": "4242",
        "InvocationID": "fabric-1",
    },
    "kubelet.service": {
        "ActiveState": "active",
        "MainPID": "77",
        "InvocationID": "k-1",
    },
}


def hosts() -> tuple[dict[str, Any], dict[str, Any]]:
    """The fault node's baseline and after snapshots."""

    old = {
        **ledger_row("RESET_GPU", 1),
        "command_id": "older/1/RESET_GPU",
        "workflow_request_id": "older",
    }
    before = {
        "boot_id": "boot-a",
        "gpu_inventory": inventory(),
        "ledger": [old],
        "quiesce_states": [],
        "gpu_fault_timers": [],
        "services": copy.deepcopy(SERVICES),
    }
    receipt = {"details": reset_receipt()}
    every = sorted([*TARGETS, *OTHERS])
    after = {
        **copy.deepcopy(before),
        "ledger": [
            old,
            ledger_row(
                "QUIESCE_GPU_SERVICES",
                3,
                evidence=probe.ledger_evidence(
                    "QUIESCE_GPU_SERVICES",
                    {"details": {"active_services": STOPPED, "kept_services": KEPT}},
                ),
            ),
            ledger_row("VERIFY_NO_GPU_CLIENTS", 4),
            ledger_row(
                "RESET_GPU", 5, evidence=probe.ledger_evidence("RESET_GPU", receipt)
            ),
            ledger_row("RESTORE_GPU_SERVICES", 6),
        ],
        "sampler": {
            "sample_count": 40,
            "observed_gpu_uuid_sets": [
                every,
                sorted(set(every) - {GPU_A}),
                sorted(set(every) - {GPU_B}),
            ],
            "last": {"gpu_uuids": every, "gpu_count": 8},
        },
    }
    return before, after


def row_of(after: dict[str, Any], operation: str) -> dict[str, Any]:
    (row,) = [
        item
        for item in after["ledger"]
        if item["operation"] == operation and item["workflow_request_id"] == REQUEST
    ]
    return row


def ledger_errors(after: dict[str, Any]) -> list[str]:
    before, _ = hosts()
    errors, _rows = verdicts.ledger_errors(
        before,
        after,
        targets=TARGETS,
        incident_id=INCIDENT,
        workflow_request_id=REQUEST,
    )
    return errors


def partner_snapshot() -> dict[str, Any]:
    return {
        "boot_id": "boot-b",
        "gpu_inventory": [
            {"uuid": f"GPU-b{index}", "pci_bdf": f"0000:{0x59 + index:02x}:00"}
            for index in range(8)
        ],
        "ledger": [ledger_row("RESET_GPU", 9, command_id="older-b")],
        "quiesce_states": [],
        "services": copy.deepcopy(SERVICES),
    }


def gpu_nodes() -> list[dict[str, Any]]:
    return [
        {"name": name, "unschedulable": False, "taints": []}
        for name in (NODE, PARTNER, "node-c")
    ]


def restarted_pods(attempt: str = "destr026-job-a002") -> list[dict[str, Any]]:
    return [
        {
            "uid": f"new-{node}",
            "node": node,
            "phase": "Running",
            "ready": True,
            "attempt_id": attempt,
        }
        for node in (NODE, PARTNER)
    ]


# --------------------------------------------------------------------------- #
# The pass shape
# --------------------------------------------------------------------------- #
def test_the_reset_contract_of_destr001_holds_for_the_ten_step_job_plan() -> None:
    workflow = pass_workflow()
    remote = [
        operation
        for operation in LINEAR_PLAN
        if operation not in {"FREEZE_EVIDENCE", "VALIDATE_GPU"}
    ]
    state = {
        **event_state("a", GPU_A),
        "workflow": workflow,
        "commands": [
            {"status": "SUCCEEDED", "step": {"operation": operation}}
            for operation in remote
        ],
    }

    assert (
        destr026.reset_case.workflow_errors(
            state, expected_steps=verdicts.WORKLOAD_RESET_STEPS
        )
        == []
    ), "the runner applies DESTR-001's chain, delegation and remote-step checks"


def test_the_single_window_two_gpu_reset_is_the_pass_shape() -> None:
    workflow = pass_workflow()
    before, after = hosts()
    errors, rows = verdicts.ledger_errors(
        before,
        after,
        targets=TARGETS,
        incident_id=INCIDENT,
        workflow_request_id=REQUEST,
    )
    control = verdicts.node_receipt(workflow, "RESET_GPU", NODE)
    recorded = rows["RESET_GPU"]["evidence"]

    assert (
        verdicts.injection_errors(
            {"a": event_state("a", GPU_A), "b": event_state("b", GPU_B)},
            node=NODE,
            targets={"a": GPU_A, "b": GPU_B},
        )
        == []
    ), "both faults in one incident and one workflow"
    assert workflow_errors(workflow) == [], "one window, two GPUs, one join"
    assert errors == [], "one quiesce, one reset of both, one restore on the node"
    assert verdicts.reset_receipt_errors(control, targets=TARGETS, source="cp") == []
    assert (
        verdicts.reset_receipt_errors(recorded, targets=TARGETS, source="ledger") == []
    )
    assert verdicts.relay_errors(control, recorded) == [], "relay matches the ledger"
    assert (
        verdicts.fabric_manager_errors(
            before, after, workflow=workflow, ledger_rows=rows, node=NODE
        )
        == []
    ), "Fabric Manager kept and untouched"
    assert verdicts.sampler_errors(inventory(), after["sampler"], targets=TARGETS) == []
    assert (
        verdicts.partner_host_errors(
            partner_snapshot(), partner_snapshot(), node=PARTNER
        )
        == []
    ), "node B untouched"
    assert verdicts.other_node_errors(gpu_nodes(), gpu_nodes(), fault_node=NODE) == []
    assert (
        verdicts.attempt_errors(
            source_attempt_id=SOURCE_ATTEMPT,
            pods=restarted_pods(),
            observations=[
                {"attempt_id": SOURCE_ATTEMPT},
                {"attempt_id": "destr026-job-a002"},
            ],
        )
        == []
    ), "the attempt changed once"


# --------------------------------------------------------------------------- #
# Negatives the brief names
# --------------------------------------------------------------------------- #
def test_two_workflows_or_two_incidents_fail_the_injection() -> None:
    split = event_state("b", GPU_B, request_id="workflow-second")
    split["incident"] = {
        "incident_id": "inc-second",
        "workflow_request_id": "workflow-second",
    }

    errors = verdicts.injection_errors(
        {"a": event_state("a", GPU_A), "b": split},
        node=NODE,
        targets={"a": GPU_A, "b": GPU_B},
    )

    assert any("landed in 2 workflows" in item for item in errors), errors
    assert any("opened 2 incidents" in item for item in errors), errors


def test_an_event_of_the_wrong_gpu_or_action_fails_the_injection() -> None:
    wrong = event_state("b", GPU_A)
    wrong["decision"] = {"official_action": "RESTART_NODE"}

    errors = verdicts.injection_errors(
        {"a": event_state("a", GPU_A), "b": wrong},
        node=NODE,
        targets={"a": GPU_A, "b": GPU_B},
    )

    assert any("resolved to GPU-a0, not GPU-a1" in item for item in errors), errors
    assert any("did not resolve to RESET_GPU" in item for item in errors), errors


def test_a_second_restart_fails_the_case() -> None:
    workflow = pass_workflow()
    restart = copy.deepcopy(workflow["step_executions"][-1])
    workflow["step_executions"].append(restart)

    assert any(
        "RESTART_WORKLOAD does not have exactly one execution: 2" in item
        for item in workflow_errors(workflow)
    ), "two executions of the join are two restarts"
    assert any(
        "restart budget did not advance exactly once" in item
        for item in workflow_errors(
            pass_workflow(), restart_budget={"restart_count": 2, "budget": 1}
        )
    ), "the budget counts the second restart"
    assert verdicts.attempt_errors(
        source_attempt_id=SOURCE_ATTEMPT,
        pods=restarted_pods("destr026-job-a003"),
        observations=[
            {"attempt_id": SOURCE_ATTEMPT},
            {"attempt_id": "destr026-job-a002"},
            {"attempt_id": "destr026-job-a003"},
        ],
    ), "a third attempt id means the job restarted twice"
    mixed = restarted_pods()
    mixed[1]["attempt_id"] = "destr026-job-a003"
    assert verdicts.attempt_errors(
        source_attempt_id=SOURCE_ATTEMPT, pods=mixed, observations=[]
    ), "the Pods of one restart share one attempt id"


def test_a_gpu_that_was_not_reset_fails_every_account_of_it() -> None:
    receipt = reset_receipt()
    receipt["physical_calls"] = receipt["physical_calls"][:-1]
    receipt["reset_successes"] = 1
    workflow = pass_workflow()
    workflow["step_executions"][5]["details"] = {"node_results": {NODE: receipt}}
    _, after = hosts()
    row_of(after, "RESET_GPU")["gpu_uuids"] = [GPU_A]
    after["sampler"]["observed_gpu_uuid_sets"].pop()

    receipt_errors = verdicts.reset_receipt_errors(
        verdicts.node_receipt(workflow, "RESET_GPU", NODE), targets=TARGETS, source="cp"
    )

    assert any(
        f"does not show {GPU_B} reset exactly once" in item for item in receipt_errors
    ), receipt_errors
    assert any("reports 1 successful resets" in item for item in receipt_errors), (
        receipt_errors
    )
    assert any("does not name both GPUs" in item for item in ledger_errors(after)), (
        "the ledger row must name both GPUs"
    )
    assert verdicts.sampler_errors(inventory(), after["sampler"], targets=TARGETS) == [
        f"the sampler never saw {GPU_B} leave and return"
    ]
    narrow = pass_workflow()
    narrow["official_steps"][5]["gpu_uuids"] = [GPU_A]
    assert any(
        "RESET_GPU does not name both faulted GPUs" in item
        for item in workflow_errors(narrow)
    ), "the step itself must name both"


@pytest.mark.parametrize(
    "mapping",
    [None, {NODE: [GPU_A]}, {NODE: list(TARGETS), PARTNER: ["GPU-b0"]}],
    ids=["absent", "one-gpu", "foreign-node"],
)
@pytest.mark.parametrize("operation", verdicts.MAPPED_NODE_ACTIONS)
def test_a_widened_node_action_without_its_mapping_fails(
    operation: str, mapping: dict[str, list[str]] | None
) -> None:
    workflow = pass_workflow()
    step = workflow["official_steps"][LINEAR_PLAN.index(operation)]
    step["parameters"] = {} if mapping is None else {"gpu_uuids_by_node": mapping}

    errors = workflow_errors(workflow)

    assert any(
        f"{operation} carries no gpu_uuids_by_node naming both GPUs" in item
        for item in errors
    ), errors


def test_a_second_window_or_any_node_action_on_node_b_fails_the_plan() -> None:
    doubled = pass_workflow()
    doubled["official_steps"].insert(6, plan_step("RESET_GPU"))
    doubled = complete(doubled["official_steps"])
    assert any(
        "exactly one RESET_GPU: 2" in item for item in workflow_errors(doubled)
    ), "a second reset step is a second window"
    spread = pass_workflow()
    spread["official_steps"][1]["node_ids"] = [NODE, PARTNER]
    assert any(
        "MARK_UNSCHEDULABLE acts on" in item for item in workflow_errors(spread)
    ), "node B is never cordoned"
    narrow = pass_workflow()
    narrow["official_steps"][2]["node_ids"] = [NODE]
    assert any(
        "STOP_WORKLOADS does not cover exactly the job's nodes" in item
        for item in workflow_errors(narrow)
    ), "the job stops whole"
    rebooted = complete([*pass_workflow()["official_steps"], plan_step("RESTART_NODE")])
    assert any("beyond reset" in item for item in workflow_errors(rebooted)), (
        "a reboot step leaves the reset-only case"
    )


def test_interleaved_resets_or_a_second_display_release_fail_the_receipt() -> None:
    reset = verdicts.RESET_COMMAND
    interleaved = reset_receipt()
    interleaved["physical_calls"] = [
        {"command": reset + GPU_A, "outcome": "busy"},
        {"command": reset + GPU_B, "outcome": "reset"},
        {"command": reset + GPU_A, "outcome": "reset"},
    ]
    twice = reset_receipt()
    twice["physical_calls"].insert(
        4, {"command": "rmmod nvidia_drm", "outcome": "released"}
    )
    twice["released_kernel_modules"] = ["nvidia_drm", "nvidia_modeset", "nvidia_drm"]

    assert any(
        "one after another" in item
        for item in verdicts.reset_receipt_errors(
            interleaved, targets=TARGETS, source="cp"
        )
    ), "one window resets them one after another"
    errors = verdicts.reset_receipt_errors(twice, targets=TARGETS, source="cp")
    assert any("released a display module more than once" in item for item in errors), (
        errors
    )
    assert any(
        "released_kernel_modules names a module twice" in item for item in errors
    ), errors
    foreign = reset_receipt()
    foreign["physical_calls"].append({"command": reset + "GPU-a5", "outcome": "reset"})
    assert any(
        "reset a GPU the case did not fault" in item
        for item in verdicts.reset_receipt_errors(foreign, targets=TARGETS, source="cp")
    ), "a reset of a GPU nobody faulted is refused"
    assert verdicts.reset_receipt_errors({}, targets=TARGETS, source="cp") == [
        "cp carries no RESET_GPU result"
    ]


def test_the_relay_must_match_the_node_ledger() -> None:
    recorded = probe.ledger_evidence("RESET_GPU", {"details": reset_receipt()})
    relayed = reset_receipt()
    relayed["physical_calls"][3]["outcome"] = "busy"

    assert verdicts.relay_errors(relayed, recorded) == [
        "the Node Agent ledger and the control plane disagree on the reset's physical calls"
    ]


def test_the_ledger_holds_one_window_of_this_workflow_and_nothing_else() -> None:
    _, after = hosts()
    second = copy.deepcopy(after)
    second["ledger"].append(ledger_row("QUIESCE_GPU_SERVICES", 7, command_id="again"))
    assert any(
        "exactly one successful QUIESCE_GPU_SERVICES" in item
        for item in ledger_errors(second)
    ), "a second quiesce is a second window"
    foreign = copy.deepcopy(after)
    foreign["ledger"].append(
        {**ledger_row("VERIFY_NO_GPU_CLIENTS", 8), "workflow_request_id": "other"}
    )
    assert any("another workflow acted" in item for item in ledger_errors(foreign)), (
        "a row of another workflow is foreign activity on the node"
    )
    outside = copy.deepcopy(after)
    row_of(outside, "RESET_GPU")["started_at"] = at(60 * 7)
    assert any(
        "inside the one quiesce window" in item for item in ledger_errors(outside)
    ), "a reset whose timestamps leave the window is refused"


def test_fabric_manager_must_be_kept_through_the_window() -> None:
    workflow = pass_workflow()
    before, after = hosts()
    _, rows = verdicts.ledger_errors(
        before,
        after,
        targets=TARGETS,
        incident_id=INCIDENT,
        workflow_request_id=REQUEST,
    )
    quiesce = workflow["step_executions"][3]["details"]
    quiesce["stopped_services"][NODE] = [*STOPPED, "nvidia-fabricmanager"]
    quiesce["kept_services"][NODE] = KEPT[1:]
    after["services"]["nvidia-fabricmanager.service"]["InvocationID"] = "fabric-2"

    errors = verdicts.fabric_manager_errors(
        before, after, workflow=workflow, ledger_rows=rows, node=NODE
    )

    assert errors == [
        "QUIESCE_GPU_SERVICES did not keep Fabric Manager running",
        "QUIESCE_GPU_SERVICES stopped Fabric Manager",
        "Fabric Manager was stopped or restarted around the reset",
    ]
    before["services"].pop("nvidia-fabricmanager.service")
    assert (
        verdicts.fabric_manager_errors(
            before, after, workflow=workflow, ledger_rows=rows, node=NODE
        )
        == []
    ), "a node without Fabric Manager has nothing to keep"


def test_node_b_and_the_other_nodes_must_stay_untouched() -> None:
    touched = partner_snapshot()
    touched["boot_id"] = "boot-b2"
    touched["ledger"].append(ledger_row("QUIESCE_GPU_SERVICES", 3))
    touched["quiesce_states"] = [{"phase": "QUIESCED"}]
    touched["services"]["nvidia-fabricmanager.service"]["InvocationID"] = "fabric-2"

    assert verdicts.partner_host_errors(partner_snapshot(), touched, node=PARTNER) == [
        f"{PARTNER} rebooted",
        f"{PARTNER} Node Agent ledger gained rows: ['QUIESCE_GPU_SERVICES']",
        f"{PARTNER} carries a GPU quiesce state",
        f"{PARTNER} Fabric Manager was restarted",
    ]
    after = gpu_nodes()
    after[2]["unschedulable"] = True
    after[0]["taints"] = [{"key": "gpu-fault.io/quarantined"}]
    assert verdicts.other_node_errors(gpu_nodes(), after, fault_node=NODE) == [
        "node-c scheduling state changed: []"
    ], "the fault node is judged by schedulability_errors, every other node here"


# --------------------------------------------------------------------------- #
# Preflight and target selection
# --------------------------------------------------------------------------- #
def preflight(**overrides: Any) -> list[str]:
    snapshot = {
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": "8",
        "boot_id": "boot-a",
    }
    owner = {
        "mode": "OWN",
        "owner": "gpu-fault-kubernetes-adapter",
        "adapter": "regional-cluster-executor",
    }
    arguments: dict[str, Any] = {
        "nodes": (NODE, PARTNER),
        "node_snapshots": {NODE: dict(snapshot), PARTNER: dict(snapshot)},
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(AGENT_OPERATIONS),
        },
        "profile": {
            "capabilities": [
                {
                    "capability": "gpuReset",
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                },
                {"capability": "workloadStop", **owner},
                {"capability": "workloadRestart", **owner},
            ]
        },
        "queue": {"depth": 0},
        "remote_commands": {},
        "gpu_workloads": [],
        "business_workloads": {},
        "recent_event": None,
        "predecessor": {"valid": True},
        "tests": {"passed": True},
        "control_env": {"aggregation_window_seconds": 5, "job_lifetime_seconds": 3600},
        "runtime_identity_errors": [],
        "gpu_identity": {
            "product": "H200",
            "source_boot_id": "boot-a",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
        "arbiter_pods": {PARTNER: ["gpu-fault-system/gpu-fault-cluster-executor-0"]},
        "dns_nodes": [NODE, "cpu-node"],
        **overrides,
    }
    return verdicts.preflight_errors(**arguments)


def test_the_preflight_admits_a_clean_pair_and_names_each_refusal() -> None:
    assert preflight() == [], "arbiters on node B do not matter: it is never quiesced"
    assert "the fault node and the job's second node are identical" in preflight(
        nodes=(NODE, NODE)
    )
    assert any(
        "hosts controller/executor arbiter Pods" in item
        for item in preflight(arbiter_pods={NODE: ["x"]})
    ), "an arbiter on the quiesced node is refused"
    assert any(
        "holds every kube-dns endpoint" in item for item in preflight(dns_nodes=[NODE])
    ), "cluster DNS must survive the quiesce"
    assert any(
        "recent XID event" in item for item in preflight(recent_event={"xid": 46})
    ), "the XID quiet window is enforced"
    assert any(
        "aggregation window 2s" in item
        for item in preflight(
            control_env={"aggregation_window_seconds": 2, "job_lifetime_seconds": 3600}
        )
    ), "a window too narrow for two writes is refused"
    assert "DESTR-015 predecessor evidence is not PASS" in preflight(
        predecessor={"valid": False}
    )
    assert any(
        "does not allow" in item
        for item in preflight(
            agent={"lifecycle_state": "ACTIVE", "allowed_operations": []}
        )
    ), "the Node Agent must allow the window's operations"


def test_target_gpus_default_to_the_first_two_and_must_belong_to_the_job() -> None:
    rows = inventory()
    allocation = {row["uuid"] for row in rows}

    assert verdicts.target_gpus(rows, pci_bdfs=("", ""), allocation=allocation) == {
        "0000:59:00": GPU_A,
        "0000:5a:00": GPU_B,
    }, "the first two inventory entries by default"
    assert verdicts.target_gpus(
        rows, pci_bdfs=("0000:5c:00.0", ""), allocation=allocation
    ) == {"0000:5c:00": "GPU-a3", "0000:59:00": GPU_A}, (
        "an explicit BDF keeps its place"
    )
    with pytest.raises(ValueError, match="not distinct"):
        verdicts.target_gpus(
            rows, pci_bdfs=("0000:59:00", "0000:59:00"), allocation=allocation
        )
    with pytest.raises(ValueError, match="no GPU of the node answers"):
        verdicts.target_gpus(rows, pci_bdfs=("0000:99:00", ""), allocation=allocation)
    with pytest.raises(ValueError, match="does not hold"):
        verdicts.target_gpus(rows, pci_bdfs=("", ""), allocation={GPU_A})


# --------------------------------------------------------------------------- #
# The verdicts against what the product really produces
# --------------------------------------------------------------------------- #
def test_the_verdict_accepts_the_plan_the_control_plane_compiles() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    job_fault(context, "xid-a1", GPU_B)
    plan = context.store.get_workflow(first["workflow_request_id"]).model_dump(
        mode="json"
    )

    workflow = complete(plan["official_steps"])

    assert workflow_errors(workflow) == [], "the compiled plan is the pass shape"


def test_the_verdict_refuses_the_late_fault_plan() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    issued = {**through(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)}
    issued[WorkflowOperation.RESET_GPU] = WorkflowStepStatus.WAITING
    start_steps(
        context, context.store.get_workflow(first["workflow_request_id"]), issued
    )
    job_fault(context, "xid-a1", GPU_B)
    plan = context.store.get_workflow(first["workflow_request_id"]).model_dump(
        mode="json"
    )

    errors = workflow_errors(complete(plan["official_steps"]))

    assert any("exactly one RESET_GPU: 2" in item for item in errors), (
        "a successor branch is a second window, not this case's shape"
    )


def test_the_node_ledger_carries_the_reset_the_verdict_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _runner = two_gpu_agent(tmp_path)
    result = reset_both(agent)
    monkeypatch.setattr(probe, "LEDGER", tmp_path / "two-gpu.db")

    (row,) = [item for item in probe.ledger_rows() if item["operation"] == "RESET_GPU"]
    evidence = row["evidence"]
    gpus = ["GPU-a", "GPU-b"]

    assert verdicts.reset_receipt_errors(evidence, targets=gpus, source="ledger") == []
    assert (
        verdicts.reset_receipt_errors(result.details, targets=gpus, source="cp") == []
    )
    assert verdicts.relay_errors(result.details, evidence) == [], "same calls both ways"
    assert all(
        set(call) <= set(probe.LEDGER_CALL_FIELDS)
        for call in evidence["physical_calls"]
    ), "the ledger read keeps command, attempt and outcome only"


def test_the_probe_bounds_the_ledger_evidence_it_reads() -> None:
    quiesce = probe.ledger_evidence(
        "QUIESCE_GPU_SERVICES",
        {
            "details": {
                "active_services": STOPPED,
                "kept_services": KEPT,
                "timer_unit": "t",
            }
        },
    )

    assert quiesce == {
        "active_services": STOPPED,
        "kept_services": ["nvidia-fabricmanager", "nvidia-persistenced"],
    }
    assert probe.ledger_evidence("VERIFY_NO_GPU_CLIENTS", {"details": {"x": 1}}) == {}
    assert probe.ledger_evidence("RESET_GPU", {"error": "no details"}) == {}
    assert probe.ledger_evidence("RESET_GPU", ["not", "a", "result"]) == {}
    calls = [{"command": "c", "outcome": "reset"}] * (probe.LEDGER_EVIDENCE_LIMIT + 5)
    bounded = probe.ledger_evidence("RESET_GPU", {"details": {"physical_calls": calls}})
    assert len(bounded["physical_calls"]) == probe.LEDGER_EVIDENCE_LIMIT


# --------------------------------------------------------------------------- #
# Runner surface
# --------------------------------------------------------------------------- #
def test_the_case_constants_follow_the_contract() -> None:
    assert destr026.CASE_ID == "GF-REGIONAL-DESTR-026"
    assert (
        destr026.PREDECESSOR_CASE_ID
        == formal_predecessor(destr026.CASE_ID)
        == ("GF-REGIONAL-DESTR-015")
    ), "the two-node join is the evidence anchor"
    metadata = RegionalCaseMetadata(
        case_id=destr026.CASE_ID,
        title="",
        category="DESTR",
        level="",
        risk="",
        automation="",
        procedure="",
        predecessor=destr026.PREDECESSOR_CASE_ID,
    )
    assert destr026.CONFIRMATION == metadata.confirmation == "DESTR026_EXECUTE"
    assert destr026.CASE.case_id == destr026.CASE_ID, "the CaseRunner names the case"
    path = ROOT / "scripts/e2e/regional/run_destr026_same_node_two_gpu_reset.py"
    assert path.stat().st_mode & 0o111 == 0o111, "the runner is executable"


def test_the_parser_offers_the_documented_live_flags() -> None:
    parser = destr026.parser()
    help_text = parser.format_help()
    for flag in (
        "--plan",
        "--execute",
        "--confirm",
        "--maintenance-window-end",
        "--site-profile",
        "--fault-node",
        "--node-b",
        "--pci-bdf-a",
        "--pci-bdf-b",
        "--host-probe-image",
        "--predecessor-evidence",
    ):
        assert flag in help_text, f"{flag} is missing from the runner help"
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr026.CONFIRMATION,
            "--fault-node",
            NODE,
            "--node-b",
            PARTNER,
        ]
    )
    assert (execute.fault_node, execute.node_b) == (NODE, PARTNER)
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])


def test_run_and_job_identities_are_stable_per_run_dir(tmp_path: Path) -> None:
    first = tmp_path / "regional-acceptance-r8-eks"
    second = tmp_path / "other" / "regional-acceptance-r9-eks"

    assert destr026.run_identity(first, 1) == destr026.run_identity(first, 1)
    assert destr026.run_identity(first, 1) != destr026.run_identity(second, 1), (
        "two rounds ending in the same lane token still differ"
    )
    assert destr026.run_identity(first, 1).startswith("destr026-eks-"), (
        "the lane token stays readable"
    )
    job, attempt = destr026.derived_identity(first, 2)
    assert job.startswith("destr026-") and attempt == f"{job}-a001", (job, attempt)


def test_plan_details_name_the_mutation_and_the_stop_conditions() -> None:
    settings = destr026.Settings(
        regional=None,  # type: ignore[arg-type]
        site_file=Path("/tmp/site.yaml"),
        manifest=Path("/tmp/job.yaml"),
        host_probe_image="image",
        fault_node=NODE,
        node_b=PARTNER,
        pci_bdf_a="",
        pci_bdf_b="",
        job_id="destr026-job",
        attempt_id=SOURCE_ATTEMPT,
        predecessor_path=Path("/tmp/none.json"),
    )
    preflight_result = {"release_id": "rel-1", "nodes": {}, "store": {}}

    details = destr026.plan_details(settings, preflight_result)

    assert details["risk"] == "destructive"
    assert (details["fault_node"], details["node_b"]) == (NODE, PARTNER)
    assert any("two workflows" in item for item in details["stop_conditions"]), (
        "a split into two workflows stops the case"
    )
    assert "late_fault_variant" in details, "the documented, not driven, variant"
    assert details["preflight_identity_digest"] == destr026.join_case.identity_digest(
        details["preflight_identity"]
    ), "the plan pins the identity digest it re-checks before execute"
