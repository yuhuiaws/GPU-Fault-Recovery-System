from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from scripts.e2e.regional import destr015_verdicts as v15
from scripts.e2e.regional import destr016_verdicts as v16
from scripts.e2e.regional import destr017_verdicts as v17
from scripts.e2e.regional import destr018_verdicts as v18
from scripts.e2e.regional import destr021_verdicts as v21
from scripts.e2e.regional import destr022_verdicts as v22
from scripts.e2e.regional import destr023_verdicts as v23
from scripts.e2e.regional import destr024_verdicts as v24
from scripts.e2e.regional import run_destr001_gpu_reset as reset
from tests.regional import test_destr015_parallel_branch_join as d15
from tests.regional import test_destr016_preempting_reboot as d16
from tests.regional import test_destr017_out_of_band_reboot_fence as d17
from tests.regional import test_destr018_lifetime_deadline as d18
from tests.regional._cov95_destr_branches import branch_host, branch_pods
from tests.regional._cov95_destr_edge_data import change
from tests.regional._cov95_destr_warm import SPARE, node_snapshot
from tests.regional.test_destr023_idle_cluster_reset import coverage
from tests.regional.test_destr024_watcher_down_fail_closed import blocked_state
from tests.regional.test_destr_reset_workload_contract import (
    host_pair,
    workload_contract,
)


@pytest.mark.parametrize(
    "defect", ["invalid-observation", "lineage", "foreign-node", "partial-clients"]
)
def test_reset_workload_exception_requires_complete_local_owned_gpu_clients(
    defect: str,
) -> None:
    contract = workload_contract()
    _, after = host_pair()
    assert reset.post_restart_client_errors(after, contract) == [], (after, contract)
    if defect == "invalid-observation":
        contract["observation"] = None
        expected = "observation or host timestamp is invalid"
    elif defect == "lineage":
        contract["source_pod_uids"] = [""]
        expected = "Pod lineage is incomplete"
    elif defect == "foreign-node":
        contract["pods"][0]["node"] = "other-node"
        contract["observation"]["containers"][0]["node_id"] = "other-node"
        expected = "ownership is not observed on the target node"
    else:
        after["compute_clients"].pop()
        expected = "do not cover the observed workload allocation"
    errors = reset.post_restart_client_errors(after, contract)
    assert len(errors) == 1 and expected in errors[0], errors


@pytest.mark.parametrize(
    "environment, accepted",
    [
        ("kubernetes", True),
        ("eks", True),
        ("hyperpod-eks", True),
        ("hyperpod-slurm", False),
    ],
)
def test_post_restart_observation_accepts_every_kubernetes_environment(
    environment: str, accepted: bool
) -> None:
    """The watcher on a HyperPod EKS cluster reports ``hyperpod-eks``; demanding
    the bare ``kubernetes`` member failed COLLECT-016 D live (2026-09-18)."""
    contract = workload_contract()
    _, after = host_pair()
    contract["observation"]["environment"] = environment
    errors = reset.post_restart_client_errors(after, contract)
    if accepted:
        assert errors == [], errors
    else:
        assert len(errors) == 1 and "identity or freshness" in errors[0], errors


@pytest.mark.parametrize("defect", ["fabric-reset", "cordon", "pod-ready"])
def test_parallel_success_requires_only_target_resets_and_usable_workload(
    defect: str,
) -> None:
    if defect == "fabric-reset":
        hosts = {
            node: {"before": branch_host(node), "after": branch_host(node, after=True)}
            for node in d15.NODES
        }
        hosts[d15.NODE_A]["after"]["ledger"].append(
            {
                "command_id": "foreign-fabric-reset",
                "operation": "RESET_ALL_GPUS_NVSWITCHES",
                "state": "FAILED",
            }
        )
        errors = v15.host_errors(hosts, nodes=d15.NODES, expected_gpu_count=8)
        expected = "shows a full fabric reset"
    elif defect == "cordon":
        nodes = {
            node: {"ready": "True", "unschedulable": node == d15.NODE_A}
            for node in d15.NODES
        }
        errors = v15.schedulability_errors(nodes, nodes=d15.NODES)
        expected = "not schedulable after RESTORE_SCHEDULING"
    else:
        pods = branch_pods(restarted=True)
        pods[0]["ready"] = False
        errors = v15.workload_errors(pods=pods, source_uids={"old"}, nodes=d15.NODES)
        expected = "not Running and Ready"
    assert any(expected in error for error in errors), errors


def test_preemption_barrier_command_must_still_be_waiting() -> None:
    commands = d16.barrier_commands()
    commands[0]["status"] = "FAILED"
    errors = v16.barrier_reason_errors(commands)
    assert any("barrier remote command is not WAITING" in error for error in errors), (
        errors
    )


def test_absorption_needs_a_known_original_workflow() -> None:
    before = d16.barrier_snapshot()
    before["workflow"]["request_id"] = ""
    errors = v16.absorb_errors(before, d16.absorbed_snapshot(), node=d16.NODE)
    assert "no workflow was observed before the absorbed fault" in errors, errors


def test_superseded_workflow_must_have_established_the_quiesce_it_hands_off() -> None:
    workflow = d16.superseded_reset_workflow()
    workflow["step_executions"] = [
        row
        for row in workflow["step_executions"]
        if row["operation"] != "QUIESCE_GPU_SERVICES"
    ]
    errors = v16.superseded_predecessor_errors(workflow)
    assert "the superseded reset has no successful QUIESCE_GPU_SERVICES" in errors, (
        errors
    )


def test_preemption_cancellation_requires_one_exact_barrier_command() -> None:
    errors = v16.cancelled_command_errors([], successor_request_id="successor")
    assert errors == ["there is not exactly one barrier remote command: 0"], errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("business_workloads",), [{"name": "foreign"}], "runs business workloads"),
        (("profile", "capabilities"), [], "gpuReset capability is not OWN"),
        (("profile", "warnings"), ["drift"], "runtime profile carries warnings"),
    ],
)
def test_fence_preflight_requires_idle_owned_unwarned_target(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    host = d17.host_before()
    frame = {
        "node": d17.NODE,
        "node_snapshot": {"ready": "True", "unschedulable": False, "taints": []},
        "agent": d17.agent_before(),
        "profile": {
            "capabilities": [
                {"capability": "gpuReset", "mode": "OWN", "owner": v17.AGENT_OWNER}
            ],
            "warnings": [],
        },
        "business_workloads": [],
        "queue": {"depth": 0, "fault_backlog_depth": 0},
        "remote_commands": {},
        "recent_events": [],
        "host_snapshot": host,
        "reboot_status": {"armed": False},
        "expected_gpu_count": len(host["gpu_inventory"]),
    }
    assert v17.preflight_errors(**frame) == [], frame
    change(frame, path, value)
    errors = v17.preflight_errors(**frame)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("defect", ["plan", "waiting-step"])
def test_fence_failure_must_have_reached_the_actual_reset_barrier(defect: str) -> None:
    workflow = d17.fenced_workflow()
    if defect == "plan":
        workflow["official_steps"] = []
        expected = "did not compile the idle-node reset plan"
    else:
        workflow["step_executions"] = [
            row
            for row in workflow["step_executions"]
            if row["operation"] != v17.FENCE_STEP_OPERATION
        ]
        expected = "never executed"
    errors = v17.workflow_errors(workflow, d17.fenced_incident(), node=d17.NODE)
    assert any(expected in error for error in errors), errors


def test_lifetime_requires_one_successful_compensation_command() -> None:
    commands = [
        row
        for row in d18.happy_commands()
        if row["step"]["operation"] != v18.COMPENSATION_STEP
    ]
    errors = v18.remote_command_errors(commands, t_cancel=d18.T_CANCEL)
    assert any(
        "expected exactly one successful RESTORE_GPU_SERVICES" in error
        for error in errors
    ), errors


def test_lifetime_support_escalation_cannot_widen_to_another_node() -> None:
    escalation = d18.happy_escalation()
    escalation["incident"]["node_ids"].append("foreign")
    errors = v18.escalation_errors(escalation, node=d18.NODE)
    assert any("support incident covers nodes" in error for error in errors), errors


@pytest.mark.parametrize("cadence", [0.0, -1.0, float("nan"), float("inf")])
def test_attempt_budget_never_accepts_nonpositive_or_nonfinite_cadence(
    cadence: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        v18.worst_case_verify_attempts(lifetime_seconds=180, cadence_seconds=cadence)


def test_lifetime_window_must_leave_room_after_containment() -> None:
    errors = v18.lifetime_margin_errors(
        lifetime_seconds=180,
        execution_timeout_seconds=180,
        cadence_seconds=30,
        step_waiting_cap_seconds=180,
        containment_allowance_seconds=180,
    )
    assert any(
        "leaves no room after the containment allowance" in error for error in errors
    ), errors


@pytest.mark.parametrize("delta", [True, "unknown", -1, float("inf"), float("nan")])
def test_counter_delta_must_be_a_finite_nondecreasing_number(delta: Any) -> None:
    errors = v18.metric_errors({"lifetime_exceeded_total": {"delta": delta}})
    assert errors == [
        f"{v18.LIFETIME_METRIC} has an unusable or decreasing counter delta"
    ], errors


@pytest.mark.parametrize("expected_count", [4, "not-a-count"])
def test_plugin_count_falls_back_to_workflow_parameters_without_guessing(
    expected_count: Any,
) -> None:
    bundle = {
        "commands": [],
        "workflow": {
            "official_steps": [
                {"operation": "FREEZE_EVIDENCE"},
                {
                    "operation": "RESTART_EFA_DEVICE_PLUGIN",
                    "parameters": {"expected_count": expected_count},
                },
            ]
        },
    }
    errors = v21.expected_count_errors(bundle, node="node-a", baseline_efa=4)
    if expected_count == 4:
        assert errors == [], errors
    else:
        assert len(errors) == 1 and "adapter would fail closed" in errors[0], errors


@pytest.mark.parametrize("defect", ["health", "reserved-at"])
def test_spare_reclaim_requires_healthy_unreserved_pool_member(defect: str) -> None:
    spare = node_snapshot(SPARE)
    if defect == "health":
        spare["labels"][v22.HYPERPOD_HEALTH_LABEL] = "Unhealthy"
        expected = "health label is not Schedulable"
    else:
        spare["annotations"][v22.SPARE_RESERVED_AT_ANNOTATION] = "2026-09-12T12:00:00Z"
        expected = "already carries a reserved-at timestamp"
    errors = v22.spare_refusals(spare)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("freshness_seconds",), 0, "no freshness window"),
        (("heartbeat", "observed_at"), None, "no valid observed_at"),
        (("heartbeat",), None, "age exists without a heartbeat"),
    ],
)
def test_coverage_heartbeat_cannot_fill_in_missing_temporal_evidence(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    sample = coverage()
    change(sample, path, value)
    errors = v23.coverage_supported_errors(sample)
    assert any(expected in error for error in errors), errors


def test_coverage_requires_explicit_timezone_and_bounded_wait() -> None:
    sample = coverage()
    sample["probed_at"] = sample["probed_at"].removesuffix("+00:00")
    assert v23.coverage_supported_errors(sample) == [
        "coverage probe has no valid probed_at timestamp"
    ], sample
    sample = coverage()
    sample["freshness_seconds"] = v23.MAX_EXPIRY_WAIT_SECONDS + 1
    with pytest.raises(ValueError, match="exceeds"):
        v23.expiry_wait_seconds(sample)


def test_managed_pod_inventory_excludes_unmanaged_pods_not_terminal_managed_pods() -> (
    None
):
    managed = {
        "metadata": {
            "namespace": "training",
            "name": "done",
            "labels": {v23.MANAGED_LABEL: "true"},
        },
        "spec": {"nodeName": "node-a"},
        "status": {"phase": "Succeeded"},
    }
    unmanaged = deepcopy(managed)
    unmanaged["metadata"]["labels"] = {}
    assert v23.managed_pod_summary([unmanaged, managed]) == [
        {
            "namespace": "training",
            "name": "done",
            "node": "node-a",
            "phase": "Succeeded",
        }
    ], (managed, unmanaged)


@pytest.mark.parametrize("defect", ["missing-workflow", "wrong-block-kind"])
def test_watcher_down_requires_actual_compile_gate_refusal(defect: str) -> None:
    state = blocked_state()
    if defect == "missing-workflow":
        state["workflow"] = {}
        expected = "no workflow was opened"
    else:
        state["workflow"]["blocked_kind"] = "unknown"
        expected = "blocked_kind"
    errors = v24.blocked_workflow_errors(state)
    assert any(expected in error for error in errors), errors


def test_watcher_down_does_not_require_previously_inactive_optional_service() -> None:
    baseline = {"services": {"optional.service": {"ActiveState": "inactive"}}}
    assert v24.host_untouched_errors(baseline, {}) == [], baseline
