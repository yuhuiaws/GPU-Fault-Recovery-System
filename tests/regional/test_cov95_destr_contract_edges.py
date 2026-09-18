from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr001_gpu_reset as reset
from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import run_destr008_warm_spare_shortage as shortage
from scripts.e2e.regional import run_destr009_workload_restart as workload
from scripts.e2e.regional import run_destr010_fabric_manager_restart as fabric
from tests.regional._cov95_destr_actions import ActionHarness
from tests.regional._cov95_destr_edge_data import change
from tests.regional._cov95_destr_idle import IdleHarness
from tests.regional._cov95_destr_warm import (
    FAULT,
    SPARE,
    failed_shortage,
    node_snapshot,
    settings_for,
    successful_failover,
)
from tests.regional._destructive_acceptance_support import (
    reset_host_pair,
    restart_state,
)
from tests.regional.test_destr002_review_fixes import SpyRegional


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("node", "ready"), "False", "not Ready"),
        (("node", "unschedulable"), True, "already unschedulable"),
        (("node", "taints"), [{"key": "foreign"}], "pre-existing taints"),
        (("workloads",), [{"name": "foreign"}], "non-system running"),
        (("state", "agent", "lifecycle_state"), "REVOKED", "not ACTIVE"),
        (("state", "profile", "capabilities", 0, "mode"), "OBSERVE", "not OWN"),
        (("state", "profile", "warnings"), ["drift"], "profile has warnings"),
        (("state", "queue", "fault_backlog_depth"), 1, "processor queue"),
        (
            ("state", "remote_commands", "open_by_cluster"),
            {"cluster-a": 1},
            "remote command queue",
        ),
        (("state", "event"), {"xid": 46}, "recent XID"),
        (("tests", "passed"), False, "regression tests failed"),
    ],
)
def test_reset_preflight_names_each_unsafe_input(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = IdleHarness(reset, tmp_path, monkeypatch)
    frame = {
        "state": h.state,
        "node": h.node,
        "workloads": [],
        "tests": {"passed": True},
    }
    assert reset.preflight_errors(**frame) == [], frame
    change(frame, path, value)
    errors = reset.preflight_errors(**frame)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("node", "ready"), "False", "not Ready"),
        (("node", "unschedulable"), True, "unschedulable"),
        (("node", "taints"), [{"key": "foreign"}], "has taints"),
        (("workloads",), [{"name": "foreign"}], "non-system"),
        (("state", "agent", "lifecycle_state"), "DRAINING", "not ACTIVE"),
        (("state", "profile", "capabilities", 0, "owner"), "foreign", "not OWN"),
        (("state", "profile", "warnings"), ["drift"], "profile has warnings"),
        (
            ("fabric", "active_workflow_incidents"),
            [{"incident_id": "busy"}],
            "active workflow",
        ),
        (("state", "queue", "fault_backlog_depth"), 2, "processor queue"),
        (
            ("state", "remote_commands", "open_by_cluster"),
            {"cluster-a": 1},
            "remote command",
        ),
    ],
)
def test_fabric_preflight_requires_healthy_idle_owned_capability(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    frame = {
        "node": {"ready": "True", "unschedulable": False, "taints": []},
        "workloads": [],
        "state": {
            "agent": {
                "lifecycle_state": "ACTIVE",
                "allowed_operations": ["RESTART_FABRIC_MANAGER"],
            },
            "profile": {
                "warnings": [],
                "capabilities": [
                    {
                        "capability": "fabricManagerRestart",
                        "mode": "OWN",
                        "owner": "gpu-fault-node-agent",
                        "adapter": "node-action",
                    }
                ],
            },
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        },
        "fabric": {"active_workflow_incidents": [], "recent_xid_events": []},
    }
    assert fabric.validate_preflight(**frame) == [], frame
    change(frame, path, value)
    errors = fabric.validate_preflight(**frame)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("event", "xid"), 79, "not XID 45"),
        (("event", "evidence_ref"), "api-replay://unit", "not backed by kmsg"),
        (("decision", "official_action"), "RESET_GPU", "RESTART_FM"),
        (("decision", "disposition"), "OBSERVE", "not EXECUTABLE"),
        (("workflow", "official_steps"), [], "two-step contract"),
        (
            ("workflow", "official_steps", 1, "execution_owner"),
            "foreign",
            "execution owners",
        ),
        (
            ("workflow", "completed_operations"),
            ["MARK_UNSCHEDULABLE"],
            "forbidden isolation",
        ),
        (("workflow", "status"), "FAILED", "not SUCCEEDED"),
        (("commands", 0, "status"), "FAILED", "not uniquely SUCCEEDED"),
        (("evidence",), [], "kernel evidence"),
        (("notifications",), [], "notification is not unique"),
        (("notifications", 0, "status"), "PENDING", "notification is not terminal"),
    ],
)
def test_fabric_workflow_rejects_missing_or_wrong_proof(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = ActionHarness(fabric, tmp_path, monkeypatch)
    state = {**deepcopy(h.state), "evidence": [{"record_id": "retained"}]}
    assert fabric.workflow_errors(state) == [], state
    change(state, path, value)
    errors = fabric.workflow_errors(state)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("event", "xid"), 45, "not XID 79"),
        (("decision", "action"), "RESET_GPU", "REBOOT_NODE"),
        (("workflow", "status"), "FAILED", "not SUCCEEDED"),
        (("workflow", "official_steps"), [], "missing RESTART_NODE"),
        (("workflow", "step_executions"), [], "successful RESTART_NODE"),
        (("submission", "state"), "PENDING", "not SUBMITTED"),
        (("submission", "action"), "REPLACE", "not REBOOT"),
        (("agent", "artifact_sha256"), "changed", "artifact changed"),
        (("agent", "boot_id"), "agent-boot-1", "boot ID did not provably change"),
        (("agent", "lifecycle_state"), "DRAINING", "did not return ACTIVE"),
    ],
)
def test_reboot_workflow_requires_complete_reboot_and_validation_proof(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    state = SpyRegional(None).wait_for_workflow()
    kwargs = {"expected_artifact": "artifact-a", "expected_boot_id": "agent-boot-1"}
    assert reboot.workflow_errors(state, **kwargs) == [], state
    change(state, path, value)
    errors = reboot.workflow_errors(state, **kwargs)
    assert any(expected in error for error in errors), errors


def test_reboot_contract_rejects_extra_physical_operation_and_reordered_validation() -> (
    None
):
    state = SpyRegional(None).wait_for_workflow()
    state["workflow"]["official_steps"].append({"operation": "RESET_GPU"})
    state["workflow"]["official_steps"].reverse()
    errors = reboot.workflow_errors(
        state, expected_artifact=None, expected_boot_id=None
    )
    assert "reboot workflow contains an unexpected physical operation" in errors, errors
    assert "reboot validation and scheduling order is not exact" in errors, errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("event", "xid"), 46, "not XID 11"),
        (("decision", "official_action"), "RESET_GPU", "RESTART_APP"),
        (("workflow", "official_steps"), [], "exactly one"),
        (("workflow", "step_executions"), [], "did not reach SUCCEEDED"),
        (
            ("workflow", "step_executions", 0, "adapter_operation_id"),
            "local/stop",
            "not remote/",
        ),
        (
            ("workflow", "step_executions", 1, "status"),
            "FAILED",
            "RESTART_WORKLOAD did not reach SUCCEEDED",
        ),
    ],
)
def test_workload_contract_requires_exact_remote_stop_restart_proof(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    state = restart_state(24)
    assert workload.workflow_errors(state) == [], state
    change(state, path, value)
    errors = workload.workflow_errors(state)
    assert any(expected in error for error in errors), errors


def test_workload_contract_cannot_include_node_mutation() -> None:
    state = restart_state(24)
    state["workflow"]["official_steps"].append({"operation": "RESET_GPU"})
    errors = workload.workflow_errors(state)
    assert "workload restart workflow contains a node mutation" in errors, errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("after", "gpu_inventory"), [], "inventory count changed"),
        (("after", "ledger", 1, "state"), "FAILED", "result is not SUCCEEDED"),
        (("after", "gpu_fault_timers"), ["stray.timer"], "timer inventory"),
        (("after", "quiesce_states"), ["owned"], "quiesce state remains"),
        (
            ("after", "services", "kubelet.service", "ActiveState"),
            "inactive",
            "service did not return active",
        ),
        (
            ("after", "sampler", "last", "gpu_uuids"),
            ["GPU-foreign"],
            "final GPU UUID set",
        ),
        (
            ("after", "ledger", 1, "gpu_uuids"),
            ["GPU-foreign"],
            "does not identify only",
        ),
        (
            ("after", "gpu_inventory", 1, "pci_bdf"),
            "0000:ff:00",
            "UUID/BDF inventory changed",
        ),
        (
            ("after", "ledger", 1, "incident_id"),
            "foreign",
            "incident_id does not match",
        ),
        (
            ("after", "ledger", 1, "workflow_request_id"),
            "foreign",
            "workflow_request_id does not match",
        ),
    ],
)
def test_reset_host_evidence_is_bound_to_target_hardware_and_command(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    before, after = reset_host_pair(minimum=7, last=8)
    after = deepcopy(after)
    after["ledger"][1].update(
        incident_id="incident-owned", workflow_request_id="workflow-owned"
    )
    kwargs = {
        "expected_gpu_count": 8,
        "target_bdf": "0000:59:00",
        "incident_id": "incident-owned",
        "workflow_request_id": "workflow-owned",
    }
    assert reset.host_errors(before, after, **kwargs) == [], (before, after)
    frame = {"before": before, "after": after}
    change(frame, path, value)
    errors = reset.host_errors(frame["before"], frame["after"], **kwargs)
    assert any(expected in error for error in errors), errors


def test_reset_host_rejects_duplicate_new_command_ids_but_ignores_inactive_optional_service() -> (
    None
):
    before, after = reset_host_pair(minimum=7, last=8)
    after = deepcopy(after)
    before["services"]["optional.service"] = {"ActiveState": "inactive"}
    assert (
        reset.host_errors(before, after, expected_gpu_count=8, target_bdf="0000:59:00")
        == []
    ), (before, after)
    after["ledger"].append(
        {"command_id": "cmd-reset", "operation": "VALIDATE_GPU", "attempt": 1}
    )
    errors = reset.host_errors(
        before, after, expected_gpu_count=8, target_bdf="0000:59:00"
    )
    assert "Node Agent ledger contains duplicate command IDs" in errors, errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("workflow", "status"), "FAILED", "not SUCCEEDED"),
        (("workflow", "official_steps"), [], "operations differ"),
        (
            ("workflow", "step_executions", 1, "status"),
            "FAILED",
            "RESTART_WORKLOAD did not reach",
        ),
        (
            (
                "workflow",
                "step_executions",
                1,
                "details",
                "notification_context",
                "source_gpu_count",
            ),
            1,
            "source GPU count",
        ),
        (
            (
                "workflow",
                "step_executions",
                1,
                "details",
                "notification_context",
                "target_gpu_count",
            ),
            1,
            "target GPU count",
        ),
        (
            (
                "workflow",
                "step_executions",
                1,
                "details",
                "notification_context",
                "restart_count",
            ),
            2,
            "restart count",
        ),
        (("notifications",), [], "notification was not persisted"),
    ],
)
def test_failover_workflow_rejects_incomplete_success_or_restart_budget_proof(
    path: tuple[str | int, ...], value: Any, expected: str, tmp_path: Path
) -> None:
    settings = settings_for(failover, tmp_path)
    state = successful_failover()
    assert failover.workflow_errors(state, settings) == [], state
    change(state, path, value)
    errors = failover.workflow_errors(state, settings)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("workflow", "status"), "SUCCEEDED", "workflow is not FAILED"),
        (
            ("workflow", "step_executions", 0, "status"),
            "FAILED",
            "STOP_WORKLOADS did not",
        ),
        (
            ("workflow", "step_executions", 1, "status"),
            "SUCCEEDED",
            "REPLACE_NODE did not",
        ),
        (
            ("workflow", "step_executions", 1, "error"),
            "other failure",
            "reason does not match",
        ),
        (("workflow", "official_steps"), [], "HEALTHY_WARM_SPARE_ONLY"),
        (("notifications",), [], "expected 1"),
        (("fault_node", "unschedulable"), False, "not kept quarantined"),
        (
            ("spare_node", "annotations", shortage.SPARE_RESERVATION_ANNOTATION),
            "incident-owned",
            "left an incident spare reservation",
        ),
        (
            ("spare_node", "annotations", shortage.SPARE_POOL_STATE_ANNOTATION),
            "ALLOCATED",
            "left the spare ALLOCATED",
        ),
    ],
)
def test_shortage_proof_requires_actual_failed_gate_without_partial_allocation(
    path: tuple[str | int, ...], value: Any, expected: str, tmp_path: Path
) -> None:
    settings = settings_for(shortage, tmp_path)
    state = failed_shortage("topology-mismatch", "event-owned")
    state["fault_node"] = {
        **node_snapshot(FAULT),
        "unschedulable": True,
        "taints": [{"key": shortage.QUARANTINE_TAINT}],
    }
    state["spare_node"] = node_snapshot(SPARE)
    kwargs = {"scenario": "topology-mismatch", "event_id": "event-owned"}
    assert shortage.scenario_errors(state, settings, **kwargs) == [], state
    change(state, path, value)
    errors = shortage.scenario_errors(state, settings, **kwargs)
    assert any(expected in error for error in errors), errors
