"""Execution-level regressions for the twelve revised health-policy categories."""

from __future__ import annotations

import json

import pytest

from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from tests.execution._raw_health_e2e_support import RawHealthRig


@pytest.fixture
def rig(tmp_path, monkeypatch):
    value = RawHealthRig(tmp_path, monkeypatch)
    try:
        yield value
    finally:
        evidence = {
            "raw_inputs": value.raw_inputs,
            "api_responses": value.responses,
            "workflows": [
                workflow.model_dump(mode="json")
                for workflow in value.store.list_workflows()
            ],
            "simulated_device_calls": [
                operation.value for operation in value.boundary.calls
            ],
            "simulated_node_final_state": value.core.node,
            "live_actions_executed": False,
        }
        value.store.close()
        evidence["cleanup"] = "fixture database closed; no live resources created"
        print("GENERAL_FAULT_EVIDENCE=" + json.dumps(evidence, default=str))


@pytest.mark.parametrize(
    "message",
    [
        "mce: uncorrectable hardware error",
        "remounting filesystem read-only",
        "efa: fatal error",
        "kernel panic",
    ],
    ids=["mce", "storage", "rdma", "lockup"],
)
def test_fatal_host_log_stops_the_owned_attempt_before_quarantine(rig, message):
    result = rig.log(message)
    workflow = rig.store.get_workflow(result["workflow_request_ids"][0])
    outcome = rig.execute(workflow)
    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    assert Op.STOP_WORKLOADS in rig.boundary.calls, rig.boundary.calls
    assert rig.boundary.running is False, "the affected attempt was stopped"
    assert rig.core.node["spec"]["unschedulable"] is True, rig.core.node
    assert (
        rig.store.get_incident(workflow.incident_id).state is IncidentState.QUARANTINED
    )


@pytest.mark.parametrize("source", ["journal", "training-log"])
def test_untrusted_fatal_text_cannot_stop_training(rig, source):
    result = rig.log("mce: uncorrectable hardware error", source=source)
    assert result["workflow_request_ids"] == [], result
    assert rig.boundary.calls == [], "no mutation is dispatched"


def test_application_log_preserves_attribution_and_watcher_ownership(rig):
    rig.gpu(
        {
            "gpu_temperature_c": 40,
            "nvlink_crc_flit_error_total": 0,
            "nvlink_recovery_error_total": 0,
            "nvlink_replay_error_total": 0,
        }
    )
    result = rig.log("NCCL timeout error", source="journal", fields={"_PID": "123"})
    finding = result["findings"][0]
    assert finding["attempt_id"] == "attempt-a", finding
    assert finding["diagnostic_parameters"]["process_state"] == "LIVE", finding
    assert finding["recommended_action"] == "NO_ACTION", finding
    assert finding["diagnostic_parameters"]["recovery_owner"] == "COMPLETION_WATCHER", (
        finding
    )
    assert (
        result["workflow_request_ids"]
        == result["incident_ids"]
        == result["marker_ids"]
        == []
    ), result
    assert rig.boundary.calls == [], rig.boundary.calls
    assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node


def test_confirmed_thermal_fault_requires_review_after_cooling(rig):
    for _ in range(3):
        rig.gpu({"gpu_temperature_c": 95})
        rig.advance(15)
    workflow = next(
        item
        for item in rig.store.list_workflows()
        if Op.QUARANTINE in [step.operation for step in item.official_steps]
    )
    outcome = rig.execute(workflow)
    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    assert rig.boundary.running is False, rig.boundary.calls
    assert rig.core.node["spec"]["unschedulable"] is True, rig.core.node
    for _ in range(3):
        rig.advance(300)
        rig.gpu({"gpu_temperature_c": 40})
    assert rig.core.node["spec"]["unschedulable"] is True, (
        "idle cooling is not proof that the cooling system was repaired"
    )
    assert Op.RESTART_WORKLOAD not in rig.boundary.calls, rig.boundary.calls
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
    assert rig.store.get_incident(workflow.incident_id).state is IncidentState.ESCALATED


def test_uncorrectable_ecc_stops_for_analysis_without_automatic_rma(rig):
    rig.gpu({"gpu_temperature_c": 40, "ecc_dbe_volatile_total": 1})
    [workflow] = rig.store.list_workflows()
    assert rig.execute(workflow).status is WorkflowStatus.SUCCEEDED
    assert rig.boundary.running is False, rig.boundary.calls
    assert Op.ESCALATE_SUPPORT not in rig.boundary.calls, (
        "DBE alone is not an RMA verdict"
    )
    assert Op.RESTART_WORKLOAD not in rig.boundary.calls, (
        "recovery needs fresh driver evidence"
    )


def test_unknown_observation_breaks_thermal_confirmation(rig):
    rig.gpu({"gpu_temperature_c": 95})
    rig.advance(15)
    rig.gpu({"gpu_temperature_c": 95}, errors=["lost exporter field"])
    rig.advance(15)
    result = rig.gpu({"gpu_temperature_c": 95})
    temperature = next(
        item
        for item in result["findings"]
        if item["canonical_name"] == "gpu_temperature_c"
    )
    assert temperature["confirmation_count"] == 1, temperature
    assert not any(
        Op.STOP_WORKLOADS in [step.operation for step in item.official_steps]
        for item in rig.store.list_workflows()
    ), "UNKNOWN cannot complete a sustained thermal confirmation"
    assert rig.boundary.running is True, rig.boundary.calls


def test_power_brake_is_not_power_cap_and_never_reboots_the_node(rig):
    for mask in (0x04, 0x80, 0x84, 0x80):
        rig.gpu({"gpu_temperature_c": 40, "clock_throttle_reasons": mask})
        rig.advance(15)
    workflow = next(
        item
        for item in rig.store.list_workflows()
        if Op.QUARANTINE in [step.operation for step in item.official_steps]
    )
    operations = [step.operation for step in workflow.official_steps]
    assert Op.RESET_GPU not in operations and Op.RESTART_NODE not in operations, (
        operations
    )
    assert rig.execute(workflow).status is WorkflowStatus.SUCCEEDED
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
    assert rig.core.node["spec"]["unschedulable"] is True, rig.core.node


@pytest.mark.parametrize(
    "values,tags,polls",
    [
        (
            {"gpu_temperature_c": 75},
            "<gpu_temp_slow_threshold>70 C</gpu_temp_slow_threshold>",
            3,
        ),
        (
            {"gpu_tlimit_c": -3, "gpu_temperature_c": 91},
            "<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>"
            "<gpu_temp_max_gpu_tlimit_threshold>0 C</gpu_temp_max_gpu_tlimit_threshold>",
            3,
        ),
        (
            {"memory_temperature_c": 91},
            "<gpu_temp_max_mem_threshold>90 C</gpu_temp_max_mem_threshold>",
            3,
        ),
        (
            {"gpu_temperature_c": 106},
            "<gpu_temp_max_threshold>105 C</gpu_temp_max_threshold>",
            1,
        ),
    ],
    ids=["absolute", "tlimit", "memory", "shutdown"],
)
def test_device_temperature_limits_reach_real_containment(rig, values, tags, polls):
    xml = (
        "<nvidia_smi_log><gpu><uuid>GPU-a</uuid>"
        "<pci><pci_bus_id>0000:81:00.0</pci_bus_id></pci>"
        f"<temperature>{tags}</temperature></gpu></nvidia_smi_log>"
    )
    for index in range(polls):
        rig.gpu(values, temperature_xml=xml)
        if index < polls - 1:
            assert not any(
                Op.STOP_WORKLOADS in [step.operation for step in item.official_steps]
                for item in rig.store.list_workflows()
            ), "unconfirmed temperature cannot stop the attempt"
        rig.advance(15)
    workflow = next(
        item
        for item in rig.store.list_workflows()
        if Op.QUARANTINE in [step.operation for step in item.official_steps]
    )
    assert rig.execute(workflow).status is WorkflowStatus.SUCCEEDED
    assert rig.boundary.running is False, rig.boundary.calls
    assert rig.core.node["spec"]["unschedulable"] is True, rig.core.node
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
    assert Op.RESTART_WORKLOAD not in rig.boundary.calls, rig.boundary.calls
    assert rig.store.get_incident(workflow.incident_id).state is IncidentState.ESCALATED


def test_thermal_warning_and_duration_execute_only_read_only_diagnostics(rig):
    for index in range(4):
        rig.gpu(
            {
                "gpu_temperature_c": 86,
                "clock_throttle_reasons": 0x20,
                "thermal_violation_total_us": index * 1_000_000,
            }
        )
        rig.advance(15)
    workflows = rig.store.list_workflows()
    assert workflows, "correlated warning evidence reaches the real planner"
    for workflow in workflows:
        assert {step.operation for step in workflow.official_steps} <= {
            Op.FREEZE_EVIDENCE,
            Op.VALIDATE_GPU,
        }
        rig.execute(workflow)
    rig.gpu({"gpu_temperature_c": 40, "thermal_violation_total_us": 3_000_000})
    for workflow in workflows:
        assert rig.execute(workflow).status is WorkflowStatus.SUCCEEDED
    assert rig.boundary.calls == [], (
        "diagnostics do not stop work or run hardware operations"
    )
    assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node


def test_temperature_without_gpu_identity_contains_without_readmission(rig):
    for _ in range(3):
        rig.post(
            "/v1/collector-events/gpu-metrics",
            {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "observed_at": rig.clock.isoformat(),
                "source": "DCGM_EXPORTER",
                "runtime_profile_version": "simulated-v1",
                "product": "H100",
                "samples": [
                    {
                        "metric_name": "temperature",
                        "canonical_name": "gpu_temperature_c",
                        "value": 95,
                        "gpu_index": "0",
                    }
                ],
            },
        )
        rig.advance(15)
    workflow = next(
        item
        for item in rig.store.list_workflows()
        if Op.QUARANTINE in [step.operation for step in item.official_steps]
    )
    outcome = rig.execute(workflow)
    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    assert rig.core.node["spec"]["unschedulable"] is True, rig.core.node
    assert rig.boundary.running is False, rig.boundary.calls
    assert Op.RESTART_WORKLOAD not in rig.boundary.calls, (
        "missing scope cannot authorize readmission"
    )
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, (
        "request identity review, not blind reset"
    )


@pytest.mark.parametrize(
    "values",
    [
        {"nvlink_crc_flit_error_total": 3},
        {"pcie_replay_total": 3},
        {
            "power_violation_total_us": 1_000_000,
            "power_usage_w": 490,
            "power_limit_w": 500,
            "gpu_utilization_percent": 95,
        },
        {"ecc_sbe_volatile_total": 1, "row_remap_correctable_total": 1},
    ],
    ids=["nvlink", "pcie", "power", "correctable-composite"],
)
def test_counter_diagnostics_never_dispatch_hardware_recovery(rig, values):
    baseline = {
        name: 0 if name.endswith("_total") or name.endswith("_us") else value
        for name, value in values.items()
    }
    rig.gpu({"gpu_temperature_c": 40, **baseline})
    rig.advance(15)
    rig.gpu({"gpu_temperature_c": 40, **values})
    workflows = rig.store.list_workflows()
    assert workflows, "the injected counter must reach the planner"
    for workflow in workflows:
        operations = [step.operation for step in workflow.official_steps]
        assert operations == [Op.FREEZE_EVIDENCE, Op.VALIDATE_GPU], operations
        rig.execute(workflow)
    assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
    assert rig.boundary.calls == [], "no reset, stress test, stop or reboot"


def test_incomplete_gpu_collection_refreshes_observation_without_hardware_ladder(rig):
    rig.gpu({"gpu_temperature_c": 40}, errors=["exporter incomplete"])
    rig.advance(181)
    rig.gpu({"gpu_temperature_c": 40}, errors=["exporter incomplete"])
    [workflow] = rig.store.list_workflows()
    assert [step.operation for step in workflow.official_steps] == [
        Op.FREEZE_EVIDENCE,
        Op.TRIGGER_HEALTH_SNAPSHOT,
        Op.VALIDATE_HOST,
    ]
    assert rig.execute(workflow).status is WorkflowStatus.RUNNING
    rig.advance(15)
    rig.gpu({"gpu_temperature_c": 40})
    outcome = rig.execute(workflow)
    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
    assert rig.boundary.calls == [Op.TRIGGER_HEALTH_SNAPSHOT], rig.boundary.calls


def test_nonfinite_loss_stops_only_its_attempt_and_does_not_restart_bad_checkpoint(rig):
    response = rig.post(
        "/v1/training-progress",
        {
            "cluster_id": "cluster-a",
            "attempt_id": "attempt-a",
            "rank": 0,
            "observed_at": rig.clock.isoformat(),
            "node_id": "node-a",
            "numerical_error": True,
            "step": 10,
            "checkpoint_ref": "s3://test/unchecked",
        },
    )
    assert response["findings"], response
    [workflow] = rig.store.list_workflows()
    outcome = rig.execute(workflow)
    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    assert rig.boundary.running is False, rig.boundary.calls
    assert Op.RESTART_WORKLOAD not in rig.boundary.calls, (
        "checkpoint is evidence, not authorization"
    )
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
    assert rig.core.node["spec"]["unschedulable"] is False, (
        "NaN does not prove bad hardware"
    )


def test_hung_validation_requires_new_application_progress_not_healthy_fabric(rig):
    def heartbeat(step):
        return rig.post(
            "/v1/training-progress",
            {
                "cluster_id": "cluster-a",
                "attempt_id": "attempt-a",
                "rank": 0,
                "observed_at": rig.clock.isoformat(),
                "node_id": "node-a",
                "step": step,
                "loss": 1.0,
            },
        )

    heartbeat(10)
    rig.advance(121)
    heartbeat(10)
    result = rig.post("/v1/training-health/cluster-a/scan", {})
    assert [item["metric_name"] for item in result["findings"]] == ["training_hang"]
    [workflow] = rig.store.list_workflows()
    rig.gpu({"gpu_temperature_c": 40})
    for _ in range(3):
        outcome = rig.execute(workflow)
    assert outcome.status is WorkflowStatus.RUNNING, outcome
    assert rig.boundary.running is True, rig.boundary.calls
    rig.advance(1)
    heartbeat(11)
    for _ in range(3):
        outcome = rig.execute(workflow)
        if outcome.status is WorkflowStatus.SUCCEEDED:
            break
    assert outcome.status is WorkflowStatus.SUCCEEDED, outcome
    assert Op.STOP_WORKLOADS not in rig.boundary.calls, rig.boundary.calls
