"""Execution-level regressions for the twelve revised health-policy categories."""

from __future__ import annotations

import json

import pytest

from gpu_fault.adapters.common import ANNOTATION_INCIDENT, quarantine_taint_value
from gpu_fault.models import (
    BlockedKind,
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    IncidentState,
    WorkflowStatus,
)
from gpu_fault.models import WorkflowOperation as Op
from tests._builders import attempt_observation, container_observation
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
        # ``efa_com_submit_admin_cmd()``: every admin command fails once the
        # queue is closed (gpu_fault.efa_kernel_log).
        "efa 0000:00:1d.0 efa_0: Admin queue is closed",
        # The same line on a live node: rdma-core's udev rule renamed the ib
        # device after its PCI location (``rdmap<bus>s<slot>``).
        "efa 0000:55:00.0 rdmap85s0: Admin queue is closed",
        # amzn-drivers 3.3.1g ``efa_com_cmd_exec()`` after an admin command
        # timed out: -ETIME is -62.
        "efa 0000:6e:00.0 rdmap110s0: Failed to process command GET_STATS "
        "(opcode 13) err -62",
        "kernel panic",
    ],
    ids=["mce", "storage", "rdma", "rdma-renamed", "rdma-timeout", "lockup"],
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


@pytest.mark.parametrize(
    "device",
    ["efa 0000:00:1d.0 efa_0", "efa 0000:55:00.0 rdmap85s0"],
    ids=["registered", "renamed"],
)
def test_degraded_efa_log_runs_read_only_fabric_diagnostics(rig, device):
    # ``efa_com_handle_single_admin_completion()``: the device answered with a
    # command id the driver never issued. The driver keeps running, so the
    # product looks at the fabric and hands the evidence over, nothing more --
    # under the registered name and the one rdma-core's udev rule gave it.
    result = rig.log(
        f"{device}: Received completion with unexpected command "
        "id[7], sq producer: 12, sq consumer: 11, cq consumer: 11"
    )
    [finding] = result["findings"]
    assert finding["category"] == "RDMA", finding
    assert finding["recommended_action"] == "RUN_DIAGNOSTICS", finding
    [workflow_id] = result["workflow_request_ids"]
    workflow = rig.store.get_workflow(workflow_id)
    operations = [step.operation for step in workflow.official_steps]
    assert operations[0] is Op.FREEZE_EVIDENCE, operations
    assert Op.VALIDATE_FABRIC in operations, operations
    assert not {Op.MARK_UNSCHEDULABLE, Op.QUARANTINE, Op.STOP_WORKLOADS} & set(
        operations
    ), operations
    assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
    assert rig.boundary.running is True, "the attempt keeps running"


@pytest.mark.parametrize("source", ["journal", "training-log"])
def test_untrusted_fatal_text_cannot_stop_training(rig, source):
    result = rig.log("mce: uncorrectable hardware error", source=source)
    assert result["workflow_request_ids"] == [], result
    assert rig.boundary.calls == [], "no mutation is dispatched"


@pytest.mark.parametrize(
    ("source", "fields"),
    [("journal", None), ("dmesg", {"_BOOT_ID": "boot-before"})],
    ids=["stdout-transport", "previous-boot"],
)
def test_untrusted_degraded_efa_text_opens_no_fabric_diagnostic(rig, source, fields):
    # The EFA driver's degraded wording is a kernel-only rule like its fatal
    # wording: a journal line another transport wrote, or a line of an earlier
    # boot, never reaches VALIDATE_FABRIC on the node and its job.
    result = rig.log(
        "efa 0000:55:00.0 rdmap85s0: Received completion with unexpected command "
        "id[7], sq producer: 12, sq consumer: 11, cq consumer: 11",
        source=source,
        fields=fields,
    )
    assert result["findings"] == [], result
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


def _observe(rig, *capabilities: CapabilityName) -> None:
    """Declare ``capabilities`` the way both live Runtime Profiles do:
    ``mode OBSERVE, owner gpu-fault-node-agent, adapter null``."""

    profile = rig.store.get_profile("simulated-v1")
    observed = set(capabilities)
    rig.store.save_profile(
        profile.model_copy(
            update={
                "capabilities": [
                    EffectiveCapability(
                        capability=item.capability,
                        mode=CapabilityMode.OBSERVE,
                        owner="gpu-fault-node-agent",
                    )
                    if item.capability in observed
                    else item
                    for item in profile.capabilities
                ]
            }
        )
    )


def test_row_remap_failure_without_a_field_diagnostic_owner_is_contained_first(rig):
    """GF-GENERAL-LIVE-019 ``remap_failure`` on a profile with
    ``memoryDiagnostics: OBSERVE`` (both live clusters, 2026-10-26): the RMA
    drain used to block at compile time for its one unowned step and leave the
    node schedulable. Now FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE and
    COLLECT_DIAGNOSTIC_BUNDLE execute, the record settles
    BLOCKED/SAFETY_SETTLED at the diagnostic step, the incident is QUARANTINED
    with the hand-off in its reasons, and one operator notice goes out."""

    _observe(rig, CapabilityName.MEMORY_DIAGNOSTICS)
    result = rig.gpu({"gpu_temperature_c": 40, "row_remap_failure": 1})
    [workflow] = rig.store.list_workflows()
    incident = rig.store.get_incident(workflow.incident_id)
    assert incident.official_action == "RUN_FIELD_DIAGNOSTIC_FOR_RMA", result
    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.blocked_reasons == ["no executable owner for memoryDiagnostics"]
    assert [step.operation for step in workflow.safety_steps] == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    assert incident.state is IncidentState.SAFETY_PENDING

    outcome = rig.execute(workflow)

    assert outcome.status is WorkflowStatus.BLOCKED, outcome
    settled = rig.store.get_workflow(workflow.request_id)
    assert settled.blocked_kind is BlockedKind.SAFETY_SETTLED
    assert settled.completed_operations == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    assert rig.boundary.calls == [Op.COLLECT_DIAGNOSTIC_BUNDLE], rig.boundary.calls
    assert rig.boundary.running is True, "the containment prefix stops no job"
    node = rig.core.node
    assert node["spec"]["unschedulable"] is True, node
    assert {(taint["key"], taint["value"]) for taint in node["spec"]["taints"]} >= {
        ("gpu-fault.io/quarantined", quarantine_taint_value(incident.incident_id))
    }
    assert node["metadata"]["annotations"][ANNOTATION_INCIDENT] == incident.incident_id
    closed = rig.store.get_incident(incident.incident_id)
    assert closed.state is IncidentState.QUARANTINED
    assert any(
        "contained (cordoned and quarantined)" in reason
        and "no executable owner for memoryDiagnostics" in reason
        for reason in closed.reasons
    ), closed.reasons
    [notice] = [
        item
        for item in rig.store.list_notifications()
        if "contained-operator-handoff" in item.deduplication_key
    ]
    assert notice.category == "OPERATOR_REVIEW"
    assert "RUN_FIELD_DIAGNOSTIC" in notice.body_text
    assert "--disposition restore" in notice.body_text


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


def test_straggler_across_two_pods_binds_to_the_lagging_ranks_node_only(rig):
    """GF-GENERAL-LIVE-027 ``straggler_group`` as live (HyperPod 2026-10-26
    17:43Z): one attempt, two Pods on two nodes, both ranks in one
    ``progress_group``, rank 1 60 steps behind rank 0 and both advancing every
    poll. The product grades the pair across the Pods and flags the lagging
    rank only; the incident names that rank's node alone and plans the
    diagnostic chain without a stop, so a records read scoped to the reference
    rank's node sees nothing while one scoped to both nodes sees the incident.
    """

    nodes = {0: "node-a", 1: "node-b"}
    pods = {0: "pod-master-0", 1: "pod-worker-0"}

    def observe():
        rig.store.save_attempt_observation(
            attempt_observation(
                "strag",
                "attempt-strag",
                rig.clock,
                expected_critical_ranks=2,
                workload_ids=["training/job/strag"],
                containers=[
                    container_observation(
                        pods[rank],
                        pods[rank],
                        rank,
                        nodes[rank],
                        container_name="pytorch",
                        role="master" if rank == 0 else "worker",
                        gpu_uuids=[f"GPU-{rank}-{index}" for index in range(8)],
                    )
                    for rank in (0, 1)
                ],
            )
        )

    def heartbeat(rank, step):
        return rig.post(
            "/v1/training-progress",
            {
                "cluster_id": "cluster-a",
                "attempt_id": "attempt-strag",
                "rank": rank,
                "observed_at": rig.clock.isoformat(),
                "node_id": nodes[rank],
                "pod_uid": pods[rank],
                "container_name": "pytorch",
                "step": step,
                "loss": 1.0,
                "labels": {"progress_group": "data-parallel"},
            },
        )

    steps = {0: 100, 1: 40}
    findings = []
    for _ in range(3):
        observe()
        for rank in (0, 1):
            assert heartbeat(rank, steps[rank])["accepted"] is True
            steps[rank] += 10
        findings.extend(rig.post("/v1/training-health/cluster-a/scan", {})["findings"])
        rig.advance(41)

    assert [
        (item["metric_name"], item["node_id"], item["device"]) for item in findings
    ] == [("training_straggler", "node-b", "rank-1")], (
        "one straggler finding, on the lagging rank's node, emitted once"
    )
    incident = rig.store.get_incident_by_event(findings[0]["event_id"])
    assert incident is not None
    assert incident.node_ids == ["node-b"], "the incident names rank 1's node only"
    assert incident.attempt_id == "attempt-strag"
    operations = [
        step.operation
        for step in rig.store.get_workflow(incident.workflow_request_id).official_steps
    ]
    assert Op.FREEZE_EVIDENCE in operations and Op.VALIDATE_FABRIC in operations, (
        operations
    )
    assert Op.STOP_WORKLOADS not in operations, operations
    states = list(IncidentState)
    assert (
        rig.store.list_incidents_by_state("cluster-a", states, node_ids={"node-a"})
        == []
    ), "a read scoped to the reference rank's node lists no straggler"
    assert [
        item.incident_id
        for item in rig.store.list_incidents_by_state(
            "cluster-a", states, node_ids={"node-a", "node-b"}
        )
    ] == [incident.incident_id]
