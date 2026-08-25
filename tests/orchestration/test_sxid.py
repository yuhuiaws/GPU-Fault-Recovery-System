from __future__ import annotations

from tests._builders import build_sxid_event, copy_model

from ._support import (
    NOW,
    ActionDisposition,
    ApplicationContext,
    Environment,
    IncidentOrchestrator,
    IncidentState,
    SxidClassification,
    SxidLinkScope,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
    event,
)


def test_fatal_trunk_sxid_compiles_full_fabric_reset_workflow(
    context: ApplicationContext,
) -> None:
    sxid_event = build_sxid_event(
        "fatal-trunk-sxid",
        NOW,
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        product="H200",
        fabric_partition="cluster-a/node-a/local-nvswitch",
        participating_gpu_uuids=["GPU-a", "GPU-b"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train-a"],
    )
    decision = context.policy.evaluate_sxid(sxid_event)

    incident, workflow = context.orchestrator.ingest(sxid_event, decision)

    assert incident.state is IncidentState.ACTION_PENDING
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    reset = workflow.official_steps[6]
    assert reset.gpu_uuids == ["GPU-a", "GPU-b"]
    assert reset.parameters == {
        "fabric_partition": "cluster-a/node-a/local-nvswitch",
        "sxid": 11001,
    }
    assert WorkflowOperation.REMEDIATE_DRIVER not in {
        step.operation for step in workflow.official_steps
    }
    assert WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE not in {
        step.operation for step in workflow.official_steps
    }


def test_sxid_site_policy_adds_version_pinned_remediation(
    context: ApplicationContext,
) -> None:
    orchestrator = IncidentOrchestrator(
        context.store,
        multi_node_aggregation_window_seconds=0,
        sxid_driver_remediation_codes={11001},
        sxid_firmware_update_codes={11001},
        target_driver_branch=575,
        target_firmware_version="92.10.14",
    )
    sxid_event = build_sxid_event(
        "fatal-trunk-remediation",
        NOW,
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        product="H200",
        fabric_partition="cluster-a/node-a/local-nvswitch",
        participating_gpu_uuids=["GPU-a", "GPU-b"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train-a"],
    )

    _, workflow = orchestrator.ingest(
        sxid_event, context.policy.evaluate_sxid(sxid_event)
    )

    operations = [step.operation for step in workflow.official_steps]
    verify_index = operations.index(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
    reset_index = operations.index(WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES)
    assert operations[verify_index + 1 : reset_index] == [
        WorkflowOperation.REMEDIATE_DRIVER,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    ]
    driver = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.REMEDIATE_DRIVER
    )
    firmware = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE
    )
    assert driver.parameters == {"target_driver_branch": 575}
    assert firmware.parameters == {"target_firmware_version": "92.10.14"}


def test_hyperpod_virtualization_only_sxid_does_not_reboot_node(
    context: ApplicationContext,
) -> None:
    base = context.store.get_profile("simulated-v1")
    profile = copy_model(
        base,
        environment=Environment.HYPERPOD_EKS,
        profile_version="hyperpod-sxid-vm-only-v1",
    )
    context.store.save_profile(profile)
    sxid_event = build_sxid_event(
        "sxid-11004-hyperpod-bare-metal",
        NOW,
        11004,
        SxidClassification.NON_FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        product="H200",
        runtime_profile_version=profile.profile_version,
    )

    decision = context.policy.evaluate_sxid(sxid_event)
    incident, workflow = context.orchestrator.ingest(sxid_event, decision)

    assert decision.disposition is ActionDisposition.BLOCKED_WORKFLOW
    assert decision.official_action == "RESTART_VM"
    assert incident.effective_action is None
    assert workflow.status is WorkflowStatus.SAFETY_PENDING
    assert WorkflowOperation.RESTART_NODE not in {
        step.operation for step in workflow.official_steps
    }


def test_sxid_20012_mechanical_step_preserves_event_identity(
    context: ApplicationContext,
) -> None:
    sxid_event = build_sxid_event(
        "sxid-20012-mechanical",
        NOW,
        20012,
        SxidClassification.NON_FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        switch_id="3",
        port="46",
        pci_bdf="0000:c1:00.0",
        product="H200",
        runtime_profile_version="simulated-v1",
    )

    decision = context.policy.evaluate_sxid(sxid_event)
    _, workflow = context.orchestrator.ingest(sxid_event, decision)

    mechanical = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.CHECK_MECHANICALS
    )
    assert mechanical.parameters == {
        "sxid": 20012,
        "xid": 20012,
        "switch_id": "3",
        "port": "46",
        "pci_bdf": "0000:c1:00.0",
    }


def test_solo_xid45_creates_owned_fabric_manager_restart(
    context: ApplicationContext,
) -> None:
    xid_event = event(
        45,
        event_id="solo-xid45",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    decision = context.policy.evaluate_xid(xid_event, xid45_window_closed=True)
    incident, workflow = context.orchestrator.ingest(xid_event, decision)

    assert decision.disposition.value == "EXECUTABLE"
    assert decision.official_action == "RESTART_FM"
    assert incident.state is IncidentState.ACTION_PENDING
    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
    ]
