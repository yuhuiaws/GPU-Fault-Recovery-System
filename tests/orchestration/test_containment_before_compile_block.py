"""A plan refused only for a missing owner still runs its containment first.

Live 2026-10-26 06:25Z (gpu-eks-cluster, ip-10-0-43-5, R13g; both clusters'
profiles declare ``memoryDiagnostics: OBSERVE``): GF-GENERAL-LIVE-019
``remap_failure`` replayed ``row_remap_failure=1`` on one H200. The node-health
family compiled the RMA drain -- FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE ->
QUARANTINE -> COLLECT_DIAGNOSTIC_BUNDLE -> RUN_FIELD_DIAGNOSTIC -> VALIDATE_GPU
-> ESCALATE_SUPPORT -- and, because RUN_FIELD_DIAGNOSTIC had no executable
owner, wrote the whole record BLOCKED/NEEDS_OPERATOR at compile time: nothing
ran, the dispatcher sweep closed it as a record "that never changed a node",
the incident went ESCALATED and a CRITICAL row-remap failure left the node
fully schedulable with no evidence frozen.

The rule now (the fault family's safety plan, applied to the node-health and
node-lifecycle families): when every compile error is a missing executable
owner and the owned steps before the first unowned one contain an isolation,
the record is SAFETY_PENDING with that containment-and-evidence prefix as its
``safety_steps``; the executor runs it and settles BLOCKED/SAFETY_SETTLED with
the incident QUARANTINED, which is the documented operator hand-off (restore or
close with node evidence). A plan whose first step is unowned, a plan refused
for any other reason (UNKNOWN workload state, ARCH-B2), a containment-only
plan and a hung-triage DAG keep today's behaviour.
"""

from __future__ import annotations

import pytest

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    BlockedKind,
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration.workflow_builder import (
    CONTAINMENT_PREFIX_OPERATIONS,
    MISSING_OWNER_REASON,
    blocked_only_by_missing_owners,
    missing_owner_capabilities,
)
from gpu_fault.workflow_quarantine import TERMINAL_QUARANTINE_NODES
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import finding, memory_store

OBSERVE_PROFILE = "observe-v1"
MEMORY_REASON = f"{MISSING_OWNER_REASON}memoryDiagnostics"


def observe_profile(
    store, *capabilities: CapabilityName, version: str = OBSERVE_PROFILE
):
    """The simulated profile with ``capabilities`` declared the live way:
    ``mode OBSERVE, owner gpu-fault-node-agent, adapter null``."""

    default = store.get_profile("simulated-v1")
    observed = set(capabilities)
    profile = default.model_copy(
        update={
            "profile_version": version,
            "capabilities": [
                EffectiveCapability(
                    capability=item.capability,
                    mode=CapabilityMode.OBSERVE,
                    owner="gpu-fault-node-agent",
                )
                if item.capability in observed
                else item
                for item in default.capabilities
            ],
        }
    )
    store.save_profile(profile)
    return profile


def remap_failure(workload_state: WorkloadState = WorkloadState.IDLE, **updates):
    """The live finding: ``row_remap_failure=1`` on a modern GPU, CRITICAL
    DRAIN with ``RUN_FIELD_DIAGNOSTIC_FOR_RMA``."""

    values = {
        "category": NodeHealthCategory.GPU,
        "severity": "critical",
        "reason": "row remap failure; field diagnosis required before any RMA conclusion",
        "recommended_action": RecoveryAction.DRAIN,
        "official_action": "RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        "metric_name": "row_remap_failure",
        "gpu_uuids": ["GPU-0"],
        "pci_bdf": "0000:81:00.0",
        "runtime_profile_version": OBSERVE_PROFILE,
        "workload_state": workload_state,
        "node_id": "node-0",
        **updates,
    }
    if workload_state is WorkloadState.ACTIVE:
        values.setdefault("affected_workload_ids", ["training/job/unit-job"])
    return finding("gpu-dcgm-node-0-GPU-0-row_remap_failure", **values)


def test_the_containment_prefix_is_the_fault_family_safety_set_plus_read_only_evidence():
    assert CONTAINMENT_PREFIX_OPERATIONS == frozenset(
        {
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        }
    ), "STOP_WORKLOADS, RESTORE_SCHEDULING and every node-mutating operation stay out"


def test_missing_owner_reasons_are_recognised_by_their_compiler_spelling():
    reasons = [MEMORY_REASON, f"{MISSING_OWNER_REASON}nvlinkDiagnostics"]
    assert missing_owner_capabilities(reasons) == [
        "memoryDiagnostics",
        "nvlinkDiagnostics",
    ]
    assert blocked_only_by_missing_owners(reasons) is True
    assert blocked_only_by_missing_owners([]) is False
    assert (
        blocked_only_by_missing_owners(
            [MEMORY_REASON, "node workload state is UNKNOWN"]
        )
        is False
    ), "any other refusal keeps the compile-time block (ARCH-B2 intact)"


def test_row_remap_failure_without_a_field_diagnostic_owner_contains_before_the_block():
    """The live shape: containment compiles, the diagnostic does not."""

    store = memory_store()
    observe_profile(store, CapabilityName.MEMORY_DIAGNOSTICS)
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(remap_failure())

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.safety_only is True
    assert workflow.blocked_kind is None
    assert workflow.blocked_reasons == [MEMORY_REASON]
    assert [step.operation for step in workflow.safety_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    quarantine = workflow.safety_steps[2]
    assert quarantine.parameters[TERMINAL_QUARANTINE_NODES] == ["node-0"], (
        "the DRAIN's terminal quarantine travels with the containment prefix"
    )
    assert all(step.node_ids == ["node-0"] for step in workflow.safety_steps), (
        "the prefix targets the finding's node"
    )
    # The owned remainder stays on the record for the operator to read; the
    # unowned diagnostic is named by ``blocked_reasons``, not compiled.
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.ESCALATE_SUPPORT,
    ]
    assert incident.state is IncidentState.SAFETY_PENDING
    assert incident.workflow_request_id == workflow.request_id
    assert store.get_workflow(workflow.request_id) == workflow
    handoff = [reason for reason in incident.reasons if "memoryDiagnostics" in reason]
    assert handoff, incident.reasons
    assert "MARK_UNSCHEDULABLE" in handoff[0] and "QUARANTINE" in handoff[0]
    assert "RUN_FIELD_DIAGNOSTIC" in handoff[0]


def test_an_active_workload_is_not_stopped_by_the_containment_prefix():
    """Containment, not remediation: the stop is a decision the operator now
    owns, as in the fault family's safety plan."""

    store = memory_store()
    observe_profile(store, CapabilityName.MEMORY_DIAGNOSTICS)
    service, _absorbed = health_service(store)

    _, workflow = service.ingest(remap_failure(WorkloadState.ACTIVE))

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert WorkflowOperation.STOP_WORKLOADS in [
        step.operation for step in workflow.official_steps
    ]
    assert [step.operation for step in workflow.safety_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
    ]


def test_a_plan_whose_first_step_is_unowned_keeps_the_compile_time_block():
    store = memory_store()
    observe_profile(
        store, CapabilityName.EVIDENCE_CAPTURE, CapabilityName.MEMORY_DIAGNOSTICS
    )
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(remap_failure())

    assert workflow.status is WorkflowStatus.BLOCKED
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow.safety_steps == []
    assert f"{MISSING_OWNER_REASON}evidenceCapture" in workflow.blocked_reasons
    assert incident.state is IncidentState.ESCALATED


def test_a_plan_with_every_owner_compiles_pending_as_before():
    store = memory_store()
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(
        remap_failure(runtime_profile_version="simulated-v1")
    )

    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.safety_steps == []
    assert workflow.blocked_reasons == []
    assert WorkflowOperation.RUN_FIELD_DIAGNOSTIC in [
        step.operation for step in workflow.official_steps
    ]
    assert incident.state is IncidentState.ACTION_PENDING


def test_a_containment_only_plan_is_unchanged_by_the_rule():
    store = memory_store()
    observe_profile(store, CapabilityName.MEMORY_DIAGNOSTICS)
    service, _absorbed = health_service(store)

    _, workflow = service.ingest(
        finding(
            "gpu-dcgm-node-0-GPU-0-ecc_dbe_volatile_total",
            category=NodeHealthCategory.GPU,
            severity="critical",
            reason="uncorrectable ECC observed",
            recommended_action=RecoveryAction.QUARANTINE,
            metric_name="ecc_dbe_volatile_total",
            gpu_uuids=["GPU-0"],
            runtime_profile_version=OBSERVE_PROFILE,
            node_id="node-0",
        )
    )

    assert workflow.status is WorkflowStatus.PENDING
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
    ]
    assert workflow.safety_steps == []


def test_an_unknown_workload_state_still_refuses_a_node_mutating_plan():
    """ARCH-B2: a node-mutating plan refused for the UNKNOWN workload state is
    not a missing-owner refusal; nothing runs and the record is replaced in
    place once the state is known (C-03)."""

    store = memory_store()
    observe_profile(store, CapabilityName.NODE_REPLACE)
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(
        finding(
            "unit-replace-unknown",
            category=NodeHealthCategory.GPU,
            severity="critical",
            reason="unit replacement",
            recommended_action=RecoveryAction.REPLACE_NODE,
            gpu_uuids=["GPU-0"],
            runtime_profile_version=OBSERVE_PROFILE,
            workload_state=WorkloadState.UNKNOWN,
            node_id="node-0",
        )
    )

    assert workflow.status is WorkflowStatus.BLOCKED
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow.safety_steps == []
    assert "node workload state is UNKNOWN" in workflow.blocked_reasons
    assert f"{MISSING_OWNER_REASON}nodeReplace" in workflow.blocked_reasons
    assert incident.state is IncidentState.ESCALATED


def test_the_rma_drain_is_not_a_node_mutating_plan_so_unknown_state_contains():
    """The DRAIN chain cordons, quarantines and collects evidence; it mutates
    nothing ARCH-B2 protects, so -- exactly as it compiles PENDING with the
    owner present -- an UNKNOWN workload state does not keep it from containing."""

    store = memory_store()
    observe_profile(store, CapabilityName.MEMORY_DIAGNOSTICS)
    service, _absorbed = health_service(store)

    _, workflow = service.ingest(remap_failure(WorkloadState.UNKNOWN))

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.blocked_reasons == [MEMORY_REASON]


@pytest.mark.parametrize(
    ("capability", "action", "unowned", "prefix"),
    [
        pytest.param(
            CapabilityName.NODE_REPLACE,
            RecoveryAction.REPLACE_NODE,
            WorkflowOperation.REPLACE_NODE,
            [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ],
            id="nodeReplace-observe",
        ),
        pytest.param(
            CapabilityName.NODE_REBOOT,
            RecoveryAction.REBOOT_NODE,
            WorkflowOperation.RESTART_NODE,
            [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ],
            id="nodeReboot-observe",
        ),
        pytest.param(
            CapabilityName.EFA_DRIVER_REMEDIATION,
            RecoveryAction.REMEDIATE_EFA_DRIVER,
            WorkflowOperation.REMEDIATE_EFA_DRIVER,
            [WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.MARK_UNSCHEDULABLE],
            id="efaDriverRemediation-observe (2026-09-04 live shape)",
        ),
    ],
)
def test_every_unowned_capability_behind_containment_contains_first(
    capability, action, unowned, prefix
):
    """Generic: the rule keys on the compiler's refusal, not on one capability."""

    store = memory_store()
    observe_profile(store, capability)
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(
        finding(
            f"unit-{capability.value}",
            category=NodeHealthCategory.GPU,
            severity="critical",
            reason=f"unit {capability.value}",
            recommended_action=action,
            gpu_uuids=["GPU-0"],
            runtime_profile_version=OBSERVE_PROFILE,
            node_id="node-0",
        )
    )

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.blocked_reasons == [f"{MISSING_OWNER_REASON}{capability.value}"]
    assert [step.operation for step in workflow.safety_steps] == prefix
    assert unowned not in [step.operation for step in workflow.official_steps]
    assert incident.state is IncidentState.SAFETY_PENDING


def test_the_node_lifecycle_family_contains_before_an_unowned_replacement(
    context,
) -> None:
    """The grouped replacement path (``families/node_lifecycle.py``) has the
    same shape: a profile that keeps ``nodeReplace`` OBSERVE (the HyperPod
    template's documented "reboot approved, replacement not yet" form) used to
    block the whole plan at compile time with the node schedulable."""

    from tests._builders import node_health_finding
    from tests.orchestration._support import NOW

    observe_profile(context.store, CapabilityName.NODE_REPLACE)
    incident, workflow = context.orchestrator.ingest_node_health(
        node_health_finding(
            "finding-replace-observe",
            "replace-observe",
            observed_at=NOW,
            category=NodeHealthCategory.GPU,
            severity="critical",
            reason="repeated hardware fault after remediation",
            recommended_action=RecoveryAction.REPLACE_NODE,
            runtime_profile_version=OBSERVE_PROFILE,
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=["training/job/job-a"],
        )
    )

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.blocked_reasons == [f"{MISSING_OWNER_REASON}nodeReplace"]
    assert [step.operation for step in workflow.safety_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
    ], "STOP_WORKLOADS is the operator's decision now, not the prefix's"
    assert WorkflowOperation.REPLACE_NODE not in [
        step.operation for step in workflow.official_steps
    ]
    assert incident.state is IncidentState.SAFETY_PENDING
    assert any("nodeReplace" in reason for reason in incident.reasons), incident.reasons
