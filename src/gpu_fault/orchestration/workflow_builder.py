from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timezone

from gpu_fault.env import env_bool
from gpu_fault.gpu_fabric_manager import graded_fabric_gpus
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    FaultIncident,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStepSpec,
    WorkloadState,
)
from gpu_fault.operation_registry import NODE_MUTATING_OPERATIONS
from gpu_fault.orchestration.xid45_policy import observe_only, solo_restart_fm
from gpu_fault.policy import (
    NVIDIA_CODE_SPECIFIC_FULL_RESET_SXIDS,
    ActionDisposition,
    FaultPolicyDecision,
    SxidClassification,
    SxidEvent,
    XidEvent,
)

# The blocked reason both families write when a node-mutating plan meets an
# UNKNOWN workload state; the fault family has spelt it this way since the
# coordinator first carried the gate, so operators and tests match on it.
WORKLOAD_STATE_UNKNOWN_REASON = "node workload state is UNKNOWN"
#: The catalog's "Solo: RESTART_FM" (XID 45).
RESTART_FABRIC_MANAGER_ACTION = "RESTART_FM"


def parse_sxid_codes(name: str) -> set[int]:
    """The positive SXID codes in the comma-separated environment variable."""

    values = set()
    for item in os.getenv(name, "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            code = int(item)
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated integers") from exc
        if code <= 0:
            raise ValueError(f"{name} values must be positive")
        values.add(code)
    return values


def _xid74_operations(
    event: XidEvent, decision: FaultPolicyDecision
) -> list[WorkflowOperation]:
    operations = [WorkflowOperation.FREEZE_EVIDENCE]
    categories = {item.rsplit(":", 1)[1] for item in decision.matched_decode_rules}
    if "fabric_reset_required" in categories:
        operations.extend(
            [
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.ESCALATE_SUPPORT,
            ]
        )
        return operations
    persistent_mechanical = "mechanical_or_hardware" in categories and any(
        count >= 2
        for key, count in (decision.nvlink_occurrence_counts.items())
        if any(
            rule.startswith(f"{key}:") and rule.endswith(":mechanical_or_hardware")
            for rule in decision.matched_decode_rules
        )
    )
    repeated_reset_required = decision.action is RecoveryAction.RESET_GPU and bool(
        categories.intersection(
            {
                "ecc_parity",
                "field_diag_if_repeated",
                "report_if_repeated",
            }
        )
    )
    field_diagnostic_required = (
        bool(
            categories.intersection(
                {
                    "marginal_channel",
                    "field_diag_if_repeated",
                }
            )
        )
        or persistent_mechanical
        or repeated_reset_required
    )
    mechanical_required = bool(
        categories.intersection(
            {
                "mechanical_or_hardware",
                "marginal_channel",
            }
        )
    )
    if mechanical_required or field_diagnostic_required:
        operations.append(WorkflowOperation.MARK_UNSCHEDULABLE)
        if event.workload_state is WorkloadState.ACTIVE:
            if event.checkpoint_manifest_ref:
                operations.append(WorkflowOperation.CHECKPOINT_WORKLOADS)
            operations.append(WorkflowOperation.STOP_WORKLOADS)
    if field_diagnostic_required:
        operations.append(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE)
    if mechanical_required or field_diagnostic_required:
        operations.extend(
            [
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            ]
        )
        if field_diagnostic_required:
            operations.append(WorkflowOperation.RUN_NVLINK74_WORKFLOW)
        operations.extend(
            [
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTORE_GPU_SERVICES,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowOperation.RESTORE_SCHEDULING,
            ]
        )
    if (
        decision.action is RecoveryAction.ESCALATE_OPERATOR
        and decision.investigatory_action == "CONTACT_SUPPORT"
    ):
        if mechanical_required or field_diagnostic_required:
            operations.append(WorkflowOperation.QUARANTINE)
        operations.append(WorkflowOperation.ESCALATE_SUPPORT)
    if (
        mechanical_required or field_diagnostic_required
    ) and event.affected_workload_ids:
        operations.append(WorkflowOperation.RESTART_WORKLOAD)
    return list(dict.fromkeys(operations))


class WorkflowBuilder:
    def __init__(
        self,
        store,
        *,
        target_driver_branch,
        target_firmware_version,
        sxid_driver_remediation_codes,
        sxid_firmware_update_codes,
        operation_capability,
        official_workflow_operation,
        pre_action_order,
        pre_action_operation,
        xid45_solo_restart_fm: bool | None = None,
    ) -> None:
        self.store = store
        self.target_driver_branch = target_driver_branch
        self.target_firmware_version = target_firmware_version
        self.sxid_driver_remediation_codes = sxid_driver_remediation_codes
        self.sxid_firmware_update_codes = sxid_firmware_update_codes
        self.operation_capability = operation_capability
        self.official_workflow_operation = official_workflow_operation
        self.pre_action_order = pre_action_order
        self.pre_action_operation = pre_action_operation
        # The site switch of ``xid45_policy``: True runs the catalog's RESTART_FM
        # on a graded NVSwitch node; unset (the default) is observe-only there.
        self.xid45_solo_restart_fm = (
            env_bool("GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM", False)
            if xid45_solo_restart_fm is None
            else xid45_solo_restart_fm
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def sxid_remediation_operations(
        self, event: XidEvent | SxidEvent
    ) -> list[WorkflowOperation]:
        if not isinstance(event, SxidEvent):
            return []
        operations = []
        if event.sxid in self.sxid_driver_remediation_codes:
            operations.append(WorkflowOperation.REMEDIATE_DRIVER)
        if event.sxid in self.sxid_firmware_update_codes:
            operations.append(WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE)
        return operations

    def requires_sxid_diagnostics(
        self,
        event: XidEvent | SxidEvent,
    ) -> bool:
        return isinstance(event, SxidEvent) and (
            event.classification
            in {
                SxidClassification.FATAL,
                SxidClassification.ALWAYS_FATAL,
            }
            or event.sxid in NVIDIA_CODE_SPECIFIC_FULL_RESET_SXIDS
        )

    def fabric_manager_restart_inventory(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> list[str]:
        """The node's GPUs when its RESTART_FM is the full GPU + NVSwitch reset.

        On a graded NVSwitch node (Hopper SXM, HGX B200/B300) a Fabric Manager
        restart alone loses the fabric state: NVIDIA's Fabric Manager user guide
        has the restarted service need every GPU of the NVLink fabric reset, and
        its H100-and-later procedure is stop the CUDA applications and GPU
        services, stop Fabric Manager, ``nvidia-smi -r``, start Fabric Manager.
        So the plan becomes the fatal-trunk SXID's chain on these GPUs
        (``catalog_operations``). A node is such a node when its latest samples
        are graded the way batch 127 grades the fabric (``graded_fabric_gpus``);
        empty -- any other node, or one whose exporter reports no field 170 --
        keeps RESTART_FABRIC_MANAGER.
        A solo XID 45 reaches the chain only when the site opted in: by default
        it is observe-only there (``xid45_site_policy``).
        """

        if decision.official_action != RESTART_FABRIC_MANAGER_ACTION or not isinstance(
            event, XidEvent
        ):
            return []
        latest = self.store.list_gpu_metrics_latest(event.cluster_id, event.node_id)
        return graded_fabric_gpus(
            ((item.sample, item.observed_at) for item in latest),
            product=event.product,
        )

    def xid45_site_policy(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> FaultPolicyDecision:
        """A solo XID 45 on a graded NVSwitch node: observe-only by default.

        ``xid45_policy`` holds the NVIDIA evidence. Unchanged: the site opted in
        (``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM``), any other decision, and
        a node whose latest samples are not graded (A100, PCIe/NVL, GH200,
        GB200/GB300, no field 170), where RESTART_FM is the Fabric Manager
        restart. Idempotent: the observe-only decision is not solo RESTART_FM.
        """

        if self.xid45_solo_restart_fm or not solo_restart_fm(event, decision):
            return decision
        graded = self.fabric_manager_restart_inventory(event, decision)
        return observe_only(decision, graded) if graded else decision

    def workflow_evidence_errors(
        self,
        event: XidEvent | SxidEvent,
        operations: list[WorkflowOperation],
        fabric_inventory: Sequence[str] = (),
    ) -> list[str]:
        """Why the plan lacks the evidence it acts on; ``fabric_inventory`` is
        the graded inventory a Fabric Manager restart resets (it stands in for
        the SXID's partition and participating GPUs)."""

        errors = []
        if (
            WorkflowOperation.RUN_NVLINK74_WORKFLOW in operations
            and isinstance(event, XidEvent)
            and (len(event.registers) != 7 or event.nvlink_link_id is None)
        ):
            errors.append(
                "XID 74 Field Diagnostic requires all seven register "
                "fields and an explicit NVLink identity"
            )
        full_reset = (
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in operations
            and not fabric_inventory
        )
        if full_reset and not event.fabric_partition:
            errors.append("full fabric reset requires fabric_partition mapping")
        if full_reset and (
            not isinstance(event, SxidEvent) or not event.participating_gpu_uuids
        ):
            errors.append("full fabric reset requires complete node GPU inventory")
        if WorkflowOperation.RESET_GPU in operations and (
            (isinstance(event, XidEvent) and not event.gpu_uuid)
            or (isinstance(event, SxidEvent) and not event.participating_gpu_uuids)
        ):
            if event.pci_bdf:
                errors.append(
                    "RESET_GPU requires an explicit GPU UUID; "
                    f"PCI BDF {event.pci_bdf} could not be resolved "
                    "from fresh DCGM GPU inventory. Check the DCGM "
                    "collector and retry after identity telemetry is "
                    "available"
                )
            else:
                errors.append("RESET_GPU requires an explicit GPU UUID")
        if (
            WorkflowOperation.RESTART_VM in operations
            and not event.affected_workload_ids
        ):
            errors.append("VM restart requires affected workload/VM ownership")
        workload_ids = list(event.affected_workload_ids)
        if isinstance(event, XidEvent) and event.job_id:
            workload_ids.append(event.job_id)
        if WorkflowOperation.RESTART_WORKLOAD in operations and not workload_ids:
            errors.append("workload restart requires an affected workload or job ID")
        if WorkflowOperation.STOP_WORKLOADS in operations and not workload_ids:
            errors.append("workload stop requires an affected workload or job ID")
        if (
            WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE in operations
            and not self.target_firmware_version
        ):
            errors.append("UPDATE_SWFW requires GPU_FAULT_TARGET_FIRMWARE_VERSION")
        return errors

    def pre_action_index(
        self,
        operations: list[WorkflowOperation],
        operation: WorkflowOperation,
    ) -> int:
        """Place containment after evidence but before remediation."""
        anchors = [WorkflowOperation.FREEZE_EVIDENCE]
        if operation is WorkflowOperation.STOP_WORKLOADS:
            anchors.extend(
                [
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    WorkflowOperation.CHECKPOINT_WORKLOADS,
                ]
            )
        index = 0
        for anchor in anchors:
            if anchor in operations:
                index = max(index, operations.index(anchor) + 1)
        return index

    def owner(self, profile, capability: CapabilityName) -> str | None:
        if profile is None:
            return None
        for item in profile.capabilities:
            if item.capability is capability and item.mode in {
                CapabilityMode.OWN,
                CapabilityMode.DELEGATE,
            }:
                return item.owner
        return None

    _RESTART_OBSERVATION_SCAN_LIMIT = 256

    def restart_step_parameters(
        self,
        cluster_id: str,
        workload_ids: list[str],
        *,
        job_id: str | None = None,
        fallback_attempt_id: str = "unknown",
        observed_at: datetime | None = None,
        node_id: str | None = None,
        fallback_gpu_uuids: list[str] | None = None,
    ) -> dict[str, object]:
        workload_set = set(workload_ids)
        # Newest attempts only: this runs inside the merge lock and the
        # window below is two minutes wide, so a cluster's whole observation
        # history has nothing to add (F-B8).
        observations = [
            state.observation
            for state in self.store.list_attempt_observation_states(
                cluster_id,
                limit=self._RESTART_OBSERVATION_SCAN_LIMIT,
                newest_first=True,
            )
        ]
        if observed_at is not None:
            observations = [
                observation
                for observation in observations
                if (
                    -30
                    <= (
                        self._utc(observed_at) - self._utc(observation.observed_at)
                    ).total_seconds()
                    <= 120
                    and observation.workload_phase.value in {"PENDING", "RUNNING"}
                    and (
                        node_id is None
                        or any(
                            container.node_id == node_id and not container.terminated
                            for container in observation.containers
                        )
                    )
                )
            ]
        if job_id:
            candidates = [
                observation
                for observation in observations
                if observation.job_id == job_id
            ]
        else:
            candidates = [
                observation
                for observation in observations
                if workload_set.intersection(observation.workload_ids)
            ]
        latest = (
            max(
                candidates,
                key=lambda item: item.observed_at,
            )
            if candidates
            else None
        )
        resolved_job_id = latest.job_id if latest is not None else job_id
        if not resolved_job_id:
            resolved_job_id = sorted(workload_set)[0] if workload_set else "unknown"
        source_observations = [
            observation
            for observation in candidates
            if latest is not None and observation.attempt_id == latest.attempt_id
        ]
        gpu_uuids = {
            gpu_uuid
            for observation in source_observations
            for container in observation.containers
            for gpu_uuid in container.gpu_uuids
        }
        if not gpu_uuids and fallback_gpu_uuids:
            # No attempt observation reached the store (a node-level
            # fault can be detected before any is recorded), but the
            # detector already named the GPUs. Without this the count
            # stays 0 and the guard falls back to a store lookup that
            # the regional executor cannot perform.
            gpu_uuids = set(fallback_gpu_uuids)
        return {
            "cluster_id": cluster_id,
            "job_id": resolved_job_id,
            "source_attempt_id": (
                latest.attempt_id if latest is not None else fallback_attempt_id
            ),
            "source_gpu_count": len(gpu_uuids),
            "restart_budget": (
                latest.restart_budget
                if latest is not None
                else int(
                    os.getenv(
                        "GPU_FAULT_DEFAULT_RESTART_BUDGET",
                        "1",
                    )
                )
            ),
        }

    def step_parameters(
        self,
        operation: WorkflowOperation,
        event: XidEvent | SxidEvent,
        incident: FaultIncident,
        decision: FaultPolicyDecision | None = None,
    ) -> dict[str, object]:
        if operation is WorkflowOperation.CHECKPOINT_WORKLOADS:
            return {"checkpoint_manifest_ref": (event.checkpoint_manifest_ref)}
        if operation is WorkflowOperation.STOP_WORKLOADS:
            return {"termination_initiator_incident_id": (incident.incident_id)}
        if operation is WorkflowOperation.RUN_NVLINK74_WORKFLOW and isinstance(
            event, XidEvent
        ):
            return {
                "xid": event.xid,
                "nvlink_link_id": event.nvlink_link_id,
                "pci_bdf": event.pci_bdf,
                "registers": [f"0x{value:x}" for value in event.registers],
                "nvlink_occurrence_counts": (
                    decision.nvlink_occurrence_counts if decision is not None else {}
                ),
                "procedure": "NVIDIA_FIELD_DIAGNOSTIC",
            }
        if operation is WorkflowOperation.CHECK_MECHANICALS:
            if isinstance(event, SxidEvent):
                return {
                    "sxid": event.sxid,
                    "xid": event.sxid,
                    "switch_id": event.switch_id,
                    "port": event.port,
                    "pci_bdf": event.pci_bdf,
                }
            return {
                "xid": event.xid,
                "nvlink_link_id": event.nvlink_link_id,
                "pci_bdf": event.pci_bdf,
                "registers": [f"0x{value:x}" for value in event.registers],
                "nvlink_occurrence_counts": (
                    decision.nvlink_occurrence_counts if decision is not None else {}
                ),
            }
        if operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE:
            if isinstance(event, XidEvent):
                return {
                    "xid": event.xid,
                    "registers": [f"0x{value:x}" for value in event.registers],
                    "pci_bdf": event.pci_bdf,
                    "nvlink_link_id": event.nvlink_link_id,
                    "nvlink_occurrence_counts": (
                        decision.nvlink_occurrence_counts
                        if event.xid == 74 and decision is not None
                        else {}
                    ),
                    "source_evidence_ref": event.evidence_ref,
                    "decode_reasons": incident.reasons,
                    "policy_version": incident.policy_version,
                }
            return {
                "sxid": event.sxid,
                "classification": event.classification.value,
                "classification_source": (event.classification_source),
                "link_scope": event.link_scope.value,
                "switch_id": event.switch_id,
                "pci_bdf": event.pci_bdf,
                "port": event.port,
                "fabric_partition": event.fabric_partition,
                "source_evidence_ref": event.evidence_ref,
            }
        if operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES and isinstance(
            event, SxidEvent
        ):
            return {
                "fabric_partition": event.fabric_partition,
                "sxid": event.sxid,
            }
        if operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES:
            # A Fabric Manager restart on a graded NVSwitch node: the node's own
            # NVSwitch fabric, named the way the SXID ingest names it.
            return {
                "fabric_partition": event.fabric_partition
                or f"{event.cluster_id}/{event.node_id}/local-nvswitch",
                "xid": event.xid,
            }
        if operation is WorkflowOperation.REMEDIATE_DRIVER:
            return {"target_driver_branch": self.target_driver_branch}
        if operation is WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE:
            return {"target_firmware_version": (self.target_firmware_version)}
        if operation is WorkflowOperation.RESTART_WORKLOAD:
            # The XID usually kills the process, so by the time the event is
            # ingested the watcher may already hold the Pod as terminated and
            # the [-30 s, +120 s] window finds no live container on the node.
            # With no fallback the step carried source_gpu_count=0, which
            # _restart_guard read as a GPU count change and waited for an
            # approval that can never be produced; the regional executor has
            # no store to recover the count from. The detector already named
            # the GPUs, so hand them over the same way the node-health family
            # does.
            if isinstance(event, XidEvent):
                event_gpu_uuids = [event.gpu_uuid] if event.gpu_uuid else []
            else:
                event_gpu_uuids = list(event.participating_gpu_uuids)
            return self.restart_step_parameters(
                event.cluster_id,
                event.affected_workload_ids,
                job_id=event.job_id,
                fallback_attempt_id=(event.attempt_id or event.event_id),
                # Compared against the watcher's observation times, which are
                # control-plane clocks; the node's observed_at is not (F-B8).
                observed_at=event.ingested_at or event.observed_at,
                node_id=event.node_id,
                fallback_gpu_uuids=event_gpu_uuids or list(incident.gpu_uuids),
            )
        return {}

    def with_pre_actions(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        operations: list[WorkflowOperation],
    ) -> list[WorkflowOperation]:
        if not decision.pre_actions:
            return operations
        result = list(operations)
        for required in self.pre_action_order:
            if required not in decision.pre_actions:
                continue
            operation = self.pre_action_operation[required]
            if operation in result:
                continue
            if operation is WorkflowOperation.STOP_WORKLOADS:
                if event.workload_state is not WorkloadState.ACTIVE:
                    continue
                if (
                    event.checkpoint_manifest_ref
                    and WorkflowOperation.CHECKPOINT_WORKLOADS not in result
                ):
                    result.insert(
                        self.pre_action_index(result, operation),
                        WorkflowOperation.CHECKPOINT_WORKLOADS,
                    )
            result.insert(
                self.pre_action_index(result, operation),
                operation,
            )
        return result

    def catalog_operations(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        effective_action: RecoveryAction | None = None,
        *,
        fabric_restart: bool = False,
    ) -> list[WorkflowOperation]:
        """The catalog workflow; ``fabric_restart`` is RESTART_FM on a node
        whose restart is the full GPU + NVSwitch reset
        (``fabric_manager_restart_inventory``)."""

        action = effective_action or decision.action
        operations = [WorkflowOperation.FREEZE_EVIDENCE]
        if isinstance(event, XidEvent) and event.xid == 74:
            return _xid74_operations(event, decision)
        if self.requires_sxid_diagnostics(event):
            operations.append(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE)
        if (
            decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
            and decision.safety_action is RecoveryAction.QUARANTINE
        ):
            operations.append(WorkflowOperation.QUARANTINE)
            return list(dict.fromkeys(operations))
        if decision.official_action == "CHECK_MECHANICALS":
            # CHECK_MECHANICALS is an operator investigation, not an
            # automatic containment action. It must notify and wait for
            # acknowledgement without cordoning or stopping a healthy
            # workload. Any later reset/reboot is a separate explicit,
            # fenced workflow.
            operations.append(WorkflowOperation.CHECK_MECHANICALS)
            return operations
        if decision.official_action == "UPDATE_SWFW":
            operations.append(WorkflowOperation.MARK_UNSCHEDULABLE)
            if event.workload_state is WorkloadState.ACTIVE:
                if event.checkpoint_manifest_ref:
                    operations.append(WorkflowOperation.CHECKPOINT_WORKLOADS)
                operations.append(WorkflowOperation.STOP_WORKLOADS)
            operations.extend(
                [
                    WorkflowOperation.QUIESCE_GPU_SERVICES,
                    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
                    WorkflowOperation.RESTORE_GPU_SERVICES,
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.VALIDATE_HOST,
                    WorkflowOperation.RESTORE_SCHEDULING,
                ]
            )
            if event.affected_workload_ids:
                operations.append(WorkflowOperation.RESTART_WORKLOAD)
            return operations
        if decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES" or (
            fabric_restart and decision.official_action == RESTART_FABRIC_MANAGER_ACTION
        ):
            operations.append(WorkflowOperation.MARK_UNSCHEDULABLE)
            if event.workload_state is WorkloadState.ACTIVE:
                if event.checkpoint_manifest_ref:
                    operations.append(WorkflowOperation.CHECKPOINT_WORKLOADS)
                operations.append(WorkflowOperation.STOP_WORKLOADS)
            operations.extend(
                [
                    WorkflowOperation.QUIESCE_GPU_SERVICES,
                    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                    *self.sxid_remediation_operations(event),
                    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                    WorkflowOperation.RESTORE_GPU_SERVICES,
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.VALIDATE_FABRIC,
                    WorkflowOperation.RESTORE_SCHEDULING,
                ]
            )
            if event.affected_workload_ids:
                operations.append(WorkflowOperation.RESTART_WORKLOAD)
            return operations
        if action is RecoveryAction.RESTART_WORKLOAD:
            operations.extend(
                [
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                ]
            )
            return operations
        if action is RecoveryAction.ESCALATE_OPERATOR:
            operations.append(WorkflowOperation.MARK_UNSCHEDULABLE)
            if event.workload_state is WorkloadState.ACTIVE:
                if event.checkpoint_manifest_ref:
                    operations.append(WorkflowOperation.CHECKPOINT_WORKLOADS)
                operations.append(WorkflowOperation.STOP_WORKLOADS)
            operations.extend(
                [
                    WorkflowOperation.QUARANTINE,
                    WorkflowOperation.ESCALATE_SUPPORT,
                ]
            )
            return operations
        if action in {
            RecoveryAction.RESET_GPU,
            RecoveryAction.REBOOT_NODE,
            RecoveryAction.REPLACE_NODE,
        }:
            operations.append(WorkflowOperation.MARK_UNSCHEDULABLE)
            if event.workload_state is WorkloadState.ACTIVE:
                if event.checkpoint_manifest_ref:
                    operations.append(WorkflowOperation.CHECKPOINT_WORKLOADS)
                operations.append(WorkflowOperation.STOP_WORKLOADS)
            if action is RecoveryAction.RESET_GPU:
                operations.extend(
                    [
                        WorkflowOperation.QUIESCE_GPU_SERVICES,
                        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                        *self.sxid_remediation_operations(event),
                        WorkflowOperation.RESET_GPU,
                        WorkflowOperation.RESTORE_GPU_SERVICES,
                        WorkflowOperation.VALIDATE_GPU,
                    ]
                )
            elif action is RecoveryAction.REBOOT_NODE:
                operations.extend(
                    [
                        WorkflowOperation.RESTART_NODE,
                        WorkflowOperation.VALIDATE_GPU,
                        WorkflowOperation.VALIDATE_HOST,
                        WorkflowOperation.VALIDATE_FABRIC,
                    ]
                )
            else:
                operations.extend(
                    [
                        WorkflowOperation.REPLACE_NODE,
                        WorkflowOperation.VALIDATE_GPU,
                        WorkflowOperation.VALIDATE_HOST,
                        WorkflowOperation.VALIDATE_FABRIC,
                    ]
                )
            operations.append(WorkflowOperation.RESTORE_SCHEDULING)
            if event.affected_workload_ids:
                operations.append(WorkflowOperation.RESTART_WORKLOAD)
            return operations

        official_operation = self.official_workflow_operation.get(
            decision.official_action or ""
        )
        if official_operation:
            operations.append(official_operation)
        return operations

    def official_operations(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        effective_action: RecoveryAction | None = None,
        *,
        fabric_restart: bool = False,
    ) -> list[WorkflowOperation]:
        """Compile the catalog workflow and honour its pre-actions.

        The policy marks containment that must precede remediation in
        ``pre_actions``. Applying it here makes that a guarantee rather
        than documentation: the action branches below happen to cover
        today's catalog, but a new branch must not be able to drop the
        cordon the policy asked for.
        """
        operations = self.catalog_operations(
            event, decision, effective_action, fabric_restart=fabric_restart
        )
        return self.with_pre_actions(event, decision, operations)

    def compile_steps(
        self,
        operations: list[WorkflowOperation],
        profile,
        node_ids: list[str],
        gpu_uuids: list[str],
        workload_ids: list[str],
        *,
        workload_state: WorkloadState | None = None,
    ) -> tuple[list[WorkflowStepSpec], list[str]]:
        """Compile ``operations`` into steps, or say why they cannot run.

        ``workload_state`` is the finding's or event's view of the node's
        workload. When it is UNKNOWN and the plan acts on the node's current
        state (``mutates_node``), the plan is refused here -- the one choke
        point both the fault family and the node-health family compile
        through -- so a reboot, a driver remediation or a plugin restart
        cannot run under a training job nobody stopped (ARCH-B2). Containment
        is exempt for the reason ``mutates_node`` gives.
        """

        steps = []
        errors = []
        if workload_state is WorkloadState.UNKNOWN and self.mutates_node(operations):
            errors.append(WORKLOAD_STATE_UNKNOWN_REASON)
        for operation in operations:
            capability = self.operation_capability[operation]
            owner = self.owner(profile, capability)
            if owner is None:
                errors.append(f"no executable owner for {capability.value}")
                continue
            steps.append(
                WorkflowStepSpec(
                    operation=operation,
                    execution_owner=owner,
                    node_ids=node_ids,
                    gpu_uuids=gpu_uuids,
                    workload_ids=workload_ids,
                )
            )
        return steps, errors

    def mutates_node(
        self,
        operations: list[WorkflowOperation],
    ) -> bool:
        """Report whether a plan acts on the node's current state.

        Containment -- cordon, quarantine taint, restoring scheduling --
        is excluded on purpose. It is reversible scheduler state, so a
        gate that blocks it because the workload state is unknown or
        because the evidence predates the current boot would leave a
        suspect node schedulable, which is the opposite of fail-closed.
        """

        return bool(set(operations) & NODE_MUTATING_OPERATIONS)
