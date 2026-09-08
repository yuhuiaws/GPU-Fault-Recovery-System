from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from functools import cached_property
from threading import RLock
from typing import Any, Sequence

from gpu_fault.env import env_bool
from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    BlockedKind,
    Environment,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowExecutionResult,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkloadState,
    bounded_reasons,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import (
    MERGE_INTENT_OPERATIONS,
    NODE_ACTION_SCOPE_OPERATIONS,
    NODE_EXCLUSIVE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
    RECOVERY_OPERATION_DOMINANCE,
    RECOVERY_OPERATION_RANK,
    TRANSIENT_GPU_INVENTORY_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
    ZERO_RANK_ACTION_OPERATIONS,
)
from gpu_fault.operation_registry import (
    OPERATION_CAPABILITY as REGISTERED_OPERATION_CAPABILITY,
)
from gpu_fault.orchestration import (
    DagBrancher,
    HardwareEscalationService,
    RecoveryArbiter,
    SxidIngestionCallbacks,
    SxidIngestionService,
    WorkflowBuilder,
)
from gpu_fault.orchestration.families import (
    DrainOperationCallbacks,
    DrainOperationService,
    EvidenceOperationService,
    GroupedFaultCallbacks,
    GroupedFaultService,
    GroupedHealthCallbacks,
    GroupedHealthService,
    NodeConflictService,
    NodeHealthCallbacks,
    NodeHealthIngestionService,
    NodeHealthPlanBuilder,
    NodeLifecycleCallbacks,
    NodeLifecycleOperationService,
    NodeScopedFaultCallbacks,
    NodeScopedFaultService,
    ResetOperationService,
    ValidationOperationService,
)
from gpu_fault.orchestration.families.identity import note_stale_event_link
from gpu_fault.orchestration.placement_hold import PlacementHoldService
from gpu_fault.orchestration.workflow_merge import (
    WorkflowMergeService,
)
from gpu_fault.policy import (
    ActionDisposition,
    DistributedXidBatch,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.watcher import AttemptObservation

LOGGER = logging.getLogger(__name__)


OPERATION_CAPABILITY = REGISTERED_OPERATION_CAPABILITY


OFFICIAL_WORKFLOW_OPERATION = {
    "RESET_GPU": WorkflowOperation.RESET_GPU,
    "RESTART_BM": WorkflowOperation.RESTART_NODE,
    "RESTART_VM": WorkflowOperation.RESTART_VM,
    "RESTART_FM": WorkflowOperation.RESTART_FABRIC_MANAGER,
    "WORKFLOW_NVLINK_ERR": (WorkflowOperation.RUN_NVLINK74_WORKFLOW),
    "WORKFLOW_NVLINK5_ERR": (WorkflowOperation.RUN_NVLINK74_WORKFLOW),
    "RESET_ALL_GPUS_AND_NVSWITCHES": (WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES),
    "CHECK_MECHANICALS": WorkflowOperation.CHECK_MECHANICALS,
    "UPDATE_SWFW": WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    "CONTACT_SUPPORT": WorkflowOperation.ESCALATE_SUPPORT,
    "CHECK_UVM": WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    "XID_154": WorkflowOperation.ESCALATE_SUPPORT,
    "DRAIN_P2P": WorkflowOperation.STOP_WORKLOADS,
}


# Containment the policy demands before remediation, in the order it
# has to be applied.
PRE_ACTION_ORDER = (
    RecoveryAction.MARK_UNSCHEDULABLE,
    RecoveryAction.DRAIN,
    RecoveryAction.STOP_WORKLOAD,
)

PRE_ACTION_OPERATION = {
    RecoveryAction.MARK_UNSCHEDULABLE: (WorkflowOperation.MARK_UNSCHEDULABLE),
    RecoveryAction.DRAIN: WorkflowOperation.MARK_UNSCHEDULABLE,
    RecoveryAction.STOP_WORKLOAD: WorkflowOperation.STOP_WORKLOADS,
}


class WorkflowFencingError(ValueError):
    pass


class IncidentOrchestrator:
    """Builds proactive workflows without performing host operations."""

    _ARBITER = RecoveryArbiter()
    _BRANCHER = DagBrancher(_ARBITER)

    def __init__(
        self,
        store: ControlPlaneStore,
        *,
        multi_node_aggregation_window_seconds: int | None = None,
        processor_drain_max_wait_seconds: int | None = None,
        multi_node_aggregation_window_max_seconds: int | None = None,
        sxid_driver_remediation_codes: set[int] | None = None,
        sxid_firmware_update_codes: set[int] | None = None,
        target_driver_branch: int | None = None,
        target_firmware_version: str | None = None,
        workflow_preemption_enabled: bool | None = None,
        fault_action_max_age_seconds: int | None = None,
    ) -> None:
        self.store = store
        self._arbiter = self._ARBITER
        self._brancher = self._BRANCHER
        self._lock = RLock()
        self._conflicts = NodeConflictService(self.store, self._arbiter)
        self._evidence_operations = EvidenceOperationService(
            self.store,
            self._active_node_exclusive_workflow,
        )
        self._validation_operations = ValidationOperationService()
        self.multi_node_aggregation_window_seconds = (
            multi_node_aggregation_window_seconds
            if multi_node_aggregation_window_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_SECONDS",
                    "5",
                )
            )
        )
        if self.multi_node_aggregation_window_seconds < 0:
            raise ValueError("multi-node aggregation window must be non-negative")
        self.multi_node_aggregation_window_max_seconds = (
            multi_node_aggregation_window_max_seconds
            if multi_node_aggregation_window_max_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_MAX_SECONDS",
                    "30",
                )
            )
        )
        if (
            self.multi_node_aggregation_window_max_seconds
            < self.multi_node_aggregation_window_seconds
        ):
            raise ValueError(
                "multi-node aggregation maximum must not be less than the base window"
            )
        self.processor_drain_max_wait_seconds = (
            processor_drain_max_wait_seconds
            if processor_drain_max_wait_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_DRAIN_MAX_WAIT_SECONDS",
                    "30",
                )
            )
        )
        if self.processor_drain_max_wait_seconds < 0:
            raise ValueError("processor drain maximum wait must be non-negative")
        self.workflow_preemption_enabled = (
            workflow_preemption_enabled
            if workflow_preemption_enabled is not None
            else env_bool("GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION", True)
        )
        self.fault_action_max_age_seconds = (
            fault_action_max_age_seconds
            if fault_action_max_age_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_FAULT_ACTION_MAX_AGE_SECONDS",
                    "900",
                )
            )
        )
        if self.fault_action_max_age_seconds <= 0:
            raise ValueError("fault action maximum age must be positive")
        self.sxid_driver_remediation_codes = (
            sxid_driver_remediation_codes
            if sxid_driver_remediation_codes is not None
            else self._parse_sxid_codes("GPU_FAULT_SXID_DRIVER_REMEDIATION_CODES")
        )
        self.sxid_firmware_update_codes = (
            sxid_firmware_update_codes
            if sxid_firmware_update_codes is not None
            else self._parse_sxid_codes("GPU_FAULT_SXID_FIRMWARE_UPDATE_CODES")
        )
        self.target_driver_branch = (
            target_driver_branch
            if target_driver_branch is not None
            else (
                int(os.environ["GPU_FAULT_TARGET_DRIVER_BRANCH"])
                if os.getenv("GPU_FAULT_TARGET_DRIVER_BRANCH")
                else None
            )
        )
        self.target_firmware_version = (
            target_firmware_version
            if target_firmware_version is not None
            else (os.getenv("GPU_FAULT_TARGET_FIRMWARE_VERSION") or None)
        )
        if self.sxid_driver_remediation_codes and self.target_driver_branch is None:
            raise ValueError(
                "SXID driver remediation codes require GPU_FAULT_TARGET_DRIVER_BRANCH"
            )
        if self.sxid_firmware_update_codes and not self.target_firmware_version:
            raise ValueError(
                "SXID firmware update codes require GPU_FAULT_TARGET_FIRMWARE_VERSION"
            )
        self._builder = WorkflowBuilder(
            self.store,
            target_driver_branch=self.target_driver_branch,
            target_firmware_version=self.target_firmware_version,
            sxid_driver_remediation_codes=(self.sxid_driver_remediation_codes),
            sxid_firmware_update_codes=(self.sxid_firmware_update_codes),
            operation_capability=OPERATION_CAPABILITY,
            official_workflow_operation=OFFICIAL_WORKFLOW_OPERATION,
            pre_action_order=PRE_ACTION_ORDER,
            pre_action_operation=PRE_ACTION_OPERATION,
        )
        self._reset_operations = ResetOperationService(
            self.store, self._builder, self._lock
        )
        self._placement_holds = PlacementHoldService(self.store, self._builder)
        # Rule A, case 2: holds opened here, and holds the observation ingest
        # could not open (logged there; the observation is still accepted).
        self.placement_holds_opened_total = 0
        self.placement_holds_failed_total = 0
        self._node_lifecycle_operations = self._create_node_lifecycle_service()
        self._escalation = HardwareEscalationService(self.store, self._builder)
        self._sxid_ingestion = SxidIngestionService(
            self.store,
            self._builder,
            self._arbiter,
            self._brancher,
            SxidIngestionCallbacks(
                attempt_observation=self._attempt_observation,
                attempt_group_key=self._attempt_group_key,
                reopen_if_terminal=self._reopen_if_terminal,
                generation_fence=self._generation_fence,
                active_job_recovery_workflow=(self._active_job_recovery_workflow),
                candidate_recovery_workflow=(self._candidate_recovery_workflow),
                claims_node_exclusively=(self._claims_node_exclusively),
                active_node_exclusive_workflow=(self._active_node_exclusive_workflow),
                merge_disposition=self._merge_disposition,
                widen_node_action_scope=(self._widen_node_action_scope),
                aggregation_deadlines=(self._aggregation_deadlines),
                prepare_preempting_successor=(self._prepare_preempting_successor),
                quiesce_parameters=self._quiesce_parameters,
            ),
            multi_node_aggregation_window_seconds=(
                self.multi_node_aggregation_window_seconds
            ),
            target_driver_branch=self.target_driver_branch,
            target_firmware_version=(self.target_firmware_version),
        )

    def _create_node_lifecycle_service(
        self,
    ) -> NodeLifecycleOperationService:
        callbacks = NodeLifecycleCallbacks(
            active_job_recovery_workflow=(self._active_job_recovery_workflow),
            active_node_exclusive_workflow=(self._active_node_exclusive_workflow),
            aggregation_deadlines=self._aggregation_deadlines,
            can_append_parallel_job_branch=(self._can_append_parallel_job_branch),
            claims_node_exclusively=self._claims_node_exclusively,
            inventory_validation_parameters=(self._inventory_validation_parameters),
            preemption_scope_matches=self._preemption_scope_matches,
            prepare_preempting_successor=(self._prepare_preempting_successor),
        )
        return NodeLifecycleOperationService(
            self.store,
            self._builder,
            self._arbiter,
            self._brancher,
            callbacks,
            aggregation_window_seconds=(self.multi_node_aggregation_window_seconds),
        )

    @cached_property
    def _drain_operations(self) -> DrainOperationService:
        callbacks = DrainOperationCallbacks(
            active_node_exclusive_workflow=(self._active_node_exclusive_workflow),
            aggregation_deadlines=self._aggregation_deadlines,
            attempt_group_key=self._attempt_group_key,
            merge_disposition=self._merge_disposition,
            node_group_key=self._node_group_key,
            ingest_node_health=self.ingest_node_health,
            incident_state_for_workflow=(self._incident_state_for_workflow),
            preempt_parallel_job_branch=(self._preempt_parallel_job_branch),
            prepare_preempting_successor=(self._prepare_preempting_successor),
        )
        return DrainOperationService(
            self.store,
            self._arbiter,
            self._brancher,
            callbacks,
            workflow_preemption_enabled=(self.workflow_preemption_enabled),
        )

    @cached_property
    def _node_scoped_faults(self) -> NodeScopedFaultService:
        callbacks = NodeScopedFaultCallbacks(
            active_node_exclusive_workflow=(self._active_node_exclusive_workflow),
            aggregation_deadlines=self._aggregation_deadlines,
            build_workflow=self._build_workflow,
            claims_node_exclusively=self._claims_node_exclusively,
            incident_state_for_workflow=(self._incident_state_for_workflow),
            merge_disposition=self._merge_disposition,
            node_group_key=self._node_group_key,
            prepare_preempting_successor=(self._prepare_preempting_successor),
            preempt_parallel_job_branch=(self._preempt_parallel_job_branch),
            reopen_if_terminal=self._reopen_if_terminal,
            runtime_effective_action=self._runtime_effective_action,
            widen_node_action_scope=self._widen_node_action_scope,
        )
        return NodeScopedFaultService(
            self.store,
            self._arbiter,
            self._brancher,
            callbacks,
            aggregation_window_seconds=(self.multi_node_aggregation_window_seconds),
            merge_actions=self._NODE_MERGE_ACTIONS,
            workflow_preemption_enabled=(self.workflow_preemption_enabled),
        )

    @cached_property
    def _grouped_faults(self) -> GroupedFaultService:
        callbacks = GroupedFaultCallbacks(
            active_job_recovery_workflow=(self._active_job_recovery_workflow),
            active_node_exclusive_workflow=(self._active_node_exclusive_workflow),
            aggregation_deadlines=self._aggregation_deadlines,
            attempt_group_key=self._attempt_group_key,
            attempt_observation=self._attempt_observation,
            build_workflow=self._build_workflow,
            claims_node_exclusively=self._claims_node_exclusively,
            generation_fence=self._generation_fence,
            incident_state_for_workflow=(self._incident_state_for_workflow),
            merge_disposition=self._merge_disposition,
            prepare_preempting_successor=(self._prepare_preempting_successor),
            preempt_parallel_job_branch=(self._preempt_parallel_job_branch),
            quiesce_parameters=self._quiesce_parameters,
            reopen_if_terminal=self._reopen_if_terminal,
            runtime_effective_action=self._runtime_effective_action,
            widen_node_action_scope=self._widen_node_action_scope,
        )
        return GroupedFaultService(
            self.store,
            self._arbiter,
            self._brancher,
            callbacks,
            aggregation_window_seconds=(self.multi_node_aggregation_window_seconds),
            workflow_preemption_enabled=(self.workflow_preemption_enabled),
            attempt_group_actions=self._ATTEMPT_GROUP_ACTIONS,
            groupable_operations=self._GROUPABLE_RECOVERY_OPERATIONS,
            official_operation=OFFICIAL_WORKFLOW_OPERATION,
        )

    @cached_property
    def _grouped_health(self) -> GroupedHealthService:
        callbacks = GroupedHealthCallbacks(
            active_job_recovery_workflow=(self._active_job_recovery_workflow),
            aggregation_deadlines=self._aggregation_deadlines,
            attempt_group_key=self._attempt_group_key,
            attempt_observation=self._attempt_observation,
            incident_state_for_workflow=(self._incident_state_for_workflow),
            ingest_node_health=self.ingest_node_health,
            is_attempt_grouped_health_finding=(self._is_attempt_grouped_health_finding),
            merge_disposition=self._merge_disposition,
            prepare_preempting_successor=(self._prepare_preempting_successor),
            preempt_parallel_job_branch=(self._preempt_parallel_job_branch),
            quiesce_parameters=self._quiesce_parameters,
            reopen_if_terminal=self._reopen_if_terminal,
            utc=self._utc,
            widen_node_action_scope=self._widen_node_action_scope,
        )
        return GroupedHealthService(
            self.store,
            self._arbiter,
            self._brancher,
            callbacks,
            aggregation_window_seconds=(self.multi_node_aggregation_window_seconds),
            workflow_preemption_enabled=(self.workflow_preemption_enabled),
        )

    @cached_property
    def _node_health(self) -> NodeHealthIngestionService:
        callbacks = NodeHealthCallbacks(
            active_node_exclusive_workflow=(self._active_node_exclusive_workflow),
            active_workflow_covers_inventory_finding=(
                self._active_workflow_covers_inventory_finding
            ),
            attempt_observation=self._attempt_observation,
            claims_node_exclusively=self._claims_node_exclusively,
            ingest_grouped_health_finding=(self._ingest_grouped_health_finding),
            ingest_grouped_node_replacement=(self._ingest_grouped_node_replacement),
            ingest_grouped_node_resource_finding=(
                self._ingest_grouped_node_resource_finding
            ),
            ingest_terminal_node_quarantine=(self._ingest_terminal_node_quarantine),
            inventory_validation_parameters=(self._inventory_validation_parameters),
            node_group_key=self._node_group_key,
            preemption_scope_matches=self._preemption_scope_matches,
            prepare_preempting_successor=(self._prepare_preempting_successor),
            sample_hung_triage_nodes=(self._sample_hung_triage_nodes),
        )
        plan_builder = NodeHealthPlanBuilder(
            self.store,
            self._builder,
            callbacks,
        )
        return NodeHealthIngestionService(
            self.store,
            self._lock,
            plan_builder,
            callbacks,
        )

    @cached_property
    def _workflow_merger(self) -> WorkflowMergeService:
        return WorkflowMergeService(
            self._arbiter,
            self._brancher,
            preemption_enabled=self.workflow_preemption_enabled,
            workload_scoped_operations=(self._WORKLOAD_SCOPED_OPERATIONS),
            node_exclusive_operations=(self._NODE_EXCLUSIVE_OPERATIONS),
            workflow_resource_claims_by_node=(self._workflow_resource_claims_by_node),
        )

    @staticmethod
    def _parse_sxid_codes(name: str) -> set[int]:
        values = set()
        for item in os.getenv(name, "").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                code = int(item)
            except ValueError as exc:
                raise ValueError(
                    f"{name} must contain comma-separated integers"
                ) from exc
            if code <= 0:
                raise ValueError(f"{name} values must be positive")
            values.add(code)
        return values

    def gpu_metric_action(
        self,
        *,
        cluster_id: str,
        node_id: str,
        metric_name: str,
        has_explicit_gpu: bool,
        default: RecoveryAction,
    ) -> RecoveryAction:
        if metric_name in {
            "row_remap_failure",
            "ecc_dbe_volatile_total",
            "ecc_dbe_aggregate_total",
        }:
            return RecoveryAction.DRAIN
        if metric_name in {
            "retired_pages_pending",
            "row_remap_pending",
        }:
            return (
                RecoveryAction.RESET_GPU
                if has_explicit_gpu
                else RecoveryAction.QUARANTINE
            )
        return default

    def compile_branch_steps(
        self,
        workflow: WorkflowRequest,
        operations: Sequence[WorkflowOperation],
        node_id: str,
        gpu_uuids: Sequence[str],
    ) -> list[WorkflowStepSpec]:
        """Steps for one node's escalated branch (F-N1), or ``[]``.

        Compiled against the workflow's runtime profile exactly like the
        whole-workflow replacement path; an empty result tells the executor
        the branch cannot be escalated in place.
        """

        if not workflow.runtime_profile_version:
            return []
        try:
            profile = self.store.get_profile(workflow.runtime_profile_version)
        except NotFoundError:
            return []
        compiled: tuple[list[WorkflowStepSpec], list[str]] = (
            self._builder.compile_steps(
                list(operations), profile, [node_id], list(gpu_uuids), []
            )
        )
        steps, errors = compiled
        if errors:
            LOGGER.warning(
                "branch escalation for %s on %s could not be compiled: %s",
                workflow.request_id,
                node_id,
                "; ".join(errors),
            )
            return []
        return steps

    def escalate_failed_hardware_remediation(
        self, workflow: WorkflowRequest
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._escalation.escalate(workflow)

    def correlate_provider_event(
        self,
        decision: FaultPolicyDecision,
        *,
        cluster_id: str,
        same_source_window: timedelta = timedelta(seconds=30),
        cross_source_window: timedelta = timedelta(minutes=5),
    ) -> FaultPolicyDecision:
        """Merge provider/raw observations with source-aware clocks."""
        marker = decision.marker
        observed_after = self._utc(marker.observed_at) - cross_source_window
        for candidate in self.store.list_recent_markers_for_nodes(
            set(marker.scope.node_ids),
            observed_after,
            source_boot_id=marker.source_boot_id,
        ):
            # Node names can repeat across regional clusters, so marker
            # correlation must stop at the incident's cluster boundary
            # before comparing shared node and GPU identities.
            try:
                candidate_incident = self.store.get_incident(candidate.incident_id)
            except NotFoundError:
                continue
            if candidate_incident.cluster_id != cluster_id:
                continue
            if candidate.marker_id == marker.marker_id:
                continue
            if not set(candidate.scope.node_ids).intersection(marker.scope.node_ids):
                continue
            candidate_gpus = set(candidate.scope.gpu_uuids)
            marker_gpus = set(marker.scope.gpu_uuids)
            if (
                candidate_gpus
                and marker_gpus
                and not candidate_gpus.intersection(marker_gpus)
            ):
                continue
            candidate_pci = set(candidate.scope.pci_bdfs)
            marker_pci = set(marker.scope.pci_bdfs)
            if (
                candidate_pci
                and marker_pci
                and not candidate_pci.intersection(marker_pci)
            ):
                continue
            same_source = candidate.event_source == marker.event_source
            candidate_keys = set(candidate.correlation_keys)
            marker_keys = set(marker.correlation_keys)
            scope_identity_matches = bool(
                (candidate_gpus & marker_gpus)
                or (candidate_pci & marker_pci)
                or (
                    set(candidate.scope.fabric_partitions)
                    & set(marker.scope.fabric_partitions)
                )
            )
            if candidate.fault_class and marker.fault_class:
                if candidate.fault_class != marker.fault_class:
                    continue
                if not (
                    candidate_keys.intersection(marker_keys)
                    or scope_identity_matches
                    or (same_source and candidate.raw_reason == marker.raw_reason)
                ):
                    continue
            elif candidate.raw_reason != marker.raw_reason:
                continue
            window = same_source_window if same_source else cross_source_window
            if (
                self._marker_time_delta(
                    candidate,
                    marker,
                    same_source=same_source,
                )
                > window
            ):
                continue
            return decision.model_copy(
                update={
                    "marker": marker.model_copy(
                        update={"incident_id": candidate.incident_id}
                    ),
                    "duplicate": True,
                }
            )
        return decision

    @staticmethod
    def _marker_time_delta(
        left,
        right,
        *,
        same_source: bool,
    ) -> timedelta:
        if (
            same_source
            and left.source_monotonic_us is not None
            and right.source_monotonic_us is not None
            and left.source_boot_id is not None
            and left.source_boot_id == right.source_boot_id
        ):
            return timedelta(
                microseconds=abs(left.source_monotonic_us - right.source_monotonic_us)
            )
        left_time = IncidentOrchestrator._event_time(left)
        right_time = IncidentOrchestrator._event_time(right)
        return abs(
            IncidentOrchestrator._utc(left_time) - IncidentOrchestrator._utc(right_time)
        )

    @staticmethod
    def _event_time(event) -> datetime:
        return (
            event.source_event_time
            or event.observed_at
            or event.collected_at
            or event.ingested_at
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def ingest(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[FaultIncident, WorkflowRequest | None]:
        with self._lock:
            existing = self.store.get_incident_by_event(event.event_id)
            if existing is not None:
                linked = self._linked_workflow(existing, event.event_id)
                if linked is not None:
                    return existing, linked[0]
                # C-04: the link says handled but its workflow is gone; the
                # build path below rebuilds and re-links (the store's own
                # duplicate check treats the link as dirty the same way).
            try:
                correlated = self.store.get_incident(decision.marker.incident_id)
            except NotFoundError:
                correlated = None
            linked = None
            if correlated is not None:
                linked = self._linked_workflow(correlated, event.event_id)
                if linked is None:
                    # C-04: the correlation target's own workflow is gone.
                    # Returning it would hand back a pair with no workflow
                    # for an event that needs one; the build path below
                    # rebuilds under the same incident id instead.
                    correlated = None
            if correlated is not None and linked is not None:
                workflow = linked[0]
                observation = self._attempt_observation(event)
                previous_generation = (
                    workflow is not None
                    and observation is not None
                    and observation.started_at is not None
                    and self._utc(observation.started_at)
                    > self._utc(self._event_time(event))
                )
                if previous_generation:
                    (
                        _,
                        _,
                        generation_ignore_reason,
                    ) = self._generation_fence(
                        event,
                        decision,
                        observation,
                        correlated,
                        workflow,
                    )
                    if generation_ignore_reason is not None:
                        read_correlated = correlated
                        correlated = correlated.model_copy(
                            update={
                                "reasons": bounded_reasons(
                                    [
                                        *correlated.reasons,
                                        generation_ignore_reason,
                                    ]
                                ),
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                        # Compare-and-set on the copy read above (ARCH-D1):
                        # ``StaleWriteError`` propagates like the
                        # ``WorkflowFencingError`` this handler already
                        # raises, and ingest retries the whole event with a
                        # fresh read of the correlated incident.
                        self.store.save_incident(
                            correlated,
                            expected=read_correlated,
                            extra_event_ids=[event.event_id],
                        )
                        return correlated, workflow
                else:
                    self.store.link_event_to_incident(
                        event.event_id,
                        correlated.incident_id,
                    )
                    return correlated, workflow

            stale = self._fence_stale_generation_without_baseline(event, decision)
            if stale is not None:
                return stale, None

            if isinstance(event, XidEvent):
                grouped = self._ingest_grouped_fault(event, decision)
                if grouped is not None:
                    return grouped
            elif isinstance(event, SxidEvent):
                grouped = self._sxid_ingestion.ingest_grouped(event, decision)
                if grouped is None:
                    grouped = self._ingest_grouped_fault(event, decision)
                if grouped is not None:
                    return grouped
            # Last resort before an independent incident: keep
            # same-node actions from overlapping each other's blast
            # radius when no workload scoped them together.
            grouped = self._ingest_node_scoped_fault(event, decision)
            if grouped is not None:
                return grouped

            incident = self._independent_incident(event, decision)

            if (
                decision.action is RecoveryAction.NO_ACTION
                and decision.safety_action is None
            ):
                incident = incident.model_copy(
                    update={"state": IncidentState.RECOVERED}
                )
                self.store.save_incident(incident)
                return incident, None

            workflow = self._build_workflow(event, decision, incident)
            incident_state = (
                IncidentState.ACTION_PENDING
                if workflow.status is WorkflowStatus.PENDING
                else IncidentState.SAFETY_PENDING
                if workflow.status is WorkflowStatus.SAFETY_PENDING
                else IncidentState.ESCALATED
            )
            incident = incident.model_copy(
                update={
                    "state": incident_state,
                    "workflow_request_id": workflow.request_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            # One transaction for incident + workflow + event link. The blind
            # ``save_workflow`` then ``save_incident`` pair was two autocommit
            # statements on Postgres: two replicas ingesting the same event made
            # two incidents, and a crash in between left an orphan workflow
            # (store review 2026-09-07, item B). The pair is built above, not in
            # the builder: ``_build_workflow`` reads the profile from the store,
            # and the builder runs under the advisory lock, where it must be pure
            # (item D). ``incident.event_id`` is this event's id, so the store's
            # link of both is one link.
            created_incident, created_workflow, _created = (
                self.store.create_incident_workflow_if_absent(
                    event.event_id, lambda: (incident, workflow)
                )
            )
            # ``created`` False means a duplicate raced another replica past the
            # ``get_incident_by_event`` check at the top; the stored pair wins.
            return created_incident, created_workflow

    def _independent_incident(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        *,
        extra_reasons: list[str] | None = None,
    ) -> FaultIncident:
        effective_action, runtime_reason = self._runtime_effective_action(
            event, decision
        )
        return FaultIncident(
            incident_id=decision.marker.incident_id,
            event_id=event.event_id,
            event_type=decision.event_type.value,
            event_source=event.event_source,
            source_boot_id=event.source_boot_id,
            cluster_id=event.cluster_id,
            node_ids=[event.node_id],
            gpu_uuids=decision.marker.scope.gpu_uuids,
            job_id=event.job_id,
            attempt_id=event.attempt_id,
            workload_identity_source=(event.workload_identity_source),
            policy_version=decision.policy_version,
            policy_source=decision.source.value,
            official_action=decision.official_action,
            effective_action=effective_action,
            safety_action=decision.safety_action,
            drill_id=event.drill_id,
            reasons=bounded_reasons(
                [
                    *decision.reasons,
                    *([runtime_reason] if runtime_reason else []),
                    *(extra_reasons or []),
                ]
            ),
        )

    def _fence_stale_generation_without_baseline(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> FaultIncident | None:
        """F-B8: fence a stale event even when no recovery workflow exists.

        ``_generation_fence`` compares a stale candidate against the ranked
        recovery workflow of the running attempt and admits only escalations.
        When there is no such workflow the attempt restart itself is the
        recovery that already answered the fault, so the comparison baseline
        is RESTART_WORKLOAD's rank. A candidate that does not out-rank it is
        recorded as an ignored incident with no workflow; the families never
        see it, so nothing acts on a fault the restart has already left behind.
        """

        observation = self._evidence_operations.attempt_observation(
            event, record_ambiguity=False
        )
        if observation is None or observation.started_at is None:
            return None
        event_time = self._utc(self._event_time(event))
        attempt_started_at = self._utc(observation.started_at)
        if attempt_started_at <= event_time:
            return None
        if self._job_recovery_workflow(observation) is not None:
            # A ranked baseline exists: the family fence compares against it.
            return None
        candidate_workflow = self._candidate_recovery_workflow(event, decision)
        candidate_rank = self._arbiter.workflow_recovery_rank(candidate_workflow)
        restart_rank = self._arbiter.RECOVERY_OPERATION_RANK[
            WorkflowOperation.RESTART_WORKLOAD
        ]
        if candidate_rank > restart_rank:
            return None
        event_name = (
            f"XID {event.xid}" if isinstance(event, XidEvent) else f"SXID {event.sxid}"
        )
        reason = (
            f"Ignored stale {event_name} action for previous attempt "
            f"generation: event_time={event_time.isoformat()}, "
            f"current_attempt={observation.attempt_id}, "
            "current_attempt_started_at="
            f"{attempt_started_at.isoformat()}, "
            f"candidate_rank={candidate_rank}, "
            f"current_recovery_rank={restart_rank} (the attempt restart is the "
            "baseline; no recovery workflow to compare); "
            "action is not an escalation"
        )
        incident = self._independent_incident(
            event, decision, extra_reasons=[reason]
        ).model_copy(update={"state": IncidentState.RECOVERED})
        # ``incident.event_id`` is this event's id and ``save_incident`` links
        # it inside the same transaction on all three backends; the separate
        # ``link_event_to_incident`` was a second autocommit for nothing, and a
        # crash between the two left the fenced incident unlinked (C-09).
        self.store.save_incident(incident)
        return incident

    def _linked_workflow(
        self, incident: FaultIncident, event_id: str
    ) -> tuple[WorkflowRequest | None] | None:
        """The workflow an event's incident points at, wrapped, or None when
        the pointer dangles (C-04). ``(None,)`` is an incident that has no
        workflow on purpose (fenced, NO_ACTION)."""

        if not incident.workflow_request_id:
            return (None,)
        try:
            return (self.store.get_workflow(incident.workflow_request_id),)
        except NotFoundError:
            note_stale_event_link(
                self.store,
                event_id=event_id,
                incident_id=incident.incident_id,
                pointer=incident.workflow_request_id,
            )
            return None

    @staticmethod
    def _sample_hung_triage_nodes(
        attempt_node_ids: list[str],
        *,
        reporting_node_id: str,
        lowest_rank_by_node: dict[str, int],
    ) -> tuple[list[str], list[str]]:
        return EvidenceOperationService.sample_hung_triage_nodes(
            attempt_node_ids,
            reporting_node_id=reporting_node_id,
            lowest_rank_by_node=lowest_rank_by_node,
        )

    def _attempt_observation(
        self,
        event: XidEvent | SxidEvent | NodeHealthFinding,
    ):
        return self._evidence_operations.attempt_observation(event)

    def _active_recovery_attempt_observation(
        self,
        event: XidEvent | SxidEvent | NodeHealthFinding,
        event_gpu_uuids: set[str],
    ):
        return self._evidence_operations.active_recovery_attempt_observation(
            event,
            event_gpu_uuids,
        )

    @staticmethod
    def _workload_cgroup_paths_by_node(
        observation,
    ) -> dict[str, list[str]]:
        return EvidenceOperationService.workload_cgroup_paths_by_node(observation)

    def _quiesce_parameters(
        self,
        parameters: dict[str, Any],
        observation,
    ) -> dict[str, Any]:
        return self._evidence_operations.quiesce_parameters(
            parameters,
            observation,
        )

    _RECOVERY_OPERATION_RANK = RECOVERY_OPERATION_RANK
    _GROUPABLE_RECOVERY_OPERATIONS = frozenset(_RECOVERY_OPERATION_RANK)
    _ATTEMPT_GROUP_ACTIONS = frozenset(
        {
            RecoveryAction.MARK_UNSCHEDULABLE,
            RecoveryAction.DRAIN,
            RecoveryAction.STOP_WORKLOAD,
            RecoveryAction.RESTART_WORKLOAD,
            RecoveryAction.RESET_GPU,
            RecoveryAction.REBOOT_NODE,
            RecoveryAction.REPLACE_NODE,
            RecoveryAction.REMEDIATE_EFA_DRIVER,
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
            RecoveryAction.RUN_DIAGNOSTICS,
            RecoveryAction.QUARANTINE,
            RecoveryAction.ESCALATE_OPERATOR,
        }
    )
    _MERGE_INTENT_OPERATIONS = MERGE_INTENT_OPERATIONS
    _NODE_WIDE_RECOVERY_OPERATIONS = NODE_WIDE_RECOVERY_OPERATIONS
    _ZERO_RANK_ACTION_OPERATIONS = ZERO_RANK_ACTION_OPERATIONS
    _RECOVERY_OPERATION_DOMINANCE = RECOVERY_OPERATION_DOMINANCE

    def _candidate_recovery_workflow(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> WorkflowRequest:
        effective_action, _ = self._runtime_effective_action(event, decision)
        incident = FaultIncident(
            incident_id=decision.marker.incident_id,
            event_id=event.event_id,
            event_type="RECOVERY_RANK_CANDIDATE",
            event_source=event.event_source,
            source_boot_id=event.source_boot_id,
            cluster_id=event.cluster_id,
            node_ids=[event.node_id],
            gpu_uuids=list(decision.marker.scope.gpu_uuids),
            policy_version=decision.policy_version,
            policy_source=decision.source.value,
            official_action=decision.official_action,
            effective_action=effective_action,
            safety_action=decision.safety_action,
            drill_id=event.drill_id,
        )
        return self._build_workflow(event, decision, incident)

    def _merge_disposition(
        self,
        existing_workflow: WorkflowRequest,
        candidate_workflow: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
        *,
        allow_job_branch_merge: bool = True,
        now: datetime | None = None,
    ) -> str:
        return self._workflow_merger.disposition(
            existing_workflow,
            candidate_workflow,
            node_id,
            gpu_uuids,
            allow_job_branch_merge=allow_job_branch_merge,
            now=now,
        )

    def _recovery_action_has_started(
        self,
        workflow: WorkflowRequest,
        indexes: list[int],
    ) -> bool:
        return self._workflow_merger.recovery_action_has_started(workflow, indexes)

    def _can_append_parallel_job_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
    ) -> bool:
        return self._workflow_merger.can_append_parallel_branch(existing, candidate)

    def _preempt_parallel_job_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
    ) -> WorkflowRequest:
        return self._workflow_merger.preempt_parallel_branch(
            existing, candidate, node_id
        )

    def _prepare_preempting_successor(
        self,
        existing_workflow: WorkflowRequest,
        candidate_workflow: WorkflowRequest,
    ) -> WorkflowRequest:
        return self._workflow_merger.prepare_preempting_successor(
            existing_workflow, candidate_workflow
        )

    @staticmethod
    def _preemption_scope_matches(
        existing_incident: FaultIncident,
        candidate_incident: FaultIncident,
    ) -> bool:
        return WorkflowMergeService.preemption_scope_matches(
            existing_incident, candidate_incident
        )

    @staticmethod
    def _incident_state_for_workflow(
        workflow: WorkflowRequest,
    ) -> IncidentState:
        if workflow.status in {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
        }:
            return IncidentState.ACTION_PENDING
        if workflow.status is WorkflowStatus.SAFETY_PENDING:
            return IncidentState.SAFETY_PENDING
        return IncidentState.ESCALATED

    @staticmethod
    def _workflow_restarted_attempt(
        workflow: WorkflowRequest,
    ) -> str | None:
        for execution in reversed(workflow.step_executions):
            if (
                execution.operation is WorkflowOperation.RESTART_WORKLOAD
                and execution.details.get("restart_attempt_id")
            ):
                return str(execution.details["restart_attempt_id"])
        return None

    def hold_attempt_on_repairing_nodes(
        self, observation: AttemptObservation
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """Open a placement hold for an attempt observed on nodes another
        incident is repairing (rule A, case 2); ``None`` when nothing was
        opened. See ``orchestration.placement_hold``."""

        with self._lock:
            held = self._placement_holds.hold(observation)
        if held is not None:
            self.placement_holds_opened_total += 1
        return held

    def _job_recovery_workflow(
        self, observation
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        for (
            incident,
            workflow,
        ) in self.store.list_job_recovery_workflow_incidents(
            observation.cluster_id,
            observation.job_id,
            observation.attempt_id,
            limit=100,
        ):
            if self._arbiter.workflow_recovery_rank(workflow) == 0:
                continue
            return incident, workflow
        return None

    def _active_job_recovery_workflow(
        self, observation
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        for (
            incident,
            workflow,
        ) in self.store.list_active_workflow_incidents(
            observation.cluster_id,
            job_id=observation.job_id,
        ):
            if self._arbiter.workflow_recovery_rank(workflow) == 0:
                continue
            if (
                incident.attempt_id != observation.attempt_id
                and self._workflow_restarted_attempt(workflow) != observation.attempt_id
            ):
                continue
            return incident, workflow
        return None

    def _generation_fence(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        observation,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[
        FaultIncident | None,
        WorkflowRequest | None,
        str | None,
    ]:
        if observation.started_at is None:
            return (
                existing_incident,
                existing_workflow,
                None,
            )
        event_time = self._utc(self._event_time(event))
        attempt_started_at = self._utc(observation.started_at)
        if attempt_started_at <= event_time:
            return (
                existing_incident,
                existing_workflow,
                None,
            )
        baseline = (
            (existing_incident, existing_workflow)
            if existing_incident is not None and existing_workflow is not None
            else self._job_recovery_workflow(observation)
        )
        if baseline is None:
            return (
                existing_incident,
                existing_workflow,
                None,
            )
        baseline_incident, baseline_workflow = baseline
        candidate_workflow = self._candidate_recovery_workflow(event, decision)
        candidate_rank = self._arbiter.workflow_recovery_rank(candidate_workflow)
        current_rank = self._arbiter.workflow_recovery_rank(baseline_workflow)
        if candidate_rank > current_rank or (
            candidate_rank == current_rank
            and not (
                self._arbiter.merge_intent(candidate_workflow)
                <= self._arbiter.merge_intent(baseline_workflow)
            )
        ):
            return (
                baseline_incident,
                baseline_workflow,
                None,
            )
        event_name = (
            f"XID {event.xid}" if isinstance(event, XidEvent) else f"SXID {event.sxid}"
        )
        reason = (
            f"Ignored stale {event_name} action for previous attempt "
            f"generation: event_time={event_time.isoformat()}, "
            f"current_attempt={observation.attempt_id}, "
            "current_attempt_started_at="
            f"{attempt_started_at.isoformat()}, "
            f"candidate_rank={candidate_rank}, "
            f"current_recovery_rank={current_rank}; "
            "action is not an escalation"
        )
        return baseline_incident, baseline_workflow, reason

    @staticmethod
    def _attempt_group_key(cluster_id: str, job_id: str, attempt_id: str) -> str:
        return json.dumps(
            [cluster_id, job_id, attempt_id],
            ensure_ascii=True,
            separators=(",", ":"),
        )

    def _aggregation_deadlines(
        self,
        now: datetime,
        existing_workflow: WorkflowRequest | None = None,
    ) -> tuple[datetime, datetime]:
        node_ids = (
            {
                node_id
                for step in existing_workflow.official_steps
                for node_id in step.node_ids
            }
            if existing_workflow is not None
            else set()
        )
        node_count = max(1, len(node_ids))
        multiplier = max(1, math.ceil(math.log2(max(2, node_count))))
        window_seconds = min(
            self.multi_node_aggregation_window_max_seconds,
            self.multi_node_aggregation_window_seconds * multiplier,
        )
        window = timedelta(seconds=window_seconds)
        maximum = (
            existing_workflow.aggregation_max_deadline
            if existing_workflow is not None
            else None
        )
        if maximum is None:
            maximum = (
                now
                + timedelta(seconds=(self.multi_node_aggregation_window_max_seconds))
                + timedelta(seconds=self.processor_drain_max_wait_seconds)
            )
        return min(now + window, maximum), maximum

    def _ingest_grouped_fault(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._grouped_faults.ingest(event, decision)

    # Actions a node-scoped merge may absorb: exactly the ones whose
    # blast radius is the host, not a single GPU or a workload.
    #
    # RESET_GPU qualifies even though it names one GPU, because it
    # quiesces host-wide GPU services and arms a single fail-safe
    # restore timer -- a second reset on the same box would restore
    # services the first still needs held down. REBOOT_NODE obviously
    # qualifies.
    #
    # RESTART_WORKLOAD does not: STOP_WORKLOADS plus RESTART_WORKLOAD
    # touch no host state, so two of them on one node do not collide
    # and merging would only erase the distinction between two GPUs'
    # faults. Everything stronger (REPLACE_NODE, QUARANTINE,
    # ESCALATE_OPERATOR) is a human decision that never executes
    # unattended, so it carries no interrupted-midpoint risk either.
    _NODE_MERGE_ACTIONS = frozenset(
        {
            RecoveryAction.RESET_GPU,
            RecoveryAction.REBOOT_NODE,
        }
    )

    # Steps whose target is the host rather than the workload. On a
    # merged incident these must cover every faulted GPU, not just the
    # GPU of whichever event happened to compile the workflow.
    _NODE_ACTION_OPERATIONS = NODE_ACTION_SCOPE_OPERATIONS

    # Steps already scoped to the workload's whole allocation by the
    # caller. Widening these to the faulted nodes would shrink them.
    _WORKLOAD_SCOPED_OPERATIONS = WORKLOAD_SCOPED_OPERATIONS

    # Steps that take exclusive possession of a node: they stop host
    # GPU services, reset or reboot the box, swap it out, or readmit it
    # to the scheduler. Two workflows running any of these against one
    # node at the same time corrupt each other regardless of which
    # fault detector produced them, so a workflow containing any of
    # them blocks every later workflow aimed at the same node.
    _NODE_EXCLUSIVE_OPERATIONS = NODE_EXCLUSIVE_OPERATIONS

    @classmethod
    def _operation_resource_claims(
        cls,
        operation: WorkflowOperation,
    ) -> frozenset[str]:
        return NodeConflictService.operation_resource_claims(operation)

    @classmethod
    def _workflow_resource_claims_by_node(
        cls,
        workflow: WorkflowRequest,
    ) -> dict[str, frozenset[str]]:
        return NodeConflictService.workflow_resource_claims_by_node(workflow)

    @classmethod
    def _steps_resource_claims_by_node(
        cls,
        steps: list[WorkflowStepSpec],
    ) -> dict[str, frozenset[str]]:
        return NodeConflictService.steps_resource_claims_by_node(steps)

    # Only node reboot/replacement are valid recovery actions for a
    # confirmed missing GPU. RESET_GPU and driver/service mutations can
    # temporarily hide a device, but they cannot safely remediate a PCI
    # function that is already absent. Absorbing such a finding into an
    # in-flight reset also cannot hot-patch the executor's snapshotted
    # validation parameters, so it risks silently missing a real card
    # loss. Queue the REBOOT_NODE candidate instead.
    _TRANSIENT_GPU_INVENTORY_OPERATIONS = TRANSIENT_GPU_INVENTORY_OPERATIONS

    _ACTIVE_WORKFLOW_STATUSES = frozenset(
        {
            WorkflowStatus.PENDING,
            WorkflowStatus.SAFETY_PENDING,
            WorkflowStatus.RUNNING,
        }
    )

    # A merge group whose workflow already reached a terminal state has
    # nothing left to merge into: rewriting a SUCCEEDED workflow's steps
    # changes nothing because the dispatcher never picks it up again.
    _TERMINAL_WORKFLOW_STATUSES = frozenset(
        {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.SUPERSEDED,
        }
    )

    @classmethod
    def _reopen_if_terminal(
        cls,
        existing_incident: FaultIncident | None,
        existing_workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident | None, WorkflowRequest | None]:
        return NodeConflictService.reopen_if_terminal(
            existing_incident,
            existing_workflow,
        )

    @classmethod
    def _claims_node_exclusively(
        cls,
        steps: list[WorkflowStepSpec],
    ) -> bool:
        return NodeConflictService.claims_node_exclusively(steps)

    def _active_node_exclusive_workflow(
        self,
        cluster_id: str,
        node_ids: set[str],
        *,
        exclude_request_ids: frozenset[str] = frozenset(),
        candidate_steps: list[WorkflowStepSpec] | None = None,
    ) -> WorkflowRequest | None:
        return self._conflicts.active_node_exclusive_workflow(
            cluster_id,
            node_ids,
            exclude_request_ids=exclude_request_ids,
            candidate_steps=candidate_steps,
        )

    def _active_workflow_covers_inventory_finding(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._conflicts.active_workflow_covers_inventory_finding(finding)

    @staticmethod
    def _node_group_key(cluster_id: str, node_id: str) -> str:
        return json.dumps(
            ["node-scope", cluster_id, node_id],
            ensure_ascii=True,
            separators=(",", ":"),
        )

    def _ingest_node_scoped_fault(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._node_scoped_faults.ingest(event, decision)

    def _widen_node_action_scope(
        self,
        workflow: WorkflowRequest,
        gpu_uuids_by_node: dict[str, list[str]],
    ) -> WorkflowRequest:
        """Point host-scoped steps at every faulted node and GPU.

        A step compiled from one event names that event's node and GPU.
        Once several faults share the workflow those steps have to cover
        the union, or the merge quietly narrows the remediation to
        whichever fault happened to compile it -- the second card never
        gets reset yet still gets validated and readmitted.

        Node agents accept a wider target only through
        ``parameters.gpu_uuids_by_node``: the fleet adapter reads the
        per-node list and refuses the step outright when a targeted node
        is missing from the mapping, so the mapping and ``node_ids`` are
        written together.

        Steps that neither touch the host nor a workload -- cordon,
        validate, uncordon, node reboot -- take the wider ``node_ids``
        without a GPU mapping, matching what the grouped SXID path
        already does with its ``fault_nodes``. Leaving them narrow would
        cordon one node and readmit it while the other was still being
        reset, and would validate only half the faulted fleet.
        """
        if not gpu_uuids_by_node:
            # Nothing to widen to. Rewriting every node step's ``gpu_uuids``
            # to ``[]`` made the node-action adapter refuse the step
            # outright (P0-57C).
            return workflow
        node_ids = sorted(gpu_uuids_by_node)
        # Nodes that carry no GPU identity join host-level steps only; the
        # node-action adapter refuses a GPU action on a node whose explicit
        # GPU list is empty (P0-57C), so GPU actions take GPU nodes alone.
        gpu_nodes = sorted(node for node, values in gpu_uuids_by_node.items() if values)
        all_gpu_uuids = {
            gpu_uuid for values in gpu_uuids_by_node.values() for gpu_uuid in values
        }
        # Finished, superseded and in-flight steps are history or a command
        # the agent already holds; widening them would make the ledger
        # disagree with what ran (F-B6). Only pending steps take the union.
        untouchable = set(resolved_step_indexes(workflow)) | {
            execution.step_index for execution in workflow.step_executions
        }
        steps = []
        for index, step in enumerate(workflow.official_steps):
            if (
                index in untouchable
                or step.operation in self._WORKLOAD_SCOPED_OPERATIONS
            ):
                steps.append(step)
                continue
            widened: dict[str, object] = {
                "node_ids": sorted(set(step.node_ids) | set(node_ids)),
                "gpu_uuids": sorted(set(step.gpu_uuids) | all_gpu_uuids),
            }
            if step.operation in self._NODE_ACTION_OPERATIONS:
                widened["node_ids"] = sorted(set(step.node_ids) | set(gpu_nodes))
                merged: dict[str, set[str]] = {}
                raw = step.parameters.get("gpu_uuids_by_node")
                if isinstance(raw, dict):
                    for node, values in raw.items():
                        if isinstance(values, list):
                            merged.setdefault(str(node), set()).update(
                                str(value) for value in values
                            )
                for node, values in gpu_uuids_by_node.items():
                    if values:
                        merged.setdefault(node, set()).update(values)
                widened["parameters"] = {
                    **step.parameters,
                    "gpu_uuids_by_node": {
                        node: sorted(values) for node, values in sorted(merged.items())
                    },
                }
            steps.append(step.model_copy(update=widened))
        return workflow.model_copy(update={"official_steps": steps})

    def _runtime_effective_action(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
    ) -> tuple[RecoveryAction | None, str | None]:
        if decision.action is not None:
            return decision.action, None
        if (
            not isinstance(event, XidEvent)
            or decision.official_action != "RESTART_VM"
            or not event.runtime_profile_version
        ):
            return None, None
        try:
            profile = self.store.get_profile(event.runtime_profile_version)
        except NotFoundError:
            return None, None
        if profile.environment not in {
            Environment.HYPERPOD_EKS,
            Environment.HYPERPOD_SLURM,
        }:
            return None, None
        return (
            RecoveryAction.REBOOT_NODE,
            "HyperPod maps NVIDIA RESTART_VM to REBOOT_NODE via "
            "BatchRebootClusterNodes; this is not EC2 stop/start",
        )

    def ingest_node_health(
        self,
        finding: NodeHealthFinding,
        *,
        workflow_request_id: str | None = None,
        _skip_attempt_grouping: bool = False,
        _skip_node_resource_merge: bool = False,
        _skip_terminal_quarantine_merge: bool = False,
        _persist: bool = True,
    ) -> tuple[FaultIncident, WorkflowRequest | None]:
        return self._node_health.ingest(
            finding,
            workflow_request_id=workflow_request_id,
            skip_attempt_grouping=_skip_attempt_grouping,
            skip_node_resource_merge=_skip_node_resource_merge,
            skip_terminal_quarantine_merge=(_skip_terminal_quarantine_merge),
            persist=_persist,
        )

    @staticmethod
    def _is_dcgm_gpu_finding(
        finding: NodeHealthFinding,
    ) -> bool:
        return (
            finding.category.value == "GPU"
            and finding.metric_name is not None
            and (
                "DCGM" in finding.policy_source
                or finding.policy_source
                in {
                    "NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT",
                    "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
                }
            )
        )

    @classmethod
    def _is_attempt_grouped_health_finding(
        cls,
        finding: NodeHealthFinding,
    ) -> bool:
        return (
            cls._is_dcgm_gpu_finding(finding)
            or finding.diagnostic_parameters.get("diagnostic_reason")
            == "EFA_TRAFFIC_HUNG_SUSPECTED"
            or (
                finding.recommended_action is RecoveryAction.QUARANTINE
                and finding.severity.value == "critical"
            )
            or (
                finding.metric_name
                in {
                    "efa_inventory_mismatch",
                    "efa_kubernetes_allocatable_mismatch",
                    "gpu_kubernetes_allocatable_mismatch",
                }
                and finding.recommended_action
                in {
                    RecoveryAction.REMEDIATE_EFA_DRIVER,
                    RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
                    RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
                    RecoveryAction.REBOOT_NODE,
                }
            )
        )

    def _ingest_terminal_node_quarantine(
        self, finding: NodeHealthFinding
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._drain_operations.ingest_terminal_node_quarantine(finding)

    def _ingest_grouped_node_resource_finding(
        self, finding: NodeHealthFinding
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._drain_operations.ingest_grouped_node_resource_finding(finding)

    def _ingest_grouped_health_finding(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._grouped_health.ingest(finding)

    @staticmethod
    def _inventory_validation_parameters(
        finding: NodeHealthFinding,
    ) -> dict:
        return ValidationOperationService.inventory_parameters(finding)

    def _ingest_grouped_node_replacement(
        self, finding: NodeHealthFinding
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        return self._node_lifecycle_operations.ingest_grouped_node_replacement(finding)

    def ingest_distributed_xids(
        self,
        batch: DistributedXidBatch,
        decisions: list[FaultPolicyDecision],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        return self._reset_operations.ingest_distributed_xids(batch, decisions)

    def simulate(
        self, request_id: str, expected_fencing_token: int
    ) -> WorkflowExecutionResult:
        with self._lock:
            workflow = self.store.get_workflow(request_id)
            incident = self.store.get_incident(workflow.incident_id)
            if expected_fencing_token != workflow.fencing_token:
                raise WorkflowFencingError(
                    "stale fencing token: expected "
                    f"{workflow.fencing_token}, got "
                    f"{expected_fencing_token}"
                )

            if workflow.status in {
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.BLOCKED,
            }:
                return WorkflowExecutionResult(
                    workflow_request_id=workflow.request_id,
                    incident_id=incident.incident_id,
                    status=workflow.status,
                    completed_operations=(workflow.completed_operations),
                )

            if workflow.status is WorkflowStatus.PENDING:
                operations = [step.operation for step in workflow.official_steps]
                final_workflow_status = WorkflowStatus.SUCCEEDED
                final_incident_state = IncidentState.RECOVERED
            elif workflow.status is WorkflowStatus.SAFETY_PENDING:
                operations = [step.operation for step in workflow.safety_steps]
                final_workflow_status = WorkflowStatus.BLOCKED
                final_incident_state = IncidentState.QUARANTINED
            else:
                raise ValueError(
                    f"workflow {request_id} is not executable from "
                    f"{workflow.status.value}"
                )

            now = datetime.now(timezone.utc)
            read_workflow = workflow
            read_incident = incident
            workflow = workflow.model_copy(
                update={
                    "status": final_workflow_status,
                    "completed_operations": operations,
                    "updated_at": now,
                }
            )
            incident = incident.model_copy(
                update={
                    "state": final_incident_state,
                    "updated_at": now,
                }
            )
            # Compare-and-set on the copy read above: a merge or lease change
            # by another replica between the read and this write surfaces as
            # ``StaleWriteError`` instead of being overwritten. It propagates,
            # like ``WorkflowFencingError`` from the same method: the caller
            # re-reads and retries, this method holds no state to retry with.
            # ``save_incident_and_workflow`` is not used because it has no
            # ``expected`` and stamps a merge revision on an existing row
            # (store review 2026-09-07, item B).
            self.store.save_workflow(workflow, expected=read_workflow)
            self.store.save_incident(incident, expected=read_incident)
            return WorkflowExecutionResult(
                workflow_request_id=workflow.request_id,
                incident_id=incident.incident_id,
                status=workflow.status,
                completed_operations=operations,
            )

    def apply_fault_action_generation_fence(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        *,
        now: datetime | None = None,
    ) -> FaultPolicyDecision:
        """Keep stale fault evidence without mutating the current node.

        Queue retention and audit semantics must not discard XID/SXID
        records. Recovery that acts on the node is only safe when the
        event belongs to the current boot generation; containment and
        support escalation stay outside the fence on purpose.
        """

        if decision.disposition is not ActionDisposition.EXECUTABLE:
            return decision
        operations = self._builder.official_operations(event, decision, decision.action)
        if not self._builder.mutates_node(operations):
            return decision

        observed_now = self._utc(now or datetime.now(timezone.utc))
        try:
            agent = self.store.get_agent(event.cluster_id, event.node_id)
        except NotFoundError:
            # The generation fence needs an authoritative current
            # incarnation. Deployments without an Agent still rely on the
            # existing runtime owner/capability gates.
            return decision

        blocked_reasons: list[str] = []
        if event.source_event_time is not None:
            event_time = self._utc(event.source_event_time)
            age_seconds = max(
                0.0,
                (observed_now - event_time).total_seconds(),
            )
            if age_seconds > self.fault_action_max_age_seconds:
                blocked_reasons.append(
                    "STALE_FAULT_GENERATION: source event age "
                    f"{age_seconds:.3f}s exceeds automatic action limit "
                    f"{self.fault_action_max_age_seconds}s"
                )

        if event.source_boot_id:
            lease_is_fresh = (
                agent.lease_expires_at is not None
                and agent.lease_expires_at > observed_now
            )
            lifecycle = getattr(
                agent.lifecycle_state,
                "value",
                str(agent.lifecycle_state),
            )
            if not lease_is_fresh or lifecycle != "ACTIVE":
                blocked_reasons.append(
                    "STALE_FAULT_GENERATION: current Agent is not "
                    "ACTIVE with a fresh heartbeat"
                )
            elif not agent.boot_id:
                blocked_reasons.append(
                    "STALE_FAULT_GENERATION: current Agent did not report a boot ID"
                )
            elif agent.boot_id != event.source_boot_id:
                blocked_reasons.append(
                    "STALE_FAULT_GENERATION: event boot ID "
                    f"{event.source_boot_id} does not match current "
                    f"Agent boot ID {agent.boot_id}"
                )

        if not blocked_reasons:
            return decision
        reason_values = list(dict.fromkeys([*decision.reasons, *blocked_reasons]))
        marker = decision.marker.model_copy(
            update={
                "trusted": False,
                "active": False,
                "recommended_action": None,
                "site_safety_action": None,
                "action_disposition": (
                    ActionDisposition.BLOCKED_MISSING_EVIDENCE.value
                ),
                "raw_reason": "; ".join(blocked_reasons),
            }
        )
        return decision.model_copy(
            update={
                "disposition": (ActionDisposition.BLOCKED_MISSING_EVIDENCE),
                "action": None,
                "safety_action": None,
                "pre_actions": [],
                "requires_operator": True,
                "reasons": reason_values,
                "marker": marker,
            }
        )

    def _build_workflow(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        incident: FaultIncident,
    ) -> WorkflowRequest:
        workload_ids = list(event.affected_workload_ids)
        node_ids = [event.node_id]
        gpu_uuids = decision.marker.scope.gpu_uuids
        blocked_reasons: list[str] = []

        safety_operations = (
            [WorkflowOperation.FREEZE_EVIDENCE]
            if (
                decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
                and decision.safety_action is None
            )
            else [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ]
        )
        official_operations = self._builder.official_operations(
            event, decision, incident.effective_action
        )

        profile = None
        if event.runtime_profile_version:
            try:
                profile = self.store.get_profile(event.runtime_profile_version)
            except NotFoundError:
                blocked_reasons.append(
                    "runtime profile does not exist: " + event.runtime_profile_version
                )
        else:
            blocked_reasons.append("runtime_profile_version is required for execution")

        # The UNKNOWN-state refusal moved into ``compile_steps`` so the
        # node-health family shares it (ARCH-B2); the official compile below
        # is handed the event's workload state. The ACTIVE check stays here:
        # it is about the event naming its workloads, not about the plan.
        if (
            self._builder.mutates_node(official_operations)
            and event.workload_state is WorkloadState.ACTIVE
            and not workload_ids
        ):
            blocked_reasons.append(
                "ACTIVE workload state requires affected_workload_ids"
            )

        safety_steps, safety_errors = self._builder.compile_steps(
            safety_operations,
            profile,
            node_ids,
            gpu_uuids,
            workload_ids,
        )
        official_steps, official_errors = self._builder.compile_steps(
            official_operations,
            profile,
            node_ids,
            gpu_uuids,
            workload_ids,
            workload_state=event.workload_state,
        )
        official_steps = [
            step.model_copy(
                update={
                    "parameters": self._builder.step_parameters(
                        step.operation,
                        event,
                        incident,
                        decision,
                    )
                }
            )
            for step in official_steps
        ]
        blocked_reasons.extend(official_errors)
        blocked_reasons.extend(
            self._builder.workflow_evidence_errors(event, official_operations)
        )

        hard_policy_block = decision.disposition in {
            ActionDisposition.BLOCKED_MISSING_EVIDENCE,
            ActionDisposition.NOT_APPLICABLE,
            ActionDisposition.SITE_SAFETY,
        }
        workflow_unresolved = (
            decision.disposition is ActionDisposition.BLOCKED_WORKFLOW
            and (len(official_operations) <= 1 or bool(official_errors))
        )
        if hard_policy_block or workflow_unresolved or blocked_reasons:
            blocked_reasons.extend(decision.reasons)
            status = (
                WorkflowStatus.SAFETY_PENDING
                if safety_steps and not safety_errors
                else WorkflowStatus.BLOCKED
            )
            blocked_reasons.extend(safety_errors)
        else:
            status = WorkflowStatus.PENDING

        return WorkflowRequest(
            incident_id=incident.incident_id,
            runtime_profile_version=event.runtime_profile_version,
            status=status,
            official_action=decision.official_action,
            fencing_token=incident.fencing_token,
            safety_steps=safety_steps,
            official_steps=official_steps,
            blocked_reasons=list(dict.fromkeys(blocked_reasons)),
            safety_only=status is WorkflowStatus.SAFETY_PENDING,
            blocked_kind=(
                BlockedKind.NEEDS_OPERATOR if status is WorkflowStatus.BLOCKED else None
            ),
        )
