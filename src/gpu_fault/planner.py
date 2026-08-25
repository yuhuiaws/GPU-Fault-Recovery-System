from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    EffectiveRuntimeProfile,
    FaultIncident,
    NodeMarker,
    PlanStep,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    TriageFinding,
    TriageOutcome,
    recovery_action_sort_key,
)


ACTION_CAPABILITY = {
    RecoveryAction.MARK_UNSCHEDULABLE: CapabilityName.SCHEDULER_DRAIN,
    RecoveryAction.DRAIN: CapabilityName.SCHEDULER_DRAIN,
    RecoveryAction.QUARANTINE: CapabilityName.SCHEDULER_DRAIN,
    RecoveryAction.STOP_WORKLOAD: CapabilityName.WORKLOAD_STOP,
    RecoveryAction.COLLECT_EVIDENCE: CapabilityName.EVIDENCE_CAPTURE,
    RecoveryAction.RESTART_WORKLOAD: CapabilityName.WORKLOAD_RESTART,
    RecoveryAction.RESET_GPU: CapabilityName.GPU_RESET,
    RecoveryAction.REBOOT_NODE: CapabilityName.NODE_REBOOT,
    RecoveryAction.REPLACE_NODE: CapabilityName.NODE_REPLACE,
    RecoveryAction.REMEDIATE_EFA_DRIVER: (CapabilityName.EFA_DRIVER_REMEDIATION),
    RecoveryAction.RESTART_EFA_DEVICE_PLUGIN: (CapabilityName.SCHEDULER_DRAIN),
    RecoveryAction.RESTART_GPU_DEVICE_PLUGIN: (CapabilityName.SCHEDULER_DRAIN),
    RecoveryAction.RUN_DIAGNOSTICS: CapabilityName.DEEP_DIAGNOSTICS,
    RecoveryAction.VALIDATE_NODE: CapabilityName.DEEP_DIAGNOSTICS,
    RecoveryAction.RESTORE_SCHEDULING: (CapabilityName.SCHEDULER_DRAIN),
    RecoveryAction.ESCALATE_OPERATOR: (CapabilityName.SUPPORT_ESCALATION),
}


class UnsupportedPlanError(ValueError):
    pass


class PlanBuilder:
    def _owner(
        self,
        profile: EffectiveRuntimeProfile,
        action: RecoveryAction,
    ) -> str:
        capability = ACTION_CAPABILITY.get(action)
        if capability is None:
            return "core-orchestrator"
        for item in profile.capabilities:
            if item.capability is capability:
                if item.mode in {
                    CapabilityMode.OWN,
                    CapabilityMode.DELEGATE,
                }:
                    return item.owner
                break
        raise UnsupportedPlanError(
            f"no executable owner for {action.value} in "
            f"profile {profile.profile_version}"
        )

    def _step(
        self,
        profile: EffectiveRuntimeProfile,
        action: RecoveryAction,
        node_ids: list[str],
        gpu_uuids: list[str] | None = None,
        parameters: dict | None = None,
    ) -> PlanStep:
        return PlanStep(
            action=action,
            node_ids=node_ids,
            gpu_uuids=gpu_uuids or [],
            execution_owner=self._owner(profile, action),
            parameters=parameters or {},
        )

    def _fallback_steps(
        self,
        profile: EffectiveRuntimeProfile,
        node_ids: list[str],
        *,
        blocked_action: RecoveryAction,
        reason: str,
        include_mark: bool,
    ) -> list[PlanStep]:
        steps = []
        for action in (
            *((RecoveryAction.MARK_UNSCHEDULABLE,) if include_mark else ()),
            RecoveryAction.QUARANTINE,
            RecoveryAction.ESCALATE_OPERATOR,
        ):
            try:
                steps.append(
                    self._step(
                        profile,
                        action,
                        node_ids,
                        parameters={
                            "blocked_action": blocked_action.value,
                            "fallback_reason": reason,
                        },
                    )
                )
            except UnsupportedPlanError:
                continue
        if not steps:
            raise UnsupportedPlanError(
                f"{blocked_action.value} is unavailable and no "
                "containment/support capability can execute"
            )
        return steps

    def from_marker(
        self,
        event: TerminalEvent,
        marker: NodeMarker,
        profile: EffectiveRuntimeProfile,
    ) -> RecoveryPlan:
        action = marker.recommended_action or RecoveryAction.QUARANTINE
        explicit_node_ids = sorted(set(marker.scope.node_ids))
        allocation_node_ids = sorted({item.node_id for item in event.allocation})
        workload_only_actions = {
            RecoveryAction.NO_ACTION,
            RecoveryAction.RESTART_WORKLOAD,
        }
        if not explicit_node_ids and action not in workload_only_actions:
            raise UnsupportedPlanError(
                f"{action.value} marker requires explicit node scope"
            )
        node_ids = explicit_node_ids or allocation_node_ids
        gpu_uuids = sorted(set(marker.scope.gpu_uuids))
        if action is RecoveryAction.RESET_GPU and not gpu_uuids:
            raise UnsupportedPlanError("RESET_GPU marker requires explicit GPU scope")
        steps: list[PlanStep] = []

        if action in {
            RecoveryAction.NO_ACTION,
            RecoveryAction.RESTART_WORKLOAD,
        }:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.RESTART_WORKLOAD,
                    node_ids,
                    parameters={"reuse_allocation": True},
                )
            )
        elif action in {
            RecoveryAction.RESET_GPU,
            RecoveryAction.REBOOT_NODE,
            RecoveryAction.REPLACE_NODE,
            RecoveryAction.REMEDIATE_EFA_DRIVER,
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        }:
            steps.extend(
                [
                    self._step(
                        profile,
                        RecoveryAction.MARK_UNSCHEDULABLE,
                        node_ids,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.COLLECT_EVIDENCE,
                        node_ids,
                    ),
                ]
            )
            if event.workload_ids:
                steps.append(
                    self._step(
                        profile,
                        RecoveryAction.STOP_WORKLOAD,
                        node_ids,
                    )
                )
            try:
                action_step = self._step(
                    profile,
                    action,
                    node_ids,
                    gpu_uuids,
                )
            except UnsupportedPlanError as exc:
                steps.extend(
                    self._fallback_steps(
                        profile,
                        node_ids,
                        blocked_action=action,
                        reason=str(exc),
                        include_mark=False,
                    )
                )
                action_step = None
            if action_step is not None:
                steps.extend(
                    [
                        action_step,
                        self._step(
                            profile,
                            RecoveryAction.VALIDATE_NODE,
                            node_ids,
                        ),
                        self._step(
                            profile,
                            RecoveryAction.RESTORE_SCHEDULING,
                            node_ids,
                        ),
                    ]
                )
            if event.workload_ids:
                if action_step is not None:
                    steps.append(
                        self._step(
                            profile,
                            RecoveryAction.RESTART_WORKLOAD,
                            node_ids,
                            parameters={"reuse_allocation": False},
                        )
                    )
        elif action is RecoveryAction.RUN_DIAGNOSTICS:
            steps.extend(
                [
                    self._step(
                        profile,
                        RecoveryAction.COLLECT_EVIDENCE,
                        node_ids,
                        gpu_uuids,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.RUN_DIAGNOSTICS,
                        node_ids,
                        gpu_uuids,
                    ),
                ]
            )
        elif action is RecoveryAction.MARK_UNSCHEDULABLE:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    node_ids,
                )
            )
        elif action is RecoveryAction.DRAIN:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    node_ids,
                )
            )
            if event.workload_ids:
                steps.append(
                    self._step(
                        profile,
                        RecoveryAction.STOP_WORKLOAD,
                        node_ids,
                    )
                )
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.QUARANTINE,
                    node_ids,
                )
            )
        elif action is RecoveryAction.STOP_WORKLOAD:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.STOP_WORKLOAD,
                    node_ids,
                )
            )
        elif action is RecoveryAction.COLLECT_EVIDENCE:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.COLLECT_EVIDENCE,
                    node_ids,
                    gpu_uuids,
                )
            )
        elif action is RecoveryAction.VALIDATE_NODE:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.VALIDATE_NODE,
                    node_ids,
                    gpu_uuids,
                )
            )
        elif action is RecoveryAction.RESTORE_SCHEDULING:
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.RESTORE_SCHEDULING,
                    node_ids,
                )
            )
        elif action is RecoveryAction.QUARANTINE:
            steps.extend(
                [
                    self._step(
                        profile,
                        RecoveryAction.MARK_UNSCHEDULABLE,
                        node_ids,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.QUARANTINE,
                        node_ids,
                    ),
                ]
            )
        elif action is RecoveryAction.ESCALATE_OPERATOR:
            steps.extend(
                [
                    self._step(
                        profile,
                        RecoveryAction.MARK_UNSCHEDULABLE,
                        node_ids,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.QUARANTINE,
                        node_ids,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.ESCALATE_OPERATOR,
                        node_ids,
                    ),
                ]
            )
        else:
            raise UnsupportedPlanError(f"unsupported recovery action: {action.value}")

        return RecoveryPlan(
            incident_id=marker.incident_id,
            attempt_id=event.attempt_id,
            trigger=f"marker:{marker.marker_id}",
            runtime_profile_version=profile.profile_version,
            steps=steps,
            avoid_node_ids=(
                node_ids
                if action
                not in {
                    RecoveryAction.NO_ACTION,
                    RecoveryAction.RESTART_WORKLOAD,
                }
                else []
            ),
            checkpoint_manifest_ref=event.checkpoint_manifest_ref,
            drill_id=marker.drill_id,
        )

    def from_triage(
        self,
        event: TerminalEvent,
        findings: list[TriageFinding],
        profile: EffectiveRuntimeProfile,
    ) -> RecoveryPlan:
        outcomes = {finding.outcome for finding in findings}
        failed = [
            finding for finding in findings if finding.outcome is TriageOutcome.FAIL
        ]
        inconclusive = [
            finding
            for finding in findings
            if finding.outcome is TriageOutcome.INCONCLUSIVE
        ]
        incident_id = f"inc-{uuid4()}"

        if failed:
            failed_nodes = {finding.node_id for finding in failed}
            action = max(
                (
                    finding.proposed_action or RecoveryAction.QUARANTINE
                    for finding in failed
                ),
                key=recovery_action_sort_key,
            )
            marker = NodeMarker(
                source="quick-triage",
                trusted=True,
                incident_id=incident_id,
                observed_at=event.ended_at,
                expires_at=event.ended_at + timedelta(days=365),
                scope={
                    "node_ids": sorted(failed_nodes),
                    "gpu_uuids": sorted(
                        {
                            gpu
                            for allocation in event.allocation
                            if allocation.node_id in failed_nodes
                            for gpu in allocation.gpu_uuids
                        }
                    ),
                },
                severity="critical",
                recommended_action=action,
                action_owner="core-policy",
                mapping_version="quick-triage-v1",
            )
            return self.from_marker(event, marker, profile)

        node_ids = sorted({item.node_id for item in event.allocation})
        if inconclusive:
            suspect_nodes = sorted({finding.node_id for finding in inconclusive})
            return RecoveryPlan(
                incident_id=incident_id,
                attempt_id=event.attempt_id,
                trigger="quick-triage:INCONCLUSIVE",
                runtime_profile_version=profile.profile_version,
                steps=[
                    self._step(
                        profile,
                        RecoveryAction.MARK_UNSCHEDULABLE,
                        suspect_nodes,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.QUARANTINE,
                        suspect_nodes,
                    ),
                    self._step(
                        profile,
                        RecoveryAction.RESTART_WORKLOAD,
                        node_ids,
                        parameters={"reuse_allocation": False},
                    ),
                ],
                avoid_node_ids=suspect_nodes,
                checkpoint_manifest_ref=event.checkpoint_manifest_ref,
            )

        if outcomes != {TriageOutcome.PASS}:
            raise ValueError("triage report has no usable outcome")
        return RecoveryPlan(
            incident_id=incident_id,
            attempt_id=event.attempt_id,
            trigger="quick-triage:PASS",
            runtime_profile_version=profile.profile_version,
            steps=[
                self._step(
                    profile,
                    RecoveryAction.RESTART_WORKLOAD,
                    node_ids,
                    parameters={"reuse_allocation": True},
                )
            ],
            checkpoint_manifest_ref=event.checkpoint_manifest_ref,
        )

    def after_incident(
        self,
        event: TerminalEvent,
        incident: FaultIncident,
        profile: EffectiveRuntimeProfile,
    ) -> RecoveryPlan:
        allocation_nodes = sorted({item.node_id for item in event.allocation})
        return RecoveryPlan(
            incident_id=incident.incident_id,
            attempt_id=event.attempt_id,
            trigger=f"incident:{incident.incident_id}",
            runtime_profile_version=profile.profile_version,
            steps=[
                self._step(
                    profile,
                    RecoveryAction.RESTART_WORKLOAD,
                    allocation_nodes,
                    parameters={
                        "reuse_allocation": (incident.state.value == "RECOVERED"),
                        "requires_incident_state": "RECOVERED",
                        "incident_id": incident.incident_id,
                    },
                )
            ],
            avoid_node_ids=(
                [] if incident.state.value == "RECOVERED" else incident.node_ids
            ),
            checkpoint_manifest_ref=event.checkpoint_manifest_ref,
        )

    def from_missing_allocation(
        self,
        event: TerminalEvent,
        profile: EffectiveRuntimeProfile,
    ) -> RecoveryPlan:
        return RecoveryPlan(
            incident_id=f"inc-{uuid4()}",
            attempt_id=event.attempt_id,
            trigger="allocation-missing:INCONCLUSIVE",
            runtime_profile_version=profile.profile_version,
            steps=[
                self._step(
                    profile,
                    RecoveryAction.COLLECT_EVIDENCE,
                    [],
                    parameters={"reason": "allocation snapshot is unavailable"},
                ),
                self._step(
                    profile,
                    RecoveryAction.ESCALATE_OPERATOR,
                    [],
                    parameters={
                        "automatic_restart_blocked": True,
                        "required_evidence": [
                            "scheduler-allocation-history",
                            "rank-to-node-mapping",
                        ],
                    },
                ),
            ],
            checkpoint_manifest_ref=event.checkpoint_manifest_ref,
        )
