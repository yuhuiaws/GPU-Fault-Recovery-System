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
        """The execution owner a profile assigns to ``action``, or a named refusal.

        Fail-closed and order-insensitive (F-G5): an action without a capability
        mapping is refused rather than handed to a phantom owner; an explicit
        ``DISABLED`` claim wins over an ``OWN`` claim for the same capability no
        matter which is listed first; two different executable owners are an
        ambiguity the planner will not resolve by list position.
        """
        capability = ACTION_CAPABILITY.get(action)
        if capability is None:
            raise UnsupportedPlanError(
                f"{action.value} has no capability mapping; refusing to plan it"
            )
        claims = [
            item for item in profile.capabilities if item.capability is capability
        ]
        if any(item.mode is CapabilityMode.DISABLED for item in claims):
            raise UnsupportedPlanError(
                f"{capability.value} is disabled in profile "
                f"{profile.profile_version}; cannot plan {action.value}"
            )
        owners = sorted(
            {
                item.owner
                for item in claims
                if item.mode in {CapabilityMode.OWN, CapabilityMode.DELEGATE}
            }
        )
        if not owners:
            raise UnsupportedPlanError(
                f"no executable owner for {action.value} in "
                f"profile {profile.profile_version}"
            )
        if len(owners) > 1:
            raise UnsupportedPlanError(
                f"ambiguous executable owner for {action.value} in profile "
                f"{profile.profile_version}: {', '.join(owners)}"
            )
        return owners[0]

    def _optional_step(
        self,
        profile: EffectiveRuntimeProfile,
        action: RecoveryAction,
        node_ids: list[str],
        gpu_uuids: list[str] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> PlanStep | None:
        """A step the plan can do without when the profile cannot execute it.

        Auxiliary steps (evidence, cordon, validation, restart) used to be
        built with the same strictness as the primary action, so a profile
        missing one auxiliary capability turned every plan into a 422 and the
        fault went unhandled (F-G5).
        """
        try:
            return self._step(profile, action, node_ids, gpu_uuids, parameters)
        except UnsupportedPlanError:
            return None

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
                self._destructive_steps(event, profile, action, node_ids, gpu_uuids)
            )
        elif action is RecoveryAction.RUN_DIAGNOSTICS:
            evidence = self._optional_step(
                profile, RecoveryAction.COLLECT_EVIDENCE, node_ids, gpu_uuids
            )
            if evidence is not None:
                steps.append(evidence)
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.RUN_DIAGNOSTICS,
                    node_ids,
                    gpu_uuids,
                )
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
                # Containment still happens without the stop; the cordon and
                # the quarantine are what keep new work off the node.
                stop = self._optional_step(
                    profile, RecoveryAction.STOP_WORKLOAD, node_ids
                )
                if stop is not None:
                    steps.append(stop)
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
            for auxiliary in (
                RecoveryAction.MARK_UNSCHEDULABLE,
                RecoveryAction.QUARANTINE,
            ):
                step = self._optional_step(profile, auxiliary, node_ids)
                if step is not None:
                    steps.append(step)
            steps.append(
                self._step(
                    profile,
                    RecoveryAction.ESCALATE_OPERATOR,
                    node_ids,
                )
            )
        else:
            raise UnsupportedPlanError(f"unsupported recovery action: {action.value}")

        return RecoveryPlan(
            # An observational marker (written before ingestion picked an
            # incident, and left that way if ingestion failed) has no incident
            # pointer; the plan mints its own rather than compiling an
            # incident with an empty id (F-G6).
            incident_id=marker.incident_id or f"inc-{uuid4()}",
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

    def _destructive_steps(
        self,
        event: TerminalEvent,
        profile: EffectiveRuntimeProfile,
        action: RecoveryAction,
        node_ids: list[str],
        gpu_uuids: list[str],
    ) -> list[PlanStep]:
        """Cordon, evidence, stop, the action, validation, release, restart.

        Only the action itself and the stop of a live workload are strict: the
        first because it is the plan, the second because running a destructive
        action under a workload nobody can stop is unsafe -- if the profile
        cannot stop it, the node is contained and escalated instead. Evidence,
        validation and the restart are dropped when the profile lacks them;
        the release step is planned exactly when the cordon was.
        """
        steps: list[PlanStep] = []
        cordon = self._optional_step(
            profile, RecoveryAction.MARK_UNSCHEDULABLE, node_ids
        )
        if cordon is not None:
            steps.append(cordon)
        evidence = self._optional_step(
            profile, RecoveryAction.COLLECT_EVIDENCE, node_ids
        )
        if evidence is not None:
            steps.append(evidence)
        if event.workload_ids:
            stop = self._optional_step(profile, RecoveryAction.STOP_WORKLOAD, node_ids)
            if stop is None:
                steps.extend(
                    self._fallback_steps(
                        profile,
                        node_ids,
                        blocked_action=action,
                        reason=(
                            "STOP_WORKLOAD is unavailable; a destructive action "
                            "cannot run under a live workload"
                        ),
                        include_mark=False,
                    )
                )
                return steps
            steps.append(stop)
        try:
            action_step = self._step(profile, action, node_ids, gpu_uuids)
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
            return steps
        steps.append(action_step)
        validate = self._optional_step(profile, RecoveryAction.VALIDATE_NODE, node_ids)
        if validate is not None:
            steps.append(validate)
        if cordon is not None:
            steps.append(
                self._step(profile, RecoveryAction.RESTORE_SCHEDULING, node_ids)
            )
        if event.workload_ids:
            restart = self._optional_step(
                profile,
                RecoveryAction.RESTART_WORKLOAD,
                node_ids,
                parameters={"reuse_allocation": False},
            )
            if restart is not None:
                steps.append(restart)
        return steps

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
            # Each failed node gets its own most-severe action and its own
            # GPUs. One marker for all failed nodes applied the single most
            # severe action, and every GPU of every failed node, to each of
            # them (F-G5).
            actions_by_node: dict[str, RecoveryAction] = {}
            for finding in failed:
                proposed = finding.proposed_action or RecoveryAction.QUARANTINE
                current = actions_by_node.get(finding.node_id)
                if current is None or recovery_action_sort_key(
                    proposed
                ) > recovery_action_sort_key(current):
                    actions_by_node[finding.node_id] = proposed
            plans = [
                self.from_marker(
                    event,
                    NodeMarker(
                        cluster_id=event.cluster_id,
                        source="quick-triage",
                        trusted=True,
                        incident_id=incident_id,
                        observed_at=event.ended_at,
                        expires_at=event.ended_at + timedelta(days=365),
                        scope={
                            "node_ids": [node_id],
                            "gpu_uuids": sorted(
                                {
                                    gpu
                                    for allocation in event.allocation
                                    if allocation.node_id == node_id
                                    for gpu in allocation.gpu_uuids
                                }
                            ),
                        },
                        severity="critical",
                        recommended_action=action,
                        action_owner="core-policy",
                        mapping_version="quick-triage-v1",
                    ),
                    profile,
                )
                for node_id, action in sorted(actions_by_node.items())
            ]
            if len(plans) == 1:
                return plans[0]
            return self._merge_node_plans(event, incident_id, plans)

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

    def _merge_node_plans(
        self,
        event: TerminalEvent,
        incident_id: str,
        plans: list[RecoveryPlan],
    ) -> RecoveryPlan:
        """One plan from per-node plans: node steps in order, one restart last."""
        steps: list[PlanStep] = []
        restart: PlanStep | None = None
        for plan in plans:
            for step in plan.steps:
                if step.action is RecoveryAction.RESTART_WORKLOAD:
                    restart = restart or step
                    continue
                steps.append(step)
        avoid = sorted({node for plan in plans for node in plan.avoid_node_ids})
        if restart is not None:
            steps.append(
                restart.model_copy(
                    update={
                        "node_ids": sorted(
                            {node for plan in plans for node in plan.steps[0].node_ids}
                        )
                    }
                )
            )
        return RecoveryPlan(
            incident_id=incident_id,
            attempt_id=event.attempt_id,
            trigger="quick-triage:FAIL",
            runtime_profile_version=plans[0].runtime_profile_version,
            steps=steps,
            avoid_node_ids=avoid,
            checkpoint_manifest_ref=event.checkpoint_manifest_ref,
        )

    def after_incident(
        self,
        event: TerminalEvent,
        incident: FaultIncident,
        profile: EffectiveRuntimeProfile,
    ) -> RecoveryPlan:
        """Restart the attempt once the incident that owns its nodes is repaired.

        The restart is gated on the incident being ``RECOVERED`` *at execution
        time* (the adapter waits until it is, and gives up if it can never be).
        Freezing ``incident.state`` here froze two execution-time decisions at
        plan time: an ``ACTION_PENDING`` incident produced a plan that, once the
        repair succeeded, restarted the job everywhere except on the node that
        had just been repaired, and without its allocation (F-G5).
        """
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
                        "reuse_allocation": True,
                        "requires_incident_state": "RECOVERED",
                        "incident_id": incident.incident_id,
                        "incident_node_ids": sorted(incident.node_ids),
                    },
                )
            ],
            avoid_node_ids=[],
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
