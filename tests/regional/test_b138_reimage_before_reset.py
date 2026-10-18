"""A HyperPod in-place re-image before the GPU reset of an in-flight remediation.

Batch 138, analysis lane (shape A, points 1-4). SageMaker
``UpdateClusterSoftware`` (or an instance group's ``ScheduledUpdateConfig``)
re-images a node in place: the Node object, its UID, name, annotations, taints
and cordon are kept (``node_installer_reconciler.py:39-42``), the boot id
changes, and the Node Agent and its on-node ledger are gone until the
installer reconciler reinstalls the Agent as a new incarnation that registers
as a same-instance reboot with ``generation + 1`` (``fleet.py:515-647``).

What the product does, and where it is still wrong:

* After a contained STOP (receipt binds node UID + boot id) the next ownership
  guarded node action is refused ``STOP_OWNERSHIP_DRIFT`` before anything
  reaches the new Agent, attested "not started"; the hand-off plans evidence
  and a ticket only (safe; pinned below).
* A ``VERIFY_NO_GPU_CLIENTS`` after the re-image is fenced by the quiesce's
  agent generation (safe; pinned below).
* A QUIESCE the Agent had accepted was recorded as a plain FAILED once the
  Agent's lease expired: the node action's fleet readiness gate runs before
  the accepted command is polled and erased its unknown outcome (batch 138
  xfail; since batch 139 the gate's FAILED keeps it).
* ``confirm-node-action`` settled a parked node action by "a reboot since the
  action" against the incident's *fault-time* boot: a re-image between the
  fault and the STOP was accepted as a reboot after an action that ran on the
  current boot, and a multi-node incident had no baseline at all, although the
  STOP receipt pins the boot every guarded action ran on (batch 138 xfails;
  since batch 139 the receipt, carried along the confirmed reboots, is the
  baseline).
* A QUARANTINED incident whose record parked with an unresolved node action
  still has a supported exit: ``restore`` then ``confirm-node-action`` on the
  validated restore (pinned below).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    node_submission_ownership_guard,
    stop_ownership_scope,
)
from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.admin.remediation_plans import build_restore_plan
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    confirm_node_action,
)
from gpu_fault.execution.node_action_uncertainty import unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import (
    NODE_ACTION_REFUSED_STAGE,
    HardwareEscalationService,
    node_action_refused_before_acting,
)
from tests._builders import (
    contained_stop_receipt,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import StubFleetRegistry
from tests.regional._late_ownership_runtime import WORKLOAD, stopped_runtime

OP = WorkflowOperation
AGENT_OWNER = "gpu-fault-node-agent"
QUIESCE_KEY = "workflow-local/1/QUIESCE_GPU_SERVICES"
NEW_BOOT = "node-a-boot-after-reimage"
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
# A port nothing listens on: an Agent the re-image took away refuses the
# connection (only reached if a fix polls the accepted command).
GONE_AGENT = "http://127.0.0.1:1"


def _quiesce_context(context: WorkflowStepContext, *executions: WorkflowStepExecution):
    """The QUIESCE that follows the contained STOP of ``stopped_runtime``."""

    stop = context.workflow.official_steps[0]
    quiesce = WorkflowStepSpec(
        operation=OP.QUIESCE_GPU_SERVICES,
        execution_owner=AGENT_OWNER,
        node_ids=["node-a"],
        gpu_uuids=["GPU-local"],
        workload_ids=[WORKLOAD],
    )
    workflow = context.workflow.model_copy(
        update={
            "official_steps": [stop, quiesce],
            "step_executions": [*context.workflow.step_executions, *executions],
        }
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=context.incident,
        step=quiesce,
        step_index=1,
        request=context.request,
        idempotency_key=QUIESCE_KEY,
    )


def _reimage(state) -> None:
    """``UpdateClusterSoftware``: same Node object and UID, a new boot."""

    state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = NEW_BOOT


def test_a_reimage_after_the_stop_refuses_the_quiesce_before_the_new_agent() -> None:
    """Point 2, re-image finished: drift is detected and attested "not started".

    The guard re-reads every receipt node; a changed boot id on the same UID
    without a product reboot record is ``STOP_OWNERSHIP_DRIFT``
    (``stop_ownership.py:612-685``, ``stop_boot_transition.py:309-387``), the
    refusal says nothing started (``stop_ownership.py:181-194``) and the
    transport returns it before the sender is called
    (``adapters/node_action/transport.py`` ``guard_new_node_submission``).
    Both ladders read it as a refusal (``escalation.py:188-230``): the
    hand-off is evidence plus a ticket, no reboot, no quarantine rung.
    """

    state, _kubernetes, validator, reset = stopped_runtime()
    context = _quiesce_context(reset)
    with stop_ownership_scope(validator):
        assert node_submission_ownership_guard(context) is None, (
            "control: the unchanged boot authorizes the quiesce"
        )
    _reimage(state)
    sent = []
    # The new incarnation registered: fleet-ready, generation bumped.
    registry = StubFleetRegistry({"node-a": GONE_AGENT}, generation=8)
    adapter = NodeActionWorkflowAdapter(
        {},
        "s" * 32,
        registry=registry,
        sender=lambda endpoint, envelope: sent.append(envelope),
    )

    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["reason"] == "STOP_OWNERSHIP_DRIFT", outcome.details
    assert outcome.details["node_action_not_started"] is True
    assert sent == [], "nothing reaches the re-imaged node's new Agent"
    assert node_action_refused_before_acting(outcome.details), (
        "the drift refusal is an attested no-start, not an unknown outcome"
    )
    record = WorkflowStepExecution(
        step_index=1,
        operation=OP.QUIESCE_GPU_SERVICES,
        status=WorkflowStepStatus.FAILED,
        phase="official",
        error=outcome.error,
        details=outcome.details,
    )
    assert unresolved_node_action(record) is False, (
        "a refusal does not park the record NEEDS_OPERATOR"
    )
    failed = context.workflow.model_copy(
        update={
            "status": WorkflowStatus.FAILED,
            "step_executions": [*context.workflow.step_executions, record],
        }
    )
    classification = HardwareEscalationService.classify(failed)
    assert classification is not None
    assert classification[0] == NODE_ACTION_REFUSED_STAGE, classification


def test_an_accepted_quiesce_whose_agent_vanished_stays_unresolved() -> None:
    """Point 3, re-image in progress after the Agent accepted the QUIESCE.

    The step's record shows the Agent accepted the command (a parsed PENDING:
    ``node_action_accepted_nodes``). The Agent is gone, its lease expired. The
    cluster executor itself skips its destructive preflight for an accepted
    command (``cluster_executor/dispatch.py:472-510``: "the work lives in the
    agent's ledger and cannot be called back"); the adapter's own readiness
    gate still answers first, and before batch 139 its plain FAILED replaced
    the PENDING record wholesale (``execution/step_bounds.py:352-415``). Now
    the gate's FAILED keeps ``outcome_unknown``, ``manual_confirmation_required``
    and the command pointer (``step_execution.keep_accepted_uncertainty``).
    """

    _state, _kubernetes, validator, reset = stopped_runtime()
    accepted = WorkflowStepExecution(
        step_index=1,
        operation=OP.QUIESCE_GPU_SERVICES,
        status=WorkflowStepStatus.WAITING,
        phase="official",
        adapter_operation_id=QUIESCE_KEY,
        details={
            "node_action_command_id": f"{QUIESCE_KEY}/node-a/agent-7",
            "node_action_state": "PENDING",
            "node_action_accepted_nodes": ["node-a"],
            "node_action_endpoint": GONE_AGENT,
        },
    )
    context = _quiesce_context(reset, accepted)
    registry = StubFleetRegistry({"node-a": GONE_AGENT}, generation=7, ready=False)
    adapter = NodeActionWorkflowAdapter({}, "s" * 32, registry=registry)

    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)

    record = WorkflowStepExecution(
        step_index=1,
        operation=OP.QUIESCE_GPU_SERVICES,
        status=outcome.status,
        phase="official",
        error=outcome.error,
        details=outcome.details or {},
    )
    assert unresolved_node_action(record), (
        "an accepted node action the Agent never answered must stay unresolved "
        f"(WAITING/outcome_unknown), got {outcome.status.value}: {outcome.error} "
        f"{outcome.details}"
    )
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "fleet consistency gate failed" in (outcome.error or ""), outcome.error
    assert outcome.details["node_action_command_id"] == f"{QUIESCE_KEY}/node-a/agent-7"
    assert outcome.details["node_action_accepted_nodes"] == ["node-a"]
    assert outcome.details["outcome_unknown"] is True
    assert "node_action_not_started" not in outcome.details


def test_a_quiesce_the_agent_never_accepted_still_fails_plainly() -> None:
    """Control for the rule above: a QUIESCE whose record shows no acceptance
    (the send never reached an Agent) keeps the gate's plain FAILED -- the
    readiness verdict is the whole truth, nothing may be running."""

    _state, _kubernetes, validator, reset = stopped_runtime()
    refused = WorkflowStepExecution(
        step_index=1,
        operation=OP.QUIESCE_GPU_SERVICES,
        status=WorkflowStepStatus.WAITING,
        phase="official",
        details={"node_action_state": "TRANSPORT_RETRY", "waiting_node": "node-a"},
    )
    context = _quiesce_context(reset, refused)
    registry = StubFleetRegistry({"node-a": GONE_AGENT}, generation=7, ready=False)
    adapter = NodeActionWorkflowAdapter({}, "s" * 32, registry=registry)

    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert "outcome_unknown" not in (outcome.details or {}), outcome.details


def test_a_verify_after_the_reimage_is_fenced_by_the_quiesce_generation() -> None:
    """Point 4, re-image finished: nothing runs on the new incarnation.

    ``VERIFY_NO_GPU_CLIENTS`` (and the RESET and RESTORE after it) pins the
    agent generation the QUIESCE recorded; the new incarnation registered at
    ``generation + 1`` (``fleet.py:515-647``), so the maintenance fence fails
    the step before any dispatch (``adapters/node_action/barriers.py:121-219``).
    """

    node = "hyperpod-i-0b138verify"
    incident = fault_incident(
        "inc-b138-verify",
        "event-b138-verify",
        cluster_id="hp-cluster",
        node_ids=[node],
        fencing_token=2,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-b138-verify",
    )
    steps = [
        workflow_step(OP.QUIESCE_GPU_SERVICES, AGENT_OWNER, node_ids=[node]),
        workflow_step(
            OP.VERIFY_NO_GPU_CLIENTS, AGENT_OWNER, node_ids=[node], gpu_uuids=["GPU-a"]
        ),
    ]
    workflow = workflow_request(
        "workflow-b138-verify",
        incident.incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=2,
        execution_epoch=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[OP.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(
                0,
                OP.QUIESCE_GPU_SERVICES,
                phase="official",
                details={
                    "agent_generations": {node: 7},
                    "maintenance_window_expires_at": (
                        NOW + timedelta(minutes=5)
                    ).isoformat(),
                },
            )
        ],
    )
    registry = StubFleetRegistry({node: f"http://{node}:9099"}, generation=7)
    registry.now = lambda: NOW
    assert registry.restart_agent() == 8, "the re-imaged node's new incarnation"
    sent = []
    adapter = NodeActionWorkflowAdapter(
        {},
        "s" * 32,
        registry=registry,
        sender=lambda endpoint, envelope: sent.append(envelope),
    )

    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=steps[1],
            step_index=1,
            request=WorkflowExecutionRequest(expected_fencing_token=2),
            idempotency_key="workflow-b138-verify/1/VERIFY_NO_GPU_CLIENTS",
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert "maintenance agent fence failed" in (outcome.error or ""), outcome.error
    assert "generation changed from 7 to 8" in (outcome.error or ""), outcome.error
    assert sent == [], "the new incarnation receives no VERIFY without a quiesce"


# --------------------------------------------------------------------------
# confirm-node-action on a parked node action of a re-imaged node


HP_NODE = "hyperpod-i-0b138fault"
SIBLING = "hyperpod-i-0b138sibling"
HP_INCIDENT = "inc-b138-quiesce"
HP_WORKFLOW = "workflow-b138-quiesce"
FAULT_BOOT = "boot-at-the-xid"
STOP_AT = NOW - timedelta(hours=1)
QUIESCE_AT = STOP_AT + timedelta(minutes=1)


def _parked_quiesce(*, node_ids: list[str], stop_boots: dict[str, str]):
    """A reset chain parked BLOCKED/NEEDS_OPERATOR on an unknown QUIESCE.

    The STOP before it completed with a contained receipt binding each node's
    boot (``stop_ownership.py`` ``capture``/``finish``); the QUIESCE was
    accepted after that receipt was re-verified, then its answer was lost.
    """

    steps = [
        workflow_step(
            OP.MARK_UNSCHEDULABLE, "gpu-fault-kubernetes-adapter", node_ids=node_ids
        ),
        workflow_step(
            OP.STOP_WORKLOADS,
            "gpu-fault-kubernetes-adapter",
            node_ids=node_ids,
            workload_ids=["training/job/training-a"],
        ),
        workflow_step(
            OP.QUIESCE_GPU_SERVICES,
            AGENT_OWNER,
            node_ids=[HP_NODE],
            gpu_uuids=["GPU-1"],
        ),
        workflow_step(OP.VERIFY_NO_GPU_CLIENTS, AGENT_OWNER, node_ids=[HP_NODE]),
        workflow_step(
            OP.RESET_GPU, AGENT_OWNER, node_ids=[HP_NODE], gpu_uuids=["GPU-1"]
        ),
        workflow_step(OP.RESTORE_GPU_SERVICES, AGENT_OWNER, node_ids=[HP_NODE]),
    ]
    receipt = contained_stop_receipt(
        cluster_id="hp-cluster",
        workflow_id=HP_WORKFLOW,
        incident_id=HP_INCIDENT,
        fencing_token=2,
        stop_step_index=1,
        stop_idempotency_key=f"{HP_WORKFLOW}/1/STOP_WORKLOADS",
        nodes=[
            {"name": node, "uid": f"{node}-uid", "boot_id": stop_boots[node]}
            for node in node_ids
        ],
        completed_at=STOP_AT.isoformat(),
    )
    workflow = workflow_request(
        HP_WORKFLOW,
        HP_INCIDENT,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=2,
        execution_epoch=3,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[OP.MARK_UNSCHEDULABLE, OP.STOP_WORKLOADS],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
            workflow_step_execution(
                1,
                OP.STOP_WORKLOADS,
                phase="official",
                details={STOP_RECEIPT_KEY: receipt},
                started_at=STOP_AT - timedelta(seconds=20),
                updated_at=STOP_AT,
            ),
            workflow_step_execution(
                2,
                OP.QUIESCE_GPU_SERVICES,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/remote-quiesce",
                error="step 2/QUIESCE_GPU_SERVICES stayed non-terminal for 600s",
                details={
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "node_action_command_id": f"{HP_WORKFLOW}/2/QUIESCE/{HP_NODE}",
                    "node_action_accepted_nodes": [HP_NODE],
                    "remote_command_id": "remote-quiesce",
                },
                started_at=QUIESCE_AT,
                updated_at=QUIESCE_AT + timedelta(minutes=10),
            ),
        ],
        blocked_reasons=["NODE_ACTION_OUTCOME_UNRESOLVED"],
        created_at=STOP_AT - timedelta(minutes=2),
        updated_at=QUIESCE_AT + timedelta(minutes=10),
    )
    incident = fault_incident(
        HP_INCIDENT,
        "kernel-log-xid-79",
        cluster_id="hp-cluster",
        node_ids=node_ids,
        gpu_uuids=["GPU-1"],
        state=IncidentState.QUARANTINED,
        workflow_request_id=HP_WORKFLOW,
        fencing_token=2,
        source_boot_id=FAULT_BOOT,
    )
    return incident, workflow


def _evidence(boot_id: str) -> tuple[KubernetesNodeEvidence, AgentEvidence]:
    """The node as kubectl and the fleet registry show it now, on ``boot_id``."""

    node = KubernetesNodeEvidence.from_mapping(
        {
            "node_id": HP_NODE,
            "uid": f"{HP_NODE}-uid",
            "boot_id": boot_id,
            "ready": True,
            "ready_since": (NOW - timedelta(minutes=20)).isoformat(),
            "unschedulable": True,
        }
    )
    agent = AgentEvidence.from_mapping(
        {
            "node_id": HP_NODE,
            "boot_id": boot_id,
            # heartbeat.py: sha256 of cluster, node, instance and boot -- never
            # a boot id, so the incident's source boot is never "retired".
            "agent_incarnation_id": f"incarnation-of-{boot_id}",
            "lifecycle_state": "ACTIVE",
            "generation": 9,
            "last_seen_at": (NOW - timedelta(seconds=15)).isoformat(),
            "lease_expires_at": (NOW + timedelta(seconds=75)).isoformat(),
        }
    )
    return node, agent


def _confirm(incident, workflow, boot_id: str, **values):
    node, agent = _evidence(boot_id)
    return confirm_node_action(
        workflow=workflow,
        incident=incident,
        node_id=HP_NODE,
        node=node,
        agent=agent,
        remote_commands=[],
        actor=ACTOR,
        reference="CHG-B138",
        now=NOW,
        **values,
    )


def test_a_reimage_before_the_stop_is_not_a_reboot_since_the_quiesce() -> None:
    """Point 1, re-image finished before the STOP; a later node action parks.

    The STOP captured the re-imaged boot, the QUIESCE ran on it, its answer
    was lost (step cap / execution cap), and the node has not rebooted since.
    Nothing settles the action -- the ledger on the node is the record. Before
    batch 139 the verb read the incident's fault-time ``source_boot_id`` and
    confirmed it ``settled_by=reboot_since_action``; it now takes the boot from
    the workflow's own STOP receipt (``node_action_baseline``) and refuses.
    """

    incident, workflow = _parked_quiesce(
        node_ids=[HP_NODE], stop_boots={HP_NODE: NEW_BOOT}
    )

    verdict = _confirm(incident, workflow, NEW_BOOT)

    assert verdict.confirmations == (), (
        "the action ran on the boot the node still runs; no reboot since it is "
        f"proven, but it was confirmed: {verdict.confirmations}"
    )
    assert any(f"still runs boot {NEW_BOOT}" in item for item in verdict.refusals), (
        verdict.refusals
    )


def test_a_reimage_after_the_quiesce_settles_it_on_a_multi_node_incident() -> None:
    """Point 3, re-image after the accepted QUIESCE of a two-node job.

    The receipt bound the node at ``FAULT_BOOT``; the node now runs a new boot
    under a new Agent incarnation. The re-image ended whatever the quiesce left
    running, exactly the ``reboot_since_action`` rule. Before batch 139 the
    verb refused for want of a baseline (a multi-node incident has no usable
    ``source_boot_id``), which left an ESCALATED record with no exit; the STOP
    receipt is that baseline now.
    """

    incident, workflow = _parked_quiesce(
        node_ids=[HP_NODE, SIBLING],
        stop_boots={HP_NODE: FAULT_BOOT, SIBLING: "sibling-boot"},
    )

    verdict = _confirm(incident, workflow, NEW_BOOT)

    assert verdict.refusals == (), verdict.refusals
    (confirmation,) = verdict.confirmations
    assert confirmation["settled_by"] == "reboot_since_action", confirmation
    assert confirmation["previous_boot_id"] == FAULT_BOOT, confirmation
    assert confirmation["previous_boot_id_source"] == (
        "stop.stop_ownership_receipt_v1"
    ), confirmation


def test_a_quarantined_parked_record_exits_through_restore_then_confirmation() -> None:
    """The supported exit of points 1-3 when a node action parked unresolved.

    The incident is QUARANTINED (MARK_UNSCHEDULABLE completed,
    ``execution/terminal_state.py`` ``failure_incident_state``); ``restore``
    refuses only PENDING/RUNNING/SAFETY_PENDING records for a QUARANTINED
    incident (``admin/remediation_plans.py:353-414``), and once the validated
    restore SUCCEEDED, ``confirm-node-action`` settles the parked QUIESCE by
    ``node_validation`` (``execution/node_action_confirmation.py:757-835``);
    the dispatcher's sweep or ``workflow-reconcile`` then ends the record.
    """

    incident, workflow = _parked_quiesce(
        node_ids=[HP_NODE, SIBLING],
        stop_boots={HP_NODE: FAULT_BOOT, SIBLING: "sibling-boot"},
    )
    isolated = {
        "metadata": {
            "name": HP_NODE,
            "annotations": {ANNOTATION_INCIDENT: HP_INCIDENT, ANNOTATION_FENCING: "2"},
        },
        "spec": {
            "unschedulable": True,
            "taints": [
                {
                    "key": QUARANTINE_TAINT,
                    "value": quarantine_taint_value(HP_INCIDENT),
                    "effect": "NoSchedule",
                }
            ],
        },
    }
    plan = build_restore_plan(
        {
            "incident": incident.model_dump(mode="json"),
            "workflow": workflow.model_dump(mode="json"),
            "open_workflows": [],
            "blocked_workflows": [workflow.model_dump(mode="json")],
        },
        incident_id=HP_INCIDENT,
        operator=ACTOR,
        reference="CHG-B138",
        live_nodes={
            HP_NODE: isolated,
            SIBLING: {"metadata": {"name": SIBLING}, "spec": {}},
        },
        now=NOW,
    )
    assert plan.restore_node_ids == (HP_NODE,), plan.node_decisions
    validated_at = NOW - timedelta(minutes=5)
    restore = workflow_request(
        "workflow-validated-restore-b138",
        HP_INCIDENT,
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=2,
        official_action="RESTORE_SCHEDULING",
        official_steps=[
            workflow_step(OP.VALIDATE_GPU, node_ids=[HP_NODE], gpu_uuids=["GPU-1"]),
            workflow_step(OP.RESTORE_SCHEDULING, node_ids=[HP_NODE]),
        ],
        completed_step_indexes=[0, 1],
        completed_operations=[OP.VALIDATE_GPU, OP.RESTORE_SCHEDULING],
        step_executions=[
            workflow_step_execution(
                0, OP.VALIDATE_GPU, phase="official", updated_at=validated_at
            ),
            workflow_step_execution(
                1, OP.RESTORE_SCHEDULING, phase="official", updated_at=validated_at
            ),
        ],
        updated_at=validated_at,
    )
    recovered = incident.model_copy(
        update={
            "state": IncidentState.RECOVERED,
            "workflow_request_id": restore.request_id,
        }
    )

    verdict = _confirm(recovered, workflow, NEW_BOOT, node_validations=[restore])

    assert verdict.refusals == (), verdict.refusals
    (confirmation,) = verdict.confirmations
    assert confirmation["settled_by"] == "validated_restore", confirmation
