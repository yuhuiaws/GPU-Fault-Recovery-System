"""Batch 138, shape B: the platform replaces a GPU node mid-remediation.

A managed node group roll or a Spot reclaim (eks-managed; a HyperPod Spot
reclaim behaves the same) takes the instance and the Kubernetes node name
away while a remediation workflow is in flight on it; a new node with another
name and instance id joins under an Agent record of its own. The departed-node
cleanup proves the departure on three fresh sources -- the Kubernetes report
chain, EC2 ``terminated`` / HyperPod ``ListClusterNodes``, an Agent that is not
alive (``gpu_fault/departed_nodes.py``) -- and ends what never changed the node
(``gpu_fault/departed_node_cleanup_actions.py``).

These tests pin the in-flight shapes where the record of a workflow that DID
change the node has no supported exit, or says something that is not true:

* a record parked BLOCKED / NEEDS_OPERATOR on an unresolved node action of the
  departed node: ``confirm-node-action`` needs a Ready node
  (``execution/node_action_confirmation.py:620-659``), ``restore`` needs every
  node in the cluster (``admin/remediation_plans.py:394-401``), the reconcile
  shapes need a verified restore successor, a never-changed record or an
  operator confirmation (``workflow_resolution.py:373-463``,
  ``workflow_reconcile.py:467-528``), the incident close refuses the open record
  (``orchestration/incident_closure.py:723-742``, ``:830-848``), and the
  cleanup reports ``workflow_changed_node``
  (``departed_node_cleanup_actions.py:534-545``);
* a two-node QUARANTINED incident one of whose nodes departed cannot be
  restored, so the surviving node keeps its cordon and taint;
* a STOP_WORKLOADS whose node vanished (or, shape A, was re-imaged) after the
  suspend landed is refused as ``node_action_not_started`` and drops its
  receipt (``adapters/kubernetes/stop_ownership.py:926-941``): the job stays
  suspended under a record that says nothing was done.

The node-action adapter's pre-dispatch gates -- the fleet readiness gate and
the quiesce maintenance-window expiry / generation fence
(``adapters/node_action/barriers.py`` 98-219) -- answer before the transport
reads an accepted command back and turn its outcome-unknown record into a plain
failure; that gap is the same code on a departed node (benign there: every rung
it climbs is refused on the absent node) and is pinned once, by the shape-A
files ``tests/regional/test_b138_reimage_before_reset.py`` (QUIESCE) and
``tests/regional/test_b138_reimage_reset_and_after.py`` (RESET_GPU).

Batch 138 wrote each gap as an ``xfail(strict=True)`` test asserting the safe
exit or the safe verdict, failing on 9132d9f4 at that assertion. Batch 139
closed them and they are plain tests now: the departure proof settles a parked
node action ``node_departed`` (the cleanup, the operator's evidence close,
``confirm-node-action``), the restore skips a proven-departed node, and a STOP
refused after its suspend keeps its receipt and says the job is suspended. The
shapes without the proof still refuse, and are pinned as such. In-memory store
only.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    stop_ownership_scope,
)
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.remediation_plans import build_restore_plan
from gpu_fault.departed_nodes import (
    EKS_PROVIDER,
    HYPERPOD_PROVIDER,
    DepartedNodeCleanupStatus,
    node_evidence,
)
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.node_action_confirmation import (
    NODE_DEPARTED,
    AgentEvidence,
    KubernetesNodeEvidence,
    apply_node_action_confirmation,
    confirm_node_action,
)
from gpu_fault.execution.node_action_uncertainty import (
    has_unresolved_node_action,
    unresolved_node_action,
)
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.incident_closure import (
    IncidentClosureService,
    IncidentNotClosable,
    NodeIsolationEvidence,
)
from gpu_fault.workflow_reconcile import (
    build_workflow_reconcile_plan,
    operator_confirmed_plan_items,
)
from tests._builders import (
    build_store,
    contained_stop_receipt,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_INSTANCE,
    EKS_NODE,
    EKS_PRESENT,
    HP_CLUSTER,
    HP_INSTANCE,
    HP_NAME,
    HP_NODE,
    HP_PRESENT,
    REGION,
    World,
    agent,
    ago,
    now,
    registration,
    world,
)
from tests.regional._late_ownership_runtime import runtime

OP = WorkflowOperation
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
REFERENCE = "CHG-B138"
WORKLOAD = "training/pytorchjob/job"
KUBERNETES = "gpu-fault-kubernetes-adapter"
NODE_AGENT = "gpu-fault-node-agent"
VALIDATION = "gpu-fault-validation-adapter"
LIFECYCLE = {
    EKS_PROVIDER: "gpu-fault-eks-adapter",
    HYPERPOD_PROVIDER: "gpu-fault-hyperpod-adapter",
}
PROVIDERS = (EKS_PROVIDER, HYPERPOD_PROVIDER)
OLD_BOOT = "boot-before-the-replacement"


class NodeNotFound(Exception):
    """What the Kubernetes client raises for a Node the cluster deleted."""

    status = 404


def _departure(store: Any, provider: str) -> tuple[World, str, str, str]:
    """``(world, cluster, departed node, present node)``: the departure proven.

    The departed node's Agent stays ACTIVE with a lease that expired twenty
    minutes ago (no path revokes it until the cleanup), twelve minutes of
    Kubernetes reports leave it out, and the provider no longer has it.
    """

    result = world(store)
    if provider == EKS_PROVIDER:
        store.save_regional_cluster(registration(EKS_CLUSTER, provider=EKS_PROVIDER))
        store.save_agent(
            agent(
                EKS_CLUSTER,
                EKS_NODE,
                instance_id=EKS_INSTANCE,
                last_seen=ago(minutes=20),
            )
        )
        result.provider.ec2[EKS_INSTANCE] = "terminated"
        result.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=12)
        return result, EKS_CLUSTER, EKS_NODE, EKS_PRESENT
    store.save_regional_cluster(registration(HP_CLUSTER))
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="node-uid-1", last_seen=ago(minutes=20))
    )
    result.provider.hyperpod[HP_NAME] = {"i-0a000002"}
    result.reports_since(HP_CLUSTER, [HP_PRESENT], minutes=12)
    return result, HP_CLUSTER, HP_NODE, HP_PRESENT


def _departure_proof(provider: str, node_id: str) -> dict[str, Any]:
    """The per-node departure record the CLI hands the closure service."""

    if provider == EKS_PROVIDER:
        departed: dict[str, Any] = {
            "node_id": node_id,
            "exists": False,
            "provider": EKS_PROVIDER,
            "instance_id": EKS_INSTANCE,
            "absent_from_provider": True,
            "ec2_state": "terminated",
        }
        target = {"cluster_id": EKS_CLUSTER, "provider": EKS_PROVIDER}
    else:
        departed = {
            "node_id": node_id,
            "exists": False,
            "provider": HYPERPOD_PROVIDER,
            "instance_id": HP_INSTANCE,
            "absent_from_provider": True,
        }
        target = {"hyperpod_cluster_name": HP_NAME}
    return node_evidence(node_id, departed=departed, target=target, aws_region=REGION)


def _parked_execution(operation: OP, node_id: str, index: int) -> WorkflowStepExecution:
    """The FAILED record a node action whose outcome nobody knows parks on.

    RESTART_NODE: the reboot was submitted, the instance was terminated before
    any confirmation, and the step's waiting cap failed it ``outcome_unknown``
    (``execution/step_bounds.py:201-225`` over the WAITING record
    ``execution/node_action_uncertainty.py:149-164`` marks unknown). RESET_GPU:
    the agent accepted the reset and its answer was lost with the node (the
    transport's ``_unknown_response``, ``adapters/node_action/transport.py``
    151-172; a barrier participant's manual failure, ``barriers.py:475-522``).
    """

    started = ago(hours=1)
    if operation is OP.RESTART_NODE:
        details: dict[str, Any] = {
            "outcome_unknown": True,
            "manual_confirmation_required": True,
            "requires_external_confirmation": True,
            "agent_baselines": {
                node_id: {
                    "boot_id": OLD_BOOT,
                    "agent_incarnation_id": f"incarnation-{node_id}",
                }
            },
            "remote_command_id": "remote-reboot",
            "remote_status": "WAITING",
            "step_waiting_timeout_seconds": 1800,
        }
        operation_id = "remote/remote-reboot"
    else:
        details = {
            "outcome_unknown": True,
            "manual_confirmation_required": True,
            "node_action_response_unknown": True,
            "node_action_command_id": f"workflow-b138/{index}/RESET_GPU/{node_id}",
            "node_action_accepted_nodes": [node_id],
            "reason": "OWNERSHIP_RESULT_UNAVAILABLE",
            "remote_command_id": "remote-reset",
        }
        operation_id = "remote/remote-reset"
    return workflow_step_execution(
        index,
        operation,
        WorkflowStepStatus.FAILED,
        phase="official",
        adapter_operation_id=operation_id,
        error=f"{operation.value} outcome is unknown",
        details=details,
        started_at=started,
        updated_at=started + timedelta(minutes=30),
    )


def _parked(
    store: Any, cluster_id: str, node_id: str, *, provider: str, operation: OP
) -> tuple[str, str]:
    """A workflow that cordoned and stopped, then parked on ``operation``.

    BLOCKED / NEEDS_OPERATOR with its incident QUARANTINED -- the verdict
    ``execution/terminal_state.py:186-199`` writes for a record with an
    unresolved node action after a completed MARK_UNSCHEDULABLE.
    """

    request_id = f"workflow-b138-{operation.value.lower()}"
    incident_id = f"incident-b138-{operation.value.lower()}"
    created = ago(hours=2)
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, KUBERNETES, node_ids=[node_id]),
        workflow_step(
            OP.STOP_WORKLOADS, KUBERNETES, node_ids=[node_id], workload_ids=[WORKLOAD]
        ),
    ]
    if operation is OP.RESET_GPU:
        steps += [
            workflow_step(OP.QUIESCE_GPU_SERVICES, NODE_AGENT, node_ids=[node_id]),
            workflow_step(OP.VERIFY_NO_GPU_CLIENTS, NODE_AGENT, node_ids=[node_id]),
            workflow_step(OP.RESET_GPU, NODE_AGENT, node_ids=[node_id]),
            workflow_step(OP.RESTORE_GPU_SERVICES, NODE_AGENT, node_ids=[node_id]),
        ]
    else:
        steps.append(
            workflow_step(OP.RESTART_NODE, LIFECYCLE[provider], node_ids=[node_id])
        )
    steps += [
        workflow_step(OP.VALIDATE_GPU, VALIDATION, node_ids=[node_id]),
        workflow_step(OP.RESTORE_SCHEDULING, KUBERNETES, node_ids=[node_id]),
        workflow_step(
            OP.RESTART_WORKLOAD, KUBERNETES, node_ids=[node_id], workload_ids=[WORKLOAD]
        ),
    ]
    parked_index = next(
        index for index, step in enumerate(steps) if step.operation is operation
    )
    completed = list(range(parked_index))
    executions = [
        workflow_step_execution(
            index,
            steps[index].operation,
            phase="official",
            started_at=created,
            updated_at=created,
        )
        for index in completed
    ]
    executions.append(_parked_execution(operation, node_id, parked_index))
    store.save_workflow(
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.BLOCKED,
            blocked_kind=BlockedKind.NEEDS_OPERATOR,
            fencing_token=1,
            execution_epoch=3,
            official_steps=steps,
            completed_step_indexes=completed,
            completed_operations=[steps[index].operation for index in completed],
            step_executions=executions,
            blocked_reasons=["NODE_ACTION_OUTCOME_UNRESOLVED"],
            created_at=created,
            updated_at=ago(minutes=30),
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{incident_id}",
            cluster_id=cluster_id,
            node_ids=[node_id],
            gpu_uuids=["GPU-b138"],
            state=IncidentState.QUARANTINED,
            workflow_request_id=request_id,
            fencing_token=1,
            source_boot_id=OLD_BOOT,
            created_at=created,
            updated_at=ago(minutes=30),
        )
    )
    return request_id, incident_id


def _inspection(store: Any, incident_id: str) -> dict[str, Any]:
    """What the CPU Pod's inspection returns to ``submit-remediation``."""

    incident = store.get_incident(incident_id)
    workflow = store.get_workflow(str(incident.workflow_request_id))
    blocked = [workflow] if workflow.status is WorkflowStatus.BLOCKED else []
    agents: dict[str, Any] = {}
    for node_id in incident.node_ids:
        try:
            record = store.get_agent(incident.cluster_id, node_id)
        except KeyError:
            record = None
        agents[node_id] = record.model_dump(mode="json") if record else None
    return {
        "incident": incident.model_dump(mode="json"),
        "workflow": workflow.model_dump(mode="json"),
        "open_workflows": [],
        "blocked_workflows": [item.model_dump(mode="json") for item in blocked],
        "agents": agents,
    }


PARKED_SHAPES = [
    pytest.param(provider, operation, id=f"{provider}-{operation.value}")
    for provider in PROVIDERS
    for operation in (OP.RESTART_NODE, OP.RESET_GPU)
]


@pytest.mark.parametrize(("provider", "operation"), PARKED_SHAPES)
def test_without_its_departure_proof_a_parked_node_action_still_has_no_exit(
    provider: str, operation: OP
) -> None:
    """Shape B, RESTART_NODE in flight / RESET_GPU claimed (pinned).

    Was the DEAD END of 9132d9f4; the departure proof is the exit since batch
    139 (the tests below). Without the proof every supported verb still
    refuses: the record is no reconcile shape (no verified restore successor,
    it changed the node, no operator confirmation:
    ``workflow_resolution.py:373-463``, ``workflow_reconcile.py:531-612``);
    ``confirm-node-action`` cannot judge a node the cluster no longer has
    (``node_action_confirmation.py`` ``_common_evidence_refusals``); the close
    cannot clear a missing node (``isolation_verdict``); and ``restore``
    refuses an incident whose node is missing (``remediation_plans.py``).
    """

    store = build_store()
    _world, cluster_id, node_id, _present = _departure(store, provider)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=provider, operation=operation
    )
    workflow = store.get_workflow(request_id)

    [item] = build_workflow_reconcile_plan(store, [request_id], now=now())["items"]
    assert item["eligible"] is False, item
    assert operator_confirmed_plan_items(store, [request_id], now=now()) == [], (
        "no confirmation exists, so the operator-confirmed shape has no item"
    )
    verdict = confirm_node_action(
        workflow=workflow,
        incident=store.get_incident(incident_id),
        node_id=node_id,
        node=KubernetesNodeEvidence.from_node(node_id, None),
        agent=AgentEvidence.from_record(node_id, store.get_agent(cluster_id, node_id)),
        remote_commands=[],
        actor=ACTOR,
        reference=REFERENCE,
        now=now(),
    )
    assert verdict.confirmations == ()
    assert f"node {node_id} is not in the cluster" in verdict.refusals, verdict.refusals
    with pytest.raises(IncidentNotClosable, match="isolation state cannot be verified"):
        IncidentClosureService(store).close_incident(
            incident_id,
            reason="node replaced by the platform",
            operator=ACTOR,
            reference=REFERENCE,
            evidence=[NodeIsolationEvidence(node_id=node_id, exists=False)],
        )
    with pytest.raises(BootstrapError, match="missing from cluster"):
        build_restore_plan(
            _inspection(store, incident_id),
            incident_id=incident_id,
            operator=ACTOR,
            reference=REFERENCE,
            live_nodes={},
            now=now(),
        )
    assert has_unresolved_node_action(store.get_workflow(request_id)), (
        "nothing above answered the record's question"
    )
    assert store.get_incident(incident_id).state is IncidentState.QUARANTINED


@pytest.mark.parametrize(("provider", "operation"), PARKED_SHAPES)
def test_confirm_node_action_settles_the_departed_nodes_action_on_its_proof(
    provider: str, operation: OP
) -> None:
    """Exit B1 (operator, step by step): ``confirm-node-action`` on a node the
    cluster no longer has takes its departure proof -- gone from Kubernetes
    and the provider, the Agent not alive -- and settles the parked action
    ``node_departed``; the existing exits then finish: the operator-confirmed
    reconcile shape plans the record, and the evidence close ends it with the
    incident."""

    store = build_store()
    _world, cluster_id, node_id, _present = _departure(store, provider)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=provider, operation=operation
    )
    workflow = store.get_workflow(request_id)
    departure = _departure_proof(provider, node_id)

    verdict = confirm_node_action(
        workflow=workflow,
        incident=store.get_incident(incident_id),
        node_id=node_id,
        node=KubernetesNodeEvidence.from_node(node_id, None),
        agent=AgentEvidence.from_record(node_id, store.get_agent(cluster_id, node_id)),
        remote_commands=[],
        actor=ACTOR,
        reference=REFERENCE,
        now=now(),
        departure=departure,
    )

    assert verdict.refusals == (), verdict.refusals
    (confirmation,) = verdict.confirmations
    assert confirmation["settled_by"] == NODE_DEPARTED
    assert confirmation["operation"] == operation.value
    assert confirmation["node_departures"][node_id]["absent_from_provider"] is True
    assert confirmation["superseded_flags"]["outcome_unknown"] is True
    confirmed = apply_node_action_confirmation(workflow, verdict, now=now())
    assert not has_unresolved_node_action(confirmed), "the departure answered it"
    store.save_workflow(confirmed, expected=workflow)
    [planned] = operator_confirmed_plan_items(store, [request_id], now=now())
    assert planned["request_id"] == request_id
    closed, changed = IncidentClosureService(store).close_incident(
        incident_id,
        reason="node replaced by the platform",
        operator=ACTOR,
        reference=REFERENCE,
        evidence=[
            NodeIsolationEvidence(node_id=node_id, exists=False, departure=departure)
        ],
    )
    assert changed is True and closed.state is IncidentState.RECOVERED
    assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_escalated_parked_record_asks_for_evidence_once_its_node_left(
    provider: str,
) -> None:
    """``--close-incident`` runs a first pass without evidence; the preview
    asks for node evidence only when a departure may answer the record -- the
    latest Kubernetes report omits the node or its Agent is past the guard --
    and the second pass with the departure proof settles and closes it."""

    store = build_store()
    _world, cluster_id, node_id, _present = _departure(store, provider)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=provider, operation=OP.RESET_GPU
    )
    incident = store.get_incident(incident_id)
    store.save_incident(
        incident.model_copy(update={"state": IncidentState.ESCALATED}),
        expected=incident,
    )
    service = IncidentClosureService(store)

    first = service.preview(incident_id)

    assert first["closable"] is False and first["evidence_required"] is True, first
    second = service.preview(
        incident_id,
        evidence=[
            NodeIsolationEvidence(
                node_id=node_id,
                exists=False,
                departure=_departure_proof(provider, node_id),
            )
        ],
    )
    assert second["closable"] is True, second
    closed, changed = service.close_incident(
        incident_id,
        reason="node replaced by the platform",
        operator=ACTOR,
        reference=REFERENCE,
        evidence=[
            NodeIsolationEvidence(
                node_id=node_id,
                exists=False,
                departure=_departure_proof(provider, node_id),
            )
        ],
    )
    assert changed is True and closed.state is IncidentState.RECOVERED
    assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED


def test_confirm_node_action_refuses_a_departure_it_cannot_prove() -> None:
    """The proof is the departed-node rule, all of it: the provider must no
    longer have the instance, the Agent must not be alive, and a step that also
    names another node is not answered by this one leaving."""

    store = build_store()
    _world, cluster_id, node_id, present = _departure(store, EKS_PROVIDER)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=EKS_PROVIDER, operation=OP.RESET_GPU
    )
    workflow = store.get_workflow(request_id)
    incident = store.get_incident(incident_id)
    proof = _departure_proof(EKS_PROVIDER, node_id)

    def judge(*, departure=proof, agent=None, record=workflow):
        return confirm_node_action(
            workflow=record,
            incident=incident,
            node_id=node_id,
            node=KubernetesNodeEvidence.from_node(node_id, None),
            agent=agent
            or AgentEvidence.from_record(node_id, store.get_agent(cluster_id, node_id)),
            remote_commands=[],
            actor=ACTOR,
            reference=REFERENCE,
            now=now(),
            departure=departure,
        )

    listed = {**proof, "absent_from_provider": False, "blockers": ["EC2 says running"]}
    unproven = judge(departure=listed)
    assert unproven.confirmations == ()
    assert any(
        "departure is unproven (EC2 says running)" in item for item in unproven.refusals
    ), unproven.refusals
    alive = judge(
        agent=AgentEvidence.from_mapping(
            {
                "node_id": node_id,
                "lifecycle_state": "ACTIVE",
                "last_seen_at": ago(minutes=2).isoformat(),
                "lease_expires_at": ago(minutes=1).isoformat(),
            }
        )
    )
    assert any("Node Agent is not departed" in item for item in alive.refusals), (
        alive.refusals
    )
    shared = workflow.model_copy(
        update={
            "official_steps": [
                step.model_copy(update={"node_ids": [node_id, present]})
                if step.operation is OP.RESET_GPU
                else step
                for step in workflow.official_steps
            ]
        }
    )
    sibling = judge(record=shared)
    assert any(f"also names {present}" in item for item in sibling.refusals), (
        sibling.refusals
    )


def test_the_close_names_what_a_departure_does_not_answer() -> None:
    """The evidence close settles a parked record only when every unresolved
    node action is on a node the evidence proves departed; a reset that also
    names the node that is still there keeps refusing, and says why."""

    store = build_store()
    _world, cluster_id, departed, present = _departure(store, EKS_PROVIDER)
    request_id, incident_id = _parked(
        store, cluster_id, departed, provider=EKS_PROVIDER, operation=OP.RESET_GPU
    )
    workflow = store.get_workflow(request_id)
    store.save_workflow(
        workflow.model_copy(
            update={
                "official_steps": [
                    step.model_copy(update={"node_ids": [departed, present]})
                    if step.operation is OP.RESET_GPU
                    else step
                    for step in workflow.official_steps
                ]
            }
        ),
        expected=workflow,
    )
    incident = store.get_incident(incident_id)
    store.save_incident(
        incident.model_copy(update={"node_ids": [departed, present]}), expected=incident
    )

    with pytest.raises(IncidentNotClosable, match="do not answer: step 4 RESET_GPU"):
        IncidentClosureService(store).close_incident(
            incident_id,
            reason="one node replaced, the other clean",
            operator=ACTOR,
            reference=REFERENCE,
            evidence=[
                NodeIsolationEvidence(
                    node_id=departed,
                    exists=False,
                    departure=_departure_proof(EKS_PROVIDER, departed),
                ),
                NodeIsolationEvidence(node_id=present),
            ],
        )
    assert has_unresolved_node_action(store.get_workflow(request_id)), (
        "the refused close settled nothing"
    )


@pytest.mark.parametrize(("provider", "operation"), PARKED_SHAPES)
def test_the_departure_proof_settles_a_parked_node_action_by_itself(
    provider: str, operation: OP
) -> None:
    """Desired exit B1 (automatic): the instance is terminated / gone from
    HyperPod, the node from Kubernetes, the Agent dead -- whatever the parked
    action did went with the instance and nothing can still be running on it.
    The cleanup that already closes the never-changed records on this proof
    ends the parked one too and closes its QUARANTINED incident with the
    departure as isolation evidence. Fails on 9132d9f4 at the ledger status:
    ``REFUSED`` (``departed_node_cleanup_actions.py:534-545``)."""

    store = build_store()
    world_, cluster_id, node_id, _present = _departure(store, provider)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=provider, operation=operation
    )

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED
    assert store.get_incident(incident_id).state is IncidentState.RECOVERED


def _with_stop_receipt(store: Any, request_id: str) -> None:
    """The parked record's STOP as it really lands: with its receipt."""

    workflow = store.get_workflow(request_id)
    receipt = contained_stop_receipt(
        workflow_id=request_id,
        workloads=[
            {
                "workload_id": WORKLOAD,
                "namespace": "training",
                "kind": "pytorchjob",
                "name": "job",
                "uid": "job-uid",
                "attempt_id": "attempt-1",
                "owners": [],
            }
        ],
    )
    executions = [
        item.model_copy(update={"details": {STOP_RECEIPT_KEY: receipt}})
        if item.operation is OP.STOP_WORKLOADS
        else item
        for item in workflow.step_executions
    ]
    store.save_workflow(
        workflow.model_copy(update={"step_executions": executions}), expected=workflow
    )


@pytest.mark.parametrize("state", [IncidentState.QUARANTINED, IncidentState.ESCALATED])
def test_the_cleaned_notice_names_the_job_the_settled_record_left_suspended(
    state: IncidentState,
) -> None:
    """G7 (user decision, batch 139): nothing restarts the job a settled record
    stopped -- not the cleanup, not a restore; its owner does. The cleaned
    notice therefore no longer says "no action is needed" and names the job.
    An ESCALATED incident is settled the same way: the cleanup hands the close
    the departure as evidence for it too."""

    store = build_store()
    world_, cluster_id, node_id, _present = _departure(store, EKS_PROVIDER)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=EKS_PROVIDER, operation=OP.RESET_GPU
    )
    _with_stop_receipt(store, request_id)
    incident = store.get_incident(incident_id)
    store.save_incident(incident.model_copy(update={"state": state}), expected=incident)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert f"workflow-closed:{request_id}" in record.actions, record.actions
    assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED
    [notice] = [store.get_notification(item) for item in world_.notifications.sent]
    assert "No action is needed" not in notice.body_text
    assert (
        "workload training/pytorchjob/job stays suspended (stopped by workflow "
        f"{request_id}): the restore does not restart it; its owner must restart it"
    ) in notice.body_text


@pytest.mark.parametrize(("provider", "operation"), PARKED_SHAPES)
def test_the_operator_closes_a_parked_record_on_the_departure_proof(
    provider: str, operation: OP
) -> None:
    """Desired exit B1 (operator, cleanup switched off or not run yet):
    ``workflow-reconcile --close-quarantined`` reads the departure proof for
    the missing node (``admin/incident_close.py:24-33``); the closure service
    already clears the isolation on it (``incident_closure.py:220-267``) and
    should settle the record the departure answered, the way it settles an
    operator-confirmed one (``incident_closure.py:723-730``, ``:459-488``).
    Fails on 9132d9f4 with ``IncidentNotClosable`` (open workflow)."""

    store = build_store()
    _world, cluster_id, node_id, _present = _departure(store, provider)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=provider, operation=operation
    )

    closed, changed = IncidentClosureService(store).close_incident(
        incident_id,
        reason="node replaced by the platform",
        operator=ACTOR,
        reference=REFERENCE,
        evidence=[
            NodeIsolationEvidence(
                node_id=node_id,
                exists=False,
                departure=_departure_proof(provider, node_id),
            )
        ],
    )

    assert changed is True and closed.state is IncidentState.RECOVERED
    assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED


def _pair(store: Any, cluster_id: str, departed: str, present: str) -> str:
    """A two-node incident whose workflow FAILED after isolating both nodes."""

    incident_id = "incident-b138-pair"
    request_id = "workflow-b138-pair"
    created = ago(hours=2)
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, KUBERNETES, node_ids=[departed, present]),
        workflow_step(
            OP.STOP_WORKLOADS,
            KUBERNETES,
            node_ids=[departed, present],
            workload_ids=[WORKLOAD],
        ),
        workflow_step(
            OP.QUIESCE_GPU_SERVICES, NODE_AGENT, node_ids=[departed, present]
        ),
    ]
    store.save_workflow(
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            execution_epoch=2,
            official_steps=steps,
            completed_step_indexes=[0, 1],
            completed_operations=[OP.MARK_UNSCHEDULABLE, OP.STOP_WORKLOADS],
            step_executions=[
                workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
                workflow_step_execution(1, OP.STOP_WORKLOADS, phase="official"),
                workflow_step_execution(
                    2,
                    OP.QUIESCE_GPU_SERVICES,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    error=f"fleet consistency gate failed: {departed}: heartbeat "
                    "is stale",
                ),
            ],
            terminal_failure_reason="fleet consistency gate failed",
            created_at=created,
            updated_at=ago(minutes=30),
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            "event-b138-pair",
            cluster_id=cluster_id,
            node_ids=[departed, present],
            state=IncidentState.QUARANTINED,
            workflow_request_id=request_id,
            fencing_token=1,
            created_at=created,
            updated_at=ago(minutes=30),
        )
    )
    return incident_id


def _isolated(node_id: str, incident_id: str) -> dict[str, Any]:
    """The surviving node as ``kubectl get node -o json`` shows it."""

    return {
        "metadata": {
            "name": node_id,
            "uid": f"uid-{node_id}",
            "annotations": {ANNOTATION_INCIDENT: incident_id, ANNOTATION_FENCING: "1"},
        },
        "spec": {
            "unschedulable": True,
            "taints": [
                {
                    "key": QUARANTINE_TAINT,
                    "value": quarantine_taint_value(incident_id),
                    "effect": "NoSchedule",
                }
            ],
        },
    }


def test_a_partly_departed_quarantine_keeps_the_surviving_node_isolated() -> None:
    """Shape B, two-node incident (a job DAG / a multi-node reset), without
    the departure proof (pinned; was the DEAD END of 9132d9f4).

    One node was reclaimed; the workflow FAILED and the other node keeps the
    incident's cordon, taint and annotations. The cleanup proves the departure
    but the incident also names a present node (``INCIDENT_NAMES_OTHER_NODES``,
    ``departed_node_cleanup_actions.py:316-342``) and points at
    ``--close-incident``, which refuses while the present node is isolated
    (``incident_closure.py:269-270``); the exit for an isolated node is the
    validated restore, which refuses the whole incident while a node is missing
    without a proven departure (``remediation_plans.py``). With the proof the
    restore releases the node that is still there (next test).
    """

    store = build_store()
    world_, cluster_id, departed, present = _departure(store, EKS_PROVIDER)
    incident_id = _pair(store, cluster_id, departed, present)

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.REFUSED, record.blockers
    assert record.blocker_classes == ["incident_names_other_nodes"]
    assert any(
        f"--close-incident {incident_id}" in item for item in record.admin_commands
    ), record.admin_commands
    with pytest.raises(IncidentNotClosable, match="still cordoned"):
        IncidentClosureService(store).close_incident(
            incident_id,
            reason="one node reclaimed, the other repaired",
            operator=ACTOR,
            reference=REFERENCE,
            evidence=[
                NodeIsolationEvidence(
                    node_id=departed,
                    exists=False,
                    departure=_departure_proof(EKS_PROVIDER, departed),
                ),
                NodeIsolationEvidence(
                    node_id=present,
                    unschedulable=True,
                    quarantine_taint_value=quarantine_taint_value(incident_id),
                    isolation_annotations={
                        ANNOTATION_INCIDENT: incident_id,
                        ANNOTATION_FENCING: "1",
                    },
                ),
            ],
        )
    with pytest.raises(BootstrapError, match=f"missing from cluster .*{departed}"):
        build_restore_plan(
            _inspection(store, incident_id),
            incident_id=incident_id,
            operator=ACTOR,
            reference=REFERENCE,
            live_nodes={present: _isolated(present, incident_id)},
            now=now(),
        )
    assert store.get_incident(incident_id).state is IncidentState.QUARANTINED


def test_the_restore_releases_the_surviving_node_of_a_partly_departed_quarantine() -> (
    None
):
    """Exit B2 (batch 139): the validated restore covers the nodes the cluster
    still has and skips a node whose departure is proven (its isolation left
    with it), the per-node decision the plan already makes for a node another
    incident owns (``remediation_plans.py`` ``departed_node_decision``).
    Failed on 9132d9f4 with ``BootstrapError: incident nodes are missing from
    cluster``."""

    store = build_store()
    _world, cluster_id, departed, present = _departure(store, EKS_PROVIDER)
    incident_id = _pair(store, cluster_id, departed, present)

    plan = build_restore_plan(
        _inspection(store, incident_id),
        incident_id=incident_id,
        operator=ACTOR,
        reference=REFERENCE,
        live_nodes={present: _isolated(present, incident_id)},
        now=now(),
        departures={departed: _departure_proof(EKS_PROVIDER, departed)},
    )

    assert plan.restore_node_ids == (present,)
    decisions = {item["node_id"]: item for item in plan.node_decisions}
    assert decisions[present]["decision"] == "restore"
    assert decisions[departed]["decision"] == "skip"
    assert decisions[departed]["departed"] is True
    assert "isolation left with it" in decisions[departed]["reason"]
    assert decisions[departed]["departure"]["absent_from_provider"] is True
    assert {node for step in plan.steps for node in step["node_ids"]} == {present}, (
        "the departed node is neither validated nor released"
    )


def test_the_restore_keeps_refusing_a_missing_node_it_cannot_prove_departed() -> None:
    """The skip rests on the departed-node rule: a proof the provider does not
    back (the instance still listed) or an Agent that is still alive leaves the
    node missing, and the restore refuses as before, naming why."""

    store = build_store()
    _world, cluster_id, departed, present = _departure(store, EKS_PROVIDER)
    incident_id = _pair(store, cluster_id, departed, present)
    proof = _departure_proof(EKS_PROVIDER, departed)
    listed = {**proof, "absent_from_provider": False, "blockers": ["EC2 is running"]}
    inspection = _inspection(store, incident_id)

    def plan(departure, inspected=inspection):
        return build_restore_plan(
            inspected,
            incident_id=incident_id,
            operator=ACTOR,
            reference=REFERENCE,
            live_nodes={present: _isolated(present, incident_id)},
            now=now(),
            departures={departed: departure},
        )

    with pytest.raises(BootstrapError, match="departure is unproven .EC2 is running"):
        plan(listed)
    alive = json.loads(json.dumps(inspection))
    alive["agents"][departed]["last_seen_at"] = ago(minutes=1).isoformat()
    with pytest.raises(BootstrapError, match="Node Agent is not departed"):
        plan(proof, alive)
    blind = {key: value for key, value in inspection.items() if key != "agents"}
    with pytest.raises(BootstrapError, match="missing from cluster"):
        plan(proof, blind)


def test_a_failed_record_that_changed_the_departed_node_ends_by_itself() -> None:
    """Shape B, the points that end FAILED (STOP refused, QUIESCE / VERIFY /
    RESTORE failed at a gate or a cap, RESTORE_SCHEDULING refused): SAFE, no
    command. Nothing is open, so the cleanup closes the QUARANTINED incident
    with the departure as isolation evidence (``departed_node_cleanup_actions``
    ``:367-411``, ``incident_closure.py:220-267``)."""

    store = build_store()
    world_, cluster_id, node_id, _present = _departure(store, EKS_PROVIDER)
    request_id, incident_id = _parked(
        store, cluster_id, node_id, provider=EKS_PROVIDER, operation=OP.RESET_GPU
    )
    parked = store.get_workflow(request_id)
    plain = [
        item.model_copy(update={"details": {"reason": "fleet consistency gate"}})
        if item.status is WorkflowStepStatus.FAILED
        else item
        for item in parked.step_executions
    ]
    store.save_workflow(
        parked.model_copy(
            update={
                "status": WorkflowStatus.FAILED,
                "blocked_kind": None,
                "step_executions": plain,
            }
        )
    )

    [record] = world_.cleanup.run_once()

    assert record.status is DepartedNodeCleanupStatus.CLEANED, record.blockers
    assert f"incident-closed:{incident_id}" in record.actions
    assert store.get_incident(incident_id).state is IncidentState.RECOVERED


def _vanishing_runtime(shape: str) -> tuple[Any, Any, Any, WorkflowStepContext]:
    """The two-node STOP of ``_late_ownership_runtime``; node-a changes under it
    the moment the suspend patch lands -- in life anywhere between the receipt
    capture and the final containment check, the whole
    STOP_PARTICIPANTS_ACTIVE wait included.

    ``deleted`` (shape B): the Node is gone, ``read_node`` answers 404.
    ``reimaged`` (shape A): the same Node object and UID on a new boot id.
    """

    state, adapter, validator, context = runtime()
    departed: set[str] = set()
    read_node = state.read_node
    patch = state.patch_namespaced_custom_object

    def read(name: str) -> Any:
        if name in departed:
            raise NodeNotFound(name)
        return read_node(name)

    def patch_and_vanish(*args: Any, **kwargs: Any) -> Any:
        result = patch(*args, **kwargs)
        if shape == "deleted":
            departed.add("node-a")
        else:
            state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = "node-a-reimaged"
        return result

    state.read_node = read
    state.patch_namespaced_custom_object = patch_and_vanish
    return state, adapter, validator, context


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param("deleted", id="B-node-deleted"),
        pytest.param("reimaged", id="A-node-reimaged"),
    ],
)
def test_a_stop_whose_node_changed_after_the_suspend_says_the_job_was_stopped(
    shape: str,
) -> None:
    """STOP_WORKLOADS in flight -- shape B (the node replaced: 404) and shape A
    (a HyperPod re-image: the boot changed on the same UID, read as drift):
    UNSAFE record. ``_set_workloads`` patches the job suspended
    (``workload_operations.py:178-196``) and only then
    ``finish_stop_ownership`` re-reads every receipt node
    (``stop_ownership.py:687-701`` -> ``_current`` ``:656-677``); the 404
    escapes as ``unverifiable_refusal`` and the drift as ``refusal`` --
    the pre-submit refusal of ``stop_ownership.py:181-194`` that sets
    ``node_action_not_started``, the key the control plane reads as proof that
    nothing was mutated -- and the receipt is dropped
    (``stop_ownership.py:926-941``). The escalation reads it as an attested
    no-op (``escalation.py:188-230``), nothing ever restarts the job (no
    RESTART_WORKLOAD runs; a passive restart needs a SUCCEEDED predecessor's
    receipt, ``stop_ownership.py:746-790``), and the departed-node cleanup (B) or
    the operator's restore (A) then ends the incident with the job suspended.
    Failed on 9132d9f4 at ``node_action_not_started``; since batch 139 the
    post-suspend refusal (``stop_ownership._suspended_refusal``) keeps the
    receipt, says ``workloads_suspended`` and attests the suspend's outcome,
    so the record neither parks on an unknown outcome nor reads as a no-op.
    Restarting the job is still the operator's (G7, not decided)."""

    state, adapter, validator, context = _vanishing_runtime(shape)

    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)

    assert ("patch-workload", "training", "job") in state.calls
    assert state.job["spec"]["runPolicy"]["suspend"] is True, "the job is stopped"
    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    details = outcome.details or {}
    assert details.get("node_action_not_started") is not True, details
    assert STOP_RECEIPT_KEY in details, "the receipt of the suspended job is kept"
    assert details["workloads_suspended"] is True
    assert details["reason"] == (
        "STOP_OWNERSHIP_UNVERIFIABLE" if shape == "deleted" else "STOP_OWNERSHIP_DRIFT"
    ), details
    assert details[STOP_RECEIPT_KEY]["contained"] is False, (
        "the receipt is the capture, never a contained one a node action could use"
    )
    record = WorkflowStepExecution(
        step_index=0,
        operation=OP.STOP_WORKLOADS,
        status=WorkflowStepStatus.FAILED,
        phase="official",
        error=outcome.error,
        details=details,
    )
    assert unresolved_node_action(record) is False, (
        "the suspend's outcome is known: nothing is in flight to confirm"
    )
    assert "were suspended" in (outcome.error or ""), outcome.error
