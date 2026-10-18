"""A HyperPod in-place re-image overtakes a remediation from RESET_GPU onwards.

Batch 138, analysis lane (shape A, in-flight points 5-10). SageMaker
``UpdateClusterSoftware`` (or an instance group's ``ScheduledUpdateConfig``)
re-images a GPU node in place: the instance id, the node name and the Kubernetes
Node object -- UID, annotations, the product's cordon and quarantine taint --
survive, while the boot id changes and the root disk with the Node Agent and its
ledger is replaced (``node_installer_reconciler.py:39-42``). The installer
reconciler reinstalls the Agent once the node has been Ready for the 600 s reboot
grace (``node_installer_reconciler.py:679-707``); the new Agent's incarnation is a
digest over the boot id (``node_agent/heartbeat.py:261-270``), so the registry
accepts it as a same-instance reboot and bumps its generation
(``fleet.py:515-647``).

What these tests pin, against the real adapters and the in-memory store:

* the STOP receipt already proves the re-image -- a node action after it is
  STOP_OWNERSHIP_DRIFT -- and since batch 139 the release of the node
  (RESTORE_SCHEDULING) is bound to that proof too: a node whose UID or boot
  moved without this workflow's confirmed reboot is not released (point 8, was
  unsafe: batch 138 xfail);
* an accepted RESET_GPU whose Agent vanished with the old root disk was turned
  into a plain FAILED -- outcome no longer unknown -- by the maintenance window
  or the generation fence that run before the result poll, and the branch
  ladder climbed to a reboot on that guess; since batch 139 the gate's FAILED
  keeps the unknown outcome and the command pointer (point 5, batch 138
  xfail);
* a RESTART_NODE overtaken by the re-image and parked at its cap has its exit:
  confirm-node-action proves it once the reinstalled Agent heartbeats (point 10,
  passing).

A RESET of a multi-node job that parks with an unknown outcome is the QUIESCE
shape of ``tests/regional/test_b138_reimage_before_reset.py``: confirm-node-action
takes its pre-action boot from the workflow's own STOP receipt since batch 139,
and the restore-first exit -- ``restore`` of the QUARANTINED incident, then the
confirmation settled by that validated restore -- is pinned there.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.adapters.kubernetes.stop_boot_transition import REBOOT_AUTHORIZATION_KEY
from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    StopOwnershipReceipt,
    node_release_ownership_guard,
    node_submission_ownership_guard,
    stop_ownership_scope,
)
from gpu_fault.cluster_executor.dispatch import CommandDispatch
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.node_action_confirmation import (
    AgentEvidence,
    KubernetesNodeEvidence,
    confirm_node_action,
)
from gpu_fault.execution.node_action_uncertainty import unresolved_node_action
from gpu_fault.hyperpod import hyperpod_submission_idempotency_key
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
from gpu_fault.node_provider import HYPERPOD_REBOOT_CONFIRMATION_SOURCE
from gpu_fault.orchestration.escalation import unknown_outcome_failure
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution.test_branch_escalation import _escalator
from tests.fleet._support import NOW as FLEET_NOW
from tests.fleet._support import SECRET, heartbeat, registry, signed
from tests.regional._late_ownership_runtime import stopped_runtime

OP = WorkflowOperation
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
STARTED = NOW - timedelta(hours=1)
NODE = "hyperpod-i-0aaa"
SIBLING = "hyperpod-i-0bbb"
OLD_BOOT = "boot-before-reimage"
NEW_BOOT = "boot-after-reimage"
OLD_INCARNATION = "incarnation-before-reimage"
NEW_INCARNATION = "incarnation-after-reimage"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"
REFERENCE = "CHG-b138"


# --------------------------------------------------------------------------- #
# point 8 -- RESTORE_SCHEDULING after the re-image
# --------------------------------------------------------------------------- #
def _isolate(node: dict[str, Any], incident_id: str, fencing_token: int) -> None:
    node["metadata"]["annotations"] = {
        ANNOTATION_INCIDENT: incident_id,
        ANNOTATION_FENCING: str(fencing_token),
    }
    node["spec"] = {
        "unschedulable": True,
        "taints": [
            {
                "key": QUARANTINE_TAINT,
                "value": quarantine_taint_value(incident_id),
                "effect": "NoSchedule",
            }
        ],
    }


def _reimaged_release():
    """A contained STOP, a SUCCEEDED reset and validation of node-a, then the
    re-image: same Node object and UID, isolation kept, a new boot id."""

    state, kubernetes, validator, reset = stopped_runtime()

    def patch_node(name: str, body: dict[str, Any]) -> None:
        current = state.nodes[name]
        annotations = current["metadata"].setdefault("annotations", {})
        for key, value in (body.get("metadata", {}).get("annotations") or {}).items():
            if value is None:
                annotations.pop(key, None)
            else:
                annotations[key] = value
        current.setdefault("spec", {}).update(deepcopy(body.get("spec") or {}))
        state.calls.append(("patch-node", name))

    state.patch_node = patch_node
    _isolate(
        state.nodes["node-a"], reset.incident.incident_id, reset.workflow.fencing_token
    )
    validate = WorkflowStepSpec(
        operation=OP.VALIDATE_GPU,
        execution_owner="gpu-fault-validation-adapter",
        node_ids=["node-a"],
        gpu_uuids=["GPU-local"],
    )
    release = WorkflowStepSpec(
        operation=OP.RESTORE_SCHEDULING,
        execution_owner=kubernetes.owner,
        node_ids=["node-a"],
    )
    workflow = reset.workflow.model_copy(
        update={
            "official_steps": [*reset.workflow.official_steps, validate, release],
            "completed_step_indexes": [0, 1, 2],
            "completed_operations": [OP.STOP_WORKLOADS, OP.RESET_GPU, OP.VALIDATE_GPU],
            "step_executions": [
                *reset.workflow.step_executions,
                WorkflowStepExecution(
                    step_index=1,
                    operation=OP.RESET_GPU,
                    status=WorkflowStepStatus.SUCCEEDED,
                    phase="official",
                ),
                WorkflowStepExecution(
                    step_index=2,
                    operation=OP.VALIDATE_GPU,
                    status=WorkflowStepStatus.SUCCEEDED,
                    phase="official",
                    details={"validated_nodes": ["node-a"]},
                ),
            ],
        }
    )
    # UpdateClusterSoftware: the Node object, its UID and every annotation and
    # taint survive; only the boot changes (node_installer_reconciler.py:39-42).
    state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = "node-a-boot-after-reimage"
    context = WorkflowStepContext(
        workflow=workflow,
        incident=reset.incident,
        step=release,
        step_index=3,
        request=reset.request,
        idempotency_key=f"{workflow.request_id}/3/RESTORE_SCHEDULING",
    )
    return state, kubernetes, validator, replace(reset, workflow=workflow), context


def test_the_stop_receipt_already_reads_the_reimage_as_drift_for_a_node_action():
    """Contrast for the release below: the receipt binds node-a's UID and boot
    (``stop_ownership.py:145-178``), and a node-mutating step after the re-image
    is refused as STOP_OWNERSHIP_DRIFT because no reboot of this workflow
    explains the new boot (``stop_ownership.py:656-677``,
    ``stop_boot_transition.py:309-387``). The evidence the release needs is on
    the record; only the release does not read it."""

    _state, _kube, validator, reset, _release = _reimaged_release()
    with stop_ownership_scope(validator):
        refused = node_submission_ownership_guard(reset)
    assert refused is not None, "a re-imaged STOP participant passed the guard"
    assert refused.details is not None
    assert refused.details["reason"] == "STOP_OWNERSHIP_DRIFT"
    assert refused.details["node_action_not_started"] is True


def test_restore_scheduling_refuses_to_release_a_node_reimaged_after_its_validation():
    """Shape A, point 8 (was unsafe, fixed in batch 139). VALIDATE_GPU
    succeeded on samples the node posted before the re-image -- the adapter
    only needs samples after the reset's completion and younger than six
    minutes, with no boot binding (``gpu_validation.py:139-155``,
    ``gpu_validation.py:320-361``); the re-image then lands before
    RESTORE_SCHEDULING. Before the fix the release ran through the real
    cluster-executor boundary with the STOP validator in scope and uncordoned
    the node, stripped the quarantine taint and the incident annotations -- a
    node on a new image whose Agent and collectors are not reinstalled yet
    went back to scheduling. Now the release re-reads the receipt's node
    (``stop_ownership.py`` ``check_release``): the boot drift the receipt
    proves refuses it, nothing is patched, and the node keeps its isolation
    for an operator's validated restore."""

    state, kubernetes, validator, _reset, release = _reimaged_release()
    dispatch = CommandDispatch(SimpleNamespace(stop_ownership_validator=validator))

    outcome = dispatch.execute_adapter(kubernetes, release)

    node = state.nodes["node-a"]
    assert outcome.status is WorkflowStepStatus.FAILED, (
        f"the re-imaged node was released: {outcome.status.value} {outcome.details}"
    )
    details = outcome.details or {}
    assert details.get("safety_rejection") is True
    assert details["reason"] == "STOP_OWNERSHIP_DRIFT", details
    assert details["node_id"] == "node-a"
    assert details["receipt_node"]["boot_id"] == "node-a-boot"
    assert details["observed_node"]["boot_id"] == "node-a-boot-after-reimage"
    assert ("patch-node", "node-a") not in state.calls, "the node was patched"
    assert node["spec"]["unschedulable"] is True, "the release uncordoned the node"
    assert any(
        taint.get("key") == QUARANTINE_TAINT for taint in node["spec"]["taints"]
    ), "the release stripped the quarantine taint"
    assert node["metadata"]["annotations"][ANNOTATION_INCIDENT] == (
        release.incident.incident_id
    ), "the release dropped the incident annotation"


def test_restore_scheduling_releases_a_node_whose_boot_did_not_move():
    """Control for the re-check: the same release on the boot the receipt
    bound uncordons node-a and strips the taint and the annotations."""

    state, kubernetes, validator, _reset, release = _reimaged_release()
    state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = "node-a-boot"
    dispatch = CommandDispatch(SimpleNamespace(stop_ownership_validator=validator))

    outcome = dispatch.execute_adapter(kubernetes, release)

    node = state.nodes["node-a"]
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["restored_nodes"] == ["node-a"]
    assert node["spec"]["unschedulable"] is False
    assert not any(
        taint.get("key") == QUARANTINE_TAINT for taint in node["spec"]["taints"]
    ), "the release keeps the quarantine taint"


def test_restore_scheduling_is_not_blocked_by_a_node_it_cannot_read():
    """Only observed drift refuses the release. A Node read that raises (here
    the API server answers 500) is no drift: the re-check stands aside and the
    restore patch, which reads the node again, decides -- an unreadable API
    never strands a compensating release."""

    state, _kubernetes, validator, _reset, release = _reimaged_release()

    class Unreadable(Exception):
        status = 500

    def unreadable(name: str) -> Any:
        raise Unreadable(name)

    state.read_node = unreadable
    with stop_ownership_scope(validator):
        assert node_release_ownership_guard(release) is None


def test_restore_scheduling_counts_a_deleted_node_as_absent():
    """A Node the cluster deleted (404) is no drift either: the re-check
    stands aside and the restore patch counts the node absent, as before."""

    state, kubernetes, validator, _reset, release = _reimaged_release()

    class Gone(Exception):
        status = 404

    def gone(name: str) -> Any:
        raise Gone(name)

    state.read_node = gone
    dispatch = CommandDispatch(SimpleNamespace(stop_ownership_validator=validator))

    outcome = dispatch.execute_adapter(kubernetes, release)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["absent_nodes"] == ["node-a"]


def test_restore_scheduling_follows_a_reboot_the_workflow_confirmed():
    """A reboot chain changes the boot on purpose: the RESTART_NODE record this
    workflow submitted under the receipt's authorization and the provider and
    the new Agent incarnation confirmed (``stop_boot_transition``, shape 1)
    accounts for the new boot, so the release proceeds."""

    state, kubernetes, validator, reset, release = _reimaged_release()
    workflow = release.workflow
    receipt = workflow.step_executions[0].details[STOP_RECEIPT_KEY]
    stopped_at = datetime.fromisoformat(receipt["completed_at"])
    digest = StopOwnershipReceipt.model_validate_json(json.dumps(receipt)).digest()
    reboot_index = len(workflow.official_steps)
    reboot = WorkflowStepSpec(
        operation=OP.RESTART_NODE,
        execution_owner="gpu-fault-hyperpod-adapter",
        node_ids=["node-a"],
    )
    key = hyperpod_submission_idempotency_key(
        workflow.request_id, reboot_index, OP.RESTART_NODE
    )
    # The chain may not end after "now"; the STOP completed a moment ago, so
    # the reboot is stamped at the receipt's own instant.
    submitted = confirmed = stopped_at
    record = WorkflowStepExecution(
        step_index=reboot_index,
        operation=OP.RESTART_NODE,
        status=WorkflowStepStatus.SUCCEEDED,
        phase="official",
        adapter_operation_id="remote/remote-reboot",
        started_at=submitted,
        updated_at=confirmed,
        details={
            "action": "REBOOT",
            "submission_idempotency_key": key,
            "externally_confirmed": True,
            "confirmation_source": HYPERPOD_REBOOT_CONFIRMATION_SOURCE,
            "submitted_nodes": ["i-0reboot"],
            "agent_baselines": {
                "node-a": {"boot_id": "node-a-boot", "agent_incarnation_id": "inc-1"}
            },
            "agent_observations": [
                {
                    "node_id": "node-a",
                    "boot_id": "node-a-boot-after-reimage",
                    "agent_incarnation_id": "inc-2",
                }
            ],
            "provider_observations": [
                {
                    "node_id": "node-a",
                    "node_logical_id": "i-0reboot",
                    "instance_id": "i-0reboot",
                    "status": "Running",
                }
            ],
            "observed_isolation": {
                "node-a": {
                    "kubernetes_node": "node-a",
                    "unschedulable": True,
                    "incident": release.incident.incident_id,
                    "fencing_token": str(workflow.fencing_token),
                }
            },
            REBOOT_AUTHORIZATION_KEY: {
                "version": 1,
                "cluster_id": release.incident.cluster_id,
                "workflow_id": workflow.request_id,
                "incident_id": release.incident.incident_id,
                "fencing_token": workflow.fencing_token,
                "execution_epoch": workflow.execution_epoch,
                "phase": "official",
                "step_index": reboot_index,
                "execution_owner": reboot.execution_owner,
                "submission_idempotency_key": key,
                "stop_receipt_sha256": digest,
                "nodes": {"node-a": {"uid": "node-a-uid", "boot_id": "node-a-boot"}},
                "checked_at": submitted.isoformat(),
            },
        },
    )
    rebooted = workflow.model_copy(
        update={
            "official_steps": [*workflow.official_steps, reboot],
            "completed_step_indexes": [*workflow.completed_step_indexes, reboot_index],
            "step_executions": [*workflow.step_executions, record],
        }
    )
    context = replace(release, workflow=rebooted)
    dispatch = CommandDispatch(SimpleNamespace(stop_ownership_validator=validator))

    outcome = dispatch.execute_adapter(kubernetes, context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert state.nodes["node-a"]["spec"]["unschedulable"] is False


# --------------------------------------------------------------------------- #
# point 5 -- an accepted RESET_GPU whose Agent went with the old root disk
# --------------------------------------------------------------------------- #
RESET_KEY = "workflow-a/1/RESET_GPU"
ACCEPTED_COMMAND = f"{RESET_KEY}/node-a/agent-1"


def _accepted_reset(variant: str):
    """QUIESCE pinned generation 1; the RESET was accepted, then the Agent
    stopped answering (the poll's record, transport.py:1199-1205 and
    transport.py:892-893, is outcome_unknown with the command pointer)."""

    clock = [FLEET_NOW]
    store = build_store()
    fleet = registry(store, now=lambda: clock[0])
    fleet.register(
        signed(
            heartbeat("node-a", boot_id=OLD_BOOT, agent_incarnation_id=OLD_INCARNATION)
        )
    )
    window = timedelta(seconds=420) if variant == "window" else timedelta(hours=1)
    steps = [
        workflow_step(
            OP.QUIESCE_GPU_SERVICES,
            "gpu-fault-node-agent",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
        ),
        workflow_step(
            OP.RESET_GPU,
            "gpu-fault-node-agent",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
            parameters={"gpu_uuids_by_node": {"node-a": ["GPU-a"]}},
        ),
    ]
    pending = {
        "node_action_command_id": ACCEPTED_COMMAND,
        "node_action_state": "PENDING",
        "node_action_accepted_nodes": ["node-a"],
        "node_action_response_unknown": True,
        "outcome_unknown": True,
        "manual_confirmation_required": True,
        "waiting_node": "node-a",
        "reason": "OWNERSHIP_PROTOCOL_UNVERIFIABLE",
    }
    workflow = workflow_request(
        "workflow-a",
        "incident-a",
        status=WorkflowStatus.RUNNING,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[OP.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(
                0,
                OP.QUIESCE_GPU_SERVICES,
                phase="official",
                details={
                    "agent_generations": {"node-a": 1},
                    "maintenance_window_started_at": FLEET_NOW.isoformat(),
                    "maintenance_window_expires_at": (FLEET_NOW + window).isoformat(),
                },
            ),
            workflow_step_execution(
                1,
                OP.RESET_GPU,
                WorkflowStepStatus.WAITING,
                phase="official",
                details=pending,
            ),
        ],
    )
    incident = fault_incident(
        "incident-a",
        "event-a",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        fencing_token=3,
        workflow_request_id="workflow-a",
    )
    if variant == "window":
        # Re-image in progress: the old record is still generation 1, its lease
        # long expired; the 420 s quiesce window ran out first.
        clock[0] = FLEET_NOW + timedelta(minutes=10)
    else:
        # Re-image finished: the reinstalled Agent registered on the same
        # instance with a new boot, so the registry bumped the generation.
        clock[0] = FLEET_NOW + timedelta(minutes=30)
        fleet.register(
            signed(
                heartbeat(
                    "node-a",
                    observed_at=clock[0],
                    boot_id=NEW_BOOT,
                    agent_incarnation_id=NEW_INCARNATION,
                )
            )
        )
        assert store.get_agent("cluster-a", "node-a").generation == 2
    sent: list[Any] = []
    adapter = NodeActionWorkflowAdapter(
        {},
        SECRET,
        registry=fleet,
        sender=lambda _endpoint, envelope: sent.append(envelope),
        maintenance_window_seconds=3600,
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=steps[1],
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=RESET_KEY,
    )
    return adapter, context, sent


@pytest.mark.parametrize("variant", ["window", "generation"])
def test_an_accepted_reset_keeps_its_unknown_outcome_past_the_window_and_the_fence(
    variant,
):
    """Shape A, point 5, both timings. ``window``: the re-image is in
    progress, the poll has been answering outcome_unknown, and the next poll
    lands after the quiesce window: ``quiesce maintenance window expired``
    (barriers.py:187-200). ``generation``: the reinstalled Agent registered:
    ``maintenance agent fence failed ... agent generation changed from 1 to 2``
    (barriers.py:202-217). Either FAILED is posted verbatim by the cluster
    executor (dispatch.py:195-230) and mapped to a FAILED step by the control
    plane (regional.py:995-1011): the uncertainty the previous record carried
    is gone, ``unresolved_node_action`` is false, and the in-place ladder
    appends a reboot (branch_escalation.py:288-322, escalation.py:499-526),
    the climb the unknown-outcome rule exists to forbid (escalation.py:96-110).
    Physically contained in shape A -- HyperPod's preflight refuses lifecycle
    calls on a cluster that is not InService or a node not Running
    (hyperpod.py:654-680), and after the re-image the STOP guard refuses the
    rung as drift -- but the record lied and the classification climbed. Since
    batch 139 the gate's FAILED keeps the unknown outcome and the command
    pointer (``step_execution.keep_accepted_uncertainty``): the record parks
    for ``confirm-node-action`` and neither ladder climbs."""

    adapter, context, sent = _accepted_reset(variant)

    outcome = adapter.execute(context)

    details = dict(outcome.details or {})
    assert sent == [], "the fenced reset was re-sent to the new incarnation"
    assert unknown_outcome_failure(details), (
        f"the accepted reset lost its unknown outcome: {outcome.status.value} "
        f"{outcome.error!r} {details}"
    )
    assert details.get("node_action_command_id") == ACCEPTED_COMMAND
    failed = WorkflowStepExecution(
        step_index=1,
        operation=OP.RESET_GPU,
        status=WorkflowStepStatus.FAILED,
        phase="official",
        details=details,
    )
    assert unresolved_node_action(failed), "the record no longer parks the reset"
    chain = context.workflow.model_copy(
        update={
            "official_steps": [
                workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
                *context.workflow.official_steps,
                workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=["node-a"]),
            ],
            "step_executions": [
                workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
                workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES),
                failed.model_copy(update={"step_index": 2}),
            ],
            "completed_step_indexes": [0, 1],
        }
    )
    assert _escalator().escalate_branch(chain, 2, outcome.error, details=details) is (
        None
    ), "the ladder climbed a rung over a reset whose outcome is unknown"


# --------------------------------------------------------------------------- #
# point 10 -- a RESTART_NODE in flight when the re-image runs
# --------------------------------------------------------------------------- #
def _node(boot_id: str, node_id: str = NODE) -> KubernetesNodeEvidence:
    return KubernetesNodeEvidence.from_mapping(
        {
            "node_id": node_id,
            "uid": f"uid-{node_id}",
            "boot_id": boot_id,
            "ready": True,
            "ready_since": (NOW - timedelta(minutes=5)).isoformat(),
            "unschedulable": True,
        }
    )


def _agent(
    boot_id: str,
    incarnation: str,
    *,
    state: str = "ACTIVE",
    alive: bool = True,
    node_id: str = NODE,
) -> AgentEvidence:
    return AgentEvidence.from_mapping(
        {
            "node_id": node_id,
            "boot_id": boot_id,
            "agent_incarnation_id": incarnation,
            "lifecycle_state": state,
            "generation": 9,
            "last_seen_at": (
                NOW - timedelta(seconds=10) if alive else STARTED - timedelta(minutes=1)
            ).isoformat(),
            "lease_expires_at": (
                NOW + timedelta(seconds=50) if alive else STARTED
            ).isoformat(),
            "retired_incarnation_ids": [OLD_INCARNATION],
        }
    )


def _parked_reboot():
    """RESTART_NODE submitted (lifecycle.py:770-835: Agent revoked, baselines
    recorded), then the re-image: SystemUpdating is never ``Running``
    (confirmation.py:139-143) and the reinstall outlasts the 1800 s cap, so the
    step fails outcome_unknown and the record parks NEEDS_OPERATOR
    (step_bounds.py:201-229, node_action_uncertainty.py:149-158,
    terminal_state.py:176-189)."""

    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[NODE]),
        workflow_step(OP.RESTART_NODE, node_ids=[NODE]),
    ]
    workflow = workflow_request(
        "workflow-reboot",
        "incident-reboot",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[OP.MARK_UNSCHEDULABLE],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE, phase="official"),
            workflow_step_execution(
                1,
                OP.RESTART_NODE,
                WorkflowStepStatus.FAILED,
                phase="official",
                adapter_operation_id="remote/remote-reboot",
                error=(
                    "step 1/RESTART_NODE stayed non-terminal for 1800s, past the "
                    "1800s per-step cap"
                ),
                details={
                    "action": "REBOOT",
                    "submitted_nodes": [NODE],
                    "revoked_agents": [NODE],
                    "agent_baselines": {
                        NODE: {
                            "boot_id": OLD_BOOT,
                            "agent_incarnation_id": OLD_INCARNATION,
                        }
                    },
                    "requires_external_confirmation": True,
                    "outcome_unknown": True,
                    "manual_confirmation_required": True,
                    "step_waiting_timeout_seconds": 1800,
                    "remote_command_id": "remote-reboot",
                },
                started_at=STARTED,
                updated_at=STARTED + timedelta(minutes=30),
            ),
        ],
    )
    incident = fault_incident(
        "incident-reboot",
        "event-reboot",
        node_ids=[NODE],
        state=IncidentState.QUARANTINED,
        workflow_request_id=workflow.request_id,
        fencing_token=1,
    )
    return workflow, incident


def test_a_parked_reboot_overtaken_by_a_reimage_is_confirmed_once_the_agent_is_back():
    """Shape A, point 10: not a dead end. Until the reconciler reinstalls the
    Agent the verdict refuses and says why (the record the lifecycle adapter
    revoked, no fresh heartbeat); once the reinstalled Agent heartbeats on the
    new boot the boot transition proves the reboot -- the re-image is one --
    from the step's own ``agent_baselines`` (node_action_confirmation.py:
    498-576, 662-706). The operator's ``restore`` then validates the
    re-imaged node before it is released."""

    workflow, incident = _parked_reboot()

    waiting = confirm_node_action(
        workflow=workflow,
        incident=incident,
        node_id=NODE,
        node=_node(NEW_BOOT),
        agent=_agent(OLD_BOOT, OLD_INCARNATION, state="REVOKED", alive=False),
        remote_commands=[],
        actor=ACTOR,
        reference=REFERENCE,
        now=NOW,
    )
    confirmed = confirm_node_action(
        workflow=workflow,
        incident=incident,
        node_id=NODE,
        node=_node(NEW_BOOT),
        agent=_agent(NEW_BOOT, NEW_INCARNATION),
        remote_commands=[],
        actor=ACTOR,
        reference=REFERENCE,
        now=NOW,
    )

    assert waiting.confirmations == ()
    assert any("fresh heartbeat is required" in item for item in waiting.refusals), (
        waiting.refusals
    )
    assert confirmed.refusals == (), confirmed.refusals
    (confirmation,) = confirmed.confirmations
    assert confirmation["settled_by"] == "boot_transition"
    assert confirmation["previous_boot_id"] == OLD_BOOT
    assert confirmation["observed_boot_id"] == NEW_BOOT
