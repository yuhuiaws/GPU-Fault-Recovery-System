from __future__ import annotations

import io
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    StopOwnershipReceipt,
)
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent.late_ownership import (
    OwnershipChallenge,
    ownership_recheck_scope,
)
from gpu_fault.node_agent.protocol import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
)
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_contract import (
    NodeIdentity,
    Participant,
    RecheckPermit,
    WorkloadIdentity,
)
from scripts.e2e.regional.late_ownership_control import check_workflow, hold_workflow
from scripts.e2e.regional.late_ownership_live import workflow_inputs
from scripts.e2e.regional.probes import late_ownership_executor_probe as probe
from tests.regional._late_ownership_runtime import runtime
from tests.regional._late_ownership_support import evidence, scope


def probe_channels(binding, state, store, starts, messages, replies, get_probe):
    def queue(kind, payload):
        replies.append(
            json.dumps(
                {"kind": kind, "scope_sha256": binding.digest(), "payload": payload}
            )
            + "\n"
        )

    class Incoming:
        def readline(self, limit):
            assert replies, "probe requested an unacknowledged causal stage"
            return replies.pop(0)

    class Outgoing(io.StringIO):
        callback = None

        def write(self, text):
            result = get_probe()
            message = json.loads(text)
            messages.append(message)
            kind, payload = message["kind"], message["payload"]
            if self.callback is not None and self.callback(kind, payload):
                return len(text)
            if kind == "holder-check":
                checked = check_workflow(store, result.workflow, run_id=binding.run_id)
                queue("holder-check-result", {**checked, "nonce": payload["nonce"]})
            elif kind == "ready":
                queue("calibrate", {})
            elif kind == "calibrated":
                queue(
                    "begin",
                    {
                        "witness_starts": [
                            item.model_dump(mode="json") for item in starts
                        ]
                    },
                )
            elif kind == "contained":
                assert state.pods == []
                queue("continue-boundary", {})
            elif kind == "stop":
                assert result.node_commands, (
                    "the accepted command must be recorded before parking"
                )
                if binding.scenario == "ownership-drift":
                    state.job["metadata"]["ownerReferences"] = [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": "new-owner",
                            "uid": "new-owner-uid",
                            "controller": False,
                        }
                    ]
                if binding.scenario == "late-sibling":
                    state.pods.append(state.pod("node-b", "late-pod"))
                    queue(
                        "observe-mutation",
                        {"sibling_uid": "late-pod", "physical_client_verified": True},
                    )
                else:
                    queue("observe-mutation", {})
            elif kind == "mutation":
                permit = RecheckPermit(
                    scope_sha256=binding.digest(),
                    boundary_id=result.stopped.boundary_id,
                    stop_sha256=result.stopped.digest(),
                    mutation_sha256=result.mutation.digest(),
                )
                queue("recheck", permit.model_dump(mode="json"))
            elif kind == "decision":
                queue("quiesce", {})
            elif kind == "services-restored":
                queue("restore-scheduling", {})
            elif kind == "actions-drained":
                queue(
                    "confirm-terminal",
                    {
                        "workflow_id": binding.workflow_id,
                        "fencing_token": binding.fencing_token,
                        "status": "SUPERSEDED",
                        "execution_epoch": binding.execution_epoch,
                    },
                )
            elif kind == "quiescence":
                queue("revoke", {})
            elif kind == "revoked":
                queue("finish", {})
            else:
                assert kind == "finished"
            return len(text)

    return Incoming(), Outgoing(), queue


@pytest.fixture
def owned_probe(monkeypatch):
    def create(scenario="ownership-drift"):
        state, kube, validator, context = runtime()
        base = scope(scenario)
        nodes = tuple(
            NodeIdentity(
                name=name,
                uid=raw["metadata"]["uid"],
                boot_id=raw["status"]["nodeInfo"]["bootID"],
            )
            for name, raw in state.nodes.items()
        )
        participants = tuple(
            Participant(
                pod_uid=pod["metadata"]["uid"],
                owner_uid="job-uid",
                node_uid=nodes[index].uid,
            )
            for index, pod in enumerate(state.pods)
        )
        binding = base.model_copy(
            update={
                "cluster_id": "cluster-local",
                "fencing_token": 1,
                "execution_epoch": 1,
                "nodes": nodes,
                "participants": participants,
                "workload": WorkloadIdentity(
                    namespace="training",
                    name="job",
                    uid="job-uid",
                    owner_uid="job-uid",
                    attempt_id="attempt",
                ),
            }
        )
        run = SimpleNamespace(
            settings=SimpleNamespace(
                nodes=tuple(node.name for node in nodes), job_id="job"
            ),
            baselines={
                node.name: {"gpu_inventory": [{"uuid": "GPU-a", "pci_bdf": "bdf"}]}
                for node in nodes
            },
            bdf={node.name: "bdf" for node in nodes},
        )
        workflow, incident = workflow_inputs(run, binding)
        store = InMemoryStore()
        held = hold_workflow(
            store,
            workflow,
            incident,
            run_id=binding.run_id,
            window_end=binding.maintenance_end,
        )
        workflow = WorkflowRequest.model_validate(held["workflow"])
        node = NodeActionWorkflowAdapter(
            {"node-a": "http://owned.invalid"}, "local-test-only-" + "x" * 32
        )
        executor = SimpleNamespace(
            adapters=[kube, node],
            stop_ownership_validator=validator,
            client=SimpleNamespace(cluster_id=binding.cluster_id),
            allowed_namespaces={"training"},
        )
        messages, replies, operations = [], [], []
        starts = tuple(
            item.model_copy(
                update={
                    "scope_sha256": binding.digest(),
                    "executor_uid": binding.executor_uid,
                    "node": nodes[index],
                    "producer": item.producer.model_copy(
                        update={"boot_id": nodes[index].boot_id}
                    ),
                    "tracee": item.tracee.model_copy(
                        update={"boot_id": nodes[index].boot_id}
                    ),
                }
            )
            for index, item in enumerate(evidence(scenario).witness_starts)
        )

        incoming, outgoing, queue = probe_channels(
            binding, state, store, starts, messages, replies, lambda: result
        )
        result = probe.ExecutorProbe(
            binding, workflow, incident, executor, incoming=incoming, outgoing=outgoing
        )
        monkeypatch.setattr(
            probe.select,
            "select",
            lambda readers, w, x, seconds: (
                (readers if seconds > 0 and replies else []),
                [],
                [],
            ),
        )
        real_kube_execute = kube.execute

        def kube_execute(current):
            operations.append((current.step.operation, current.step.node_ids))
            if current.step.operation is WorkflowOperation.STOP_WORKLOADS:
                return real_kube_execute(current)
            return WorkflowStepOutcome.succeeded()

        monkeypatch.setattr(kube, "execute", kube_execute)
        monkeypatch.setattr(node, "read_ownership_capability", lambda *a: True)

        def node_execute(current):
            operations.append((current.step.operation, current.step.node_ids))
            if current.step.operation is WorkflowOperation.RESET_GPU:
                challenge = OwnershipChallenge(
                    command_id=current.idempotency_key,
                    workflow_id=binding.workflow_id,
                    incident_id=binding.incident_id,
                    node_id=current.step.node_ids[0],
                    boot_id=nodes[0].boot_id,
                    agent_generation=1,
                    fencing_token=1,
                    command_sha256="a" * 64,
                    nonce="c" * 64,
                    sequence=1,
                    boundary="AGENT_PRE_SPAWN",
                    expires_at=binding.maintenance_end,
                )
                with ownership_recheck_scope(challenge):
                    refused = validator.check(current)
                if refused is not None:
                    return refused
            return WorkflowStepOutcome.succeeded(
                details={"node_action_command_id": current.idempotency_key}
            )

        monkeypatch.setattr(node, "execute", node_execute)

        def terminal(cluster_id, node_id, command_id):
            return NodeActionSubmission(
                command_id=command_id,
                state=NodeActionExecutionState.SUCCEEDED,
                result=NodeActionResult(
                    command_id=command_id,
                    operation=result.node_operations[command_id],
                    status=NodeActionStatus.SUCCEEDED,
                ),
            )

        monkeypatch.setattr(node, "read_action_result", terminal)
        return SimpleNamespace(
            probe=result,
            state=state,
            kube=kube,
            node=node,
            validator=validator,
            scope=binding,
            messages=messages,
            replies=replies,
            queue=queue,
            outgoing=outgoing,
            operations=operations,
            starts=starts,
            original_context=context,
            store=store,
        )

    return create


@pytest.mark.parametrize(
    "scenario", ["unchanged-owner", "ownership-drift", "late-sibling"]
)
def test_gpu_probe_runs_real_stop_and_native_callback_protocol_then_compensates(
    owned_probe, scenario
):
    owned = owned_probe(scenario)
    owned.probe.run()
    messages = [value for value in owned.messages if value["kind"] != "holder-check"]
    assert [value["kind"] for value in messages] == [
        "ready",
        "calibrated",
        "contained",
        "stop",
        "mutation",
        "decision",
        "services-restored",
        "actions-drained",
        "quiescence",
        "revoked",
        "finished",
    ]
    assert any(value["kind"] == "holder-check" for value in owned.messages), (
        "the probe must verify its CPU workflow holder during execution"
    )
    decision = messages[5]["payload"]
    assert (
        decision["decision"]
        == {
            "unchanged-owner": "ALLOWED",
            "ownership-drift": "STOP_OWNERSHIP_DRIFT",
            "late-sibling": "STOP_PARTICIPANTS_CHANGED",
        }[scenario]
    )
    assert decision["checked_node_uids"] == [node.uid for node in owned.scope.nodes]
    assert (
        messages[3]["payload"]["queued_command_id"]
        == owned.probe.stopped.queued_command_id
    )
    assert owned.probe.revoked and owned.validator.before_recheck is None
    operations = [operation for operation, _nodes in owned.operations]
    assert operations.count(WorkflowOperation.RESET_GPU) == (
        2 if scenario == "unchanged-owner" else 1
    )
    assert operations.index(WorkflowOperation.RESTORE_GPU_SERVICES) < operations.index(
        WorkflowOperation.RESTORE_SCHEDULING
    )
    assert owned.replies == []
    with pytest.raises(BoundaryDenied, match="cleanup"):
        owned.probe.run_step(0, cleanup=True)


def test_probe_refuses_legacy_agent_before_stop(owned_probe, monkeypatch):
    owned = owned_probe()
    monkeypatch.setattr(owned.node, "read_ownership_capability", lambda *a: False)
    with pytest.raises(BoundaryDenied, match="Agent"):
        owned.probe.run()
    assert owned.messages == [] and owned.operations == []


@pytest.mark.parametrize(
    "defect", ["eof", "oversized", "newline", "kind", "scope", "payload", "keys"]
)
def test_probe_incoming_protocol_refuses_late_or_unbound_messages(owned_probe, defect):
    owned = owned_probe()
    owned.queue("expected", {})
    if defect == "eof":
        owned.replies[0] = ""
    elif defect == "oversized":
        owned.replies[0] = "x" * (probe.MAX_MESSAGE_BYTES + 1) + "\n"
    elif defect == "newline":
        owned.replies[0] = "{}"
    else:
        data = json.loads(owned.replies[0])
        if defect == "kind":
            data["kind"] = "old-callback"
        elif defect == "scope":
            data["scope_sha256"] = "a" * 64
        elif defect == "payload":
            data["payload"] = []
        else:
            data["extra"] = True
        owned.replies[0] = json.dumps(data) + "\n"
    with pytest.raises(BoundaryDenied):
        owned.probe.receive("expected")
    assert owned.probe.revoked, (
        "invalid control input must revoke the probe's execution authority"
    )


def test_probe_timeout_and_oversized_output_never_release_mutation(owned_probe):
    owned = owned_probe()
    with pytest.raises(BoundaryDenied, match="responding"):
        owned.probe.receive("expected")
    with pytest.raises(BoundaryDenied, match="bound"):
        owned.probe.emit("ready", {"data": "x" * probe.MAX_MESSAGE_BYTES})
    owned.probe.cleanup_deadline = time.monotonic() - 1
    with pytest.raises(BoundaryDenied, match="window"):
        owned.probe.receive("finish", cleanup=True)
    assert owned.operations == []


def test_controller_loss_and_expiry_revoke_the_lease_guard(owned_probe, monkeypatch):
    owned = owned_probe()
    assert owned.probe.lease_reason() is None
    owned.queue("abort", {})
    monkeypatch.setattr(
        probe.select, "select", lambda *args: ([owned.probe.incoming], [], [])
    )
    assert owned.probe.lease_reason() is not None and owned.probe.revoked


@pytest.mark.parametrize(
    "defect",
    ["none", "wrong-command", "no-result", "wrong-operation", "state", "pending"],
)
def test_drainage_requires_exact_authenticated_terminal_command_receipt(
    owned_probe, monkeypatch, defect
):
    owned = owned_probe()
    current = owned.probe
    current.node_commands = {"owned-command": "node-a"}
    current.node_operations = {"owned-command": WorkflowOperation.RESET_GPU}
    current.cleanup_deadline = time.monotonic() + 10
    value = owned.node.read_action_result("cluster-local", "node-a", "owned-command")
    if defect == "none":
        value = None
    elif defect == "wrong-command":
        value = value.model_copy(update={"command_id": "other"})
    elif defect == "no-result":
        value = value.model_copy(update={"result": None})
    elif defect == "wrong-operation":
        value = value.model_copy(
            update={
                "result": value.result.model_copy(
                    update={"operation": WorkflowOperation.RESTART_NODE}
                )
            }
        )
    elif defect == "state":
        value = value.model_copy(update={"state": NodeActionExecutionState.FAILED})
    else:
        value = value.model_copy(
            update={"state": NodeActionExecutionState.PENDING, "result": None}
        )
        owned.queue("abort", {})
    monkeypatch.setattr(owned.node, "read_action_result", lambda *args: value)
    with pytest.raises(BoundaryDenied):
        current.drain_actions()


def test_uid_bound_node_reads_and_writes_never_follow_replacement(owned_probe):
    owned = owned_probe()
    core = owned.probe.kube.core
    with pytest.raises(BoundaryDenied, match="target"):
        core.read_node("foreign-node")
    with pytest.raises(BoundaryDenied, match="structured"):
        core.patch_node("node-a", [])
    owned.state.nodes["node-a"]["metadata"]["uid"] = "new-node"
    with pytest.raises(BoundaryDenied, match="UID"):
        core.read_node("node-a")


def test_node_patch_carries_server_checked_uid_and_resource_version(owned_probe):
    owned = owned_probe()
    patches = []
    owned.state.patch_node = (
        lambda name, body, **kw: patches.append((name, body)) or body
    )
    original = {"spec": {"unschedulable": True}}
    result = owned.probe.kube.core.patch_node("node-a", original)
    assert result["metadata"] == {"uid": "node-a-uid", "resourceVersion": "1"}
    assert original == {"spec": {"unschedulable": True}}
    assert patches[0][0] == "node-a"


@pytest.mark.parametrize("defect", ["scope", "unsupported", "node", "adapters"])
def test_probe_inputs_are_bound_before_any_product_execution(owned_probe, defect):
    owned = owned_probe()
    current = owned.probe
    executor = current.executor
    workflow = current.workflow
    binding = owned.scope
    if defect == "scope":
        binding = binding.model_copy(update={"fencing_token": 2})
    elif defect == "unsupported":
        workflow = workflow.model_copy(
            update={
                "official_steps": [
                    workflow.official_steps[0].model_copy(
                        update={"operation": WorkflowOperation.REPLACE_NODE}
                    )
                ]
            }
        )
    elif defect == "node":
        workflow = workflow.model_copy(
            update={
                "official_steps": [
                    step.model_copy(update={"node_ids": ["foreign"]})
                    for step in workflow.official_steps
                ]
            }
        )
    else:
        executor.adapters = []
    with pytest.raises(BoundaryDenied):
        probe.ExecutorProbe(binding, workflow, current.incident, executor)
    assert owned.operations == []


def test_current_workload_refuses_ambiguous_owner_and_callback_requires_armed_session(
    owned_probe,
):
    owned = owned_probe()
    owned.state.job["metadata"]["ownerReferences"] = [{"uid": "a"}, {"uid": "b"}]
    with pytest.raises(BoundaryDenied, match="ambiguous"):
        owned.probe.current_workload()
    owned.state.job["metadata"].pop("ownerReferences")
    context = replace(
        owned.original_context,
        step=owned.probe.workflow.official_steps[4],
        step_index=4,
    )
    challenge = OwnershipChallenge(
        command_id="queued",
        workflow_id=owned.scope.workflow_id,
        incident_id=owned.scope.incident_id,
        node_id="node-a",
        boot_id=owned.scope.nodes[0].boot_id,
        agent_generation=1,
        fencing_token=1,
        command_sha256="a" * 64,
        nonce="b" * 64,
        sequence=1,
        expires_at=owned.scope.maintenance_end,
        boundary="AGENT_PRE_SPAWN",
    )
    owned.probe.at_physical_boundary(context, None)
    with (
        ownership_recheck_scope(challenge),
        pytest.raises(BoundaryDenied, match="armed"),
    ):
        owned.probe.at_physical_boundary(context, None)


def test_actual_stop_receipt_is_preserved_in_the_probe_workflow(owned_probe):
    owned = owned_probe("unchanged-owner")
    outcome = owned.probe.run_step(0)
    receipt = StopOwnershipReceipt.model_validate_json(
        json.dumps(outcome.details[STOP_RECEIPT_KEY])
    )
    assert receipt.contained, (
        "STOP must preserve a receipt proving source workload containment"
    )
    assert {pod.uid for pod in receipt.pods} == {
        pod.pod_uid for pod in owned.scope.participants
    }
    assert (
        owned.probe.workflow.step_executions[-1].status is WorkflowStepStatus.SUCCEEDED
    )
    assert owned.probe.workflow.completed_step_indexes == [0]


@pytest.mark.parametrize("defect", ["owner", "fence", "epoch", "terminal", "nonce"])
def test_cpu_ownership_loss_or_stale_ack_prevents_product_stop(owned_probe, defect):
    owned = owned_probe()
    current = owned.store.get_workflow(owned.scope.workflow_id)
    changes = {
        "owner": {"execution_owner_id": "other"},
        "fence": {"fencing_token": current.fencing_token + 1},
        "epoch": {"execution_epoch": current.execution_epoch + 1},
        "terminal": {"status": WorkflowStatus.SUPERSEDED},
    }
    if defect == "nonce":

        def callback(kind, payload):
            if kind == "holder-check":
                owned.queue(
                    "holder-check-result",
                    {
                        **payload,
                        "nonce": "0" * 64,
                        "holder_valid": True,
                        "lifetime_deadline_at": owned.scope.maintenance_end.isoformat(),
                    },
                )
                return True
            return False

        owned.outgoing.callback = callback
    else:
        owned.store.save_workflow(
            current.model_copy(update=changes[defect]), expected=current
        )
    with pytest.raises(BoundaryDenied, match="lease was lost"):
        owned.probe.run_step(0)
    assert owned.probe.revoked and owned.operations == []
    assert owned.state.job["spec"]["runPolicy"]["suspend"] is False
