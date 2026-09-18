from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import (
    node_submission_ownership_guard,
    stop_ownership_scope,
)
from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.cluster_executor.dispatch import CommandDispatch
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    OwnershipChallenge,
    command_identity,
    sign_permit,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    sign_node_action,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.regional import BatchedStep, RemoteActionCommand, RemoteCommandStatus
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY
from tests._builders import build_store
from tests.execution.test_branch_escalation import (
    BUNDLE,
    REBOOT,
    RESET,
    RESTART_JOB,
    VALIDATIONS,
    _escalator,
    _job_dag,
    _ok,
    _run,
)
from tests.node_agent._support import SECRET, FakeRunner, node_action_executor
from tests.regional._late_ownership_runtime import stopped_runtime
from tests.regional.test_late_ownership_agent_boundary import LocalRegistry


@pytest.mark.parametrize("drift", ["owner", "late-sibling", "missing-validator"])
def test_real_ownership_refusal_cannot_trigger_either_escalation_ladder(drift):
    state, _kube, validator, context = stopped_runtime()
    if drift == "owner":
        state.job["metadata"]["uid"] = "replaced"
    elif drift == "late-sibling":
        state.pods.append(state.pod("node-b", "late"))
    else:
        validator = None
    with stop_ownership_scope(validator):
        outcome = node_submission_ownership_guard(context)
    assert outcome is not None and outcome.details["manual_confirmation_required"]
    assert_refusal_does_not_escalate(outcome)


def assert_refusal_does_not_escalate(outcome):
    store = build_store()
    _incident, workflow = _job_dag(store, node_c_operation=BUNDLE)
    outcomes = {RESET: outcome, **_ok(BUNDLE, REBOOT, RESTART_JOB, *VALIDATIONS)}
    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())
    assert result.status is WorkflowStatus.FAILED
    assert saved.branch_escalation_counts == {}
    assert not any(
        call.endswith(("/RESTART_NODE", "/REPLACE_NODE", "/RESTART_WORKLOAD"))
        for call in adapter.calls
    ), "ownership refusal must not create any stronger provider or restart rung"
    classified = HardwareEscalationService.classify(saved)
    assert classified is not None and classified[1] is RecoveryAction.ESCALATE_OPERATOR


@pytest.mark.parametrize("slow", ["compute", "device"])
def test_permit_expiring_during_final_client_recheck_cannot_spawn_or_escalate(
    tmp_path, slow
):
    _state, _kube, validator, context = stopped_runtime()
    context = replace(
        context, step=context.step.model_copy(update={"gpu_uuids": ["GPU-a"]})
    )
    clock = {"now": datetime.now(timezone.utc), "permitted": False}
    calls = []

    class Runner(FakeRunner):
        def __call__(self, command, **kwargs):
            if (
                clock["permitted"]
                and slow == "compute"
                and "--query-compute-apps=gpu_uuid,pid,process_name" in command
            ):
                clock["now"] += timedelta(seconds=4)
                calls.append("long-compute-read")
            return super().__call__(command, **kwargs)

    hardware = Runner()

    def devices(targets):
        if clock["permitted"] and slow == "device":
            clock["now"] += timedelta(seconds=4)
            calls.append("long-device-read")
        return []

    agent = node_action_executor(
        tmp_path,
        "expired-after-clients.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        runner=hardware,
        require_final_ownership=True,
        agent_generation=1,
        now=lambda: clock["now"],
        device_client_finder=devices,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        device_client_samples=1,
    )
    command = NodeActionCommand(
        command_id=context.idempotency_key + "/node-a/agent-1",
        workflow_request_id=context.workflow.request_id,
        incident_id=context.incident.incident_id,
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=["GPU-a"],
        ownership_guard=OWNERSHIP_PROTOCOL,
        issued_at=clock["now"],
        expires_at=clock["now"] + timedelta(seconds=90),
    )
    challenge = OwnershipChallenge(
        command_id=command.command_id,
        workflow_id=command.workflow_request_id,
        incident_id=command.incident_id,
        node_id=command.node_id,
        agent_generation=1,
        fencing_token=1,
        command_sha256=command_identity(command),
        boot_id=agent.ownership_gate.boot_id,
        nonce="a" * 64,
        sequence=1,
        boundary="AGENT_PRE_SPAWN",
        expires_at=command.expires_at,
    )
    permit = sign_permit(challenge, SECRET, allowed=True, reason="OK", now=clock["now"])

    def granted(value):
        assert value == command
        clock["permitted"] = True
        return permit

    agent.ownership_gate.require = granted
    result = agent.execute(
        SignedNodeAction(command=command, signature=sign_node_action(command, SECRET))
    )
    assert calls == [f"long-{slow}-read"]
    assert result.status is NodeActionStatus.FAILED and not result.retryable
    assert result.details["reason"] == "OWNERSHIP_PERMIT_EXPIRED"
    assert (
        result.details["manual_confirmation_required"]
        and result.details["node_action_not_started"]
    )
    assert not any("--gpu-reset" in command for command in hardware.commands), (
        "permit expiry during final recheck must prevent a GPU reset"
    )
    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=LocalRegistry(), sender=lambda *args: result
    )
    with stop_ownership_scope(validator):
        folded = adapter.execute(context)
    assert (
        folded.status is WorkflowStepStatus.FAILED
        and folded.details["manual_confirmation_required"]
    )
    assert_refusal_does_not_escalate(folded)


@pytest.mark.parametrize(
    "operation",
    [
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_WORKLOAD,
        WorkflowOperation.RESTART_VM,
    ],
)
@pytest.mark.parametrize("drift", ["owner", "late-sibling"])
def test_common_dispatch_blocks_non_agent_mutations_before_adapter_call(
    operation, drift
):
    state, _kube, validator, context = stopped_runtime()
    if drift == "owner":
        state.job["metadata"]["uid"] = "replaced"
    else:
        state.pods.append(state.pod("node-b", "late"))
    effects = []
    adapter = SimpleNamespace(
        execute=lambda current: effects.append(current)
        or WorkflowStepOutcome.succeeded()
    )
    dispatch = CommandDispatch(SimpleNamespace(stop_ownership_validator=validator))
    result = dispatch.execute_adapter(
        adapter,
        replace(context, step=context.step.model_copy(update={"operation": operation})),
    )
    assert result.status is WorkflowStepStatus.FAILED and effects == []
    assert (
        result.details["safety_rejection"]
        and result.details["manual_confirmation_required"]
    )
    if operation is WorkflowOperation.RESTART_WORKLOAD:
        assert result.details["restart_submitted"] is False


@pytest.mark.parametrize(
    "operation",
    [WorkflowOperation.RESTORE_SCHEDULING, WorkflowOperation.RESTORE_GPU_SERVICES],
)
def test_compensating_restore_is_not_blocked_by_a_lost_ownership_validator(operation):
    context = stopped_runtime()[-1]
    calls = []
    dispatch = CommandDispatch(SimpleNamespace(stop_ownership_validator=None))
    result = dispatch.execute_adapter(
        SimpleNamespace(
            execute=lambda current: calls.append(current)
            or WorkflowStepOutcome.succeeded()
        ),
        replace(context, step=context.step.model_copy(update={"operation": operation})),
    )
    assert result.status is WorkflowStepStatus.SUCCEEDED and len(calls) == 1


@pytest.mark.parametrize("drift", ["owner", "late-sibling"])
def test_batched_step_rechecks_containment_after_earlier_step_before_any_reset(
    tmp_path, monkeypatch, drift
):
    state, _kube, validator, context = stopped_runtime()
    sent, progress = [], []

    def send(endpoint, signed):
        assert signed.command.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
        sent.append(signed.command.operation)
        if drift == "owner":
            state.job["metadata"]["uid"] = "replaced-after-first-step"
        else:
            state.pods.append(state.pod("node-b", "late-after-first-step"))
        return NodeActionResult(
            command_id=signed.command.command_id,
            operation=signed.command.operation,
            status=NodeActionStatus.SUCCEEDED,
            details={"verified_no_gpu_clients": True},
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=LocalRegistry(), sender=send
    )
    steps = [
        context.workflow.official_steps[0],
        context.step.model_copy(
            update={"operation": WorkflowOperation.VERIFY_NO_GPU_CLIENTS}
        ),
        context.step,
        context.step.model_copy(
            update={"operation": WorkflowOperation.REMEDIATE_DRIVER}
        ),
    ]
    workflow = context.workflow.model_copy(update={"official_steps": steps})

    def key(index):
        return f"{workflow.request_id}/{index}/{steps[index].operation.value}"

    command = RemoteActionCommand(
        command_id="owned-batch",
        cluster_id=context.incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=context.incident.incident_id,
        step_index=1,
        fencing_token=workflow.fencing_token,
        idempotency_key=key(1),
        step=steps[1],
        workflow=workflow,
        incident=context.incident,
        lease_token="local-test-lease",
        lease_owner="owned-executor",
        status=RemoteCommandStatus.LEASED,
        batched_steps=[
            BatchedStep(step_index=index, step=steps[index], idempotency_key=key(index))
            for index in (2, 3)
        ],
    )
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        str(tmp_path / "owned-claim.json"),
    )
    client = SimpleNamespace(
        cluster_id=context.incident.cluster_id,
        progress=lambda *args: progress.append(args[-1]),
    )
    executor = ClusterActionExecutor(
        client, [adapter], executor_id="owned-executor", allowed_namespaces={"training"}
    )
    executor.stop_ownership_validator = validator
    result = executor.dispatch.execute(command)
    assert result.status is RemoteCommandStatus.FAILED, result
    assert result.details["manual_confirmation_required"]
    assert result.details["batched_step_index"] == 2
    assert result.details["reason"] == (
        "STOP_OWNERSHIP_DRIFT" if drift == "owner" else "STOP_PARTICIPANTS_CHANGED"
    )
    assert sent == [WorkflowOperation.VERIFY_NO_GPU_CLIENTS]
    assert len(progress) == 2
    assert "3" not in result.details[BATCHED_RESULTS_KEY]
