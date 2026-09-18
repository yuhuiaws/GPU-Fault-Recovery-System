"""The installed Agent checkpoint plus real kernel exec observation, on owned children."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import pytest

from gpu_fault.adapters.kubernetes.stop_ownership import (
    node_submission_ownership_guard,
    stop_ownership_scope,
)
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    OwnershipPermit,
    OwnershipRefused,
    ownership_recheck_scope,
    sign_permit,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    SignedNodeAction,
    sign_node_action,
)
from scripts.e2e.regional.late_ownership_barrier import process_identity
from scripts.e2e.regional.late_ownership_trace import (
    AttachedExecWitness,
    parse_exec_trace,
    physical_actions,
)
from tests.node_agent._support import (
    SECRET,
    FakeRunner,
    no_device_clients,
    node_action_executor,
)
from tests.regional._late_ownership_runtime import stopped_runtime

EXECUTABLE = Path("/usr/bin/true")
CALIBRATION = (str(EXECUTABLE), "--owned-physical-calibration")


def agent_child(peer, unused, directory):
    unused.close()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, os.getppid(), 0, 0, 0) != 0:
        peer.send({"error": "owned ptrace unavailable"})
        return

    class PhysicalRunner(FakeRunner):
        def __call__(self, command, **kwargs):
            if "--gpu-reset" in command:
                # The model deliberately reports zero resets. Only this harmless
                # child's real exec, observed independently, establishes action.
                return subprocess.run(
                    [str(EXECUTABLE), *command[1:]],
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=2,
                )
            return super().__call__(command, **kwargs)

    runner = PhysicalRunner()
    agent = node_action_executor(
        directory,
        "owned-agent.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        now=None,
        runner=runner,
        require_final_ownership=True,
        agent_generation=1,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        device_client_samples=1,
    )
    ready = Event()
    agent.ownership_gate.monotonic = lambda: ready.set() or time.monotonic()
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    peer.send({"ready": True})
    try:
        while True:
            request = peer.recv()
            action = request["action"]
            if action == "exit":
                return
            if action == "calibrate":
                completed = subprocess.run(
                    CALIBRATION, check=False, capture_output=True, timeout=2
                )
                peer.send({"calibrated": completed.returncode == 0})
            elif action == "queue":
                signed = SignedNodeAction.model_validate_json(request["envelope"])
                future = pool.submit(agent.execute, signed)
                assert ready.wait(5), (
                    "the native Agent did not park at its physical checkpoint"
                )
                challenge = agent.ownership_gate.challenge(signed.command.command_id)
                assert challenge is not None
                peer.send(
                    {
                        "challenge": challenge.model_dump_json(),
                        "future_done": future.done(),
                    }
                )
            elif action == "permit":
                permit = OwnershipPermit.model_validate_json(request["permit"])
                agent.ownership_gate.authorize(permit)
                result = future.result(5)
                peer.send(
                    {
                        "result": result.model_dump(mode="json"),
                        "model_reset_calls": sum(
                            "--gpu-reset" in item for item in runner.commands
                        ),
                    }
                )
            elif action == "late":
                try:
                    agent.ownership_gate.authorize(
                        OwnershipPermit.model_validate_json(request["permit"])
                    )
                except OwnershipRefused:
                    peer.send({"late_refused": True})
                else:
                    peer.send({"late_refused": False})
            else:
                raise AssertionError("unapproved local test command")
    except EOFError:
        pass
    finally:
        agent.ownership_gate.cancel_all()
        pool.shutdown(wait=True)
        peer.close()


@pytest.fixture
def owned_agent(tmp_path):
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    worker = context.Process(target=agent_child, args=(child, parent, tmp_path))
    worker.start()
    child.close()

    def request(action, **payload):
        parent.send({"action": action, **payload})
        assert parent.poll(10), "owned Agent child did not acknowledge the causal stage"
        return parent.recv()

    try:
        assert parent.poll(10) and parent.recv() == {"ready": True}
        yield worker, request
    finally:
        if worker.is_alive():
            parent.send({"action": "exit"})
        worker.join(10)
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        parent.close()
        assert not worker.is_alive(), "owned Agent child was not reaped"
        worker.close()


@pytest.mark.parametrize(
    "drift", ["unchanged", "wrong-uid", "replaced-owner", "late-sibling"]
)
def test_queued_guard_rejection_has_independent_physical_no_action_receipt(
    tmp_path, owned_agent, drift
):
    from gpu_fault.node_agent.late_ownership import OwnershipChallenge

    state, _kube, validator, context = stopped_runtime()
    child, request = owned_agent
    directory = tmp_path / "witness"
    directory.mkdir(mode=0o700)
    witness = AttachedExecWitness(
        directory,
        process_identity(child.pid),
        deadline=time.monotonic() + 40,
        executable=EXECUTABLE,
    )
    now = datetime.now(timezone.utc)
    command = NodeActionCommand(
        command_id=context.idempotency_key,
        workflow_request_id=context.workflow.request_id,
        incident_id=context.incident.incident_id,
        fencing_token=context.workflow.fencing_token,
        operation=WorkflowOperation.RESET_GPU,
        node_id="node-a",
        gpu_uuids=["GPU-a"],
        agent_generation=1,
        ownership_guard=OWNERSHIP_PROTOCOL,
        issued_at=now,
        expires_at=now + timedelta(seconds=90),
    )
    signed = SignedNodeAction(
        command=command, signature=sign_node_action(command, SECRET)
    )
    try:
        witness.start()
        assert request("calibrate") == {"calibrated": True}
        assert (
            physical_actions(
                parse_exec_trace(witness.snapshot()),
                nvidia_smi=EXECUTABLE,
                calibration_argv=CALIBRATION,
            )
            == ()
        )
        queued = request("queue", envelope=signed.model_dump_json())
        assert queued["future_done"] is False
        challenge = OwnershipChallenge.model_validate_json(queued["challenge"])
        assert challenge.boundary == "AGENT_PRE_SPAWN"
        assert challenge.command_id == command.command_id
        if drift == "wrong-uid":
            state.job["metadata"]["uid"] = "replaced-after-enqueue"
        elif drift == "replaced-owner":
            state.job["metadata"]["ownerReferences"] = [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": "replacement",
                    "uid": "replacement-owner",
                    "controller": False,
                }
            ]
        elif drift == "late-sibling":
            state.pods.append(state.pod("node-b", "late-after-enqueue"))
        with stop_ownership_scope(validator), ownership_recheck_scope(challenge):
            refused = node_submission_ownership_guard(context)
        assert (refused is None) is (drift == "unchanged")
        permit = sign_permit(
            challenge,
            SECRET,
            allowed=refused is None,
            reason="OK" if refused is None else refused.details["reason"],
            now=datetime.now(timezone.utc),
        )
        terminal = request("permit", permit=permit.model_dump_json())
        assert terminal["model_reset_calls"] == 0
        result = terminal["result"]
        assert result["status"] == ("SUCCEEDED" if drift == "unchanged" else "FAILED")
        if refused is not None:
            assert refused.status is WorkflowStepStatus.FAILED
            assert result["retryable"] is False
            assert result["details"]["manual_confirmation_required"] is True
            assert result["details"]["agent_queue_ownership_checked"] is True
        assert request("late", permit=permit.model_dump_json()) == {
            "late_refused": True
        }
        events = parse_exec_trace(witness.snapshot())
        actions = physical_actions(
            events, nvidia_smi=EXECUTABLE, calibration_argv=CALIBRATION
        )
        assert len(actions) == int(drift == "unchanged"), (
            "only a continuous kernel trace, not the deliberately false model counter, proves physical action"
        )
        if actions:
            assert actions[0].operation == "RESET_GPU" and actions[0].returncode == 0
        assert not list(directory.iterdir()), (
            "raw exec data must remain private and anonymous"
        )
    finally:
        witness.close()
    assert child.is_alive(), "tracer cleanup must not signal the Node Agent tracee"
