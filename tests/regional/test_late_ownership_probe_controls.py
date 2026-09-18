from __future__ import annotations

import io
import json
import time

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent.late_ownership import (
    OwnershipChallenge,
    ownership_recheck_scope,
)
from gpu_fault.node_agent.protocol import NodeActionExecutionState
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.late_ownership_contract import RecheckPermit
from scripts.e2e.regional.probes import late_ownership_executor_probe as probe
from tests.regional import test_late_ownership_executor_probe as support

owned_probe = support.owned_probe


def challenge(owned):
    return OwnershipChallenge(
        command_id="queued",
        workflow_id=owned.scope.workflow_id,
        incident_id=owned.scope.incident_id,
        node_id=owned.scope.nodes[0].name,
        boot_id=owned.scope.nodes[0].boot_id,
        agent_generation=1,
        fencing_token=1,
        command_sha256="a" * 64,
        nonce="b" * 64,
        sequence=1,
        expires_at=owned.scope.maintenance_end,
        boundary="AGENT_PRE_SPAWN",
    )


@pytest.mark.parametrize("interrupt", [False, True])
def test_waiting_steps_keep_actual_command_pointers_and_honor_controller_loss(
    owned_probe, monkeypatch, interrupt
):
    owned = owned_probe()
    replies = iter(
        [
            WorkflowStepOutcome.waiting(
                details={
                    "node_results": {
                        "node-a": {"node_action_command_id": "accepted"},
                        "node-b": {"node_action_command_id": None},
                    }
                }
            ),
            WorkflowStepOutcome.succeeded(),
        ]
    )

    def run(current):
        if interrupt:
            owned.queue("abort", {})
        return next(replies)

    monkeypatch.setattr(owned.node, "execute", run)
    if interrupt:
        with pytest.raises(BoundaryDenied, match="interrupted"):
            owned.probe.run_step(3)
        assert owned.probe.revoked, "controller loss must revoke the waiting probe"
    else:
        assert owned.probe.run_step(3).status is WorkflowStepStatus.SUCCEEDED
    assert owned.probe.node_commands == {"accepted": "node-a"}
    assert owned.probe.node_operations == {
        "accepted": WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    }


def test_expired_cleanup_deadline_does_not_run_even_a_compensating_command(owned_probe):
    owned = owned_probe()
    owned.probe.cleanup_deadline = time.monotonic() - 1
    with pytest.raises(BoundaryDenied, match="bounded window"):
        owned.probe.run_step(5, cleanup=True)
    assert owned.operations == []


@pytest.mark.parametrize(
    "defect", ["late", "changed-owner", "source-present", "unknown-pods"]
)
def test_native_callback_rejects_stale_session_and_unproven_source_absence(
    owned_probe, monkeypatch, defect
):
    owned = owned_probe()
    stopped = owned.probe.run_step(0)
    receipt = probe.StopOwnershipReceipt.model_validate_json(
        json.dumps(stopped.details[probe.STOP_RECEIPT_KEY])
    )
    owned.probe.starts = owned.starts
    if defect == "late":
        owned.probe.permit = RecheckPermit(
            scope_sha256=owned.scope.digest(),
            boundary_id="a" * 64,
            stop_sha256="b" * 64,
            mutation_sha256="c" * 64,
        )
    elif defect == "changed-owner":
        owned.state.job["metadata"]["uid"] = "replaced"
    elif defect == "source-present":
        owned.state.pods = [
            owned.state.pod("node-a", owned.scope.participants[0].pod_uid)
        ]
    else:
        monkeypatch.setattr(
            owned.state, "list_namespaced_pod", lambda *a: {"items": None}
        )
    with ownership_recheck_scope(challenge(owned)), pytest.raises(BoundaryDenied):
        owned.probe.at_physical_boundary(owned.probe.context(4), receipt)
    assert not any(message["kind"] == "stop" for message in owned.messages), (
        "a stale callback or unproven source absence must not emit STOP"
    )


@pytest.mark.parametrize(
    "defect", ["no-client", "absent", "placement", "phase", "owner", "permit"]
)
def test_observed_late_sibling_and_recheck_release_are_bound_to_actual_callback(
    owned_probe, defect
):
    owned = owned_probe("late-sibling")
    outcome = owned.probe.run_step(0)
    receipt = probe.StopOwnershipReceipt.model_validate_json(
        json.dumps(outcome.details[probe.STOP_RECEIPT_KEY])
    )
    owned.probe.starts = owned.starts

    def callback(kind, payload):
        if kind == "stop":
            pod = owned.state.pod("node-b", "late")
            if defect == "placement":
                pod["spec"]["nodeName"] = "node-a"
            if defect == "phase":
                pod["status"]["phase"] = "Pending"
            if defect == "owner":
                pod["metadata"]["ownerReferences"] = []
            owned.state.pods = [] if defect == "absent" else [pod]
            owned.queue(
                "observe-mutation",
                {
                    "sibling_uid": "late",
                    "physical_client_verified": defect != "no-client",
                },
            )
            return True
        if kind == "mutation" and defect == "permit":
            owned.queue(
                "recheck",
                RecheckPermit(
                    scope_sha256=owned.scope.digest(),
                    boundary_id="f" * 64,
                    stop_sha256=owned.probe.stopped.digest(),
                    mutation_sha256=owned.probe.mutation.digest(),
                ).model_dump(mode="json"),
            )
            return True
        return False

    owned.outgoing.callback = callback
    with ownership_recheck_scope(challenge(owned)), pytest.raises(BoundaryDenied):
        owned.probe.at_physical_boundary(owned.probe.context(4), receipt)
    assert owned.probe.permit is None


def test_action_drain_deadline_never_becomes_a_zero_pending_receipt(
    owned_probe, monkeypatch
):
    owned = owned_probe()
    current = owned.probe
    current.node_commands = {"pending": "node-a"}
    current.node_operations = {"pending": WorkflowOperation.RESET_GPU}
    current.cleanup_deadline = time.monotonic() - 1
    with pytest.raises(BoundaryDenied, match="terminal quiescence"):
        current.drain_actions()
    current.cleanup_deadline = time.monotonic() + 5
    terminal = owned.node.read_action_result("cluster-local", "node-a", "pending")
    pending = terminal.model_copy(
        update={"state": NodeActionExecutionState.PENDING, "result": None}
    )
    readings = iter([pending, terminal])
    monkeypatch.setattr(owned.node, "read_action_result", lambda *a: next(readings))
    current.drain_actions()


@pytest.mark.parametrize(
    "defect",
    ["calibration-interrupt", "calibration-timeout", "witnesses", "containment-ack"],
)
def test_probe_cannot_begin_the_physical_stage_without_all_acknowledgements(
    owned_probe, monkeypatch, defect
):
    owned = owned_probe()
    if defect.startswith("calibration"):

        def pending(current):
            if defect == "calibration-interrupt":
                owned.queue("abort", {})
            return WorkflowStepOutcome.waiting()

        monkeypatch.setattr(owned.node, "execute", pending)
    else:

        def callback(kind, payload):
            if defect == "witnesses" and kind == "calibrated":
                owned.queue("begin", {"witness_starts": []})
                return True
            if defect == "containment-ack" and kind == "contained":
                owned.queue("continue-boundary", {"unexpected": True})
                return True
            return False

        owned.outgoing.callback = callback
    with pytest.raises(BoundaryDenied):
        owned.probe.run()
    assert not any(message["kind"] == "stop" for message in owned.messages), (
        "missing acknowledgements must prevent physical-stage STOP"
    )


@pytest.mark.parametrize(
    "defect", ["early-step", "wrong-refusal", "services", "scheduling", "terminal"]
)
def test_failed_or_unconfirmed_compensation_never_emits_quiescence(
    owned_probe, monkeypatch, defect
):
    owned = owned_probe()
    original = owned.probe.run_step

    def run(index, *, cleanup=False):
        operation = owned.probe.workflow.official_steps[index].operation
        if (
            (defect == "early-step" and index == 0)
            or (
                defect == "services"
                and operation is WorkflowOperation.RESTORE_GPU_SERVICES
            )
            or (
                defect == "scheduling"
                and operation is WorkflowOperation.RESTORE_SCHEDULING
            )
        ):
            return WorkflowStepOutcome.failed("local control refusal")
        result = original(index, cleanup=cleanup)
        if defect == "wrong-refusal" and operation is WorkflowOperation.RESET_GPU:
            return WorkflowStepOutcome.failed(
                "unrelated failure", details={"reason": "OTHER"}
            )
        return result

    monkeypatch.setattr(owned.probe, "run_step", run)
    if defect == "terminal":

        def callback(kind, payload):
            if kind == "actions-drained":
                owned.queue("confirm-terminal", {"workflow_id": "another-owner"})
                return True
            return False

        owned.outgoing.callback = callback
    with pytest.raises(BoundaryDenied):
        owned.probe.run()
    assert not any(message["kind"] == "quiescence" for message in owned.messages), (
        "failed compensation must not emit a quiescence receipt"
    )


@pytest.mark.parametrize("defect", ["none", "cluster", "nodes", "duplicate", "adapter"])
def test_readonly_probe_entry_requires_exact_cluster_nodes_and_installed_adapter(
    owned_probe, monkeypatch, capsys, defect
):
    owned = owned_probe()
    request = {
        "inspect_only": True,
        "cluster_id": owned.scope.cluster_id,
        "nodes": ["node-a", "node-b"],
    }
    if defect == "cluster":
        request["cluster_id"] = "foreign"
    elif defect == "nodes":
        request["nodes"] = "unknown"
    elif defect == "duplicate":
        request["nodes"] = ["node-a", "node-a"]
    elif defect == "adapter":
        owned.probe.executor.adapters = []
    monkeypatch.setattr(
        probe, "executor_from_environment", lambda: owned.probe.executor
    )
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    assert probe.main() == int(defect != "none")
    response = json.loads(capsys.readouterr().out)
    if defect == "none":
        assert response == {
            "protocol_ready": True,
            "cluster_id": owned.scope.cluster_id,
            "nodes": ["node-a", "node-b"],
        }
    else:
        assert response == {"error_kind": "BoundaryDenied"}
    assert owned.operations == []


def test_probe_entry_assembles_scoped_models_without_running_any_live_io(
    owned_probe, monkeypatch, capsys
):
    owned = owned_probe()
    request = {
        "scope": owned.scope.model_dump(mode="json"),
        "workflow": owned.probe.workflow.model_dump(mode="json"),
        "incident": owned.probe.incident.model_dump(mode="json"),
    }
    calls = []
    monkeypatch.setattr(
        probe, "executor_from_environment", lambda: owned.probe.executor
    )
    monkeypatch.setattr(
        probe.ExecutorProbe, "run", lambda self: calls.append(self.scope)
    )
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    assert probe.main() == 0 and capsys.readouterr().out == ""
    assert calls == [owned.scope] and owned.operations == []
