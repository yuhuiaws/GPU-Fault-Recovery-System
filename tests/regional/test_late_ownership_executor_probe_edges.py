from __future__ import annotations

import io
import json
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import late_ownership_executor_probe as probe
from tests.regional import test_late_ownership_executor_probe as support

owned_probe = support.owned_probe


def step_index(owned: Any, operation: WorkflowOperation) -> int:
    return next(
        index
        for index, step in enumerate(owned.probe.workflow.official_steps)
        if step.operation is operation
    )


def test_negative_case_skips_second_node_even_when_its_step_is_ordered_first(
    owned_probe: Any,
) -> None:
    owned = owned_probe("ownership-drift")
    original = owned.store.get_workflow(owned.scope.workflow_id)
    steps = list(original.official_steps)
    second_node_index = next(
        index
        for index, step in enumerate(steps)
        if step.operation is WorkflowOperation.MARK_UNSCHEDULABLE
        and step.node_ids == [owned.scope.nodes[1].name]
    )
    second_node_step = steps.pop(second_node_index)
    steps.insert(1, second_node_step)
    revised = original.model_copy(update={"official_steps": steps})
    owned.store.save_workflow(revised, expected=original)
    owned.probe.workflow = owned.store.get_workflow(owned.scope.workflow_id)
    owned.probe.run()
    assert 1 not in owned.probe.workflow.completed_step_indexes, (
        "the early second-node step must not be falsely marked completed"
    )
    assert not any(
        execution.step_index == 1 for execution in owned.probe.workflow.step_executions
    ), "the skipped step must not be executed or recorded as attempted"
    assert (
        WorkflowOperation.MARK_UNSCHEDULABLE,
        [owned.scope.nodes[1].name],
    ) not in owned.operations, "the negative scenario may not mutate the second node"
    kinds = [message["kind"] for message in owned.messages]
    assert kinds[-1] == "finished" and owned.probe.revoked, (
        "skipping an out-of-scenario step must preserve the complete cleanup protocol"
    )
    assert kinds.count("holder-check") > 1, (
        "execution must retain real CPU holder checks"
    )
    assert owned.replies == [], "all causal messages must be consumed"


def test_boot_drift_blocks_node_reads_and_patches_before_mutation(
    owned_probe: Any,
) -> None:
    owned = owned_probe()
    node = owned.scope.nodes[0].name
    owned.state.nodes[node]["status"]["nodeInfo"]["bootID"] = "replacement-boot"
    with pytest.raises(BoundaryDenied, match="UID or boot"):
        owned.probe.kube.core.read_node(node)
    with pytest.raises(BoundaryDenied, match="UID or boot"):
        owned.probe.kube.core.patch_node(node, {"spec": {"unschedulable": True}})
    assert owned.operations == [], "a different boot cannot authorize product execution"
    assert all(call[0] == "get-node" for call in owned.state.calls), (
        "boot refusal must precede every Node mutation"
    )


def test_node_patch_preserves_callers_resource_version_and_original_body(
    owned_probe: Any,
) -> None:
    owned = owned_probe()
    node = owned.scope.nodes[0]
    body = {"metadata": {"resourceVersion": "earlier"}, "spec": {"unschedulable": True}}
    original = deepcopy(body)
    patches: list[dict[str, Any]] = []

    def conflict(name: str, patch: dict[str, Any], **kwargs: Any) -> None:
        assert name == node.name, "the patch must target the scoped Node"
        patches.append(deepcopy(patch))
        raise RuntimeError("local fake: resourceVersion conflict")

    owned.state.patch_node = conflict
    with pytest.raises(RuntimeError, match="resourceVersion conflict"):
        owned.probe.kube.core.patch_node(node.name, body)
    assert body == original, "the UID wrapper must not mutate the caller's patch"
    assert patches == [
        {
            "metadata": {"uid": node.uid, "resourceVersion": "earlier"},
            "spec": {"unschedulable": True},
        }
    ], "the wrapper must not replace an earlier CAS precondition with a fresh one"


def test_drain_requires_inner_result_command_identity(
    owned_probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = owned_probe()
    current = owned.probe
    node = owned.scope.nodes[0].name
    current.node_commands = {"accepted": node}
    current.node_operations = {"accepted": WorkflowOperation.RESET_GPU}
    current.cleanup_deadline = time.monotonic() + 5
    terminal = owned.node.read_action_result(owned.scope.cluster_id, node, "accepted")
    wrong = terminal.model_copy(
        update={
            "result": terminal.result.model_copy(
                update={"command_id": "another-command"}
            )
        }
    )
    monkeypatch.setattr(owned.node, "read_action_result", lambda *args: wrong)
    with pytest.raises(BoundaryDenied, match="terminal receipt is incomplete"):
        current.drain_actions()
    assert current.node_commands == {"accepted": node}, (
        "drainage may not erase unresolved identity"
    )
    assert not any(message["kind"] == "quiescence" for message in owned.messages), (
        "a matching envelope cannot stand in for the actual command result"
    )


def test_drain_boot_change_refuses_before_polling_a_replacement_agent(
    owned_probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = owned_probe()
    node = owned.scope.nodes[0].name
    owned.probe.node_commands = {"accepted": node}
    owned.probe.node_operations = {"accepted": WorkflowOperation.RESET_GPU}
    owned.probe.cleanup_deadline = time.monotonic() + 5
    owned.state.nodes[node]["status"]["nodeInfo"]["bootID"] = "new-boot"
    reads: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        owned.node, "read_action_result", lambda *args: reads.append(args)
    )
    with pytest.raises(BoundaryDenied, match="UID or boot"):
        owned.probe.drain_actions()
    assert not reads, (
        "a replaced Agent cannot resolve the original command's physical state"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"holder_valid": False},
        {"holder_valid": 1},
        {"lifetime_deadline_at": None},
        {"lifetime_deadline_at": "2000-01-01T00:00:00Z"},
    ],
)
def test_invalid_holder_receipt_blocks_stop_before_workload_mutation(
    owned_probe: Any, change: dict[str, Any]
) -> None:
    owned = owned_probe()

    def callback(kind: str, payload: dict[str, Any]) -> bool:
        if kind != "holder-check":
            return False
        owned.queue(
            "holder-check-result",
            {
                **payload,
                "holder_valid": True,
                "lifetime_deadline_at": owned.scope.maintenance_end.isoformat(),
                **change,
            },
        )
        return True

    owned.outgoing.callback = callback
    with pytest.raises(BoundaryDenied, match="lease was lost"):
        owned.probe.run_step(0)
    assert owned.probe.revoked and not owned.operations, (
        "invalid CPU evidence must revoke before calling any product adapter"
    )
    assert owned.state.job["spec"]["runPolicy"]["suspend"] is False, (
        "a malformed holder proof cannot stop the workload"
    )


def test_adapter_exception_restores_the_outer_lease_guard(
    owned_probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = owned_probe()
    seen: list[Any] = []

    def execute(context: Any) -> WorkflowStepOutcome:
        seen.append(active_lease_guard.get())
        raise TimeoutError("local fake: adapter outcome unknown")

    def outer_guard() -> str:
        return "outer-owner"

    monkeypatch.setattr(owned.node, "execute", execute)
    token = active_lease_guard.set(outer_guard)
    try:
        with pytest.raises(TimeoutError):
            owned.probe.run_step(
                step_index(owned, WorkflowOperation.VERIFY_NO_GPU_CLIENTS)
            )
        assert active_lease_guard.get() is outer_guard, (
            "an exception cannot leak the probe's lease guard into another workflow"
        )
    finally:
        active_lease_guard.reset(token)
    assert seen == [owned.probe.lease_reason], (
        "the adapter must run under the real lease guard"
    )
    assert owned.probe.outcomes == [], (
        "an unknown adapter outcome is not a completed step"
    )


def test_cleanup_cannot_authorize_an_additional_reset(owned_probe: Any) -> None:
    owned = owned_probe()
    owned.probe.cleanup_deadline = time.monotonic() + 30
    with pytest.raises(BoundaryDenied, match="new hardware action"):
        owned.probe.run_step(
            step_index(owned, WorkflowOperation.RESET_GPU), cleanup=True
        )
    assert owned.operations == [] and owned.messages == [], (
        "a cleanup window cannot authorize or request a fresh physical command"
    )


@pytest.mark.parametrize("raw", ["[]\n", "null\n", '"expected"\n'])
def test_incoming_non_object_message_irreversibly_revokes(
    owned_probe: Any, raw: str
) -> None:
    owned = owned_probe()
    owned.replies.append(raw)
    with pytest.raises(BoundaryDenied, match="stale or out of order"):
        owned.probe.receive("expected")
    assert owned.probe.revoked, "non-object control input must revoke authority"
    owned.queue("expected", {})
    with pytest.raises(BoundaryDenied, match="lease was lost"):
        owned.probe.run_step(0)
    assert not owned.operations, (
        "a later valid frame must not restore revoked authority"
    )


@pytest.mark.parametrize("failed_node", [0, 1])
def test_failed_calibration_cannot_announce_ready_or_start_containment(
    owned_probe: Any, monkeypatch: pytest.MonkeyPatch, failed_node: int
) -> None:
    owned = owned_probe()
    execute = owned.node.execute
    calibrations: list[str] = []

    def fail_calibration(context: Any) -> WorkflowStepOutcome:
        if "/calibration/" in context.idempotency_key:
            calibrations.append(context.idempotency_key)
            if context.step.node_ids == [owned.scope.nodes[failed_node].name]:
                return WorkflowStepOutcome.failed(
                    "local fake: calibration was not executed"
                )
        outcome: WorkflowStepOutcome = execute(context)
        return outcome

    monkeypatch.setattr(owned.node, "execute", fail_calibration)
    with pytest.raises(BoundaryDenied):
        owned.probe.run()
    assert calibrations == [
        f"{owned.scope.workflow_id}/calibration/{node.name}"
        for node in owned.scope.nodes[: failed_node + 1]
    ], "calibration must stop immediately on either node's failure"
    assert not any(
        message["kind"] in {"calibrated", "contained", "stop", "decision", "quiescence"}
        for message in owned.messages
    ), "failed calibration must not be promoted into a completed causal stage"
    assert owned.state.job["spec"]["runPolicy"]["suspend"] is False, (
        "failed calibration must precede any workload mutation"
    )


def test_incident_cluster_must_match_before_installing_the_probe(
    owned_probe: Any,
) -> None:
    owned = owned_probe()
    wrong_incident = owned.probe.incident.model_copy(
        update={"cluster_id": "another-cluster"}
    )
    original_core = owned.probe.kube.core
    original_validator_core = owned.validator.core
    original_callback = owned.validator.before_recheck
    with pytest.raises(BoundaryDenied, match="scope"):
        probe.ExecutorProbe(
            owned.scope,
            owned.probe.workflow,
            wrong_incident,
            owned.probe.executor,
            incoming=owned.probe.incoming,
            outgoing=owned.probe.outgoing,
        )
    assert owned.operations == [] and owned.messages == [], (
        "a foreign incident must not receive the current cluster's execution authority"
    )
    assert owned.probe.kube.core is original_core, (
        "refusal must not replace the installed core"
    )
    assert owned.validator.core is original_validator_core, (
        "refusal must not rebind validation"
    )
    assert owned.validator.before_recheck == original_callback, (
        "refusal must preserve the existing native callback"
    )


@pytest.mark.parametrize("encoding", ["ascii", "utf8"])
def test_entry_byte_cap_precedes_json_decode_and_executor_loading(
    owned_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    encoding: str,
) -> None:
    owned = owned_probe()
    request = {
        "inspect_only": True,
        "cluster_id": owned.scope.cluster_id,
        "nodes": [node.name for node in owned.scope.nodes],
        "padding": "",
    }
    size = len(json.dumps(request, ensure_ascii=False).encode())
    remaining = probe.MAX_MESSAGE_BYTES + 1 - size
    character = "x" if encoding == "ascii" else "\u00e9"
    count, tail = divmod(remaining, len(character.encode()))
    request["padding"] = character * count + "x" * tail
    raw = json.dumps(request, ensure_ascii=False)
    assert len(raw.encode()) == probe.MAX_MESSAGE_BYTES + 1, (
        "the input control must exceed the byte bound by exactly one byte"
    )
    calls: list[str] = []

    def decode(value: str) -> Any:
        calls.append("decode")
        return json.loads(value)

    def load() -> Any:
        calls.append("load")
        return owned.probe.executor

    monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=io.StringIO(raw)))
    monkeypatch.setattr(probe, "json", SimpleNamespace(loads=decode, dumps=json.dumps))
    monkeypatch.setattr(probe, "executor_from_environment", load)
    assert probe.main() == 1, (
        "oversized input must fail closed before environment access"
    )
    assert json.loads(capsys.readouterr().out) == {"error_kind": "BoundaryDenied"}, (
        "oversized input must produce only a sanitized refusal"
    )
    assert not calls, "byte validation must precede decoding and runtime construction"
    assert owned.operations == [], "an oversized request cannot run product operations"


def test_calibration_waits_for_both_successes_without_changing_the_scope_deadline(
    owned_probe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = owned_probe()
    execute = owned.node.execute
    counts: dict[str, int] = {}
    deadline = owned.scope.maintenance_end

    def waiting_then_success(context: Any) -> WorkflowStepOutcome:
        if "/calibration/" in context.idempotency_key:
            node = context.step.node_ids[0]
            counts[node] = counts.get(node, 0) + 1
            if counts[node] == 1:
                return WorkflowStepOutcome.waiting()
        outcome: WorkflowStepOutcome = execute(context)
        return outcome

    monkeypatch.setattr(owned.node, "execute", waiting_then_success)
    owned.probe.run()
    assert counts == {node.name: 2 for node in owned.scope.nodes}, (
        "both nodes must complete their existing bounded calibration polls"
    )
    assert owned.probe.scope.maintenance_end == deadline, (
        "polling cannot extend authority"
    )
    kinds = [message["kind"] for message in owned.messages]
    assert kinds.count("calibrated") == 1 and kinds[-1] == "finished", (
        "successful calibration must preserve the existing proof sequence"
    )
    assert not owned.replies, (
        "the full causal protocol must finish without orphaned messages"
    )


@pytest.mark.parametrize("prefetched", [False, True])
def test_exact_entry_limit_preserves_buffered_source_loader_compatibility(
    owned_probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prefetched: bool,
) -> None:
    owned = owned_probe()
    request = {
        "inspect_only": True,
        "cluster_id": owned.scope.cluster_id,
        "nodes": [node.name for node in owned.scope.nodes],
    }
    body = json.dumps(request)
    line = " " * (probe.MAX_MESSAGE_BYTES - len(body.encode()) - 1) + body + "\n"
    prefix = "local-pinned-source-prefix\n" if prefetched else ""
    stream = io.TextIOWrapper(io.BytesIO((prefix + line).encode()), encoding="utf-8")
    calls: list[str] = []

    def load() -> Any:
        calls.append("load")
        return owned.probe.executor

    with stream:
        if prefetched:
            assert stream.read(len(prefix)) == prefix, (
                "the source loader consumes only its prefix"
            )
        monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=stream))
        monkeypatch.setattr(probe, "executor_from_environment", load)
        assert probe.main() == 0, (
            "a valid maximum-size line must remain readable after source loading"
        )
    assert calls == ["load"] and owned.operations == [], (
        "inspection must construct once and perform no product operation"
    )
    assert json.loads(capsys.readouterr().out) == {
        "protocol_ready": True,
        "cluster_id": owned.scope.cluster_id,
        "nodes": [node.name for node in owned.scope.nodes],
    }, "the existing read-only response ABI must be unchanged"
