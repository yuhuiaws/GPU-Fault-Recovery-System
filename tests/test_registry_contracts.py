from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault.app.ingest import telemetry as telemetry_module
from gpu_fault.app.ingest.telemetry import TelemetryIngestionService
from gpu_fault.channel_registry import (
    BATCHABLE_CHANNEL_PATHS,
    CHANNEL_REGISTRY,
    COLLECTOR_EVENT_PREFIX,
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    ChannelLane,
    ChannelPool,
    ChannelPriorityMode,
    validate_channel_registry,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.operations.registry import (
    OPERATION_HANDLERS,
    validate_operation_handlers,
)
from gpu_fault.operation_registry import (
    CONTAINMENT_ONLY_OPERATIONS,
    DESTRUCTIVE_OPERATIONS,
    GENERATION_STABLE_COMMAND_OPERATIONS,
    HARDWARE_ESCALATION_RELEVANT_OPERATIONS,
    HOST_PROC_ROOT_OPERATIONS,
    MAINTENANCE_GENERATION_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
    OPERATION_REGISTRY,
    OperationAdapter,
    OperationResourceClaim,
    OperationScope,
    validate_operation_registry,
)
from gpu_fault.processor.batching import TELEMETRY_BATCH_SIZE_BY_PATH
from gpu_fault.store.postgres import processor_claims

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/gpu_fault"

# Hand-written membership. Comparing a derived export against the same
# comprehension that produced it proves nothing; these lists are the
# independent statement of intent, so flipping a flag on a registry row
# fails here and forces the change to be argued for.
EXPECTED_CONTAINMENT_ONLY = {
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUARANTINE,
    WorkflowOperation.RESTORE_SCHEDULING,
}
EXPECTED_HOST_PROC_ROOT = {
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
}
EXPECTED_MAINTENANCE_GENERATION = {
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    WorkflowOperation.RESTORE_GPU_SERVICES,
}
# Long-running mutating node actions whose command_id must survive an agent
# generation change: the restarted agent answers from its ledger for the id
# it already holds, where a generation-suffixed id would have been a brand
# new command and a second driver/firmware install (R4, 2026-09-08 review).
EXPECTED_GENERATION_STABLE_COMMAND = {
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
}
# A set literal this large is unreadable at a call site and drifts the
# moment an operation is added, so re-listing one is what the duplication
# guard below rejects. Two- and three-member families stay inline.
INLINE_SET_MEMBER_LIMIT = 3


def test_batchable_telemetry_paths_have_a_batch_size() -> None:
    expected = {
        path
        for path in BATCHABLE_CHANNEL_PATHS
        if CHANNEL_REGISTRY[path].pool in {ChannelPool.GPU, ChannelPool.HOST}
    }

    assert set(TELEMETRY_BATCH_SIZE_BY_PATH) == expected
    assert all(size > 0 for size in TELEMETRY_BATCH_SIZE_BY_PATH.values())


def _workflow_operation_set_literals(
    path: Path,
) -> list[tuple[int, frozenset[WorkflowOperation]]]:
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Set):
            continue
        members = [
            getattr(WorkflowOperation, element.attr, None)
            for element in node.elts
            if isinstance(element, ast.Attribute)
            and isinstance(element.value, ast.Name)
            and element.value.id == "WorkflowOperation"
        ]
        if len(members) != len(node.elts) or None in members:
            continue
        found.append((node.lineno, frozenset(members)))
    return found


def _sql_strings(module) -> list[str]:
    source = Path(module.__file__).read_text()
    strings = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal_text = node.value
        elif isinstance(node, ast.JoinedStr):
            literal_text = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
        else:
            continue
        if "gpu_fault_processor_queue" not in literal_text:
            continue
        strings.append(
            eval(  # noqa: S307 - module's own f-strings, no input
                compile(ast.Expression(node), "<sql>", "eval"), vars(module)
            )
        )
    return strings


def test_containment_and_node_mutating_sets_are_disjoint_and_complete() -> None:
    assert CONTAINMENT_ONLY_OPERATIONS == EXPECTED_CONTAINMENT_ONLY
    assert NODE_MUTATING_OPERATIONS == (
        DESTRUCTIVE_OPERATIONS - EXPECTED_CONTAINMENT_ONLY
    )
    assert not (NODE_MUTATING_OPERATIONS & CONTAINMENT_ONLY_OPERATIONS)
    assert NODE_MUTATING_OPERATIONS <= DESTRUCTIVE_OPERATIONS
    for operation in CONTAINMENT_ONLY_OPERATIONS:
        semantics = OPERATION_REGISTRY[operation]
        # Containment earns its exemption by touching nothing but the
        # scheduler. An operation that also claims the GPU, the driver or
        # the node must fall on the node-mutating side of the gate.
        assert semantics.resource_claims == frozenset(
            {OperationResourceClaim.SCHEDULER_MUTATION}
        )
        assert semantics.scope is OperationScope.NODE
    for operation, semantics in OPERATION_REGISTRY.items():
        if not semantics.destructive:
            continue
        if semantics.resource_claims - {OperationResourceClaim.SCHEDULER_MUTATION}:
            assert operation in NODE_MUTATING_OPERATIONS


def test_registry_semantic_sets_match_declared_membership() -> None:
    assert HOST_PROC_ROOT_OPERATIONS == EXPECTED_HOST_PROC_ROOT
    assert MAINTENANCE_GENERATION_OPERATIONS == EXPECTED_MAINTENANCE_GENERATION
    assert GENERATION_STABLE_COMMAND_OPERATIONS == EXPECTED_GENERATION_STABLE_COMMAND
    # Escalating to hardware support is only defensible for operations the
    # fleet has actually attempted on the device.
    assert HARDWARE_ESCALATION_RELEVANT_OPERATIONS
    assert not (HARDWARE_ESCALATION_RELEVANT_OPERATIONS & CONTAINMENT_ONLY_OPERATIONS)


def test_generation_stable_commands_are_unpinned_mutating_node_actions() -> None:
    # A maintenance-generation-scoped action already pins the generation the
    # quiesce captured, so its command_id never moves; the stable-command
    # flag is for the mutating node actions that read the generation live.
    # A non-mutating action must keep the suffix: re-collecting on the fresh
    # agent is the useful answer there, not an INTERRUPTED replay.
    assert not (
        GENERATION_STABLE_COMMAND_OPERATIONS & MAINTENANCE_GENERATION_OPERATIONS
    )
    for operation in GENERATION_STABLE_COMMAND_OPERATIONS:
        semantics = OPERATION_REGISTRY[operation]
        assert OperationAdapter.NODE_ACTION in semantics.adapters, operation
        assert operation in NODE_MUTATING_OPERATIONS, operation
    # Every mutating node action is covered one way or the other: pinned by
    # the quiesce or named without the generation. QUIESCE itself produces
    # the pin and re-quiescing is idempotent; RESTART_FABRIC_MANAGER is the
    # DESTR-019 subject whose suffix the case's verdicts read.
    covered = GENERATION_STABLE_COMMAND_OPERATIONS | MAINTENANCE_GENERATION_OPERATIONS
    uncovered = {
        operation
        for operation, semantics in OPERATION_REGISTRY.items()
        if OperationAdapter.NODE_ACTION in semantics.adapters
        and operation in NODE_MUTATING_OPERATIONS
        and operation not in covered
    }
    assert uncovered == {
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
    }


def test_large_registry_sets_are_not_relisted_as_literals() -> None:
    derived = {
        "DESTRUCTIVE_OPERATIONS": DESTRUCTIVE_OPERATIONS,
        "NODE_MUTATING_OPERATIONS": NODE_MUTATING_OPERATIONS,
        "HOST_PROC_ROOT_OPERATIONS": HOST_PROC_ROOT_OPERATIONS,
        "MAINTENANCE_GENERATION_OPERATIONS": (MAINTENANCE_GENERATION_OPERATIONS),
        "GENERATION_STABLE_COMMAND_OPERATIONS": (GENERATION_STABLE_COMMAND_OPERATIONS),
        "HARDWARE_ESCALATION_RELEVANT_OPERATIONS": (
            HARDWARE_ESCALATION_RELEVANT_OPERATIONS
        ),
    }
    duplicated = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "operation_registry.py":
            continue
        for lineno, members in _workflow_operation_set_literals(path):
            if len(members) <= INLINE_SET_MEMBER_LIMIT:
                continue
            for name, expected in derived.items():
                if members == expected:
                    duplicated.append(
                        f"{path.relative_to(ROOT)}:{lineno} re-lists {name}"
                    )
    assert not duplicated


@pytest.mark.parametrize(
    ("operation", "updates", "message"),
    [
        (
            WorkflowOperation.REMEDIATE_DRIVER,
            {"maintenance_generation_scoped": True},
            "already pins its command_id",
        ),
        (
            WorkflowOperation.REMEDIATE_DRIVER,
            {"adapters": frozenset({OperationAdapter.KUBERNETES})},
            "generation-stable command_id but is not a node action",
        ),
        # A stable id exists to keep a mutation from running twice; a
        # non-mutating action wants the suffix (re-run on the fresh agent).
        (
            WorkflowOperation.REMEDIATE_DRIVER,
            {"destructive": False},
            "generation-stable command_id but is not node-mutating",
        ),
        # Containment is destructive in the audit sense but never a node
        # mutation, so a scheduler-only claim set is refused the same way.
        (
            WorkflowOperation.REMEDIATE_DRIVER,
            {"resource_claims": frozenset({OperationResourceClaim.SCHEDULER_MUTATION})},
            "generation-stable command_id but is not node-mutating",
        ),
    ],
)
def test_generation_stable_command_invariants_fail_closed(
    monkeypatch, operation, updates, message
) -> None:
    monkeypatch.setitem(
        OPERATION_REGISTRY, operation, replace(OPERATION_REGISTRY[operation], **updates)
    )

    with pytest.raises(RuntimeError, match=message):
        validate_operation_registry()


def test_node_action_handlers_match_adapter_and_executor() -> None:
    validate_operation_handlers(NodeActionExecutor)
    assert all(
        callable(getattr(NodeActionExecutor, name, None))
        for name in OPERATION_HANDLERS.values()
    )


def test_spoolable_collector_channels_have_batch_handlers() -> None:
    service = TelemetryIngestionService(object(), object(), object(), object())
    expected = {
        path
        for path, channel in CHANNEL_REGISTRY.items()
        if path.startswith(COLLECTOR_EVENT_PREFIX) and channel.spoolable
    }
    # Bidirectional: a spoolable channel with no batch path would 422 on
    # every drain, and a batch path that is not spoolable would accept
    # items the collector is never allowed to hold on disk.
    assert service.telemetry_batch_paths == expected
    assert set(service.telemetry_batch_handlers) <= expected


@pytest.mark.parametrize(
    ("path", "updates", "message"),
    [
        (
            GPU_METRICS_PATH,
            {"path": "/not-a-v1-channel"},
            "invalid processor channel path",
        ),
        (
            GPU_METRICS_PATH,
            {"edge_filtered": False},
            "edge-filtered mode and flag disagree",
        ),
        (
            GPU_METRICS_PATH,
            {"routine_reasons": frozenset()},
            "edge-filtered channel has no routine vocabulary",
        ),
        (
            GPU_INVENTORY_PATH,
            {"batchable": False},
            "spoolable channel must be batchable",
        ),
        (
            GPU_INVENTORY_PATH,
            {
                "summary_lane_suffix": "invalid-summary",
                "lane": ChannelLane.GPU_INVENTORY,
            },
            "summary suffix requires EDGE_SUMMARY lane",
        ),
        (GPU_INVENTORY_PATH, {"spool_weight": -1}, "spool weight cannot be negative"),
    ],
)
def test_channel_registry_invariants_fail_closed(
    monkeypatch, path, updates, message
) -> None:
    monkeypatch.setitem(
        CHANNEL_REGISTRY, path, replace(CHANNEL_REGISTRY[path], **updates)
    )

    with pytest.raises(RuntimeError, match=message):
        validate_channel_registry()


def test_unhandled_spoolable_channel_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telemetry_module,
        "SPOOLABLE_CHANNEL_PATHS",
        frozenset(
            {
                *telemetry_module.SPOOLABLE_CHANNEL_PATHS,
                "/v1/collector-events/invented-channel",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="invented-channel"):
        TelemetryIngestionService(object(), object(), object(), object())


def test_postgres_fault_claim_paths_match_channel_registry() -> None:
    expected = {
        path
        for path, channel in CHANNEL_REGISTRY.items()
        if path.startswith(COLLECTOR_EVENT_PREFIX)
        and channel.priority_mode is ChannelPriorityMode.FAULT
    }
    assert expected <= set(processor_claims._FAULT_CLAIM_PATHS)

    # No store module may name a collector fault path itself: the claim
    # SQL has to grow a new FAULT channel automatically, or lane fairness
    # silently stops treating it as a fault.
    hardcoded = []
    for path in sorted((SRC / "store").rglob("*.py")):
        for match in re.findall(
            r"'(/v1/collector-events/[a-z0-9-]+)'", path.read_text()
        ):
            hardcoded.append(f"{path.relative_to(ROOT)}: {match}")
    assert not hardcoded

    # Every rendered claim query that filters on fault paths must list
    # all of them and nothing that is not a registered FAULT channel.
    checked = 0
    for sql in _sql_strings(processor_claims):
        found = set(re.findall(r"'(/v1/collector-events/[a-z0-9-]+)'", sql))
        if not found:
            continue
        checked += 1
        assert found == expected
    assert checked >= 2
