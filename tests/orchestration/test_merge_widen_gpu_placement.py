"""A merge widens a shared node action onto the GPUs it can place (batch 155).

``RecoveryArbiter.merged_gpu_scope`` gave every node of a multi-node
node-action step without ``gpu_uuids_by_node`` the step's whole GPU list, and
the orchestrator's widening wrote that scope into the step's mapping. The
multi-node reset barrier hands each node's agent exactly its entry, so the
widened reset sent node-b's GPU to node-a's agent -- the mistake batch 149
fixed in ``disposition.widen_in_place``. Such a step's GPUs are now placed with
the workflow's own per-node lists (``arbitration.step_gpu_shares``). When one
of them stays unplaced, none is attributed to a node and the merge leaves the
step without a mapping (``arbitration.gpu_shares_unknown``). Written as a
partial mapping instead, the next merges -- ``widen_in_place``, the widening
itself, ``merge_sxid_step_scope`` -- completed it without the unplaced GPUs.
The barrier refuses the unmapped multi-node reset before any command is sent.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from gpu_fault.operation_registry import WORKLOAD_SCOPED_OPERATIONS
from gpu_fault.orchestration import arbitration
from gpu_fault.orchestration.disposition import widen_in_place
from gpu_fault.orchestration.ingest.sxid import merge_sxid_step_scope
from gpu_fault.policy import SxidClassification, SxidEvent
from tests._builders import (
    build_context,
    build_sxid_event,
    copy_model,
    workflow_request,
    workflow_step,
)
from tests.orchestration._support import _node_event, ingest
from tests.orchestration.test_scope_fidelity import _widen
from tests.regional._cov95_runtime_adapter import context_for, fleet_adapter

RESET = WorkflowOperation.RESET_GPU
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
VALIDATE = WorkflowOperation.VALIDATE_GPU
VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS
PAIR = ["node-a", "node-b"]
#: The merged fault's GPU, on node-a.
NEW_GPU = "GPU-a2"
PER_NODE = {"node-a": ["GPU-a", NEW_GPU], "node-b": ["GPU-b"]}
#: What the barrier answers for a multi-node reset without a mapping.
NO_MAPPING = "multi-node GPU reset requires parameters.gpu_uuids_by_node"
#: Later faults: node-a, node-b, node-a again.
MERGES = (("node-a", NEW_GPU), ("node-b", "GPU-b2"), ("node-a", "GPU-a3"))
NOW = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)


def _shared(
    operation: WorkflowOperation = RESET,
    gpu_uuids: tuple[str, ...] = ("GPU-a", "GPU-b"),
    *,
    mapping: dict[str, list[str]] | None = None,
) -> WorkflowStepSpec:
    """A node action over both nodes. Without ``mapping`` its flat GPU list
    does not say which node holds which GPU."""

    return workflow_step(
        operation,
        node_ids=list(PAIR),
        gpu_uuids=list(gpu_uuids),
        parameters={} if mapping is None else {"gpu_uuids_by_node": mapping},
    )


def _one(
    operation: WorkflowOperation, node_id: str, *gpu_uuids: str
) -> WorkflowStepSpec:
    return workflow_step(operation, node_ids=[node_id], gpu_uuids=list(gpu_uuids))


def _workflow(*steps: WorkflowStepSpec) -> WorkflowRequest:
    return workflow_request(
        "wf-b155", "inc-b155", WorkflowStatus.PENDING, official_steps=list(steps)
    )


def _reset(workflow: WorkflowRequest) -> WorkflowStepSpec:
    return next(step for step in workflow.official_steps if step.operation is RESET)


def _merge(
    workflow: WorkflowRequest,
    node_id: str = "node-a",
    gpu_uuids: tuple[str, ...] = (NEW_GPU,),
) -> tuple[dict[str, list[str]], WorkflowRequest]:
    """The scope a fault on ``node_id`` widens to, and the widened workflow."""

    scope = arbitration.RecoveryArbiter().merged_gpu_scope(
        workflow, node_id, set(gpu_uuids)
    )
    return scope, _widen(workflow, scope)


#: What else in the workflow places GPU-a on node-a and GPU-b on node-b.
EVIDENCE = {
    "single-node-steps": (
        _one(VALIDATE, "node-a", "GPU-a"),
        _one(VALIDATE, "node-b", "GPU-b"),
    ),
    "explicit-mapping": (
        _shared(QUIESCE, mapping={"node-a": ["GPU-a"], "node-b": ["GPU-b"]}),
    ),
}
#: A shared reset some GPU of which nothing places.
UNPLACEABLE = {
    "unplaced": _workflow(_shared()),
    "partly-placed": _workflow(
        _shared(gpu_uuids=("GPU-a", "GPU-b", "GPU-x")), *EVIDENCE["single-node-steps"]
    ),
}


# ------------------------------------------------------------ the merged scope


@pytest.mark.parametrize("evidence", sorted(EVIDENCE))
def test_the_merged_scope_puts_a_shared_steps_gpus_on_the_nodes_holding_them(
    evidence: str,
) -> None:
    scope, _ = _merge(_workflow(_shared(), *EVIDENCE[evidence]))

    assert scope == PER_NODE, "node-b's GPU is not node-a's, nor node-a's node-b's"


@pytest.mark.parametrize("case", sorted(UNPLACEABLE))
def test_an_unplaceable_shared_step_stays_in_scope_without_attributing_a_gpu(
    case: str,
) -> None:
    scope, _ = _merge(UNPLACEABLE[case])

    assert scope == {"node-a": [NEW_GPU], "node-b": []}, "fail closed: never guess"


# ------------------------------------------------------------ the widened step


@pytest.mark.parametrize("evidence", sorted(EVIDENCE))
def test_the_widened_shared_reset_maps_each_node_to_its_own_gpus(evidence: str) -> None:
    _, widened = _merge(_workflow(_shared(), *EVIDENCE[evidence]))

    reset = _reset(widened)
    assert reset.node_ids == PAIR
    assert reset.gpu_uuids == ["GPU-a", NEW_GPU, "GPU-b"]
    assert reset.parameters["gpu_uuids_by_node"] == PER_NODE


@pytest.mark.parametrize("case", sorted(UNPLACEABLE))
def test_an_unplaceable_shared_reset_stays_without_a_mapping(case: str) -> None:
    workflow = UNPLACEABLE[case]

    _, widened = _merge(workflow)

    reset = _reset(widened)
    assert reset.node_ids == PAIR
    assert reset.gpu_uuids == sorted({*_reset(workflow).gpu_uuids, NEW_GPU}), (
        "the step's flat list keeps every GPU it had"
    )
    assert "gpu_uuids_by_node" not in reset.parameters, (
        "an entry for node-a or node-b would drop the unplaced GPU from the reset"
    )


def test_a_shared_quiesce_stays_unmapped_too() -> None:
    _, widened = _merge(_workflow(_shared(QUIESCE)))

    (quiesce,) = widened.official_steps
    assert quiesce.node_ids == PAIR
    assert quiesce.gpu_uuids == ["GPU-a", NEW_GPU, "GPU-b"]
    assert "gpu_uuids_by_node" not in quiesce.parameters, quiesce.parameters


def test_other_steps_and_new_nodes_do_not_map_an_unplaceable_step() -> None:
    """node-a has GPUs in the scope from another step, node-c from the merged
    fault. The shared reset still takes node-c and its GPU, without a mapping;
    the single-node step's mapping names its own node's GPUs."""

    workflow = _workflow(_one(QUIESCE, "node-a", "GPU-a3"), _shared())

    scope, widened = _merge(workflow, "node-c", ("GPU-c",))

    assert scope == {"node-a": ["GPU-a3"], "node-b": [], "node-c": ["GPU-c"]}
    reset = _reset(widened)
    assert reset.node_ids == ["node-a", "node-b", "node-c"]
    assert reset.gpu_uuids == ["GPU-a", "GPU-a3", "GPU-b", "GPU-c"]
    assert "gpu_uuids_by_node" not in reset.parameters, reset.parameters
    assert widened.official_steps[0].parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-a3"],
        "node-c": ["GPU-c"],
    }, "the single-node step's own node keeps its GPUs"


# ------------------------------------------------------------ unchanged shapes


def test_a_mapped_shared_step_keeps_its_mapping() -> None:
    workflow = _workflow(_shared(mapping={"node-a": ["GPU-a"], "node-b": ["GPU-b"]}))

    scope, widened = _merge(workflow)

    assert scope == PER_NODE
    assert _reset(widened).parameters["gpu_uuids_by_node"] == PER_NODE


def test_a_single_node_step_still_places_its_gpus_on_its_node() -> None:
    workflow = _workflow(_one(QUIESCE, "node-a", "GPU-a"))

    scope, widened = _merge(workflow, "node-b", ("GPU-b",))

    assert scope == {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
    (quiesce,) = widened.official_steps
    assert quiesce.node_ids == PAIR
    assert quiesce.parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-a"],
        "node-b": ["GPU-b"],
    }


def test_an_sxid_still_adds_its_node_entry_to_a_mapped_shared_step() -> None:
    workflow = _workflow(_shared(mapping={"node-a": ["GPU-a"], "node-b": ["GPU-b"]}))

    merged = merge_sxid_step_scope(workflow, _sxid("node-a", "GPU-a9"))

    assert _reset(merged).parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-a", "GPU-a9"],
        "node-b": ["GPU-b"],
    }


# ------------------------------------------------------------ what the agents receive


def _execute(
    step: WorkflowStepSpec,
) -> tuple[WorkflowStepOutcome, list[tuple[str, WorkflowOperation, list[str]]]]:
    """Run ``step`` through the fleet adapter's reset barrier (prepare, then
    commit); every signed command as (node, operation, GPUs)."""

    sent: list[tuple[str, WorkflowOperation, list[str]]] = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        command = signed.command
        sent.append((command.node_id, command.operation, list(command.gpu_uuids)))
        return NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.SUCCEEDED,
        )

    adapter = fleet_adapter(sender=send)
    context = context_for(adapter)
    context = replace(
        context,
        step=step,
        workflow=context.workflow.model_copy(update={"official_steps": [step]}),
    )
    outcome = adapter.execute(context)
    if outcome.status is WorkflowStepStatus.WAITING:
        outcome = adapter.execute(context)
    return outcome, sent


def _assert_fail_closed(workflow: WorkflowRequest, gpus: set[str]) -> None:
    """The shared reset carries no mapping, keeps every GPU in its flat list,
    and the barrier refuses it before any command is sent."""

    reset = _reset(workflow)
    assert "gpu_uuids_by_node" not in reset.parameters, reset.parameters
    assert gpus <= set(reset.gpu_uuids), reset.gpu_uuids
    outcome, sent = _execute(reset)
    assert outcome.status is WorkflowStepStatus.FAILED
    assert NO_MAPPING in (outcome.error or "")
    assert sent == [], "no agent is asked about a GPU it may not hold"


@pytest.mark.parametrize("evidence", sorted(EVIDENCE))
def test_the_barrier_sends_each_agent_only_its_own_gpus(evidence: str) -> None:
    _, widened = _merge(_workflow(_shared(), *EVIDENCE[evidence]))

    outcome, sent = _execute(_reset(widened))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert sent == [
        ("node-a", VERIFY, ["GPU-a", NEW_GPU]),
        ("node-b", VERIFY, ["GPU-b"]),
        ("node-a", RESET, ["GPU-a", NEW_GPU]),
        ("node-b", RESET, ["GPU-b"]),
    ]


@pytest.mark.parametrize("case", sorted(UNPLACEABLE))
def test_an_unplaceable_widened_reset_is_refused_before_any_command(case: str) -> None:
    workflow = UNPLACEABLE[case]

    _, widened = _merge(workflow)

    _assert_fail_closed(widened, {*_reset(workflow).gpu_uuids, NEW_GPU})


# ------------------------------------------------------------ merge after merge


def _sxid(node_id: str, gpu: str) -> SxidEvent:
    return build_sxid_event(
        f"sxid-{node_id}-{gpu}",
        NOW,
        12028,
        SxidClassification.FATAL,
        "unit",
        node_id=node_id,
        participating_gpu_uuids=[gpu],
    )


def test_merges_on_a_b_a_never_complete_an_unplaceable_reset() -> None:
    workflow = UNPLACEABLE["unplaced"]
    gpus = {"GPU-a", "GPU-b"}

    for node_id, gpu in MERGES:
        _, workflow = _merge(workflow, node_id, (gpu,))
        gpus.add(gpu)
        _assert_fail_closed(workflow, gpus)


def test_one_later_merge_does_not_complete_an_unplaceable_reset() -> None:
    """Other steps give both nodes GPUs: as a partial mapping, the withheld
    reset took {node-a: [...], node-b: [GPU-b7]} at the next fault on node-a,
    without GPU-b."""

    workflow = _workflow(
        _one(QUIESCE, "node-a", "GPU-a"), _one(QUIESCE, "node-b", "GPU-b7"), _shared()
    )
    gpus = {"GPU-a", "GPU-b"}

    for gpu in (NEW_GPU, "GPU-a3"):
        _, workflow = _merge(workflow, "node-a", (gpu,))
        gpus.add(gpu)
        _assert_fail_closed(workflow, gpus)


def test_widen_in_place_then_the_widening_never_complete_an_unplaceable_reset() -> None:
    """The applier's WIDEN_IN_PLACE runs before the family's widening and adds
    the event node's entry to any mapping dict; the scope comes from the
    workflow as it was before the disposition."""

    workflow = UNPLACEABLE["unplaced"]
    gpus = {"GPU-a", "GPU-b"}

    for node_id, gpu in MERGES:
        candidate = workflow_request(
            "wf-candidate",
            "inc-candidate",
            official_steps=[
                workflow_step(
                    RESET,
                    node_ids=[node_id],
                    gpu_uuids=[gpu],
                    parameters={"gpu_uuids_by_node": {node_id: [gpu]}},
                )
            ],
        )
        steps = widen_in_place(
            workflow,
            candidate,
            node_id,
            {gpu},
            workload_scoped_operations=WORKLOAD_SCOPED_OPERATIONS,
        )
        scope = arbitration.RecoveryArbiter().merged_gpu_scope(workflow, node_id, {gpu})
        workflow = _widen(workflow.model_copy(update={"official_steps": steps}), scope)
        gpus.add(gpu)
        _assert_fail_closed(workflow, gpus)


@pytest.mark.parametrize("dag", [False, True], ids=["after-the-widening", "dag"])
def test_sxid_merges_on_a_then_b_never_complete_an_unplaceable_reset(dag: bool) -> None:
    """A flat plan widens before ``merge_sxid_step_scope``; a DAG does not.
    Each SXID wrote its node's entry: two of them made a complete mapping."""

    workflow = UNPLACEABLE["unplaced"].model_copy(update={"dag_enabled": dag})
    gpus = {"GPU-a", "GPU-b"}

    for node_id, gpu in (("node-a", "GPU-a9"), ("node-b", "GPU-b9")):
        if not dag:
            _, workflow = _merge(workflow, node_id, (gpu,))
            gpus.add(gpu)
        workflow = merge_sxid_step_scope(workflow, _sxid(node_id, gpu))
        assert _reset(workflow).parameters["sxids_by_node"][node_id] == [12028]
        _assert_fail_closed(workflow, gpus)


# ------------------------------------------------------------ through the ingest


@pytest.mark.parametrize(
    ("placed", "expected"),
    [(True, PER_NODE), (False, None)],
    ids=["placed", "unplaced"],
)
def test_an_xid_merged_into_a_pending_shared_reset_persists_per_node_gpus(
    placed: bool, expected: dict[str, list[str]] | None
) -> None:
    """A second XID on node-a folds into the PENDING workflow whose reset spans
    node-a and node-b without a mapping; with node-b's own validation placing
    GPU-b, the persisted reset names each node's GPUs, else it stays unmapped
    (``None``: no ``gpu_uuids_by_node`` at all)."""

    context = build_context()
    _, _, workflow = ingest(
        context, _node_event(48, event_id="b155-first", gpu_uuid="GPU-a")
    )
    assert "gpu_uuids_by_node" not in _reset(workflow).parameters
    steps = [
        step.model_copy(
            update={"node_ids": list(PAIR), "gpu_uuids": ["GPU-a", "GPU-b"]}
        )
        if step.operation is RESET
        else step
        for step in workflow.official_steps
    ]
    if placed:
        steps.append(_one(VALIDATE, "node-b", "GPU-b"))
    context.store.save_workflow(copy_model(workflow, official_steps=steps))

    _, _, merged = ingest(
        context, _node_event(48, event_id="b155-second", gpu_uuid=NEW_GPU)
    )

    assert merged.request_id == workflow.request_id
    persisted = context.store.get_workflow(workflow.request_id)
    assert _reset(persisted).parameters.get("gpu_uuids_by_node") == expected
    named_for_node_a = {
        gpu
        for step in persisted.official_steps
        for gpu in step.parameters.get("gpu_uuids_by_node", {}).get("node-a", [])
    }
    assert "GPU-b" not in named_for_node_a, named_for_node_a


# ------------------------------------------------------------ the helpers


@pytest.mark.parametrize(
    ("step", "evidence", "expected"),
    [
        pytest.param(
            _one(RESET, "node-a", "GPU-a"), (), {"node-a": ["GPU-a"]}, id="single-node"
        ),
        pytest.param(
            workflow_step(RESET, node_ids=[], gpu_uuids=["GPU-a"]), (), {}, id="no-node"
        ),
        pytest.param(
            _shared(gpu_uuids=()), (), {"node-a": [], "node-b": []}, id="no-gpu"
        ),
        pytest.param(
            _shared(),
            EVIDENCE["single-node-steps"],
            {"node-a": ["GPU-a"], "node-b": ["GPU-b"]},
            id="single-node-steps",
        ),
        pytest.param(
            _shared(),
            EVIDENCE["explicit-mapping"],
            {"node-a": ["GPU-a"], "node-b": ["GPU-b"]},
            id="explicit-mapping",
        ),
        pytest.param(
            _shared(gpu_uuids=("GPU-a",)),
            EVIDENCE["single-node-steps"],
            {"node-a": ["GPU-a"], "node-b": []},
            id="a-node-holding-none",
        ),
        pytest.param(_shared(), (), None, id="unplaced"),
        pytest.param(
            _shared(gpu_uuids=("GPU-a", "GPU-b", "GPU-x")),
            EVIDENCE["single-node-steps"],
            None,
            id="partly-placed",
        ),
        pytest.param(
            _shared(),
            (_one(VALIDATE, "node-a", "GPU-a"), _one(VALIDATE, "node-c", "GPU-b")),
            None,
            id="placed-off-the-step",
        ),
        pytest.param(
            _shared(),
            (*EVIDENCE["single-node-steps"], _one(VALIDATE, "node-b", "GPU-a")),
            None,
            id="placed-on-two-of-its-nodes",
        ),
    ],
)
def test_step_gpu_shares_places_every_gpu_or_answers_unknown(
    step: WorkflowStepSpec,
    evidence: tuple[WorkflowStepSpec, ...],
    expected: dict[str, list[str]] | None,
) -> None:
    workflow = _workflow(step, *evidence)

    assert arbitration.step_gpu_shares(workflow, step) == expected


@pytest.mark.parametrize(
    ("step", "evidence", "unknown"),
    [
        pytest.param(_shared(), (), True, id="unplaced-shared"),
        pytest.param(_shared(), EVIDENCE["single-node-steps"], False, id="placed"),
        pytest.param(
            _shared(mapping={"node-a": ["GPU-a"]}), (), False, id="own-mapping"
        ),
        pytest.param(_one(RESET, "node-a", "GPU-a"), (), False, id="single-node"),
    ],
)
def test_gpu_shares_unknown_only_for_an_unplaceable_unmapped_shared_step(
    step: WorkflowStepSpec, evidence: tuple[WorkflowStepSpec, ...], unknown: bool
) -> None:
    workflow = _workflow(step, *evidence)

    assert arbitration.gpu_shares_unknown(workflow, step) is unknown


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        pytest.param(
            _shared(mapping={"node-a": ["GPU-a"], "node-b": ["GPU-b"]}),
            {"node-a": ["GPU-a", "GPU-a9"], "node-b": ["GPU-b"], "node-c": ["GPU-c"]},
            id="own-mapping",
        ),
        pytest.param(
            _one(QUIESCE, "node-a", "GPU-a"),
            {"node-a": ["GPU-a", "GPU-a9"], "node-c": ["GPU-c"]},
            id="single-node",
        ),
        pytest.param(_shared(), None, id="unplaced-shared"),
    ],
)
def test_widened_gpu_mapping_takes_the_scope_nodes_that_carry_gpus(
    step: WorkflowStepSpec, expected: dict[str, list[str]] | None
) -> None:
    """The step's own mapping plus every scope node with GPUs; a shared step
    whose GPUs nothing places gets no mapping at all (``None``)."""

    scope = {"node-a": ["GPU-a", "GPU-a9"], "node-b": [], "node-c": ["GPU-c"]}

    mapping = arbitration.widened_gpu_mapping(_workflow(step), step, scope)

    assert mapping == expected
