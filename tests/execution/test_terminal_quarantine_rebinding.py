"""Terminal holds stay on failed nodes while readmission follows real spares."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_PREVIOUS_UNSCHEDULABLE,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.node_rebinding import rebind_nodes
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.regional import RemoteIncidentOwnershipReport
from gpu_fault.store import InMemoryStore
from gpu_fault.workflow_quarantine import (
    TERMINAL_QUARANTINE_NODES,
    inherit_terminal_quarantine,
    replacement_ancestors,
    terminal_quarantine_covered,
    terminal_quarantine_nodes,
)
from tests._builders import (
    active_workflow_executor,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._cov95_runtime_restart import ApiError
from tests.execution._support import RESTART_PARAMETERS, UnusedApi
from tests.hyperpod._cov95_runtime_failover import FailoverHarness
from tests.hyperpod.test_hyperpod_spares import FakeCore, kubernetes_node

OP = WorkflowOperation
KUBE_OWNER = "gpu-fault-kubernetes-adapter"
PROVIDER_TAINT = {
    "key": "sagemaker.amazonaws.com/node-health-status",
    "value": "Unschedulable",
    "effect": "NoSchedule",
}


def step(
    operation: OP,
    nodes: tuple[str, ...] = ("node-a",),
    *,
    dependencies: tuple[int, ...] = (),
    **values: Any,
) -> WorkflowStepSpec:
    parameters = values.pop("parameters", {})
    if operation is OP.RESTART_WORKLOAD:
        parameters = {**RESTART_PARAMETERS, **parameters}
    owner = (
        KUBE_OWNER
        if operation in {OP.MARK_UNSCHEDULABLE, OP.QUARANTINE, OP.RESTORE_SCHEDULING}
        else "owner-a"
    )
    return workflow_step(
        operation,
        owner,
        node_ids=list(nodes),
        depends_on_step_indexes=list(dependencies),
        parameters=parameters,
        **values,
    )


def plan(
    steps: list[WorkflowStepSpec], *, dag: bool = False, **values: Any
) -> WorkflowRequest:
    return workflow_request(
        "peer-workflow",
        "peer-incident",
        official_steps=steps,
        dag_enabled=dag,
        **values,
    )


def hold(nodes: tuple[str, ...] = ("node-a",)) -> WorkflowRequest:
    return plan([step(OP.QUARANTINE, nodes)])


def recovery(
    *, dag: bool, operation: OP = OP.REPLACE_NODE, with_quarantine: bool = True
) -> WorkflowRequest:
    operations = [OP.FREEZE_EVIDENCE, OP.MARK_UNSCHEDULABLE, OP.STOP_WORKLOADS]
    if with_quarantine:
        operations.append(OP.QUARANTINE)
    operations.extend(
        [operation, OP.VALIDATE_GPU, OP.RESTORE_SCHEDULING, OP.RESTART_WORKLOAD]
    )
    return plan(
        [
            step(op, dependencies=(index - 1,) if dag and index else ())
            for index, op in enumerate(operations)
        ],
        dag=dag,
    )


def incident_for(
    workflow: WorkflowRequest, nodes: tuple[str, ...] = ("node-a",)
) -> FaultIncident:
    return fault_incident(
        workflow.incident_id,
        "peer-event",
        node_ids=list(nodes),
        workflow_request_id=workflow.request_id,
        fencing_token=workflow.fencing_token,
        state=IncidentState.ACTION_PENDING,
    )


def context_for(
    workflow: WorkflowRequest, index: int, incident: FaultIncident | None = None
) -> WorkflowStepContext:
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident or incident_for(workflow),
        step=workflow.official_steps[index],
        step_index=index,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key=f"{workflow.request_id}/{index}",
    )


def isolated_node(
    name: str, *, incident_id: str = "peer-incident", fence: int = 3
) -> dict[str, Any]:
    node = kubernetes_node()
    node["metadata"]["name"] = name
    node["metadata"]["annotations"] = {
        ANNOTATION_INCIDENT: incident_id,
        ANNOTATION_FENCING: str(fence),
        ANNOTATION_PREVIOUS_UNSCHEDULABLE: "false",
    }
    node["spec"]["taints"] = [
        deepcopy(PROVIDER_TAINT),
        {
            "key": QUARANTINE_TAINT,
            "value": quarantine_taint_value(incident_id),
            "effect": "NoSchedule",
        },
    ]
    return node


class RecordingCore(FakeCore):
    def __init__(self, nodes: dict[str, Any]) -> None:
        super().__init__(nodes)
        self.reads: list[str] = []

    def read_node(self, node_id: str) -> dict[str, Any]:
        self.reads.append(node_id)
        if node_id not in self.nodes:
            raise ApiError(404)
        return super().read_node(node_id)


class RecordingAdapter:
    def __init__(self, replacement_details: dict[str, Any] | None = None) -> None:
        self.replacement_details = replacement_details or {}
        self.calls: list[tuple[OP, list[str]]] = []

    def supports(self, value: WorkflowStepSpec) -> bool:
        return value.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append((context.step.operation, list(context.step.node_ids)))
        return WorkflowStepOutcome.succeeded(
            details=(
                self.replacement_details
                if context.step.operation is OP.REPLACE_NODE
                else {}
            )
        )


def scheduler(
    core: RecordingCore, *, store: InMemoryStore | None = None, provider: Any = None
) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=store,
        ownership_provider=provider,
    )


def execute_plan(
    workflow: WorkflowRequest,
    core: RecordingCore,
    adapter: RecordingAdapter,
    nodes: tuple[str, ...] = ("node-a",),
) -> tuple[Any, InMemoryStore]:
    store = InMemoryStore()
    store.save_incident(incident_for(workflow, nodes))
    store.save_workflow(workflow)
    executor = active_workflow_executor(
        store,
        [adapter, scheduler(core, store=store)],
        {value.operation for value in workflow.official_steps},
    )
    return execute_workflow(executor, workflow.request_id), store


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
@pytest.mark.parametrize("with_quarantine", [False, True], ids=["missing-q", "has-q"])
@pytest.mark.parametrize("operation", [OP.REPLACE_NODE, OP.RESTART_NODE])
def test_inherited_hold_preserves_only_the_replacement_readmission_tail(
    dag: bool, with_quarantine: bool, operation: OP
) -> None:
    candidate = recovery(dag=dag, operation=operation, with_quarantine=with_quarantine)
    before = candidate.model_copy(deep=True)

    inherited = inherit_terminal_quarantine(hold(), candidate)

    live = {
        value.operation
        for index, value in enumerate(inherited.official_steps)
        if index not in inherited.superseded_step_indexes
    }
    readmission = {OP.RESTORE_SCHEDULING, OP.RESTART_WORKLOAD}
    assert readmission.intersection(live) == (
        readmission if operation is OP.REPLACE_NODE else set()
    ), "only replacement can retain future readmission of a held node"
    assert terminal_quarantine_nodes(inherited) == {"node-a"}, (
        "preserved readmission must not erase the original terminal hold"
    )
    assert terminal_quarantine_covered(inherited, hold()), (
        "a coherent replacement tail must not continually queue another successor"
    )
    assert inherit_terminal_quarantine(hold(), inherited) == inherited, (
        "repeated inheritance must not duplicate containment or shift indexes"
    )
    assert candidate == before, "inheritance must not mutate its input plan"


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
def test_missing_quarantine_precedes_hardware_and_remaps_retired_dependencies(
    dag: bool,
) -> None:
    candidate = plan(
        [
            step(OP.FREEZE_EVIDENCE),
            step(OP.MARK_UNSCHEDULABLE, dependencies=(0,)),
            step(OP.RESET_GPU, dependencies=(1,)),
            step(OP.REPLACE_NODE, dependencies=(2,)),
            step(OP.VALIDATE_GPU, dependencies=(3,)),
            step(OP.RESTORE_SCHEDULING, dependencies=(4,)),
            step(OP.RESTART_WORKLOAD, dependencies=(5,)),
        ],
        dag=dag,
        superseded_step_indexes=[2],
    )
    inherited = inherit_terminal_quarantine(hold(), candidate)
    core = RecordingCore({name: isolated_node(name) for name in ("node-a", "spare-a")})
    actions = RecordingAdapter({"node_rebindings": {"node-a": "spare-a"}})

    result, store = execute_plan(inherited, core, actions)

    assert inherited.official_steps[2].operation is OP.QUARANTINE, (
        "missing terminal containment must precede retired and live hardware"
    )
    assert inherited.superseded_step_indexes == [3], (
        "retirement must follow the old RESET after insertion, not retire quarantine"
    )
    assert inherited.official_steps[4].depends_on_step_indexes == [2, 3], (
        "replacement must depend on containment and its remapped predecessor"
    )
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert actions.calls == [
        (OP.FREEZE_EVIDENCE, ["node-a"]),
        (OP.REPLACE_NODE, ["node-a"]),
        (OP.VALIDATE_GPU, ["spare-a"]),
        (OP.RESTART_WORKLOAD, ["spare-a"]),
    ], "retired hardware must not run and the live tail must follow the binding"
    assert store.get_incident(inherited.incident_id).node_ids == [
        "spare-a",
        "node-a",
    ], "the incident must continue to own the failed-node hold"
    assert core.nodes["node-a"]["spec"]["unschedulable"] is True, (
        "only the spare may be restored"
    )


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
@pytest.mark.parametrize("after_stop", [False, True], ids=["pre-stop", "post-stop"])
def test_late_quarantine_branch_stays_on_the_failed_node(
    dag: bool, after_stop: bool
) -> None:
    existing = recovery(dag=dag)
    if after_stop:
        existing = existing.model_copy(
            update={
                "completed_step_indexes": [0, 1, 2],
                "completed_operations": [
                    OP.FREEZE_EVIDENCE,
                    OP.MARK_UNSCHEDULABLE,
                    OP.STOP_WORKLOADS,
                ],
                "step_executions": [workflow_step_execution(2, OP.STOP_WORKLOADS)],
            }
        )
    incoming = plan(
        [step(OP.FREEZE_EVIDENCE), step(OP.MARK_UNSCHEDULABLE), step(OP.QUARANTINE)]
    )
    joined = DagBrancher(RecoveryArbiter()).append_parallel_job_branch(
        existing, incoming
    )
    replacement = next(
        index
        for index, value in enumerate(joined.official_steps)
        if value.operation is OP.REPLACE_NODE
    )

    rebound, incident = rebind_nodes(
        InMemoryStore(),
        joined,
        incident_for(joined),
        {"node-a": "spare-a"},
        is_safety=False,
        after_index=replacement,
    )

    for before, after in zip(
        joined.official_steps[len(existing.official_steps) :],
        rebound.official_steps[len(existing.official_steps) :],
        strict=True,
    ):
        assert after.node_ids == ["node-a"], (
            "late evidence and containment belong to the failed node, not its spare"
        )
        assert after.branch_node_ids == before.branch_node_ids == ["node-a"], (
            "replacement must not change fixed branch identity"
        )
    restored = [
        value.node_ids
        for index, value in enumerate(rebound.official_steps)
        if value.operation is OP.RESTORE_SCHEDULING
        and index not in rebound.superseded_step_indexes
    ]
    assert restored == [["spare-a"]], "late quarantine must preserve spare readmission"
    assert terminal_quarantine_nodes(rebound) == {"node-a"}, (
        "rebinding must neither lose the hold nor transfer it to the spare"
    )
    assert set(incident.node_ids) == {"node-a", "spare-a"}, (
        "incident scope must include the retained original and replacement"
    )


def test_forward_dag_readmission_executes_on_the_bound_spare() -> None:
    candidate = plan(
        [
            step(OP.QUARANTINE, parameters={TERMINAL_QUARANTINE_NODES: ["node-a"]}),
            step(OP.RESTORE_SCHEDULING, dependencies=(3,)),
            step(OP.RESTART_WORKLOAD, dependencies=(1,)),
            step(OP.REPLACE_NODE, dependencies=(0,)),
        ],
        dag=True,
        completed_step_indexes=[0],
        completed_operations=[OP.QUARANTINE],
    )
    inherited = inherit_terminal_quarantine(hold(), candidate)
    core = RecordingCore({name: isolated_node(name) for name in ("node-a", "spare-a")})
    actions = RecordingAdapter({"node_rebindings": {"node-a": "spare-a"}})

    result, store = execute_plan(inherited, core, actions)

    assert inherited.superseded_step_indexes == [], (
        "forward-edge replacement ancestors authorize preserving the future tail"
    )
    assert result.status is WorkflowStatus.SUCCEEDED, result
    final = store.get_workflow(inherited.request_id)
    assert final.official_steps[1].node_ids == ["spare-a"], (
        "DAG execution order, not list position, determines future readmission"
    )
    assert (OP.RESTART_WORKLOAD, ["spare-a"]) in actions.calls, (
        "the forward-edge join must also follow the actual replacement"
    )
    assert core.nodes["node-a"]["spec"]["unschedulable"] is True, (
        "forward edges must not release the failed node"
    )


@pytest.mark.parametrize(
    "defect", ["unrelated", "parallel", "superseded", "cycle", "bad-index"]
)
def test_unproven_replacement_ancestry_never_preserves_readmission(defect: str) -> None:
    candidate = plan(
        [
            step(OP.QUARANTINE, parameters={TERMINAL_QUARANTINE_NODES: ["node-a"]}),
            step(OP.REPLACE_NODE, dependencies=(0,)),
            step(OP.FREEZE_EVIDENCE, dependencies=(1,)),
            step(OP.RESTORE_SCHEDULING, dependencies=(2,)),
            step(OP.RESTART_WORKLOAD, dependencies=(3,)),
        ],
        dag=True,
    )
    if defect == "unrelated":
        candidate.official_steps[1] = step(OP.REPLACE_NODE, ("node-b",))
    elif defect == "parallel":
        candidate.official_steps[2] = step(OP.FREEZE_EVIDENCE, dependencies=(0,))
    elif defect == "superseded":
        candidate.superseded_step_indexes = [1]
    elif defect == "cycle":
        candidate.official_steps[1] = step(OP.REPLACE_NODE, dependencies=(2,))
    else:
        candidate.official_steps[2] = step(OP.FREEZE_EVIDENCE, dependencies=(99,))

    inherited = inherit_terminal_quarantine(hold(), candidate)

    assert "node-a" not in replacement_ancestors(candidate, 3), (
        "an unrelated, retired, cyclic or invalid replacement is not authority"
    )
    assert not terminal_quarantine_covered(candidate, hold()), (
        "coverage must reject the same unproven readmission as suppression"
    )
    assert {3, 4} <= set(inherited.superseded_step_indexes), (
        "unsafe ancestry must retire both original-node readmission operations"
    )


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
@pytest.mark.parametrize(
    "bindings",
    [None, {}, {"node-a": "spare-a"}, {"node-a": "node-a", "node-b": "node-b"}],
    ids=["absent", "empty", "partial", "self"],
)
@pytest.mark.parametrize("node_state", ["isolated", "unisolated", "absent"])
def test_incomplete_binding_refuses_restore_before_any_api_or_restart(
    dag: bool, bindings: dict[str, str] | None, node_state: str
) -> None:
    nodes = ("node-a", "node-b")
    workflow = plan(
        [
            step(
                OP.QUARANTINE,
                nodes,
                parameters={TERMINAL_QUARANTINE_NODES: list(nodes)},
            ),
            step(OP.REPLACE_NODE, nodes, dependencies=(0,)),
            step(OP.VALIDATE_GPU, nodes, dependencies=(1,)),
            step(OP.RESTORE_SCHEDULING, nodes, dependencies=(2,)),
            step(OP.RESTART_WORKLOAD, nodes, dependencies=(3,)),
        ],
        dag=dag,
        completed_step_indexes=[0],
        completed_operations=[OP.QUARANTINE],
    )
    core_nodes = {name: isolated_node(name) for name in (*nodes, "spare-a")}
    for name in nodes:
        if node_state == "absent":
            core_nodes.pop(name)
        elif node_state == "unisolated":
            core_nodes[name] = kubernetes_node(unschedulable=False)
    core = RecordingCore(core_nodes)
    before = deepcopy(core.nodes)
    actions = RecordingAdapter(
        {} if bindings is None else {"node_rebindings": bindings}
    )

    result, store = execute_plan(workflow, core, actions, nodes)

    final = store.get_workflow(workflow.request_id)
    refusal = next(
        value
        for value in final.step_executions
        if value.operation is OP.RESTORE_SCHEDULING
    )
    assert result.status is WorkflowStatus.FAILED, result
    assert OP.REPLACE_NODE in result.completed_operations, (
        "the regression must reach a successful replacement result before refusal"
    )
    assert refusal.details.get("reason") == "TERMINAL_QUARANTINE_HOLD", (
        "failure must be the held-node gate, not an unrelated adapter or budget error"
    )
    assert core.reads == [] and core.patches == [], (
        "even a partially rebound shared restore must reject before its first API call"
    )
    assert core.nodes == before, "original and spare scheduling state must be untouched"
    assert all(
        operation is not OP.RESTART_WORKLOAD for operation, _ in actions.calls
    ), "missing replacement identity must never implicitly admit another attempt"


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
def test_shared_restore_keeps_replaced_and_unheld_nodes_only(dag: bool) -> None:
    nodes = ("node-a", "node-b", "node-c")
    candidate = plan(
        [
            step(OP.QUARANTINE, ("node-a", "node-b")),
            step(OP.REPLACE_NODE, dependencies=(0,)),
            step(
                OP.RESTORE_SCHEDULING,
                nodes,
                dependencies=(1,),
                parameters={
                    "gpu_uuids_by_node": {node: [f"GPU-{node}"] for node in nodes}
                },
            ),
            step(OP.RESTART_WORKLOAD, nodes, dependencies=(2,)),
        ],
        dag=dag,
    )
    inherited = inherit_terminal_quarantine(hold(("node-a", "node-b")), candidate)

    rebound, _ = rebind_nodes(
        InMemoryStore(),
        inherited,
        incident_for(inherited, nodes),
        {"node-a": "spare-a"},
        is_safety=False,
        after_index=1,
    )
    core = RecordingCore({name: isolated_node(name) for name in (*nodes, "spare-a")})
    outcome = scheduler(core).execute(context_for(rebound, 2))

    assert inherited.official_steps[2].node_ids == ["node-a", "node-c"], (
        "replacement exempts A, unrelated C remains restorable, but held B is removed"
    )
    assert set(inherited.official_steps[2].parameters["gpu_uuids_by_node"]) == {
        "node-a",
        "node-c",
    }, "narrowing must remove only the held, unreplaced GPU mapping"
    assert inherited.official_steps[3].node_ids == list(nodes), (
        "shared workload restart retains its existing multi-node semantics"
    )
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert {name for name, _ in core.patches} == {"spare-a", "node-c"}, (
        "Kubernetes may restore only the bound spare and unaffected node"
    )
    assert all(
        core.nodes[name]["spec"]["unschedulable"] for name in ("node-a", "node-b")
    ), "neither terminal-held original may be uncordoned"
    assert terminal_quarantine_nodes(rebound) == {"node-a", "node-b"}, (
        "partial replacement must retain both original holds"
    )


@pytest.mark.parametrize("history", ["completed", "superseded", "waiting", "failed"])
def test_rebinding_never_rewrites_resolved_or_submitted_step_identity(
    history: str,
) -> None:
    workflow = inherit_terminal_quarantine(
        hold(),
        plan(
            [
                step(OP.QUARANTINE),
                step(OP.REPLACE_NODE),
                step(OP.VALIDATE_GPU, gpu_uuids=["GPU-old"]),
                step(OP.RESTORE_SCHEDULING),
            ]
        ),
    )
    if history == "completed":
        workflow.completed_step_indexes = [2]
        workflow.completed_operations = [OP.VALIDATE_GPU]
    elif history == "superseded":
        workflow.superseded_step_indexes = [2]
    else:
        workflow.step_executions = [
            workflow_step_execution(
                2,
                OP.VALIDATE_GPU,
                WorkflowStepStatus.WAITING
                if history == "waiting"
                else WorkflowStepStatus.FAILED,
                adapter_operation_id="original-validation-operation",
                details={"source_node": "node-a"},
            )
        ]
    before = workflow.model_copy(deep=True)

    rebound, _ = rebind_nodes(
        InMemoryStore(),
        workflow,
        incident_for(workflow),
        {"node-a": "spare-a"},
        is_safety=False,
        after_index=1,
    )

    assert rebound.official_steps[2] == before.official_steps[2], (
        "existing execution history must keep its original node and GPU scope"
    )
    assert rebound.step_executions == before.step_executions, (
        "a spare binding must not rewrite recorded outcomes"
    )
    assert rebound.official_steps[3].node_ids == ["spare-a"], (
        "the independent, unsubmitted restore must still follow the replacement"
    )
    assert workflow == before, "rebinding must not mutate its caller's copy"


@pytest.mark.parametrize(
    "inventory_present", [False, True], ids=["node-wide", "spare-gpus"]
)
@pytest.mark.parametrize(
    "pending_snapshot", [False, True], ids=["fresh", "snapshot-retry"]
)
def test_production_failover_result_rebinds_and_restores_only_the_spare(
    inventory_present: bool, pending_snapshot: bool
) -> None:
    harness = FailoverHarness()
    original = harness.context
    replacement = original.step
    workflow = original.workflow.model_copy(
        update={
            "official_steps": [
                step(OP.QUARANTINE),
                replacement,
                step(
                    OP.VALIDATE_GPU,
                    gpu_uuids=["GPU-old"],
                    parameters={"gpu_uuids_by_node": {"node-a": ["GPU-old"]}},
                ),
                step(OP.RESTORE_SCHEDULING),
            ]
        }
    )
    workflow = inherit_terminal_quarantine(hold(), workflow).model_copy(
        update={"completed_step_indexes": [0], "completed_operations": [OP.QUARANTINE]}
    )
    harness.context = replace(
        original, workflow=workflow, step=workflow.official_steps[1], step_index=1
    )
    for name, node in harness.core.nodes.items():
        node["metadata"]["name"] = name
    if pending_snapshot:
        harness.actions.outcomes[OP.TRIGGER_HEALTH_SNAPSHOT] = (
            WorkflowStepOutcome.waiting()
        )
        pending = harness.execute()
        assert pending.status is WorkflowStepStatus.WAITING, pending
        assert not pending.details.get("node_rebindings"), (
            "a pending health snapshot is not a completed replacement binding"
        )
        workflow = workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        1,
                        OP.REPLACE_NODE,
                        WorkflowStepStatus.WAITING,
                        adapter_operation_id=pending.adapter_operation_id,
                        details=deepcopy(pending.details),
                    )
                ]
            }
        )
        harness.context = replace(harness.context, workflow=workflow)
        harness.actions.outcomes[OP.TRIGGER_HEALTH_SNAPSHOT] = (
            WorkflowStepOutcome.succeeded()
        )
    if inventory_present:
        harness.store.save_gpu_inventory_snapshot(
            GpuInventorySnapshot(
                cluster_id=original.incident.cluster_id,
                node_id="spare-a",
                observed_at=datetime.now(timezone.utc),
                source=GpuMetricSource.DCGM_EXPORTER,
                source_boot_id="spare-boot",
                devices=[
                    GpuInventoryDevice(
                        gpu_index=0, gpu_uuid="GPU-spare", pci_bdf="0000:01:00.0"
                    )
                ],
            )
        )

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["node_rebindings"] == {"node-a": "spare-a"}, (
        "the production result builder must identify every failed-node replacement"
    )
    assert outcome.details["provider_mutation_submitted"] is False, outcome
    rebound, incident = rebind_nodes(
        harness.store,
        workflow,
        original.incident,
        outcome.details["node_rebindings"],
        is_safety=False,
        after_index=1,
    )
    restored = harness.scheduler.execute(context_for(rebound, 3, incident))
    expected_gpus = ["GPU-spare"] if inventory_present else []
    assert restored.status is WorkflowStepStatus.SUCCEEDED, restored
    assert rebound.official_steps[2].gpu_uuids == expected_gpus, (
        "validation must use spare inventory or node-wide scope, never old GPU UUIDs"
    )
    assert rebound.official_steps[2].parameters["gpu_uuids_by_node"] == {
        "spare-a": expected_gpus
    }, "the per-node GPU mapping must move with the actual node"
    assert harness.core.nodes["node-a"]["spec"]["unschedulable"] is True, (
        "production spare success does not admit the terminal-held original"
    )
    assert harness.core.nodes["spare-a"]["spec"]["unschedulable"] is False, (
        "the real Kubernetes restoration path must admit the successfully bound spare"
    )
    assert set(incident.node_ids) == {"node-a", "spare-a"}, (
        "the incident retains both containment and replacement scope"
    )
    assert len(harness.spares.calls) == 1 and harness.spares.releases == [], (
        "snapshot replay must retain its one existing spare allocation"
    )
    assert harness.provider.submissions == [], "provider replacement remains forbidden"


class OwnershipProvider:
    def __init__(self, report: RemoteIncidentOwnershipReport) -> None:
        self.report = report
        self.calls: list[str] = []

    def incident_ownership(self, incident_id: str) -> RemoteIncidentOwnershipReport:
        self.calls.append(incident_id)
        return self.report


def takeover_adapter(
    mode: str, *, prior_state: str = "held"
) -> tuple[RecordingCore, KubernetesWorkflowAdapter]:
    core = RecordingCore(
        {"node-a": isolated_node("node-a", incident_id="prior-incident", fence=1)}
    )
    terminal = prior_state != "active"
    held = prior_state != "recovered"
    status = WorkflowStatus.SUCCEEDED if terminal else WorkflowStatus.RUNNING
    state = IncidentState.QUARANTINED if held else IncidentState.RECOVERED
    report = RemoteIncidentOwnershipReport(
        incident_id="prior-incident",
        known=prior_state != "unknown",
        workflow_request_id="prior-workflow",
        workflow_status=status.value,
        terminal=terminal,
        incident_state=state.value,
        quarantine_hold=held,
    )
    if mode == "storeless":
        return core, scheduler(core, provider=OwnershipProvider(report))
    store = InMemoryStore()
    if prior_state != "unknown":
        operation = OP.QUARANTINE if held else OP.MARK_UNSCHEDULABLE
        store.save_incident(
            fault_incident(
                "prior-incident",
                "prior-event",
                state=state,
                fencing_token=1,
                workflow_request_id="prior-workflow",
            )
        )
        store.save_workflow(
            workflow_request(
                "prior-workflow",
                "prior-incident",
                status,
                fencing_token=1,
                official_steps=[step(operation)],
                completed_step_indexes=[0],
                completed_operations=[operation],
            )
        )
    return core, scheduler(core, store=store)


@pytest.mark.parametrize("mode", ["local", "storeless"])
@pytest.mark.parametrize(
    ("shape", "allowed"),
    [
        ("legacy-hold", True),
        ("retired-restore", True),
        ("other-node-restore", True),
        ("replacement-tail", True),
        ("live-restore", False),
        ("legacy-temporary", False),
        ("other-node-hold", False),
        ("metadata-mismatch", False),
        ("missing-name", False),
        ("malformed", False),
    ],
)
def test_takeover_uses_actual_node_and_effective_terminal_hold(
    mode: str, shape: str, allowed: bool
) -> None:
    core, adapter = takeover_adapter(mode)
    workflow = plan(
        [
            step(OP.MARK_UNSCHEDULABLE),
            step(OP.QUARANTINE, parameters={TERMINAL_QUARANTINE_NODES: ["node-a"]}),
        ]
    )
    if shape in {"legacy-hold", "legacy-temporary"}:
        workflow.official_steps[1] = step(OP.QUARANTINE)
    if shape in {"retired-restore", "live-restore", "legacy-temporary"}:
        workflow.official_steps.append(step(OP.RESTORE_SCHEDULING))
        if shape == "retired-restore":
            workflow.superseded_step_indexes = [2]
    elif shape == "other-node-restore":
        workflow.official_steps.append(step(OP.RESTORE_SCHEDULING, ("node-b",)))
    elif shape == "replacement-tail":
        workflow.official_steps.extend(
            [step(OP.REPLACE_NODE), step(OP.RESTORE_SCHEDULING)]
        )
    elif shape == "other-node-hold":
        workflow.official_steps[1] = step(
            OP.QUARANTINE,
            ("node-b",),
            parameters={TERMINAL_QUARANTINE_NODES: ["node-b"]},
        )
    elif shape == "metadata-mismatch":
        core.nodes["node-a"]["metadata"]["name"] = "node-b"
    elif shape == "missing-name":
        core.nodes["node-a"]["metadata"].pop("name")
    elif shape == "malformed":
        workflow.official_steps[1].parameters[TERMINAL_QUARANTINE_NODES] = ["node-b"]
    before = deepcopy(core.nodes)

    outcome = adapter.execute(context_for(workflow, 0))

    assert outcome.status is (
        WorkflowStepStatus.SUCCEEDED if allowed else WorkflowStepStatus.FAILED
    ), (shape, outcome)
    if not allowed:
        assert outcome.details.get("safety_rejection") is True, outcome
        assert core.nodes == before and core.patches == [], (
            "unproven node-specific holds must not transfer isolation ownership"
        )
        return
    annotations = core.nodes["node-a"]["metadata"]["annotations"]
    assert annotations[ANNOTATION_INCIDENT] == workflow.incident_id, (
        "legitimate terminal-held takeover must acquire this incident's ownership"
    )
    assert annotations[ANNOTATION_PREVIOUS_UNSCHEDULABLE] == "false", (
        "takeover must retain the original baseline, not sample the current cordon"
    )
    assert PROVIDER_TAINT in core.nodes["node-a"]["spec"]["taints"], (
        "node-specific takeover must not remove provider-owned isolation"
    )
    assert core.nodes["node-a"]["spec"]["unschedulable"] is True, (
        "acquiring a held node never makes it schedulable"
    )


@pytest.mark.parametrize("mode", ["local", "storeless"])
@pytest.mark.parametrize("prior_state", ["active", "unknown"])
def test_effective_hold_does_not_override_unknown_or_active_ownership(
    mode: str, prior_state: str
) -> None:
    core, adapter = takeover_adapter(mode, prior_state=prior_state)
    workflow = plan([step(OP.MARK_UNSCHEDULABLE), step(OP.QUARANTINE)])
    before = deepcopy(core.nodes)

    outcome = adapter.execute(context_for(workflow, 0))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert core.nodes == before and core.patches == [], (
        "a preservation plan is not authority to steal an active or unknown node"
    )


@pytest.mark.parametrize("mode", ["local", "storeless"])
def test_legacy_recovered_isolation_still_restores_its_original_baseline(
    mode: str,
) -> None:
    core, adapter = takeover_adapter(mode, prior_state="recovered")
    workflow = plan([step(OP.MARK_UNSCHEDULABLE), step(OP.RESTORE_SCHEDULING)])

    acquired = adapter.execute(context_for(workflow, 0))
    restored = adapter.execute(context_for(workflow, 1))

    assert acquired.status is WorkflowStepStatus.SUCCEEDED, acquired
    assert restored.status is WorkflowStepStatus.SUCCEEDED, restored
    assert core.nodes["node-a"]["spec"]["unschedulable"] is False, (
        "the new hold gate must not strand ordinary recovered nodes"
    )
    assert core.nodes["node-a"]["spec"]["taints"] == [PROVIDER_TAINT], (
        "restoration removes only gpu-fault isolation, not provider taints"
    )
    assert core.nodes["node-a"]["metadata"]["annotations"] == {}, (
        "legacy restoration still clears its own isolation annotations"
    )


@pytest.mark.parametrize("node_state", ["absent", "unisolated", "unreserved-spare"])
def test_unheld_legacy_restore_retains_noop_and_unreserved_spare_controls(
    node_state: str,
) -> None:
    workflow = plan([step(OP.RESTORE_SCHEDULING)])
    node = isolated_node("node-a")
    if node_state == "unisolated":
        node = kubernetes_node(unschedulable=False)
    elif node_state == "unreserved-spare":
        node["metadata"]["labels"]["gpu-fault.io/spare"] = "true"
        node["metadata"]["annotations"]["gpu-fault.io/spare-pool-state"] = "AVAILABLE"
    core = RecordingCore({} if node_state == "absent" else {"node-a": node})

    outcome = scheduler(core).execute(context_for(workflow, 0))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    if node_state == "unreserved-spare":
        assert core.nodes["node-a"]["spec"]["unschedulable"] is True, (
            "clearing gpu-fault isolation must not activate an unreserved spare"
        )
    else:
        assert core.patches == [], "absent and unisolated unheld nodes remain no-ops"


def test_same_incident_newer_fence_still_refuses_held_takeover() -> None:
    workflow = hold()
    core = RecordingCore({"node-a": isolated_node("node-a", fence=4)})
    before = deepcopy(core.nodes)

    outcome = scheduler(core).execute(context_for(workflow, 0))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert "newer workflow generation" in (outcome.error or ""), outcome
    assert core.nodes == before and core.patches == [], (
        "a terminal hold cannot authorize a stale fencing token"
    )


@pytest.mark.parametrize("raw", [None, "node-a", ["node-b"], ["node-a", None]])
def test_malformed_quarantine_refuses_restore_without_reading_the_node(
    raw: Any,
) -> None:
    workflow = plan(
        [
            step(OP.QUARANTINE, parameters={TERMINAL_QUARANTINE_NODES: raw}),
            step(OP.RESTORE_SCHEDULING),
        ]
    )
    core = RecordingCore({"node-a": isolated_node("node-a")})

    outcome = scheduler(core).execute(context_for(workflow, 1))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details.get("safety_rejection") is True, outcome
    assert "scope is malformed" in (outcome.error or ""), outcome
    assert core.reads == [] and core.patches == [], (
        "invalid scope cannot fall through to any ownership or no-op path"
    )


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
def test_unmarked_temporary_quarantine_after_replacement_still_follows_the_spare(
    dag: bool,
) -> None:
    workflow = plan(
        [
            step(OP.REPLACE_NODE),
            step(OP.QUARANTINE, dependencies=(0,)),
            step(OP.RESTORE_SCHEDULING, dependencies=(1,)),
        ],
        dag=dag,
    )

    rebound, incident = rebind_nodes(
        InMemoryStore(),
        workflow,
        incident_for(workflow),
        {"node-a": "spare-a"},
        is_safety=False,
        after_index=0,
    )

    assert terminal_quarantine_nodes(workflow) == set(), (
        "legacy temporary isolation with readmission is not a persistent hold"
    )
    assert [value.node_ids for value in rebound.official_steps[1:]] == [
        ["spare-a"],
        ["spare-a"],
    ], "only explicitly fixed containment is exempt from normal replacement rebinding"
    assert incident.node_ids == ["spare-a"], (
        "unheld legacy incidents must not acquire an artificial failed-node hold"
    )


@pytest.mark.parametrize("history", ["completed", "submitted"])
def test_missing_quarantine_cannot_renumber_existing_execution_identity(
    history: str,
) -> None:
    candidate = plan([step(OP.REPLACE_NODE), step(OP.RESTORE_SCHEDULING)], dag=True)
    if history == "completed":
        candidate.completed_step_indexes = [0]
        candidate.completed_operations = [OP.REPLACE_NODE]
    else:
        candidate.step_executions = [
            workflow_step_execution(
                0,
                OP.REPLACE_NODE,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="already-submitted-replacement",
            )
        ]
    before = candidate.model_copy(deep=True)

    with pytest.raises(ValueError, match="cannot insert quarantine"):
        inherit_terminal_quarantine(hold(), candidate)

    assert candidate == before, (
        "failed inheritance must leave submitted indexes and outcomes untouched"
    )
