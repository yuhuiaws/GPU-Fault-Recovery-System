"""The DAG shape gate, and what it re-checks when the graph moves under it.

``ProductionWorkflowExecutor._validate_dag`` guards three things, and only its
step cap was covered. The other two are the ones a malformed plan trips: a
dependency cycle and a dependency index that names no step both stall the
ready-set scan forever rather than failing, because a step whose dependencies
can never be a subset of the resolved set is simply never ready -- the workflow
returns RUNNING on every pass and nothing says why.

The revalidation is the same gate applied to a moving target. ``_execute_dag``
re-reads the record from the store after every step, so a branch appended by
another writer while this executor holds the lease arrives mid-call. The
contract is that the graph is validated once per shape, keyed on
``dag_revision``: a bumped revision is re-validated and its new steps are
executed in the same call, instead of the executor finishing on the shape it
validated when it started.
"""

from __future__ import annotations

import pytest

from gpu_fault.execution import (
    ProductionWorkflowExecutor,
    WorkflowExecutionError,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step,
)
from tests.execution._support import workflow_state

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
VALIDATE = WorkflowOperation.VALIDATE_GPU


class _DagAdapter:
    """Records what ran, and optionally appends a step while step 0 runs.

    The append goes through the store on the record the executor already holds
    a lease on, which is what an ingest-time branch append does: read the
    current workflow, add the step, bump ``dag_revision``, save.
    """

    def __init__(
        self, store: InMemoryStore, *, appends: WorkflowStepSpec | None = None
    ) -> None:
        self.store = store
        self.appends = appends
        self.calls: list[tuple[WorkflowOperation, int]] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return bool(step.execution_owner == "owner-a")

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append((context.step.operation, context.step_index))
        if self.appends is not None:
            appended, self.appends = self.appends, None
            current = self.store.get_workflow(context.workflow.request_id)
            self.store.save_workflow(
                copy_model(
                    current,
                    official_steps=[*current.official_steps, appended],
                    dag_revision=current.dag_revision + 1,
                )
            )
        return WorkflowStepOutcome.succeeded()


def _dag_workflow(
    store: InMemoryStore, dependencies: list[list[int]]
) -> WorkflowRequest:
    _, workflow = workflow_state(store, [FREEZE])
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        official_steps=[
            workflow_step(FREEZE, depends_on_step_indexes=list(values))
            for values in dependencies
        ],
    )
    store.save_workflow(workflow)
    return workflow


def _executor(store: InMemoryStore, adapter: _DagAdapter) -> ProductionWorkflowExecutor:
    return active_workflow_executor(store, [adapter], {FREEZE, VALIDATE})


def test_a_dag_whose_steps_depend_on_each_other_is_rejected_before_any_step_runs() -> (
    None
):
    store = build_store()
    workflow = _dag_workflow(store, [[1], [0]])
    adapter = _DagAdapter(store)

    with pytest.raises(WorkflowExecutionError, match="cycle"):
        execute_workflow(_executor(store, adapter), workflow.request_id)

    assert adapter.calls == []


def test_a_dag_dependency_on_a_step_that_does_not_exist_is_rejected() -> None:
    store = build_store()
    workflow = _dag_workflow(store, [[3], []])
    adapter = _DagAdapter(store)

    with pytest.raises(WorkflowExecutionError, match="invalid DAG dependencies"):
        execute_workflow(_executor(store, adapter), workflow.request_id)

    assert adapter.calls == []


def test_a_step_appended_at_a_new_dag_revision_runs_in_the_same_execution() -> None:
    store = build_store()
    workflow = _dag_workflow(store, [[]])
    adapter = _DagAdapter(
        store, appends=workflow_step(VALIDATE, depends_on_step_indexes=[0])
    )

    result = execute_workflow(_executor(store, adapter), workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert adapter.calls == [(FREEZE, 0), (VALIDATE, 1)]
    final = store.get_workflow(workflow.request_id)
    assert final.dag_revision == 1
    assert sorted(final.completed_step_indexes) == [0, 1]


def test_a_step_appended_at_a_new_dag_revision_is_revalidated() -> None:
    store = build_store()
    workflow = _dag_workflow(store, [[]])
    adapter = _DagAdapter(
        store, appends=workflow_step(VALIDATE, depends_on_step_indexes=[7])
    )

    with pytest.raises(WorkflowExecutionError, match="invalid DAG dependencies"):
        execute_workflow(_executor(store, adapter), workflow.request_id)

    assert adapter.calls == [(FREEZE, 0)]
