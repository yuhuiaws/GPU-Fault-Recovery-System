"""A reboot or replacement refused for want of observed isolation is handed over.

The node lifecycle adapter (HyperPod, and plain EKS through the same class)
reads the Kubernetes Node right before a RESTART_NODE or REPLACE_NODE and
refuses when the node is not cordoned, quarantine-tainted and annotated by this
incident and this workflow's fencing token, or cannot be read at all
(``adapters/hyperpod/lifecycle._observe_isolation``). The refusal comes before
the preflight, the agent revocation and the provider call. It used to climb
like a failed reboot: the whole record to a ``workflow-replace-after-<id>``
successor (which re-isolated the node and, given a healthy warm spare, failed
the node over without a reboot ever having run), a DAG branch to an in-place
REPLACE_NODE rung that met the same refusal, and either way a ticket whose
reason read ``REPLACE_NODE failed``. Now the refusal carries
``isolation_not_observed``: a node that was read but is not isolated gets the
containment hand-off (``containment_or_release``: FREEZE_EVIDENCE ->
MARK_UNSCHEDULABLE -> QUARANTINE -> ESCALATE_SUPPORT, the isolation put back,
one ticket), a node that cannot be observed the refused-containment hand-off
(``containment_refused``: FREEZE_EVIDENCE -> ESCALATE_SUPPORT), neither ladder
takes a rung, and the reason names the isolation problem -- and, for a reboot,
what the provider-submission ledger holds for the step. Kept: an attested
ownership refusal, an unknown outcome, and a reboot the provider or its
preflight refused (still REPLACE_NODE).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import refusal as ownership_refusal
from gpu_fault.cluster_executor.regional_client import RegionalHyperPodSubmissionStore
from gpu_fault.execution import WorkflowExecutionRequest
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.node_action_uncertainty import unresolved_node_action
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodNodeFailure,
    HyperPodSubmissionRecord,
    HyperPodSubmissionResult,
    hyperpod_submission_idempotency_key,
)
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import (
    CONTAINMENT_REFUSED_STAGE,
    CONTAINMENT_STAGE,
    NODE_ACTION_REFUSED_STAGE,
    HardwareEscalationService,
)
from gpu_fault.orchestration.failure_outcome import (
    ISOLATION_NOT_ISOLATED,
    ISOLATION_UNOBSERVABLE,
    LIFECYCLE_ISOLATION_DETAIL,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import (
    FakeHyperPodLifecycle,
    isolated_kubernetes_adapter,
    workflow_state,
)
from tests.orchestration._cov95_runtime_escalation import EscalationHarness

OP = WorkflowOperation
CLUSTER = "hp-cluster"
#: The reboot-after successor's shape: it isolates the node, then reboots it.
REBOOT_CHAIN = [
    OP.FREEZE_EVIDENCE,
    OP.MARK_UNSCHEDULABLE,
    OP.QUARANTINE,
    OP.RESTART_NODE,
    OP.VALIDATE_GPU,
    OP.VALIDATE_HOST,
    OP.VALIDATE_FABRIC,
    OP.RESTORE_SCHEDULING,
]
REISOLATING_HAND_OFF = [
    OP.FREEZE_EVIDENCE,
    OP.MARK_UNSCHEDULABLE,
    OP.QUARANTINE,
    OP.ESCALATE_SUPPORT,
]
OBSERVATION_HAND_OFF = [OP.FREEZE_EVIDENCE, OP.ESCALATE_SUPPORT]


def _not_found(_name: str) -> Any:
    error = KeyError("node-a")
    error.status = 404  # type: ignore[attr-defined]
    raise error


def _cordon_lifted(core: Any) -> None:
    core.node["spec"]["unschedulable"] = False


def _foreign_taint(core: Any) -> None:
    core.node["spec"]["taints"] = [
        {"key": "gpu-fault.io/quarantined", "value": "x", "effect": "NoSchedule"}
    ]


def _foreign_incident(core: Any) -> None:
    core.node["metadata"]["annotations"]["gpu-fault.io/incident-id"] = "incident-b"


def _stale_token(core: Any) -> None:
    core.node["metadata"]["annotations"]["gpu-fault.io/fencing-token"] = "2"


def _unresolvable(core: Any) -> None:
    core.read_node = _not_found


#: (case, how the node differs, refusal kind, the problem the reason names)
REFUSALS = [
    pytest.param(
        _cordon_lifted,
        ISOLATION_NOT_ISOLATED,
        "spec.unschedulable is not true",
        id="cordon-lifted",
    ),
    pytest.param(
        _foreign_taint,
        ISOLATION_NOT_ISOLATED,
        "quarantine taint for this incident is missing",
        id="foreign-taint-value",
    ),
    pytest.param(
        _foreign_incident,
        ISOLATION_NOT_ISOLATED,
        "incident annotation is 'incident-b', not this incident",
        id="foreign-incident-annotation",
    ),
    pytest.param(
        _stale_token,
        ISOLATION_NOT_ISOLATED,
        "fencing token annotation does not match this workflow",
        id="stale-fencing-token",
    ),
    pytest.param(
        _unresolvable,
        ISOLATION_UNOBSERVABLE,
        "no Kubernetes node resolves (tried node-a)",
        id="node-not-resolvable",
    ),
    pytest.param(
        None,
        ISOLATION_UNOBSERVABLE,
        "the lifecycle adapter has no Kubernetes client to read the node",
        id="no-kubernetes-client",
    ),
]


def _lifecycle_step(
    operation: WorkflowOperation,
    lifecycle: Any,
    *,
    node: Any = None,
    kubernetes: bool = True,
):
    """``operation`` run once by the real lifecycle adapter, the scheduler's
    node changed by ``node``; returns the record and its last execution."""

    store = build_store()
    _, workflow = workflow_state(store, [operation])
    adapter_kubernetes = isolated_kubernetes_adapter() if kubernetes else None
    if adapter_kubernetes is not None and node is not None:
        node(adapter_kubernetes.core)
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, kubernetes_adapter=adapter_kubernetes
    )
    parameters = (
        {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
        if operation is OP.REPLACE_NODE
        else {}
    )
    step = copy_model(
        workflow.official_steps[0], execution_owner=adapter.owner, parameters=parameters
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active_workflow_executor(store, [adapter], {operation}).execute(
        workflow.request_id,
        WorkflowExecutionRequest(
            expected_fencing_token=3,
            isolation_verified_nodes=["node-a"],
            confirm_cluster_name=CLUSTER,
        ),
    )
    saved = store.get_workflow(workflow.request_id)
    return saved, saved.step_executions[-1]


def _hand_off(operations, failed_index: int, execution):
    """The escalation of a record whose ``failed_index`` step ended as
    ``execution`` did (the simulated profile compiles the successor)."""

    h = EscalationHarness(
        operations, failed_index=failed_index, details=dict(execution.details)
    )
    executions = list(h.workflow.step_executions)
    executions[-1] = executions[-1].model_copy(update={"error": execution.error})
    h.amend(step_executions=executions)
    return HardwareEscalationService.classify(h.workflow), h.escalate()


@pytest.mark.parametrize("operation", [OP.RESTART_NODE, OP.REPLACE_NODE])
@pytest.mark.parametrize(("node", "kind", "problem"), REFUSALS)
def test_an_isolation_refusal_is_handed_over_with_the_isolation_put_back(
    operation: WorkflowOperation, node: Any, kind: str, problem: str
) -> None:
    lifecycle = FakeHyperPodLifecycle()

    workflow, execution = _lifecycle_step(
        operation, lifecycle, node=node, kubernetes=node is not None
    )

    assert workflow.status is WorkflowStatus.FAILED
    assert execution.status is WorkflowStepStatus.FAILED
    assert (lifecycle.calls, lifecycle.preflight_kwargs) == (0, {}), (
        "refused before the preflight and the provider call"
    )
    assert execution.details["safety_rejection"] is True
    assert execution.details[LIFECYCLE_ISOLATION_DETAIL] == kind
    reason = f"{operation.value} not submitted, isolation not observed: {problem}"
    assert execution.details["node_failures"] == {"node-a": [reason]}
    stage = (
        CONTAINMENT_STAGE
        if kind == ISOLATION_NOT_ISOLATED
        else (CONTAINMENT_REFUSED_STAGE)
    )
    classification = HardwareEscalationService.classify(workflow)
    assert classification is not None
    assert classification[:3] == (
        stage,
        RecoveryAction.ESCALATE_OPERATOR,
        OP.ESCALATE_SUPPORT,
    ), "no reboot or replacement rung"

    chain = (
        REBOOT_CHAIN
        if operation is OP.RESTART_NODE
        else [*REBOOT_CHAIN[:3], OP.REPLACE_NODE, *REBOOT_CHAIN[4:]]
    )
    _classified, result = _hand_off(chain, 3, execution)

    assert result is not None
    incident, successor = result
    assert successor.request_id == "workflow-support-after-source-workflow"
    assert [step.operation for step in successor.official_steps] == (
        REISOLATING_HAND_OFF if kind == ISOLATION_NOT_ISOLATED else OBSERVATION_HAND_OFF
    )
    assert incident.reasons[0] == (
        f"{stage} remediation failed; failed operation or validation: "
        f"{operation.value}; nodes: node-a: {reason}"
    ), "the reason names the isolation problem, not a failed reboot"


def test_a_dag_branch_takes_no_in_place_rung_on_an_isolation_refusal() -> None:
    lifecycle = FakeHyperPodLifecycle()
    _workflow, refused = _lifecycle_step(OP.RESTART_NODE, lifecycle, node=_stale_token)
    rejected = {"revoked_agents": ["node-a"], "reactivated_agents": ["node-a"]}

    def branch(details: dict[str, Any], error: str):
        steps = [
            workflow_step(operation, node_ids=["node-a"], gpu_uuids=["GPU-a"])
            for operation in (
                OP.MARK_UNSCHEDULABLE,
                OP.QUARANTINE,
                OP.RESTART_NODE,
                OP.VALIDATE_GPU,
                OP.RESTORE_SCHEDULING,
            )
        ]
        workflow = workflow_request(
            "wf-reboot",
            "inc-reboot",
            status=WorkflowStatus.RUNNING,
            fencing_token=3,
            official_steps=steps,
            completed_step_indexes=[0, 1],
            completed_operations=[OP.MARK_UNSCHEDULABLE, OP.QUARANTINE],
            step_executions=[
                workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
                workflow_step_execution(1, OP.QUARANTINE),
                workflow_step_execution(
                    2,
                    OP.RESTART_NODE,
                    WorkflowStepStatus.FAILED,
                    error=error,
                    details=details,
                ),
            ],
        )
        return BranchEscalator(
            DagBrancher(RecoveryArbiter()),
            lambda _workflow, operations, node, gpus: [
                workflow_step(operation, node_ids=[node], gpu_uuids=list(gpus))
                for operation in operations
            ],
        ).escalate_branch(workflow, 2, error, details=details)

    exhausted = branch(dict(refused.details), refused.error or "")
    climbed = branch(rejected, "HyperPod rejected nodes: node-a")

    assert exhausted is not None and exhausted.outcome == "exhausted"
    assert "isolation could not be observed" in exhausted.reason
    assert OP.REPLACE_NODE not in {
        step.operation for step in exhausted.workflow.official_steps
    }
    assert climbed is not None and climbed.outcome == "escalated", (
        "a reboot the provider refused keeps its replacement rung"
    )
    assert OP.REPLACE_NODE in {
        step.operation for step in climbed.workflow.official_steps
    }


class _Ledger:
    """The regional client's transport: ``_get`` answers ``None`` for a key
    the control plane has no record of (``get_remote_hyperpod_submission``)."""

    def _get(self, _path: str) -> Any:
        return None


class _BrokenLedger:
    def get_hyperpod_submission(self, _cluster_name: str, _key: str) -> Any:
        raise RuntimeError("regional control plane request failed")


def _recorded_store() -> Any:
    store = build_store()
    store.save_hyperpod_submission(
        HyperPodSubmissionRecord(
            cluster_name=CLUSTER,
            idempotency_key=hyperpod_submission_idempotency_key(
                "workflow-active", 0, OP.RESTART_NODE
            ),
            action=HyperPodAction.REBOOT,
            requested_node_identifiers=["node-a"],
        )
    )
    return store


@pytest.mark.parametrize(
    ("ledger", "state", "note"),
    [
        pytest.param(
            _recorded_store,
            "on_record",
            "a reboot submission for this step is on record; the node may "
            "already have rebooted",
            id="submission-on-record",
        ),
        pytest.param(build_store, "none", None, id="store-not-found"),
        pytest.param(
            lambda: RegionalHyperPodSubmissionStore(_Ledger()),  # type: ignore[arg-type]
            "none",
            None,
            id="regional-key-error",
        ),
        pytest.param(
            _BrokenLedger,
            "unknown",
            "whether a reboot submission for this step is on record is unknown "
            "(the ledger lookup failed)",
            id="lookup-failed",
        ),
    ],
)
def test_a_refused_reboot_says_what_the_submission_ledger_holds(
    ledger: Any, state: str, note: str | None
) -> None:
    """The lost-WAITING case: a reboot submitted on an earlier attempt whose
    record the step no longer holds, the node uncordoned by the provider's
    bootstrap. The ending stays the containment hand-off; the reason says the
    node may already have rebooted."""

    lifecycle = FakeHyperPodLifecycle()
    lifecycle.store = ledger()  # type: ignore[attr-defined]

    workflow, execution = _lifecycle_step(
        OP.RESTART_NODE, lifecycle, node=_cordon_lifted
    )

    assert execution.details["provider_submission_record"] == state
    reason = execution.details["node_failures"]["node-a"][0]
    assert reason.startswith(
        "RESTART_NODE not submitted, isolation not observed: "
        "spec.unschedulable is not true"
    ), reason
    assert reason.endswith(f"; {note}") if note else "; " not in reason, reason
    classification = HardwareEscalationService.classify(workflow)
    assert classification is not None
    assert classification[0] == CONTAINMENT_STAGE


@pytest.mark.parametrize(
    ("details", "stage"),
    [
        pytest.param(
            dict(ownership_refusal("OWNERSHIP_LEASE_LOST").details or {}),
            NODE_ACTION_REFUSED_STAGE,
            id="attested-ownership-refusal",
        ),
        pytest.param(
            {
                "requires_external_confirmation": True,
                "outcome_unknown": True,
                "manual_confirmation_required": True,
                "step_waiting_timeout_seconds": 1800,
            },
            "manual_confirmation_required",
            id="unknown-outcome",
        ),
    ],
)
def test_the_endings_judged_before_the_ladder_are_kept(
    details: dict[str, Any], stage: str
) -> None:
    h = EscalationHarness(REBOOT_CHAIN, failed_index=3, details=details)

    classification = HardwareEscalationService.classify(h.workflow)

    assert classification is not None and classification[0] == stage
    if stage == "manual_confirmation_required":
        assert unresolved_node_action(h.workflow.step_executions[-1]), (
            "an unknown reboot still parks for the operator's confirmation"
        )


class _RefusingPreflight(FakeHyperPodLifecycle):
    def preflight(self, *_args, **kwargs):
        self.preflight_kwargs = kwargs
        return SimpleNamespace(
            safe_to_submit=False,
            gate_failures=["nodes have non-actionable status: node-a:ShuttingDown"],
        )


class _RejectingProvider(FakeHyperPodLifecycle):
    def execute_step(self, step, **kwargs):
        self.calls += 1
        return HyperPodSubmissionResult(
            idempotency_key=kwargs["idempotency_key"],
            action=HyperPodAction.REBOOT,
            cluster_name=CLUSTER,
            requested_node_logical_ids=step.node_ids,
            successful_node_logical_ids=[],
            failures=[
                HyperPodNodeFailure(
                    node_logical_id="node-a",
                    error_code="ValidationException",
                    message="node cannot be rebooted",
                )
            ],
        )


@pytest.mark.parametrize(
    ("lifecycle", "error"),
    [
        pytest.param(
            _RefusingPreflight,
            "HyperPod preflight failed: nodes have non-actionable status: "
            "node-a:ShuttingDown",
            id="preflight-gate",
        ),
        pytest.param(
            _RejectingProvider,
            "HyperPod rejected nodes: node-a",
            id="provider-rejected",
        ),
    ],
)
def test_a_reboot_the_provider_cannot_perform_keeps_its_replacement_rung(
    lifecycle: type[FakeHyperPodLifecycle], error: str
) -> None:
    workflow, execution = _lifecycle_step(OP.RESTART_NODE, lifecycle())

    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.error == error
    assert LIFECYCLE_ISOLATION_DETAIL not in execution.details
    classification = HardwareEscalationService.classify(workflow)
    assert classification is not None
    assert classification[:3] == (
        "reboot",
        RecoveryAction.REPLACE_NODE,
        OP.REPLACE_NODE,
    )
