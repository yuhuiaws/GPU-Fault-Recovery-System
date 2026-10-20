"""A step built for one node names only the GPUs on that node (batch 149).

A single-node VALIDATE_GPU needs fresh per-GPU telemetry for every GPU it names
(``adapters/gpu_validation.py``, unchanged). The escalation successor narrows
its source's nodes to the failed ones; when the source workflow had not
recorded which GPU sits on which node, the successor inherited every GPU of
the source. Batch 146 made a multi-node attempt incident name all ranks' GPUs
(16 on two 8-GPU nodes), so the reboot successor of a reset that failed on one
rank's node -- and the support successor of a passive STOP that failed on one
node, and the operator restore after it -- named the other node's eight GPUs
on a node that can never report them: the step waited out its per-step cap and
the ladder replaced a healthy node. These tests drive the realistic shapes
through ``escalate`` and the validated restore and fail on dadeb4c2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricSource,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.validated_restore import build_validated_restore_workflow
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NODE_A = "gpu-node-a"
NODE_B = "gpu-node-b"
GPUS_A = [f"GPU-a{index}" for index in range(8)]
GPUS_B = [f"GPU-b{index}" for index in range(8)]
ALL_GPUS = sorted([*GPUS_A, *GPUS_B])
JOB = "training/pytorchjob/llm-pretrain"
ATTEMPT = "llm-pretrain-attempt-a001"
RESET_GPU = "GPU-a3"
VALIDATIONS = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
)
RESTART_CONTEXT = {
    "cluster_id": "cluster-a",
    "job_id": JOB,
    "source_attempt_id": ATTEMPT,
    "source_gpu_count": 16,
    "restart_budget": 1,
}
STEP = WorkflowOperation


def _inventory(store, node_id: str, gpus: list[str]) -> None:
    store.save_gpu_inventory_snapshot(
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id=node_id,
            observed_at=datetime.now(timezone.utc),
            source=GpuMetricSource.DCGM_EXPORTER,
            source_boot_id=f"boot-{node_id}",
            devices=[
                GpuInventoryDevice(
                    gpu_index=index, gpu_uuid=gpu, pci_bdf=f"0000:{index + 16:02x}:00.0"
                )
                for index, gpu in enumerate(gpus)
            ],
        )
    )


def _telemetry(store, node_id: str, gpus: list[str], observed_at=None) -> None:
    at = observed_at or datetime.now(timezone.utc) - timedelta(hours=2)
    store.observe_gpu_metrics(
        [
            (
                ("cluster-a", node_id, gpu, "gpu_temperature_c"),
                GpuMetricLatest(
                    cluster_id="cluster-a",
                    node_id=node_id,
                    observed_at=at,
                    source=GpuMetricSource.DCGM_EXPORTER,
                    sample=GpuMetricSample(
                        metric_name="DCGM_FI_DEV_GPU_TEMP",
                        canonical_name="gpu_temperature_c",
                        value=41.0,
                        gpu_index=str(index),
                        gpu_uuid=gpu,
                    ),
                ),
            )
            for index, gpu in enumerate(gpus)
        ]
    )


def _attempt(store) -> None:
    store.save_attempt_observation(
        attempt_observation(
            JOB,
            ATTEMPT,
            datetime.now(timezone.utc),
            expected_critical_ranks=2,
            workload_ids=[JOB],
            containers=[
                container_observation(
                    "pod-uid-master-0",
                    "llm-pretrain-master-0",
                    0,
                    NODE_A,
                    gpu_uuids=GPUS_A,
                    gpu_count=8,
                ),
                container_observation(
                    "pod-uid-worker-0",
                    "llm-pretrain-worker-0",
                    1,
                    NODE_B,
                    gpu_uuids=GPUS_B,
                    gpu_count=8,
                ),
            ],
        )
    )


def _attempt_incident(incident_id: str, workflow_id: str, **values):
    defaults = dict(
        event_type="TRAINING_ATTEMPT_TERMINAL",
        node_ids=[NODE_A, NODE_B],
        gpu_uuids=ALL_GPUS,
        job_id=JOB,
        attempt_id=ATTEMPT,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow_id,
        fencing_token=1,
        policy_version="passive-recovery-v1",
        policy_source="completion-handler",
    )
    defaults.update(values)
    return fault_incident(incident_id, f"event-{incident_id}", **defaults)


def _failed(store, incident, steps, failed_index, *, details=None, error=None):
    """Save ``incident`` and its FAILED workflow: every step before
    ``failed_index`` succeeded, ``failed_index`` failed."""

    executions = [
        workflow_step_execution(index, steps[index].operation, phase="official")
        for index in range(failed_index)
    ]
    executions.append(
        workflow_step_execution(
            failed_index,
            steps[failed_index].operation,
            WorkflowStepStatus.FAILED,
            phase="official",
            error=error or f"{steps[failed_index].operation.value} failed",
            details=dict(details or {}),
        )
    )
    workflow = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_steps=steps,
        completed_step_indexes=list(range(failed_index)),
        completed_operations=[steps[index].operation for index in range(failed_index)],
        step_executions=executions,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


def _passive_reset_steps():
    """The recovery workflow compiled from a marker plan: the marker names
    rank 0's node and the GPU that faulted; the incident names the attempt."""

    one = [NODE_A]
    return [
        workflow_step(STEP.FREEZE_EVIDENCE, node_ids=[NODE_A, NODE_B]),
        workflow_step(
            STEP.STOP_WORKLOADS, node_ids=[NODE_A, NODE_B], workload_ids=[JOB]
        ),
        workflow_step(STEP.QUIESCE_GPU_SERVICES, node_ids=one, gpu_uuids=[RESET_GPU]),
        workflow_step(STEP.VERIFY_NO_GPU_CLIENTS, node_ids=one, gpu_uuids=[RESET_GPU]),
        workflow_step(STEP.RESET_GPU, node_ids=one, gpu_uuids=[RESET_GPU]),
        workflow_step(STEP.RESTORE_GPU_SERVICES, node_ids=one, gpu_uuids=[RESET_GPU]),
        workflow_step(STEP.VALIDATE_GPU, node_ids=one),
        workflow_step(STEP.RESTORE_SCHEDULING, node_ids=one),
        workflow_step(
            STEP.RESTART_WORKLOAD,
            node_ids=[NODE_A, NODE_B],
            workload_ids=[JOB],
            parameters=dict(RESTART_CONTEXT),
        ),
    ]


def _validations(workflow):
    return [step for step in workflow.official_steps if step.operation in VALIDATIONS]


class _HealthyNodes:
    """GPU telemetry as the validation adapter reads it: every GPU of a node
    reports ``gpu_temperature_c`` at ``sampled_at``; collectors are fresh."""

    def __init__(self, gpus_by_node: dict[str, list[str]], sampled_at: datetime):
        self.gpus_by_node = gpus_by_node
        self.sampled_at = sampled_at
        self.store = self

    def list_collector_statuses(self, cluster_id, node_id):
        return [
            CollectorStatus(
                cluster_id=cluster_id,
                node_id=node_id,
                collector=kind,
                observed_at=self.sampled_at,
                ingested_at=self.sampled_at,
                last_success_at=self.sampled_at,
            )
            for kind in (CollectorKind.GPU_METRICS, CollectorKind.HOST_TELEMETRY)
        ]

    def list_telemetry_metrics_latest(self, cluster_id, node_id):
        return []

    def latest(self, cluster_id, node_id):
        return [
            SimpleNamespace(
                observed_at=self.sampled_at,
                source=GpuMetricSource.DCGM_EXPORTER,
                sample=SimpleNamespace(
                    canonical_name="gpu_temperature_c", gpu_uuid=gpu, value=41.0
                ),
            )
            for gpu in self.gpus_by_node.get(node_id, [])
        ]

    def findings(self, cluster_id, node_id):
        return []

    def get_agent(self, cluster_id, node_id):
        raise KeyError(node_id)

    def get_workflow(self, request_id):
        raise KeyError(request_id)

    def get_incident(self, incident_id):
        raise KeyError(incident_id)


def _validate_after_reboot(incident, workflow, healthy):
    """Run the successor's VALIDATE_GPU as the executor would after its
    RESTART_NODE succeeded, with every GPU of the healthy node reporting."""

    rebooted_at = datetime.now(timezone.utc) - timedelta(seconds=90)
    reboot = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is STEP.RESTART_NODE
    )
    index = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is STEP.VALIDATE_GPU
    )
    ran = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                reboot, STEP.RESTART_NODE, phase="official", updated_at=rebooted_at
            )
        ],
    )
    adapter = GpuValidationAdapter(
        _HealthyNodes(healthy, rebooted_at + timedelta(seconds=60))
    )
    return adapter.execute(
        WorkflowStepContext(
            incident=incident,
            workflow=ran,
            step=workflow.official_steps[index],
            step_index=index,
            request=WorkflowExecutionRequest(expected_fencing_token=1),
            idempotency_key=f"{workflow.request_id}/validate-gpu",
        )
    )


def test_a_reset_that_failed_on_one_rank_node_reboots_it_with_its_own_gpus_only():
    """The marker plan reset GPU-a3 on rank 0's node; the reset failed. The
    reboot successor isolates and reboots that node, and its validation asks
    it for its own eight GPUs -- not for rank 1's eight on the other node."""

    context = build_context()
    store = context.store
    _inventory(store, NODE_A, GPUS_A)
    _inventory(store, NODE_B, GPUS_B)
    incident = _attempt_incident("inc-passive-recovery", "workflow-passive-recovery")
    workflow = _failed(store, incident, _passive_reset_steps(), 4)

    result = context.orchestrator.escalate_failed_hardware_remediation(workflow)

    assert result is not None
    successor_incident, successor = result
    assert successor.request_id == f"workflow-reboot-after-{workflow.request_id}"
    assert successor_incident.effective_action is RecoveryAction.REBOOT_NODE
    assert successor_incident.node_ids == [NODE_A]
    assert sorted(successor_incident.gpu_uuids) == GPUS_A, (
        "the successor incident is about rank 0's node: its eight GPUs, every one "
        "of which the attempt incident named, and none of rank 1's"
    )
    assert successor_incident.gpu_uuids[0] == RESET_GPU, "the reset GPU leads"
    validations = _validations(successor)
    assert [step.operation for step in validations] == list(VALIDATIONS)
    for step in validations:
        assert step.node_ids == [NODE_A], step
        assert sorted(step.gpu_uuids) == GPUS_A, (step.operation, step.gpu_uuids)
    assert not {
        gpu
        for step in successor.official_steps
        if step.node_ids == [NODE_A]
        for gpu in step.gpu_uuids
    } & set(GPUS_B), "no step of rank 0's node names a GPU of rank 1's node"

    outcome = _validate_after_reboot(
        successor_incident, successor, {NODE_A: GPUS_A, NODE_B: GPUS_B}
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


def test_a_step_naming_a_siblings_gpus_can_only_time_out_into_a_replacement():
    """The rule the fix keeps and why the old scope mattered: the reboot
    successor as dadeb4c2 built it named all sixteen GPUs on rank 0's node.
    The node reported its own eight after the reboot and still WAITED on rank
    1's eight; at the per-step cap the ladder replaced the healthy node."""

    context = build_context()
    store = context.store
    incident = _attempt_incident(
        "inc-reboot-after",
        "workflow-reboot-after-workflow-passive-recovery",
        node_ids=[NODE_A],
        state=IncidentState.ACTION_PENDING,
        effective_action=RecoveryAction.REBOOT_NODE,
    )
    one = [NODE_A]
    steps = [
        workflow_step(STEP.FREEZE_EVIDENCE, node_ids=one, gpu_uuids=ALL_GPUS),
        workflow_step(STEP.MARK_UNSCHEDULABLE, node_ids=one, gpu_uuids=ALL_GPUS),
        workflow_step(STEP.QUARANTINE, node_ids=one, gpu_uuids=ALL_GPUS),
        workflow_step(STEP.RESTART_NODE, node_ids=one, gpu_uuids=ALL_GPUS),
        workflow_step(STEP.VALIDATE_GPU, node_ids=one, gpu_uuids=ALL_GPUS),
    ]
    old_successor = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_steps=steps,
    )

    waiting = _validate_after_reboot(incident, old_successor, {NODE_A: GPUS_A})

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["pending_nodes"] == [NODE_A]
    assert (
        sorted(waiting.details["node_pending"][NODE_A]["missing_recent_metrics_by_gpu"])
        == GPUS_B
    ), "a single-node step still needs every GPU it names"

    capped = _failed(
        store,
        incident,
        steps,
        4,
        details=waiting.details,
        error="step 4/VALIDATE_GPU stayed non-terminal for 601s, past the 600s cap",
    )
    result = HardwareEscalationService(store, context.orchestrator._builder).escalate(
        capped
    )

    assert result is not None
    replacement_incident, replacement = result
    assert replacement.request_id == f"workflow-replace-after-{capped.request_id}"
    assert replacement_incident.node_ids == [NODE_A], (
        "the timeout escalates the healthy node the step was waiting on"
    )


def test_a_stop_that_failed_on_one_node_gives_its_support_successor_that_nodes_gpus():
    """Batch 146's path: the attempt-failure containment (FREEZE_EVIDENCE /
    STOP_WORKLOADS over both nodes, all sixteen GPUs) whose STOP failed on rank
    0's node with an unknown outcome. The support successor names that node's
    GPUs -- placed from the attempt's container observations -- and the
    operator restore after the manual confirmation validates only those."""

    context = build_context()
    store = context.store
    _attempt(store)
    incident = _attempt_incident(
        "inc-attempt-failure-9a02",
        "workflow-attempt-stop-9a02",
        event_type="TRAINING_ATTEMPT_FAILURE",
    )
    steps = [
        workflow_step(
            STEP.FREEZE_EVIDENCE, node_ids=[NODE_A, NODE_B], gpu_uuids=ALL_GPUS
        ),
        workflow_step(
            STEP.STOP_WORKLOADS,
            node_ids=[NODE_A, NODE_B],
            gpu_uuids=ALL_GPUS,
            workload_ids=[JOB],
        ),
    ]
    workflow = _failed(
        store,
        incident,
        steps,
        1,
        details={"failed_nodes": [NODE_A], "manual_confirmation_required": True},
        error="stop of the attempt on gpu-node-a did not confirm",
    )

    result = context.orchestrator.escalate_failed_hardware_remediation(workflow)

    assert result is not None
    support_incident, support = result
    assert support.request_id == f"workflow-support-after-{workflow.request_id}"
    assert support_incident.node_ids == [NODE_A]
    assert support_incident.gpu_uuids == GPUS_A, support_incident.gpu_uuids
    assert all(
        not set(step.gpu_uuids) & set(GPUS_B)
        for step in support.official_steps
        if step.node_ids == [NODE_A]
    ), [(step.operation, step.gpu_uuids) for step in support.official_steps]

    quarantined = support_incident.model_copy(
        update={"state": IncidentState.QUARANTINED}
    )
    _, restore = build_validated_restore_workflow(
        quarantined,
        operator="arn:aws:sts::123456789012:assumed-role/Admin/ops",
        reference="CHG-149",
        now=datetime.now(timezone.utc),
        node_gpu_uuids={NODE_A: GPUS_A},
    )

    for step in _validations(restore):
        assert step.node_ids == [NODE_A]
        assert step.gpu_uuids == GPUS_A, (step.operation, step.gpu_uuids)


def test_without_any_placement_the_failed_node_validates_node_wide():
    """Nothing places a GPU -- no inventory, no telemetry, no observation,
    only multi-node steps. The successor names none of the sixteen on the
    failed node (any of them may be the other node's) and validates it
    node-wide, instead of inheriting the source's whole list."""

    context = build_context()
    store = context.store
    incident = _attempt_incident("inc-unplaced", "workflow-unplaced")
    steps = [
        workflow_step(
            STEP.FREEZE_EVIDENCE, node_ids=[NODE_A, NODE_B], gpu_uuids=ALL_GPUS
        ),
        workflow_step(STEP.RESET_GPU, node_ids=[NODE_A, NODE_B], gpu_uuids=ALL_GPUS),
    ]
    workflow = _failed(store, incident, steps, 1, details={"failed_nodes": [NODE_A]})

    successor_incident, successor = (
        context.orchestrator.escalate_failed_hardware_remediation(workflow)
    )

    assert successor_incident.node_ids == [NODE_A]
    assert successor_incident.gpu_uuids == []
    for step in _validations(successor):
        assert step.node_ids == [NODE_A]
        assert step.gpu_uuids == [], "node-wide: fresh node telemetry, every finding"


def test_a_partly_known_node_validates_its_full_known_inventory():
    """Only rank 0's node reports (its inventory); rank 1's node has no
    record. The other node's GPUs are named nowhere, and the failed node's
    validation asks for every GPU it reports."""

    context = build_context()
    store = context.store
    _inventory(store, NODE_A, GPUS_A)
    incident = _attempt_incident(
        "inc-half-known", "workflow-half-known", gpu_uuids=[*GPUS_B, RESET_GPU]
    )
    steps = [
        workflow_step(STEP.RESET_GPU, node_ids=[NODE_A, NODE_B], gpu_uuids=ALL_GPUS)
    ]
    workflow = _failed(store, incident, steps, 0, details={"failed_nodes": [NODE_A]})

    successor_incident, successor = (
        context.orchestrator.escalate_failed_hardware_remediation(workflow)
    )

    assert sorted(successor_incident.gpu_uuids) == GPUS_A
    for step in _validations(successor):
        assert step.node_ids == [NODE_A]
        assert sorted(step.gpu_uuids) == GPUS_A, step.gpu_uuids


def test_a_step_narrowed_to_one_node_is_no_proof_a_siblings_gpu_sits_there():
    """A RESTORE_SCHEDULING trimmed to the readmitted node keeps the union of
    GPUs (``workflow_quarantine.suppress_readmission``); read as a single-node
    step it would put rank 1's GPUs on rank 0's node. What the nodes report
    themselves decides."""

    context = build_context()
    store = context.store
    _telemetry(store, NODE_A, GPUS_A)
    _telemetry(store, NODE_B, GPUS_B)
    incident = _attempt_incident("inc-narrowed", "workflow-narrowed")
    steps = [
        workflow_step(STEP.RESET_GPU, node_ids=[NODE_A], gpu_uuids=[RESET_GPU]),
        workflow_step(STEP.RESTORE_SCHEDULING, node_ids=[NODE_A], gpu_uuids=ALL_GPUS),
    ]
    workflow = _failed(store, incident, steps, 0)

    successor_incident, successor = (
        context.orchestrator.escalate_failed_hardware_remediation(workflow)
    )

    assert sorted(successor_incident.gpu_uuids) == GPUS_A
    for step in _validations(successor):
        assert sorted(step.gpu_uuids) == GPUS_A, step.gpu_uuids


def test_a_single_node_source_keeps_a_gpu_nothing_else_places():
    """Fail-closed where it cannot be a sibling's: every node of the source is
    the failed node, so a GPU of its incident that no inventory lists (fallen
    off the bus before any sample) stays required on it."""

    context = build_context()
    store = context.store
    _inventory(store, NODE_A, [gpu for gpu in GPUS_A if gpu != "GPU-a7"])
    incident = _attempt_incident(
        "inc-one-node",
        "workflow-one-node",
        node_ids=[NODE_A],
        gpu_uuids=[RESET_GPU, "GPU-a7"],
        event_type="XID",
    )
    steps = [workflow_step(STEP.RESET_GPU, node_ids=[NODE_A], gpu_uuids=[RESET_GPU])]
    workflow = _failed(store, incident, steps, 0)

    successor_incident, successor = (
        context.orchestrator.escalate_failed_hardware_remediation(workflow)
    )

    assert successor_incident.gpu_uuids == [RESET_GPU, "GPU-a7"]
    for step in _validations(successor):
        assert step.gpu_uuids == [RESET_GPU, "GPU-a7"], step.gpu_uuids


def test_the_restore_of_an_older_successor_drops_gpus_its_lineage_places_elsewhere():
    """A support successor an earlier release created with all sixteen GPUs on
    rank 0's node. ``submit-remediation`` reads its lineage's inventories; a
    GPU rank 1's node reports is not named on rank 0's validation, while a GPU
    no inventory lists stays required (absence is no proof)."""

    legacy = fault_incident(
        "inc-support-after-workflow-attempt-stop-9a02",
        "support-after-workflow-attempt-stop-9a02",
        event_type="TRAINING_ATTEMPT_FAILURE",
        node_ids=[NODE_A],
        gpu_uuids=[*ALL_GPUS, "GPU-gone"],
        state=IncidentState.QUARANTINED,
        workflow_request_id="workflow-support-after-workflow-attempt-stop-9a02",
        fencing_token=2,
    )

    _, restore = build_validated_restore_workflow(
        legacy,
        operator="arn:aws:sts::123456789012:assumed-role/Admin/ops",
        reference="CHG-149",
        now=datetime.now(timezone.utc),
        node_gpu_uuids={NODE_A: GPUS_A, NODE_B: GPUS_B},
    )

    for step in _validations(restore):
        assert step.node_ids == [NODE_A]
        assert step.gpu_uuids == [*GPUS_A, "GPU-gone"], step.gpu_uuids
    assert restore.official_steps[-1].operation is STEP.RESTORE_SCHEDULING


def test_a_release_only_node_takes_its_gpus_out_of_the_validated_nodes_steps():
    """The restore preview of a two-node incident whose rank 1 node is only
    quiesce-tainted (another incident's cordon holds it): rank 0's node is
    validated alone and names only its own GPUs."""

    incident = _attempt_incident(
        "inc-two-node", "workflow-two-node", state=IncidentState.QUARANTINED
    )

    _, restore = build_validated_restore_workflow(
        incident,
        operator="arn:aws:sts::123456789012:assumed-role/Admin/ops",
        reference="CHG-149",
        now=datetime.now(timezone.utc),
        node_gpu_uuids={NODE_A: GPUS_A, NODE_B: GPUS_B},
        release_only_node_ids=[NODE_B],
    )

    for step in _validations(restore):
        assert step.node_ids == [NODE_A]
        assert step.gpu_uuids == GPUS_A, step.gpu_uuids
    release = restore.official_steps[-1]
    assert release.operation is STEP.RESTORE_SCHEDULING
    assert release.node_ids == [NODE_A, NODE_B]


def _legacy_reboot_successor(store):
    """The source (both nodes, all sixteen GPUs) and the reboot successor an
    earlier release built from it: rank 0's node, every step naming all 16."""

    source = _attempt_incident(
        "inc-passive-recovery", "workflow-passive-recovery", attempt_id=None
    )
    _failed(store, source, _passive_reset_steps(), 4)
    one = [NODE_A]
    legacy = fault_incident(
        "inc-reboot-after-workflow-passive-recovery",
        "reboot-after-workflow-passive-recovery",
        event_type="TRAINING_ATTEMPT_TERMINAL",
        node_ids=one,
        gpu_uuids=ALL_GPUS,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-reboot-after-workflow-passive-recovery",
        fencing_token=1,
        effective_action=RecoveryAction.REBOOT_NODE,
    )
    steps = [
        workflow_step(operation, node_ids=one, gpu_uuids=ALL_GPUS)
        for operation in (
            STEP.FREEZE_EVIDENCE,
            STEP.MARK_UNSCHEDULABLE,
            STEP.QUARANTINE,
            STEP.RESTART_NODE,
            STEP.VALIDATE_GPU,
        )
    ]
    return legacy, _failed(store, legacy, steps, 3, error="node did not come back")


def test_an_older_successor_that_fails_again_hands_on_only_its_nodes_gpus():
    """Repeated escalations compounded the inherited scope: every step of the
    older successor names all sixteen GPUs on rank 0's node, so its own steps
    "place" them there. What the nodes of its lineage report decides."""

    context = build_context()
    store = context.store
    _inventory(store, NODE_A, GPUS_A)
    _inventory(store, NODE_B, GPUS_B)
    _, legacy = _legacy_reboot_successor(store)

    replacement_incident, replacement = (
        context.orchestrator.escalate_failed_hardware_remediation(legacy)
    )

    assert replacement.request_id == f"workflow-replace-after-{legacy.request_id}"
    assert replacement_incident.node_ids == [NODE_A]
    assert sorted(replacement_incident.gpu_uuids) == GPUS_A
    for step in _validations(replacement):
        assert sorted(step.gpu_uuids) == GPUS_A, (step.operation, step.gpu_uuids)


def test_the_lineage_walk_is_bounded_and_stops_at_an_unreadable_link():
    from gpu_fault.orchestration.escalation import escalation_lineage_nodes

    context = build_context()
    store = context.store
    legacy, _ = _legacy_reboot_successor(store)
    orphan = legacy.model_copy(update={"event_id": "reboot-after-workflow-gone"})

    assert escalation_lineage_nodes(store, legacy) == {NODE_A, NODE_B}
    assert escalation_lineage_nodes(store, orphan) == set()
    chained = legacy.model_copy(
        update={"event_id": f"replace-after-{legacy.workflow_request_id}"}
    )
    assert escalation_lineage_nodes(store, chained, limit=1) == {NODE_A}, (
        "one hop reaches the older successor only"
    )
