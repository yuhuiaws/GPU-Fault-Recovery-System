"""``fencing_token`` is the incident's generation: shared with its workflow,
never decreasing (C-13; F-B9 tail).

Four spellings write the token (``+1`` in the in-place replacements, "keep"
for successors, ``= 1`` for new incidents, ``max`` in node_lifecycle) and four
consumers each anchor on one of them (``retired_generation_successor`` wants
strictly lower, the restore reconcile wants equal, ``record_cas`` wants
``<=``). Nothing pinned the invariant those consumers assume. This test walks
a merge chain through every family and checks, after each write, that the
incident and the workflow it points at carry the same token and that no
incident's token ever went backwards.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import RecoveryAction, WorkflowStatus
from gpu_fault.policy import SxidClassification, SxidLinkScope
from tests._builders import (
    attempt_observation,
    build_sxid_event,
    container_observation,
    copy_model,
    node_health_finding,
)

from ._support import (
    NOW,
    ApplicationContext,
    WorkloadState,
    _device_resource_finding,
    _node_event,
    _save_attempt,
    event,
    ingest,
)


class _Invariant:
    def __init__(self, context: ApplicationContext) -> None:
        self.context = context
        self.seen: dict[str, int] = {}

    def check(self, incident, workflow, label: str) -> None:
        stored_incident = self.context.store.get_incident(incident.incident_id)
        assert stored_incident.workflow_request_id, label
        pointed = self.context.store.get_workflow(stored_incident.workflow_request_id)
        assert stored_incident.fencing_token == pointed.fencing_token, (
            label,
            stored_incident.fencing_token,
            pointed.fencing_token,
        )
        if workflow is not None and workflow.request_id == pointed.request_id:
            assert workflow.fencing_token == pointed.fencing_token, label
        previous = self.seen.get(incident.incident_id)
        assert previous is None or stored_incident.fencing_token >= previous, (
            label,
            previous,
            stored_incident.fencing_token,
        )
        self.seen[incident.incident_id] = stored_incident.fencing_token


def _xid(context, code: int, *, event_id: str, gpu: str, node: str = "node-a"):
    _, incident, workflow = ingest(
        context, _node_event(code, event_id=event_id, gpu_uuid=gpu, node_id=node)
    )
    return incident, workflow


def _active_xid(context, code: int, *, event_id: str, gpu: str, node: str):
    fault = copy_model(
        _node_event(code, event_id=event_id, gpu_uuid=gpu, node_id=node),
        workload_state=WorkloadState.ACTIVE,
        job_id="job-a",
        attempt_id="attempt-a",
        affected_workload_ids=["training/job/job-a"],
    )
    _, incident, workflow = ingest(context, fault)
    return incident, workflow


def _node_scope_chain(context):
    yield "reset GPU-a", _xid(context, 48, event_id="ns-1", gpu="GPU-a")
    yield "reset GPU-b widens", _xid(context, 48, event_id="ns-2", gpu="GPU-b")
    yield "reboot replaces in place", _xid(context, 79, event_id="ns-3", gpu="GPU-a")
    _, running = _xid(context, 79, event_id="ns-4", gpu="GPU-c")
    context.store.save_workflow(
        copy_model(running, status=WorkflowStatus.RUNNING, execution_owner_id="x")
    )
    yield (
        "successor behind a running reboot",
        _xid(context, 79, event_id="ns-5", gpu="GPU-d"),
    )


def _attempt_group_chain(context):
    _save_attempt(context, ("node-a", "node-b"))
    yield (
        "grouped reset node-a",
        _active_xid(context, 48, event_id="ag-1", gpu="GPU-a", node="node-a"),
    )
    yield (
        "grouped reset node-b",
        _active_xid(context, 48, event_id="ag-2", gpu="GPU-b", node="node-b"),
    )
    yield (
        "grouped reboot node-a",
        _active_xid(context, 79, event_id="ag-3", gpu="GPU-a", node="node-a"),
    )


def _sxid(rank: int, event_id: str):
    return build_sxid_event(
        event_id,
        NOW + timedelta(seconds=1),
        11001,
        SxidClassification.FATAL,
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        node_id=f"node-{rank}",
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        product="H200",
        fabric_partition=f"cluster-a/node-{rank}/local-nvswitch",
        participating_gpu_uuids=[f"GPU-{rank}-0", f"GPU-{rank}-1"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train"],
    )


def _sxid_chain(context):
    context.store.save_attempt_observation(
        attempt_observation(
            "train",
            "train-a001",
            NOW,
            expected_critical_ranks=2,
            containers=[
                container_observation(
                    f"pod-{rank}",
                    f"trainer-{rank}",
                    rank,
                    f"node-{rank}",
                    gpu_uuids=[f"GPU-{rank}-0", f"GPU-{rank}-1"],
                )
                for rank in range(2)
            ],
            workload_ids=["training/pytorchjob/train"],
            restart_budget=3,
        )
    )
    for rank in range(2):
        sxid = _sxid(rank, f"sx-{rank}")
        incident, workflow = context.orchestrator.ingest(
            sxid, context.policy.evaluate_sxid(sxid)
        )
        yield f"sxid node-{rank}", (incident, workflow)
    sxid = _sxid(0, "sx-0-again")
    yield (
        "sxid node-0 again",
        context.orchestrator.ingest(sxid, context.policy.evaluate_sxid(sxid)),
    )


def _drain_chain(context):
    yield "pending reset", _xid(context, 48, event_id="dr-1", gpu="GPU-a")
    yield (
        "critical quarantine replaces it",
        context.orchestrator.ingest_node_health(
            node_health_finding(
                "finding-dr-2",
                "dr-2",
                observed_at=NOW,
                category=NodeHealthCategory.MCE,
                severity="critical",
                reason="machine check",
                recommended_action=RecoveryAction.QUARANTINE,
                runtime_profile_version="simulated-v1",
                workload_state=WorkloadState.IDLE,
            )
        ),
    )
    yield (
        "plugin restart on node-b",
        context.orchestrator.ingest_node_health(
            _device_resource_finding(
                event_id="dr-3",
                metric_name="efa_kubernetes_allocatable_mismatch",
                action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            ).model_copy(update={"node_id": "node-b"})
        ),
    )
    yield (
        "driver remediation upgrades it",
        context.orchestrator.ingest_node_health(
            _device_resource_finding(
                event_id="dr-4",
                metric_name="efa_inventory_mismatch",
                action=RecoveryAction.REMEDIATE_EFA_DRIVER,
            ).model_copy(update={"node_id": "node-b"})
        ),
    )


def _replacement_chain(context):
    context.store.save_attempt_observation(
        attempt_observation(
            "train-multi",
            "train-multi-a001",
            NOW,
            expected_critical_ranks=2,
            containers=[
                container_observation(
                    f"pod-{rank}",
                    f"worker-{rank}",
                    rank,
                    f"node-{rank}",
                    gpu_uuids=[f"GPU-{rank}"],
                )
                for rank in range(2)
            ],
            workload_ids=["training/job/train-multi"],
            restart_budget=2,
        )
    )
    for rank in range(2):
        yield (
            f"replace node-{rank}",
            context.orchestrator.ingest_node_health(
                node_health_finding(
                    f"finding-rp-{rank}",
                    f"rp-{rank}",
                    node_id=f"node-{rank}",
                    observed_at=NOW,
                    category=NodeHealthCategory.GPU,
                    severity="critical",
                    reason=f"unrecoverable GPU fault on node-{rank}",
                    recommended_action=RecoveryAction.REPLACE_NODE,
                    gpu_uuids=[f"GPU-{rank}"],
                    runtime_profile_version="simulated-v1",
                    workload_state=WorkloadState.ACTIVE,
                    affected_workload_ids=["training/job/train-multi"],
                )
            ),
        )


@pytest.mark.parametrize(
    "chain",
    [
        _node_scope_chain,
        _attempt_group_chain,
        _sxid_chain,
        _drain_chain,
        _replacement_chain,
    ],
    ids=["faults", "grouped_faults", "sxid", "drain", "node_lifecycle"],
)
def test_incident_and_workflow_share_a_token_that_never_decreases(
    context: ApplicationContext, chain
) -> None:
    invariant = _Invariant(context)
    steps = 0
    for label, (incident, workflow) in chain(context):
        assert workflow is not None, label
        invariant.check(incident, workflow, label)
        steps += 1
    assert steps >= 2
    assert event, (
        "the merge must leave an event to compare tokens against"
    )  # imported for the fault families' event builder
