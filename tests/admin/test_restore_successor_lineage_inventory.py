"""The in-Pod restore reads the inventories of an escalation successor's lineage.

A support successor that an earlier release created for a STOP that failed on
one node of a two-node attempt names all sixteen GPUs of the attempt on that
one node (batch 149). ``submit-remediation --disposition restore`` validates
the node before it lifts the isolation, and a single-node VALIDATE_GPU that
names the other node's GPUs can only time out. The script walks the
successor's ``<rung>-after-<workflow>`` lineage and reads the inventories of
the nodes there; the validated restore leaves out the GPUs another node of
the lineage reports, and nothing else. An original incident reads only its
own nodes, as before.
"""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from gpu_fault.admin.submit_remediation import REMEDIATION_SCRIPT
from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import build_store, fault_incident, workflow_request, workflow_step

NODE_A = "gpu-node-a"
NODE_B = "gpu-node-b"
GPUS_A = [f"GPU-a{index}" for index in range(8)]
GPUS_B = [f"GPU-b{index}" for index in range(8)]
ALL_GPUS = sorted([*GPUS_A, *GPUS_B])
SOURCE_WORKFLOW = "workflow-attempt-stop-9a02"
SUCCESSOR_EVENT = f"support-after-{SOURCE_WORKFLOW}"


def _source(store) -> None:
    source = fault_incident(
        "inc-attempt-failure-9a02",
        "cluster-a/attempt-a001/TrainingAttemptFailureDetected",
        event_type="TRAINING_ATTEMPT_FAILURE",
        node_ids=[NODE_A, NODE_B],
        gpu_uuids=ALL_GPUS,
        state=IncidentState.ESCALATED,
        workflow_request_id=SOURCE_WORKFLOW,
        fencing_token=1,
    )
    store.save_incident_and_workflow(
        source,
        workflow_request(
            SOURCE_WORKFLOW,
            source.incident_id,
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            official_steps=[
                workflow_step(
                    WorkflowOperation.STOP_WORKLOADS,
                    node_ids=[NODE_A, NODE_B],
                    gpu_uuids=ALL_GPUS,
                )
            ],
        ),
    )


def _successor(store, *, event_id: str = SUCCESSOR_EVENT):
    incident = fault_incident(
        f"inc-{event_id}",
        event_id,
        event_type="TRAINING_ATTEMPT_FAILURE",
        node_ids=[NODE_A],
        gpu_uuids=ALL_GPUS,
        state=IncidentState.QUARANTINED,
        workflow_request_id=f"workflow-{event_id}",
        fencing_token=1,
    )
    previous = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=1,
    )
    store.save_incident_and_workflow(incident, previous)
    return incident, previous


def _run(monkeypatch, capsys, store, payload, inventory):
    context = ApplicationContext(store=store)
    reads = []

    def snapshot(cluster_id, node_id):
        reads.append((cluster_id, node_id))
        return SimpleNamespace(
            devices=[SimpleNamespace(gpu_uuid=gpu) for gpu in inventory[node_id]]
        )

    monkeypatch.setattr(store, "get_gpu_inventory_snapshot", snapshot)
    monkeypatch.setattr(context.dispatcher, "wake", lambda: None)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    exec(compile(REMEDIATION_SCRIPT, "<restore-successor-lineage>", "exec"), {})
    return json.loads(capsys.readouterr().out), reads


def _restore_payload(incident, previous):
    return {
        "mode": "restore",
        "incident_id": incident.incident_id,
        "expected": {
            "fencing_token": 1,
            "workflow_request_id": previous.request_id,
            "state": "QUARANTINED",
            "node_ids": [NODE_A],
        },
        "operator": "test-operator",
        "reference": "CHG-149",
        "runtime_profile_version": "simulated-v1",
    }


def _validations(store, result):
    created = store.get_workflow(result["workflow"]["request_id"])
    return [
        (step.operation, step.node_ids, step.gpu_uuids)
        for step in created.official_steps
        if step.operation is not WorkflowOperation.RESTORE_SCHEDULING
    ]


def test_the_restore_of_an_older_successor_validates_only_its_nodes_gpus(
    monkeypatch, capsys
):
    store = build_store()
    _source(store)
    incident, previous = _successor(store)

    result, reads = _run(
        monkeypatch,
        capsys,
        store,
        _restore_payload(incident, previous),
        {NODE_A: GPUS_A, NODE_B: GPUS_B},
    )

    assert reads == [("cluster-a", NODE_A), ("cluster-a", NODE_B)], (
        "the successor's own node, then its lineage's"
    )
    validations = _validations(store, result)
    assert [operation for operation, _, _ in validations] == [
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    ]
    assert all(
        nodes == [NODE_A] and gpus == GPUS_A for _, nodes, gpus in validations
    ), validations
    assert store.get_incident(incident.incident_id).gpu_uuids == ALL_GPUS, (
        "the record keeps what it named; only the steps are scoped"
    )


def test_inspect_reports_the_lineage_inventories_the_restore_plan_uses(
    monkeypatch, capsys
):
    store = build_store()
    _source(store)
    incident, _ = _successor(store)
    payload = {
        "mode": "inspect",
        "incident_id": incident.incident_id,
        "action_incident_id": f"{incident.incident_id}-action",
    }

    result, _ = _run(
        monkeypatch, capsys, store, payload, {NODE_A: GPUS_A, NODE_B: GPUS_B}
    )

    assert result["node_gpu_uuids"] == {NODE_A: GPUS_A, NODE_B: GPUS_B}


@pytest.mark.parametrize(
    "event_id",
    ["event-xid-79", "support-after-workflow-gone"],
    ids=["original-incident", "unreadable-lineage"],
)
def test_without_a_readable_lineage_every_incident_gpu_stays_required(
    monkeypatch, capsys, event_id
):
    store = build_store()
    _source(store)
    incident, previous = _successor(store, event_id=event_id)

    result, reads = _run(
        monkeypatch,
        capsys,
        store,
        _restore_payload(incident, previous),
        {NODE_A: GPUS_A, NODE_B: GPUS_B},
    )

    assert reads == [("cluster-a", NODE_A)]
    assert all(gpus == ALL_GPUS for _, _, gpus in _validations(store, result)), (
        "no other node's inventory was read, so no GPU is proven elsewhere"
    )


def test_the_walk_follows_a_chain_of_successors(monkeypatch, capsys):
    """replace-after-reboot-after-<the attempt stop>: every hop's nodes count."""

    store = build_store()
    _source(store)
    reboot_event = f"reboot-after-{SOURCE_WORKFLOW}"
    reboot, _ = _successor(store, event_id=reboot_event)
    reboot_workflow = store.get_workflow(reboot.workflow_request_id)
    store.save_workflow(
        reboot_workflow.model_copy(
            update={
                "status": WorkflowStatus.FAILED,
                "official_steps": [
                    workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[NODE_A])
                ],
            }
        )
    )
    incident, previous = _successor(
        store, event_id=f"replace-after-{reboot.workflow_request_id}"
    )

    result, reads = _run(
        monkeypatch,
        capsys,
        store,
        _restore_payload(incident, previous),
        {NODE_A: GPUS_A, NODE_B: GPUS_B},
    )

    assert reads == [("cluster-a", NODE_A), ("cluster-a", NODE_B)]
    assert all(gpus == GPUS_A for _, _, gpus in _validations(store, result)), (
        "the attempt's other node is two hops back and still places its GPUs"
    )
