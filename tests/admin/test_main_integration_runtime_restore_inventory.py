from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from gpu_fault.admin.submit_remediation import REMEDIATION_SCRIPT
from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import build_store, fault_incident, workflow_request


@pytest.mark.parametrize(
    ("nodes", "inventory"),
    [
        (["node-a"], {"node-a": ["GPU-a"]}),
        (["node-a"], {"node-a": ["GPU-new"]}),
        (["node-a", "node-b"], {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}),
        (["node-a", "node-b"], {"node-a": ["GPU-a"], "node-b": None}),
        (["node-a", "node-b"], {"node-a": ["GPU-a"], "node-b": "read-error"}),
    ],
    ids=["partial-dropout", "all-absent", "complete", "missing-node", "store-error"],
)
def test_in_pod_restore_reads_only_its_cluster_and_preserves_missing_gpu_scope(
    monkeypatch, capsys, nodes, inventory
):
    store = build_store()
    incident = fault_incident(
        "inc-runtime-admin",
        "event-runtime-admin",
        node_ids=nodes,
        gpu_uuids=["GPU-a", "GPU-b"],
        state=IncidentState.QUARANTINED,
        workflow_request_id="workflow-runtime-quarantine",
        fencing_token=7,
    )
    previous = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=7,
    )
    store.save_incident_and_workflow(incident, previous)
    context = ApplicationContext(store=store)
    reads = []
    woken = []

    def snapshot(cluster_id, node_id):
        reads.append((cluster_id, node_id))
        value = inventory[node_id]
        if value == "read-error":
            raise RuntimeError("inventory temporarily unavailable")
        if value is None:
            return None
        return SimpleNamespace(devices=[SimpleNamespace(gpu_uuid=gpu) for gpu in value])

    monkeypatch.setattr(store, "get_gpu_inventory_snapshot", snapshot)
    monkeypatch.setattr(context.dispatcher, "wake", lambda: woken.append(True))
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    payload = {
        "mode": "restore",
        "incident_id": incident.incident_id,
        "expected": {
            "fencing_token": 7,
            "workflow_request_id": previous.request_id,
            "state": "QUARANTINED",
            "node_ids": nodes,
        },
        "operator": "test-operator",
        "reference": "CHG-runtime-admin",
        "runtime_profile_version": "simulated-v1",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    exec(compile(REMEDIATION_SCRIPT, "<runtime-restore-inventory>", "exec"), {})

    result = json.loads(capsys.readouterr().out)
    created = store.get_workflow(result["workflow"]["request_id"])
    assert result["no_op"] is False
    assert reads == [("cluster-a", node) for node in nodes]
    assert created.fencing_token == 7
    assert created.runtime_profile_version == "simulated-v1"
    assert store.get_incident(incident.incident_id).gpu_uuids == ["GPU-a", "GPU-b"]
    assert woken == [True]
    validations = [
        step
        for step in created.official_steps
        if step.operation is WorkflowOperation.VALIDATE_GPU
    ]
    if nodes == ["node-a", "node-b"] and inventory["node-b"] == ["GPU-b"]:
        assert [(step.node_ids, step.gpu_uuids) for step in validations] == [
            (["node-a"], ["GPU-a"]),
            (["node-b"], ["GPU-b"]),
        ]
    else:
        assert [(step.node_ids, step.gpu_uuids) for step in validations] == [
            (nodes, ["GPU-a", "GPU-b"])
        ], "an absent or unreadable inventory must never authorize dropping a GPU"
