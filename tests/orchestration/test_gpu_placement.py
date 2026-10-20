"""The placement sources of ``orchestration/gpu_placement.py`` and their limits.

Every source is read defensively: an unreadable inventory, telemetry or
observation contributes nothing (the GPU stays unplaced and follows the
documented decision), another attempt's containers place nothing, and a
container read without a node or without GPUs places nothing either.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from gpu_fault.models import IncidentState, WorkflowOperation
from gpu_fault.orchestration.gpu_placement import (
    escalation_gpu_scope,
    scope_validation_steps,
    store_gpu_placement,
)
from gpu_fault.orchestration.validated_restore import build_validated_restore_workflow
from tests._builders import fault_incident, workflow_request, workflow_step

ATTEMPT = "attempt-a001"


class _Store:
    def __init__(self, *, snapshots=None, latest=None, observations=None):
        self.snapshots = snapshots or {}
        self.latest = latest or {}
        self.observations = observations

    def get_gpu_inventory_snapshot(self, cluster_id, node_id):
        value = self.snapshots.get(node_id)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return None
        return SimpleNamespace(devices=[SimpleNamespace(gpu_uuid=gpu) for gpu in value])

    def list_gpu_metrics_latest(self, cluster_id, node_id):
        value = self.latest.get(node_id, [])
        if isinstance(value, Exception):
            raise value
        return [SimpleNamespace(sample=SimpleNamespace(gpu_uuid=gpu)) for gpu in value]

    def list_attempt_observations(self, cluster_id):
        if isinstance(self.observations, Exception):
            raise self.observations
        return self.observations or []


def _container(node_id, gpus):
    return SimpleNamespace(node_id=node_id, gpu_uuids=list(gpus))


def test_every_source_places_and_an_unreadable_one_contributes_nothing(caplog):
    store = _Store(
        snapshots={"node-a": ["GPU-a0"], "node-b": RuntimeError("store down")},
        latest={"node-a": ["GPU-a1", None], "node-b": RuntimeError("store down")},
        observations=[
            SimpleNamespace(
                attempt_id="another-attempt", containers=[_container("node-c", ["X"])]
            ),
            SimpleNamespace(
                attempt_id=ATTEMPT,
                containers=[
                    _container("node-b", ["GPU-b0"]),
                    _container(None, ["GPU-orphan"]),
                    _container("node-c", []),
                ],
            ),
        ],
    )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.orchestration.gpu_placement"
    ):
        placement = store_gpu_placement(
            store, "cluster-a", ["node-a", "node-b"], attempt_id=ATTEMPT
        )

    assert placement == {"node-a": {"GPU-a0", "GPU-a1"}, "node-b": {"GPU-b0"}}
    assert [record.getMessage() for record in caplog.records] == [
        "GPU placement: the GPU inventory snapshot unavailable for node-b",
        "GPU placement: the latest GPU telemetry unavailable for node-b",
    ]


def test_unreadable_observations_and_no_attempt_place_from_the_nodes_only():
    store = _Store(snapshots={"node-a": ["GPU-a0"]}, observations=RuntimeError("x"))

    assert store_gpu_placement(store, "cluster-a", ["node-a"], attempt_id=ATTEMPT) == {
        "node-a": {"GPU-a0"}
    }
    assert store_gpu_placement(store, "cluster-a", ["node-a", "node-z"]) == {
        "node-a": {"GPU-a0"}
    }, "a node with nothing known is left out"


def test_two_failed_nodes_with_store_placement_get_their_own_shares_and_mapping():
    """A drain of both nodes: each node's share, the node-action mapping, and
    the GPU a third node reports is no failed node's."""

    source = fault_incident(
        "inc-three",
        "event-three",
        node_ids=["node-a", "node-b", "node-c"],
        gpu_uuids=["GPU-a0", "GPU-b0", "GPU-c0"],
    )
    workflow = workflow_request(
        "wf-three",
        source.incident_id,
        official_steps=[
            workflow_step(
                WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
                node_ids=["node-a", "node-b", "node-c"],
                gpu_uuids=["GPU-a0", "GPU-b0", "GPU-c0"],
            )
        ],
    )
    store = _Store(
        snapshots={"node-a": ["GPU-a0"], "node-b": ["GPU-b0"], "node-c": ["GPU-c0"]}
    )

    scope = escalation_gpu_scope(workflow, source, ["node-a", "node-b"], store)

    assert scope.gpu_uuids == ["GPU-a0", "GPU-b0"]
    assert scope.gpu_uuids_by_node == {"node-a": ["GPU-a0"], "node-b": ["GPU-b0"]}
    assert scope.validation_gpu_uuids_by_node == scope.gpu_uuids_by_node


def test_a_failed_node_holding_none_of_the_scope_validates_its_full_inventory():
    source = fault_incident(
        "inc-other", "event-other", node_ids=["node-a", "node-b"], gpu_uuids=["GPU-b0"]
    )
    workflow = workflow_request(
        "wf-other",
        source.incident_id,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS, node_ids=["node-a", "node-b"]
            )
        ],
    )
    store = _Store(snapshots={"node-a": ["GPU-a1", "GPU-a0"], "node-b": ["GPU-b0"]})

    scope = escalation_gpu_scope(workflow, source, ["node-a"], store)

    assert scope.gpu_uuids == []
    assert scope.gpu_uuids_by_node == {}
    assert scope.validation_gpu_uuids_by_node == {"node-a": ["GPU-a0", "GPU-a1"]}


def test_only_validation_steps_whose_nodes_are_all_scoped_are_rewritten():
    steps = [
        workflow_step(
            WorkflowOperation.VALIDATE_GPU, node_ids=["node-a"], gpu_uuids=["X"]
        ),
        workflow_step(
            WorkflowOperation.VALIDATE_FABRIC,
            node_ids=["node-a", "node-b"],
            gpu_uuids=["X"],
        ),
        workflow_step(
            WorkflowOperation.RESTART_NODE, node_ids=["node-a"], gpu_uuids=["X"]
        ),
    ]

    scoped = scope_validation_steps(steps, {"node-a": ["GPU-a0"]})

    assert [step.gpu_uuids for step in scoped] == [["GPU-a0"], ["X"], ["X"]]


def test_an_unusable_release_only_inventory_drops_nothing():
    incident = fault_incident(
        "inc-two",
        "event-two",
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a0", "GPU-b0"],
        state=IncidentState.QUARANTINED,
    )

    for broken in (None, "GPU-b0", ["GPU-b0", ""]):
        _, restore = build_validated_restore_workflow(
            incident,
            operator="test-operator",
            reference=None,
            now=datetime.now(timezone.utc),
            node_gpu_uuids={"node-a": ["GPU-a0"], "node-b": broken},
            release_only_node_ids=["node-b"],
        )

        assert restore.official_steps[0].gpu_uuids == ["GPU-a0", "GPU-b0"], (
            broken,
            "an inventory that proves nothing removes nothing",
        )


def test_a_store_without_telemetry_reads_places_nothing():
    """A store that keeps no GPU telemetry (a minimal stub) is no placement
    evidence; it does not fail the escalation either."""

    assert (
        store_gpu_placement(object(), "cluster-a", ["node-a"], attempt_id=ATTEMPT) == {}
    )
