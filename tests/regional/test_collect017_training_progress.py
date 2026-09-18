"""Live training proof uses advancing collectives, not merely stable Pod UIDs."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import collect017_training as training


def snapshot(step: int = 10):
    return {
        "pods": [
            {
                "name": f"trainer-{index}",
                "uid": f"pod-{index}",
                "node": f"node-{index}",
                "attempt_id": "attempt-a",
                "phase": "Running",
                "ready": True,
            }
            for index in range(3)
        ],
        "heartbeat_logs": {
            f"trainer-{index}": (
                f"HEARTBEAT rank={index * 8} step={step} all_reduce=300"
            )
            for index in range(3)
        },
    }


class Workload:
    settings = SimpleNamespace(
        attempt_id="attempt-a", expected_gpu_count=24, expected_pods=3
    )

    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.last = snapshot()
        self.reads = 0

    def snapshot(self):
        self.reads += 1
        self.last = next(self.snapshots, self.last)
        return self.last


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        training,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ),
    )
    return now


def test_unchanged_samples_wait_until_every_original_pod_advances(clock):
    workload = Workload([snapshot(10), snapshot(11)])
    result = training.wait_training_progress(workload, snapshot(10))
    assert workload.reads == 2, "a repeated heartbeat is not evidence of progress"
    proof = result["progress_evidence"]
    assert set(proof["before"]) == set(proof["after"]), "all original Pods are proved"
    assert all(item["step"] == 11 for item in proof["after"].values()), (
        "the accepted proof records the actual post-baseline collective step"
    )


def test_stalled_collectives_cannot_pass_with_unchanged_uids(clock):
    workload = Workload([snapshot(10)])
    with pytest.raises(RuntimeError, match="did not advance"):
        training.wait_training_progress(workload, snapshot(10), timeout_seconds=4)
    assert workload.reads == 2, "the wait must respect its finite deadline"


@pytest.mark.parametrize(
    "defect",
    [
        "missing-pod",
        "duplicate-uid",
        "missing-name",
        "new-uid",
        "new-name",
        "new-node",
        "new-attempt",
        "not-running",
        "unready",
        "no-node",
        "no-logs",
        "no-step",
        "wrong-world",
        "regressed-step",
    ],
)
def test_identity_and_collective_failures_stop_immediately(clock, defect):
    current = snapshot(11)
    pod = current["pods"][0]
    if defect == "missing-pod":
        current["pods"].pop()
    elif defect == "duplicate-uid":
        pod["uid"] = current["pods"][1]["uid"]
    elif defect == "missing-name":
        pod["name"] = ""
    elif defect in {"new-uid", "new-name", "new-node"}:
        field = defect.removeprefix("new-")
        pod[field] = "replacement"
        if field == "name":
            current["heartbeat_logs"]["replacement"] = current["heartbeat_logs"][
                "trainer-0"
            ]
    elif defect == "new-attempt":
        pod["attempt_id"] = "attempt-b"
    elif defect == "not-running":
        pod["phase"] = "Succeeded"
    elif defect == "unready":
        pod["ready"] = False
    elif defect == "no-node":
        pod["node"] = None
    elif defect == "no-logs":
        current["heartbeat_logs"] = {}
    else:
        current["heartbeat_logs"]["trainer-0"] = {
            "no-step": "HEARTBEAT all_reduce=300",
            "wrong-world": "HEARTBEAT step=11 all_reduce=36",
            "regressed-step": "HEARTBEAT step=9 all_reduce=300",
        }[defect]
    workload = Workload([current])
    with pytest.raises(RuntimeError):
        training.wait_training_progress(workload, snapshot(10))
    assert workload.reads == 1, "identity loss cannot be hidden by a later reread"


def test_snapshot_that_returns_after_the_deadline_is_not_accepted(clock):
    class LateWorkload(Workload):
        def snapshot(self):
            clock[0] = 5
            return snapshot(11)

    with pytest.raises(RuntimeError, match="did not advance"):
        training.wait_training_progress(
            LateWorkload([]), snapshot(10), timeout_seconds=4
        )


def test_incident_must_bind_the_same_cluster_job_attempt_and_planned_workflow():
    state = {
        "incident": {
            "incident_id": "incident",
            "cluster_id": "cluster-a",
            "job_id": "job",
            "attempt_id": "attempt-a",
            "node_ids": ["node-a"],
        },
        "workflow": {"request_id": "workflow", "incident_id": "incident"},
    }
    options = {
        "cluster_id": "cluster-a",
        "node": "node-a",
        "job_id": "job",
        "attempt_id": "attempt-a",
        "workflow_id": "workflow",
    }
    assert training.incident_attempt_errors(state, **options) == [], (
        "the current planned workflow and managed attempt are valid"
    )
    for collection, field in (
        ("incident", "incident_id"),
        ("incident", "cluster_id"),
        ("incident", "job_id"),
        ("incident", "attempt_id"),
        ("incident", "node_ids"),
        ("workflow", "request_id"),
    ):
        changed = deepcopy(state)
        changed[collection][field] = [] if field == "node_ids" else "other"
        assert training.incident_attempt_errors(changed, **options), (
            f"mismatched {collection}/{field} cannot prove the managed scenario"
        )
    assert training.incident_attempt_errors(state, **{**options, "workflow_id": ""}), (
        "a missing planned workflow identity cannot bind a later node workflow"
    )
