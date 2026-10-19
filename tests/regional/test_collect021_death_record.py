"""COLLECT-021's Pod-death proof survives a superseded attempt observation.

Live on plain EKS (release fdceb5c4d093) the Store held the reading with the
killed container for about two seconds: the passive containment's
STOP_WORKLOADS deleted the Pods one after another, a Completion Watcher pass
between the deletes published the attempt with the surviving Pod only, and the
TIMED_OUT terminal reading later carried ``containers: []``. The case polled for
that one reading and failed ("never went FAILED") while the product had recorded
the death durably in its failure containment and its terminal event.
"""

from __future__ import annotations

import copy
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import Environment
from gpu_fault.watcher import (
    AllocationCompleteness,
    AttemptObservation,
    ContainerObservation,
    FailureDetectedEvent,
    WorkloadPhase,
    failure_containment_ids,
)
from scripts.e2e.regional import collect021_death as death
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CLUSTER = "gpu-eks-cluster"
JOB = "c021-1792439874-2"
ATTEMPT = f"{JOB}-a001"
KILLED_NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER_NODE = "ip-10-0-45-43.us-west-2.compute.internal"
KILLED_POD = "pod-master-0"
KILLED_AFTER = datetime(2026, 10, 19, 19, 59, 8, tzinfo=timezone.utc)
INCIDENT_ID = failure_containment_ids(
    f"{CLUSTER}/{ATTEMPT}/TrainingAttemptFailureDetected"
)[0]


def container(pod_uid: str, rank: int, node: str, **state: Any) -> dict[str, Any]:
    return {
        "pod_uid": pod_uid,
        "pod_name": pod_uid,
        "container_name": "pytorch",
        "role": "worker",
        "rank": rank,
        "node_id": node,
        "terminated": False,
        "exit_code": None,
        "gpu_uuids": [],
        **state,
    }


def observation(phase: str, containers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cluster_id": CLUSTER,
        "job_id": JOB,
        "attempt_id": ATTEMPT,
        "workload_phase": phase,
        "observed_at": "2026-10-19T19:59:15.200000Z",
        "containers": containers,
    }


def running_before_kill() -> dict[str, Any]:
    return observation(
        "RUNNING",
        [
            container(KILLED_POD, 0, KILLED_NODE),
            container("pod-worker-0", 1, PEER_NODE),
        ],
    )


def survivor_only() -> dict[str, Any]:
    """The live reading after the stop deleted the killed Pod first."""

    return observation("RUNNING", [container("pod-worker-0", 1, PEER_NODE)])


def failure_incident(**overrides: Any) -> dict[str, Any]:
    incident = {
        "incident_id": INCIDENT_ID,
        "event_id": f"{CLUSTER}/{ATTEMPT}/TrainingAttemptFailureDetected",
        "event_type": "TRAINING_ATTEMPT_FAILURE_DETECTED",
        "cluster_id": CLUSTER,
        "job_id": JOB,
        "attempt_id": ATTEMPT,
        "node_ids": [KILLED_NODE, PEER_NODE],
        "reasons": [
            "critical container exited non-zero",
            "first_failed_rank=0",
            "exit_code=137",
            "workload log evidence: workload-log/5d68eeb68485d82fdb599852",
        ],
        "created_at": "2026-10-19T19:59:13.948525Z",
        "state": "RECOVERED",
    }
    incident.update(overrides)
    return incident


def judge(state: dict[str, Any], *, rank: int | None = 0) -> tuple[dict, list[str]]:
    return death.death_record(
        state,
        cluster_id=CLUSTER,
        node=KILLED_NODE,
        job_id=JOB,
        attempt_id=ATTEMPT,
        pod_uid=KILLED_POD,
        rank=rank,
        killed_after=KILLED_AFTER,
    )


def test_the_pre_kill_reading_ranks_the_killed_pod_on_its_node() -> None:
    reading = running_before_kill()
    assert death.killed_pod_rank(reading, pod_uid=KILLED_POD, node=KILLED_NODE) == 0
    assert death.killed_pod_rank(reading, pod_uid=KILLED_POD, node=PEER_NODE) is None
    assert death.killed_pod_rank({}, pod_uid=KILLED_POD, node=KILLED_NODE) is None
    reading["containers"][0]["rank"] = True
    assert (
        death.killed_pod_rank(reading, pod_uid=KILLED_POD, node=KILLED_NODE) is None
    ), "only an integer rank binds the Pod"


def test_failure_containment_proves_the_death_the_store_superseded() -> None:
    record, errors = judge(
        {"observations": [survivor_only()], "failure_incident": failure_incident()}
    )
    assert errors == []
    assert record["proven_by"] == "failure_incident"
    assert record["recorded_at"] == "2026-10-19T19:59:13.948525Z"
    assert record["observation"] == survivor_only(), "the Store's reading is kept"
    assert record["killed_rank"] == 0
    assert record["killed_after"] == KILLED_AFTER.isoformat()


def test_the_terminal_paths_containment_names_the_rank_too() -> None:
    record, errors = judge(
        {
            "observations": [observation("FAILED", [])],
            "failure_incident": failure_incident(
                reasons=[
                    "terminal TIMED_OUT reported before failure detection",
                    "rank 0 exit_code=137",
                ]
            ),
        }
    )
    assert errors == [] and record["proven_by"] == "failure_incident"


def test_an_observation_with_the_killed_container_still_proves_the_death() -> None:
    reading = running_before_kill()
    reading["containers"][0].update(terminated=True, exit_code=137)
    record, errors = judge({"observations": [reading], "failure_incident": None})
    assert errors == []
    assert record["proven_by"] == "observation"
    assert record["recorded_at"] == reading["observed_at"]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"incident_id": "inc-other"}, "identity"),
        ({"job_id": "another-job"}, "identity"),
        ({"event_type": "XID"}, "identity"),
        ({"node_ids": [PEER_NODE]}, "node"),
        (
            {
                "reasons": [
                    "critical container exited non-zero",
                    "first_failed_rank=1",
                    "exit_code=137",
                ]
            },
            "rank 0",
        ),
        (
            {
                "reasons": [
                    "critical container exited non-zero",
                    "first_failed_rank=0",
                    "exit_code=0",
                ]
            },
            "non-zero",
        ),
        (
            {"reasons": ["workload adapter reported FAILED", "first_failed_rank=0"]},
            "rank 0",
        ),
        ({"created_at": "2026-10-19T19:50:00Z"}, "predates the kill"),
        ({"created_at": None}, "predates the kill"),
    ],
)
def test_each_failure_containment_reading_is_checked(
    change: dict[str, Any], fragment: str
) -> None:
    record, errors = judge(
        {
            "observations": [survivor_only()],
            "failure_incident": failure_incident(**change),
        }
    )
    assert record["proven_by"] is None
    assert any(fragment in item for item in errors), (fragment, errors)
    assert any("killed Pod" in item for item in errors), (
        "the reasons the Store's reading does not prove the death are kept too"
    )


def test_no_containment_and_no_ranked_pod_prove_nothing() -> None:
    _record, errors = judge({"observations": [survivor_only()]})
    assert any("no failure containment" in item for item in errors), errors
    _record, errors = judge(
        {"observations": [], "failure_incident": failure_incident()}, rank=None
    )
    assert any("did not rank the killed Pod" in item for item in errors), errors
    assert any("no observation" in item for item in errors), errors


def test_a_record_is_rejudged_and_must_precede_the_late_xid() -> None:
    record, errors = judge(
        {"observations": [survivor_only()], "failure_incident": failure_incident()}
    )
    assert errors == []
    arguments = {
        "node": KILLED_NODE,
        "job_id": JOB,
        "attempt_id": ATTEMPT,
        "pod_uid": KILLED_POD,
    }
    xid = datetime(2026, 10, 19, 19, 59, 30, tzinfo=timezone.utc)
    assert death.death_record_errors(record, xid_injected_at=xid, **arguments) == []
    assert (
        death.death_record_errors(
            record,
            job_id=None,
            **{key: value for key, value in arguments.items() if key != "job_id"},
        )
        == []
    ), "the record carries its own job identity"
    early = xid - timedelta(seconds=30)
    assert death.death_record_errors(record, xid_injected_at=early, **arguments) == [
        "the Pod death was not recorded before the late XID"
    ]
    forged = copy.deepcopy(record)
    forged["killed_rank"] = 1
    assert any(
        "rank 1" in item for item in death.death_record_errors(forged, **arguments)
    ), "a re-judged record is held to its own rank"
    assert death.death_record_errors(None, **arguments), "no record proves nothing"


def test_the_terminal_event_names_the_killed_ranks_exit() -> None:
    event = {
        "rank_exit_status": [
            {"rank": 0, "node_id": KILLED_NODE, "exit_code": 137, "signal": None}
        ]
    }
    assert death.terminal_death_errors(event, rank=0, node=KILLED_NODE) == []
    assert death.terminal_death_errors(event, rank=None, node=None) == []
    zero = {"rank_exit_status": [{"rank": 0, "node_id": KILLED_NODE, "exit_code": 0}]}
    for reading, rank, node in [
        (event, 1, KILLED_NODE),
        (event, 0, PEER_NODE),
        ({"rank_exit_status": []}, 0, None),
        (zero, 0, KILLED_NODE),
        (None, 0, KILLED_NODE),
    ]:
        assert death.terminal_death_errors(reading, rank=rank, node=node) == [
            "terminal event does not record the killed rank's non-zero exit"
        ], (reading, rank, node)


class Regional:
    """The CPU-side probe answers a scripted sequence of Store readings."""

    def __init__(self, readings: list[dict[str, Any]]) -> None:
        self.settings = type("Settings", (), {"cluster_id": CLUSTER})()
        self.readings = readings
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        self.calls.append((script, args))
        return copy.deepcopy(
            self.readings[min(len(self.calls), len(self.readings)) - 1]
        )


def wait(regional: Regional, **overrides: Any) -> dict[str, Any]:
    arguments = {
        "node": KILLED_NODE,
        "job_id": JOB,
        "attempt_id": ATTEMPT,
        "pod_uid": KILLED_POD,
        "rank": 0,
        "killed_after": KILLED_AFTER,
        "timeout_seconds": 60,
        "poll_seconds": 0,
        **overrides,
    }
    return death.wait_death_record(regional, **arguments)  # type: ignore[arg-type]


def test_the_wait_returns_once_the_containment_exists() -> None:
    """The live sequence: the dying reading was already superseded at the
    first poll, and the containment proves the death on the next one."""

    regional = Regional(
        [
            {"observations": [survivor_only()], "failure_incident": None},
            {"observations": [survivor_only()], "failure_incident": failure_incident()},
        ]
    )
    record = wait(regional)
    assert record["proven_by"] == "failure_incident"
    assert len(regional.calls) == 2
    assert all(
        script == death.DEATH_PROBE and args == (CLUSTER, JOB, ATTEMPT)
        for script, args in regional.calls
    ), regional.calls


def test_the_wait_names_why_the_death_is_unproven() -> None:
    regional = Regional(
        [{"observations": [observation("FAILED", [])], "failure_incident": None}]
    )
    with pytest.raises(RegionalFixtureError, match="never recorded") as raised:
        wait(regional, timeout_seconds=0)
    assert "no failure containment" in str(raised.value)
    assert "killed Pod" in str(raised.value)


def run_probe(
    monkeypatch: pytest.MonkeyPatch, context: ApplicationContext
) -> dict[str, Any]:
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=context.store)),
    )
    monkeypatch.setattr(sys, "argv", ["probe", CLUSTER, JOB, ATTEMPT])
    output = io.StringIO()
    with redirect_stdout(output):
        exec(death.DEATH_PROBE, {"__name__": "__main__"})
    return json.loads(output.getvalue())


def stored_observation(
    attempt_id: str, *containers: ContainerObservation
) -> AttemptObservation:
    return AttemptObservation(
        cluster_id=CLUSTER,
        environment=Environment.EKS,
        job_id=JOB,
        attempt_id=attempt_id,
        workload_phase=WorkloadPhase.RUNNING,
        observed_at=datetime.now(timezone.utc),
        expected_critical_ranks=2,
        runtime_profile_version="simulated-v1",
        containers=list(containers),
    )


def test_the_probe_reads_the_products_own_containment_and_reading(
    monkeypatch: pytest.MonkeyPatch, context: ApplicationContext
) -> None:
    """The emitted probe runs against the real Store API, and the containment
    the real failure-detected handler opens is what the record parses."""

    killed_after = datetime.now(timezone.utc) - timedelta(seconds=1)
    survivor = ContainerObservation(
        pod_uid="pod-worker-0",
        pod_name="pod-worker-0",
        container_name="pytorch",
        role="worker",
        rank=1,
        node_id=PEER_NODE,
    )
    context.store.save_attempt_observation(stored_observation(ATTEMPT, survivor))
    context.store.save_attempt_observation(
        stored_observation(f"{ATTEMPT}-r-1", survivor)
    )
    assert run_probe(monkeypatch, context)["failure_incident"] is None, (
        "a containment not yet opened is a pending proof, not an error"
    )
    context.completion.handle_failure_detected(
        FailureDetectedEvent(
            cluster_id=CLUSTER,
            job_id=JOB,
            attempt_id=ATTEMPT,
            detected_at=datetime.now(timezone.utc),
            runtime_profile_version="simulated-v1",
            workload_ids=[f"gpu-fault-system/pytorchjob/gpu-fault-{JOB}"],
            node_ids=[KILLED_NODE, PEER_NODE],
            first_failed_rank=0,
            node_id=KILLED_NODE,
            exit_code=137,
            reason="critical container exited non-zero",
            allocation_completeness=AllocationCompleteness.INCOMPLETE,
        )
    )
    state = run_probe(monkeypatch, context)
    assert [item["attempt_id"] for item in state["observations"]] == [ATTEMPT], (
        "only the killed attempt's reading is returned"
    )
    record, errors = death.death_record(
        state,
        cluster_id=CLUSTER,
        node=KILLED_NODE,
        job_id=JOB,
        attempt_id=ATTEMPT,
        pod_uid=KILLED_POD,
        rank=0,
        killed_after=killed_after,
    )
    assert errors == [], errors
    assert record["proven_by"] == "failure_incident"
    assert record["failure_incident"]["incident_id"] == INCIDENT_ID
