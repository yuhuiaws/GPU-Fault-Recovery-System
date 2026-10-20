"""A rank that died keeps the GPUs it was read with while it ran (batch 146).

The Completion Watcher learns a training container's GPUs from the
``gpu-fault.io/gpu-uuids`` annotation or by exec'ing ``nvidia-smi`` in the
running container, and an exited container cannot be exec'd. Live
(GF-REGIONAL-COLLECT-021 on plain EKS and on HyperPod) the reading of the rank
that was killed therefore carried no GPU at all: the attempt-failure incident
named only the survivors' GPUs (8 of 16 on plain EKS, 16 of 24 on HyperPod) and
the TIMED_OUT terminal allocation listed rank 0 with ``gpu_uuids: []`` -- the
information lost was exactly the GPUs of the rank that failed.

The Pods below have the shape the submit CLI and the PyTorchJob operator give
them (rank from ``rank-offset`` + replica index, ``nvidia.com/gpu: 8``, no GPU
annotation), and the exec double answers per Pod UID, so a UUID that shows up
under another Pod would be visible.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_outbox import KubernetesCompletionOutbox
from gpu_fault.models import Environment, TerminalEvent, WorkflowOperation
from gpu_fault.passive import PassiveWorkflowCompiler
from gpu_fault.service import CompletionService
from gpu_fault.watcher import FailureDetectedEvent, failure_containment_ids
from tests.completion._support import NOW, Clock, FakeCoreApi, FakeSink

CLUSTER = "gpu-cluster"
JOB = "c021-1792454434-3"
ATTEMPT = f"{JOB}-a001"
NAMESPACE = "gpu-fault-system"
PROFILE = "simulated-v1"
FAILURE_PATH = "/v1/attempts/failure-detected"
TERMINAL_PATH = "/v1/attempts/terminal"
OBSERVATION_PATH = "/v1/workload-observations"
PASSIVE_INCIDENT, _ = failure_containment_ids(
    f"{CLUSTER}/{ATTEMPT}/TrainingAttemptFailureDetected"
)
PROVIDERS = [
    pytest.param(
        Environment.EKS,
        (
            "ip-10-0-45-43.us-west-2.compute.internal",
            "ip-10-0-43-5.us-west-2.compute.internal",
        ),
        id="plain-eks",
    ),
    pytest.param(
        Environment.HYPERPOD_EKS,
        ("hyperpod-i-00000000000000001", "hyperpod-i-00000000000000002"),
        id="hyperpod",
    ),
]


def gpus(pod_uid: str) -> list[str]:
    """What ``nvidia-smi --query-gpu=uuid`` prints inside that Pod's container."""

    return [f"GPU-{pod_uid}-{index}" for index in range(8)]


def uid(rank: int) -> str:
    return f"uid-{'master-0' if rank == 0 else f'worker-{rank - 1}'}"


def training_pod(
    rank: int,
    node: str,
    *,
    pod_uid: str | None = None,
    exit_code: int | None = None,
    initiator: str | None = None,
) -> dict[str, Any]:
    """One rank of a managed two-rank PyTorchJob (master-0 and worker-0)."""

    role = "master" if rank == 0 else "worker"
    index = 0 if rank == 0 else rank - 1
    annotations = {
        "gpu-fault.io/expected-critical-ranks": "2",
        "gpu-fault.io/training-container": "pytorch",
        "gpu-fault.io/runtime-profile-version": PROFILE,
        "gpu-fault.io/restart-budget": "1",
        "gpu-fault.io/rank-offset": "0" if rank == 0 else "1",
    }
    if initiator:
        annotations["gpu-fault.io/termination-initiator-incident-id"] = initiator
    if exit_code is None:
        phase, state = "Running", {"running": {"startedAt": NOW.isoformat()}}
    else:
        phase = "Failed"
        state = {"terminated": {"exitCode": exit_code, "finishedAt": NOW.isoformat()}}
    return {
        "metadata": {
            "name": f"gpu-fault-{JOB}-{role}-{index}",
            "namespace": NAMESPACE,
            "uid": pod_uid or uid(rank),
            "labels": {
                "gpu-fault.io/managed": "true",
                "gpu-fault.io/job-id": JOB,
                "gpu-fault.io/attempt-id": ATTEMPT,
                "gpu-fault.io/role": role,
                "gpu-fault.io/critical": "true",
                "training.kubeflow.org/job-name": f"gpu-fault-{JOB}",
                "training.kubeflow.org/replica-type": role,
                "training.kubeflow.org/replica-index": str(index),
            },
            "annotations": annotations,
        },
        "spec": {
            "nodeName": node,
            "containers": [
                {
                    "name": "pytorch",
                    "resources": {
                        "requests": {"nvidia.com/gpu": "8"},
                        "limits": {"nvidia.com/gpu": "8"},
                    },
                }
            ],
        },
        "status": {
            "phase": phase,
            "startTime": NOW.isoformat(),
            "containerStatuses": [
                {"name": "pytorch", "state": state, "restartCount": 0}
            ],
        },
    }


class Exec:
    """The ``nvidia-smi`` exec: answers per Pod UID, fails for ``failing``."""

    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.failing = failing
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, pod: dict[str, Any], container_name: str) -> list[str]:
        pod_uid = pod["metadata"]["uid"]
        state = pod["status"]["containerStatuses"][0]["state"]
        self.calls.append((pod_uid, "running" in state and container_name == "pytorch"))
        if pod_uid in self.failing:
            raise RuntimeError("error dialing backend: container exec timed out")
        return gpus(pod_uid)


def watcher(
    core: FakeCoreApi, sink: Any, environment: Environment, clock: Clock, resolver: Exec
) -> KubernetesCompletionController:
    return KubernetesCompletionController(
        core,
        sink,
        cluster_id=CLUSTER,
        environment=environment,
        cleanup_timeout_seconds=30,
        now=clock,
        publish_observations=True,
        gpu_uuid_resolver=resolver,
    )


def posted(sink: FakeSink, path: str) -> list[dict[str, Any]]:
    return [payload for item, payload in sink.posts if item == path]


def rank_gpus(containers: list[dict[str, Any]]) -> dict[int, list[str]]:
    return {item["rank"]: sorted(item["gpu_uuids"]) for item in containers}


def control_plane() -> tuple[ApplicationContext, CompletionService]:
    """The production completion wiring: plans compile into workflows."""

    context = ApplicationContext()
    return context, CompletionService(
        context.store, workflow_compiler=PassiveWorkflowCompiler(context.store)
    )


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_dead_rank_keeps_its_gpus_through_containment_and_terminal(
    environment: Environment, nodes: tuple[str, str]
) -> None:
    """The live COLLECT-021 sequence: rank 0 is SIGKILLed (exit 137), the
    passive containment annotates both Pods and deletes them with zero grace,
    the dead rank's first, and the attempt ends TIMED_OUT at the cleanup
    deadline. Every record of the failure names all 16 GPUs."""

    clock = Clock()
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    sink = FakeSink()
    resolver = Exec()
    subject = watcher(core, sink, environment, clock, resolver)

    subject.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [training_pod(0, nodes[0], exit_code=137), training_pod(1, nodes[1])]
    subject.run_once()
    clock.value += timedelta(seconds=1)
    core.pods = [
        training_pod(0, nodes[0], exit_code=137, initiator=PASSIVE_INCIDENT),
        training_pod(1, nodes[1], initiator=PASSIVE_INCIDENT),
    ]
    subject.run_once()
    clock.value += timedelta(seconds=1)
    core.pods = [training_pod(1, nodes[1], initiator=PASSIVE_INCIDENT)]
    subject.run_once()
    clock.value += timedelta(seconds=30)
    core.pods = []
    subject.run_once()

    expected = {0: gpus(uid(0)), 1: gpus(uid(1))}
    every_gpu = sorted(expected[0] + expected[1])
    assert resolver.calls == [(uid(0), True), (uid(1), True)], (
        f"one exec per running container and none into the exited one: {resolver.calls}"
    )
    [failure] = posted(sink, FAILURE_PATH)
    assert (failure["first_failed_rank"], failure["exit_code"]) == (0, 137), failure
    assert failure["gpu_uuids"] == every_gpu, (
        f"the failure event lost the dead rank's GPUs: {failure['gpu_uuids']}"
    )
    assert failure["allocation_completeness"] == "COMPLETE", failure
    dead_readings = [
        item
        for reading in posted(sink, OBSERVATION_PATH)
        for item in reading["containers"]
        if item["rank"] == 0 and item["terminated"]
    ]
    assert len(dead_readings) == 4, "the FAILED, stop, partial and terminal readings"
    assert all(sorted(item["gpu_uuids"]) == expected[0] for item in dead_readings), (
        f"a reading of the exited rank dropped its GPUs: {dead_readings}"
    )
    [terminal] = posted(sink, TERMINAL_PATH)
    assert terminal["terminal_status"] == "TIMED_OUT", terminal
    assert rank_gpus(terminal["allocation"]) == expected, terminal["allocation"]
    assert sum(item["gpu_count"] for item in terminal["allocation"]) == 16

    context, completion = control_plane()
    containment = completion.handle_failure_detected(
        FailureDetectedEvent.model_validate(failure)
    )
    incident = context.store.get_incident(containment.incident_id)
    assert incident.incident_id == PASSIVE_INCIDENT, incident.incident_id
    assert incident.gpu_uuids == every_gpu, (
        f"the attempt-failure incident names {len(incident.gpu_uuids)} of 16 GPUs"
    )
    workflow = context.store.get_workflow(containment.workflow_request_id)
    assert [(step.operation, step.gpu_uuids) for step in workflow.official_steps] == [
        (WorkflowOperation.FREEZE_EVIDENCE, every_gpu),
        (WorkflowOperation.STOP_WORKLOADS, every_gpu),
    ], workflow.official_steps
    decision = completion.handle_terminal(TerminalEvent.model_validate(terminal))
    plan = context.store.get_plan(str(decision.recovery_plan_id))
    recovery = context.store.get_incident(plan.incident_id)
    assert recovery.event_type == "TRAINING_ATTEMPT_TERMINAL", recovery.event_type
    assert recovery.gpu_uuids == every_gpu, (
        f"the recovery incident names {len(recovery.gpu_uuids)} of 16 GPUs"
    )
    restart = context.store.get_workflow(str(plan.workflow_request_id))
    [source_count] = [
        step.parameters["source_gpu_count"]
        for step in restart.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    ]
    assert source_count == 16, restart.official_steps


@pytest.mark.parametrize("exec_failed", [False, True], ids=["dead-first", "no-exec"])
def test_a_rank_never_read_with_gpus_keeps_none(exec_failed: bool) -> None:
    """Nothing is invented: a rank that died before the watcher ever saw it
    running, or whose exec never answered while it ran, has no known GPU. Its
    allocation stays visibly incomplete -- the declared count without UUIDs."""

    clock = Clock()
    nodes = ("node-a", "node-b")
    dead_first = training_pod(0, nodes[0], exit_code=137)
    core = FakeCoreApi(
        [training_pod(0, nodes[0]) if exec_failed else dead_first]
        + [training_pod(1, nodes[1])]
    )
    sink = FakeSink()
    resolver = Exec(failing=frozenset({uid(0)}) if exec_failed else frozenset())
    subject = watcher(core, sink, Environment.EKS, clock, resolver)

    subject.run_once()
    if exec_failed:
        clock.value += timedelta(seconds=5)
        core.pods = [dead_first, training_pod(1, nodes[1])]
        subject.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [dead_first, training_pod(1, nodes[1], exit_code=137)]
    subject.run_once()

    [failure] = posted(sink, FAILURE_PATH)
    assert failure["gpu_uuids"] == gpus(uid(1)), (
        f"only the survivor's GPUs are known: {failure['gpu_uuids']}"
    )
    assert failure["allocation_completeness"] == "INCOMPLETE", failure
    [terminal] = posted(sink, TERMINAL_PATH)
    assert terminal["terminal_status"] == "FAILED", terminal
    assert {
        item["rank"]: (item["gpu_count"], item["gpu_uuids"])
        for item in terminal["allocation"]
    } == {0: (8, []), 1: (8, gpus(uid(1)))}, terminal["allocation"]


@pytest.mark.parametrize("replacement_runs", [False, True], ids=["exited", "running"])
def test_a_replacement_pod_is_read_fresh(replacement_runs: bool) -> None:
    """The job controller's replacement for rank 0 has a new UID: it is read
    from its own Pod, never handed the GPUs of the Pod it replaced."""

    clock = Clock()
    core = FakeCoreApi([training_pod(0, "node-a"), training_pod(1, "node-b")])
    sink = FakeSink()
    resolver = Exec()
    subject = watcher(core, sink, Environment.HYPERPOD_EKS, clock, resolver)

    subject.run_once()
    clock.value += timedelta(seconds=5)
    replacement = training_pod(
        0,
        "node-a",
        pod_uid="uid-master-0-replacement",
        exit_code=None if replacement_runs else 1,
    )
    core.pods = [replacement, training_pod(1, "node-b")]
    subject.run_once()

    latest = posted(sink, OBSERVATION_PATH)[-1]
    rank0 = [item for item in latest["containers"] if item["rank"] == 0]
    assert [item["pod_uid"] for item in rank0] == ["uid-master-0-replacement"], rank0
    assert rank0[0]["gpu_uuids"] == (
        gpus("uid-master-0-replacement") if replacement_runs else []
    ), f"the replacement was handed its predecessor's GPUs: {rank0}"
    if not replacement_runs:
        [failure] = posted(sink, FAILURE_PATH)
        assert failure["gpu_uuids"] == gpus(uid(1)), failure["gpu_uuids"]
        assert failure["allocation_completeness"] == "INCOMPLETE", failure


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_rank_that_died_while_the_watcher_restarted_keeps_its_gpus(
    environment: Environment, nodes: tuple[str, str]
) -> None:
    """The persisted attempt record of a running attempt keeps each Pod's GPU
    UUIDs. A rank that died while the watcher was down is read exited by the new
    process, which has never exec'd anything, and still keeps its GPUs."""

    clock = Clock()
    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    delivered = FakeSink()
    first = watcher(
        core, KubernetesCompletionOutbox(core, delivered), environment, clock, Exec()
    )
    first.run_once()
    assert json.loads(core.config_map_data["active-attempts.json"]), (
        "the running attempt was not persisted"
    )

    clock.value += timedelta(seconds=20)
    core.pods = [training_pod(0, nodes[0], exit_code=137), training_pod(1, nodes[1])]
    resolver = Exec()
    restarted = watcher(
        core, KubernetesCompletionOutbox(core, delivered), environment, clock, resolver
    )
    restarted.run_once()

    assert resolver.calls == [(uid(1), True)], resolver.calls
    [failure] = posted(delivered, FAILURE_PATH)
    assert failure["gpu_uuids"] == sorted(gpus(uid(0)) + gpus(uid(1))), (
        f"the restart lost the dead rank's GPUs: {failure['gpu_uuids']}"
    )
    assert failure["allocation_completeness"] == "COMPLETE", failure


def test_a_running_rank_keeps_its_restored_gpus_while_exec_fails() -> None:
    """The same rule for a container that is still running: after a restart
    the in-memory exec cache is empty, and a hung ``nvidia-smi`` -- what a GPU
    fault does -- left the reading without GPUs until an exec answered, so a
    fault naming one of those GPUs matched no container."""

    clock = Clock()
    core = FakeCoreApi([training_pod(0, "node-a"), training_pod(1, "node-b")])
    delivered = FakeSink()
    first = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        Environment.EKS,
        clock,
        Exec(),
    )
    first.run_once()

    clock.value += timedelta(seconds=20)
    resolver = Exec(failing=frozenset({uid(0), uid(1)}))
    restarted = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        Environment.EKS,
        clock,
        resolver,
    )
    restarted.run_once()

    assert [item for item, _ in resolver.calls] == [uid(0), uid(1)], resolver.calls
    latest = posted(delivered, OBSERVATION_PATH)[-1]
    assert latest["workload_phase"] == "RUNNING", latest
    assert rank_gpus(latest["containers"]) == {0: gpus(uid(0)), 1: gpus(uid(1))}, (
        latest["containers"]
    )


def test_a_restart_after_the_failed_reading_reads_the_dead_rank_fresh() -> None:
    """The limit of the restart support: a FAILED reading is not an active one,
    so publishing it drops the persisted record. A watcher that restarts after
    that and still finds the exited Pod knows nothing of its GPUs; it reports
    them missing (INCOMPLETE) instead of guessing. The containment incident is
    the first process's event, delivered with every GPU."""

    clock = Clock()
    core = FakeCoreApi([training_pod(0, "node-a"), training_pod(1, "node-b")])
    delivered = FakeSink()
    first = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        Environment.EKS,
        clock,
        Exec(),
    )
    first.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [training_pod(0, "node-a", exit_code=137), training_pod(1, "node-b")]
    first.run_once()
    assert json.loads(core.config_map_data["active-attempts.json"]) == {}, (
        "the FAILED reading should have dropped the persisted record"
    )

    clock.value += timedelta(seconds=5)
    restarted = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        Environment.EKS,
        clock,
        Exec(),
    )
    restarted.run_once()

    first_failure, second_failure = posted(delivered, FAILURE_PATH)
    assert first_failure["gpu_uuids"] == sorted(gpus(uid(0)) + gpus(uid(1)))
    assert second_failure["gpu_uuids"] == gpus(uid(1)), second_failure["gpu_uuids"]
    assert second_failure["allocation_completeness"] == "INCOMPLETE", second_failure
