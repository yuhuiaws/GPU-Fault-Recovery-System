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

Batch 154 closes the window batch 146 left open: publishing the FAILED reading
used to drop the persisted record, so a watcher restarted between that reading
and the terminal one read the dead rank without GPUs -- or, when the stop had
already deleted the Pods, never reported the attempt's terminal at all. The
record of the last active reading is now kept until the terminal reading.
"""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
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
PASSIVE_INCIDENT, PASSIVE_WORKFLOW = failure_containment_ids(
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
    core: FakeCoreApi,
    sink: Any,
    environment: Environment,
    clock: Clock,
    resolver: Exec,
    *,
    stopper: Any = None,
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
        workload_stopper=stopper,
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


def persisted(core: FakeCoreApi) -> dict[str, Any]:
    """The attempt's persisted record: its phase and each rank's GPU UUIDs."""

    records = json.loads(core.config_map_data["active-attempts.json"])
    if not records:
        return {}
    [record] = records.values()
    return {
        "phase": record["workload_phase"],
        "gpus": {pod["rank"]: sorted(pod.get("uuids") or []) for pod in record["pods"]},
    }


def control_plane_verdict(
    failures: list[dict[str, Any]], terminal: dict[str, Any]
) -> tuple[list[str], list[str], int]:
    """Feed the delivered events to the control plane in order: the containment
    incident's GPUs, the recovery incident's GPUs and the restart's
    ``source_gpu_count``. A re-sent failure event is the same incident."""

    context, completion = control_plane()
    for failure in failures:
        containment = completion.handle_failure_detected(
            FailureDetectedEvent.model_validate(failure)
        )
        assert containment.incident_id == PASSIVE_INCIDENT, containment
    decision = completion.handle_terminal(TerminalEvent.model_validate(terminal))
    plan = context.store.get_plan(str(decision.recovery_plan_id))
    recovery = context.store.get_incident(plan.incident_id)
    assert recovery.event_type == "TRAINING_ATTEMPT_TERMINAL", recovery.event_type
    restart = context.store.get_workflow(str(plan.workflow_request_id))
    [source_count] = [
        step.parameters["source_gpu_count"]
        for step in restart.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    ]
    return (
        context.store.get_incident(PASSIVE_INCIDENT).gpu_uuids,
        recovery.gpu_uuids,
        source_count,
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


class SuspendedPyTorchJob:
    """The PyTorchJob as the containment's STOP_WORKLOADS leaves it: annotated
    with the initiator and its step key, and suspended."""

    def get_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, name: str, **_: Any
    ) -> dict[str, Any]:
        assert (group, plural, namespace, name) == (
            "kubeflow.org",
            "pytorchjobs",
            NAMESPACE,
            f"gpu-fault-{JOB}",
        ), (group, version, namespace, plural, name)
        return {
            "metadata": {
                "name": name,
                "namespace": namespace,
                "annotations": {
                    "gpu-fault.io/termination-initiator-incident-id": PASSIVE_INCIDENT,
                    "gpu-fault.io/operation-id": f"{PASSIVE_WORKFLOW}/1/STOP_WORKLOADS",
                },
            },
            "spec": {"runPolicy": {"suspend": True}},
        }


class StopNotYetReadable(SuspendedPyTorchJob):
    """The first read of the PyTorchJob times out; later reads see our stop."""

    def __init__(self) -> None:
        self.reads = 0

    def get_namespaced_custom_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.reads += 1
        if self.reads == 1:
            raise TimeoutError("read timed out")
        return super().get_namespaced_custom_object(*args, **kwargs)


class RefusesTheFirstFailedReading(FakeSink):
    """A control plane that refuses one FAILED workload observation, once."""

    def __init__(self) -> None:
        super().__init__()
        self.refused = False

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            path == OBSERVATION_PATH
            and payload["workload_phase"] == "FAILED"
            and not self.refused
        ):
            self.refused = True
            raise RuntimeError("control plane unavailable")
        return super().post(path, payload)


def failed_attempt(
    environment: Environment,
    nodes: tuple[str, str],
    *,
    delivered: FakeSink,
    clock: Clock,
    resolver: Exec | None = None,
) -> tuple[FakeCoreApi, KubernetesCompletionController]:
    """A first watcher process that saw both ranks run and then published the
    FAILED reading of rank 0's SIGKILL (exit 137)."""

    core = FakeCoreApi([training_pod(0, nodes[0]), training_pod(1, nodes[1])])
    first = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        environment,
        clock,
        resolver or Exec(),
    )
    first.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [training_pod(0, nodes[0], exit_code=137), training_pod(1, nodes[1])]
    first.run_once()
    return core, first


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_restart_after_the_failed_reading_keeps_the_dead_ranks_gpus(
    environment: Environment, nodes: tuple[str, str]
) -> None:
    """The window batch 146 left open: the watcher restarts after it published
    the FAILED reading and before the terminal one, while the containment is
    still stopping the Pods. The record of the last RUNNING reading is kept
    until the terminal reading, so the restarted process -- which never exec'd
    the dead rank -- still knows its GPUs: the failure event it sends again,
    the TIMED_OUT terminal allocation and the recovery incident compiled from
    it name all 16. Publishing the terminal reading drops the record."""

    clock = Clock()
    delivered = FakeSink()
    core, _first = failed_attempt(environment, nodes, delivered=delivered, clock=clock)
    after_failed_reading = persisted(core)

    clock.value += timedelta(seconds=5)
    resolver = Exec()
    restarted = watcher(
        core, KubernetesCompletionOutbox(core, delivered), environment, clock, resolver
    )
    restarted.run_once()
    clock.value += timedelta(seconds=1)
    core.pods = [
        training_pod(0, nodes[0], exit_code=137, initiator=PASSIVE_INCIDENT),
        training_pod(1, nodes[1], initiator=PASSIVE_INCIDENT),
    ]
    restarted.run_once()
    clock.value += timedelta(seconds=1)
    core.pods = [training_pod(1, nodes[1], initiator=PASSIVE_INCIDENT)]
    restarted.run_once()
    clock.value += timedelta(seconds=30)
    core.pods = []
    restarted.run_once()

    expected = {0: gpus(uid(0)), 1: gpus(uid(1))}
    every_gpu = sorted(expected[0] + expected[1])
    assert resolver.calls == [(uid(1), True)], (
        f"the restarted watcher execs the running survivor only: {resolver.calls}"
    )
    first_failure, second_failure = posted(delivered, FAILURE_PATH)
    assert first_failure["gpu_uuids"] == every_gpu, first_failure["gpu_uuids"]
    assert second_failure["gpu_uuids"] == every_gpu, (
        f"the restart lost the dead rank's GPUs: {second_failure['gpu_uuids']}"
    )
    assert second_failure["allocation_completeness"] == "COMPLETE", second_failure
    [terminal] = posted(delivered, TERMINAL_PATH)
    assert terminal["terminal_status"] == "TIMED_OUT", terminal
    assert rank_gpus(terminal["allocation"]) == expected, terminal["allocation"]
    containment, recovery, source_count = control_plane_verdict(
        [first_failure, second_failure], terminal
    )
    assert containment == every_gpu, f"the containment names {len(containment)} GPUs"
    assert recovery == every_gpu, (
        f"the recovery incident names {len(recovery)} of 16 GPUs"
    )
    assert source_count == 16, source_count
    assert after_failed_reading == {"phase": "RUNNING", "gpus": expected}, (
        f"the FAILED reading must keep the last RUNNING record: {after_failed_reading}"
    )
    assert persisted(core) == {}, "the terminal reading must drop the record"


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_restart_after_the_stop_deleted_the_pods_still_reports_every_gpu(
    environment: Environment, nodes: tuple[str, str]
) -> None:
    """The same window when the containment's STOP_WORKLOADS deleted both Pods
    while the watcher was down. With the record dropped the restarted process
    knew nothing of the attempt and never sent its terminal. Restored from the
    kept record, the attempt takes the path of a restored attempt whose Pods are
    gone (F7): our own stop suspended the PyTorchJob, so the STOPPED terminal
    naming the containment incident is sent at once, and its allocation and the
    recovery incident carry all 16 GPUs."""

    clock = Clock()
    delivered = FakeSink()
    core, _first = failed_attempt(environment, nodes, delivered=delivered, clock=clock)

    clock.value += timedelta(seconds=20)
    core.pods = []
    resolver = Exec()
    restarted = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        environment,
        clock,
        resolver,
        stopper=SimpleNamespace(custom=SuspendedPyTorchJob(), batch=None),
    )
    restarted.run_once()

    expected = {0: gpus(uid(0)), 1: gpus(uid(1))}
    every_gpu = sorted(expected[0] + expected[1])
    assert resolver.calls == [], resolver.calls
    terminals = posted(delivered, TERMINAL_PATH)
    assert len(terminals) == 1, f"the restarted watcher sent {len(terminals)} terminals"
    [terminal] = terminals
    assert (
        terminal["terminal_status"],
        terminal["termination_initiator_incident_id"],
    ) == ("STOPPED", PASSIVE_INCIDENT), terminal
    assert rank_gpus(terminal["allocation"]) == expected, terminal["allocation"]
    [failure] = posted(delivered, FAILURE_PATH)
    containment, recovery, source_count = control_plane_verdict([failure], terminal)
    assert containment == every_gpu, f"the containment names {len(containment)} GPUs"
    assert recovery == every_gpu, (
        f"the recovery incident names {len(recovery)} of 16 GPUs"
    )
    assert source_count == 16, source_count
    assert persisted(core) == {}, "the terminal reading must drop the record"


@pytest.mark.parametrize(("environment", "nodes"), PROVIDERS)
def test_a_restart_with_the_stop_unproven_waits_out_the_grace_then_reports_every_gpu(
    environment: Environment, nodes: tuple[str, str]
) -> None:
    """The rare branch of the same window: the Pods are gone and the first read
    of the workload object does not prove our stop (here it times out). The
    restored attempt then takes the missing grace like any restored attempt
    (F7): its last RUNNING reading is republished with a fresh instant -- the
    control plane reads the attempt as RUNNING again until the grace ends, the
    fail-closed cost of not knowing -- and the fresh read at the end of the
    grace finds our stop, so the STOPPED terminal names the containment
    incident and every GPU. Before, no terminal was ever sent."""

    clock = Clock()
    delivered = FakeSink()
    core, _first = failed_attempt(environment, nodes, delivered=delivered, clock=clock)
    workload = StopNotYetReadable()
    clock.value += timedelta(seconds=20)
    core.pods = []
    restarted = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        environment,
        clock,
        Exec(),
        stopper=SimpleNamespace(custom=workload, batch=None),
    )
    before = len(posted(delivered, OBSERVATION_PATH))
    restarted.run_once()
    clock.value += timedelta(seconds=5)
    restarted.run_once()
    clock.value += timedelta(seconds=300)
    restarted.run_once()

    expected = {0: gpus(uid(0)), 1: gpus(uid(1))}
    every_gpu = sorted(expected[0] + expected[1])
    readings = posted(delivered, OBSERVATION_PATH)[before:]
    assert [item["workload_phase"] for item in readings] == [
        "RUNNING",
        "RUNNING",
        "STOPPED",
    ], readings
    assert rank_gpus(readings[0]["containers"]) == expected, readings[0]
    assert workload.reads == 2, "one read at the first missing pass, one at the end"
    [terminal] = posted(delivered, TERMINAL_PATH)
    assert (
        terminal["terminal_status"],
        terminal["termination_initiator_incident_id"],
    ) == ("STOPPED", PASSIVE_INCIDENT), terminal
    assert rank_gpus(terminal["allocation"]) == expected, terminal["allocation"]
    [failure] = posted(delivered, FAILURE_PATH)
    _containment, recovery, source_count = control_plane_verdict([failure], terminal)
    assert recovery == every_gpu, (
        f"the recovery incident names {len(recovery)} of 16 GPUs"
    )
    assert source_count == 16, source_count
    assert persisted(core) == {}, "the terminal reading must drop the record"


def test_a_replayed_failed_reading_keeps_the_record_for_a_restart() -> None:
    """The outbox replay dropped the persisted record too, for any non-active
    reading it delivered: a FAILED reading the control plane refused once and
    the next pass replayed reopened the same window. Only the watcher knows
    which reading is the terminal one, so only it drops the record."""

    clock = Clock()
    delivered = RefusesTheFirstFailedReading()
    nodes = ("node-a", "node-b")
    core, first = failed_attempt(
        Environment.EKS, nodes, delivered=delivered, clock=clock
    )
    assert delivered.refused, "the FAILED reading should have been refused once"
    clock.value += timedelta(seconds=5)
    first.run_once()
    assert first.sink.last_replay["replayed"] == 1, first.sink.last_replay

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
    every_gpu = sorted(gpus(uid(0)) + gpus(uid(1)))
    assert first_failure["gpu_uuids"] == every_gpu, first_failure["gpu_uuids"]
    assert second_failure["gpu_uuids"] == every_gpu, (
        f"the replay dropped the record of the dead rank's GPUs: "
        f"{second_failure['gpu_uuids']}"
    )
    assert second_failure["allocation_completeness"] == "COMPLETE", second_failure


@pytest.mark.parametrize("case", ["never-read", "replacement"])
def test_a_restart_in_the_failure_window_invents_no_gpu(case: str) -> None:
    """The kept record hands over only what was read for the same Pod UID,
    container and rank: a rank whose exec never answered while it ran has no
    GPU after the restart either, and a replacement Pod is read from its own
    Pod, never given the GPUs its predecessor's record holds. Both allocations
    stay visibly INCOMPLETE."""

    clock = Clock()
    nodes = ("node-a", "node-b")
    delivered = FakeSink()
    core, _first = failed_attempt(
        Environment.HYPERPOD_EKS,
        nodes,
        delivered=delivered,
        clock=clock,
        resolver=Exec(
            failing=frozenset({uid(0)}) if case == "never-read" else frozenset()
        ),
    )
    dead = training_pod(
        0,
        nodes[0],
        pod_uid=uid(0) if case == "never-read" else "uid-master-0-replacement",
        exit_code=137 if case == "never-read" else 1,
    )
    clock.value += timedelta(seconds=5)
    core.pods = [dead, training_pod(1, nodes[1])]
    restarted = watcher(
        core,
        KubernetesCompletionOutbox(core, delivered),
        Environment.HYPERPOD_EKS,
        clock,
        Exec(),
    )
    restarted.run_once()
    clock.value += timedelta(seconds=5)
    core.pods = [dead, training_pod(1, nodes[1], exit_code=137)]
    restarted.run_once()

    failure = posted(delivered, FAILURE_PATH)[-1]
    assert failure["gpu_uuids"] == gpus(uid(1)), (
        f"only the survivor's GPUs are known: {failure['gpu_uuids']}"
    )
    assert failure["allocation_completeness"] == "INCOMPLETE", failure
    [terminal] = posted(delivered, TERMINAL_PATH)
    assert terminal["terminal_status"] == "FAILED", terminal
    assert {
        item["rank"]: (item["gpu_count"], item["gpu_uuids"])
        for item in terminal["allocation"]
    } == {0: (8, []), 1: (8, gpus(uid(1)))}, terminal["allocation"]
