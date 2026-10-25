"""An event's cgroup path attributes it to a managed container by runtime id.

A GPU event may carry the faulting process's ``cgroup_path`` (one of the four
attribution fields of the event API, design v2 §5.5). The control plane
compared it only with the container's own ``cgroup_path`` -- the
``gpu-fault.io/cgroup-path`` Pod annotation nothing sets -- so a path naming a
managed job's container matched no container at all: the event found no
attempt (``NO_ACTIVE_MANAGED_ATTEMPT``), its workload state stayed UNKNOWN and
an XID 46 went SAFETY_PENDING (freeze, cordon, quarantine, wait for an
operator) instead of the attributed stop, reset and restart of the job. Fail
closed, no automatic recovery. None of the product's own collectors sets the
field; an external producer that does met this.

The runtime names a container's cgroup after the id the kubelet reports
(``cri-containerd-<id>.scope``, ``crio-<id>.scope``, ``docker-<id>.scope``, the
cgroupfs ``<id>``), so a container without an explicit path matches a path
that names its id. A path naming another container, CRI-O's monitor
(``crio-conmon-<id>.scope``) or no container is not this one's; an explicit
path keeps its comparison.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.telemetry import WorkloadTopologyService
from gpu_fault.workload_container_scope import cgroup_in_container
from tests._builders import (
    attempt_observation,
    build_context,
    build_store,
    container_observation,
)
from tests.orchestration._cov95_runtime_builder import xid
from tests.orchestration._cov95_runtime_evidence import EvidenceHarness
from tests.orchestration._cross_fault_support import (
    NOW,
    observation,
    post_faults,
    xid_payload,
)

TRAINER = hashlib.sha256(b"trainer-container").hexdigest()
OTHER = hashlib.sha256(b"another-job-container").hexdigest()
POD = "kubepods-burstable-pod835a5f72_cf56_4102_912f_3dbb2ddfa995.slice"
POD_DIR = f"/kubepods.slice/kubepods-burstable.slice/{POD}"
OWN = f"{POD_DIR}/cri-containerd-{TRAINER}.scope"

PATHS = [
    pytest.param(OWN, True, id="its-own-scope"),
    pytest.param(f"{OWN}/", True, id="trailing-slash"),
    pytest.param(f"{OWN}/init.scope", True, id="a-cgroup-below-it"),
    pytest.param(
        "/kubepods/burstable/pod835a5f72-cf56-4102-912f-3dbb2ddfa995/" + TRAINER,
        True,
        id="cgroupfs-driver",
    ),
    pytest.param(f"{POD_DIR}/cri-containerd-{OTHER}.scope", False, id="other-job"),
    pytest.param(f"{POD_DIR}/crio-conmon-{TRAINER}.scope", False, id="runtime-monitor"),
    pytest.param(POD_DIR, False, id="pod-level"),
    pytest.param("/system.slice/customer-monitor.service", False, id="host-service"),
]


def managed(**values):
    return container_observation(
        "pod-a",
        "trainer",
        0,
        "node-a",
        gpu_uuids=["GPU-a"],
        container_id=f"containerd://{TRAINER}",
        **values,
    )


@pytest.mark.parametrize(("path", "matches"), PATHS)
def test_a_managed_container_is_named_by_its_runtime_id(path, matches) -> None:
    assert cgroup_in_container(path, managed()) is matches


@pytest.mark.parametrize(
    ("path", "matches"),
    [
        ("/job/a/child", True),
        ("/job", True),
        ("/job/a", True),
        ("/another", False),
        (OWN, False),
    ],
)
def test_an_explicit_path_keeps_its_comparison(path, matches) -> None:
    assert cgroup_in_container(path, managed(cgroup_path="/job/a/")) is matches


@pytest.mark.parametrize(("path", "matches"), PATHS)
def test_the_workload_context_resolves_the_event_to_the_managed_job(
    path, matches
) -> None:
    service = WorkloadTopologyService(build_store())
    service.observe(
        attempt_observation(
            "job-a",
            "attempt-a",
            NOW,
            workload_ids=["training/job-a"],
            containers=[managed()],
        )
    )

    result = service.resolve("cluster-a", "node-a", NOW, cgroup_path=path)

    assert result.workload_state == ("ACTIVE" if matches else "IDLE"), result
    assert result.job_ids == (["job-a"] if matches else []), result


@pytest.mark.parametrize(("path", "matches"), PATHS)
def test_the_attempt_observation_follows_the_same_identity(path, matches) -> None:
    h = EvidenceHarness()
    job = h.observation(containers=[managed()])

    result = h.service.attempt_observation(
        xid(observed_at=h.now, gpu_uuid=None, cgroup_path=path)
    )

    assert result == (job if matches else None), result


def test_an_event_naming_the_jobs_container_stops_and_restarts_the_job() -> None:
    """An XID 46 posted with the faulting process's cgroup, as an external
    producer posts it, on a node running a managed job."""

    context = build_context()
    context.store.save_attempt_observation(
        observation().model_copy(update={"containers": [managed()]})
    )
    payload = {**xid_payload(46, "xid-46-cgroup"), "cgroup_path": OWN}

    (response,) = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))

    workflow = context.store.get_workflow(response["workflow_request_id"])
    assert workflow.status is WorkflowStatus.PENDING, (
        "attributed: no SAFETY_PENDING over an UNKNOWN workload state",
        workflow.status,
        workflow.blocked_reasons,
    )
    operations = [step.operation for step in workflow.official_steps]
    assert WorkflowOperation.STOP_WORKLOADS in operations, operations
    assert WorkflowOperation.RESTART_WORKLOAD in operations, operations
    incident = context.store.get_incident(response["incident_id"])
    assert (incident.job_id, incident.attempt_id) == ("train", "train-a001"), incident
