"""The passive communication monitor scopes a managed job by its runtime container ids.

The control plane assigns a node a monitor target only when every one of the
attempt's containers there names its cgroup (``CommunicationTarget.
cgroup_paths``), and that came from a ``gpu-fault.io/cgroup-path`` Pod
annotation nothing sets: on every managed Pod the lease carried
``expected_target_count=1`` with no target, the Agent never sampled the job, and
the hung survey's cached history stayed ``UNAVAILABLE`` -- read-only evidence
missing, nothing sampled wrongly.

A target can now name the attempt's containers by the runtime id the kubelet
reports (``container_ids``), which the Agent resolves to cgroups on the node
with the hung chain's resolver. A v1 Agent validates its plan with
``extra="forbid"`` and would refuse the whole lease over the new field, so the
ids go only to an Agent whose heartbeat advertises ``communication-monitor/v2``;
a v1 Agent keeps receiving byte for byte the plan it accepts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import Field, field_validator

from gpu_fault.app.context import ApplicationContext
from gpu_fault.communication_monitoring import (
    COMMUNICATION_MONITOR_V1,
    COMMUNICATION_MONITOR_V2,
    CommunicationTarget,
    observation_plan,
)
from gpu_fault.models import StrictModel
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.communication_history import CommunicationHistory
from gpu_fault.node_agent.communication_runtime import CommunicationRuntime
from gpu_fault.node_agent.observation_supervisor import ObservationSupervisor
from gpu_fault.node_agent.observation_worker import ObservationAgent
from gpu_fault.node_agent.passive_communication import sample_communication
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_store,
    container_observation,
)
from tests.execution._flight_series_support import NOW
from tests.fleet._support import heartbeat, registry, signed
from tests.node_agent.test_communication_monitor import use_imported_package
from tests.node_agent.test_heartbeat_cadence import _reporter
from tests.node_agent.test_hung_collection_hardening import stat_text

TRAINER = hashlib.sha256(b"trainer-container").hexdigest()
PEER = hashlib.sha256(b"second-trainer-container").hexdigest()
OTHER = hashlib.sha256(b"another-job-container").hexdigest()
POD = "kubepods-burstable-pod835a5f72_cf56_4102_912f_3dbb2ddfa995.slice"
POD_DIR = f"/kubepods.slice/kubepods-burstable.slice/{POD}"


def scope(identifier: str, runtime: str = "cri-containerd") -> str:
    return f"{POD_DIR}/{runtime}-{identifier}.scope"


class V1Target(StrictModel):
    """``CommunicationTarget`` as every Agent before this fix validates it."""

    job_id: str = Field(min_length=1, max_length=256)
    attempt_id: str = Field(min_length=1, max_length=256)
    cgroup_paths: list[str] = Field(min_length=1, max_length=256)
    gpu_uuids: list[str] = Field(default_factory=list, max_length=256)
    ras_coordinator: bool = False

    @field_validator("cgroup_paths")
    @classmethod
    def scoped_paths(cls, values: list[str]) -> list[str]:
        if any(
            not path.startswith("/") or path == "/" or ".." in path.split("/")
            for path in values
        ):
            raise ValueError("communication observation needs explicit cgroups")
        return values


class V1Plan(StrictModel):
    """``CommunicationPlan`` as every Agent before this fix validates it."""

    version: Literal["communication-monitor/v1"] = "communication-monitor/v1"
    cluster_id: str
    node_id: str
    boot_id: str
    generation: int
    issued_at: datetime
    expires_at: datetime
    targets: list[V1Target] = Field(default_factory=list, max_length=8)
    expected_target_count: int = Field(ge=0)
    node_addresses: list[str] = Field(default_factory=list, max_length=8)


class Store:
    def __init__(self, *containers: Any, observed_at: datetime = NOW) -> None:
        self.containers = list(containers)
        self.observed_at = observed_at

    def list_attempt_observations(self, cluster: str):
        return [
            attempt_observation(
                "job", "attempt", self.observed_at, containers=self.containers
            )
        ]


def record(node: str = "node-a") -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node,
        cluster_id="cluster-a",
        boot_id="boot",
        generation=1,
        allowed_operations=[Op.COLLECT_HUNG_SUMMARY],
        lifecycle_state=SimpleNamespace(value="ACTIVE"),
        endpoint=f"https://{node}:9443",
    )


def managed(rank: int = 0, identifier: str = TRAINER, **values: Any):
    """A running container of a managed Pod: the kubelet's id, no annotation."""

    return container_observation(
        f"pod-{rank}",
        "trainer",
        rank,
        values.pop("node_id", "node-a"),
        container_id=f"containerd://{identifier}",
        gpu_uuids=["GPU-a"],
        **values,
    )


def test_today_a_v1_agent_gets_no_target_for_a_managed_pod() -> None:
    plan = observation_plan(Store(managed()), record(), NOW)

    assert plan.version == COMMUNICATION_MONITOR_V1, plan
    assert plan.expected_target_count == 1 and plan.targets == [], plan
    V1Plan.model_validate(plan.model_dump(mode="json"))


def test_a_v2_agent_is_sent_the_managed_pods_container_ids() -> None:
    plan = observation_plan(
        Store(managed(0), managed(1, PEER)),
        record(),
        NOW,
        version=COMMUNICATION_MONITOR_V2,
    )

    assert plan.version == COMMUNICATION_MONITOR_V2, plan
    assert plan.expected_target_count == 1, plan
    (target,) = plan.targets
    assert target.container_ids == sorted([TRAINER, PEER]), target
    assert target.cgroup_paths == [], target
    assert target.gpu_uuids == ["GPU-a"] and target.ras_coordinator is True, target


def test_an_annotated_pod_keeps_the_exact_v1_plan_a_v1_agent_accepts() -> None:
    annotated = container_observation(
        "pod-0", "trainer", 0, "node-a", cgroup_path="/job", gpu_uuids=["GPU-a"]
    )

    v1 = observation_plan(Store(annotated), record(), NOW)
    v2 = observation_plan(
        Store(annotated), record(), NOW, version=COMMUNICATION_MONITOR_V2
    )

    (target,) = v1.model_dump(mode="json")["targets"]
    assert target == {
        "job_id": "job",
        "attempt_id": "attempt",
        "cgroup_paths": ["/job"],
        "gpu_uuids": ["GPU-a"],
        "ras_coordinator": True,
    }, target
    V1Plan.model_validate(v1.model_dump(mode="json"))
    assert v2.model_dump(mode="json")["targets"] == [target], v2


def test_a_v1_agent_would_refuse_the_whole_plan_over_a_container_id() -> None:
    """Why the ids need the capability check: the old model forbids the field."""

    plan = observation_plan(
        Store(managed()), record(), NOW, version=COMMUNICATION_MONITOR_V2
    )

    with pytest.raises(ValueError):
        V1Plan.model_validate(plan.model_dump(mode="json"))


def test_a_mixed_node_names_both_its_cgroups_and_its_container_ids() -> None:
    annotated = container_observation(
        "pod-1", "trainer", 1, "node-a", cgroup_path="/job", gpu_uuids=["GPU-a"]
    )

    v1 = observation_plan(Store(managed(0), annotated), record(), NOW)
    v2 = observation_plan(
        Store(managed(0), annotated), record(), NOW, version=COMMUNICATION_MONITOR_V2
    )

    assert v1.targets == [] and v1.expected_target_count == 1, v1
    (target,) = v2.targets
    assert target.cgroup_paths == ["/job"] and target.container_ids == [TRAINER]


@pytest.mark.parametrize(
    "unscoped",
    [
        pytest.param({"container_id": None}, id="no-id-yet"),
        pytest.param({"container_id": "containerd://abc123"}, id="short-id"),
        pytest.param({"cgroup_path": "/"}, id="root-cgroup"),
        pytest.param({"cgroup_path": "/job/../other"}, id="escaping-cgroup"),
    ],
)
def test_one_unscoped_container_leaves_the_node_untargeted(unscoped) -> None:
    other = container_observation("pod-1", "trainer", 1, "node-a", **unscoped)

    plan = observation_plan(
        Store(managed(0), other), record(), NOW, version=COMMUNICATION_MONITOR_V2
    )

    assert plan.targets == [] and plan.expected_target_count == 1, plan


def test_more_containers_than_a_target_holds_leave_the_node_untargeted() -> None:
    containers = [
        managed(rank, hashlib.sha256(str(rank).encode()).hexdigest())
        for rank in range(65)
    ]

    plan = observation_plan(
        Store(*containers), record(), NOW, version=COMMUNICATION_MONITOR_V2
    )

    assert plan.targets == [] and plan.expected_target_count == 1, plan


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"container_ids": ["abc123"]}, id="short-id"),
        pytest.param({"container_ids": [f"containerd://{TRAINER}"]}, id="prefixed"),
        pytest.param({"container_ids": [], "cgroup_paths": []}, id="no-scope"),
    ],
)
def test_a_target_must_name_a_scope_it_can_prove(value) -> None:
    with pytest.raises(ValueError):
        CommunicationTarget.model_validate(
            {"job_id": "job", "attempt_id": "attempt", **value}
        )


def test_the_heartbeat_route_answers_each_agent_in_the_version_it_speaks() -> None:
    now = datetime.now(timezone.utc)
    store = build_store()
    store.save_attempt_observation(
        attempt_observation("job", "attempt", now, containers=[managed()])
    )
    ctx = ApplicationContext(
        store=store,
        execution_token="e" * 32,
        fleet_registry=registry(store, now=lambda: now),
    )

    async def scenario() -> dict[str, Any]:
        plans: dict[str, Any] = {}
        async with asgi_client(ctx) as client:
            for offset, version in enumerate(
                (COMMUNICATION_MONITOR_V1, COMMUNICATION_MONITOR_V2, "monitor/v9")
            ):
                value = heartbeat("node-a", observed_at=now).model_copy(
                    update={
                        "communication_monitor_version": version,
                        "allowed_operations": [Op.COLLECT_HUNG_SUMMARY],
                        "observed_at": now + timedelta(seconds=offset),
                    }
                )
                response = await client.post(
                    "/v1/fleet/agents/heartbeat",
                    json=signed(value).model_dump(mode="json"),
                )
                assert response.status_code == 200, response.text
                plans[version] = response.json().get("communication_plan")
        return plans

    plans = asyncio.run(scenario())

    assert plans[COMMUNICATION_MONITOR_V1]["targets"] == [], plans
    V1Plan.model_validate(plans[COMMUNICATION_MONITOR_V1])
    (target,) = plans[COMMUNICATION_MONITOR_V2]["targets"]
    assert target["container_ids"] == [TRAINER], plans
    assert plans[COMMUNICATION_MONITOR_V2]["version"] == COMMUNICATION_MONITOR_V2
    assert plans["monitor/v9"] is None, "an unknown monitor version gets no lease"


def test_the_agent_advertises_the_version_that_reads_container_ids() -> None:
    sent = []
    reporter = _reporter(
        communication_plan_sink=lambda plan: None,
        sender=lambda _url, envelope: sent.append(envelope) or {},
    )

    reporter.report_once()

    (envelope,) = sent
    assert (
        envelope.heartbeat.communication_monitor_version == COMMUNICATION_MONITOR_V2
    ), envelope.heartbeat


def _runtime(tmp_path: Path) -> CommunicationRuntime:
    root = tmp_path / "proc"
    boot = root / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("boot")
    return CommunicationRuntime(
        SimpleNamespace(
            node_ids={"node-a"},
            agent_generation=1,
            proc_root=root,
            infiniband_root=tmp_path / "ib",
            diagnostic_output_dir=tmp_path / "evidence",
            python_stack_tool="",
            diagnostic_s3_uri=None,
            diagnostic_retention_seconds=600,
            diagnostic_max_archives=10,
        )
    )


def test_the_agent_accepts_a_v2_lease_with_container_ids(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    now = datetime.now(timezone.utc)
    plan = observation_plan(
        Store(managed(), observed_at=now),
        record(),
        now,
        version=COMMUNICATION_MONITOR_V2,
    )

    runtime.accept_plan(plan.model_dump(mode="json"))

    health = runtime.health()
    assert health["lease_valid"] is True, health
    assert health["assigned_targets"] == 1 == health["expected_targets"], health


def rank(root: Path, pid: int, cgroup: str, environment: bytes = b"") -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cgroup").write_text(f"0::{cgroup}\n")
    (directory / "stat").write_text(stat_text(pid, 1000))
    (directory / "environ").write_bytes(environment)
    (directory / "status").write_text("State:\tS\n")


def sample(tmp_path: Path, target: dict[str, Any]) -> dict[str, Any]:
    agent = ObservationAgent(
        {
            "proc_root": str(tmp_path / "proc"),
            "infiniband_root": str(tmp_path / "ib"),
            "diagnostic_output_dir": str(tmp_path / "evidence"),
        }
    )
    agent.runner = lambda argv, **kwargs: CompletedProcess(argv, 1, "", "")
    plan = {
        "version": COMMUNICATION_MONITOR_V2,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "boot_id": "boot",
        "generation": 1,
        "issued_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(seconds=90)).isoformat(),
        "expected_target_count": 1,
        "targets": [target],
    }
    return sample_communication(agent, target, plan)


def history(tmp_path: Path) -> list[dict[str, Any]]:
    store = CommunicationHistory(tmp_path / "evidence" / "communication-history.db")
    return store.read(
        store.scope("job", "attempt", "boot"),
        since=0,
        until=datetime.now(timezone.utc).timestamp() + 60,
    )


def target(*identifiers: str) -> dict[str, Any]:
    return {
        "job_id": "job",
        "attempt_id": "attempt",
        "container_ids": list(identifiers),
    }


def test_the_agent_samples_every_process_of_the_named_container_and_nothing_else(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    rank(proc, 99, scope(TRAINER))
    for local in range(2):
        rank(
            proc, 100 + local, scope(TRAINER), f"RANK={local}\0WORLD_SIZE=2\0".encode()
        )
    rank(proc, 900, scope(OTHER), b"RANK=0\0WORLD_SIZE=8\0")
    rank(proc, 950, f"{POD_DIR}/crio-conmon-{TRAINER}.scope")

    result = sample(tmp_path, target(TRAINER))

    assert result["status"] == "SAMPLED" and result["process_count"] == 3, result
    assert result["scan_complete"] is True, result
    assert result["workload_scope"]["resolved_container_count"] == 1, result
    (row,) = history(tmp_path)
    document = row["document"]
    assert sorted(item["pid"] for item in document["processes"]) == [99, 100, 101]
    assert all(
        item["process_identity"]["verified"] for item in document["processes"]
    ), document["processes"]
    assert document["workload_scope"]["basis"] == "RUNTIME_CONTAINER_ID", document


def test_one_container_found_of_two_is_an_incomplete_sample(tmp_path: Path) -> None:
    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0WORLD_SIZE=2\0")

    result = sample(tmp_path, target(TRAINER, PEER))

    assert result["status"] == "SAMPLED" and result["process_count"] == 1, result
    assert result["scan_complete"] is False, "an unresolved container hides ranks"
    assert result["workload_scope"]["unresolved_container_ids"] == [PEER], result


def test_no_container_found_is_scope_unknown_and_writes_no_history(
    tmp_path: Path,
) -> None:
    rank(tmp_path / "proc", 900, scope(OTHER), b"RANK=0\0WORLD_SIZE=8\0")

    result = sample(tmp_path, target(TRAINER))

    assert result["status"] == "SCOPE_UNKNOWN", result
    assert result["workload_scope"]["reason"] == "CONTAINER_CGROUP_NOT_FOUND", result
    assert result["dump_triggers"] == 0, result
    assert not (tmp_path / "evidence" / "communication-history.db").exists() or (
        history(tmp_path) == []
    ), "a scope never found is not recorded as an empty job"


def test_an_explicit_target_samples_as_before(tmp_path: Path) -> None:
    rank(tmp_path / "proc", 100, "/training/job", b"RANK=0\0WORLD_SIZE=1\0")

    result = sample(
        tmp_path,
        {"job_id": "job", "attempt_id": "attempt", "cgroup_paths": ["/training/job"]},
    )

    assert result["status"] == "SAMPLED" and result["process_count"] == 1, result
    assert result["workload_scope"] == {"basis": "EXPLICIT_CGROUP_PATHS"}, result
    assert json.dumps(result), "the result stays JSON for the healthz view"


def test_the_owned_worker_reads_a_v2_target_from_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child the runtime starts gets the target as the lease dumped it."""

    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0WORLD_SIZE=1\0")
    monkeypatch.setenv("PATH", str(tmp_path))
    use_imported_package(monkeypatch)
    now = datetime.now(timezone.utc)
    plan = observation_plan(
        Store(managed(), observed_at=now),
        record(),
        now,
        version=COMMUNICATION_MONITOR_V2,
    )
    (lease_target,) = plan.targets

    result = ObservationSupervisor().run(
        {
            "operation": "sample",
            "settings": {
                "proc_root": str(tmp_path / "proc"),
                "infiniband_root": str(tmp_path / "ib"),
                "diagnostic_output_dir": str(tmp_path / "evidence"),
                "python_stack_tool": "",
            },
            "target": lease_target.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
        },
        10,
    )

    assert result["status"] == "SAMPLED" and result["process_count"] == 1, result
    assert result["workload_scope"]["resolved_container_count"] == 1, result
