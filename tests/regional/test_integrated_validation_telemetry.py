"""Validation telemetry of the integrated suite follows the reset/reboot barrier.

Batch 80 (r4 timeline): post-reset/reboot validation waited for the harness's
30 s refresh, whose samples carried the refresh script's start time, so every
r4 completion landed 5-30 s after a refresh. The in-Pod watcher now publishes a
node's healthy telemetry right after the control plane records its reset or
reboot, stamped with the actual publish time; the periodic refresh stays as
the fallback and stamps every node with its own write time.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from scripts.perf import seed_regional_action_workflows as seed
from tests._builders import build_store
from tests.regional.test_regional_action_capacity import release_agent_identity

T0 = datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc)
RUN = "run-a"


class Ticks:
    """A clock that advances by a fixed step on every read."""

    def __init__(self, start: datetime, step: timedelta) -> None:
        self.value = start
        self.step = step
        self.reads: list[datetime] = []

    def __call__(self) -> datetime:
        self.value += self.step
        self.reads.append(self.value)
        return self.value


def seeded_context(clusters: int = 1, workflows: int = 4) -> ApplicationContext:
    context = ApplicationContext(store=build_store())
    for cluster_index in range(clusters):
        for workflow_index in range(workflows):
            context.store.save_agent(
                seed.release_bound_synthetic_agent(
                    release_agent_identity(),
                    cluster_id=f"perf-cap-{cluster_index:03d}",
                    node_id=seed.integrated_node_id(RUN, cluster_index, workflow_index),
                    run_id=RUN,
                    now=T0,
                )
            )
    return context


def reboot_workflow(
    context: ApplicationContext, cluster_index: int, workflow_index: int
) -> WorkflowRequest:
    node_id = seed.integrated_node_id(RUN, cluster_index, workflow_index)
    scope = f"{RUN}-c{cluster_index:03d}-w{workflow_index:02d}"
    workflow = WorkflowRequest(
        request_id=f"workflow-integrated-{scope}",
        incident_id=f"inc-integrated-{scope}",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[
            WorkflowStepSpec(
                operation=WorkflowOperation.RESTART_NODE,
                execution_owner=seed.HYPERPOD_OWNER,
                node_ids=[node_id],
            ),
            WorkflowStepSpec(
                operation=WorkflowOperation.VALIDATE_GPU,
                execution_owner="gpu-fault-validation-adapter",
                node_ids=[node_id],
                depends_on_step_indexes=[0],
            ),
        ],
        created_at=T0,
        updated_at=T0,
    )
    context.store.save_incident_and_workflow(
        FaultIncident(
            incident_id=workflow.incident_id,
            event_id=f"integrated-{scope}",
            event_type="ACTION_CAPACITY_TEST",
            cluster_id=f"perf-cap-{cluster_index:03d}",
            node_ids=[node_id],
            policy_version="action-capacity-v1",
            policy_source="CONTROLLED_DRILL",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=workflow.request_id,
            created_at=T0,
            updated_at=T0,
        ),
        workflow,
    )
    return workflow


def record_reboot(
    context: ApplicationContext, workflow: WorkflowRequest, barrier: datetime
) -> WorkflowRequest:
    """What the dispatcher saves when it sees the RESTART_NODE command succeed."""

    recorded = workflow.model_copy(
        update={
            "step_executions": [
                WorkflowStepExecution(
                    step_index=0,
                    operation=WorkflowOperation.RESTART_NODE,
                    status=WorkflowStepStatus.SUCCEEDED,
                    started_at=barrier - timedelta(seconds=120),
                    updated_at=barrier,
                ),
                WorkflowStepExecution(
                    step_index=1,
                    operation=WorkflowOperation.VALIDATE_GPU,
                    status=WorkflowStepStatus.WAITING,
                    started_at=barrier,
                    updated_at=barrier,
                ),
            ],
            "completed_step_indexes": [0],
            "updated_at": barrier,
        }
    )
    context.store.save_workflow(recorded)
    return recorded


def telemetry_stamps(
    context: ApplicationContext, node_id: str
) -> tuple[set[datetime], set[datetime], set[datetime]]:
    gpu = {
        item.observed_at for item in context.gpu_metrics.latest("perf-cap-000", node_id)
    }
    host = {
        item.observed_at
        for item in context.store.list_telemetry_metrics_latest("perf-cap-000", node_id)
    }
    collectors = {
        item.last_success_at
        for item in context.store.list_collector_statuses("perf-cap-000", node_id)
    }
    return gpu, host, collectors


def test_periodic_refresh_stamps_every_node_with_its_own_write_time() -> None:
    context = seeded_context(workflows=2)
    clock = Ticks(T0 + timedelta(minutes=5), timedelta(seconds=3))

    refreshed = seed.refresh_integrated_agent_heartbeats(
        context,
        run_id=RUN,
        clusters=1,
        workflows_per_cluster=2,
        lease_seconds=600,
        clock=clock,
    )

    assert refreshed == 2
    for workflow_index, written_at in enumerate(clock.reads):
        node_id = seed.integrated_node_id(RUN, 0, workflow_index)
        gpu, host, collectors = telemetry_stamps(context, node_id)
        assert gpu == host == collectors == {written_at}, (
            f"node w{workflow_index:02d} must carry its own write time, "
            "not the refresh start"
        )
        agent = next(
            item
            for item in context.store.list_agents("perf-cap-000")
            if item.node_id == node_id
        )
        assert agent.last_seen_at == written_at
        assert agent.lease_expires_at == written_at + timedelta(seconds=600)
    assert clock.reads[0] < clock.reads[1], "the two nodes were written apart"


def test_a_recorded_reboot_triggers_an_immediate_publish_after_its_barrier() -> None:
    context = seeded_context(workflows=4)
    rebooting = reboot_workflow(context, 0, 2)
    idle = reboot_workflow(context, 0, 0)
    clock = Ticks(T0 + timedelta(minutes=1), timedelta(milliseconds=50))
    barrier = T0 + timedelta(minutes=3)
    polls: list[int] = []
    before_record: list[tuple[set[datetime], set[datetime], set[datetime]]] = []

    def between_polls(_seconds: float) -> None:
        polls.append(len(polls) + 1)
        if len(polls) == 1:
            # The executor's result alone is not the barrier: nothing is
            # published until the dispatcher's step record exists.
            before_record.append(
                telemetry_stamps(context, seed.integrated_node_id(RUN, 0, 2))
            )
            clock.value = barrier - timedelta(milliseconds=200)
            record_reboot(context, rebooting, barrier)
            clock.value = barrier + timedelta(milliseconds=100)

    stops: Iterator[bool] = iter([False, False, False, True])

    summary = seed.watch_validation_barriers(
        context,
        run_id=RUN,
        clusters=1,
        workflows_per_cluster=4,
        stopped=lambda: next(stops),
        clock=clock,
        sleep=between_polls,
    )

    assert before_record == [(set(), set(), set())], (
        "no publish may precede the recorded reset/reboot step"
    )
    node_id = seed.integrated_node_id(RUN, 0, 2)
    gpu, host, collectors = telemetry_stamps(context, node_id)
    assert len(gpu) == 1 and gpu == host == collectors, (
        "the node published one package with one stamp"
    )
    (published_at,) = gpu
    assert published_at > barrier, "the publish must postdate the reboot barrier"
    assert published_at - barrier < timedelta(seconds=1), (
        "the publish must follow the barrier at once, not at the next refresh"
    )
    assert summary["publishes"] == 1
    assert summary["nodes_published"] == 1
    assert summary["barrier_to_publish_seconds"] == [
        (published_at - barrier).total_seconds()
    ]
    assert summary["ended_by"] == "stopped"
    idle_node = idle.official_steps[0].node_ids[0]
    assert telemetry_stamps(context, idle_node) == (set(), set(), set()), (
        "a node without a completed reset/reboot gets no event-driven publish"
    )


def test_a_publish_not_later_than_its_barrier_is_repeated_until_it_is() -> None:
    """The barrier is the dispatcher Pod's clock, the stamp the api-ha Pod's:
    a publish that does not postdate the barrier would not count, so the
    watcher publishes again until its stamp is strictly later."""

    context = seeded_context(workflows=1)
    workflow = reboot_workflow(context, 0, 0)
    barrier = T0 + timedelta(minutes=3)
    record_reboot(context, workflow, barrier)
    clock = Ticks(barrier - timedelta(milliseconds=300), timedelta(milliseconds=50))
    stops = iter([False] * 6 + [True])

    summary = seed.watch_validation_barriers(
        context,
        run_id=RUN,
        clusters=1,
        workflows_per_cluster=1,
        stopped=lambda: next(stops),
        clock=clock,
        sleep=lambda _seconds: None,
    )

    delays = summary["barrier_to_publish_seconds"]
    assert isinstance(delays, list) and len(delays) >= 2, delays
    assert delays[0] <= 0 < delays[-1], (
        "the first publish did not postdate the barrier; the last one must"
    )
    assert all(item <= 0 for item in delays[:-1]), (
        "publishing must stop once a stamp is later than the barrier"
    )
    gpu, host, collectors = telemetry_stamps(
        context, seed.integrated_node_id(RUN, 0, 0)
    )
    assert gpu == host == collectors and min(gpu) > barrier, (
        "the stored samples must be strictly newer than the barrier"
    )


def test_the_watcher_republishes_only_for_a_newer_barrier() -> None:
    context = seeded_context(workflows=1)
    workflow = reboot_workflow(context, 0, 0)
    barrier = T0 + timedelta(minutes=3)
    recorded = record_reboot(context, workflow, barrier)
    clock = Ticks(barrier + timedelta(seconds=1), timedelta(milliseconds=50))
    rounds: list[int] = []

    def between_polls(_seconds: float) -> None:
        rounds.append(1)
        if len(rounds) == 2:
            later = recorded.model_copy(
                update={
                    "step_executions": [
                        recorded.step_executions[0].model_copy(
                            update={"updated_at": clock.value + timedelta(seconds=5)}
                        ),
                        recorded.step_executions[1],
                    ],
                    "updated_at": clock.value + timedelta(seconds=5),
                }
            )
            context.store.save_workflow(later)
            clock.value += timedelta(seconds=6)

    stops = iter([False, False, False, False, True])
    summary = seed.watch_validation_barriers(
        context,
        run_id=RUN,
        clusters=1,
        workflows_per_cluster=1,
        stopped=lambda: next(stops),
        clock=clock,
        sleep=between_polls,
    )

    assert summary["publishes"] == 2, (
        "one publish per barrier: the same barrier seen again is not republished"
    )


def test_the_watcher_ends_once_every_run_workflow_is_terminal() -> None:
    context = seeded_context(workflows=1)
    workflow = reboot_workflow(context, 0, 0)
    context.store.save_workflow(
        workflow.model_copy(
            update={"status": WorkflowStatus.SUCCEEDED, "updated_at": T0}
        )
    )

    summary = seed.watch_validation_barriers(
        context,
        run_id=RUN,
        clusters=1,
        workflows_per_cluster=1,
        stopped=lambda: False,
        clock=Ticks(T0, timedelta(milliseconds=50)),
        sleep=lambda _seconds: None,
    )

    assert summary["ended_by"] == "all-terminal"
    assert summary["publishes"] == 0
