"""The correlated suite's node return follows the recorded reboot (§8.3).

Both escalated reboots of a correlated cluster are followed by post-action
validation that accepts only telemetry observed after the reboot, and a
synthetic node publishes nothing on its own. The runner's 30 s refresh was the
only publisher and stamped every sample with the refresh script's start, so a
reboot recorded while a refresh ran left the later nodes older than the
barrier until the next refresh (the batch-87 finding for the integrated suite).
Now an in-Pod watch publishes a node's node-return package right after the
dispatcher records the reboot, stamped with the actual write time, and the
periodic refresh stamps every Agent with its own write time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from scripts.perf import regional_validation_telemetry as telemetry
from scripts.perf import seed_regional_action_workflows as seed
from tests._builders import build_store
from tests.regional.test_integrated_validation_telemetry import Ticks
from tests.regional.test_regional_action_capacity import release_agent_identity
from tests.regional.test_validation_telemetry_watcher import (
    FakeSession,
    Launcher,
    control,
    session_summary,
)

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)
RUN = "corract-a"
PREEMPT = f"corr-node-{RUN}-c000-preempt"
RESET = f"corr-node-{RUN}-c000-reset"


def seeded_context() -> ApplicationContext:
    context = ApplicationContext(store=build_store())
    for node_id in (PREEMPT, RESET):
        context.store.save_agent(
            seed.release_bound_synthetic_agent(
                release_agent_identity(),
                cluster_id="perf-cap-000",
                node_id=node_id,
                run_id=RUN,
                now=T0,
            )
        )
    return context


def telemetry_stamps(
    context: ApplicationContext, node_id: str
) -> tuple[set[datetime], set[datetime], set[datetime], set[str]]:
    latest = context.gpu_metrics.latest("perf-cap-000", node_id)
    host = context.store.list_telemetry_metrics_latest("perf-cap-000", node_id)
    statuses = context.store.list_collector_statuses("perf-cap-000", node_id)
    return (
        {item.observed_at for item in latest},
        {item.observed_at for item in host},
        {item.last_success_at for item in statuses},
        {item.sample.gpu_uuid for item in latest},
    )


def escalated_reboot(context: ApplicationContext) -> WorkflowRequest:
    """The preempt lane's record: a failed fabric reset, then its reboot rung."""

    workflow = WorkflowRequest(
        request_id=f"workflow-corr-{RUN}-c000-preempt",
        incident_id=f"inc-corr-{RUN}-c000-preempt",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[
            WorkflowStepSpec(
                operation=WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                execution_owner=seed.NODE_OWNER,
                node_ids=[PREEMPT],
            ),
            WorkflowStepSpec(
                operation=WorkflowOperation.RESTART_NODE,
                execution_owner=seed.HYPERPOD_OWNER,
                node_ids=[PREEMPT],
                depends_on_step_indexes=[0],
            ),
            WorkflowStepSpec(
                operation=WorkflowOperation.VALIDATE_GPU,
                execution_owner="gpu-fault-validation-adapter",
                node_ids=[PREEMPT],
                depends_on_step_indexes=[1],
            ),
        ],
        created_at=T0,
        updated_at=T0,
    )
    context.store.save_incident_and_workflow(
        FaultIncident(
            incident_id=workflow.incident_id,
            event_id=f"corr-{RUN}-c000-preempt",
            event_type="SXID_11001",
            cluster_id="perf-cap-000",
            node_ids=[PREEMPT],
            policy_version="correlated-v1",
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
) -> None:
    """What the dispatcher saves once the reboot rung's command succeeded."""

    context.store.save_workflow(
        workflow.model_copy(
            update={
                "step_executions": [
                    WorkflowStepExecution(
                        step_index=0,
                        operation=WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                        status=WorkflowStepStatus.FAILED,
                        started_at=barrier - timedelta(seconds=200),
                        updated_at=barrier - timedelta(seconds=150),
                    ),
                    WorkflowStepExecution(
                        step_index=1,
                        operation=WorkflowOperation.RESTART_NODE,
                        status=WorkflowStepStatus.SUCCEEDED,
                        started_at=barrier - timedelta(seconds=120),
                        updated_at=barrier,
                    ),
                    WorkflowStepExecution(
                        step_index=2,
                        operation=WorkflowOperation.VALIDATE_GPU,
                        status=WorkflowStepStatus.WAITING,
                        started_at=barrier,
                        updated_at=barrier,
                    ),
                ],
                "completed_step_indexes": [1],
                "updated_at": barrier,
            }
        )
    )


def test_the_periodic_refresh_stamps_every_agent_with_its_own_write_time() -> None:
    context = seeded_context()
    clock = Ticks(T0 + timedelta(minutes=5), timedelta(seconds=3))

    refreshed = seed.refresh_run_agent_heartbeats(
        context,
        run_id=RUN,
        clusters=1,
        lease_seconds=600,
        publish_telemetry=True,
        clock=clock,
    )

    assert refreshed == 2
    agents = {item.node_id: item for item in context.store.list_agents("perf-cap-000")}
    assert {agents[node].last_seen_at for node in (PREEMPT, RESET)} == set(
        clock.reads
    ), "each Agent's heartbeat is its own write time"
    assert len(set(clock.reads)) == 2, "the two Agents were written apart"
    for node_id in (PREEMPT, RESET):
        written_at = agents[node_id].last_seen_at
        gpu, host, collectors, uuids = telemetry_stamps(context, node_id)
        assert gpu == host == collectors == {written_at}, (
            f"{node_id} must carry its own write time, not the refresh start"
        )
        assert uuids == {"GPU-corr-000"}, "the GPU its attempt identity names"
        assert agents[node_id].lease_expires_at == written_at + timedelta(seconds=600)


def test_a_recorded_reboot_rung_publishes_the_node_return_package_at_once() -> None:
    context = seeded_context()
    rebooting = escalated_reboot(context)
    clock = Ticks(T0 + timedelta(minutes=1), timedelta(milliseconds=50))
    barrier = T0 + timedelta(minutes=4)
    before_record: list[tuple[object, ...]] = []
    polls: list[int] = []

    def between_polls(_seconds: float) -> None:
        polls.append(len(polls) + 1)
        if len(polls) == 1:
            # The failed fabric reset is no barrier, and the reboot command's
            # result alone is none either: nothing until the step is recorded.
            before_record.append(telemetry_stamps(context, PREEMPT))
            clock.value = barrier - timedelta(milliseconds=200)
            record_reboot(context, rebooting, barrier)
            clock.value = barrier + timedelta(milliseconds=100)

    stops: Iterator[bool] = iter([False, False, False, True])

    summary = seed.watch_node_return_barriers(
        context,
        run_id=RUN,
        clusters=1,
        stopped=lambda: next(stops),
        clock=clock,
        sleep=between_polls,
    )

    assert before_record == [(set(), set(), set(), set())], (
        "no publish may precede the recorded reboot step"
    )
    gpu, host, collectors, uuids = telemetry_stamps(context, PREEMPT)
    assert len(gpu) == 1 and gpu == host == collectors, (
        "the node published one node-return package with one stamp"
    )
    (published_at,) = gpu
    assert barrier < published_at < barrier + timedelta(seconds=1), (
        "the publish follows the reboot barrier at once, not at the next refresh"
    )
    assert uuids == {"GPU-corr-000"}, "the package carries the attempt's GPU"
    assert telemetry_stamps(context, RESET) == (set(), set(), set(), set()), (
        "a node without a recorded reboot gets no event-driven publish"
    )
    assert summary["mode"] == seed.NODE_RETURN_WATCH_MODE
    assert summary["nodes_watched"] == 2 and summary["nodes_published"] == 1
    assert summary["barrier_to_publish_seconds"] == [
        (published_at - barrier).total_seconds()
    ]
    assert summary["ended_by"] == "stopped"


def test_terminal_workflows_do_not_end_the_node_return_watch() -> None:
    """Successor and escalation records make the run's workflow count open-ended:
    only the runner (or the deadline) ends the session."""

    context = seeded_context()
    workflow = escalated_reboot(context)
    context.store.save_workflow(
        workflow.model_copy(update={"status": WorkflowStatus.SUCCEEDED})
    )
    stops = iter([False, False, False, True])

    summary = seed.watch_node_return_barriers(
        context,
        run_id=RUN,
        clusters=1,
        stopped=lambda: next(stops),
        clock=Ticks(T0, timedelta(milliseconds=50)),
        sleep=lambda _seconds: None,
    )

    assert summary["ended_by"] == "stopped", summary
    assert summary["terminal_workflows"] == 1 and summary["polls"] == 3


def test_the_session_runs_the_node_return_watch_mode() -> None:
    session = FakeSession(
        {**session_summary([0.3]), "mode": telemetry.NODE_RETURN_WATCH_MODE},
        exits_early=False,
    )
    launcher = Launcher([session])
    running = telemetry.ValidationTelemetryWatcher(
        run_id=RUN,
        clusters=32,
        deadline_seconds=2400,
        mode=telemetry.NODE_RETURN_WATCH_MODE,
        popen=launcher,
        control_fn=control,
    )
    running.start()
    deadline = time.monotonic() + 10
    while not launcher.argv and time.monotonic() < deadline:
        time.sleep(0.01)

    summary = running.stop()

    (argv,) = launcher.argv
    assert f"ACTION_SEED_MODE={telemetry.NODE_RETURN_WATCH_MODE}" in argv, argv
    assert not any(item.startswith("ACTION_WORKFLOWS_PER_CLUSTER=") for item in argv), (
        "the correlated watch has no fixed workflows per cluster"
    )
    assert summary["publishes"] == 1, "the session's own summary is taken"
    assert telemetry.NODE_RETURN_WATCH_MODE == seed.NODE_RETURN_WATCH_MODE
    assert telemetry.WATCH_MODE == seed.VALIDATION_WATCH_MODE


def test_the_bootstrap_runs_the_real_node_return_watch_until_stdin_closes(
    tmp_path: Path,
) -> None:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        # The checkout's src, as pytest's pythonpath gives the test itself.
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "HOME": str(tmp_path),
        "ACTION_RUN_ID": RUN,
        "ACTION_CLUSTERS": "1",
        "ACTION_SEED_MODE": telemetry.NODE_RETURN_WATCH_MODE,
        "ACTION_WATCH_POLL_SECONDS": "0.05",
    }
    process = subprocess.Popen(
        [sys.executable, "-c", telemetry.BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=environment,
    )
    assert process.stdin is not None
    process.stdin.write(telemetry.session_input(telemetry.SEED_SCRIPT.read_bytes()))
    process.stdin.flush()
    time.sleep(1.0)
    assert process.poll() is None, "the watch ended while its stdin was still open"
    output, errors = process.communicate(timeout=120)  # closes stdin: EOF

    assert process.returncode == 0, errors.decode()
    summary = json.loads(output.decode().strip().splitlines()[-1])
    assert summary["mode"] == telemetry.NODE_RETURN_WATCH_MODE
    assert summary["ended_by"] == "stopped", "stdin EOF is the stop signal"
    assert summary["publishes"] == 0
