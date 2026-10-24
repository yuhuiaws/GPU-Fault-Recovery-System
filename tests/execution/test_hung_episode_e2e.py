from datetime import timedelta

import pytest

from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import IncidentState, WorkflowStatus, workflow_is_open
from gpu_fault.models import WorkflowOperation as Op
from tests.execution._health_policy_e2e_support import JOB
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture
from tests.execution._raw_health_e2e_support import RawHealthRig


def traffic(rig, rate):
    batch = HostTelemetryBatch(
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=rig.clock,
        received_at=rig.clock,
        runtime_profile_version="simulated-v1",
        workload_state="ACTIVE",
        affected_workload_ids=[JOB],
        samples=[
            HostMetricSample(name="efa_traffic_bytes_per_second", value=rate),
            HostMetricSample(name="efa_traffic_observation_valid", value=1),
        ],
    )
    return rig.post(
        "/v1/collector-events/host-telemetry", batch.model_dump(mode="json")
    )


def hung_workflows(rig):
    return [
        row
        for row in rig.store.list_workflows()
        if Op.COLLECT_HUNG_TRIAGE in {step.operation for step in row.official_steps}
    ]


def low_wave(rig):
    rig.advance(10)
    traffic(rig, 0)
    for _ in range(10):
        rig.advance(60)
        traffic(rig, 0)


@pytest.mark.parametrize("finish_first", [True, False], ids=["finished", "in-flight"])
def test_api_sqlite_noise_dedup_stable_rearm_and_second_evidence_round(
    tmp_path, monkeypatch, finish_first
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:
        for _ in range(3):
            traffic(rig, 16_000_000)
            rig.advance(60)
        low_wave(rig)
        [first] = hung_workflows(rig)
        rig.bind_task_fabric(first)
        rig.host()
        rig.gpu({"gpu_temperature_c": 40})
        if finish_first:
            for _ in range(8):
                rig.execute(first)
            assert (
                rig.store.get_workflow(first.request_id).status
                is WorkflowStatus.SUCCEEDED
            )
            saved = rig.store.get_workflow(first.request_id)
            rig.store.save_workflow(
                saved.model_copy(
                    update={"lifetime_deadline_at": rig.clock + timedelta(seconds=1)}
                )
            )
        previous_state = rig.store.get_incident(first.incident_id).state
        rig.advance(10)
        traffic(rig, 2)
        low_wave(rig)
        assert len(hung_workflows(rig)) == 1, "noise must not start another diagnostic"
        for _ in range(3):
            rig.advance(60)
            traffic(rig, 4_000_000)
        assert rig.store.get_incident(first.incident_id).state is previous_state, (
            "traffic rearm must not declare NCCL or training recovered"
        )
        low_wave(rig)
        workflows = hung_workflows(rig)
        assert len(workflows) == 2, workflows
        second = next(row for row in workflows if row.request_id != first.request_id)
        assert second.predecessor_workflow_id == first.request_id, second
        assert second.incident_id != first.incident_id, (
            "a new episode needs its own lifecycle budget"
        )
        if finish_first:
            assert (
                second.lifetime_deadline_at
                != rig.store.get_workflow(first.request_id).lifetime_deadline_at
            ), "new episodes must not inherit an exhausted predecessor lifetime"
        assert not second.preempt_predecessor, second
        assert not {
            Op.RESTART_WORKLOAD,
            Op.STOP_WORKLOADS,
            Op.RESET_GPU,
            Op.RESTART_NODE,
        } & {step.operation for step in second.official_steps}, second
        original = next(
            step
            for step in first.official_steps
            if step.operation is Op.COLLECT_HUNG_SUMMARY
        )
        following = next(
            step
            for step in second.official_steps
            if step.operation is Op.COLLECT_HUNG_SUMMARY
        )
        assert (
            original.parameters["efa_stall_episode"]["episode_id"]
            != following.parameters["efa_stall_episode"]["episode_id"]
        )
        rig.advance(60)
        traffic(rig, 0)
        assert len(hung_workflows(rig)) == 2, "same second wave must be deduplicated"
        rig.host()
        rig.gpu({"gpu_temperature_c": 40})
        if not finish_first:
            # The predecessor must finish first: the dispatcher holds a
            # successor while ``workflow_is_open`` says its predecessor is
            # (``held("predecessor")``, test_dispatcher_single_instance). The
            # executor does not gate on it; an executor call that stood here
            # was held only by the unbound-scope VALIDATE_FABRIC wait.
            predecessor = rig.store.get_workflow(first.request_id)
            assert workflow_is_open(predecessor.status, predecessor.blocked_kind), (
                predecessor
            )
            for _ in range(8):
                rig.execute(first)
        rig.bind_task_fabric(second)
        for _ in range(8):
            rig.execute(second)
        assert (
            rig.store.get_workflow(second.request_id).status is WorkflowStatus.SUCCEEDED
        )
        assert (
            rig.store.get_incident(second.incident_id).state is IncidentState.ESCALATED
        )
    finally:
        rig.store.close()


def test_raw_counter_recovery_with_default_edge_filter_rearms_without_extra_probes(
    chain,
):
    rig = chain(nodes=1)
    first = rig.trigger()
    rig.collector.edge_filter_enabled = True
    assert rig.collector.health_summary_seconds == 300, (
        "exercise the deployed default cadence"
    )
    followups = 0
    for _ in range(12):
        batch = rig.traffic(2_000_000, seconds=15)
        followups += "efa-traffic-recovery-window" in batch.edge_filter_reasons
    state = rig.store.get_efa_traffic_state("cluster-a/node-a")
    assert state.stall_episode is None and state.last_stable_traffic_at is not None, (
        state
    )
    assert 3 <= followups <= 6, "bounded retransmission of already collected samples"
    assert (
        rig.store.get_incident(first.incident_id).state is not IncidentState.RECOVERED
    )
    assert all(not actor.calls for actor in rig.actors), (
        "rearm delivery does not invoke diagnostic tools"
    )
    batch = rig.traffic(2_000_000, seconds=45)
    assert "efa-traffic-recovery-window" not in batch.edge_filter_reasons, batch


def test_recovered_first_reporter_allows_new_wave_on_a_different_node():
    from datetime import datetime, timedelta, timezone

    from gpu_fault.health_recovery_contract import is_hung_finding
    from gpu_fault.host_health import NodeHealthPolicy
    from tests._builders import build_context
    from tests.host_health._support import efa_telemetry, observe_running_attempt

    context = build_context()
    policy = NodeHealthPolicy(context.store)
    start = datetime.now(timezone.utc)
    rounds = []
    samples = [
        *[(offset, 16_000_000, 16_000_000) for offset in (0, 60, 120)],
        *[(offset, 0, 16_000_000) for offset in range(130, 731, 60)],
        *[(offset, 4_000_000, 16_000_000) for offset in (740, 800, 860)],
        *[(offset, 4_000_000, 0) for offset in range(870, 1471, 60)],
    ]
    for offset, first, second in samples:
        stamp = start + timedelta(seconds=offset)
        observe_running_attempt(context.store, stamp, ("node-a", "node-b"))
        for node, rate in (("node-a", first), ("node-b", second)):
            batch = efa_telemetry(f"sample-{node}-{offset}", rate, stamp).model_copy(
                update={"node_id": node}
            )
            for finding in policy.evaluate_metrics(batch):
                if is_hung_finding(finding):
                    rounds.append(context.orchestrator.ingest_node_health(finding)[1])
    assert len(rounds) == 2 and rounds[0].request_id != rounds[1].request_id, rounds
    assert rounds[1].predecessor_workflow_id == rounds[0].request_id, rounds
    assert rounds[1].incident_id != rounds[0].incident_id, rounds
    second_state = context.store.get_efa_traffic_state("cluster-a/node-b")
    assert second_state.last_stable_traffic_at is None, (
        "node B has its first local stall"
    )
