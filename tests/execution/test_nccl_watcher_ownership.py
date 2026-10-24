from datetime import timedelta

import pytest

from gpu_fault.adapters import ControlPlaneEvidenceAdapter
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from gpu_fault.telemetry import EvidenceKind
from gpu_fault.watcher import CompletionWatcher
from tests._builders import container_observation
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture
from tests.execution._raw_health_e2e_support import RawHealthRig

MESSAGES = [
    "NCCL WARN invalid usage error",
    "NCCL WARN collective timeout error",
    "torch.distributed backend failed",
    "RuntimeError: ordinary application exception",
]


@pytest.mark.parametrize("message", MESSAGES)
def test_logs_then_nonzero_exit_follow_generic_watcher_containment_and_restart(
    tmp_path, monkeypatch, message
):
    rig = RawHealthRig(tmp_path, monkeypatch)
    try:
        rig.context.workflow_executor.adapters.append(ControlPlaneEvidenceAdapter())
        observation = rig.store.list_attempt_observations("cluster-a")[0]
        peer = container_observation(
            "pod-b", "trainer", 1, "node-b", gpu_uuids=["GPU-b"], host_pid=456
        )
        observation = observation.model_copy(
            update={
                "expected_critical_ranks": 2,
                "containers": [*observation.containers, peer],
            }
        )
        rig.store.save_attempt_observation(observation)
        result = rig.log(message, source="journal", fields={"_PID": "123"})
        assert result["workflow_request_ids"] == result["marker_ids"] == [], result
        watcher = CompletionWatcher()
        live = watcher.observe(observation)
        assert live.failure_detected is None and live.commands == [], live

        rig.clock += timedelta(seconds=1)
        failure = observation.model_copy(
            update={
                "observed_at": rig.clock,
                "containers": [
                    observation.containers[0].model_copy(
                        update={
                            "terminated": True,
                            "exit_code": 1,
                            "finished_at": rig.clock,
                        }
                    ),
                    peer,
                ],
            }
        )
        rig.store.save_attempt_observation(failure)
        failed = watcher.observe(failure)
        assert failed.failure_detected is not None, failed
        receipt = rig.post(
            "/v1/attempts/failure-detected",
            failed.failure_detected.model_dump(mode="json"),
        )
        replay = rig.post(
            "/v1/attempts/failure-detected",
            failed.failure_detected.model_dump(mode="json"),
        )
        assert (
            replay["duplicate"] is True
            and replay["workflow_request_id"] == receipt["workflow_request_id"]
        ), replay
        workflow = rig.store.get_workflow(receipt["workflow_request_id"])
        assert [step.operation for step in workflow.official_steps] == [
            Op.FREEZE_EVIDENCE,
            Op.STOP_WORKLOADS,
        ], workflow
        assert rig.execute(workflow).status is WorkflowStatus.SUCCEEDED

        rig.clock += timedelta(seconds=1)
        terminal_observation = failure.model_copy(
            update={
                "observed_at": rig.clock,
                "termination_initiator_incident_id": receipt["incident_id"],
                "containers": [
                    failure.containers[0],
                    peer.model_copy(
                        update={
                            "terminated": True,
                            "exit_code": 143,
                            "finished_at": rig.clock,
                        }
                    ),
                ],
            }
        )
        rig.store.save_attempt_observation(terminal_observation)
        terminal = watcher.observe(terminal_observation).terminal_event
        assert terminal is not None, terminal_observation
        decision = rig.post("/v1/attempts/terminal", terminal.model_dump(mode="json"))
        plan = rig.store.get_plan(decision["recovery_plan_id"])
        assert [step.action.value for step in plan.steps] == ["RESTART_WORKLOAD"], plan
        assert decision["matched_marker_ids"] == [], decision
        assert not any(
            step.operation
            in {Op.COLLECT_HUNG_SUMMARY, Op.COLLECT_HUNG_TRIAGE, Op.VALIDATE_FABRIC}
            for workflow in rig.store.list_workflows()
            for step in workflow.official_steps
        ), "ordinary exit handling must not acquire an NCCL diagnostic chain"
    finally:
        rig.store.close()


@pytest.mark.parametrize("moment", ["before-summary", "after-recheck"])
@pytest.mark.parametrize("state", ["terminal", "peer-failed", "cleanup"])
def test_known_exit_or_cleanup_stops_new_hung_collection(
    chain, monkeypatch, moment, state
):
    rig = chain(mode="fr-lag")
    workflow = rig.trigger()
    rig.healthy()

    def finish():
        current = rig.store.list_attempt_observations("cluster-a")[0]
        updates = {"observed_at": rig.clock}
        if state == "cleanup":
            updates["termination_initiator_incident_id"] = "watcher-containment"
        elif state == "terminal":
            updates["workload_phase"] = "FAILED"
            updates["containers"] = [
                row.model_copy(
                    update={
                        "terminated": True,
                        "exit_code": 1,
                        "finished_at": rig.clock,
                    }
                )
                for row in current.containers
            ]
        else:
            updates["containers"] = [
                current.containers[0].model_copy(
                    update={
                        "terminated": True,
                        "exit_code": 1,
                        "finished_at": rig.clock,
                    }
                ),
                *current.containers[1:],
            ]
        # Validate enum and container state, just as the observation API does.
        rig.store.save_attempt_observation(
            type(current).model_validate({**current.model_dump(), **updates})
        )

    if moment == "before-summary":
        finish()
    else:
        sender = rig.node_adapter.sender

        def send(endpoint, envelope):
            result = sender(endpoint, envelope)
            command = envelope.command
            if (
                command.parameters.get("survey_pass") == "RECHECK"
                and command.node_id == rig.nodes[-1]
            ):
                finish()
            return result

        monkeypatch.setattr(rig.node_adapter, "sender", send)
    result = rig.run(workflow)
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert all(
        command.operation is Op.COLLECT_HUNG_SUMMARY for command in rig.commands
    ), rig.commands
    assert all(actor.requests == 0 for actor in rig.actors), (
        "no fresh FR triggers after known exit/cleanup"
    )
    if moment == "before-summary":
        assert rig.commands == [], rig.commands
    saved = rig.store.get_workflow(workflow.request_id)
    assert any(
        entry.details.get("recovery_owner") == "COMPLETION_WATCHER"
        for entry in saved.step_executions
    ), saved


def test_ordinary_log_is_retained_without_creating_fault_markers(chain):
    rig = chain()
    result = rig.log("NCCL WARN invalid usage error", source="training-log")
    evidence = rig.store.list_raw_evidence(
        "cluster-a", node_id="node-a", kind=EvidenceKind.NODE_LOGS
    )
    assert any(
        entry["message"] == "NCCL WARN invalid usage error"
        for record in evidence
        for entry in record.payload["entries"]
    ), evidence
    assert result["marker_ids"] == result["workflow_request_ids"] == [], result


def test_persistent_rdma_suspicion_remains_independent_of_ordinary_errors(chain):
    rig = chain(mode="fr-lag")
    result = rig.log("NCCL WARN invalid usage error", source="training-log")
    assert result["workflow_request_ids"] == [], result
    workflow = rig.trigger()
    assert all(
        not step.parameters.get("preserve_error_evidence_first")
        for step in workflow.official_steps
    ), workflow
    rig.healthy()
    assert rig.run(workflow).status is WorkflowStatus.SUCCEEDED
    assert any(
        command.operation is Op.COLLECT_HUNG_TRIAGE for command in rig.commands
    ), rig.commands


def test_peer_failure_during_low_traffic_window_does_not_launch_hung(chain):
    rig = chain()
    rig.traffic(seconds=0)
    rig.traffic(2_000_000, 15)
    rig.traffic(seconds=130)
    for _ in range(9):
        rig.traffic()
    rig.clock += timedelta(seconds=60)
    current = rig.store.list_attempt_observations("cluster-a")[0]
    rig.store.save_attempt_observation(
        current.model_copy(
            update={
                "observed_at": rig.clock,
                "containers": [
                    current.containers[0],
                    current.containers[1].model_copy(
                        update={
                            "terminated": True,
                            "exit_code": 1,
                            "finished_at": rig.clock,
                        }
                    ),
                    *current.containers[2:],
                ],
            }
        )
    )
    rig.write_counters()
    rig.collector.collect_once()
    assert rig.hung_workflows() == [], (
        "exit cleanup belongs to the watcher even while peers still have zero RDMA"
    )
