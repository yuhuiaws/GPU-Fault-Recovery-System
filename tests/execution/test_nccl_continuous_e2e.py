import pytest

from gpu_fault.hung_summary_codec import decode_summary
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.watcher import WorkloadPhase
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture

FORBIDDEN = {
    Op.STOP_WORKLOADS,
    Op.RESTART_WORKLOAD,
    Op.RESET_GPU,
    Op.RESTART_NODE,
    Op.REPLACE_NODE,
    Op.MARK_UNSCHEDULABLE,
    Op.QUARANTINE,
}


def settled(rig, workflow):
    result = rig.run(workflow)
    assert result.status is WorkflowStatus.SUCCEEDED, result
    saved = rig.store.get_workflow(workflow.request_id)
    assert not FORBIDDEN & {step.operation for step in saved.official_steps}, saved
    assert rig.store.get_incident(saved.incident_id).state is IncidentState.ESCALATED, (
        saved
    )
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
    return saved


def decision(workflow):
    bundle = next(
        step
        for step in workflow.official_steps
        if step.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
    )
    return bundle, bundle.parameters["hung_triage_decision"]


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "fr-lag",
        "fr-pickle",
        "fr-advancing",
        "fr-second-missing",
        "fr-whole-group",
        "unranked-io",
        "pyspy-error",
        "same-role",
        "different-role",
        "pid-reuse",
        "foreign-process",
        "normalized-cpu",
        "tid-namespace",
    ],
)
def test_raw_rdma_to_real_collection_archive_and_operator_handoff(chain, mode):
    rig = chain(nodes=5 if mode in {"same-role", "different-role"} else 3, mode=mode)
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    bundle, report = decision(saved)
    assert report["root_cause_confirmed"] is False, report
    survey = next(
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    )
    assert survey.details["hung_survey"]["responding_node_count"] == len(rig.nodes), (
        survey
    )
    assert {
        item.node_id
        for item in rig.commands
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    } == set(rig.nodes)
    assert report["targeted_capture_eligible"] is (
        mode in {"fr-lag", "fr-pickle", "pid-reuse"}
    ), report
    if mode in {"fr-lag", "fr-pickle", "pid-reuse"}:
        assert report["culprit_ranks"] == [0], report
        assert bundle.node_ids == ["node-a", "node-b"], bundle
    if mode == "fr-whole-group":
        assert report["stalled_process_groups"] == ["0:dp"], report
        assert report["culprit_ranks"] == [], report
    if mode == "fr-advancing":
        assert report["evidence_branches"]["fr_temporal"]["process_groups"][0][
            "completion_advanced_ranks"
        ] == [0], report
    if mode == "unranked-io":
        assert any(
            item["node_id"] == "node-a" and item["pid"] == 100 and item["rank"] is None
            for item in report["candidate_processes"]
        ), report
    if mode == "pyspy-error":
        assert all(
            item["stack"]["valid_samples"] == 0
            for item in report["evidence_branches"]["process_observations"]["processes"]
        ), report
    if mode == "same-role":
        assert any(
            "ROLE_COHORT_STACK_DIFFERENCE" in item["reasons"]
            for item in report["candidate_processes"]
        ), report
    if mode == "different-role":
        assert not any(
            "ROLE_COHORT_STACK_DIFFERENCE" in item["reasons"]
            for item in report["candidate_processes"]
        ), report
    if mode == "normalized-cpu":
        processes = report["evidence_branches"]["process_observations"]["processes"]
        assert [item["behavior"]["cpu_cores"] for item in processes] == [1, 1, 1], (
            processes
        )
    if mode == "tid-namespace":
        processes = report["evidence_branches"]["process_observations"]["processes"]
        assert all(
            item["stack"]["stable_threads"][0]["host_thread_id"] == 100
            for item in processes
        ), processes
    archives = rig.archives()
    assert set(archives) == set(bundle.node_ids), archives
    receipts = next(
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
    ).details["node_results"]
    for node, (manifest, digest) in archives.items():
        assert receipts[node]["sha256"] == digest, receipts
        assert manifest["fault_context"]["hung_triage_decision"] == report, manifest
        assert "strace" in manifest["omitted_intrusive_tools"], manifest
    if mode == "pid-reuse":
        manifest = archives["node-a"][0]
        assert manifest["gpu_processes"] == [], manifest
        assert any(
            item["reason"] == "process_identity_changed_or_missing"
            for item in manifest["skipped_processes"]
        ), manifest
    if mode == "foreign-process":
        assert all(
            argv[-1] != "900"
            for actor in rig.actors
            for argv in actor.calls
            if argv[0] == "/unit/py-spy"
        ), rig.commands


def test_expansion_budget_coverage_and_bundle_are_persisted_end_to_end(chain):
    rig = chain(nodes=9)
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    waves = [
        step.node_ids
        for step in saved.official_steps
        if step.operation is Op.COLLECT_HUNG_TRIAGE
    ]
    assert [len(nodes) for nodes in waves] == [4, 4], waves
    assert set(waves[0]).isdisjoint(waves[1]), waves
    bundle, report = decision(saved)
    assert report["active_responding_node_count"] == 8, report
    assert len(report["not_sampled_nodes"]) == 1, report
    assert len(rig.archives()) == len(bundle.node_ids) == 8, bundle
    for item in saved.step_executions:
        if item.operation is Op.COLLECT_HUNG_TRIAGE:
            assert all(
                "encoding" in value for value in item.details["node_results"].values()
            ), item
            assert all(
                len(decode_summary(value)["ranks"]) == 1
                for value in item.details["node_results"].values()
            ), item


@pytest.mark.parametrize("failure", ["summary", "triage", "corrupt", "old-epoch"])
def test_partial_evidence_still_reaches_review_without_claiming_coverage(
    chain, failure
):
    rig = chain()
    workflow = rig.trigger()
    rig.healthy()
    setattr(
        rig,
        {
            "summary": "summary_unreachable",
            "triage": "triage_unreachable",
            "corrupt": "corrupt_node",
            "old-epoch": "old_epoch_node",
        }[failure],
        "node-c",
    )
    saved = settled(rig, workflow)
    _, report = decision(saved)
    assert report["targeted_capture_eligible"] is False, report
    if failure in {"summary", "old-epoch"}:
        survey = next(
            item
            for item in saved.step_executions
            if item.operation is Op.COLLECT_HUNG_SUMMARY
        )
        assert "node-c" in survey.details["hung_survey"]["missing_nodes"], survey
    else:
        assert "node-c" in report["undetermined_nodes"], report


def test_fabric_waits_for_missing_telemetry_and_never_closes_from_process_data_alone(
    chain,
):
    rig = chain()
    workflow = rig.trigger()
    rig.healthy(omit_gpu="node-c")
    result = rig.run(workflow, attempts=2)
    assert result.status is WorkflowStatus.RUNNING, result
    assert (
        rig.store.get_incident(workflow.incident_id).state
        is not IncidentState.RECOVERED
    )
    assert Op.ESCALATE_SUPPORT not in rig.boundary.calls, rig.boundary.calls
    rig.healthy()
    settled(rig, workflow)


def test_bundle_failure_cannot_be_reported_as_training_recovered(chain):
    rig = chain()
    workflow = rig.trigger()
    rig.healthy()
    rig.bundle_unreachable = "node-b"
    result = rig.run(workflow)
    saved = rig.store.get_workflow(workflow.request_id)
    assert rig.store.get_incident(saved.incident_id).state is IncidentState.ESCALATED, (
        saved
    )
    # The partial bundle is evidence, not the end: the hand-off still runs and
    # the record keeps the bundle's failure as its verdict.
    assert result.status is WorkflowStatus.FAILED, result
    assert Op.ESCALATE_SUPPORT in rig.boundary.calls, rig.boundary.calls
    assert saved.terminal_failure_reason and "node-b" in (
        saved.terminal_failure_reason
    ), saved
    assert not FORBIDDEN & {step.operation for step in saved.official_steps}, saved
    receipts = [
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
    ]
    assert receipts and receipts[0].details["failed_nodes"] == ["node-b"], receipts
    assert set(receipts[0].details["node_results"]) == {"node-a", "node-c"}, receipts
    assert set(rig.archives()) == {"node-a", "node-c"}, rig.archives()


@pytest.mark.parametrize(
    "case", ["always-low", "invalid-counter", "new-attempt", "traffic-recovers"]
)
def test_raw_counter_negative_paths_do_not_start_hung_collection(chain, case):
    rig = chain()
    rig.traffic(seconds=0)
    if case == "always-low":
        for _ in range(14):
            rig.traffic()
    else:
        rig.traffic(2_000_000, 15)
        rig.traffic(seconds=130)
        for _ in range(9):
            rig.traffic()
        if case == "new-attempt":
            previous = rig.store.list_attempt_observations("cluster-a")[0]
            rig.store.save_attempt_observation(
                previous.model_copy(update={"workload_phase": WorkloadPhase.STOPPED})
            )
            rig.attempt = "attempt-b"
        rig.traffic(
            2_000_000 if case == "traffic-recovers" else 0,
            invalid=case == "invalid-counter",
        )
    assert rig.hung_workflows() == [], rig.store.list_workflows()
    assert rig.commands == [], rig.commands
    if case == "new-attempt":
        state = rig.store.get_efa_traffic_state("cluster-a/node-a")
        assert state.attempt_id == "attempt-b" and state.had_active_traffic is False, (
            state
        )


def test_nccl_log_is_evidence_only_without_rdma_history(chain):
    rig = chain()
    result = rig.log("NCCL WARN collective timeout error", source="training-log")
    assert result["findings"][0]["recommended_action"] == "NO_ACTION", result
    assert result["workflow_request_ids"] == result["marker_ids"] == [], result
    assert rig.hung_workflows() == [] and rig.commands == [], rig.commands


@pytest.mark.parametrize("window", [0, 15])
def test_multiple_reporting_nodes_share_one_raw_triggered_collection(
    chain, monkeypatch, window
):
    monkeypatch.setenv("GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_SECONDS", str(window))
    rig = chain()
    rig.traffic_all(seconds=0)
    rig.traffic_all(2_000_000, 15)
    rig.traffic_all(seconds=130)
    for _ in range(10):
        rig.traffic_all()
    [workflow] = rig.hung_workflows()
    rig.healthy()
    settled(rig, workflow)
    summaries = [
        item.node_id
        for item in rig.commands
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    ]
    assert sorted(summaries) == sorted(rig.nodes * 2), summaries


@pytest.mark.parametrize("timeout", [False, True], ids=["resume", "pending-timeout"])
def test_executor_restart_retains_completed_nodes_and_pending_deadlines(
    chain, monkeypatch, timeout
):
    from gpu_fault.adapters.node_action import hung_survey

    class Slices:
        value = 0

        def monotonic(self):
            self.value += 6
            return self.value

    rig = chain()
    workflow = rig.trigger()
    rig.healthy()
    monkeypatch.setattr(hung_survey, "time", Slices())
    result = rig.run(workflow, attempts=1)
    assert result.status is WorkflowStatus.RUNNING, result
    saved = rig.store.get_workflow(workflow.request_id)
    survey = next(
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    )
    assert set(survey.details["node_results"]) == {"node-a"}, survey
    rig.restart_executor()
    if timeout:
        rig.pending_node = "node-b"
        rig.run(workflow, attempts=1)
        rig.advance(31)
        rig.run(workflow)
        rig.advance(31)
    saved = settled(rig, workflow)
    summaries = [
        item.node_id
        for item in rig.commands
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    ]
    assert summaries.count("node-a") == 2, summaries
    if timeout:
        survey = next(
            item
            for item in saved.step_executions
            if item.operation is Op.COLLECT_HUNG_SUMMARY
        )
        assert survey.details["node_failures"]["node-b"] == "NODE_TIMEOUT", survey
        assert survey.details["hung_survey"]["responding_node_count"] == 2, survey


def test_long_observation_gap_is_not_counted_as_continuous_idle(chain):
    rig = chain()
    rig.traffic(seconds=0)
    rig.traffic(2_000_000, 15)
    rig.traffic(seconds=130)
    rig.traffic(seconds=700)
    assert rig.hung_workflows() == [], rig.store.list_workflows()
    state = rig.store.get_efa_traffic_state("cluster-a/node-a")
    assert state.zero_since == rig.clock, state


@pytest.mark.parametrize("operation", [Op.COLLECT_HUNG_SUMMARY, Op.COLLECT_HUNG_TRIAGE])
def test_lost_response_and_agent_restart_replay_without_recapturing(chain, operation):
    rig = chain()
    workflow = rig.trigger()
    rig.healthy()
    rig.lose_response = (operation, "node-a")
    settled(rig, workflow)
    counts = [
        item for item in rig.command_io_counts if item[1:3] == ("node-a", operation)
    ]
    assert len(counts) == (3 if operation is Op.COLLECT_HUNG_SUMMARY else 2), counts
    assert counts[0][0] == counts[1][0], counts
    assert counts[0][3] > 0 and counts[1][3] == 0, counts


def test_missing_workload_scope_skips_attachments_and_requires_operator_review(chain):
    rig = chain()
    rig.scope_known = False
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    survey = next(
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    )
    assert survey.details["hung_survey"]["responding_node_count"] == 0, survey
    assert rig.archives() == {}, "unknown workload scope must not authorize attachment"
    assert {item.operation for item in rig.commands} == {Op.COLLECT_HUNG_SUMMARY}, (
        rig.commands
    )


def test_whole_task_communication_summary_prioritizes_an_unsampled_lagging_node(chain):
    rig = chain(nodes=9, mode="cooperative")
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    survey = next(
        item
        for item in reversed(saved.step_executions)
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    )
    assert survey.details["hung_survey"]["candidate_ranks"] == [8], survey
    first_wave = next(
        step
        for step in saved.official_steps
        if step.operation is Op.COLLECT_HUNG_TRIAGE
    )
    assert rig.nodes[-1] in first_wave.node_ids, first_wave
    assert any(
        item.operation is Op.COLLECT_HUNG_TRIAGE and item.node_id == rig.nodes[-1]
        for item in rig.commands
    ), rig.commands


def test_summary_byte_budget_exhaustion_is_unknown_not_full_coverage(
    chain, monkeypatch
):
    from gpu_fault.adapters.node_action import hung_survey

    rig = chain()
    workflow = rig.trigger()
    rig.healthy()
    monkeypatch.setattr(hung_survey, "MAX_RESULT_BYTES", 1)
    saved = settled(rig, workflow)
    survey = next(
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    )
    assert set(survey.details["node_failures"].values()) == {
        "RESULT_BUDGET_EXCEEDED"
    }, survey
    assert survey.details["hung_survey"]["responding_node_count"] == 0, survey
    assert survey.details["hung_survey"]["expected_node_count"] == 3, survey
    assert rig.archives() == {}, "no valid scope evidence means no attachment"
