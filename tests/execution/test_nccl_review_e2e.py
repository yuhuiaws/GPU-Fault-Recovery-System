import hashlib
from pathlib import Path

import pytest

from gpu_fault.hung_summary_codec import decode_summary
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from tests.execution._nccl_chain_support import chain_fixture as chain_fixture
from tests.execution.test_nccl_continuous_e2e import decision, settled
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent.test_process_diagnostics_e2e import proc_stat


def recheck(workflow):
    return next(
        item
        for item in reversed(workflow.step_executions)
        if item.operation is Op.COLLECT_HUNG_SUMMARY
    )


def test_inherited_rank_and_same_start_tick_child_do_not_remove_training_rank(chain):
    rig = chain(mode="cooperative")
    actor = rig.actors[0]
    for pid in (101, 102):
        child = process(actor.root / "proc", pid)
        (child / "stat").write_text(proc_stat(pid, 0))
        (child / "environ").write_bytes(
            (actor.proc / "environ").read_bytes() if pid == 101 else b""
        )
        (child / "root").symlink_to(actor.proc / "root", target_is_directory=True)
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    survey = recheck(saved)
    report = survey.details["hung_survey"]
    assert report["rank_inventory_complete"] is True, report
    assert report["valid_rank_count"] == 3 and report["duplicate_ranks"] == [], report
    assert report["auxiliary_process_count"] == 2, report
    rows = decode_summary(survey.details["node_results"]["node-a"])["ranks"]
    assert {row["pid"] for row in rows} == {100, 101, 102}, rows
    assert next(row for row in rows if row["pid"] == 101)["communication"] == [], rows


def test_process_bound_registration_resolves_rank_without_environment(chain):
    rig = chain(mode="cooperative")
    for actor in rig.actors:
        (actor.proc / "environ").write_bytes(
            b"GPU_FAULT_RANK_PROGRESS_FILE=/tmp/context.json\0"
        )
    workflow = rig.trigger()
    rig.healthy()
    report = recheck(settled(rig, workflow)).details["hung_survey"]
    assert report["valid_rank_count"] == 3, report
    assert report["participant_without_rank_count"] == 0, report
    assert report["communication_time_series_coverage_fraction"] == 1, report


def test_pg_status_by_numeric_id_and_both_original_dumps_survive_rotation(chain):
    rig = chain(mode="fr-pg-status")
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    _, report = decision(saved)
    assert report["culprit_ranks"] == [0] and report["targeted_capture_eligible"], (
        report
    )
    triage = next(
        item
        for item in saved.step_executions
        if item.operation is Op.COLLECT_HUNG_TRIAGE
    )
    row = decode_summary(triage.details["node_results"]["node-a"])["ranks"][0]
    series = row["flight_recorder_series"]
    originals = []
    for point in ("first", "second"):
        group = series[point]["groups"][0]
        assert group["progress"]["watermark_source"] == "PG_STATUS", group
        assert group["progress"]["last_completed"] == 99, group
        assert any(
            entry["collective_seq_id"] == 100 for entry in group["semantic_entries"]
        ), "bounded semantic evidence must retain the pending operation"
        evidence = series[point]["raw_evidence"]
        assert evidence["status"] == "PRESERVED", evidence
        raw = Path(evidence["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == evidence["sha256"], evidence
        originals.append(evidence["path"])
    assert originals[0] != originals[1], originals


def test_missing_clock_bound_cannot_authorize_cross_node_targeted_capture(chain):
    rig = chain(mode="fr-lag")
    rig.clock_bound_known = False
    workflow = rig.trigger()
    rig.healthy()
    _, report = decision(settled(rig, workflow))
    assert report["targeted_capture_eligible"] is False, report
    group = report["evidence_branches"]["fr_temporal"]["process_groups"][0]
    assert group["time_comparable"] is False, group
    assert group["time_reason"] == "MISSING_CLOCK_BOUND_OR_INVALID_WINDOW", group


def validation_record(workflow):
    return next(
        item
        for item in workflow.step_executions
        if item.operation is Op.VALIDATE_FABRIC
    )


def test_management_link_is_not_a_substitute_for_task_nic_binding(chain):
    rig = chain()
    rig.fabric_scope_known = False
    workflow = rig.trigger()
    rig.healthy()
    result = rig.run(workflow)
    assert result.status is WorkflowStatus.RUNNING, result
    saved = rig.store.get_workflow(workflow.request_id)
    validation = next(
        item for item in saved.step_executions if item.operation is Op.VALIDATE_FABRIC
    )
    assert all(
        item["reason"] == "TASK_NIC_PORT_SCOPE_UNAVAILABLE"
        for item in validation.details["node_pending"].values()
    ), validation


def test_rank_thread_owned_qps_bind_the_task_ports_end_to_end(chain):
    """The summary's own QP evidence binds the scope; no binding is injected."""

    rig = chain()
    rig.fabric_scope_known = False
    rig.qp_owner_thread = 131
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    step = next(
        item for item in saved.official_steps if item.operation is Op.VALIDATE_FABRIC
    )
    assert {
        node: (scope["basis"], scope["rdma_ports"])
        for node, scope in step.parameters["training_fabric_by_node"].items()
    } == {node: ("TASK_DEVICE_BINDING", ["efa0/1"]) for node in rig.nodes}, step
    validation = validation_record(saved)
    assert validation.details["validated_nodes"] == rig.nodes, validation
    assert validation.details["fabric_evidence_scope"] == "SCOPED_TELEMETRY_ONLY", (
        validation
    )
    assert "fabric_status" not in validation.details, validation


@pytest.mark.parametrize(
    "message,kind",
    [
        ("NCCL WARN collective timeout error", "COLLECTIVE_TIMEOUT"),
        ("NCCL WARN invalid usage error", "EXPLICIT_NCCL_ERROR"),
    ],
)
def test_explicit_errors_are_evidence_only_while_processes_live(chain, message, kind):
    rig = chain(mode="fr-lag")
    result = rig.log(message, source="training-log")
    assert result["findings"][0]["diagnostic_parameters"]["nccl_event_kind"] == kind, (
        result
    )
    assert result["workflow_request_ids"] == result["marker_ids"] == [], result
    assert rig.hung_workflows() == [] and rig.commands == [], rig.commands
    assert all(actor.requests == 0 for actor in rig.actors), (
        "saving existing FR must not trigger a new dump"
    )


def test_recheck_is_a_barrier_before_any_active_collection(chain):
    rig = chain(mode="cooperative", nodes=9)
    workflow = rig.trigger()
    rig.healthy()
    saved = settled(rig, workflow)
    operations = [command.operation for command in rig.commands]
    assert operations[:18] == [Op.COLLECT_HUNG_SUMMARY] * 18, operations
    assert [command.parameters["survey_pass"] for command in rig.commands[:18]] == (
        ["BASELINE"] * 9 + ["RECHECK"] * 9
    )
    report = recheck(saved).details["hung_survey"]
    assert report["candidate_ranks"] == [8], report
