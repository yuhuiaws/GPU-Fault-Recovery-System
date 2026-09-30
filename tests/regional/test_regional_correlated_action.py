from __future__ import annotations

import json
from datetime import datetime, timezone

from gpu_fault.models import (
    RecoveryAction,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import SxidEvent, XidEvent
from gpu_fault.training_models import TrainingProgressHeartbeat
from gpu_fault.watcher import AttemptObservation
from scripts.perf import benchmark_correlated_action_scenario as benchmark
from scripts.perf import regional_capacity_suite as capacity_suite
from scripts.perf import regional_correlated_action_suite as suite
from tests._builders import build_context, copy_model, workflow_step_execution

RESTART_SAFETY_PARAMETERS = {
    "cluster_id",
    "job_id",
    "source_attempt_id",
    "source_gpu_count",
    "restart_budget",
}


def test_correlated_action_checks_the_deployed_preemption_variable(monkeypatch) -> None:
    scripts = []

    def fake_control(*args, **_kwargs):
        if args[0] == "get":
            return "api-a worker-a"
        scripts.append(args[-1])
        return "true\n"

    monkeypatch.setattr(suite, "control", fake_control)

    suite.validate_preemption_enabled()

    assert scripts, "preemption validation inspected no control-plane Pod scripts"
    assert all(
        "GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION" in script for script in scripts
    ), "preemption validation omitted the deployed environment variable"
    assert all(
        "GPU_FAULT_WORKFLOW_PREEMPTION_ENABLED" not in script for script in scripts
    ), "preemption validation still referenced the obsolete environment variable"


def test_correlated_action_seed_uses_release_bound_agent_identity(monkeypatch) -> None:
    captured = {}
    identity = {
        "agent_protocol_version": 3,
        "node_action_key_version": 2,
        "agent_version": "0.10.0",
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "b" * 64,
        "installer_bundle_sha256": "c" * 64,
        "policy_version": "catalog-v1",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "d" * 64,
        "allowed_operations": [WorkflowOperation.RESET_GPU.value],
    }

    def fake_control(*args, **kwargs):
        if args[0] == "get":
            return "api-a"
        captured["args"] = args
        captured["stdin"] = kwargs["stdin"]
        return json.dumps(
            {"agents_created": 4, "agent_identity_source": "release-state"}
        )

    monkeypatch.setattr(suite, "control", fake_control)

    result = suite.seed_agents("run-a", 2, agent_identity=identity)

    assert result["agents_created"] == 4
    assert "ACTION_SEED_MODE=correlated-agents" in captured["args"]
    identity_argument = next(
        item
        for item in captured["args"]
        if isinstance(item, str) and item.startswith("ACTION_AGENT_IDENTITY_JSON=")
    )
    assert json.loads(identity_argument.partition("=")[2]) == identity
    assert b"release_bound_synthetic_agent" in captured["stdin"]


def test_correlated_action_purge_uses_registered_scope(tmp_path, monkeypatch) -> None:
    captured = {}
    (tmp_path / "registry-registration-intent.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "cluster_ids": ["perf-cap-007"],
                "data_empty_before_registration": True,
            }
        )
    )

    def fake_control(*args, **kwargs):
        if args[0] == "get":
            return "api-a"
        captured["request"] = json.loads(args[-1])
        captured["checked"] = kwargs.get("check", True)
        return '{"total": 0}'

    monkeypatch.setattr(capacity_suite, "control", fake_control)

    suite.purge_scenario_rows("run-a", artifacts=tmp_path)

    assert captured == {
        "request": {
            "run_id": "run-a",
            "cluster_ids": ["perf-cap-007"],
            "cleanup": True,
            "force_nonterminal": True,
        },
        "checked": True,
    }


def test_correlated_action_database_audit_script_compiles(monkeypatch) -> None:
    captured = {}

    def fake_control(*args, **kwargs):
        if args[0] == "get":
            return "api-a"
        script = kwargs["stdin"].decode()
        compile(script, "<correlated-action-audit>", "exec")
        captured["script"] = script
        return "{}\n"

    monkeypatch.setattr(suite, "control", fake_control)

    assert suite.database_audit("run-a") == {}
    assert "escalation_rungs" in captured["script"]
    assert "validated_after_reboot" in captured["script"]
    assert "duplicate_workflow_steps" in captured["script"]
    assert "incident_by_id" in captured["script"]


def test_correlated_action_payloads_share_attempt_identity() -> None:
    observed_at = datetime.now(timezone.utc)
    identity = benchmark.attempt_identity("run-a", 3)
    observation = benchmark.observation_payload("perf-cap-003", identity, observed_at)
    heartbeat = benchmark.heartbeat_payload("perf-cap-003", identity, observed_at)
    weak = benchmark.xid_payload(
        xid=11,
        event_id="weak",
        cluster_id="perf-cap-003",
        identity=identity,
        observed_at=observed_at,
        profile_version="hyperpod-v1",
    )
    strong = benchmark.xid_payload(
        xid=48,
        event_id="strong",
        cluster_id="perf-cap-003",
        identity=identity,
        observed_at=observed_at,
        profile_version="hyperpod-v1",
    )

    assert observation["attempt_id"] == heartbeat["attempt_id"]
    assert observation["workload_ids"] == weak["affected_workload_ids"]
    assert weak["affected_workload_ids"] == strong["affected_workload_ids"]
    assert observation["restart_budget"] == 1
    assert weak["drill_id"] == strong["drill_id"] == "perf-capacity"
    assert weak["xid"] == 11
    assert strong["xid"] == 48
    sxid = benchmark.sxid_payload(
        event_id="sxid",
        cluster_id="perf-cap-003",
        identity=identity,
        observed_at=observed_at,
        profile_version="hyperpod-v1",
    )
    AttemptObservation(**observation)
    TrainingProgressHeartbeat(**heartbeat)
    XidEvent(**weak)
    XidEvent(**strong)
    SxidEvent(**sxid)


def test_correlated_action_ownership_requires_terminal_success() -> None:
    paths = []

    class Client:
        def get(self, path):
            paths.append(path)
            return {"terminal": True, "workflow_status": "SUCCEEDED"}

    report = benchmark.incident_ownership(Client(), "incident-a")

    assert report["terminal"] is True
    assert benchmark.ownership_succeeded(report), (
        "SUCCEEDED ownership report was not recognized"
    )
    assert paths == ["/v1/regional/executors/incident-ownership?incident_id=incident-a"]
    assert not benchmark.ownership_succeeded(
        {"terminal": False, "workflow_status": "RUNNING"}
    ), "non-terminal ownership report was treated as succeeded"
    assert not benchmark.ownership_succeeded(
        {"terminal": True, "workflow_status": "FAILED"}
    ), "failed ownership report was treated as succeeded"


def test_correlated_action_job_is_indexed_and_pinned() -> None:
    manifest = suite.scenario_job(
        clusters=32,
        run_id="run-a",
        scenario_max_seconds=1800,
        active_deadline_seconds=2100,
        executor_protocol_version=2,
        executor_artifact_sha256="a" * 64,
        executor_compatibility_digest="b" * 64,
        runtime_profile_version="profile-release-a",
        connection_secret="isolated-connection",
    )
    spec = manifest["spec"]
    container = spec["template"]["spec"]["containers"][0]
    environment = {item["name"]: item for item in container["env"]}

    assert spec["completionMode"] == "Indexed"
    assert spec["completions"] == 32
    assert spec["activeDeadlineSeconds"] == 2100
    assert environment["CORRELATED_ACTION_RUN_ID"]["value"] == "run-a"
    assert environment["SCENARIO_MAX_SECONDS"]["value"] == "1800"
    assert environment["EXECUTOR_PROTOCOL_VERSION"]["value"] == "2"
    assert environment["RUNTIME_PROFILE_VERSION"]["value"] == "profile-release-a"
    assert (
        environment["GPU_FAULT_CONTROL_PLANE_URL"]["valueFrom"]["secretKeyRef"]["name"]
        == "isolated-connection"
    )
    assert {item["key"] for item in spec["template"]["spec"]["tolerations"]} == {
        "node.kubernetes.io/unschedulable",
        "gpu-fault.io/quarantined",
    }


def test_correlated_action_terminal_audit_requires_full_drain() -> None:
    complete = {
        "workflow_count": 5,
        "terminal_workflow_count": 5,
        "command_count": 12,
        "terminal_command_count": 12,
        "permanent_budget_waiters": 0,
    }

    assert suite.audit_is_terminal(complete), "fully drained audit was not terminal"
    assert not suite.audit_is_terminal({**complete, "terminal_workflow_count": 4}), (
        "non-terminal workflow was ignored"
    )
    assert not suite.audit_is_terminal({**complete, "permanent_budget_waiters": 1}), (
        "permanent budget waiter was ignored"
    )


def test_correlated_action_verdict_requires_full_chain() -> None:
    summary = {
        "job_status": "Complete",
        "job_succeeded": 2,
        "job_failed": 0,
        "pod_results": [
            {
                "aggregate_shared_incident": True,
                "aggregate_shared_workflow": True,
                "strong_sent": True,
                "primary_reset_failed": True,
                "primary_reboot_succeeded": True,
                "reset_attempt_started": True,
                "reset_gpu_failed": True,
                "reset_reboot_succeeded": True,
                "idle_terminal": True,
                "duplicate_claims": 0,
                "claim_errors": 0,
                "result_errors": 0,
                "ownership_errors": 0,
            }
            for _ in range(2)
        ],
        "audit": {
            "incident_count": 4,
            # Clean-boundary preemption stays in the weak record: per cluster
            # one preempted record (its fabric reset escalates to an in-record
            # RESTART_NODE rung), one reset-lane record and the reset lane's
            # workflow-reboot-after successor.
            "workflow_count": 6,
            "command_count": 24,
            "weak_workflow_count": 2,
            "strong_workflow_count": 2,
            "in_record_preemption_count": 2,
            "cross_record_preemption_count": 0,
            "reset_gpu_workflow_count": 2,
            "reboot_workflow_count": 4,
            "preemption_pair_count": 2,
            "weak_superseded_count": 2,
            "strong_preempt_failed_fabric_reset_count": 2,
            "reset_gpu_failed_count": 2,
            "reboot_succeeded_count": 4,
            "reboot_bound_source_count": 4,
            "in_record_reboot_count": 2,
            "successor_reboot_count": 2,
            "orphan_reboot_count": 0,
            "validated_reboot_count": 4,
            "replace_rung_count": 0,
            "exhausted_branch_count": 0,
            "budget_refused_rung_count": 0,
            "exhausted_other_reason_count": 0,
            "unrecovered_exhausted_count": 0,
            "duplicate_reboot_rungs": 0,
            "unexpected_reboot_rung_sources": 0,
            "terminal_workflow_count": 6,
            "terminal_command_count": 24,
            "duplicate_idempotency_keys": 0,
            "duplicate_workflow_steps": 0,
            "fencing_mismatches": 0,
            "permanent_budget_waiters": 0,
        },
    }

    assert suite.verdict(summary, 2) == ("PASS", [])
    # A preemption that landed on an in-flight physical step is a separate
    # successor record (DESTR-016's shape); the audit counts it as well.
    cross = json.loads(json.dumps(summary))
    cross["audit"].update(
        {
            "workflow_count": 7,
            "terminal_workflow_count": 7,
            "in_record_preemption_count": 1,
            "cross_record_preemption_count": 1,
        }
    )
    assert suite.verdict(cross, 2) == ("PASS", [])
    # A reboot that no failed reset owns (a reboot-after created after the
    # in-record ladder was exhausted) fails the run.
    orphaned = json.loads(json.dumps(summary))
    orphaned["audit"].update({"orphan_reboot_count": 1})
    status, errors = suite.verdict(orphaned, 2)
    assert status == "FAIL"
    assert "orphan_reboot_count is nonzero" in errors
    # A node that never came back walked on to the REPLACE_NODE rung or
    # exhausted its ladder: the reboot did not release it.
    replaced = json.loads(json.dumps(summary))
    replaced["audit"].update(
        {
            "validated_reboot_count": 3,
            "replace_rung_count": 1,
            "exhausted_branch_count": 1,
            "exhausted_other_reason_count": 1,
            "unrecovered_exhausted_count": 1,
        }
    )
    status, errors = suite.verdict(replaced, 2)
    assert status == "FAIL"
    assert "validated_reboot_count does not equal 4" in errors
    assert "replace_rung_count is nonzero" in errors
    assert "unrecovered_exhausted_count is nonzero" in errors
    # A rung the remediation budget refused retires the branch; the record's
    # reboot-after successor still validates the node (live 2026-09-23).
    refused = json.loads(json.dumps(summary))
    refused["audit"].update(
        {
            "workflow_count": 7,
            "terminal_workflow_count": 7,
            "in_record_reboot_count": 1,
            "successor_reboot_count": 3,
            "exhausted_branch_count": 1,
            "budget_refused_rung_count": 1,
        }
    )
    assert suite.verdict(refused, 2) == ("PASS", [])
    summary["audit"]["weak_superseded_count"] = 1
    status, errors = suite.verdict(summary, 2)
    assert status == "FAIL"
    assert "weak_superseded_count does not equal 2" in errors
    summary["audit"]["weak_superseded_count"] = 2
    summary["audit"]["in_record_preemption_count"] = 1
    status, errors = suite.verdict(summary, 2)
    assert status == "FAIL"
    assert "preemption shape counts do not add up to 2" in errors


def test_correlated_action_verdict_rejects_incomplete_job_and_claim_errors() -> None:
    summary = {
        "job_status": "Failed",
        "job_succeeded": 1,
        "job_failed": 1,
        "pod_results": [
            {
                "aggregate_shared_incident": True,
                "aggregate_shared_workflow": True,
                "strong_sent": True,
                "primary_reset_failed": True,
                "primary_reboot_succeeded": True,
                "reset_attempt_started": True,
                "reset_gpu_failed": True,
                "reset_reboot_succeeded": True,
                "idle_terminal": True,
                "duplicate_claims": 0,
                "claim_errors": 1,
                "result_errors": 0,
                "ownership_errors": 0,
            }
        ],
        "audit": {
            "incident_count": 2,
            "workflow_count": 4,
            "command_count": 12,
            "weak_workflow_count": 1,
            "strong_workflow_count": 1,
            "in_record_preemption_count": 0,
            "cross_record_preemption_count": 1,
            "reset_gpu_workflow_count": 1,
            "reboot_workflow_count": 2,
            "preemption_pair_count": 1,
            "weak_superseded_count": 1,
            "strong_preempt_failed_fabric_reset_count": 1,
            "reset_gpu_failed_count": 1,
            "reboot_succeeded_count": 2,
            "reboot_bound_source_count": 2,
            "in_record_reboot_count": 1,
            "successor_reboot_count": 1,
            "orphan_reboot_count": 0,
            "validated_reboot_count": 2,
            "replace_rung_count": 0,
            "exhausted_branch_count": 0,
            "budget_refused_rung_count": 0,
            "exhausted_other_reason_count": 0,
            "unrecovered_exhausted_count": 0,
            "duplicate_reboot_rungs": 0,
            "unexpected_reboot_rung_sources": 0,
            "terminal_workflow_count": 4,
            "terminal_command_count": 12,
            "duplicate_idempotency_keys": 0,
            "duplicate_workflow_steps": 0,
            "fencing_mismatches": 0,
            "permanent_budget_waiters": 0,
        },
    }

    status, errors = suite.verdict(summary, 1)

    assert status == "FAIL"
    assert "scenario job did not complete" in errors
    assert "scenario claim errors are nonzero" in errors


def test_correlated_action_chain_preempts_and_escalates_fabric_reset() -> None:
    observed_at = datetime.now(timezone.utc)
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    identity = benchmark.attempt_identity("run-a", 0, "preempt")
    observation = AttemptObservation(
        **benchmark.observation_payload(
            "cluster-a", identity, observed_at, profile_version="simulated-v1"
        )
    )
    context.store.save_attempt_observation(observation)
    weak_event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-weak",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=observed_at,
            profile_version="simulated-v1",
        )
    )
    weak_decision = context.policy.evaluate_xid(weak_event)
    weak_incident, weak = context.orchestrator.ingest(weak_event, weak_decision)
    aggregate_event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-aggregate",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=observed_at,
            profile_version="simulated-v1",
        )
    )
    aggregate_decision = context.policy.evaluate_xid(aggregate_event)
    aggregate_incident, aggregate = context.orchestrator.ingest(
        aggregate_event, aggregate_decision
    )
    assert aggregate_incident.incident_id == weak_incident.incident_id
    assert aggregate.request_id == weak.request_id
    inherited_operations = {
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
    }
    inherited_indexes = [
        index
        for index, step in enumerate(aggregate.official_steps)
        if step.operation in inherited_operations
    ]
    running = copy_model(
        # The merge above bumped merge_revision; copy the merged row so the
        # version guard of save_workflow accepts the write (item B).
        aggregate,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="scenario-executor",
        completed_step_indexes=inherited_indexes,
        completed_operations=[
            aggregate.official_steps[index].operation for index in inherited_indexes
        ],
        step_executions=[
            workflow_step_execution(index, aggregate.official_steps[index].operation)
            for index in inherited_indexes
        ],
    )
    context.store.save_workflow(running)

    strong_event = SxidEvent(
        **benchmark.sxid_payload(
            event_id="corr-live-run-a-c000-strong",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=observed_at,
            profile_version="simulated-v1",
        )
    )
    strong_decision = context.policy.evaluate_sxid(strong_event)
    strong_incident, strong = context.orchestrator.ingest(strong_event, strong_decision)

    # Clean boundary (PREEMPT-002): the stronger fabric reset preempts inside
    # the record, exactly as the XID family does for a stronger XID. The weak
    # plan's pending steps are retired, its completed containment stays, and
    # the fabric reset runs as the node's successor branch behind one join.
    assert strong_incident.incident_id == weak_incident.incident_id
    assert strong.request_id == weak.request_id, (
        "a clean-boundary preemption stays in the record it preempts"
    )
    assert strong.dag_enabled, "the successor branch makes the plan a DAG"
    weak_pending = {
        index
        for index, step in enumerate(aggregate.official_steps)
        if index not in inherited_indexes
        and step.operation is not WorkflowOperation.RESTART_WORKLOAD
    }
    assert set(strong.superseded_step_indexes) == weak_pending, (
        "every not-yet-started step of the weak plan is retired"
    )
    assert set(inherited_indexes) <= set(strong.completed_step_indexes)
    assert not set(inherited_indexes) & set(strong.superseded_step_indexes), (
        "completed containment is history, not retired work"
    )
    for index in inherited_indexes:
        assert (
            strong.official_steps[index].operation
            is aggregate.official_steps[index].operation
        )
    successor_branch = f"branch:{identity['node_id']}:successor:1"
    successor_operations = [
        step.operation
        for step in strong.official_steps
        if step.branch_id == successor_branch
    ]
    assert {
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    } <= set(successor_operations), successor_operations
    assert any(event.kind is WorkflowEventKind.PREEMPTION for event in strong.events), (
        "the record says who preempted whom"
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in strong.official_steps
        )
        == 1
    ), "one join restart for the attempt"
    reset_index = next(
        index
        for index, step in enumerate(strong.official_steps)
        if step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
        and step.branch_id == successor_branch
    )
    failed = copy_model(
        strong,
        status=WorkflowStatus.FAILED,
        step_executions=[
            *strong.step_executions,
            workflow_step_execution(
                reset_index,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                WorkflowStepStatus.FAILED,
                error="synthetic reset failure",
            ),
        ],
    )
    context.store.save_workflow(failed)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None
    incident, reboot = escalated
    assert incident.effective_action is RecoveryAction.REBOOT_NODE
    assert reboot.request_id == f"workflow-reboot-after-{failed.request_id}"
    source_restart = next(
        step
        for step in failed.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    reboot_restart = next(
        step
        for step in reboot.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert {
        name: reboot_restart.parameters[name] for name in RESTART_SAFETY_PARAMETERS
    } == {name: source_restart.parameters[name] for name in RESTART_SAFETY_PARAMETERS}
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_NODE
            for step in reboot.official_steps
        )
        == 1
    ), "fabric reset failure did not create one node reboot step"


def test_correlated_action_reset_gpu_failure_escalates_to_reboot() -> None:
    observed_at = datetime.now(timezone.utc)
    context = build_context()
    identity = benchmark.attempt_identity("run-a", 0, "reset")
    context.store.save_attempt_observation(
        AttemptObservation(
            **benchmark.observation_payload(
                "cluster-a", identity, observed_at, profile_version="simulated-v1"
            )
        )
    )
    event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-reset",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=observed_at,
            profile_version="simulated-v1",
        )
    )
    decision = context.policy.evaluate_xid(event)
    _incident, workflow = context.orchestrator.ingest(event, decision)
    reset_index = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    failed = copy_model(
        workflow,
        status=WorkflowStatus.FAILED,
        step_executions=[
            workflow_step_execution(
                reset_index,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="synthetic reset failure",
            )
        ],
    )
    context.store.save_workflow(failed)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None
    incident, reboot = escalated
    assert incident.effective_action is RecoveryAction.REBOOT_NODE
    assert reboot.request_id == f"workflow-reboot-after-{failed.request_id}"
    source_restart = next(
        step
        for step in failed.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    reboot_restart = next(
        step
        for step in reboot.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert {
        name: reboot_restart.parameters[name] for name in RESTART_SAFETY_PARAMETERS
    } == {name: source_restart.parameters[name] for name in RESTART_SAFETY_PARAMETERS}
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_NODE
            for step in reboot.official_steps
        )
        == 1
    ), "GPU reset failure did not create one node reboot step"


def test_correlated_action_escalation_blocks_without_restart_context() -> None:
    observed_at = datetime.now(timezone.utc)
    context = build_context()
    identity = benchmark.attempt_identity("run-a", 0, "missing-restart-context")
    context.store.save_attempt_observation(
        AttemptObservation(
            **benchmark.observation_payload(
                "cluster-a", identity, observed_at, profile_version="simulated-v1"
            )
        )
    )
    event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-missing-restart-context",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=observed_at,
            profile_version="simulated-v1",
        )
    )
    decision = context.policy.evaluate_xid(event)
    _incident, workflow = context.orchestrator.ingest(event, decision)
    reset_index = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    failed = copy_model(
        workflow,
        status=WorkflowStatus.FAILED,
        official_steps=[
            (
                step.model_copy(update={"parameters": {}})
                if step.operation is WorkflowOperation.RESTART_WORKLOAD
                else step
            )
            for step in workflow.official_steps
        ],
        step_executions=[
            workflow_step_execution(
                reset_index,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="synthetic reset failure",
            )
        ],
    )
    context.store.save_workflow(failed)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None
    _incident, reboot = escalated
    assert reboot.status is WorkflowStatus.BLOCKED
    assert reboot.blocked_reasons == [
        "failed workflow has no complete restart safety context"
    ]


def test_correlated_action_escalation_blocks_if_any_restart_context_is_incomplete() -> (
    None
):
    observed_at = datetime.now(timezone.utc)
    context = build_context()
    identity = benchmark.attempt_identity("run-a", 0, "mixed-restart-context")
    context.store.save_attempt_observation(
        AttemptObservation(
            **benchmark.observation_payload(
                "cluster-a", identity, observed_at, profile_version="simulated-v1"
            )
        )
    )
    event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-mixed-restart-context",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=observed_at,
            profile_version="simulated-v1",
        )
    )
    decision = context.policy.evaluate_xid(event)
    _incident, workflow = context.orchestrator.ingest(event, decision)
    reset_index = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    restart_step = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    incomplete_restart = restart_step.model_copy(
        update={
            "parameters": {
                name: value
                for name, value in restart_step.parameters.items()
                if name != "source_attempt_id"
            }
        }
    )
    failed = copy_model(
        workflow,
        status=WorkflowStatus.FAILED,
        official_steps=[*workflow.official_steps, incomplete_restart],
        step_executions=[
            workflow_step_execution(
                reset_index,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="synthetic reset failure",
            )
        ],
    )
    context.store.save_workflow(failed)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None
    _incident, reboot = escalated
    assert reboot.status is WorkflowStatus.BLOCKED
    assert reboot.blocked_reasons == [
        "failed workflow has no complete restart safety context"
    ]


def _escalated_record(request_id, *, incident_id, failed_operation, status="SUCCEEDED"):
    """One §8.3 record after its failed reset escalated in place to a reboot."""

    node = "corr-node-run-a-c000-preempt"
    steps = [
        ("MARK_UNSCHEDULABLE", "branch:initial"),
        ("STOP_WORKLOADS", "shared"),
        ("RESET_GPU", "branch:initial"),
        ("RESTART_WORKLOAD", "join"),
        (failed_operation, f"branch:{node}:successor:1"),
        ("RESTART_NODE", f"branch:{node}:successor:2"),
        ("VALIDATE_GPU", f"branch:{node}:successor:2"),
        ("VALIDATE_HOST", f"branch:{node}:successor:2"),
        ("VALIDATE_FABRIC", f"branch:{node}:successor:2"),
        ("RESTORE_SCHEDULING", f"branch:{node}:successor:2"),
    ]
    executions = [
        {"step_index": 0, "operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"},
        {"step_index": 1, "operation": "STOP_WORKLOADS", "status": "SUCCEEDED"},
        {"step_index": 4, "operation": failed_operation, "status": "FAILED"},
        {"step_index": 5, "operation": "RESTART_NODE", "status": "SUCCEEDED"},
        {"step_index": 6, "operation": "VALIDATE_GPU", "status": "SUCCEEDED"},
        {"step_index": 7, "operation": "VALIDATE_HOST", "status": "SUCCEEDED"},
        {"step_index": 8, "operation": "VALIDATE_FABRIC", "status": "SUCCEEDED"},
        {"step_index": 9, "operation": "RESTORE_SCHEDULING", "status": "SUCCEEDED"},
        {"step_index": 3, "operation": "RESTART_WORKLOAD", "status": "SUCCEEDED"},
    ]
    in_record = failed_operation == "RESET_ALL_GPUS_NVSWITCHES"
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": status,
        "fencing_token": 1,
        "official_steps": [
            {"operation": operation, "branch_id": branch} for operation, branch in steps
        ],
        "step_executions": executions,
        "superseded_step_indexes": [2] if in_record else [],
        "completed_step_indexes": [0, 1, 3, 4, 5, 6, 7, 8, 9],
        "exhausted_branch_ids": [],
        "events": [
            *([{"kind": "PREEMPTION", "code": "PLAN_REWRITE"}] if in_record else []),
            {
                "kind": "BRANCH_ESCALATION",
                "code": "BRANCH_ESCALATED",
                "details": {
                    "from_operation": failed_operation,
                    "to_operation": "RESTART_NODE",
                    "rung_count": 1,
                },
            },
        ],
    }


def _run_audit_script(workflows, commands):
    """Execute the audit's classification on in-memory rows (no database)."""

    import io
    from contextlib import redirect_stdout
    from datetime import datetime, timezone

    namespace = {
        "json": json,
        "datetime": datetime,
        "timezone": timezone,
        "workflows": workflows,
        "commands": commands,
        "incidents": [{"incident_id": item["incident_id"]} for item in workflows],
        "incident_by_id": {
            item["incident_id"]: {"fencing_token": 1} for item in workflows
        },
    }
    output = io.StringIO()
    with redirect_stdout(output):
        exec(suite.database_audit_script_suffix(), namespace)  # noqa: S102
    return json.loads(output.getvalue().strip().splitlines()[-1])


def _failed_reset_lane(request_id, *, incident_id):
    """The reset lane: a plain record whose RESET_GPU failed, and the
    workflow-reboot-after successor the whole-workflow escalation created."""

    failed = {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": "FAILED",
        "fencing_token": 1,
        "official_steps": [
            {"operation": operation, "branch_id": None}
            for operation in (
                "MARK_UNSCHEDULABLE",
                "STOP_WORKLOADS",
                "RESET_GPU",
                "RESTORE_GPU_SERVICES",
                "VALIDATE_GPU",
            )
        ],
        "step_executions": [
            {"step_index": 0, "operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"},
            {"step_index": 1, "operation": "STOP_WORKLOADS", "status": "SUCCEEDED"},
            {"step_index": 2, "operation": "RESET_GPU", "status": "FAILED"},
        ],
        "superseded_step_indexes": [],
        "exhausted_branch_ids": [],
        "events": [],
    }
    successor = {
        "request_id": f"workflow-reboot-after-{request_id}",
        "incident_id": f"inc-reboot-after-{request_id}",
        "status": "SUCCEEDED",
        "fencing_token": 1,
        "official_steps": [
            {"operation": operation, "branch_id": None}
            for operation in (
                "MARK_UNSCHEDULABLE",
                "QUARANTINE",
                "STOP_WORKLOADS",
                "RESTART_NODE",
                "VALIDATE_GPU",
                "VALIDATE_HOST",
                "VALIDATE_FABRIC",
                "RESTORE_SCHEDULING",
                "RESTART_WORKLOAD",
            )
        ],
        "step_executions": [
            {"step_index": index, "operation": operation, "status": "SUCCEEDED"}
            for index, operation in enumerate(
                (
                    "MARK_UNSCHEDULABLE",
                    "QUARANTINE",
                    "STOP_WORKLOADS",
                    "RESTART_NODE",
                    "VALIDATE_GPU",
                    "VALIDATE_HOST",
                    "VALIDATE_FABRIC",
                    "RESTORE_SCHEDULING",
                    "RESTART_WORKLOAD",
                )
            )
        ],
        "superseded_step_indexes": [],
        "exhausted_branch_ids": [],
        "events": [],
    }
    return failed, successor


def _pod_result() -> dict:
    return {
        key: True
        for key in (
            "aggregate_shared_incident",
            "aggregate_shared_workflow",
            "strong_sent",
            "primary_reset_failed",
            "primary_reboot_succeeded",
            "reset_attempt_started",
            "reset_gpu_failed",
            "reset_reboot_succeeded",
            "idle_terminal",
        )
    }


def test_correlated_action_audit_binds_each_failed_reset_to_one_reboot() -> None:
    # Live 2026-09-23 (1 cluster): the preempted DAG record escalated its
    # failed fabric reset to an in-record RESTART_NODE rung; the plain reset
    # lane failed and the whole-workflow escalation created its
    # workflow-reboot-after successor. Both reboots validated the returned
    # node.
    preempted = _escalated_record(
        "workflow-a",
        incident_id="incident-a",
        failed_operation="RESET_ALL_GPUS_NVSWITCHES",
    )
    reset_lane, successor = _failed_reset_lane("workflow-b", incident_id="incident-b")
    workflows = [preempted, reset_lane, successor]
    commands = [
        {
            "workflow_request_id": item["request_id"],
            "incident_id": item["incident_id"],
            "step_index": execution["step_index"],
            "idempotency_key": f"{item['request_id']}:{execution['step_index']}",
            "fencing_token": 1,
            "status": "SUCCEEDED",
        }
        for item in workflows
        for execution in item["step_executions"]
    ]

    audit = _run_audit_script(workflows, commands)

    assert audit["workflow_count"] == 3
    assert audit["in_record_preemption_count"] == 1
    assert audit["cross_record_preemption_count"] == 0
    assert audit["weak_superseded_count"] == 1
    assert audit["strong_preempt_failed_fabric_reset_count"] == 1
    assert audit["reset_gpu_workflow_count"] == 1
    assert audit["reset_gpu_failed_count"] == 1
    assert audit["reboot_workflow_count"] == 2
    assert audit["reboot_bound_source_count"] == 2
    assert audit["in_record_reboot_count"] == 1
    assert audit["successor_reboot_count"] == 1
    assert audit["orphan_reboot_count"] == 0
    assert audit["reboot_succeeded_count"] == 2
    assert audit["validated_reboot_count"] == 2
    assert audit["replace_rung_count"] == 0
    assert audit["exhausted_branch_count"] == 0
    assert audit["duplicate_reboot_rungs"] == 0
    assert audit["unexpected_reboot_rung_sources"] == 0
    assert audit["terminal_workflow_count"] == 3
    assert audit["fencing_mismatches"] == 0
    summary = {
        "job_status": "Complete",
        "job_succeeded": 1,
        "job_failed": 0,
        "pod_results": [_pod_result()],
        "audit": {**audit, "incident_count": 2},
    }
    assert suite.verdict(summary, 1) == ("PASS", [])


def test_correlated_action_audit_flags_a_node_that_never_returned() -> None:
    # Live 2026-09-23: the synthetic node published nothing after its reboot,
    # VALIDATE_GPU waited to the step cap, the branch walked on to
    # REPLACE_NODE, exhausted, and the failed record then got a
    # workflow-reboot-after successor nobody's failed reset owns.
    stranded = _escalated_record(
        "workflow-a",
        incident_id="incident-a",
        failed_operation="RESET_ALL_GPUS_NVSWITCHES",
    )
    stranded["step_executions"] = [
        item
        for item in stranded["step_executions"]
        if item["step_index"] not in {7, 8, 9}
    ] + [{"step_index": 6, "operation": "VALIDATE_GPU", "status": "FAILED"}]
    stranded["events"].append(
        {
            "kind": "BRANCH_ESCALATION",
            "code": "BRANCH_ESCALATED",
            "details": {
                "from_operation": "VALIDATE_GPU",
                "to_operation": "REPLACE_NODE",
                "rung_count": 2,
            },
        }
    )
    stranded["events"].append(
        {
            "kind": "BRANCH_ESCALATION",
            "code": "BRANCH_EXHAUSTED",
            "reason": "VALIDATE_GPU failed (past the 600s per-step cap); no further rung",
            "details": {"exhausted": True, "rung_count": 2},
        }
    )
    stranded["exhausted_branch_ids"] = [
        "branch:corr-node-run-a-c000-preempt:successor:3"
    ]
    stranded["status"] = "FAILED"
    _, late_successor = _failed_reset_lane("workflow-a", incident_id="incident-a")

    audit = _run_audit_script([stranded, late_successor], [])

    assert audit["reboot_workflow_count"] == 2
    assert audit["reboot_bound_source_count"] == 1
    assert audit["in_record_reboot_count"] == 1
    assert audit["successor_reboot_count"] == 0
    assert audit["orphan_reboot_count"] == 1
    assert audit["reboot_succeeded_count"] == 0
    assert audit["validated_reboot_count"] == 0
    assert audit["replace_rung_count"] == 1
    assert audit["exhausted_branch_count"] == 1
    assert audit["budget_refused_rung_count"] == 0
    assert audit["exhausted_other_reason_count"] == 1
    assert audit["unrecovered_exhausted_count"] == 1


def test_correlated_action_audit_accepts_a_budget_refused_rung() -> None:
    # Live 2026-09-23 (perf-cap-005 of 32): the fabric reset failed, the
    # cluster remediation budget could not take the RESTART_NODE rung, the
    # branch was exhausted, the record failed and its workflow-reboot-after
    # successor rebooted and validated the node.
    refused = _escalated_record(
        "workflow-a",
        incident_id="incident-a",
        failed_operation="RESET_ALL_GPUS_NVSWITCHES",
        status="FAILED",
    )
    # The refused rung's steps are never appended: the plan ends with the
    # failed fabric reset and the branch is retired.
    refused["official_steps"] = refused["official_steps"][:5]
    refused["step_executions"] = [
        item for item in refused["step_executions"] if item["step_index"] <= 4
    ]
    refused["events"] = [
        {"kind": "PREEMPTION", "code": "PREEMPTED"},
        {
            "kind": "BRANCH_ESCALATION",
            "code": "BRANCH_EXHAUSTED",
            "reason": (
                "RESET_ALL_GPUS_NVSWITCHES failed on corr-node-run-a-c000-preempt "
                "(synthetic reset failure); escalating to RESTART_NODE; the cluster "
                "remediation budget cannot take the next rung (reboot scope full)"
            ),
            "details": {"exhausted": True, "rung_count": 0},
        },
    ]
    refused["exhausted_branch_ids"] = [
        "branch:corr-node-run-a-c000-preempt:successor:1"
    ]
    _, refused_successor = _failed_reset_lane("workflow-a", incident_id="incident-a")
    reset_lane, successor = _failed_reset_lane("workflow-b", incident_id="incident-b")

    audit = _run_audit_script([refused, refused_successor, reset_lane, successor], [])

    assert audit["in_record_preemption_count"] == 1
    assert audit["strong_preempt_failed_fabric_reset_count"] == 1
    assert audit["reboot_bound_source_count"] == 2
    assert audit["in_record_reboot_count"] == 0
    assert audit["successor_reboot_count"] == 2
    assert audit["orphan_reboot_count"] == 0
    assert audit["validated_reboot_count"] == 2
    assert audit["exhausted_branch_count"] == 1
    assert audit["budget_refused_rung_count"] == 1
    assert audit["exhausted_other_reason_count"] == 0
    assert audit["unrecovered_exhausted_count"] == 0
    assert audit["replace_rung_count"] == 0
    summary = {
        "job_status": "Complete",
        "job_succeeded": 1,
        "job_failed": 0,
        "pod_results": [_pod_result()],
        "audit": {
            **audit,
            "incident_count": 2,
            "command_count": 1,
            "terminal_command_count": 1,
        },
    }
    assert suite.verdict(summary, 1) == ("PASS", [])


def test_correlated_scenario_binds_both_reboot_shapes() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.posts: list[tuple[str, dict]] = []

        def post(self, path: str, payload: dict) -> dict:
            self.posts.append((path, payload))
            return {}

    client = FakeClient()
    state = benchmark.ScenarioState(executor_id="correlated-action-000")
    state.strong_workflow_id = "workflow-a"
    state.strong_sent = True
    state.reset_workflow_id = "workflow-b"

    def restart_command(workflow_id: str, incident_id: str) -> dict:
        return {
            "command_id": f"command-{workflow_id}",
            "lease_token": "lease",
            "workflow": {"request_id": workflow_id},
            "incident": {"incident_id": incident_id},
            "step": {"operation": "RESTART_NODE", "node_ids": ["node-a"]},
        }

    for workflow_id, incident_id in (
        ("workflow-a", "incident-a"),
        ("workflow-reboot-after-workflow-b", "inc-reboot-after-workflow-b"),
    ):
        benchmark.handle_scenario_command(
            client,
            restart_command(workflow_id, incident_id),
            state,
            weak_workflow_id="workflow-a",
            strong_event_id="strong",
            cluster_id="perf-cap-000",
            primary_identity=benchmark.attempt_identity("run-a", 0, "preempt"),
            observed_at=datetime.now(timezone.utc),
            profile_version="hyperpod-v1",
        )

    assert state.primary_reboot_workflow_id == "workflow-a"
    assert state.primary_reboot_incident_id == "incident-a"
    assert state.reset_reboot_workflow_id == "workflow-reboot-after-workflow-b"
    assert state.reset_reboot_incident_id == "inc-reboot-after-workflow-b"
    assert state.operations == {"RESTART_NODE": 2}
    assert all(payload["status"] == "SUCCEEDED" for _, payload in client.posts), (
        "the reboot rung itself is reported as succeeded"
    )
