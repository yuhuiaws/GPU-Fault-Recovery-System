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


def test_correlated_action_purge_removes_policy_decisions(monkeypatch) -> None:
    captured = {}

    def fake_control(*args, **kwargs):
        if args[0] == "get":
            return "api-a"
        captured["stdin"] = kwargs["stdin"].decode()
        return ""

    monkeypatch.setattr(suite, "control", fake_control)

    suite.purge_scenario_rows("run-a")

    assert "kind='xid_policy_decision'" in captured["stdin"]
    assert "corr-live-run-a-%" in captured["stdin"]


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
    assert "deterministic_reboot_sources" in captured["script"]
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
            # one preempted record, its reboot successor, the reset-lane
            # record and its reboot successor.
            "workflow_count": 8,
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
            "reboot_successor_pair_count": 4,
            "unique_reboot_predecessor_count": 4,
            "duplicate_reboot_successors": 0,
            "unexpected_reboot_predecessors": 0,
            "terminal_workflow_count": 8,
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
            "workflow_count": 9,
            "terminal_workflow_count": 9,
            "in_record_preemption_count": 1,
            "cross_record_preemption_count": 1,
        }
    )
    assert suite.verdict(cross, 2) == ("PASS", [])
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
            "workflow_count": 5,
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
            "reboot_successor_pair_count": 2,
            "unique_reboot_predecessor_count": 2,
            "duplicate_reboot_successors": 0,
            "unexpected_reboot_predecessors": 0,
            "terminal_workflow_count": 5,
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
