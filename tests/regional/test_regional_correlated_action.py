from __future__ import annotations

import json
from datetime import datetime, timezone

from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestrator import IncidentOrchestrator
from gpu_fault.policy import SxidEvent, XidEvent
from gpu_fault.training_models import TrainingProgressHeartbeat
from gpu_fault.watcher import AttemptObservation
from scripts.perf import benchmark_correlated_action_scenario as benchmark
from scripts.perf import regional_correlated_action_suite as suite
from tests._builders import build_context, copy_model, workflow_step_execution

NOW = datetime.now(timezone.utc)


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


def test_correlated_action_payloads_share_attempt_identity() -> None:
    identity = benchmark.attempt_identity("run-a", 3)
    observation = benchmark.observation_payload("perf-cap-003", identity, NOW)
    heartbeat = benchmark.heartbeat_payload("perf-cap-003", identity, NOW)
    weak = benchmark.xid_payload(
        xid=11,
        event_id="weak",
        cluster_id="perf-cap-003",
        identity=identity,
        observed_at=NOW,
        profile_version="hyperpod-v1",
    )
    strong = benchmark.xid_payload(
        xid=48,
        event_id="strong",
        cluster_id="perf-cap-003",
        identity=identity,
        observed_at=NOW,
        profile_version="hyperpod-v1",
    )

    assert observation["attempt_id"] == heartbeat["attempt_id"]
    assert observation["workload_ids"] == weak["affected_workload_ids"]
    assert weak["affected_workload_ids"] == strong["affected_workload_ids"]
    assert weak["drill_id"] == strong["drill_id"] == "perf-capacity"
    assert weak["xid"] == 11
    assert strong["xid"] == 48
    sxid = benchmark.sxid_payload(
        event_id="sxid",
        cluster_id="perf-cap-003",
        identity=identity,
        observed_at=NOW,
        profile_version="hyperpod-v1",
    )
    AttemptObservation(**observation)
    TrainingProgressHeartbeat(**heartbeat)
    XidEvent(**weak)
    XidEvent(**strong)
    SxidEvent(**sxid)


def test_correlated_action_job_is_indexed_and_pinned() -> None:
    manifest = suite.scenario_job(
        clusters=32,
        run_id="run-a",
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
    assert environment["CORRELATED_ACTION_RUN_ID"]["value"] == "run-a"
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


def test_correlated_action_verdict_requires_full_chain() -> None:
    summary = {
        "pod_results": [
            {
                "strong_sent": True,
                "primary_reset_failed": True,
                "reset_attempt_started": True,
                "reset_gpu_failed": True,
                "idle_terminal": True,
                "duplicate_claims": 0,
                "result_errors": 0,
            }
            for _ in range(2)
        ],
        "audit": {
            "weak_superseded_count": 2,
            "strong_preempt_failed_fabric_reset_count": 2,
            "reset_gpu_failed_count": 2,
            "reboot_succeeded_count": 4,
            "workflow_count": 10,
            "terminal_workflow_count": 10,
            "command_count": 24,
            "terminal_command_count": 24,
            "duplicate_idempotency_keys": 0,
            "fencing_mismatches": 0,
            "permanent_budget_waiters": 0,
        },
    }

    assert suite.verdict(summary, 2) == ("PASS", [])
    summary["audit"]["weak_superseded_count"] = 1
    status, errors = suite.verdict(summary, 2)
    assert status == "FAIL"
    assert "weak_superseded_count does not equal cluster count" in errors


def test_correlated_action_chain_preempts_and_escalates_fabric_reset() -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    identity = benchmark.attempt_identity("run-a", 0, "preempt")
    observation = AttemptObservation(
        **benchmark.observation_payload(
            "cluster-a", identity, NOW, profile_version="simulated-v1"
        )
    )
    context.store.save_attempt_observation(observation)
    weak_event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-weak",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=NOW,
            profile_version="simulated-v1",
        )
    )
    weak_decision = context.policy.evaluate_xid(weak_event)
    _weak_incident, weak = context.orchestrator.ingest(weak_event, weak_decision)
    inherited_operations = {
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.STOP_WORKLOADS,
    }
    inherited_indexes = [
        index
        for index, step in enumerate(weak.official_steps)
        if step.operation in inherited_operations
    ]
    running = copy_model(
        weak,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="scenario-executor",
        completed_step_indexes=inherited_indexes,
        completed_operations=[
            weak.official_steps[index].operation for index in inherited_indexes
        ],
        step_executions=[
            workflow_step_execution(index, weak.official_steps[index].operation)
            for index in inherited_indexes
        ],
    )
    context.store.save_workflow(running)

    strong_event = SxidEvent(
        **benchmark.sxid_payload(
            event_id="corr-live-run-a-c000-strong",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=NOW,
            profile_version="simulated-v1",
        )
    )
    strong_decision = context.policy.evaluate_sxid(strong_event)
    _strong_incident, strong = context.orchestrator.ingest(
        strong_event, strong_decision
    )

    assert strong.request_id != weak.request_id
    assert strong.predecessor_workflow_id == weak.request_id
    assert strong.preempt_predecessor is True
    assert strong.inherited_step_indexes, (
        "strong workflow did not inherit completed predecessor steps"
    )
    reset_index = next(
        index
        for index, step in enumerate(strong.official_steps)
        if step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
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
    assert any(
        step.operation is WorkflowOperation.RESTART_NODE
        for step in reboot.official_steps
    ), "fabric reset failure did not escalate to node reboot"


def test_correlated_action_reset_gpu_failure_escalates_to_reboot() -> None:
    context = build_context()
    identity = benchmark.attempt_identity("run-a", 0, "reset")
    context.store.save_attempt_observation(
        AttemptObservation(
            **benchmark.observation_payload(
                "cluster-a", identity, NOW, profile_version="simulated-v1"
            )
        )
    )
    event = XidEvent(
        **benchmark.xid_payload(
            xid=48,
            event_id="corr-live-run-a-c000-reset",
            cluster_id="cluster-a",
            identity=identity,
            observed_at=NOW,
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
    assert any(
        step.operation is WorkflowOperation.RESTART_NODE
        for step in reboot.official_steps
    ), "GPU reset failure did not escalate to node reboot"
