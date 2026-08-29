from __future__ import annotations

from tests._builders import (
    active_workflow_executor,
    asgi_client,
    attempt_observation,
    build_store,
    container_observation,
    copy_model,
    fault_incident,
    node_action_result,
    workflow_request,
    workflow_step,
)

from ._support import (
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    NOW,
    ApplicationContext,
    FakeAdapter,
    FakeCoreApi,
    GpuValidationAdapter,
    HyperPodLifecycleStepAdapter,
    IncidentState,
    KubernetesWorkflowAdapter,
    NodeActionWorkflowAdapter,
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    RecordingOwnershipProvider,
    SqliteStore,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowLeaseError,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    _hung_signal_without_dump,
    _real_flight_signal,
    _storeless_isolation_context,
    context_module,
    pytest,
    timedelta,
    workflow_state,
)


def test_hung_triage_rewrites_bundle_to_culprit_and_control() -> None:
    store = build_store()
    store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            NOW,
            expected_critical_ranks=3,
            containers=[
                container_observation(
                    "pod-a", "worker-0", 0, "node-a", host_pid=100, gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-b", "worker-1", 1, "node-b", host_pid=200, gpu_uuids=["GPU-b"]
                ),
                container_observation(
                    "pod-c", "worker-2", 2, "node-c", host_pid=300, gpu_uuids=["GPU-c"]
                ),
            ],
            workload_ids=["training/job/job-a"],
            runtime_profile_version="active-v1",
        )
    )
    incident = fault_incident(
        "incident-hung",
        "event-hung",
        "NODE_HEALTH",
        node_ids=["node-a", "node-b", "node-c"],
        job_id="job-a",
        attempt_id="attempt-a",
        policy_version="v1",
        policy_source="SITE_EFA_TRAFFIC",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
    )
    workflow = workflow_request(
        "workflow-hung",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        1,
        dag_enabled=True,
        dag_revision=1,
        official_steps=[
            workflow_step(
                WorkflowOperation.FREEZE_EVIDENCE,
                node_ids=["node-a", "node-b", "node-c"],
            ),
            workflow_step(
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                node_ids=["node-a", "node-b", "node-c"],
                depends_on_step_indexes=[0],
            ),
            workflow_step(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                node_ids=["node-a", "node-b", "node-c"],
                depends_on_step_indexes=[1],
                parameters={"capture_process_state": False},
            ),
            workflow_step(
                WorkflowOperation.VALIDATE_FABRIC,
                node_ids=["node-a", "node-b", "node-c"],
                depends_on_step_indexes=[1],
            ),
        ],
    )
    executor = active_workflow_executor(store, [], frozenset())
    details = {
        "node_results": {
            "node-a": {
                "ranks": [
                    {
                        "pid": 100,
                        "rank": 0,
                        "gpu_uuid": "GPU-a",
                        "flight_recorder": {
                            "last_entry": {
                                "pg_name": "default",
                                "collective_seq_id": 9,
                                "time_discovered_started": "t",
                            }
                        },
                    }
                ]
            },
            "node-b": {
                "ranks": [
                    {
                        "pid": 200,
                        "rank": 1,
                        "gpu_uuid": "GPU-b",
                        "flight_recorder": {
                            "last_entry": {
                                "pg_name": "default",
                                "collective_seq_id": 10,
                                "time_discovered_started": "t",
                            }
                        },
                    }
                ]
            },
            "node-c": {
                "ranks": [
                    {
                        "pid": 300,
                        "rank": 2,
                        "gpu_uuid": "GPU-c",
                        "flight_recorder": {
                            "last_entry": {
                                "pg_name": "default",
                                "collective_seq_id": 10,
                                "time_discovered_started": "t",
                            }
                        },
                    }
                ]
            },
        },
        "undetermined_nodes": [],
    }

    rewritten = executor._rewrite_hung_triage_bundle(
        workflow, incident, triage_index=1, triage_details=details
    )

    bundle = rewritten.official_steps[2]
    assert rewritten.dag_revision == 2
    assert rewritten.official_steps[3].depends_on_step_indexes == [1]
    assert bundle.node_ids == ["node-a", "node-b"]
    assert bundle.gpu_uuids == ["GPU-a", "GPU-b"]
    assert bundle.parameters["capture_process_state"] is True
    assert bundle.parameters["target_pids_by_node"] == {
        "node-a": [100],
        "node-b": [200],
    }
    assert bundle.parameters["max_processes"] == 2
    assert bundle.parameters["hung_triage_decision"]["classification"] == "CONFIRMED"
    assert "not_sampled_nodes" not in bundle.parameters["hung_triage_decision"]

    partial = executor._rewrite_hung_triage_bundle(
        workflow,
        incident,
        triage_index=1,
        triage_details={**details, "not_sampled_nodes": ["node-d", "node-e"]},
    )

    # A verdict reached over part of the attempt has to say so: the
    # nodes that were never attached to are not evidence of health.
    assert partial.official_steps[2].parameters["hung_triage_decision"][
        "not_sampled_nodes"
    ] == ["node-d", "node-e"]


def test_hung_triage_many_lagging_ranks_marks_fabric_suspected() -> None:
    signals = [
        {
            "rank": rank,
            "node_id": f"node-{rank}",
            "flight_recorder": {
                "last_entry": {
                    "pg_name": "default",
                    "collective_seq_id": 9 if rank < 2 else 10,
                    "time_discovered_started": "t",
                }
            },
        }
        for rank in range(20)
    ]

    decision = ProductionWorkflowExecutor._classify_hung_signals(
        signals, undetermined_nodes=[]
    )

    assert decision["classification"] == "FABRIC_SUSPECTED"
    assert decision["culprit_ranks"] == []


def test_hung_triage_orders_weak_candidates_by_efa_zero_time() -> None:
    signals = [
        {
            "rank": rank,
            "node_id": f"node-{rank}",
            "flight_recorder": {
                "last_entry": {
                    "pg_name": "default",
                    "collective_seq_id": 10,
                    "time_discovered_started": "t",
                }
            },
            "proc": {
                "cpu_ticks_delta": 10,
                "thread_states": {"D": 1 if rank < 2 else 0},
            },
            "gpu": {"utilization_gpu_percent": 80},
        }
        for rank in range(4)
    ]

    decision = ProductionWorkflowExecutor._classify_hung_signals(
        signals,
        undetermined_nodes=[],
        efa_zero_pending_at_by_node={
            "node-0": NOW,
            "node-1": NOW - timedelta(seconds=5),
        },
    )

    assert decision["classification"] == "WEAK"
    assert decision["culprit_ranks"][:2] == [1, 0]


def test_hung_triage_confirms_the_rank_that_stopped_enqueueing() -> None:
    signals = [
        _real_flight_signal(rank, seq=21, state="scheduled") for rank in range(1, 24)
    ]
    signals.insert(0, _real_flight_signal(0, seq=20, state="completed"))
    signals[0]["gpu"] = {"utilization_gpu_percent": 0.0}

    decision = ProductionWorkflowExecutor._classify_hung_signals(
        signals, undetermined_nodes=[]
    )

    assert decision["classification"] == "CONFIRMED"
    assert decision["culprit_ranks"] == [0]
    assert decision["mode_collective_seq_id"] == 21
    assert decision["pg_name"] == "0:default_pg"


def test_hung_triage_ignores_missing_start_times_shared_by_all() -> None:
    signals = [
        _real_flight_signal(rank, seq=21, state="scheduled") for rank in range(24)
    ]
    signals[0]["gpu"] = {"utilization_gpu_percent": 0.0}
    signals[0]["proc"] = {
        "cpu_ticks_delta": 1,
        "thread_states": {"S": 9},
        "voluntary_ctxt_switches_delta": 0,
        "wchan_unchanged": True,
    }

    decision = ProductionWorkflowExecutor._classify_hung_signals(
        signals, undetermined_nodes=[]
    )

    # No rank lags the sequence, so the verdict must come from the
    # CPU/GPU evidence rather than declaring the whole fabric suspect.
    assert decision["classification"] == "WEAK"
    assert decision["culprit_ranks"] == [0]


def test_hung_triage_without_dumps_falls_through_to_cpu_gpu() -> None:
    signals = [
        _hung_signal_without_dump(rank, spinning=rank != 0) for rank in range(24)
    ]

    decision = ProductionWorkflowExecutor._classify_hung_signals(
        signals, undetermined_nodes=[]
    )

    assert decision["classification"] == "WEAK"
    assert decision["culprit_ranks"] == [0]
    assert decision["flight_recorder_unavailable"] is True
    assert decision["control_ranks"] and decision["control_ranks"] != [0]


def test_hung_triage_weak_ignores_traits_shared_by_every_rank() -> None:
    signals = [_hung_signal_without_dump(rank, spinning=True) for rank in range(24)]

    decision = ProductionWorkflowExecutor._classify_hung_signals(
        signals, undetermined_nodes=[]
    )

    assert decision["classification"] == "UNDETERMINED"
    assert decision["culprit_ranks"] == []
    assert decision["flight_recorder_unavailable"] is True
    assert "no rank stood out" in decision["reason"]


@pytest.mark.parametrize("graded_by_node_agent", [True, False])
def test_dcgm_config_severity_failure_does_not_drain(
    graded_by_node_agent: bool,
) -> None:
    """A node agent that grades the failure returns WARN; one that
    predates the grading still returns FAIL, and the control plane
    must re-derive CONFIG severity from the findings rather than
    quarantine a healthy node."""
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RUN_DCGM_DIAGNOSTIC])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    findings = [
        {
            "test_name": "software",
            "status": "FAIL",
            "entities": ["GPU", str(gpu)],
            "error_codes": ["29"],
            "messages": [
                f"Persistence Mode: Persistence mode for GPU {gpu} is disabled."
            ],
            "error_severities": ["5"],
            "result_path": f"$.results[{gpu}]",
        }
        for gpu in range(8)
    ]
    details: dict = {
        "diagnostic_outcome": ("WARN" if graded_by_node_agent else "FAIL"),
        "returncode": 226,
        "parse_error": None,
        "diagnostic_findings": findings,
        "evidence_ref": "file:///diagnostics/dcgm.json",
        "sha256": "a" * 64,
        "recommended_actions": [
            {
                "action_code": "GPU_HOST_CONFIG_REMEDIATION",
                "priority": "REVIEW",
                "instruction": "Fix host configuration.",
                "trigger_tests": ["software"],
            }
        ],
    }
    if graded_by_node_agent:
        details["configuration_only_failures"] = True

    def sender(_, envelope):
        return node_action_result(
            envelope.command.command_id, envelope.command.operation, details=details
        )

    outcome = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, "s" * 32, sender=sender, store=store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="workflow/dcgm-config-only",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert "failed_nodes" not in outcome.details
    assert outcome.details["configuration_only_nodes"] == ["node-a"]
    assert outcome.details["control_plane_action"] == ("COOLDOWN_AND_VALIDATE")
    notification = store.list_notifications()[0]
    assert "GPU_HOST_CONFIG_REMEDIATION" in notification.body_text
    assert "COOLDOWN_AND_VALIDATE" in notification.body_text


def test_sqlite_expired_lease_is_taken_over_and_fences_stale_owner(tmp_path) -> None:
    path = tmp_path / "lease.db"
    first = SqliteStore(str(path))
    _, workflow = workflow_state(first, [WorkflowOperation.RESTART_NODE])
    second = SqliteStore(str(path))
    lease = timedelta(seconds=30)

    claimed_a = first.claim_workflow(
        workflow.request_id,
        "executor-a",
        workflow.fencing_token,
        now=NOW,
        lease_duration=lease,
    )
    with pytest.raises(WorkflowLeaseError, match="another executor"):
        second.claim_workflow(
            workflow.request_id,
            "executor-b",
            workflow.fencing_token,
            now=NOW + timedelta(seconds=10),
            lease_duration=lease,
        )

    claimed_b = second.claim_workflow(
        workflow.request_id,
        "executor-b",
        workflow.fencing_token,
        now=NOW + timedelta(seconds=31),
        lease_duration=lease,
    )

    assert claimed_a.execution_epoch == 1
    assert claimed_b.execution_epoch == 2
    assert claimed_b.execution_owner_id == "executor-b"
    with pytest.raises(WorkflowLeaseError, match="stale"):
        first.save_workflow_if_leased(
            copy_model(claimed_a, status=WorkflowStatus.SUCCEEDED),
            "executor-a",
            claimed_a.execution_epoch,
            now=NOW + timedelta(seconds=32),
        )


def test_mechanical_inspection_waits_for_incident_fenced_annotation() -> None:
    core = FakeCoreApi()
    store = build_store()
    sent = []
    adapter = KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=store,
        alert_sender=sent.append,
    )
    incident, workflow = workflow_state(store, [WorkflowOperation.CHECK_MECHANICALS])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        parameters={
            "nvlink_link_id": 3,
            "pci_bdf": "0000:59:00.0",
            "nvlink_occurrence_counts": {"register1.bit8": 1},
        },
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="mechanical/incident-active",
    )

    waiting = adapter.execute(context)
    core.node["metadata"]["annotations"][ANNOTATION_MECHANICAL_INSPECTION_COMPLETE] = (
        "old-incident:3"
    )
    stale = adapter.execute(context)
    core.node["metadata"]["annotations"][ANNOTATION_MECHANICAL_INSPECTION_COMPLETE] = (
        "incident-active:3"
    )
    confirmed = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert stale.status is WorkflowStepStatus.WAITING
    assert waiting.details["required_annotation_value"] == ("incident-active:3")
    notification = store.list_notifications()[0]
    assert "xid74-mechanical-zh-v1" in notification.body_text
    assert "register1.bit8=1" in notification.body_text
    assert sent[0] == notification.notification_id
    assert confirmed.status is WorkflowStepStatus.SUCCEEDED
    assert confirmed.details["confirmed_nodes"] == ["node-a"]


def test_storeless_adapter_refuses_live_predecessor() -> None:
    core = FakeCoreApi()
    provider = RecordingOwnershipProvider({"incident-dead": False})
    adapter, context = _storeless_isolation_context(core, provider)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "already isolated by another incident" in (outcome.error or "")
    assert outcome.details["safety_rejection"] is True
    assert (
        core.node["metadata"]["annotations"]["gpu-fault.io/incident-id"]
        == "incident-dead"
    )


def test_storeless_adapter_fails_closed_when_control_plane_errors() -> None:
    # An unreachable control plane must not license stealing a node from
    # a workflow that could still be running.
    core = FakeCoreApi()
    provider = RecordingOwnershipProvider(
        {"incident-dead": True}, error=RuntimeError("connection refused")
    )
    adapter, context = _storeless_isolation_context(core, provider)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "already isolated by another incident" in (outcome.error or "")
    assert outcome.details["safety_rejection"] is True


def test_storeless_adapter_without_provider_refuses_takeover() -> None:
    core = FakeCoreApi()
    adapter, context = _storeless_isolation_context(core, None)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "already isolated by another incident" in (outcome.error or "")
    assert outcome.details["safety_rejection"] is True


def test_same_incident_stale_node_generation_is_always_rejected() -> None:
    core = FakeCoreApi()
    provider = RecordingOwnershipProvider({"incident-active": True})
    adapter, context = _storeless_isolation_context(core, provider)
    core.node["metadata"]["annotations"].update(
        {
            "gpu-fault.io/incident-id": "incident-active",
            "gpu-fault.io/fencing-token": "4",
        }
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "newer workflow generation" in (outcome.error or "")
    assert outcome.details["safety_rejection"] is True
    assert provider.queried == []


def test_reboot_confirmation_prefers_structured_source_boot_id() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    incident = copy_model(
        incident,
        event_id="collector-without-kernel-id-shape",
        source_boot_id="11111111-2222-3333-4444-555555555555",
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=workflow.official_steps[0],
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="structured-boot-id",
    )

    assert (
        HyperPodLifecycleStepAdapter._source_boot_id(context)
        == "11111111-2222-3333-4444-555555555555"
    )


def test_execute_api_requires_token() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.FREEZE_EVIDENCE])
    adapter = FakeAdapter(
        {WorkflowOperation.FREEZE_EVIDENCE: (WorkflowStepOutcome.succeeded())}
    )
    context = ApplicationContext(
        store=store,
        production_executor_config=ProductionExecutorConfig(
            enabled=True,
            executor_id="executor-a",
            allowed_operations=frozenset({WorkflowOperation.FREEZE_EVIDENCE}),
        ),
        production_adapters=[adapter],
        execution_token="x" * 32,
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            denied = await client.post(
                f"/v1/workflows/{workflow.request_id}/execute",
                json={"expected_fencing_token": 3},
            )
            accepted = await client.post(
                f"/v1/workflows/{workflow.request_id}/execute",
                headers={"X-GPU-Fault-Execution-Token": "x" * 32},
                json={"expected_fencing_token": 3},
            )
            health = await client.get("/healthz")

        assert denied.status_code == 403
        assert accepted.status_code == 200
        assert accepted.json()["simulation_only"] is False
        assert health.json()["executor"] == "active"

    import asyncio

    asyncio.run(scenario())


def test_active_environment_requires_durable_configuration(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("GPU_FAULT_EXECUTOR_MODE", "active")
    monkeypatch.setenv("GPU_FAULT_ALLOW_SINGLE_CLUSTER", "true")
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    # This test is about the durable-configuration gates, not about
    # alerting; the alert-channel gate has its own test below.
    monkeypatch.setenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", "true")

    import pytest

    with pytest.raises(RuntimeError, match="STORE_URL"):
        ApplicationContext.from_environment()

    monkeypatch.setenv("GPU_FAULT_STORE_URL", f"sqlite:///{tmp_path / 'active.db'}")
    monkeypatch.delenv("GPU_FAULT_EXECUTION_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="EXECUTION_TOKEN"):
        ApplicationContext.from_environment()

    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "t" * 32)
    monkeypatch.delenv("GPU_FAULT_ALLOWED_OPERATIONS", raising=False)
    with pytest.raises(RuntimeError, match="ALLOWED_OPERATIONS"):
        ApplicationContext.from_environment()

    monkeypatch.setenv("GPU_FAULT_ALLOWED_OPERATIONS", "FREEZE_EVIDENCE")
    context = ApplicationContext.from_environment()

    assert context.executor_mode == "active"
    assert context.dispatcher.config.enabled


def test_active_environment_requires_an_alert_channel(monkeypatch, tmp_path) -> None:
    """An active control plane must be able to reach a human.

    The deployed configuration had email off, the notification
    dispatcher off and the metrics collector at zero replicas, so a
    recovery that silently never executed looked exactly like a healthy
    one. Startup now fails closed; the escape hatch is an explicit
    statement in the manifest that alerting lives outside this process.
    """
    import pytest

    monkeypatch.setenv("GPU_FAULT_EXECUTOR_MODE", "active")
    monkeypatch.setenv("GPU_FAULT_ALLOW_SINGLE_CLUSTER", "true")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", f"sqlite:///{tmp_path / 'alerting.db'}")
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "t" * 32)
    monkeypatch.setenv("GPU_FAULT_ALLOWED_OPERATIONS", "FREEZE_EVIDENCE")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.delenv("GPU_FAULT_EMAIL_SENDER", raising=False)
    monkeypatch.delenv("GPU_FAULT_EMAIL_RECIPIENTS", raising=False)
    monkeypatch.delenv("GPU_FAULT_ALLOW_EMAIL", raising=False)
    monkeypatch.delenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", raising=False)

    with pytest.raises(RuntimeError, match="no external alert channel"):
        ApplicationContext.from_environment()

    monkeypatch.setenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", "true")
    acknowledged = ApplicationContext.from_environment()
    assert acknowledged.executor_mode == "active"
    assert not acknowledged.advisory_notifications.delivers_externally()

    # A configured channel needs no acknowledgement: email on, with a
    # sender and recipients, and synchronous delivery so the dispatcher
    # is not required to drain the outbox.
    monkeypatch.delenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", raising=False)
    monkeypatch.setenv("GPU_FAULT_ALLOW_EMAIL", "true")
    monkeypatch.setenv("GPU_FAULT_EMAIL_SENDER", "gpu-fault@example.com")
    monkeypatch.setenv("GPU_FAULT_EMAIL_RECIPIENTS", "oncall@example.com")
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY", "false")
    delivering = ApplicationContext.from_environment()
    assert delivering.advisory_notifications.delivers_externally()


def test_validation_sample_age_matches_gpu_metrics_silence(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("GPU_FAULT_EXECUTOR_MODE", "active")
    monkeypatch.setenv("GPU_FAULT_ALLOW_SINGLE_CLUSTER", "true")
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", f"sqlite:///{tmp_path / 'validation-age.db'}"
    )
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "t" * 32)
    monkeypatch.setenv("GPU_FAULT_ALLOWED_OPERATIONS", "FREEZE_EVIDENCE")
    monkeypatch.setenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", "true")
    monkeypatch.setenv("GPU_FAULT_GPU_METRICS_SILENT_AFTER_SECONDS", "480")
    monkeypatch.setenv("GPU_FAULT_POST_ACTION_VALIDATION_MAX_SAMPLE_AGE_SECONDS", "480")

    context = ApplicationContext.from_environment()
    adapter = next(
        item
        for item in context.workflow_executor.adapters
        if isinstance(item, GpuValidationAdapter)
    )

    assert adapter.max_sample_age == timedelta(seconds=480)
    assert adapter.post_action_max_sample_age == timedelta(seconds=480)


def test_active_environment_selects_postgres_store(monkeypatch) -> None:
    selected = build_store()
    captured = {}
    monkeypatch.setenv("GPU_FAULT_EXECUTOR_MODE", "active")
    monkeypatch.setenv("GPU_FAULT_ALLOW_SINGLE_CLUSTER", "true")
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://gpu-fault:secret@postgres/gpu-fault"
    )
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "t" * 32)
    monkeypatch.setenv("GPU_FAULT_ALLOWED_OPERATIONS", "FREEZE_EVIDENCE")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "false")
    monkeypatch.setenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", "true")

    def fake_store(_url, **kwargs):
        captured.update(kwargs)
        return selected

    monkeypatch.setattr(context_module, "PostgresStore", fake_store)

    context = ApplicationContext.from_environment()

    assert context.store is selected
    assert context.executor_mode == "active"
    assert captured["initialize_schema"] is False
