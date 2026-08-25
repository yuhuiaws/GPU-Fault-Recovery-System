from __future__ import annotations

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step_execution,
)

from ._support import (
    CollectorKind,
    CollectorStatus,
    FakeAdapter,
    GpuValidationAdapter,
    SimpleNamespace,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStepStatus,
    _preempting_successor,
    datetime,
    pytest,
    timedelta,
    timezone,
    workflow_state,
)


def test_gpu_validation_reports_every_failed_node() -> None:
    now = datetime.now(timezone.utc)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now, sample=SimpleNamespace(canonical_name=name)
                )
                for name in {"gpu_temperature_c"}
            ]

        def findings(self, cluster_id, node_id):
            return (
                [SimpleNamespace(finding_id=f"finding-{node_id}")]
                if node_id in {"node-a", "node-c"}
                else []
            )

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_GPU])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-validation-adapter",
        node_ids=["node-a", "node-b", "node-c"],
    )
    outcome = GpuValidationAdapter(
        ValidationMetrics(), store=ValidationMetrics.store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/multi-node",
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["failed_nodes"] == ["node-a", "node-c"]
    assert set(outcome.details["node_failures"]) == {"node-a", "node-c"}


def test_gpu_validation_ignores_findings_outside_reset_scope() -> None:
    now = datetime.now(timezone.utc)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            return [
                SimpleNamespace(finding_id="historical-other-gpu", gpu_uuid="GPU-other")
            ]

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_GPU])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-validation-adapter",
        gpu_uuids=["GPU-target"],
    )
    outcome = GpuValidationAdapter(
        ValidationMetrics(), store=ValidationMetrics.store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/target-gpu",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED


def test_gpu_validation_waits_for_post_action_telemetry() -> None:
    action_completed_at = datetime.now(timezone.utc)

    class ValidationStore:
        sample_at = action_completed_at - timedelta(seconds=1)

        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=self.sample_at,
                    ingested_at=self.sample_at,
                    last_success_at=self.sample_at,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=self.store.sample_at,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            return []

    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.VALIDATE_GPU,
        ],
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.RESET_GPU, updated_at=action_completed_at
            ),
            workflow_step_execution(
                1,
                WorkflowOperation.RESTORE_GPU_SERVICES,
                updated_at=action_completed_at + timedelta(seconds=5),
            ),
        ],
    )
    step = copy_model(
        workflow.official_steps[2], execution_owner="gpu-fault-validation-adapter"
    )
    adapter = GpuValidationAdapter(ValidationMetrics(), store=ValidationMetrics.store)
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=2,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/post-action",
    )

    waiting = adapter.execute(context)
    ValidationMetrics.store.sample_at = action_completed_at + timedelta(seconds=1)
    ready = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["pending_nodes"] == ["node-a"]
    assert ready.status is WorkflowStepStatus.SUCCEEDED


def test_post_action_validation_accepts_edge_filter_window() -> None:
    now = datetime.now(timezone.utc)
    action_completed_at = now - timedelta(minutes=4)
    sample_at = now - timedelta(minutes=3)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=collector,
                    observed_at=sample_at,
                    ingested_at=sample_at,
                    last_success_at=sample_at,
                )
                for collector in {
                    CollectorKind.GPU_METRICS,
                    CollectorKind.HOST_TELEMETRY,
                }
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(name=name, value=value, observed_at=sample_at)
                for name, value in {
                    "gpu_inventory_expected_count": 8,
                    "gpu_inventory_active_count": 8,
                    "load1_per_cpu": 0.1,
                    "memory_used_percent": 10,
                    "filesystem_used_percent": 20,
                }.items()
            ]

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=sample_at,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            return []

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    step = copy_model(
        workflow.official_steps[1], execution_owner="gpu-fault-validation-adapter"
    )
    workflow = copy_model(
        workflow,
        official_steps=[workflow.official_steps[0], step],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.RESTART_NODE, updated_at=action_completed_at
            )
        ],
    )
    adapter = GpuValidationAdapter(ValidationMetrics(), store=ValidationMetrics.store)
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/edge-filter-window",
    )

    assert adapter.execute(context).status is WorkflowStepStatus.SUCCEEDED

    ordinary = workflow_state(build_store(), [WorkflowOperation.VALIDATE_GPU])
    ordinary_step = copy_model(
        ordinary[1].official_steps[0], execution_owner="gpu-fault-validation-adapter"
    )
    ordinary_context = WorkflowStepContext(
        workflow=copy_model(ordinary[1], official_steps=[ordinary_step]),
        incident=ordinary[0],
        step=ordinary_step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/normal-window",
    )

    assert adapter.execute(ordinary_context).status is (WorkflowStepStatus.WAITING)


@pytest.mark.parametrize(
    ("active_count", "expected_status"),
    [(8, WorkflowStepStatus.SUCCEEDED), (7, WorkflowStepStatus.FAILED)],
)
def test_post_reboot_gpu_inventory_requires_fresh_expected_count(
    active_count, expected_status
) -> None:
    now = datetime.now(timezone.utc)
    rebooted_at = now - timedelta(seconds=5)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=collector,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
                for collector in {
                    CollectorKind.GPU_METRICS,
                    CollectorKind.HOST_TELEMETRY,
                }
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    name="gpu_inventory_active_count",
                    value=active_count,
                    observed_at=now,
                ),
                SimpleNamespace(
                    name="gpu_inventory_expected_count",
                    value=8,
                    observed_at=now - timedelta(minutes=10),
                ),
            ]

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            return []

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    validation_step = copy_model(
        workflow.official_steps[1], execution_owner="gpu-fault-validation-adapter"
    )
    workflow = copy_model(
        workflow,
        official_steps=[workflow.official_steps[0], validation_step],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.RESTART_NODE, updated_at=rebooted_at
            )
        ],
    )

    outcome = GpuValidationAdapter(
        ValidationMetrics(), store=ValidationMetrics.store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=validation_step,
            step_index=1,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/post-reboot-inventory",
        )
    )

    assert outcome.status is expected_status
    if expected_status is WorkflowStepStatus.FAILED:
        assert outcome.details["failed_nodes"] == ["node-a"]
        assert (
            "gpu_inventory_active_count=7,expected=8"
            in (outcome.details["node_failures"]["node-a"])
        )


def test_post_reboot_inventory_rejects_pre_reboot_sample() -> None:
    now = datetime.now(timezone.utc)
    rebooted_at = now - timedelta(seconds=5)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=collector,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
                for collector in {
                    CollectorKind.GPU_METRICS,
                    CollectorKind.HOST_TELEMETRY,
                }
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    name="gpu_inventory_active_count",
                    value=8,
                    observed_at=rebooted_at - timedelta(seconds=1),
                )
            ]

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            return []

    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESTART_NODE, WorkflowOperation.VALIDATE_GPU]
    )
    validation_step = copy_model(
        workflow.official_steps[1],
        execution_owner="gpu-fault-validation-adapter",
        parameters={
            "inventory_requirements_by_node": {
                "node-a": {
                    "active_metric": "gpu_inventory_active_count",
                    "expected_count": 8,
                }
            }
        },
    )
    workflow = copy_model(
        workflow,
        official_steps=[workflow.official_steps[0], validation_step],
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.RESTART_NODE, updated_at=rebooted_at
            )
        ],
    )

    outcome = GpuValidationAdapter(
        ValidationMetrics(), store=ValidationMetrics.store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=validation_step,
            step_index=1,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/pre-reboot-inventory",
        )
    )

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["pending_nodes"] == ["node-a"]
    assert (
        "missing_post_action_inventory_metric"
        in (outcome.details["node_pending"]["node-a"])
    )


@pytest.mark.parametrize(
    ("recovery_operation", "validation_operation", "active_metric"),
    [
        (
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
            WorkflowOperation.VALIDATE_FABRIC,
            "efa_inventory_active_count",
        ),
        (
            WorkflowOperation.REMEDIATE_EFA_DRIVER,
            WorkflowOperation.VALIDATE_FABRIC,
            "efa_inventory_active_count",
        ),
        (
            WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
            WorkflowOperation.VALIDATE_GPU,
            "gpu_inventory_active_count",
        ),
    ],
)
def test_targeted_inventory_recovery_satisfies_validation_gate(
    recovery_operation, validation_operation, active_metric
) -> None:
    now = datetime.now(timezone.utc)
    recovered_at = now - timedelta(seconds=5)
    expected = 16 if active_metric.startswith("efa_") else 8

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=collector,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
                for collector in {
                    CollectorKind.GPU_METRICS,
                    CollectorKind.HOST_TELEMETRY,
                }
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(name=active_metric, value=expected, observed_at=now),
                SimpleNamespace(name="network_link_up", value=1, observed_at=now),
            ]

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            names = (
                {
                    "nvlink_crc_aggregate_error_total",
                    "nvlink_recovery_aggregate_error_total",
                    "nvlink_replay_aggregate_error_total",
                }
                if validation_operation is WorkflowOperation.VALIDATE_FABRIC
                else {"gpu_temperature_c"}
            )
            return [
                SimpleNamespace(
                    observed_at=now, sample=SimpleNamespace(canonical_name=name)
                )
                for name in names
            ]

        def findings(self, cluster_id, node_id):
            return []

    store = build_store()
    incident, workflow = workflow_state(
        store, [recovery_operation, validation_operation]
    )
    validation_step = copy_model(
        workflow.official_steps[1],
        execution_owner="gpu-fault-validation-adapter",
        parameters={
            "inventory_requirements_by_node": {
                "node-a": {"metrics": {active_metric: expected}}
            }
        },
    )
    workflow = copy_model(
        workflow,
        official_steps=[workflow.official_steps[0], validation_step],
        step_executions=[
            workflow_step_execution(0, recovery_operation, updated_at=recovered_at)
        ],
    )

    outcome = GpuValidationAdapter(
        ValidationMetrics(), store=ValidationMetrics.store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=validation_step,
            step_index=1,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/targeted-inventory",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED


def test_temperature_warning_validation_observes_cooldown() -> None:
    now = datetime.now(timezone.utc)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

    class ValidationMetrics:
        store = ValidationStore()

        def __init__(self, warning_active: bool, *, severity: str = "WARNING") -> None:
            self.warning_active = warning_active
            self.severity = severity

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            if not self.warning_active:
                return []
            return [
                SimpleNamespace(
                    finding_id="temperature-warning",
                    canonical_name="gpu_temperature_c",
                    severity=SimpleNamespace(value=self.severity),
                    gpu_uuid="GPU-a",
                ),
                SimpleNamespace(
                    finding_id="thermal-throttle-warning",
                    canonical_name="clock_throttle_reasons",
                    severity=SimpleNamespace(value=self.severity),
                    gpu_uuid="GPU-a",
                ),
                SimpleNamespace(
                    finding_id="thermal-stress-composite",
                    canonical_name="composite:THERMAL_STRESS",
                    severity=SimpleNamespace(value=self.severity),
                    gpu_uuid="GPU-a",
                ),
            ]

    def validate(
        *, warning_active: bool, created_at: datetime, severity: str = "WARNING"
    ) -> WorkflowStepOutcome:
        store = build_store()
        incident, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_GPU])
        workflow = copy_model(workflow, created_at=created_at)
        step = copy_model(
            workflow.official_steps[0],
            execution_owner="gpu-fault-validation-adapter",
            gpu_uuids=["GPU-a"],
        )
        return GpuValidationAdapter(
            ValidationMetrics(warning_active, severity=severity),
            store=ValidationMetrics.store,
            temperature_warning_grace=timedelta(minutes=2),
        ).execute(
            WorkflowStepContext(
                workflow=workflow,
                incident=incident,
                step=step,
                step_index=0,
                request=WorkflowExecutionRequest(expected_fencing_token=3),
                idempotency_key="validation/temperature-warning",
            )
        )

    cooling = validate(warning_active=True, created_at=now - timedelta(seconds=30))
    persistent = validate(warning_active=True, created_at=now - timedelta(minutes=3))
    critical = validate(
        warning_active=True, severity="CRITICAL", created_at=now - timedelta(seconds=30)
    )
    cleared = validate(warning_active=False, created_at=now - timedelta(minutes=3))

    assert cooling.status is WorkflowStepStatus.WAITING
    assert cooling.details["pending_nodes"] == ["node-a"]
    assert persistent.status is WorkflowStepStatus.FAILED
    assert persistent.details["failed_nodes"] == ["node-a"]
    assert critical.status is WorkflowStepStatus.FAILED
    assert cleared.status is WorkflowStepStatus.SUCCEEDED


def test_correctable_memory_warning_validation_observes_grace() -> None:
    now = datetime.now(timezone.utc)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return []

    class ValidationMetrics:
        store = ValidationStore()

        def __init__(self, severity: str) -> None:
            self.severity = severity

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now,
                    sample=SimpleNamespace(canonical_name="gpu_temperature_c"),
                )
            ]

        def findings(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    finding_id="sbe-growth",
                    canonical_name="ecc_sbe_volatile_total",
                    severity=SimpleNamespace(value=self.severity),
                    gpu_uuid="GPU-a",
                ),
                SimpleNamespace(
                    finding_id="correctable-memory-composite",
                    canonical_name=("composite:CORRECTABLE_MEMORY_DEGRADATION"),
                    severity=SimpleNamespace(value=self.severity),
                    gpu_uuid="GPU-a",
                ),
            ]

    def validate(severity: str) -> WorkflowStepOutcome:
        store = build_store()
        incident, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_GPU])
        workflow = copy_model(workflow, created_at=now - timedelta(seconds=30))
        step = copy_model(
            workflow.official_steps[0],
            execution_owner="gpu-fault-validation-adapter",
            gpu_uuids=["GPU-a"],
        )
        return GpuValidationAdapter(
            ValidationMetrics(severity),
            store=ValidationMetrics.store,
            transient_warning_grace=timedelta(minutes=2),
        ).execute(
            WorkflowStepContext(
                workflow=workflow,
                incident=incident,
                step=step,
                step_index=0,
                request=WorkflowExecutionRequest(expected_fencing_token=3),
                idempotency_key=("validation/correctable-memory-warning"),
            )
        )

    warning = validate("WARNING")
    critical = validate("CRITICAL")

    assert warning.status is WorkflowStepStatus.WAITING
    assert warning.details["pending_nodes"] == ["node-a"]
    assert critical.status is WorkflowStepStatus.FAILED


def test_fabric_validation_ignores_findings_outside_reset_scope() -> None:
    now = datetime.now(timezone.utc)

    class ValidationStore:
        def list_collector_statuses(self, cluster_id, node_id):
            return [
                CollectorStatus(
                    cluster_id=cluster_id,
                    node_id=node_id,
                    collector=collector,
                    observed_at=now,
                    ingested_at=now,
                    last_success_at=now,
                )
                for collector in {
                    CollectorKind.GPU_METRICS,
                    CollectorKind.HOST_TELEMETRY,
                }
            ]

        def list_telemetry_metrics_latest(self, cluster_id, node_id):
            return [SimpleNamespace(observed_at=now, name="network_link_up", value=1)]

    class ValidationMetrics:
        store = ValidationStore()

        def latest(self, cluster_id, node_id):
            return [
                SimpleNamespace(
                    observed_at=now, sample=SimpleNamespace(canonical_name=name)
                )
                for name in {
                    "nvlink_crc_aggregate_error_total",
                    "nvlink_recovery_aggregate_error_total",
                    "nvlink_replay_aggregate_error_total",
                }
            ]

        def findings(self, cluster_id, node_id):
            return [
                SimpleNamespace(finding_id="historical-other-gpu", gpu_uuid="GPU-other")
            ]

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_FABRIC])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-validation-adapter",
        gpu_uuids=["GPU-target"],
    )
    outcome = GpuValidationAdapter(
        ValidationMetrics(), store=ValidationMetrics.store
    ).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/target-fabric",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED


def test_local_validation_waiting_is_safe_to_preempt() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.VALIDATE_GPU])
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="validation/poll",
            )
        ],
    )
    store.save_workflow(workflow)
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {
            WorkflowOperation.VALIDATE_GPU: (
                WorkflowStepOutcome.waiting(operation_id="validation/poll")
            )
        }
    )
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.VALIDATE_GPU}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUPERSEDED
    assert adapter.calls == []
