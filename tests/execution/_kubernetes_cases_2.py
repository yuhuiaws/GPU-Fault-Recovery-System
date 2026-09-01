from __future__ import annotations

from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_step,
)

from ._support import (
    NOW,
    RESTART_PARAMETERS,
    EvidenceKind,
    FakeBatchApi,
    KubernetesWorkflowAdapter,
    SimpleNamespace,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepStatus,
    _managed_job_recovery_workflow,
    _run_managed_job_recovery_step,
    gzip,
    pytest,
    workflow_state,
)


def test_kubernetes_stop_captures_training_log_before_delete() -> None:
    class Core:
        def __init__(self) -> None:
            self.events = []
            self.pods = [
                {
                    "metadata": {
                        "name": "worker-0",
                        "namespace": "gpu-fault-system",
                        "uid": "pod-uid-0",
                        "annotations": {"gpu-fault.io/training-container": ("trainer")},
                    },
                    "spec": {"nodeName": "node-a", "containers": [{"name": "trainer"}]},
                }
            ]

        def list_namespaced_pod(self, *_args, **_kwargs):
            return SimpleNamespace(items=list(self.pods))

        def read_namespaced_pod_log(self, *_args, **kwargs):
            self.events.append(("read", kwargs))
            return "2026-08-13T00:00:00Z first\nlast\n"

        def patch_namespaced_pod(self, name, namespace, body):
            self.events.append(("patch", namespace, name, body))

        def delete_namespaced_pod(self, name, namespace, grace_period_seconds):
            self.events.append(("delete", namespace, name))
            self.pods = []

    class Custom:
        def __init__(self) -> None:
            self.workload = {
                "metadata": {
                    "resourceVersion": "1",
                    "labels": {
                        "gpu-fault.io/job-id": "training-job",
                        "gpu-fault.io/attempt-id": "attempt-a",
                    },
                    "annotations": {},
                },
                "spec": {"runPolicy": {"suspend": False}},
                "status": {
                    "conditions": [{"type": "Running", "status": "True"}],
                    "replicaStatuses": {"Master": {"active": 1}},
                },
            }

        def get_namespaced_custom_object(self, *_args):
            return self.workload

        def patch_namespaced_custom_object(
            self, _group, _version, _namespace, _plural, _name, body
        ):
            self.workload["metadata"]["annotations"].update(
                body["metadata"]["annotations"]
            )
            self.workload["spec"] = body["spec"]

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    core = Core()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=Custom(), store=store
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["gpu-fault-system/pytorchjob/training-job"],
        parameters={"termination_initiator_incident_id": incident.incident_id},
    )
    context = WorkflowStepContext(
        workflow=copy_model(workflow, official_steps=[step]),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="workflow-log-capture/STOP_WORKLOADS",
    )

    waiting = adapter.execute(context)
    evidence = store.list_raw_evidence(
        "cluster-a", node_id="node-a", kind=EvidenceKind.WORKLOAD_LOG
    )

    assert waiting.status is WorkflowStepStatus.WAITING
    assert [event[0] for event in core.events] == ["read", "patch", "delete"]
    assert len(evidence) == 1
    assert evidence[0].attempt_ids == ["attempt-a"]
    assert evidence[0].payload["tail"].endswith("last\n")
    assert (
        waiting.details["workload_log_evidence"][0]["record_id"]
        == evidence[0].record_id
    )
    assert waiting.details["workload_log_errors"] == []


def test_workload_log_s3_upload_is_gzipped() -> None:
    uploaded = {}

    def upload(destination, value):
        uploaded["destination"] = destination
        uploaded["value"] = value
        return destination

    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        workload_log_s3_uri="s3://audit-bucket/workload-logs",
        workload_log_uploader=upload,
    )

    destination = adapter._upload_workload_log(
        data=b"training output\n",
        cluster_id="cluster-a",
        incident_id="incident-a",
        workflow_id="workflow-a",
        pod_uid="pod-a",
        container_name="trainer",
        captured_at=NOW,
    )

    assert destination.startswith("s3://audit-bucket/workload-logs/cluster-a/")
    assert gzip.decompress(uploaded["value"]) == b"training output\n"


def test_kubernetes_stop_retries_workload_patch_conflict() -> None:
    class ConflictBatch(FakeBatchApi):
        def __init__(self) -> None:
            super().__init__()
            self.patch_calls = 0
            self.resource_version = "1"

        def read_namespaced_job(self, name, namespace):
            value = super().read_namespaced_job(name, namespace)
            value["metadata"]["resourceVersion"] = self.resource_version
            return value

        def patch_namespaced_job(self, name, namespace, body):
            self.patch_calls += 1
            if self.patch_calls == 1:
                self.resource_version = "2"
                error = RuntimeError("job was modified")
                error.status = 409
                raise error
            assert body["metadata"]["resourceVersion"] == "2"
            return super().patch_namespaced_job(name, namespace, body)

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    batch = ConflictBatch()
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["gpu-fault-system/job/training-job"],
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)

    result = execute_workflow(
        active_workflow_executor(store, [adapter], {WorkflowOperation.STOP_WORKLOADS}),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.RUNNING
    assert batch.patch_calls == 2
    assert batch.suspend_patches == [True]


def test_kubernetes_stop_treats_missing_workload_as_already_stopped() -> None:
    class MissingBatch(FakeBatchApi):
        def read_namespaced_job(self, name, namespace):
            error = KeyError((namespace, name))
            error.status = 404
            raise error

    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    batch = MissingBatch()
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["gpu-fault-system/job/training-job"],
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)

    result = execute_workflow(
        active_workflow_executor(store, [adapter], {WorkflowOperation.STOP_WORKLOADS}),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    persisted = store.get_workflow(workflow.request_id)
    execution = persisted.step_executions[-1]
    assert execution.status is WorkflowStepStatus.SUCCEEDED
    assert execution.details["already_absent_workloads"] == [
        "gpu-fault-system/job/training-job"
    ]
    assert batch.suspend_patches == []


def test_kubernetes_restart_fails_closed_when_source_workload_is_missing() -> None:
    class MissingCustom:
        def get_namespaced_custom_object(self, *_args):
            error = KeyError("missing PyTorchJob")
            error.status = 404
            raise error

    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTART_WORKLOAD])
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=UnusedApi(),
        custom_api=MissingCustom(),
        store=store,
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["gpu-fault-system/pytorchjob/training-job"],
    )
    context = WorkflowStepContext(
        workflow=copy_model(workflow, official_steps=[step]),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="workflow-missing-source/RESTART_WORKLOAD",
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details == {
        "reason": "RESTART_SOURCE_WORKLOAD_NOT_FOUND",
        "source_workload_ids": ["gpu-fault-system/pytorchjob/training-job"],
    }
    assert outcome.error == (
        "restart source workload is missing: gpu-fault-system/pytorchjob/training-job"
    )


@pytest.mark.parametrize(
    "previous_status",
    [WorkflowStatus.BLOCKED, WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED],
)
def test_kubernetes_stop_takes_over_terminal_incident(
    previous_status: WorkflowStatus,
) -> None:
    store = build_store()
    previous_incident, previous_workflow = workflow_state(
        store, [WorkflowOperation.RESTART_WORKLOAD]
    )
    previous_workflow = copy_model(previous_workflow, status=previous_status)
    store.save_workflow(previous_workflow)

    current_incident = copy_model(
        previous_incident,
        incident_id="incident-current",
        event_id="event-current",
        workflow_request_id="workflow-current",
    )
    current_workflow = copy_model(
        previous_workflow,
        request_id="workflow-current",
        incident_id="incident-current",
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "gpu-fault-kubernetes-adapter",
                workload_ids=["gpu-fault-system/job/training-job"],
            )
        ],
    )
    store.save_incident(current_incident)
    store.save_workflow(current_workflow)
    batch = FakeBatchApi()
    batch.annotations = {
        "gpu-fault.io/incident-id": (previous_incident.incident_id),
        "gpu-fault.io/workflow-id": (previous_workflow.request_id),
        "gpu-fault.io/operation-id": (
            f"{previous_workflow.request_id}/0/RESTART_WORKLOAD"
        ),
        "gpu-fault.io/execution-epoch": "1",
        "gpu-fault.io/workflow-step-index": "0",
    }
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )

    result = execute_workflow(
        active_workflow_executor(store, [adapter], {WorkflowOperation.STOP_WORKLOADS}),
        current_workflow.request_id,
    )

    assert result.status is WorkflowStatus.RUNNING
    assert batch.suspend_patches == [True]
    assert batch.annotations["gpu-fault.io/incident-id"] == (
        current_incident.incident_id
    )


def test_kubernetes_stop_rejects_active_incident_takeover() -> None:
    store = build_store()
    previous_incident, previous_workflow = workflow_state(
        store, [WorkflowOperation.RESTART_WORKLOAD]
    )
    current_incident = copy_model(
        previous_incident,
        incident_id="incident-current",
        event_id="event-current",
        workflow_request_id="workflow-current",
    )
    current_workflow = copy_model(
        previous_workflow,
        request_id="workflow-current",
        incident_id="incident-current",
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "gpu-fault-kubernetes-adapter",
                workload_ids=["gpu-fault-system/job/training-job"],
            )
        ],
    )
    store.save_incident(current_incident)
    store.save_workflow(current_workflow)
    batch = FakeBatchApi()
    batch.annotations = {"gpu-fault.io/incident-id": (previous_incident.incident_id)}
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )

    result = execute_workflow(
        active_workflow_executor(store, [adapter], {WorkflowOperation.STOP_WORKLOADS}),
        current_workflow.request_id,
    )

    assert result.status is WorkflowStatus.FAILED
    assert "controlled by another incident" in (result.error or "")
    assert batch.suspend_patches == []


def test_kubernetes_restart_clones_terminal_job_idempotently() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_WORKLOAD])
    batch = FakeBatchApi()
    batch.active = 0
    batch.failed = 1
    batch.job_spec_extra = {
        "completionMode": "Indexed",
        "completions": 2,
        "parallelism": 2,
    }
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=["gpu-fault-system/job/training-job"],
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "training-job",
            "source_attempt_id": "attempt-a",
            "source_gpu_count": 2,
            "restart_budget": 1,
        },
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    active = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_WORKLOAD}
    )

    result = execute_workflow(active, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert len(batch.created) == 1
    restarted = next(iter(batch.created.values()))
    assert restarted["spec"]["suspend"] is False
    assert restarted["spec"]["completionMode"] == "Indexed"
    assert restarted["spec"]["completions"] == 2
    assert restarted["spec"]["parallelism"] == 2
    assert "resourceVersion" not in restarted["metadata"]
    assert restarted["spec"]["template"]["metadata"]["labels"][
        "gpu-fault.io/attempt-id"
    ].startswith("attempt-a-r-")
    retry_name = restarted["metadata"]["name"]
    persisted = store.get_workflow(workflow.request_id)
    assert persisted.step_executions[-1].details["restarted_workload_ids"] == [
        f"gpu-fault-system/job/{retry_name}"
    ]
    assert (
        restarted["spec"]["template"]["metadata"]["annotations"][
            "gpu-fault.io/workload-ids"
        ]
        == f'["gpu-fault-system/job/{retry_name}"]'
    )


@pytest.mark.parametrize(
    "operation,parameters",
    [
        (WorkflowOperation.STOP_WORKLOADS, {}),
        (WorkflowOperation.RESTART_WORKLOAD, RESTART_PARAMETERS),
    ],
)
def test_kubernetes_refuses_workload_with_hyperpod_auto_resume(
    operation: WorkflowOperation, parameters: dict[str, object]
) -> None:
    store = build_store()
    batch = FakeBatchApi()
    batch.annotations = {"sagemaker.amazonaws.com/enable-job-auto-resume": "true"}
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    workflow = _managed_job_recovery_workflow(
        store,
        operation,
        adapter,
        workload_ids=["gpu-fault-system/job/training-job"],
        parameters=parameters,
    )

    result = _run_managed_job_recovery_step(store, workflow, adapter, operation)

    assert result.status is WorkflowStatus.FAILED
    assert "enable-job-auto-resume is enabled" in (result.error or "")
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["managed_job_recovery_workloads"] == [
        "gpu-fault-system/job/training-job"
    ]
    assert execution.details["required_annotation"] == (
        "sagemaker.amazonaws.com/enable-job-auto-resume"
    )
    assert execution.details["required_annotation_value"] == ("absent or false")
    assert execution.details["remediation_commands"] == [
        "kubectl annotate job training-job -n gpu-fault-system "
        "sagemaker.amazonaws.com/enable-job-auto-resume-"
    ]
    assert batch.suspend_patches == []
    assert batch.created == {}


@pytest.mark.parametrize(
    "annotations", [{}, {"sagemaker.amazonaws.com/enable-job-auto-resume": ("false")}]
)
def test_kubernetes_stops_workload_without_hyperpod_auto_resume(
    annotations: dict[str, str],
) -> None:
    store = build_store()
    batch = FakeBatchApi()
    batch.annotations = dict(annotations)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    workflow = _managed_job_recovery_workflow(
        store,
        WorkflowOperation.STOP_WORKLOADS,
        adapter,
        workload_ids=["gpu-fault-system/job/training-job"],
    )

    result = _run_managed_job_recovery_step(
        store, workflow, adapter, WorkflowOperation.STOP_WORKLOADS
    )

    assert result.status is WorkflowStatus.RUNNING
    assert batch.suspend_patches == [True]


def test_kubernetes_auto_resume_guard_runs_before_pod_termination() -> None:
    class TerminationCore:
        def __init__(self) -> None:
            self.listed: list[str] = []

        def list_namespaced_pod(self, namespace, label_selector):
            self.listed.append(namespace)
            return {"items": []}

    store = build_store()
    core = TerminationCore()
    batch = FakeBatchApi()
    batch.annotations = {"sagemaker.amazonaws.com/enable-job-auto-resume": "true"}
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=batch, custom_api=UnusedApi(), store=store
    )
    workflow = _managed_job_recovery_workflow(
        store,
        WorkflowOperation.STOP_WORKLOADS,
        adapter,
        workload_ids=["gpu-fault-system/job/training-job"],
        parameters={"termination_initiator_incident_id": "incident-active"},
    )

    result = _run_managed_job_recovery_step(
        store, workflow, adapter, WorkflowOperation.STOP_WORKLOADS
    )

    assert result.status is WorkflowStatus.FAILED
    assert core.listed == []
    assert batch.suspend_patches == []


def test_kubernetes_auto_resume_guard_reports_only_violating_workloads() -> None:
    class TwoJobBatch(FakeBatchApi):
        def read_namespaced_job(self, name, namespace):
            value = super().read_namespaced_job("training-job", namespace)
            value = {
                **value,
                "metadata": {
                    **value["metadata"],
                    "name": name,
                    "annotations": (
                        {"sagemaker.amazonaws.com/enable-job-auto-resume": "true"}
                        if name == "managed-job"
                        else {}
                    ),
                },
            }
            return value

    store = build_store()
    batch = TwoJobBatch()
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    workflow = _managed_job_recovery_workflow(
        store,
        WorkflowOperation.STOP_WORKLOADS,
        adapter,
        workload_ids=[
            "gpu-fault-system/job/plain-job",
            "gpu-fault-system/job/managed-job",
        ],
    )

    result = _run_managed_job_recovery_step(
        store, workflow, adapter, WorkflowOperation.STOP_WORKLOADS
    )

    assert result.status is WorkflowStatus.FAILED
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.details["managed_job_recovery_workloads"] == [
        "gpu-fault-system/job/managed-job"
    ]
    assert batch.suspend_patches == []


def test_pytorch_restart_rebinds_node_placement_to_spare() -> None:
    workload = {
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "template": {
                        "spec": {
                            "nodeName": "node-source",
                            "nodeSelector": {"kubernetes.io/hostname": "node-source"},
                        }
                    }
                },
                "Worker": {
                    "template": {
                        "spec": {
                            "nodeSelector": {"kubernetes.io/hostname": "node-source"}
                        }
                    }
                },
            }
        }
    }

    spec = KubernetesWorkflowAdapter._restart_custom_spec(
        "pytorchjob", workload, "attempt-restarted", 2, 1, {"node-source": "node-spare"}
    )

    master = spec["pytorchReplicaSpecs"]["Master"]["template"]
    worker = spec["pytorchReplicaSpecs"]["Worker"]["template"]
    assert master["spec"]["nodeName"] == "node-spare"
    assert master["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-spare"
    assert worker["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-spare"
    assert (
        master["metadata"]["labels"]["gpu-fault.io/attempt-id"] == "attempt-restarted"
    )
