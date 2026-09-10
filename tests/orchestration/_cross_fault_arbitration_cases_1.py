from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import XidEvent
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import (
    asgi_client,
    build_context,
    build_store,
    container_observation,
    copy_model,
    workflow_step_execution,
)
from tests.orchestration._cross_fault_support import (
    NOW,
    arbitration_workflow,
    dcgm_pcie_batch,
    fabric_sxid12020_collector_event,
    freeze_fault_ingestion_time,
    kernel_xid11_collector_event,
    observation,
    post_faults,
    sxid_payload,
    xid_payload,
)


def two_node_observation() -> AttemptObservation:
    return copy_model(
        observation(),
        expected_critical_ranks=2,
        containers=[
            container_observation(
                "pod-a", "trainer-a", 0, "node-a", role="master", gpu_uuids=["GPU-a"]
            ),
            container_observation(
                "pod-b", "trainer-b", 1, "node-b", gpu_uuids=["GPU-b"]
            ),
        ],
    )


def restarted_attempt_context(
    *,
    baseline_xid: int = 11,
    baseline_status: WorkflowStatus = WorkflowStatus.SUCCEEDED,
) -> tuple[ApplicationContext, str, str]:
    context = build_context()
    old = observation(started_at=NOW - timedelta(minutes=1))
    context.store.save_attempt_observation(old)
    baseline = asyncio.run(
        post_faults(
            context,
            [("/v1/gpu-events/xid", xid_payload(baseline_xid, "baseline-recovery"))],
        )
    )[0]
    workflow = context.store.get_workflow(baseline["workflow_request_id"])
    current_attempt_id = "train-a002"
    context.store.save_workflow(
        copy_model(
            workflow,
            status=baseline_status,
            execution_owner_id="executor-a"
            if baseline_status is WorkflowStatus.RUNNING
            else None,
            step_executions=[
                workflow_step_execution(
                    0,
                    WorkflowOperation.RESTART_WORKLOAD,
                    details={"restart_attempt_id": current_attempt_id},
                )
            ],
        )
    )
    context.store.save_attempt_observation(
        copy_model(
            old,
            workload_phase=WorkloadPhase.STOPPED,
            observed_at=NOW + timedelta(seconds=10),
        )
    )
    context.store.save_attempt_observation(
        observation(
            attempt_id=current_attempt_id,
            started_at=NOW + timedelta(seconds=60),
            observed_at=NOW + timedelta(seconds=70),
        )
    )
    return (context, baseline["incident_id"], baseline["workflow_request_id"])


def test_attempt_observation_gpu_count_prefers_declared_resources() -> None:
    declared = copy_model(
        observation(),
        containers=[
            container_observation(
                f"pod-{index}", f"trainer-{index}", index, f"node-{index}", gpu_count=8
            )
            for index in range(3)
        ],
    )

    assert declared.gpu_count == 24


def test_delayed_xid_replay_uses_explicit_attempt_allocation() -> None:
    context = build_context()
    replayed = copy_model(
        observation(
            observed_at=NOW + timedelta(minutes=2),
            started_at=NOW - timedelta(minutes=1),
        ),
        expected_critical_ranks=3,
        containers=[
            container_observation(
                f"pod-{index}",
                f"trainer-{index}",
                index,
                "node-a" if index == 0 else f"node-{index}",
                gpu_count=8,
            )
            for index in range(3)
        ],
    )
    context.store.save_attempt_observation(replayed)
    payload = xid_payload(11, "delayed-explicit-xid")
    payload.update(
        {
            "gpu_uuid": None,
            "observed_at": NOW.isoformat(),
            "ingested_at": (NOW + timedelta(minutes=3)).isoformat(),
            "job_id": replayed.job_id,
            "attempt_id": replayed.attempt_id,
            "workload_state": "ACTIVE",
            "affected_workload_ids": replayed.workload_ids,
        }
    )

    response = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]
    workflow = context.store.get_workflow(response["workflow_request_id"])
    restart = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )

    assert restart.parameters["source_gpu_count"] == 24


def test_same_rank_incompatible_actions_do_not_absorb() -> None:
    existing = arbitration_workflow(
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE, request_id="existing"
    )
    candidate = arbitration_workflow(
        WorkflowOperation.REPLACE_NODE, request_id="candidate"
    )

    assert (
        IncidentOrchestrator(build_store())._merge_disposition(
            existing, candidate, "node-a", {"GPU-a"}
        )
        == "REPLACE_IN_PLACE"
    )


def test_running_workflow_widens_scope_before_reset_is_submitted() -> None:
    existing = arbitration_workflow(
        WorkflowOperation.RESET_GPU,
        request_id="existing",
        status=WorkflowStatus.RUNNING,
    )
    candidate = arbitration_workflow(
        WorkflowOperation.RESET_GPU, request_id="candidate", gpu_uuid="GPU-b"
    )

    assert (
        IncidentOrchestrator(build_store())._merge_disposition(
            existing, candidate, "node-a", {"GPU-b"}
        )
        == "WIDEN_IN_PLACE"
    )


def test_running_workflow_absorbs_same_action_and_scope() -> None:
    existing = arbitration_workflow(
        WorkflowOperation.RESET_GPU,
        request_id="existing",
        status=WorkflowStatus.RUNNING,
    )
    candidate = arbitration_workflow(
        WorkflowOperation.RESET_GPU, request_id="candidate"
    )

    assert (
        IncidentOrchestrator(build_store())._merge_disposition(
            existing, candidate, "node-a", {"GPU-a"}
        )
        == "ABSORB"
    )
    assert (
        IncidentOrchestrator._incident_state_for_workflow(existing)
        is IncidentState.ACTION_PENDING
    )


@pytest.mark.parametrize("metric_first", [True, False])
def test_dcgm_kernel_and_fabric_manager_share_attempt_workflow(
    metric_first: bool, monkeypatch
) -> None:
    freeze_fault_ingestion_time(monkeypatch)
    context = build_context()
    context.store.save_attempt_observation(observation())

    async def scenario() -> tuple[dict, dict, dict]:
        async with asgi_client(context) as client:
            baseline = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=dcgm_pcie_batch("dcgm-pcie-baseline", 0, NOW),
            )
            assert baseline.status_code == 200, baseline.text

            metric_request = (
                "/v1/collector-events/gpu-metrics",
                dcgm_pcie_batch("dcgm-pcie-fault", 3, NOW + timedelta(seconds=15)),
            )
            kernel_request = (
                "/v1/collector-events/nvidia-kernel",
                {
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "record_id": "kmsg-xid-11",
                    "observed_at": (NOW + timedelta(seconds=15)).isoformat(),
                    "collected_at": (NOW + timedelta(seconds=15)).isoformat(),
                    "message": ("NVRM: Xid (PCI:0000:b9:00): 11, Ch 00000001"),
                    "product": "H200",
                    "runtime_profile_version": "simulated-v1",
                },
            )
            fabric_request = (
                "/v1/collector-events/fabric-manager",
                {
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "record_id": "fm-sxid-12020",
                    "observed_at": (NOW + timedelta(seconds=16)).isoformat(),
                    "collected_at": (NOW + timedelta(seconds=16)).isoformat(),
                    "message": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 12020, "
                        "Fatal, Link 46 egress sequence ID error"
                    ),
                    "source": "journal",
                    "unit": "nvidia-fabricmanager.service",
                    "runtime_profile_version": "simulated-v1",
                    "product": "H200",
                },
            )
            ordered = (
                [metric_request, kernel_request, fabric_request]
                if metric_first
                else [kernel_request, fabric_request, metric_request]
            )
            responses = []
            for path, payload in ordered:
                response = await client.post(path, json=payload)
                assert response.status_code == 200, response.text
                responses.append(response.json())
            by_path = {
                path: response for (path, _), response in zip(ordered, responses)
            }
            return (
                by_path["/v1/collector-events/gpu-metrics"],
                by_path["/v1/collector-events/nvidia-kernel"],
                by_path["/v1/collector-events/fabric-manager"],
            )

    metric, kernel, fabric = asyncio.run(scenario())

    assert metric["new_findings"][0]["canonical_name"] == ("pcie_replay_total")
    assert kernel["normalized"]["xid_events"][0]["event_source"] == ("KERNEL_LOG")
    assert (
        fabric["normalized"]["sxid_events"][0]["event_source"] == "FABRIC_MANAGER_LOG"
    )
    workflows = context.store.list_workflows()
    assert len(workflows) == 1
    metric_event_id = "gpu-" + metric["new_findings"][0]["finding_id"]
    incident = context.store.get_incident_by_event(metric_event_id)
    assert incident is not None
    assert kernel["decisions"][0]["incident_id"] == (incident.incident_id)
    assert fabric["decisions"][0]["incident_id"] == (incident.incident_id)
    workflow = workflows[0]
    assert incident.job_id == "train"
    assert incident.attempt_id == "train-a001"
    assert "PCIe replay rate exceeded" in " ".join(incident.reasons)
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in workflow.official_steps
    }
    quiesce = next(
        (
            step
            for step in workflow.official_steps
            if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
        ),
        None,
    )
    if quiesce is not None:
        assert quiesce.parameters["workload_cgroup_paths_by_node"] == {
            "node-a": ["/kubepods/pod-train-a001/trainer"]
        }
    assert workflow.not_before is not None


def test_three_collectors_concurrently_merge_atomically(monkeypatch) -> None:
    freeze_fault_ingestion_time(monkeypatch)
    context = build_context()
    context.store.save_attempt_observation(observation())

    async def scenario() -> list[dict]:
        async with asgi_client(context) as client:
            baseline = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=dcgm_pcie_batch("dcgm-concurrent-baseline", 0, NOW),
            )
            assert baseline.status_code == 200, baseline.text
            responses = await asyncio.gather(
                client.post(
                    "/v1/collector-events/gpu-metrics",
                    json=dcgm_pcie_batch(
                        "dcgm-concurrent-fault", 3, NOW + timedelta(seconds=15)
                    ),
                ),
                client.post(
                    "/v1/collector-events/nvidia-kernel",
                    json=kernel_xid11_collector_event(),
                ),
                client.post(
                    "/v1/collector-events/fabric-manager",
                    json=fabric_sxid12020_collector_event(),
                ),
            )
        for response in responses:
            assert response.status_code == 200, response.text
        return [response.json() for response in responses]

    metric, kernel, fabric = asyncio.run(scenario())
    metric_event_id = "gpu-" + metric["new_findings"][0]["finding_id"]
    incident = context.store.get_incident_by_event(metric_event_id)
    assert incident is not None
    assert kernel["decisions"][0]["incident_id"] == (incident.incident_id)
    assert fabric["decisions"][0]["incident_id"] == (incident.incident_id)
    workflows = context.store.list_workflows()
    assert len(workflows) == 1
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in workflows[0].official_steps
    }


def test_running_xid_workflow_adds_terminal_quarantine_branch() -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(11, "xid-before-dcgm-dbe"))]
        )
    )[0]
    predecessor = context.store.get_workflow(first["workflow_request_id"])
    context.store.save_workflow(
        copy_model(
            predecessor, status=WorkflowStatus.RUNNING, execution_owner_id="executor-a"
        )
    )
    metric_batch = {
        "batch_id": "dcgm-volatile-dbe",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": (NOW + timedelta(seconds=15)).isoformat(),
        "source": "DCGM_EXPORTER",
        "samples": [
            {
                "metric_name": "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
                "canonical_name": "ecc_dbe_volatile_total",
                "value": 1,
                "gpu_index": "0",
                "gpu_uuid": "GPU-a",
                "pci_bdf": "0000:b9:00.0",
            }
        ],
        "runtime_profile_version": "simulated-v1",
        "product": "H200",
    }

    async def submit_metric() -> dict:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/gpu-metrics", json=metric_batch
            )
        assert response.status_code == 200, response.text
        return response.json()

    result = asyncio.run(submit_metric())
    metric_event_id = "gpu-" + result["new_findings"][0]["finding_id"]
    incident = context.store.get_incident_by_event(metric_event_id)
    assert incident is not None
    successor = context.store.get_workflow(incident.workflow_request_id)
    assert successor.request_id == predecessor.request_id
    assert successor.predecessor_workflow_id is None
    assert WorkflowOperation.QUARANTINE in {
        step.operation for step in successor.official_steps
    }
    restart_index = next(
        index
        for index, step in enumerate(successor.official_steps)
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert restart_index in successor.superseded_step_indexes


@pytest.mark.parametrize("xid_first", [True, False])
def test_xid_and_sxid_share_one_attempt_workflow(xid_first: bool) -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    xid = ("/v1/gpu-events/xid", xid_payload(11, f"xid-11-{xid_first}"))
    sxid = ("/v1/gpu-events/sxid", sxid_payload(f"sxid-11001-{xid_first}"))

    results = asyncio.run(
        post_faults(context, [xid, sxid] if xid_first else [sxid, xid])
    )

    assert {item["incident_id"] for item in results} == {results[0]["incident_id"]}
    assert {item["workflow_request_id"] for item in results} == {
        results[0]["workflow_request_id"]
    }
    assert len(context.store.list_workflows()) == 1
    incident = context.store.get_incident(results[0]["incident_id"])
    assert incident.job_id == "train"
    assert incident.attempt_id == "train-a001"
    assert incident.workload_identity_source == ("SOLE_ACTIVE_ATTEMPT_ON_NODE")
    workflow = context.store.get_workflow(results[0]["workflow_request_id"])
    operations = [step.operation for step in workflow.official_steps]
    assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in operations
    assert operations.count(WorkflowOperation.RESTART_WORKLOAD) == 1
    restart = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    assert restart.parameters["job_id"] == "train"
    assert restart.parameters["source_attempt_id"] == "train-a001"


def test_cross_node_sxid_upgrade_preserves_prior_xid_gpu() -> None:
    context = build_context()
    context.store.save_attempt_observation(two_node_observation())
    xid = xid_payload(11, "xid-node-a-before-sxid-node-b")
    sxid = sxid_payload("sxid-node-b-after-xid-node-a")
    sxid.update(
        {
            "node_id": "node-b",
            "fabric_partition": "cluster-a/node-b/local-nvswitch",
            "participating_gpu_uuids": ["GPU-b"],
        }
    )

    results = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid), ("/v1/gpu-events/sxid", sxid)]
        )
    )

    assert results[0]["incident_id"] == results[1]["incident_id"]
    assert results[0]["workflow_request_id"] == results[1]["workflow_request_id"]
    incident = context.store.get_incident(results[0]["incident_id"])
    assert incident.node_ids == ["node-a", "node-b"]
    assert incident.gpu_uuids == ["GPU-a", "GPU-b"]
    workflow = context.store.get_workflow(results[0]["workflow_request_id"])
    reset = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    )
    assert reset.parameters["gpu_uuids_by_node"] == {"node-b": ["GPU-b"]}


@pytest.mark.parametrize("xid_first", [True, False])
def test_node_reboot_is_not_downgraded_by_sxid_reset(xid_first: bool) -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    xid = ("/v1/gpu-events/xid", xid_payload(79, f"xid-79-{xid_first}"))
    sxid = ("/v1/gpu-events/sxid", sxid_payload(f"sxid-reset-{xid_first}"))

    results = asyncio.run(
        post_faults(context, [xid, sxid] if xid_first else [sxid, xid])
    )

    assert results[0]["workflow_request_id"] == (results[1]["workflow_request_id"])
    workflow = context.store.get_workflow(results[0]["workflow_request_id"])
    operations = {step.operation for step in workflow.official_steps}
    assert WorkflowOperation.RESTART_NODE in operations
    assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES not in operations


def test_multiple_active_attempts_disable_cross_type_grouping(caplog) -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    context.store.save_attempt_observation(
        observation(job_id="other", attempt_id="other-a001")
    )

    results = asyncio.run(
        post_faults(
            context,
            [
                ("/v1/gpu-events/xid", xid_payload(11, "ambiguous-xid")),
                ("/v1/gpu-events/sxid", sxid_payload("ambiguous-sxid")),
            ],
        )
    )

    assert results[0]["incident_id"] != results[1]["incident_id"]
    assert results[0]["workflow_request_id"] != (results[1]["workflow_request_id"])
    assert (
        context.orchestrator._evidence_operations.ambiguous_attempt_ownership_total()
        == 2
    )
    assert (
        sum(
            "ambiguous attempt ownership" in record.message for record in caplog.records
        )
        == 2
    )


def test_stale_observation_is_excluded_from_attempt_ownership() -> None:
    context = build_context()
    stale = copy_model(
        observation(job_id="stale-job", attempt_id="stale-a001"),
        observed_at=NOW - timedelta(minutes=10),
    )
    fresh = observation()
    context.store.save_attempt_observation(stale)
    context.store.save_attempt_observation(fresh)
    event = XidEvent.model_validate(
        {**xid_payload(11, "freshness-xid"), "ingested_at": NOW.isoformat()}
    )

    selected = context.orchestrator._evidence_operations.attempt_observation(event)
    snapshot = context.orchestrator._evidence_operations.ownership_metric_snapshot(
        now=NOW
    )

    assert selected is not None
    assert selected.attempt_id == fresh.attempt_id
    assert (
        context.orchestrator._evidence_operations.ambiguous_attempt_ownership_total()
        == 0
    )
    assert snapshot["current"][("cluster-a", "node-a")] == 1
    assert snapshot["stale"][("cluster-a", "node-a")] == 1


def test_gpu_ownership_disambiguates_multiple_active_attempts() -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    other = observation(job_id="other", attempt_id="other-a001")
    other = copy_model(
        other, containers=[copy_model(other.containers[0], gpu_uuids=["GPU-b"])]
    )
    context.store.save_attempt_observation(other)

    results = asyncio.run(
        post_faults(
            context,
            [
                ("/v1/gpu-events/xid", xid_payload(11, "gpu-owned-xid")),
                ("/v1/gpu-events/sxid", sxid_payload("gpu-owned-sxid")),
            ],
        )
    )

    assert results[0]["incident_id"] == results[1]["incident_id"]
    incident = context.store.get_incident(results[0]["incident_id"])
    assert incident.job_id == "train"
    assert incident.attempt_id == "train-a001"


def test_running_stronger_workflow_ignores_later_weaker_action() -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/sxid", sxid_payload("running-strong-sxid"))]
        )
    )[0]
    workflow = context.store.get_workflow(first["workflow_request_id"])
    context.store.save_workflow(
        copy_model(
            workflow, status=WorkflowStatus.RUNNING, execution_owner_id="executor-a"
        )
    )

    second = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(11, "later-weaker-xid"))]
        )
    )[0]

    assert second["incident_id"] == first["incident_id"]
    assert second["workflow_request_id"] == first["workflow_request_id"]
    assert len(context.store.list_workflows()) == 1


def test_later_weaker_event_does_not_reverse_existing_preemption() -> None:
    """Once the fabric reset preempted the GPU reset, a weaker event changes nothing.

    The trunk SXID on the running XID 48 workflow preempts the pending reset
    inside the DAG (same record, reset retired, fabric reset branch queued).
    A later XID 11 -- a bare job restart -- is absorbed: no step comes back
    from the superseded set, the action stays the fabric reset, and the
    duplicate of that event is a duplicate of the same record.
    """
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    context.store.save_attempt_observation(observation())
    low = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(48, "reverse-preempt-low"))]
        )
    )[0]
    low_workflow = context.store.get_workflow(low["workflow_request_id"])
    context.store.save_workflow(copy_model(low_workflow, status=WorkflowStatus.RUNNING))
    reset_index = next(
        index
        for index, step in enumerate(low_workflow.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    high = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/sxid", sxid_payload("reverse-preempt-high"))]
        )
    )[0]
    assert high["workflow_request_id"] == low_workflow.request_id, (
        "the stronger action preempts inside the record, not beside it"
    )
    high_workflow = context.store.get_workflow(high["workflow_request_id"])
    assert reset_index in high_workflow.superseded_step_indexes, (
        "the pending GPU reset is retired by the fabric reset"
    )
    assert high_workflow.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES", (
        "the record takes the stronger action"
    )
    incident_before_weaker = context.store.get_incident(high["incident_id"])

    weaker = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(11, "reverse-preempt-weaker"))]
        )
    )[0]
    duplicate = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(11, "reverse-preempt-weaker"))]
        )
    )[0]

    assert weaker["incident_id"] == high["incident_id"]
    assert weaker["workflow_request_id"] == high_workflow.request_id
    current = context.store.get_workflow(high_workflow.request_id)
    assert current.status is WorkflowStatus.RUNNING
    assert current.superseded_step_indexes == high_workflow.superseded_step_indexes, (
        "a weaker event brings nothing back from the superseded set"
    )
    node_actions = {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
    }
    assert [
        (index, step.operation, step.branch_id)
        for index, step in enumerate(current.official_steps)
        if step.operation in node_actions
    ] == [
        (index, step.operation, step.branch_id)
        for index, step in enumerate(high_workflow.official_steps)
        if step.operation in node_actions
    ], "a weaker event plans no node action and moves none"
    for operation in (
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
    ):
        assert (
            len(
                [step for step in current.official_steps if step.operation is operation]
            )
            == 1
        ), f"still one {operation.value} for the attempt"
    assert current.official_action == high_workflow.official_action
    current_incident = context.store.get_incident(high["incident_id"])
    assert current_incident.official_action == incident_before_weaker.official_action
    assert current_incident.effective_action == incident_before_weaker.effective_action
    assert current_incident.workflow_request_id == high_workflow.request_id
    assert duplicate["duplicate"]
    assert duplicate["incident_id"] == high["incident_id"]
    assert duplicate["workflow_request_id"] == high_workflow.request_id
    assert len(context.store.list_workflows(limit=10)) == 1


def test_disjoint_node_scope_uses_independent_workflows_or_shared_dag() -> None:
    independent = build_context()
    independent.orchestrator = IncidentOrchestrator(
        independent.store, workflow_preemption_enabled=True
    )
    node_a = asyncio.run(
        post_faults(
            independent, [("/v1/gpu-events/xid", xid_payload(48, "independent-node-a"))]
        )
    )[0]
    node_a_workflow = independent.store.get_workflow(node_a["workflow_request_id"])
    independent.store.save_workflow(
        copy_model(node_a_workflow, status=WorkflowStatus.RUNNING)
    )
    node_b_payload = xid_payload(79, "independent-node-b")
    node_b_payload.update(
        {"node_id": "node-b", "gpu_uuid": "GPU-b", "pci_bdf": "0000:ca:00.0"}
    )
    node_b = asyncio.run(
        post_faults(independent, [("/v1/gpu-events/xid", node_b_payload)])
    )[0]
    node_b_workflow = independent.store.get_workflow(node_b["workflow_request_id"])

    assert node_a["incident_id"] != node_b["incident_id"]
    assert node_a_workflow.request_id != node_b_workflow.request_id
    assert node_b_workflow.predecessor_workflow_id is None
    assert not node_b_workflow.preempt_predecessor
    claimed_a = independent.store.claim_workflow(
        node_a_workflow.request_id, "executor-node-a", node_a_workflow.fencing_token
    )
    claimed_b = independent.store.claim_workflow(
        node_b_workflow.request_id, "executor-node-b", node_b_workflow.fencing_token
    )
    assert claimed_a.execution_owner_id == "executor-node-a"
    assert claimed_b.execution_owner_id == "executor-node-b"
    assert (
        independent.store.get_incident(node_a["incident_id"]).workflow_request_id
        == node_a_workflow.request_id
    )
    assert (
        independent.store.get_incident(node_b["incident_id"]).workflow_request_id
        == node_b_workflow.request_id
    )

    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    context.store.save_attempt_observation(two_node_observation())
    low = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(48, "disjoint-node-low"))]
        )
    )[0]
    low_workflow = context.store.get_workflow(low["workflow_request_id"])
    context.store.save_workflow(copy_model(low_workflow, status=WorkflowStatus.RUNNING))
    high_payload = xid_payload(79, "disjoint-node-high")
    high_payload.update(
        {"node_id": "node-b", "gpu_uuid": "GPU-b", "pci_bdf": "0000:ca:00.0"}
    )

    high = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", high_payload)]))[0]

    high_workflow = context.store.get_workflow(high["workflow_request_id"])
    assert high_workflow.request_id == low_workflow.request_id
    assert high_workflow.predecessor_workflow_id is None
    assert not high_workflow.preempt_predecessor
    assert high_workflow.dag_enabled
    operations = [step.operation for step in high_workflow.official_steps]
    assert operations.count(WorkflowOperation.STOP_WORKLOADS) == 1
    assert operations.count(WorkflowOperation.RESTART_WORKLOAD) == 1
    assert operations.count(WorkflowOperation.CHECKPOINT_WORKLOADS) <= 1
    reset = next(
        step
        for step in high_workflow.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    )
    reboot = next(
        step
        for step in high_workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_NODE
    )
    restart = next(
        step
        for step in high_workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    restores = [
        (index, step)
        for index, step in enumerate(high_workflow.official_steps)
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING
    ]
    assert reset.node_ids == ["node-a"]
    assert reboot.node_ids == ["node-b"]
    assert len(restores) == 2
    assert {tuple(step.node_ids) for _, step in restores} == {("node-a",), ("node-b",)}
    assert set(restart.depends_on_step_indexes) == {index for index, _ in restores}
