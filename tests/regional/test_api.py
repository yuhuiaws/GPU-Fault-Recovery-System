from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from gpu_fault.models import CapabilityMode, CapabilityName, TerminalEvent
from gpu_fault.watcher import AllocationCompleteness, FailureDetectedEvent
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
)


def test_installation_resource_registry_api_is_execution_token_protected() -> None:
    token = "installation-registry-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    context.regional_mode = True
    resource = InstallationResource(
        site_id="site-a",
        resource_key="aws/nlb",
        resource_type="nlb",
        resource_id="gpu-fault-site-a",
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
    )
    snapshot = InstallationResourceSnapshot(site_id="site-a", resources=[resource])

    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            forbidden = await client.post(
                "/v1/installation-resources/sync", json=snapshot.model_dump(mode="json")
            )
            assert forbidden.status_code == 403, "registry sync allowed no token"

            synced = await client.post(
                "/v1/installation-resources/sync",
                json=snapshot.model_dump(mode="json"),
                headers={"X-GPU-Fault-Execution-Token": token},
            )
            assert synced.status_code == 200, "registry sync failed"

            listed = await client.get(
                "/v1/installation-resources",
                params={"site_id": "site-a"},
                headers={"X-GPU-Fault-Execution-Token": token},
            )
            assert listed.status_code == 200, "registry list failed"
            assert listed.json()[0]["resource_key"] == "aws/nlb", (
                "registry list returned the wrong resource"
            )

            pending = resource.model_copy(
                update={"status": InstallationResourceStatus.DELETE_PENDING}
            )
            updated = await client.put(
                "/v1/installation-resources/site-a/aws/nlb",
                json=pending.model_dump(mode="json"),
                headers={"X-GPU-Fault-Execution-Token": token},
            )
            assert updated.status_code == 200, "registry status update failed"
            assert updated.json()["status"] == "DELETE_PENDING", (
                "registry status was not persisted"
            )

    asyncio.run(run_scenario())


def test_synthetic_replacement_requires_flag_and_execution_token(monkeypatch) -> None:
    token = "synthetic-replacement-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    payload = {
        "event_id": "warm-spare-e2e",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_profile_version": "simulated-v1",
        "job_id": "job-a",
        "attempt_id": "job-a-a001",
        "affected_workload_ids": ["training/job/job-a"],
        "gpu_uuids": ["GPU-a"],
        "reason": "synthetic warm-spare E2E",
        "synthetic": True,
    }

    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            disabled = await client.post(
                "/v1/admin/test/node-replacement",
                json=payload,
                headers={"X-GPU-Fault-Execution-Token": token},
            )
            assert disabled.status_code == 404

            monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
            forbidden = await client.post(
                "/v1/admin/test/node-replacement", json=payload
            )
            assert forbidden.status_code == 403

            accepted = await client.post(
                "/v1/admin/test/node-replacement",
                json=payload,
                headers={"X-GPU-Fault-Execution-Token": token},
            )
            assert accepted.status_code == 200
            workflow_id = accepted.json()["workflow_request_ids"][0]
            workflow = context.store.get_workflow(workflow_id)
            replace_step = next(
                step
                for step in workflow.official_steps
                if step.operation.value == "REPLACE_NODE"
            )
            assert replace_step.parameters == {
                "replacement_strategy": ("HEALTHY_WARM_SPARE_ONLY")
            }

    asyncio.run(run_scenario())


def test_a_grouped_node_health_marker_points_at_the_surviving_incident(
    monkeypatch,
) -> None:
    """A merged finding's marker has to name the incident that owns it.

    ``NodeHealthFinding.marker()`` guesses ``inc-<event_id>`` because it is
    built before ingestion picks an incident, and a finding that lands on a node
    with a running attempt is grouped into that attempt's incident, which keeps
    its own generated id. The marker was then left pointing at a record nobody
    ever persisted, which defeats both guards that read it: the completion
    handler cannot see that recovery is already owned by the incident's
    workflow, so a terminal attempt inside the marker window starts a *second*
    replacement for the same fault, and ``marker_blocks_spare`` can never reach
    the ``SUCCEEDED`` workflow that would return the node to the spare pool.
    """
    monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
    token = "synthetic-replacement-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    context.store.save_attempt_observation(
        attempt_observation(
            "train",
            "train-a001",
            datetime.now(timezone.utc),
            containers=[
                container_observation(
                    "pod-train-a001",
                    "trainer-train-a001",
                    0,
                    "node-a",
                    gpu_uuids=["GPU-a"],
                )
            ],
            workload_ids=["gpu-fault-system/pytorchjob/train"],
        )
    )
    payload = {
        "event_id": "warm-spare-grouped",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_profile_version": "simulated-v1",
        "job_id": "train",
        "attempt_id": "train-a001",
        "affected_workload_ids": ["gpu-fault-system/pytorchjob/train"],
        "gpu_uuids": ["GPU-a"],
        "reason": "synthetic warm-spare E2E",
        "synthetic": True,
    }

    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            accepted = await client.post(
                "/v1/admin/test/node-replacement",
                json=payload,
                headers={"X-GPU-Fault-Execution-Token": token},
            )
            assert accepted.status_code == 200, accepted.text

    asyncio.run(run_scenario())

    incident = context.store.get_incident_by_event("warm-spare-grouped")
    assert incident is not None, "grouped finding did not produce an incident"
    assert incident.incident_id != "inc-warm-spare-grouped", (
        "finding was not grouped into the attempt's incident"
    )
    marker = next(
        item
        for item in context.store.list_markers()
        if item.marker_id == "marker-warm-spare-grouped"
    )
    assert marker.incident_id == incident.incident_id, (
        "marker points at an incident that was never persisted"
    )
    assert context.store.list_markers_for_incident(incident.incident_id) == [marker], (
        "marker is not reachable from the incident that owns the finding"
    )


def test_http_failure_detection_creates_one_containment_workflow(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    event = FailureDetectedEvent(
        cluster_id=failed_event.cluster_id,
        job_id=failed_event.job_id,
        attempt_id=failed_event.attempt_id,
        detected_at=failed_event.ended_at,
        runtime_profile_version=failed_event.runtime_profile_version,
        workload_ids=["training/pytorchjob/distributed-training"],
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a", "GPU-b"],
        first_failed_rank=0,
        node_id="node-a",
        exit_code=1,
        reason="critical container exited non-zero",
        allocation_completeness=AllocationCompleteness.COMPLETE,
    )

    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                "/v1/attempts/failure-detected", json=event.model_dump(mode="json")
            )
            second = await client.post(
                "/v1/attempts/failure-detected", json=event.model_dump(mode="json")
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert not first.json()["duplicate"]
        assert second.json()["duplicate"]
        assert (
            second.json()["workflow_request_id"]
            == (first.json()["workflow_request_id"])
        )

    asyncio.run(run_scenario())


def test_http_terminal_to_triage_to_simulated_execution(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/attempts/terminal", json=failed_event.model_dump(mode="json")
            )
            assert response.status_code == 200
            pending = response.json()
            assert pending["status"] == "PENDING_TRIAGE"

            triage = await client.post(
                "/v1/triage-results",
                json={
                    "request_id": pending["diagnostic_request_id"],
                    "attempt_id": failed_event.attempt_id,
                    "completed_at": failed_event.ended_at.isoformat(),
                    "findings": [
                        {"node_id": "node-a", "outcome": "PASS"},
                        {"node_id": "node-b", "outcome": "PASS"},
                    ],
                },
            )
            assert triage.status_code == 200
            plan_id = triage.json()["recovery_plan_id"]

            plan = await client.get(f"/v1/recovery-plans/{plan_id}")
            assert plan.status_code == 200
            assert plan.json()["status"] == "PENDING"

            operation = await client.post(f"/v1/recovery-plans/{plan_id}/simulate")
            assert operation.status_code == 200
            assert operation.json()["status"] == "SUCCEEDED"

    asyncio.run(run_scenario())


def test_http_xid_marker_is_reused_by_failed_attempt(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            xid = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "xid-http-79",
                    "cluster_id": failed_event.cluster_id,
                    "node_id": "node-a",
                    "gpu_uuid": "GPU-a",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                    "observed_at": failed_event.ended_at.isoformat(),
                    "xid": 79,
                },
            )
            assert xid.status_code == 200
            assert xid.json()["action"] == "REBOOT_NODE"
            workflow_id = xid.json()["workflow_request_id"]

            workflow = await client.get(f"/v1/workflows/{workflow_id}")
            assert workflow.status_code == 200
            assert workflow.json()["status"] == "PENDING"

            proactive = await client.post(
                f"/v1/workflows/{workflow_id}/simulate",
                json={"expected_fencing_token": 1},
            )
            assert proactive.status_code == 200
            assert proactive.json()["status"] == "SUCCEEDED"

            terminal = await client.post(
                "/v1/attempts/terminal", json=failed_event.model_dump(mode="json")
            )
            assert terminal.status_code == 200
            decision = terminal.json()
            assert decision["status"] == "PLAN_CREATED"
            assert decision["matched_marker_ids"] == ["marker-xid-http-79"]

            plan = await client.get(
                f"/v1/recovery-plans/{decision['recovery_plan_id']}"
            )
            assert plan.status_code == 200
            assert [step["action"] for step in plan.json()["steps"]] == [
                "RESTART_WORKLOAD"
            ]
            assert (
                plan.json()["steps"][0]["parameters"]["requires_incident_state"]
                == "RECOVERED"
            )

    asyncio.run(run_scenario())


def test_http_managed_hyperpod_advisory_preview_and_send(
    context: ApplicationContext, ended_at
) -> None:
    default = context.store.get_profile("simulated-v1")
    managed = copy_model(
        default,
        cluster_id="hp-cluster",
        profile_version="hp-managed-v1",
        capabilities=[
            copy_model(
                item,
                mode=CapabilityMode.DELEGATE,
                owner="hyperpod-managed-node-recovery",
                adapter="hyperpod-managed",
            )
            if item.capability is CapabilityName.NODE_REBOOT
            else item
            for item in default.capabilities
        ],
    )
    context.store.save_profile(managed)

    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            xid = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "managed-xid-79",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "gpu_uuid": "GPU-a",
                    "observed_at": ended_at.isoformat(),
                    "xid": 79,
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "hp-managed-v1",
                    "workload_state": "IDLE",
                    "evidence_ref": "s3://evidence/managed-xid-79.json",
                },
            )
            assert xid.status_code == 200
            incident_id = xid.json()["incident_id"]
            auto_notification_id = xid.json()["advisory_notification_id"]
            assert auto_notification_id

            preview = await client.post(
                f"/v1/incidents/{incident_id}/advisory-notifications"
            )
            assert preview.status_code == 200
            notification = preview.json()
            assert (
                "Decision owner: customer administrator" in (notification["body_text"])
            )
            assert notification["evidence_refs"] == [
                "s3://evidence/managed-xid-79.json"
            ]

            duplicate_preview = await client.post(
                f"/v1/incidents/{incident_id}/advisory-notifications"
            )
            assert (
                duplicate_preview.json()["notification_id"]
                == (notification["notification_id"])
            )
            assert notification["notification_id"] == (auto_notification_id)

            delivery = await client.post(
                f"/v1/advisory-notifications/{notification['notification_id']}/send"
            )
            assert delivery.status_code == 200
            assert delivery.json()["status"] == "SKIPPED"

            dispatch = await client.post(
                "/v1/advisory-notifications/dispatch", json={"limit": 10}
            )
            assert dispatch.status_code == 200
            assert dispatch.json()["attempted"] == 2
            assert dispatch.json()["skipped"] == 2

    asyncio.run(run_scenario())


def test_http_advisory_rejects_non_managed_incident(
    context: ApplicationContext, ended_at
) -> None:
    async def run_scenario() -> None:
        async with asgi_client(context) as client:
            xid = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "custom-xid-79",
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "observed_at": ended_at.isoformat(),
                    "xid": 79,
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": "simulated-v1",
                    "workload_state": "IDLE",
                },
            )
            incident_id = xid.json()["incident_id"]
            assert xid.json()["advisory_notification_id"] is None
            response = await client.post(
                f"/v1/incidents/{incident_id}/advisory-notifications"
            )
            assert response.status_code == 409
            assert "no HyperPod-managed" in response.json()["detail"]

    asyncio.run(run_scenario())


def test_create_app_configures_root_logging(monkeypatch) -> None:
    """The API process must configure logging, not inherit silence.

    uvicorn's default LOGGING_CONFIG has no ``root`` entry, so without
    this the control plane runs with root at WARNING and no handlers:
    every LOGGER.info is discarded and every warning reaches stderr
    through logging.lastResort with no timestamp or logger name. A
    misconfigured deployment then looks indistinguishable from a
    healthy one in the logs.
    """

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        root.handlers = []
        root.setLevel(logging.WARNING)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", "INFO")

        create_app(build_context())

        assert root.handlers, "create_app left root without a handler"
        assert logging.getLogger("gpu_fault.notification_service").isEnabledFor(
            logging.INFO
        )
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_create_app_keeps_an_existing_logging_setup(monkeypatch) -> None:
    """Never stomp on a host process's own logging configuration."""

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        existing = logging.NullHandler()
        root.handlers = [existing]
        root.setLevel(logging.ERROR)
        monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", "DEBUG")

        create_app(build_context())

        assert root.handlers == [existing]
        assert root.level == logging.ERROR
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
