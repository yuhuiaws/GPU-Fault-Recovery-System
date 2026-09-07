from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from gpu_fault.adapters import (
    ControlPlaneEvidenceAdapter,
    ManagedRecoveryObserverAdapter,
)
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    Environment,
    NotificationResult,
    NotificationStatus,
    WorkflowOperation,
)
from gpu_fault.store import SqliteStore
from tests._builders import asgi_client, copy_model

NOW = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)


class RecordingNotifier:
    def __init__(self) -> None:
        self.notifications = []

    def send(self, notification):
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="simulated-ses-xid94",
        )


def hyperpod_managed_job_profile():
    base = default_simulated_profile()
    return copy_model(
        base,
        cluster_id="hp-cluster",
        environment=Environment.HYPERPOD_EKS,
        profile_version="hp-job-auto-recovery-v1",
        capabilities=[
            copy_model(
                item,
                mode=CapabilityMode.DELEGATE,
                owner="hyperpod-managed-job-recovery",
                adapter="hyperpod-managed",
            )
            if item.capability
            in {CapabilityName.WORKLOAD_STOP, CapabilityName.WORKLOAD_RESTART}
            else item
            for item in base.capabilities
        ],
    )


def test_active_xid94_to_managed_job_recovery_and_admin_email() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    context.store.save_profile(hyperpod_managed_job_profile())

    async def scenario() -> None:
        async with asgi_client(context) as client:
            active_event = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "active-monitor-xid94",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": NOW.isoformat(),
                    "xid": 94,
                    "gpu_uuid": "GPU-94",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "job_id": "training-job-94",
                    "runtime_profile_version": ("hp-job-auto-recovery-v1"),
                    "workload_state": "ACTIVE",
                    "affected_workload_ids": ["training-job-94"],
                    "evidence_ref": ("s3://gpu-evidence/xid94/event.json"),
                },
            )
            assert active_event.status_code == 200
            decision = active_event.json()
            assert decision["official_action"] == "RESTART_APP"
            assert decision["action"] == "RESTART_WORKLOAD"
            assert decision["containment"] == "APPLICATION"
            assert decision["severity"] == "warning"
            assert decision["advisory_notification_id"]
            assert decision["investigatory_notification_id"]

            workflow = await client.get(
                f"/v1/workflows/{decision['workflow_request_id']}"
            )
            assert workflow.status_code == 200
            workflow_body = workflow.json()
            operations = [item["operation"] for item in workflow_body["official_steps"]]
            assert operations == [
                "FREEZE_EVIDENCE",
                "STOP_WORKLOADS",
                "RESTART_WORKLOAD",
            ]
            assert not {
                "MARK_UNSCHEDULABLE",
                "RESET_GPU",
                "RESTART_NODE",
                "REPLACE_NODE",
            }.intersection(operations)
            restart_step = workflow_body["official_steps"][-1]
            assert restart_step["execution_owner"] == ("hyperpod-managed-job-recovery")
            stop_step = workflow_body["official_steps"][-2]
            assert stop_step["execution_owner"] == ("hyperpod-managed-job-recovery")

            notification = await client.get(
                "/v1/advisory-notifications/" + decision["advisory_notification_id"]
            )
            assert notification.status_code == 200
            assert "RESTART_APP" in notification.json()["body_text"]
            assert "customer administrator" in (notification.json()["body_text"])

            dispatch = await client.post(
                "/v1/advisory-notifications/dispatch", json={"limit": 10}
            )
            assert dispatch.status_code == 200
            assert dispatch.json()["sent"] == 1
            assert len(notifier.notifications) == 2
            investigatory = next(
                item
                for item in notifier.notifications
                if item.notification_id == decision["investigatory_notification_id"]
            )
            assert "NVIDIA Immediate Action：RESTART_APP" in (investigatory.body_text)
            assert "NVIDIA Investigatory Action：IGNORE (sympathetic)" in (
                investigatory.body_text
            )

            terminal = await client.post(
                "/v1/attempts/terminal",
                json={
                    "cluster_id": "hp-cluster",
                    "environment": "hyperpod-eks",
                    "job_id": "training-job-94",
                    "attempt_id": "training-job-94-attempt-1",
                    "terminal_status": "FAILED",
                    "ended_at": NOW.isoformat(),
                    "rank_exit_status": [
                        {
                            "rank": 0,
                            "exit_code": 1,
                            "node_id": "worker-1",
                            "finished_at": NOW.isoformat(),
                        }
                    ],
                    "allocation": [
                        {"node_id": "worker-1", "rank": 0, "gpu_uuids": ["GPU-94"]}
                    ],
                    "runtime_profile_version": ("hp-job-auto-recovery-v1"),
                },
            )
            assert terminal.status_code == 200
            terminal_decision = terminal.json()
            assert terminal_decision["status"] == "NO_ACTION"
            assert terminal_decision["recovery_plan_id"] is None
            assert "without a second restart" in (terminal_decision["reason"])

    asyncio.run(scenario())


def test_active_executor_xid94_waits_for_managed_recovery(tmp_path) -> None:
    profile = hyperpod_managed_job_profile()
    profile = copy_model(
        profile,
        capabilities=[
            copy_model(
                item, owner="gpu-fault-control-plane", adapter="control-plane-evidence"
            )
            if item.capability is CapabilityName.EVIDENCE_CAPTURE
            else item
            for item in profile.capabilities
        ],
    )
    store = SqliteStore(str(tmp_path / "active-xid94.db"))
    managed_owner = "hyperpod-managed-job-recovery"
    context = ApplicationContext(
        store=store,
        production_executor_config=ProductionExecutorConfig(
            enabled=True,
            executor_id="executor-xid94",
            allowed_operations=frozenset(
                {
                    WorkflowOperation.FREEZE_EVIDENCE,
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                }
            ),
        ),
        production_adapters=[
            ControlPlaneEvidenceAdapter(),
            ManagedRecoveryObserverAdapter({managed_owner}),
        ],
        execution_token="e" * 32,
    )
    context.store.save_profile(profile)

    async def scenario() -> None:
        headers = {"X-GPU-Fault-Execution-Token": "e" * 32}
        async with asgi_client(context) as client:
            event = await client.post(
                "/v1/gpu-events/xid",
                json={
                    "event_id": "active-executor-xid94",
                    "cluster_id": "hp-cluster",
                    "node_id": "worker-1",
                    "observed_at": NOW.isoformat(),
                    "xid": 94,
                    "gpu_uuid": "GPU-94",
                    "product": "H100",
                    "driver_branch": 575,
                    "cuda_version": "12.9",
                    "runtime_profile_version": (profile.profile_version),
                    "workload_state": "ACTIVE",
                    "affected_workload_ids": ["training-job-94"],
                },
            )
            workflow_id = event.json()["workflow_request_id"]

            first = await client.post("/v1/workflows/dispatch", headers=headers)
            assert first.json()["waiting"] == 1
            workflow = (await client.get(f"/v1/workflows/{workflow_id}")).json()
            stop_operation_id = workflow["step_executions"][-1]["adapter_operation_id"]

            stop_confirmed = await client.post(
                f"/v1/workflows/{workflow_id}/execute",
                headers=headers,
                json={
                    "expected_fencing_token": 1,
                    "confirmed_adapter_operation_ids": [stop_operation_id],
                },
            )
            assert stop_confirmed.json()["status"] == "RUNNING"
            workflow = (await client.get(f"/v1/workflows/{workflow_id}")).json()
            restart_operation_id = workflow["step_executions"][-1][
                "adapter_operation_id"
            ]

            completed = await client.post(
                f"/v1/workflows/{workflow_id}/execute",
                headers=headers,
                json={
                    "expected_fencing_token": 1,
                    "confirmed_adapter_operation_ids": [restart_operation_id],
                },
            )
            incident = (
                await client.get(f"/v1/incidents/{event.json()['incident_id']}")
            ).json()

            assert completed.json()["status"] == "SUCCEEDED"
            assert incident["state"] == "RECOVERED"

    asyncio.run(scenario())
