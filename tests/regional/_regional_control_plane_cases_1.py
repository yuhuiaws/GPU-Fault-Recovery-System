from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.execution.restart_budget_preflight import (
    release_unattempted_restart_reservations,
    reservation_id,
)
from gpu_fault.fleet import (
    AgentHeartbeat,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.hyperpod import HyperPodAction, HyperPodSubmissionRecord
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import (
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import NotFoundError, SqliteStore, WorkflowLeaseError
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    build_store,
    copy_model,
    workflow_step_execution,
)
from tests.regional._regional_support import (
    NOW,
    TOKEN_A,
    TOKEN_B,
    RecordingNotifier,
    regional_environment,
    registration,
    remote_context,
    terminal,
    workflow_state,
)


@pytest.mark.parametrize("durable", [False, True])
def test_attempt_index_is_cluster_scoped(tmp_path, durable: bool) -> None:
    store = SqliteStore(str(tmp_path / "regional.db")) if durable else build_store()
    first = terminal("cluster-a")
    second = terminal("cluster-b")

    assert store.save_event_if_absent(first)
    assert store.save_event_if_absent(second)
    assert store.get_event_by_attempt("cluster-a", "same-attempt") == first
    assert store.get_event_by_attempt("cluster-b", "same-attempt") == second


def test_running_observations_are_scoped_without_eager_decisions() -> None:
    store = build_store()
    for cluster_id in ("cluster-a", "cluster-b"):
        store.save_attempt_observation(
            attempt_observation(
                "same-job",
                "same-attempt",
                NOW,
                cluster_id=cluster_id,
                workload_ids=["training/job/same-job"],
                runtime_profile_version="profile-v1",
                restart_budget=1,
            )
        )

    assert {
        item.cluster_id for item in store.list_attempt_observations("cluster-a")
    } == {"cluster-a"}
    assert {
        item.cluster_id for item in store.list_attempt_observations("cluster-b")
    } == {"cluster-b"}
    for cluster_id in ("cluster-a", "cluster-b"):
        with pytest.raises(NotFoundError):
            store.get_decision_by_attempt(cluster_id, "same-attempt")
        with pytest.raises(NotFoundError):
            store.get_restart_budget(cluster_id, "same-job")


def test_regional_api_authenticates_cluster_and_rejects_spoofing() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))

    async def scenario() -> None:
        async with asgi_client(context) as client:
            accepted = await client.post(
                "/v1/regional/executors/claim",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={"executor_id": "executor-a"},
            )
            assert accepted.status_code == 200

            wrong_token = await client.post(
                "/v1/regional/executors/claim",
                headers={
                    "Authorization": f"Bearer {TOKEN_B}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={"executor_id": "executor-a"},
            )
            assert wrong_token.status_code == 403

            spoofed_payload = await client.post(
                "/v1/attempts/terminal",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json=terminal("cluster-b").model_dump(mode="json"),
            )
            assert spoofed_payload.status_code == 403

            nested_spoof = await client.post(
                "/v1/regional/executors/claim",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={
                    "executor_id": "executor-a",
                    "batch": {"items": [{"cluster_id": "cluster-b"}]},
                },
            )
            assert nested_spoof.status_code == 403

    asyncio.run(scenario())


def test_regional_spare_health_uses_fleet_registry() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    secret = "fleet-secret-" + "x" * 32
    context.fleet_registry = FleetRegistry(context.store, secret, now=lambda: NOW)
    heartbeat = AgentHeartbeat(
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=3,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="policy-v1",
        runtime_profile_version="profile-v1",
        config_digest="b" * 64,
        allowed_operations=[
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ],
        observed_at=NOW,
    )
    context.fleet_registry.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, secret)
        )
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/regional/executors/spares/health",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={"cluster_id": "cluster-a", "node_aliases": ["node-a"]},
            )

        assert response.status_code == 200
        assert response.json() == {"ready": True, "reasons": []}

    asyncio.run(scenario())


def test_regional_spare_health_rejects_cross_cluster_payload() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/regional/executors/spares/health",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={"cluster_id": "cluster-b", "node_aliases": ["probe-node"]},
            )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "authenticated cluster does not match all payload cluster_id values"
        )

    asyncio.run(scenario())


def test_remote_command_lease_and_result_advance_adapter() -> None:
    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    context = workflow_state()

    waiting = adapter.execute(context)
    assert waiting.status.value == "WAITING"
    assert waiting.details["mutation_submitted_by_control_plane"] is False

    assert not store.claim_remote_commands(
        "cluster-b", "wrong-cluster", limit=1, lease_seconds=60
    )
    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    assert len(claimed) == 1
    command = claimed[0]
    renewed = store.renew_remote_command_lease(
        "cluster-a",
        command.command_id,
        "executor-a",
        command.lease_token,
        lease_seconds=120,
    )
    assert renewed.lease_expires_at > command.lease_expires_at

    with pytest.raises(WorkflowLeaseError):
        store.complete_remote_command(
            "cluster-a",
            command.command_id,
            RemoteCommandResult(
                lease_token="stale-token", status=RemoteCommandStatus.SUCCEEDED
            ),
        )

    store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=command.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={"node": "node-a"},
        ),
    )
    succeeded = adapter.execute(context)
    assert succeeded.status.value == "SUCCEEDED"
    assert succeeded.details == {"node": "node-a"}
    assert store.get_remote_command(command.command_id).last_lease_owner == "executor-a"


def test_empty_namespace_allowlist_fails_closed() -> None:
    store = build_store()
    store.save_regional_cluster(
        copy_model(registration("cluster-a", TOKEN_A), allowed_namespaces=[])
    )
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    base = workflow_state()
    step = copy_model(base.step, workload_ids=["training/pytorchjob/training-a"])

    outcome = adapter.execute(replace(base, step=step))

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "no allowed workload namespaces" in outcome.error
    assert store.list_remote_commands() == []


def test_remote_command_digest_includes_step_parameters() -> None:
    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    base = workflow_state()
    first = copy_model(base.step, parameters={"target_driver_branch": 575})
    second = copy_model(base.step, parameters={"target_driver_branch": 580})

    adapter.execute(replace(base, step=first))
    outcome = adapter.execute(replace(base, step=second))

    # The digest tells the two parameterisations apart, but the first command
    # is still open for this step, so the second is held rather than minted
    # (open-command invariant, ARCH-D5).
    commands = store.list_remote_commands()
    assert len(commands) == 1, "a second command was minted beside an open one"
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OPEN_SIBLING_COMMAND", outcome.details
    assert outcome.details["held_command_id"] != commands[0].command_id, (
        "the rewritten parameters must produce a different command digest"
    )
    assert outcome.details["remote_command_id"] == commands[0].command_id, (
        "the hold must name the open sibling"
    )


def test_remote_command_digest_ignores_dag_scheduling_metadata() -> None:
    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    base = workflow_state()
    initial = base.step
    rewritten = copy_model(
        initial, branch_id="branch:node-a", depends_on_step_indexes=[3, 7]
    )

    adapter.execute(replace(base, step=initial))
    adapter.execute(replace(base, step=rewritten))

    assert len(store.list_remote_commands()) == 1


def test_remote_command_digest_includes_rebound_nodes() -> None:
    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    base = workflow_state()
    rebound = copy_model(base.step, node_ids=["node-b"])

    adapter.execute(replace(base, step=base.step))
    outcome = adapter.execute(replace(base, step=rebound))

    # Rebinding the node changes the digest; the open first command holds the
    # second instead of letting two actions target the step (ARCH-D5).
    commands = store.list_remote_commands()
    assert len(commands) == 1, "a second command was minted beside an open one"
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OPEN_SIBLING_COMMAND", outcome.details
    assert outcome.details["held_command_id"] != commands[0].command_id, (
        "the rebound node must produce a different command digest"
    )


def test_failed_remote_restart_leaves_the_reservation_to_terminalization() -> None:
    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    base = workflow_state()
    step = copy_model(
        base.step,
        operation=WorkflowOperation.RESTART_WORKLOAD,
        workload_ids=["training/pytorchjob/training-a"],
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "training-a",
            "source_attempt_id": "training-a-a001",
            "source_gpu_count": 8,
            "restart_budget": 1,
        },
    )
    workflow = copy_model(base.workflow, official_steps=[step])
    reservation = reservation_id(workflow, 0)
    context = replace(base, workflow=workflow, step=step, idempotency_key=reservation)
    # The preflight's reservation; dispatch only signs it.
    store.reserve_job_restart("cluster-a", "training-a", 1, reservation)
    adapter.execute(context)
    command = store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=1,
        lease_seconds=60,
        execution_owners={"gpu-fault-kubernetes-adapter"},
    )[0]
    # What the cluster executor forwards from the data-plane guard's refusal:
    # the adapter's outcome details, marker included.
    store.complete_remote_command(
        "cluster-a",
        command.command_id,
        RemoteCommandResult(
            lease_token=command.lease_token,
            status=RemoteCommandStatus.FAILED,
            error="restart rejected",
            details={"restart_submitted": False},
        ),
    )

    failed = adapter.execute(context)
    held = store.get_restart_budget("cluster-a", "training-a")

    # Dispatch neither reserves nor releases: the reservation stays with the
    # step until the workflow's terminal write decides whether the restart
    # ever left the gate.
    assert failed.status is WorkflowStepStatus.FAILED
    assert failed.details["restart_submitted"] is False
    assert held.restart_count == 1
    assert held.reservation_ids == [reservation]

    # The terminal write reads the marker off the step record and releases.
    release_unattempted_restart_reservations(
        store,
        copy_model(
            workflow,
            step_executions=[
                workflow_step_execution(
                    0,
                    WorkflowOperation.RESTART_WORKLOAD,
                    WorkflowStepStatus.FAILED,
                    details=failed.details,
                )
            ],
        ),
    )
    released = store.get_restart_budget("cluster-a", "training-a")
    assert released.restart_count == 0
    assert released.reservation_ids == []


def test_remote_claim_filters_execution_owners_and_legacy_api() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        context.store, owners={"gpu-fault-kubernetes-adapter", "gpu-fault-node-agent"}
    )
    adapter.execute(remote_context("kubernetes", "gpu-fault-kubernetes-adapter"))
    adapter.execute(remote_context("node", "gpu-fault-node-agent"))

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        headers = {
            "Authorization": f"Bearer {TOKEN_A}",
            "X-GPU-Fault-Cluster-ID": "cluster-a",
        }
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            legacy = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={"executor_id": "legacy-executor"},
            )
            assert legacy.status_code == 200
            legacy_commands = legacy.json()["commands"]
            assert len(legacy_commands) == 1
            assert (
                legacy_commands[0]["step"]["execution_owner"]
                == "gpu-fault-kubernetes-adapter"
            )

            node = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={
                    "executor_id": "node-executor",
                    "execution_owners": ["gpu-fault-node-agent"],
                },
            )
            assert node.status_code == 200
            node_commands = node.json()["commands"]
            assert len(node_commands) == 1
            assert node_commands[0]["step"]["execution_owner"] == "gpu-fault-node-agent"

    asyncio.run(scenario())


def test_executor_protocol_gate_blocks_before_command_lease(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION", "2")
    monkeypatch.setenv("GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS", "")
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    RegionalRemoteWorkflowAdapter(
        context.store, owners={"gpu-fault-kubernetes-adapter"}
    ).execute(remote_context("kubernetes", "gpu-fault-kubernetes-adapter"))

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        headers = {
            "Authorization": f"Bearer {TOKEN_A}",
            "X-GPU-Fault-Cluster-ID": "cluster-a",
        }
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            legacy = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={"executor_id": "legacy"},
            )
            assert legacy.status_code == 503
            assert "expected 2, got 1" in legacy.text
            assert context.store.remote_command_stats()["by_status"]["PENDING"] == 1

            current = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={"executor_id": "current", "executor_protocol_version": 2},
            )
            assert current.status_code == 200
            assert len(current.json()["commands"]) == 1

    asyncio.run(scenario())


def test_executor_readiness_rejects_an_executor_that_cannot_claim() -> None:
    """Readiness must fail for the executor that claims nothing.

    The old probe fetched the anonymous /healthz, which answers 200 no
    matter which executor asks, so an executor missing the node-action
    adapter stayed Ready next to commands it could never claim.
    """

    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        context.store, owners={"gpu-fault-kubernetes-adapter", "gpu-fault-node-agent"}
    )
    adapter.execute(remote_context("kubernetes", "gpu-fault-kubernetes-adapter"))
    adapter.execute(remote_context("node", "gpu-fault-node-agent"))

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        headers = {
            "Authorization": f"Bearer {TOKEN_A}",
            "X-GPU-Fault-Cluster-ID": "cluster-a",
        }
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            anonymous = await client.post(
                "/v1/regional/executors/readiness",
                json={"executor_id": "e1", "execution_owners": []},
            )
            # 401 with no credential, 403 with the wrong one -- either
            # way the anonymous /healthz probe's 200 is gone.
            assert anonymous.status_code == 401

            wrong_cluster = await client.post(
                "/v1/regional/executors/readiness",
                headers={
                    "Authorization": f"Bearer {TOKEN_B}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json={"executor_id": "e1", "execution_owners": []},
            )
            assert wrong_cluster.status_code == 403

            partial = await client.post(
                "/v1/regional/executors/readiness",
                headers=headers,
                json={
                    "executor_id": "kubernetes-only",
                    "execution_owners": ["gpu-fault-kubernetes-adapter"],
                },
            )
            assert partial.status_code == 503
            body = partial.json()
            assert body["ready"] is False
            assert body["unsupported_execution_owners"] == ["gpu-fault-node-agent"]
            assert body["pending_commands"] == 2

            complete = await client.post(
                "/v1/regional/executors/readiness",
                headers=headers,
                json={
                    "executor_id": "full",
                    "execution_owners": [
                        "gpu-fault-kubernetes-adapter",
                        "gpu-fault-node-agent",
                    ],
                    "last_successful_claim_age_seconds": 3.0,
                },
            )
            assert complete.status_code == 200
            assert complete.json()["ready"] is True

            stale = await client.post(
                "/v1/regional/executors/readiness",
                headers=headers,
                json={
                    "executor_id": "full",
                    "execution_owners": [
                        "gpu-fault-kubernetes-adapter",
                        "gpu-fault-node-agent",
                    ],
                    "last_successful_claim_age_seconds": 4000.0,
                },
            )
            assert stale.status_code == 503
            assert any(
                "last successful claim" in reason for reason in stale.json()["reasons"]
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_unclaimed_remote_commands_are_dead_lettered(backend: str, tmp_path) -> None:
    """A command nobody claims must fail, not wait forever.

    Before this the workflow stayed WAITING with no timeout, no terminal
    state and no signal, so a recovery that never executed looked exactly
    like one still in progress.
    """

    store = (
        build_store()
        if backend == "memory"
        else SqliteStore(str(tmp_path / "store.sqlite"))
    )
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(store, owners={"gpu-fault-node-agent"})
    waiting = adapter.execute(remote_context("orphan", "gpu-fault-node-agent"))
    assert waiting.status is WorkflowStepStatus.WAITING

    fresh = store.expire_unclaimed_remote_commands(
        older_than=datetime.now(timezone.utc) - timedelta(seconds=900), limit=10
    )
    expired = store.expire_unclaimed_remote_commands(
        older_than=datetime.now(timezone.utc) + timedelta(seconds=1), limit=10
    )

    assert fresh == 0
    assert expired == 1
    stats = store.remote_command_stats()
    assert stats["unclaimed_expired_total"] == 1
    assert stats["by_status"]["FAILED"] == 1
    # The workflow must now see a failure through the existing FAILED
    # branch, which is the whole reason this lands on FAILED rather than
    # on a status the orchestrator would not recognise.
    outcome = adapter.execute(remote_context("orphan", "gpu-fault-node-agent"))
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "no cluster executor claimed" in (outcome.error or "")

    health = store.remote_command_cluster_health("cluster-a")
    assert health["pending_total"] == 0
    assert health["open_total"] == 0


def test_regional_api_sends_gpu_reset_completion_once() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    remote = RegionalRemoteWorkflowAdapter(
        context.store, owners={"gpu-fault-node-agent"}
    )
    base = remote_context("gpu-reset", "gpu-fault-node-agent")
    step = copy_model(
        base.step,
        operation=WorkflowOperation.RESET_GPU,
        gpu_uuids=["GPU-a"],
        workload_ids=["training/pytorchjob/training-a"],
    )
    workflow = copy_model(base.workflow, official_steps=[step])
    remote.execute(
        replace(
            base,
            workflow=workflow,
            step=step,
            idempotency_key="workflow-gpu-reset/0/RESET_GPU",
        )
    )
    command = context.store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=1,
        lease_seconds=60,
        execution_owners={"gpu-fault-node-agent"},
    )[0]
    wake_calls = []
    context.dispatcher.wake = lambda: wake_calls.append(True)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        headers = {
            "Authorization": f"Bearer {TOKEN_A}",
            "X-GPU-Fault-Cluster-ID": "cluster-a",
        }
        payload = RemoteCommandResult(
            lease_token=command.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={
                "node_results": {
                    "node-a": {"status": "SUCCEEDED", "reset_gpu_uuids": ["GPU-a"]}
                }
            },
        ).model_dump(mode="json")
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.post(
                f"/v1/regional/executors/{command.command_id}/result",
                headers=headers,
                json=payload,
            )
            repeated = await client.post(
                f"/v1/regional/executors/{command.command_id}/result",
                headers=headers,
                json=payload,
            )
        assert first.status_code == 200
        assert repeated.status_code == 200

    asyncio.run(scenario())
    assert wake_calls == [True, True]
    assert len(notifier.notifications) == 1
    assert "系统动作：RESET_GPU" in notifier.notifications[0].body_text


def test_regional_api_sends_executor_workload_restart_notification() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    remote = RegionalRemoteWorkflowAdapter(
        context.store, owners={"gpu-fault-kubernetes-adapter"}
    )
    base = remote_context("workload-restart", "gpu-fault-kubernetes-adapter")
    step = copy_model(
        base.step,
        operation=WorkflowOperation.RESTART_WORKLOAD,
        workload_ids=["training/pytorchjob/training-a"],
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "training-a",
            "source_attempt_id": "training-a-a001",
            "source_gpu_count": 8,
            "restart_budget": 2,
        },
    )
    workflow = copy_model(base.workflow, official_steps=[step])
    # The preflight's reservation; dispatch only signs it.
    context.store.reserve_job_restart(
        "cluster-a", "training-a", 2, "workflow-workload-restart/0/RESTART_WORKLOAD"
    )
    remote.execute(
        replace(
            base,
            workflow=workflow,
            step=step,
            idempotency_key=("workflow-workload-restart/0/RESTART_WORKLOAD"),
        )
    )
    command = context.store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=1,
        lease_seconds=60,
        execution_owners={"gpu-fault-kubernetes-adapter"},
    )[0]

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                f"/v1/regional/executors/{command.command_id}/result",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json=RemoteCommandResult(
                    lease_token=command.lease_token,
                    status=RemoteCommandStatus.SUCCEEDED,
                    details={
                        "notification_id": "executor-local-notification",
                        "notification_context": {
                            "job_id": "training-a",
                            "source_attempt_id": "attempt-a",
                            "restart_attempt_id": "attempt-a-r-1",
                            "source_gpu_count": 24,
                            "target_gpu_count": 24,
                            "restart_count": 1,
                            "restart_budget": 2,
                        },
                    },
                ).model_dump(mode="json"),
            )
        assert response.status_code == 200

    asyncio.run(scenario())
    assert len(notifier.notifications) == 1
    assert "系统动作：RESTART_WORKLOAD" in (notifier.notifications[0].body_text)


def test_regional_context_loads_registry_and_remote_adapter(
    monkeypatch, tmp_path
) -> None:
    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_ENABLE_KUBERNETES_ADAPTER", "false")

    context = ApplicationContext.from_environment()

    assert context.regional_mode
    assert (
        context.store.get_regional_cluster("cluster-a").hyperpod_cluster_name
        == "hp-cluster-a"
    )
    assert any(
        isinstance(adapter, RegionalRemoteWorkflowAdapter)
        for adapter in context.workflow_executor.adapters
    )


def test_regional_context_rejects_local_kubernetes_adapter(
    monkeypatch, tmp_path
) -> None:
    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_ENABLE_KUBERNETES_ADAPTER", "true")

    with pytest.raises(
        RuntimeError, match="must not enable.*KubernetesWorkflowAdapter"
    ):
        ApplicationContext.from_environment()


def test_agent_registry_requires_artifact_and_config_pins(
    monkeypatch, tmp_path
) -> None:
    regional_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_ENABLE_AGENT_REGISTRY", "true")
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "s" * 32)
    monkeypatch.delenv("GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256", raising=False)
    monkeypatch.delenv("GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST", raising=False)

    with pytest.raises(RuntimeError, match="non-empty.*pins"):
        ApplicationContext.from_environment()


@pytest.mark.parametrize("durable", [False, True])
def test_hyperpod_submission_reservation_is_exclusive(tmp_path, durable: bool) -> None:
    store = SqliteStore(str(tmp_path / "submissions.db")) if durable else build_store()
    record = HyperPodSubmissionRecord(
        cluster_name="hp-cluster-a",
        idempotency_key="workflow-1/restart-node/0",
        action=HyperPodAction.REBOOT,
        requested_node_identifiers=["node-a"],
    )

    first, reserved_first = store.reserve_hyperpod_submission(record)
    second, reserved_second = store.reserve_hyperpod_submission(
        copy_model(record, state="SUBMITTED")
    )

    assert reserved_first
    # A second caller -- another replica, or the same executor after a
    # restart -- must lose the race and see the first record, not
    # overwrite it with its own INTENDED/SUBMITTED state.
    assert not reserved_second
    assert first.state == "INTENDED"
    assert second.state == "INTENDED"
    assert (
        store.get_hyperpod_submission("hp-cluster-a", "workflow-1/restart-node/0").state
        == "INTENDED"
    )


@pytest.mark.parametrize("durable", [False, True])
def test_hyperpod_submissions_are_cluster_scoped(tmp_path, durable: bool) -> None:
    store = SqliteStore(str(tmp_path / "submissions.db")) if durable else build_store()
    shared_key = "workflow-1/restart-node/0"
    for cluster_name in ("hp-cluster-a", "hp-cluster-b"):
        _, reserved = store.reserve_hyperpod_submission(
            HyperPodSubmissionRecord(
                cluster_name=cluster_name,
                idempotency_key=shared_key,
                action=HyperPodAction.REBOOT,
                requested_node_identifiers=[f"{cluster_name}/node"],
            )
        )
        # Idempotency keys are workflow-scoped, so two clusters can
        # legitimately produce the same one; neither may block the other.
        assert reserved

    assert store.get_hyperpod_submission(
        "hp-cluster-a", shared_key
    ).requested_node_identifiers == ["hp-cluster-a/node"]
    assert store.get_hyperpod_submission(
        "hp-cluster-b", shared_key
    ).requested_node_identifiers == ["hp-cluster-b/node"]
