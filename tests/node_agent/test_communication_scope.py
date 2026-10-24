import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app.context import ApplicationContext
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.communication_runtime import CommunicationRuntime
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_store,
    container_observation,
)
from tests.fleet._support import heartbeat, registry, signed


def test_authenticated_heartbeat_returns_only_current_task_scope_and_legacy_stays_compatible():
    now = datetime.now(timezone.utc)
    store = build_store()
    store.save_attempt_observation(
        attempt_observation(
            "job",
            "attempt",
            now,
            containers=[
                container_observation(
                    "pod", "trainer", 0, "node-a", cgroup_path="/job"
                ),
                container_observation(
                    "other-pod", "trainer", 1, "node-b", cgroup_path="/other"
                ),
            ],
        )
    )
    ctx = ApplicationContext(
        store=store,
        execution_token="e" * 32,
        fleet_registry=registry(store, now=lambda: now),
    )

    async def scenario():
        async with asgi_client(ctx) as client:
            value = heartbeat("node-a", observed_at=now)
            legacy = await client.post(
                "/v1/fleet/agents/heartbeat", json=signed(value).model_dump(mode="json")
            )
            assert (
                legacy.status_code == 200 and "communication_plan" not in legacy.json()
            ), legacy.text
            value = value.model_copy(
                update={
                    "communication_monitor_version": "communication-monitor/v1",
                    "allowed_operations": [Op.COLLECT_HUNG_SUMMARY],
                    "observed_at": now + timedelta(seconds=1),
                }
            )
            result = await client.post(
                "/v1/fleet/agents/heartbeat", json=signed(value).model_dump(mode="json")
            )
            assert result.status_code == 200, result.text
            plan = result.json()["communication_plan"]
            assert plan["boot_id"] == value.boot_id and plan["node_id"] == "node-a", (
                plan
            )
            assert plan["targets"][0]["cgroup_paths"] == ["/job"], plan
            assert (
                "communication_plan"
                not in store.get_agent("cluster-a", "node-a").model_dump()
            ), "short-lived scopes must not be persisted as fleet identity"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changed", ["boot_id", "generation", "node_id", "lease", "expired"]
)
def test_observer_rejects_foreign_or_overlong_scope(tmp_path, changed):
    root = tmp_path / "proc"
    boot = root / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("boot")
    agent = SimpleNamespace(
        node_ids={"node-a"},
        agent_generation=1,
        proc_root=root,
        infiniband_root=tmp_path / "ib",
        diagnostic_output_dir=tmp_path / "evidence",
        python_stack_tool="",
        diagnostic_s3_uri=None,
        diagnostic_retention_seconds=600,
        diagnostic_max_archives=10,
    )
    runtime = CommunicationRuntime(agent)
    now = datetime.now(timezone.utc)
    plan = dict(
        cluster_id="cluster-a",
        node_id="node-a",
        boot_id="boot",
        generation=1,
        issued_at=now,
        expires_at=now + timedelta(seconds=90),
        targets=[],
        expected_target_count=0,
    )
    if changed == "lease":
        plan["expires_at"] = now + timedelta(seconds=91)
    elif changed == "expired":
        plan["issued_at"] = now - timedelta(seconds=120)
        plan["expires_at"] = now - timedelta(seconds=30)
    else:
        plan[changed] = 2 if changed == "generation" else "other"
    with pytest.raises(ValueError):
        runtime.accept_plan(plan)
    assert runtime.health()["lease_valid"] is False, runtime.health()
