"""One rule for "departed": the administrator's reconcile and the cleanup share it.

``gpu_fault.departed_nodes`` is the definition both callers judge with -- the
CLI on what it reads from the deploy host, the control plane's cleanup on what
it reads itself -- so the admin modules must hand out the very same objects,
and the rule itself must refuse everything short of proof.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault import departed_nodes as rule
from gpu_fault.admin import workflow_reconcile_departed_agents as departed_agents
from gpu_fault.admin import workflow_reconcile_departed_nodes as departed_admin
from gpu_fault.admin import workflow_reconcile_never_dispatched as never_dispatched
from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentRecord,
    AgentTransitionRequest,
    FleetRegistry,
)
from tests._builders import build_store

HP_CLUSTER = "gpu-hyperpod"
HP_INSTANCE = "i-0a000001"
HP_NODE = f"hyperpod-{HP_INSTANCE}"


def now() -> datetime:
    return datetime.now(timezone.utc)


def ago(**delta: float) -> datetime:
    return now() - timedelta(**delta)


def agent(
    cluster_id: str, node_id: str, *, instance_id: str, last_seen: datetime
) -> AgentRecord:
    return AgentRecord(
        cluster_id=cluster_id,
        node_id=node_id,
        endpoint="https://10.0.38.180:9099",
        agent_version="2026.10.14",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="regional-v1",
        config_digest="b" * 64,
        allowed_operations=[],
        node_instance_id=instance_id,
        agent_incarnation_id=f"incarnation-{node_id}",
        first_seen_at=last_seen - timedelta(days=1),
        last_seen_at=last_seen,
        lease_expires_at=last_seen + timedelta(seconds=120),
        generation=4,
    )


def test_the_cli_and_the_cleanup_judge_with_the_same_objects() -> None:
    assert departed_agents.DEPARTED_AGENT_GUARD_AGE is rule.DEPARTED_AGENT_GUARD_AGE
    assert departed_agents.GUARD_REASON is rule.GUARD_REASON
    assert departed_agents.RETIREABLE_LIFECYCLES is rule.RETIREABLE_LIFECYCLES
    assert departed_admin.HYPERPOD_NODE_NAME is rule.HYPERPOD_NODE_NAME
    assert departed_admin.INSTANCE_ID is rule.INSTANCE_ID
    assert departed_admin.cluster_provider_of is rule.cluster_provider_of
    assert never_dispatched.node_evidence is rule.node_evidence
    assert (
        never_dispatched.KUBERNETES_PRESENT_BLOCKER is rule.KUBERNETES_PRESENT_BLOCKER
    )
    assert rule.DEPARTED_AGENT_GUARD_AGE == timedelta(minutes=10)


def test_the_provider_verdicts_never_read_absent_by_default() -> None:
    assert rule.hyperpod_instance_id(HP_NODE) == HP_INSTANCE
    assert rule.hyperpod_instance_id("ip-10-0-1-2.ec2.internal") is None
    assert rule.recorded_instance_id(" i-0e000001 ") == "i-0e000001"
    assert rule.recorded_instance_id("0b9b4c2e-node-uid") is None
    assert rule.recorded_instance_id(None) is None
    assert rule.hyperpod_listing_blocker(HP_INSTANCE, {HP_INSTANCE}) is not None
    assert rule.hyperpod_listing_blocker(HP_INSTANCE, set()) is None
    for state in ("terminated", "not-found"):
        assert rule.ec2_state_blocker("i-0a", state) is None
    for state in ("running", "stopped", "shutting-down", "pending"):
        assert rule.ec2_state_blocker("i-0a", state) == (
            f"node is missing from Kubernetes but EC2 instance i-0a is {state}"
        )


@pytest.mark.parametrize(
    ("lifecycle", "seen", "lease", "expected"),
    [
        ("ACTIVE", {"minutes": 20}, {"minutes": 18}, []),
        ("DRAINING", {"minutes": 20}, None, []),
        ("ACTIVE", {"minutes": 4}, {"minutes": 2}, [rule.GUARD_REASON]),
        ("ACTIVE", {"minutes": 20}, {"minutes": -5}, [rule.LEASE_REASON]),
        ("REVOKED", {"minutes": 20}, None, ["agent lifecycle state is REVOKED"]),
    ],
)
def test_an_agent_is_retired_only_after_the_guard_and_its_lease(
    lifecycle: str,
    seen: dict[str, float],
    lease: dict[str, float] | None,
    expected: list[str],
) -> None:
    evaluated = now()
    reasons = rule.agent_retirement_reasons(
        lifecycle,
        (evaluated - timedelta(**seen)).isoformat().replace("+00:00", "Z"),
        (evaluated - timedelta(**lease)) if lease is not None else None,
        evaluated_at=evaluated,
    )
    assert reasons == expected


def test_a_heartbeat_without_a_timezone_is_never_judged_old() -> None:
    reasons = rule.agent_retirement_reasons(
        "ACTIVE", "2026-10-14T01:00:00", None, evaluated_at=now()
    )
    assert reasons == ["agent last_seen_at is missing or not timezone-aware"]


def _registry() -> tuple[Any, FleetRegistry]:
    store = build_store()
    store.save_agent(
        agent(HP_CLUSTER, HP_NODE, instance_id="i-0a000001", last_seen=ago(minutes=30))
    )
    return store, FleetRegistry(store, "s" * 32)


def _evidence(**overrides: Any) -> dict[str, Any]:
    return {
        "node_id": HP_NODE,
        "instance_id": HP_INSTANCE,
        "absent_from_kubernetes": True,
        "absent_from_provider": True,
        **overrides,
    }


def _retire(store: Any, registry: FleetRegistry, **overrides: Any) -> Any:
    record = store.get_agent(HP_CLUSTER, HP_NODE)
    values: dict[str, Any] = {
        "cluster_id": HP_CLUSTER,
        "node_id": HP_NODE,
        "expected_generation": record.generation,
        "expected_lifecycle": record.lifecycle_state.value,
        "expected_last_seen_at": record.last_seen_at,
        "evidence": _evidence(),
        "transition_id": "departed-node-cleanup/ref",
        "reason": "operator reconciliation ref: retired Agent",
        "now": now(),
        **overrides,
    }
    return rule.retire_departed_agent(registry, store, **values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_generation": 3}, "generation changed"),
        ({"expected_lifecycle": "DRAINING"}, "lifecycle state changed"),
        ({"expected_last_seen_at": ago(minutes=31)}, "heartbeat changed"),
        ({"evidence": _evidence(absent_from_provider=False)}, "does not prove"),
        ({"evidence": _evidence(instance_id="i-0e000009")}, "instance changed"),
        ({"now": ago(minutes=25)}, "younger than the 10-minute"),
    ],
)
def test_the_retirement_refuses_anything_that_moved_or_is_not_proof(
    overrides: dict[str, Any], message: str
) -> None:
    store, registry = _registry()

    with pytest.raises(ValueError, match=message):
        _retire(store, registry, **overrides)

    assert store.get_agent(HP_CLUSTER, HP_NODE).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )


def test_a_drain_another_transition_started_is_finished_under_it() -> None:
    store, registry = _registry()
    registry.drain_agent(
        HP_CLUSTER,
        HP_NODE,
        AgentTransitionRequest(
            expected_generation=4, transition_id="operator/drain", reason="earlier"
        ),
    )

    revoked = _retire(store, registry)

    assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
    assert revoked.transition_id == "operator/drain"
    assert revoked.generation == 5, "no second drain"
