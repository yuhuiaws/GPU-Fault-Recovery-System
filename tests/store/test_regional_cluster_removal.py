from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.fleet import AgentRecord
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.store import InMemoryStore, SqliteStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_deleting_a_regional_cluster_removes_its_stale_agents(
    backend: str, tmp_path
) -> None:
    store = (
        InMemoryStore()
        if backend == "memory"
        else SqliteStore(str(tmp_path / "store.sqlite"))
    )
    now = datetime.now(timezone.utc)
    store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id="gpu-a",
            region="us-east-1",
            hyperpod_cluster_name="hp-a",
            eks_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
            token_sha256="a" * 64,
            agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        )
    )
    store.save_agent(
        AgentRecord(
            cluster_id="gpu-a",
            node_id="node-a",
            endpoint="http://node-a:9099",
            agent_protocol_version=3,
            node_action_key_version=2,
            agent_version="0.10.0",
            artifact_sha256="b" * 64,
            policy_version="catalog-a",
            runtime_profile_version="hyperpod-v1",
            config_digest="c" * 64,
            allowed_operations=[],
            first_seen_at=now,
            last_seen_at=now,
            lease_expires_at=now + timedelta(minutes=5),
        )
    )

    store.delete_regional_cluster("gpu-a")

    assert store.list_regional_clusters() == []
    assert store.list_agents("gpu-a") == []
