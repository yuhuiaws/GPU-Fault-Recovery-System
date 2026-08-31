from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.regional import RegionalClusterRegistration, RegionalRegistryRevision
from gpu_fault.regional_registry import sync_regional_cluster_registry
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import build_store

NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _registration(cluster_id: str) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        hyperpod_cluster_name=f"hyperpod-{cluster_id}",
        eks_cluster_arn=(f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}"),
        token_sha256="a" * 64,
        allowed_namespaces=["training"],
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
    )


def test_synthetic_registration_expires_and_stops_authenticating() -> None:
    token = "t" * 32
    now = datetime.now(timezone.utc)
    registration = RegionalClusterRegistration(
        cluster_id="perf-cap-000",
        region="us-west-2",
        hyperpod_cluster_name="perf-cap-000",
        eks_cluster_arn=("arn:aws:eks:us-west-2:000000000000:cluster/perf-cap-000"),
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        synthetic=True,
        synthetic_run_id="run-a",
        synthetic_expires_at=now + timedelta(days=1),
        agent_endpoint_allowed_cidrs=["127.0.0.1/32"],
    )

    assert registration.is_active(now), "synthetic registration was inactive before TTL"
    assert not registration.is_active(now + timedelta(days=2)), (
        "synthetic registration remained active after TTL"
    )
    assert registration.authenticates(token), (
        "active synthetic registration rejected its token"
    )


def test_registry_sync_prunes_expired_synthetic_registration() -> None:
    store = build_store()
    store.save_regional_cluster(_registration("cluster-stale"))

    configured = sync_regional_cluster_registry(
        store,
        [
            {
                "cluster_id": "perf-cap-000",
                "region": "us-west-2",
                "hyperpod_cluster_name": "perf-cap-000",
                "eks_cluster_arn": (
                    "arn:aws:eks:us-west-2:000000000000:cluster/perf-cap-000"
                ),
                "token": "t" * 32,
                "synthetic": True,
                "synthetic_run_id": "run-a",
                "synthetic_expires_at": (NOW - timedelta(seconds=1)).isoformat(),
                "agent_endpoint_allowed_cidrs": ["127.0.0.1/32"],
            }
        ],
        now=NOW,
    )

    assert configured == []
    assert store.list_regional_clusters() == []


def test_durable_revision_supersedes_stale_bootstrap_secret() -> None:
    store = build_store()
    active = _registration("cluster-active")
    revision = RegionalRegistryRevision.build(
        generation=1,
        registrations=[active],
        previous_generation=None,
        required_member_ids=[],
        reason="runtime authority",
        created_at=NOW,
    )
    store.publish_regional_registry_revision(revision, expected_generation=0)

    configured = sync_regional_cluster_registry(
        store,
        [
            {
                "cluster_id": "cluster-stale",
                "region": "us-west-2",
                "hyperpod_cluster_name": "hyperpod-stale",
                "eks_cluster_arn": (
                    "arn:aws:eks:us-west-2:123456789012:cluster/cluster-stale"
                ),
                "token": "s" * 32,
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
        now=NOW,
    )

    assert configured == [active]
    assert store.list_regional_clusters() == [active]


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_registry_sync_prunes_stale_durable_identity(kind: str, tmp_path) -> None:
    store = (
        build_store()
        if kind == "memory"
        else SqliteStore(str(tmp_path / "registry.db"))
    )
    store.save_regional_cluster(_registration("cluster-stale"))

    configured = sync_regional_cluster_registry(
        store,
        [
            {
                "cluster_id": "cluster-a",
                "region": "us-west-2",
                "hyperpod_cluster_name": "hyperpod-a",
                "eks_cluster_arn": (
                    "arn:aws:eks:us-west-2:123456789012:cluster/cluster-a"
                ),
                "token": "a" * 32,
                "allowed_namespaces": ["training"],
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
    )

    assert [item.cluster_id for item in configured] == ["cluster-a"]
    assert [item.cluster_id for item in store.list_regional_clusters()] == ["cluster-a"]
    with pytest.raises(NotFoundError):
        store.get_regional_cluster("cluster-stale")
    store.delete_regional_cluster("cluster-stale")
    if hasattr(store, "close"):
        store.close()


def test_registry_sync_migrates_legacy_sqlite_record_before_model_decode(
    tmp_path,
) -> None:
    path = tmp_path / "legacy-registry.db"
    SqliteStore(str(path)).close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO objects(kind, key, payload)
            VALUES ('regional_cluster', ?, ?)
            """,
            (
                "cluster-a",
                json.dumps(
                    {
                        "cluster_id": "cluster-a",
                        "region": "us-west-2",
                        "hyperpod_cluster_name": "hyperpod-a",
                        "eks_cluster_arn": (
                            "arn:aws:eks:us-west-2:123456789012:cluster/cluster-a"
                        ),
                        "token_sha256": "a" * 64,
                        "enabled": True,
                        "allowed_namespaces": ["training"],
                    }
                ),
            ),
        )

    store = SqliteStore(str(path))
    configured = sync_regional_cluster_registry(
        store,
        [
            {
                "cluster_id": "cluster-a",
                "region": "us-west-2",
                "hyperpod_cluster_name": "hyperpod-a",
                "eks_cluster_arn": (
                    "arn:aws:eks:us-west-2:123456789012:cluster/cluster-a"
                ),
                "token": "a" * 32,
                "allowed_namespaces": ["training"],
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
    )

    assert configured[0].agent_endpoint_allowed_cidrs == ["10.0.0.0/16"]
    assert store.get_regional_cluster("cluster-a").agent_endpoint_allowed_cidrs == [
        "10.0.0.0/16"
    ]
    store.close()
