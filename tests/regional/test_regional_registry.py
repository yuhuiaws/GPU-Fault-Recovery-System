from __future__ import annotations

import pytest

from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.regional_registry import sync_regional_cluster_registry
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import build_store


def _registration(cluster_id: str) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        hyperpod_cluster_name=f"hyperpod-{cluster_id}",
        eks_cluster_arn=(f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}"),
        token_sha256="a" * 64,
        allowed_namespaces=["training"],
    )


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
