from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from gpu_fault.admin.cluster_batch_join import effective_deploy_concurrency
from gpu_fault.admin.config import AdminConfigError, admin_config_lock
from gpu_fault.admin.membership_lock import membership_operation_lock
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    site_operation_lock,
)


def test_membership_operation_lock_serializes_same_site(tmp_path) -> None:
    site = SimpleNamespace(source=tmp_path / "site.yaml")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with membership_operation_lock(site):
            first_entered.set()
            assert release_first.wait(timeout=2), "first lock holder was not released"

    def second() -> None:
        assert first_entered.wait(timeout=2), "first lock holder did not enter"
        with membership_operation_lock(site):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2), "first membership lock was not acquired"
    time.sleep(0.05)
    assert not second_entered.is_set(), "second membership operation bypassed the lock"
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert second_entered.is_set(), "second membership operation never resumed"


def test_membership_and_config_operations_share_one_site_lock(tmp_path) -> None:
    site = SimpleNamespace(source=tmp_path / "site.yaml")

    with membership_operation_lock(site):
        with pytest.raises(AdminConfigError, match="administrator mutation"):
            with admin_config_lock(tmp_path):
                pass


def test_site_operation_lock_accepts_verified_inherited_descriptor(
    tmp_path, monkeypatch
) -> None:
    with site_operation_lock(tmp_path, wait=False) as descriptor:
        monkeypatch.setenv(SITE_OPERATION_LOCK_FD_ENV, str(descriptor))
        with site_operation_lock(tmp_path, wait=False) as inherited:
            assert inherited == descriptor, (
                "nested site operation did not reuse the verified inherited lock"
            )


def test_auto_upgrade_max_unavailable_counts_against_the_node_budget() -> None:
    """`0` is auto, and auto is not one node per cluster.

    The wave engine reads 0 as "take the size cap the node count allows", so a
    thousand-node cluster can have 32 nodes down at once. Reading the same 0 here
    as a per-cluster limit of 1 made the 64-node global budget divide by 1 and
    admit four such clusters in parallel -- a budget of 64 spending 128.
    """

    context = SimpleNamespace(
        site=SimpleNamespace(
            release_config={"release": {"upgrade_max_unavailable": 0}}
        ),
        concurrency=4,
    )
    attempts = [
        SimpleNamespace(
            execution=SimpleNamespace(
                local={"nodes": [f"node-{index}" for index in range(1000)]}
            )
        )
        for _ in range(4)
    ]

    assert effective_deploy_concurrency(context, attempts) == 2


def test_large_batch_join_respects_global_node_budget() -> None:
    context = SimpleNamespace(
        site=SimpleNamespace(
            release_config={"release": {"upgrade_max_unavailable": 32}}
        ),
        concurrency=4,
    )
    attempts = [
        SimpleNamespace(
            execution=SimpleNamespace(
                local={"nodes": [f"node-{index}" for index in range(1000)]}
            )
        )
        for _ in range(4)
    ]

    assert effective_deploy_concurrency(context, attempts) == 2
