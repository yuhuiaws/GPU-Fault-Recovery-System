"""``stale_health_signal_states``: one node's health-signal rows that no claim
wrote since a bound, found by exact key prefix (batch 160).

The control plane uses it to age out device-scoped signals whose device
stopped reporting (``host_device_signals.py``). The prefix match must be
exact -- the production database's default collation is not byte order, so a
key-range walk cannot stand in for it -- and a mount-point device carrying
``/`` in the key must still be found under its node.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import SqliteStore
from tests._builders import build_store

NOW = datetime(2030, 3, 1, 12, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
PREFIX = f"{CLUSTER}/node-a/"
DM0 = f"{CLUSTER}/node-a/disk_io_await_ms/dm-0"
NVME = f"{CLUSTER}/node-a/disk_io_util_percent/nvme0n1"
MOUNT = (
    f"{CLUSTER}/node-a/local_filesystem_used_percent//fsx/HIGH_LOCAL_FILESYSTEM_USAGE"
)
NODE_RULE = f"{CLUSTER}/node-a/cpu_usage_percent/node"
# Another node whose name extends this one, and another cluster with the
# same node name: neither belongs to ``node-a`` of ``cluster-a``.
LONGER_NODE = f"{CLUSTER}/node-ab/disk_io_await_ms/dm-0"
OTHER_CLUSTER = "cluster-b/node-a/disk_io_await_ms/dm-0"


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    instance = SqliteStore(str(tmp_path / "stale-health-signals.db"))
    try:
        yield instance
    finally:
        instance.close()


def claim(store, key: str, at: datetime, *, active: bool = True) -> None:
    store.claim_health_signal_transitions([(key, active, at, 0.0)], received_at=at)


def keys(states) -> list[str]:
    return [state.signal_key for state in states]


def test_only_rows_of_the_exact_prefix_older_than_the_bound_are_returned(store):
    old = NOW - timedelta(minutes=20)
    for key in (DM0, NVME, MOUNT, NODE_RULE, LONGER_NODE, OTHER_CLUSTER):
        claim(store, key, old)
    # A row the node keeps writing is not stale, whatever its key.
    claim(store, NVME, NOW - timedelta(seconds=30))

    stale = store.stale_health_signal_states(
        PREFIX, stale_before=NOW - timedelta(minutes=15)
    )

    assert keys(stale) == sorted([DM0, MOUNT, NODE_RULE]), (
        "exact prefix, mount-point device included, fresh row and other "
        "nodes/clusters excluded"
    )
    assert all(state.active for state in stale)


def test_the_bound_is_the_control_plane_clock_of_the_last_claim(store):
    # Node time says long ago, the control plane received it just now: the
    # receive clock (``clock_at``) is the one the row was last written on.
    received = NOW - timedelta(seconds=10)
    store.claim_health_signal_transitions(
        [(DM0, True, NOW - timedelta(hours=2), 0.0)], received_at=received
    )
    # Judged on node time the row would be two hours stale at either bound.
    assert store.stale_health_signal_states(PREFIX, stale_before=received) == []
    assert keys(
        store.stale_health_signal_states(
            PREFIX, stale_before=received + timedelta(seconds=1)
        )
    ) == [DM0]


def test_inactive_rows_are_returned_too_and_the_limit_bounds_the_read(store):
    old = NOW - timedelta(hours=1)
    claim(store, DM0, old)
    # A clear after an activation leaves an inactive row behind (a clear with
    # no row to clear writes nothing).
    claim(store, NVME, old, active=True)
    claim(store, NVME, old + timedelta(seconds=1), active=False)

    stale = store.stale_health_signal_states(PREFIX, stale_before=NOW)
    assert keys(stale) == [DM0, NVME]
    assert [state.active for state in stale] == [True, False], (
        "the caller decides what an inactive stale row means"
    )
    assert len(store.stale_health_signal_states(PREFIX, stale_before=NOW, limit=1)) == 1


def test_an_empty_prefix_or_an_unknown_node_reads_nothing(store):
    claim(store, DM0, NOW - timedelta(hours=1))
    assert store.stale_health_signal_states("cluster-a/node-z/", stale_before=NOW) == []
    with pytest.raises(ValueError, match="prefix"):
        store.stale_health_signal_states("", stale_before=NOW)
    with pytest.raises(ValueError, match="limit"):
        store.stale_health_signal_states(PREFIX, stale_before=NOW, limit=0)
