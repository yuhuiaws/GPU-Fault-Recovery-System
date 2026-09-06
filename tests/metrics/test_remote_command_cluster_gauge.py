"""F-D12: the remote-command backlog is exported per cluster as well as in total.

The fleet-wide ``gpu_fault_remote_command_total{status}`` family stays as it
is (alerts read it); the new family carries the ``cluster_id`` label so a
single stuck cluster is visible instead of being averaged into the fleet.
"""

from __future__ import annotations

from gpu_fault.app.builtin_metric_contributors import remote_command_metric_lines
from tests._builders import build_context
from tests.regional._regional_support import (
    TOKEN_A,
    TOKEN_B,
    enqueue_remote_command,
    registration,
)


def test_remote_command_backlog_is_exported_per_cluster_and_status() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    enqueue_remote_command(context.store, "remote-" + "a" * 24, cluster_id="cluster-a")
    enqueue_remote_command(context.store, "remote-" + "b" * 24, cluster_id="cluster-b")
    enqueue_remote_command(context.store, "remote-" + "c" * 24, cluster_id="cluster-b")

    runtime = type("Runtime", (), {"context": context})()
    text = "\n".join(remote_command_metric_lines(runtime))

    assert 'gpu_fault_remote_command_total{status="PENDING"} 3' in text, text
    assert (
        'gpu_fault_remote_command_by_cluster_total{cluster_id="cluster-a",status="PENDING"} 1'
        in text
    ), text
    assert (
        'gpu_fault_remote_command_by_cluster_total{cluster_id="cluster-b",status="PENDING"} 2'
        in text
    ), text
