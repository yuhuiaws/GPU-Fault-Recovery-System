from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    gpu_node_items,
    node_inventory_scope,
)
from gpu_fault_release.regional_release_mutation_preflight import (
    preflight_upgrade_mutations,
)


def test_candidate_preflight_is_bounded_across_clusters() -> None:
    lock = threading.Lock()
    four_started = threading.Event()
    active = 0
    maximum = 0
    finished: list[str] = []

    def preflight(target, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_started.set()
        assert four_started.wait(timeout=5), "cluster preflights did not overlap"
        time.sleep(0.01)
        with lock:
            finished.append(target.cluster_id)
            active -= 1

    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=tuple(SimpleNamespace(cluster_id=f"gpu-{i}") for i in range(7)),
            agent_config_digest="a" * 64,
            namespace="gpu-fault-system",
        ),
        _get_json=lambda _arguments: {"items": []},
        _gpu=lambda _target, *arguments: list(arguments),
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="b" * 64,
        _preflight_node_runtime=preflight,
    )
    preflight_upgrade_mutations(
        release, ReleaseExecutionPlan(nodes=(ReleaseComponent.AGENT,))
    )

    assert maximum == 4
    assert sorted(finished) == [f"gpu-{i}" for i in range(7)]


def test_node_inventory_scopes_do_not_leak_between_parallel_derivations() -> None:
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    reads = 0

    def get_json(_arguments):
        nonlocal reads
        with lock:
            reads += 1
            return {"items": [{"generation": reads}]}

    target = SimpleNamespace(cluster_id="gpu", hyperpod_cluster_name="hp")
    release = SimpleNamespace(
        _gpu=lambda _target, *args: list(args), _get_json=get_json
    )

    def derive():
        with node_inventory_scope(release):
            first = gpu_node_items(release, target)
            barrier.wait()
            second = gpu_node_items(release, target)
            barrier.wait()
            assert first == second, "another thread replaced the pinned inventory"
            return first

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(derive)
        second = executor.submit(derive)
        assert first.result() != second.result(), "threads shared one pinned snapshot"
    assert reads == 2
    gpu_node_items(release, target)
    assert reads == 3, "inventory escaped its derivation scope"
