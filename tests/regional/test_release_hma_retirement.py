from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_mutation_preflight import (
    preflight_upgrade_mutations,
)


@pytest.mark.parametrize(
    "listing",
    [
        {},
        {"items": None},
        {"items": [None]},
        {"items": [{"metadata": {}}]},
        {"items": [{"metadata": {"name": "gpu-fault-hma-watcher"}}]},
        {
            "items": [
                {
                    "metadata": {"name": "gpu-fault-hma-cloudwatch-consumer"},
                    "spec": {"replicas": 0},
                }
            ]
        },
    ],
)
def test_retired_or_unknown_collectors_block_candidate_preflight(listing: dict) -> None:
    reads = []
    rendered = []
    target = SimpleNamespace(cluster_id="gpu-a")

    def get_json(arguments):
        reads.append(arguments)
        return listing

    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(target,), namespace="gpu-fault-system"),
        executor_wheel_cm="candidate-wheel",
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=get_json,
        _preflight_gpu_deployments=lambda *args, **kwargs: rendered.append(args),
    )
    with pytest.raises(ReleaseError, match="retired"):
        preflight_upgrade_mutations(
            release, ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,))
        )
    assert rendered == [], "candidate must not proceed past the retirement barrier"
    assert reads == [["-n", "gpu-fault-system", "get", "deployments"]], (
        "retirement must use one read-only LIST in the solution namespace"
    )


def test_retired_collector_discovery_failure_propagates_without_mutation() -> None:
    rendered = []

    def denied(_arguments):
        raise ReleaseError("retired collector discovery forbidden")

    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
            namespace="gpu-fault-system",
        ),
        executor_wheel_cm="candidate-wheel",
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=denied,
        _preflight_gpu_deployments=lambda *args, **kwargs: rendered.append(args),
    )
    with pytest.raises(ReleaseError, match="forbidden"):
        preflight_upgrade_mutations(
            release, ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,))
        )
    assert rendered == [], "discovery failure must not be treated as absence"


@pytest.mark.parametrize(
    "items", [[], [{"metadata": {"name": "gpu-fault-completion-watcher"}}]]
)
def test_absent_retired_collectors_allow_candidate_preflight(items: list) -> None:
    rendered = []
    target = SimpleNamespace(cluster_id="gpu-a")
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(target,), namespace="gpu-fault-system"),
        executor_wheel_cm="candidate-wheel",
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=lambda _arguments: {"items": items},
        _preflight_gpu_deployments=lambda *args, **kwargs: rendered.append(args),
    )
    preflight_upgrade_mutations(
        release, ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,))
    )
    assert rendered == [(target, "candidate-wheel")]
