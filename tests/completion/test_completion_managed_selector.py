"""The managed label is the definition of a job, not a filter option.

Every submitted training job carries ``gpu-fault.io/managed=true``, so the
completion watcher lists by that selector unconditionally. The former
observation-only mode (``GPU_FAULT_COMPLETION_OBSERVE_UNMANAGED``), which
dropped the selector and grouped unlabelled JobSet/Kubeflow Pods into
synthetic attempts, is gone: these tests pin its absence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gpu_fault
from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.completion_observation import completion_list_arguments
from gpu_fault.telemetry import ATTEMPT_COVERAGE_PATH
from tests.completion.test_completion_controller import (
    NOW,
    FakeCoreApi,
    FakeSink,
    controller,
    pod,
)

MANAGED_SELECTOR = "gpu-fault.io/managed=true"


class SelectorRecordingCoreApi(FakeCoreApi):
    """Records the list arguments so a test can pin the selector itself."""

    def __init__(self, pods=None) -> None:
        super().__init__(pods)
        self.list_arguments: list[tuple[str, dict]] = []

    def list_pod_for_all_namespaces(self, **kwargs):
        self.list_arguments.append(("list_pod_for_all_namespaces", dict(kwargs)))
        return super().list_pod_for_all_namespaces(**kwargs)

    def list_namespaced_pod(self, namespace, **kwargs):
        self.list_arguments.append(
            ("list_namespaced_pod", {"namespace": namespace, **kwargs})
        )
        return super().list_pod_for_all_namespaces(**kwargs)


def unlabelled_jobset_pod() -> dict:
    """A running JobSet Pod that was never submitted through the framework."""

    value = pod(
        0,
        extra_labels={"jobset.sigs.k8s.io/jobset-name": "set-a"},
        owner_references=[
            {"kind": "JobSet", "name": "set-a", "uid": "jobset-uid", "controller": True}
        ],
    )
    labels = value["metadata"]["labels"]
    labels.pop("gpu-fault.io/managed")
    labels.pop("gpu-fault.io/attempt-id")
    labels.pop("gpu-fault.io/job-id")
    return value


def test_pods_are_always_listed_by_the_managed_selector() -> None:
    """Cluster-wide and per namespace, the LIST always carries the selector.

    There is no switch that drops it: an unlabelled Pod is never listed, so
    it is neither an attempt nor a running Pod the coverage heartbeat
    counts. The watch stream reuses ``completion_list_arguments`` and is
    covered by the same call, not driven here.
    """

    cluster_wide = SelectorRecordingCoreApi([pod(0)])
    controller(cluster_wide, FakeSink()).run_once()
    namespaced = SelectorRecordingCoreApi([pod(0)])
    KubernetesCompletionController(
        namespaced,
        FakeSink(),
        cluster_id="hp-cluster",
        namespace="training",
        now=lambda: NOW,
    ).run_once()

    assert cluster_wide.list_arguments == [
        ("list_pod_for_all_namespaces", {"label_selector": MANAGED_SELECTOR})
    ], cluster_wide.list_arguments
    assert namespaced.list_arguments == [
        (
            "list_namespaced_pod",
            {"namespace": "training", "label_selector": MANAGED_SELECTOR},
        )
    ], namespaced.list_arguments
    assert completion_list_arguments(None) == {"label_selector": MANAGED_SELECTOR}
    assert completion_list_arguments("training") == {
        "label_selector": MANAGED_SELECTOR,
        "namespace": "training",
    }


def test_an_unlabelled_pod_is_neither_grouped_nor_observed() -> None:
    """An unlabelled JobSet Pod is outside the contract, whatever the API returns.

    The selector keeps such Pods off the list; should the API server hand
    one back anyway (defence in depth; this fake ignores the selector), it
    is not turned into a synthetic
    attempt: no observation is published and nothing is stopped. It still
    counts as a running Pod for the coverage heartbeat -- "every non-finished
    Pod counts" is fail-closed on purpose -- so the pass claims no IDLE either,
    where the same pass over an empty list does.
    """

    sink = FakeSink()
    subject = KubernetesCompletionController(
        FakeCoreApi([unlabelled_jobset_pod()]),
        sink,
        cluster_id="hp-cluster",
        now=lambda: NOW,
        publish_observations=True,
        workload_stopper=object(),
    )
    idle_sink = FakeSink()
    idle = KubernetesCompletionController(
        FakeCoreApi([]),
        idle_sink,
        cluster_id="hp-cluster",
        now=lambda: NOW,
        publish_observations=True,
    )

    assert subject.run_once() == []
    assert idle.run_once() == []

    assert sink.posts == [], sink.posts
    assert [path for path, _payload in idle_sink.posts] == [ATTEMPT_COVERAGE_PATH]


def test_the_constructor_no_longer_offers_an_observe_unmanaged_switch() -> None:
    with pytest.raises(TypeError, match="observe_unmanaged_workloads"):
        KubernetesCompletionController(
            FakeCoreApi([unlabelled_jobset_pod()]),
            FakeSink(),
            cluster_id="hp-cluster",
            observe_unmanaged_workloads=True,  # type: ignore[call-arg]
        )


def test_the_observe_unmanaged_switch_is_gone_from_the_runtime_inventory() -> None:
    """The generated inventory is what a deployment is validated against.

    ``GPU_FAULT_COMPLETION_OBSERVE_UNMANAGED`` and the two settings that only
    it gave a meaning to (the observation runtime profile and the retention
    cycles) are not read anywhere any more, so a site that still sets them is
    told by ``validate_gpu_fault_environment`` rather than silently ignored.
    """

    inventory = json.loads(
        (
            Path(gpu_fault.__file__).resolve().parent / "data" / "env-inventory.json"
        ).read_text(encoding="utf-8")
    )
    removed = {
        "GPU_FAULT_COMPLETION_OBSERVE_UNMANAGED",
        "GPU_FAULT_COMPLETION_OBSERVATION_RUNTIME_PROFILE",
        "GPU_FAULT_COMPLETION_OBSERVATION_RETENTION_CYCLES",
    }
    present = removed & set(inventory["variables"])
    assert not present, (
        f"the observation-only switch is dead surface under the managed-label "
        f"contract and must not stay in the runtime inventory: {sorted(present)}"
    )
