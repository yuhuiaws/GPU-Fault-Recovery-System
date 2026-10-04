"""Edge shapes of the release-state readers and the persisted-state writer.

``kubectl get -o json`` answers come back as a ``List``, as one object, or
empty; the readers must index what is there and skip what is not a named
document rather than fail on it. The deployment snapshot primes only the
planes the plan touches, the Running-replica profile probe skips Pods that
are not Running, the previous-snapshot verifier refuses a chunk whose
annotations are not a mapping, and the persisted-state writer leaves a
``previous`` it cannot externalize alone.
"""

from __future__ import annotations

import base64
import copy
import json
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.release_state_snapshot import (
    PREVIOUS_SNAPSHOT_CHUNK_KEY,
    encode_previous_snapshot,
)
from gpu_fault_release import regional_release_state as state
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from tests.regional._cov95_release_support import RecordingRunner, ResourceRelease


def test_documents_by_name_skips_unnamed_and_non_mapping_items() -> None:
    listed = {
        "kind": "List",
        "items": [
            "not-a-mapping",
            {"metadata": {}},
            {"metadata": {"name": "a", "uid": "1"}},
            {"metadata": {"name": "b"}},
        ],
    }

    assert sorted(state.documents_by_name(listed)) == ["a", "b"]
    assert state.documents_by_name({}) == {}
    single = {"kind": "ConfigMap", "metadata": {"name": "solo"}}
    assert state.documents_by_name(single) == {"solo": single}


class SnapshotRelease(ResourceRelease):
    """A release whose deployment snapshot priming is switched on."""

    _deployment_snapshot_enabled = True


def test_a_cpu_only_plan_primes_no_gpu_deployment_read() -> None:
    release = SnapshotRelease(cluster_ids=("gpu-a", "gpu-b"))
    release.documents[("cpu", "deployment", "")] = {"kind": "List", "items": []}
    for cluster_id in ("gpu-a", "gpu-b"):
        release.documents[(cluster_id, "deployment", "")] = {"items": []}

    state.prime_deployment_snapshot(
        release, ReleaseExecutionPlan(nodes=(ReleaseComponent.SCHEMA,))
    )
    assert release.reads == [("cpu", "deployment", "")]

    release.reads.clear()
    state.prime_deployment_snapshot(
        release, ReleaseExecutionPlan(nodes=(ReleaseComponent.EXECUTOR,))
    )
    assert sorted(release.reads) == [
        ("cpu", "deployment", ""),
        ("gpu-a", "deployment", ""),
        ("gpu-b", "deployment", ""),
    ]


def test_a_disabled_snapshot_reads_nothing() -> None:
    release = ResourceRelease()

    state.prime_deployment_snapshot(release, None)

    assert release.reads == []


def test_a_list_answer_without_items_is_cached_only_under_its_own_key() -> None:
    cache: dict[tuple[str, ...], Future[dict[str, Any]]] = {}
    runner = RecordingRunner(lambda _arguments, _kwargs: json.dumps({"items": "no"}))
    release = SimpleNamespace(
        runner=runner, _json_read_cache=cache, _json_read_cache_lock=threading.Lock()
    )
    arguments = ["kubectl", "-n", "gpu-fault-system", "get", "configmap"]

    value = state.get_json(release, arguments)

    assert value == {"items": "no"}
    assert list(cache) == [tuple(arguments)]
    assert state.get_json(release, arguments) == value
    assert len(runner.calls) == 1, "the second read is a cache hit"


def test_a_list_answer_files_each_named_item_for_per_name_reads() -> None:
    cache: dict[tuple[str, ...], Future[dict[str, Any]]] = {}
    listed = {
        "kind": "List",
        "items": [
            {"metadata": {"name": "one"}, "data": {"k": "1"}},
            "not-a-mapping",
            {"metadata": {"name": ""}},
        ],
    }
    runner = RecordingRunner(lambda _arguments, _kwargs: json.dumps(listed))
    release = SimpleNamespace(
        runner=runner, _json_read_cache=cache, _json_read_cache_lock=threading.Lock()
    )
    prefix = ["kubectl", "-n", "gpu-fault-system", "get", "configmap"]

    state.get_json(release, prefix)
    one = state.get_json(release, [*prefix, "one"])

    assert one == {"metadata": {"name": "one"}, "data": {"k": "1"}}
    assert len(runner.calls) == 1, "the per-name read was served from the list"


def test_a_selector_read_is_never_filed_under_a_per_name_key() -> None:
    cache: dict[tuple[str, ...], Future[dict[str, Any]]] = {}
    listed = {"kind": "List", "items": [{"metadata": {"name": "api-ha-1"}}]}
    runner = RecordingRunner(lambda _arguments, _kwargs: json.dumps(listed))
    release = SimpleNamespace(
        runner=runner, _json_read_cache=cache, _json_read_cache_lock=threading.Lock()
    )
    selected = ["kubectl", "-n", "gpu-fault-system", "get", "pod", "-l", "app=api"]

    state.get_json(release, selected)

    assert list(cache) == [tuple(selected)], (
        "a selector's answer is only the answer to that selector"
    )


def test_a_pod_that_is_not_running_does_not_answer_the_profile_version() -> None:
    release = ResourceRelease()
    release.documents[("cpu", "pods", "")] = {
        "items": [
            {"metadata": {"name": "pending-pod"}, "status": {"phase": "Pending"}},
            {"metadata": {}, "status": {"phase": "Running"}},
        ]
    }

    assert state.effective_cpu_runtime_profile_version(release) is None
    assert release.runner.calls == [], "no exec reaches a Pod that is not Running"


class StoredChunkRunner(RecordingRunner):
    """Previous-snapshot chunks already stored, answered to ``probe_output``."""

    def __init__(self, objects: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.objects = objects

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        name = arguments[arguments.index("get") + 2]
        stored = self.objects.get(name)
        return 0, json.dumps(stored) if stored is not None else "", ""


def _stored_chunks(previous: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reference, chunks = encode_previous_snapshot(previous)
    return {
        name: {
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": "gpu-fault-system",
                "uid": f"uid-{name}",
                "annotations": {
                    state.PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION: reference["sha256"]
                },
            },
            "immutable": True,
            "binaryData": {
                PREVIOUS_SNAPSHOT_CHUNK_KEY: base64.b64encode(data).decode()
            },
        }
        for name, data in chunks
    }


def test_a_stored_chunk_whose_annotations_are_not_a_mapping_is_refused() -> None:
    previous = {"release_id": "older", "metadata": {"pin": "p"}}
    objects = _stored_chunks(previous)
    for item in objects.values():
        item["metadata"]["annotations"] = ["not", "a", "mapping"]
    release = ResourceRelease()
    release.runner = StoredChunkRunner(objects)

    with pytest.raises(ReleaseError, match="annotations are invalid"):
        state.ensure_previous_snapshot(release, previous)
    assert release.runner.calls == [], "nothing is rewritten over a suspect chunk"


def test_intact_stored_chunks_are_verified_not_rewritten() -> None:
    previous = {"release_id": "older", "metadata": {"pin": "p"}}
    release = ResourceRelease()
    release.runner = StoredChunkRunner(_stored_chunks(previous))

    reference = state.ensure_previous_snapshot(release, previous)

    assert reference["chunks"], "the reference names its chunks"
    assert release.runner.calls == []


def test_persisted_state_leaves_a_previous_it_cannot_externalize_alone() -> None:
    release = ResourceRelease()
    release.state = {"previous": "not-a-mapping", "previous_snapshot_sha256": "keep"}

    persisted, text = state.render_persisted_state(release)

    assert persisted == {
        "previous": "not-a-mapping",
        "previous_snapshot_sha256": "keep",
    }
    assert json.loads(text) == persisted
    assert copy.deepcopy(release.state) == persisted, "the live state is untouched"
