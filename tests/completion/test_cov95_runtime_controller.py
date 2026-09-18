from __future__ import annotations

import json
from datetime import timedelta

import pytest

from gpu_fault.completion_pod_parsing import CompletionControllerError
from tests.completion._cov95_runtime_support import make_controller
from tests.completion._support import Clock, FakeCoreApi, FakeSink, pod


@pytest.mark.parametrize(
    "options",
    [
        {"cluster_id": ""},
        {"poll_interval_seconds": 0},
        {"watch_timeout_seconds": 0},
        {"emergency_fallback_seconds": 0},
        {"reconcile_debounce_seconds": -1},
        {"coverage_heartbeat_interval_seconds": -1},
        {"terminal_retention_seconds": 59},
        {"watcher_max_attempts": 99},
        {"attempt_missing_grace_seconds": 0},
    ],
)
def test_controller_rejects_invalid_bounds_before_reading_workloads(options):
    core, sink = FakeCoreApi(), FakeSink()
    with pytest.raises(CompletionControllerError):
        make_controller(core, sink, **options)
    assert core.list_calls == 0
    assert sink.posts == []


def test_unreadable_persisted_state_is_an_actionable_startup_error() -> None:
    class CorruptSink(FakeSink):
        def load_attempt_observations(self):
            raise ValueError("invalid persisted JSON")

    with pytest.raises(CompletionControllerError, match="cannot restore persisted"):
        make_controller(sink=CorruptSink())


@pytest.mark.parametrize(
    "key,value,diagnostic",
    [
        ("gpu-fault.io/expected-critical-ranks", "unknown", "requires integer"),
        ("gpu-fault.io/restart-budget", "unknown", "requires non-negative"),
        ("gpu-fault.io/training-container", "missing", "training container"),
        ("gpu-fault.io/rank", "first", "requires integer rank"),
        ("gpu-fault.io/workload-log-snapshot", "{broken", "invalid"),
        ("gpu-fault.io/workload-log-snapshot", "[]", "requires object"),
        ("gpu-fault.io/gpu-uuids", '["GPU-1", 1]', "JSON string list"),
        (
            "gpu-fault.io/workload-ids",
            '["default/job/train", false]',
            "JSON string list",
        ),
    ],
)
def test_malformed_attempt_does_not_block_a_healthy_sibling(
    key, value, diagnostic, caplog
) -> None:
    malformed = pod(0, attempt_id="bad")
    malformed["metadata"]["annotations"][key] = value
    good = pod(0, attempt_id="good", exit_code=0)
    good["metadata"]["uid"] = "good-uid"
    core, sink = FakeCoreApi([malformed, good]), FakeSink()
    controller = make_controller(core, sink)
    controller.run_once()
    assert controller.reconcile_failures_total == 1
    terminals = [payload for path, payload in sink.posts if path.endswith("/terminal")]
    assert [item["attempt_id"] for item in terminals] == ["good"]
    assert all(item["terminal_status"] == "SUCCEEDED" for item in terminals), (
        "invalid metadata must not synthesize a sibling failure"
    )
    assert any(
        diagnostic_part in caplog.text for diagnostic_part in diagnostic.split("|")
    ), "the rejected field needs an actionable diagnostic"


def test_unannotated_training_container_is_inferred_only_when_unambiguous(caplog):
    current = pod(0)
    current["metadata"]["annotations"].pop("gpu-fault.io/training-container")
    core, sink = FakeCoreApi([current]), FakeSink()
    controller = make_controller(core, sink)
    controller.run_once()
    assert controller.reconcile_failures_total == 0
    observations = [
        payload
        for path, payload in sink.posts
        if path.endswith("workload-observations")
    ]
    assert observations[-1]["containers"][0]["container_name"] == "trainer"
    current["spec"]["containers"].append({"name": "sidecar"})
    controller.run_once()
    assert controller.reconcile_failures_total == 1
    assert "must identify its training container" in caplog.text


@pytest.mark.parametrize("quantity", [None, "not-a-quantity"])
def test_invalid_gpu_quantity_is_not_reported_as_zero(quantity, caplog) -> None:
    current = pod(0)
    current["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = quantity
    controller = make_controller(FakeCoreApi([current]))
    controller.run_once()
    assert controller.reconcile_failures_total == 1
    assert "invalid nvidia.com/gpu quantity" in caplog.text


def test_missing_rank_rejects_attempt_without_a_global_offset(caplog) -> None:
    controller = make_controller(FakeCoreApi([pod(None)]))
    controller.run_once()
    assert controller.reconcile_failures_total == 1
    assert "requires integer rank" in caplog.text


@pytest.mark.parametrize("discovered", [[], ["not-a-gpu"], ["GPU-1", "GPU-1"]])
def test_invalid_discovery_uses_backoff_then_recovers_without_duplicate_calls(
    discovered,
):
    clock, sink = Clock(), FakeSink()
    current = pod(0, include_gpu_uuids=False, gpu_count=1)
    core, calls = FakeCoreApi([current]), []

    def resolver(item, container):
        calls.append((item["metadata"]["uid"], container))
        return discovered if len(calls) == 1 else [" ", " MIG-local "]

    controller = make_controller(core, sink, now=clock, gpu_uuid_resolver=resolver)
    controller.run_once()
    controller.run_once()
    assert len(calls) == 1
    clock.value += timedelta(seconds=3)
    controller.run_once()
    controller.run_once()
    assert len(calls) == 2
    observations = [
        payload
        for path, payload in sink.posts
        if path.endswith("workload-observations")
    ]
    assert observations[-1]["containers"][0]["gpu_uuids"] == ["MIG-local"]


def test_missing_resolver_keeps_gpu_allocation_unknown() -> None:
    core, sink = FakeCoreApi([pod(0, include_gpu_uuids=False, gpu_count=1)]), FakeSink()
    controller = make_controller(core, sink)
    controller.run_once()
    observations = [
        payload
        for path, payload in sink.posts
        if path.endswith("workload-observations")
    ]
    assert observations[-1]["containers"][0]["gpu_uuids"] == []
    assert observations[-1]["containers"][0]["gpu_count"] == 1


def test_valid_snapshot_comma_lists_and_non_workload_owners_preserve_observation():
    current = pod(0)
    annotations = current["metadata"]["annotations"]
    annotations["gpu-fault.io/gpu-uuids"] = " GPU-1, ,MIG-2 "
    annotations["gpu-fault.io/workload-log-snapshot"] = json.dumps(
        {"record_id": "log-1"}
    )
    current["metadata"]["ownerReferences"] = [
        {"kind": "ReplicaSet", "name": "ignored"},
        {"kind": "Job", "name": ""},
    ]
    sink = FakeSink()
    make_controller(FakeCoreApi([current]), sink).run_once()
    payload = next(
        payload
        for path, payload in sink.posts
        if path.endswith("workload-observations")
    )
    assert payload["workload_ids"] == []
    assert payload["containers"][0]["gpu_uuids"] == ["GPU-1", "MIG-2"]
    assert payload["containers"][0]["workload_log_snapshot"] == {"record_id": "log-1"}
