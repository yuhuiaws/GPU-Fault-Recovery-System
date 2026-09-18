from __future__ import annotations

import copy
import io
import json
import os
import sys
from pathlib import Path

import pytest

from gpu_fault.admin import node_key_custody_process as process
from gpu_fault.admin.node_key_custody_models import CustodyError

NS = 1_000_000_000


def proof():
    return {
        "sha256": "a" * 64,
        "mtime_ns": 10 * NS + 250_000_000,
        "ctime_ns": 10 * NS + 250_000_000,
        "process_start_ticks": 1050,
        "ticks_per_second": 100,
        "boottime_ns": 100 * NS,
        "realtime_before_ns": 100 * NS,
        "realtime_after_ns": 100 * NS + 1000,
    }


def test_same_second_mount_precedes_the_actual_kernel_process_start():
    value = proof()
    assert value["mtime_ns"] > 10 * NS, "the old seconds-only comparison rejected this"
    assert process.projection_precedes_start(
        value, started_at="1970-01-01T00:00:10Z", expected_digest="a" * 64
    ), "a proved mount before PID1 startup must pass despite second-precision CRI time"


@pytest.mark.parametrize("field", ["mtime_ns", "ctime_ns"])
@pytest.mark.parametrize("later", [10 * NS + 505_000_000, 11 * NS])
def test_after_start_or_same_tick_ambiguity_is_not_rounded_back(field, later):
    value = proof()
    value[field] = later
    assert (
        process.projection_precedes_start(
            value, started_at="1970-01-01T00:00:10Z", expected_digest="a" * 64
        )
        is False
    )


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("1970-01-01T00:00:10Z", (10 * NS, 11 * NS)),
        ("1970-01-01T00:00:10.123456789Z", (10_123_456_789, 10_123_456_790)),
        ("1970-01-01T01:00:10.125+01:00", (10_125_000_000, 10_126_000_000)),
    ],
)
def test_timestamp_precision_is_an_integer_interval(stamp, expected):
    assert process.started_interval(stamp) == expected


@pytest.mark.parametrize(
    "stamp", ["invalid", "2026-99-99T01:00:00Z", "1970-01-01T00:00:10.1234567890Z"]
)
def test_invalid_container_timestamp_does_not_authorize_a_proof(stamp):
    with pytest.raises(CustodyError):
        process.started_interval(stamp)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ctime_ns", True),
        ("mtime_ns", -1),
        ("process_start_ticks", 10.5),
        ("ticks_per_second", 0),
        ("ticks_per_second", 2_000_000),
        ("realtime_after_ns", 1),
        ("realtime_after_ns", 101 * NS),
        ("process_start_ticks", 99999999),
    ],
)
def test_bad_kernel_clock_evidence_fails_closed(field, value):
    document = proof()
    document[field] = value
    with pytest.raises(CustodyError):
        process.projection_precedes_start(
            document, started_at="1970-01-01T00:00:10Z", expected_digest="a" * 64
        )


def test_an_inconsistent_kernel_and_container_epoch_is_not_retried_as_a_key_lag():
    with pytest.raises(CustodyError, match="bound container clock"):
        process.projection_precedes_start(
            proof(), started_at="1970-01-01T00:00:40Z", expected_digest="a" * 64
        )


def test_wrong_projected_bytes_do_not_prove_the_signed_key():
    assert (
        process.projection_precedes_start(
            proof(), started_at="1970-01-01T00:00:10Z", expected_digest="b" * 64
        )
        is False
    )


@pytest.mark.parametrize("field", ["uid", "containerID", "state", "ready"])
def test_a_container_replaced_during_the_read_cannot_supply_activation_evidence(field):
    status = {
        "name": "api",
        "containerID": "containerd://original",
        "ready": True,
        "state": {"running": {"startedAt": "1970-01-01T00:00:10Z"}},
    }
    pod = {
        "metadata": {
            "name": "api",
            "uid": "pod",
            "labels": {"app": "gpu-fault-api-ha"},
        },
        "spec": {"containers": [{"name": "api"}]},
        "status": {"containerStatuses": [status]},
    }
    after = copy.deepcopy(pod)
    if field == "uid":
        after["metadata"]["uid"] = "replacement"
    else:
        after["status"]["containerStatuses"][0][field] = (
            False if field == "ready" else {} if field == "state" else "replacement"
        )
    with pytest.raises(CustodyError, match="restarted"):
        process.projected_key_precedes_process(
            plane="cpu",
            pod=pod,
            namespace="unit-system",
            node_id="node-a",
            expected_digest="a" * 64,
            kubectl=["kubectl"],
            run=lambda *_args, **_kwargs: json.dumps(proof()),
            read_pod=lambda _name: after,
        )


def test_atomic_projection_replacement_is_detected_even_when_the_open_fd_is_stable(
    tmp_path, monkeypatch
):
    key = tmp_path / "node-a"
    key.write_bytes(b"a" * 64)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"b" * 64)
    original_open = Path.open

    class ReplacingRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            value = self.handle.read(size)
            os.replace(replacement, key)
            return value

    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *args, **kwargs: ReplacingRead(
            original_open(path, *args, **kwargs)
        )
        if path == key
        else original_open(path, *args, **kwargs),
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"node_id":"node-a"}'))
    with pytest.raises(RuntimeError, match="unstable"):
        exec(compile(process.PROJECTED_KEY_PROBE, "<projection-probe>", "exec"), {})
