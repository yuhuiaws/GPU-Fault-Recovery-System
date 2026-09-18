"""Default producer factories reject missing identity before any host call."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from gpu_fault.collector_registry import COLLECTOR_REGISTRY
from gpu_fault.collectors import training_progress
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize(
    "command", ["kernel", "dcgm", "nvidia-smi", "host", "logs", "fabric-manager"]
)
def test_node_producer_factory_refuses_missing_identity_before_host_operations(command):
    sink = support.RecordingSink()
    with pytest.raises(SystemExit, match="node-id.*required"):
        COLLECTOR_REGISTRY[command].build(
            sink, support.collector_context(), argparse.Namespace(node_id=None)
        )
    assert sink.requests == []


def test_training_loop_recovers_delivery_without_losing_the_next_progress_read(
    monkeypatch, tmp_path, caplog
):
    clock = support.Clock()
    path = tmp_path / "progress.json"
    path.write_text('{"step":1}')
    sink = support.RecordingSink(CollectorError("fake delivery unavailable"))
    collector = training_progress.TrainingProgressCollector(
        sink,
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=0,
        progress_path=str(path),
        now=clock.now,
    )
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            raise support.StopLoop
        sink.error = None
        path.write_text('{"step":2}')
        clock.sleep(seconds)

    monkeypatch.setattr(training_progress, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(support.StopLoop):
        collector.run()
    assert [payload["step"] for _, payload in sink.requests] == [1, 2]
    assert waits == [15, 15]
    assert "training progress collection failed" in caplog.text
