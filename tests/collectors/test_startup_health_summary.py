"""Every log collector posts its health summary on its first collection.

Live 2026-09-12: a fresh bootstrap's first verify ran three minutes after the
agents came up and read the nodes as silent, because the health-summary kinds
post unconditionally only every 300 s and an idle cluster produces no log lines
or metric edges before that. The first summary now goes out at start-up; the
later ones keep their stable phase.
"""

from __future__ import annotations

import io
import json
import subprocess
from datetime import timedelta

import pytest

from gpu_fault.collectors.logs import node as node_module
from gpu_fault.collectors.logs.fabric_manager import FabricManagerLogCollector
from gpu_fault.collectors.logs.kernel import KernelLogCollector
from gpu_fault.collectors.logs.node import NodeLogCollector

from ._support import NOW, RecordingSink, context


def _summaries(sink: RecordingSink) -> list[dict]:
    return [
        payload
        for _path, payload in sink.requests
        if "health-summary" in (payload.get("edge_filter_reasons") or [])
    ]


def test_the_kernel_collector_posts_a_summary_when_its_stream_opens(
    monkeypatch,
) -> None:
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: io.StringIO(""))

    def stop(_seconds: float) -> None:
        raise KeyboardInterrupt

    sink = RecordingSink()
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=stop,
    )

    with pytest.raises(KeyboardInterrupt):
        collector.run()

    assert len(_summaries(sink)) == 1, (
        f"an empty kmsg stream must still announce the collector once: {sink.requests}"
    )


def test_the_fabric_manager_collector_posts_a_summary_on_its_first_collection(
    tmp_path,
) -> None:
    log = tmp_path / "fabricmanager.log"
    log.write_text("", encoding="utf-8")
    clock = [NOW]
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(tmp_path / "state.json"),
        now=lambda: clock[0],
    )

    collector.collect_once()
    first = len(sink.requests)
    clock[0] = NOW + timedelta(seconds=1)
    collector.collect_once()

    assert first == 1 and len(sink.requests) == 1, (
        f"exactly one summary at start-up, none a second later: {sink.requests}"
    )
    assert "summary_id" in sink.requests[0][1], "the start-up post is the summary"


def test_the_node_log_collector_posts_a_summary_on_its_first_collection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        node_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )

    def journalctl(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    # A journal cursor from an earlier run: the first collection is clean, so
    # the periodic slot carries the summary (an error-only first collection
    # is posted too, on the same slot, as a collection error).
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"journal_since": (NOW - timedelta(seconds=60)).isoformat()}),
        encoding="utf-8",
    )
    clock = [NOW]
    collector = NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=[],
        state_path=str(state),
        now=lambda: clock[0],
        runner=journalctl,
    )

    batch = collector.collect_once()
    clock[0] = NOW + timedelta(seconds=1)
    later = collector.collect_once()

    assert batch.edge_filter_reasons == ["health-summary"], (
        f"the first empty collection is the start-up summary: {batch.edge_filter_reasons}"
    )
    assert later.edge_filter_reasons == [], (
        "a second later there is nothing to say until the next stable phase"
    )
    assert len(collector.sink.requests) == 1, "one summary was posted"
    assert json.loads(json.dumps(collector.sink.requests[0][1]))[
        "edge_filter_reasons"
    ] == ["health-summary"], "the posted payload names the summary"
