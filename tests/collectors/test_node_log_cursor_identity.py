"""A journald cursor is carried whole and identifies the entry whole (F-M3).

A ``__CURSOR`` is typically 150-190 characters (``s=..;i=..;b=..;m=..;t=..;x=..``).
The control plane's readable-id sanitiser truncates to 160 characters, and the
review found the cursor passing through it (P3-38I): two cursors that differed
only in their ``t=``/``x=`` tail could have spelled one event. The collector
never truncated the cursor itself; the control plane now digests ``entry_id``
instead of sanitising it. These cases pin both halves.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import gpu_fault.collectors.logs.node as node_module
from gpu_fault.collectors.logs.node import NodeLogCollector
from gpu_fault.host_health import NodeHealthPolicy, NodeLogBatch, NodeLogEntry
from tests._builders import build_store

from ._support import NOW, RecordingSink, context

LONG_CURSOR_HEAD = (
    "s=6f3a0b2c9d8e4f1a8b7c6d5e4f3a2b1c;i=1a2b3c4d5e6f;"
    "b=9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b;m=1f2e3d4c5b6a7980f1e2d3c4b5a69788;"
    "t=5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a;x="
)
MCE_LINE = {
    "__REALTIME_TIMESTAMP": "1753012800000000",
    "_TRANSPORT": "kernel",
    "MESSAGE": "mce: [Hardware Error]: Machine check events logged",
}


@pytest.fixture(autouse=True)
def _journalctl_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        node_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )


def _cursor(tail: str) -> str:
    cursor = LONG_CURSOR_HEAD + tail
    assert len(cursor) > 160, "the fixture cursor must be longer than the sanitiser cap"
    return cursor


def _collect(cursor: str, tmp_path: Path) -> NodeLogBatch:
    line = json.dumps({**MCE_LINE, "__CURSOR": cursor})

    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=line, stderr="")

    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="node-a",
        boot_id="boot-1",
        training_log_paths=[str(tmp_path / "none.log")],
        now=lambda: NOW,
        runner=runner,
    ).collect_once()


def test_a_long_journal_cursor_is_carried_whole(tmp_path: Path) -> None:
    cursor = _cursor("0123456789abcdef0123456789abcdef")

    batch = _collect(cursor, tmp_path)

    assert [entry.entry_id for entry in batch.entries] == [cursor]


def test_two_cursors_that_differ_only_past_160_characters_are_two_events() -> None:
    policy = NodeHealthPolicy(build_store())
    first, second = (
        _cursor("0123456789abcdef0123456789abcdef"),
        _cursor("0123456789abcdef0123456789abcdee"),
    )
    assert first[:160] == second[:160], "the fixture must differ only in the tail"

    def findings(cursor: str) -> list:
        return policy.evaluate_logs(
            NodeLogBatch(
                batch_id="logs-node-a-1",
                cluster_id="cluster-a",
                node_id="node-a",
                collected_at=NOW,
                runtime_profile_version="simulated-v1",
                entries=[
                    NodeLogEntry(
                        entry_id=cursor,
                        source="dmesg",
                        observed_at=NOW,
                        message=MCE_LINE["MESSAGE"],
                    )
                ],
            )
        )

    found_first, found_second = findings(first), findings(second)

    assert len(found_first) == len(found_second) == 1
    assert found_first[0].event_id != found_second[0].event_id
    assert found_first[0].evidence_ref.endswith(first), (
        "the evidence reference must carry the whole cursor"
    )
