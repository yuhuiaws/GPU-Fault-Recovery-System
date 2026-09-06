"""Node-log entry ids carry the node's identity (F-B7 / P0-38A).

A training-log entry used to be identified by ``sha256(path:offset)`` and a
journal line without a ``__CURSOR`` by ``sha256(line)``. Every rank of a
distributed job writes the same path on its own node, and two nodes print the
same kernel line verbatim, so two nodes' faults shared one id and the second
one was folded into the first one's incident and never recovered.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import gpu_fault.collectors.logs.node as node_module
from gpu_fault.collectors.logs.node import NodeLogCollector
from gpu_fault.host_health import NodeLogBatch

from ._support import NOW, RecordingSink, context

_EMPTY_JOURNAL = subprocess.CompletedProcess([], 0, stdout="", stderr="")
NCCL_LINE = "NCCL WARN collective timeout error\n"
MCE_LINE = {
    "__REALTIME_TIMESTAMP": "1753012800000000",
    "_TRANSPORT": "kernel",
    "MESSAGE": "mce: [Hardware Error]: Machine check events logged",
}


@pytest.fixture(autouse=True)
def _journalctl_present(monkeypatch) -> None:
    monkeypatch.setattr(
        node_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )


def _journal_runner(*lines: dict[str, str]):
    stdout = "\n".join(json.dumps(item) for item in lines)

    def runner(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    return runner


def _collector(
    node_id: str, log: Path, *, boot_id: str = "boot-1", runner=None
) -> NodeLogCollector:
    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id=node_id,
        boot_id=boot_id,
        training_log_paths=[str(log)],
        initial_tail_bytes=4096,
        now=lambda: NOW,
        runner=runner or (lambda *_args, **_kwargs: _EMPTY_JOURNAL),
    )


def _collect(node_id: str, log: Path, **options) -> NodeLogBatch:
    return _collector(node_id, log, **options).collect_once()


def test_training_log_entry_ids_differ_between_nodes_reading_the_same_path(
    tmp_path,
) -> None:
    # Two ranks, two nodes, one path: /var/log/train.log at offset 0 on both.
    # The shared tmp file stands in for the same path on two hosts.
    log = tmp_path / "train.log"
    log.write_text(NCCL_LINE, encoding="utf-8")

    node_a = _collect("node-a", log).entries
    node_b = _collect("node-b", log).entries

    assert len(node_a) == len(node_b) == 1
    assert node_a[0].entry_id != node_b[0].entry_id


def test_training_log_entry_id_changes_when_the_file_is_rotated(tmp_path) -> None:
    # After a rotation the replacement file starts at offset 0 again; the id
    # must not repeat the pre-rotation line's id or the new fault dedupes into
    # the old incident (possibly a SUCCEEDED one).
    log = tmp_path / "train.log"
    log.write_text(NCCL_LINE, encoding="utf-8")
    collector = _collector("node-a", log)

    before = collector.collect_once().entries
    rotated = tmp_path / "train.log.new"
    rotated.write_text(NCCL_LINE, encoding="utf-8")
    os.replace(rotated, log)
    after = collector.collect_once().entries

    assert len(before) == len(after) == 1
    assert before[0].entry_id != after[0].entry_id


def test_training_log_entry_id_is_stable_for_a_re_read_within_one_boot(
    tmp_path,
) -> None:
    # The boundary entry is read again after a budget cut and de-duplicated
    # by id on the control plane; the identity must therefore stay put when
    # nothing about the file changed.
    log = tmp_path / "train.log"
    log.write_text(NCCL_LINE, encoding="utf-8")

    first = _collect("node-a", log).entries
    second = _collect("node-a", log).entries

    assert first[0].entry_id == second[0].entry_id


def test_training_log_entry_id_differs_between_boots(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text(NCCL_LINE, encoding="utf-8")

    first_boot = _collect("node-a", log, boot_id="boot-1").entries
    second_boot = _collect("node-a", log, boot_id="boot-2").entries

    assert first_boot[0].entry_id != second_boot[0].entry_id


def test_journal_fallback_id_differs_between_nodes_printing_the_same_line(
    tmp_path,
) -> None:
    # No ``__CURSOR``: the fallback used to hash the raw line, and two nodes
    # logging the same machine-check text verbatim shared one id.
    runner = _journal_runner(MCE_LINE)
    missing = tmp_path / "none.log"

    entries_a = _collect("node-a", missing, runner=runner).entries
    entries_b = _collect("node-b", missing, runner=runner).entries

    assert len(entries_a) == len(entries_b) == 1
    assert entries_a[0].source == "dmesg"
    assert entries_a[0].entry_id != entries_b[0].entry_id


def test_journal_cursor_is_still_used_verbatim_when_present(tmp_path) -> None:
    cursor = "s=abc;i=1;b=boot;m=1;t=1;x=1"
    runner = _journal_runner({**MCE_LINE, "__CURSOR": cursor})

    entries = _collect("node-a", tmp_path / "none.log", runner=runner).entries

    assert [item.entry_id for item in entries] == [cursor]
