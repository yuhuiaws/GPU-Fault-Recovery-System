"""journald hands back a byte array for a MESSAGE that is not UTF-8 (ARCH-G9).

``str()`` of that list is ``"[78, 86, ...]"``, which matches no log rule, so a
driver line with one stray byte in it disappeared from the batch.
"""

from __future__ import annotations

from ._support import NOW, NodeLogCollector, RecordingSink, context, json, subprocess


def test_a_byte_array_message_is_decoded_and_counted(tmp_path, monkeypatch) -> None:
    from gpu_fault.collectors.logs import node as module

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/journalctl")
    raw = list(b"machine check hardware error \xff on cpu 3")
    line = json.dumps(
        {
            "__CURSOR": "c-1",
            "__REALTIME_TIMESTAMP": str(int(NOW.timestamp() * 1_000_000) - 1000),
            "MESSAGE": raw,
            "_TRANSPORT": "kernel",
        }
    )

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(list(command), 0, stdout=line, stderr="")

    collector = NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
        runner=runner,
    )

    batch = collector.collect_once()

    assert len(batch.entries) == 1, "the byte-array message matched no rule"
    assert batch.entries[0].message.startswith("machine check hardware error"), (
        batch.entries[0].message
    )
    assert "�" in batch.entries[0].message, "the bad byte was not replaced"
    assert collector.binary_messages_total == 1, "the decode was not counted"
