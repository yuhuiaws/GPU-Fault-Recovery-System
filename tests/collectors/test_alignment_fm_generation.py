"""File generations separate reused offsets without making retries new events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_fault.channel_registry import FABRIC_MANAGER_PATH
from gpu_fault.collectors.logs.fabric_manager import FabricManagerLogCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import CollectorError

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
LINE = b"SXid (PCI:0000:ab:00.0): 99999, Non-fatal, private repeated software event\n"
PADDING = b"private rotation padding\n" * 3


class Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.fail = False

    def post(self, path: str, payload: dict) -> dict:
        if path == FABRIC_MANAGER_PATH:
            self.events.append(payload)
            if self.fail:
                raise CollectorError("private sink delivery rejected")
        return {"accepted": True}


def reader(
    root: Path, sink: Sink, *, paths: list[str] | None = None
) -> FabricManagerLogCollector:
    return FabricManagerLogCollector(
        sink,
        CollectorContext(cluster_id="private-fm-alignment"),
        node_id="private-node",
        boot_id="private-boot",
        journal_enabled=False,
        log_paths=paths or [str(root / "*.log")],
        state_path=str(root / "cursor.json"),
        now=lambda: NOW,
    )


def state(root: Path, log: Path) -> dict:
    return json.loads((root / "cursor.json").read_text())["files"][str(log)]


def emitted_zero(root: Path, sink: Sink) -> Path:
    log = root / "active.log"
    log.write_bytes(b"")
    reader(root, sink).collect_once()
    log.write_bytes(LINE + PADDING)
    reader(root, sink).collect_once()
    assert len(sink.events) == 1 and sink.events[0]["fields"]["offset"] == "0", (
        "the pre-truncation zero offset must really have been emitted"
    )
    return log


def test_identical_timestamp_free_line_after_copytruncate_is_a_new_record(
    tmp_path: Path,
) -> None:
    sink = Sink()
    log = emitted_zero(tmp_path, sink)
    before = state(tmp_path, log)
    log.write_bytes(LINE)
    assert log.stat().st_ino == before["inode"]
    reader(tmp_path, sink).collect_once()
    after = state(tmp_path, log)
    first, second = sink.events
    assert first["message"] == second["message"]
    assert first["observed_at"] == second["observed_at"], (
        "uniqueness must not depend on a new message timestamp"
    )
    assert first["fields"]["offset"] == second["fields"]["offset"] == "0"
    assert first["record_id"] != second["record_id"], (
        "copytruncate cannot reuse an already accepted event identity"
    )
    assert first["evidence_ref"] != second["evidence_ref"], (
        "the new raw evidence must not overwrite the preceding generation"
    )
    assert after["generation"] == before["generation"] + 1
    assert after["offset"] == len(LINE)
    reader(tmp_path, sink).collect_once()
    assert len(sink.events) == 2, "restart must retain the new EOF and generation"
    assert state(tmp_path, log) == after


def test_empty_truncation_is_checkpointed_before_a_later_append(tmp_path: Path) -> None:
    sink = Sink()
    log = emitted_zero(tmp_path, sink)
    original = state(tmp_path, log)
    log.write_bytes(b"")
    reader(tmp_path, sink).collect_once()
    truncated = state(tmp_path, log)
    assert truncated["generation"] == original["generation"] + 1
    assert truncated["offset"] == 0
    assert len(sink.events) == 1
    log.write_bytes(LINE)
    reader(tmp_path, sink).collect_once()
    assert state(tmp_path, log)["generation"] == truncated["generation"]
    assert len({event["record_id"] for event in sink.events}) == 2


def test_delivery_retry_after_truncation_retains_the_new_generation(
    tmp_path: Path,
) -> None:
    sink = Sink()
    log = emitted_zero(tmp_path, sink)
    original = state(tmp_path, log)
    log.write_bytes(LINE)
    sink.fail = True
    with pytest.raises(CollectorError):
        reader(tmp_path, sink).collect_once()
    pending = state(tmp_path, log)
    rejected = sink.events[-1]
    assert pending["generation"] == original["generation"] + 1
    assert pending["offset"] == 0, "unaccepted delivery must not advance the cursor"
    sink.fail = False
    reader(tmp_path, sink).collect_once()
    assert sink.events[-1]["record_id"] == rejected["record_id"], (
        "restart of a failed delivery must retry the original identity"
    )
    assert sink.events[-1]["evidence_ref"] == rejected["evidence_ref"]
    assert state(tmp_path, log)["generation"] == pending["generation"]


def test_generation_is_durable_before_first_delivery_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = Sink()
    log = emitted_zero(tmp_path, sink)
    old_generation = state(tmp_path, log)["generation"]
    log.write_bytes(LINE)
    original_post = sink.post

    def checked_post(path: str, payload: dict) -> dict:
        if path == FABRIC_MANAGER_PATH:
            saved = state(tmp_path, log)
            assert saved["generation"] == old_generation + 1, (
                "a sink ACK must never precede the durable generation boundary"
            )
            assert saved["offset"] == 0
            assert payload["fields"]["generation"] == str(saved["generation"])
        return original_post(path, payload)

    monkeypatch.setattr(sink, "post", checked_post)
    reader(tmp_path, sink).collect_once()
    assert len(sink.events) == 2


@pytest.mark.parametrize("old_name", ["active.log", "zz.log"])
def test_rename_after_lost_delivery_keeps_the_record_id_regardless_of_glob_order(
    tmp_path: Path, old_name: str
) -> None:
    sink = Sink()
    log = tmp_path / old_name
    log.write_bytes(b"")
    reader(tmp_path, sink).collect_once()
    log.write_bytes(LINE)
    sink.fail = True
    with pytest.raises(CollectorError):
        reader(tmp_path, sink).collect_once()
    rejected = sink.events[-1]
    recorded = state(tmp_path, log)
    rotated = tmp_path / "middle.log"
    log.rename(rotated)
    log.write_bytes(b"")
    sink.fail = False
    reader(tmp_path, sink).collect_once()
    assert sink.events[-1]["record_id"] == rejected["record_id"], (
        "a rename is not a new generation of the old inode"
    )
    assert state(tmp_path, rotated)["generation"] == recorded["generation"]
    assert state(tmp_path, log)["generation"] != recorded["generation"]
    assert state(tmp_path, rotated)["offset"] == len(LINE)


def test_missing_checkpoint_does_not_recycle_a_previous_truncation_generation(
    tmp_path: Path,
) -> None:
    sink = Sink()
    log = emitted_zero(tmp_path, sink)
    log.write_bytes(LINE)
    reader(tmp_path, sink).collect_once()
    old_generation = state(tmp_path, log)["generation"]
    (tmp_path / "cursor.json").unlink()
    reader(tmp_path, sink).collect_once()
    assert len(sink.events) == 2, "lost state must baseline at EOF, not replay history"
    assert state(tmp_path, log)["generation"] != old_generation
    with log.open("ab") as stream:
        stream.write(PADDING)
    reader(tmp_path, sink).collect_once()
    log.write_bytes(LINE)
    reader(tmp_path, sink).collect_once()
    assert len({event["record_id"] for event in sink.events}) == 3


def test_reused_rotation_names_keep_each_inodes_generation_and_acknowledged_offset(
    tmp_path: Path,
) -> None:
    sink = Sink()
    active = emitted_zero(tmp_path, sink)
    first = state(tmp_path, active)
    rotated = tmp_path / "rotated.log"
    active.rename(rotated)
    active.write_bytes(LINE)
    reader(tmp_path, sink).collect_once()
    second = state(tmp_path, active)
    assert len(sink.events) == 2
    assert state(tmp_path, rotated) == first
    oldest = tmp_path / "oldest.log"
    rotated.rename(oldest)
    active.rename(rotated)
    active.write_bytes(b"")
    reader(tmp_path, sink).collect_once()
    assert len(sink.events) == 2, "reused rotation names must not replay old SXIDs"
    assert state(tmp_path, oldest) == first
    assert state(tmp_path, rotated) == second
    assert state(tmp_path, active)["generation"] not in {
        first["generation"],
        second["generation"],
    }
    with rotated.open("ab") as stream:
        stream.write(LINE)
    reader(tmp_path, sink).collect_once()
    assert len(sink.events) == 3
    assert sink.events[-1]["fields"]["offset"] == str(second["offset"])
    assert sink.events[-1]["fields"]["generation"] == str(second["generation"])


def test_stat_open_rotation_cannot_emit_content_under_the_wrong_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = Sink()
    active = emitted_zero(tmp_path, sink)
    original = state(tmp_path, active)
    old = tmp_path / "rotated.log"
    real_open = Path.open
    rotated = False

    def rotating_open(path: Path, mode: str = "r", *args, **kwargs):
        nonlocal rotated
        if path == active and mode == "rb" and not rotated:
            rotated = True
            active.rename(old)
            active.write_bytes(LINE + PADDING + LINE)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", rotating_open)
    reader(tmp_path, sink).collect_once()
    assert rotated, "the fixture must rotate the source between stat and open"
    assert len(sink.events) == 1, "a stat/open identity race must defer this file"
    reader(tmp_path, sink).collect_once()
    assert len(sink.events) == 3
    assert state(tmp_path, old) == original
    assert all(
        event["fields"]["inode"] == str(active.stat().st_ino)
        for event in sink.events[1:]
    ), "post-rotation events must identify the inode actually read"


def test_legacy_checkpoint_replay_remains_idempotent_without_generation(
    tmp_path: Path,
) -> None:
    sink = Sink()
    log = tmp_path / "active.log"
    log.write_bytes(LINE)
    stat = log.stat()
    legacy = json.dumps(
        {
            "files": {
                str(log): {"device": stat.st_dev, "inode": stat.st_ino, "offset": 0}
            }
        }
    )
    checkpoint = tmp_path / "cursor.json"
    checkpoint.write_text(legacy)
    reader(tmp_path, sink).collect_once()
    first = sink.events[-1]
    checkpoint.write_text(legacy)
    reader(tmp_path, sink).collect_once()
    assert sink.events[-1]["record_id"] == first["record_id"]
    assert sink.events[-1]["evidence_ref"] == first["evidence_ref"]
    assert state(tmp_path, log)["generation"] == 0
