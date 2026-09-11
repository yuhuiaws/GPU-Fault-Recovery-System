"""The Fabric Manager log collector's two fences against replaying history.

A node reboot on the shared acceptance site re-read the whole
``fabricmanager.log`` and posted a four-day-old Always-Fatal SXID as a new
fault: the root filesystem's ``st_dev`` had changed across the boot and the
collector took that for a different file, and nothing questioned a line that
old. The checkpoint now follows the inode, and stale lines advance it silently.
"""

from __future__ import annotations

from ._support import (
    NOW,
    FabricManagerLogCollector,
    RecordingSink,
    context,
    json,
    timedelta,
)


def _write_fabric_file_state(state, log, *, offset: int) -> None:
    stat = log.stat()
    state.write_text(
        json.dumps(
            {
                "journal_cursor": None,
                "files": {
                    str(log): {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "offset": offset,
                    }
                },
            }
        )
    )


def test_fabric_manager_keeps_its_checkpoint_when_the_device_number_changes(tmp_path):
    """st_dev is boot-scoped: one node's root moved from 66305 to 66306 across a
    reboot. Same inode and no truncation is the same file, so the checkpoint
    stands; resetting it replayed a four-day-old Always-Fatal SXID as new."""

    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "fabric-manager-state.json"
    log.write_text(
        "[2026-07-15T04:18:55.513286+00:00] nvidia-nvswitch0: "
        "SXid (PCI:0000:59:00.0): 23001, Fatal, Link 12 Always Fatal\n"
    )
    stat = log.stat()
    state.write_text(
        json.dumps(
            {
                "journal_cursor": None,
                "files": {
                    str(log): {
                        "device": stat.st_dev + 1,
                        "inode": stat.st_ino,
                        "offset": stat.st_size,
                    }
                },
            }
        )
    )
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    stats = collector.collect_once()

    assert stats.observed == 0 and stats.delivered == 0, (
        "a device-number change replayed the log from offset 0"
    )
    saved = json.loads(state.read_text())["files"][str(log)]
    assert saved["device"] == stat.st_dev and saved["offset"] == stat.st_size, saved


def test_fabric_manager_skips_lines_older_than_the_max_line_age(tmp_path):
    """A line whose own timestamp is older than max_line_age advances the
    checkpoint but is not reported: history re-read after a lost checkpoint is
    not a fault. A current line in the same read is still delivered."""

    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "fabric-manager-state.json"
    old_line = (
        f"[{(NOW - timedelta(days=4)).isoformat()}] nvidia-nvswitch0: "
        "SXid (PCI:0000:59:00.0): 23001, Fatal, Link 12 Always Fatal\n"
    )
    fresh_line = (
        f"[{(NOW - timedelta(seconds=30)).isoformat()}] nvidia-nvswitch3: "
        "SXid (PCI:0000:c1:00.0): 12020, Fatal, Link 46 egress sequence ID error\n"
    )
    log.write_text(old_line + fresh_line)
    _write_fabric_file_state(state, log, offset=0)
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    stats = collector.collect_once()

    assert stats.delivered == 1 and stats.observed == 1, stats
    assert collector.stale_lines_skipped == 1, "the old line was not counted"
    assert "12020" in sink.requests[0][1]["message"], "the wrong line was delivered"
    assert (
        json.loads(state.read_text())["files"][str(log)]["offset"] == log.stat().st_size
    ), "the checkpoint did not move past the stale line"
