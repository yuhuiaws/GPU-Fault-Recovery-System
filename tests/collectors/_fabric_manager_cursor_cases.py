"""Fabric Manager cursor identity, rotation and incomplete-record contracts."""

from __future__ import annotations

from ._support import (
    NOW,
    FabricManagerLogCollector,
    RecordingSink,
    context,
    json,
    logging,
    subprocess,
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


def _fabric_sxid_journal(cursor: str) -> str:
    return json.dumps(
        {
            "__CURSOR": cursor,
            "__REALTIME_TIMESTAMP": "1753012800000000",
            "_SYSTEMD_UNIT": "nvidia-fabricmanager.service",
            "MESSAGE": (
                "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, "
                "Fatal, Link 46 egress sequence ID error"
            ),
        }
    )


def test_fabric_manager_file_record_ids_are_node_scoped(tmp_path) -> None:
    """Two nodes at the same path and offset must not mint one record id.

    ``nvidia_logs.py`` scopes the derived ``event_id`` to the cluster only, so node
    uniqueness is the collector's job. Identically imaged nodes share the
    device and inode of ``/var/log/fabricmanager.log``, so ``dev:ino:offset``
    folded node B's SXID into node A's event -- the P0-38A class that
    ``node.py:_entry_identity`` already fixed for training logs.
    """

    log = tmp_path / "fabricmanager.log"
    log.write_text(
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Non-fatal, Link 12\n"
    )
    record_ids: dict[str, str] = {}
    references: dict[str, str] = {}
    for node_id in ("worker-a", "worker-b"):
        state = tmp_path / f"state-{node_id}.json"
        _write_fabric_file_state(state, log, offset=0)
        sink = RecordingSink()
        FabricManagerLogCollector(
            sink,
            context(),
            node_id=node_id,
            boot_id="boot-shared-by-the-image",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        ).collect_once()
        record_ids[node_id] = sink.requests[0][1]["record_id"]
        references[node_id] = sink.requests[0][1]["evidence_ref"]

    assert record_ids["worker-a"] != record_ids["worker-b"], (
        "two nodes reading the same file offset shared one record id"
    )
    assert all(value.startswith("fm-file-") for value in record_ids.values()), (
        "the record id prefix that names the source was dropped"
    )
    assert "worker-a" in references["worker-a"], (
        "the evidence reference does not say which node the line came from"
    )


def test_fabric_manager_journal_asks_for_untruncated_fields(tmp_path) -> None:
    """Without ``--all`` journalctl nulls any field over 4096 bytes.

    ``_message_text(None)`` is ``""``, which matches no SXID pattern, so a long
    Fabric Manager line was dropped without a trace on both the cold ``--since``
    query and the resumed ``--after-cursor`` one.
    """

    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        stdout = "" if "--after-cursor" in command else _fabric_sxid_journal("cursor-1")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    collector = FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
        runner=runner,
    )

    collector.collect_once()
    collector.collect_once()

    assert len(calls) == 2, f"expected a cold and a resumed journal query: {calls}"
    assert all("--all" in command for command in calls), (
        f"a journal query can still truncate fields over 4096 bytes: {calls}"
    )


def test_fabric_manager_reads_the_tail_of_a_renamed_log(tmp_path) -> None:
    """A rotation must not baseline the old inode at EOF under its new name.

    logrotate renames ``fabricmanager.log`` to ``fabricmanager.1.log``; the glob
    then sees the old inode under a name that has no checkpoint, and baselining
    it at EOF threw away every line written since the last poll -- including the
    fatal SXID that made Fabric Manager rotate in the first place.
    """

    log = tmp_path / "fabricmanager.log"
    log.write_text("Fabric Manager successfully configured\n")
    state = tmp_path / "state.json"
    sink = RecordingSink()

    def build() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(tmp_path / "fabricmanager*.log")],
            state_path=str(state),
            now=lambda: NOW,
        )

    build().collect_once()
    with log.open("a") as stream:
        stream.write(
            "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
            "Link 46 written before the rotation\n"
        )
    log.rename(tmp_path / "fabricmanager.1.log")
    log.write_text("")

    stats = build().collect_once()

    assert stats.delivered == 1, (
        "the tail written between the last poll and the rotation was lost"
    )
    events = [payload for _path, payload in sink.requests if "message" in payload]
    assert events[0]["message"].endswith("written before the rotation"), (
        f"the wrong line was delivered: {sink.requests}"
    )
    assert (
        str(tmp_path / "fabricmanager.1.log")
        in (json.loads(state.read_text())["files"])
    ), "the rotated file kept no checkpoint of its own"


def test_fabric_manager_journal_survives_an_unreadable_configured_file(
    tmp_path, monkeypatch, caplog
) -> None:
    """One 0600 log file must not discard the journal SXIDs of the round.

    ``collect_once`` builds ``[*journal, *files]`` before it delivers anything,
    so a ``PermissionError`` from the file tail threw away the journal records
    of that round as well -- every round, permanently, for a file whose mode
    never changes.
    """

    from pathlib import Path

    log = tmp_path / "fabricmanager.log"
    log.write_text("Fabric Manager successfully configured\n")
    state = tmp_path / "state.json"
    _write_fabric_file_state(state, log, offset=0)
    real_open = Path.open

    def refuse(self, *args, **kwargs):
        if str(self) == str(log):
            raise PermissionError(13, "Permission denied", str(log))
        return real_open(self, *args, **kwargs)

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=_fabric_sxid_journal("cursor-1"), stderr=""
        )

    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
        runner=runner,
    )
    monkeypatch.setattr(Path, "open", refuse)

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        stats = collector.collect_once()

    assert stats.delivered == 1, (
        "an unreadable file starved the journal source of the whole round"
    )
    assert sink.requests[0][1]["source"] == "journal", (
        f"the journal SXID was not the delivered record: {sink.requests}"
    )
    assert "unreadable" in caplog.text.lower(), (
        "the unreadable file was skipped without a warning"
    )


def test_fabric_manager_file_offsets_are_bytes_and_resync_to_a_line_boundary(
    tmp_path, caplog
) -> None:
    """Offsets are byte offsets, and a stored one mid-character re-syncs once.

    The offsets used to be ``TextIOWrapper.tell()`` cookies, so a stored value
    could land inside a multibyte character; ``seek(offset - 1)`` then read a
    replacement character instead of the newline. The read resumes at the next
    line boundary, says so once, and the line that follows is delivered exactly
    once.
    """

    first = "光纤管理器启动 [2026-08-13T12:34:56Z] configured\n"
    second = (
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Non-fatal, "
        "Link 12 SAW_MVB error\n"
    )
    log = tmp_path / "fabricmanager.log"
    log.write_bytes((first + second).encode("utf-8"))
    state = tmp_path / "state.json"
    # Two bytes into the first character: a legacy cookie, not a line boundary.
    _write_fabric_file_state(state, log, offset=2)
    sink = RecordingSink()

    def build() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        stats = build().collect_once()
    resumed = build().collect_once()

    assert stats.delivered == 1, f"the SXID after the resync was lost: {sink.requests}"
    assert sink.requests[0][1]["fields"]["offset"] == str(len(first.encode("utf-8"))), (
        "the record offset is not the byte offset of the line"
    )
    assert json.loads(state.read_text())["files"][str(log)]["offset"] == (
        log.stat().st_size
    ), "the committed offset is not a byte count"
    assert "line boundary" in caplog.text, (
        "resuming inside a line was not reported once"
    )
    assert resumed.observed == 0, "the same line was read twice"


def test_fabric_manager_keeps_its_offset_at_a_line_the_daemon_is_still_writing(
    tmp_path, caplog
) -> None:
    """A resume boundary inside an unterminated line must not be consumed.

    The resync branch skipped to the next newline and committed
    ``stream.tell()`` even when the "line" it skipped had no newline at all --
    it was the tail of a record the daemon was still writing. The bytes were
    consumed while the line was still incomplete, so the completed line was
    skipped a second time on the next round, and the one-per-collector warning
    never said so again. Only a complete line may advance the checkpoint, which
    is the invariant the main read loop already holds.
    """

    from pathlib import Path

    opening = "Fabric Manager daemon started, build 550.90.07\n"
    fatal = (
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Fatal, Link 12 SAW_MVB error"
    )
    log = tmp_path / "fabricmanager.log"
    log.write_bytes((opening + fatal).encode("utf-8"))
    state = tmp_path / "state.json"
    # A legacy cookie that landed inside the line still being written.
    _write_fabric_file_state(state, log, offset=len(opening.encode("utf-8")) + 10)
    sink = RecordingSink()

    def build() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        first_round = build().collect_once()
        held = json.loads(state.read_text())["files"][str(log)]["offset"]
        # The daemon finishes the fatal line and logs the next one.
        with Path(log).open("ab") as stream:
            stream.write(
                b"\nnvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
                b"Link 46 egress sequence ID error\n"
            )
        second_round = build().collect_once()

    assert first_round.delivered == 0, (
        f"an incomplete line was delivered as a record: {sink.requests}"
    )
    assert held == len(opening.encode("utf-8")) + 10, (
        "the checkpoint advanced over a line the daemon was still writing, so "
        f"the completed line is skipped again: {held}"
    )
    assert second_round.delivered == 1, (
        f"the fatal SXID after the completed line was lost: {sink.requests}"
    )
    events = [payload for _path, payload in sink.requests if "message" in payload]
    assert "12020" in events[0]["message"], (
        f"the delivered record is not the fatal SXID: {sink.requests}"
    )
    assert caplog.text.count("line boundary") == 1, (
        f"the skipped line was reported {caplog.text.count('line boundary')} times"
    )


def test_fabric_manager_decodes_a_binary_journal_message(tmp_path) -> None:
    """``journalctl --output=json`` emits MESSAGE as an array of byte values.

    Any field that is not valid UTF-8 arrives as ``[110, 118, ...]``, and
    ``str()`` of that list matches no SXID pattern, so the line was dropped
    without a trace -- the same hole ``node.py:_message_text`` already closed
    for training logs, and reachable here now that ``--all`` stops journalctl
    from nulling long fields.
    """

    text = (
        "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
        "Link 46 egress sequence ID error"
    )
    entry = json.dumps(
        {
            "__CURSOR": "s=binary;i=1",
            "__REALTIME_TIMESTAMP": "1753012800000000",
            "_SYSTEMD_UNIT": "nvidia-fabricmanager.service",
            "MESSAGE": list(text.encode("utf-8")) + [255],
        }
    )
    sink = RecordingSink()
    stats = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        log_paths=[],
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=entry + "\n", stderr=""
        ),
    ).collect_once()

    assert stats.delivered == 1, (
        f"a binary MESSAGE field dropped the fatal SXID: {sink.requests}"
    )
    assert text in sink.requests[0][1]["message"], (
        f"the MESSAGE byte array was stringified instead of decoded: {sink.requests}"
    )


def test_fabric_manager_reports_a_stall_inside_an_unfinished_record(
    tmp_path, caplog
) -> None:
    """Holding the offset must not look like a healthy collector.

    Nothing is read from the file until the daemon terminates the line, and a
    daemon killed mid-append never will. The hold was silent, so an operator saw
    a collector reporting success and no SXIDs; it now says so once per file and
    offset instead of every round.
    """

    opening = "Fabric Manager daemon started, build 550.90.07\n"
    torn = "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Fatal, Link 12 SAW_MVB"
    log = tmp_path / "fabricmanager.log"
    log.write_bytes((opening + torn).encode("utf-8"))
    state = tmp_path / "state.json"
    offset = len(opening.encode("utf-8")) + 10
    _write_fabric_file_state(state, log, offset=offset)
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

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        first = collector.collect_once()
        second = collector.collect_once()

    assert (first.delivered, second.delivered) == (0, 0), (
        f"an unfinished record was delivered: {sink.requests}"
    )
    assert caplog.text.count("has not finished writing") == 1, (
        f"the stalled read was reported {caplog.text.count('has not finished writing')}"
        f" time(s) instead of once: {caplog.text}"
    )
    from gpu_fault.collectors.logs.fabric_manager_receipts import identity_sha256

    assert (
        f"file_sha256={identity_sha256(str(log))}" in caplog.text
        and f"byte {offset}" in caplog.text
    ), "the warning must bind the exact stalled file and offset"
    assert str(log) not in caplog.text, "the warning must not expose the source path"
