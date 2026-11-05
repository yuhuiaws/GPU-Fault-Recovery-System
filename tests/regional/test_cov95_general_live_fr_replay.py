"""GF-GENERAL-LIVE Flight Recorder replay: the responder source and ledger oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_fr_replay as replay

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
NODES = (NODE, PEER)


def ledger(ranks: list[int], **rank_fields: Any) -> dict[str, Any]:
    return {
        "version": replay.LEDGER_VERSION,
        "status": "serving",
        "errors": [],
        "pipe_base": replay.PIPE_BASE,
        "ranks": {
            str(rank): {
                "pipe": f"{replay.PIPE_BASE}{rank}.pipe",
                "requests": 2,
                "events": [],
                **rank_fields,
            }
            for rank in ranks
        },
    }


def test_a_responder_that_would_end_its_own_heredoc_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "responder.py"
    script.write_text(f"print('x')\n{replay.HEREDOC}\nprint('y')\n", encoding="utf-8")
    monkeypatch.setattr(replay, "RESPONDER_SCRIPT", script)

    with pytest.raises(ValueError, match="the responder would end its own heredoc"):
        replay.responder_source()


def test_a_ledger_serving_no_attempt_nodes_rank_block_belongs_to_none() -> None:
    blocks = replay.node_ranks(NODES)

    assert replay.ledger_node(ledger(blocks[NODE]), NODES) == NODE
    assert replay.ledger_node(ledger([*blocks[NODE], 99]), NODES) is None


def test_responder_errors_name_a_foreign_rank_block_and_failed_dump_writes() -> None:
    blocks = replay.node_ranks(NODES)
    failed = [{"action": "failed", "error": "ENOSPC"}]
    ledgers = {"pod-a": ledger(blocks[NODE], events=failed), "pod-x": ledger([99])}

    errors = replay.responder_errors(ledgers, NODES)

    first = blocks[NODE][0]
    assert f"pod-a: rank {first} dump writes failed: {failed}" in errors, errors
    assert "pod-x: the responder serves ranks of no attempt node" in errors, errors
    assert f"{PEER}: no responder ledger for its Pod" in errors, errors


def test_request_errors_skip_what_is_not_a_ledger() -> None:
    blocks = replay.node_ranks(NODES)
    ledgers = {"pod-a": "unreadable", "pod-b": ledger(blocks[PEER], requests=1)}

    assert replay.request_errors(ledgers, NODES, sampled_nodes=[PEER]) == [
        f"{PEER}: rank {rank} received 1 dump request(s), the product's triage "
        "should have asked 2"
        for rank in sorted(map(str, blocks[PEER]))
    ]


def test_answered_watermarks_keep_only_answered_dumps_per_rank() -> None:
    events = [
        {"action": "answered", "watermarks": {"seq": 1}},
        {"action": "failed", "watermarks": {"seq": 2}},
        {"action": "answered", "watermarks": "not-a-record"},
        {"action": "answered", "watermarks": {"seq": 3}},
    ]
    ledgers = {"pod-a": None, "pod-b": ledger([4, 5], events=events)}

    assert replay.answered_watermarks(ledgers) == {
        4: [{"seq": 1}, {"seq": 3}],
        5: [{"seq": 1}, {"seq": 3}],
    }
    assert replay.answered_watermarks({"pod-c": {"ranks": {}}}) == {}
