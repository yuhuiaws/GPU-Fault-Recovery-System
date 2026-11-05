"""GF-GENERAL-LIVE memory: a failed contributor must reach a delivered batch."""

from __future__ import annotations

from scripts.e2e.regional import general_live_memory as memory


def test_a_contributor_failure_must_reach_a_delivered_batch() -> None:
    observed = {
        "batches": [
            {"collection_errors": ["meminfo: PermissionError"]},
            {"collection_errors": ["swap: FileNotFoundError: /proc/swaps"]},
        ]
    }

    assert memory.contributor_error_errors(observed, "swap") == []
    assert memory.contributor_error_errors(observed, "pressure") == [
        "no delivered batch carries a pressure collection error"
    ]
