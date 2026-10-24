from __future__ import annotations

import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals


@pytest.mark.parametrize(
    ("signatures", "collective", "expected"),
    [
        (["compute"] * 5, False, "UNDETERMINED"),
        (["collective"] * 5, True, "UNDETERMINED"),
        (["collective"] * 3 + ["other", "different"], True, "UNDETERMINED"),
        (["collective"] * 16 + ["other"] * 4, True, "UNDETERMINED"),
        (["collective"] * 8 + ["other"] * 2, True, "UNDETERMINED"),
    ],
)
def test_stack_consensus_distinguishes_collective_evidence_from_ambiguous_outliers(
    signatures, collective, expected
):
    signals = [
        {
            "rank": rank,
            "node_id": f"node-{rank}",
            "python_stack": {
                "signature": signature,
                "collective_frames": ["all_reduce"] if collective else [],
            },
            "proc": {"cpu_ticks_delta": 0},
            "gpu": {"utilization_gpu_percent": 0},
        }
        for rank, signature in enumerate(signatures)
    ]
    verdict = classify_hung_signals(signals, undetermined_nodes=["unreachable-node"])
    assert verdict["classification"] == expected, verdict
    assert verdict["undetermined_nodes"] == ["unreachable-node"], verdict
    assert verdict["root_cause_confirmed"] is False, verdict
    assert verdict["targeted_capture_eligible"] is False, verdict
    if expected == "WEAK":
        assert verdict["culprit_ranks"] == [
            rank for rank, signature in enumerate(signatures) if signature == "other"
        ], verdict
        assert verdict["control_ranks"] == [0], verdict
        assert verdict["confidence"] == "WEAK", verdict
    else:
        assert verdict["culprit_ranks"] == [], (
            "inconclusive stacks must not invent a rank-level collection target",
            verdict,
        )
