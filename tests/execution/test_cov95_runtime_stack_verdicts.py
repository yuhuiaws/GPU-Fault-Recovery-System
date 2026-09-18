from __future__ import annotations

import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals


@pytest.mark.parametrize(
    ("signatures", "collective", "expected"),
    [
        (["compute"] * 5, False, "NOT_COLLECTIVE_HANG"),
        (["collective"] * 5, True, "UNDETERMINED"),
        (["collective"] * 3 + ["other", "different"], True, "UNDETERMINED"),
        (["collective"] * 16 + ["other"] * 4, True, "UNDETERMINED"),
        (["collective"] * 8 + ["other"] * 2, True, "PLAUSIBLE"),
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
    if expected == "PLAUSIBLE":
        assert verdict["culprit_ranks"] == [8, 9], verdict
        assert verdict["control_ranks"] == [0], verdict
        assert verdict["confidence"] == "PLAUSIBLE", verdict
    else:
        assert verdict["culprit_ranks"] == [], (
            "inconclusive stacks must not invent a rank-level collection target",
            verdict,
        )
