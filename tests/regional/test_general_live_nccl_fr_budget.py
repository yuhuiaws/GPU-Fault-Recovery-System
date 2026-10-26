"""GF-GENERAL-LIVE-011 ``fr_enabled``: a triage result the node lost, attributed.

Live (HyperPod, 2026-10-25 23:05Z, release d5d7610460e0) the eight-rank, two-dump
triage result of each node weighed ~1.2 MB against the Node Agent worker's 1 MiB
result line and came back as ``RESULT_SIZE_LIMIT``: the step's ``node_results``
was empty, both nodes sat under ``node_failures`` with ``ObservationIncomplete:
RESULT_SIZE_LIMIT``, and the oracle read "sampled no rank process" over sixteen
fresh dumps where the product reads. The gap is named, and a lost result is
told from a rank the triage failed to find or a dump it failed to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional import general_live_nccl as nc
from tests.regional.test_general_live_nccl import (
    NODE,
    PEER,
    SINCE,
    SPECS,
    FakeNcclJob,
    _ctx,
    _patch_job,
    hung_records,
)

__all__ = ["_patch_job"]


def test_fr_enabled_is_an_expected_fail_naming_the_result_budget_gap() -> None:
    """Eight ranks with two dumps each overflow the worker's 1 MiB result line.

    The dump-location gap before it was closed and proven live; the live run
    after that fix (HyperPod, 2026-10-25) lost both nodes' whole triage result
    to ``RESULT_SIZE_LIMIT`` while sixteen fresh dumps sat where the product
    reads. The gap names the cliff, the live record and the node-side fix.
    """

    spec = SPECS["fr_enabled"]
    assert spec.expected_outcome == "FAIL"
    for needle in (
        "observation_worker.py",
        "OBSERVATION_RESULT_BUDGET_BYTES",
        "1 MiB",
        "RESULT_SIZE_LIMIT",
        "ObservationIncomplete",
        "node_results = {}",
        "sampled no rank process",
        "triage_budget.py",
        "evidence_trimmed",
        "/tmp/nccl_trace_rank_*",
    ):
        assert needle in spec.product_gap, needle
    assert "dump_missing" not in spec.product_gap, "the location gap is closed"
    assert SPECS["fr_disabled"].expected_outcome == "PASS"
    assert SPECS["fr_disabled"].product_gap == ""


def _legacy_forensics(*pods: str) -> dict[str, Any]:
    """The trainer's account on the live HyperPod run: dumps where the product
    reads, requested twice per rank (PyTorch's own log lines)."""

    return {
        pod: {
            "home": "/root",
            "files": [
                {
                    "path": f"/tmp/torch_nccl_debug_pipe_{rank}.pipe",
                    "type": "fifo",
                    "size": 0,
                    "mtime": 1,
                },
                {
                    "path": f"/tmp/nccl_trace_rank_{rank}",
                    "type": "regular file",
                    "size": 29500,
                    "mtime": 1792969554,
                },
            ],
            "log_lines": [
                "[rank0]:[E1025 23:05:49.666318541 ProcessGroupNCCL.cpp:1575] [PG ID 0"
                " PG GUID 0(default_pg) Rank 0] ProcessGroupNCCL preparing to dump"
                " debug info. Include stack trace: 1",
                "[rank0]:[E1025 23:05:54.666791597 ProcessGroupNCCL.cpp:1575] [PG ID 0"
                " PG GUID 0(default_pg) Rank 0] ProcessGroupNCCL preparing to dump"
                " debug info. Include stack trace: 1",
            ],
            "error": None,
        }
        for rank, pod in enumerate(pods)
    }


def test_fr_errors_name_a_triage_result_the_executor_rejected(tmp_path: Path) -> None:
    """The live HyperPod record: ``node_results`` empty, both nodes under
    ``node_failures`` with ``ObservationIncomplete: RESULT_SIZE_LIMIT``. The miss
    is the lost result, not a rank the triage failed to find or a dump it
    failed to read."""

    cause = "ObservationIncomplete: RESULT_SIZE_LIMIT"
    lost = {NODE: cause, PEER: cause}
    records = hung_records(SINCE, triage_failures=lost)
    assert nc.triage_node_failures(records) == lost
    assert nc.fr_errors(records, expect_configured=True) == [
        "COLLECT_HUNG_TRIAGE sampled no rank process",
        f"{NODE}: no COLLECT_HUNG_TRIAGE result reached the record: {cause}",
        f"{PEER}: no COLLECT_HUNG_TRIAGE result reached the record: {cause}",
    ]
    assert nc.triage_node_failures(hung_records(SINCE)) == {}

    FakeNcclJob.forensics = _legacy_forensics("master", "worker")
    ctx, _calls = _ctx(SPECS["fr_enabled"], tmp_path, triage_failures=lost)
    result = SPECS["fr_enabled"].execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["fr_dumps"] == [
        "COLLECT_HUNG_TRIAGE sampled no rank process",
        f"{NODE}: no COLLECT_HUNG_TRIAGE result reached the record: {cause}",
        f"{PEER}: no COLLECT_HUNG_TRIAGE result reached the record: {cause}",
        "the trainer's 2 FR dump file(s) are under /tmp/nccl_trace_rank_* where the"
        " product reads and were requested (4 'preparing to dump debug info'"
        " line(s)); the product's triage result never reached the record --"
        f" {NODE}: {cause}; {PEER}: {cause}",
    ], result["stages"]["fr_dumps"]
