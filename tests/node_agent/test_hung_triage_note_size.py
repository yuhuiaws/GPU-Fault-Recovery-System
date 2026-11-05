"""The trimming note weighs itself: ``bytes_after`` is the size of what is sent.

``bounded_triage_result`` used to write ``bytes_after`` and ``within_budget``
from the document measured while its ``evidence_trimmed`` note still held the
size before the note was added. When writing the measured size added a digit,
the document sent was one byte heavier than its note said: the sparse four-row
document of ``test_cov95_triage_skeleton_shapes`` under a 2,048-byte budget
recorded 1,197 bytes and weighed 1,198. A document within a byte or two of the
budget could say ``within_budget: true`` while over it, or flip to ``false``
and grow by the byte that word adds. The ladder also stopped at a step after
which the document fitted only without its note.

The size checked here is the encoding the worker writes and the control plane's
codec weighs -- compact JSON, ``json.dumps(..., separators=(",", ":"))`` --
measured by the test itself, not by ``triage_budget.encoded_size``.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault.node_agent.operations.triage_budget import (
    SKELETON_STEP,
    bounded_triage_result,
)

STACK_STEP = "stack_series.samples:last"


def sent_bytes(document: dict[str, Any]) -> int:
    """The compact JSON the worker writes and the control plane's codec weighs."""

    return len(json.dumps(document, separators=(",", ":")).encode())


def node_result(*, padding: int, stack_samples: int = 0) -> dict[str, Any]:
    """One rank whose 4 KiB stack text only the floor drops, beside a node-level
    field of ``padding`` bytes that no step touches. ``stack_samples`` 1 KiB
    samples give ``stack_series.samples:last`` something to cut first."""

    row: dict[str, Any] = {
        "node_id": "node-a",
        "pid": 4001,
        "rank": 0,
        "status": "OK",
        "python_stack": {"status": "COLLECTED", "threads": ["x" * 4096]},
    }
    if stack_samples:
        row["stack_series"] = {
            "version": "stack-series/v1",
            "samples": [{"threads": ["s" * 1024]} for _ in range(stack_samples)],
        }
    return {
        "status": "COMPLETE",
        "node_id": "node-a",
        "padding": "p" * padding,
        "ranks": [row],
    }


@pytest.mark.parametrize("boundary", [1_000, 10_000, 100_000])
def test_bytes_after_is_the_size_sent_on_both_sides_of_a_digit_boundary(
    boundary: int,
) -> None:
    """Every padding from six bytes under the boundary to six over: the note's
    size gains its digit inside the sweep and is still the size sent."""

    budget = boundary + 2_048
    # Near the boundary every number in the note already has the digit count
    # the sweep starts with, so one byte of padding is one byte of document.
    probe_padding = boundary - 400
    probe = bounded_triage_result(node_result(padding=probe_padding), budget)
    start = probe_padding + boundary - sent_bytes(probe) - 6

    sizes = []
    for padding in range(start, start + 12):
        bounded = bounded_triage_result(node_result(padding=padding), budget)

        note = bounded["evidence_trimmed"]
        size = sent_bytes(bounded)
        assert [step["step"] for step in note["steps"]] == [SKELETON_STEP], note
        assert note["bytes_after"] == size, (padding, size, note)
        assert note["within_budget"] is True, note
        sizes.append(size)

    assert min(sizes) < boundary <= max(sizes), (
        f"the sweep must cross {boundary}: {sizes}"
    )


def test_a_document_over_the_budget_by_its_own_note_says_so_with_the_size_sent():
    """At the floor nothing is left to cut: a document over the budget only by
    its note is returned as it is, and the note names its real size (``false``
    is a byte longer than ``true``)."""

    document = node_result(padding=2_000)
    fitted = bounded_triage_result(copy.deepcopy(document), 3_000)
    size = sent_bytes(fitted)
    assert fitted["evidence_trimmed"]["within_budget"] is True, fitted
    assert len(str(size)) == len(str(3_000)), (
        f"every budget below has as many digits as the probe's: {size}"
    )

    exact = bounded_triage_result(copy.deepcopy(document), size)
    over = bounded_triage_result(copy.deepcopy(document), size - 1)

    assert exact["evidence_trimmed"]["bytes_after"] == sent_bytes(exact) == size
    assert exact["evidence_trimmed"]["within_budget"] is True, exact
    note = over["evidence_trimmed"]
    assert [step["step"] for step in note["steps"]] == [SKELETON_STEP], note
    assert note["within_budget"] is False, note
    assert note["bytes_after"] == sent_bytes(over) == size + 1, (size, note)
    assert note["steps"][-1]["bytes"] <= size - 1, (
        "the floor fitted without its note: only the note is over"
    )


def test_a_step_that_fits_only_without_its_note_does_not_stop_the_ladder() -> None:
    """The ladder stops at the first step after which the document fits with
    its note: a budget one byte short of that takes the next step."""

    document = node_result(padding=2_000, stack_samples=3)
    generous = bounded_triage_result(copy.deepcopy(document), 8_000)
    size = sent_bytes(generous)
    assert [step["step"] for step in generous["evidence_trimmed"]["steps"]] == [
        STACK_STEP
    ], generous["evidence_trimmed"]
    assert len(str(size - 1)) == len(str(8_000)), (
        f"every budget below has as many digits as the probe's: {size}"
    )

    at_line = bounded_triage_result(copy.deepcopy(document), size)
    short = bounded_triage_result(copy.deepcopy(document), size - 1)

    note = at_line["evidence_trimmed"]
    assert [step["step"] for step in note["steps"]] == [STACK_STEP], note
    assert note["within_budget"] is True, note
    assert note["bytes_after"] == sent_bytes(at_line) == size, note
    note = short["evidence_trimmed"]
    assert [step["step"] for step in note["steps"]] == [STACK_STEP, SKELETON_STEP], (
        "the stack step fitted only without its note, so the floor follows"
    )
    assert note["steps"][0]["bytes"] <= size - 1, note
    assert note["within_budget"] is True, note
    assert note["bytes_after"] == sent_bytes(short) <= size - 1, note
