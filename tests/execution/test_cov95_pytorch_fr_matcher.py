"""``gpu_fault.pytorch_fr_matcher``: the collective branch of PyTorch 2.10's
``Op.match``, one predicate at a time.

``tools/verify_fr_analyzer.py`` checks the adapted rules against the real
PyTorch matcher where torch is installed; these tests pin each rule's verdict
without torch, in upstream's order: the first predicate that fails decides.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from gpu_fault.pytorch_fr_matcher import COLLECTIVES, match_collective, valid_entry


def _entry(kind: str, **fields: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "profiling_name": f"nccl:{kind}",
        "input_sizes": [[4]],
        "output_sizes": [[4]],
        "input_dtypes": ["Float"],
        "output_dtypes": ["Float"],
        "state": "completed",
    }
    values.update(fields)
    return values


@pytest.mark.parametrize(
    "shapes",
    [[[4], "4"], [[1] * 33], [[4.0]], [[True]], [[-1]], [[2**63]]],
    ids=["not-a-list", "too-many-dims", "float", "bool", "negative", "overflow"],
)
def test_a_malformed_tensor_shape_is_not_a_valid_entry(shapes: list[Any]) -> None:
    for field in ("input_sizes", "output_sizes"):
        entry = _entry("all_reduce", **{field: shapes})
        assert valid_entry(entry) is False, (field, shapes)
        assert match_collective(entry, _entry("all_reduce"), 2) == "UNDECIDED", (
            "an entry that cannot be read is never a verdict either way"
        )


@pytest.mark.parametrize(
    "profiling_name", ["", "all_reduce", "gloo:all_reduce", "nccl:all_reduce:extra"]
)
def test_a_profiling_name_outside_nccl_and_xccl_is_undecided(
    profiling_name: str,
) -> None:
    left = _entry("all_reduce", profiling_name=profiling_name)
    assert match_collective(left, _entry("all_reduce"), 2) == "UNDECIDED"
    assert match_collective(_entry("all_reduce"), left, 2) == "UNDECIDED"


def test_an_operation_the_rules_do_not_know_is_undecided() -> None:
    assert match_collective(_entry("send"), _entry("send"), 2) == "UNDECIDED"
    assert match_collective(_entry("all_reduce"), _entry("barrier"), 2) == "UNDECIDED"
    assert (
        match_collective(
            _entry("all_reduce", profiling_name="xccl:all_reduce"),
            _entry("all_reduce"),
            2,
        )
        == "FULLY_MATCHED"
    ), "xccl names the same collectives as nccl"


def test_a_coalesced_marker_matches_only_another_coalesced_marker() -> None:
    marker = _entry("coalesced", input_sizes=None)
    assert match_collective(marker, copy.deepcopy(marker), 2) == "FULLY_MATCHED", (
        "a coalesced marker carries no sizes to compare"
    )
    assert (
        match_collective(marker, _entry("all_reduce"), 2) == "SIZE_OR_SYNTAX_MISMATCH"
    )


@pytest.mark.parametrize(
    ("kind", "left", "right", "size", "verdict"),
    [
        ("broadcast", {}, {"output_sizes": [[7]]}, 2, "SIZE_OR_SYNTAX_MISMATCH"),
        ("gather", {}, {"output_sizes": [[7]]}, 2, "FULLY_MATCHED"),
        (
            "all_reduce",
            {"output_sizes": [[8]]},
            {"output_sizes": [[8]]},
            2,
            "SIZE_OR_SYNTAX_MISMATCH",
        ),
        ("all_gather", {}, {}, 2, "SIZE_OR_SYNTAX_MISMATCH"),
        (
            "all_gather",
            {"output_sizes": [[8]]},
            {"output_sizes": [[8]]},
            2,
            "FULLY_MATCHED",
        ),
        ("reduce_scatter", {}, {}, 2, "SIZE_OR_SYNTAX_MISMATCH"),
        (
            "reduce_scatter",
            {"input_sizes": [[8]]},
            {"input_sizes": [[8]]},
            2,
            "FULLY_MATCHED",
        ),
        (
            "all_reduce",
            {},
            {"input_dtypes": ["Half"], "output_dtypes": ["Half"]},
            2,
            "COLLECTIVE_DTYPE_MISMATCH",
        ),
        ("all_reduce", {}, {"state": "started"}, 2, "COLLECTIVE_STATE_MISMATCH"),
        ("all_to_all", {}, {"input_sizes": [[7]]}, 2, "UNDECIDED"),
        ("scatter", {}, {"input_sizes": [[7]]}, 2, "FULLY_MATCHED"),
    ],
    ids=[
        "outputs-differ",
        "gather-ignores-outputs",
        "all-reduce-input-is-not-its-output",
        "all-gather-output-is-not-input-times-size",
        "all-gather-output-is-input-times-size",
        "reduce-scatter-input-is-not-output-times-size",
        "reduce-scatter-input-is-output-times-size",
        "dtypes-differ",
        "states-differ",
        "all-to-all-is-never-decided",
        "scatter-ignores-inputs",
    ],
)
def test_each_collective_rule_decides_in_upstream_order(
    kind: str, left: dict[str, Any], right: dict[str, Any], size: int, verdict: str
) -> None:
    assert kind in COLLECTIVES
    assert (
        match_collective(_entry(kind, **left), _entry(kind, **right), size) == verdict
    )


@pytest.mark.parametrize("size", [0, 65537])
def test_a_group_size_outside_the_bound_is_undecided(size: int) -> None:
    assert match_collective(_entry("all_reduce"), _entry("all_reduce"), size) == (
        "UNDECIDED"
    )
