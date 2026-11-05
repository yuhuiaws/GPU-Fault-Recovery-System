"""GF-GENERAL-LIVE restart budget: every broken ledger invariant is named."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_restart_budget as rb
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from tests.regional._general_live_fakes import budget_ledger_store, make_context
from tests.regional.test_general_live_restart_budget import (
    RecordingRegional,
    concurrent_document,
    missing_document,
)


def concurrent_errors(mutate: Any) -> list[str]:
    _, document = concurrent_document()
    assert rb.concurrent_claim_errors(document) == [], "the fixture passes as built"
    broken = copy.deepcopy(document)
    mutate(broken)
    return rb.concurrent_claim_errors(broken)


def missing_errors(mutate: Any) -> list[str]:
    document = missing_document()
    assert rb.missing_context_errors(document) == [], "the fixture passes as built"
    broken = copy.deepcopy(document)
    mutate(broken)
    return rb.missing_context_errors(broken)


def set_path(document: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    *parents, last = path
    for key in parents:
        document = document[key]
    document[last] = value


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (
            ("phases", "claim", "ledger"),
            None,
            "after the concurrent claim: the test job has no restart ledger",
        ),
        (
            ("phases", "claim", "results", 1),
            {"index": 1, "reserved": False},
            "claimer 1 was refused the one restart",
        ),
        (
            ("phases", "claim", "results", 2),
            {"index": 2, "reserved": True, "restart_count": 2, "reservation_ids": []},
            "claimer 2 saw 2 [] (a double deduction or a stranger)",
        ),
        (
            ("phases", "rival", "results", 0),
            {"index": 0, "error": "PoolTimeout: no connection"},
            "rival claimer 0 failed: PoolTimeout: no connection",
        ),
        (
            ("phases", "release", "results", 0),
            {"index": 0, "error": "OperationalError: gone"},
            "releaser 0 failed: OperationalError: gone",
        ),
        (
            ("phases", "false_release", "results"),
            [{"index": 5, "error": "LockNotAvailable: busy"}],
            "false release 5 failed: LockNotAvailable: busy",
        ),
        (
            ("phases", "false_release", "results"),
            [{"index": 6, "restart_count": -1}],
            "false_release: restart_count went negative",
        ),
    ],
)
def test_concurrent_claim_errors_name_each_broken_result(
    path: tuple[Any, ...], value: Any, expected: str
) -> None:
    errors = concurrent_errors(lambda document: set_path(document, path, value))

    assert expected in errors, errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (
            ("preflight", "status"),
            "SUCCEEDED",
            "the preflight refusal is SUCCEEDED, not FAILED",
        ),
        (
            ("preflight", "missing_parameters"),
            ["source_gpu_count", "source_node_ids"],
            "the refusal does not name exactly the omitted parameters",
        ),
        (
            ("ledger", "before"),
            {"restart_count": 0, "reservation_ids": []},
            "the test job already had a restart ledger before the probe",
        ),
        (
            ("ledger", "after_gate"),
            {"restart_count": 0, "reservation_ids": []},
            "the adapter gate wrote a restart ledger row",
        ),
        (
            ("control", "preflight"),
            {"reason": "RESTART_BUDGET_EXHAUSTED"},
            "the complete context was refused: {'reason': 'RESTART_BUDGET_EXHAUSTED'}",
        ),
    ],
)
def test_missing_context_errors_name_each_broken_phase(
    path: tuple[Any, ...], value: Any, expected: str
) -> None:
    errors = missing_errors(lambda document: set_path(document, path, value))

    assert expected in errors, errors


def test_an_authorizing_gate_for_a_context_less_step_is_named() -> None:
    errors = missing_errors(
        lambda document: document.update(
            gate={"authorized": True, "reason": rb.CONTEXT_MISSING}
        )
    )

    assert errors == [
        f"the adapter gate answered {rb.CONTEXT_MISSING} for a step without "
        f"context, expected {rb.CONTEXT_MISSING}"
    ]


def test_a_second_write_reuses_the_registered_cleanup(tmp_path: Path) -> None:
    regional = RecordingRegional(budget_ledger_store())
    spec = {item.variant: item for item in att.VARIANTS}["concurrent_claim"]
    ctx = make_context(spec, tmp_path=tmp_path, regional=regional, probes={})
    budget = rb.SeededRestartBudget(ctx, tag="cc")
    budget.concurrent()

    # The row the race left is this run's own residue: the probe refuses a
    # second seeding of it, and the one registered cleanup still removes it.
    with pytest.raises(VariantNotRun, match="the cleanup removes it") as caught:
        budget.missing_context("source_node_ids")

    assert caught.value.reason == "FIXTURE_NOT_READY"
    assert ctx.cleanup.pending() == [budget.cleanup_name]
    assert [arguments[0] for arguments, _ in regional.calls] == [
        "concurrent",
        "missing-context",
    ]
    assert regional.calls[1][0][-1] == "source_node_ids"
