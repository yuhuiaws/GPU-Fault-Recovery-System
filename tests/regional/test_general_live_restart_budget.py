"""GF-GENERAL-LIVE-029 store-level restart-budget harness over the product's ledger.

The budget probe runs in-process over an in-memory store: the concurrent claim,
the rival step, the raced release, the context-less preflight refusal and the
cleanup all drive the real ``reserve_job_restart``/``release_job_restart`` and
``restart_budget_preflight`` code, so the verdict helpers are exercised on the
documents the product actually produces, then on scripted deviations.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.store import NotFoundError
from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_restart_budget as rb
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from tests.regional._general_live_fakes import (
    FakeRegional,
    budget_ledger_store,
    budget_probe_run,
    make_context,
)

CLUSTER = "gpu-eks-cluster"
JOB = "gl29cc-abcdef12-a1"
WORKFLOW = f"{JOB}-workflow"
FIRST = f"{WORKFLOW}/0/RESTART_WORKLOAD"
MC_JOB = "gl29mc-abcdef12-a1"
MC_WORKFLOW = f"{MC_JOB}-workflow"


def concurrent_document(
    store: Any | None = None, claimers: str = "4"
) -> tuple[Any, dict[str, Any]]:
    store = store or budget_ledger_store()
    document = budget_probe_run(
        store, ["concurrent", CLUSTER, JOB, "1", WORKFLOW, claimers]
    )
    return store, document


def missing_document(missing: str = "source_gpu_count") -> dict[str, Any]:
    return budget_probe_run(
        budget_ledger_store(),
        [
            "missing-context",
            CLUSTER,
            MC_JOB,
            "1",
            MC_WORKFLOW,
            f"{MC_JOB}-a001",
            missing,
        ],
    )


class RecordingRegional(FakeRegional):
    """A FakeRegional whose cpu_python runs the budget probe over one store."""

    def __init__(self, store: Any) -> None:
        super().__init__()
        self.store = store
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def cpu_python(
        self, script: str, *arguments: str, **options: Any
    ) -> dict[str, Any]:
        self.calls.append((arguments, options))
        assert script == rb.BUDGET_PROBE, "the harness ships the budget probe"
        return budget_probe_run(self.store, arguments)


# --------------------------------------------------------------------------- #
# the probe over the product's ledger API
# --------------------------------------------------------------------------- #
def test_concurrent_probe_holds_the_ledger_invariants() -> None:
    store, document = concurrent_document()
    assert rb.concurrent_claim_errors(document) == [], document
    claim = document["phases"]["claim"]
    assert [item["reserved"] for item in claim["results"]] == [True] * 4
    assert claim["ledger"]["restart_count"] == 1
    assert claim["ledger"]["reservation_ids"] == [FIRST]
    assert all(
        item["reserved"] is False for item in document["phases"]["rival"]["results"]
    ), "all((item['reserved'] is False for item in document['phases']['rival']"
    assert document["phases"]["rival"]["ledger"]["reservation_ids"] == [FIRST]
    assert document["phases"]["release"]["ledger"]["restart_count"] == 0
    assert document["phases"]["false_release"]["ledger"]["reservation_ids"] == []
    state = store.get_restart_budget(CLUSTER, JOB)
    assert state.restart_count == 0 and state.reservation_ids == []


def test_cleanup_releases_this_runs_reservations_then_removes_the_row() -> None:
    store = budget_ledger_store()
    store.reserve_job_restart(CLUSTER, JOB, 1, FIRST)
    document = budget_probe_run(store, ["cleanup", CLUSTER, JOB, WORKFLOW])
    assert document["released"] == [FIRST]
    assert document["deleted"] is True and document["clean"] is True
    assert rb.cleanup_errors(document) == []
    with pytest.raises(NotFoundError):
        store.get_restart_budget(CLUSTER, JOB)
    again = budget_probe_run(store, ["cleanup", CLUSTER, JOB, WORKFLOW])
    assert again["clean"] is True and again["released"] == []


def test_cleanup_keeps_a_row_holding_a_strangers_reservation() -> None:
    store = budget_ledger_store()
    stranger = "other-workflow/0/RESTART_WORKLOAD"
    store.reserve_job_restart(CLUSTER, JOB, 1, stranger)
    document = budget_probe_run(store, ["cleanup", CLUSTER, JOB, WORKFLOW])
    assert document["released"] == [] and document["deleted"] is False
    assert document["clean"] is False
    assert rb.cleanup_errors(document), "a remaining row is a cleanup error"
    assert store.get_restart_budget(CLUSTER, JOB).reservation_ids == [stranger]


def test_missing_context_is_refused_before_any_ledger_write() -> None:
    document = missing_document()
    assert rb.missing_context_errors(document) == [], document
    preflight = document["preflight"]
    assert preflight["reason"] == rb.CONTEXT_MISSING and preflight["status"] == "FAILED"
    assert preflight["missing_parameters"] == ["source_gpu_count"]
    assert "source_gpu_count" in preflight["error"]
    assert document["ledger"]["before"] is None
    assert document["ledger"]["after_preflight"] is None
    assert document["ledger"]["after_gate"] is None
    assert document["gate"]["authorized"] is False
    assert document["gate"]["reason"] == rb.CONTEXT_MISSING
    assert document["unreserved"]["reason"] == rb.RESERVATION_MISSING
    assert document["control"]["preflight"] is None
    assert document["control"]["gate"]["authorized"] is True
    assert document["ledger"]["after_control"]["reservation_ids"] == [
        document["reservation_id"]
    ]
    assert document["reservation_id"] == f"{MC_WORKFLOW}/0/RESTART_WORKLOAD"


def test_missing_context_names_every_omitted_parameter() -> None:
    document = missing_document("source_gpu_count,restart_budget")
    assert document["preflight"]["missing_parameters"] == [
        "restart_budget",
        "source_gpu_count",
    ]
    assert rb.missing_context_errors(document) == []


def test_probe_refuses_a_non_test_job_a_preexisting_row_and_bad_claimers() -> None:
    store = budget_ledger_store()
    with pytest.raises(ValueError, match="test prefix"):
        budget_probe_run(
            store, ["concurrent", CLUSTER, "training-job-7", "1", "wf", "4"]
        )
    with pytest.raises(ValueError, match="test prefix"):
        budget_probe_run(store, ["cleanup", CLUSTER, "training-job-7", "wf"])
    with pytest.raises(ValueError, match="2..8"):
        budget_probe_run(store, ["concurrent", CLUSTER, JOB, "1", WORKFLOW, "1"])
    store.reserve_job_restart(CLUSTER, JOB, 1, "stale/0/RESTART_WORKLOAD")
    document = budget_probe_run(store, ["concurrent", CLUSTER, JOB, "1", WORKFLOW, "4"])
    assert document["refused"].startswith("preexisting ledger"), (
        "document['refused'].startswith('preexisting ledger')"
    )
    assert rb.concurrent_claim_errors(document) == [
        "the probe refused: " + document["refused"]
    ]
    missing = budget_probe_run(
        store, ["missing-context", CLUSTER, JOB, "1", WORKFLOW, "a", "source_gpu_count"]
    )
    assert rb.missing_context_errors(missing) == [
        "the probe refused: " + missing["refused"]
    ]
    assert store.get_restart_budget(CLUSTER, JOB).restart_count == 1, (
        "a refused probe writes nothing"
    )
    read = budget_probe_run(store, ["read", CLUSTER, JOB])
    assert read["ledger"]["reservation_ids"] == ["stale/0/RESTART_WORKLOAD"]


# --------------------------------------------------------------------------- #
# verdict helpers on scripted deviations
# --------------------------------------------------------------------------- #
def test_concurrent_claim_errors_flag_each_broken_invariant() -> None:
    _, good = concurrent_document()
    doubled = copy.deepcopy(good)
    doubled["phases"]["claim"]["ledger"] = {
        "restart_count": 2,
        "reservation_ids": [FIRST, "x"],
    }
    assert any("expected 1" in item for item in rb.concurrent_claim_errors(doubled)), (
        "any(('expected 1' in item for item in rb.concurrent_claim_errors(doubl"
    )
    granted = copy.deepcopy(good)
    granted["phases"]["rival"]["results"][0]["reserved"] = True
    assert any(
        "past the budget" in item for item in rb.concurrent_claim_errors(granted)
    ), "any(('past the budget' in item for item in rb.concurrent_claim_errors("
    leaked = copy.deepcopy(good)
    leaked["phases"]["release"]["results"][1]["restart_count"] = 1
    assert any(
        "after the release" in item for item in rb.concurrent_claim_errors(leaked)
    ), "any(('after the release' in item for item in rb.concurrent_claim_error"
    negative = copy.deepcopy(good)
    negative["phases"]["false_release"]["ledger"] = {
        "restart_count": -1,
        "reservation_ids": [],
    }
    assert any("negative" in item for item in rb.concurrent_claim_errors(negative)), (
        "any(('negative' in item for item in rb.concurrent_claim_errors(negativ"
    )
    failed = copy.deepcopy(good)
    failed["phases"]["claim"]["results"][0] = {"index": 0, "error": "PoolTimeout: x"}
    assert any("PoolTimeout" in item for item in rb.concurrent_claim_errors(failed)), (
        "any(('PoolTimeout' in item for item in rb.concurrent_claim_errors(fail"
    )
    short = copy.deepcopy(good)
    short["claimers"] = 1
    assert any(">= 2" in item for item in rb.concurrent_claim_errors(short)), (
        "any(('>= 2' in item for item in rb.concurrent_claim_errors(short)))"
    )


def test_missing_context_errors_flag_a_wrong_reason_or_a_ledger_write() -> None:
    good = missing_document()
    wrong = copy.deepcopy(good)
    wrong["preflight"]["reason"] = "RESTART_BUDGET_EXHAUSTED"
    assert any(
        rb.CONTEXT_MISSING in item for item in rb.missing_context_errors(wrong)
    ), "any((rb.CONTEXT_MISSING in item for item in rb.missing_context_errors("
    admitted = copy.deepcopy(good)
    admitted["preflight"] = None
    assert any("admitted" in item for item in rb.missing_context_errors(admitted)), (
        "any(('admitted' in item for item in rb.missing_context_errors(admitted"
    )
    wrote = copy.deepcopy(good)
    wrote["ledger"]["after_preflight"] = {"restart_count": 1, "reservation_ids": ["r"]}
    assert any(
        "wrote a restart ledger" in item for item in rb.missing_context_errors(wrote)
    ), "any(('wrote a restart ledger' in item for item in rb.missing_context_e"
    open_gate = copy.deepcopy(good)
    open_gate["unreserved"] = {"authorized": True}
    assert any(
        "fail closed" in item for item in rb.missing_context_errors(open_gate)
    ), "any(('fail closed' in item for item in rb.missing_context_errors(open_"
    no_control = copy.deepcopy(good)
    no_control["control"]["gate"] = {"authorized": False, "reason": "X"}
    assert any(
        "control" in item or "authorized" in item
        for item in rb.missing_context_errors(no_control)
    ), "any(('control' in item or 'authorized' in item for item in rb.missing_"


# --------------------------------------------------------------------------- #
# the helper class
# --------------------------------------------------------------------------- #
def test_seeded_budget_registers_cleanup_once_and_never_retries_writes(
    tmp_path: Path,
) -> None:
    store = budget_ledger_store()
    regional = RecordingRegional(store)
    spec = {item.variant: item for item in att.VARIANTS}["concurrent_claim"]
    ctx = make_context(spec, tmp_path=tmp_path, regional=regional, probes={})
    budget = rb.SeededRestartBudget(ctx, tag="cc")
    assert budget.job_id.startswith("gl29cc-") and budget.job_id.endswith("-a1")
    assert budget.cluster_id == CLUSTER
    document = budget.concurrent()
    assert rb.concurrent_claim_errors(document) == []
    assert ctx.cleanup.pending() == [budget.cleanup_name]
    budget.read()
    assert ctx.cleanup.pending() == [budget.cleanup_name], "registered once"
    cleaned = budget.cleanup()
    assert cleaned["clean"] is True
    writes = [
        options for arguments, options in regional.calls if arguments[0] != "read"
    ]
    assert writes and all(options.get("attempts") == 1 for options in writes), writes
    assert (ctx.case_dir / "budget-concurrent.json").is_file(), "evidence written"
    assert (ctx.case_dir / "budget-cleanup.json").is_file(), "cleanup receipt written"
    with pytest.raises(ValueError, match="alphanumeric"):
        rb.SeededRestartBudget(ctx, tag="c-c")


def test_seeded_budget_refusal_is_not_run_and_keeps_the_cleanup(tmp_path: Path) -> None:
    store = budget_ledger_store()
    regional = RecordingRegional(store)
    spec = {item.variant: item for item in att.VARIANTS}["missing_context"]
    ctx = make_context(spec, tmp_path=tmp_path, regional=regional, probes={})
    budget = rb.SeededRestartBudget(ctx, tag="mc")
    store.reserve_job_restart(
        CLUSTER, budget.job_id, 1, f"{budget.workflow_id}/0/RESTART_WORKLOAD"
    )
    with pytest.raises(VariantNotRun, match="preexisting ledger") as caught:
        budget.missing_context()
    assert caught.value.reason == "FIXTURE_NOT_READY"
    assert ctx.cleanup.pending() == [budget.cleanup_name]
    ctx.cleanup.run()
    assert ctx.cleanup.failed == []
    with pytest.raises(NotFoundError):
        store.get_restart_budget(CLUSTER, budget.job_id)
