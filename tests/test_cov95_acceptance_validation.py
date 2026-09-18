from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tests._cov95_acceptance_support import (
    analysis,
    backend_with_output,
    manual_case,
    review,
)
from tools.codex_acceptance import InvalidAcceptanceInput, InvalidCodexOutput


def replace_field(document: Any, path: tuple[Any, ...], value: Any) -> None:
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


@pytest.mark.parametrize(
    "path,value,problem",
    [
        (("schema_version",), 2, "schema_version"),
        (("results",), {}, "JSON array"),
        (("results",), [], "result count"),
        (("results", 0), [], "JSON object"),
        (("results", 0, "case_id"), "unknown", "unknown case_id"),
        (("results", 1, "case_id"), "case-a", "duplicates case_id"),
        (("results", 0, "status"), "PARTIAL", "unsupported status"),
        (("results", 0, "summary"), " ", "non-empty string"),
        (("results", 0, "failure_details"), "bad", "JSON array"),
        (("results", 0, "evidence"), {}, "JSON array"),
        (("results", 0, "evidence"), [{}], "fields do not match"),
        (("results", 0, "evidence"), [], "PASS requires evidence"),
        (("results", 0, "evidence", 0, "source"), "", "non-empty string"),
        (("results", 0, "test_process"), [], "non-empty JSON array"),
        (("results", 0, "test_process"), {}, "non-empty JSON array"),
        (("results", 0, "test_process", 0, "step_id"), "bad_id", "step_id is invalid"),
        (
            ("results", 0, "test_process", 0, "kind"),
            "execute-anything",
            "kind is unsupported",
        ),
        (
            ("results", 0, "test_process", 0, "depends_on"),
            ["inspect"],
            "depends on itself",
        ),
        (
            ("results", 0, "test_process", 0, "depends_on"),
            ["missing", "missing"],
            "contains duplicates",
        ),
        (
            ("results", 0, "test_process", 0, "depends_on"),
            ["missing"],
            "unknown dependencies",
        ),
        (
            ("results", 0, "test_process", 0, "executor_ref"),
            "shell",
            "executor_ref must be",
        ),
        (
            ("results", 0, "test_process", 0, "status"),
            "UNVERIFIED",
            "status is unsupported",
        ),
        (("results", 0, "test_process", 0, "evidence"), [], "PASS requires evidence"),
        (("results", 0, "test_process", 0, "expected"), "", "non-empty string"),
        (
            ("results", 0, "affected_dependents"),
            ["case-b", "case-b"],
            "contains duplicates",
        ),
        (("results", 0, "affected_dependents"), ["case-a"], "own case_id"),
        (("results", 0, "affected_dependents"), ["other"], "unknown case IDs"),
        (("results", 0, "blockers"), ["missing observation"], "PASS cannot include"),
        (("results", 0, "human_actions"), ["manual action"], "PASS cannot include"),
    ],
)
def test_analysis_refuses_unbound_or_incomplete_structured_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: tuple[Any, ...],
    value: Any,
    problem: str,
) -> None:
    payload = {"schema_version": 1, "results": [analysis("case-a"), analysis("case-b")]}
    replace_field(payload, path, value)
    instance, transport = backend_with_output(monkeypatch, tmp_path, payload)
    with pytest.raises(InvalidCodexOutput, match=problem):
        instance.analyze_manual_cases([manual_case("case-a"), manual_case("case-b")])
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "updates,problem",
    [
        ({"status": "FAIL"}, "FAIL requires failure_details"),
        (
            {"status": "FAIL", "failure_details": ["mismatch"]},
            "FAIL requires reproduction",
        ),
        (
            {
                "status": "FAIL",
                "failure_details": ["mismatch"],
                "reproduction": ["inspect the fixture"],
                "evidence": [],
            },
            "FAIL requires evidence",
        ),
        ({"status": "BLOCKED"}, "BLOCKED requires blockers"),
        ({"status": "NEEDS_HUMAN"}, "NEEDS_HUMAN requires human_actions"),
    ],
)
def test_nonpassing_status_still_requires_its_explanation_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: dict[str, Any],
    problem: str,
) -> None:
    result = analysis()
    result.update(updates)
    instance, _transport = backend_with_output(
        monkeypatch, tmp_path, {"schema_version": 1, "results": [result]}
    )
    with pytest.raises(InvalidCodexOutput, match=problem):
        instance.analyze_manual_cases([manual_case()])


@pytest.mark.parametrize("fault", ["duplicate", "cycle", "live-pass", "human-fail"])
def test_test_process_cannot_duplicate_steps_cycle_or_claim_unexecuted_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    result = analysis()
    steps = result["test_process"]
    if fault in {"duplicate", "cycle"}:
        steps.append(copy.deepcopy(steps[0]))
        if fault == "cycle":
            steps[1]["step_id"] = "second"
            steps[0]["depends_on"] = ["second"]
            steps[1]["depends_on"] = ["inspect"]
    else:
        steps[0].update(
            kind="requires-live-action" if fault == "live-pass" else "requires-human",
            executor_ref="controlled-live-executor"
            if fault == "live-pass"
            else "human",
            status="PASS" if fault == "live-pass" else "FAIL",
        )
    instance, _transport = backend_with_output(
        monkeypatch, tmp_path, {"schema_version": 1, "results": [result]}
    )
    problem = (
        "duplicates step_id"
        if fault == "duplicate"
        else "dependency cycle"
        if fault == "cycle"
        else "cannot claim a read-only execution result"
    )
    with pytest.raises(InvalidCodexOutput, match=problem):
        instance.analyze_manual_cases([manual_case()])


def test_analysis_returns_typed_results_in_requested_order_without_input_aliasing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second = analysis("case-a"), analysis("case-b")
    original = copy.deepcopy(first)
    instance, transport = backend_with_output(
        monkeypatch, tmp_path, {"schema_version": 1, "results": [second, first]}
    )
    result = instance.analyze_manual_cases(
        [manual_case(" case-a "), manual_case("case-b")]
    )
    assert list(result) == ["case-a", "case-b"]
    assert result["case-a"].as_dict() == original
    transport.payload["results"][1]["summary"] = "changed after parsing"
    assert result["case-a"].summary == original["summary"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", None),
        ("title", " "),
        ("risk", 1),
        ("procedure", ""),
        ("expected", []),
        ("expected", None),
        ("expected", [" "]),
        ("category", 4),
        ("automation", "command"),
    ],
)
def test_invalid_manual_case_never_invokes_the_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any
) -> None:
    instance, transport = backend_with_output(monkeypatch, tmp_path, {})
    item = manual_case()
    item[field] = value
    with pytest.raises(InvalidAcceptanceInput):
        instance.analyze_manual_cases([item])
    assert transport.calls == []


@pytest.mark.parametrize(
    "evidence", [[None], [{}], [{"source": "x", "observation": "y", "extra": 1}]]
)
def test_invalid_review_input_is_rejected_before_any_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: Any
) -> None:
    instance, transport = backend_with_output(monkeypatch, tmp_path, review())
    with pytest.raises(InvalidAcceptanceInput, match="evidence"):
        instance.review_evidence(manual_case(), evidence)
    assert transport.calls == []


@pytest.mark.parametrize(
    "updates,problem",
    [
        ({"status": "PASS", "missing_evidence": ["missing"]}, "PASS cannot include"),
        ({"status": "FAIL"}, "FAIL requires contradictions"),
        ({"status": "BLOCKED"}, "BLOCKED requires missing_evidence"),
        ({"status": "NEEDS_HUMAN"}, "NEEDS_HUMAN requires human_actions"),
    ],
)
def test_independent_review_cannot_contradict_its_declared_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: dict[str, Any],
    problem: str,
) -> None:
    payload = review()
    payload["review"].update(updates)
    instance, _transport = backend_with_output(monkeypatch, tmp_path, payload)
    with pytest.raises(InvalidCodexOutput, match=problem):
        instance.review_evidence(manual_case(), [])
