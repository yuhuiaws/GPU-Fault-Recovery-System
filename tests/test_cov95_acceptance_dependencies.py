from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest

from tests._cov95_acceptance_support import (
    backend_with_output,
    dependency,
    manual_case,
    regional_case,
    review,
)
from tools.codex_acceptance import (
    CodexAcceptanceBackend,
    InvalidAcceptanceInput,
    InvalidCodexOutput,
    MandatoryOrderEdge,
    build_codex_environment,
)


def proposal() -> dict[str, Any]:
    first, second = dependency("case-a"), dependency("case-b")
    second["depends_on"] = ["case-a"]
    return {"schema_version": 1, "trusted": False, "results": [first, second]}


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("case_id", "other", "unknown case_id"),
        ("depends_on", ["case-a"], "self-dependency"),
        ("depends_on", ["case-b", "case-b"], "contains duplicates"),
        ("depends_on", ["other"], "unknown dependencies"),
        ("depends_on", None, "JSON array"),
        ("confidence", True, "must be a number"),
        ("confidence", "0.5", "must be a number"),
        ("confidence", 2, "between 0 and 1"),
        ("confidence", float("nan"), "between 0 and 1"),
        ("locks", {}, "JSON array"),
        ("locks", [{"resource": "x", "mode": "write"}], "mode is unsupported"),
        (
            "locks",
            [{"resource": "x", "mode": "shared"}, {"resource": "x", "mode": "shared"}],
            "duplicates resource lock",
        ),
        (
            "locks",
            [
                {"resource": "x", "mode": "shared"},
                {"resource": "x", "mode": "exclusive"},
            ],
            "conflicting modes",
        ),
        ("executor", "unrestricted-shell", "unsupported executor"),
        ("rationale", "", "non-empty string"),
    ],
)
def test_dependency_rows_must_be_scoped_acyclic_typed_and_noncontradictory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
    problem: str,
) -> None:
    payload = proposal()
    payload["results"][0][field] = value
    instance, transport = backend_with_output(monkeypatch, tmp_path, payload)
    with pytest.raises(InvalidCodexOutput, match=problem):
        instance.propose_dependencies(
            [regional_case("case-a"), regional_case("case-b")], []
        )
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "trusted",
        "numeric-trust",
        "schema",
        "shape",
        "count",
        "duplicate",
        "cycle",
        "mandatory",
    ],
)
def test_dependency_proposal_cannot_promote_itself_or_remove_required_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    payload = proposal()
    if fault in {"trusted", "numeric-trust"}:
        payload["trusted"] = True if fault == "trusted" else 0
    elif fault == "schema":
        payload["schema_version"] = 2
    elif fault == "shape":
        payload["results"] = {}
    elif fault == "count":
        payload["results"].pop()
    elif fault == "duplicate":
        payload["results"][1]["case_id"] = "case-a"
    elif fault == "cycle":
        payload["results"][0]["depends_on"] = ["case-b"]
    else:
        payload["results"][1]["depends_on"] = []
    instance, _transport = backend_with_output(monkeypatch, tmp_path, payload)
    with pytest.raises(InvalidCodexOutput):
        instance.propose_dependencies(
            [regional_case("case-a"), regional_case("case-b")],
            [MandatoryOrderEdge("case-a", "case-b")],
        )


@pytest.mark.parametrize(
    "edges,problem",
    [
        ([["case-a", "case-b"]], "two-item tuple"),
        ([("case-a", "case-a")], "self-edge"),
        ([("case-a", "unknown")], "unknown case"),
        ([("case-a", "case-b"), ("case-a", "case-b")], "duplicate mandatory"),
    ],
)
def test_invalid_mandatory_edges_never_reach_the_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, edges: Any, problem: str
) -> None:
    instance, transport = backend_with_output(monkeypatch, tmp_path, proposal())
    with pytest.raises(InvalidAcceptanceInput, match=problem):
        instance.propose_dependencies(
            [regional_case("case-a"), regional_case("case-b")], edges
        )
    assert transport.calls == []


def test_proposal_preserves_order_locks_and_untrusted_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = proposal()
    expected = copy.deepcopy(payload["results"])
    payload["results"].reverse()
    first = regional_case("case-a")
    first["problem"] = first.pop("summary")
    first["expected"] = "preserve the prerequisite"
    instance, transport = backend_with_output(monkeypatch, tmp_path, payload)
    result = instance.propose_dependencies(
        [first, regional_case("case-b")], [("case-a", "case-b")]
    )
    assert result.trusted is False
    assert list(result.by_case()) == ["case-a", "case-b"]
    assert result.as_dict() == {"trusted": False, "results": expected}
    assert "preserve the prerequisite" in transport.calls[0][1]["input"]


@pytest.mark.parametrize(
    "options,problem",
    [
        ({"codex_binary": " "}, "codex_binary"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"max_output_bytes": 0}, "max_output_bytes"),
    ],
)
def test_backend_rejects_invalid_transport_configuration(
    tmp_path: Path, options: dict[str, Any], problem: str
) -> None:
    with pytest.raises(InvalidAcceptanceInput, match=problem):
        CodexAcceptanceBackend(tmp_path, environment={"HOME": str(tmp_path)}, **options)


def test_backend_rejects_missing_repository_and_blank_home(tmp_path: Path) -> None:
    with pytest.raises(InvalidAcceptanceInput, match="existing directory"):
        CodexAcceptanceBackend(
            tmp_path / "missing", environment={"HOME": str(tmp_path)}
        )
    for environment in ({}, {"HOME": " "}):
        with pytest.raises(InvalidAcceptanceInput, match="HOME"):
            build_codex_environment(environment)
    assert build_codex_environment({"HOME": str(tmp_path)})["PATH"] == os.defpath


@pytest.mark.parametrize("updates", [{"schema_version": 2}, {"review": None}])
def test_review_envelope_must_match_its_declared_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, updates: dict[str, Any]
) -> None:
    payload = review()
    payload.update(updates)
    instance, _transport = backend_with_output(monkeypatch, tmp_path, payload)
    with pytest.raises(InvalidCodexOutput):
        instance.review_evidence(manual_case(), [])
