from __future__ import annotations

from pathlib import Path

import pytest

from tests._cov95_acceptance_support import (
    analysis,
    backend_with_output,
    dependency,
    manual_case,
    regional_case,
    review,
)
from tools.codex_acceptance import InvalidCodexOutput


def test_non_utf8_structured_output_uses_the_backend_error_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance, transport = backend_with_output(monkeypatch, tmp_path, {})
    transport.raw = b"\xff"
    with pytest.raises(InvalidCodexOutput, match="UTF-8|JSON"):
        instance.analyze_manual_cases([manual_case()])
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "confidence", [10**400, -(10**400)], ids=["huge-positive", "huge-negative"]
)
def test_unbounded_numeric_confidence_is_rejected_as_invalid_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, confidence: int
) -> None:
    result = dependency()
    result["confidence"] = confidence
    instance, transport = backend_with_output(
        monkeypatch,
        tmp_path,
        {"schema_version": 1, "trusted": False, "results": [result]},
    )
    with pytest.raises(InvalidCodexOutput, match="confidence"):
        instance.propose_dependencies([regional_case()], [])
    assert len(transport.calls) == 1


@pytest.mark.parametrize("operation", ["analysis", "review", "dependencies"])
def test_boolean_schema_version_cannot_be_accepted_as_numeric_version_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    payload = (
        review()
        if operation == "review"
        else {
            "schema_version": 1,
            "results": [analysis() if operation == "analysis" else dependency()],
        }
    )
    if operation == "dependencies":
        payload["trusted"] = False
    payload["schema_version"] = True
    instance, transport = backend_with_output(monkeypatch, tmp_path, payload)
    with pytest.raises(InvalidCodexOutput, match="schema_version"):
        if operation == "analysis":
            instance.analyze_manual_cases([manual_case()])
        elif operation == "review":
            instance.review_evidence(manual_case(), [])
        else:
            instance.propose_dependencies([regional_case()], [])
    assert transport.schemas[0]["properties"]["schema_version"]["type"] == "integer"
