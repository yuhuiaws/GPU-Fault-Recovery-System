"""Scope-independent predecessor facts stay bound to one parsed document."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.regional_live_fixture import (
    predecessor_evidence,
    predecessor_evidence_facts,
    read_predecessor_evidence,
)

CASE_ID = "GF-REGIONAL-DESTR-009"
IDENTITY = {"release_id": "release-a", "cluster_id": "cluster-a"}


@pytest.fixture(autouse=True)
def formal_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "formal")
    monkeypatch.delenv(SELECTION_REFERENCE_ENV, raising=False)


def evidence_document(**changes: Any) -> dict[str, Any]:
    return {
        "case_id": CASE_ID,
        "verdict": "PASS",
        "status": "COMPLETED",
        "execution_scope": "formal",
        "formal_sequence_satisfied": True,
        **IDENTITY,
        **changes,
    }


def write_evidence(tmp_path: Path, **changes: Any) -> Path:
    path = tmp_path / "predecessor.json"
    path.write_text(json.dumps(evidence_document(**changes)), encoding="utf-8")
    return path


def selective_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-SHARED-RUNNER")


def test_selective_sequence_waiver_keeps_valid_predecessor_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_evidence(tmp_path)
    selective_scope(monkeypatch)
    result = predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert result["execution_allowed"] is True, "the explicit sequence waiver applies"
    assert result["verdict"] == "SKIPPED_BY_OPERATOR", "a waiver is not a case PASS"
    assert result["formal_sequence_satisfied"] is False, "selective is not formal"
    assert result["evidence_valid"] is True, "a bound completed PASS remains reusable"
    assert result["evidence_verdict"] == "PASS" and result["evidence_error"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"case_id": "GF-REGIONAL-DESTR-010"},
        {"verdict": "FAIL"},
        {"verdict": "PASS "},
        {"status": "RUNNING"},
        {"status": "FAILED"},
        {"status": None},
        {"release_id": "release-old"},
        {"cluster_id": "cluster-foreign"},
        {"release_id": ["release-a"]},
        {"cluster_id": None},
        {"execution_scope": "unknown"},
        {"execution_scope": None},
        {"execution_scope": ""},
        {"execution_scope": []},
        {"formal_sequence_satisfied": "true"},
        {"formal_sequence_satisfied": 1},
        {"formal_sequence_satisfied": None},
    ],
    ids=[
        "wrong-case",
        "failed",
        "malformed-verdict",
        "running",
        "failed-execution",
        "unknown-status",
        "old-release",
        "foreign-cluster",
        "malformed-release",
        "missing-cluster",
        "unknown-scope",
        "null-scope",
        "empty-scope",
        "malformed-scope",
        "string-flag",
        "numeric-flag",
        "null-flag",
    ],
)
def test_invalid_facts_do_not_become_reusable_under_a_sequence_waiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    path = write_evidence(tmp_path, **changes)
    selective_scope(monkeypatch)
    result = predecessor_evidence(path, CASE_ID, **IDENTITY)
    facts = predecessor_evidence_facts(path, CASE_ID, **IDENTITY)
    assert result["execution_allowed"] is True, "only the sequence check is waived"
    assert result["evidence_valid"] is False, "invalid evidence cannot supply facts"
    assert result["evidence_error"], "invalid evidence must retain its own diagnostic"
    assert facts["evidence_valid"] is False and facts["evidence_error"], (
        "the standalone facts API must apply the same fail-closed checks"
    )


@pytest.mark.parametrize(
    "body",
    [
        "not-json",
        "null",
        "[]",
        '{"case_id":"wrong","case_id":"' + CASE_ID + '","verdict":"PASS"}',
        '{"case_id":"' + CASE_ID + '","verdict":"PASS","extra":NaN}',
        '{"case_id":"' + CASE_ID + '","verdict":"PASS","extra":Infinity}',
        '{"case_id":"' + CASE_ID + '","verdict":"PASS","nested":{"x":1,"x":2}}',
    ],
    ids=[
        "syntax",
        "null",
        "array",
        "duplicate-case",
        "nan",
        "infinity",
        "nested-duplicate",
    ],
)
def test_ambiguous_or_malformed_json_never_supplies_pass_facts(
    tmp_path: Path, body: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(body, encoding="utf-8")
    result = predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert result["verdict"] == "INVALID", "a parsing failure is not an absent case"
    assert result["execution_allowed"] is False and result["evidence_valid"] is False


@pytest.mark.parametrize("failure", ["missing", "permission", "invalid-utf8"])
def test_unreadable_evidence_keeps_the_waiver_separate_from_its_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "unreadable.json"
    if failure == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif failure == "permission":
        path.touch()

        def denied(_path: Path) -> bytes:
            raise PermissionError("untrusted diagnostic contents")

        monkeypatch.setattr(Path, "read_bytes", denied)
    selective_scope(monkeypatch)
    result = predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert result["execution_allowed"] is True and result["evidence_valid"] is False
    assert result["evidence_error"], "failed reads need a separate evidence diagnostic"
    assert "untrusted diagnostic contents" not in result["evidence_error"], (
        "an I/O error must not echo uncontrolled data into evidence"
    )


def test_formal_gate_and_facts_use_exactly_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_evidence(tmp_path)
    first = json.dumps(evidence_document(formal_sequence_satisfied=False))
    second = json.dumps(evidence_document(formal_sequence_satisfied=True))
    reads: list[Path] = []

    def changing_file(target: Path) -> bytes:
        reads.append(target)
        return (first if len(reads) == 1 else second).encode("utf-8")

    monkeypatch.setattr(Path, "read_bytes", changing_file)
    result = predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert reads == [path], "facts and sequence authorization must share one parse"
    assert result["evidence_valid"] is True, "the completed bound facts are valid"
    assert result["execution_allowed"] is False, "the same file did not prove sequence"
    assert result["evidence_sha256"] == hashlib.sha256(first.encode()).hexdigest()


def test_reused_document_is_the_one_validated_by_the_public_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_evidence(tmp_path)
    first = evidence_document(workflow_official_steps=[{"operation": "STOP_WORKLOADS"}])
    text = json.dumps(first)
    reads: list[Path] = []

    def changing_file(target: Path) -> bytes:
        reads.append(target)
        current = (
            text
            if len(reads) == 1
            else json.dumps(
                evidence_document(cluster_id="foreign", workflow_official_steps=[])
            )
        )
        return current.encode("utf-8")

    monkeypatch.setattr(Path, "read_bytes", changing_file)
    document, facts = read_predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert reads == [path], "recorded steps must not trigger another path read"
    assert document == first and facts["evidence_valid"] is True, (
        "the caller must receive exactly the validated document"
    )
    assert facts["evidence_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_selective_predecessor_facts_are_valid_but_cannot_prove_formal_sequence(
    tmp_path: Path,
) -> None:
    path = write_evidence(
        tmp_path, execution_scope="selective", formal_sequence_satisfied=False
    )
    result = predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert result["evidence_valid"] is True, "selective facts remain facts"
    assert result["execution_allowed"] is False, "selective evidence is not formal"


@pytest.mark.parametrize("field", ["release_id", "cluster_id"])
def test_empty_expected_identity_cannot_turn_into_an_unbound_check(
    tmp_path: Path, field: str
) -> None:
    path = write_evidence(tmp_path)
    result = predecessor_evidence(path, CASE_ID, **{**IDENTITY, field: ""})
    assert result["execution_allowed"] is False and result["evidence_valid"] is False


def test_legacy_absent_scope_and_status_keep_the_unbound_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"case_id": CASE_ID, "verdict": "PASS"}), encoding="utf-8"
    )
    result = predecessor_evidence(path, CASE_ID)
    assert result["execution_allowed"] is True, "omitted legacy fields remain readable"


def test_evidence_digest_binds_original_bytes_including_line_endings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "windows-evidence.json"
    data = (
        json.dumps(evidence_document(), indent=2).replace("\n", "\r\n").encode("utf-8")
    )
    path.write_bytes(data)
    document, facts = read_predecessor_evidence(path, CASE_ID, **IDENTITY)
    assert document == evidence_document() and facts["evidence_valid"] is True, (
        "ordinary JSON line endings must not change the parsed facts"
    )
    assert facts["evidence_sha256"] == hashlib.sha256(data).hexdigest(), (
        "content binding must hash actual bytes, not newline-normalized text"
    )
