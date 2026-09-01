from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import tools.run_fault_test_cases as runner
from tests._script_loader import lazy_script_module
from tools.run_fault_test_cases import (
    CURRENT_STATUS_VALUES,
    DEFAULT_REPORT_DIR,
    load_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/fault"
INDEX = EVIDENCE / "index.json"
CATALOG = ROOT / "testcases/fault-scenarios.yaml"
INDEX_SCRIPT = lazy_script_module(
    "build_fault_evidence_index", ROOT / "scripts/build-fault-evidence-index.py"
)


def test_fault_evidence_index_is_current() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build-fault-evidence-index.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_report_verdict_vocabulary_matches_readme() -> None:
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    section = readme.split("标准 verdict 只有：", 1)[1].lstrip().split("\n\n", 1)[0]
    documented = set(re.findall(r"^- `([A-Z_]+)`$", section, re.MULTILINE))

    assert documented, "README verdict section format no longer parses"
    assert documented == set(INDEX_SCRIPT.VERDICTS)


def test_cross_level_verdict_map_covers_case_vocabulary() -> None:
    union = set().union(*INDEX_SCRIPT.REPORT_TO_CASE_VERDICTS.values())

    assert union == CURRENT_STATUS_VALUES
    assert INDEX_SCRIPT.REPORT_TO_CASE_VERDICTS["PASS"] == {"PASS"}
    assert INDEX_SCRIPT.REPORT_TO_CASE_VERDICTS["PASS_WITH_LIMITATIONS"] == {"PASS"}
    assert "PASS" not in INDEX_SCRIPT.REPORT_TO_CASE_VERDICTS["FAIL"]
    assert {"BLOCKED", "NOT_RUN"} <= INDEX_SCRIPT.REPORT_TO_CASE_VERDICTS["FAIL"]


def test_report_verdict_rejects_conflicting_catalog_case(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    report = evidence / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "verdict": "PASS",
                "executed_at": "2026-08-24T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    manifest = evidence / "manifest.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        'curated_at: "2026-08-24"\n'
        "provenance: test\n"
        "provenance_note: test fixture\n"
        "reports:\n"
        "  - file: report.json\n"
        "    case_ids: [GF-TEST-BLOCKED]\n"
        "    verdict: PASS\n"
        '    executed_at: "2026-08-24T00:00:00Z"\n'
        "    limitations: [test fixture]\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "schema_version: 1\n"
        "test_cases:\n"
        "  - id: GF-TEST-BLOCKED\n"
        "    evidence:\n"
        "      verdict: BLOCKED\n",
        encoding="utf-8",
    )
    module = sys.modules[INDEX_SCRIPT.build_index.__module__]
    monkeypatch.setattr(module, "EVIDENCE_DIR", evidence)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setattr(module, "INDEX", evidence / "index.json")
    monkeypatch.setattr(module, "CATALOG", catalog)

    with pytest.raises(
        INDEX_SCRIPT.EvidenceIndexError, match="conflicts with catalog evidence.verdict"
    ):
        INDEX_SCRIPT.build_index()


def test_public_catalog_contains_no_private_report_pointers() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    assert index["reports"] == []
    assert index["verdict_counts"] == {}

    for case in load_catalog(CATALOG):
        evidence = case.get("evidence") or {}
        assert set(evidence) <= {"verdict"}, case["id"]


def test_public_fault_evidence_tree_contains_only_schema_files() -> None:
    files = {path.name for path in EVIDENCE.iterdir() if path.is_file()}

    assert files == {"README.md", "manifest.yaml", "index.json"}
    manifest = (EVIDENCE / "manifest.yaml").read_text(encoding="utf-8")
    assert "reports: []" in manifest
    assert "private evidence store" in manifest


def test_runner_writes_raw_reports_to_ignored_artifacts() -> None:
    assert DEFAULT_REPORT_DIR == ROOT / "artifacts/fault"
    assert not (ROOT / "reports").exists()
    assert "artifacts/*" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_runner_emits_the_standard_v2_envelope(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "fault-report.json"

    monkeypatch.setattr(
        runner, "run_case", lambda case, **_kwargs: {"id": case["id"], "status": "PASS"}
    )

    result = runner.main(["--case", "GF-POL-001", "--report", str(report)])
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert result == 0
    assert payload["schema_version"] == 2
    assert payload["report_type"] == "fault-test-run"
    assert payload["verdict"] == "PASS"
    assert payload["limitations"] == []
    assert "executed_at" in payload
    assert "generated_at" not in payload
    assert payload["summary"]["workers"] == 1
    assert payload["results"][0]["scheduling"]["locks"]


def test_runner_isolates_local_cases_but_preserves_cap005_environment(
    tmp_path: Path, monkeypatch
) -> None:
    reports = iter([tmp_path / "local.json", tmp_path / "cap005.json"])
    observed = []

    def fake_run_case(case, **kwargs):
        observed.append((case["id"], kwargs.get("environment")))
        return {"id": case["id"], "status": "PASS"}

    monkeypatch.setattr(runner, "run_case", fake_run_case)

    assert (
        runner.main(
            ["--case", "GF-POL-001", "--report", str(next(reports)), "--workers", "2"]
        )
        == 0
    )
    assert (
        runner.main(
            [
                "--case",
                "GF-REGIONAL-CAP-005",
                "--include-live",
                "--report",
                str(next(reports)),
                "--workers",
                "2",
            ]
        )
        == 0
    )

    assert observed == [
        ("GF-POL-001", runner.build_isolated_environment()),
        ("GF-REGIONAL-CAP-005", None),
    ]
