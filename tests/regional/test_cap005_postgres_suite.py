"""CAP-005: the isolated PostgreSQL suite reports what ran, red or green."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_cap005_postgres_suite as cap005

BASE_URL = "postgresql://user:secret@aurora.example/gpu_fault"


def test_isolation_needs_the_test_url_to_name_the_generated_database() -> None:
    database = "gpu_fault_cap005_0123456789ab"
    # The old `database != production_database` was always true; the URL that
    # pytest actually receives is what decides isolation.
    same_as_production = cap005.isolation_facts(BASE_URL, BASE_URL, database)
    assert same_as_production["isolated"] is False

    isolated = cap005.isolation_facts(
        BASE_URL, cap005.database_url(BASE_URL, database), database
    )
    assert isolated["isolated"] is True
    assert isolated["test_url_database"] == database
    assert isolated["production_database"] == "gpu_fault"


def _script(monkeypatch, tmp_path: Path, *, drop_error: Exception | None = None):
    calls: list[str] = []

    def create(_base_url: str, database: str) -> None:
        calls.append(f"create {database}")

    def drop(_base_url: str, database: str) -> None:
        calls.append(f"drop {database}")
        if drop_error is not None:
            raise drop_error

    def run_pytest(
        argv: list[str], *, cwd: Path, env: dict[str, str], junit: Path
    ) -> int:
        calls.append(f"pytest {argv[0]}")
        assert env["GPU_FAULT_TEST_POSTGRES_URL"] != BASE_URL
        junit.write_text(
            '<testsuite tests="3" failures="1" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(cap005, "_create_database", create)
    monkeypatch.setattr(cap005, "_drop_database", drop)
    monkeypatch.setattr(cap005, "_database_exists", lambda _u, _d: False)
    monkeypatch.setattr(cap005, "_run_pytest", run_pytest)
    return calls


def test_a_red_suite_still_reports_its_junit_counts(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _script(monkeypatch, tmp_path)

    summary = cap005.run_suite(BASE_URL, tmp_path)

    assert summary["status"] == "FAIL"
    assert summary["suites"]["postgres"] == {
        "tests": 3,
        "failures": 1,
        "errors": 0,
        "skipped": 0,
    }
    assert summary["exit_codes"] == {"postgres": 1, "contract": 1}
    assert "postgres pytest exited 1" in summary["errors"]
    assert summary["database_dropped"] is True
    assert [item for item in calls if item.startswith("drop")], "database not dropped"


def test_a_drop_failure_is_merged_into_the_report(tmp_path: Path, monkeypatch) -> None:
    _script(monkeypatch, tmp_path, drop_error=RuntimeError("still connected"))

    summary = cap005.run_suite(BASE_URL, tmp_path)

    assert summary["status"] == "FAIL"
    assert "drop database: RuntimeError: still connected" in summary["errors"]
    assert summary["database_dropped"] is False
    assert "suites" in summary, "the suite counts must survive a failed drop"


def test_main_raises_after_printing_the_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _script(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", BASE_URL)
    monkeypatch.setenv("GPU_FAULT_CAP005_WORKDIR", str(tmp_path))

    with pytest.raises(cap005.SuiteError):
        cap005.main()

    assert '"status": "FAIL"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("suites", "fragment"),
    [
        ({"postgres": None}, "left no JUnit report"),
        (
            {"postgres": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}},
            "ran no tests",
        ),
        (
            {"postgres": {"tests": 2, "failures": 0, "errors": 0, "skipped": 1}},
            "not clean",
        ),
    ],
)
def test_clean_suites_names_what_is_wrong(
    suites: dict[str, Any], fragment: str
) -> None:
    assert any(fragment in item for item in cap005.clean_suites(suites)), (
        "clean_suites names the offending suite"
    )
