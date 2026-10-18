"""Regional acceptance runner report edges: a report whose summary or case
status is not a mapping is refused after it has been written for inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests import test_cov95_acceptance_runner_branches as runner_branches
from tests.test_run_regional_acceptance import plan_for, planned_case
from tools import run_regional_acceptance as runner
from tools.regional_acceptance_plan import PlanMode

template = runner_branches.template
calls = runner_branches.calls


@pytest.mark.parametrize(
    ("report", "fragment"),
    [
        ({"verdict": "PASS", "summary": ["not", "a", "mapping"]}, "summary is not"),
        (
            {"verdict": "PASS", "summary": {"total": 1, "case_status": "PASS"}},
            "case_status is not",
        ),
    ],
)
def test_main_refuses_a_malformed_report_after_writing_it(
    template: Any,
    calls: list[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: dict[str, Any],
    fragment: str,
) -> None:
    plan = plan_for(planned_case(template, 1), mode=PlanMode.COLLECT_ALL)
    monkeypatch.setattr(
        runner, "compile_regional_acceptance_plan", lambda **_kwargs: plan
    )
    monkeypatch.setattr(runner, "build_report", lambda *a, **k: report)
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match=fragment):
        runner.main(["--workers", "1", "--report", str(output)])
    assert output.is_file(), "the raw report is kept for inspection"
    assert len(calls) == 1, "the planned case still executed once"
