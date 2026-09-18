from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests._cov95_acceptance_plan import RETIRED, A, B, PlanInputs
from tests.test_run_regional_acceptance import (
    AnalysisBackend,
    execute,
    plan_for,
    planned_case,
)
from tools import run_regional_acceptance as runner
from tools.codex_acceptance import CodexAcceptanceError, EvidenceObservation
from tools.regional_acceptance_plan import ExecutorKind, PlanMode


@pytest.fixture
def template(tmp_path):
    return PlanInputs(tmp_path).compile().case(A)


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def run(case, **kwargs):
        calls.append((case, kwargs))
        return {"id": case["id"], "status": "PASS"}

    monkeypatch.setattr(runner, "run_case", run)
    return calls


@pytest.mark.parametrize(
    ("first_result", "continues"),
    [
        ({"status": "FAIL", "analysis_status": "PASS"}, False),
        ({"status": "BLOCKED"}, False),
        ({"status": "PASS", "independent_review": {"status": "FAIL"}}, False),
        ({"status": "PASS", "independent_review": {"status": "PASS"}}, True),
        ({"status": "NOT_RUN", "analysis_status": "PASS"}, True),
        ({"status": "NOT_RUN", "proxy_check": {"status": "PASS"}}, True),
        ({"status": "NOT_RUN", "proxy_check": {"status": "FAIL"}}, False),
        ({"status": "NOT_RUN"}, False),
    ],
)
def test_diagnostic_dependencies_follow_evidence_without_promoting_execution(
    template, first_result, continues, monkeypatch
) -> None:
    first, second = planned_case(template, 1), planned_case(template, 2)
    second = replace(
        second, execution=replace(second.execution, depends_on=(first.id,))
    )
    launched = []

    def run(case, **_kwargs):
        launched.append(case["id"])
        return {
            "id": case["id"],
            **(first_result if case["id"] == first.id else {"status": "PASS"}),
        }

    monkeypatch.setattr(runner, "run_case", run)
    results = execute(plan_for(first, second, mode=PlanMode.COLLECT_ALL))
    assert results[1]["status"] == ("PASS" if continues else "BLOCKED"), (
        "a diagnostic dependency must reflect the predecessor's evidence status"
    )
    assert launched == ([first.id, second.id] if continues else [first.id]), (
        "blocked dependencies must never reach the case transport"
    )
    assert results[0]["execution_status"] == first_result["status"], (
        "analysis success must not rewrite the physical execution status"
    )


@pytest.mark.parametrize(
    "raw_evidence",
    [
        None,
        [],
        [
            None,
            {"source": 1, "observation": "invalid"},
            {"source": "fixture.json", "observation": "bounded observation"},
        ],
    ],
)
@pytest.mark.parametrize("review_fails", [False, True])
def test_manual_review_uses_only_structured_observations_and_fails_closed(
    template, calls, raw_evidence, review_fails
) -> None:
    observed = []

    class Backend(AnalysisBackend):
        def analyze_manual_cases(self, cases):
            analyses = super().analyze_manual_cases(cases)
            result = {}
            for case_id, analysis in analyses.items():
                value = analysis.as_dict()
                value["evidence"] = raw_evidence
                result[case_id] = SimpleNamespace(
                    status="PASS", as_dict=lambda value=value: value
                )
            return result

        def review_evidence(self, case, evidence):
            observed.extend(evidence)
            if review_fails:
                raise CodexAcceptanceError("fixture review unavailable")
            return super().review_evidence(case, evidence)

    case = replace(planned_case(template, 1), executor=ExecutorKind.CODEX_MANUAL)
    result = execute(
        plan_for(case, mode=PlanMode.COLLECT_ALL),
        backend=Backend(),
        review_with_codex=True,
    )[0]
    assert observed == (
        [EvidenceObservation("fixture.json", "bounded observation")]
        if raw_evidence
        else []
    ), "the independent review must not receive malformed observation records"
    assert result["independent_review"]["status"] == (
        "BLOCKED" if review_fails else "PASS"
    ), "review errors must remain explicit rather than inheriting the first analysis"
    assert result["status"] == ("BLOCKED" if review_fails else "NOT_RUN"), (
        "manual analysis and review are never execution PASS"
    )
    assert calls == [], "read-only manual analysis cannot execute the procedure"


def test_review_without_a_backend_fails_and_review_without_targets_is_inert(
    template, calls
) -> None:
    case = replace(planned_case(template, 1), executor=ExecutorKind.HUMAN)
    plan = plan_for(case, mode=PlanMode.COLLECT_ALL)
    with pytest.raises(ValueError, match="requires a Codex"):
        execute(plan, review_with_codex=True)
    results = execute(plan, review_with_codex=True, backend=AnalysisBackend())
    assert results[0]["status"] == "NOT_RUN", "a human case remains unexecuted"
    assert "independent_review" not in results[0], "do not invent a review target"
    assert calls == [], "report-only cases never reach an execution transport"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--workers", "0"),
        ("--workers", "-1"),
        ("--review-workers", "0"),
        ("--codex-batch-size", "0"),
        ("--codex-timeout-seconds", "0"),
        ("--codex-timeout-seconds", "-1"),
        ("--codex-timeout-seconds", "nan"),
        ("--codex-timeout-seconds", "inf"),
    ],
)
def test_public_parser_rejects_nonpositive_or_nonfinite_limits(flag, value) -> None:
    with pytest.raises(SystemExit) as error:
        runner.build_parser().parse_args([flag, value])
    assert error.value.code == 2, "invalid bounds must stop before backend construction"


def test_public_parser_accepts_explicit_bounded_limits() -> None:
    args = runner.build_parser().parse_args(
        [
            "--workers",
            "2",
            "--review-workers",
            "1",
            "--codex-batch-size",
            "4",
            "--codex-timeout-seconds",
            "2.5",
        ]
    )
    assert (
        args.workers,
        args.review_workers,
        args.codex_batch_size,
        args.codex_timeout_seconds,
    ) == (2, 1, 4, 2.5), (
        "validated limits must reach the execution configuration unchanged"
    )


@pytest.mark.parametrize("explicit_output", [False, True])
def test_advisory_proposal_entry_uses_only_fake_backend_and_keeps_source_binding(
    explicit_output, tmp_path, monkeypatch, calls
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.order["phases"][0]["entries"][1]["predecessor"] = A
    inputs.catalog["test_cases"][1]["execution"] = {
        "parallel_safe": False,
        "depends_on": [A],
    }
    inputs.write()
    observed = {}

    def propose(cases, mandatory_edges):
        observed["cases"] = cases
        observed["edges"] = mandatory_edges
        return SimpleNamespace(as_dict=lambda: {"trusted": False, "results": []})

    def backend(root, **kwargs):
        observed["backend"] = (root, kwargs)
        return SimpleNamespace(propose_dependencies=propose)

    monkeypatch.setattr(runner, "CodexAcceptanceBackend", backend)
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(runner, "DEFAULT_REPORT_DIR", output_dir)
    args = [
        "--propose-dependencies",
        "--order",
        str(inputs.order_path),
        "--catalog",
        str(inputs.catalog_path),
        "--codex-binary",
        "fixture-codex",
        "--codex-timeout-seconds",
        "2",
    ]
    output = tmp_path / "proposal.json"
    if explicit_output:
        args.extend(["--proposal-output", str(output)])
    assert runner.main(args) == 0, "the fake read-only proposal should be written"
    if not explicit_output:
        outputs = list(output_dir.glob("regional-dependency-proposal-*.json"))
        assert len(outputs) == 1, "default proposal naming must create one report"
        output = outputs[0]
    result = json.loads(output.read_text())
    assert [case["id"] for case in observed["cases"]] == [A, B], (
        "retired cases must not be sent for dependency proposals"
    )
    assert RETIRED not in {edge.before for edge in observed["edges"]}, (
        "retired entries must not become mandatory ordering constraints"
    )
    assert (A, B) in [(edge.before, edge.after) for edge in observed["edges"]], (
        "catalog dependency constraints must accompany the advisory proposal"
    )
    assert result["trusted"] is False, "advisory output must never authorize a DAG"
    assert (
        result["sources"]["order_sha256"]
        == hashlib.sha256(inputs.order_path.read_bytes()).hexdigest()
    ), "the proposal must bind the exact input order"
    assert output.stat().st_mode & 0o777 == 0o600, "advisory reports remain private"
    assert calls == [], "proposal mode must not execute any case"


def test_default_report_path_and_runtime_limits_reach_fake_execution(
    template, tmp_path, monkeypatch, calls
) -> None:
    plan = plan_for(planned_case(template, 1), mode=PlanMode.COLLECT_ALL)
    monkeypatch.setattr(
        runner, "compile_regional_acceptance_plan", lambda **_kwargs: plan
    )
    monkeypatch.setattr(runner, "DEFAULT_REPORT_DIR", tmp_path / "reports")
    assert runner.main(["--workers", "2", "--codex-batch-size", "2"]) == 0, (
        "a complete fake automated case should exit successfully"
    )
    paths = list((tmp_path / "reports").glob("regional-acceptance-*.json"))
    assert len(paths) == 1, "default report naming must produce one private result"
    result = json.loads(paths[0].read_text())
    assert result["environment"]["workers"] == 2, "retain the selected worker budget"
    assert result["summary"]["case_status"] == {"PASS": 1}, (
        "report the actual fake result"
    )
    assert len(calls) == 1, "one planned case must execute once"


def test_nested_tuple_sanitization_and_output_cap_preserve_public_metadata(
    monkeypatch,
) -> None:
    payload = {
        "entries": (
            {"name": "UNIT_TOKEN", "value": "fixture-value"},
            {"token_ref": "fixture-ref"},
        ),
        "kind": "Secret",
        "binaryData": {"fixture": "not-a-real-secret"},
    }
    assert runner.sanitize_value(payload) == {
        "entries": [
            {"name": "UNIT_TOKEN", "value": "<redacted>"},
            {"token_ref": "fixture-ref"},
        ],
        "kind": "Secret",
        "binaryData": "<redacted>",
    }, "redact credential values while keeping permitted reference metadata"
    monkeypatch.setattr(runner, "MAX_CAPTURED_OUTPUT_CHARS", 32)
    result = runner.redact_text("x" * 42)
    assert result == "x" * 32 + "\n<output truncated by regional acceptance runner>", (
        "captured public output must have a bounded payload and visible truncation"
    )
