"""PREEMPT-001..009 contract runner: the pytest verdict becomes case evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.e2e.regional import run_preemption_contracts as contracts


def fake_runner(returncode: int):
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        assert command[1:4] == ["-m", "pytest", "-q"]
        return subprocess.CompletedProcess(
            command, returncode, stdout="1 passed\n", stderr=""
        )

    return run


@pytest.mark.parametrize(("returncode", "verdict"), [(0, "PASS"), (1, "FAIL")])
def test_run_case_writes_case_evidence_under_run_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], returncode: int, verdict: str
) -> None:
    case_id = "GF-REGIONAL-PREEMPT-002"

    code = contracts.run_case(
        case_id, run_dir=tmp_path, release_id="rel-1", runner=fake_runner(returncode)
    )

    assert code == returncode
    path = tmp_path / "cases" / case_id / f"{case_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["case_id"] == case_id
    assert document["verdict"] == verdict
    assert document["release_id"] == "rel-1"
    assert document["pytest_nodeids"] == list(contracts.CASE_NODEIDS[case_id])
    assert document["pytest_returncode"] == returncode
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "1 passed"
    assert json.loads(lines[-1]) == {
        "case_id": case_id,
        "verdict": verdict,
        "pytest_nodeids": list(contracts.CASE_NODEIDS[case_id]),
    }


def test_run_case_without_run_dir_keeps_the_stdout_contract_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    code = contracts.run_case("GF-REGIONAL-PREEMPT-001", runner=fake_runner(0))

    assert code == 0
    assert not (tmp_path / "cases").exists(), (
        "without --run-dir the runner must write no evidence"
    )
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["verdict"] == "PASS"


def test_parser_accepts_the_optional_run_dir() -> None:
    parsed = contracts.build_parser().parse_args(
        ["--case", "GF-REGIONAL-PREEMPT-003", "--run-dir", "/tmp/run"]
    )
    assert parsed.run_dir == Path("/tmp/run")
    assert (
        contracts.build_parser()
        .parse_args(["--case", "GF-REGIONAL-PREEMPT-003"])
        .run_dir
        is None
    )
