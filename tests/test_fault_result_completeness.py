from __future__ import annotations

import json

import pytest

from tools import run_fault_test_cases as runner

IDENTITY = "a" * 64


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    case = next(
        item
        for item in runner.load_catalog(runner.DEFAULT_CATALOG)
        if item["id"] == "GF-POL-001"
    )
    payload = {
        "schema_version": 1,
        "source_identity": IDENTITY,
        "session": {
            "source_identity": IDENTITY,
            "exitstatus": 0,
            "collected_nodeids": [case["pytest_nodeid"]],
            "discovered_nodeids": [case["pytest_nodeid"]],
        },
        "records": {
            case["pytest_nodeid"]: {
                "status": "PASS",
                "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
            }
        },
    }
    monkeypatch.setattr(runner, "source_identity", lambda root: IDENTITY)
    return tmp_path / "results.json", case, payload


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_fault_case_cannot_pass_with_an_uncompleted_pytest_phase(
    artifact, phase
) -> None:
    path, case, payload = artifact
    payload["records"][case["pytest_nodeid"]]["phases"].pop(phase)
    path.write_text(json.dumps(payload), encoding="utf-8")
    results = runner.load_pytest_results([case], path=path)
    assert results[case["id"]]["status"] == "FAIL", (
        "a partial pytest run is not acceptance"
    )


@pytest.mark.parametrize(
    "session",
    [
        None,
        {"source_identity": "b" * 64, "exitstatus": 0},
        {"source_identity": IDENTITY, "exitstatus": 1},
        {"source_identity": IDENTITY, "exitstatus": False},
    ],
)
def test_fault_result_reuse_refuses_source_drift_or_failed_session(
    artifact, session
) -> None:
    path, case, payload = artifact
    payload["session"] = session
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="session is incomplete or source changed"):
        runner.load_pytest_results([case], path=path)


@pytest.mark.parametrize("legacy", [False, True])
def test_complete_result_preserves_existing_case_mapping(artifact, legacy) -> None:
    path, case, payload = artifact
    if legacy:
        payload.pop("session")
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.load_pytest_results([case], path=path)
    assert result[case["id"]]["status"] == "PASS", (
        "complete source-bound records still work"
    )


@pytest.mark.parametrize(
    "collected", [None, [], ["tests/unit.py::missing"], ["duplicate", "duplicate"]]
)
def test_modern_result_reuse_requires_the_entire_collection(artifact, collected):
    path, case, payload = artifact
    payload["session"]["collected_nodeids"] = collected
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="complete collection"):
        runner.load_pytest_results([case], path=path)
