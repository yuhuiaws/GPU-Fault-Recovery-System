from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_collect016_training_recovery as runner
from tests.test_acceptance_receipt_alignment import IDENTITY, complete_report
from tools import run_fault_test_cases as fault_runner


@pytest.mark.parametrize(
    "defect", ["none", "skip", "partial", "source", "filtered", "missing"]
)
def test_collect016_negative_contracts_require_fresh_complete_local_receipts(
    tmp_path, monkeypatch, defect
):
    monkeypatch.setattr(fault_runner, "source_identity", lambda root: IDENTITY)
    captured = []

    def execute(command, **kwargs):
        captured.append(command)
        assert kwargs["timeout"] == 300
        assert kwargs["env"]["GPU_FAULT_TEST_POSTGRES_URL"] == ""
        assert kwargs["env"]["KUBECONFIG"] == "/dev/null"
        selectors = [item for item in command if item.startswith("tests/")]
        report = complete_report(selectors)
        if defect == "skip":
            report["records"][selectors[-1]]["phases"] = {
                "setup": "skipped",
                "teardown": "passed",
            }
        elif defect == "partial":
            report["session"]["collected_nodeids"].remove(selectors[-1])
            report["records"].pop(selectors[-1])
        elif defect == "source":
            report["source_identity"] = "b" * 64
        elif defect == "filtered":
            report["session"]["selection"]["keyword"] = "not negative"
        if defect != "missing":
            Path(kwargs["env"]["PYTEST_GPU_FAULT_CASE_REPORT"]).write_text(
                json.dumps(report)
            )
        return subprocess.CompletedProcess(command, 0, stdout="local fixture\n")

    monkeypatch.setattr(fault_runner.subprocess, "run", execute)
    proof = runner.local_contract_proof(tmp_path)
    assert len(captured) == 1
    assert proof["validation_level"] == "LOCAL"
    assert proof["proof_scope"] == "local-contract"
    assert proof["live_negative_actions_performed"] is False
    assert proof["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert set(proof["groups"]) == {
        "reset_and_restart",
        "dcgm_classification",
        "dcgm_control_action",
        "client_intersection",
    }
    if defect == "none":
        assert proof["source_identity"] == IDENTITY
        assert len(proof["pytest_records"]) == sum(
            len(group) for group in runner.LOCAL_CONTRACT_SELECTORS.values()
        )
        assert not proof["receipt_errors"]
    else:
        assert proof["receipt_errors"]
        assert all(group["verdict"] == "FAIL" for group in proof["groups"].values()), (
            "an invalid shared receipt cannot authorize any local contract group"
        )
    assert json.loads((tmp_path / "local-contracts.json").read_text()) == proof
    assert (tmp_path / "focused-tests.log").stat().st_mode & 0o777 == 0o600


def test_collect016_does_not_promote_local_negative_contracts_to_live_actions():
    plan = runner.plan_details(
        SimpleNamespace(),
        {"predecessor": {}, "release_id": "local-fixture", "candidate_nodes": []},
    )
    assert plan["proof_scopes"]["D"] == "live-positive-XID109-reset"
    assert plan["proof_scopes"]["negative_gpu_holder_injection"] is False
    assert "LOCAL" in plan["proof_scopes"]["dcgm_and_client_negative_contracts"]
