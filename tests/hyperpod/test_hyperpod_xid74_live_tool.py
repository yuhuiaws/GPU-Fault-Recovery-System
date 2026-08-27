import json
from pathlib import Path

import pytest

import scripts.e2e.hyperpod.run_hyperpod_xid74_case as live_tool
from scripts.e2e.hyperpod.run_hyperpod_xid74_case import (
    CASES,
    evaluate_assertions,
    injection_manifest,
    validate_case_flags,
    wait_for_evidence,
)
from tools.run_fault_test_cases import load_catalog

ROOT = Path(__file__).resolve().parents[2]


def snapshot(
    *,
    attempt_id: str,
    pod_uid: str,
    unschedulable: bool = False,
    taints: list[dict] | None = None,
) -> dict:
    return {
        "node": {
            "ready": True,
            "unschedulable": unschedulable,
            "taints": taints or [],
            "annotations": {},
        },
        "managed_pods": [{"uid": pod_uid, "attempt_id": attempt_id, "node": "node-a"}],
    }


def support_evidence(case_name: str) -> dict:
    return {
        "event": {"xid": 74, "registers": CASES[case_name]["registers"]},
        "decision": {"action": "ESCALATE_OPERATOR", "disposition": "EXECUTABLE"},
        "workflow": {
            "status": "SUCCEEDED",
            "completed_operations": ["FREEZE_EVIDENCE", "ESCALATE_SUPPORT"],
        },
        "notifications": [
            {
                "notification": {"body_text": "xid74-support-zh-v2"},
                "delivery": {"status": "SENT"},
            }
        ],
    }


def test_xid74_live_cases_have_exactly_seven_registers() -> None:
    assert all(len(case["registers"]) == 7 for case in CASES.values())


def test_xid74_injection_contains_no_test_id() -> None:
    manifest = injection_manifest(
        "gpu-fault-system",
        "node-a",
        "safe-only",
        "0000:5A:00",
        CASES["safe-only"]["registers"],
        "123456",
    )
    message = manifest["spec"]["containers"][0]["env"][0]["value"]

    assert message == (
        "NVRM: Xid (PCI:0000:5A:00): 74, "
        "pid=1234, name=python, Link 3, "
        "0x1 0x0 0x0 0x0 0x0 0x0 0x0"
    )
    assert "test" not in message.lower()


def test_snapshot_filters_managed_pods_by_job_id(monkeypatch) -> None:
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[:3] == ["kubectl", "get", "node"]:
            return json.dumps(
                {
                    "metadata": {"annotations": {}},
                    "spec": {"taints": []},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            )
        return json.dumps({"items": []})

    monkeypatch.setattr(live_tool, "run", run)

    live_tool.snapshot("gpu-fault-system", "node-a", "training-job")

    assert commands[1][6] == (
        "gpu-fault.io/managed=true,gpu-fault.io/job-id=training-job"
    )


def test_mutating_xid74_cases_require_explicit_risk_class() -> None:
    assert CASES["safe-only"]["mode"] == "monitor"
    assert CASES["all-zero"]["mode"] == "support"
    assert CASES["ecc-parity"]["mode"] == "monitor"
    assert CASES["ecc-third"]["mode"] == "reset"


def test_destructive_xid74_cases_require_explicit_flags() -> None:
    with pytest.raises(ValueError, match="--allow-reset"):
        validate_case_flags(
            CASES["mechanical-first"],
            allow_support=False,
            allow_reset=False,
            require_email_sent=False,
        )
    with pytest.raises(ValueError, match="--allow-support"):
        validate_case_flags(
            CASES["marginal-channel"],
            allow_support=False,
            allow_reset=True,
            require_email_sent=True,
        )

    validate_case_flags(
        CASES["marginal-channel"],
        allow_support=True,
        allow_reset=True,
        require_email_sent=True,
    )


def test_xid74_tool_case_ids_exist_in_catalog() -> None:
    catalog = {
        case["id"]: case
        for case in load_catalog(ROOT / "testcases" / "fault-scenarios.yaml")
    }

    assert {case["test_case_id"] for case in CASES.values()} <= set(catalog)
    for case_name, case in CASES.items():
        assert catalog[case["test_case_id"]]["operator_case"] == case_name


def test_evidence_poll_retries_transient_query_error(monkeypatch) -> None:
    calls = 0

    def read_evidence(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary Aurora DNS failure")
        return {"found": True, "workflow": {"status": "SUCCEEDED"}}

    monkeypatch.setattr(live_tool, "read_evidence", read_evidence)
    monkeypatch.setattr(live_tool.time, "sleep", lambda _: None)

    evidence = wait_for_evidence(
        "gpu-fault-system",
        "cluster-a",
        "node-a",
        "2026-07-26T00:00:00+00:00",
        CASES["ecc-third"]["registers"],
        timeout=1,
    )

    assert calls == 2
    assert evidence["workflow"]["status"] == "SUCCEEDED"


def test_support_case_asserts_complete_support_chain() -> None:
    before = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after = snapshot(attempt_id="training-a001", pod_uid="pod-a001")

    assertions = evaluate_assertions(
        case=CASES["all-zero"],
        evidence=support_evidence("all-zero"),
        before=before,
        after=after,
        require_email_sent=True,
    )

    assert assertions
    assert all(assertions.values())


def test_support_case_fails_if_managed_workload_disappears() -> None:
    before = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after["managed_pods"] = []

    assertions = evaluate_assertions(
        case=CASES["all-zero"],
        evidence=support_evidence("all-zero"),
        before=before,
        after=after,
    )

    assert assertions["managed_pod_identities_unchanged"] is False


def test_support_case_fails_if_workload_was_restarted() -> None:
    before = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after = snapshot(attempt_id="training-a002", pod_uid="pod-a002")
    evidence = {
        "event": {"xid": 74, "registers": CASES["secondary-only"]["registers"]},
        "decision": {"action": "ESCALATE_OPERATOR", "disposition": "EXECUTABLE"},
        "workflow": {
            "status": "SUCCEEDED",
            "completed_operations": [
                "FREEZE_EVIDENCE",
                "ESCALATE_SUPPORT",
                "RESTART_WORKLOAD",
            ],
        },
        "notifications": [
            {
                "notification": {"body_text": "xid74-support-zh-v2"},
                "delivery": {"status": "SENT"},
            }
        ],
    }

    assertions = evaluate_assertions(
        case=CASES["secondary-only"], evidence=evidence, before=before, after=after
    )

    assert assertions["workload_restart_not_executed"] is False
    assert assertions["managed_pod_identities_unchanged"] is False


def test_monitor_case_fails_when_pod_uid_changes() -> None:
    before = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after = snapshot(attempt_id="training-a001", pod_uid="pod-replaced")
    evidence = {
        "event": {"xid": 74, "registers": CASES["safe-only"]["registers"]},
        "decision": {
            "action": "NO_ACTION",
            "disposition": "MONITOR_ONLY",
            "workflow_request_id": None,
        },
    }

    assertions = evaluate_assertions(
        case=CASES["safe-only"], evidence=evidence, before=before, after=after
    )

    assert assertions["managed_pod_identities_unchanged"] is False


def test_ecc_third_occurrence_requires_reset_chain() -> None:
    evidence = {
        "event": {"xid": 74, "registers": CASES["ecc-third"]["registers"]},
        "decision": {
            "action": "RESET_GPU",
            "disposition": "EXECUTABLE",
            "nvlink_occurrence_counts": {"register1.bit4": 3},
        },
        "workflow": {
            "status": "SUCCEEDED",
            "completed_operations": [
                "FREEZE_EVIDENCE",
                "MARK_UNSCHEDULABLE",
                "COLLECT_DIAGNOSTIC_BUNDLE",
                "RUN_NVLINK74_WORKFLOW",
                "RESET_GPU",
                "RESTORE_SCHEDULING",
                "RESTART_WORKLOAD",
            ],
        },
        "notifications": [],
    }
    before = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after = snapshot(attempt_id="training-a002", pod_uid="pod-a002")

    assertions = evaluate_assertions(
        case=CASES["ecc-third"], evidence=evidence, before=before, after=after
    )

    assert all(assertions.values())


def test_marginal_channel_requires_remediation_and_quarantine() -> None:
    evidence = support_evidence("marginal-channel")
    evidence["workflow"]["completed_operations"] = [
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "RUN_NVLINK74_WORKFLOW",
        "RESET_GPU",
        "QUARANTINE",
        "ESCALATE_SUPPORT",
        "RESTART_WORKLOAD",
    ]
    before = snapshot(attempt_id="training-a001", pod_uid="pod-a001")
    after = snapshot(
        attempt_id="training-a002",
        pod_uid="pod-a002",
        unschedulable=True,
        taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
    )

    assertions = evaluate_assertions(
        case=CASES["marginal-channel"],
        evidence=evidence,
        before=before,
        after=after,
        require_email_sent=True,
    )

    assert all(assertions.values())
