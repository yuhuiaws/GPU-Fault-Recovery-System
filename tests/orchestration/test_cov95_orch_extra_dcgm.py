from __future__ import annotations

import pytest

from gpu_fault.dcgm_diagnostic_analysis import (
    MAX_FINDINGS,
    MAX_MESSAGE_LENGTH,
    MAX_MESSAGES_PER_FINDING,
    build_dcgm_recommendations,
    dcgm_failures_are_configuration_only,
    extract_dcgm_diagnostic_findings,
    normalize_dcgm_status,
)
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ({}, None),
        ([], None),
        (1.5, None),
        (0, None),
        (True, None),
        ("unexpected", None),
        (" passed ", "PASS"),
        ("SUCCESS", "PASS"),
        ("succeeded", "PASS"),
        ("failed", "FAIL"),
        ("warning", "WARN"),
        ("skipped", "SKIP"),
        ("not_run", "NOT_RUN"),
        ("notrun", "NOT_RUN"),
    ],
)
def test_dcgm_status_normalization_rejects_unknown_shapes_and_keeps_known_outcomes(
    value, expected
):
    assert normalize_dcgm_status(value) == expected


def test_nested_diagnostic_inherits_names_and_extracts_bounded_structured_details():
    payload = {
        "suite": {
            "name": "memory",
            "tests": [
                {
                    "name": {},
                    "status": "unknown",
                    "result": "FAILED",
                    "entity_ids": {"groups": [*range(30)]},
                    "error_code": {"primary": 42, "nested": [7, 7, None]},
                    "meta": {
                        "severity": [2],
                        "warnings": [
                            {"message": f"message-{index}-" + "x" * 600}
                            for index in range(30)
                        ],
                    },
                }
            ],
        }
    }
    findings = extract_dcgm_diagnostic_findings(payload)
    assert len(findings) == 1
    found = findings[0]
    assert found["test_name"] == "memory"
    assert found["status"] == "FAIL"
    assert found["entities"] == []
    assert found["error_codes"] == ["42", "7"]
    assert found["error_severities"] == ["2"]
    assert found["result_path"] == "$.suite.tests[0]"
    assert len(found["messages"]) == MAX_MESSAGES_PER_FINDING
    assert all(len(message) == MAX_MESSAGE_LENGTH for message in found["messages"]), (
        "nested diagnostic messages exceeded their stored bound"
    )
    assert found["messages"][0].startswith("message-0-"), "message order changed"


def test_flattened_entity_and_error_values_are_deduplicated_and_bounded():
    payload = {
        "name": "unit diagnostic",
        "status": "FAIL",
        "gpu_ids": {"groups": [*range(30)]},
        "error_id": [12, {"codes": [12, 13, False]}, None, ""],
        "severity": {"kind": [5, 5]},
    }
    (finding,) = extract_dcgm_diagnostic_findings(payload)
    assert finding["entities"] == [
        str(index) for index in range(MAX_MESSAGES_PER_FINDING)
    ]
    assert finding["error_codes"] == ["12", "13", "False"]
    assert finding["error_severities"] == ["5"]
    assert dcgm_failures_are_configuration_only([finding]) is True


def test_finding_limit_stops_recording_after_the_first_bounded_population():
    findings = extract_dcgm_diagnostic_findings(
        [
            {"name": f"test-{index}", "result": "WARN"}
            for index in range(MAX_FINDINGS + 7)
        ]
    )
    assert len(findings) == MAX_FINDINGS
    assert [item["test_name"] for item in findings] == [
        f"test-{index}" for index in range(MAX_FINDINGS)
    ]
    assert findings[-1]["result_path"] == f"$[{MAX_FINDINGS - 1}]"


@pytest.mark.parametrize(
    "payload", [None, [], {"status": "FAIL"}, {"name": "", "status": "FAIL"}]
)
def test_unnamed_or_absent_results_do_not_become_diagnostic_findings(payload):
    assert extract_dcgm_diagnostic_findings(payload) == []


def test_warning_message_arrays_preserve_their_evidence_and_select_specific_guidance():
    findings = extract_dcgm_diagnostic_findings(
        {
            "name": "unit-check",
            "status": "FAIL",
            "warnings": ["GPU temperature exceeded the limit", "cooling fan stalled"],
        }
    )
    assert findings[0]["messages"] == [
        "GPU temperature exceeded the limit",
        "cooling fan stalled",
    ], "string-array DCGM warnings were silently discarded"
    recommendations = build_dcgm_recommendations(
        findings, returncode=0, parse_error=None
    )
    assert recommendations[0]["action_code"] == "THERMAL_COOLING_INSPECTION"


@pytest.mark.parametrize("parse_error", [None, "unit incomplete JSON"])
def test_configuration_failure_does_not_hide_an_unreadable_execution_result(
    parse_error,
):
    findings = extract_dcgm_diagnostic_findings(
        {
            "name": "software",
            "status": "FAIL",
            "warnings": [{"warning": "unit host configuration", "error_severity": 5}],
        }
    )
    result = build_dcgm_recommendations(
        findings,
        returncode=2,
        parse_error=parse_error,
        configuration_only=dcgm_failures_are_configuration_only(findings),
    )
    codes = {item["action_code"] for item in result}
    assert "GPU_HOST_CONFIG_REMEDIATION" in codes
    assert ("DCGM_EXECUTION_REVIEW" in codes) is (parse_error is not None)
    configuration = next(
        item for item in result if item["action_code"] == "GPU_HOST_CONFIG_REMEDIATION"
    )
    assert configuration["priority"] == "REVIEW"


def test_recommendations_group_tests_by_action_and_priority_without_duplicate_triggers():
    findings = extract_dcgm_diagnostic_findings(
        [
            {"name": "thermal-a", "status": "FAIL"},
            {"name": "thermal-a", "status": "FAIL"},
            {"name": "thermal-b", "status": "FAIL"},
            {"name": "thermal-warn", "status": "WARN"},
            {"name": "thermal-pass", "status": "PASS"},
            {"name": "context_create", "status": "FAIL"},
            {"name": "compute", "status": "FAIL"},
        ]
    )
    results = build_dcgm_recommendations(findings, returncode=0, parse_error=None)
    thermal = [
        item for item in results if item["action_code"] == "THERMAL_COOLING_INSPECTION"
    ]
    assert [(item["priority"], item["trigger_tests"]) for item in thermal] == [
        ("IMMEDIATE", ["thermal-a", "thermal-b"]),
        ("REVIEW", ["thermal-warn"]),
    ]
    assert {item["action_code"] for item in results} == {
        "THERMAL_COOLING_INSPECTION",
        "GPU_CLIENT_DRIVER_CHECK",
        "GPU_FIELD_DIAGNOSTIC",
    }
