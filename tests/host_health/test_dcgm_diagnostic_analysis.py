from __future__ import annotations

import pytest

from gpu_fault.dcgm_diagnostic_analysis import (
    build_dcgm_recommendations,
    dcgm_failures_are_configuration_only,
    extract_dcgm_diagnostic_findings,
)


@pytest.mark.parametrize(
    ("test_name", "message", "expected_action"),
    [
        ("Thermal", "GPU temperature exceeded the limit", "THERMAL_COOLING_INSPECTION"),
        ("PCIe", "AER error detected", "PCIE_AER_INSPECTION"),
        ("NVLink", "NVSwitch fabric link failed", "NVLINK_NVSWITCH_INSPECTION"),
        ("Memory", "Uncorrectable ECC memory error", "GPU_MEMORY_FIELD_DIAGNOSTIC"),
        ("Deployment", "DCGM library mismatch", "DRIVER_DCGM_REMEDIATION"),
        ("Unknown Test", "unclassified failure", "DEEP_DIAGNOSTIC_REVIEW"),
    ],
)
def test_dcgm_failure_maps_to_fixed_guidance(
    test_name: str, message: str, expected_action: str
) -> None:
    findings = extract_dcgm_diagnostic_findings(
        {
            "DCGM GPU Diagnostic": {
                "tests": [
                    {
                        "name": test_name,
                        "results": [
                            {
                                "gpu_ids": "0,1",
                                "status": "Fail",
                                "warnings": [
                                    {
                                        "warning": message,
                                        "error_id": {"code": 42, "field_id": 150},
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    )
    recommendations = build_dcgm_recommendations(
        findings, returncode=0, parse_error=None
    )

    assert findings == [
        {
            "test_name": test_name,
            "status": "FAIL",
            "entities": ["0,1"],
            "error_codes": ["42", "150"],
            "messages": [message],
            "error_severities": [],
            "result_path": ("$.DCGM GPU Diagnostic.tests[0].results[0]"),
        }
    ]
    assert recommendations[0]["action_code"] == expected_action
    assert recommendations[0]["priority"] == "IMMEDIATE"
    assert recommendations[0]["trigger_tests"] == [test_name]


def test_dcgm_execution_failure_has_fixed_guidance() -> None:
    recommendations = build_dcgm_recommendations(
        [], returncode=2, parse_error="JSONDecodeError: invalid document"
    )

    assert recommendations[0]["action_code"] == ("DCGM_EXECUTION_REVIEW")
    assert recommendations[0]["priority"] == "IMMEDIATE"


def _persistence_mode_payload() -> dict:
    """Real 2026-08-10 payload captured before the HyperPod node
    persistence baseline was provisioned: every GPU fails the
    Deployment/software check with CONFIG severity only.

    The missing ``nvidia-persistenced`` unit is not causal. The same
    fleet was revalidated on 2026-08-19 with no NVIDIA unit, all GPUs
    in persistence mode, and a clean Level 1 diagnostic.
    """
    return {
        "DCGM Diagnostic": {
            "test_categories": [
                {
                    "category": "Deployment",
                    "tests": [
                        {
                            "name": "software",
                            "results": [
                                {
                                    "entity_group": "GPU",
                                    "entity_id": gpu,
                                    "status": "Fail",
                                    "warnings": [
                                        {
                                            "error_category": 8,
                                            "error_id": 29,
                                            "error_severity": 5,
                                            "warning": (
                                                "Persistence Mode: "
                                                "Persistence mode "
                                                f"for GPU {gpu} is "
                                                "disabled."
                                            ),
                                        }
                                    ],
                                }
                                for gpu in range(8)
                            ],
                        }
                    ],
                }
            ]
        }
    }


def test_config_severity_failures_are_configuration_only() -> None:
    findings = extract_dcgm_diagnostic_findings(_persistence_mode_payload())

    assert len(findings) == 8
    assert all(finding["error_severities"] == ["5"] for finding in findings)
    assert dcgm_failures_are_configuration_only(findings)


def test_config_severity_guidance_does_not_drain() -> None:
    findings = extract_dcgm_diagnostic_findings(_persistence_mode_payload())
    recommendations = build_dcgm_recommendations(
        findings, returncode=226, parse_error=None, configuration_only=True
    )

    assert [item["action_code"] for item in recommendations] == [
        "GPU_HOST_CONFIG_REMEDIATION"
    ]
    assert recommendations[0]["priority"] == "REVIEW"
    assert recommendations[0]["trigger_tests"] == ["software"]


def test_isolate_severity_alongside_config_still_drains() -> None:
    payload = _persistence_mode_payload()
    payload["DCGM Diagnostic"]["test_categories"].append(
        {
            "category": "Hardware",
            "tests": [
                {
                    "name": "memory",
                    "results": [
                        {
                            "entity_id": 3,
                            "status": "Fail",
                            "warnings": [
                                {
                                    "error_id": 12,
                                    "error_severity": 2,
                                    "warning": (
                                        "Uncorrectable ECC memory error detected"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    findings = extract_dcgm_diagnostic_findings(payload)

    assert not dcgm_failures_are_configuration_only(findings)
    recommendations = build_dcgm_recommendations(
        findings, returncode=226, parse_error=None
    )
    action_codes = {item["action_code"] for item in recommendations}
    assert "GPU_MEMORY_FIELD_DIAGNOSTIC" in action_codes
    assert "DCGM_EXECUTION_REVIEW" in action_codes


def test_missing_severity_is_not_configuration_only() -> None:
    """Older DCGM builds omit ``error_severity``; unknown severity
    must stay fail-closed rather than silently keep a node in
    service."""
    findings = extract_dcgm_diagnostic_findings(
        {
            "DCGM Diagnostic": {
                "test_categories": [
                    {
                        "category": "Hardware",
                        "tests": [
                            {
                                "name": "pcie",
                                "results": [
                                    {
                                        "entity_id": 0,
                                        "status": "Fail",
                                        "warnings": [{"warning": "AER error"}],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    )

    assert findings[0]["error_severities"] == []
    assert not dcgm_failures_are_configuration_only(findings)


def test_no_failures_are_not_configuration_only() -> None:
    findings = extract_dcgm_diagnostic_findings(
        {
            "DCGM Diagnostic": {
                "test_categories": [
                    {
                        "category": "Deployment",
                        "tests": [
                            {
                                "name": "software",
                                "results": [{"entity_id": 0, "status": "Pass"}],
                            }
                        ],
                    }
                ]
            }
        }
    )

    assert not dcgm_failures_are_configuration_only(findings)
