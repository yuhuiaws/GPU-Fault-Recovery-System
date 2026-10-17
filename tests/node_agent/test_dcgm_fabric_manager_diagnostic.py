"""DCGM's Fabric Manager training failure is never "configuration only".

Live (plain EKS p5en, DCGM 4.7.0 host engine): ``dcgmi diag -r 1`` graded the
software test FAIL on all eight GPUs with "Fabric Manager (Cluster UUID: ...,
Clique ID: 0, Health Mask: 0): training in progress" on the summary. DCGM files
that message (``DCGM_FR_FABRIC_MANAGER_TRAINING_ERROR``, 112) under severity
CONFIG, so the product minted GPU_HOST_CONFIG_REMEDIATION -- "Do not drain the
node for this" -- beside the DRAIN that rightly quarantined a node on which CUDA
could not start; had every failure carried the CONFIG severity, the node would
have been graded WARN and kept in service. The failure now grades FAIL and its
one guidance says what the product does: the fabric recovery.
"""

from __future__ import annotations

import json

import pytest

from gpu_fault.dcgm_diagnostic_analysis import (
    build_dcgm_recommendations,
    dcgm_failures_are_configuration_only,
    extract_dcgm_diagnostic_findings,
    finding_is_fabric_manager_training,
)

from ._support import (
    CompletedProcess,
    WorkflowOperation,
    command,
    envelope,
    node_action_executor,
)

LIVE_MESSAGE = (
    "Fabric Manager (Cluster UUID: 00000000-0000-0000-0000-000000000000, Clique "
    "ID: 0, Health Mask: 0): training in progress. Ensure that the FabricManager "
    "is running without errors."
)


def _warning(message: str, severity: int = 5) -> dict:
    return {
        "error_category": 6,
        "error_id": 112,
        "error_severity": severity,
        "warning": message,
    }


def _payload(*, per_gpu_message: bool, message: str = LIVE_MESSAGE) -> dict:
    results = [
        {
            "entity_group": "GPU",
            "entity_id": gpu,
            "status": "Fail",
            **({"warnings": [_warning(message)]} if per_gpu_message else {}),
        }
        for gpu in range(8)
    ]
    return {
        "DCGM Diagnostic": {
            "test_categories": [
                {
                    "category": "Deployment",
                    "tests": [
                        {
                            "name": "software",
                            "results": results,
                            "test_summary": {
                                "status": "Fail",
                                "warnings": [_warning(message)],
                            },
                        }
                    ],
                }
            ]
        }
    }


@pytest.mark.parametrize("per_gpu_message", [False, True], ids=["live", "all-config"])
def test_an_untrained_fabric_grades_fail_with_the_fabric_guidance(
    tmp_path, per_gpu_message: bool
) -> None:
    def runner(argv, **_kwargs):
        stdout = json.dumps(_payload(per_gpu_message=per_gpu_message))
        return CompletedProcess(argv, 226, stdout=stdout, stderr="")

    agent = node_action_executor(
        tmp_path,
        "fabric-diag.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / "quick-fabric"),
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))

    assert result.details["diagnostic_outcome"] == "FAIL", (
        "a node whose GPUs have not registered with the fabric cannot run CUDA"
    )
    assert result.details["configuration_only_failures"] is False
    actions = [item["action_code"] for item in result.details["recommended_actions"]]
    assert actions == ["NVLINK_FABRIC_RECOVERY"], (
        "no 'do not drain' host-configuration advice, no hostengine review for the "
        "non-zero exit the failure itself causes, no driver advice for the per-GPU "
        f"results of the same test: {actions}"
    )
    (guidance,) = result.details["recommended_actions"]
    assert guidance["priority"] == "IMMEDIATE"
    assert "reset every GPU and NVSwitch" in guidance["instruction"], (
        "the guidance names the product's recovery"
    )
    assert "submit-remediation --disposition restore" in guidance["instruction"]
    assert guidance["trigger_tests"] == ["software"]


@pytest.mark.parametrize(
    "message",
    [
        LIVE_MESSAGE,
        "Fabric Manager (Cluster UUID: x, Clique ID: 0): training not started.",
        "Fabric Manager (Cluster UUID: x, Clique ID: 0): Training completed with an "
        "error: Timeout.",
        "Fabric Manager (Cluster UUID: x, Clique ID: 0): Receiving an unknown "
        "fabric manager state from NVML.",
    ],
)
def test_every_dcgm_variant_of_the_message_is_recognised(message: str) -> None:
    (summary,) = [
        finding
        for finding in extract_dcgm_diagnostic_findings(
            _payload(per_gpu_message=False, message=message)
        )
        if finding["messages"]
    ]

    assert finding_is_fabric_manager_training(summary), message
    assert not dcgm_failures_are_configuration_only([summary]), (
        "CONFIG by DCGM's table, never configuration-only here"
    )


def test_host_configuration_failures_keep_their_guidance() -> None:
    findings = extract_dcgm_diagnostic_findings(
        _payload(per_gpu_message=True, message="Persistence Mode: disabled")
    )

    assert dcgm_failures_are_configuration_only(findings), (
        "a CONFIG-only failure that is not the fabric still keeps the node"
    )
    actions = {
        item["action_code"]
        for item in build_dcgm_recommendations(
            findings, returncode=226, parse_error=None, configuration_only=True
        )
    }
    assert actions == {"GPU_HOST_CONFIG_REMEDIATION"}


def test_a_field_id_that_happens_to_be_112_is_not_the_fabric() -> None:
    """The extractor reads field ids too; field 112 is the throttle reasons."""

    finding = {
        "test_name": "Diagnostic",
        "status": "FAIL",
        "entities": [],
        "error_codes": ["112"],
        "messages": ["Clocks throttled: thermal"],
        "error_severities": ["2"],
        "result_path": "$",
    }

    assert not finding_is_fabric_manager_training(finding), (
        "a number alone never names the fabric"
    )
