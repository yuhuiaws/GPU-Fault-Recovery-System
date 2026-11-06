"""Intent attached to newly compiled health workflows, independent of Xid actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gpu_fault.gpu_fabric_health import FABRIC_HEALTH
from gpu_fault.gpu_field_rules import HW_POWER_BRAKE_SLOWDOWN, HW_THERMAL_SLOWDOWN
from gpu_fault.gpu_memory_policy import MEMORY_METRICS
from gpu_fault.gpu_nvlink5_policy import (
    NVLINK5_RECOVERY_EVENTS,
    NVLINK5_RECOVERY_SUCCESSFUL,
)
from gpu_fault.gpu_pcie_link_policy import PCIE_CORRECTABLE_ERRORS, PCIE_LINK_DOWNTRAIN
from gpu_fault.host_diagnostics import HOST_DIAGNOSTIC_FAILURE
from gpu_fault.models import RecoveryAction

HEALTH_CONTRACT = "health_recovery_contract"
THERMAL_NAMES = frozenset(
    {
        "gpu_temperature_c",
        "gpu_tlimit_c",
        "memory_temperature_c",
        "thermal_violation_total_us",
        "clock_throttle_reasons",
        "composite:THERMAL_STRESS",
    }
)
#: Newer DCGM fields whose WARNING ``RUN_DIAGNOSTICS`` is the read-only
#: diagnosis of the family each one grades with, as their modules say: 1423 like
#: the 241 thermal violation time (``gpu_environment_policy.
#: THERMAL_DURATION_NAMES``), 1424 below the brake verdict like the 240 power
#: violation time (VALIDATE_GPU's ``POWER_WARNING_NAMES``), the downtrain and
#: 1501 like the 202 replay rate (``gpu_pcie_link_policy``), 1211/1213 like the
#: Hopper NVLink counters (``gpu_nvlink5_policy``) and a degraded fabric, a
#: WARNING plus an operator notice that no product reset undoes
#: (``gpu_fabric_health``). Off this list a GPU diagnostic compiles
#: RUN_DCGM_DIAGNOSTIC, whose failure -- or a failed VALIDATE_GPU behind it --
#: drains the node (``orchestration/escalation.py``). Their CRITICAL grades --
#: the confirmed brake's DRAIN, 1212's DRAIN, an unhealthy fabric's full reset
#: -- are no ``RUN_DIAGNOSTICS`` and keep their own chains.
READ_ONLY_FIELD_NAMES = frozenset(
    {
        HW_THERMAL_SLOWDOWN,
        HW_POWER_BRAKE_SLOWDOWN,
        PCIE_LINK_DOWNTRAIN,
        PCIE_CORRECTABLE_ERRORS,
        NVLINK5_RECOVERY_SUCCESSFUL,
        NVLINK5_RECOVERY_EVENTS,
        FABRIC_HEALTH,
    }
)
OBSERVABILITY_NAMES = frozenset(
    {"dcgm_field_completeness", "collector_observation_unknown"}
)
COOLDOWN_SECONDS = 300
COOLDOWN_MAX_GAP_SECONDS = 360
COOLDOWN_TIMEOUT_SECONDS = 1200


def is_hung_finding(finding: Any) -> bool:
    return (
        finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS
        and finding.diagnostic_parameters.get("diagnostic_reason")
        in {"EFA_TRAFFIC_HUNG_SUSPECTED", "TRAINING_HUNG"}
    )


def recovery_contract(finding: Any) -> dict[str, Any] | None:
    name = finding.metric_name or ""
    if name in THERMAL_NAMES and finding.recommended_action is RecoveryAction.DRAIN:
        return {
            "mode": "ENVIRONMENT_REVIEW",
            "metric": name,
            "detected_at": finding.observed_at.isoformat(),
            "node_id": finding.node_id,
        }
    if name in OBSERVABILITY_NAMES:
        return {
            "mode": "OBSERVABILITY",
            "detected_at": finding.observed_at.isoformat(),
            "collectors": finding.diagnostic_parameters.get(
                "collectors", ["GPU_METRICS"]
            ),
        }
    if is_hung_finding(finding):
        return {
            "mode": "HUNG_DIAGNOSTIC",
            "metric": name,
            "detected_at": finding.observed_at.isoformat(),
        }
    if finding.diagnostic_parameters.get("training_diagnostic"):
        return {
            "mode": "TRAINING",
            "detected_at": finding.observed_at.isoformat(),
            **finding.diagnostic_parameters["training_diagnostic"],
        }
    if finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS and (
        finding.category.value == "GPU"
        and (
            name
            in THERMAL_NAMES
            | MEMORY_METRICS
            | READ_ONLY_FIELD_NAMES
            | {
                "pcie_replay_total",
                "power_violation_total_us",
                "host_gpu_utilization_percent",
                "gpu_inventory_identity_changed",
                "gpu_expected_count_unknown",
                "gpu_expected_count_contradicted",
            }
            or name.startswith("composite:")
            or name.startswith("nvlink_")
            and name.endswith("_error_total")
        )
        or finding.policy_source == "SITE_APPLICATION_LOG"
    ):
        return {"mode": "DIAGNOSTIC_ONLY", "metric": name}
    return None


def prevents_hardware_escalation(parameters: dict[str, Any]) -> bool:
    contract = parameters.get(HEALTH_CONTRACT)
    return isinstance(contract, dict) and contract.get("mode") in {
        "THERMAL_COOLDOWN",
        "THERMAL_UNRESOLVED",
        "ENVIRONMENT_REVIEW",
        "OBSERVABILITY",
        "TRAINING",
        "DIAGNOSTIC_ONLY",
        "HUNG_DIAGNOSTIC",
    }


def failed_step_is_diagnostic(
    workflow: Any, index: int, details: Mapping[str, Any] | None
) -> bool:
    return (details or {}).get(HOST_DIAGNOSTIC_FAILURE) is True or (
        0 <= index < len(workflow.official_steps)
        and prevents_hardware_escalation(workflow.official_steps[index].parameters)
    )


def diagnostic_workflow_failures(workflow: Any, failures: list[Any]) -> bool:
    return bool(failures) and all(
        failed_step_is_diagnostic(workflow, item.step_index, item.details)
        for item in failures
    )
