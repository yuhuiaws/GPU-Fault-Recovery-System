from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

MAX_FINDINGS = 100
MAX_MESSAGES_PER_FINDING = 20
MAX_MESSAGE_LENGTH = 512

_STATUS_ALIASES = {
    "PASSED": "PASS",
    "SUCCESS": "PASS",
    "SUCCEEDED": "PASS",
    "FAILED": "FAIL",
    "WARNING": "WARN",
    "SKIPPED": "SKIP",
    "NOT RUN": "NOT_RUN",
    "NOTRUN": "NOT_RUN",
}
_VALID_STATUSES = {"PASS", "FAIL", "WARN", "SKIP", "NOT_RUN"}
_STATUS_KEYS = {
    "status",
    "result",
    "testresult",
    "overallresult",
}
_ENTITY_KEYS = {
    "entity",
    "entityid",
    "entitygroup",
    "gpuid",
    "gpuids",
    "gpuuuid",
}
_ERROR_CODE_KEYS = {
    "code",
    "errorcode",
    "errorid",
    "fieldid",
}
_SEVERITY_KEYS = {
    "errorseverity",
    "severity",
}
# dcgmErrorSeverity_t（NVIDIA DCGM bindings, dcgm_errors.py）：
# 0 NONE / 1 MONITOR / 2 ISOLATE / 3 UNKNOWN / 4 TRIAGE
# 5 CONFIG（"This error can be configured"）/ 6 RESET
DCGM_ERROR_SEVERITY_CONFIG = "5"
_MESSAGE_KEYS = {
    "error",
    "errors",
    "info",
    "message",
    "messages",
    "reason",
    "warning",
    "warnings",
}

_GUIDANCE_RULES = (
    (
        "THERMAL_COOLING_INSPECTION",
        ("thermal", "temperature", "overheat", "cooling"),
        "Keep the node drained; inspect airflow, fans, heat sinks, "
        "ambient temperature, and device temperature limits before "
        "revalidation.",
    ),
    (
        "NVLINK_NVSWITCH_INSPECTION",
        ("nvlink", "nvswitch", "fabric"),
        "Keep the node drained; inspect NVLink and NVSwitch health, "
        "Fabric Manager logs, link counters, and topology before a "
        "fabric validation.",
    ),
    (
        "PCIE_AER_INSPECTION",
        ("pcie", "pci express", "aer"),
        "Keep the node drained; inspect PCIe link state, AER records, "
        "replay counters, and the host or baseboard connection before "
        "GPU validation.",
    ),
    (
        "GPU_MEMORY_FIELD_DIAGNOSTIC",
        (
            "memory",
            "memtest",
            "memory bandwidth",
            "row remap",
            "ecc",
        ),
        "Keep the node drained; collect ECC and row-remap evidence, "
        "then run the configured NVIDIA Field Diagnostic before an "
        "RMA or return-to-service decision.",
    ),
    (
        "DRIVER_DCGM_REMEDIATION",
        (
            "deployment",
            "dcgm",
            "driver",
            "software",
            "library",
            "cuda",
        ),
        "Verify NVIDIA driver, DCGM, CUDA library compatibility and "
        "service health; remediate only through the pinned driver or "
        "software maintenance workflow.",
    ),
    (
        "GPU_CLIENT_DRIVER_CHECK",
        ("context create", "context_create", "permissions"),
        "Check active GPU clients, device permissions, driver state, "
        "and container runtime GPU access before rerunning diagnostics.",
    ),
    (
        "GPU_FIELD_DIAGNOSTIC",
        (
            "diagnostic",
            "sm stress",
            "targeted stress",
            "pulse test",
            "compute",
        ),
        "Keep the node drained; collect the NVIDIA bug report and run "
        "the configured NVIDIA Field Diagnostic before returning the "
        "GPU to service.",
    ),
)

_FALLBACK_GUIDANCE = (
    "DEEP_DIAGNOSTIC_REVIEW",
    "Keep the node drained; review the preserved DCGM evidence and "
    "run the configured deep diagnostic before returning it to "
    "service.",
)

_CONFIGURATION_GUIDANCE = (
    "GPU_HOST_CONFIG_REMEDIATION",
    "DCGM reports a configuration-severity condition only "
    "(dcgmErrorSeverity_t=CONFIG); the GPU is not diagnosed as "
    "faulty. Do not drain the node for this: apply the host "
    "configuration fix named in the message through the node "
    "provisioning workflow, then rerun the diagnostic.",
)


def normalize_dcgm_status(value: Any) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    normalized = str(value).strip().upper().replace("_", " ")
    normalized = _STATUS_ALIASES.get(normalized, normalized)
    if normalized in _VALID_STATUSES:
        return normalized
    return None


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z]", "", str(value).lower())


def _bounded_text(value: Any) -> str | None:
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text[:MAX_MESSAGE_LENGTH] if text else None
    return None


def _flatten_values(value: Any, *, limit: int) -> list[str]:
    values: list[str] = []

    def visit(item: Any) -> None:
        if len(values) >= limit:
            return
        text = _bounded_text(item)
        if text is not None:
            values.append(text)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return list(dict.fromkeys(values))


def _direct_status(value: dict[str, Any]) -> str | None:
    for key, item in value.items():
        if _normalized_key(key) in _STATUS_KEYS:
            status = normalize_dcgm_status(item)
            if status is not None:
                return status
    return None


def _test_name(value: dict[str, Any]) -> str | None:
    for key, item in value.items():
        if _normalized_key(key) in {"name", "testname"}:
            return _bounded_text(item)
    return None


def _details(
    value: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    entities: list[str] = []
    error_codes: list[str] = []
    messages: list[str] = []
    severities: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = _normalized_key(key)
                if normalized in _ENTITY_KEYS:
                    entities.extend(
                        _flatten_values(child, limit=MAX_MESSAGES_PER_FINDING)
                    )
                elif normalized in _SEVERITY_KEYS:
                    severities.extend(
                        _flatten_values(child, limit=MAX_MESSAGES_PER_FINDING)
                    )
                elif normalized in _ERROR_CODE_KEYS:
                    error_codes.extend(
                        _flatten_values(child, limit=MAX_MESSAGES_PER_FINDING)
                    )
                elif normalized in _MESSAGE_KEYS:
                    text = _bounded_text(child)
                    if text is not None:
                        messages.append(text)
                    else:
                        visit(child)
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return (
        list(dict.fromkeys(entities))[:MAX_MESSAGES_PER_FINDING],
        list(dict.fromkeys(error_codes))[:MAX_MESSAGES_PER_FINDING],
        list(dict.fromkeys(messages))[:MAX_MESSAGES_PER_FINDING],
        list(dict.fromkeys(severities))[:MAX_MESSAGES_PER_FINDING],
    )


def extract_dcgm_diagnostic_findings(
    payload: Any,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def visit(
        value: Any,
        *,
        path: str,
        inherited_test_name: str | None,
    ) -> None:
        if len(findings) >= MAX_FINDINGS:
            return
        if isinstance(value, dict):
            current_test_name = _test_name(value) or inherited_test_name
            status = _direct_status(value)
            if status is not None and current_test_name is not None:
                (
                    entities,
                    error_codes,
                    messages,
                    severities,
                ) = _details(value)
                findings.append(
                    {
                        "test_name": current_test_name,
                        "status": status,
                        "entities": entities,
                        "error_codes": error_codes,
                        "messages": messages,
                        "error_severities": severities,
                        "result_path": path,
                    }
                )
            for key, item in value.items():
                visit(
                    item,
                    path=f"{path}.{key}",
                    inherited_test_name=current_test_name,
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(
                    item,
                    path=f"{path}[{index}]",
                    inherited_test_name=inherited_test_name,
                )

    visit(payload, path="$", inherited_test_name=None)
    return findings


def finding_is_configuration_only(
    finding: dict[str, Any],
) -> bool:
    """True when DCGM itself grades the finding as configuration.

    ``dcgmErrorSeverity_t`` value ``CONFIG`` means "this error can be
    configured" — the GPU is not diagnosed as faulty, so the finding
    must not drain a node. A finding that reports no severity at all
    (older DCGM builds) is deliberately *not* treated as
    configuration: unknown severity stays fail-closed.
    """
    severities = [
        str(item).strip()
        for item in (finding.get("error_severities") or [])
        if str(item).strip()
    ]
    if not severities:
        return False
    return all(severity == DCGM_ERROR_SEVERITY_CONFIG for severity in severities)


def dcgm_failures_are_configuration_only(
    findings: list[dict[str, Any]],
) -> bool:
    """True when every failing check is configuration-severity.

    Used to keep a healthy node in service when the only thing DCGM
    objects to is host configuration (e.g. persistence mode disabled),
    which otherwise fails the DCGM diagnostic on every node forever.
    """
    failures = [finding for finding in findings if finding.get("status") == "FAIL"]
    if not failures:
        return False
    return all(finding_is_configuration_only(finding) for finding in failures)


def build_dcgm_recommendations(
    findings: list[dict[str, Any]],
    *,
    returncode: int,
    parse_error: str | None,
    configuration_only: bool = False,
) -> list[dict[str, Any]]:
    triggers: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    actionable = [
        finding for finding in findings if finding["status"] in {"FAIL", "WARN"}
    ]
    for finding in actionable:
        searchable = " ".join(
            [
                str(finding["test_name"]),
                *finding["messages"],
            ]
        ).lower()
        if finding_is_configuration_only(finding):
            rule = _CONFIGURATION_GUIDANCE
            priority = "REVIEW"
        else:
            rule = next(
                (
                    (code, instruction)
                    for code, keywords, instruction in (_GUIDANCE_RULES)
                    if any(keyword in searchable for keyword in keywords)
                ),
                _FALLBACK_GUIDANCE,
            )
            priority = "IMMEDIATE" if finding["status"] == "FAIL" else "REVIEW"
        triggers[(rule[0], priority, rule[1])].add(str(finding["test_name"]))

    # A non-zero dcgmi exit status is expected when the diagnostic
    # reports a configuration-severity failure; asking the operator to
    # inspect hostengine health in that case is a false alarm.
    if returncode != 0 and configuration_only and not parse_error:
        return _sorted_recommendations(triggers)

    if returncode != 0 or parse_error:
        reason = (
            "dcgmi returned a non-zero status"
            if returncode != 0
            else "DCGM JSON could not be parsed"
        )
        triggers[
            (
                "DCGM_EXECUTION_REVIEW",
                "IMMEDIATE",
                f"{reason}; verify dcgmi and hostengine health, preserve "
                "stderr, and rerun the diagnostic before returning the "
                "node to service.",
            )
        ].add("dcgmi diag")

    return _sorted_recommendations(triggers)


def _sorted_recommendations(
    triggers: dict[tuple[str, str, str], set[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "action_code": action_code,
            "priority": priority,
            "instruction": instruction,
            "trigger_tests": sorted(test_names),
        }
        for (
            action_code,
            priority,
            instruction,
        ), test_names in sorted(triggers.items())
    ]
