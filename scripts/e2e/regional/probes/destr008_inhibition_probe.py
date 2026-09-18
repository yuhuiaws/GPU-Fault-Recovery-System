"""Read-only installed capability checks, delivered as pinned acceptance source."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from typing import Any

from gpu_fault.regional_compatibility import (
    ACTIVATION_FORBIDDEN_PARAMETER,
    ACTIVATION_INHIBITION_CAPABILITY,
    ACTIVATION_INHIBITION_PROTOCOL_VERSION,
    ACTIVATION_INHIBITION_VERSION,
    CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    REMOTE_STEP_BATCHING_PROTOCOL_VERSION,
)


def _inhibition_version_matches(value: object) -> bool:
    return type(value) is int and value == ACTIVATION_INHIBITION_VERSION


def _keyword_supported(method: Any, keyword: str, *, default: object) -> bool:
    parameter = inspect.signature(method).parameters.get(keyword)
    return (
        parameter is not None
        and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and type(parameter.default) is type(default)
        and parameter.default == default
    )


def _api_inhibition_checks() -> dict[str, bool]:
    model = importlib.import_module(
        "gpu_fault.host_health"
    ).SyntheticNodeReplacementRequest
    field = model.model_fields.get(ACTIVATION_FORBIDDEN_PARAMETER)
    payload = {
        "event_id": "feature-proof",
        "cluster_id": "feature-proof",
        "node_id": "feature-proof",
        "observed_at": "2000-01-01T00:00:00Z",
        "runtime_profile_version": "feature-proof",
        "job_id": "feature-proof",
        "attempt_id": "feature-proof",
        "affected_workload_ids": ["feature-proof"],
        "reason": "read-only inhibition capability proof",
    }
    default = model.model_validate(payload)
    guarded = model.model_validate({**payload, ACTIVATION_FORBIDDEN_PARAMETER: True})
    invalid_rejected = True
    invalid_values: tuple[object, ...] = (False, None, 0, 1, "true", {}, [])
    for invalid in invalid_values:
        try:
            model.model_validate({**payload, ACTIVATION_FORBIDDEN_PARAMETER: invalid})
        except ValueError:
            continue
        invalid_rejected = False
    versioned = (
        importlib.import_module("gpu_fault.app.routes.collector_events"),
        importlib.import_module("gpu_fault.app.routes.regional"),
        importlib.import_module(
            "gpu_fault.orchestration.families.health"
        ).NodeHealthPlanBuilder,
        importlib.import_module(
            "gpu_fault.orchestration.families.node_lifecycle"
        ).NodeLifecycleOperationService,
    )
    claims = (
        ("gpu_fault.store.memory.remote_commands", "MemoryRemoteCommandMixin"),
        ("gpu_fault.store.sqlite.remote_commands", "SqliteRemoteCommandMixin"),
        ("gpu_fault.store.postgres.remote_commands", "PostgresRemoteCommandMixin"),
    )
    return {
        "request_field": field is not None and field.frozen is True,
        "literal_true_only": (
            getattr(guarded, ACTIVATION_FORBIDDEN_PARAMETER, None) is True
            and guarded.model_dump(mode="json").get(ACTIVATION_FORBIDDEN_PARAMETER)
            is True
            and invalid_rejected
        ),
        "opt_in_only": ACTIVATION_FORBIDDEN_PARAMETER
        not in default.model_dump(mode="json"),
        "propagation_and_claim_version": all(
            _inhibition_version_matches(
                getattr(item, "activation_inhibition_version", None)
            )
            for item in versioned
        ),
        "claim_protocol_argument": all(
            _keyword_supported(
                getattr(importlib.import_module(module), name).claim_remote_commands,
                "executor_protocol_version",
                default=LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
            )
            for module, name in claims
        ),
    }


def _executor_inhibition_checks() -> dict[str, bool]:
    module = importlib.import_module("gpu_fault.hyperpod_spares")
    coordinator = module.HyperPodSpareCoordinator
    adapter = importlib.import_module(
        "gpu_fault.adapters.hyperpod.lifecycle"
    ).HyperPodLifecycleStepAdapter
    refused = True
    present_values: tuple[object, ...] = (True, False, None, 0, 1, "true", {}, [])
    for value in present_values:
        try:
            module.require_spare_activation_permitted(value)
        except module.SpareActivationForbidden:
            continue
        refused = False
    absence_allowed = module.require_spare_activation_permitted() is None
    return {
        "coordinator_version": _inhibition_version_matches(
            getattr(coordinator, "activation_inhibition_version", None)
        ),
        "confirmation_version": _inhibition_version_matches(
            getattr(adapter, "activation_inhibition_version", None)
        ),
        "allocate_keyword": _keyword_supported(
            coordinator.allocate,
            ACTIVATION_FORBIDDEN_PARAMETER,
            default=module.ACTIVATION_NOT_SPECIFIED,
        ),
        "reserve_keyword": _keyword_supported(
            coordinator.reserve,
            ACTIVATION_FORBIDDEN_PARAMETER,
            default=module.ACTIVATION_NOT_SPECIFIED,
        ),
        "present_values_refused": refused,
        "absence_allowed": absence_allowed,
    }


def activation_inhibition_feature_proof(*, component: str) -> dict[str, Any]:
    """Inspect installed code without constructing a service or reading credentials."""
    if component not in {"api", "executor"}:
        raise ValueError("inhibition proof component must be api or executor")
    checks = {
        "protocol": (
            type(CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION) is int
            and CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
            >= ACTIVATION_INHIBITION_PROTOCOL_VERSION
        ),
        "capability_version": _inhibition_version_matches(
            ACTIVATION_INHIBITION_VERSION
        ),
    }
    try:
        checks.update(
            _api_inhibition_checks()
            if component == "api"
            else _executor_inhibition_checks()
        )
        checks["inspection_complete"] = True
    except Exception:
        # Raw import/configuration errors can contain provider diagnostics.
        checks["inspection_complete"] = False
    return {
        "capability": ACTIVATION_INHIBITION_CAPABILITY,
        "capability_version": ACTIVATION_INHIBITION_VERSION,
        "component": component,
        "marker": ACTIVATION_FORBIDDEN_PARAMETER,
        "minimum_executor_protocol_version": ACTIVATION_INHIBITION_PROTOCOL_VERSION,
        "executor_protocol_version": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        "step_batching_protocol_version": REMOTE_STEP_BATCHING_PROTOCOL_VERSION,
        "supported": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    source_sha256 = globals().get("_PROBE_SHA256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("inhibition capability probe requires pinned source")
    if len(sys.argv) != 2 or sys.argv[1] not in {"cpu", "gpu"}:
        raise ValueError("inhibition capability probe requires cpu or gpu")
    component = {"cpu": "api", "gpu": "executor"}[sys.argv[1]]
    result = activation_inhibition_feature_proof(component=component)
    result["probe_sha256"] = source_sha256
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
