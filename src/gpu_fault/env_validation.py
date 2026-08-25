from __future__ import annotations

import json
import logging
import os
from functools import cache
from importlib.resources import files
from typing import Mapping


LOGGER = logging.getLogger(__name__)
PREFIX = "GPU_FAULT_"
POLICY_ENV = "GPU_FAULT_UNKNOWN_ENV_POLICY"


@cache
def environment_inventory() -> tuple[frozenset[str], tuple[str, ...]]:
    resource = files("gpu_fault").joinpath("data/env-inventory.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    return (
        frozenset(str(item) for item in document["variables"]),
        tuple(str(item) for item in document.get("dynamic_prefixes", [])),
    )


def unknown_gpu_fault_environment(
    values: Mapping[str, str],
) -> list[str]:
    known, dynamic_prefixes = environment_inventory()
    pytest_active = "PYTEST_CURRENT_TEST" in values
    return sorted(
        name
        for name in values
        if name.startswith(PREFIX)
        and name not in known
        and not any(name.startswith(prefix) for prefix in dynamic_prefixes)
        and not (pytest_active and name.startswith("GPU_FAULT_TEST_"))
    )


def validate_gpu_fault_environment(
    values: Mapping[str, str] | None = None,
    *,
    process_name: str,
) -> None:
    environment = os.environ if values is None else values
    unknown = unknown_gpu_fault_environment(environment)
    if not unknown:
        return
    policy = environment.get(POLICY_ENV, "error").strip().lower()
    if policy not in {"error", "warn", "ignore"}:
        raise RuntimeError(f"{POLICY_ENV} must be error, warn or ignore")
    message = (
        f"{process_name} received unknown GPU_FAULT_* "
        f"environment variable(s): {', '.join(unknown)}"
    )
    if policy == "ignore":
        return
    if policy == "warn":
        LOGGER.warning(message)
        return
    raise RuntimeError(message)
