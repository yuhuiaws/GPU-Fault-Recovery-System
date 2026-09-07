from __future__ import annotations

import json
import logging
import os
from functools import cache
from importlib.resources import files
from typing import Any, Mapping

# The token sets are re-exported on purpose: this module is where the schema
# check reads them, and they must be the very objects ``env_bool`` accepts.
from gpu_fault.env import BOOLEAN_TOKENS as BOOLEAN_TOKENS
from gpu_fault.env import TRUE_TOKENS as TRUE_TOKENS
from gpu_fault.env import env_bool, invalid_boolean_message

LOGGER = logging.getLogger(__name__)
PREFIX = "GPU_FAULT_"
POLICY_ENV = "GPU_FAULT_UNKNOWN_ENV_POLICY"
TRAINING_HEALTH_MONITOR_ENV = "GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR"


def training_health_monitor_enabled(
    values: Mapping[str, str] | None = None,
) -> bool:
    """Whether the training-health monitor is switched on.

    The monitor reads live training workload state, so it stays off until an
    operator asks for it. Both the processor's periodic runner and the
    non-processor worker start-up ask here, so the default is declared once
    instead of at each call site.
    """

    return env_bool(TRAINING_HEALTH_MONITOR_ENV, False, environ=values)


@cache
def _inventory_document() -> Mapping[str, Any]:
    resource = files("gpu_fault").joinpath("data/env-inventory.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError("environment inventory must be a JSON object")
    return document


@cache
def environment_inventory() -> tuple[frozenset[str], tuple[str, ...]]:
    document = _inventory_document()
    return (
        frozenset(str(item) for item in document["variables"]),
        tuple(str(item) for item in document.get("dynamic_prefixes", [])),
    )


@cache
def environment_value_kinds() -> Mapping[str, tuple[str, frozenset[str]]]:
    """The value kind of every variable whose type the code pins down.

    ``scripts/generate-env-reference.py`` derives these from the coercion each
    read site applies, so the schema cannot drift from the code that consumes it.
    A variable missing here has no single inferable type and is only checked for
    being a known name. For a boolean, the second element is the set of tokens
    its own read sites accept as enabled — several switches recognise ``true``
    only, so ``yes`` there means "silently off" rather than "on".
    """

    kinds = _inventory_document().get("value_kinds") or {}
    if not isinstance(kinds, dict):
        raise RuntimeError("environment inventory value_kinds must be an object")
    return {
        str(name): (
            str(entry["kind"]),
            frozenset(str(token) for token in entry.get("true_tokens", ())),
        )
        for name, entry in kinds.items()
    }


@cache
def environment_value_bounds() -> Mapping[str, tuple[float | None, float | None]]:
    """The range every read site of a numeric variable already enforces.

    ``scripts/generate-env-reference.py`` derives these the same way it derives
    the kinds: from the guard the reading code itself applies, so a bound is
    never invented here. Where two read sites disagree the widest range wins, so
    a value refused below is one no read site can use.

    The point is *when* it is refused. A zero scan budget or a five-second lease
    parses as an integer and then raises inside whichever service constructs
    first -- minutes into a rollout, in one role, often as a crash loop behind a
    Pod that already reported Ready. The same value is a misconfiguration at
    process start, which is where an operator is still watching.
    """

    document = _inventory_document().get("value_bounds") or {}
    if not isinstance(document, dict):
        raise RuntimeError("environment inventory value_bounds must be an object")
    return {
        str(name): (entry.get("minimum"), entry.get("maximum"))
        for name, entry in document.items()
    }


def _out_of_range(name: str, value: float) -> str | None:
    minimum, maximum = environment_value_bounds().get(name, (None, None))
    if minimum is not None and value < minimum:
        if maximum is not None:
            return f"{name} must be between {_plain(minimum)} and {_plain(maximum)}"
        return f"{name} must be at least {_plain(minimum)}"
    if maximum is not None and value > maximum:
        if minimum is not None:
            return f"{name} must be between {_plain(minimum)} and {_plain(maximum)}"
        return f"{name} must be at most {_plain(maximum)}"
    return None


def _plain(value: float) -> str:
    """Render a bound the way the operator wrote it, not as ``1.0``."""

    return str(int(value)) if float(value).is_integer() else str(value)


def invalid_gpu_fault_environment_values(
    values: Mapping[str, str],
) -> list[str]:
    """Describe every configured value its own read sites cannot use.

    Messages never quote the value: some ``GPU_FAULT_*`` variables carry tokens
    and passwords, and this text reaches logs and operator terminals.
    """

    problems: list[str] = []
    kinds = environment_value_kinds()
    for name in sorted(values):
        entry = kinds.get(name)
        if entry is None:
            continue
        kind, true_tokens = entry
        text = (values[name] or "").strip()
        if not text:
            # Empty means "unset" everywhere: each read site falls back to its
            # own default rather than parsing the blank.
            continue
        if kind in {"integer", "number"}:
            try:
                number: float = int(text) if kind == "integer" else float(text)
            except ValueError:
                problems.append(
                    f"{name} must be an integer"
                    if kind == "integer"
                    else f"{name} must be a number"
                )
            else:
                # A parsable value can still be one the reading code refuses.
                out_of_range = _out_of_range(name, number)
                if out_of_range is not None:
                    problems.append(out_of_range)
        elif kind == "boolean":
            token = text.lower()
            if token not in BOOLEAN_TOKENS:
                problems.append(invalid_boolean_message(name))
            elif true_tokens and token in TRUE_TOKENS - true_tokens:
                # Refusing only the enabled-looking tokens keeps this sound: no
                # deployment writes "on" meaning off, while "no"/"0"/"off" are
                # left alone because they mean off to every reader.
                problems.append(
                    f"{name} is only enabled by "
                    f"{'/'.join(sorted(true_tokens))}; the configured value "
                    "would leave it switched off"
                )
    return problems


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
    _validate_names(environment, process_name=process_name)
    _validate_values(environment, process_name=process_name)


def _validate_values(
    environment: Mapping[str, str],
    *,
    process_name: str,
) -> None:
    """Refuse to start on a value no read site can use.

    A malformed value used to surface wherever the owning module first parsed it:
    minutes into a rollout, as a crash loop in one role, or as a 500 per request
    while the Pod stayed Ready. There is no policy knob here — an unusable value
    is a misconfiguration in every mode, and starting anyway is exactly the
    fail-open behaviour this system is not allowed to have.
    """

    problems = invalid_gpu_fault_environment_values(environment)
    if problems:
        raise RuntimeError(
            f"{process_name} received unusable GPU_FAULT_* "
            f"environment value(s): {'; '.join(problems)}"
        )


def _validate_names(
    environment: Mapping[str, str],
    *,
    process_name: str,
) -> None:
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
