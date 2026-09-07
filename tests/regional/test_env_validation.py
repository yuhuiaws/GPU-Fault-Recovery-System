from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from gpu_fault import env_validation
from gpu_fault.env import TRUE_TOKENS
from gpu_fault.env_validation import (
    TRAINING_HEALTH_MONITOR_ENV,
    environment_inventory,
    environment_value_bounds,
    environment_value_kinds,
    invalid_gpu_fault_environment_values,
    training_health_monitor_enabled,
    unknown_gpu_fault_environment,
    validate_gpu_fault_environment,
)

ROOT = Path(__file__).resolve().parents[2]
REGIONAL_GENERATED = ROOT / "deploy" / "control-plane" / "regional" / "generated"
PLACEHOLDERS = ("REPLACE_WITH_", "{{", "${", "$(")


def test_unknown_gpu_fault_environment_fails_closed() -> None:
    values = {"GPU_FAULT_PROCESSOR_MOD": "active-active"}

    with pytest.raises(RuntimeError, match="GPU_FAULT_PROCESSOR_MOD"):
        validate_gpu_fault_environment(values, process_name="test-process")


def test_dynamic_notification_ttl_is_allowed() -> None:
    values = {"GPU_FAULT_NOTIFICATION_TTL_SECONDS_HEALTH_TREND": "60"}

    assert unknown_gpu_fault_environment(values) == []


def test_unknown_environment_can_warn(caplog: pytest.LogCaptureFixture) -> None:
    values = {
        "GPU_FAULT_UNKNOWN_ENV_POLICY": "warn",
        "GPU_FAULT_UNKNOWN_SETTING": "value",
    }

    with caplog.at_level(logging.WARNING):
        validate_gpu_fault_environment(values, process_name="test-process")

    assert "GPU_FAULT_UNKNOWN_SETTING" in caplog.text


def test_pytest_only_environment_is_scoped_to_pytest() -> None:
    values = {
        "PYTEST_CURRENT_TEST": "tests/test_example.py::test_case",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://example",
    }

    assert unknown_gpu_fault_environment(values) == []


def test_deploy_manifests_only_use_known_environment() -> None:
    known, prefixes = environment_inventory()
    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str) and name.startswith("GPU_FAULT_"):
                names.add(name)
            if value.get("kind") == "ConfigMap":
                names.update(
                    key
                    for key in (value.get("data") or {})
                    if key.startswith("GPU_FAULT_")
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in (ROOT / "deploy").rglob("*.yaml"):
        for document in yaml.load_all(
            path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        ):
            visit(document)

    assert (
        sorted(
            name
            for name in names
            if name not in known
            and not any(name.startswith(prefix) for prefix in prefixes)
        )
        == []
    )


def test_all_runtime_entrypoints_validate_environment() -> None:
    for relative in (
        "src/gpu_fault/app/context.py",
        "src/gpu_fault/node_agent/app.py",
        "src/gpu_fault/cluster_executor.py",
        "src/gpu_fault/collectors_cli.py",
        "src/gpu_fault/completion_controller.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "validate_gpu_fault_environment" in text


def test_typed_values_are_validated_before_the_process_starts() -> None:
    """A value its own reader cannot parse must stop the process, not a request.

    `GPU_FAULT_API_PORT=eight` used to reach uvicorn and `...RESPONSE_TIMEOUT`
    used to be parsed per request, so a typo produced a Ready Pod that failed
    every call instead of a deployment that visibly refused to roll out.
    """

    problems = invalid_gpu_fault_environment_values(
        {
            "GPU_FAULT_API_PORT": "eight",
            "GPU_FAULT_PROCESSOR_RESPONSE_TIMEOUT_SECONDS": "soon",
            "GPU_FAULT_STORE_URL": "postgresql://example/db",
        }
    )

    assert problems == [
        "GPU_FAULT_API_PORT must be an integer",
        "GPU_FAULT_PROCESSOR_RESPONSE_TIMEOUT_SECONDS must be a number",
    ]


def test_well_formed_values_are_accepted() -> None:
    assert (
        invalid_gpu_fault_environment_values(
            {
                "GPU_FAULT_API_PORT": " 8080 ",
                "GPU_FAULT_PROCESSOR_RESPONSE_TIMEOUT_SECONDS": "115.5",
                "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false",
                "GPU_FAULT_TELEMETRY_SPOOL": "1",
                # Empty means unset: every read site falls back to its default.
                "GPU_FAULT_NODE_ALLOW_GPU_RESET": "",
            }
        )
        == []
    )


def test_boolean_typo_is_rejected() -> None:
    """`ture` used to be a silently disabled safety switch."""

    problems = invalid_gpu_fault_environment_values(
        {"GPU_FAULT_NODE_ALLOW_GPU_RESET": "ture"}
    )

    assert problems == [
        "GPU_FAULT_NODE_ALLOW_GPU_RESET must be one of 0/1/false/no/off/on/true/yes"
    ]


def test_enabled_token_a_switch_does_not_recognise_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A switch that reads `true` only turns `yes` into "silently off".

    Every switch now reads through ``env_bool``, so the packaged inventory no
    longer carries such a switch; the rule stays because the inventory is
    derived from the code and a future narrow read site would bring it back.
    Refusing is sound in one direction only: an operator never writes an
    enabled-looking token meaning disabled, while `no`/`0`/`off` mean disabled to
    every reader and stay acceptable.
    """

    def narrow_kinds() -> dict[str, tuple[str, frozenset[str]]]:
        return {"GPU_FAULT_NODE_ALLOW_GPU_RESET": ("boolean", frozenset({"true"}))}

    monkeypatch.setattr(env_validation, "environment_value_kinds", narrow_kinds)

    assert invalid_gpu_fault_environment_values(
        {"GPU_FAULT_NODE_ALLOW_GPU_RESET": "yes"}
    ) == [
        "GPU_FAULT_NODE_ALLOW_GPU_RESET is only enabled by true; "
        "the configured value would leave it switched off"
    ]
    assert (
        invalid_gpu_fault_environment_values({"GPU_FAULT_NODE_ALLOW_GPU_RESET": "no"})
        == []
    )


def test_a_parsable_value_the_read_site_refuses_fails_at_start_up() -> None:
    """The range is checked where the operator is still watching.

    ``0`` is a perfectly good integer, so the type check passes it through and the
    refusal used to happen inside whichever service constructs the dispatcher --
    one role, minutes into a rollout, behind a Pod that already reported Ready.
    """

    assert invalid_gpu_fault_environment_values(
        {"GPU_FAULT_NOTIFICATION_DISPATCH_SCAN_LIMIT": "0"}
    ) == ["GPU_FAULT_NOTIFICATION_DISPATCH_SCAN_LIMIT must be at least 1"]
    assert (
        invalid_gpu_fault_environment_values(
            {"GPU_FAULT_NOTIFICATION_DISPATCH_SCAN_LIMIT": "1"}
        )
        == []
    )

    with pytest.raises(RuntimeError, match="must be at least 30"):
        validate_gpu_fault_environment(
            {"GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS": "5"},
            process_name="test-process",
        )


def test_every_declared_bound_is_the_one_its_read_site_enforces() -> None:
    """The bounds are derived, so this pins what they were derived from.

    Each entry below is the guard in the reading code, quoted by hand. A generator
    that started inferring a bound from something else -- an arithmetic
    expression, a comparison against another setting, a guard nested under a
    feature switch -- would refuse a value production is entitled to use, so the
    derivation is checked against the source rather than against itself.
    """

    bounds = environment_value_bounds()

    assert bounds["GPU_FAULT_NOTIFICATION_DISPATCH_SCAN_LIMIT"] == (1, None)
    # ``if lease_duration < 30: raise`` in gpu_fault/execution/config.py.
    assert bounds["GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS"] == (30, None)
    # ``if not 3 <= self.hung_pyspy_sample_count <= 5: raise`` in host_health.py.
    assert bounds["GPU_FAULT_HUNG_PYSPY_SAMPLE_COUNT"] == (3, 5)
    # ``if not 0 < self.efa_baseline_alpha <= 1: raise``: the upper bound is
    # expressible and inclusive, the lower one is "greater than 0", which has no
    # inclusive form for a real number and is therefore not recorded at all.
    assert bounds["GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA"] == (None, 1)
    assert (
        invalid_gpu_fault_environment_values(
            {"GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA": "0.0"}
        )
        == []
    )
    assert invalid_gpu_fault_environment_values(
        {"GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA": "1.5"}
    ) == ["GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA must be at most 1"]


def test_an_untyped_variable_is_never_range_checked() -> None:
    """No bound without the coercion that makes the text a number.

    A variable the kind inference left alone is deliberately unvalidated, and a
    range on a value nothing parses would be a guess.
    """

    kinds = environment_value_kinds()

    assert set(environment_value_bounds()) <= set(kinds)
    for name in environment_value_bounds():
        assert kinds[name][0] in {"integer", "number"}, f"{name} bounded but untyped"


def test_value_problems_never_quote_the_value() -> None:
    """Some GPU_FAULT_* variables carry tokens, and this text reaches logs."""

    secret = "AKIAsecret-token-value"

    problems = invalid_gpu_fault_environment_values(
        {"GPU_FAULT_NODE_ACTION_KEY_VERSION": secret}
    )

    assert problems == ["GPU_FAULT_NODE_ACTION_KEY_VERSION must be an integer"]
    assert secret not in " ".join(problems), "an environment value reached a message"


def test_value_validation_runs_even_when_unknown_names_only_warn() -> None:
    values = {
        "GPU_FAULT_UNKNOWN_ENV_POLICY": "warn",
        "GPU_FAULT_UNKNOWN_SETTING": "value",
        "GPU_FAULT_API_PORT": "http",
    }

    with pytest.raises(RuntimeError, match="GPU_FAULT_API_PORT must be an integer"):
        validate_gpu_fault_environment(values, process_name="test-process")


@pytest.mark.parametrize("token", ["1", "yes", "on", "TRUE"])
def test_training_health_monitor_accepts_every_enabled_token(token: str) -> None:
    """``=1`` used to leave the monitor silently off."""

    assert training_health_monitor_enabled({TRAINING_HEALTH_MONITOR_ENV: token}), (
        f"{token!r} must enable the monitor"
    )


def test_training_health_monitor_typo_is_loud() -> None:
    with pytest.raises(ValueError, match=TRAINING_HEALTH_MONITOR_ENV):
        training_health_monitor_enabled({TRAINING_HEALTH_MONITOR_ENV: "ture"})


def test_every_boolean_switch_accepts_the_same_enabled_tokens() -> None:
    """One parser, one token set: no switch may still be ``true``-only.

    Reads the packaged inventory, which ``scripts/generate-env-reference.py``
    derives from the read sites, so this is red until the inventory is
    regenerated after the last ``== "true"`` read is gone.
    """

    true_only = sorted(
        name
        for name, (kind, true_tokens) in environment_value_kinds().items()
        if kind == "boolean" and true_tokens != TRUE_TOKENS
    )

    assert true_only == [], (
        "these switches still accept a narrower set of enabled tokens: "
        + ", ".join(true_only)
    )


def test_hyperpod_safety_switches_are_typed() -> None:
    """The switches the design forbids enabling must be schema-checked.

    A misspelled value on one of these reads as "off", which is safe for the
    replace guard and unsafe for the ones that gate recovery, so both directions
    have to fail loudly instead.
    """

    kinds = environment_value_kinds()

    for name in (
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE",
        "GPU_FAULT_ALLOW_HYPERPOD_REBOOT",
        "GPU_FAULT_ALLOW_HYPERPOD_MUTATION",
        "GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY",
        "GPU_FAULT_NODE_ALLOW_GPU_RESET",
        "GPU_FAULT_PROXY_HEADERS_ENABLED",
    ):
        assert kinds[name][0] == "boolean", f"{name} lost its inferred type"


def test_generated_regional_values_satisfy_the_typed_schema() -> None:
    """The rendered production manifests are the values that actually ship."""

    values: dict[str, str] = {}

    def visit(document: object) -> None:
        if isinstance(document, dict):
            name = document.get("name")
            value = document.get("value")
            if isinstance(name, str) and isinstance(value, str):
                values[name] = value
            for key, child in document.items():
                if key == "data" and isinstance(child, dict):
                    values.update(
                        {
                            key: item
                            for key, item in child.items()
                            if isinstance(key, str) and isinstance(item, str)
                        }
                    )
                visit(child)
        elif isinstance(document, list):
            for child in document:
                visit(child)

    for path in sorted(REGIONAL_GENERATED.glob("*.yaml")):
        for document in yaml.load_all(
            path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        ):
            visit(document)
    shipped = {
        name: value
        for name, value in values.items()
        if name.startswith("GPU_FAULT_")
        and not any(marker in value for marker in PLACEHOLDERS)
    }

    assert len(shipped) >= 100, "the regional manifest scan found almost nothing"
    assert invalid_gpu_fault_environment_values(shipped) == []


def test_inventory_is_packaged_as_json() -> None:
    inventory = json.loads(
        (ROOT / "src/gpu_fault/data/env-inventory.json").read_text(encoding="utf-8")
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert len(inventory["variables"]) >= 500
    assert inventory["schema_version"] == 3
    assert len(inventory["value_kinds"]) >= 250
    assert set(inventory["value_kinds"]) <= set(inventory["variables"])
    assert len(inventory["value_bounds"]) >= 20
    assert set(inventory["value_bounds"]) <= set(inventory["value_kinds"])
    assert '"data/*.json"' in pyproject
