from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from gpu_fault.env_validation import (
    environment_inventory,
    unknown_gpu_fault_environment,
    validate_gpu_fault_environment,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_inventory_is_packaged_as_json() -> None:
    inventory = json.loads(
        (ROOT / "src/gpu_fault/data/env-inventory.json").read_text(encoding="utf-8")
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert len(inventory["variables"]) >= 500
    assert '"data/*.json"' in pyproject
