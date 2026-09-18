from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import ha009_verdicts as verdicts
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from tests.regional.test_cov95_ha_rotation_verdicts import observations


@pytest.mark.parametrize(
    "name",
    [
        "DEPLOYMENTS",
        "role_status",
        "enabled_roles",
        "enabled_pods",
        "deployments_steady",
        "deployments_rolled",
        "rotation_errors",
        "observation_errors",
        "steady_deployments",
    ],
)
def test_verdict_extraction_preserves_the_runner_public_api(name: str) -> None:
    assert getattr(ha009, name) is getattr(verdicts, name)


@pytest.mark.parametrize("filename", ["ha009_verdicts.py", Path(ha009.__file__).name])
def test_script_style_imports_keep_the_stronger_rotation_verdicts(
    filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = Path(ha009.__file__).parent
    monkeypatch.syspath_prepend(str(directory))
    imported = runpy.run_path(str(directory / filename), run_name="ha009-import-test")
    arguments = observations()
    assert imported["rotation_errors"](**arguments) == []
    arguments["propagation"]["pods"] = {}
    assert "Secret propagation does not cover every expected Pod" in imported[
        "rotation_errors"
    ](**arguments)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("uid", "replacement", "UID changed"),
        ("replicas", 0, "replica target"),
        ("pods", [], "Pod observation is incomplete"),
        ("ready", 0, "not fully Ready"),
        ("updated", 0, "not fully Ready"),
        ("available", 0, "not fully Ready"),
    ],
)
def test_extracted_verdicts_keep_the_stronger_deployment_identity_and_population_checks(
    field: str, value: Any, fragment: str
) -> None:
    arguments = observations()
    assert verdicts.rotation_errors(**arguments) == []
    arguments["deployments_after"][verdicts.DEPLOYMENTS[0]][field] = value
    errors = verdicts.rotation_errors(**arguments)
    assert any(fragment in error for error in errors), errors


@pytest.mark.parametrize(
    ("section", "field", "fragment"),
    [
        ("propagation", "pods", "Secret propagation"),
        ("idle_observation", "samples", "post-idle samples"),
        ("idle_observation", "auth_failures_in_logs", "authentication log reads"),
    ],
)
def test_extracted_verdicts_require_all_expected_pods_in_every_observation(
    section: str, field: str, fragment: str
) -> None:
    arguments = observations()
    assert verdicts.rotation_errors(**arguments) == []
    arguments[section][field] = {}
    errors = verdicts.rotation_errors(**arguments)
    assert any(fragment in error for error in errors), errors


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("fresh_connection", False, "fresh exec-child SQL"),
        ("metrics_status", 503, "/metrics was not successfully observed"),
        ("healthz_status", "200", "/healthz returned"),
        (
            "metrics",
            {"gpu_fault_postgres_pool_connections_errors_total": float("nan")},
            "invalid value",
        ),
        (
            "metrics",
            {"gpu_fault_postgres_pool_connections_errors_total": float("inf")},
            "invalid value",
        ),
        (
            "metrics",
            {"gpu_fault_postgres_pool_connections_errors_total": -1},
            "invalid value",
        ),
        (
            "metrics",
            {"gpu_fault_postgres_pool_connections_errors_total": True},
            "invalid value",
        ),
    ],
)
def test_extracted_verdicts_keep_fresh_sql_and_finite_metric_requirements(
    field: str, value: Any, fragment: str
) -> None:
    arguments = observations()
    assert verdicts.rotation_errors(**arguments) == []
    first = next(iter(arguments["idle_observation"]["samples"].values()))
    first[0][field] = value
    errors = verdicts.rotation_errors(**arguments)
    assert any(fragment in error for error in errors), errors
