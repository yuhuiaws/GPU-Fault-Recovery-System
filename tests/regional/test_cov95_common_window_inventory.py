from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import deployment_window_guard as guard
from scripts.e2e.regional import executor_env_window as window
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_common_windows import change_variable
from tests.regional.test_deployment_window_safety import WindowAPI


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        ((), [], "inventory is malformed"),
        (("metadata",), None, "identity is missing"),
        (("spec",), [], "identity is missing"),
        (("status",), None, "identity is missing"),
        (("metadata", "uid"), "", "identity or replica"),
        (("metadata", "resourceVersion"), 2, "identity or replica"),
        (("metadata", "generation"), True, "identity or replica"),
        (("metadata", "generation"), 0, "identity or replica"),
        (("metadata", "deletionTimestamp"), "deleting", "identity or replica"),
        (("spec", "replicas"), 0, "identity or replica"),
        (("spec", "replicas"), True, "identity or replica"),
        (("spec", "template", "spec", "containers"), {}, "containers are missing"),
        (
            ("spec", "template", "spec", "containers", 0, "env"),
            {},
            "environment is malformed",
        ),
        (
            ("spec", "template", "spec", "containers", 0, "env"),
            [None],
            "environment is malformed",
        ),
    ],
)
def test_inventory_refuses_unusable_identity_or_environment(
    path, replacement, message: str
) -> None:
    api = WindowAPI(window)
    value = api.deployment
    if not path:
        value = replacement
    else:
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
    with pytest.raises(RegionalFixtureError, match=message):
        guard.deployment_snapshot(
            value,
            plane="gpu",
            deployment=window.DEPLOYMENT,
            container=window.CONTAINER,
            variables=window.ALLOWED_VARIABLES,
        )
    assert api.patches == [], "malformed inventory cannot authorize a mutation"


def test_window_variable_must_be_a_string_literal() -> None:
    api = WindowAPI(window)
    change_variable(api, window.ALLOWED_VARIABLES[0], 10)
    with pytest.raises(RegionalFixtureError, match="literal string"):
        window.deployment_env(api)


@pytest.mark.parametrize("identity", [{}, {"cluster_id": "cluster", "release_id": ""}])
def test_scope_requires_both_release_and_cluster(identity: dict) -> None:
    api = SimpleNamespace(evidence_identity=lambda: identity)
    with pytest.raises(RegionalFixtureError, match="release identity"):
        guard.window_scope(
            api, plane="gpu", deployment="executor", container="executor"
        )


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ([], "variables are missing"),
        ({}, "variables are missing"),
        ({"LIMIT": None}, "variables are malformed"),
        ({"LIMIT": {"present": 1, "value": "1"}}, "variables are malformed"),
        ({"LIMIT": {"present": True, "value": None}}, "variables are malformed"),
        ({"LIMIT": {"present": False, "value": "1"}}, "variables are malformed"),
    ],
)
def test_restore_requires_recorded_presence_and_exact_literal(
    variables, message: str
) -> None:
    record = {
        "schema_version": 2,
        "scope": {"identity": "fixture"},
        "baseline": {"uid": "owned", "variables": variables},
        "assignments": {"LIMIT": "10"},
    }
    with pytest.raises(RegionalFixtureError, match=message):
        guard.require_window_record(
            record, record["scope"], {"uid": "owned"}, ("LIMIT",)
        )


@pytest.mark.parametrize("drift", ["uid", "value"])
def test_compare_and_swap_refuses_changes_since_survey(drift: str) -> None:
    api = WindowAPI(window)
    expected = window.deployment_env(api)
    name = window.ALLOWED_VARIABLES[0]
    if drift == "uid":
        api.deployment["metadata"]["uid"] = "foreign-deployment"
    else:
        change_variable(api, name, "foreign-value")
    with pytest.raises(RegionalFixtureError, match="changed"):
        guard.apply_window_variables(
            api,
            plane="gpu",
            deployment=window.DEPLOYMENT,
            container=window.CONTAINER,
            expected=expected,
            desired={name: "10"},
        )
    assert api.patches == [], "stale inventory must not overwrite a new owner"


def test_unchanged_environment_needs_no_write() -> None:
    api = WindowAPI(window)
    original = copy.deepcopy(api.deployment)
    guard.apply_window_variables(
        api,
        plane="gpu",
        deployment=window.DEPLOYMENT,
        container=window.CONTAINER,
        expected=window.deployment_env(api),
        desired={window.ALLOWED_VARIABLES[0]: None},
    )
    assert api.deployment == original, "deleting an absent variable must be a no-op"
    assert api.patches == [], "an unchanged environment must not roll replicas"
