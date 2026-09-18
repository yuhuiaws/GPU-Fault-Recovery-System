from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpu_fault.admin import grafana
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    BootstrapState,
)
from tests.admin._cov95_join_support import target
from tests.admin.test_admin_grafana import (
    AMP,
    HYPERPOD_WORKSPACE,
    REGION,
    SITE,
    Http,
    Runner,
)
from tests.admin.test_admin_grafana_admin_grant import IdentityRunner


@pytest.fixture
def cpu():
    return replace(target("a"), role="cpu")


class Responses(Runner):
    def __init__(self, responses=None, **options):
        super().__init__(**options)
        self.responses = responses or {}

    def run(self, arguments, **options):
        response = self.responses.get(tuple(arguments[1:3]))
        if response is not None:
            self.calls.append((list(arguments), options))
            if isinstance(response, BaseException):
                raise response
            return json.dumps(response)
        return super().run(arguments, **options)


@pytest.mark.parametrize(
    "role",
    [
        {
            "Arn": "arn:aws:iam::123456789012:role/gpu-fault-site-a-grafana",
            "Tags": [{"Key": SITE_TAG_KEY, "Value": "foreign"}],
        },
        {"Arn": "arn:aws:iam::123456789012:role/gpu-fault-site-a-grafana"},
        {
            "Arn": "arn:aws:iam::111122223333:role/gpu-fault-site-a-grafana",
            "Tags": [{"Key": SITE_TAG_KEY, "Value": SITE}],
        },
        {"Tags": [{"Key": SITE_TAG_KEY, "Value": SITE}]},
    ],
)
def test_grafana_bootstrap_never_rewrites_an_unproven_existing_role(cpu, role):
    runner = Responses({("iam", "get-role"): {"Role": role}})
    with pytest.raises(BootstrapError, match="site|identity|ownership"):
        grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE)
    assert runner.operations() == ["list-workspaces", "get-role"]


def test_grafana_bootstrap_reuses_only_the_owned_role(cpu):
    runner = Responses(
        {
            ("iam", "get-role"): {
                "Role": {
                    "Arn": "arn:aws:iam::123456789012:role/gpu-fault-site-a-grafana",
                    "Tags": [{"Key": SITE_TAG_KEY, "Value": SITE}],
                }
            }
        }
    )
    result = grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE)
    assert result["role_ownership"] == "REUSED"
    assert "create-role" not in runner.operations()
    assert "put-role-policy" in runner.operations()


def test_multiple_grafana_site_claims_fail_before_mutation(cpu):
    runner = Runner(
        workspaces=[
            {**HYPERPOD_WORKSPACE, "id": "g-one", "tags": {SITE_TAG_KEY: SITE}},
            {**HYPERPOD_WORKSPACE, "id": "g-two", "tags": {SITE_TAG_KEY: SITE}},
        ]
    )
    with pytest.raises(grafana.GrafanaIdentityError, match="multiple Grafana"):
        grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE)
    assert runner.operations() == ["list-workspaces"]


@pytest.mark.parametrize(
    "operation", [("iam", "get-role"), ("grafana", "describe-workspace")]
)
def test_grafana_read_failure_is_not_workspace_or_role_absence(cpu, operation):
    runner = Responses({operation: BootstrapError("example permission denied")})
    options = {"requested_id": "g-example"} if operation[0] == "grafana" else {}
    with pytest.raises(BootstrapError, match="permission denied"):
        grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE, **options)
    assert not any(
        name.startswith(("create", "put")) for name in runner.operations()
    ), "unproven Grafana read authorized a mutation"


def test_grafana_creation_requires_returned_workspace_identity(cpu):
    runner = Responses({("grafana", "create-workspace"): {"workspace": {}}})
    with pytest.raises(BootstrapError, match="returned no workspace id"):
        grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE)
    assert "describe-workspace" not in runner.operations()


@pytest.mark.parametrize(
    "state", ["FAILED", "DELETING", "timeout", "eventually-active"]
)
def test_grafana_creation_waits_only_for_admissible_bounded_readiness(
    cpu, monkeypatch, state
):
    runner = Responses()
    clock = SimpleNamespace(value=0.0)
    original = runner.run
    sleeps = []

    def read(arguments, **options):
        output = original(arguments, **options)
        if arguments[1:3] == ["grafana", "describe-workspace"]:
            document = json.loads(output)
            if state == "timeout" or state == "eventually-active" and not sleeps:
                document["workspace"]["status"] = "CREATING"
            elif state != "eventually-active":
                document["workspace"]["status"] = state
            return json.dumps(document)
        return output

    def sleep(seconds):
        sleeps.append(seconds)
        clock.value += 5 if state == "eventually-active" else 600

    monkeypatch.setattr(runner, "run", read)
    monkeypatch.setattr(
        grafana, "time", SimpleNamespace(monotonic=lambda: clock.value, sleep=sleep)
    )
    if state == "eventually-active":
        result = grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE)
        assert result["ownership"] == "CREATED"
    else:
        with pytest.raises(BootstrapError, match="entered|did not become ACTIVE"):
            grafana.ensure_grafana_workspace(runner, cpu=cpu, site_id=SITE)
    assert sleeps == ([5] if state in {"timeout", "eventually-active"} else [])


@pytest.fixture
def provisioning(tmp_path):
    directory = tmp_path / grafana.DASHBOARDS_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "example.json").write_text(
        json.dumps({"uid": "example", "title": "Example", "panels": []})
    )
    return {
        "workspace": {
            "workspace_id": HYPERPOD_WORKSPACE["id"],
            "endpoint": HYPERPOD_WORKSPACE["endpoint"],
        },
        "amp_workspace_id": AMP,
        "region": REGION,
        "dashboards_dir": directory,
    }


def test_invalid_dashboard_json_is_rejected_before_token_creation(provisioning):
    (provisioning["dashboards_dir"] / "example.json").write_text("invalid")
    runner = Runner()
    http = Http()
    with pytest.raises(BootstrapError, match="not valid JSON"):
        grafana.provision_grafana(runner, **provisioning, http=http)
    assert runner.calls == []
    assert http.requests == []


def test_missing_grafana_token_never_reaches_http(provisioning):
    runner = Runner()
    runner.token_key = ""
    http = Http()
    with pytest.raises(BootstrapError, match="token was not returned"):
        grafana.provision_grafana(runner, **provisioning, http=http)
    assert http.requests == []


def test_grafana_tolerates_nondocument_success_and_checks_unhealthy_datasource(
    provisioning,
):
    http = Http(
        {
            ("GET", "/api/org"): grafana.HttpResponse(200, "non-json-success"),
            ("GET", f"/api/datasources/uid/{grafana.DATASOURCE_UID}/health"): {
                "status": "ERROR"
            },
        }
    )
    result = grafana.provision_grafana(Runner(), **provisioning, http=http)
    assert result["status"] == "PROVISIONED"
    assert "/api/ds/query" in http.paths("POST")


def test_grafana_failed_token_revocation_does_not_mask_provisioning(
    provisioning, capsys
):
    runner = Responses(
        {
            ("grafana", "delete-workspace-service-account-token"): BootstrapError(
                "example revocation unavailable"
            )
        }
    )
    result = grafana.provision_grafana(runner, **provisioning, http=Http())
    assert result["status"] == "PROVISIONED"
    assert "token was not revoked" in capsys.readouterr().err


def test_degraded_grafana_access_output_never_looks_provisioned(tmp_path):
    state = BootstrapState(tmp_path / "bootstrap.json", site_id=SITE)
    state.record(
        "grafana_install",
        {"grafana": {"status": "DEGRADED", "reason": "example unavailable"}},
    )
    assert grafana.grafana_access_lines(state) == [
        "Grafana dashboards: DEGRADED; example unavailable"
    ]


def test_identity_center_without_store_identifier_is_not_an_admin_grant():
    runner = IdentityRunner(instances=[{"InstanceArn": "example-instance"}])
    result = grafana.grant_admin(
        runner,
        region=REGION,
        workspace_id=HYPERPOD_WORKSPACE["id"],
        email="example@example.invalid",
    )
    assert result["status"] == "not-derivable"
    assert "no IdentityStoreId" in result["reason"]
    assert "update-permissions" not in runner.operations()


def test_empty_identity_lookup_tries_both_bound_attributes():
    runner = IdentityRunner()
    original = runner.run

    def read(arguments, **options):
        if arguments[1] == "identitystore":
            runner.calls.append((list(arguments), options))
            return "{}"
        return original(arguments, **options)

    runner.run = read
    result = grafana.grant_admin(
        runner,
        region=REGION,
        workspace_id=HYPERPOD_WORKSPACE["id"],
        email="example@example.invalid",
    )
    assert result["status"] == "not-derivable"
    assert runner.operations().count("get-user-id") == 2
